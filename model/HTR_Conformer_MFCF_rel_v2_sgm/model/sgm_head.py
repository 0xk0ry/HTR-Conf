# sgm_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sgm_vocab(converter, add_tokens=("<pad>", "<eos>", "<bos_left>", "<bos_right>")):
    # converter.character is typically an ordered list or str of your symbols
    # exclude the CTC blank; keep only real symbols
    base = list(converter.character)
    stoi = {ch: i for i, ch in enumerate(base)}
    for t in add_tokens:
        if t not in stoi:
            stoi[t] = len(stoi)
    itos = [''] * len(stoi)
    for k, v in stoi.items():
        itos[v] = k
    pad_id = stoi["<pad>"]
    eos_id = stoi["<eos>"]
    bos_l_id = stoi["<bos_left>"]
    bos_r_id = stoi["<bos_right>"]
    return stoi, itos, pad_id, eos_id, bos_l_id, bos_r_id


def texts_to_ids(texts, stoi):
    return [torch.tensor([stoi[ch] for ch in t], dtype=torch.long) for t in texts]


def make_context_batch(texts, stoi, sub_str_len=5, device='cuda'):
    """
    texts: list[str], length B
    returns:
      left_ctx  [B, Lmax, S], right_ctx [B, Lmax, S], tgt_ids [B, Lmax], tgt_mask [B, Lmax]
    """
    ids = texts_to_ids(texts, stoi)
    # Ensure all per-sample id tensors are on the target device to avoid CPU/CUDA cat issues
    ids = [t.to(device) for t in ids]
    B = len(ids)
    Lmax = max(t.size(0) for t in ids)
    S = sub_str_len

    left = torch.full(
        (B, Lmax, S), fill_value=stoi["<pad>"], dtype=torch.long, device=device)
    right = torch.full(
        (B, Lmax, S), fill_value=stoi["<pad>"], dtype=torch.long, device=device)
    tgt = torch.full(
        (B, Lmax),    fill_value=stoi["<pad>"], dtype=torch.long, device=device)
    mask = torch.zeros((B, Lmax),   dtype=torch.float32,      device=device)

    for b, seq in enumerate(ids):
        L = seq.size(0)
        tgt[b, :L] = seq
        mask[b, :L] = 1.0
        for i in range(L):
            # left window: ... c_{i-2}, c_{i-1} with BOS when missing
            l_start = max(0, i - S)
            l_ctx = seq[l_start:i]
            need = S - l_ctx.size(0)
            if need > 0:
                l_ctx = torch.cat(
                    [torch.tensor([stoi["<bos_left>"]] * need, device=device), l_ctx], dim=0)
            left[b, i] = l_ctx[-S:]

            # right window: c_{i+1}, c_{i+2}, ... with EOS when missing
            r_end = min(L, i + 1 + S)
            r_ctx = seq[i+1:r_end]
            need = S - r_ctx.size(0)
            if need > 0:
                r_ctx = torch.cat([r_ctx, torch.tensor(
                    [stoi["<eos>"]] * need, device=device)], dim=0)
            right[b, i] = r_ctx[:S]

    return left, right, tgt, mask


class SGMHead(nn.Module):
    def __init__(self, d_vis, vocab_size_sgm, d_txt=256, sub_str_len=5, num_heads=8, p_drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size_sgm
        self.sub_str_len = sub_str_len
        self.emb = nn.Embedding(vocab_size_sgm, d_txt)
        self.dir_left = nn.Parameter(torch.randn(1,1,d_txt))
        self.dir_right = nn.Parameter(torch.randn(1,1,d_txt))

        self.txt_proj = nn.Linear(d_txt, d_vis)
        self.q_norm  = nn.LayerNorm(d_vis)
        self.k_proj  = nn.Linear(d_vis, d_vis)
        self.v_proj  = nn.Linear(d_vis, d_vis)
        self.kv_norm = nn.LayerNorm(d_vis)

        # Keep an MHA instance for reference but perform attention manually to allow custom masking
        self.mha = nn.MultiheadAttention(d_vis, num_heads, dropout=p_drop, batch_first=True)
        self.num_heads = num_heads
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(d_vis, vocab_size_sgm)
        # tie weights
        self.classifier.weight = self.emb.weight  # optional: remove if dims differ

    def _context_to_query(self, ctx_ids, dir_token):
        E = self.emb(ctx_ids)               # [B,L,S,d_txt]
        # tiny 1D conv contextualizer over S
        q = E.transpose(2,3)                # [B,L,d_txt,S]
        q = F.conv1d(q.reshape(-1, q.size(2), q.size(3)), 
                     weight=torch.nn.init.xavier_uniform_(torch.empty(q.size(2), q.size(2), 3, device=q.device)),
                     padding=1).reshape(E.size(0), E.size(1), -1, E.size(3)).mean(2)
        q = q + dir_token
        q = self.q_norm(self.txt_proj(q))   # [B,L,D]
        return q

    def _cross_attend(self, Q, F, vis_mask=None, return_attn=False):
        """
        Custom multi-head cross-attention with optional multiplicative visual mask.
        Q: [B, L, D], F (keys/values): [B, N, D]
        vis_mask: optional [B, L, N] multiplicative mask in [0,1] applied to scores before softmax
        return_attn: whether to return attention weights [B, L, N]
        """
        B, L, D = Q.shape
        _, N, _ = F.shape

        K = self.kv_norm(self.k_proj(F))   # [B, N, D]
        V = self.kv_norm(self.v_proj(F))   # [B, N, D]

        H = self.num_heads
        Dh = D // H

        # Split into heads
        q = Q.view(B, L, H, Dh).transpose(1, 2)   # [B, H, L, Dh]
        k = K.view(B, N, H, Dh).transpose(1, 2)   # [B, H, N, Dh]
        v = V.view(B, N, H, Dh).transpose(1, 2)   # [B, H, N, Dh]

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, N]
        tau = torch.clamp(self.tau, 0.25, 4.0)
        scores = scores / (Dh ** 0.5 * tau)

        if vis_mask is not None:
            # vis_mask: [B, L, N] -> expand across heads
            m = vis_mask.unsqueeze(1)  # [B, 1, L, N]
            scores = scores * m

        attn = torch.softmax(scores, dim=-1)  # [B, H, L, N]
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)           # [B, H, L, Dh]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.dropout(out)

        if return_attn:
            # Average attention across heads for interpretability
            attn_avg = attn.mean(dim=1)  # [B, L, N]
            return out, attn_avg
        return out

    def forward(self, vis_tokens, left_ctx_ids, right_ctx_ids, tgt_ids, tgt_mask, vis_mask=None, return_attn: bool = False):
        """
        vis_tokens: [B, N, D]; left_ctx_ids/right_ctx_ids: [B, L, S]; tgt_ids: [B, L]
        tgt_mask: [B, L] with 1 for real labels, 0 for padding (no loss).
        Returns: dict(loss_sgm=...), logits_l, logits_r, [optional] attn_l, attn_r
        """
        Ql = self._context_to_query(
            left_ctx_ids,  self.dir_left)          # [B, L, D]
        Qr = self._context_to_query(
            right_ctx_ids, self.dir_right)         # [B, L, D]
        if return_attn:
            # [B, L, D], [B, L, N]
            Fl, attn_l = self._cross_attend(Ql, vis_tokens, vis_mask=vis_mask, return_attn=True)
            # [B, L, D], [B, L, N]
            Fr, attn_r = self._cross_attend(Qr, vis_tokens, vis_mask=vis_mask, return_attn=True)
        else:
            # [B, L, D]
            Fl = self._cross_attend(Ql, vis_tokens, vis_mask=vis_mask, return_attn=False)
            # [B, L, D]
            Fr = self._cross_attend(Qr, vis_tokens, vis_mask=vis_mask, return_attn=False)

        # [B, L, V]
        logits_l = self.classifier(Fl)
        logits_r = self.classifier(Fr)

        p_l = F.softmax(logits_l / 2.0, dim=-1)
        p_r = F.softmax(logits_r / 2.0, dim=-1)
        agree = F.mse_loss(p_l[:,:-1,:], p_r[:,1:,:], reduction='none').sum(-1)
        agree = (agree * tgt_mask[:,:-1]).sum() / (tgt_mask[:,:-1].sum().clamp_min(1.))
        # Base SGM loss placeholder (cross-entropy can be added externally); keep agreement term
        loss_sgm = 0.1 * agree

        out = {'loss_sgm': loss_sgm, 'logits_l': logits_l, 'logits_r': logits_r}
        if return_attn:
            out['attn_l'] = attn_l  # [B, L, N]
            out['attn_r'] = attn_r  # [B, L, N]
        return out
