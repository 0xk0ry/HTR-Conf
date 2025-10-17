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
    """
    Training-only semantic guidance head.
    - Visual features: F [B, N, D] from your ViT encoder (after final norm).
    - Context windows: left_ctx, right_ctx each [B, L, S] of char ids
      where L = label length (per sample max, padded), S = sub_str_len.
    - Predicts the L labels; computes CE loss for left and right guidance.
    """

    def __init__(self, d_vis, vocab_size_sgm, d_txt=256, sub_str_len=5, num_heads=8, p_drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size_sgm
        self.sub_str_len = sub_str_len
        self.emb = nn.Embedding(vocab_size_sgm, d_txt)

        # direction tokens
        self.dir_left = nn.Parameter(torch.randn(1, 1, d_txt))
        self.dir_right = nn.Parameter(torch.randn(1, 1, d_txt))

        # tiny text encoder -> pooled query
        self.txt_proj = nn.Linear(d_txt, d_vis)   # project to visual dim D
        self.q_norm = nn.LayerNorm(d_vis)
        self.kv_norm = nn.LayerNorm(d_vis)

        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(d_vis, vocab_size_sgm)

        # optional: share one attention for both dirs via simple dot-product
        # (MultiheadAttention would also work; this batched matmul is faster & simpler here)

    def _context_to_query(self, ctx_ids, dir_token):  # ctx_ids: [B, L, S]
        # embed & mean-pool over window
        E = self.emb(ctx_ids)                         # [B, L, S, d_txt]
        q = E.mean(dim=2)                             # [B, L, d_txt]
        # add learned direction bias (broadcast over B,L)
        q = q + dir_token
        q = self.txt_proj(q)                          # [B, L, d_vis]
        return self.q_norm(q)

    # Q: [B, L, D], F: [B, N, D]
    def _cross_attend(self, Q, F):
        # scaled dot-product attention: A = softmax(QK^T)
        K = self.kv_norm(F)
        V = K
        attn = torch.einsum('bld,bnd->bln', Q, K) / \
            (K.size(-1) ** 0.5)   # [B, L, N]
        A = attn.softmax(dim=-1)
        # [B, L, D]
        out = torch.einsum('bln,bnd->bld', A, V)
        return self.dropout(out)

    def forward(self, vis_tokens, left_ctx_ids, right_ctx_ids, tgt_ids, tgt_mask):
        """
        vis_tokens: [B, N, D]; left_ctx_ids/right_ctx_ids: [B, L, S]; tgt_ids: [B, L]
        tgt_mask: [B, L] with 1 for real labels, 0 for padding (no loss).
        Returns: dict(loss_sgm=...), logits_l, logits_r
        """
        Ql = self._context_to_query(
            left_ctx_ids,  self.dir_left)          # [B, L, D]
        Qr = self._context_to_query(
            right_ctx_ids, self.dir_right)         # [B, L, D]

        # [B, L, D]
        Fl = self._cross_attend(Ql, vis_tokens)
        # [B, L, D]
        Fr = self._cross_attend(Qr, vis_tokens)

        # [B, L, V]
        logits_l = self.classifier(Fl)
        logits_r = self.classifier(Fr)

        # CE over valid positions only (mask out pads)
        loss_l = F.cross_entropy(
            logits_l.view(-1, self.vocab_size), tgt_ids.view(-1), reduction='none'
        ).view_as(tgt_ids)
        loss_r = F.cross_entropy(
            logits_r.view(-1, self.vocab_size), tgt_ids.view(-1), reduction='none'
        ).view_as(tgt_ids)

        loss_masked = (loss_l + loss_r) * tgt_mask
        denom = torch.clamp(tgt_mask.sum(), min=1.0)
        loss_sgm = loss_masked.sum() / (2.0 * denom)

        return {'loss_sgm': loss_sgm, 'logits_l': logits_l, 'logits_r': logits_r}
