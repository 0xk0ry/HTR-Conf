import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tcm_vocab(converter, add_tokens=("<pad>", "<eos>", "<bos_left>", "<bos_right>")):
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
    ids = texts_to_ids(texts, stoi)
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
            l_start = max(0, i - S)
            l_ctx = seq[l_start:i]
            need = S - l_ctx.size(0)
            if need > 0:
                l_ctx = torch.cat(
                    [torch.tensor([stoi["<bos_left>"]] * need, device=device), l_ctx], dim=0)
            left[b, i] = l_ctx[-S:]

            r_end = min(L, i + 1 + S)
            r_ctx = seq[i+1:r_end]
            need = S - r_ctx.size(0)
            if need > 0:
                r_ctx = torch.cat([r_ctx, torch.tensor(
                    [stoi["<eos>"]] * need, device=device)], dim=0)
            right[b, i] = r_ctx[:S]

    return left, right, tgt, mask


class TCMHead(nn.Module):
    def __init__(self, d_vis, vocab_size_tcm, d_txt=256, sub_str_len=5, num_heads=8, p_drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size_tcm
        self.sub_str_len = sub_str_len
        self.emb = nn.Embedding(vocab_size_tcm, d_txt)

        self.dir_left = nn.Parameter(torch.randn(1, 1, d_txt))
        self.dir_right = nn.Parameter(torch.randn(1, 1, d_txt))

        self.ctx_conv = nn.Conv1d(d_txt, d_txt, kernel_size=3, padding=1)

        self.txt_proj = nn.Linear(d_txt, d_vis)
        self.q_norm = nn.LayerNorm(d_vis)
        self.kv_norm = nn.LayerNorm(d_vis)
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(d_vis, vocab_size_tcm)

    def _context_to_query(self, ctx_ids, dir_token):
        E = self.emb(ctx_ids)
        B, L, S, D = E.shape
        x = E.view(B*L, S, D).transpose(1, 2)
        x = self.ctx_conv(x)
        x = x.mean(dim=-1)
        x = x.view(B, L, D)

        x = x + dir_token
        x = self.txt_proj(x)
        return self.q_norm(x)

    def _cross_attend(self, Q, F):
        K = self.kv_norm(F)
        V = K
        attn = torch.einsum('bld,bnd->bln', Q, K) / \
            (K.size(-1) ** 0.5)
        A = attn.softmax(dim=-1)
        out = torch.einsum('bln,bnd->bld', A, V)
        return self.dropout(out)

    def forward(self,
                vis_tokens,
                left_ctx_ids,
                right_ctx_ids,
                tgt_ids,
                tgt_mask,
                focus_mask=None):
        Ql = self._context_to_query(left_ctx_ids,  self.dir_left)
        Qr = self._context_to_query(right_ctx_ids, self.dir_right)

        Fl = self._cross_attend(Ql, vis_tokens)
        Fr = self._cross_attend(Qr, vis_tokens)

        logits_l = self.classifier(Fl)
        logits_r = self.classifier(Fr)

        loss_l = F.cross_entropy(
            logits_l.view(-1, self.vocab_size),
            tgt_ids.view(-1),
            reduction='none'
        ).view_as(tgt_ids)
        loss_r = F.cross_entropy(
            logits_r.view(-1, self.vocab_size),
            tgt_ids.view(-1),
            reduction='none'
        ).view_as(tgt_ids)

        if focus_mask is not None:
            weights = tgt_mask * (1.0 + focus_mask)
        else:
            weights = tgt_mask

        loss_masked = (loss_l + loss_r) * weights
        denom = torch.clamp(weights.sum(), min=1.0)
        loss_tcm = loss_masked.sum() / (2.0 * denom)

        return {'loss_tcm': loss_tcm,
                'logits_l': logits_l,
                'logits_r': logits_r}
