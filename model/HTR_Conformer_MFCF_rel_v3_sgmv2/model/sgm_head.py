# sgm_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sgm_vocab_size_sgm(converter, add_tokens=("<pad>", "<eos>", "<bos_left>", "<bos_right>")):
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
    def __init__(self, d_vis, d_txt, vocab_size_sgm, band_width=16, rel_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_txt, d_vis, bias=False)
        self.k_proj = nn.Linear(d_vis, d_vis, bias=False)
        self.v_proj = nn.Linear(d_vis, d_vis, bias=False)
        self.vis2txt = nn.Linear(d_vis, d_txt, bias=False)  # low-rank link
        self.emb = nn.Embedding(vocab_size_sgm, d_txt)
        self.classifier = nn.Linear(d_txt, vocab_size_sgm, bias=False)
        self.classifier.weight = self.emb.weight  # weight tying
        self.band_width = band_width
        self.rel_bias = rel_bias
        # optional learned slope for signed bias
        self.dir_alpha = nn.Parameter(torch.tensor(0.2))
        self.ctx_dw = nn.Conv1d(d_txt, d_txt, kernel_size=3, padding=1, groups=d_txt, bias=False)
        self.ctx_pw = nn.Conv1d(d_txt, d_txt, kernel_size=1, bias=False)
        self.ctx_act = nn.SiLU()  # or GELU
        self.ctx_ln  = nn.LayerNorm(d_txt)
    @torch.no_grad()
    def _centers_linear(self, B, L, N, device):
        # fallback linear alignment: pos i -> round(i * (N-1)/(L-1))
        base = torch.linspace(0, N-1, L, device=device).long()
        return base[None, :].expand(B, L)

    def _band_mask(self, centers, N, width):
        B, L = centers.shape
        grid = torch.arange(N, device=centers.device)[None, None, :]  # [1,1,N]
        dist = (grid - centers[..., None]).abs()                      # [B,L,N]
        return (dist <= width)                                        # True keep

    def forward(self, vis_tokens, txt_ctx_ids, pos_centers=None, use_band=True):
        B, N, Dv = vis_tokens.shape
        E = self.emb(txt_ctx_ids)                          # [B,L,S,Dt]

        # --- context mixer to build Q ---
        B, L, S, Dt = E.shape
        x = E.view(B * L, S, Dt).transpose(1, 2)           # [B*L, Dt, S]
        y = self.ctx_pw(self.ctx_act(self.ctx_dw(x)))
        x = x + y
        q_txt = self.ctx_ln(x.mean(dim=2)).view(B, L, Dt)
        Q = self.q_proj(q_txt)                             # [B,L,Dv]

        # --- shared K/V ---
        K = self.k_proj(vis_tokens)                        # [B,N,Dv]
        V = self.v_proj(vis_tokens)                        # [B,N,Dv]

        # --- attention logits ---
        logits = torch.einsum('bld,bnd->bln', Q, K) * (Dv ** -0.5)

        # signed relative bias
        if self.rel_bias:
            centers = pos_centers if pos_centers is not None else self._centers_linear(B, L, N, Q.device)
            idx = torch.arange(N, device=Q.device)[None, None, :]
            signed = (centers[..., None] - idx).float()
            logits = logits + self.dir_alpha * (signed / (self.band_width + 1e-6))

        # banded mask
        if use_band:
            centers = pos_centers if pos_centers is not None else self._centers_linear(B, L, N, Q.device)
            keep = self._band_mask(centers, N, self.band_width)
            logits = logits.masked_fill(~keep, float('-inf'))

        A = logits.softmax(-1)                             # [B,L,N]
        F = self.dropout(torch.einsum('bln,bnd->bld', A, V))
        H = self.vis2txt(F)                                # [B,L,Dt]
        return self.classifier(H), H, A
