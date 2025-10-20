import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import random
import re
import warnings


class RelativePositionBias1D(nn.Module):
    """T5-style learnable 1D relative position bias shared or per head.

    Produces a bias tensor of shape [1, num_heads, N, N] given current N.
    max_rel_positions controls the clipping window for relative distances.
    """

    def __init__(self, num_heads: int, max_rel_positions: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel_positions = max(1, int(max_rel_positions))
        self.bias = nn.Embedding(2 * self.max_rel_positions - 1, num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, N: int) -> torch.Tensor:
        device = self.bias.weight.device
        # relative distance matrix in range [-(N-1), (N-1)]
        coords = torch.arange(N, device=device)
        rel = coords[:, None] - coords[None, :]  # [N, N]
        # clip to window and shift to [0, 2*max-2]
        rel = rel.clamp(-self.max_rel_positions + 1, self.max_rel_positions - 1)
        rel = rel + (self.max_rel_positions - 1)
        # lookup and reshape to [1, H, N, N]
        bias = self.bias(rel)  # [N, N, H]
        return bias.permute(2, 0, 1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # num_patches argument is repurposed here as the max relative positions window
        max_rel_positions = max(1, int(num_patches)) if num_patches is not None else 1024
        self.rel_pos_bias = RelativePositionBias1D(num_heads=num_heads, max_rel_positions=max_rel_positions)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # add relative position bias before softmax
        attn = attn + self.rel_pos_bias(N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """Original ViT style block (retained for compatibility)."""

    def __init__(
        self,
        dim,
        num_heads,
        num_patches,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.attn = Attention(
            dim,
            num_patches,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class FeedForward(nn.Module):
    """Position-wise Feed Forward with configurable expansion (used for Conformer macaron)."""

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=nn.SiLU):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lin2(self.act(self.lin1(x))))


class ConvModule(nn.Module):
    """
    Unified-activation conv block (Swish everywhere), no GLU.
    pw (D -> eD) -> Swish -> dw(k) -> GN -> Swish -> pw (eD -> D)
    input: (B, N, D) ; internal convs on (B, C, N)
    """
    def __init__(self, dim, kernel_size=3, dropout=0.1, drop_path=0.0,
                 expansion=1.0, pre_norm=False, activation=nn.SiLU):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim) if pre_norm else None
        hidden = int(round(dim * expansion))

        self.pw1 = nn.Conv1d(dim, hidden, kernel_size=1, bias=True)
        self.act1 = activation()

        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=hidden, bias=True)
        self.gn = nn.GroupNorm(1, hidden, eps=1e-5)
        self.act2 = activation()

        self.pw2 = nn.Conv1d(hidden, dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        z = x.transpose(1, 2)            # (B, D, N)
        z = self.pw1(z)
        z = self.act1(z)
        z = self.dw(z)
        z = self.gn(z)
        z = self.act2(z)
        z = self.pw2(z)
        z = self.dropout(z).transpose(1, 2)
        return self.drop_path(z)



class Downsample1D(nn.Module):
    """
    Depthwise 1D stride-2 low-pass on the sequence length (N).
    Input:  [B, N, D]  ->  Output: [B, N/2, D]
    """

    def __init__(self, dim, kernel_size=3, stride=2, lowpass_init=True):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size//2,
                            groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        if lowpass_init:
            with torch.no_grad():
                w = torch.zeros_like(self.dw.weight)
                w[:, 0, :] = 1.0 / kernel_size  # simple box filter per channel
                self.dw.weight.copy_(w)

    def forward(self, x):                      # x: [B, N, D]
        x = x.transpose(1, 2)                  # [B, D, N]
        x = self.pw(self.dw(x))                # [B, D, N/2]
        return x.transpose(1, 2)               # [B, N/2, D]


class Upsample1D(nn.Module):
    """
    Lightweight upsampler: nearest/linear + 1x1 mix to smooth.
    Input:  [B, N_low, D]  ->  Output: [B, N_high, D]
    """

    def __init__(self, dim, mode: str = 'nearest'):
        super().__init__()
        assert mode in ('nearest', 'linear'), "Upsample1D mode must be 'nearest' or 'linear'"
        self.mode = mode
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, target_len: int):
        x = x.transpose(1, 2)                            # [B, D, N_low]
        if self.mode == 'nearest':
            x = F.interpolate(x, size=target_len, mode='nearest')
        else:
            # 1D linear interpolation
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x = self.proj(x)
        return x.transpose(1, 2)                         # [B, N_high, D]

class SqueezeformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 num_patches,
                 mlp_ratio=4.0,
                 ff_dropout=0.1,
                 attn_dropout=0.0,
                 conv_dropout=0.0,
                 conv_kernel_size=3,
                 conv_expansion=1.0,          # NEW
                 norm_layer=nn.LayerNorm,
                 drop_path=0.0,
                 layerscale_init=1e-5):       # NEW
        super().__init__()

        ff_hidden = int(dim * mlp_ratio)

        self.attn = Attention(dim, num_patches, num_heads=num_heads,
                              qkv_bias=True, attn_drop=attn_dropout, proj_drop=ff_dropout)

        self.ffn1 = FeedForward(dim, ff_hidden, dropout=ff_dropout, activation=nn.SiLU)
        self.conv = ConvModule(dim, kernel_size=conv_kernel_size,
                               dropout=conv_dropout, drop_path=0.0,
                               expansion=conv_expansion, pre_norm=False, activation=nn.SiLU)
        self.ffn2 = FeedForward(dim, ff_hidden, dropout=ff_dropout, activation=nn.SiLU)

        # post-LNs
        self.postln_attn = norm_layer(dim, elementwise_affine=True)
        self.postln_ffn1 = norm_layer(dim, elementwise_affine=True)
        self.postln_conv = norm_layer(dim, elementwise_affine=True)
        self.postln_ffn2 = norm_layer(dim, elementwise_affine=True)

        # stochastic depth
        self.dp_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_ffn1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_conv = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_ffn2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # LayerScale on each residual branch (tiny init)
        self.ls_attn = LayerScale(dim, init_values=layerscale_init)
        self.ls_ffn1 = LayerScale(dim, init_values=layerscale_init)
        self.ls_conv = LayerScale(dim, init_values=layerscale_init)
        self.ls_ffn2 = LayerScale(dim, init_values=layerscale_init)

    def forward(self, x):
        # 1) MHA (residual -> PostLN)
        x = self.postln_attn(x + self.ls_attn(self.dp_attn(self.attn(x))))

        # 2) 1/2 FFN (macaron) (residual -> PostLN)
        x = self.postln_ffn1(x + self.ls_ffn1(0.5 * self.dp_ffn1(self.ffn1(x))))

        # 3) Conv (residual -> PostLN)
        x = self.postln_conv(x + self.ls_conv(self.dp_conv(self.conv(x))))

        # 4) 1/2 FFN (residual -> PostLN)
        x = self.postln_ffn2(x + self.ls_ffn2(0.5 * self.dp_ffn2(self.ffn2(x))))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)


class MaskedAutoencoderViT(nn.Module):
    """HTR encoder with selectable backend (ViT or Conformer)."""

    def __init__(
        self,
        nb_cls=80,
        img_size=[512, 32],
        patch_size=[8, 32],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        temporal_unet: bool = True,
        down_after: int = 2,    # downsample after this many blocks
        up_after: int = 4,    # upsample after this many blocks
        ds_kernel: int = 3,
        max_seq_len: int = 1024,
        # upsampler config
        upsample_mode: str = 'nearest',   # 'nearest' or 'linear'
    ):
        super().__init__()

        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.embed_dim = embed_dim
        # Use a configurable max sequence length for relative position window
        self.max_rel_pos = int(max_seq_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(embed_dim, num_heads, self.max_rel_pos,
                            mlp_ratio=mlp_ratio,
                            ff_dropout=dropout, attn_dropout=dropout,
                            conv_dropout=dropout, conv_kernel_size=conv_kernel_size,
                            conv_expansion=1.0,                 # Swish@1.0 to start
                            norm_layer=norm_layer, drop_path=dpr[i],
                            layerscale_init=1e-5)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.temporal_unet = temporal_unet
        self.down_after = down_after
        self.up_after = up_after
        if self.temporal_unet:
            self.down1 = Downsample1D(embed_dim, kernel_size=ds_kernel)
            self.up1 = Upsample1D(embed_dim, mode=upsample_mode)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # no absolute pos embed initialization needed for relative PE setup
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # --- Backward compatibility for loading older checkpoints ---
    def _upgrade_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Map legacy parameter names to the current module structure, in-place.
        - blocks.*.conv_layer_norm.* -> blocks.*.conv_module.layer_norm.*
        """
        if not isinstance(state_dict, dict):
            return state_dict

        remapped = {}
        to_delete = []

        # 0) Strip an optional leading 'module.' from DataParallel checkpoints
        has_module_prefix = all(k.startswith("module.")
                                for k in state_dict.keys())
        if has_module_prefix:
            for k, v in list(state_dict.items()):
                new_k = k[len("module."):]
                if new_k not in state_dict:
                    remapped[new_k] = v
                to_delete.append(k)
            # apply moving keys
            for k in to_delete:
                state_dict.pop(k, None)
            state_dict.update(remapped)
            remapped.clear()
            to_delete.clear()

        # Direct string replace should be safe and faster than regex for this case
        conv_legacy_patterns = [
            (".conv_layer_norm.", ".conv_module.layer_norm."),
            (".conv_pointwise_conv1.", ".conv_module.pointwise_conv1."),
            (".conv_depthwise_conv.", ".conv_module.depthwise_conv."),
            (".conv_norm.", ".conv_module.norm."),
            (".conv_pointwise_conv2.", ".conv_module.pointwise_conv2."),
        ]

        for k, v in list(state_dict.items()):
            for old, new in conv_legacy_patterns:
                if old in k and new not in k:
                    new_k = k.replace(old, new)
                    # Only set if target key not already present
                    if new_k not in state_dict:
                        remapped[new_k] = v
                    to_delete.append(k)
                    break

        if remapped or to_delete:
            # apply removals then additions to avoid key overlap issues
            for k in to_delete:
                state_dict.pop(k, None)
            state_dict.update(remapped)

        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remap legacy keys before delegating to PyTorch loader
        upgraded = self._upgrade_state_dict_keys(dict(state_dict))
        # Drop absolute positional embedding if present in older checkpoints
        if 'pos_embed' in upgraded:
            upgraded.pop('pos_embed')
        # Tolerate presence of unused absolute-PE shaped keys inside nested modules
        pruned_keys = [k for k in upgraded.keys() if k.endswith('.pos_embed')]
        for k in pruned_keys:
            upgraded.pop(k, None)
        try:
            return super().load_state_dict(upgraded, strict=strict)
        except RuntimeError as e:
            warnings.warn(
                f"Strict checkpoint load failed ({e}); retrying with strict=False to ignore positional/relative bias/head mismatches.")
            return super().load_state_dict(upgraded, strict=False)

    # ---- MMS helpers ----
    # ---------------------------
    # 1-D Multiple Masking
    # ---------------------------

    def _mask_random_1d(self, B: int, L: int, ratio: float, device) -> torch.Tensor:
        """Random token masking on 1-D sequence. Returns bool [B, L], True = masked."""
        if ratio <= 0.0 or ratio > 1.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        num = int(round(ratio * L))
        if num <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        noise = torch.rand(B, L, device=device)
        idx = noise.argsort(dim=1)[:, :num]         # per-sample masked indices
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def _mask_block_1d(self, B: int, L: int, ratio: float, device,
                       min_block: int = 2) -> torch.Tensor:
        """
        Blockwise masking in 1-D (contiguous segments), no spacing constraints.
        Returns bool [B, L], True = masked.
        """
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            covered = int(mask[b].sum().item())
            # cap iterations to avoid infinite loops on tiny targets
            # More reasonable upper bound
            max_iterations = min(10000, target * 3)
            for iteration in range(max_iterations):
                if covered >= target:
                    break
                # choose a block length
                remain = max(1, target - covered)
                blk = random.randint(min_block, max(min_block, min(remain, L)))
                start = random.randint(0, max(0, L - blk))
                seg = mask[b, start:start+blk]
                prev = int(seg.sum().item())
                seg[:] = True
                covered += int(seg.sum().item()) - prev

                # Early exit if we're not making progress
                if iteration > 100 and covered < target * 0.1:
                    break
        return mask

    def _mask_span_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
        """
        Span masking in 1-D (YOUR OLD SEMANTICS, but robust):
        - place contiguous spans of random length s ∈ [1, max_span]
        - enforce an Algorithm-1-like spacing policy via k depending on ratio
        - continue until ~ratio*L tokens are covered
        Returns bool [B, L], True = masked.
        """
        if ratio <= 0.0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        L = int(L)
        max_span = int(max(1, min(max_span, L)))
        target = int(round(ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        # spacing policy similar to Alg.1 (adapted to 1-D)
        def spacing_for(R):
            if R <= 0.4:
                # use k = span length (separates spans when ratio small)
                return None
            elif R <= 0.7:
                return 1
            else:
                return 0
        fixed_k = spacing_for(ratio)

        for b in range(B):
            used = torch.zeros(L, dtype=torch.bool, device=device)
            covered = int(used.sum().item())
            for _ in range(10000):
                if covered >= target:
                    break
                s = random.randint(1, max_span)
                if s > L:
                    s = L
                l = random.randint(0, L - s)
                r = l + s - 1
                k = s if fixed_k is None else fixed_k
                # check spacing neighborhood
                left_ok = (l - k) < 0 or not used[max(0, l - k):l].any()
                right_ok = (
                    r + 1) >= L or not used[r+1:min(L, r + 1 + k)].any()
                if left_ok and right_ok:
                    used[l:r+1] = True
                    covered = int(used.sum().item())
            mask[b] = used
        return mask

    def _mask_span_old_1d(self, B: int, L: int, ratio: float, max_span: int, device) -> torch.Tensor:
        if ratio <= 0.0 or max_span <= 0 or L <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        span_total = int(L * ratio)
        num_spans = span_total // max(1, max_span)
        if num_spans <= 0:
            return torch.zeros(B, L, dtype=torch.bool, device=device)

        s = min(max_span, L)  # fixed length (old behavior)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for _ in range(num_spans):
            start = torch.randint(0, L - s + 1, (1,), device=device).item()
            mask[:, start:start + s] = True    # same start for the whole batch

        return mask

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    # inside MaskedAutoencoderViT.forward_features(...)
    def forward_features(self, x, use_masking=False,
                         mask_mode="span_old",   # "random" | "block" | "span_old"
                         mask_ratio=0.5, max_span_length=8,
                         ratios=None, block_params=None):
        # [B,C,W,H] -> your [B,N,D] after reshape
        x = self.patch_embed(x)
        B, C, W, H = x.shape
        # Ensure dimensions are correct before reshaping
        assert C == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {C}"
        x = x.view(B, C, -1).permute(0, 2, 1)         # [B,N,D]

        if use_masking:
            if mask_mode == "random":
                keep = (~self._mask_random_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            elif mask_mode == "block":
                keep = (~self._mask_block_1d(B, x.size(1),
                        mask_ratio, x.device)).float().unsqueeze(-1)
            else:
                keep = (~self._mask_span_old_1d(B, x.size(1), mask_ratio,
                        max_span_length, x.device)).float().unsqueeze(-1)

        # Relative positional encoding is applied inside Attention; no absolute PE added here

        skip_hi = None
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)

            # ---- Downsample after 'down_after' blocks
            if self.temporal_unet and i == self.down_after:
                skip_hi = x                                # keep high-res skip
                # pad one token if odd length to keep /2 exact
                if (x.size(1) % 2) == 1:
                    x = torch.cat([x, x[:, -1:, :]], dim=1)
                x = self.down1(x)                          # [B, N/2, D]

            # ---- Upsample & fuse after 'up_after' blocks
            if self.temporal_unet and i == self.up_after:
                assert skip_hi is not None, "Upsample requires a stored skip."
                # back to high-res
                x = self.up1(x, target_len=skip_hi.size(1))
                # fuse by simple addition
                x = x + skip_hi

        return self.norm(x)

    def forward(self, x, use_masking=False, return_features=False, mask_mode="span_old", mask_ratio=None, max_span_length=None):
        feats = self.forward_features(
            # [B, N, D]
            x, use_masking=use_masking, mask_mode=mask_mode, mask_ratio=mask_ratio, max_span_length=max_span_length)
        logits = self.head(feats)               # [B, N, nb_cls]  → CTC
        if return_features:
            return logits, feats
        return logits


def create_model(nb_cls, img_size, mlp_ratio=4, **kwargs):
    model = MaskedAutoencoderViT(
        nb_cls,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_kernel_size=7,
        temporal_unet=True,      # enable #1
        down_after=3,            # blocks 1-2 high-res
        up_after=7,              # blocks 3-4 low-res
        ds_kernel=3,
        max_seq_len=128,
        upsample_mode='nearest',
        **kwargs,
    )
    return model