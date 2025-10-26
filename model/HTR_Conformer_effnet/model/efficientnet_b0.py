import torch
import torch.nn as nn


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 feature extractor that outputs [B, embed_dim, 1, W'] using a selected feature stage.

    - Uses timm's efficientnet_b0 with features_only=True to grab a stage.
    - Accepts 1-channel (grayscale) input images (in_chans=1).
    - Projects native channels to `nb_feat` via a 1x1 conv + BN + activation.

    Args:
        nb_feat (int): Output channel dimension (should match the model's embed_dim).
        pretrained (bool): Whether to load ImageNet-pretrained weights for EfficientNet-B0.
        stage_index (int): Feature stage index (0: s=2, 1: s=4, 2: s=8, 3: s=16, 4: s=32). Default 1.
    """

    def __init__(self, nb_feat: int = 512, pretrained: bool = False, stage_index: int = 1):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for EfficientNetB0 backbone. Please install timm >= 0.6"
            ) from e

        assert 0 <= int(stage_index) <= 4, "stage_index must be in [0,4] for EfficientNet-B0"

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=[int(stage_index)],
            in_chans=1,
        )
        in_channels = self.backbone.feature_info.channels()[-1]

        self.proj = nn.Conv2d(in_channels, nb_feat, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(nb_feat, eps=1e-5)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[0]                  # [B, C_in, H', W']
        feats = self.proj(feats)
        feats = self.bn(feats)
        feats = self.act(feats)
        feats = feats.mean(dim=2, keepdim=True)      # collapse H' -> 1  => [B, C, 1, W']
        return feats

# ---- Custom gradual downsampler to 1 x 128 ---------------------------------
class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.0):
        super().__init__()
        self.enabled = se_ratio and se_ratio > 0.0
        if self.enabled:
            hidden = max(1, int(ch * se_ratio))
            self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)
            self.act = nn.SiLU(inplace=True)
            self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)
            self.gate = nn.Sigmoid()

    def forward(self, x):
        if not self.enabled:
            return x
        s = x.mean((2, 3), keepdim=True)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)


class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1), expand: int = 4,
                 kernel_size: int = 3, se_ratio: float = 0.0):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == (1, 1)) and (in_ch == out_ch)

        self.has_expand = expand != 1
        if self.has_expand:
            self.pw_expand = nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid, eps=1e-5),
                nn.SiLU(inplace=True),
            )
        else:
            self.pw_expand = nn.Identity()

        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid, bias=False),
            nn.BatchNorm2d(mid, eps=1e-5),
            nn.SiLU(inplace=True),
        )

        self.se = SqueezeExcite(mid, se_ratio)

        self.pw_project = nn.Sequential(
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5),
        )

    def forward(self, x):
        out = self.pw_expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.pw_project(out)
        if self.use_res:
            out = out + x
        return out


class EfficientNet1x128(nn.Module):
    """
    Custom EfficientNet-like backbone that downsamples 64x512 -> 1x128 gradually.

    - Uses MBConv-style blocks with anisotropic strides to control height/width separately.
    - Output is [B, embed_dim, 1, 128] for input [B, 1, 64, 512].
    - No timm dependency.

    Args:
        embed_dim (int): Output channel dimension.
        se_ratio (float): Squeeze-Excite ratio (0.0 disables SE).
        expand (int): Expansion ratio for MBConv mid channels.
    """

    def __init__(self,
                 embed_dim: int = 512,
                 se_ratio: float = 0.25,
                 expand: int = 4,
                 norm: str = 'bn',            # 'bn' or 'gn'
                 drop_path_rate: float = 0.1):
        super().__init__()

        # --- Norm factory ---
        def norm2d(ch: int):
            if norm == 'gn':
                # choose a valid group count up to 32 that divides ch
                for g in [32, 16, 8, 4, 2, 1]:
                    if ch % g == 0:
                        return nn.GroupNorm(g, ch, eps=1e-5)
                return nn.GroupNorm(1, ch, eps=1e-5)
            else:
                return nn.BatchNorm2d(ch, eps=1e-5, track_running_stats=True)

        # --- DropPath ---
        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

        # --- Blocks ---
        class FusedMBConv(nn.Module):
            def __init__(self, in_ch, out_ch, stride=(1, 1), k=3, dp=0.0):
                super().__init__()
                self.use_res = (stride == (1, 1)) and (in_ch == out_ch)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k//2, bias=False)
                self.bn1 = norm2d(out_ch)
                self.act1 = nn.SiLU(inplace=True)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
                self.bn2 = norm2d(out_ch)
                self.dp = DropPath(dp)

            def forward(self, x):
                out = self.act1(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.use_res:
                    out = x + self.dp(out)
                return nn.functional.silu(out, inplace=True)

        class MBConvAA(nn.Module):
            def __init__(self, in_ch, out_ch, stride=(1, 1), expand=4, k=3, se_ratio=0.25, dp=0.0,
                         anti_alias: bool = False, aa_kind: str = 'avg'):
                super().__init__()
                self.pre_down = None
                # Anti-alias pre-pool for stride (2,2)
                if anti_alias and stride == (2, 2):
                    if aa_kind == 'avg':
                        self.pre_down = nn.AvgPool2d(kernel_size=2, stride=2)
                    else:
                        # simple blur pool using 1D separable [1 2 1] filter
                        kernel = torch.tensor([1., 2., 1.], dtype=torch.float32)
                        filt = (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0)
                        filt = filt / filt.sum()
                        self.register_buffer('blur', filt)
                        self.pre_down = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1,
                                                  groups=in_ch, bias=False)
                        with torch.no_grad():
                            self.pre_down.weight.copy_(self.blur.repeat(in_ch, 1, 1, 1))
                    stride = (1, 1)

                mid = in_ch * expand
                self.use_res = (stride == (1, 1)) and (in_ch == out_ch)

                self.pw_expand = nn.Sequential(
                    nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
                    norm2d(mid),
                    nn.SiLU(inplace=True),
                )
                self.dw = nn.Sequential(
                    nn.Conv2d(mid, mid, kernel_size=k, stride=stride, padding=k//2, groups=mid, bias=False),
                    norm2d(mid),
                    nn.SiLU(inplace=True),
                )
                self.se = SqueezeExcite(mid, se_ratio)
                self.pw_project = nn.Sequential(
                    nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
                    norm2d(out_ch),
                )
                self.dp = DropPath(dp)

            def forward(self, x):
                if self.pre_down is not None:
                    x = self.pre_down(x)
                out = self.pw_expand(x)
                out = self.dw(out)
                out = self.se(out)
                out = self.pw_project(out)
                if self.use_res:
                    out = x + self.dp(out)
                return nn.functional.silu(out, inplace=True)

        # Schedule DropPath across blocks (linearly to drop_path_rate)
        # Total blocks: 2 per stage * 5 stages = 10
        total_blocks = 10
        dp_rates = [i * (drop_path_rate / (total_blocks - 1)) for i in range(total_blocks)] if total_blocks > 1 else [0.0]
        dp_iter = iter(dp_rates)

        def stage(in_ch, out_ch, num_blocks, stride, k=3, fused=False, aa=False, aa_kind='avg'):
            blocks = []
            # kernel size widening after H <= 8
            _k = k
            for bi in range(num_blocks):
                dp = next(dp_iter)
                block_stride = stride if bi == 0 else (1, 1)
                if fused:
                    blocks.append(FusedMBConv(in_ch if bi == 0 else out_ch, out_ch, stride=block_stride, k=_k, dp=dp))
                else:
                    blocks.append(MBConvAA(in_ch if bi == 0 else out_ch, out_ch, stride=block_stride,
                                           expand=expand, k=_k, se_ratio=se_ratio, dp=dp,
                                           anti_alias=aa, aa_kind=aa_kind))
                # widen kernel once we are in late stages (we'll control per-stage below)
            return nn.Sequential(*blocks)

        ch = 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, stride=(2, 1), padding=1, bias=False),  # 64x512 -> 32x512
            norm2d(ch),
            nn.SiLU(inplace=True),
        )

        # Stage configs
        # Stage1 (fused): 32x512 -> 16x512, use fused MBConv, kernel 3
        self.stage1 = stage(ch, 64, num_blocks=2, stride=(2, 1), k=3, fused=True)
        # Stage2 (classic MBConv): 16x512 -> 8x512 (keep modest width)
        self.stage2 = stage(64, 96, num_blocks=2, stride=(2, 1), k=3, fused=False)
        # Stage3: anti-aliased (2,2), keep width modest (128): 8x512 -> 4x256
        self.stage3 = stage(96, 128, num_blocks=2, stride=(2, 2), k=3, fused=False, aa=True, aa_kind='avg')
        # Stage4: smaller width than before (144 vs 192), widen DW kernel to 5: 4x256 -> 2x256
        self.stage4 = stage(128, 144, num_blocks=2, stride=(2, 1), k=5, fused=False)
        # Stage5: smaller width (176 vs 256), anti-aliased (2,2), kernel 5: 2x256 -> 1x128
        self.stage5 = stage(144, 176, num_blocks=2, stride=(2, 2), k=5, fused=False, aa=True, aa_kind='avg')

        # Cheap horizontal context before head: 1x5 depthwise conv
        self.horiz_context = nn.Sequential(
            nn.Conv2d(176, 176, kernel_size=(1, 5), padding=(0, 2), groups=176, bias=False),
            norm2d(176),
            nn.SiLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(176, embed_dim, kernel_size=1, bias=False),
            norm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.horiz_context(x)
        x = self.head(x)  # [B, embed_dim, 1, 128]
        return x

