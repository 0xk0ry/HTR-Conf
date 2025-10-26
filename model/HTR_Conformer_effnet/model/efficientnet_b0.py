import torch
import torch.nn as nn


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 feature extractor that outputs a [B, C=embed_dim, W', H'] feature map.

    - Uses timm's efficientnet_b0 with features_only=True to grab a selected stage.
      By default, we take the stride-4 stage (index=1) so that for input W=512 we get W'=128.
    - Accepts 1-channel (grayscale) input images (in_chans=1).
    - Projects the native 1280 channels to `nb_feat` via a 1x1 conv + BN + activation.

    Args:
        nb_feat (int): Output channel dimension (should match the model's embed_dim).
        pretrained (bool): Whether to load ImageNet-pretrained weights for EfficientNet-B0.
        stage_index (int): Which feature stage to use from timm features_only output.
            Common choices (for B0):
              0 -> stride 2, 1 -> stride 4, 2 -> stride 8, 3 -> stride 16, 4 -> stride 32.
            Default 1 (stride 4) yields W' = W/4 (e.g., 512 -> 128).
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

        # Grab only the final feature map (index 4) from EfficientNet-B0
        # Configure to accept grayscale input directly
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=[int(stage_index)],
            in_chans=1,
        )

        # timm FeatureInfo helper exposes native channel dims per selected index
        in_channels = self.backbone.feature_info.channels()[-1]

        # Project to the requested embedding dimension
        self.proj = nn.Conv2d(in_channels, nb_feat, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(nb_feat, eps=1e-5)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]  (standard CHW)
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

    def __init__(self, embed_dim: int = 512, se_ratio: float = 0.0, expand: int = 4):
        super().__init__()

        def stage(in_ch, out_ch, num_blocks, stride, k=3):
            blocks = []
            # first block with given stride
            blocks.append(MBConv(in_ch, out_ch, stride=stride, expand=expand, kernel_size=k, se_ratio=se_ratio))
            # remaining with stride 1
            for _ in range(num_blocks - 1):
                blocks.append(MBConv(out_ch, out_ch, stride=(1, 1), expand=expand, kernel_size=k, se_ratio=se_ratio))
            return nn.Sequential(*blocks)

        ch = 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, stride=(2, 1), padding=1, bias=False),  # 64x512 -> 32x512
            nn.BatchNorm2d(ch, eps=1e-5),
            nn.SiLU(inplace=True),
        )

        self.stage1 = stage(ch, 64, num_blocks=2, stride=(2, 1))    # 32x512 -> 16x512
        self.stage2 = stage(64, 96, num_blocks=2, stride=(2, 1))    # 16x512 -> 8x512
        self.stage3 = stage(96, 128, num_blocks=2, stride=(2, 2))   # 8x512 -> 4x256
        self.stage4 = stage(128, 192, num_blocks=2, stride=(2, 1))  # 4x256 -> 2x256
        self.stage5 = stage(192, 256, num_blocks=2, stride=(2, 2))  # 2x256 -> 1x128

        self.head = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim, eps=1e-5),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.head(x)  # [B, embed_dim, 1, 128]
        return x

