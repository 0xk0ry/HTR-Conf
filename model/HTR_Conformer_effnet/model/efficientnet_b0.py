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

