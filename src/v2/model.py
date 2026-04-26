"""RGB-D dual-stream multi-task model for Nutrition5k.

Spec: §3.

Architecture:
    rgb     -> ConvNeXt-Base   (ImageNet-22k pretrained)  -> feat_rgb (1024)
    depth   -> ConvNeXt-Tiny   (channel-mean adapted to 2ch) -> feat_d (768)
    concat  -> MLP(1792 -> 512) -> z (512)
    z       -> head_scalar  (Linear 512 -> 5)
    z       -> head_ingr    (Linear 512 -> n_ingredients)
    z       -> head_mass    (Linear 512 -> n_ingredients)

The depth encoder's first conv is reinitialized with channel-mean of the
RGB-pretrained weights, then duplicated to 2 in_channels (depth, valid_mask).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import convnext_base, convnext_tiny, ConvNeXt_Base_Weights, ConvNeXt_Tiny_Weights


def _adapt_first_conv_to_2ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Replace a 3-channel first conv with a 2-channel version using RGB-channel-mean init."""
    w = conv.weight.detach()  # (out, 3, kh, kw)
    mean_w = w.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
    new_w = mean_w.repeat(1, 2, 1, 1)     # (out, 2, kh, kw)
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(new_w)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class NutritionRGBDModel(nn.Module):
    def __init__(self, n_ingredients: int, dropout: float = 0.1, hidden_dim: int = 512):
        super().__init__()
        # RGB encoder — ConvNeXt-Base
        rgb_w = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.rgb_enc = convnext_base(weights=rgb_w)
        rgb_feat_dim = 1024
        self.rgb_enc.classifier = nn.Identity()  # leave global pool's flatten in classifier
        # ConvNeXt's classifier = LayerNorm2d -> Flatten -> Linear; we replaced with Identity.
        # We use features-level output after avgpool-flatten manually below.
        self.rgb_avgpool = nn.AdaptiveAvgPool2d(1)
        self.rgb_norm = nn.LayerNorm(rgb_feat_dim)

        # Depth encoder — ConvNeXt-Tiny, adapted first conv
        d_w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.d_enc = convnext_tiny(weights=d_w)
        # Patchify conv lives at features[0][0] for ConvNeXt
        old_conv = self.d_enc.features[0][0]
        self.d_enc.features[0][0] = _adapt_first_conv_to_2ch(old_conv)
        self.d_enc.classifier = nn.Identity()
        d_feat_dim = 768
        self.d_avgpool = nn.AdaptiveAvgPool2d(1)
        self.d_norm = nn.LayerNorm(d_feat_dim)

        # Fusion MLP
        in_dim = rgb_feat_dim + d_feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Heads
        self.head_scalar = nn.Linear(hidden_dim, 5)
        self.head_ingr = nn.Linear(hidden_dim, n_ingredients)
        self.head_mass = nn.Linear(hidden_dim, n_ingredients)

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.rgb_enc.features(rgb)
        x = self.rgb_avgpool(x).flatten(1)
        return self.rgb_norm(x)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.d_enc.features(depth)
        x = self.d_avgpool(x).flatten(1)
        return self.d_norm(x)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, *, use_depth: bool = True):
        feat_rgb = self.encode_rgb(rgb)
        if use_depth:
            feat_d = self.encode_depth(depth)
        else:
            feat_d = torch.zeros(rgb.size(0), 768, device=rgb.device, dtype=feat_rgb.dtype)
        z = self.fuse(torch.cat([feat_rgb, feat_d], dim=1))
        return {
            "scalar": self.head_scalar(z),
            "ingr_logits": self.head_ingr(z),
            "ingr_mass": self.head_mass(z),
        }

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        backbone_params = list(self.rgb_enc.parameters()) + list(self.d_enc.parameters())
        head_params = list(self.fuse.parameters()) + \
                      list(self.head_scalar.parameters()) + \
                      list(self.head_ingr.parameters()) + \
                      list(self.head_mass.parameters()) + \
                      list(self.rgb_norm.parameters()) + list(self.d_norm.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
