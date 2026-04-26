# tests/v2/test_model.py
from __future__ import annotations
import torch
from src.v2.model import NutritionRGBDModel


def test_forward_shapes():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 2, 224, 224)
    out = m(rgb, depth)
    assert out["scalar"].shape == (2, 5)
    assert out["ingr_logits"].shape == (2, 555)
    assert out["ingr_mass"].shape == (2, 555)


def test_no_depth_zeros_d_branch():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(2, 3, 224, 224)
    depth_zero = torch.zeros(2, 2, 224, 224)
    depth_random = torch.randn(2, 2, 224, 224)
    out_zero = m(rgb, depth_zero, use_depth=False)
    out_rand = m(rgb, depth_random, use_depth=False)
    # Output must be identical when use_depth=False
    for k in out_zero:
        assert torch.allclose(out_zero[k], out_rand[k])


def test_param_count_roughly_120m():
    m = NutritionRGBDModel(n_ingredients=555)
    n = sum(p.numel() for p in m.parameters())
    # Target ~120M ± 30M (ConvNeXt-Base ~89M + ConvNeXt-Tiny ~28M + heads)
    assert 90_000_000 < n < 200_000_000, f"got {n} params"


def test_backward_no_nan():
    m = NutritionRGBDModel(n_ingredients=555).train()
    rgb = torch.randn(2, 3, 224, 224, requires_grad=False)
    depth = torch.randn(2, 2, 224, 224, requires_grad=False)
    out = m(rgb, depth)
    loss = sum(o.mean() for o in out.values())
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
