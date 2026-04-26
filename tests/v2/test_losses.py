# tests/v2/test_losses.py
from __future__ import annotations
import math
import torch
from src.v2.losses import (
    masked_huber,
    bce_with_pos_weight,
    atwater_loss,
    kcal_consistency_loss,
    UncertaintyWeighter,
)


def test_masked_huber_zero_when_pred_eq_target_at_mask():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 5.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_masked_huber_ignores_zero_mask():
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 1000.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_bce_with_pos_weight_shape():
    logits = torch.randn(4, 555)
    target = torch.zeros(4, 555); target[:, :3] = 1
    pos_weight = torch.full((555,), 50.0)
    loss = bce_with_pos_weight(logits, target, pos_weight)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_atwater_zero_when_consistent():
    # kcal = 9*fat + 4*carb + 4*protein
    fat = torch.tensor([10.0]); carb = torch.tensor([20.0]); protein = torch.tensor([5.0])
    kcal = 9 * fat + 4 * carb + 4 * protein
    loss = atwater_loss(kcal, fat, carb, protein)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_kcal_consistency_zero_when_equal():
    a = torch.tensor([300.0, 200.0])
    b = torch.tensor([300.0, 200.0])
    loss = kcal_consistency_loss(a, b)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_uncertainty_weighter_init_zero():
    w = UncertaintyWeighter(["a", "b", "c"])
    losses = {"a": torch.tensor(2.0), "b": torch.tensor(4.0), "c": torch.tensor(6.0)}
    total, parts = w(losses)
    # With s_t=0, total = 0.5*L_a + 0.5*L_b + 0.5*L_c + 0
    expected = 0.5 * (2 + 4 + 6)
    assert torch.isclose(total, torch.tensor(expected))


def test_uncertainty_weighter_clamp():
    w = UncertaintyWeighter(["a"], s_floor=-2.0)
    with torch.no_grad():
        w.log_var["a"].fill_(-100.0)
    losses = {"a": torch.tensor(1.0)}
    total, _ = w(losses)
    assert torch.isfinite(total)


def test_masked_huber_correct_average_excludes_masked_positions():
    """Masked positions must NOT contaminate the average even when their loss would be huge."""
    pred   = torch.tensor([[2.0, 999.0, 3.0]])    # mid position masked, large fake error
    target = torch.tensor([[1.0,   5.0, 2.0]])
    mask   = torch.tensor([[1.0,   0.0, 1.0]])
    # delta=1.0; pos 0: |1|=1 → quad: 0.5 ; pos 2: |1|=1 → quad: 0.5
    # average over 2 unmasked positions: (0.5+0.5)/2 = 0.5
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.5))


def test_masked_huber_handles_nan_at_masked_positions():
    """NaN at masked positions must not propagate to the loss."""
    pred   = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, float("nan")]])
    mask   = torch.tensor([[1.0, 0.0]])
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isfinite(loss).item()
    assert torch.isclose(loss, torch.tensor(0.0))
