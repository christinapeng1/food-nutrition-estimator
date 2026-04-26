# src/v2/losses.py
"""Loss functions for the multi-task RGB-D Nutrition5k model.

Spec: §5.1, §5.2.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 delta: float = 1.0) -> torch.Tensor:
    """Huber loss applied only at mask=1 positions, averaged over masked count.

    Args:
        pred:   any shape (typically (B, V))
        target: same shape
        mask:   same shape, 0/1
        delta:  Huber transition point (in target units)
    """
    target_safe = torch.where(mask.bool(), target, torch.zeros_like(target))
    diff = (pred - target_safe) * mask
    abs_diff = diff.abs()
    quad = 0.5 * (abs_diff ** 2)
    lin = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff <= delta, quad, lin)
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def bce_with_pos_weight(logits: torch.Tensor, target: torch.Tensor,
                        pos_weight: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits, mean over (B, V), with per-class pos_weight."""
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="mean"
    )


def atwater_loss(kcal: torch.Tensor, fat: torch.Tensor,
                 carb: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
    """Soft physical regularizer: |kcal - (9·fat + 4·carb + 4·protein)|.

    All inputs in raw kcal/g units, shape (B,) or (B, 1).
    """
    derived = 9.0 * fat + 4.0 * carb + 4.0 * protein
    return F.smooth_l1_loss(kcal, derived, reduction="mean", beta=1.0)


def kcal_consistency_loss(direct: torch.Tensor, derived: torch.Tensor) -> torch.Tensor:
    """Couple direct head A kcal and derived (Σ mass × density) kcal, both raw kcal."""
    return F.smooth_l1_loss(direct, derived, reduction="mean", beta=1.0)


class UncertaintyWeighter(nn.Module):
    """Multi-task uncertainty weighting (Kendall, Gal, Cipolla 2018).

    For each task t with raw loss L_t:
        L = Σ_t  (1 / (2 · exp(s_t))) · L_t  +  0.5 · s_t

    s_t are learnable scalars; gradient signs balance task losses automatically.
    Floor clamp prevents `exp(-large_negative)` blow-up.
    """

    def __init__(self, task_names: List[str], s_floor: float = -2.0, s_init: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.s_floor = s_floor
        self.log_var = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(s_init, dtype=torch.float32))
            for name in self.task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = torch.zeros((), device=next(iter(losses.values())).device)
        parts: Dict[str, torch.Tensor] = {}
        for name in self.task_names:
            if name not in losses:
                raise KeyError(
                    f"UncertaintyWeighter: task '{name}' not in provided losses dict. "
                    f"Available: {list(losses.keys())}"
                )
            s = torch.clamp(self.log_var[name], min=self.s_floor)
            l = losses[name]
            scaled = 0.5 * torch.exp(-s) * l + 0.5 * s
            parts[f"weighted_{name}"] = scaled.detach()
            parts[f"raw_{name}"] = l.detach()
            parts[f"s_{name}"] = s.detach()
            total = total + scaled
        return total, parts
