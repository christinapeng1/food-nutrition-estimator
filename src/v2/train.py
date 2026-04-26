# src/v2/train.py
"""Main training loop with AMP + EMA + cosine schedule + correctness gates.

Spec: §5.

Usage:
    python -m src.v2.train --config src/v2/configs/main.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import Nutrition5kRGBD, build_default_eval_transform, build_default_train_transform
from .losses import (
    UncertaintyWeighter, atwater_loss, bce_with_pos_weight,
    kcal_consistency_loss, masked_huber,
)
from .model import NutritionRGBDModel
from .stats import TrainStats
from .vocab import Vocab


logger = logging.getLogger("nutrition5k.train")


@dataclass
class TrainConfig:
    run_id: str
    out_root: str
    imagery_root: str
    metadata_cafe1: str
    metadata_cafe2: str
    train_ids_path: str
    val_ids_path: str          # built once (10% of train)
    available_ids_path: str
    vocab_csv: str
    stats_path: str            # produced by scripts/compute_train_stats.py
    ckpt_root: str = "checkpoints/v2"  # default; override in config to make absolute
    use_depth: bool = True
    n_epochs: int = 50
    batch_size: int = 64
    grad_accum: int = 1
    lr_backbone: float = 3e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    ema_decay: float = 0.9999
    early_stop_patience: int = 10
    seed: int = 42
    bf16: bool = True
    num_workers: int = 6
    log_every: int = 50
    overfit_micro: bool = False  # G3 mode

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        from dataclasses import fields
        with open(path) as f:
            d = yaml.safe_load(f)
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"Unknown config keys in {path}: {sorted(extra)}")
        return cls(**d)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow


def build_pos_weight(loader: DataLoader, vocab_size: int, device: torch.device) -> torch.Tensor:
    """pos_weight per class = (num_negatives / num_positives), capped to 100."""
    pos = torch.zeros(vocab_size); n = 0
    for b in loader:
        pos += b["y_ingr_binary"].sum(dim=0)
        n += b["y_ingr_binary"].size(0)
        if n > 500: break
    pos = pos / max(n, 1)  # frequency
    neg = 1.0 - pos
    pw = (neg / pos.clamp(min=1e-3)).clamp(max=100.0)
    return pw.to(device)


def lr_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def make_loaders(cfg: TrainConfig, vocab: Vocab, stats: TrainStats):
    avail = set([ln.strip() for ln in Path(cfg.available_ids_path).read_text().splitlines() if ln.strip()])
    train_ids = [ln.strip() for ln in Path(cfg.train_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    val_ids = [ln.strip() for ln in Path(cfg.val_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    if cfg.overfit_micro:
        train_ids = train_ids[:8]; val_ids = val_ids[:8]

    md = [cfg.metadata_cafe1, cfg.metadata_cafe2]
    # In overfit_micro mode, use eval transform on TRAIN too — augmentation is noise that
    # prevents the model from memorizing 8 specific images. Also use the SAME 8 dishes for
    # val so we can directly observe whether overfitting succeeded.
    train_transform = (build_default_eval_transform()
                       if cfg.overfit_micro
                       else build_default_train_transform())
    if cfg.overfit_micro:
        val_ids = train_ids
    train_ds = Nutrition5kRGBD(train_ids, md, cfg.imagery_root, vocab, stats,
                               transform=train_transform,
                               require_depth=cfg.use_depth)
    val_ds = Nutrition5kRGBD(val_ids, md, cfg.imagery_root, vocab, stats,
                             transform=build_default_eval_transform(),
                             require_depth=cfg.use_depth)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def compute_total_loss(model, batch, weighter: UncertaintyWeighter,
                       densities: torch.Tensor, stats: TrainStats,
                       pos_weight: torch.Tensor, use_depth: bool) -> tuple[torch.Tensor, dict]:
    rgb = batch["rgb"]; depth = batch["depth"]
    out = model(rgb, depth, use_depth=use_depth)
    sc_pred_z = out["scalar"]
    sc_target_z = batch["y_scalar"]
    L_scalar = masked_huber(sc_pred_z, sc_target_z,
                            mask=torch.ones_like(sc_target_z), delta=1.0)
    L_ingr_cls = bce_with_pos_weight(out["ingr_logits"], batch["y_ingr_binary"], pos_weight)
    L_ingr_mass = masked_huber(out["ingr_mass"], batch["y_ingr_mass"],
                               batch["y_ingr_mask"], delta=1.0)
    # Atwater + kcal_consist on raw kcal scale
    sc_pred_raw = sc_pred_z * torch.tensor(stats.scalar_std, device=sc_pred_z.device) \
                  + torch.tensor(stats.scalar_mean, device=sc_pred_z.device)
    direct_kcal = sc_pred_raw[:, 0]
    fat = sc_pred_raw[:, 2]; carb = sc_pred_raw[:, 3]; protein = sc_pred_raw[:, 4]
    # Divide raw-units losses by std_kcal so gradients backprop'd through the inverse z-score
    # are in the same magnitude as the z-scored Huber losses. Without this, the chain rule
    # multiplies gradients by std_kcal≈222, making atwater/kcal_consist dominate updates ~500x.
    kcal_scale = float(stats.scalar_std[0])
    L_atwater = atwater_loss(direct_kcal, fat, carb, protein) / kcal_scale
    mass_raw = torch.expm1(out["ingr_mass"] * stats.mass_log1p_std + stats.mass_log1p_mean).clamp(min=0)
    # Mask by GT presence during TRAINING so derived_kcal sums only over actually-present
    # ingredients. Without this, at init the mass head outputs ~expm1(mass_log1p_mean)≈8g
    # per slot × 555 slots × density → derived_kcal ≈ 8000 kcal vs direct ≈ 250 kcal,
    # making L_kcal_consist ≈ 4 orders of magnitude larger than other losses and
    # poisoning training. (Eval-time uses sigmoid(ingr_logits) > 0.5 instead.)
    derived_kcal = (mass_raw * densities[None, :] * batch["y_ingr_mask"]).sum(dim=1)
    L_kcal_consist = kcal_consistency_loss(direct_kcal, derived_kcal) / kcal_scale

    losses = {
        "scalar": L_scalar, "ingr_cls": L_ingr_cls, "ingr_mass": L_ingr_mass,
        "atwater": L_atwater, "kcal_consist": L_kcal_consist,
    }
    total, parts = weighter(losses)
    return total, {**losses, **parts}


def evaluate_val(model, val_loader, weighter, densities, stats, pos_weight, use_depth, device) -> dict:
    model.eval()
    sums = {}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)
            total, parts = compute_total_loss(model, batch, weighter, densities, stats, pos_weight, use_depth)
            B = batch["rgb"].size(0)
            for k, v in parts.items():
                if torch.is_tensor(v):
                    sums[k] = sums.get(k, 0.0) + float(v) * B
            n += B
    return {k: v / max(n, 1) for k, v in sums.items()}


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_root) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist resolved config for reproducibility
    (out_dir / "config.yaml").write_text(yaml.dump(asdict(cfg), default_flow_style=False))
    ckpt_dir = Path(cfg.ckpt_root) / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "train.log"); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(fh); logger.addHandler(sh); logger.setLevel(logging.INFO)
    logger.info("config: %s", json.dumps(asdict(cfg), indent=2))

    vocab = Vocab.from_csv(cfg.vocab_csv)
    vocab.save(ckpt_dir / "vocab.json")
    stats = TrainStats.load(cfg.stats_path)
    stats.save(ckpt_dir / "train_stats.json")

    train_loader, val_loader = make_loaders(cfg, vocab, stats)
    logger.info("train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = NutritionRGBDModel(n_ingredients=vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
    )
    weighter = UncertaintyWeighter(["scalar", "ingr_cls", "ingr_mass", "atwater", "kcal_consist"]).to(device)
    # Uncertainty weighter log-variances need a smaller LR than task heads (they're
    # dimensionless scalars that should drift slowly). Excluded from cosine schedule
    # — keeping s_t free to adapt throughout training.
    optimizer.add_param_group({"params": list(weighter.parameters()), "lr": cfg.lr_head * 0.1, "weight_decay": 0.0})
    densities = torch.tensor(vocab.idx_to_density, dtype=torch.float32, device=device)
    pos_weight = build_pos_weight(train_loader, vocab.size, device)

    ema = EMA(model, decay=cfg.ema_decay)
    total_steps = len(train_loader) * cfg.n_epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)
    best_val = math.inf; bad = 0

    step = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        for it, batch in enumerate(train_loader):
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
                total, parts = compute_total_loss(
                    model, batch, weighter, densities, stats, pos_weight, cfg.use_depth
                )

            if not torch.isfinite(total):
                logger.error("NaN/Inf loss at step=%d epoch=%d — aborting", step, epoch)
                return

            (total / cfg.grad_accum).backward()

            # Adjust LR per-param-group for warmup/cosine — backbone + head only.
            # Weighter LR is fixed (see above) so s_t can keep adapting after head LR decays.
            # For overfit_micro: use constant LR after warmup (cosine would zero out the LR
            # before the model can finish memorizing the tiny train set).
            if cfg.overfit_micro:
                sched_factor = min(1.0, (step + 1) / max(warmup_steps, 1))
            else:
                sched_factor = lr_schedule(step, total_steps, warmup_steps)
            for pg, base in zip(optimizer.param_groups[:2], [cfg.lr_backbone, cfg.lr_head]):
                pg["lr"] = base * sched_factor

            if (it + 1) % cfg.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(weighter.parameters()), max_norm=5.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                step += 1

                if step % cfg.log_every == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.4f scalar=%.4f cls=%.4f mass=%.4f atw=%.4f kc=%.4f gn=%.2f lr=%.2e "
                        "s={s:.2f},{si:.2f},{sm:.2f},{sa:.2f},{sk:.2f}".format(
                            s=float(parts["s_scalar"]), si=float(parts["s_ingr_cls"]),
                            sm=float(parts["s_ingr_mass"]), sa=float(parts["s_atwater"]),
                            sk=float(parts["s_kcal_consist"]),
                        ),
                        step, epoch, float(total),
                        float(parts["scalar"]), float(parts["ingr_cls"]),
                        float(parts["ingr_mass"]), float(parts["atwater"]),
                        float(parts["kcal_consist"]), float(grad_norm),
                        optimizer.param_groups[0]["lr"],
                    )

        # End of epoch — eval on val with EMA weights
        val_model = copy.deepcopy(model)
        val_model.load_state_dict(ema.state_dict(), strict=True)
        val_metrics = evaluate_val(val_model, val_loader, weighter, densities, stats,
                                   pos_weight, cfg.use_depth, device)
        # val_metrics["scalar"] is the masked Huber loss on z-scored 5-vec — used as a
        # robust proxy for per-target z-score MAE for early-stopping/G4. They differ in
        # the |err| > delta region (Huber saturates linearly there).
        val_score = val_metrics.get("scalar", math.inf)
        logger.info("epoch=%d val_huber_z=%.4f val_total_metrics=%s", epoch, val_score,
                    {k: round(v, 4) for k, v in val_metrics.items()})

        # G4 sanity (epoch 1)
        if epoch == 0 and val_score >= 1.0:
            logger.error("G4 FAIL: val Huber(z-score) >= 1.0 at epoch 1 (%.4f). Stop and diagnose.", val_score)
            torch.save({"model": model.state_dict()}, ckpt_dir / "g4_fail_last.pt")
            return

        # Save best (atomic: write temp, then rename)
        if val_score < best_val:
            best_val = val_score; bad = 0
            tmp = ckpt_dir / "best_tmp.pt"
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "val_score": val_score,
            }, tmp)
            tmp.rename(ckpt_dir / "best.pt")
            # Also dump just the EMA weights for downstream eval scripts that load them directly
            tmp2 = ckpt_dir / "ema_tmp.pt"
            torch.save({"model": ema.state_dict(), "epoch": epoch}, tmp2)
            tmp2.rename(ckpt_dir / "ema.pt")
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                logger.info("Early stop at epoch %d", epoch); break

    # Always save last
    torch.save({"model": model.state_dict()}, ckpt_dir / "last.pt")
    torch.save({"model": ema.state_dict()}, ckpt_dir / "last_ema.pt")
    logger.info("done. best val %.4f", best_val)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overfit-micro", action="store_true", help="G3 mode: 8-dish overfit test")
    args = p.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.overfit_micro:
        cfg.overfit_micro = True
        cfg.n_epochs = 100; cfg.early_stop_patience = 999; cfg.batch_size = 8
        cfg.lr_head = 1e-3; cfg.lr_backbone = 1e-4
        cfg.log_every = 1     # 1 step per epoch in overfit mode; log every step
        cfg.bf16 = False      # FP32 for cleaner debugging precision (8 dishes is tiny)
        cfg.num_workers = 0   # avoid dataloader-worker NFS cleanup tracebacks on tiny set
    main(cfg)


if __name__ == "__main__":
    cli()
