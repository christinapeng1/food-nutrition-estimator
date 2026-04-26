# src/v2/dataset.py
"""Nutrition5k RGB-D dataset + transforms.

Spec: docs/superpowers/specs/2026-04-26-nutrition5k-sota-design.md §3, §4.

Returns dict per dish:
    rgb           (3, 224, 224) float32, ImageNet-normalized
    depth         (2, 224, 224) float32, [normalized_depth, valid_mask]
    y_scalar      (5,)          float32, z-scored kcal/mass/fat/carb/protein
    y_ingr_binary (V,)          float32, multi-label one-hot
    y_ingr_mass   (V,)          float32, log1p+z-scored at present positions; 0 elsewhere
    y_ingr_mask   (V,)          float32, 1 where ingredient is present, else 0
    dish_id       str

Test-time we also expose RAW (unnormalized) versions for evaluation:
    y_scalar_raw  (5,)          raw kcal/mass/fat/carb/protein
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .stats import TrainStats
from .vocab import Vocab


DEPTH_CLIP_MIN = 200.0   # mm
DEPTH_CLIP_MAX = 800.0


@dataclass
class DishLabels:
    dish_id: str
    kcal: float
    mass: float
    fat: float
    carb: float
    protein: float
    ingr_ids: List[str] = field(default_factory=list)
    ingr_grams: List[float] = field(default_factory=list)


def parse_dish_metadata_row(line: str, vocab: Vocab) -> DishLabels:
    """Parse a single CSV row from dish_metadata_cafe{1,2}.csv.

    Format:
        dish_id,kcal,mass,fat,carb,protein,
        [ingr_id,name,grams,kcal,fat,carb,protein] × N
    Ingredients with id not in vocab are silently dropped.
    """
    parts = line.strip().split(",")
    if len(parts) < 6:
        raise ValueError(f"row too short: {line[:80]}")
    dish_id = parts[0]
    kcal = float(parts[1]); mass = float(parts[2])
    fat = float(parts[3]);  carb = float(parts[4]); protein = float(parts[5])
    ingr_part = parts[6:]
    if len(ingr_part) % 7 != 0:
        # Drop trailing partial entry — defensive
        ingr_part = ingr_part[: (len(ingr_part) // 7) * 7]
    ingr_ids: List[str] = []
    ingr_grams: List[float] = []
    for i in range(0, len(ingr_part), 7):
        ingr_id = ingr_part[i].strip()
        try:
            grams = float(ingr_part[i + 2])
        except ValueError:
            continue
        if ingr_id in vocab.id_to_idx:
            ingr_ids.append(ingr_id)
            ingr_grams.append(grams)
    return DishLabels(dish_id, kcal, mass, fat, carb, protein,
                      ingr_ids=ingr_ids, ingr_grams=ingr_grams)


def _load_metadata_dict(metadata_csvs: Sequence[Path | str], vocab: Vocab) -> dict[str, DishLabels]:
    out: dict[str, DishLabels] = {}
    for csv in metadata_csvs:
        with open(csv) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = parse_dish_metadata_row(ln, vocab)
                    out[row.dish_id] = row
                except Exception:
                    continue
    return out


# ---------- Transforms ----------

def build_default_train_transform() -> Callable:
    return Nutrition5kTransform(train=True)


def build_default_eval_transform() -> Callable:
    return Nutrition5kTransform(train=False)


class Nutrition5kTransform:
    """Joint RGB+depth transform; depth gets a separate path."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, train: bool, size: int = 224, resize: int = 256):
        self.train = train
        self.size = size
        self.resize = resize
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2)
        self.rand_aug = T.RandAugment(num_ops=2, magnitude=9)

    def __call__(
        self,
        rgb_pil: Image.Image,
        depth_arr: np.ndarray,        # uint16, raw mm
        valid_mask: np.ndarray,       # bool
        depth_mean: float,
        depth_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize keeping aspect
        rgb_pil = TF.resize(rgb_pil, self.resize, antialias=True)
        depth_t = torch.from_numpy(depth_arr.astype(np.float32))[None, None, ...]   # (1,1,H,W)
        mask_t = torch.from_numpy(valid_mask.astype(np.float32))[None, None, ...]
        depth_t = F.interpolate(depth_t, size=self.resize, mode="bilinear", align_corners=False)[0, 0]
        mask_t = F.interpolate(mask_t, size=self.resize, mode="nearest")[0, 0]

        # Crop
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(rgb_pil, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            rgb_pil = TF.resized_crop(rgb_pil, i, j, h, w, [self.size, self.size], antialias=True)
            depth_t = depth_t[i:i+h, j:j+w][None, None]
            mask_t = mask_t[i:i+h, j:j+w][None, None]
            depth_t = F.interpolate(depth_t, size=self.size, mode="bilinear", align_corners=False)[0, 0]
            mask_t = F.interpolate(mask_t, size=self.size, mode="nearest")[0, 0]
        else:
            rgb_pil = TF.center_crop(rgb_pil, [self.size, self.size])
            cy = (depth_t.shape[-2] - self.size) // 2
            cx = (depth_t.shape[-1] - self.size) // 2
            depth_t = depth_t[cy:cy+self.size, cx:cx+self.size]
            mask_t = mask_t[cy:cy+self.size, cx:cx+self.size]

        # HFlip
        if self.train and random.random() < 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            depth_t = torch.flip(depth_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])

        # RGB color aug
        if self.train:
            rgb_pil = self.rand_aug(rgb_pil)
            rgb_pil = self.color_jitter(rgb_pil)

        rgb = TF.to_tensor(rgb_pil)
        rgb = TF.normalize(rgb, self.IMAGENET_MEAN, self.IMAGENET_STD)

        # Depth scale aug
        if self.train:
            depth_t = depth_t * random.uniform(0.95, 1.05)

        # Apply mask: invalid → 0 (post-z-score this means "mean", which is the safest fill)
        depth_norm = (depth_t - depth_mean) / max(depth_std, 1e-6)
        depth_norm = depth_norm * mask_t   # zero out invalid
        depth_out = torch.stack([depth_norm.float(), mask_t.float()], dim=0)
        return rgb, depth_out


# ---------- Dataset ----------

class Nutrition5kRGBD(Dataset):
    """RGB-D Nutrition5k dish-level dataset.

    Args:
        dish_ids:        list of dish IDs to include
        metadata_csvs:   list of dish_metadata_*.csv paths
        imagery_root:    Path to data/sample/imagery
        vocab:           Vocab (for ingredient indexing + density)
        stats:           TrainStats (for z-score normalization of labels & depth)
        transform:       Nutrition5kTransform (or any (rgb_pil, depth_arr, mask, mean, std) -> (rgb, depth))
        require_depth:   if True, dishes without depth_raw.png are filtered
    """

    def __init__(
        self,
        dish_ids: Sequence[str],
        metadata_csvs: Sequence[Path | str],
        imagery_root: Path,
        vocab: Vocab,
        stats: TrainStats,
        transform: Optional[Callable] = None,
        require_depth: bool = True,
    ):
        self.imagery_root = Path(imagery_root)
        self.vocab = vocab
        self.stats = stats
        self.transform = transform
        meta = _load_metadata_dict(metadata_csvs, vocab)
        self.dishes: List[DishLabels] = []
        for did in dish_ids:
            row = meta.get(did)
            if row is None:
                continue
            rgb_p = self.imagery_root / did / "rgb.png"
            if not rgb_p.is_file():
                continue
            d_p = self.imagery_root / did / "depth_raw.png"
            if require_depth and (not d_p.is_file() or d_p.stat().st_size == 0):
                continue
            self.dishes.append(row)

    def __len__(self) -> int:
        return len(self.dishes)

    def _read_depth(self, dish_id: str) -> Tuple[np.ndarray, np.ndarray]:
        d_p = self.imagery_root / dish_id / "depth_raw.png"
        if d_p.is_file():
            arr = np.array(Image.open(d_p)).astype(np.float32)  # (H,W) uint16 -> float32 mm
        else:
            arr = np.zeros((480, 640), dtype=np.float32)
        valid = (arr > 0).astype(np.float32)
        arr = np.clip(arr, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX) * valid   # invalid stays 0
        return arr, valid

    def __getitem__(self, i: int):
        d = self.dishes[i]
        rgb_p = self.imagery_root / d.dish_id / "rgb.png"
        rgb_pil = Image.open(rgb_p).convert("RGB")
        depth_arr, valid = self._read_depth(d.dish_id)
        if self.transform is None:
            self.transform = build_default_eval_transform()
        rgb_t, depth_t = self.transform(rgb_pil, depth_arr, valid,
                                        self.stats.depth_mean, self.stats.depth_std)

        # Labels — 5 scalars
        y_scalar_raw = np.array([d.kcal, d.mass, d.fat, d.carb, d.protein], dtype=np.float32)
        y_scalar = self.stats.scalar_z(y_scalar_raw).astype(np.float32)

        # Labels — 555-dim
        V = self.vocab.size
        y_ingr_binary = np.zeros(V, dtype=np.float32)
        y_ingr_mass = np.zeros(V, dtype=np.float32)
        y_ingr_mask = np.zeros(V, dtype=np.float32)
        for ingr_id, grams in zip(d.ingr_ids, d.ingr_grams):
            idx = self.vocab.id_to_idx[ingr_id]
            y_ingr_binary[idx] = 1.0
            y_ingr_mass[idx] = self.stats.mass_log1p_z(np.array([grams], dtype=np.float32))[0]
            y_ingr_mask[idx] = 1.0

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "y_scalar": torch.from_numpy(y_scalar),
            "y_scalar_raw": torch.from_numpy(y_scalar_raw),
            "y_ingr_binary": torch.from_numpy(y_ingr_binary),
            "y_ingr_mass": torch.from_numpy(y_ingr_mass),
            "y_ingr_mask": torch.from_numpy(y_ingr_mask),
            "dish_id": d.dish_id,
        }
