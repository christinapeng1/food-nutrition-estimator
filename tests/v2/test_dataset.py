# tests/v2/test_dataset.py
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from src.v2.vocab import Vocab
from src.v2.stats import TrainStats
from src.v2.dataset import (
    parse_dish_metadata_row,
    DishLabels,
    Nutrition5kRGBD,
    build_default_train_transform,
    build_default_eval_transform,
)


def _dummy_stats():
    return TrainStats(
        scalar_mean=np.array([250., 200., 12., 25., 15.], dtype=np.float32),
        scalar_std=np.array([180., 130., 8., 18., 12.], dtype=np.float32),
        depth_mean=450.0, depth_std=80.0,
        mass_log1p_mean=2.0, mass_log1p_std=1.5,
    )


def test_parse_metadata_row(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    line = "dish_X,300.0,193.0,12.4,28.2,18.6,ingr_0000000508,soy sauce,3.4,1.8,0.02,0.17,0.28"
    row = parse_dish_metadata_row(line, v)
    assert row.dish_id == "dish_X"
    assert math.isclose(row.kcal, 300.0)
    assert math.isclose(row.mass, 193.0)
    assert len(row.ingr_ids) == 1
    assert row.ingr_ids[0] == "ingr_0000000508"
    assert math.isclose(row.ingr_grams[0], 3.4)


def test_parse_metadata_handles_multiple_ingredients(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    line = "dish_Y,100,50,1,2,3,ingr_0000000026,white rice,30,33,0.1,7,0.6,ingr_0000000508,soy sauce,5,2.6,0.03,0.25,0.4"
    row = parse_dish_metadata_row(line, v)
    assert len(row.ingr_ids) == 2
    assert row.ingr_grams == [30.0, 5.0]


def test_dataset_returns_correct_shapes(repo_root, ingredients_csv, dish_csv_cafe1, monkeypatch):
    # We test against real metadata but only need a few dishes that have RGB locally.
    v = Vocab.from_csv(ingredients_csv)
    s = _dummy_stats()
    img_root = repo_root / "data" / "sample" / "imagery"
    if not img_root.is_dir():
        pytest.skip("no imagery available")

    # Pick first 2 dish IDs that have rgb.png
    dish_ids = [d.name for d in img_root.iterdir() if (d / "rgb.png").is_file()][:2]
    if len(dish_ids) < 2:
        pytest.skip("need at least 2 dishes with rgb.png")

    ds = Nutrition5kRGBD(
        dish_ids=dish_ids,
        metadata_csvs=[dish_csv_cafe1],
        imagery_root=img_root,
        vocab=v,
        stats=s,
        transform=build_default_eval_transform(),
        require_depth=False,    # sample test allows no depth
    )
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["rgb"].shape == (3, 224, 224)
    assert sample["depth"].shape == (2, 224, 224)
    assert sample["y_scalar"].shape == (5,)
    assert sample["y_ingr_binary"].shape == (v.size,)
    assert sample["y_ingr_mass"].shape == (v.size,)
    assert sample["y_ingr_mask"].shape == (v.size,)
    # Mass mask consistency
    assert int(sample["y_ingr_mask"].sum()) >= 1


def test_label_construction_sums_to_total_mass(ingredients_csv, dish_csv_cafe1):
    """Per-ingredient grams must sum (within rounding) to total dish mass."""
    v = Vocab.from_csv(ingredients_csv)
    rows = []
    with open(dish_csv_cafe1) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(parse_dish_metadata_row(ln, v))
            except Exception:
                pass
    n_checked = 0
    for r in rows[:50]:
        if not r.ingr_grams:
            continue
        s = sum(r.ingr_grams)
        # Allow a 2% relative slack for floating-point + dataset rounding
        assert abs(s - r.mass) <= max(1.0, 0.02 * r.mass), \
            f"{r.dish_id}: ingr_sum={s} vs total_mass={r.mass}"
        n_checked += 1
    assert n_checked >= 10, "too few rows checked"
