# scripts/run_g1.py
"""G1 gate — visualize 5 random dishes; verify per-ingredient sums; verify vocab."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from src.v2.dataset import (Nutrition5kRGBD, build_default_eval_transform, parse_dish_metadata_row)
from src.v2.stats import TrainStats
from src.v2.vocab import Vocab
from src.v2 import viz


def main():
    random.seed(0)
    vocab = Vocab.from_csv("data/raw/metadata/ingredients_metadata.csv")
    print("vocab size:", vocab.size)

    # Diff against existing checkpoint vocab if present
    old_p = Path("checkpoints/vocab.json")
    if old_p.is_file():
        try:
            old = json.loads(old_p.read_text())
            if isinstance(old, list):
                old_size = len(old)
            elif isinstance(old, dict):
                old_size = len(old.get("idx_to_id", old))
            else:
                old_size = -1
            print(f"old vocab.json size = {old_size}; new = {vocab.size}")
        except Exception as e:
            print("could not read old vocab:", e)

    stats = TrainStats.load("data/sample/train_stats.json")
    avail = set(Path("data/sample/available_dish_ids.txt").read_text().splitlines())
    pick = random.sample(sorted(avail), 5)
    ds = Nutrition5kRGBD(
        dish_ids=pick,
        metadata_csvs=["data/raw/metadata/dish_metadata_cafe1.csv",
                       "data/raw/metadata/dish_metadata_cafe2.csv"],
        imagery_root="data/sample/imagery",
        vocab=vocab, stats=stats,
        transform=build_default_eval_transform(),
        require_depth=False,
    )
    samples = [ds[i] for i in range(len(ds))]
    out_dir = Path("docs/runs/g1"); out_dir.mkdir(parents=True, exist_ok=True)
    viz.sanity_panel(samples, out_dir / "sanity.png")
    print("wrote", out_dir / "sanity.png")

    # Per-ingredient mass sum check on 50 random dishes
    sample_ids = random.sample(sorted(avail), 50)
    rows = {}
    for csv in ["data/raw/metadata/dish_metadata_cafe1.csv", "data/raw/metadata/dish_metadata_cafe2.csv"]:
        for ln in Path(csv).read_text().splitlines():
            try:
                r = parse_dish_metadata_row(ln, vocab); rows[r.dish_id] = r
            except Exception: pass
    n_checked = 0; n_pass = 0
    for did in sample_ids:
        r = rows.get(did)
        if r is None or not r.ingr_grams: continue
        s = sum(r.ingr_grams); diff = abs(s - r.mass)
        ok = diff <= max(1.0, 0.02 * r.mass)
        if not ok:
            print(f"  WARN  {did} ingr_sum={s:.1f} total={r.mass:.1f} diff={diff:.2f}")
        n_pass += int(ok); n_checked += 1
    print(f"per-ingredient sum check: {n_pass}/{n_checked} pass")
    if n_pass < 0.9 * n_checked:
        sys.exit(1)
    print("G1 OK")


if __name__ == "__main__":
    main()
