# scripts/compute_train_stats.py
"""Compute train-set z-score statistics for scalars / depth / mass.

Run once before training; output is consumed by train.py and evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when script is invoked directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from src.v2.dataset import DEPTH_CLIP_MAX, DEPTH_CLIP_MIN, parse_dish_metadata_row
from src.v2.stats import TrainStats
from src.v2.vocab import Vocab


def main(args):
    avail = set([ln.strip() for ln in Path(args.available_dish_ids).read_text().splitlines() if ln.strip()])
    train_ids = [ln.strip() for ln in Path(args.train_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    vocab = Vocab.from_csv(args.vocab_csv)

    rows = {}
    for csv in [args.metadata_cafe1, args.metadata_cafe2]:
        with open(csv) as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try:
                    r = parse_dish_metadata_row(ln, vocab)
                    rows[r.dish_id] = r
                except Exception: pass

    scalars = []
    grams = []
    img_root = Path(args.imagery_root)

    depth_sum = 0.0; depth_sqsum = 0.0; depth_n = 0
    n = 0
    for did in train_ids:
        r = rows.get(did)
        if r is None: continue
        scalars.append([r.kcal, r.mass, r.fat, r.carb, r.protein])
        grams.extend([g for g in r.ingr_grams if g > 0])
        d_p = img_root / did / "depth_raw.png"
        if d_p.is_file():
            arr = np.array(Image.open(d_p)).astype(np.float32)
            valid = arr > 0
            arr_v = np.clip(arr[valid], DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
            depth_sum += float(arr_v.sum())
            depth_sqsum += float((arr_v ** 2).sum())
            depth_n += int(arr_v.size)
        n += 1
        if args.max and n >= args.max: break

    arr = np.asarray(scalars, dtype=np.float32)
    s_mean = arr.mean(axis=0); s_std = arr.std(axis=0) + 1e-6
    g = np.asarray(grams, dtype=np.float32)
    log1p = np.log1p(g)
    m_mean = float(log1p.mean()); m_std = float(log1p.std() + 1e-6)
    d_mean = depth_sum / max(depth_n, 1)
    d_var = max(depth_sqsum / max(depth_n, 1) - d_mean ** 2, 1e-6)
    d_std = float(np.sqrt(d_var))

    stats = TrainStats(s_mean, s_std, d_mean, d_std, m_mean, m_std)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats.save(args.out)
    print(json.dumps({
        "n_dishes": n, "scalar_mean": s_mean.tolist(), "scalar_std": s_std.tolist(),
        "depth_mean": d_mean, "depth_std": d_std,
        "mass_log1p_mean": m_mean, "mass_log1p_std": m_std,
    }, indent=2))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--train-ids-path", default="data/sample/splits/train_ids.txt")
    p.add_argument("--available-dish-ids", default="data/sample/available_dish_ids.txt")
    p.add_argument("--imagery-root", default="data/sample/imagery")
    p.add_argument("--vocab-csv", default="data/raw/metadata/ingredients_metadata.csv")
    p.add_argument("--metadata-cafe1", default="data/raw/metadata/dish_metadata_cafe1.csv")
    p.add_argument("--metadata-cafe2", default="data/raw/metadata/dish_metadata_cafe2.csv")
    p.add_argument("--out", default="data/sample/train_stats.json")
    p.add_argument("--max", type=int, default=0, help="cap number of dishes processed (debug)")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
