# Nutrition5k SOTA Technical Report

**Goal**: beat the Direct Prediction baseline from the Nutrition5k paper (Thames et al., CVPR 2021).

**Result**: **4 of 5 scalar metrics beat baseline** (kcal 61.9 vs 70 / mass 37.9 vs 40 / fat 4.7 vs 6 / carb 6.2 vs 10), trained in 14 minutes on a single A6000.

---

## Contents

1. [Task & objective](#1-task--objective)
2. [Data](#2-data)
3. [Preprocessing & augmentation](#3-preprocessing--augmentation)
4. [Model architecture](#4-model-architecture)
5. [Loss functions](#5-loss-functions)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Main results](#8-main-results)
9. [Ablation](#9-ablation)
11. [Limitations & future work](#11-limitations--future-work)
12. [Reproducibility](#12-reproducibility)

---

## 1. Task & objective

**Input**: overhead RGB-D image of a dish.

**Output**:
- 5 scalars: kcal / mass / fat / carb / protein
- 555-way multi-label ingredient classification
- per-ingredient grams (for present ingredients)

**Baseline** (Google Nutrition5k Direct Prediction, InceptionV2 ~25M):

| Metric | Baseline | Hard floor | Stretch |
|---|---|---|---|
| kcal MAE | 70 | ≤ 70 | ≤ 60 |
| mass MAE | 40 g | ≤ 40 | ≤ 35 |
| fat / carb / protein | 6 / 10 / 5 g | report all | — |

**Resources**: single A6000 (48 GB) on Brown CCV `gpu2803`, 1–2 day budget.

---

## 2. Data

### 2.1 Nutrition5k

- Source: public GCS bucket `gs://nutrition5k_dataset/`
- Total: 5006 dishes, but **only 3490** have a realsense_overhead view
- Labels: industrial-grade food-scale measurements (kcal/mass/macros) + ingredient list + per-ingredient grams

### 2.2 Splits we use

| Split | Official | Intersection with `available` |
|---|---|---|
| Train | 4059 | 2755 → split 90/10 → 2479 train + 276 val |
| Test | 709 | **507** (final eval) |

Overhead RGB-D only; side-angle videos out of scope (time budget).

### 2.3 Engineering: parallel depth download

The original assignment uses serial `gsutil cp` — 3490 files takes about an hour. Our approach:

```bash
awk '...' dish_list.txt | xargs -P 16 -I LINE bash -c '
  IFS=$'"'"'\t'"'"' read -r src dst <<< "LINE"
  [ -f "$dst" ] && [ -s "$dst" ] && exit 0
  curl -sSL -o "$dst" "$src"
'
```

**Key tricks**:
- Public HTTPS (`storage.googleapis.com/nutrition5k_dataset/...`) — no `gcloud auth` needed
- `xargs -P 16` for 16-way parallel curl
- Idempotent (skip files already on disk)
- `IFS=$'\t'` uses ANSI-C quoting (**not** `$"\t"`, which is bash i18n lookup)

**Measured**: 3490 files in 35 seconds. One dish (`dish_1564159636`) has a 0-byte file in GCS itself (dataset-level corruption); final usable count is 3489/3490.

---

## 3. Preprocessing & augmentation

### 3.1 RGB

```
PNG → PIL → [0,1] → ImageNet mean/std → resize 256×256
                          ↓
                train:  RandomResizedCrop(0.7-1.0) → 224
                eval:   CenterCrop 224
```

**RGB and depth must be resized to the same size** — otherwise the synced crop coordinates from RandomResizedCrop become inconsistent across the two streams.

### 3.2 Depth (with a **critical fix**)

Empirically measured pixel distribution over 30 random dishes (valid pixels only): p1 = 3021, p50 = 3571, p99 = 4867 mm. The RealSense rig sits ~3–5 m above the dish.

The spec originally clipped to `[200, 800]` mm (mobile ToF range — wrong). Corrected to `[2500, 6000]` mm:

```
16-bit PNG (480, 640, mm)
  → valid_mask = (depth > 0)
  → clip(depth, 2500, 6000) * mask    # invalid pixels stay 0
  → resize 256×256 (depth: bilinear, mask: nearest)
  → synced crop with RGB → 224×224
  → (depth - mean) / (std + 1e-6) * mask
  → stack([depth_norm, mask], dim=0)  →  (2, 224, 224)
```

**If you don't fix the clip range**: 96% of valid pixels saturate at the cap, and the depth signal is gone.

### 3.3 Labels

| Key | Shape | Meaning |
|---|---|---|
| `y_scalar` | (5,) | (kcal, mass, fat, carb, protein), **z-scored** |
| `y_scalar_raw` | (5,) | raw units (used at eval) |
| `y_ingr_binary` | (555,) | multi-label one-hot |
| `y_ingr_mass` | (555,) | log1p(grams) z-scored; 0 where absent |
| `y_ingr_mask` | (555,) | 0/1 |

**Why log1p on mass**: per-ingredient grams are heavy-tailed (most < 5g, a few > 100g); plain z-score would let the tail dominate the loss.

### 3.4 Augmentation

| Aug | Scope | Strength |
|---|---|---|
| RandomResizedCrop | RGB+Depth synced | scale=(0.7, 1.0) |
| HorizontalFlip | RGB+Depth synced | p=0.5 |
| ColorJitter | RGB only | 0.2 |
| RandAugment | RGB only | n=2, m=9 |
| Depth random scale | Depth only | × U(0.95, 1.05) |
| MixUp / CutMix | — | **disabled** (breaks physical-quantity semantics) |

### 3.5 z-score statistics

`scripts/compute_train_stats.py` computes once and saves to `train_stats.json`:

```json
{
  "scalar_mean": [253.95, 215.69, 12.71, 19.33, 17.85],
  "scalar_std":  [222.19, 163.47, 13.44, 22.99, 20.19],
  "depth_mean":  3709.5,  "depth_std":  371.4,
  "mass_log1p_mean": 2.205,  "mass_log1p_std": 1.623
}
```

---

## 4. Model architecture

### 4.1 Overview

```
RGB (3, 224, 224) ─► ConvNeXt-Base (89M, ImageNet-1K)
                  ─► AvgPool + LN              ─► feat_rgb (1024)
                                                          │
Depth+Mask (2, 224, 224) ─► ConvNeXt-Tiny (28M, first conv adapted to 2 channels)
                          ─► AvgPool + LN       ─► feat_d (768)
                                                          │
                          concat → MLP(1792→512) → z (512)
                                                          │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        head_scalar       head_ingr        head_mass
        (512→5)           (512→555)        (512→555)
```

**Total parameters**: 116.9 M

### 4.2 Key design choices

**1. Late fusion (not 4-channel concat)**: RGB / depth / mask have different statistics and missing-value handling; separate streams keep their own inductive biases, and the "no-depth" ablation becomes trivial (zero out `feat_d`).

**2. Depth encoder uses Tiny, not Base**: depth has lower information density; Tiny avoids overfitting on 2479 training samples.

**3. First-conv channel-mean init for the depth encoder**:

```python
mean_w = w.mean(dim=1, keepdim=True)   # (out, 3, kh, kw) → (out, 1, kh, kw)
new_w = mean_w.repeat(1, 2, 1, 1)      # (out, 2, kh, kw)
```

Preserves the "style" of the pretrained weights.

**4. Three heads**:

| Head | Output | Loss | Supervision |
|---|---|---|---|
| `head_scalar` | (5,) | Huber | all dishes |
| `head_ingr` | (555,) logits | BCE + pos_weight | all dishes |
| `head_mass` | (555,) | masked Huber | only at GT-positive ingredient slots |

**5. Two-path kcal (design vs reality)**:

- **Original design**: `kcal_direct` (head_scalar) and `kcal_derived = Σ mass × density` constrain each other during training; the headline at eval is a 50/50 average.
- **Reality**: at init the mass head outputs ~8 g per slot; 555 slots × 8 g × ~2 cal/g ≈ 9000 kcal, while real average is 250. The `kcal_consist` loss is 4 orders of magnitude larger than the others and overwhelms training.
- **Fix**: train with a GT mask so `kcal_derived` only sums over present ingredients; eval with a predicted mask (sigmoid > 0.5); the headline kcal **uses direct only** — the derived path needs ingredient F1 ≥ 0.6 to be useful.

---

## 5. Loss functions

### 5.1 Five tasks

| Task | Formula |
|---|---|
| `L_scalar` | Huber(δ=1.0) on the 5-vector (z-scored) |
| `L_ingr_cls` | BCE-with-logits + pos_weight |
| `L_ingr_mass` | **masked** Huber, only at GT-positive slots |
| `L_atwater` | smooth_l1(direct_kcal, 9·fat + 4·carb + 4·protein) **/ std_kcal** |
| `L_kcal_consist` | smooth_l1(direct, derived) **/ std_kcal** |

### 5.2 Critical fix: divide by std_kcal

`L_atwater` is computed in raw kcal units (reasonable), but the gradient back-propagates through the inverse z-score with a factor of `std_kcal ≈ 222` — making this loss's gradient ~200× larger than the z-scored Huber's. Dividing by `std_kcal` cancels the chain-rule scaling, so all 5 losses end up at the same magnitude.

### 5.3 Joint loss: uncertainty weighting (Kendall 2018)

Each task gets a learnable log-variance:

$$L = \sum_t \left[\frac{L_t}{2 e^{s_t}} + \frac{s_t}{2}\right]$$

- `s_floor = -2` clamp prevents `exp(-s)` overflow
- Weighter LR = `lr_head × 0.1 = 3e-5`, **not on the cosine schedule**

### 5.4 BCE pos_weight

The 555-way multi-label problem is sparse (~5–7 ingredients per dish). Per-class `(neg/pos)` frequency ratio is used as `pos_weight`, capped at 100 to prevent gradient explosion.

---

## 6. Training

### 6.1 Hyperparameters

| Item | Value |
|---|---|
| Optimizer | AdamW, weight_decay=0.05 |
| LR (backbone / head / weighter) | 3e-5 / 3e-4 / 3e-5 |
| Schedule | warmup 5% → cosine (backbone + head only) |
| Batch | 64 |
| Epochs | 50 |
| Mixed precision | bf16 |
| Grad clip | max_norm = 5.0 |
| EMA decay | 0.9999 (**not actually used** — see §10) |
| Seed | 42 |

### 6.2 Training-loop critical bits

**The NaN check must come before `backward()`**:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    total, parts = compute_total_loss(...)

if not torch.isfinite(total):
    logger.error("NaN/Inf — aborting"); return

(total / cfg.grad_accum).backward()
```

Otherwise NaN poisons every parameter's `.grad`, the optimizer writes NaN weights, and training silently corrupts itself.

**G4 gate**: at the end of epoch 0, if `val_score ≥ 1.0` (near random baseline), abort immediately.

**Atomic checkpoint**: write to a temp path then `rename`, to survive SLURM preemption mid-write.

---

## 7. Evaluation

### 7.1 Procedure

1. Load `best.pt` (**model weights, not EMA** — see §10)
2. Eval transform (CenterCrop, no randomness)
3. **TTA**: 6 forwards per dish = 3 crops × 2 flips, averaged
4. Inverse z-score
5. Ingredient threshold sigmoid > 0.5
6. Headline kcal uses direct only

### 7.2 Reported metrics

5 scalars: MAE / %MAE / 95% bootstrap CI; ingredient F1 micro+macro; top-5 IoU; per-ingredient mass MAE.

---

## 8. Main results

### 8.1 Training curve

| Epoch | val_huber_z |
|---|---|
| 0 | 0.4955 |
| 10 | 0.4363 |
| 30 | 0.3228 |
| 49 | **0.2456** (−50%) |

Val decreases monotonically and is still decreasing at termination — **could train longer**.

### 8.2 Test set (n=507)

| Metric | Ours | 95% CI | Google baseline | Δ |
|---|---|---|---|---|
| **kcal MAE** | **61.9** | [56.0, 67.8] | 70 | **−11.5%** ✅ |
| **mass MAE** | **37.9** | [34.3, 41.7] | 40 | **−5.3%** ✅ |
| **fat MAE** | **4.7** | [4.3, 5.2] | 6 | **−21.7%** ✅ |
| **carb MAE** | **6.2** | [5.6, 6.7] | 10 | **−38.5%** ✅ |
| protein MAE | 6.0 | [5.3, 6.6] | 5 | +20% (slightly worse) |

**4/5 beat baseline**; both hard floors met.

Auxiliary: ingredient F1 micro = 0.331 / macro = 0.263, top-5 IoU = 0.230, per-ingredient mass MAE = 21.4 g.

---

## 9. Ablation

### 9.1 Design

A single ablation: **no-depth**. Same config as main except `use_depth=false`; in `forward()`, `feat_d` is zeroed.

### 9.2 Results (paired bootstrap, n=1000)

| Metric | Main (RGB+D) | No-Depth | Δ | 95% CI | Significant? |
|---|---|---|---|---|---|
| kcal MAE | 61.9 | **57.6** | −4.2 | [−6.9, −1.6] | ★ depth **HURTS** |
| mass MAE | **37.9** | 46.6 | +8.6 | [+5.9, +11.3] | ★ depth **HELPS** |
| fat / carb / protein | — | — | small | — | not sig |

### 9.3 Physical interpretation

- **Mass benefits from depth** (+8.6 g, 23%): mass is a volumetric quantity, and depth provides a direct geometric signal.
- **Kcal is hurt by depth** (−4.2 kcal): kcal is dominated by **ingredient identity** (salad vs fried rice → density is roughly determined). RGB color/texture is the strong identity signal. Within 50 epochs, the depth stream and the RGB stream **compete** in the fusion MLP.
- **Macros unaffected**: determined by ingredient identity, which depth doesn't change.

### 9.4 Implication

Future work: **head-specific depth gating** — depth flows only into the mass head, with a weakened path to the kcal head.

---

## 11. Limitations & future work

### 11.1 Limitations

- Test only on 507 / 709 (the missing 202 dishes have no overhead view)
- Single seed
- 50 epochs is short (curve still going down)
- EMA decay misconfigured — never actually used
- Derived-kcal pathway never paid off
- Wild OOD photos not optimized
- Ingredient F1 is low (0.33)

### 11.2 Improvements

**Low-cost, high return**:
- Train longer (100–200 epochs)
- EMA decay → 0.999
- Head-specific depth gating
- Replace BCE with focal loss

**Medium-cost**:
- Backbone upgrade to DINOv2-L / SigLIP-2-L
- Add side-angle video frames
- Per-ingredient mass as a sequence head

**Research directions**:
- VLM fine-tune (Qwen2.5-VL / InternVL3)
- Explicit 3D volume estimation → density × volume
- Self-supervised pretrain on Nutrition5k

---

## 12. Reproducibility

### 12.1 Environment

```bash
git clone <repo> && cd food-nutrition-estimator
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# torch must match the GPU driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader
.venv/bin/python -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.11.0 torchvision==0.26.0
```

### 12.2 Data

```bash
mkdir -p data/raw && cd data/raw
gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/{metadata,dish_ids,scripts} .
cd ../..

gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/ | \
    sed 's|gs://.*/||;s|/||' | grep ^dish_ > data/sample/available_dish_ids.txt

bash scripts/download_rgb.sh        # ~1 GB
bash scripts/download_depth.sh      # ~1.5 GB, 35 s
.venv/bin/python scripts/verify_depth.py
```

### 12.3 Train + evaluate

```bash
# split + stats
.venv/bin/python scripts/build_splits.py
.venv/bin/python scripts/compute_train_stats.py

# main training (14 min)
.venv/bin/python -m src.v2.train --config src/v2/configs/main.yaml

# evaluate
.venv/bin/python -m src.v2.evaluate \
    --checkpoint checkpoints/v2/main_seed42/best.pt \
    --vocab      checkpoints/v2/main_seed42/vocab.json \
    --stats      checkpoints/v2/main_seed42/train_stats.json \
    --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
    --output-dir docs/runs/main_seed42/eval/

# unit tests
.venv/bin/python -m pytest tests/v2/ -v   # 38 passed
```

### 12.4 Repository layout

```
src/v2/         # vocab/stats/dataset/model/losses/metrics/tta/evaluate/train/viz + configs/
tests/v2/       # 38 unit tests
scripts/        # data download, stats, G1 sanity
docs/
├── final_report.md           # English headline report
├── technical_report_zh.md    # Chinese version of this doc
├── technical_report_en.md    # this document
├── superpowers/{specs, plans} # design spec & implementation plan
├── runs/<run_id>/             # config + train.log + eval/
└── ablations/<name>/
checkpoints/v2/<run_id>/       # best.pt + vocab.json + train_stats.json
```

---

## Summary

- **Result**: 4/5 metrics beat the Google baseline; 14 minutes on a single A6000.
- **Method**: dual-stream ConvNeXt RGB-D + multi-task + uncertainty weighting + TTA.
- **Biggest takeaway**: depth helps mass but hurts kcal — head-specific gating is the obvious next step.
- **Honest disclosures**: derived-kcal pathway never used, protein slightly worse, single seed, 50 epochs is short.

Lowest-effort highest-return next step: train longer + fix EMA + head-specific depth gating. Expected: another 10–15% MAE reduction.
