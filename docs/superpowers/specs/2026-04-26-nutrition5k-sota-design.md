# Nutrition5k SOTA — Design Spec

**Date**: 2026-04-26
**Owner**: xcong2 (Brown CCV)
**Status**: Approved (brainstorming complete; pending implementation plan)

---

## 1. Goal & Success Criteria

Beat the Google Nutrition5k paper's "Direct Prediction" baseline on the official RGB test split, using overhead RGB-D imagery only.

**Reference baseline (Thames et al., CVPR 2021)**: calorie MAE ≈ 70 kcal, mass MAE ≈ 40 g, fat ≈ 6 g, carbs ≈ 10 g, protein ≈ 5 g.

**Targets:**
- Hard floor (must beat): calorie MAE ≤ 70 kcal AND mass MAE ≤ 40 g
- Stretch goal: calorie MAE ≤ 60 kcal AND mass MAE ≤ 35 g
- Final report includes per-target MAE + %MAE, ingredient F1 (macro + micro), per-ingredient mass MAE, top-5 ingredient accuracy, all with 95% bootstrap CI.

## 2. Constraints

- **Compute**: single A6000 (48 GB) on `gpu2803`; `module load cuda` available
- **Wall-clock budget**: target 1–2 days. Total GPU-hour ceiling = 30 h (= 3 main-run trains × ~6 h + 2 ablations × ~6 h). If main run hits the hard floor on the first attempt, retrain budget is reallocated to ablations.
- **Modality**: RGB + depth (depth_raw.png from realsense_overhead). Side-angle videos out of scope.
- **Data scope**: rgb_train_ids ∩ available = 2755 dishes; rgb_test_ids ∩ available = 507 (of 709 official). Reported on the 507 subset; explicitly disclosed in report.
- **External data**: ImageNet-22k pretraining allowed (ConvNeXt weights). No external food datasets in this run.
- **No GPU subagents**: training is single-process. Subagents are only for code authoring, code review, downloads, and doc generation.

## 3. Architecture

### 3.1 Overview

Dual-stream RGB-D late fusion → 3 prediction heads + auxiliary derived-kcal path.

```
RGB (3,224,224) ──► ConvNeXt-Base (ImageNet-22k) ──► feat_rgb (1024)
                                                                │
Depth+Mask (2,224,224) ──► ConvNeXt-Tiny (RGB-pretrained) ──► feat_d (768)
                                                                │
                                       concat ─► MLP(1792→512) ─► z (512)
                                                                       │
       ┌───────────────────────────────────────────────────────────────┤
       ▼                       ▼                       ▼
  Reg head A (5):         Cls head (555):       Reg head B (555):
  kcal/mass/fat/          ingredient            per-ingredient mass
  carbs/protein           multi-label           (masked supervision)
  Linear 512→5            Linear 512→555        Linear 512→555
```

### 3.2 Key choices

1. **Late fusion (not 4-channel concat)**: depth has different statistics + missing-value handling than RGB; late fusion lets each stream keep its own normalization and inductive bias, and enables clean "no-depth" ablation by zeroing `feat_d`.
2. **Depth encoder is Tiny, not Base**: depth has lower information density than RGB; Tiny avoids overfit. First conv adapted from 3-channel RGB-pretrained weights via channel-mean (then duplicated to 2 channels for depth+mask).
3. **Per-ingredient mass head**: 555-way regression with **masked Huber loss** — only penalize positions where the dish actually contains that ingredient. Mass labels are log1p-z-score normalized to handle long tail.
4. **Two kcal paths** (consistency-trained, ensembled at inference):
   - Direct: head A's kcal output (inverse z-score → raw kcal)
   - Derived: `kcal_derived = Σ_i inv_log1p_zscore(pred_mass_i) × density_i`, where `density_i` (cal/g) comes from `ingredients_metadata.csv`. The mass head output must be inverse-transformed (z-score → log1p → expm1) back to raw grams before density multiplication.
   - **Reported kcal** (in §6 metrics): 50/50 average of direct and derived. Direct-only and derived-only are also logged for diagnostic purposes but the headline number uses the average.
5. **Total params** ≈ 120 M; batch=64 RGB+Depth at 224² fits comfortably in 48 GB with bf16.

## 4. Data Pipeline

### 4.1 Splits

| Split | Official | With overhead imagery | Used as |
|---|---|---|---|
| Train | 4059 | 2755 | 2480 train + 275 val (90/10 dish-level) |
| Test | 709 | 507 | held-out test, evaluated once |

### 4.2 Depth download

Sequential `gsutil cp` is hours; use parallel `gcloud storage cp` driven by `xargs -P 16`:

```bash
awk '{print "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/"$1"/depth_raw.png"}' \
    data/sample/available_dish_ids.txt | \
    xargs -P 16 -I{} sh -c 'src={}; dst=data/sample/imagery/$(basename $(dirname $src))/depth_raw.png; gcloud storage cp "$src" "$dst"'
```

Expected wall-clock: 5–15 min. Run this in a background subagent while main thread writes code.

### 4.3 Preprocessing

**RGB**: PIL → [0,1] → ImageNet mean/std → resize 256 → train RandomResizedCrop(0.7-1.0)→224, val CenterCrop 224.

**Depth**: 16-bit raw PNG (mm) → float32 → clip [200, 800] mm → produce `valid_mask = (depth > 0)` → z-score using train-set statistics (cached) → synced spatial transforms with RGB. Final tensor shape (2, 224, 224) = stack(depth_normalized, valid_mask).

**Missing depth dish**: skip (decision logged); should be 0 if download is clean.

### 4.4 Augmentation (train only)

| Aug | Scope | Strength |
|---|---|---|
| RandomResizedCrop (0.7–1.0) | RGB+Depth synced | default |
| HorizontalFlip | RGB+Depth synced | p=0.5 |
| ColorJitter | RGB | brightness/contrast/saturation 0.2 |
| RandAugment | RGB | n=2, m=9 |
| Depth random scale | Depth | × U(0.95, 1.05) |
| MixUp / CutMix | — | DISABLED (breaks physical mass/kcal semantics) |

### 4.5 Labels (per dish)

```python
y_scalar      = (kcal, mass, fat, carb, protein)      # (5,) z-scored from train stats
y_ingr_binary = one_hot(present_ids)                  # (555,) ∈ {0,1}
y_ingr_mass   = log1p(grams) z-scored                 # (555,) NaN where absent
y_ingr_mask   = (~isnan(y_ingr_mass)).float()         # (555,) ∈ {0,1}
```

**Vocab**: 555 ingredients from `ingredients_metadata.csv`, fixed order (ascending by `id`). Saved to `checkpoints/v2/<run_id>/vocab.json`. **Compatibility check** with existing `checkpoints/vocab.json` is part of Gate G1 (must match or document the diff).

## 5. Loss & Training

### 5.1 Per-task losses

| Name | Input | Formula | Notes |
|---|---|---|---|
| `L_scalar` | 5 z-scored scalars | Huber(δ=1.0) | Robust to outliers |
| `L_ingr_cls` | 555 logits | BCE-with-logits + pos_weight | Sparse multi-label, pos_weight from train freq |
| `L_ingr_mass` | 555 mass | masked Huber(δ=1.0) on mask=1 only | log1p-z-scored target |
| `L_atwater` | derived kcal vs label | \|kcal_pred − (9·fat + 4·carb + 4·protein)\| in raw kcal units | Soft physical regularizer |
| `L_kcal_consist` | direct vs derived kcal | Huber in raw kcal units | Couples head A and head B |

### 5.2 Joint loss — uncertainty weighting (Kendall et al. 2018)

Each task t has a learnable log-variance `s_t`:

```
L = Σ_t [ (1 / (2 · exp(s_t))) · L_t + 0.5 · s_t ]
```

Tasks: {scalar, ingr_cls, ingr_mass, atwater, kcal_consist}. Initialize `s_t = 0`. Trained jointly with the network. **Failure mode** (one task collapses, `s_t → +∞`): apply a floor `s_t ≥ -2` clamp; if collapse persists → fixed-weight fallback (manual 1.0/1.0/1.0/0.1/0.5).

### 5.3 Optimizer & schedule

| Item | Value |
|---|---|
| Optimizer | AdamW, weight_decay=0.05 |
| LR | 3e-4 (heads) / 3e-5 (backbone) — 2 param groups |
| Schedule | Linear warmup 5% of total steps → cosine decay to 0 |
| Batch size | 64 (grad accum to 128 if val plateau) |
| Epochs | 50 max, early stop patience=10 on val combined score |
| Weight EMA | decay=0.9999, EMA weights used for eval/test |
| Precision | bf16 (A6000 sm_86 native) |
| Seed | 42 main run; ablation matches |

### 5.4 Validation & model selection

- Val combined score = mean of per-target z-score MAE on 5 scalars
- Save EMA + raw weights at best val combined score
- Test runs **once** at the end (no test-set tuning)

### 5.5 Test-time augmentation (TTA)

HFlip + 3-crop (center, top-left, bottom-right) → 6 forward passes → average scalars and ingredient logits. Per-dish prediction = TTA-averaged output. Reported numbers use TTA + EMA.

## 6. Evaluation Protocol

Reported on rgb_test_ids ∩ available (n=507):

| Metric | Unit | Target |
|---|---|---|
| Calorie MAE / %MAE | kcal / % | ≤ 60 kcal stretch, ≤ 70 hard floor |
| Mass MAE / %MAE | g / % | ≤ 35 g stretch, ≤ 40 g hard floor |
| Fat / Carb / Protein MAE | g | report all |
| Ingredient F1 (macro+micro) | — | report; threshold = 0.5 |
| Per-ingredient mass MAE | g | computed only at GT-positive positions |
| Top-5 ingredient accuracy | % | top-5 pred IoU with GT set |

All metrics report mean + 95% bootstrap CI (n=1000).

Outputs:
- `data/v2/predictions.csv` (per-dish prediction)
- `data/v2/groundtruth.csv` (per-dish GT)
- `data/v2/eval_results.json` (full metric table + CI)

## 7. Ablation Plan

**Budget**: 1 main run (~6 h) + 1–2 ablations (~6 h each). Combined with the §2 retrain ceiling (3 main-run trains), worst-case total = 30 GPU-h.

| Priority | Ablation | What changes | Expected info |
|---|---|---|---|
| ★ 1 (must) | **No depth** | Forward sets `feat_d = 0`; depth encoder frozen + zeroed | Quantifies depth contribution; validates Q4 decision |
| 2 (if time) | No per-ingredient mass head | Remove head B + L_ingr_mass + L_kcal_consist | Quantifies whether decomposition is the win |
| 3 (skip) | No Atwater consistency | weight=0 | Lower priority; effect likely small |
| 4 (skip) | No TTA | center-crop only inference | Engineering only |
| 5 (skip) | EfficientNet-B3 backbone | Match `best_model.pt` backbone | Optional sanity |

Each ablation re-runs Gates G1–G4. Significance: paired bootstrap (n=1000) on per-dish absolute errors; report ΔMAE and 95% CI of Δ.

## 8. Correctness Gates

| Gate | When | Pass criterion | On fail |
|---|---|---|---|
| **G1** | dataset.py written | Visualize 5 random samples (RGB+Depth+labels) — manual inspection passes; vocab matches existing or diff documented; per-ingredient masses sum to total ± 1 g | Fix code, retry |
| **G2** | model.py written | Dummy forward (B=2) → output shapes match spec; backward no NaN; param count ≈ 120 M ± 10% | Fix architecture |
| **G3** | train.py written | **Overfit 8-dish micro-batch**: train loss → 0 in ≤ 100 iters; grad norm < 100; LR schedule prints sanely | Fix loss/optim |
| **G4** | Main run epoch 1 end | Train loss decreasing; val z-score MAE < 1.0; pred vs GT scatter not collapsed to constant; no NaN | Stop training, diagnose |
| **G5** | Main run complete | Test calorie MAE ≤ 70 kcal | Diagnose, retrain (≤ 2 retries left) |
| **G6** | Each ablation complete | G1–G4 pass + paired bootstrap test vs main | Fix config, rerun |

## 9. Iteration Loop (G5 fail recovery)

Diagnose order, ranked by empirical frequency:

1. Uncertainty `s_t` collapsed for some task → check `s_t` curves; clamp floor or switch to fixed-weight
2. Backbone LR off → grad-norm ratio backbone:head off by >100× → retune
3. Label/data misalignment → rerun G1 on 50 dishes, double-check shuffle/seed
4. Depth not contributing → PCA(feat_d) → if 2D viz is noise, freeze depth encoder for 5 epochs warmup
5. Per-ingredient mass head sparse-supervision slow → bump mass loss weight, try `mass_loss_weight × (mask sum / 555)` normalization

Each retry must produce `docs/runs/<run_id>/diagnosis_<n>.md` with: hypothesis, change set, expected delta, actual delta. **No "try a different seed" retries.**

Hard cap: 3 main-run trains. If still > 70 kcal after 3, freeze and report what we have.

## 10. Execution Playbook

### 10.1 Repository layout

```
src/v2/
├── model.py
├── dataset.py
├── losses.py
├── train.py
├── evaluate.py
├── viz.py
└── configs/{main.yaml, ablation_no_depth.yaml}
docs/runs/<run_id>/{config.yaml, train.log, eval.json, summary.md, diagnosis_*.md}
docs/ablations/<name>/summary.md
checkpoints/v2/<run_id>/{best.pt, ema.pt, vocab.json}
```

### 10.2 Subagent strategy

GPU is single, so training stays sequential. Subagents handle parallel non-GPU work:

| Phase | Parallelism | Subagent jobs |
|---|---|---|
| 0. Env | background | gcloud SDK install + auth + parallel depth download |
| 1. Code authoring | 4× parallel | dataset.py, model.py, losses.py, evaluate.py — each authored by an isolated subagent given a self-contained slice of this spec |
| 2. Integration + review | 1× | code-reviewer subagent reads merged repo against spec; flags discrepancies |
| 3. Training | 0 | main thread on GPU, gates inline |
| 4. Doc-while-train | 1× | doc subagent reads training log + writes summary.md while training continues |
| 5. Ablation | 0 | main thread serial training; doc subagent writes ablation summary in parallel |

Subagent rules:
- Self-contained prompts (paste relevant spec sections; don't rely on conversation context)
- No GPU usage in subagents
- Failure → I read the error and decide; no auto-retry
- Code-author subagents must include unit tests for their module that pass on CPU

### 10.3 Documentation discipline

| Trigger | File | Required content |
|---|---|---|
| Phase begin | `docs/runs/<phase>.md` (top) | Goal, expected output |
| Phase end | same file (bottom) | Actual output, deviations, next |
| Each train run | `docs/runs/<run_id>/summary.md` | Config, test metrics, failure modes, next |
| Each diagnosis | `docs/runs/<run_id>/diagnosis_<n>.md` | Hypothesis, changes, observed delta |
| Each ablation | `docs/ablations/<name>/summary.md` | What was disabled, ΔMAE table, significance |
| Final | `docs/final_report.md` | Headline table, ablation table, qualitative panel, limitations |

## 11. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| Depth download fails or rate-limits | Retry once with smaller `-P`; if persistent, fall back to RGB-only main run + skip ablation |
| `vocab.json` from old checkpoint differs from new build | Document diff in G1; new run uses new vocab; old checkpoint kept as `predict.py` legacy path |
| Uncertainty weights collapse | s_t floor clamp + fixed-weight fallback; documented in §5.2 |
| Per-ingredient mass head doesn't learn | log1p + masked loss + warmup; if still flat after 5 epochs, weight↑ or freeze head A briefly |
| Out-of-distribution wild photos (e.g., `food_photos/omurice.png`) | Acknowledged: this spec optimizes Nutrition5k benchmark, not wild generalization. Logged as future work. |
| 1–2 day budget overrun | Hard cap 3 main-run retrains; ablation list trimmed in priority order |

## 12. Out of Scope

- Side-angle video frames (not downloaded; would 6× data but breaks compute budget)
- Foundation-model VLM fine-tuning (DINOv2-L, SigLIP-2, Qwen-VL, etc.)
- Self-supervised pretraining on Nutrition5k itself
- True 3D volume estimation from depth (geometric portion estimation)
- The 202 test dishes without overhead RGB-D (videos-only)
- Ensembling beyond TTA

## 13. Deliverables

1. Code: `src/v2/` complete, with unit tests for dataset/model/losses passing on CPU
2. Trained checkpoints: main + at least 1 ablation, EMA weights, in `checkpoints/v2/`
3. Reports: `docs/runs/`, `docs/ablations/`, and a top-level `docs/final_report.md`
4. Reproducibility: every run has `config.yaml` and a recorded `train.log`; seeds fixed
