# food-nutrition-estimator

Multi-task RGB-D nutrition estimation on the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset.

**Headline**: 4 of 5 scalar metrics beat the Google Direct Prediction baseline (Thames et al., CVPR 2021), trained in 14 minutes on a single A6000.

| Metric | Ours | 95% CI | Google baseline | Δ |
|---|---|---|---|---|
| **kcal MAE** | **61.9** | [56.0, 67.8] | 70 | **−11.5%** ✅ |
| **mass MAE** | **37.9** | [34.3, 41.7] | 40 | **−5.3%** ✅ |
| **fat MAE** | **4.7** | [4.3, 5.2] | 6 | **−21.7%** ✅ |
| **carb MAE** | **6.2** | [5.6, 6.7] | 10 | **−38.5%** ✅ |
| protein MAE | 6.0 | [5.3, 6.6] | 5 | +20% (slightly worse) |

n=507 dishes (rgb_test_ids ∩ available overhead RGB-D).

## Method

Dual-stream ConvNeXt-Base (RGB) + ConvNeXt-Tiny (Depth, 2-channel adapted) → late fusion → 3 prediction heads:
- 5 nutrition scalars (kcal, mass, fat, carb, protein)
- 555-way ingredient multi-label classification
- per-ingredient mass regression

Multi-task uncertainty weighting (Kendall et al. 2018), with several engineering fixes — see [docs/technical_report_en.md](docs/technical_report_en.md) (English) or [docs/technical_report_zh.md](docs/technical_report_zh.md) (Chinese) for the full design and findings, including a non-trivial ablation result (depth helps mass but hurts kcal).

## Quickstart

### 1. Environment

```bash
git clone https://github.com/Oliver-Cong02/food-nutrition-estimator.git
cd food-nutrition-estimator

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# torch must match the GPU driver — pick the cu* index that matches `nvidia-smi`
nvidia-smi --query-gpu=driver_version --format=csv,noheader
.venv/bin/python -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.11.0 torchvision==0.26.0
```

### 2. Data

```bash
# Metadata + dish_ids (small)
mkdir -p data/raw
gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/{metadata,dish_ids,scripts} data/raw/

# Available overhead-RGB-D dishes
mkdir -p data/sample
gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/ | \
    sed 's|gs://.*/||;s|/||' | grep ^dish_ > data/sample/available_dish_ids.txt

# RGB + Depth (~2.5 GB, parallel curl, ~1 minute)
bash scripts/download_rgb.sh
bash scripts/download_depth.sh
.venv/bin/python scripts/verify_depth.py
```

### 3. Train + evaluate

```bash
# 90/10 train/val split + train statistics
.venv/bin/python scripts/build_splits.py
.venv/bin/python scripts/compute_train_stats.py

# Main training (~14 min on A6000)
.venv/bin/python -m src.v2.train --config src/v2/configs/main.yaml

# No-depth ablation (~10 min)
.venv/bin/python -m src.v2.train --config src/v2/configs/ablation_no_depth.yaml

# Evaluate on the held-out test split
.venv/bin/python -m src.v2.evaluate \
    --checkpoint checkpoints/v2/main_seed42/best.pt \
    --vocab      checkpoints/v2/main_seed42/vocab.json \
    --stats      checkpoints/v2/main_seed42/train_stats.json \
    --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
    --output-dir docs/runs/main_seed42/eval/

# Unit tests (38 should pass on CPU)
.venv/bin/python -m pytest tests/v2/ -v
```

## Repository layout

```
src/v2/                         # core code
├── vocab.py / stats.py         # 555-ingredient vocab + z-score helpers
├── dataset.py                  # RGB-D dataset + augmentation
├── model.py                    # dual-stream ConvNeXt + 3 heads
├── losses.py                   # 5 task losses + UncertaintyWeighter
├── metrics.py                  # MAE, F1, top-k IoU, paired bootstrap CI
├── tta.py                      # 6-pass test-time augmentation
├── evaluate.py                 # eval CLI
├── train.py                    # training loop (AMP + EMA + cosine + G4 gate)
├── viz.py                      # sanity & scatter plots
└── configs/{main, ablation_no_depth}.yaml
tests/v2/                       # 38 unit tests
scripts/                        # download, stats, G1 sanity
docs/
├── final_report.md             # English headline report
├── technical_report_en.md      # full English technical report
├── technical_report_zh.md      # 中文详细技术报告
├── superpowers/{specs, plans}  # design spec & 22-task implementation plan
├── runs/<run_id>/              # per-run config, train.log, eval/
└── ablations/<name>/           # bootstrap significance
```

## Pretrained weights

Not committed (LFS not used). Reproduce them via §3 above — training takes ~14 minutes on a single A6000.

## License

For research / educational use only. Built on top of the Nutrition5k dataset (Google Research) which has its own license terms.
