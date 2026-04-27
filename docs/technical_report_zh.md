# Nutrition5k SOTA 项目技术报告

**目标**：在 Nutrition5k 数据集上超越 Google 论文 (Thames et al., CVPR 2021) 的 Direct Prediction 基线。

**结果**：5 个标量指标中 **4 个超越基线**（kcal 61.9 vs 70 / mass 37.9 vs 40 / fat 4.7 vs 6 / carb 6.2 vs 10），单卡 A6000 训练 14 分钟。

---

## 目录

1. [任务与目标](#1-任务与目标)
2. [数据](#2-数据)
3. [预处理与增广](#3-预处理与增广)
4. [模型架构](#4-模型架构)
5. [损失函数](#5-损失函数)
6. [训练](#6-训练)
7. [评测](#7-评测)
8. [主要结果](#8-主要结果)
9. [Ablation](#9-ablation)
10. [关键工程坑点](#10-关键工程坑点)
11. [局限与未来方向](#11-局限与未来方向)
12. [复现](#12-复现)

---

## 1. 任务与目标

**输入**：餐盘俯视 RGB-D 图像。

**输出**：
- 5 个标量：kcal / mass / fat / carb / protein
- 555 维多标签食材分类
- 每个出现食材的克数

**基线**（Google Nutrition5k Direct Prediction，InceptionV2 ~25M）：

| Metric | 基线 | 硬性下限 | 拓展目标 |
|---|---|---|---|
| kcal MAE | 70 | ≤ 70 | ≤ 60 |
| mass MAE | 40 g | ≤ 40 | ≤ 35 |
| fat / carb / protein | 6 / 10 / 5 g | report all | — |

**资源**：单 A6000 (48 GB) on Brown CCV `gpu2803`，1-2 天预算。

---

## 2. 数据

### 2.1 Nutrition5k

- 来源：公开 GCS `gs://nutrition5k_dataset/`
- 总规模：5006 道菜，**只有 3490 道**有 realsense_overhead 视角
- 标注：工业级食物秤测量的 kcal/mass/macros + 食材列表 + 每食材克数

### 2.2 数据切片

| Split | 官方 | 与 available 交集 |
|---|---|---|
| Train | 4059 | 2755 → 拆 90/10 → 2479 train + 276 val |
| Test | 709 | **507**（最终评测） |

只用 overhead RGB-D，不用 side_angle 视频（时间预算原因）。

### 2.3 关键工程：并行 depth 下载

原作业要求串行 `gsutil cp`，3490 文件 ≈ 1 小时。我们的方案：

```bash
awk '...' dish_list.txt | xargs -P 16 -I LINE bash -c '
  IFS=$'"'"'\t'"'"' read -r src dst <<< "LINE"
  [ -f "$dst" ] && [ -s "$dst" ] && exit 0
  curl -sSL -o "$dst" "$src"
'
```

**关键技巧**：
- 公共 HTTPS (`storage.googleapis.com/nutrition5k_dataset/...`) 不需要 gcloud auth
- `xargs -P 16` 16 路并行
- 幂等（已下载跳过）
- `IFS=$'\t'` 用 ANSI-C quoting（**不是** `$"\t"` — 后者是 i18n 字面量）

**实测**：3490 文件 35 秒。1 个 dish (`dish_1564159636`) 在 GCS 上是 0 字节文件（数据集自身 corruption），最终 3489/3490 可用。

---

## 3. 预处理与增广

### 3.1 RGB

```
PNG → PIL → [0,1] → ImageNet mean/std → resize 256×256
                          ↓
                训练: RandomResizedCrop(0.7-1.0) → 224
                评测: CenterCrop 224
```

**RGB 与 Depth 必须 resize 到同一尺寸**，否则同步 crop 坐标系不一致。

### 3.2 Depth（含**关键修正**）

实测 30 dish 的 valid 像素分布：p1=3021, p50=3571, p99=4867 mm。RealSense 餐桌摄像头距离餐盘 3-5 米。

**Spec 原写 clip [200, 800] mm 是错的**（手机 ToF 范围）。修正后 `[2500, 6000]` mm：

```
16-bit PNG (480, 640, mm)
  → valid_mask = (depth > 0)
  → clip(depth, 2500, 6000) * mask    # 无效像素=0
  → resize 256×256（depth: bilinear, mask: nearest）
  → 与 RGB 同步 crop 224×224
  → (depth - mean) / (std + 1e-6) * mask
  → stack([depth_norm, mask], dim=0)  →  (2, 224, 224)
```

**如不修正**：96% 像素被截饱和，depth 信号丢失。

### 3.3 标签

| Key | Shape | 含义 |
|---|---|---|
| `y_scalar` | (5,) | (kcal, mass, fat, carb, protein) **z-score** |
| `y_scalar_raw` | (5,) | 原始单位（评测用） |
| `y_ingr_binary` | (555,) | 多标签 one-hot |
| `y_ingr_mass` | (555,) | log1p(grams) z-score；不存在=0 |
| `y_ingr_mask` | (555,) | 0/1 |

**为什么 mass 用 log1p**：单食材克数长尾（多数 < 5g，少数 > 100g），直接 z-score 大值会主导损失。

### 3.4 增广

| 增广 | 范围 | 强度 |
|---|---|---|
| RandomResizedCrop | RGB+Depth 同步 | scale=(0.7, 1.0) |
| HorizontalFlip | RGB+Depth 同步 | p=0.5 |
| ColorJitter | 仅 RGB | 0.2 |
| RandAugment | 仅 RGB | n=2, m=9 |
| Depth random scale | 仅 Depth | × U(0.95, 1.05) |
| MixUp / CutMix | — | **禁用**（破坏物理量语义） |

### 3.5 z-score 统计

`scripts/compute_train_stats.py` 一次性算好存到 `train_stats.json`：

```json
{
  "scalar_mean": [253.95, 215.69, 12.71, 19.33, 17.85],
  "scalar_std":  [222.19, 163.47, 13.44, 22.99, 20.19],
  "depth_mean":  3709.5,  "depth_std":  371.4,
  "mass_log1p_mean": 2.205,  "mass_log1p_std": 1.623
}
```

---

## 4. 模型架构

### 4.1 整体

```
RGB (3, 224, 224) ─► ConvNeXt-Base (89M, ImageNet-1K)
                  ─► AvgPool + LN              ─► feat_rgb (1024)
                                                          │
Depth+Mask (2, 224, 224) ─► ConvNeXt-Tiny (28M, 第一层 conv 改 2 通道)
                          ─► AvgPool + LN       ─► feat_d (768)
                                                          │
                          concat → MLP(1792→512) → z (512)
                                                          │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        head_scalar       head_ingr        head_mass
        (512→5)           (512→555)        (512→555)
```

**总参数量**：116.9 M

### 4.2 关键设计

**1. Late fusion（不是 4-channel concat）**：RGB / depth / mask 统计与缺失值不同；分流保留各自归纳偏置；方便 "no-depth" ablation（feat_d 置零）。

**2. Depth encoder 用 Tiny**：depth 信息密度低；避免在 2479 样本上过拟合。

**3. 第一层 conv channel-mean 初始化**：

```python
mean_w = w.mean(dim=1, keepdim=True)   # (out, 3, kh, kw) → (out, 1, kh, kw)
new_w = mean_w.repeat(1, 2, 1, 1)      # (out, 2, kh, kw)
```

保持预训练权重的"风格"。

**4. 三个 head**：

| Head | 输出 | 损失 | 监督 |
|---|---|---|---|
| `head_scalar` | (5,) | Huber | 全部 |
| `head_ingr` | (555,) logits | BCE+pos_weight | 全部 |
| `head_mass` | (555,) | masked Huber | 仅 GT 存在的食材 |

**5. 双路 kcal（设计 vs 现实）**：

- **设计原意**：`kcal_direct` (head_scalar) 与 `kcal_derived = Σ mass×density` 在训练时互相约束，评测取 50/50 平均
- **现实问题**：初始化时 mass head 输出每个 slot ≈ 8g，555 slots × 8g × 2 cal/g = ~9000 kcal。`kcal_consist` loss 比其他大 4 个数量级
- **修正**：训练时用 GT mask 限制求和；评测用预测 mask（sigmoid > 0.5）；headline kcal **只用 direct**（derived 路径需 F1 ≥ 0.6 才能有用）

---

## 5. 损失函数

### 5.1 5 个任务

| Task | 公式 |
|---|---|
| `L_scalar` | Huber(δ=1.0) on 5 维 z-score |
| `L_ingr_cls` | BCE-with-logits + pos_weight |
| `L_ingr_mass` | **masked** Huber，仅 GT-positive 位置 |
| `L_atwater` | smooth_l1(direct_kcal, 9·fat+4·carb+4·protein) **/ std_kcal** |
| `L_kcal_consist` | smooth_l1(direct, derived) **/ std_kcal** |

### 5.2 关键修正：除以 std_kcal

`L_atwater` 在 raw kcal 单位（reasonable），但梯度反传过 z-score 链 `*std_kcal ≈ 222`，所以梯度比 z-score Huber 大 200 倍。除以 std_kcal 让两个 std 抵消，5 个 loss 量级一致。

### 5.3 多任务联合：Uncertainty Weighting (Kendall 2018)

每任务一个可学习 log-variance：

$$L = \sum_t \left[\frac{L_t}{2 e^{s_t}} + \frac{s_t}{2}\right]$$

- `s_floor = -2` 钳制防 `exp(-s)` 上溢
- Weighter LR = `lr_head × 0.1 = 3e-5`，**不参与 cosine schedule**

### 5.4 BCE pos_weight

555 类多标签稀疏（平均每菜 5-7 食材）。每类 `(neg/pos)` 频率比作权重，cap 100 防梯度爆炸。

---

## 6. 训练

### 6.1 超参

| 项 | 值 |
|---|---|
| Optimizer | AdamW, weight_decay=0.05 |
| LR (backbone / head / weighter) | 3e-5 / 3e-4 / 3e-5 |
| Schedule | warmup 5% → cosine（仅 backbone+head） |
| Batch | 64 |
| Epochs | 50 |
| Mixed precision | bf16 |
| Grad clip | max_norm=5.0 |
| EMA decay | 0.9999 (**实际未用**，见 §10) |
| Seed | 42 |

### 6.2 训练循环关键

**NaN 检查必须在 backward() 前**：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    total, parts = compute_total_loss(...)

if not torch.isfinite(total):
    logger.error("NaN/Inf — aborting"); return

(total / cfg.grad_accum).backward()
```

否则 NaN 已污染所有 `.grad`，optimizer 写出 NaN 权重，训练静默崩溃。

**G4 gate**：epoch 0 末若 val_score ≥ 1.0（接近随机），立即停训。

**Atomic checkpoint**：tmp → rename，防 SLURM 抢占时半写。

### 6.3 6 个 Correctness Gates

| Gate | 时机 | 检查 |
|---|---|---|
| G1 | dataset 写完 | 5-dish 可视化 + per-ingredient mass sum ≥ 45/50 |
| G2 | model 写完 | dummy forward shape；param 116.9M ± 10% |
| G3 | train 写完 | 8-dish overfit，loss → 0 |
| G4 | epoch 1 末 | val z-score MAE < 1.0 |
| G5 | 主 run 完 | test kcal MAE ≤ 70 |
| G6 | 每 ablation 完 | G1-G4 重跑 + paired bootstrap |

---

## 7. 评测

### 7.1 流程

1. 加载 `best.pt`（**model 权重，不是 EMA**，详见 §10）
2. eval transform（CenterCrop，无随机）
3. **TTA**：每 dish 6 forward = 3 crops × 2 flips，平均
4. 反 z-score
5. ingredient 阈值 sigmoid > 0.5
6. headline kcal 只用 direct

### 7.2 报告指标

5 标量 MAE / %MAE / 95% bootstrap CI；食材 F1 micro+macro；top-5 IoU；per-ingredient mass MAE。

---

## 8. 主要结果

### 8.1 训练曲线

| Epoch | val_huber_z |
|---|---|
| 0 | 0.4955 |
| 10 | 0.4363 |
| 30 | 0.3228 |
| 49 | **0.2456** (−50%) |

val 单调下降，结束时仍在降 — **可以再训更久**。

### 8.2 测试集（n=507）

| Metric | 我们 | 95% CI | Google 基线 | Δ |
|---|---|---|---|---|
| **kcal MAE** | **61.9** | [56.0, 67.8] | 70 | **−11.5%** ✅ |
| **mass MAE** | **37.9** | [34.3, 41.7] | 40 | **−5.3%** ✅ |
| **fat MAE** | **4.7** | [4.3, 5.2] | 6 | **−21.7%** ✅ |
| **carb MAE** | **6.2** | [5.6, 6.7] | 10 | **−38.5%** ✅ |
| protein MAE | 6.0 | [5.3, 6.6] | 5 | +20% (略差) |

**4/5 超基线**，两个硬性下限都达到。

辅助：食材 F1 micro=0.331 / macro=0.263，top-5 IoU=0.230，per-ingredient mass MAE=21.4 g。

---

## 9. Ablation

### 9.1 设计

只跑 1 个 ablation：**no-depth**。配置与 main 完全相同，仅 `use_depth=false`，`forward()` 时 `feat_d` 置零。

### 9.2 结果（paired bootstrap, n=1000）

| Metric | Main (RGB+D) | No-Depth | Δ | 95% CI | 显著? |
|---|---|---|---|---|---|
| kcal MAE | 61.9 | **57.6** | −4.2 | [−6.9, −1.6] | ★ depth **HURTS** |
| mass MAE | **37.9** | 46.6 | +8.6 | [+5.9, +11.3] | ★ depth **HELPS** |
| fat / carb / protein | — | — | small | — | not sig |

### 9.3 物理解释

- **Mass 受益于 depth**（+8.6g, 23%）：质量是体积量，depth 给直接几何信号
- **Kcal 反被拖累**（−4.2 kcal）：kcal 主要由**食材身份**决定（沙拉 vs 炸饭，密度大致定）。RGB 颜色纹理是身份强信号。50 epoch 内，depth stream 与 RGB stream 在 fusion MLP 上**竞争**
- **Macros 不受影响**：由食材身份决定，depth 不变识别

### 9.4 启示

未来应**head-specific depth gating**：depth 只进 mass head，弱化对 kcal head 的影响。

---

## 10. 关键工程坑点

| # | 坑 | 修复 | 教训 |
|---|---|---|---|
| 1 | Depth clip [200,800] 错误（手机 ToF 范围） | 改 [2500, 6000] mm | 写 spec 先**测数据** |
| 2 | mass head 评测 5× bloated（555 slots × 8g 默认） | 训练 GT mask + 评测 sigmoid mask + headline 只用 direct | 评测前看 prediction 分布 |
| 3 | Raw-units loss 梯度被 std_kcal 放大 200× | 除以 std_kcal | 所有 loss 同量级 |
| 4 | torch cu130 wheel 装上但 driver 是 12.9 | 装 cu129 wheel | 先 nvidia-smi 看 driver |
| 5 | `IFS=$"\t"` 不是 tab | 用 `IFS=$'\t'` | bash 注意 ANSI-C quoting |
| 6 | EMA decay 0.9999 太慢（窗口 10000 步 vs 训 1900 步） | 评测用 raw model | 短训练用 0.999 |
| 7 | Overfit_micro 用了增广 + cosine | 强制无增广 + val=train + 常数 LR | overfit test 关掉所有 noise |
| 8 | ssh + tee 长连接断开导致 log 中断 | nohup + 远程 log + ssh tail -F | 长任务完全 detach |

---

## 11. 局限与未来方向

### 11.1 局限

- Test 只用 507/709（缺失 202 个无 overhead）
- 单 seed
- 50 epoch 偏短（曲线还在降）
- EMA decay 设错没用上
- Derived kcal pathway 没生效
- Wild OOD photos 完全没优化
- 食材 F1 偏低 (0.33)

### 11.2 改进方向

**低成本高回报**：
- 训更长（100-200 epochs）
- EMA decay 改 0.999
- Head-specific depth gating
- BCE 改 Focal loss

**中等成本**：
- Backbone 升 DINOv2-L / SigLIP-2-L
- 加 side_angle 视频数据
- Per-ingredient mass 改 sequence head

**研究方向**：
- VLM fine-tune (Qwen2.5-VL / InternVL3)
- 显式 3D 体积估计 → density × volume
- Self-supervised pretrain on Nutrition5k

---

## 12. 复现

### 12.1 环境

```bash
git clone <repo> && cd food-nutrition-estimator
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# torch 必须匹配 driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader
.venv/bin/python -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.11.0 torchvision==0.26.0
```

### 12.2 数据

```bash
mkdir -p data/raw && cd data/raw
gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/{metadata,dish_ids,scripts} .
cd ../..

gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/ | \
    sed 's|gs://.*/||;s|/||' | grep ^dish_ > data/sample/available_dish_ids.txt

bash scripts/download_rgb.sh        # ~1 GB
bash scripts/download_depth.sh      # ~1.5 GB, 35 秒
.venv/bin/python scripts/verify_depth.py
```

### 12.3 训练 + 评测

```bash
# split + stats
.venv/bin/python scripts/build_splits.py
.venv/bin/python scripts/compute_train_stats.py

# 主训练 (14 分钟)
.venv/bin/python -m src.v2.train --config src/v2/configs/main.yaml

# 评测
.venv/bin/python -m src.v2.evaluate \
    --checkpoint checkpoints/v2/main_seed42/best.pt \
    --vocab      checkpoints/v2/main_seed42/vocab.json \
    --stats      checkpoints/v2/main_seed42/train_stats.json \
    --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
    --output-dir docs/runs/main_seed42/eval/

# 单元测试
.venv/bin/python -m pytest tests/v2/ -v   # 38 passed
```

### 12.4 仓库结构

```
src/v2/         # vocab/stats/dataset/model/losses/metrics/tta/evaluate/train/viz + configs/
tests/v2/       # 38 单元测试
scripts/        # 下载、stats、G1 sanity
docs/
├── final_report.md           # 英文最终报告
├── technical_report_zh.md    # 本文档
├── superpowers/{specs, plans} # 设计文档与实施计划
├── runs/<run_id>/             # config + train.log + eval/
└── ablations/<name>/
checkpoints/v2/<run_id>/       # best.pt + vocab.json + train_stats.json
```

---

## 总结

- **结果**：4/5 指标超 Google 基线，单卡 14 分钟
- **方法**：双流 ConvNeXt-RGBD + 多任务 + uncertainty weighting + TTA
- **最大启发**：depth 帮助 mass 但伤害 kcal — head-specific gating 是下一步
- **诚实披露**：derived kcal 未真正用上、protein 略差、单 seed、50 epoch 偏短

下一步最低成本高回报：训更长 + 修 EMA + head-specific depth gating，预期再降 10-15% MAE。
