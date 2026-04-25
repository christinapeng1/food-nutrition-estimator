import sys
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(".")
from dataset import load_metadata, build_ingredient_vocab, Nutrition5kDataset
from cnn import NutritionCNN

# --- Config ---
CSV_PATH      = "../data/raw/metadata/dish_metadata_cafe1.csv"
IMAGERY_ROOT  = "../data/sample/imagery"
CHECKPOINT    = "../checkpoints/best_model.pt"
TEST_IDS_PATH = "../data/raw/dish_ids/splits/rgb_test_ids.txt"
PREDS_CSV     = "../data/predictions.csv"
GT_CSV        = "../data/groundtruth.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load official test split ---
with open(TEST_IDS_PATH) as f:
    test_ids = set(f.read().splitlines())
print(f"Official test IDs: {len(test_ids)}")

# --- Data (must match train.py exactly) ---
df       = load_metadata(CSV_PATH, IMAGERY_ROOT)
vocab    = build_ingredient_vocab(df, top_n=200)
NUM_INGR = len(vocab)

test_df = df[df["dish_id"].isin(test_ids)].reset_index(drop=True)
print(f"Evaluating on {len(test_df)} dishes")

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds     = Nutrition5kDataset(test_df, IMAGERY_ROOT, vocab, transform=val_tf)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

# --- Load model ---
model = NutritionCNN(num_ingredients=NUM_INGR).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Loaded checkpoint")

# --- Collect predictions ---
val_dish_ids = []
cal_preds,  cal_gt  = [], []
mass_preds, mass_gt = [], []
fat_preds,  fat_gt  = [], []
carb_preds, carb_gt = [], []
prot_preds, prot_gt = [], []

with torch.no_grad():
    for imgs, ingr_lbl, mass_vec, dish_gt, dish_ids in test_loader:
        imgs    = imgs.to(device)
        dish_gt = dish_gt.to(device)

        _, _, dish_pred = model(imgs)

        val_dish_ids.extend(dish_ids)

        cal_preds.extend((dish_pred[:, 0]  * 1000).cpu().numpy())
        cal_gt.extend((dish_gt[:, 0]       * 1000).cpu().numpy())

        mass_preds.extend((dish_pred[:, 1] * 1000).cpu().numpy())
        mass_gt.extend((dish_gt[:, 1]      * 1000).cpu().numpy())

        fat_preds.extend((dish_pred[:, 2]  * 200).cpu().numpy())
        fat_gt.extend((dish_gt[:, 2]       * 200).cpu().numpy())

        carb_preds.extend((dish_pred[:, 3] * 200).cpu().numpy())
        carb_gt.extend((dish_gt[:, 3]      * 200).cpu().numpy())

        prot_preds.extend((dish_pred[:, 4] * 200).cpu().numpy())
        prot_gt.extend((dish_gt[:, 4]      * 200).cpu().numpy())

# --- Save CSVs ---
with open(PREDS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(len(val_dish_ids)):
        writer.writerow([
            val_dish_ids[i],
            round(cal_preds[i],  4),
            round(mass_preds[i], 4),
            round(fat_preds[i],  4),
            round(carb_preds[i], 4),
            round(prot_preds[i], 4),
        ])

with open(GT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(len(val_dish_ids)):
        writer.writerow([
            val_dish_ids[i],
            round(cal_gt[i],  4),
            round(mass_gt[i], 4),
            round(fat_gt[i],  4),
            round(carb_gt[i], 4),
            round(prot_gt[i], 4),
        ])

print(f"Saved predictions → {PREDS_CSV}")
print(f"Saved groundtruth → {GT_CSV}")
print("\nNow run:")
print("  python ../data/raw/scripts/compute_eval_statistics.py \\")
print("      ../data/groundtruth.csv ../data/predictions.csv ../data/eval_results.json")