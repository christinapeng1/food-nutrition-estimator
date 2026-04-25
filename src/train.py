import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(".")
from dataset import load_metadata, build_ingredient_vocab, Nutrition5kDataset
from cnn import NutritionCNN

# --- Config ---
CSV_PATH       = "../data/raw/metadata/dish_metadata_cafe1.csv"
IMAGERY_ROOT   = "../data/sample/imagery"
CHECKPOINT     = "../checkpoints/best_model.pt"
TRAIN_IDS_PATH = "../data/raw/dish_ids/splits/rgb_train_ids.txt"
TEST_IDS_PATH  = "../data/raw/dish_ids/splits/rgb_test_ids.txt"
NUM_INGR       = None
BATCH_SIZE     = 16
EPOCHS         = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4

# Loss weights
LAMBDA_CLS  = 1.0
LAMBDA_MASS = 0.5
LAMBDA_REG  = 2.0

os.makedirs("../checkpoints", exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load official splits ---
with open(TRAIN_IDS_PATH) as f:
    train_ids = set(f.read().splitlines())
with open(TEST_IDS_PATH) as f:
    test_ids = set(f.read().splitlines())

print(f"Official train IDs: {len(train_ids)}")
print(f"Official test IDs:  {len(test_ids)}")

# --- Data ---
df       = load_metadata(CSV_PATH, IMAGERY_ROOT)
vocab    = build_ingredient_vocab(df, top_n=200)
NUM_INGR = len(vocab)
print(f"Actual vocab size: {NUM_INGR}")
print(f"Total dishes with imagery: {len(df)}")

with open("../checkpoints/vocab.json", "w") as f:
    json.dump(vocab, f)

# Split by official IDs
train_df = df[df["dish_id"].isin(train_ids)].reset_index(drop=True)
test_df  = df[df["dish_id"].isin(test_ids)].reset_index(drop=True)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = Nutrition5kDataset(train_df, IMAGERY_ROOT, vocab, transform=train_tf)
test_ds  = Nutrition5kDataset(test_df,  IMAGERY_ROOT, vocab, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Model ---
model     = NutritionCNN(num_ingredients=NUM_INGR).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
bce_loss  = nn.BCEWithLogitsLoss()
huber     = nn.HuberLoss(delta=0.5)


def compute_loss(ingr_logits, ingr_mass_pred, dish_pred,
                 ingr_labels,  ingr_mass_gt,  dish_gt):
    l_cls  = bce_loss(ingr_logits, ingr_labels)
    mask   = ingr_labels.bool()
    l_mass = huber(ingr_mass_pred[mask], ingr_mass_gt[mask]) if mask.any() else torch.tensor(0.0).to(device)
    l_reg  = huber(dish_pred, dish_gt)
    return LAMBDA_CLS * l_cls + LAMBDA_MASS * l_mass + LAMBDA_REG * l_reg


# --- Training loop ---
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for imgs, ingr_lbl, mass_gt, dish_gt, _ in train_loader:
        imgs, ingr_lbl, mass_gt, dish_gt = [t.to(device) for t in (imgs, ingr_lbl, mass_gt, dish_gt)]
        optimizer.zero_grad()
        ingr_logits, ingr_mass_pred, dish_pred = model(imgs)
        loss = compute_loss(ingr_logits, ingr_mass_pred, dish_pred,
                            ingr_lbl, mass_gt, dish_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validate on test set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, ingr_lbl, mass_gt, dish_gt, _ in test_loader:
            imgs, ingr_lbl, mass_gt, dish_gt = [t.to(device) for t in (imgs, ingr_lbl, mass_gt, dish_gt)]
            ingr_logits, ingr_mass_pred, dish_pred = model(imgs)
            val_loss += compute_loss(ingr_logits, ingr_mass_pred, dish_pred,
                                     ingr_lbl, mass_gt, dish_gt).item()

    train_loss /= len(train_loader)
    val_loss   /= len(test_loader)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | test={val_loss:.4f} | ✓ saved")
    else:
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | test={val_loss:.4f}")

print(f"\nDone. Best test loss: {best_val_loss:.4f}")
print(f"Model saved to {CHECKPOINT}")