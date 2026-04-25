import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_metadata(csv_path, imagery_root):
    """Parse metadata CSV, keeping only dishes that have an rgb.png on disk."""
    available = set(os.listdir(imagery_root))

    dishes = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            dish_id = parts[0]
            if dish_id not in available:
                continue
            if not os.path.exists(f"{imagery_root}/{dish_id}/rgb.png"):
                continue

            dish = {
                "dish_id":   dish_id,
                "calories":  float(parts[1]),
                "mass_g":    float(parts[2]),
                "fat_g":     float(parts[3]),
                "carb_g":    float(parts[4]),
                "protein_g": float(parts[5]),
            }

            ingredients = []
            offset = 6
            while offset + 6 < len(parts):
                if not parts[offset].startswith("ingr_"):
                    break
                ingredients.append({
                    "ingr_id":   parts[offset],
                    "ingr_name": parts[offset + 1],
                    "grams":     float(parts[offset + 2]),
                })
                offset += 7
            dish["ingredients"] = ingredients
            dishes.append(dish)

    return pd.DataFrame(dishes)


def build_ingredient_vocab(df, top_n=100):
    """Build vocabulary of the top_n most frequent ingredients."""
    counts = {}
    for ingrs in df["ingredients"]:
        for ingr in ingrs:
            counts[ingr["ingr_name"]] = counts.get(ingr["ingr_name"], 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    vocab = [name for name, _ in top]
    print(f"Vocab size: {len(vocab)}, most common: {vocab[:5]}")
    return vocab


class Nutrition5kDataset(Dataset):
    def __init__(self, df, imagery_root, vocab, transform=None):
        self.df           = df.reset_index(drop=True)
        self.imagery_root = imagery_root
        self.vocab        = {name: i for i, name in enumerate(vocab)}
        self.transform    = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = f"{self.imagery_root}/{row['dish_id']}/rgb.png"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Dish-level regression targets
        reg_targets = torch.tensor([
            row["calories"]  / 1000.0,
            row["mass_g"]    / 1000.0,
            row["fat_g"]     / 200.0,
            row["carb_g"]    / 200.0,
            row["protein_g"] / 200.0,
        ], dtype=torch.float32)

        # Ingredient presence (multi-label)
        ingr_vec = torch.zeros(len(self.vocab))
        # Per-ingredient mass
        mass_vec = torch.zeros(len(self.vocab))

        for ingr in row["ingredients"]:
            name = ingr["ingr_name"]
            if name in self.vocab:
                i = self.vocab[name]
                ingr_vec[i] = 1.0
                mass_vec[i] = ingr["grams"] / 500.0

        return img, ingr_vec, mass_vec, reg_targets, row["dish_id"]