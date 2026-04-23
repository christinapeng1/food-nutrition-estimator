import pandas as pd
import numpy as np

dishes = []
with open("data/raw/metadata/dish_metadata_cafe1.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        dishes.append({
            "dish_id":   parts[0],
            "mass_g":    float(parts[1]),
            "calories":  float(parts[2]),
            "fat_g":     float(parts[3]),
            "carb_g":    float(parts[4]),
            "protein_g": float(parts[5]),
        })

df = pd.DataFrame(dishes)

# Filter out obvious outliers
df = df[df["calories"] < 2000]

# Sample 100 dishes evenly across calorie range (stratified)
df["calorie_bin"] = pd.cut(df["calories"], bins=10)
sample = (
    df.groupby("calorie_bin", observed=True)
      .apply(lambda x: x.sample(min(len(x), 10), random_state=42))
      .reset_index(drop=True)
      .head(100)
)

print(f"Sample size: {len(sample)}")
print(sample["calories"].describe())

# Save the dish IDs we want to download
sample["dish_id"].to_csv("data/sample/sample_dish_ids.txt", index=False, header=False)
print("\nSaved to data/sample/sample_dish_ids.txt")