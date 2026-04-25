import sys
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(".")
from dataset import load_metadata, build_ingredient_vocab
from cnn import NutritionCNN

# --- Config ---
CSV_PATH     = "../data/raw/metadata/dish_metadata_cafe1.csv"
IMAGERY_ROOT = "../data/sample/imagery"
CHECKPOINT   = "../checkpoints/best_model.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load vocab (must match training) ---
df       = load_metadata(CSV_PATH, IMAGERY_ROOT)
vocab    = build_ingredient_vocab(df, top_n=200)
NUM_INGR = len(vocab)

# --- Load model ---
model = NutritionCNN(num_ingredients=NUM_INGR).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Model loaded\n")

# --- Transform ---
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x   = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        ingr_logits, ingr_mass_pred, dish_pred = model(x)

    # Dish-level nutrition (un-normalize)
    calories = dish_pred[0, 0].item() * 1000
    mass     = dish_pred[0, 1].item() * 1000
    fat      = dish_pred[0, 2].item() * 200
    carbs    = dish_pred[0, 3].item() * 200
    protein  = dish_pred[0, 4].item() * 200

    # Ingredients above threshold
    probs      = torch.sigmoid(ingr_logits[0])
    mass_preds = ingr_mass_pred[0]
    all_ingrs  = [(vocab[i], probs[i].item(), mass_preds[i].item() * 500)
                  for i in range(len(vocab))]
    all_ingrs.sort(key=lambda x: -x[1])
    top5 = all_ingrs[:5]
    # detected   = [(vocab[i], probs[i].item(), mass_preds[i].item() * 500)
    #               for i in range(len(vocab)) if probs[i].item() > 0.35]
    # detected.sort(key=lambda x: -x[1])

    # Print results
    print("=" * 40)
    print(f"  Calories: {calories:.1f} kcal")
    print(f"  Mass:     {mass:.1f} g")
    print(f"  Fat:      {fat:.1f} g")
    print(f"  Carbs:    {carbs:.1f} g")
    print(f"  Protein:  {protein:.1f} g")
    print("=" * 40)
    # print(f"  Detected {len(detected)} ingredients:")
    # for name, prob, mass_g in detected:
    #     print(f"    {name:<25} {prob:.0%} confidence  ~{mass_g:.1f}g")
    print("  Top 5 ingredients:")
    for name, prob, mass_g in top5:
        print(f"    {name:<25} {prob:.0%} confidence  ~{mass_g:.1f}g")
    print("=" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your/image.jpg")
    else:
        predict(sys.argv[1])