with open("../data/sample/available_dish_ids.txt") as f:
    lines = f.read().splitlines()

clean_ids = []
for line in lines:
    dish_id = line.rstrip("/").split("/")[-1]
    if dish_id.startswith("dish_"):
        clean_ids.append(dish_id)

print(f"Found {len(clean_ids)} clean dish IDs")

with open("../data/sample/available_dish_ids.txt", "w") as f:
    for dish_id in clean_ids:
        f.write(dish_id + "\n")

print("Saved clean IDs to available_dish_ids.txt")