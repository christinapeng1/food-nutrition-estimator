mkdir -p data/sample/imagery

while IFS= read -r dish_id; do
  mkdir -p "data/sample/imagery/$dish_id"
  gsutil cp \
    "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/$dish_id/rgb.png" \
    "data/sample/imagery/$dish_id/rgb.png"
done < data/sample/sample_dish_ids.txt