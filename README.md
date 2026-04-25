# food-nutrition-estimator

## Set up Virtual Environment
```
git clone https://github.com/christinapeng1/food-nutrition-estimator.git
cd food-nutrition-estimator

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Try the Model
```
python predict.py [path_to_image]
```

## Repository Structure
```
food-nutrition-estimator/
├── checkpoints/
│   └── best_model.pt          # saved model weights (best validation loss)
├── data/
│   ├── raw/
│   │   ├── dish_ids/          # official Nutrition5k train/test split IDs
│   │   ├── metadata/          # dish_metadata_cafe1.csv and cafe2.csv
│   │   └── scripts/           # official Nutrition5k evaluation script
│   └── sample/
│       ├── imagery/           # downloaded rgb.png images per dish
│       └──available_dish_ids.txt  # all dish IDs with realsense overhead images
├── food_photos/               # test images for inference
├── scripts/
│   ├── clean_dish_ids.py      # cleans gsutil ls output to plain dish IDs
│   └── download_available_dishes.sh  # downloads rgb.png for all available dishes
├── src/
│   ├── cnn.py                 # EfficientNet-B3 multi-head model architecture
│   ├── dataset.py             # dataset loading, metadata parsing, vocab building
│   ├── train.py               # training loop with official train/test splits
│   ├── evaluate.py            # generates predictions.csv and groundtruth.csv
│   └── predict.py             # run inference on a single food image
├── .gitignore
├── README.md
└── requirements.txt
```

