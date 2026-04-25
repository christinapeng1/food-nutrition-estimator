import torch
import torch.nn as nn
import torchvision.models as models


class NutritionCNN(nn.Module):
    def __init__(self, num_ingredients):
        super().__init__()

        # Backbone
        backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
        in_feats = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.encoder = backbone

        # Shared fully connected layers
        self.shared = nn.Sequential(
            nn.Linear(in_feats, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Head 1: ingredient presence (multi-label classification)
        self.head_ingr_cls  = nn.Linear(1024, num_ingredients)

        # Head 2: per-ingredient mass (regression)
        self.head_ingr_mass = nn.Linear(1024, num_ingredients)

        # Head 3: dish-level nutrition (calories, mass, fat, carb, protein)
        self.head_dish_reg  = nn.Linear(1024, 5)

    def forward(self, x):
        feats        = self.encoder(x)
        h            = self.shared(feats)
        ingr_logits  = self.head_ingr_cls(h)
        ingr_mass    = self.head_ingr_mass(h)
        dish_reg     = self.head_dish_reg(h)
        return ingr_logits, ingr_mass, dish_reg