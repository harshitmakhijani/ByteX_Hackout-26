"""
Standalone test script for Model 1 (Sentinel-2 Algae Classifier & Segmentation).
Usage:
    python ml/test_model1.py <path_to_image>
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "model1_algae_classifier.pt")
CONFIG_PATH = os.path.join(WEIGHTS_DIR, "model1_config.json")

# Fallback to root if not in weights
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model1_algae_classifier.pt")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(BASE_DIR, "..", "model1_config.json")

# --- Load Config ---
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CLASS_NAMES = config.get("class_names", ["Low", "Moderate", "High"])
IMG_SIZE = config.get("img_size", 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model Architecture ---
class AlgaeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# --- Load Model Weights ---
model = AlgaeClassifier(num_classes=len(CLASS_NAMES)).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
print(f"Loaded Model 1 from {MODEL_PATH} on {device}")


def convert_to_rgb(img_raw):
    """Normalize and convert any image mode or array to RGB PIL Image."""
    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")
    arr = np.array(img_raw, dtype=np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-6:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr_uint8 = arr.astype(np.uint8)
    if arr_uint8.ndim == 2:
        rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
    else:
        rgb = arr_uint8[..., :3] if arr_uint8.shape[-1] >= 3 else np.repeat(arr_uint8, 3, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def segment_algae_rgb(img_np, farm_area_ha=10.0):
    """ExG chlorophyll segmentation."""
    img = img_np.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2.0 * g - r - b
    exg_range = exg.max() - exg.min()
    exg_norm = (exg - exg.min()) / exg_range if exg_range > 1e-6 else np.zeros_like(exg)
    is_water = ((r + g + b) / 3.0) < 0.7

    algae_binary = np.zeros_like(exg, dtype=np.uint8)
    algae_binary[(exg_norm > 0.45) & is_water] = 1

    density = np.zeros_like(exg, dtype=np.uint8)
    density[(exg_norm > 0.45) & (exg_norm <= 0.60) & is_water] = 1
    density[(exg_norm > 0.60) & (exg_norm <= 0.75) & is_water] = 2
    density[(exg_norm > 0.75) & is_water] = 3

    water_px = is_water.sum()
    algae_px = algae_binary.sum()
    coverage = (algae_px / water_px * 100.0) if water_px > 0 else 0.0
    algae_ha = farm_area_ha * coverage / 100.0

    return {
        "coverage_percentage": round(float(coverage), 2),
        "algae_area_hectares": round(float(algae_ha), 2),
        "mask": algae_binary,
        "density_mask": density,
    }


def run_inference(image_input, farm_area_ha=10.0):
    """Full inference pipeline on an image path or PIL image."""
    if isinstance(image_input, str):
        pil_img = Image.open(image_input)
    else:
        pil_img = image_input

    rgb_img = convert_to_rgb(pil_img)

    # Classification
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tfm(rgb_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    # Segmentation
    img_np = np.array(rgb_img)
    seg = segment_algae_rgb(img_np, farm_area_ha)

    return {
        "severity": CLASS_NAMES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "coverage_percentage": seg["coverage_percentage"],
        "algae_area_hectares": seg["algae_area_hectares"],
        "class_probabilities": {
            name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probs)
        },
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"\nAnalyzing: {test_path}...")
        res = run_inference(test_path)
        print("\n--- Model 1 Results ---")
        print(f"Severity Prediction : {res['severity']} ({res['confidence']}%)")
        print(f"Algae Coverage      : {res['coverage_percentage']}%")
        print(f"Algae Area          : {res['algae_area_hectares']} ha (on 10 ha farm)")
        print(f"Class Probabilities : {res['class_probabilities']}")
    else:
        print("Run with an image path: python ml/test_model1.py <image_path>")
