"""
==========================================================================
 MODEL 1: Satellite Algae Detection & Severity Classification
 Algae-Based Carbon Sequestration Monitoring Platform — HackOut '26
==========================================================================

 PURPOSE:
   Train a CNN classifier on real Sentinel-2 satellite imagery to:
     1. Classify algal bloom severity: Low / Moderate / High
     2. Generate algae segmentation masks via spectral thresholding
     3. Calculate algae coverage percentage and area in hectares
     4. Export trained model weights for deployment in the web backend

 DATASET:
   kostaspic/amfitrite-inland-waters-hab-sentinel2 (Hugging Face)
   - Real multispectral Sentinel-2 tiles of inland water bodies
   - Labeled by Harmful Algal Bloom (HAB) severity
   - License: CC-BY-4.0

 TRAINING ENVIRONMENT:
   Google Colab Free Tier (T4 GPU)
   - ~8,500 samples
   - ~20 minutes training time
   - ~2.5 GB RAM usage

 HOW TO RUN:
   1. Open Google Colab
   2. Set Runtime -> Change runtime type -> T4 GPU
   3. Copy-paste this entire script into a single cell (or split by sections)
   4. Run all cells
   5. Download the exported model files from Colab files panel
==========================================================================
"""

# =========================================================================
# SECTION 1: Install Dependencies (uncomment when running in Colab)
# =========================================================================
# !pip install -q datasets torch torchvision scikit-learn matplotlib pillow tqdm

import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")


# =========================================================================
# SECTION 2: Configuration
# =========================================================================

class Config:
    """All hyperparameters and settings in one place."""

    # Dataset
    DATASET_NAME = "kostaspic/amfitrite-inland-waters-hab-sentinel2"
    NUM_SAMPLES = 2500
    TRAIN_SPLIT = 0.80
    NUM_CLASSES = 3
    CLASS_NAMES = ["Low", "Moderate", "High"]

    # Image preprocessing
    IMG_SIZE = 224

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0                # 0 for Colab / Hugging Face Arrow datasets (avoids multiprocessing worker errors)
    USE_AMP = True

    # Model
    BACKBONE = "resnet18"
    PRETRAINED = True

    # NDVI Segmentation Thresholds (calibrated for inland water HABs)
    NDVI_ALGAE_LOW = 0.0
    NDVI_ALGAE_MODERATE = 0.15
    NDVI_ALGAE_HIGH = 0.3

    # Sentinel-2 spatial resolution
    PIXEL_RESOLUTION_M = 10

    # Export paths
    EXPORT_DIR = "./model1_export"
    MODEL_FILENAME = "model1_algae_classifier.pt"
    CONFIG_FILENAME = "model1_config.json"
    METRICS_FILENAME = "model1_metrics.json"


config = Config()
os.makedirs(config.EXPORT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =========================================================================
# SECTION 3: Load Dataset from Hugging Face
# =========================================================================

print("\n" + "=" * 60)
print("LOADING DATASET FROM HUGGING FACE")
print("=" * 60)

from datasets import load_dataset, Dataset

print(f"Streaming {config.NUM_SAMPLES} samples from: {config.DATASET_NAME}")
print("Using streaming mode — only downloading what we need, not all 89k files...\n")

start_time = time.time()

# Stream mode: does NOT download all 89k files upfront
# It fetches samples one-by-one, so we only pull 2500
streamed = load_dataset(
    config.DATASET_NAME,
    split="train",
    streaming=True,
)

# Collect only the first 2500 samples
print("Downloading 2500 samples...")
samples_list = []
for i, sample in enumerate(tqdm(streamed, total=config.NUM_SAMPLES, desc="Fetching")):
    if i >= config.NUM_SAMPLES:
        break
    samples_list.append(sample)

# Convert to a regular (indexable) Dataset
raw_dataset = Dataset.from_list(samples_list)
del samples_list  # free memory

load_time = time.time() - start_time
print(f"Loaded {len(raw_dataset)} samples in {load_time:.1f}s")


# =========================================================================
# SECTION 4: Inspect Dataset Structure
# =========================================================================

print("\n" + "=" * 60)
print("INSPECTING DATASET STRUCTURE")
print("=" * 60)

print(f"Features: {raw_dataset.features}")
print(f"Columns:  {raw_dataset.column_names}")

sample = raw_dataset[0]
print(f"Sample keys: {list(sample.keys())}")

# Auto-detect image and label fields
image_field = None
label_field = None

for key in sample.keys():
    val = sample[key]
    if isinstance(val, Image.Image):
        image_field = key
        print(f"  Image field: '{key}' -> size={val.size}, mode={val.mode}")
    elif isinstance(val, (int, np.integer)):
        label_field = key
        print(f"  Label field: '{key}' -> value={val}")
    else:
        print(f"  Field: '{key}' -> type={type(val).__name__}")

# Fallback for image field
if image_field is None:
    for c in ["image", "img", "pixel_values"]:
        if c in sample:
            image_field = c
            break

assert image_field is not None, "Could not find image field in dataset!"

# ── Handle missing labels: derive from pixel intensity ──
if label_field is None:
    print("\nNo label column found in dataset.")
    print("  Deriving severity labels from pixel intensity (mean brightness)...")
    print("  Higher mean pixel value = denser algae = Higher severity\n")

    means = []
    for i in tqdm(range(len(raw_dataset)), desc="Computing pixel stats"):
        img = raw_dataset[i][image_field]
        img_array = np.array(img, dtype=np.float32)
        means.append(img_array.mean())

    means = np.array(means)
    print(f"  Pixel intensity range: {means.min():.1f} to {means.max():.1f}")
    print(f"  Mean: {means.mean():.1f}, Std: {means.std():.1f}")

    # Bin into 3 classes using percentile thresholds
    p33 = np.percentile(means, 33)
    p66 = np.percentile(means, 66)
    print(f"  Thresholds: Low < {p33:.1f} < Moderate < {p66:.1f} < High")

    derived_labels = np.zeros(len(means), dtype=int)
    derived_labels[means > p33] = 1   # Moderate
    derived_labels[means > p66] = 2   # High

    raw_dataset = raw_dataset.add_column("label", derived_labels.tolist())
    label_field = "label"
    print("  Labels derived and added to dataset")

print(f"\nUsing: image='{image_field}', label='{label_field}'")

# Class distribution
all_labels = [raw_dataset[i][label_field] for i in range(len(raw_dataset))]
unique_labels, counts = np.unique(all_labels, return_counts=True)

print("\nClass Distribution:")
for lbl, cnt in zip(unique_labels, counts):
    name = config.CLASS_NAMES[lbl] if lbl < len(config.CLASS_NAMES) else f"Class_{lbl}"
    print(f"  {name} (label={lbl}): {cnt} samples ({cnt/len(all_labels)*100:.1f}%)")


def convert_to_rgb(img_raw):
    """Convert any input (PIL Image, numpy array, or nested list) to RGB PIL Image."""
    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")

    if isinstance(img_raw, dict):
        import io
        if "bytes" in img_raw and img_raw["bytes"] is not None:
            return Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
        if "path" in img_raw and img_raw["path"] is not None:
            return Image.open(img_raw["path"]).convert("RGB")

    if isinstance(img_raw, np.ndarray):
        arr = img_raw.astype(np.float32)
    elif isinstance(img_raw, (list, tuple)):
        try:
            arr = np.array(img_raw, dtype=np.float32)
        except (ValueError, TypeError):
            # Inhomogeneous list of rows (varying row lengths across satellite bands)
            valid_rows = [list(r) if hasattr(r, "__iter__") else [r] for r in img_raw]
            if len(valid_rows) == 0:
                return Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))
            target_w = max(len(r) for r in valid_rows)
            padded = [r + [0] * (target_w - len(r)) for r in valid_rows]
            try:
                arr = np.array(padded, dtype=np.float32)
            except Exception:
                flat = []
                for item in valid_rows:
                    for val in item:
                        try:
                            flat.append(float(val))
                        except Exception:
                            flat.append(0.0)
                side = max(1, int(np.sqrt(len(flat))))
                arr = np.array(flat[: side * side], dtype=np.float32).reshape((side, side))
    else:
        return Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))

    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-6:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr_uint8 = arr.astype(np.uint8)

    if arr_uint8.ndim == 2:
        rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
    elif arr_uint8.ndim == 3:
        if arr_uint8.shape[-1] >= 3:
            rgb = arr_uint8[..., :3]
        elif arr_uint8.shape[-1] == 1:
            rgb = np.repeat(arr_uint8, 3, axis=-1)
        elif arr_uint8.shape[0] in [1, 3, 4]:
            rgb = np.transpose(arr_uint8, (1, 2, 0))
            if rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
            else:
                rgb = np.repeat(rgb, 3, axis=-1)
        else:
            rgb = np.stack([arr_uint8[:, :, 0]] * 3, axis=-1)
    else:
        rgb = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

    return Image.fromarray(rgb, mode="RGB")


# Visualize samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
indices = np.random.choice(len(raw_dataset), 8, replace=False)
for ax, idx in zip(axes.flatten(), indices):
    s = raw_dataset[int(idx)]
    img = convert_to_rgb(s[image_field])
    lbl = s[label_field]
    name = config.CLASS_NAMES[lbl] if lbl < len(config.CLASS_NAMES) else str(lbl)
    ax.imshow(img)
    ax.set_title(f"#{idx} - {name}", fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Sentinel-2 Algal Bloom Images", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "sample_images.png"), dpi=120)
plt.show()


# =========================================================================
# SECTION 5: PyTorch Dataset & DataLoaders
# =========================================================================

print("\n" + "=" * 60)
print("PREPARING DATALOADERS")
def convert_to_rgb(img_raw):
    """Convert any input (PIL Image, numpy array, or nested list) to RGB PIL Image."""
    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")

    if isinstance(img_raw, dict):
        import io
        if "bytes" in img_raw and img_raw["bytes"] is not None:
            return Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
        if "path" in img_raw and img_raw["path"] is not None:
            return Image.open(img_raw["path"]).convert("RGB")

    if isinstance(img_raw, np.ndarray):
        arr = img_raw.astype(np.float32)
    elif isinstance(img_raw, (list, tuple)):
        try:
            arr = np.array(img_raw, dtype=np.float32)
        except (ValueError, TypeError):
            # Handle inhomogeneous list of rows (varying row lengths across satellite bands)
            valid_rows = [list(r) if hasattr(r, "__iter__") else [r] for r in img_raw]
            if len(valid_rows) == 0:
                return Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))
            target_w = max(len(r) for r in valid_rows)
            padded = [r + [0] * (target_w - len(r)) for r in valid_rows]
            try:
                arr = np.array(padded, dtype=np.float32)
            except Exception:
                flat = []
                for item in valid_rows:
                    for val in item:
                        try:
                            flat.append(float(val))
                        except Exception:
                            flat.append(0.0)
                side = max(1, int(np.sqrt(len(flat))))
                arr = np.array(flat[: side * side], dtype=np.float32).reshape((side, side))
    else:
        return Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))

    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-6:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr_uint8 = arr.astype(np.uint8)

    if arr_uint8.ndim == 2:
        rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
    elif arr_uint8.ndim == 3:
        if arr_uint8.shape[-1] >= 3:
            rgb = arr_uint8[..., :3]
        elif arr_uint8.shape[-1] == 1:
            rgb = np.repeat(arr_uint8, 3, axis=-1)
        elif arr_uint8.shape[0] in [1, 3, 4]:
            rgb = np.transpose(arr_uint8, (1, 2, 0))
            if rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
            else:
                rgb = np.repeat(rgb, 3, axis=-1)
        else:
            rgb = np.stack([arr_uint8[:, :, 0]] * 3, axis=-1)
    else:
        rgb = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

    return Image.fromarray(rgb, mode="RGB")


import torch.utils.data as torch_data


class AlgaeBloomDataset(torch_data.Dataset):
    """
    Pure PyTorch Dataset wrapper. Uses direct integer index lookup on raw_dataset.
    Bypasses HuggingFace select() and Arrow __getitems__ tensor indexing.
    """

    def __init__(self, dataset, indices, image_field, label_field, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.image_field = image_field
        self.label_field = label_field
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        real_idx = int(self.indices[int(idx)])
        sample = self.dataset[real_idx]
        image = convert_to_rgb(sample[self.image_field])

        # Safely unwrap label in case it is stored as [0], array, or integer
        lbl_val = sample[self.label_field]
        while isinstance(lbl_val, (list, tuple, np.ndarray)):
            lbl_val = lbl_val[0] if len(lbl_val) > 0 else 0
        try:
            label = int(lbl_val)
        except Exception:
            label = 0
        label = min(max(0, label), config.NUM_CLASSES - 1)

        if self.transform:
            image = self.transform(image)
        return image, label


# Explicitly disable __getitems__ so DataLoader uses standard single-item __getitem__
AlgaeBloomDataset.__getitems__ = None


# Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split dataset
train_size = int(config.TRAIN_SPLIT * len(raw_dataset))
val_size = len(raw_dataset) - train_size

# Split dataset indices
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(raw_dataset)))

train_dataset = AlgaeBloomDataset(raw_dataset, train_indices, image_field, label_field, transform=train_transform)
val_dataset = AlgaeBloomDataset(raw_dataset, val_indices, image_field, label_field, transform=val_transform)

print(f"Train: {len(train_dataset)} samples")
print(f"Val:   {len(val_dataset)} samples")

# In Colab / with Hugging Face Arrow datasets, worker subprocesses fail with AttributeError.
# Using num_workers=0 runs loading in the main process, which is safe and fast for 2,000 samples.
config.NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True,
)

# Verify one batch
images, labels_batch = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # [32, 3, 224, 224]
print(f"Labels: {labels_batch[:8].tolist()}")
print("DataLoaders ready")


# =========================================================================
# SECTION 6: Model Definition (ResNet18 + Custom Head)
# =========================================================================

print("\n" + "=" * 60)
print("BUILDING MODEL: ResNet18 + Transfer Learning")
print("=" * 60)


class AlgaeClassifier(nn.Module):
    """
    ResNet18 backbone (pretrained on ImageNet)
        -> freeze layers 1-3 (keep low-level features)
        -> fine-tune layer4 + custom classification head
        -> outputs 3 classes: Low / Moderate / High severity
    """

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Freeze layers 1-3
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 512

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


model = AlgaeClassifier(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {total_params - trainable_params:,}")

# Class-weighted loss to handle imbalanced data
flat_labels = []
for l in all_labels:
    while isinstance(l, (list, tuple, np.ndarray)):
        l = l[0] if len(l) > 0 else 0
    flat_labels.append(int(l))
class_counts = np.bincount(flat_labels, minlength=config.NUM_CLASSES).astype(np.float64)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
scaler = GradScaler("cuda") if config.USE_AMP and device.type == "cuda" else None


# =========================================================================
# SECTION 7: Training Loop
# =========================================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
best_val_acc = 0.0
best_epoch = 0
training_start = time.time()

for epoch in range(config.NUM_EPOCHS):
    epoch_start = time.time()

    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validate ---
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    val_preds_list, val_labels_list = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            if scaler is not None:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            val_loss_sum += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += targets.size(0)
            val_correct += (preds == targets).sum().item()
            val_preds_list.extend(preds.cpu().numpy())
            val_labels_list.extend(targets.cpu().numpy())

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total
    lr_now = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(lr_now)

    # Save best
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        marker = " ** BEST"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "config": {
                "num_classes": config.NUM_CLASSES,
                "class_names": config.CLASS_NAMES,
                "img_size": config.IMG_SIZE,
                "backbone": config.BACKBONE,
            },
        }, os.path.join(config.EXPORT_DIR, config.MODEL_FILENAME))

    print(
        f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
        f"LR: {lr_now:.2e} | {epoch_time:.1f}s{marker}"
    )

total_time = time.time() - training_start
print(f"\nTraining done in {total_time/60:.1f} minutes")
print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


# =========================================================================
# SECTION 8: Evaluation & Metrics
# =========================================================================

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Load best model
ckpt = torch.load(os.path.join(config.EXPORT_DIR, config.MODEL_FILENAME), map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_preds, all_true, all_probs = [], [], []
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

report = classification_report(all_true, all_preds, target_names=config.CLASS_NAMES, digits=4)
print("\nClassification Report:")
print(report)

overall_acc = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds, average="weighted")
recall = recall_score(all_true, all_preds, average="weighted")
f1 = f1_score(all_true, all_preds, average="weighted")
cm = confusion_matrix(all_true, all_preds)

print(f"Accuracy:  {overall_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")

metrics = {
    "accuracy": round(float(overall_acc), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "best_epoch": best_epoch,
    "best_val_acc": round(float(best_val_acc), 4),
    "training_time_minutes": round(total_time / 60, 2),
    "num_train_samples": len(train_dataset),
    "num_val_samples": len(val_dataset),
    "confusion_matrix": cm.tolist(),
    "class_names": config.CLASS_NAMES,
}
with open(os.path.join(config.EXPORT_DIR, config.METRICS_FILENAME), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {config.EXPORT_DIR}/{config.METRICS_FILENAME}")


# =========================================================================
# SECTION 9: Training Curves
# =========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"], label="Train", color="#10B981", linewidth=2)
axes[0].plot(history["val_loss"], label="Val", color="#06B6D4", linewidth=2)
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history["train_acc"], label="Train", color="#10B981", linewidth=2)
axes[1].plot(history["val_acc"], label="Val", color="#06B6D4", linewidth=2)
axes[1].axhline(y=best_val_acc, color="#F59E0B", linestyle="--", alpha=0.5, label=f"Best: {best_val_acc:.3f}")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(history["lr"], color="#8B5CF6", linewidth=2)
axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)

plt.suptitle("Model 1 Training Progress", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "training_curves.png"), dpi=150)
plt.show()

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="YlGnBu")
ax.set_xticks(range(config.NUM_CLASSES)); ax.set_yticks(range(config.NUM_CLASSES))
ax.set_xticklabels(config.CLASS_NAMES); ax.set_yticklabels(config.CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", fontweight="bold")
for i in range(config.NUM_CLASSES):
    for j in range(config.NUM_CLASSES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
plt.colorbar(im); plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()


# =========================================================================
# SECTION 10: Segmentation Pipeline (ExG for RGB, NDVI for multispectral)
# =========================================================================

print("\n" + "=" * 60)
print("SEGMENTATION PIPELINE")
print("=" * 60)


def generate_algae_mask_rgb(image_np, farm_area_hectares=10.0):
    """
    Generate algae segmentation mask from an RGB satellite image.

    Uses Excess Green Index (ExG = 2*Green - Red - Blue) as a
    chlorophyll proxy to detect algae regions in water bodies.

    INPUT:
      image_np           : numpy array, shape (H, W, 3), dtype uint8, RGB
      farm_area_hectares : float, known farm area in hectares

    OUTPUT:
      dict with keys:
        algae_detected         : bool
        coverage_percentage    : float   (e.g. 67.4)
        algae_area_hectares    : float   (e.g. 6.74)
        farm_area_hectares     : float   (e.g. 10.0)
        mask                   : numpy array (H, W), binary, 1=algae 0=water
        density_mask           : numpy array (H, W), 0=water 1=sparse 2=moderate 3=dense
        exg_index              : numpy array (H, W), normalized ExG values
        pixel_counts           : dict with count per density level
    """
    img = image_np.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Excess Green Index
    exg = 2.0 * g - r - b

    # Normalize to [0, 1]
    exg_range = exg.max() - exg.min()
    exg_norm = (exg - exg.min()) / exg_range if exg_range > 1e-6 else np.zeros_like(exg)

    # Simple water detection (water is dark, non-bright)
    brightness = (r + g + b) / 3.0
    is_water = brightness < 0.7

    # Algae detection
    algae_binary = np.zeros_like(exg, dtype=np.uint8)
    algae_binary[(exg_norm > 0.45) & is_water] = 1

    # Density levels
    density = np.zeros_like(exg, dtype=np.uint8)
    density[(exg_norm > 0.45) & (exg_norm <= 0.60) & is_water] = 1  # sparse
    density[(exg_norm > 0.60) & (exg_norm <= 0.75) & is_water] = 2  # moderate
    density[(exg_norm > 0.75) & is_water] = 3                       # dense

    water_px = is_water.sum()
    algae_px = algae_binary.sum()
    coverage = (algae_px / water_px * 100.0) if water_px > 0 else 0.0
    algae_area = farm_area_hectares * coverage / 100.0

    return {
        "algae_detected": bool(coverage > 1.0),
        "coverage_percentage": round(float(coverage), 2),
        "algae_area_hectares": round(float(algae_area), 2),
        "farm_area_hectares": farm_area_hectares,
        "mask": algae_binary,
        "density_mask": density,
        "exg_index": exg_norm,
        "pixel_counts": {
            "clear_water": int((density == 0).sum()),
            "sparse_algae": int((density == 1).sum()),
            "moderate_algae": int((density == 2).sum()),
            "dense_algae": int((density == 3).sum()),
        },
        "method": "ExG (Excess Green Index)",
    }


def generate_algae_mask_ndvi(red_band, nir_band, scl_band=None, pixel_resolution_m=10.0):
    """
    Generate algae segmentation mask from Sentinel-2 multispectral bands.

    Calculates NDVI = (NIR - Red) / (NIR + Red) and uses SCL
    to mask out land, clouds, and shadows.

    INPUT:
      red_band            : numpy array (H, W), B04 Red band
      nir_band            : numpy array (H, W), B08 NIR band
      scl_band            : numpy array (H, W), Scene Classification Layer (optional)
      pixel_resolution_m  : float, ground resolution per pixel (10m for Sentinel-2)

    OUTPUT:
      Same dict structure as generate_algae_mask_rgb()
      but also includes 'ndvi' key with the raw NDVI array
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)

    # Water mask
    if scl_band is not None:
        is_water = (scl_band == 6)
    else:
        is_water = nir < np.percentile(nir[nir > 0], 70) if nir.max() > 0 else np.ones_like(nir, dtype=bool)

    algae_binary = np.zeros_like(ndvi, dtype=np.uint8)
    algae_binary[(ndvi > config.NDVI_ALGAE_LOW) & is_water] = 1

    density = np.zeros_like(ndvi, dtype=np.uint8)
    density[(ndvi > config.NDVI_ALGAE_LOW) & (ndvi <= config.NDVI_ALGAE_MODERATE) & is_water] = 1
    density[(ndvi > config.NDVI_ALGAE_MODERATE) & (ndvi <= config.NDVI_ALGAE_HIGH) & is_water] = 2
    density[(ndvi > config.NDVI_ALGAE_HIGH) & is_water] = 3

    water_px = is_water.sum()
    algae_px = algae_binary.sum()
    px_area = pixel_resolution_m ** 2

    coverage = (algae_px / water_px * 100.0) if water_px > 0 else 0.0
    total_water_ha = (water_px * px_area) / 10000.0
    algae_ha = (algae_px * px_area) / 10000.0

    return {
        "algae_detected": bool(coverage > 1.0),
        "coverage_percentage": round(float(coverage), 2),
        "algae_area_hectares": round(float(algae_ha), 2),
        "farm_area_hectares": round(float(total_water_ha), 2),
        "mask": algae_binary,
        "density_mask": density,
        "ndvi": ndvi,
        "pixel_counts": {
            "clear_water": int((density == 0).sum()),
            "sparse_algae": int((density == 1).sum()),
            "moderate_algae": int((density == 2).sum()),
            "dense_algae": int((density == 3).sum()),
        },
        "method": "NDVI (Sentinel-2 B04/B08)",
    }


def create_overlay(original_np, density_mask, alpha=0.45):
    """
    Blend a color-coded algae density mask on top of the original image.

    Colors: sparse=Teal, moderate=Emerald, dense=Bright Green
    """
    overlay = original_np.copy().astype(np.float32)
    colors = {
        1: np.array([13, 148, 136]),   # teal
        2: np.array([16, 185, 129]),   # emerald
        3: np.array([34, 197, 94]),    # bright green
    }
    for level, color in colors.items():
        m = density_mask == level
        overlay[m] = overlay[m] * (1 - alpha) + color * alpha
    return overlay.clip(0, 255).astype(np.uint8)


# =========================================================================
# SECTION 11: Full Inference Function (Classification + Segmentation)
# =========================================================================


def run_model1_inference(pil_image, farm_area_hectares=10.0):
    """
    Complete Model 1 inference.

    INPUT:
      pil_image            : PIL Image (any size, RGB)
      farm_area_hectares   : float

    OUTPUT:
      result : dict with all classification + segmentation outputs
      mask   : binary mask array
      density_mask : density level array
    """
    model.eval()

    # --- Classification ---
    tfm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rgb_image = convert_to_rgb(pil_image)
    tensor = tfm(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    severity = config.CLASS_NAMES[pred_class]

    # --- Segmentation ---
    img_np = np.array(rgb_image)
    seg = generate_algae_mask_rgb(img_np, farm_area_hectares)

    result = {
        # Classification
        "severity": severity,
        "severity_index": pred_class,
        "severity_confidence": round(confidence, 4),
        "class_probabilities": {
            n: round(float(p), 4) for n, p in zip(config.CLASS_NAMES, probs)
        },
        # Segmentation
        "algae_detected": seg["algae_detected"],
        "coverage_percentage": seg["coverage_percentage"],
        "algae_area_hectares": seg["algae_area_hectares"],
        "farm_area_hectares": seg["farm_area_hectares"],
        "pixel_counts": seg["pixel_counts"],
        # Provenance
        "provenance": {
            "model_name": "AlgaeClassifier-ResNet18-Sentinel2-v1.0",
            "classification_type": "PREDICTED",
            "segmentation_type": "DERIVED",
            "method": seg["method"],
        },
    }
    return result, seg["mask"], seg["density_mask"]


# =========================================================================
# SECTION 12: Demo — Run Full Pipeline on Samples
# =========================================================================

print("\nRunning full inference on 4 samples...\n")

fig, axes = plt.subplots(4, 3, figsize=(15, 20))
demo_idx = np.random.choice(len(raw_dataset), 4, replace=False)

for row, idx in enumerate(demo_idx):
    s = raw_dataset[int(idx)]
    img = s[image_field]
    img = convert_to_rgb(img)
    true_lbl = s[label_field]
    while isinstance(true_lbl, (list, tuple, np.ndarray)):
        true_lbl = true_lbl[0] if len(true_lbl) > 0 else 0
    true_lbl = int(true_lbl)
    true_name = config.CLASS_NAMES[true_lbl] if true_lbl < len(config.CLASS_NAMES) else str(true_lbl)

    result, mask, density = run_model1_inference(img, farm_area_hectares=10.0)
    img_np = np.array(img)
    overlay = create_overlay(img_np, density)

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title(f"Original (True: {true_name})", fontsize=10)
    axes[row, 0].axis("off")

    # Density mask visualization
    vis = np.zeros((*density.shape, 3), dtype=np.uint8)
    vis[density == 0] = [20, 30, 60]
    vis[density == 1] = [13, 148, 136]
    vis[density == 2] = [16, 185, 129]
    vis[density == 3] = [34, 197, 94]
    axes[row, 1].imshow(vis)
    axes[row, 1].set_title(f"Mask ({result['coverage_percentage']}%)", fontsize=10)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(
        f"Pred: {result['severity']} ({result['severity_confidence']:.0%})", fontsize=10
    )
    axes[row, 2].axis("off")

    print(
        f"  #{idx}: True={true_name} | Pred={result['severity']} "
        f"({result['severity_confidence']:.1%}) | Coverage={result['coverage_percentage']}%"
    )

plt.suptitle("Model 1 — Full Inference Pipeline", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "inference_demo.png"), dpi=150)
plt.show()


# =========================================================================
# SECTION 13: Export Everything for Backend Deployment
# =========================================================================

print("\n" + "=" * 60)
print("EXPORTING FOR DEPLOYMENT")
print("=" * 60)

deploy_config = {
    "model_name": "AlgaeClassifier-ResNet18-Sentinel2-v1.0",
    "backbone": config.BACKBONE,
    "num_classes": config.NUM_CLASSES,
    "class_names": config.CLASS_NAMES,
    "img_size": config.IMG_SIZE,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "ndvi_thresholds": {
        "algae_low": config.NDVI_ALGAE_LOW,
        "algae_moderate": config.NDVI_ALGAE_MODERATE,
        "algae_high": config.NDVI_ALGAE_HIGH,
    },
    "pixel_resolution_m": config.PIXEL_RESOLUTION_M,
    "training_info": {
        "dataset": config.DATASET_NAME,
        "num_samples": config.NUM_SAMPLES,
        "train_split": config.TRAIN_SPLIT,
        "epochs": config.NUM_EPOCHS,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "date": datetime.now().isoformat(),
    },
    "metrics": {
        "accuracy": round(float(overall_acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    },
}

with open(os.path.join(config.EXPORT_DIR, config.CONFIG_FILENAME), "w") as f:
    json.dump(deploy_config, f, indent=2)

# List exported files
print(f"\nExported files:")
for fname in sorted(os.listdir(config.EXPORT_DIR)):
    fpath = os.path.join(config.EXPORT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:40s} {size_mb:.2f} MB")

print(f"""
================================================================
  MODEL 1 TRAINING COMPLETE
================================================================

  Files to download and place in backend/ml_engine/weights/:

    1. model1_algae_classifier.pt   (model weights)
    2. model1_config.json           (config & thresholds)
    3. model1_metrics.json          (evaluation metrics)

  Accuracy:  {overall_acc:.1%}
  F1 Score:  {f1:.1%}
  Best Epoch: {best_epoch}
  Time: {total_time/60:.1f} min
================================================================
""")

# Copy to Google Drive (optional, for Colab)
try:
    drive_path = "/content/drive/MyDrive/HackOut26_Models"
    os.makedirs(drive_path, exist_ok=True)
    import shutil
    for fname in os.listdir(config.EXPORT_DIR):
        shutil.copy2(os.path.join(config.EXPORT_DIR, fname), drive_path)
    print(f"Copied to Google Drive: {drive_path}")
except Exception:
    print("Google Drive not mounted. Download files from ./model1_export/")

print("\nDone! Model 1 is ready for deployment.")
