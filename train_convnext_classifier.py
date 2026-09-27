"""
train_convnext_classifier.py
----------------------------

Fine-tune ConvNeXt Tiny on soybean disease dataset.

Outputs:
--------
best_convnext_model.pth
training_curves.png
classification_report.txt
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report


# ==========================================================
# PATHS
# ==========================================================

DATA_DIR = r"C:\Users\sayan_dey\Desktop\Sayan Dey - 4th Semester\Srinka Maam Approach\datasets\closeup\Final Dataset"

OUTPUT_DIR = r"D:/soybean_disease_detection/convnext_training"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

EPOCHS = 50

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")

# ==========================================================
# SET SEED
# ==========================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ==========================================================
# DATASET
# ==========================================================


class SoybeanDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.class_names = sorted(
            [
                folder
                for folder in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, folder))
            ]
        )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        self.samples = []

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)

            label = self.class_to_idx[class_name]

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(class_dir, file_name)

                    self.samples.append((image_path, label))

        print(f"\nFound {len(self.class_names)} classes")
        print(f"Found {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================================
# TRANSFORMS
# ==========================================================

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ==========================================================
# DATASET
# ==========================================================

dataset = SoybeanDataset(DATA_DIR, transform=train_transform)

# ==========================================================
# TRAIN / VAL SPLIT
# ==========================================================

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

print(f"\nTrain samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ==========================================================
# DATALOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ==========================================================
# LOAD CONVNEXT
# ==========================================================

print("\nLoading ConvNeXt Tiny...")

weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(weights=weights)

# ==========================================================
# MODIFY CLASSIFIER
# ==========================================================

num_classes = len(dataset.class_names)

in_features = model.classifier[2].in_features

model.classifier[2] = nn.Linear(in_features, num_classes)

model = model.to(device)

print("Model loaded.")

# ==========================================================
# LOSS
# ==========================================================

criterion = nn.CrossEntropyLoss()

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================================
# SCHEDULER
# ==========================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================================================
# TRAINING STORAGE
# ==========================================================

train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []

best_val_accuracy = 0.0

best_model_weights = copy.deepcopy(model.state_dict())

# ==========================================================
# TRAINING LOOP
# ==========================================================

print("\nStarting training...")

for epoch in range(EPOCHS):
    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # ======================================================
    # TRAIN
    # ======================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = correct / total

    # ======================================================
    # VALIDATION
    # ======================================================

    model.eval()

    running_val_loss = 0.0

    val_correct = 0

    val_total = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    val_loss = running_val_loss / len(val_loader)

    val_accuracy = val_correct / val_total

    scheduler.step()

    # ======================================================
    # STORE
    # ======================================================

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)

    val_accuracies.append(val_accuracy)

    # ======================================================
    # PRINT
    # ======================================================

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(model.state_dict())

        model_path = os.path.join(OUTPUT_DIR, "best_convnext_model.pth")

        torch.save(model.state_dict(), model_path)

        print("\nBest model updated.")

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

model.load_state_dict(best_model_weights)

# ==========================================================
# FINAL REPORT
# ==========================================================

report = classification_report(
    all_labels, all_preds, target_names=dataset.class_names, digits=4
)

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)

print(report)

# ==========================================================
# SAVE REPORT
# ==========================================================

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

with open(report_path, "w") as f:
    f.write(report)

# ==========================================================
# PLOT CURVES
# ==========================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# ==========================================================
# LOSS CURVE
# ==========================================================

plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_losses, label="Train Loss")

plt.plot(epochs_range, val_losses, label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

# ==========================================================
# ACCURACY CURVE
# ==========================================================

plt.subplot(1, 2, 2)

plt.plot(epochs_range, train_accuracies, label="Train Accuracy")

plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.tight_layout()

curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")

plt.savefig(curve_path)

plt.close()

# ==========================================================
# DONE
# ==========================================================

print("\n=================================================")
print("CONVNEXT TRAINING COMPLETED")
print("=================================================")

print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}")

print("\nSaved files:")

print(f"  {model_path}")
print(f"  {report_path}")
print(f"  {curve_path}")
