#!/usr/bin/env python3
# -*- coding: utf-8-*-
# 2026-07-09 15:26
# Ruan Ding Wang Yong


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# Import CNN class
from CNN_model import CNN

# Device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

IMAGE_SIZE = (231, 310)
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Load dataset

dataset = datasets.ImageFolder("cwt_images", transform=transform)

class_names = dataset.classes

print("Classes:", class_names)

# Split train / test
# (same ratio as training)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Load model
model = CNN().to(device)

model.load_state_dict(torch.load("cnn_model_3_dropout.pth", map_location=device))

model.eval()

print("Model loaded successfully.\n")


# Prediction
all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())


# Show first 20 predictions
print("=" * 50)
print("Prediction Results")
print("=" * 50)

for i in range(min(20, len(all_preds))):

    print(
        f"Image {i+1:02d} | "
        f"True: {class_names[all_labels[i]]:<8} | "
        f"Predict: {class_names[all_preds[i]]}"
    )


# Classification Report
print("\nClassification Report\n")

print(classification_report(all_labels, all_preds, target_names=class_names))


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
