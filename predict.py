#!/usr/bin/env python3
# -*- coding: utf-8-*-
# 2026-07-09 16:26
# Ruan Ding Wang Yong


import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from CNN_model import CNN

# Device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# Parameters
IMAGE_SIZE = (231, 310)
BATCH_SIZE = 16
MODEL_PATH = "cnn_model_3_dropout.pth"
DATASET_PATH = "cwt_images"


# Dataset
def create_test_loader():

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=transform,
    )

    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader, class_names


# Load Model
def load_model():

    model = CNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    print("Model loaded successfully.\n")

    return model


# Prediction
def predict(model, test_loader):

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


# Result
def show_results(labels, preds, class_names):

    print("=" * 50)
    print("Prediction Results")
    print("=" * 50)

    for i in range(min(20, len(preds))):

        print(
            f"Image {i+1:02d} | "
            f"True: {class_names[labels[i]]:<8} | "
            f"Predict: {class_names[preds[i]]}"
        )

    print("\nClassification Report\n")

    print(
        classification_report(
            labels,
            preds,
            target_names=class_names,
        )
    )


# Confusion Matrix
def plot_confusion_matrix(labels, preds, class_names):

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    plt.show()


# Main
def main():

    test_loader, class_names = create_test_loader()

    model = load_model()

    labels, preds = predict(model, test_loader)

    show_results(labels, preds, class_names)

    plot_confusion_matrix(labels, preds, class_names)


if __name__ == "__main__":
    main()
