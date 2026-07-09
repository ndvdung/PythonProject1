#!/usr/bin/env python3
# -*- coding: utf-8-*-
# 2026-07-09 13:57
# Ruan Ding Wang Yong


import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# hyperpara declare
BATCH_SIZE = 16

LR = 0.001

EPOCHS = 35

IMAGE_SIZE = (231, 310)

# generic transform
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# dataset declaration
dataset = datasets.ImageFolder(root="cwt_images", transform=transform)

print(dataset.classes)

print(dataset.class_to_idx)

# split
train_size = int(0.8 * len(dataset))

test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# CNN
import torch
import torch.nn as nn


class CNN(nn.Module):
    """三层卷积神经网络"""

    def __init__(self, num_classes=4):
        super(CNN, self).__init__()

        # -------------------------
        # 第一层卷积
        # -------------------------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # 第二层卷积
        # -------------------------
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # 第三层卷积
        # -------------------------
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # 全连接层
        # -------------------------
        self.flatten = nn.Flatten()

        # Input image: 231 × 310
        # After 3 MaxPool(2×2):
        # 231 -> 115 -> 57 -> 28
        # 310 -> 155 -> 77 -> 38
        self.fc1 = nn.Linear(3 * 28 * 38, 512)

        self.relu_fc = nn.ReLU()

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """前向传播"""

        # 第一层
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 第三层
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # 展平
        x = self.flatten(x)

        # 全连接
        x = self.fc1(x)
        x = self.relu_fc(x)

        # 输出层
        x = self.fc2(x)

        return x


# create model
model = CNN().to(device)

print(model)

summary(model, input_size=(1, 3, 231, 310))

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)


# training and testing
train_loss_history = []

test_loss_history = []

train_acc_history = []

test_acc_history = []

# epoch
for epoch in range(EPOCHS):

    # training

    model.train()

    train_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, pred = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    train_acc = correct / total

    train_loss /= len(train_loader)

    # evaluating

    model.eval()

    test_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (pred == labels).sum().item()

    test_acc = correct / total

    test_loss /= len(test_loader)

    # append list

    train_loss_history.append(train_loss)

    test_loss_history.append(test_loss)

    train_acc_history.append(train_acc)

    test_acc_history.append(test_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" Train Loss={train_loss:.4f}"
        f" Test Loss={test_loss:.4f}"
        f" Train Acc={train_acc:.4f}"
        f" Test Acc={test_acc:.4f}"
    )


# loss curve
plt.figure(figsize=(8, 5))

plt.plot(train_loss_history, label="Train")

plt.plot(test_loss_history, label="Test")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("Loss_1.png")
plt.show()


# accu curve
plt.figure(figsize=(8, 5))

plt.plot(train_acc_history, label="Train")

plt.plot(test_acc_history, label="Test")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("Accu_1.png")
plt.show()
