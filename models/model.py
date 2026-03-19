#!/usr/bin/env python3
"""Train a neural network to predict March Madness game outcomes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import HIDDEN_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DATA_DIR


def load_data(train_csv, test_csv):
    """Load training and testing CSVs into PyTorch TensorDatasets."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = torch.tensor(train_df.iloc[:, :-2].values, dtype=torch.float32)
    Y_train = torch.tensor(train_df.iloc[:, -2:].values, dtype=torch.float32)

    X_test = torch.tensor(test_df.iloc[:, :-2].values, dtype=torch.float32)
    Y_test = torch.tensor(test_df.iloc[:, -2:].values, dtype=torch.float32)

    return TensorDataset(X_train, Y_train), TensorDataset(X_test, Y_test)


class MarchMadnessNN(nn.Module):
    """Fully connected network for predicting game win probabilities."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the model and save a loss curve plot."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, torch.max(Y_batch, 1)[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss)

        if epoch % 10 == 0:
            accuracy = evaluate_model(model, test_loader)
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.savefig("training_loss.png")
    plt.close()


def evaluate_model(model, test_loader):
    """Return accuracy percentage on the test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            predicted = torch.argmax(Y_pred, dim=1)
            actual = torch.argmax(Y_batch, dim=1)
            correct += (predicted == actual).sum().item()
            total += Y_batch.size(0)

    return (correct / total) * 100


if __name__ == "__main__":
    train_csv = os.path.join(MODEL_DATA_DIR, "training_data.csv")
    test_csv = os.path.join(MODEL_DATA_DIR, "testing_data.csv")
    train_dataset, test_dataset = load_data(train_csv, test_csv)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_size = train_dataset[0][0].shape[0]
    model = MarchMadnessNN(input_size)

    train_model(model, train_loader, test_loader)

    final_accuracy = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    torch.save(model.state_dict(), "model.pth")
