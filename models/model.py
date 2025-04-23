#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

def load_data(train_csv, test_csv):
    # Load CSVs into Pandas
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract features (X) and labels (Y)
    X_train = train_df.iloc[:, :-2].values  # All columns except last two (Win probabilities)
    Y_train = train_df.iloc[:, -2:].values  # Last two columns (Win probability vector)

    X_test = test_df.iloc[:, :-2].values
    Y_test = test_df.iloc[:, -2:].values

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset

# Step 2: Define the PyTorch Model
class MarchMadnessNN(nn.Module):
    def __init__(self, input_size):
        super(MarchMadnessNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),  # More neurons
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

loss_history = []

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, torch.max(Y_batch, 1)[1])  # Convert Y_batch to class labels
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
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)  # Get probabilities
            predicted = torch.argmax(Y_pred, dim=1)  # Get the index of the max probability
            actual = torch.argmax(Y_batch, dim=1)  # Get the actual class labels
            correct += (predicted == actual).sum().item()
            total += Y_batch.size(0)

    return (correct / total) * 100  # Accuracy percentage

# Step 5: Run Everything
if __name__ == "__main__":
    train_dataset, test_dataset = load_data("data/training_data.csv", "data/testing_data.csv")

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define model
    input_size = train_dataset[0][0].shape[0]  # Number of input features
    model = MarchMadnessNN(input_size)

    # Train the model
    train_model(model, train_loader, test_loader, epochs=30, lr=0.0005)  # epcohs = 15

    # Final Accuracy
    final_accuracy = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    torch.save(model.state_dict(), "model_heavy.pth")
