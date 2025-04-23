import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

def load_data(train_csv, test_csv, scaler=None):
    # Load CSVs into Pandas
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract features (X) and labels (Y)
    X_train = train_df.iloc[:, :-2].values  # All columns except last two
    Y_train = train_df.iloc[:, -2:].values   # Last two columns (one-hot encoded probabilities)
    
    X_test = test_df.iloc[:, :-2].values
    Y_test = test_df.iloc[:, -2:].values

    # Optionally normalize features
    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset, scaler

# Revised Neural Network Model
class MarchMadnessNN(nn.Module):
    def __init__(self, input_size):
        super(MarchMadnessNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation; CrossEntropyLoss expects raw logits
        return x

loss_history = []

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Example scheduler

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            # Convert one-hot labels to class indices
            loss = criterion(outputs, torch.max(Y_batch, 1)[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Update learning rate
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
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(Y_batch, dim=1)
            correct += (predicted == actual).sum().item()
            total += Y_batch.size(0)
    return (correct / total) * 100

if __name__ == "__main__":
    # Load data and perform scaling
    train_dataset, test_dataset, scaler = load_data("data/training_data.csv", "data/testing_data.csv")

    # Optionally, split train_dataset into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define model
    input_size = train_dataset[0][0].shape[0]
    model = MarchMadnessNN(input_size)

    # Train the model
    train_model(model, train_loader, test_loader, epochs=1000, lr=0.0005)

    # Final Evaluation
    final_accuracy = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # Save the model weights
    torch.save(model.state_dict(), "march_madness_nn.pth")

    # Example Prediction
    model.eval()
    sample, _ = test_dataset[0]
    with torch.no_grad():
        logits = model(sample.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    print("Predicted class:", predicted_class.item())
    print("Predicted probabilities:", probabilities.numpy())
