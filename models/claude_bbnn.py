import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def load_data(train_csv, test_csv):
    # Load CSVs into Pandas
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract features (X) and labels (Y)
    X_train = train_df.iloc[:, :-2].values  # All columns except last two (Win probabilities)
    Y_train = train_df.iloc[:, -2:].values  # Last two columns (Win probability vector)

    X_test = test_df.iloc[:, :-2].values
    Y_test = test_df.iloc[:, -2:].values

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # Create validation set from training data
    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size
    
    train_dataset = TensorDataset(X_train[:train_size], Y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], Y_train[train_size:])
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, val_dataset, test_dataset, scaler

# Define the PyTorch Model with improvements
class MarchMadnessNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(MarchMadnessNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 2),
            nn.Softmax(dim=1)  # Use Softmax for probability distribution
        )
        
        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, test_loader, epochs=100, lr=0.001, patience=20):
    criterion = nn.BCELoss()  # Binary Cross Entropy for probability outputs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_pred = model(X_batch)
                val_loss = criterion(Y_pred, Y_batch)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            accuracy = evaluate_model(model, test_loader)
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_history.png")
    plt.show()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            predicted = torch.argmax(Y_pred, dim=1)
            actual = torch.argmax(Y_batch, dim=1)
            correct += (predicted == actual).sum().item()
            total += Y_batch.size(0)
            
            # Store predictions and actuals for further analysis
            predictions.extend(Y_pred.numpy())
            actuals.extend(Y_batch.numpy())
    
    accuracy = (correct / total) * 100
    return accuracy

def make_predictions(model, feature_data, scaler):
    """
    Make predictions on new data
    
    Args:
        model: Trained PyTorch model
        feature_data: Numpy array of features
        scaler: The scaler used to normalize training data
    
    Returns:
        probabilities: Win probabilities for each team
    """
    # Normalize the input data
    normalized_data = scaler.transform(feature_data)
    
    # Convert to tensor
    X = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    return predictions.numpy()

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    train_dataset, val_dataset, test_dataset, scaler = load_data(
        "data/training_data.csv", 
        "data/testing_data.csv"
    )
    
    # Create DataLoaders
    batch_size = 64  # Increased batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define model
    input_size = train_dataset[0][0].shape[0]  # Number of input features
    model = MarchMadnessNN(input_size)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        epochs=1000, 
        lr=0.001, 
        patience=30
    )
    
    # Final Accuracy
    final_accuracy = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    
    # Save model
    torch.save({'model_state_dict': model.state_dict(), }, "march_madness_nn.pth")
    
    # Example prediction
    example_features = test_dataset[0][0].numpy().reshape(1, -1)
    predictions = make_predictions(model, example_features, scaler)
    print(f"Example prediction probabilities: {predictions}")