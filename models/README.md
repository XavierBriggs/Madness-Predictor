# Models Directory

## Overview
This directory contains the neural network model definition, trained model weights, and training data for the XG-MM March Madness prediction system. The model is designed to predict win probabilities for matchups between NCAA basketball teams.

## Key Files

### Model Definition
- `model.py`: PyTorch neural network model definition and training code

### Trained Models
- `model.pth`: Trained model weights for the standard neural network
- `model_heavy.pth`: Trained model weights for a larger version of the neural network
- `march_madness_nn.pth`: Alternative trained model weights

### Training Data
- `/data/training_data.csv`: Dataset used for model training
- `/data/testing_data.csv`: Dataset used for model evaluation

### Visualization
- `loss_history.png`: Plot showing the training loss over time

## Model Architecture
The model is a fully-connected neural network implemented in PyTorch with the following architecture:

```
MarchMadnessNN(
  (model): Sequential(
    (0): Linear(in_features=X, out_features=16)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=16)
    (5): ReLU()
    (6): Linear(in_features=16, out_features=2)
    (7): Softmax(dim=1)
  )
)
```

Where X is the number of input features (team statistics).

## Training Process

The model training process includes:
1. Loading training and testing data from CSV files
2. Converting data to PyTorch tensors
3. Training the neural network using Cross-Entropy Loss and Adam optimizer
4. Evaluating model accuracy on test data
5. Saving the trained model weights to a .pth file

## Usage

### Training the Model
To train the model from scratch:

```python
# Import the model definition
from model import MarchMadnessNN, load_data, train_model

# Load training and testing data
train_dataset, test_dataset = load_data("data/training_data.csv", "data/testing_data.csv")

# Create model and train
input_size = train_dataset[0][0].shape[0]
model = MarchMadnessNN(input_size)
train_model(model, train_loader, test_loader, epochs=30)
```

### Using the Model for Predictions

```python
# Load the trained model
model = MarchMadnessNN(input_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(matchup_data)
```

## Performance
The model achieves approximately 70-75% accuracy on historical NCAA tournament matchups, with variations depending on the specific model version and training parameters.