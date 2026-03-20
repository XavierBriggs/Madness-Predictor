"""Central configuration for the Madness-Predictor pipeline."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_DATA_DIR = os.path.join(MODELS_DIR, "data")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PREDICTIONS_DIR = os.path.join(SRC_DIR, "predictions")

# Season
CURRENT_SEASON = 2026

# Model hyperparameters
HIDDEN_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

# Model file
MODEL_PATH = os.path.join(MODELS_DIR, "model.pth")
