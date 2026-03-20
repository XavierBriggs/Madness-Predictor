# Models

Neural network definition, training code, and saved weights.

## Files

| File | Description |
|------|-------------|
| `model.py` | `MarchMadnessNN` — 3-layer fully connected network (16-16-16) with softmax output |
| `model.pth` | Trained weights (gitignored — regenerate with `python model.py`) |
| `training_loss.png` | Training loss curve |
| `data/` | Training and testing CSVs (gitignored) |

## Architecture

```
Input (26 features) → Linear(16) → ReLU → Linear(16) → ReLU → Linear(16) → ReLU → Linear(2) → Softmax
```

Trained with cross-entropy loss, Adam optimizer (lr=0.0005, 30 epochs, batch size 32). Achieves ~70-75% accuracy on historical tournament matchups.
