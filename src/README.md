# Source

Prediction and tournament simulation scripts.

## Files

| File | Description |
|------|-------------|
| `predictor.py` | Loads model, runs predictions on a round's matchups, outputs win probabilities |
| `generate_rr.py` | Advances winners from predictions to build the next round's bracket |
| `play_round.sh` | Convenience script: predict round → generate next round → build matchups |
| `predictions/` | Output CSVs with per-round predictions and confidence scores |

## Usage

```bash
# Predict a single round and advance winners
./play_round.sh <round_number>

# Or run steps individually
python predictor.py --round 3
python generate_rr.py --round 4
```
