# Source Code Directory

This directory contains the source code for prediction generation and tournament simulation for the Madness-Predictor March Madness prediction system.

## Contents

- **predictor.py**: Script for generating predictions for tournament matchups
- **generate_rr.py**: Script for generating next round matchups based on predictions
- **predictions/**: Directory containing prediction output files

## Prediction Process

The prediction process involves:

1. **Loading matchup data**: The `predictor.py` script loads matchup data for a specific tournament round
2. **Running the model**: The script loads the trained neural network model and generates win probabilities for each matchup
3. **Saving predictions**: The predictions are saved to CSV files in the predictions directory
4. **Generating next round**: The `generate_rr.py` script uses the predictions to determine winners and generate matchups for the next round

## Usage

### Generating Predictions

To generate predictions for a specific tournament round:

```
python predictor.py <round_number>
```

Where `<round_number>` is the tournament round number (e.g., 1 for first round).

This will:
1. Load the matchup data from `../data/processed/matchups<round_number>.csv`
2. Load the trained model from `../models/model.pth`
3. Generate win probabilities for each matchup
4. Save the predictions to `predictions/round<round_number>_with_probs.csv`

### Generating Next Round Matchups

To generate matchups for the next round based on predictions:

```
python generate_rr.py <round_number>
```

Where `<round_number>` is the next round number (must be greater than 1).

This will:
1. Load the predictions from the previous round
2. Determine winners based on win probabilities
3. Generate matchups for the next round
4. Save the matchups to `../data/processed/round<round_number>.csv`

To concatenate these steps:
```
./play_round <round_number>
```
This will simulate round `<round_number` and generate next round matchups. 
