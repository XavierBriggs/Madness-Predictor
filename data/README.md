# Data

Raw NCAA datasets and processing scripts that produce the team feature matrix and training data.

## Scripts

| Script | Purpose |
|--------|---------|
| `process_quarter_data.py` | Computes 14 per-possession metrics per team per quarter from raw game logs |
| `generate_training_data.py` | Pairs historical matchups into train/test splits for the neural network |
| `generate_matchup_data.py` | Builds feature vectors for a given tournament round's matchups |
| `process_data.sh` | Runs the full data pipeline (process → train data → round 1 matchups) |
| `score_distributions.py` | Utility for analyzing historical score differentials |

## Pipeline

```
raw/*.csv → process_quarter_data.py → processed/teams.csv
                                           ↓
raw/*.csv → generate_training_data.py → models/data/{training,testing}_data.csv
                                           ↓
processed/round{r}.csv → generate_matchup_data.py → processed/matchups{r}.csv
```
