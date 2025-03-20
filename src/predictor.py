import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
import torch
import pandas as pd
import numpy as np
from model import MarchMadnessNN; 


file_path = "matchups.csv"
matchups = pd.read_csv(file_path)

# Convert matchups data to tensor (assumes all columns are features)
feature_columns = matchups.columns  # Select all columns (since the last two are missing)
matchup_data = torch.tensor(matchups[feature_columns].values, dtype=torch.float32)
input_size = matchups.shape[1]

model = MarchMadnessNN(input_size)  # Initialize model
model.load_state_dict(torch.load('../models/model.pth'))
model.eval()  

# for game in matchups.csv
    # run model on game
    # store output somewhere

# Run model predictions
with torch.no_grad():
    predictions = model(matchup_data).numpy()  

# Append predictions to the dataframe
matchups["Win_Prob_Team1"] = predictions[:, 0]
matchups["Win_Prob_Team2"] = predictions[:, 1]

# Save updated CSV
output_path = "matchups_with_predictions.csv"
matchups.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")

# Load original round1.csv
round1_path = "../data/raw/round1.csv"
round1_df = pd.read_csv(round1_path)

# Load matchups_with_predictions.csv (output from model)
predictions_path = "matchups_with_predictions.csv"
predictions_df = pd.read_csv(predictions_path)

# Extract only the win probabilities
win_probs = predictions_df[["Win_Prob_Team1", "Win_Prob_Team2"]]

# Append probabilities to round1.csv
round1_df = pd.concat([round1_df, win_probs], axis=1)
round1_df.drop(columns=["team1_id","team2_id"], inplace=True)


# Save updated round1.csv
output_path = "round1_with_probs.csv"
round1_df.to_csv(output_path, index=False)

print(f"Updated round1.csv with win probabilities saved as {output_path}")
