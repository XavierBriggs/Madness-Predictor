#!/usr/bin/env python3
"""Run the trained model on a tournament round's matchups and output win probabilities."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import torch
import pandas as pd
from model import MarchMadnessNN
from config import PROCESSED_DIR, MODEL_PATH, PREDICTIONS_DIR


def predict_round(round_num):
    """Load matchup data for a round, run the model, and save predictions."""
    matchup_path = os.path.join(PROCESSED_DIR, f"matchups{round_num}.csv")
    if not os.path.exists(matchup_path):
        print(f"Error: {matchup_path} not found. Generate matchup data first.")
        sys.exit(1)

    matchups = pd.read_csv(matchup_path)
    matchup_data = torch.tensor(matchups.values, dtype=torch.float32)

    model = MarchMadnessNN(matchups.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        predictions = model(matchup_data).numpy()

    matchups["Win_Prob_Team1"] = predictions[:, 0]
    matchups["Win_Prob_Team2"] = predictions[:, 1]

    round_path = os.path.join(PROCESSED_DIR, f"round{round_num}.csv")
    round_df = pd.read_csv(round_path)
    win_probs = matchups[["Win_Prob_Team1", "Win_Prob_Team2"]]
    round_df = pd.concat([round_df, win_probs], axis=1)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    output_path = os.path.join(PREDICTIONS_DIR, f"round{round_num}_with_probs.csv")
    round_df.to_csv(output_path, index=False)
    print(f"Updated round{round_num}.csv with win probabilities saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tournament round outcomes")
    parser.add_argument("--round", type=int, required=True, choices=range(1, 7),
                        help="Tournament round number (1-6)")
    args = parser.parse_args()
    predict_round(args.round)
