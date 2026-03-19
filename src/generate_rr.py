#!/usr/bin/env python3
"""Generate next-round matchups by advancing winners from the current round's predictions."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import PREDICTIONS_DIR, PROCESSED_DIR


def generate_next_round(round_num):
    """Read predictions from the previous round and build the next round's bracket."""
    prev_round = round_num - 1
    preds_path = os.path.join(PREDICTIONS_DIR, f"round{prev_round}_with_probs.csv")

    if not os.path.exists(preds_path):
        print(f"Error: {preds_path} not found. Run predictor.py for round {prev_round} first.")
        sys.exit(1)

    round_data = pd.read_csv(preds_path)

    required_columns = ["team", "away_team", "team1_id", "team2_id", "Win_Prob_Team1", "Win_Prob_Team2"]
    for col in required_columns:
        if col not in round_data.columns:
            print(f"Error: Column '{col}' not found in {preds_path}")
            sys.exit(1)

    winners = []
    for i in range(0, len(round_data), 2):
        game1 = round_data.iloc[i]
        game2 = round_data.iloc[i + 1]

        if game1["Win_Prob_Team1"] > game1["Win_Prob_Team2"]:
            winner1 = (game1["team"], game1["team1_id"])
        else:
            winner1 = (game1["away_team"], game1["team2_id"])

        if game2["Win_Prob_Team1"] > game2["Win_Prob_Team2"]:
            winner2 = (game2["team"], game2["team1_id"])
        else:
            winner2 = (game2["away_team"], game2["team2_id"])

        winners.append([winner1[0], winner2[0], winner1[1], winner2[1]])

    round_next = pd.DataFrame(winners, columns=["team", "away_team", "team1_id", "team2_id"])
    output_path = os.path.join(PROCESSED_DIR, f"round{round_num}.csv")
    round_next.to_csv(output_path, index=False)
    print(f"Round {round_num} matchups saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate next round matchups from predictions")
    parser.add_argument("--round", type=int, required=True, choices=range(2, 7),
                        help="Next round number to generate (2-6)")
    args = parser.parse_args()
    generate_next_round(args.round)
