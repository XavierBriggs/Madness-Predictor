#!/usr/bin/env python3
"""Build feature vectors for a tournament round's matchups by joining team stats."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import CURRENT_SEASON, PROCESSED_DIR


def generate_matchups(round_num):
    """Merge team statistics for each matchup in the given round."""
    round_path = os.path.join(PROCESSED_DIR, f"round{round_num}.csv")
    teams_path = os.path.join(PROCESSED_DIR, "teams.csv")

    round_df = pd.read_csv(round_path)
    teams = pd.read_csv(teams_path)

    matchups = round_df.merge(teams, left_on="team1_id", right_on="TeamID", suffixes=("_Team1", ""))
    matchups = matchups.merge(teams, left_on="team2_id", right_on="TeamID", suffixes=("", "_Team2"))

    matchups = matchups[
        (matchups["Season"] == CURRENT_SEASON) & (matchups["QuarterID"] == 4)
    ]

    matchups = matchups.drop_duplicates(subset=["team1_id", "team2_id"])

    column_rename_map = {}
    for col in teams.columns:
        if col != "team_id":
            column_rename_map[col + "_Team1"] = f"Team1_{col}"
            column_rename_map[col + "_Team2"] = f"Team2_{col}"

    matchups.rename(columns=column_rename_map, inplace=True)
    matchups.drop(
        columns=["team", "away_team", "team1_id", "team2_id",
                 "Season", "QuarterID", "TeamID",
                 "Team2_Season", "Team2_QuarterID", "Team2_TeamID"],
        inplace=True,
    )

    output_path = os.path.join(PROCESSED_DIR, f"matchups{round_num}.csv")
    matchups.to_csv(output_path, index=False)
    print(f"Matchups file generated: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate matchup feature vectors for a round")
    parser.add_argument("--round", type=int, required=True, choices=range(1, 7),
                        help="Tournament round number (1-6)")
    args = parser.parse_args()
    generate_matchups(args.round)
