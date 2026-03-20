#!/usr/bin/env python3
"""Generate training and testing datasets by pairing historical tournament matchups."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import erf

from config import RAW_DIR, PROCESSED_DIR, MODEL_DATA_DIR


def generate_training_data(tourney_results_csv, team_feature_matrix_csv,
                           train_csv, test_csv, test_size=0.2, random_seed=3627):
    """Create train/test datasets from historical matchups with soft win probability labels."""
    tourney_results = pd.read_csv(tourney_results_csv)
    team_features = pd.read_csv(team_feature_matrix_csv)

    if "QuarterID" not in team_features.columns:
        team_features["QuarterID"] = 4

    tourney_results["QuarterID"] = pd.cut(
        tourney_results["DayNum"],
        bins=[-1, 33, 66, 99, 132],
        labels=[1, 2, 3, 4]
    ).astype(int)

    team_map = {}
    for _, row in team_features.iterrows():
        key = (row["Season"], row["QuarterID"], row["TeamID"])
        team_map[key] = row.drop(["Season", "QuarterID", "TeamID"]).values

    X = []
    Y = []

    for _, row in tourney_results.iterrows():
        season = row["Season"]
        quarter = row["QuarterID"]
        team_w = row["WTeamID"]
        team_l = row["LTeamID"]
        score_w = row["WScore"]
        score_l = row["LScore"]
        delta = score_w - score_l

        if (season, quarter, team_w) in team_map and (season, quarter, team_l) in team_map:
            team_w_vector = team_map[(season, quarter, team_w)]
            team_l_vector = team_map[(season, quarter, team_l)]

            sigma = 10
            p = lambda d: 0.5 + 0.5 * erf(d / sigma)  # noqa: E731

            if np.random.rand() > 0.5:
                matchup_vector = list(team_w_vector) + list(team_l_vector)
                y_label = [p(delta), 1 - p(delta)]
            else:
                matchup_vector = list(team_l_vector) + list(team_w_vector)
                y_label = [1 - p(delta), p(delta)]

            X.append(matchup_vector)
            Y.append(y_label)

    feature_columns = list(team_features.columns[3:])
    columns = [f"Team1_{col}" for col in feature_columns] + [f"Team2_{col}" for col in feature_columns]
    X_df = pd.DataFrame(X, columns=columns)
    Y_df = pd.DataFrame(Y, columns=["Win_Prob_Team1", "Win_Prob_Team2"])

    data = pd.concat([X_df, Y_df], axis=1)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)

    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    print(f"Training data saved as {train_csv}")
    print(f"Testing data saved as {test_csv}")


if __name__ == "__main__":
    generate_training_data(
        os.path.join(RAW_DIR, "MRegularSeasonDetailedResults_with_poss.csv"),
        os.path.join(PROCESSED_DIR, "teams.csv"),
        os.path.join(MODEL_DATA_DIR, "training_data.csv"),
        os.path.join(MODEL_DATA_DIR, "testing_data.csv"),
    )
