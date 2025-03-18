# import teams.csv as df indexed by [season][team_id]
# loop through each line of tourney results 
    # append each matchup ; concat vectors from df 
    # generate ouptput either (1,0) or probabilties using delta points 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import erf


def generate_training_data(tourney_results_csv, team_feature_matrix_csv, train_csv, test_csv, test_size=0.2, random_seed=42):
    # Load the March Madness tournament results
    tourney_results = pd.read_csv(tourney_results_csv)
    
    # Load the team feature matrix (with TeamID included)
    team_features = pd.read_csv(team_feature_matrix_csv)
    
    # Create a mapping (Season, TeamID) -> Team Stats
    team_map = {}
    for _, row in team_features.iterrows():
        key = (row["Season"], row["TeamID"])
        team_map[key] = row.drop(["Season", "TeamID"]).values  # Drop identifiers, store only stats
    
    # Prepare training data
    X = []
    Y = []
    
    for _, row in tourney_results.iterrows():
        season = row["Season"]
        team_w = row["WTeamID"]  # Winning Team
        team_l = row["LTeamID"]  # Losing Team
        score_w = row["WScore"]
        score_l = row["LScore"]
        delta = score_w - score_l
        
        # Get team vectors from mapping
        if (season, team_w) in team_map and (season, team_l) in team_map:
            team_w_vector = team_map[(season, team_w)]
            team_l_vector = team_map[(season, team_l)]

            sigma = 10
            p = lambda delta : 0.5 + 0.5 * erf(delta / sigma) 
            
            # Randomly shuffle order of teams
            if np.random.rand() > 0.5:
                matchup_vector = list(team_w_vector) + list(team_l_vector)
                y_label = [p(delta), 1 - p(delta)]
                y_label = [1, 0]  # Team1 won
            else:
                matchup_vector = list(team_l_vector) + list(team_w_vector)
                y_label = [1-p(delta), p(delta)]  # Team2 won
            
            X.append(matchup_vector)
            Y.append(y_label)
    
    # Convert to DataFrame
    feature_columns = list(team_features.columns[2:])  # Skip Season and TeamID
    columns = [f"Team1_{col}" for col in feature_columns] + [f"Team2_{col}" for col in feature_columns]
    X_df = pd.DataFrame(X, columns=columns)
    Y_df = pd.DataFrame(Y, columns=["Win_Prob_Team1", "Win_Prob_Team2"])
    
    # Combine X and Y into a single DataFrame
    data = pd.concat([X_df, Y_df], axis=1)
    
    # Train-test split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    
    # Save training and testing data
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    
    print(f"Training data saved as {train_csv}")
    print(f"Testing data saved as {test_csv}")

# usage
generate_training_data("raw\MNCAATourneyDetailedResults.csv", "processed/teams.csv", "../models/data/training_data.csv", "../models/data/testing_data.csv")
