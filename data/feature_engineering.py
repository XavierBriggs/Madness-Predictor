import pandas as pd
import numpy as np

def compute_team_features(csv_file, rolling_window=5):
    # Load data and drop any rows with missing values
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)

    # Define QuarterID based on DayNum
    df["QuarterID"] = pd.cut(df["DayNum"], bins=[-1, 33, 66, 99, 132], labels=[1, 2, 3, 4]).astype(int)

    # Compute possessions for winning and losing teams
    df["WPoss"] = df["WFGA"] - df["WOR"] + df["WTO"] + (0.44 * df["WFTA"])
    df["LPoss"] = df["LFGA"] - df["LOR"] + df["LTO"] + (0.44 * df["LFTA"])

    # Small constant to avoid division by zero
    epsilon = 1e-6

    # Compute advanced metrics for both winning (W) and losing (L) teams
    for prefix in ["W", "L"]:
        # Offensive and defensive rating
        df[f"{prefix}ORTG"] = df[f"{prefix}Score"] / (df[f"{prefix}Poss"] + epsilon) * 100
        opponent_prefix = "L" if prefix == "W" else "W"
        df[f"{prefix}DRTG"] = df[f"{opponent_prefix}Score"] / (df[f"{prefix}Poss"] + epsilon) * 100
        df[f"{prefix}NetRTG"] = df[f"{prefix}ORTG"] - df[f"{prefix}DRTG"]
        df[f"{prefix}AssistRatio"] = df[f"{prefix}Ast"] / (df[f"{prefix}Poss"] + epsilon) * 100
        df[f"{prefix}TurnoverRatio"] = df[f"{prefix}TO"] / (df[f"{prefix}Poss"] + epsilon) * 100

    # Helper function to compute rolling averages
    def rolling_avg(df, team_col, stats, window):
        df_sorted = df.sort_values(by=["Season", "DayNum"])
        rolling_result = df_sorted.groupby(["Season", team_col])[stats].rolling(window=window, min_periods=1).mean()
        rolling_result = rolling_result.reset_index(level=[0,1], drop=True)
        return rolling_result.reindex(df.index)

    rolling_stats = ["ORTG", "DRTG", "NetRTG", "AssistRatio", "TurnoverRatio"]
    
    for prefix, team_col in [("W", "WTeamID"), ("L", "LTeamID")]:
        stat_cols = [f"{prefix}{stat}" for stat in rolling_stats]
        rolling_avg_df = rolling_avg(df, team_col, stat_cols, window=rolling_window)
        new_col_names = [f"{prefix}R_{stat}" for stat in rolling_stats]
        df[new_col_names] = rolling_avg_df

    # Home/Away feature
    df["HomeAdv"] = df["WLoc"].map({"H": 1, "A": -1, "N": 0})

    # Compute differences between team metrics
    df["eFG%_Diff"] = df["WeFG%"] - df["LeFG%"]
    df["TO%_Diff"] = df["WTO%"] - df["LTO%"]
    df["OREB%_Diff"] = df["WORB%"] - df["LORB%"]
    df["FTR_Diff"] = df["WFTR"] - df["LFTR"]

    # Create a long-format dataset to ensure each team has its own row
    team_features = pd.concat([
        df.rename(columns={col: col[1:] for col in df.columns if col.startswith("W")})
        .assign(TeamID=df["WTeamID"], OpponentID=df["LTeamID"], Result=1),  # 1 for win
        df.rename(columns={col: col[1:] for col in df.columns if col.startswith("L")})
        .assign(TeamID=df["LTeamID"], OpponentID=df["WTeamID"], Result=0)   # 0 for loss
    ])

    # Define final feature columns
    feature_columns = [
        "Season", "DayNum", "TeamID", "QuarterID", "HomeAdv", "ORTG", "DRTG", "NetRTG",
        "AssistRatio", "TurnoverRatio", "R_ORTG", "R_DRTG", "R_NetRTG", "R_AssistRatio",
        "R_TurnoverRatio", "eFG%_Diff", "TO%_Diff", "OREB%_Diff", "FTR_Diff", "Result"
    ]

    # Keep only necessary columns
    final_df = team_features[feature_columns].copy()

    return final_df

# Usage example:
if __name__ == "__main__":
    processed_data = compute_team_features("raw/MRegularSeasonDetailedResults_with_poss.csv")
    processed_data.to_csv("processed/team_features.csv", index=False)
    print("Feature Engineering Completed.")
