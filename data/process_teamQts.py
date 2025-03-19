#!/usr/bin/env python3
import pandas as pd

def compute_team_per_possession_matrix_quarters(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    df = df[df["Season"] != 2021]

    df = df.sort_values(["Season", "DayNum"])

    # Define quarter thresholds (inclusive)
    quarters = [(33, 1), (66, 2), (99, 3), (132, 4)]
    
    # Estimate possessions for winning and losing teams
    df["WPoss"] = df["WFGA"] - (df["WOR"]/(df["WOR"] + df["LDR"]))*(df["WFGA"] - df["WFGM"]) * 1.07 + df["WTO"] + (0.44 * df["WFTA"])
    df["LPoss"] = df["LFGA"] - (df["LOR"]/(df["LOR"] + df["WDR"]))*(df["LFGA"] - df["LFGM"]) * 1.07 + df["LTO"] + (0.44 * df["LFTA"])

    # Compute per-possession stats for both winning and losing teams
    for prefix in ["W", "L"]:
        df[f"{prefix}TS%"] = 0.5 * df[f"{prefix}Score"]/(df[f"{prefix}FGA"] + 0.44*df[f"{prefix}FTA"])
        df[f"{prefix}eFG%"] = (df[f"{prefix}FGM"] + 0.5 * df[f"{prefix}FGM3"]) / df[f"{prefix}FGA"]
        df[f"{prefix}TO%"] = df[f"{prefix}TO"] / df[f"{prefix}Poss"]
        df[f"{prefix}OREB%"] = df[f"{prefix}OR"] / (df[f"{prefix}OR"] + df[f"{'L' if prefix == 'W' else 'W'}DR"])
        df[f"{prefix}DREB%"] = df[f"{prefix}DR"] / (df[f"{prefix}DR"] + df[f"{'L' if prefix == 'W' else 'W'}OR"])
        df[f"{prefix}FTR"] = df[f"{prefix}FTA"] / df[f"{prefix}FGA"]
        df[f"{prefix}3PAr"] = df[f"{prefix}FGA3"] / df[f"{prefix}FGA"]
        df[f"{prefix}AST/TO"] = df[f"{prefix}Ast"] / (df[f"{prefix}TO"] + 1e-6)
        df[f"{prefix}STL%"] = df[f"{prefix}Stl"] / df[f"{prefix}Poss"]
        df[f"{prefix}BLK%"] = df[f"{prefix}Blk"] / df[f"{prefix}FGA"]

    # Aggregate per team per season
    team_stats = {}
    # Loop over each game row.
    for idx, row in df.iterrows():
        season = row["Season"]
        day = row["DayNum"]
        for prefix in ["W", "L"]:
            team = row[f"{prefix}TeamID"]
            score = row[f"{prefix}Score"]
            poss = row[f"{prefix}Poss"]
            # Get per-possession values (already computed above)
            stats_list = {
                "TS%": row[f"{prefix}TS%"],
                "eFG%": row[f"{prefix}eFG%"],
                "TO%": row[f"{prefix}TO%"],
                "OREB%": row[f"{prefix}OREB%"],
                "DREB%": row[f"{prefix}DREB%"],
                "FTR": row[f"{prefix}FTR"],
                "3PAr": row[f"{prefix}3PAr"],
                "AST/TO": row[f"{prefix}AST/TO"],
                "STL%": row[f"{prefix}STL%"],
                "BLK%": row[f"{prefix}BLK%"]
            }
            key = (season, team)
            if key not in team_stats:
                # Initialize accumulators for each quarter.
                team_stats[key] = {q_label: {
                    "Games": 0,
                    "TotalPoss": 0,
                    "TS": 0,
                    "eFG": 0,
                    "TO": 0,
                    "OREB": 0,
                    "DREB": 0,
                    "FTR": 0,
                    "3PAr": 0,
                    "AST_TO": 0,
                    "STL": 0,
                    "BLK": 0,
                    "Points": 0
                } for (_, q_label) in quarters}

            # For each quarter, if this game happened by that day (using cutoff), add the game’s stats.
            for cutoff, q_label in quarters:
                if day <= cutoff:
                    team_stats[key][q_label]["Games"] += 1
                    team_stats[key][q_label]["TotalPoss"] += poss
                    team_stats[key][q_label]["TS"] += stats_list["TS%"] * poss
                    team_stats[key][q_label]["eFG"] += stats_list["eFG%"] * poss
                    team_stats[key][q_label]["TO"] += stats_list["TO%"] * poss
                    team_stats[key][q_label]["OREB"] += stats_list["OREB%"] * poss
                    team_stats[key][q_label]["DREB"] += stats_list["DREB%"] * poss
                    team_stats[key][q_label]["FTR"] += stats_list["FTR"] * poss
                    team_stats[key][q_label]["3PAr"] += stats_list["3PAr"] * poss
                    team_stats[key][q_label]["AST_TO"] += stats_list["AST/TO"] * poss
                    team_stats[key][q_label]["STL"] += stats_list["STL%"] * poss
                    team_stats[key][q_label]["BLK"] += stats_list["BLK%"] * poss
                    team_stats[key][q_label]["Points"] += score

    # Build the dataset: one row per team per quarter.
    team_features = []
    for (season, team), quarters_dict in team_stats.items():
        for q_label in sorted(quarters_dict.keys()):
            stats = quarters_dict[q_label]
            if stats["TotalPoss"] > 0:
                ts_avg     = stats["TS"] / stats["TotalPoss"]
                efg_avg    = stats["eFG"] / stats["TotalPoss"]
                to_avg     = stats["TO"] / stats["TotalPoss"]
                oreb_avg   = stats["OREB"] / stats["TotalPoss"]
                dreb_avg   = stats["DREB"] / stats["TotalPoss"]
                ftr_avg    = stats["FTR"] / stats["TotalPoss"]
                threepar_avg = stats["3PAr"] / stats["TotalPoss"]
                ast_to_avg = stats["AST_TO"] / stats["TotalPoss"]
                stl_avg    = stats["STL"] / stats["TotalPoss"]
                blk_avg    = stats["BLK"] / stats["TotalPoss"]
                ppp        = stats["Points"] / stats["TotalPoss"]
            else:
                ts_avg = efg_avg = to_avg = oreb_avg = dreb_avg = ftr_avg = threepar_avg = ast_to_avg = stl_avg = blk_avg = ppp = 0

            row = [season, team, q_label, ts_avg, efg_avg, to_avg, oreb_avg, dreb_avg,
                   ftr_avg, threepar_avg, ast_to_avg, stl_avg, blk_avg, ppp]
            team_features.append(row)

    # Create a DataFrame from the aggregated data.
    cols = ["Season", "TeamID", "Quarter", "TS%", "eFG%", "TO%", "OREB%", "DREB%",
            "FTR", "3PAr", "AST/TO", "STL%", "BLK%", "PointsPerPoss"]
    team_matrix = pd.DataFrame(team_features, columns=cols)

    # --- Now compute Adjusted Offensive and Defensive ratings ---
    # For each season and quarter, compute the season average PointsPerPoss.
    season_quarter_avg = {}
    for season in team_matrix["Season"].unique():
        for q_label in sorted(set(team_matrix["Quarter"])):
            subset = team_matrix[(team_matrix["Season"]==season) & (team_matrix["Quarter"]==q_label)]
            season_quarter_avg[(season, q_label)] = subset["PointsPerPoss"].mean() if not subset.empty else 0

    # Compute adjusted ratings for each team relative to its season-quarter peers.
    adj_list = []
    for (season, q_label), group in team_matrix.groupby(["Season", "Quarter"]):
        overall_avg = season_quarter_avg[(season, q_label)]
        for idx, row in group.iterrows():
            team = row["TeamID"]
            opponents = group[group["TeamID"] != team]
            if not opponents.empty:
                avg_opponent = opponents["PointsPerPoss"].mean()
            else:
                avg_opponent = overall_avg
            # Adjusted ratings are computed relative to the season-quarter average.
            adjO = row["PointsPerPoss"] * (avg_opponent / overall_avg) if overall_avg != 0 else 0
            adjD = row["PointsPerPoss"] * (avg_opponent / overall_avg) if overall_avg != 0 else 0
            new_row = list(row) + [adjO, adjD]
            adj_list.append(new_row)
    final_cols = cols + ["AdjO", "AdjD"]
    final_df = pd.DataFrame(adj_list, columns=final_cols)
    return final_df

# Usage example:
team_matrix_quarters = compute_team_per_possession_matrix_quarters("raw/MRegularSeasonDetailedResults_with_poss.csv")
team_matrix_quarters.to_csv("processed/teams_quarters.csv", index=False)
print("Done")