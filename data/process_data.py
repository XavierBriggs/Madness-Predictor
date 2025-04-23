#!/usr/bin/env python3

import numpy as np
import pandas as pd

def refactor_year(file):
    total_df = pd.read_csv(f"./raw/{file}")
    new_df = total_df[total_df["Season"] > 2009]

    return new_df.to_csv(f"./raw/{file}_filtered.csv", index=False)

def add_rank_col(game_file, rank_file):
    game_data = pd.read_csv(f"./raw/{game_file}")
    rank_data = pd.read_csv(f'./raw/{rank_file}')

    rank_data = rank_data[rank_data['SystemName'].isin(['MOR', 'POM'])]

    # Ensure that the key columns are numeric.
    for col in ['Season', 'TeamID', 'RankingDayNum']:
        rank_data[col] = pd.to_numeric(rank_data[col])
    for col in ['Season', 'WTeamID', 'LTeamID', 'DayNum']:
        game_data[col] = pd.to_numeric(game_data[col])

    ranking_pivot = rank_data.pivot_table(index=['Season', 'RankingDayNum', 'TeamID'], 
                                         columns='SystemName', 
                                         values='OrdinalRank').reset_index()
    
    
    ranking_pivot = ranking_pivot.rename(columns={'MOR': 'Rank_MOR', 'POM': 'Rank_KPOM'})
    ranking_pivot = ranking_pivot.sort_values(['Season', 'TeamID', 'RankingDayNum']).copy()

    # ----- Merge for Winning Teams (WTeam) using groupby merge_asof -----
    winners_list = []
    # Group by Season and WTeamID so that within each group DayNum is sorted.
    for (season, team), group in game_data.groupby(['Season', 'WTeamID']):
        group = group.reset_index().sort_values('DayNum')
        # Extract ranking records for this season and team.
        ranking_subset = ranking_pivot[
            (ranking_pivot['Season'] == season) & (ranking_pivot['TeamID'] == team)
        ].sort_values('RankingDayNum')
        # Merge using merge_asof for the current group.
        merged = pd.merge_asof(
            group, 
            ranking_subset, 
            left_on='DayNum', 
            right_on='RankingDayNum', 
            direction='backward',
            suffixes=('','_r')
        )
        winners_list.append(merged)
    winners_df = pd.concat(winners_list)
    winners_df.drop(columns=['Season_r'], inplace=True)
    

    winners_df = winners_df.reset_index(drop=True)
    winners_df = winners_df.sort_values(['Season', 'WTeamID', 'DayNum'])
    winners_df = winners_df.rename(columns={'Rank_MOR': 'WTeam_MOR', 'Rank_KPOM': 'WTeam_KPOM'})
    
    # ----- Merge for Losing Teams (LTeam) using groupby merge_asof -----
    losers_list = []
    # Use winners_df (which now has the winning teams' rankings) and merge for losing teams.
    for (season, team), group in winners_df.groupby(['Season', 'LTeamID']):
        group = group.reset_index().sort_values('DayNum')
        ranking_subset = ranking_pivot[
            (ranking_pivot['Season'] == season) & (ranking_pivot['TeamID'] == team)
        ].sort_values('RankingDayNum')
        merged = pd.merge_asof(
            group,
            ranking_subset,
            left_on='DayNum',
            right_on='RankingDayNum',
            direction='backward',
            suffixes=('','_r')
        )
        losers_list.append(merged)
    full_df = pd.concat(losers_list)
    full_df.drop(columns=['Season_r'], inplace=True)

    full_df = full_df.reset_index(drop=True).sort_values(['Season', 'DayNum','WTeamID' ])
    full_df = full_df.rename(columns={'Rank_MOR': 'LTeam_MOR', 'Rank_KPOM': 'LTeam_KPOM'})
    # full_df[['WTeam_MOR', 'WTeam_KPOM']] = full_df.groupby(['Season', 'WTeamID'])[['WTeam_MOR', 'WTeam_KPOM']].bfill()
    # full_df[['LTeam_MOR', 'LTeam_KPOM']] = full_df.groupby(['Season', 'LTeamID'])[['LTeam_MOR', 'LTeam_KPOM']].bfill()
    full_df[['WTeam_MOR', 'WTeam_KPOM']] = full_df.groupby(['Season', 'WTeamID'])[['WTeam_MOR', 'WTeam_KPOM']].transform(lambda x: x.bfill().ffill())
    full_df[['LTeam_MOR', 'LTeam_KPOM']] = full_df.groupby(['Season', 'LTeamID'])[['LTeam_MOR', 'LTeam_KPOM']].transform(lambda x: x.bfill().ffill())


    drop_columns = ['level_0', 'index', 'RankingDayNum', 'RankingDayNum_r', 'TeamID', 'TeamID_r']
    full_df.drop(columns=drop_columns, inplace=True)
    



    
    
    
    # Save the updated dataset

    output_file = f"{game_file}_addedrank.csv"
    return full_df.to_csv(f"./raw/{output_file}", index=False)


def add_poss_stats(file):
    total_df = pd.read_csv(f"./raw/{file}")


    total_df["WPoss"] = np.round(total_df["WFGA"] - (total_df["WOR"] / (total_df["WOR"] + total_df["LDR"])) * (total_df["WFGA"] - total_df["WFGM"]) * 1.07 + total_df["WTO"] + 0.44 * total_df["WFTA"])
    total_df["LPoss"] = np.round(total_df["LFGA"] - (total_df["LOR"] / (total_df["LOR"] + total_df["WDR"])) * (total_df["LFGA"] - total_df["LFGM"]) * 1.07 + total_df["LTO"] + 0.44 * total_df["LFTA"])

    
    total_df["WeFG%"] = (total_df["WFGM"] + 0.5 * total_df['WFGM3'])/total_df["WFGA"]
    total_df["LeFG%"] = (total_df["LFGM"] + 0.5 * total_df['LFGM3'])/total_df["LFGA"]
    

    total_df["WORB%"] = (total_df["WOR"])/(total_df["WOR"] + total_df['LDR'])
    total_df["LORB%"] = (total_df["LOR"])/(total_df["LOR"] + total_df['WDR'])
    total_df["WDRB%"] = (total_df["WDR"])/(total_df["LOR"] + total_df['WDR'])
    total_df["LDRB%"] = (total_df["LDR"])/(total_df["WOR"] + total_df['LDR'])


    total_df["WFTR"] = total_df['WFTA']/total_df['WFGA']
    total_df["LFTR"] = total_df['LFTA']/total_df['LFGA']
    total_df["WFT%"] = total_df['WFTM']/total_df['WFTA']
    total_df["LFT%"] = total_df['LFTM']/total_df['LFTA']
    

    total_df["WTO%"] = total_df["WTO"]/total_df["WPoss"]
    total_df["LTO%"] = total_df["LTO"]/total_df["LPoss"]


    # Save the updated dataset
    output_file = f"{file}_with_poss.csv"
    return total_df.to_csv(f"./raw/{output_file}", index=False)


def concat_data(file1, file2):

    df1 = pd.read_csv(file1)

    df2 = pd.read_csv(file2)

    concatdf = pd.concat([df2, df1])

    return concatdf.to_csv(f"./raw/MRegularDetailed*MNCAATourneydetailed.csv", index=False)
    

def split_quarter():
    pass
    

def main():
    # refactor_year("MRegularSeasonDetailedResults.csv")
    # refactor_year("MMasseyOrdinals_filtered.csv")
    # add_rank_col("MNCAATourneyDetailedResults.csv","MMasseyOrdinals_filtered.csv")
    # add_poss_stats("MNCAATourneyDetailedResults_addedrank.csv")
    concat_data("raw/MNCAATourneyDetailedResults_with_poss.csv", "raw/MRegularSeasonDetailedResults_with_poss.csv")


if __name__ == "__main__":
    main()