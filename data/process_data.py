#!/usr/bin/env python3


import pandas as pd

def refactor_year(file):
    total_df = pd.read_csv(f"./raw/{file}")
    new_df = total_df[total_df["Season"] > 2007]

    return new_df.to_csv(f"./raw/{file}_filtered.csv", index=False)

def main():
    refactor_year("MRegularSeasonDetailedResults.csv")
    refactor_year("MMasseyOrdinals.csv")

if __name__ == "__main__":
    main()