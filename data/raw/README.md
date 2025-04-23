# Raw Data Directory

## Overview
This directory contains the original NCAA basketball tournament data files used as input for the prediction model. These files serve as the foundation for all data processing and model training.

## Data Files

- `MConferenceTourneyGames.csv`: Conference tournament game results
- `MNCAATourneyDetailedResults.csv`: Detailed NCAA tournament game results
- `MNCAATourneyDetailedResults_addedrank.csv`: Tournament results with team ranking information
- `MNCAATourneyDetailedResults_with_poss.csv`: Tournament results with possession statistics
- `MNCAATourneySeeds.csv`: Historical seeding information for tournament teams
- `MNCAATourneySlots.csv`: Tournament bracket structure and slot assignments
- `MRegularSeasonDetailedResults_with_poss.csv`: Regular season game results with possession data
- `MTeamConferences.csv`: Team conference affiliations by season
- `MTeams.csv`: Master list of team information and identifiers
- `MMasseyOrdinals_filtered.csv`: Filtered team ranking data from various ranking systems

## Data Format
Each CSV file contains structured data with consistent formats. Key identifiers include:
- `Season`: The season year (e.g., 2023 for the 2022-2023 season)
- `TeamID`: Unique identifier for each team
- `DayNum`: Day number within the season

## Usage
These raw data files are processed by scripts in the parent directory to create the structured datasets used for model training and prediction. Do not modify these files directly, as they serve as the source of truth for all data processing.

## Data Sources
The data is derived from historical NCAA basketball tournament records and regular season statistics, formatted for machine learning applications.
