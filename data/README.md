# Data Directory

## Overview
This directory contains all data-related files for the Madness-Predictor March Madness prediction project, including raw NCAA tournament data, data processing scripts, and processed datasets ready for model training.

## Directory Structure
- `/raw`: Original NCAA basketball data files
- `/processed`: Cleaned and transformed data files ready for model consumption

## Key Files

### Scripts
- `generate_matchup_data.py`: Creates matchup data for tournament predictions
- `generate_training_data.py`: Prepares training datasets for the neural network model
- `process_quarter_data.py`: Processes data by season quarters for more accurate predictions
- `process_data.sh`: Shell script to automate the data processing pipeline
- `score_distributions.py`: Analyzes score distributions from historical games

### Data Flow
1. Raw NCAA data is stored in the `/raw` directory
2. Processing scripts transform raw data into structured formats
3. Processed data is saved to the `/processed` directory
4. Training data is generated for model training
5. Matchup data is created for tournament predictions

## Usage
To process raw data and generate training datasets:

```bash
# Run the data processing pipeline
./process_data.sh

# Generate training data from processed files
python generate_training_data.py
```

## Data Description
The processed data includes team statistics, tournament results, and matchup information formatted for the neural network model. Features include offensive and defensive metrics, tempo-adjusted statistics, and historical performance indicators.
