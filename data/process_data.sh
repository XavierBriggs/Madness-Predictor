#!/bin/bash
set -e

echo "Processing game data..."
python3 process_quarter_data.py

echo "Generating training data..."
python3 generate_training_data.py

echo "Generating round 1 matchups..."
python3 generate_matchup_data.py --round 1

echo "Complete"
