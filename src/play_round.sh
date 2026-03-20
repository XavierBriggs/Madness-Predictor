#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: ./play_round.sh <round_number>"
    exit 1
fi

ROUND=$1
NEXT=$((ROUND + 1))

echo "Running predictor.py for round $ROUND..."
python3 predictor.py --round $ROUND

echo "Running generate_rr.py for round $NEXT..."
python3 generate_rr.py --round $NEXT

echo "Running generate_matchup_data.py for round $NEXT..."
cd ../data || exit
python3 generate_matchup_data.py --round $NEXT
cd ../src || exit

echo "Round $ROUND processing complete!"
