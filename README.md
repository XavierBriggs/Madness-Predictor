# Madness-Predictor: March Madness Prediction System

A machine learning-based system for predicting NCAA March Madness basketball tournament outcomes using team performance metrics and neural networks.

## Project Overview

XG-MM analyzes basketball team performance data on a quarterly basis, generates feature matrices, and uses a neural network model to predict game outcomes in the NCAA March Madness tournament. The system processes historical game data, calculates advanced basketball metrics, and makes predictions for tournament matchups.

## Directory Structure

- **data/**: Contains raw data, processed datasets, and data processing scripts
- **models/**: Contains the neural network model definition and trained model files
- **src/**: Contains source code for prediction generation and tournament simulation

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/XG-MM.git
   cd XG-MM
   ```

2. Install required dependencies:
   ```
   pip install torch pandas numpy matplotlib
   ```

### Usage

1. Process raw data:
   ```
   cd data
   python process_quarter_data.py
   ```

2. Train the model:
   ```
   cd models
   python model.py
   ```

3. Generate predictions for a tournament round:
   ```
   cd src
   python predictor.py <round_number>
   ```

4. Generate next round matchups based on predictions:
   ```
   cd src
   python generate_rr.py <round_number>
   ```

## Model Architecture

The system uses a neural network with multiple fully connected layers and ReLU activations. The output layer uses a softmax activation to produce win probabilities for each team in a matchup.

## Data Features

The system calculates and uses the following basketball metrics:
- True Shooting Percentage (TS%)
- Effective Field Goal Percentage (eFG%)
- Turnover Percentage (TO%)
- Offensive and Defensive Rebound Percentages (OREB%, DREB%)
- Free Throw Rate (FTR)
- Three-Point Attempt Rate (3PAr)
- Assist to Turnover Ratio (AST/TO)
- Steal and Block Percentages (STL%, BLK%)
- Adjusted Offensive and Defensive Ratings

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute this software, provided that proper attribution is given.

## Contact
- Notre Dame SAC MBB Memebers
- Xavier Briggs - xbriggs@nd.edu [linkedin.com/in/xavierbriggs05](https://linkedin.com/in/xavierbriggs05)
- George Kyrollos - gkyrollo@nd.edu (https://www.linkedin.com/in/george-kyrollos-744047286/)
- Andre Mayard - amayard@nd.edu https://www.linkedin.com/in/andre-mayard-907876285/



