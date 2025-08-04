"""
NBA Rookie Rotation Predictor - Player Name Interface

This module provides a command-line interface for predicting whether an NBA rookie
will become a rotation player (15+ minutes per game) based on their statistical
performance. The prediction is made using a trained machine learning model that
analyzes key performance metrics.

The model uses features such as points per game, assists, rebounds, advanced
metrics (VORP, BPM, PER), and usage rate to make predictions about a player's
future role in the NBA.

Author: Bejamin Wong
Date: 2025-08-04
Version: 1.0
"""

import pandas as pd
import joblib
import argparse
import os
import sys

# Configuration constants
DATA_PATH = "data/processed/labeled_rookie_data.csv"  # Path to the labeled dataset
MODEL_PATH = "data/processed/final_model.joblib"      # Path to the trained model
FEATURES = ['PTS', 'AST', 'TRB', 'VORP', 'BPM', 'PER', 'USG%']  # Features used for prediction

def load_model():
    """
    Load the trained machine learning model from disk.
    
    Returns:
        object: The loaded scikit-learn model
        
    Raises:
        SystemExit: If the model file is not found
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at: {MODEL_PATH}")
        print("Please ensure the model has been trained and saved correctly.")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_data():
    """
    Load and preprocess the labeled rookie dataset.
    
    The dataset contains historical rookie statistics with binary labels
    indicating whether each player became a rotation player (1) or not (0).
    
    Returns:
        pandas.DataFrame: Preprocessed dataset with normalized player names
        
    Raises:
        SystemExit: If the data file is not found
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at: {DATA_PATH}")
        print("Please ensure the dataset has been processed and saved correctly.")
        sys.exit(1)
    
    try:
        # Load the CSV file
        df = pd.read_csv(DATA_PATH)
        
        # Normalize player names for consistent matching
        # Convert to string, strip whitespace, and convert to lowercase
        df['Player'] = df['Player'].astype(str).str.strip().str.lower()
        
        print(f"Successfully loaded dataset with {len(df)} players")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def predict_for_player(player_name):
    """
    Predict whether a player will become a rotation player based on their name.
    
    This function performs the following steps:
    1. Loads the dataset and trained model
    2. Searches for the player in the dataset using fuzzy matching
    3. Extracts the player's statistical features
    4. Makes a prediction using the trained model
    5. Displays the results and supporting statistics
    
    Args:
        player_name (str): The name of the player to predict for
        
    Returns:
        None: Results are printed to console
    """
    # Load required data and model
    print(f"Searching for player: {player_name}")
    df = load_data()
    model = load_model()

    # Normalize user input for consistent matching
    player_input = player_name.strip().lower()

    # Find partial matches in the dataset
    # This allows for flexible name matching (e.g., "lebron" matches "LeBron James")
    matches = df[df['Player'].str.contains(player_input, na=False)]

    # Handle case where no matches are found
    if matches.empty:
        print(f"\nNo matches found for '{player_name}'.")
        print("This could mean:")
        print("- The player is not in our dataset")
        print("- There's a typo in the name")
        print("- The player name needs to be more specific")
        print("\nSample players in dataset:")
        sample_players = df['Player'].sample(min(10, len(df))).tolist()
        for player in sample_players:
            print(f"  - {player.title()}")
        return

    # Handle case where multiple matches are found
    elif len(matches) > 1:
        print(f"\nMultiple matches found for '{player_name}':")
        for player in matches['Player'].unique():
            print(f"  - {player.title()}")
        print("\nPlease be more specific with the player name.")
        return

    # Get the single matching player
    player_row = matches.iloc[0]
    
    # Extract the features used by the model
    # Reshape to 2D array as required by scikit-learn
    input_features = player_row[FEATURES].values.reshape(1, -1)

    # Make prediction using the trained model
    prediction = model.predict(input_features)[0]

    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    print(f"\nPlayer: {player_row['Player'].title()}")
    print(f"Season: {player_row.get('Season', 'N/A')}")
    
    print("\nStatistical Features Used:")
    print("-" * 30)
    for feature in FEATURES:
        value = player_row[feature]
        print(f"{feature:>6}: {value:>8.2f}")
    
    print("\nModel Prediction:")
    print("-" * 20)
    if prediction == 1:
        print(f" {player_name.title()} is predicted to become a rotation player (15+ MPG)")
        print("   This means the model expects them to play significant minutes in the NBA.")
    else:
        print(f" {player_name.title()} is NOT predicted to become a rotation player")
        print("   This means the model expects them to have limited playing time.")
    
    print("\nNote: This prediction is based on historical data and statistical patterns.")
    print("Individual player development can vary significantly.")

def main():
    """
    Main function that handles command-line argument parsing and initiates prediction.
    
    This function sets up the argument parser for the command-line interface
    and calls the prediction function with the provided player name.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Predict NBA rookie rotation status from player name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_rotation_from_name.py --player "Jayson Tatum"
  python predict_rotation_from_name.py --player "lebron"
  python predict_rotation_from_name.py --player "curry"
        """
    )
    
    parser.add_argument(
        "--player", 
        type=str, 
        required=True, 
        help="Player name to predict for (e.g., 'Jayson Tatum', 'lebron')"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute prediction
    predict_for_player(args.player)

if __name__ == "__main__":
    main()
