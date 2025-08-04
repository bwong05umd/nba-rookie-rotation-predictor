import pandas as pd
import joblib
import argparse
import os
import sys

# Config
DATA_PATH = "data/processed/labeled_rookie_data.csv"
MODEL_PATH = "data/processed/final_model.joblib"
FEATURES = ['PTS', 'AST', 'TRB', 'VORP', 'BPM', 'PER', 'USG%']

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found at: {MODEL_PATH}")
        sys.exit(1)
    return joblib.load(MODEL_PATH)

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f" Data not found at: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    df['Player'] = df['Player'].astype(str).str.strip().str.lower()
    return df

def predict_for_player(player_name):
    df = load_data()
    model = load_model()

    # Normalize user input
    player_input = player_name.strip().lower()

    # Find partial matches
    matches = df[df['Player'].str.contains(player_input, na=False)]

    if matches.empty:
        print(f" No matches found for '{player_name}'.")
        print(" Sample players in dataset:")
        print(df['Player'].sample(10).tolist())
        return

    elif len(matches) > 1:
        print(f" Multiple matches found for '{player_name}':")
        print(matches['Player'].unique())
        print(" Please be more specific.")
        return

    player_row = matches.iloc[0]
    input_features = player_row[FEATURES].values.reshape(1, -1)

    prediction = model.predict(input_features)[0]

    print("\n Player Stats Used:")
    print(player_row[FEATURES])

    print("\n Model Prediction:")
    if prediction == 1:
        print(f" {player_name.title()} is predicted to become a rotation player (15+ MPG).")
    else:
        print(f" {player_name.title()} is NOT predicted to become a rotation player.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rookie rotation status from name")
    parser.add_argument("--player", type=str, required=True, help="Player name (e.g., 'Jayson Tatum')")
    args = parser.parse_args()

    predict_for_player(args.player)
