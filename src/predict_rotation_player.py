import pandas as pd
import joblib

# Load the trained model
model = joblib.load("data/processed/final_model.joblib")

# Features used in training (must match the model's expectations!)
feature_cols = ['PTS', 'AST', 'TRB', 'VORP', 'BPM', 'PER', 'USG%']

# Sample rookie stats — you can edit this to test new rookies
new_player = {
    'PTS': 450,
    'AST': 100,
    'TRB': 150,
    'VORP': 0.3,
    'BPM': 0.8,
    'PER': 14.0,
    'USG%': 19.0
}

# Convert to DataFrame and reorder columns just in case
df = pd.DataFrame([new_player])[feature_cols]

# Predict
prediction = model.predict(df)[0]

# Output
print("Prediction:", "Rotation Player" if prediction == 1 else "Not a Rotation Player")
