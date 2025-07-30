import pandas as pd

# Load the CSV you moved from Kaggle
df = pd.read_csv("data/raw/nba_season_stats.csv")

# Print out the first 5 rows
print(df.head())

# Print all the column names
print("\nColumns:")
print(df.columns)
