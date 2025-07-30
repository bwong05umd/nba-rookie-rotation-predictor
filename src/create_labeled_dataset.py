import pandas as pd

# Load the CSV
df = pd.read_csv("data/raw/nba_season_stats.csv")

# Drop unnecessary rows
df = df.dropna(subset=['MP', 'G', 'Player', 'Year'])
df = df[df['G'] > 0]

# Ensure numeric types
df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
df['G'] = pd.to_numeric(df['G'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Compute Minutes Per Game
df['MPG'] = df['MP'] / df['G']

# Identify each player's rookie year
rookie_year_df = df.groupby('Player')['Year'].min().reset_index()
rookie_year_df.columns = ['Player', 'Rookie_Year']

# Merge rookie year back into full dataset
df = df.merge(rookie_year_df, on='Player')

# Filter players whose rookie year is between 2010 and 2020
df = df[df['Rookie_Year'].between(2010, 2020)]

# Define label: did player play ≥15 MPG in any of first 3 years?
def is_rotation_player(player_df):
    first_3 = player_df.sort_values('Year').head(3)
    return int((first_3['MPG'] >= 15).any())

labels = df.groupby('Player').apply(is_rotation_player).reset_index()
labels.columns = ['Player', 'is_rotation_player']

# Merge label into full dataset
df_labeled = df.merge(labels, on='Player')

# Keep only rookie year stats for each player
rookie_df = df_labeled[df_labeled['Year'] == df_labeled['Rookie_Year']]

# Select relevant columns
columns_to_keep = [
    'Player', 'Year', 'MP', 'G', 'MPG', 'Pos', 'Age', 'Tm',
    'PER', 'USG%', 'VORP', 'BPM', 'PTS', 'AST', 'TRB', 'is_rotation_player'
]
rookie_df = rookie_df[[col for col in columns_to_keep if col in rookie_df.columns]]

# Save to CSV
rookie_df.to_csv("data/processed/labeled_rookie_data.csv", index=False)
print("Labeled dataset saved to data/processed/labeled_rookie_data.csv")

