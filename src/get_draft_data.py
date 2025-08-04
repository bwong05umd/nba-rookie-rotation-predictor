from nba_api.stats.endpoints import drafthistory
import pandas as pd

print(" Requesting data from DraftHistory endpoint...")

try:
    response = drafthistory.DraftHistory(league_id='00')
    data_frames = response.get_data_frames()
    print(f" Number of dataframes returned: {len(data_frames)}")

    if data_frames:
        df = data_frames[0]
        print(" First 5 rows:")
        print(df.head())

        print("\n Available columns:")
        print(df.columns)
    else:
        print(" No data returned from endpoint.")
except Exception as e:
    print(f" Error occurred: {e}")
