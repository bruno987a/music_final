import pandas as pd
import sqlite3
from pathlib import Path

# Define paths to CSV files in a universal way
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# read CSVs
genres = pd.read_csv(DATA_DIR / "genre_with_main_identity.csv")
tracks_small = pd.read_csv(DATA_DIR / "tracks_small.csv")
features = pd.read_csv(DATA_DIR / "reduced_features.csv", index_col=0)  # track_id index

# set up SQLite database file
DB = sqlite3.connect(DATA_DIR / "app.db")

# create tables
# with 'replace' feature in case they already exist
genres.to_sql("genre_with_main_identity", DB, if_exists="replace", index=False)
tracks_small.to_sql("tracks_small", DB, if_exists="replace", index=False)

# convert index into normal column
features.reset_index(names="track_id", inplace=True)
features.to_sql("features", DB, if_exists="replace", index=False)

DB.close()
print("Database built: data/app.db")