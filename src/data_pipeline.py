# src/data_pipeline.py
import pandas as pd
import os

RAW_PATH = "data_raw/Electric_Vehicle_Population_Data.csv"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_clean(path):
    print("Loading raw file...")
    df = pd.read_csv(path, low_memory=False)
    print("Initial shape:", df.shape)

    # Drop duplicates (in case any VIN repeats)
    df = df.drop_duplicates(subset=['VIN (1-10)'])
    print("After removing duplicates:", df.shape)

    # Clean up column names (remove spaces, lowercase)
    df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "").lower() for c in df.columns]

    # Make sure model_year is numeric
    df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
    df = df.dropna(subset=['model_year'])
    df['model_year'] = df['model_year'].astype(int)

    # Aggregate: total EVs per state per model year
    agg_state = df.groupby(['state', 'model_year']).size().reset_index(name='registrations')
    agg_state = agg_state.sort_values(['state', 'model_year'])
    agg_state.to_csv(os.path.join(OUT_DIR, 'ev_registrations_by_state_year.csv'), index=False)
    print("✅ Saved data/ev_registrations_by_state_year.csv")

    # Also aggregate by county (optional)
    if 'county' in df.columns:
        agg_county = df.groupby(['county', 'model_year']).size().reset_index(name='registrations')
        agg_county = agg_county.sort_values(['county', 'model_year'])
        agg_county.to_csv(os.path.join(OUT_DIR, 'ev_registrations_by_county_year.csv'), index=False)
        print("✅ Saved data/ev_registrations_by_county_year.csv")

    return agg_state.head()

if __name__ == "__main__":
    summary = load_and_clean(RAW_PATH)
    print("\nSample output:\n", summary)
