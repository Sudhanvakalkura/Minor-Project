# src/inspect_raw.py
import pandas as pd

# Path to your CSV inside data_raw
fp = 'data_raw/Electric_Vehicle_Population_Data.csv'

print("Loading:", fp)
df = pd.read_csv(fp, low_memory=False)
print("Rows, Cols:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head().to_string())
