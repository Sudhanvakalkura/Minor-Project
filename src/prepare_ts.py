import pandas as pd
import os

# Input and output
IN = 'data/ev_registrations_by_state_year.csv'
OUT = 'data/ts'
os.makedirs(OUT, exist_ok=True)

# Load the dataset
df = pd.read_csv(IN)
df['model_year'] = df['model_year'].astype(int)
df = df.sort_values(['state', 'model_year'])
df = df[df['model_year'] <= 2023]

# National total by year
national = df.groupby('model_year')['registrations'].sum().reset_index()
national.to_csv(os.path.join(OUT, 'national_total_year.csv'), index=False)
print("✅ Saved national totals")

# Save one CSV per state
for state in df['state'].unique():
    s = df[df['state'] == state][['model_year', 'registrations']]
    s.to_csv(os.path.join(OUT, f'{state}_year.csv'), index=False)
print("✅ Saved per-state CSVs in data/ts/")
