import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

# Example data
df = pd.read_csv('data/ev_registrations_by_state_year.csv')
state = 'WA'
df_state = df[df['state'] == state].groupby('model_year')['registrations'].sum().reset_index()

# Simulate external variables (like fuel price, charging station density)
np.random.seed(42)
df_state['fuel_price'] = np.random.uniform(3.0, 5.0, len(df_state))
df_state['stations'] = np.random.randint(50, 300, len(df_state))

# Run Granger causality test
print("🚗 Granger Causality Test Results (Does X cause Y?)")
for var in ['fuel_price', 'stations']:
    print(f"\nTesting if {var} causes EV registrations:")
    result = grangercausalitytests(df_state[['registrations', var]], maxlag=2, verbose=False)
    for lag, test in result.items():
        p_value = test[0]['ssr_chi2test'][1]
        if p_value < 0.05:
            print(f"  ✅ Lag {lag}: Significant causality (p={p_value:.4f})")
        else:
            print(f"  ❌ Lag {lag}: No significant causality (p={p_value:.4f})")
# --- SAVE Granger causality p-values to CSV for dashboard ---
import pandas as pd
from pathlib import Path

# Collect results (replace with your actual computed p-values)
results = [
    {"Variable": "Fuel Price", "Lag": 1, "p-value": 0.4356},
    {"Variable": "Fuel Price", "Lag": 2, "p-value": 0.8164},
    {"Variable": "Charging Stations", "Lag": 1, "p-value": 0.0994},
    {"Variable": "Charging Stations", "Lag": 2, "p-value": 0.6246},
]

granger_df = pd.DataFrame(results)

# Save to /reports/results/
output_dir = Path("reports/results")
output_dir.mkdir(parents=True, exist_ok=True)
granger_df.to_csv(output_dir / "granger_results.csv", index=False)

print("✅ Granger causality results saved to reports/results/granger_results.csv")
