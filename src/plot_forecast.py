# src/plot_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load actual data and forecast
actual = pd.read_csv('data/ev_registrations_by_state_year.csv')
forecast = pd.read_csv('reports/forecast_next_year.csv')

# Select the state you want to visualize
STATE = 'WA'  # Change if you want another state

# Combine actual + forecast
state_data = actual[actual['state'] == STATE].copy()
forecast_state = forecast[forecast['state'] == STATE]

forecast_state.rename(columns={'predicted_registrations': 'registrations'}, inplace=True)
forecast_state['is_forecast'] = True
state_data['is_forecast'] = False

combined = pd.concat([state_data, forecast_state])

# Create output folder
os.makedirs('reports/figs', exist_ok=True)

# Plot setup
plt.figure(figsize=(8, 5))

# Plot actual (blue solid line)
plt.plot(state_data['model_year'], state_data['registrations'],
         marker='o', color='tab:blue', label='Actual')

# Plot forecast (orange dashed line)
plt.plot(forecast_state['model_year'], forecast_state['registrations'],
         marker='x', color='tab:orange', linestyle='--', label='Forecast')

# Titles and labels
plt.title(f'EV Registrations for {STATE} (with forecast)')
plt.xlabel('Model Year')
plt.ylabel('Registrations')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
out_path = f'reports/figs/forecast_{STATE}.png'
plt.savefig(out_path)
print(f"✅ Saved {out_path}")
plt.show()
