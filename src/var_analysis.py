import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Example simulated external data
df = pd.read_csv('data/ev_registrations_by_state_year.csv')
state = 'WA'
df_state = df[df['state'] == state].groupby('model_year')['registrations'].sum().reset_index()

# Add dummy variables (e.g., fuel price, charging station density)
np.random.seed(42)
df_state['fuel_price'] = np.random.uniform(3.0, 5.0, len(df_state))
df_state['stations'] = np.random.randint(50, 300, len(df_state))

df_state = df_state.dropna()

# Train/test split
train = df_state[:-2]
test = df_state[-2:]

# Fit VAR model
model = VAR(train[['registrations', 'fuel_price', 'stations']])
results = model.fit(maxlags=2)

# Forecast next 2 years
forecast = results.forecast(train[['registrations', 'fuel_price', 'stations']].values, steps=2)
forecast_df = pd.DataFrame(forecast, columns=['registrations_pred', 'fuel_price_pred', 'stations_pred'])

# Evaluate
y_true = test['registrations'].values
y_pred = forecast_df['registrations_pred'].values
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"✅ VAR Forecasting for {state}")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.plot(df_state['model_year'], df_state['registrations'], label='Actual', marker='o')
plt.plot(test['model_year'], y_pred, label='Forecast', marker='x', linestyle='--')
plt.title(f"VAR Forecast for EV Registrations - {state}")
plt.xlabel("Model Year")
plt.ylabel("Registrations")
plt.legend()
plt.tight_layout()
# --- SAVE PLOT ---
import os
os.makedirs("reports/figs", exist_ok=True)
plt.savefig("reports/figs/var_forecast_WA.png", dpi=300, bbox_inches="tight")
print("✅ VAR forecast plot saved to reports/figs/var_forecast_WA.png")

plt.show()
