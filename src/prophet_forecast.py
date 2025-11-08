import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv('data/ev_registrations_by_state_year.csv')
df = df[df['model_year'] <= 2023]
state = 'WA'
df_state = df[df['state'] == state].groupby('model_year')['registrations'].sum().reset_index()

# Prepare data for Prophet
df_prophet = df_state.rename(columns={'model_year': 'ds', 'registrations': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

# Train/test split
train = df_prophet[:-2]
test = df_prophet[-2:]

# Initialize and train model
model = Prophet(yearly_seasonality=False)
model.fit(train)

# Forecast next 3 years
future = model.make_future_dataframe(periods=3, freq='Y')
forecast = model.predict(future)

# Evaluate
y_true = test['y'].values
y_pred = forecast['yhat'][-2:].values
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"✅ Prophet Forecasting for {state}")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
model.plot(forecast, ax=ax)
plt.title(f"Prophet Forecast for EV Registrations - {state}")
plt.xlabel("Year")
plt.ylabel("Registrations")
plt.tight_layout()
# --- SAVE PLOT ---
import os
import matplotlib.pyplot as plt

os.makedirs("reports/figs", exist_ok=True)
plt.savefig("reports/figs/prophet_forecast_WA.png", dpi=300, bbox_inches="tight")
print("✅ Prophet forecast plot saved to reports/figs/prophet_forecast_WA.png")

plt.show()
