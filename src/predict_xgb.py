# src/predict_xgb.py
import pandas as pd
import joblib
import numpy as np
import os

# Load the trained model
model = joblib.load('models/xgb_ev_forecast.pkl')

# Load your cleaned data
df = pd.read_csv('data/ev_registrations_by_state_year.csv')
df = df.sort_values(['state', 'model_year'])

# Create lag features (same as during training)
for lag in range(1, 4):
    df[f'lag_{lag}'] = df.groupby('state')['registrations'].shift(lag)

# Get latest row for each state (most recent year’s data)
latest = df.groupby('state').tail(1).copy()

# Predict the next year’s registrations for each state
predictions = []
for _, row in latest.iterrows():
    state = row['state']
    model_year = int(row['model_year'])
    features = row[['lag_1', 'lag_2', 'lag_3']].values.reshape(1, -1)
    pred_next = model.predict(features)[0]
    predictions.append({
        'state': state,
        'model_year': model_year + 1,
        'predicted_registrations': int(pred_next)
    })

forecast = pd.DataFrame(predictions)

# Save to file
os.makedirs('reports', exist_ok=True)
forecast.to_csv('reports/forecast_next_year.csv', index=False)

print("\nPredicted EV registrations for next year:")
print(forecast)
print("\n✅ Saved forecast to reports/forecast_next_year.csv")
