# src/run_all_arima_arimax.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt

# Make sure output folders exist
os.makedirs('reports/figs', exist_ok=True)
os.makedirs('reports/results', exist_ok=True)

states_dir = 'data/ts'
nat_df = pd.read_csv(os.path.join(states_dir, 'national_total_year.csv'))

results = []

# List all state files except national_total_year.csv
for file in os.listdir(states_dir):
    if not file.endswith('_year.csv') or file == 'national_total_year.csv':
        continue

    state = file.replace('_year.csv', '')
    print(f"\n=== Processing {state} ===")

    # Load state data
    state_df = pd.read_csv(os.path.join(states_dir, file))
    merged = pd.merge(state_df, nat_df, on='model_year', suffixes=('_state', '_nat'))
    merged = merged.sort_values('model_year')
    merged['model_year'] = merged['model_year'].astype(int)
    merged = merged.set_index(pd.PeriodIndex(merged['model_year'], freq='Y'))

    y = merged['registrations_state'].astype(float)
    X = merged[['registrations_nat']].astype(float)
    if len(y) < 4:
        print(f"Skipping {state}: not enough data points")
        continue

    # Split data
    h = 1
    train_y, test_y = y.iloc[:-h], y.iloc[-h:]
    train_X, test_X = X.iloc[:-h], X.iloc[-h:]

    # --- ARIMA ---
    try:
        arima_model = SARIMAX(train_y, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        arima_res = arima_model.fit(disp=False)
        arima_pred = arima_res.get_forecast(steps=h)
        arima_forecast = arima_pred.predicted_mean.iloc[0]
        arima_mae = abs(test_y.iloc[0] - arima_forecast)
        arima_rmse = ((test_y.iloc[0] - arima_forecast)**2)**0.5
        results.append([state, 'ARIMA', arima_mae, arima_rmse, arima_forecast])
    except Exception as e:
        print(f"ARIMA failed for {state}: {e}")

    # --- ARIMAX ---
    try:
        arimax_model = SARIMAX(train_y, exog=train_X, order=(1,1,1),
                               enforce_stationarity=False, enforce_invertibility=False)
        arimax_res = arimax_model.fit(disp=False)
        arimax_pred = arimax_res.get_forecast(steps=h, exog=test_X)
        arimax_forecast = arimax_pred.predicted_mean.iloc[0]
        arimax_mae = abs(test_y.iloc[0] - arimax_forecast)
        arimax_rmse = ((test_y.iloc[0] - arimax_forecast)**2)**0.5
        results.append([state, 'ARIMAX', arimax_mae, arimax_rmse, arimax_forecast])

        # Save plots
        plt.figure(figsize=(8,4))
        plt.plot(y.index.to_timestamp(), y.values, label='Actual', marker='o')
        plt.plot(train_y.index.to_timestamp(), arimax_res.fittedvalues, label='Fitted', marker='.')
        forecast_index = [(train_y.index[-1] + i + 1).to_timestamp() for i in range(h)]
        plt.plot(forecast_index, [arimax_forecast], marker='x', linestyle='--', color='orange', label='Forecast')
        plt.title(f'ARIMAX Forecast - {state}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'reports/figs/arimax_{state}.png')
        plt.close()
    except Exception as e:
        print(f"ARIMAX failed for {state}: {e}")

# Save results to CSV
res_df = pd.DataFrame(results, columns=['State', 'Model', 'MAE', 'RMSE', 'Forecast'])
res_df.to_csv('reports/results/arima_comparison.csv', index=False)
print("\n✅ Results saved to reports/results/arima_comparison.csv")
print(res_df.head())
