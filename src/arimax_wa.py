import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ensure output folder exists
os.makedirs('reports/figs', exist_ok=True)

# Load state and national data
state_df = pd.read_csv('data/ts/WA_year.csv')
nat_df = pd.read_csv('data/ts/national_total_year.csv')

# Merge on model_year
merged = pd.merge(state_df, nat_df, on='model_year', suffixes=('_state', '_nat'))
merged = merged.sort_values('model_year')

# Use yearly PeriodIndex
merged['model_year'] = merged['model_year'].astype(int)
merged = merged.set_index(pd.PeriodIndex(merged['model_year'], freq='Y'))

# Target and exogenous variable
y = merged['registrations_state'].astype(float)
X = merged[['registrations_nat']].astype(float)

# Train-test split (last year as test)
h = 1
train_y, test_y = y.iloc[:-h], y.iloc[-h:]
train_X, test_X = X.iloc[:-h], X.iloc[-h:]

# Fit ARIMAX
order = (1, 1, 1)
model = SARIMAX(train_y, exog=train_X, order=order,
                enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

# Forecast next h steps
pred = res.get_forecast(steps=h, exog=test_X)
forecast_mean = pred.predicted_mean.iloc[0]
ci = pred.conf_int().iloc[0]

print(f"Predicted next year registrations (ARIMAX): {forecast_mean:.2f}")
print(f"95% Confidence Interval: [{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}]")

# Compute error manually
true_val = test_y.iloc[0]
mae = abs(true_val - forecast_mean)
rmse = ((true_val - forecast_mean) ** 2) ** 0.5
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot actual, fitted, forecast
plt.figure(figsize=(9,5))
plt.plot(y.index.to_timestamp(), y.values, marker='o', label='Actual')
plt.plot(train_y.index.to_timestamp(), res.fittedvalues, marker='.', label='Fitted')

# Forecast visualization
forecast_index = [(train_y.index[-1] + i + 1).to_timestamp() for i in range(h)]
plt.plot(forecast_index, [forecast_mean], marker='x', linestyle='--', color='tab:orange', label='Forecast')
plt.fill_between(forecast_index, ci.iloc[0], ci.iloc[1], color='orange', alpha=0.2)

plt.axvline(train_y.index[-1].to_timestamp(), color='gray', linestyle='--', label='Train/Test split')
plt.title('ARIMAX Forecast for WA (with national total as exogenous variable)')
plt.xlabel('Year')
plt.ylabel('Registrations')
plt.legend()
plt.tight_layout()

outpath = 'reports/figs/arimax_WA_forecast.png'
plt.savefig(outpath)
print(f"Saved plot to {outpath}")
plt.show()
