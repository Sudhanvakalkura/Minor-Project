# src/arima_wa.py  (corrected)
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ensure output folder exists
os.makedirs('reports/figs', exist_ok=True)

# Load data
df = pd.read_csv('data/ts/WA_year.csv')
df = df.sort_values('model_year')

# Use a PeriodIndex so statsmodels can produce forecasts with proper time index
df['model_year'] = df['model_year'].astype(int)
df = df.set_index(pd.PeriodIndex(df['model_year'], freq='Y'))
series = df['registrations'].astype(float)

# Train-test split (last 1 year as test)
h = 1
if len(series) <= h:
    raise SystemExit("Not enough data points for train/test split. Need more years.")
train = series.iloc[:-h]
test = series.iloc[-h:]

# Fit ARIMA model (adjust order as needed)
order = (1, 1, 1)
model = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

# Forecast next h steps
pred = res.get_forecast(steps=h)
forecast_mean = pred.predicted_mean.iloc[0]
ci = pred.conf_int().iloc[0]

# Print forecast and CI
print(f"Predicted next year registrations: {forecast_mean:.2f}")
print(f"95% Confidence Interval: [{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}]")

# Evaluate against the held-out test
# Get prediction at test index (use same index)
y_pred_series = res.get_prediction(start=test.index[0], end=test.index[0]).predicted_mean
y_pred_val = y_pred_series.iloc[0]
y_true_val = test.iloc[0]

mae = mean_absolute_error([y_true_val], [y_pred_val])
rmse = mean_squared_error([y_true_val], [y_pred_val]) ** 0.5  # compute sqrt(MSE
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot actual, fitted, and forecast
plt.figure(figsize=(9,5))
plt.plot(series.index.to_timestamp(), series.values, marker='o', label='Actual')
plt.plot(train.index.to_timestamp(), res.fittedvalues, marker='.', label='Fitted')
# Plot forecast point(s)
forecast_index = [ (train.index[-1] + i + 1).to_timestamp() for i in range(h) ]
plt.plot(forecast_index, [forecast_mean], marker='x', linestyle='--', color='tab:orange', label='Forecast')
# CI shading
plt.fill_between(forecast_index, ci.iloc[0], ci.iloc[1], color='orange', alpha=0.2)

plt.axvline(train.index[-1].to_timestamp(), color='gray', linestyle='--', label='Train/Test split')
plt.title('ARIMA Forecast for WA')
plt.xlabel('Year')
plt.ylabel('Registrations')
plt.legend()
plt.tight_layout()

outpath = 'reports/figs/arima_WA_forecast.png'
plt.savefig(outpath)
print(f"Saved plot to {outpath}")
plt.show()
