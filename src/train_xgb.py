# src/train_xgb.py
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

# Load cleaned data
df = pd.read_csv('data/ev_registrations_by_state_year.csv')
df = df.sort_values(['state', 'model_year'])

# Create lag features (use previous years as predictors)
def make_lag_features(df, lags=3):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df.groupby('state')['registrations'].shift(lag)
    return df

df_feat = make_lag_features(df, lags=3)
df_feat = df_feat.dropna()  # remove first few rows without lags

# Feature and target columns
X = df_feat[['lag_1', 'lag_2', 'lag_3']]
y = df_feat['registrations']

# Split into train/test — last 2 years as test
latest_year = df_feat['model_year'].max()
train = df_feat[df_feat['model_year'] < latest_year - 1]
test = df_feat[df_feat['model_year'] >= latest_year - 1]

X_train, y_train = train[X.columns], train[y.name]
X_test, y_test = test[X.columns], test[y.name]

print("Training rows:", X_train.shape[0], "Testing rows:", X_test.shape[0])

# Train the XGBoost model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

print(f"✅ MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_ev_forecast.pkl')
print("✅ Model saved to models/xgb_ev_forecast.pkl")
