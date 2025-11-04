# src/feature_importance.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model
model = joblib.load('models/xgb_ev_forecast.pkl')

# Feature names (must match training)
features = ['lag_1', 'lag_2', 'lag_3']

# Get feature importance from model
importance = model.feature_importances_
df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
df_imp = df_imp.sort_values('Importance', ascending=False)

print("Feature importances:\n", df_imp)

# Plot
os.makedirs('reports/figs', exist_ok=True)
plt.figure(figsize=(6,4))
plt.barh(df_imp['Feature'], df_imp['Importance'], color='teal')
plt.gca().invert_yaxis()
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('reports/figs/feature_importance_xgb.png')
plt.show()

print("\n✅ Saved reports/figs/feature_importance_xgb.png")
