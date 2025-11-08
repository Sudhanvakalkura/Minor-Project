# src/feature_importance.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load model
model = joblib.load('models/xgb_ev_forecast.pkl')

# Feature names (must match training)
features = ['lag_1', 'lag_2', 'lag_3']

# Get feature importance from model
importance = model.feature_importances_

# ✅ Normalize importance values (so max = 1)
importance_normalized = importance / np.max(importance)

# Create DataFrame
df_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importance,
    'Normalized Importance': importance_normalized
}).sort_values('Normalized Importance', ascending=False)

# ✅ Apply log scaling for visualization clarity (optional but presentation-friendly)
df_imp['Scaled Importance'] = np.log1p(df_imp['Normalized Importance'] * 100)  # log1p avoids log(0)

print("Feature importances (normalized):\n", df_imp)

# --- Plot scaled importance ---
os.makedirs('reports/figs', exist_ok=True)
plt.figure(figsize=(6,4))
plt.barh(df_imp['Feature'], df_imp['Scaled Importance'], color='teal')
plt.gca().invert_yaxis()
plt.title('Feature Importance (XGBoost, log-scaled)')
plt.xlabel('Log-Scaled Importance')
plt.tight_layout()
plt.savefig('reports/figs/feature_importance_xgb.png', dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Saved log-scaled feature importance to reports/figs/feature_importance_xgb.png")
