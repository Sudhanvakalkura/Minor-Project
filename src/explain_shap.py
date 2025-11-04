import pandas as pd
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import tempfile

# =====================================
# 1. Load trained XGBoost model
# =====================================
print("✅ Loading trained model...")
model = joblib.load("models/xgb_ev_forecast.pkl")
print("✅ Model loaded successfully")

# =====================================
# 2. Load dataset and recreate lag features
# =====================================
df = pd.read_csv("data/ev_registrations_by_state_year.csv")
print(f"Data for SHAP: {len(df)} samples, {len(df.columns)} features")

# Filter for one state (used during training)
df = df[df["state"] == "WA"].copy()  # Change WA if needed
df = df.sort_values("model_year")

# Recreate lag features (same as training script)
df["lag_1"] = df["registrations"].shift(1)
df["lag_2"] = df["registrations"].shift(2)
df["lag_3"] = df["registrations"].shift(3)
df = df.dropna()

X = df[["lag_1", "lag_2", "lag_3"]]
print(f"✅ Using features: {list(X.columns)}")

# =====================================
# 3. Clean booster to fix SHAP compatibility
# =====================================
booster = model.get_booster()

# Save & reload booster safely
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
    booster.save_model(tmp.name)
    tmp_path = tmp.name

clean_booster = xgb.Booster()
clean_booster.load_model(tmp_path)
print("✅ Cleaned XGBoost booster successfully")

# =====================================
# 4. Run SHAP
# =====================================
explainer = shap.TreeExplainer(clean_booster)
shap_values = explainer(X)

# =====================================
# 5. Save SHAP plots
# =====================================
os.makedirs("reports/figs", exist_ok=True)
print("✅ Generating SHAP plots...")

# SHAP summary plot
shap.summary_plot(shap_values.values, X, show=False)
plt.title("SHAP Summary Plot for XGBoost Model")
plt.savefig("reports/figs/shap_summary.png", bbox_inches="tight", dpi=300)
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values.values, X, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP Values)")
plt.savefig("reports/figs/shap_bar.png", bbox_inches="tight", dpi=300)
plt.close()

print("✅ SHAP plots saved to reports/figs/")
print("✅ Done.")
