# app.py  —  Final EV Growth Dashboard (cloud-ready)
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import io
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="EV Growth Dashboard", layout="wide", page_icon="🚗")
st.title("🚗 Electric Vehicle Growth and Forecast Dashboard")
st.markdown("Analyze EV registration trends and forecasts across U.S. states. (Prophet, VAR, Granger, SHAP integrated)")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data" / "ev_registrations_by_state_year.csv"
FORECAST_XGB = BASE_DIR / "reports" / "forecast_next_year.csv"
ARIMA_COMP = BASE_DIR / "reports" / "results" / "arima_comparison.csv"
MODEL_COMP_ALL = BASE_DIR / "reports" / "results" / "model_comparison_all.csv"
SHAP_SUMMARY = BASE_DIR / "reports" / "figs" / "shap_summary.png"
SHAP_BAR = BASE_DIR / "reports" / "figs" / "shap_bar.png"
FEATURE_IMP = BASE_DIR / "reports" / "figs" / "feature_importance_xgb.png"
FORECAST_WA = BASE_DIR / "reports" / "figs" / "forecast_WA.png"
PROPHET_IMG = BASE_DIR / "reports" / "figs" / "prophet_forecast_WA.png"
VAR_IMG = BASE_DIR / "reports" / "figs" / "var_forecast_WA.png"
GRANGER_CSV = BASE_DIR / "reports" / "results" / "granger_results.csv"

# ---------------- HELPERS ----------------
@st.cache_data
def load_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Failed to read {path.name}: {e}")
            return None
    return None

def safe_img_display(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing image: {path.name}. Generate it by running the script.")

def run_python_script(script_rel_path: Path, timeout=240):
    script_path = BASE_DIR / script_rel_path
    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    try:
        proc = subprocess.run(["python", str(script_path)], capture_output=True, text=True, cwd=str(BASE_DIR), timeout=timeout)
        out = proc.stdout + "\n" + proc.stderr
        return proc.returncode == 0, out
    except Exception as e:
        return False, str(e)

def download_button_for_file(path: Path, label: str, file_name: str):
    if path.exists():
        with open(path, "rb") as f:
            data = f.read()
        st.download_button(label, data, file_name=file_name)
    else:
        st.info(f"{path.name} not found for download.")

# ---------------- LOAD DATA ----------------
df = load_csv(DATA_PATH)
forecast_xgb = load_csv(FORECAST_XGB)
arima_comp = load_csv(ARIMA_COMP)
model_comp_all = load_csv(MODEL_COMP_ALL)
granger_df = load_csv(GRANGER_CSV)

if df is None:
    st.error(f"Data file not found at {DATA_PATH}. Make sure the CSV is present in the data/ folder.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
states = sorted(df['state'].unique())
default_state = "WA" if "WA" in states else states[0]
state = st.sidebar.selectbox("Select a State", states, index=states.index(default_state))
years = sorted(df['model_year'].unique())
st.sidebar.write(f"Data covers {years[0]} – {years[-1]}")

# Model selection
model_options = ["None"]
if forecast_xgb is not None and 'state' in forecast_xgb.columns:
    model_options.append("XGBoost (saved forecast)")
if PROPHET_IMG.exists() or (BASE_DIR / "src" / "prophet_forecast.py").exists():
    model_options.append("Prophet")
if VAR_IMG.exists() or (BASE_DIR / "src" / "var_analysis.py").exists():
    model_options.append("VAR")
if MODEL_COMP_ALL.exists():
    # include models from CSV if present
    for m in model_comp_all['Model'].unique():
        if f"Comparison: {m}" not in model_options:
            model_options.append(f"Comparison: {m}")

selected_model = st.sidebar.selectbox("Select forecast view", model_options)



st.sidebar.markdown("---")
st.sidebar.write("File status:")
for p in [DATA_PATH, FORECAST_XGB, MODEL_COMP_ALL, ARIMA_COMP, FEATURE_IMP, SHAP_SUMMARY, SHAP_BAR, PROPHET_IMG, VAR_IMG, GRANGER_CSV]:
    st.sidebar.write(f"- {p.name} — {'OK' if p.exists() else 'MISSING'}")

# ---------------- MAIN PANEL ----------------
st.subheader(f"📈 EV Registrations in {state}")
state_df = df[df['state'] == state].sort_values("model_year")

fig, ax = plt.subplots(figsize=(9, 4))
sns.lineplot(data=state_df, x='model_year', y='registrations', marker='o', ax=ax)
ax.set_title(f"EV Registrations Over Time — {state}")
ax.set_xlabel("Model Year")
ax.set_ylabel("Registrations")
st.pyplot(fig)

st.markdown("### 🔮 Forecasts / Predictions")
left, right = st.columns(2)

with left:
    # XGBoost saved forecast
    if forecast_xgb is not None and state in forecast_xgb['state'].values:
        st.subheader("XGBoost - Saved Forecast")
        pred_row = forecast_xgb[forecast_xgb['state'] == state].iloc[0]
        try:
            pred_year = int(pred_row['model_year'])
            pred_val = int(pred_row['predicted_registrations'])
            st.metric(label=f"Predicted Registrations for {pred_year}", value=pred_val)
        except Exception:
            st.write(pred_row.to_dict())
        # show forecast image if available
        forecast_img = BASE_DIR / "reports" / "figs" / f"forecast_{state}.png"
        safe_img_display(forecast_img, caption=f"XGBoost: Forecast {state}")

    else:
        st.info("No saved XGBoost forecast found for this state (reports/forecast_next_year.csv).")

    # show model comparison in an expander
    with st.expander("Show Model Comparison Table"):
        if model_comp_all is not None:
            st.dataframe(model_comp_all.set_index('Model'))
        elif arima_comp is not None:
            st.dataframe(arima_comp)
        else:
            st.info("No model comparison CSV found (reports/results/model_comparison_all.csv or arima_comparison.csv).")

with right:
    # show selected model images or messages
    if selected_model == "Prophet":
        if PROPHET_IMG.exists():
            st.subheader("Prophet Forecast")
            safe_img_display(PROPHET_IMG, caption="Prophet Forecast")
        else:
            st.info("Prophet image not found. Run Prophet script (sidebar) to generate it.")
    elif selected_model == "VAR":
        if VAR_IMG.exists():
            st.subheader("VAR Forecast")
            safe_img_display(VAR_IMG, caption="VAR Forecast")
        else:
            st.info("VAR image not found. Run VAR script (sidebar) to generate it.")
    elif selected_model.startswith("Comparison:"):
        model_name = selected_model.replace("Comparison:", "").strip()
        st.subheader(f"Comparison metrics: {model_name}")
        if model_comp_all is not None:
            sub = model_comp_all[model_comp_all['Model'] == model_name]
            if not sub.empty:
                st.table(sub.set_index('Model'))
            else:
                st.info(f"No metrics found for {model_name} in model_comparison_all.csv")

# ---------------- Interpretability / SHAP ----------------
st.markdown("### 🧠 Model Interpretability")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if FEATURE_IMP.exists():
        st.image(str(FEATURE_IMP), caption="Feature importance (XGBoost)", use_container_width=True)
    else:
        st.info("Feature importance image missing.")
with c2:
    safe_img_display(SHAP_SUMMARY, caption="SHAP summary plot")
with c3:
    safe_img_display(SHAP_BAR, caption="SHAP bar plot")

# ---------------- Granger causality ----------------
st.markdown("### 🔎 Granger Causality (p-values)")
if granger_df is not None:
    # Expecting columns like ['variable', 'lag', 'p_value']
    try:
        display_df = granger_df.copy()
        # Nicely format p-values
        if 'p_value' in display_df.columns:
            display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{float(x):.4f}")
        st.table(display_df)
    except Exception:
        st.write(granger_df)
else:
    st.info("No persisted Granger results found. Click 'Run Granger' in the sidebar to compute.")
    if st.button("Run Granger tests now"):
        ok, out = run_python_script(Path("src/granger_test.py"))
        st.text("Success" if ok else "Failed")
        st.code(out[:1000])
        time.sleep(0.5)
        gr_df = load_csv(GRANGER_CSV)
        if gr_df is not None:
            st.success("Granger results saved.")
            st.table(gr_df)
        else:
            st.warning("No granger_results.csv created; check script output above.")

# ---------------- Downloads ----------------
st.markdown("### 📁 Download data & figures")
dl1, dl2 = st.columns(2)
with dl1:
    if MODEL_COMP_ALL.exists():
        with open(MODEL_COMP_ALL, "rb") as f:
            btn = st.download_button("Download Model Comparison CSV", f.read(), file_name="model_comparison_all.csv")
    elif ARIMA_COMP.exists():
        with open(ARIMA_COMP, "rb") as f:
            st.download_button("Download ARIMA comparison CSV", f.read(), file_name="arima_comparison.csv")
    else:
        st.info("No comparison CSV available for download.")
with dl2:
    if SHAP_SUMMARY.exists():
        with open(SHAP_SUMMARY, "rb") as f:
            st.download_button("Download SHAP summary image", f.read(), file_name="shap_summary.png")
    if SHAP_BAR.exists():
        with open(SHAP_BAR, "rb") as f:
            st.download_button("Download SHAP bar image", f.read(), file_name="shap_bar.png")

# ---------------- Raw data toggle ----------------
with st.expander("Show Raw Data"):
    st.dataframe(state_df)

# ---------------- Notes / How to generate missing artifacts ----------------
st.markdown("### Notes & How to Generate Missing Artifacts")
st.write("""
- If images/CSVs are missing, run the scripts from the project root (or use the sidebar buttons):
  - `python src/explain_shap.py` → generates SHAP images (ensure it saves to reports/figs)
  - `python src/prophet_forecast.py` → should save `reports/figs/prophet_forecast_WA.png` and optionally forecasting metrics
  - `python src/var_analysis.py` → should save `reports/figs/var_forecast_WA.png` and optionally metrics
  - `python src/granger_test.py` → should save `reports/results/granger_results.csv` with columns ['variable','lag','p_value']
- To create a consolidated `reports/results/model_comparison_all.csv`, collect MAE/RMSE/MAPE from each script and save a CSV with columns: Model, MAE, RMSE, MAPE, Remarks.
- After generating artifacts, refresh the app (or re-open) to display them.
""")
