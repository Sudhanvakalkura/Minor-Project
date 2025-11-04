# 🚗 Electric Vehicle (EV) Growth Forecasting and Analysis

This project analyzes and forecasts Electric Vehicle (EV) registration growth trends across U.S. states using a combination of **machine learning** and **time-series forecasting** techniques.  
It also includes an interactive **Streamlit Dashboard** to visualize EV trends, model forecasts, and causality analysis results.

---

## 📊 **Project Overview**

The project explores the evolution of EV registrations across multiple U.S. states, leveraging open datasets from the **U.S. Department of Energy’s Alternative Fuels Data Center**.

We implemented and compared multiple forecasting approaches:

- **XGBoost Regression** — Feature-based machine learning model  
- **ARIMA** — Univariate time-series forecasting  
- **ARIMAX** — Time-series model with exogenous variables  
- **Prophet** — Additive model capturing trends and seasonality  
- **VAR (Vector AutoRegression)** — Multivariate forecasting including external factors (e.g., fuel prices, charging stations)  
- **Granger Causality Test** — To identify variables that *cause* changes in EV registrations  

---

## ⚙️ **Tech Stack**

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3.11 |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, xgboost |
| Time Series | statsmodels, pmdarima, prophet |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Reporting | ReportLab |
| Deployment | Streamlit Cloud / GitHub |

## 🧠 **Key Features**

- 📈 **Trend Visualization:** EV registration trends across multiple years and states  
- 🔮 **Forecasting Models:** Compare ARIMA, ARIMAX, Prophet, VAR, and XGBoost  
- 🧮 **Causality Analysis:** Identify driving factors via Granger causality tests  
- 💡 **Feature Interpretability:** SHAP-based feature importance for explainable ML  
- 📊 **Interactive Dashboard:** Built with Streamlit for real-time exploration  

---

## 🚀 **How to Run**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/ev-growth-project.git
cd ev-growth-project

