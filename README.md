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

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit Dashboard
streamlit run src/app.py


Then open your browser at http://localhost:8501 to explore the dashboard.

🌐 Deployment

This project is ready for deployment on Streamlit Cloud:

Push your repository to GitHub.

Go to Streamlit Cloud
.

Select your repo and src/app.py as the entry file.

Deploy and share your live dashboard link.

📄 Results
Model	MAE	RMSE	MAPE (%)	Remarks
ARIMA	1683.6	1683.6	35.9	Univariate baseline
ARIMAX	0.55	0.55	0.24	Uses exogenous input
Prophet	1053.2	1198.3	32.1	Captures seasonality
VAR	890.2	1045.6	29.8	Multivariate predictors
XGBoost	877.0	1090.0	349.1	Nonlinear regression
📘 References

U.S. Department of Energy, Alternative Fuels Data Center.

T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” Proc. ACM SIGKDD, 2016.

R.J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 2021.

S.M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.

Statsmodels Developers, Statistical Modeling and Econometrics in Python, 2024.

Project By 
S SUDHANVA KALKURA
