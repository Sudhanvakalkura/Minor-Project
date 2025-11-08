# src/generate_report.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

# Create output folder
os.makedirs('reports', exist_ok=True)

# PDF setup
doc = SimpleDocTemplate("reports/final_ev_report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("<b>Electric Vehicle Growth Analysis and Forecast Report</b>", styles['Title']))
story.append(Spacer(1, 20))

# Introduction
story.append(Paragraph(
    "This report presents the results of data analysis and forecasting on electric vehicle registrations "
    "across U.S. states. The project includes data cleaning, exploratory data visualization, XGBoost-based "
    "forecasting, and model interpretation through feature importance.", styles['BodyText']))
story.append(Spacer(1, 12))

# Key model performance metrics (from earlier training run)
story.append(Paragraph("<b>Model Performance</b>", styles['Heading2']))
story.append(Paragraph("• Mean Absolute Error (MAE): 877.00", styles['BodyText']))
story.append(Paragraph("• Root Mean Squared Error (RMSE): 1090.05", styles['BodyText']))
story.append(Paragraph("• Mean Absolute Percentage Error (MAPE): 349.14%", styles['BodyText']))
story.append(Spacer(1, 12))

# EDA section
story.append(Paragraph("<b>Exploratory Data Analysis</b>", styles['Heading2']))
story.append(Paragraph("The figures below show trends in EV registrations by year and state.", styles['BodyText']))
story.append(Spacer(1, 8))
for fig in [
    "reports/figs/total_registrations_by_year.png",
    "reports/figs/top10_states.png",
    "reports/figs/top3_state_trends.png"
]:
    if os.path.exists(fig):
        story.append(Image(fig, width=400, height=250))
        story.append(Spacer(1, 12))

# Forecast
story.append(Paragraph("<b>Forecast Results</b>", styles['Heading2']))
story.append(Paragraph("The following figure shows the predicted next-year EV registrations for Washington State.", styles['BodyText']))
story.append(Spacer(1, 8))
if os.path.exists("reports/figs/forecast_WA.png"):
    story.append(Image("reports/figs/forecast_WA.png", width=400, height=250))
story.append(Spacer(1, 12))

# Feature importance
story.append(Paragraph("<b>Model Explainability</b>", styles['Heading2']))
story.append(Paragraph("Feature importance analysis identifies which lag features have the most influence on forecasts.", styles['BodyText']))
story.append(Spacer(1, 8))
if os.path.exists("reports/figs/feature_importance_xgb.png"):
    story.append(Image("reports/figs/feature_importance_xgb.png", width=400, height=250))
story.append(Spacer(1, 12))

# Conclusion
story.append(Paragraph("<b>Conclusion</b>", styles['Heading2']))
story.append(Paragraph(
    "The model demonstrates that recent-year trends (lag_1) are the most influential predictor of EV growth. "
    "Forecasting indicates a continued upward trajectory in electric vehicle registrations, reflecting strong "
    "consumer adoption momentum and expanding infrastructure support across states.", styles['BodyText']))

# Build the PDF
doc.build(story)
print("✅ Report generated: reports/final_ev_report.pdf")
