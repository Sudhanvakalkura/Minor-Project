# src/eda_basic.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the cleaned data
df = pd.read_csv('data/ev_registrations_by_state_year.csv')

# Make sure data is sorted
df = df.sort_values(['state', 'model_year'])

# 1️⃣ Total EV registrations by year (all states combined)
yearly = df.groupby('model_year')['registrations'].sum().reset_index()
print("\nTotal registrations by year:\n", yearly.head())

# Create output folder
os.makedirs('reports/figs', exist_ok=True)

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly, x='model_year', y='registrations', marker='o')
plt.title('Total Electric Vehicle Registrations by Model Year (All States)')
plt.xlabel('Model Year')
plt.ylabel('Registrations')
plt.grid(True)
plt.tight_layout()
plt.savefig('reports/figs/total_registrations_by_year.png')
print("✅ Saved reports/figs/total_registrations_by_year.png")

# 2️⃣ Top 10 states by total registrations
top_states = df.groupby('state')['registrations'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 states:\n", top_states)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
plt.title('Top 10 States by Total EV Registrations')
plt.xlabel('Registrations')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('reports/figs/top10_states.png')
print("✅ Saved reports/figs/top10_states.png")

# 3️⃣ Yearly trend for top 3 states
top3 = top_states.index[:3]
df_top3 = df[df['state'].isin(top3)]
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_top3, x='model_year', y='registrations', hue='state', marker='o')
plt.title('EV Registrations Over Time for Top 3 States')
plt.xlabel('Model Year')
plt.ylabel('Registrations')
plt.legend(title='State')
plt.tight_layout()
plt.savefig('reports/figs/top3_state_trends.png')
print("✅ Saved reports/figs/top3_state_trends.png")

plt.show()
