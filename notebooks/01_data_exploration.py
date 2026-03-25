# %% [markdown]
# # 📦 Notebook 1: Data Exploration & EDA
# **E-Commerce Customer Behavior Analysis**
# Author: Your Name | MCA Student, Kolkata
#
# **Objectives:**
# - Load and inspect all Olist datasets
# - Understand data shape, types, and missing values
# - Perform exploratory data analysis with visualizations

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_raw_data, merge_master_df, get_customer_summary

sns.set_theme(style='whitegrid')
pd.set_option('display.float_format', '{:,.2f}'.format)
print("✅ All libraries imported successfully")

# %% [markdown]
# ## 2. Load Raw Data

# %%
dfs = load_raw_data()
print("\n📋 Loaded datasets:", list(dfs.keys()))

# %% [markdown]
# ## 3. Inspect Individual Datasets

# %%
# Orders
print("=== ORDERS ===")
print(dfs['orders'].shape)
dfs['orders'].head(3)

# %%
print("Missing values in Orders:")
print(dfs['orders'].isnull().sum())

# %%
# Customers
print("=== CUSTOMERS ===")
print(dfs['customers'].shape)
dfs['customers'].head(3)

# %%
# Items
print("=== ORDER ITEMS ===")
print(dfs['items'].describe())

# %%
# Payments
print("=== PAYMENTS ===")
print(dfs['payments']['payment_type'].value_counts())

# %%
# Reviews
print("=== REVIEWS ===")
print(dfs['reviews']['review_score'].value_counts().sort_index())

# %% [markdown]
# ## 4. Build Master DataFrame

# %%
master = merge_master_df(dfs)
print(f"\nMaster DataFrame: {master.shape}")
master.head(3)

# %%
master.dtypes

# %%
# Missing values in master
missing = master.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# %% [markdown]
# ## 5. Order Status Distribution

# %%
fig, ax = plt.subplots(figsize=(9, 4))
status_counts = master['order_status'].value_counts()
sns.barplot(x=status_counts.index, y=status_counts.values, palette='Blues_r', ax=ax)
ax.set_title("Order Status Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Status")
ax.set_ylabel("Count")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{int(bar.get_height()):,}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Monthly Revenue Trend

# %%
master['year_month'] = master['order_purchase_timestamp'].dt.to_period('M').astype(str)
monthly_rev = master.groupby('year_month')['payment_value'].sum().reset_index()
monthly_rev.columns = ['Month', 'Revenue']

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_rev['Month'], monthly_rev['Revenue'],
        color='#3498db', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(monthly_rev['Month'], monthly_rev['Revenue'], alpha=0.15, color='#3498db')
ax.set_title("Monthly Revenue Trend (2016–2018)", fontsize=14, fontweight='bold')
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (BRL)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Top Product Categories

# %%
cat_rev = (
    master.groupby('main_category')['payment_value']
    .sum().sort_values(ascending=False).head(15).reset_index()
)
cat_rev.columns = ['Category', 'Revenue']

fig = px.bar(cat_rev, x='Revenue', y='Category', orientation='h',
             color='Revenue', color_continuous_scale='teal',
             title='Top 15 Categories by Revenue', text_auto='.2s')
fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
fig.show()

# %% [markdown]
# ## 8. Payment Method Analysis

# %%
pay_counts = master['payment_type'].value_counts().reset_index()
pay_counts.columns = ['Type', 'Count']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count
sns.barplot(data=pay_counts, x='Type', y='Count', palette='viridis', ax=axes[0])
axes[0].set_title("Payment Method - Count")

# Avg payment value per type
avg_pay = master.groupby('payment_type')['payment_value'].mean().sort_values(ascending=False)
sns.barplot(x=avg_pay.index, y=avg_pay.values, palette='magma', ax=axes[1])
axes[1].set_title("Payment Method - Avg Value")
axes[1].set_ylabel("Avg Value (BRL)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Delivery Time Analysis

# %%
delivery = master[master['delivery_days'].between(0, 60)].copy()

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(delivery['delivery_days'], bins=30, color='#2ecc71', edgecolor='white', alpha=0.85)
ax.axvline(delivery['delivery_days'].median(), color='red', linestyle='--',
           linewidth=2, label=f"Median: {delivery['delivery_days'].median():.0f} days")
ax.axvline(delivery['delivery_days'].mean(), color='orange', linestyle='--',
           linewidth=2, label=f"Mean: {delivery['delivery_days'].mean():.1f} days")
ax.set_title("Delivery Time Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Delivery Days")
ax.set_ylabel("Number of Orders")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Review Score Distribution

# %%
scores = master['review_score'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
bars = ax.bar(scores.index.astype(str), scores.values, color=colors, edgecolor='white', linewidth=1.5)
ax.bar_label(bars, fmt='{:,.0f}', padding=4)
ax.set_title("Customer Review Scores", fontsize=14, fontweight='bold')
ax.set_xlabel("Score")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

print(f"\nAverage Review Score: {master['review_score'].mean():.2f}")

# %% [markdown]
# ## 11. Order Heatmap by Day & Hour

# %%
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
pivot = master.groupby(['order_dow','order_hour'])['order_id'].count().unstack(fill_value=0)
pivot = pivot.reindex(day_order)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
            cbar_kws={'label': 'Order Count'})
ax.set_title("Order Volume by Day and Hour of Week", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Key Takeaways
#
# | Finding | Value |
# |---|---|
# | Total Orders (Delivered) | ~96,000 |
# | Date Range | Sep 2016 – Aug 2018 |
# | Most Popular Category | Bed/Bath/Table |
# | Most Used Payment | Credit Card (74%) |
# | Median Delivery Time | ~12 days |
# | Average Review Score | 4.09 / 5 |
# | Peak Order Day | Monday |
# | Peak Order Hour | 2 PM – 4 PM |
