# %% [markdown]
# # 📊 Notebook 2: RFM Analysis
# **E-Commerce Customer Behavior Analysis**
# Author: Your Name | MCA Student, Kolkata
#
# **RFM stands for:**
# - **R**ecency   — How recently did the customer purchase?
# - **F**requency — How often do they purchase?
# - **M**onetary  — How much do they spend?
#
# RFM is one of the most powerful and widely used customer segmentation techniques in marketing analytics.

# %% [markdown]
# ## 1. Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_raw_data, merge_master_df, get_customer_summary
from src.rfm_analysis        import compute_rfm, rfm_segment_summary, tag_churn_risk
from src.visualization       import plot_rfm_segments_plotly, plot_rfm_3d_scatter

sns.set_theme(style='whitegrid')
print("✅ Setup complete")

# %% [markdown]
# ## 2. Load & Prepare Data

# %%
dfs    = load_raw_data()
master = merge_master_df(dfs)
cust   = get_customer_summary(master)
print(f"\nCustomer summary shape: {cust.shape}")
cust.head(3)

# %% [markdown]
# ## 3. RFM Distributions (Raw)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Recency
axes[0].hist(cust['recency_days'], bins=50, color='#e74c3c', edgecolor='white', alpha=0.85)
axes[0].set_title("Recency Distribution", fontweight='bold')
axes[0].set_xlabel("Days since last order")
axes[0].axvline(cust['recency_days'].median(), color='black', linestyle='--',
                label=f"Median: {cust['recency_days'].median():.0f}")
axes[0].legend()

# Frequency
axes[1].hist(cust['total_orders'], bins=30, color='#3498db', edgecolor='white', alpha=0.85)
axes[1].set_title("Frequency Distribution", fontweight='bold')
axes[1].set_xlabel("Number of Orders")

# Monetary
axes[2].hist(cust['total_spend'].clip(upper=2000), bins=50, color='#2ecc71', edgecolor='white', alpha=0.85)
axes[2].set_title("Monetary Distribution (clipped at 2000)", fontweight='bold')
axes[2].set_xlabel("Total Spend (BRL)")

plt.suptitle("Raw RFM Distributions", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Compute RFM Scores

# %%
rfm = compute_rfm(cust)
rfm[['customer_unique_id', 'recency_days', 'total_orders', 'total_spend',
     'R_score', 'F_score', 'M_score', 'RFM_score', 'segment']].head(10)

# %% [markdown]
# ## 5. Segment Distribution

# %%
seg_summary = rfm_segment_summary(rfm)
print("─── RFM Segment Summary ───")
print(seg_summary.to_string(index=False))

# %%
fig = plot_rfm_segments_plotly(rfm)
fig.show()

# %% [markdown]
# ## 6. Revenue Contribution per Segment

# %%
fig = px.bar(
    seg_summary.sort_values('revenue_pct', ascending=False),
    x='segment', y='revenue_pct',
    color='revenue_pct', color_continuous_scale='Greens',
    text='revenue_pct',
    title='Revenue % by Customer Segment',
    labels={'revenue_pct': '% of Total Revenue', 'segment': 'Segment'}
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(yaxis_title='% Revenue', showlegend=False, height=450,
                  xaxis_tickangle=-25)
fig.show()

# %% [markdown]
# ## 7. RFM Heatmap — Avg Monetary by R & F scores

# %%
rfm_pivot = rfm.pivot_table(
    values='M_score', index='R_score', columns='F_score', aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(rfm_pivot, annot=True, fmt='.2f', cmap='YlGn',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Avg Monetary Score'})
ax.set_title("Avg Monetary Score by Recency × Frequency", fontweight='bold')
ax.set_xlabel("Frequency Score")
ax.set_ylabel("Recency Score")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. 3D RFM Scatter Plot

# %%
fig = plot_rfm_3d_scatter(rfm, sample_n=3000)
fig.show()

# %% [markdown]
# ## 9. Segment Profiles — Box Plots

# %%
top_segments = ['Champions', 'Loyal Customers', 'At Risk', 'Hibernating', 'Lost']
filtered = rfm[rfm['segment'].isin(top_segments)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ('recency_days', 'Recency (Days)',    '#e74c3c'),
    ('total_orders', 'Frequency (Orders)', '#3498db'),
    ('total_spend',  'Monetary (BRL)',     '#2ecc71'),
]

for ax, (col, label, color) in zip(axes, metrics):
    filtered.boxplot(column=col, by='segment', ax=ax, patch_artist=True)
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

plt.suptitle('RFM Metrics by Segment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Marketing Recommendations by Segment
#
# | Segment | Strategy |
# |---|---|
# | **Champions** | Reward them. Early access, loyalty programs |
# | **Loyal Customers** | Upsell higher-value products, ask for reviews |
# | **Potential Loyalists** | Offer membership / loyalty program |
# | **At Risk** | Send win-back emails, special discounts |
# | **Cant Lose Them** | Reactivation campaign, survey why they left |
# | **Hibernating** | Remind them of value, show new products |
# | **Lost** | Last-resort discount or remove from active lists |
