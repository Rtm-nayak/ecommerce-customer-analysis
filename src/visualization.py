# =============================================================================
# src/visualization.py
# Reusable Plotting Functions
# Author: Your Name | MCA, Kolkata
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Global Style ──────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
           '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#f9f9f9',
    'font.family'      : 'DejaVu Sans',
    'axes.titlesize'   : 14,
    'axes.labelsize'   : 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Monthly Revenue Trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_revenue(master_df):
    """Line chart of total monthly revenue."""
    df = master_df.copy()
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

    monthly = df.groupby('year_month')['payment_value'].sum().reset_index()
    monthly.columns = ['Month', 'Revenue']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly['Month'], monthly['Revenue'], color=PALETTE[1],
            linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(monthly['Month'], monthly['Revenue'],
                    alpha=0.15, color=PALETTE[1])
    ax.set_title('Monthly Revenue Trend', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (BRL)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top Product Categories
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_categories(master_df, top_n=15):
    """Horizontal bar chart of top N product categories by revenue."""
    cat_rev = (
        master_df.groupby('main_category')['payment_value']
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cat_rev))]
    cat_rev.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title(f'Top {top_n} Categories by Revenue', fontweight='bold')
    ax.set_xlabel('Total Revenue (BRL)')
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Payment Type Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_payment_distribution(master_df):
    """Donut chart for payment method breakdown."""
    pay = master_df['payment_type'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        pay.values,
        labels=pay.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=PALETTE[:len(pay)],
        wedgeprops={'width': 0.6, 'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}
    )
    ax.set_title('Payment Method Distribution', fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. RFM Segment Distribution (Plotly — interactive)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rfm_segments_plotly(rfm_df):
    """Interactive treemap of RFM customer segments."""
    from src.rfm_analysis import SEGMENT_COLORS

    seg = rfm_df.groupby('segment').agg(
        count   = ('customer_unique_id', 'count'),
        revenue = ('total_spend',        'sum')
    ).reset_index()

    seg['color'] = seg['segment'].map(SEGMENT_COLORS).fillna('#95a5a6')
    seg['label'] = seg.apply(
        lambda r: f"{r['segment']}<br>{r['count']:,} customers<br>R${r['revenue']:,.0f}", axis=1
    )

    fig = px.treemap(
        seg,
        path=['segment'],
        values='count',
        color='revenue',
        color_continuous_scale='Greens',
        title='RFM Customer Segments (by count, colored by revenue)',
        hover_data={'revenue': ':,.0f', 'count': ':,'},
    )
    fig.update_traces(textinfo='label+percent entry')
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. RFM Score 3D Scatter (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rfm_3d_scatter(rfm_df, sample_n=3000):
    """3D scatter of R, F, M scores colored by segment."""
    df = rfm_df.sample(min(sample_n, len(rfm_df)), random_state=42)

    fig = px.scatter_3d(
        df,
        x='R_score', y='F_score', z='M_score',
        color='segment',
        opacity=0.7,
        title='RFM Scores 3D View',
        labels={'R_score': 'Recency', 'F_score': 'Frequency', 'M_score': 'Monetary'},
        hover_data=['recency_days', 'total_orders', 'total_spend']
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), height=600)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Churn Status Pie
# ─────────────────────────────────────────────────────────────────────────────
def plot_churn_status(rfm_df):
    """Donut chart of Active / At Risk / Churned customers."""
    status = rfm_df['churn_status'].value_counts().reset_index()
    status.columns = ['Status', 'Count']

    color_map = {'Active': '#2ecc71', 'At Risk': '#f39c12', 'Churned': '#e74c3c'}
    fig = px.pie(
        status, names='Status', values='Count',
        hole=0.5,
        color='Status',
        color_discrete_map=color_map,
        title='Customer Churn Status'
    )
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Review Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_review_scores(master_df):
    """Bar chart of review score (1–5) counts."""
    scores = master_df['review_score'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(scores.index.astype(str), scores.values,
                  color=PALETTE[:5], edgecolor='white', linewidth=1.5)
    ax.bar_label(bars, fmt='{:,.0f}', padding=4, fontsize=10)
    ax.set_title('Customer Review Score Distribution', fontweight='bold')
    ax.set_xlabel('Review Score')
    ax.set_ylabel('Number of Reviews')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Orders by Day of Week & Hour Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_order_heatmap(master_df):
    """Heatmap of order count by day-of-week × hour."""
    df = master_df.copy()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    pivot = df.groupby(['order_dow', 'order_hour'])['order_id'].count().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
                cbar_kws={'label': 'Order Count'})
    ax.set_title('Order Volume by Day and Hour', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. State-wise Revenue Map (Plotly Choropleth)
# ─────────────────────────────────────────────────────────────────────────────
def plot_state_revenue(master_df):
    """Bar chart of top 10 states by revenue (choropleth needs geojson)."""
    state_rev = (
        master_df.groupby('customer_state')['payment_value']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    state_rev.columns = ['State', 'Revenue']

    fig = px.bar(
        state_rev, x='State', y='Revenue',
        color='Revenue', color_continuous_scale='Blues',
        title='Top 10 States by Revenue',
        text_auto='.2s'
    )
    fig.update_layout(yaxis_tickprefix='R$', showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. Delivery Time Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_delivery_time(master_df):
    """Histogram of delivery time in days."""
    df = master_df[master_df['delivery_days'].between(0, 60)].copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['delivery_days'], bins=30, color=PALETTE[0], edgecolor='white',
            linewidth=0.8, alpha=0.85)
    ax.axvline(df['delivery_days'].median(), color='red', linestyle='--',
               linewidth=2, label=f"Median: {df['delivery_days'].median():.0f} days")
    ax.set_title('Delivery Time Distribution', fontweight='bold')
    ax.set_xlabel('Delivery Days')
    ax.set_ylabel('Number of Orders')
    ax.legend()
    plt.tight_layout()
    return fig
