# =============================================================================
# dashboard/app.py
# Interactive Streamlit Dashboard
# Author: Your Name | MCA, Kolkata
# Run:  streamlit run dashboard/app.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import load_raw_data, merge_master_df, get_customer_summary
from src.rfm_analysis        import compute_rfm, rfm_segment_summary, tag_churn_risk
from src.visualization       import (
    plot_rfm_segments_plotly, plot_rfm_3d_scatter,
    plot_churn_status, plot_state_revenue
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_all():
    dfs    = load_raw_data()
    master = merge_master_df(dfs)
    cust   = get_customer_summary(master)
    rfm    = compute_rfm(cust)
    rfm    = tag_churn_risk(rfm)
    return master, cust, rfm

try:
    master, cust, rfm = load_all()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Could not load data: {e}")
    st.info("Make sure all Olist CSV files are placed in the `data/` folder.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
    st.title("🛒 E-Commerce\nAnalytics")
    st.markdown("---")

    if data_loaded:
        page = st.radio("📌 Navigate", [
            "🏠 Overview",
            "📈 Sales Trends",
            "👥 Customer Segments",
            "⚠️ Churn Analysis",
            "🗺️ Geo Analysis",
        ])

        st.markdown("---")
        st.markdown("**Filters**")
        years = sorted(master['order_year'].dropna().unique().astype(int).tolist())
        sel_years = st.multiselect("Year", years, default=years)
    else:
        page = "🏠 Overview"

# ─────────────────────────────────────────────────────────────────────────────
# Filter data
# ─────────────────────────────────────────────────────────────────────────────
if data_loaded:
    mdf = master[master['order_year'].isin(sel_years)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview" and data_loaded:
    st.title("🛒 E-Commerce Customer Behavior Dashboard")
    st.caption("Olist Brazilian E-Commerce Dataset  |  End-to-End Analytics Project")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Orders",     f"{mdf['order_id'].nunique():,}")
    col2.metric("Unique Customers", f"{mdf['customer_unique_id'].nunique():,}")
    col3.metric("Total Revenue",    f"R${mdf['payment_value'].sum():,.0f}")
    col4.metric("Avg Order Value",  f"R${mdf['payment_value'].mean():.2f}")
    col5.metric("Avg Review Score", f"{mdf['review_score'].mean():.2f} ⭐")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📦 Orders by Status")
        status_counts = master['order_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.bar(status_counts, x='Status', y='Count',
                     color='Count', color_continuous_scale='Blues',
                     text_auto=True)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💳 Payment Methods")
        pay = mdf['payment_type'].value_counts().reset_index()
        pay.columns = ['Type', 'Count']
        fig = px.pie(pay, names='Type', values='Count', hole=0.45)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Monthly Revenue Trend")
    mdf['year_month'] = mdf['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly = mdf.groupby('year_month')['payment_value'].sum().reset_index()
    monthly.columns = ['Month', 'Revenue']
    fig = px.line(monthly, x='Month', y='Revenue', markers=True,
                  color_discrete_sequence=['#3498db'])
    fig.update_layout(yaxis_tickprefix='R$', height=350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Sales Trends
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Sales Trends" and data_loaded:
    st.title("📈 Sales Trends Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top Categories by Revenue")
        cat_rev = (
            mdf.groupby('main_category')['payment_value']
            .sum().sort_values(ascending=False).head(12).reset_index()
        )
        cat_rev.columns = ['Category', 'Revenue']
        fig = px.bar(cat_rev, x='Revenue', y='Category', orientation='h',
                     color='Revenue', color_continuous_scale='Teal',
                     text_auto='.2s')
        fig.update_layout(showlegend=False, height=450,
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⭐ Review Score Distribution")
        scores = mdf['review_score'].value_counts().sort_index().reset_index()
        scores.columns = ['Score', 'Count']
        fig = px.bar(scores, x='Score', y='Count',
                     color='Score', color_continuous_scale='RdYlGn',
                     text_auto=True)
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕐 Order Volume Heatmap (Day × Hour)")
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = mdf.groupby(['order_dow','order_hour'])['order_id'].count().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)

    fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                    labels={'x': 'Hour of Day', 'y': 'Day', 'color': 'Orders'})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Customer Segments
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Customer Segments" and data_loaded:
    st.title("👥 RFM Customer Segmentation")
    st.markdown("---")

    seg_summary = rfm_segment_summary(rfm)

    st.subheader("📊 Segment Summary Table")
    st.dataframe(
        seg_summary.style.background_gradient(subset=['revenue_pct'], cmap='Greens'),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗺️ Segment Treemap")
        fig = plot_rfm_segments_plotly(rfm)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Revenue by Segment")
        rev_seg = rfm.groupby('segment')['total_spend'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(rev_seg, x='segment', y='total_spend',
                     color='total_spend', color_continuous_scale='Greens',
                     text_auto='.2s')
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔵 3D RFM Score View")
    fig = plot_rfm_3d_scatter(rfm)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Churn Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ Churn Analysis" and data_loaded:
    st.title("⚠️ Customer Churn Analysis")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    active   = (rfm['churn_status'] == 'Active').sum()
    at_risk  = (rfm['churn_status'] == 'At Risk').sum()
    churned  = (rfm['churn_status'] == 'Churned').sum()
    col1.metric("✅ Active Customers",   f"{active:,}",  f"{active/len(rfm)*100:.1f}%")
    col2.metric("⚠️ At Risk Customers",  f"{at_risk:,}", f"{at_risk/len(rfm)*100:.1f}%")
    col3.metric("❌ Churned Customers",  f"{churned:,}", f"{churned/len(rfm)*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn Status Overview")
        fig = plot_churn_status(rfm)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Avg Spend by Churn Status")
        churn_spend = rfm.groupby('churn_status')['total_spend'].mean().reset_index()
        fig = px.bar(churn_spend, x='churn_status', y='total_spend',
                     color='churn_status',
                     color_discrete_map={'Active':'#2ecc71','At Risk':'#f39c12','Churned':'#e74c3c'},
                     text_auto='.2f')
        fig.update_layout(showlegend=False, yaxis_tickprefix='R$', height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 At-Risk Customers Sample")
    at_risk_df = rfm[rfm['churn_status'] == 'At Risk'][[
        'customer_unique_id', 'recency_days', 'total_orders',
        'total_spend', 'avg_review_score', 'segment'
    ]].sort_values('total_spend', ascending=False).head(20)
    st.dataframe(at_risk_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Geo Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Geo Analysis" and data_loaded:
    st.title("🗺️ Geographic Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 States by Revenue")
        fig = plot_state_revenue(mdf)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Cities by Orders")
        city_orders = (
            mdf.groupby('customer_city')['order_id']
            .count().sort_values(ascending=False).head(10).reset_index()
        )
        city_orders.columns = ['City', 'Orders']
        fig = px.bar(city_orders, x='Orders', y='City', orientation='h',
                     color='Orders', color_continuous_scale='Purples',
                     text_auto=True)
        fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Avg Review Score by State")
    state_score = mdf.groupby('customer_state')['review_score'].mean().reset_index()
    state_score.columns = ['State', 'Avg Score']
    fig = px.bar(state_score.sort_values('Avg Score', ascending=False),
                 x='State', y='Avg Score',
                 color='Avg Score', color_continuous_scale='RdYlGn',
                 text_auto='.2f')
    fig.update_layout(height=400, yaxis_range=[3.5, 5.0])
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85rem'>"
    "Built with ❤️ using Python & Streamlit | MCA Project — Kolkata"
    "</div>",
    unsafe_allow_html=True
)
