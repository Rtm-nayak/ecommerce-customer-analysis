# =============================================================================
# src/rfm_analysis.py
# RFM (Recency, Frequency, Monetary) Analysis
# Author: Your Name | MCA, Kolkata
# =============================================================================

import pandas as pd
import numpy as np


# ─── RFM Segment Labels ───────────────────────────────────────────────────────
SEGMENT_MAP = {
    r'5[45]'        : 'Champions',
    r'[34][45]'     : 'Loyal Customers',
    r'[45][1-3]'    : 'Potential Loyalists',
    r'51'           : 'Recent Customers',
    r'41'           : 'Promising',
    r'[34][12]'     : 'Customers Needing Attention',
    r'[12][45]'     : 'At Risk',
    r'[12][1-3]'    : 'Cant Lose Them',
    r'[12][12]'     : 'Hibernating',
    r'11'           : 'Lost',
}

SEGMENT_COLORS = {
    'Champions'                   : '#2ecc71',
    'Loyal Customers'             : '#27ae60',
    'Potential Loyalists'         : '#1abc9c',
    'Recent Customers'            : '#3498db',
    'Promising'                   : '#2980b9',
    'Customers Needing Attention' : '#f39c12',
    'At Risk'                     : '#e67e22',
    'Cant Lose Them'              : '#e74c3c',
    'Hibernating'                 : '#c0392b',
    'Lost'                        : '#7f8c8d',
}


def compute_rfm(customer_summary_df):
    """
    Compute RFM scores from the customer summary DataFrame.

    Parameters
    ----------
    customer_summary_df : pd.DataFrame
        Output of get_customer_summary() with columns:
        recency_days, total_orders, total_spend

    Returns
    -------
    pd.DataFrame with RFM scores and segment label added.
    """
    df = customer_summary_df.copy()

    # ── Recency Score: lower recency_days = better ────────────────────────────
    df['R_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1])
    df['R_score'] = df['R_score'].astype(int)

    # ── Frequency Score: higher total_orders = better ────────────────────────
    # Use rank to handle duplicates gracefully
    df['F_score'] = pd.qcut(
        df['total_orders'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]
    )
    df['F_score'] = df['F_score'].astype(int)

    # ── Monetary Score: higher total_spend = better ───────────────────────────
    df['M_score'] = pd.qcut(
        df['total_spend'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]
    )
    df['M_score'] = df['M_score'].astype(int)

    # ── RFM Combined Score ────────────────────────────────────────────────────
    df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str)
    df['RFM_total'] = df['R_score'] + df['F_score'] + df['M_score']

    # ── Assign Segment ────────────────────────────────────────────────────────
    df['segment'] = df['RFM_score'].apply(_assign_segment)

    print("\n📊 RFM Segment Distribution:")
    seg_counts = df['segment'].value_counts()
    for seg, count in seg_counts.items():
        pct = count / len(df) * 100
        print(f"   {seg:<30} {count:>6,}  ({pct:.1f}%)")

    return df


def _assign_segment(rfm_str):
    """Map an RFM score string to a human-readable segment label."""
    import re
    for pattern, label in SEGMENT_MAP.items():
        if re.match(pattern, rfm_str):
            return label
    return 'Others'


def rfm_segment_summary(rfm_df):
    """
    Produce a summary table of each RFM segment:
    count, avg recency, avg frequency, avg spend, % of revenue.
    """
    total_revenue = rfm_df['total_spend'].sum()

    summary = rfm_df.groupby('segment').agg(
        customer_count   = ('customer_unique_id', 'count'),
        avg_recency_days = ('recency_days',        'mean'),
        avg_orders       = ('total_orders',        'mean'),
        avg_spend        = ('total_spend',         'mean'),
        total_revenue    = ('total_spend',         'sum'),
    ).reset_index()

    summary['revenue_pct'] = (summary['total_revenue'] / total_revenue * 100).round(1)
    summary['avg_recency_days'] = summary['avg_recency_days'].round(0).astype(int)
    summary['avg_orders']       = summary['avg_orders'].round(2)
    summary['avg_spend']        = summary['avg_spend'].round(2)

    summary.sort_values('total_revenue', ascending=False, inplace=True)
    return summary


def tag_churn_risk(rfm_df, recency_threshold=180):
    """
    Tag customers as:
      - 'Churned'   : no order in > recency_threshold days
      - 'At Risk'   : 90 < recency_days <= recency_threshold
      - 'Active'    : recency_days <= 90

    Parameters
    ----------
    recency_threshold : int, default 180 (6 months)
    """
    df = rfm_df.copy()

    def _tag(row):
        if row['recency_days'] > recency_threshold:
            return 'Churned'
        elif row['recency_days'] > 90:
            return 'At Risk'
        else:
            return 'Active'

    df['churn_status'] = df.apply(_tag, axis=1)

    print("\n⚠️  Churn Analysis:")
    for status, count in df['churn_status'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {status:<12} {count:>6,}  ({pct:.1f}%)")

    return df


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)
    n = 1000
    test_df = pd.DataFrame({
        'customer_unique_id': [f'cust_{i}' for i in range(n)],
        'recency_days':       np.random.randint(1, 365, n),
        'total_orders':       np.random.randint(1, 20,  n),
        'total_spend':        np.random.uniform(50, 5000, n),
    })

    rfm     = compute_rfm(test_df)
    summary = rfm_segment_summary(rfm)
    rfm     = tag_churn_risk(rfm)

    print("\n--- Segment Summary ---")
    print(summary.to_string(index=False))
