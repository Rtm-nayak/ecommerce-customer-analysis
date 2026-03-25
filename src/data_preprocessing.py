# =============================================================================
# src/data_preprocessing.py
# E-Commerce Customer Behavior Analysis
# Author: Your Name | MCA, Kolkata
# =============================================================================

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_raw_data():
    """
    Load all Olist CSV files from the data/ directory.
    Returns a dictionary of DataFrames.
    """
    files = {
        'orders':    'olist_orders_dataset.csv',
        'items':     'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'products':  'olist_products_dataset.csv',
        'reviews':   'olist_order_reviews_dataset.csv',
        'payments':  'olist_order_payments_dataset.csv',
        'sellers':   'olist_sellers_dataset.csv',
        'category':  'product_category_name_translation.csv',
    }

    dfs = {}
    for key, filename in files.items():
        path = os.path.join(DATA_PATH, filename)
        try:
            dfs[key] = pd.read_csv(path)
            print(f"✅ Loaded {key:12s} → {dfs[key].shape[0]:>7,} rows, {dfs[key].shape[1]} cols")
        except FileNotFoundError:
            print(f"⚠️  File not found: {filename} — skipping.")
    return dfs


def clean_orders(orders_df):
    """
    Clean the orders DataFrame:
    - Parse datetime columns
    - Drop cancelled/unavailable orders
    - Remove rows with missing delivery dates
    """
    df = orders_df.copy()

    # Parse timestamps
    datetime_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Keep only delivered orders for behaviour analysis
    df = df[df['order_status'] == 'delivered'].copy()

    # Drop rows where purchase timestamp is missing
    df.dropna(subset=['order_purchase_timestamp'], inplace=True)

    # Add useful derived columns
    df['order_year']  = df['order_purchase_timestamp'].dt.year
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['order_dow']   = df['order_purchase_timestamp'].dt.day_name()
    df['order_hour']  = df['order_purchase_timestamp'].dt.hour

    # Delivery time in days
    df['delivery_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    print(f"\n🧹 Orders after cleaning: {len(df):,}")
    return df


def merge_master_df(dfs):
    """
    Merge all datasets into one master DataFrame for analysis.
    """
    orders    = clean_orders(dfs['orders'])
    items     = dfs['items']
    customers = dfs['customers']
    payments  = dfs['payments']
    reviews   = dfs['reviews']
    products  = dfs['products']
    category  = dfs['category']

    # Aggregate items per order (total price, freight, item count)
    items_agg = items.groupby('order_id').agg(
        total_price   = ('price', 'sum'),
        total_freight = ('freight_value', 'sum'),
        item_count    = ('order_item_id', 'count')
    ).reset_index()

    # Aggregate payments per order
    payments_agg = payments.groupby('order_id').agg(
        payment_value = ('payment_value', 'sum'),
        payment_type  = ('payment_type', lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
        installments  = ('payment_installments', 'mean')
    ).reset_index()

    # Take the latest review per order
    reviews_agg = reviews.sort_values('review_creation_date', ascending=False) \
                         .drop_duplicates('order_id')[['order_id', 'review_score']]

    # Translate product categories
    products = products.merge(category, on='product_category_name', how='left')

    # Products per item (get category for each order)
    items_with_cat = items.merge(
        products[['product_id', 'product_category_name_english']],
        on='product_id', how='left'
    )
    cat_agg = items_with_cat.groupby('order_id')['product_category_name_english'] \
                             .agg(lambda x: x.mode()[0] if len(x) > 0 else 'unknown') \
                             .reset_index() \
                             .rename(columns={'product_category_name_english': 'main_category'})

    # Build master table
    master = orders.merge(customers,     on='customer_id',  how='left')
    master = master.merge(items_agg,     on='order_id',     how='left')
    master = master.merge(payments_agg,  on='order_id',     how='left')
    master = master.merge(reviews_agg,   on='order_id',     how='left')
    master = master.merge(cat_agg,       on='order_id',     how='left')

    print(f"\n📦 Master DataFrame shape: {master.shape}")
    print(f"   Columns: {list(master.columns)}")
    return master


def get_customer_summary(master_df):
    """
    Aggregate the master DataFrame to one row per customer.
    Useful for segmentation and churn analysis.
    """
    snapshot_date = master_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    summary = master_df.groupby('customer_unique_id').agg(
        total_orders      = ('order_id',                  'nunique'),
        total_spend       = ('payment_value',             'sum'),
        avg_order_value   = ('payment_value',             'mean'),
        avg_review_score  = ('review_score',              'mean'),
        first_order_date  = ('order_purchase_timestamp',  'min'),
        last_order_date   = ('order_purchase_timestamp',  'max'),
        state             = ('customer_state',            'first'),
        city              = ('customer_city',             'first'),
        fav_category      = ('main_category',             lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
        fav_payment       = ('payment_type',              lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
    ).reset_index()

    summary['recency_days'] = (snapshot_date - summary['last_order_date']).dt.days
    summary['customer_lifetime_days'] = (
        summary['last_order_date'] - summary['first_order_date']
    ).dt.days

    print(f"\n👥 Unique customers: {len(summary):,}")
    return summary


if __name__ == '__main__':
    dfs    = load_raw_data()
    master = merge_master_df(dfs)
    cust   = get_customer_summary(master)
    print("\n--- Customer Summary Sample ---")
    print(cust.head(3).to_string())
