import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

NP_SEED = 42
np.random.seed(NP_SEED)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_PATH, exist_ok=True)

NUM_ORDERS = 5000
NUM_CUSTOMERS = 4000
NUM_PRODUCTS = 200

# 1. Customers
cust_ids = [f'C_{i}' for i in range(NUM_ORDERS)]
cust_unique_ids = [f'CU_{i % NUM_CUSTOMERS}' for i in range(NUM_ORDERS)]
states = np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], NUM_ORDERS, p=[0.5, 0.2, 0.1, 0.1, 0.1])
cities = np.random.choice(['Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba'], NUM_ORDERS)

customers = pd.DataFrame({
    'customer_id': cust_ids,
    'customer_unique_id': cust_unique_ids,
    'customer_state': states,
    'customer_city': cities
})

# 2. Orders
order_ids = [f'O_{i}' for i in range(NUM_ORDERS)]
start_date = datetime(2017, 1, 1)

timestamps = [start_date + timedelta(days=int(np.random.randint(0, 600)), hours=int(np.random.randint(0, 24))) for _ in range(NUM_ORDERS)]
delivered_dates = [t + timedelta(days=int(np.random.randint(1, 15))) for t in timestamps]

orders = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': cust_ids,
    'order_status': ['delivered'] * NUM_ORDERS,
    'order_purchase_timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
    'order_delivered_customer_date': [t.strftime('%Y-%m-%d %H:%M:%S') for t in delivered_dates],
    'order_approved_at': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
    'order_delivered_carrier_date': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
    'order_estimated_delivery_date': [t.strftime('%Y-%m-%d %H:%M:%S') for t in delivered_dates]
})

# 3. Order Items
prod_ids = [f'P_{i}' for i in range(NUM_PRODUCTS)]
items_list = []
for order in order_ids:
    num_items = np.random.randint(1, 4)
    for i in range(num_items):
        items_list.append({
            'order_id': order,
            'order_item_id': i + 1,
            'product_id': np.random.choice(prod_ids),
            'price': np.random.uniform(10.0, 500.0),
            'freight_value': np.random.uniform(2.0, 50.0)
        })
items = pd.DataFrame(items_list)

# 4. Products
categories = ['beleza_saude', 'esporte_lazer', 'moveis_decoracao', 'informatica_acessorios']
products = pd.DataFrame({
    'product_id': prod_ids,
    'product_category_name': np.random.choice(categories, NUM_PRODUCTS)
})

# 5. Category Translation
category = pd.DataFrame({
    'product_category_name': categories,
    'product_category_name_english': ['health_beauty', 'sports_leisure', 'furniture_decor', 'computers_accessories']
})

# 6. Payments
payments = pd.DataFrame({
    'order_id': order_ids,
    'payment_type': np.random.choice(['credit_card', 'boleto', 'voucher', 'debit_card'], NUM_ORDERS, p=[0.7, 0.2, 0.05, 0.05]),
    'payment_installments': np.random.randint(1, 12, NUM_ORDERS),
    'payment_value': np.random.uniform(15.0, 600.0, NUM_ORDERS)
})

# 7. Reviews
reviews = pd.DataFrame({
    'review_id': [f'R_{i}' for i in range(NUM_ORDERS)],
    'order_id': order_ids,
    'review_score': np.random.choice([1, 2, 3, 4, 5], NUM_ORDERS, p=[0.1, 0.05, 0.1, 0.3, 0.45]),
    'review_creation_date': [datetime(2018, 1, 1).strftime('%Y-%m-%d %H:%M:%S')] * NUM_ORDERS
})

# 8. Sellers
sellers = pd.DataFrame(columns=['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state'])


# Save all
files = {
    'olist_orders_dataset.csv': orders,
    'olist_order_items_dataset.csv': items,
    'olist_customers_dataset.csv': customers,
    'olist_products_dataset.csv': products,
    'olist_order_reviews_dataset.csv': reviews,
    'olist_order_payments_dataset.csv': payments,
    'olist_sellers_dataset.csv': sellers,
    'product_category_name_translation.csv': category
}

for filename, df in files.items():
    df.to_csv(os.path.join(DATA_PATH, filename), index=False)
    print(f"Generated {filename}")
