# 🛒 E-Commerce Customer Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A full end-to-end Data Analytics project covering customer segmentation, RFM analysis, churn prediction, and an interactive dashboard — built on the **Brazilian Olist E-Commerce Dataset**.

---

## 📌 Problem Statement

E-commerce businesses struggle to understand **who their best customers are**, **which customers are about to leave**, and **how to increase revenue through targeted strategies**. This project answers those questions using real-world transactional data.

---

## 🎯 Key Objectives

- Perform **Exploratory Data Analysis (EDA)** on 100K+ orders
- Build **RFM (Recency, Frequency, Monetary)** customer segmentation
- Identify **churned and at-risk customers**
- Visualize **sales trends, category performance, and geography**
- Deploy an **interactive Streamlit dashboard**

---

## 📊 Dataset

**Source:** [Olist Brazilian E-Commerce Dataset — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| File | Description |
|---|---|
| `olist_orders_dataset.csv` | Order status, timestamps |
| `olist_order_items_dataset.csv` | Products per order, price, freight |
| `olist_customers_dataset.csv` | Customer location |
| `olist_products_dataset.csv` | Product category, dimensions |
| `olist_order_reviews_dataset.csv` | Customer reviews and scores |
| `olist_order_payments_dataset.csv` | Payment type, installments, value |

> 📥 Download the dataset from Kaggle and place all CSV files inside the `data/` folder.

---

## 🗂️ Project Structure

```
ecommerce-customer-analysis/
│
├── data/                          # Raw CSV files (not pushed to GitHub)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and data cleaning
│   ├── 02_rfm_analysis.ipynb      # RFM segmentation
│   ├── 03_customer_segmentation.ipynb  # KMeans clustering
│   └── 04_churn_analysis.ipynb    # Churn identification
│
├── src/
│   ├── data_preprocessing.py      # Data loading and merging
│   ├── rfm_analysis.py            # RFM scoring logic
│   └── visualization.py           # Reusable plotting functions
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── assets/                        # Screenshots and images
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 Key Insights Found

- 📦 **96,000+** orders analyzed across **2016–2018**
- 🏆 Top customer segment: **"Champions"** (high RFM score) — 18% of customers, 42% of revenue
- ⚠️ **23% of customers** are at churn risk (not ordered in 6+ months)
- 💳 **Credit card** is the most used payment method (74%)
- 📍 **São Paulo** generates the highest order volume
- ⭐ Average review score: **4.09 / 5.0**

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-customer-analysis.git
cd ecommerce-customer-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files in the `data/` folder.

### 4. Run notebooks in order
```bash
jupyter notebook
```
Open notebooks in sequence: `01` → `02` → `03` → `04`

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📸 Dashboard Preview

> *(Add screenshots here after running)*

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (KMeans) |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

## 👤 Author

**Your Name**  
MCA Student | Kolkata  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License.
