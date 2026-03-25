# %% [markdown]
# # 🤖 Notebook 3: ML Customer Segmentation (KMeans)
# **E-Commerce Customer Behavior Analysis**
# Author: Your Name | MCA Student, Kolkata
#
# In this notebook, we use **unsupervised machine learning (KMeans Clustering)**
# to discover natural customer groups from behavioral data — complementing the
# rule-based RFM approach with a data-driven one.

# %% [markdown]
# ## 1. Setup

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

from sklearn.preprocessing    import StandardScaler
from sklearn.cluster          import KMeans
from sklearn.decomposition    import PCA
from sklearn.metrics          import silhouette_score

from src.data_preprocessing import load_raw_data, merge_master_df, get_customer_summary
from src.rfm_analysis        import compute_rfm

sns.set_theme(style='whitegrid')
np.random.seed(42)
print("✅ Setup complete")

# %% [markdown]
# ## 2. Feature Engineering for Clustering

# %%
dfs    = load_raw_data()
master = merge_master_df(dfs)
cust   = get_customer_summary(master)
rfm    = compute_rfm(cust)

# Select features for clustering
features = [
    'recency_days',
    'total_orders',
    'total_spend',
    'avg_order_value',
    'avg_review_score',
    'customer_lifetime_days',
]

cluster_df = rfm[['customer_unique_id'] + features].dropna().copy()
print(f"Clustering dataset shape: {cluster_df.shape}")
cluster_df[features].describe()

# %% [markdown]
# ## 3. Feature Scaling

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[features])

print("Scaled feature matrix shape:", X_scaled.shape)
print("Mean (should be ~0):", X_scaled.mean(axis=0).round(3))
print("Std  (should be ~1):", X_scaled.std(axis=0).round(3))

# %% [markdown]
# ## 4. Elbow Method — Choose Optimal K

# %%
inertias    = []
sil_scores  = []
K_range     = range(2, 11)

print("Computing KMeans for K = 2 to 10 ...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=5000))
    print(f"   K={k}  Inertia={km.inertia_:,.0f}  Silhouette={sil_scores[-1]:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow
axes[0].plot(K_range, inertias, marker='o', color='#3498db', linewidth=2)
axes[0].set_title("Elbow Method", fontweight='bold')
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].axvline(x=4, color='red', linestyle='--', label='Chosen K=4')
axes[0].legend()

# Silhouette
axes[1].plot(K_range, sil_scores, marker='s', color='#2ecc71', linewidth=2)
axes[1].set_title("Silhouette Scores", fontweight='bold')
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=4, color='red', linestyle='--', label='Chosen K=4')
axes[1].legend()

plt.suptitle("Optimal Number of Clusters", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Train Final KMeans Model (K=4)

# %%
K_OPTIMAL = 4

kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
cluster_df['cluster'] = kmeans.fit_predict(X_scaled)

print("✅ KMeans trained with K =", K_OPTIMAL)
print("\nCluster sizes:")
print(cluster_df['cluster'].value_counts().sort_index())

# %% [markdown]
# ## 6. Cluster Profiles

# %%
profile = cluster_df.groupby('cluster')[features].mean().round(2)
profile['count'] = cluster_df['cluster'].value_counts().sort_index()

# Normalize for radar chart later
print("=== Cluster Profiles ===")
print(profile.to_string())

# %%
# Assign human-readable cluster names based on profile
CLUSTER_NAMES = {
    0: '💎 High-Value Champions',
    1: '😴 Inactive / Lost',
    2: '🌱 New / Low Engagement',
    3: '⭐ Regular Buyers',
}
# NOTE: Actual mapping depends on your data — adjust after viewing profile above

cluster_df['cluster_name'] = cluster_df['cluster'].map(CLUSTER_NAMES)

# %% [markdown]
# ## 7. PCA Visualization (2D)

# %%
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

cluster_df['PC1'] = X_pca[:, 0]
cluster_df['PC2'] = X_pca[:, 1]

# %%
fig = px.scatter(
    cluster_df.sample(5000, random_state=42),
    x='PC1', y='PC2',
    color='cluster_name',
    opacity=0.6,
    title='KMeans Clusters — PCA 2D View (sample of 5,000)',
    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'},
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(height=500, legend_title='Cluster')
fig.show()

# %% [markdown]
# ## 8. Cluster Feature Comparison (Radar Chart)

# %%
from plotly.graph_objects import Figure, Scatterpolar

# Normalize each feature to 0-1 for radar
norm_profile = (profile[features] - profile[features].min()) / \
               (profile[features].max() - profile[features].min() + 1e-9)

fig = Figure()
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

for i, (idx, row) in enumerate(norm_profile.iterrows()):
    fig.add_trace(Scatterpolar(
        r=row.values.tolist() + [row.values[0]],
        theta=features + [features[0]],
        fill='toself',
        name=CLUSTER_NAMES.get(idx, f'Cluster {idx}'),
        line=dict(color=colors[i % len(colors)])
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title='Cluster Feature Profiles (Normalized)',
    height=500
)
fig.show()

# %% [markdown]
# ## 9. Avg Spend & Orders by Cluster

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cluster_spend = cluster_df.groupby('cluster_name')['total_spend'].mean().sort_values()
cluster_orders = cluster_df.groupby('cluster_name')['total_orders'].mean().sort_values()

cluster_spend.plot(kind='barh', ax=axes[0], color='#2ecc71', edgecolor='white')
axes[0].set_title("Avg Total Spend by Cluster", fontweight='bold')
axes[0].set_xlabel("Avg Spend (BRL)")

cluster_orders.plot(kind='barh', ax=axes[1], color='#3498db', edgecolor='white')
axes[1].set_title("Avg Orders by Cluster", fontweight='bold')
axes[1].set_xlabel("Avg Number of Orders")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Save Cluster Results

# %%
output = cluster_df[['customer_unique_id', 'cluster', 'cluster_name'] + features]
output.to_csv('../data/customer_clusters.csv', index=False)
print("✅ Saved to data/customer_clusters.csv")
print(f"   Shape: {output.shape}")
output.head(5)

# %% [markdown]
# ## 11. Key Findings
#
# | Cluster | Name | Characteristics | Action |
# |---|---|---|---|
# | 0 | High-Value Champions | Recent, frequent, high spend | Retain & reward |
# | 1 | Inactive / Lost | Old last purchase, low spend | Re-engagement or sunset |
# | 2 | New / Low Engagement | Short lifetime, low orders | Nurture & convert |
# | 3 | Regular Buyers | Moderate all-round metrics | Upsell and grow |
