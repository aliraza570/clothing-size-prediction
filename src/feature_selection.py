# ===============================
# Import Libraries
# ===============================
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt  # ✅ Fixed
import networkx as nx
import seaborn as sns
import os

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/processed/feature_engineered_dataset.csv")
print("Original Dataset Shape:", df.shape)

# ===============================
# Convert numeric size to categories (S, M, L, XL, XXL)
# ===============================
df['size_class'] = pd.qcut(df['size'], q=5, labels=['S','M','L','XL','XXL'])
print("\nSample of size_class mapping:")
print(df[['size','size_class']].head(10))

# ===============================
# Separate Target
# ===============================
target = 'size_class'
y = df[target]
X = df.drop(columns=['size','size_class'])
print("\nFeatures Shape:", X.shape)

# ===============================
# 1️⃣ Remove Highly Correlated Features
# ===============================
corr_threshold = 0.9
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
print("\nHighly correlated features to drop (corr > 0.9):", to_drop_corr)

X_reduced = X.drop(columns=to_drop_corr)
print("Shape after correlation filter:", X_reduced.shape)

# ===============================
# 2️⃣ VIF (Variance Inflation Factor)
# ===============================
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

X_vif = X_reduced.copy()
vif_threshold = 5  # Safe threshold for VIF
while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df['VIF'].max()
    if max_vif > vif_threshold:
        remove_feature = vif_df.sort_values('VIF', ascending=False)['feature'].iloc[0]
        print("Removing feature due to high VIF:", remove_feature, "VIF =", max_vif)
        X_vif = X_vif.drop(columns=[remove_feature])
    else:
        break

print("Shape after VIF filtering:", X_vif.shape)
print("\nFinal Features after VIF:\n", X_vif.columns.tolist())

# ===============================
# 3️⃣ Model-based Feature Importance (Random Forest)
# ===============================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_vif, y)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_vif.columns).sort_values(ascending=False)
print("\nTop Features by Random Forest Importance:\n", importances.head(10))

# Select top features automatically using median threshold
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_selected = X_vif[X_vif.columns[selector.get_support()]]

print("\nShape after model-based feature selection (median threshold):", X_selected.shape)
print("\nSelected Features:\n", X_selected.columns.tolist())

# ===============================
# 4️⃣ Class Imbalance Acknowledge (no balancing applied)
# ===============================
print("\nClass Distribution (acknowledge only, no balancing applied):")
class_counts = y.value_counts()
class_percent = y.value_counts(normalize=True) * 100
print(class_counts)
print("\nPercentage of each class:\n", class_percent)

# ===============================
# 5️⃣ Save final selected features dataset INCLUDING TARGET
# ===============================
final_df = pd.concat([X_selected, y], axis=1)  # ✅ Target included
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv("data/processed/feature_selected_dataset.csv", index=False)
print("\nFinal dataset saved to: data/processed/feature_selected_dataset.csv")

# ===============================
# 1️⃣ Horizontal Feature Importance Bar Plot
# ===============================
rf.fit(X, df[target])
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(12,8), dpi=300)
colors = sns.color_palette("Spectral", len(importances))
bars = plt.barh(importances.index, importances.values, color=colors)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f"{width*100:.1f}%", va='center', fontsize=9)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top Feature Importance (Random Forest)", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/feature_importance_barplot.png", dpi=300)
plt.show()

# ===============================
# 2️⃣ Correlation Heatmap (without color bar)
# ===============================
corr_matrix = X.corr()
plt.figure(figsize=(12,10), dpi=300)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.8,
    cbar=False,
    square=True
)
plt.title("Feature selected Correlation", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/feature_correlation_heatmap.png", dpi=300)
plt.show()

# ===============================
# 3️⃣ Feature Network Plot (Correlation-based)
# ===============================
threshold = 0.3  # show only strong correlations
G = nx.Graph()
for i in X.columns:
    G.add_node(i)
for i in X.columns:
    for j in X.columns:
        if i != j:
            corr_val = corr_matrix.loc[i,j]
            if abs(corr_val) >= threshold:
                color = 'green' if corr_val > 0 else 'red'
                G.add_edge(i, j, weight=abs(corr_val), color=color)

pos = nx.spring_layout(G, k=1, seed=42)
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight']*3 for u,v in edges]

plt.figure(figsize=(12,12), dpi=300)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1500,
    node_color=sns.color_palette("Set3", n_colors=len(G.nodes())),
    edge_color=colors,
    width=weights,
    font_size=10
)
plt.title("Feature Relationship Network Plot", fontsize=16)
plt.tight_layout()
plt.savefig("reports/figures/feature_network_plot.png", dpi=300)
plt.show()