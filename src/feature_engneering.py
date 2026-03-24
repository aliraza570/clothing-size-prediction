import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import numpy as np

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/processed/train_dataset.csv")

# ===============================
# Drop unnecessary columns
# ===============================
df = df.drop(columns=['item_id','review_summary','review_text'], errors='ignore')
df = df.drop_duplicates()

# ===============================
# Separate target
# ===============================
target = "size"

y = df[target]
X = df.drop(columns=[target])

original_columns = X.columns.tolist()

# ===============================
# Detect numeric & categorical columns
# ===============================
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ===============================
# Handle missing values
# ===============================
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# ===============================
# Outlier Handling (IQR)
# ===============================
for col in num_cols:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X[col] = np.clip(X[col], lower, upper)

# ===============================
# Feature Creation
# ===============================
if "height" in X.columns and "weight" in X.columns:

    X["BMI"] = X["weight"] / ((X["height"] / 100) ** 2)

    # division safe
    X["height_weight_ratio"] = X["height"] / (X["weight"] + 1e-5)

if "chest" in X.columns and "waist" in X.columns:

    X["chest_waist_ratio"] = X["chest"] / (X["waist"] + 1e-5)

if "waist" in X.columns and "hips" in X.columns:

    X["waist_hip_ratio"] = X["waist"] / (X["hips"] + 1e-5)

# ===============================
# Encode Categorical Columns
# ===============================
label_encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col].astype(str))

    label_encoders[col] = le

# ===============================
# Detect numeric columns again
# ===============================
num_cols_updated = X.select_dtypes(include=['int64','float64']).columns.tolist()

# ===============================
# Save numeric data BEFORE scaling
# ===============================
before_scaling = X[num_cols_updated].copy()

# ===============================
# Feature Scaling
# ===============================
scaler = StandardScaler()

X[num_cols_updated] = scaler.fit_transform(X[num_cols_updated])

after_scaling = X[num_cols_updated].copy()

# ===============================
# Combine features + target
# ===============================
final_df = pd.concat([X, y], axis=1)

# ===============================
# Feature Engineering Report
# ===============================

created_features = [col for col in X.columns if col not in original_columns]

print("\nNew Features Created:", created_features)

print("Total New Features:", len(created_features))

print("Final Dataset Shape:", final_df.shape)

print("Final Columns:", final_df.columns.tolist())

# ===============================
# Correlation Heatmap (Improved)
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns
import os

corr_matrix = final_df.corr(numeric_only=True)

plt.figure(figsize=(16,12), dpi=300)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=1,
    cbar=False,  # remove side color bar
    annot_kws={"size":10}
)

plt.title("Feature Correlation Heatmap (After Feature Engineering)", fontsize=18)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()

os.makedirs("figures", exist_ok=True)

plt.savefig(
    "reports/figures/feature_engineering_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
corr_matrix = final_df.corr(numeric_only=True)

G = nx.Graph()

threshold = 0.3  # minimum correlation to draw edge

# Add edges
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):

        corr = corr_matrix.iloc[i,j]

        if abs(corr) > threshold:

            color = "green" if corr > 0 else "red"

            G.add_edge(
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                weight=abs(corr),
                color=color
            )

# Layout
pos = nx.spring_layout(G, seed=42)

edge_colors = [G[u][v]['color'] for u,v in G.edges()]

plt.figure(figsize=(12,10), dpi=300)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="skyblue",
    node_size=3000,
    font_size=10,
    edge_color=edge_colors,
    width=2
)

plt.title("Feature Relationship Network", fontsize=18)

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_relationship_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# create folder if not exists
os.makedirs("data/processed", exist_ok=True)

# save csv
final_df.to_csv("data/processed/feature_engineered_dataset.csv", index=False)

print("\nDataset successfully saved!")
print("Location: data/processed/feature_engineered_dataset.csv")