# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/EDA_final_dataset.csv")

print("First 10 Rows:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# TARGET AND FEATURES
# ==========================================

target = "size"

X = df.drop(columns=[target])
y = df[target]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# SHAPES
# ==========================================

print("\nTrain Test Split Shapes")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# ==========================================
# DATA LEAKAGE CHECK
# ==========================================

train_index = set(X_train.index)
test_index = set(X_test.index)

leakage = train_index.intersection(test_index)

if len(leakage) == 0:
    print("\nNo Data Leakage Detected")
else:
    print("\nWarning: Data Leakage Found")

# ==========================================
# SAVE TRAIN TEST DATASETS
# ==========================================

os.makedirs("data/processed", exist_ok=True)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data/processed/train_dataset.csv", index=False)
test_df.to_csv("data/processed/test_dataset.csv", index=False)

print("\nTrain dataset saved")
print("Test dataset saved")

# ==========================================
# CREATE FIGURE FOLDER
# ==========================================

os.makedirs("reports/figures", exist_ok=True)

# ==========================================
# TRAIN TEST TARGET DISTRIBUTION
# ==========================================

train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()

dist_df = pd.DataFrame({
    "Train": train_counts,
    "Test": test_counts
}).fillna(0)

# percentage calculation
total = dist_df.sum(axis=1)
train_percent = (dist_df["Train"] / total) * 100
test_percent = (dist_df["Test"] / total) * 100

# ==========================================
# STACKED BAR PLOT
# ==========================================

ax = dist_df.plot(
    kind="bar",
    stacked=True,
    figsize=(12,7),
    color=["#4CAF50", "#FF7043"],
    width=0.8
)

plt.title(
    "Train vs Test Target Distribution (Stacked)",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Size Categories")
plt.ylabel("Number of Samples")

plt.legend(
    title="Dataset Split",
    fontsize=10
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# ==========================================
# ADD PERCENTAGE LABELS
# ==========================================

for i, (train, test) in enumerate(zip(train_percent, test_percent)):
    
    height = dist_df.iloc[i].sum()
    
    ax.text(
        i,
        height + 5,
        f"{train:.1f}% / {test:.1f}%",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

plt.tight_layout()

# ==========================================
# SAVE FIGURE
# ==========================================

plt.savefig(
    "reports/figures/train_test_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nStacked Bar Plot saved in reports/figures/")