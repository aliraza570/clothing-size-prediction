# ==========================================
# Import Libraries
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from catboost import CatBoostClassifier

# ==========================================
# Create Required Folders
# ==========================================
os.makedirs("data/processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/processed/feature_selected_dataset.csv")

print("First 10 Rows:\n")
print(df.head(10))

print("\nDataset Shape:", df.shape)

# ==========================================
# Define Target
# ==========================================
target = "size_class"

X = df.drop(columns=[target])
y = df[target]

# ==========================================
# Encode Target
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================================
# Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ==========================================
# CatBoost Model
# ==========================================
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)

# ==========================================
# Train Model
# ==========================================
model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================
y_pred = model.predict(X_test).ravel()

# ==========================================
# Evaluation Metrics
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nModel Performance")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ==========================================
# Cross Validation
# ==========================================
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import os

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# ==========================================
# Train Model on Full Dataset
# ==========================================

model.fit(X, y_encoded)


# ==========================================
# Create Folder if Not Exists
# ==========================================

os.makedirs("data/processed", exist_ok=True)

# ==========================================
# Save Trained Model
# ==========================================

joblib.dump(model, "data/processed/catboost_model.pkl")

print("\nModel successfully saved at: data/processed/catboost_model.pkl")
# ==========================================
# GRAPH 1 — Confusion Matrix
# ==========================================
plt.figure(figsize=(8,6), dpi=300)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix — CatBoost Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix_catboost.png")

plt.show()
plt.close()

# ==========================================
# GRAPH 2 — Feature Importance
# ==========================================
importance = model.get_feature_importance()

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

top_feat = feat_imp.head(15)

plt.figure(figsize=(10,6), dpi=300)

palette = sns.color_palette("viridis", len(top_feat))

ax = sns.barplot(
    data=top_feat,
    x="Importance",
    y="Feature",
    palette=palette
)

# Percentage labels
total = top_feat["Importance"].sum()

for i, v in enumerate(top_feat["Importance"]):
    percentage = (v / total) * 100
    ax.text(
        v + 0.5,
        i,
        f"{percentage:.1f}%",
        va="center"
    )

plt.title("Top Feature Importance — CatBoost")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("reports/figures/feature_importance_catboost.png")

plt.show()
plt.close()


# ==========================================
# GRAPH 3 — Class Distribution
# ==========================================
plt.figure(figsize=(8,6), dpi=300)

palette = sns.color_palette("Set2")

ax = sns.countplot(
    x=pd.Series(y),
    palette=palette
)

# Percentage labels
total = len(y)

for p in ax.patches:
    count = p.get_height()
    percentage = 100 * count / total

    ax.annotate(
        f"{percentage:.1f}%",
        (p.get_x() + p.get_width()/2, count),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title("Target Class Distribution")
plt.xlabel("Size Class")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("reports/figures/class_distribution.png")

plt.show()
plt.close()

print("\nAll graphs saved inside 'figures' folder")


# ==========================================
# Create Folder
# ==========================================
os.makedirs("reports/figures", exist_ok=True)


# ==========================================
# Prepare Data
# ==========================================

classes = np.unique(y_test)

# Binarize labels
y_test_bin = label_binarize(y_test, classes=classes)

# Model probabilities
y_prob = model.predict_proba(X_test)


# ==========================================
# 1️⃣ ROC CURVE
# ==========================================

plt.figure(figsize=(8,6), dpi=300)

colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        color=colors[i % len(colors)],
        lw=2,
        label=f"Class {classes[i]} (AUC = {roc_auc:.2f})"
    )

# Random baseline
plt.plot([0,1],[0,1],'k--',label="Random Model")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — CatBoost Model")

plt.legend()

plt.tight_layout()
plt.savefig("reports/figures/roc_curve.png")

plt.show()
plt.close()


# ==========================================
# 2️⃣ PRECISION-RECALL CURVE
# ==========================================

plt.figure(figsize=(8,6), dpi=300)

for i in range(len(classes)):

    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    ap_score = average_precision_score(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    plt.plot(
        recall,
        precision,
        color=colors[i % len(colors)],
        lw=2,
        label=f"Class {classes[i]} (AP = {ap_score:.2f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision–Recall Curve — CatBoost Model")

plt.legend()

plt.tight_layout()
plt.savefig("reports/figures/precision_recall_curve.png")

plt.show()
plt.close()


# ==========================================
# 3️⃣ SHAP FEATURE IMPORTANCE
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\nAll plots saved inside 'reports/figures'")