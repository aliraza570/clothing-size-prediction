# ===============================
# Import Libraries
# ===============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/processed/feature_selected_dataset.csv")

print("First 10 Rows:\n")
print(df.head(10))

print("\nDataset Shape:", df.shape)

# ===============================
# Define Target
# ===============================
target = "size_class"

X = df.drop(columns=[target])
y = df[target]

# ===============================
# Encode Target
# ===============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===============================
# Scale Data (for SVM)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Cross Validation Setup
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = ['accuracy','precision_weighted','recall_weighted','f1_weighted']

# ===============================
# Models
# ===============================
models = {

    "CatBoost": CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        C=1,
        kernel="rbf",
        gamma="scale"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42
    )
}

# ===============================
# Model Evaluation
# ===============================
results = []

for name, model in models.items():

    print(f"\nRunning {name}...")

    if name == "SVM":
        scores = cross_validate(
            model,
            X_scaled,
            y_encoded,
            cv=cv,
            scoring=scoring
        )
    else:
        scores = cross_validate(
            model,
            X,
            y_encoded,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

    accuracy = scores['test_accuracy'].mean()
    precision = scores['test_precision_weighted'].mean()
    recall = scores['test_recall_weighted'].mean()
    f1 = scores['test_f1_weighted'].mean()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# ===============================
# Comparison Table
# ===============================
results_df = pd.DataFrame(results)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)

# ===============================
# Best Model
# ===============================
best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

print("\n==============================")
print("BEST MODEL")
print("==============================")

print(best_model)

# ===============================
# Save Results
# ===============================

results_df.to_csv("data/processed/model_comparison_results.csv", index=False)

print("\nModel comparison results saved to:")
print("data/processed/model_comparison_results.csv")
