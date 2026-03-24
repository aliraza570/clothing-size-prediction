import pandas as pd
import matplotlib.pyplot as plt
import os

# Read JSON file (JSON Lines format)
df = pd.read_json("data/raw/modcloth_final_data.json", lines=True)

# Show first 5 rows
print("First 5 Rows:")
print(df.head())

# Shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# Info (No .sum() here)
print("\nDataset Info:")
df.info()

# Column Names
print("\nColumn Names:")
print(df.columns.tolist())

# Duplicate Count
print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nmissing value check")
print(df.isna().sum())

df = df.drop(columns=['item_id', 'review_summary', 'review_text'], errors='ignore')

print("Updated Shape (Rows, Columns):")
print(df.shape)

# Show remaining columns
print("\nRemaining Columns:")
print(df.columns.tolist())

# Separate Numerical & Categorical
# -------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# -------------------------------
# Numerical Imputation (Median)
# -------------------------------
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# -------------------------------
# Categorical Imputation (Mode)
# -------------------------------
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# -------------------------------
# Final Missing Values Check
# -------------------------------
print("Remaining Missing Values:\n")
print(df.isnull().sum())

print("\nUpdated Shape:")
print(df.shape)

print("\nImputation Completed Successfully")

# Remove duplicates
df = df.drop_duplicates()

# Reset index (important step)
df.reset_index(drop=True, inplace=True)

print("Duplicates Removed")
print("New Shape:", df.shape)

# Numerical columns detect karo
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Outlier detection & handling (IQR capping)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Outlier count (optional)
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} → Outliers: {outliers.shape[0]}")

    # Handling (capping)
    df[col] = df[col].clip(lower, upper)

print("\nOutliers Detected & Handled Successfully")

# Report folder create karo
save_dir = "reports/figures"
os.makedirs(save_dir, exist_ok=True)

# Before copy
df_before = df.copy()

# Numerical columns (exclude waist & shoe related)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Filter: remove columns containing 'shoe' or 'waist'
num_cols = [col for col in num_cols if ('shoe' not in col.lower()) and ('waist' not in col.lower())]

# Plot each column
for col in num_cols:
    plt.figure(figsize=(8, 5), dpi=300)

    # Boxplot Before & After
    data = [df_before[col], df[col]]

    box = plt.boxplot(
        data,
        labels=["Before", "After"],
        patch_artist=True
    )

    # Colors
    colors = ['#FFB6C1', '#90EE90']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Median line (red dashed)
    plt.axhline(
        y=df[col].median(),
        color='red',
        linestyle='--',
        label='Median'
    )

    # Supporting details
    plt.title(f"Outlier Impact - {col}")
    plt.ylabel(col)
    plt.legend()
    plt.grid(False)

    plt.show()

    # Target column
target = 'size'

# Check if exists
if target in df.columns:

    # Data type
    print("Data Type:", df[target].dtype)

    # Unique values (important for classification)
    print("\nUnique Sizes:")
    print(df[target].unique())

    # Value counts (distribution)
    print("\nSize Distribution:")
    print(df[target].value_counts())

    # Percentage distribution
    print("\nPercentage Distribution:")
    print(df[target].value_counts(normalize=True) * 100)

    # Missing values
    print("\nMissing Values:", df[target].isnull().sum())

else:
    print("Target column 'size' not found in dataset.")

   # Target column
target = 'size'

if target in df.columns:

    # Distribution
    dist = df[target].value_counts()

    # Colors (contrasting)
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0']

    # Pie Chart
    plt.figure(figsize=(8, 8), dpi=300)
    wedges, texts, autotexts = plt.pie(
        dist,
        labels=dist.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:len(dist)]
    )

    # Feature name as title
    plt.title(f"Distribution of Feature: {target}")

    # Legend with size names
    plt.legend(
        wedges,
        [f"Size - {label}" for label in dist.index],
        title="Size Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # Save plot
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/size_distribution.png", dpi=300, bbox_inches='tight')

    plt.show()

    print("Plot saved successfully")

else:
    print("Target column 'size' not found in dataset.")

    # Save cleaned data to processed folder as CSV

df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Cleaned data saved successfully in data/processed/cleaned_data.csv")