import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
import os
from scipy.stats import mannwhitneyu
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/processed/cleaned_data.csv")

print(df.head(10))
print(df.shape)

# DESCRIPTIVE STATISTICS (numeric only)
print(df.describe())

# SHAPE
print("\nShape:", df.shape)

# NULL VALUES
print("\nNull Values:")
print(df.isnull().sum())

# SKEWNESS (numeric columns only)
num_cols = df.select_dtypes(include=['int64', 'float64'])

print("\nSkewness:")
print(num_cols.skew())

# KURTOSIS
print("\nKurtosis:")
print(num_cols.kurt())

# VARIANCE
print("\nVariance:")
print(num_cols.var())

## Folder create (if not exists)
os.makedirs("reports/figures", exist_ok=True)

# Color lists
hist_colors = ["#A29BFE", "#74B9FF", "#FF7675", "#55EFC4", "#FAB1A0"]
kde_colors  = ["#6C5CE7", "#00CEC9", "#E17055", "#00B894", "#D63031"]
border_colors = ["#341F97", "#0984E3", "#D63031", "#00CEC9", "#E17055"]

for i, col in enumerate(num_cols):
    plt.figure(figsize=(10, 6), dpi=300)

    h_color = hist_colors[i % len(hist_colors)]
    k_color = kde_colors[i % len(kde_colors)]
    b_color = border_colors[i % len(border_colors)]

    sns.histplot(df[col], bins=30, kde=True,
                color=h_color,
                edgecolor=b_color,
                line_kws={"color": k_color, "linewidth": 2})

    plt.title(f"Skewness of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(False)

    # Save plot (ONLY ADDED PART)
    plt.savefig(f"reports/figures/hist_{col}.png", dpi=300, bbox_inches='tight')

    plt.show()

    print(f"{col} -> Skew: {df[col].skew():.2f}")

# Folder create (if not exists)
os.makedirs("reports/figures", exist_ok=True)

for col in num_cols:
    data = df[col].dropna()

    # Q-Q data (theoretical & sample)
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")

    plt.figure(figsize=(8, 6), dpi=300)

    # Background (modern)
    plt.gca().set_facecolor("#F5F5F5")

    # Middle reference line
    plt.plot(osm, slope*osm + intercept, color="#2C3E50", linewidth=2)

    # Deviation scatter (colorful)
    deviation = osr - (slope*osm + intercept)

    colors = np.where(np.abs(deviation) > np.std(deviation),
                      '#FF6B6B',  # out of range (red)
                      '#4ECDC4')  # near line (green)

    plt.scatter(osm, osr, c=colors, s=30)

    # Title
    plt.title(f"Check Kurtosis of {col}", fontsize=14)

    # Grid off
    plt.grid(False)

    # Save plot (ONLY ADDED)
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(f"reports/figures/qq_{col}.png", dpi=300, bbox_inches='tight')

    plt.show()

    # Supporting detail
    print(f"{col} -> Skewness: {df[col].skew():.2f}")
    print(f"{col} -> Kurtosis: {df[col].kurt():.2f}")


for col in num_cols:
    data = df[col].dropna()
    
    stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))

    print(f"{col}")
    print(f"Statistic = {stat:.4f}, p-value = {p:.4f}")

    if p > 0.05:
        print("Data looks NORMAL\n")
    else:
        print("Data NOT normal\n")

# Create folder once (loop se pehle)
os.makedirs("reports/figures", exist_ok=True)

for col in num_cols:
    data = df[col].dropna()
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data)+1) / len(sorted_data)

    mean = np.mean(data)
    std = np.std(data)

    normal_cdf = norm.cdf(sorted_data, mean, std)

    plt.figure(figsize=(10, 6), dpi=300)

    # Clean background
    plt.gca().set_facecolor("#F8F9FA")

    # Actual ECDF
    plt.plot(sorted_data, y,
             color="#6C5CE7",
             linewidth=2,
             label="Actual Data (ECDF)")

    # Normal CDF
    plt.plot(sorted_data, normal_cdf,
             color="#00B894",
             linestyle="--",
             linewidth=2,
             label="Normal Distribution")

    # Fill deviation area
    plt.fill_between(sorted_data, y, normal_cdf,
                     color="#FF7675",
                     alpha=0.15)

    # Maximum Tail Gap Line
    gap_index = np.argmax(np.abs(y - normal_cdf))
    x_gap = sorted_data[gap_index]
    y_actual = y[gap_index]
    y_normal = normal_cdf[gap_index]

    plt.vlines(x_gap,
               ymin=min(y_actual, y_normal),
               ymax=max(y_actual, y_normal),
               color="#D63031",
               linewidth=3,
               label="Maximum Tail Gap")

    # Title & Labels
    plt.title(f"Check Normality of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Cumulative Probability")

    plt.legend()
    plt.grid(False)

    # ✅ SAVE BEFORE SHOW
    plt.savefig(f"reports/figures/ecdf_normality_{col}.png",
                dpi=300,
                bbox_inches='tight')

    plt.show()
    plt.close()   # memory clean (important for many columns)

    # Supporting statistics
    max_gap = np.max(np.abs(y - normal_cdf))

    print(f"{col}")
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    print(f"Maximum CDF Gap: {max_gap:.4f}")

    if max_gap < 0.05:
        print("Conclusion: Approximately Normal\n")
    else:
        print("Conclusion: Not Normal\n")

target = df["size"]

results = []

for col in num_cols:
    if col != "size":
        corr, p = spearmanr(df[col], target)

        results.append({
            "Feature": col,
            "Spearman_Correlation": corr,
            "P_Value": p,
            "Significant": p < 0.05
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by absolute correlation (strongest first)
results_df["Abs_Corr"] = results_df["Spearman_Correlation"].abs()
results_df = results_df.sort_values("Abs_Corr", ascending=False)

print(results_df)

# Save results (optional)
results_df.to_csv("data/processed/spearman_results.csv", index=False)
print("Results saved to processed/spearman_results.csv")

# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Spearman correlation matrix
corr_matrix = numeric_df.corr(method="spearman")

plt.figure(figsize=(12, 8), dpi=300)

sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar=False)

plt.title("Correlation between feature", fontsize=14)

plt.xlabel("Features")
plt.ylabel("Features")

plt.show()

os.makedirs("reports/figures", exist_ok=True)

target = df["size"]

for col in df.select_dtypes(include=['int64', 'float64']).columns:

    if col in ["size", "waist", "shoe size"]:
        continue

    corr, p = spearmanr(df[col], target)

    plt.figure(figsize=(8, 6), dpi=300)

    # Color based on relationship
    if corr > 0:
        box_color = "#55EFC4"  # green (positive)
        relation = "Positive Relationship"
        text_color = "green"
    else:
        box_color = "#FF7675"  # red (negative)
        relation = "Negative Relationship"
        text_color = "red"

    sns.boxplot(y=df[col], color=box_color)

    # Title
    plt.title(f" Correlation of: {col}", fontsize=14)

    # Supporting detail
    detail = (
        f"Correlation: {corr:.2f}\n"
        f"P-Value: {p:.5f}\n"
        f"Relation: {relation}"
    )

    plt.text(0, df[col].median(),
             detail,
             fontsize=11,
             color=text_color,
             bbox=dict(facecolor="white", alpha=0.8))

    plt.grid(False)

    # save figure
    plt.savefig(f"reports/figures/box_{col}.png",
                dpi=300,
                bbox_inches="tight")

    plt.show()

    # Categorical columns detect
cat_cols = df.select_dtypes(include=['object', 'category']).columns

print("Categorical Columns:", list(cat_cols))

for col in cat_cols:
    print("\nColumn:", col)
    print("Data Type:", df[col].dtype)
    print("Shape (Rows):", df[col].shape)
    print("Unique Values:", df[col].nunique())

    print("\nTop Values (Frequency):")
    print(df[col].value_counts().head())

    print("\nFrequency Distribution (Full):")
    print(df[col].value_counts())


for col in cat_cols:
    print("\nColumn:", col)

    # frequency
    freq = df[col].value_counts()
    total = len(df)

    # percentage
    percentage = (freq / total) * 100

    print(pd.DataFrame({
        "Count": freq,
        "Percentage": percentage
    }))

    # imbalance insight
    print("\nImbalance Insight:")
    if percentage.max() > 70:
        print("⚠ Major imbalance (one class dominates)")
    elif percentage.max() > 50:
        print("⚠ Moderate imbalance")
    else:
        print("✔ Balanced distribution")

        os.makedirs("reports/figures", exist_ok=True)

# DONUT CHARTS (IMBALANCE CHECK) WITH SAVE FIGURE

os.makedirs("reports/figures", exist_ok=True)

for col in cat_cols:
    plt.figure(figsize=(8, 8), dpi=300)
    
    freq = df[col].value_counts()
    total = len(df)
    
    # Clean labels by escaping any $ symbols
    labels = [str(x).replace('$', r'\$') for x in freq.index]
    
    plt.pie(
        freq,
        labels=labels,
        autopct='%1.1f%%',
        colors=sns.color_palette("viridis", len(freq)),
        startangle=140,
        wedgeprops={'edgecolor': 'black'}
    )
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    
    # Clean title as well
    clean_col = str(col).replace('$', r'\$')
    plt.title(f"Imbalance Check - {clean_col}", fontsize=14)
    
    # safe filename (remove spaces and special chars)
    safe_col = str(col).replace(" ", "_").replace("/", "_").replace("$", "_")
    
    plt.savefig(
        f"reports/figures/donut_{safe_col}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.show()
    
    distribution = (freq / total) * 100
    
    print(f"\nClass Distribution for {col} (Percentage):")
    print(distribution)
    
    max_percent = distribution.max()
    
    print("\nSupporting Insight:")
    
    if max_percent > 70:
        print("Major Imbalance")
    elif max_percent > 50:
        print("Moderate Imbalance")
    else:
        print("Balanced")
    
    print(f"Highest Class Percentage: {max_percent:.2f}%\n")

# Chi-square tests
results = []

# 📌 HAR COLUMN SE SIRF TOP 20 CATEGORIES LO
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Total categorical columns: {len(cat_cols)}")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# ==============================
# CHI SQUARE TEST
# ==============================

chi_results = []

for col1, col2 in itertools.combinations(cat_cols, 2):

    top1 = df[col1].value_counts().nlargest(20).index
    top2 = df[col2].value_counts().nlargest(20).index

    df_filtered = df[df[col1].isin(top1) & df[col2].isin(top2)]

    if len(df_filtered) > 50:

        table = pd.crosstab(
            df_filtered[col1].astype(str),
            df_filtered[col2].astype(str)
        )

        chi2, p, dof, _ = chi2_contingency(table)

        chi_results.append({
            "Feature1": col1,
            "Feature2": col2,
            "P_Value": p,
            "Significant": p < 0.05
        })

chi_df = pd.DataFrame(chi_results)

print("\nChi-Square Results")
print(chi_df.sort_values("P_Value"))

chi_df.to_csv("data/processed/chi_square_results.csv", index=False)


# ==============================
# NETWORK GRAPH
# ==============================

G = nx.Graph()

for _, row in chi_df[chi_df["Significant"]].iterrows():
    G.add_edge(row["Feature1"], row["Feature2"])

plt.figure(figsize=(10,8), dpi=300)

pos = nx.spring_layout(G, k=0.6, seed=42)

nx.draw_networkx_nodes(
    G, pos,
    node_size=2000,
    node_color="#74B9FF",
    edgecolors="black"
)

nx.draw_networkx_edges(
    G, pos,
    width=2,
    edge_color="#55EFC4"
)

nx.draw_networkx_labels(
    G, pos,
    font_size=10
)

plt.title("Categorical Network (Significant Relationships)")
plt.axis("off")

plt.savefig("reports/figures/categorical_network.png", dpi=300)
plt.show()

# ======================================
# FEATURE vs TARGET CORRELATION
# ======================================

target = "size"

results = []

features = df.drop(columns=[target]).columns

for col in features:

    data = df[[col, target]].dropna()

    # categorical ko numeric me convert
    if data[col].dtype == "object":
        data[col] = data[col].astype("category").cat.codes

    try:
        corr = data[col].corr(data[target], method="spearman")

        results.append({
            "Feature": col,
            "Correlation": corr,
            "Importance": abs(corr)
        })

    except:
        pass


corr_df = pd.DataFrame(results)

corr_df = corr_df.sort_values("Importance", ascending=True)

print("\nFeature Importance (Spearman Correlation)")
print(corr_df)

# ======================================
# SAVE CORRELATION RESULTS
# ======================================

import os

os.makedirs("data/processed", exist_ok=True)

corr_df.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

# ======================================
# FEATURE IMPORTANCE BAR PLOT
# ======================================

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(10,7), dpi=300)

bars = plt.barh(
    corr_df["Feature"],
    corr_df["Importance"]
)

# percentage labels
for i, value in enumerate(corr_df["Importance"]):
    plt.text(value + 0.01, i, f"{value:.2f}", va="center")

plt.title("Feature Importance vs Target (Spearman Correlation)")
plt.xlabel("Correlation Strength")
plt.ylabel("Features")

plt.tight_layout()

# ======================================
# SAVE FIGURE
# ======================================

plt.savefig(
    "reports/figures/eda_feature_importance_barplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Feature importance plot saved")

# ======================================
# FINAL EDA SAVE
# ======================================

df.to_csv(
    "data/processed/EDA_final_dataset.csv",
    index=False
)

print("EDA dataset saved successfully")