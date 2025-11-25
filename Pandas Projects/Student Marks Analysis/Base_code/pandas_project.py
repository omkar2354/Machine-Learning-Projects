"""
pandas_project.py

Complete EDA + visualizations + simple ML on student_marks_big_dataset.csv (2k rows).
- Windows paths supported.
- Uses non-interactive matplotlib backend (saves images, no GUI).
- Prints verbose output at the end for demo / presentation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend to avoid Tkinter errors when saving plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    classification_report, confusion_matrix
)

# -------------------------
# Config / Paths (edit if needed)
# -------------------------
DATA_PATH = r"C:\PYTHON LECTURE\student_marks_big_dataset.csv"   # <-- set your CSV path here
OUT_DIR = r"C:\PYTHON LECTURE\pandas_report_outputs"            # <-- outputs will be saved here
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# seaborn style
sns.set(style="whitegrid", context="talk")

def save_fig(fig_path):
    """Helper to save and close matplotlib figures safely."""
    try:
        plt.tight_layout()
        plt.savefig(fig_path)
    except Exception as e:
        print(f"Warning: failed to save {fig_path}: {e}")
    finally:
        plt.close()

# -------------------------
# Load data
# -------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(5).to_string(index=False))

# -------------------------
# Basic cleaning & derived columns
# -------------------------
for col in ["Math", "Science", "English", "Attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Recompute Average
df["Average"] = df[["Math", "Science", "English"]].mean(axis=1).round(2)

# Create Grade if missing or has NaNs
if "Grade" not in df.columns or df["Grade"].isnull().any():
    bins = [0, 60, 70, 80, 90, 100]
    labels = ["F", "D", "C", "B", "A"]
    df["Grade"] = pd.cut(df["Average"], bins=bins, labels=labels, include_lowest=True)

# -------------------------
# Descriptive statistics
# -------------------------
desc = df[["Math", "Science", "English", "Attendance", "Average"]].describe().round(2)
print("\n=== Descriptive statistics ===")
print(desc.to_string())
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

# -------------------------
# Correlation matrix + heatmap
# -------------------------
corr = df[["Math", "Science", "English", "Attendance", "Average"]].corr().round(2)
print("\n=== Correlation matrix ===")
print(corr.to_string())

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation matrix")
save_fig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))

# -------------------------
# Subject histograms + boxplots
# -------------------------
for col in ["Math", "Science", "English"]:
    plt.figure(figsize=(10,4))
    sns.histplot(df[col].dropna(), bins=25, kde=True)
    plt.title(f"{col} Score Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    save_fig(os.path.join(PLOTS_DIR, f"{col.lower()}_distribution.png"))

plt.figure(figsize=(8,6))
sns.boxplot(data=df[["Math","Science","English"]].dropna())
plt.title("Boxplots of Subject Scores")
save_fig(os.path.join(PLOTS_DIR, "score_boxplots.png"))

# -------------------------
# Grade distribution
# -------------------------
grade_counts = df["Grade"].value_counts().reindex(["A","B","C","D","F"]).fillna(0)
plt.figure(figsize=(6,4))
sns.barplot(x=grade_counts.index.astype(str), y=grade_counts.values)
plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Count")
save_fig(os.path.join(PLOTS_DIR, "grade_distribution.png"))

# -------------------------
# Average distribution
# -------------------------
plt.figure(figsize=(8,4))
sns.histplot(df["Average"].dropna(), bins=30, kde=True)
plt.title("Distribution of Student Averages")
plt.xlabel("Average Mark")
save_fig(os.path.join(PLOTS_DIR, "average_distribution.png"))

# -------------------------
# Pairplot (sampled) - safe try/except
# -------------------------
try:
    pair_sample = df.sample(n=min(300, df.shape[0]), random_state=1)
    pairplot = sns.pairplot(pair_sample[["Math","Science","English","Attendance","Average"]], corner=True)
    pairplot.fig.suptitle("Pairplot (sample of dataset)", y=1.02)
    pairplot.savefig(os.path.join(PLOTS_DIR, "pairplot_sample.png"))
    plt.close()
except Exception as e:
    print("Pairplot skipped (too slow or failed):", e)

# -------------------------
# Scatter with regression (regplot)
# -------------------------
plt.figure(figsize=(6,4))
sns.regplot(data=df, x="Math", y="English", scatter_kws={"s":8}, line_kws={"color":"red"})
plt.title("Math vs English (regression line)")
save_fig(os.path.join(PLOTS_DIR, "math_vs_english_regression.png"))

plt.figure(figsize=(6,4))
sns.regplot(data=df, x="Science", y="Average", scatter_kws={"s":8}, line_kws={"color":"red"})
plt.title("Science vs Average (regression line)")
save_fig(os.path.join(PLOTS_DIR, "science_vs_average_regression.png"))

# -------------------------
# Regression: predict English from Math + Science
# -------------------------
print("\nRunning regression: predict English from Math & Science")
features_reg = ["Math", "Science"]
Xr_df = df[features_reg].dropna()
Xr = Xr_df.values
yr = df.loc[Xr_df.index, "English"].values  # align indices safely

r2_reg, mse_reg = None, None
lr = None

if Xr.shape[0] < 10:
    print("Not enough rows for regression; skipping.")
else:
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
    lr = LinearRegression()
    lr.fit(Xr_train, yr_train)
    yr_pred = lr.predict(Xr_test)

    r2_reg = r2_score(yr_test, yr_pred)
    mse_reg = mean_squared_error(yr_test, yr_pred)
    print(f"Linear Regression (predict English from Math & Science): R^2 = {r2_reg:.4f}, MSE = {mse_reg:.4f}")
    print("Regression coefficients:", dict(zip(features_reg, lr.coef_.round(3))))
    print("Regression intercept:", round(lr.intercept_,3))

    plt.figure(figsize=(6,6))
    plt.scatter(yr_test, yr_pred, s=8, alpha=0.6)
    min_val = min(yr_test.min(), yr_pred.min())
    max_val = max(yr_test.max(), yr_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True English")
    plt.ylabel("Predicted English")
    plt.title("Regression: True vs Predicted English (test set)")
    save_fig(os.path.join(PLOTS_DIR, "regression_english_true_vs_pred.png"))

# -------------------------
# Classification: predict Grade
# -------------------------
print("\nRunning classification: predict Grade")
present_grades = sorted(df["Grade"].dropna().unique(), key=lambda x: str(x))
classification_report_text = "Not run"
if len(present_grades) < 2:
    print("Not enough grade classes for classification. Skipping classification.")
else:
    grade_map = {g: i for i, g in enumerate(present_grades)}
    df["Grade_label"] = df["Grade"].map(grade_map)

    Xc_df = df[["Math","Science","English","Attendance"]].dropna()
    yc = df.loc[Xc_df.index, "Grade_label"].values

    if len(np.unique(yc)) < 2 or Xc_df.shape[0] < 20:
        print("Not enough labeled rows for reliable classification. Skipping classification.")
    else:
        stratify_param = yc if len(np.unique(yc)) > 1 else None
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_df.values, yc, test_size=0.25, random_state=42, stratify=stratify_param)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(Xc_train, yc_train)
        yc_pred = clf.predict(Xc_test)

        classification_report_text = classification_report(yc_test, yc_pred, target_names=[str(g) for g in present_grades], zero_division=0)
        print("\nClassification report (Grade prediction):")
        print(classification_report_text)

        # Confusion matrix
        cm = confusion_matrix(yc_test, yc_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=present_grades, yticklabels=present_grades, cmap="Blues")
        plt.xlabel("Predicted Grade")
        plt.ylabel("True Grade")
        plt.title("Confusion Matrix: Grade Prediction")
        save_fig(os.path.join(PLOTS_DIR, "grade_confusion_matrix.png"))

        # Save a sample of test results
        n_sample = min(200, Xc_test.shape[0])
        sample_idx = np.random.choice(range(Xc_test.shape[0]), size=n_sample, replace=False)
        results_df = pd.DataFrame({
            "Math": Xc_test[sample_idx, 0],
            "Science": Xc_test[sample_idx, 1],
            "English": Xc_test[sample_idx, 2],
            "Attendance": Xc_test[sample_idx, 3],
            "True_Grade": [present_grades[i] for i in yc_test[sample_idx]],
            "Predicted_Grade": [present_grades[i] for i in yc_pred[sample_idx]]
        })
        results_df.to_csv(os.path.join(OUT_DIR, "grade_predictions_sample.csv"), index=False)

# -------------------------
# Save model summary + outputs
# -------------------------
summary = {
    "regression_R2_english_from_math_science": r2_reg,
    "regression_MSE_english_from_math_science": mse_reg,
    "classification_report": classification_report_text
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)

# -------------------------
# EXTRA VERBOSE OUTPUT FOR DEMO (prints to console)
# -------------------------
print("\n--- VERBOSE DEMO OUTPUT (extra prints) ---\n")

# 1) Full head and tail
print("Top 10 rows:")
print(df.head(10).to_string(index=False))
print("\nBottom 5 rows:")
print(df.tail(5).to_string(index=False))

# 2) More descriptive stats (including percentiles)
print("\nDescriptive stats (extended):")
print(df[["Math","Science","English","Attendance","Average"]].describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())

# 3) Value counts for Grades
print("\nGrade counts:")
print(df["Grade"].value_counts().sort_index().to_string())

# 4) Regression full details (if run)
if lr is not None:
    print("\nRegression model: English ~ Math + Science")
    print(" Coefficients:", dict(zip(features_reg, lr.coef_.round(4))))
    print(" Intercept:", round(lr.intercept_,4))
    print(" R2 (test):", round(r2_reg,4))
    print(" MSE (test):", round(mse_reg,4))

# 5) Classification report (if run)
if classification_report_text != "Not run":
    print("\nClassification report (full):")
    print(classification_report_text)

# 6) Show a small sample of predictions CSV (if exists)
pred_csv = os.path.join(OUT_DIR, "grade_predictions_sample.csv")
if os.path.exists(pred_csv):
    print("\nSample of saved grade_predictions_sample.csv:")
    try:
        sample_preds = pd.read_csv(pred_csv)
        print(sample_preds.head(10).to_string(index=False))
    except Exception as e:
        print("Could not read sample CSV:", e)

# 7) List saved plot files (first 30)
print("\nSaved plot files (first 30):")
plots = []
for root, _, files in os.walk(PLOTS_DIR):
    for f in files:
        plots.append(os.path.join(root, f))
plots = sorted(plots)
for p in plots[:30]:
    print(" -", p)
if len(plots) > 30:
    print(f" ... (+{len(plots)-30} more)")

print("\n--- END OF VERBOSE DEMO OUTPUT ---\n")

print("Saved outputs to:", OUT_DIR)
print("Done — all plots saved under:", PLOTS_DIR)
