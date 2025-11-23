"""
model_training.py
- Safe: loads existing model/scaler if present
- Trains & saves only if model/scaler are missing (or if --force flag used)
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --------- Paths (use your Colab/VSCode paths) ----------
MODEL_PATH  = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/RandomForest_model.pkl"          # final model file
SCALER_PATH = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/standard_scaler.pkl"             # final scaler file
DATA_PATH   = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/ecommerce_sales.csv"             # put your CSV in project folder

# ---------- Helper: load data & minimal preprocessing ----------
def load_and_prep(path=DATA_PATH):
    df = pd.read_csv(path)
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format="%d-%m-%Y", errors='coerce')
    # recreate used engineered features (match exactly your notebook)
    df['Revenue_per_Unit'] = df['Revenue'] / df['Units_Sold']
    df['CTR_Impact'] = df['Clicks'] / (df['Impressions'] + 1)
    df['CPC_Efficiency'] = df['Ad_Spend'] / (df['Clicks'] + 1)
    df['ROI'] = (df['Revenue'] - df['Ad_Spend']) / (df['Ad_Spend'] + 1)
    df['Discount_Effect'] = df['Units_Sold'] * df['Discount_Applied']
    # one-hot for Category & Region (drop_first to match notebook)
    df = pd.get_dummies(df, columns=['Category','Region'], drop_first=True)
    # build feature set and target (use exact feature_cols used earlier)
    feature_cols = [
        'Units_Sold','Discount_Applied','Clicks','Impressions','Conversion_Rate',
        'Ad_CTR','Ad_CPC','Ad_Spend','Revenue_per_Unit','CTR_Impact','CPC_Efficiency',
        'ROI','Discount_Effect','Category_Clothing','Category_Electronics',
        'Category_Home Appliances','Category_Toys','Region_Europe','Region_North America'
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].copy()
    y = df['Revenue']
    return X, y

# ---------- Train function ----------
def train_and_save(X, y, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # scale numeric cols (list must match training)
    numeric_cols = [
        'Units_Sold','Discount_Applied','Clicks','Impressions','Conversion_Rate',
        'Ad_CTR','Ad_CPC','Ad_Spend','Revenue_per_Unit','CTR_Impact','CPC_Efficiency',
        'ROI','Discount_Effect'
    ]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # train (use smaller parameters if you want a lighter model for demo)
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # save both
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved -> {model_path}")
    print(f"Scaler saved -> {scaler_path}")

    # optional: print a quick test score
    preds = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error, r2_score
    print("MAE (test):", round(mean_absolute_error(y_test, preds), 4))
    print("R2  (test):", round(r2_score(y_test, preds), 4))

# ---------- Main logic: avoid double/training unless forced ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain and overwrite model/scaler")
    args = parser.parse_args()

    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)

    if model_exists and scaler_exists and not args.force:
        print("Model and scaler already exist. Skipping training.")
        print(f"Model path: {os.path.abspath(MODEL_PATH)}")
        print(f"Scaler path: {os.path.abspath(SCALER_PATH)}")
        print("If you want to retrain and overwrite, run: python model_training.py --force")
    else:
        print("Training model (this may take time)...")
        X, y = load_and_prep(DATA_PATH)
        train_and_save(X, y)
