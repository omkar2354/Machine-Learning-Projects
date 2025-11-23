# app.py — realistic demo (uses median price-per-unit by category, editable)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --------- Paths (use your Colab/VSCode paths) ----------
MODEL_PATH  = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/RandomForest_model.pkl"          # final model file
SCALER_PATH = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/standard_scaler.pkl"             # final scaler file
DATA_PATH   = "C:/Streamlit/Ecoomerce_retail_sales_revenue_prediction/ecommerce_sales.csv"  # your dataset

# ---------- helpers ----------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # reference df for realistic medians
    df_ref = pd.read_csv(DATA_PATH)
    df_ref['Revenue_per_Unit'] = df_ref['Revenue'] / df_ref['Units_Sold']
    medians = df_ref.groupby('Category')['Revenue_per_Unit'].median().to_dict()
    overall_median = df_ref['Revenue_per_Unit'].median()
    return model, scaler, medians, overall_median

model, scaler, median_price_by_cat, overall_median = load_resources()

st.title("E-commerce Revenue Prediction")

# ---------- input controls ----------
st.sidebar.header("Edit demo row (live)")
units = st.sidebar.number_input("Units Sold", min_value=1, max_value=5000, value=120, step=1)
discount = st.sidebar.slider("Discount Applied", 0.0, 0.9, 0.10, 0.01)
clicks = st.sidebar.number_input("Clicks", min_value=0, max_value=1000, value=30, step=1)
imps = st.sidebar.number_input("Impressions", min_value=0, max_value=10000, value=250, step=1)
conv = st.sidebar.slider("Conversion Rate", 0.0, 5.0, 0.12, 0.01)
ad_ctr = st.sidebar.slider("Ad CTR", 0.0, 1.0, 0.13, 0.01)
ad_cpc = st.sidebar.number_input("Ad CPC", min_value=0.0, value=1.20, step=0.01)
ad_spend = st.sidebar.number_input("Ad Spend", min_value=0.0, value=100.0, step=0.5)
category = st.sidebar.selectbox("Category", options=["Electronics","Clothing","Toys","Home Appliances"])
region = st.sidebar.selectbox("Region", options=["Asia","Europe","North America"])
use_median = st.sidebar.checkbox("Use median price-per-unit for this category (recommended)", value=True)

# allow manual override of price-per-unit
manual_price = st.sidebar.number_input("Manual price-per-unit (optional)", min_value=0.0, value=0.0, step=0.1)

# build row
demo = {
    "Units_Sold": units,
    "Discount_Applied": discount,
    "Clicks": clicks,
    "Impressions": imps,
    "Conversion_Rate": conv,
    "Ad_CTR": ad_ctr,
    "Ad_CPC": ad_cpc,
    "Ad_Spend": ad_spend,
    "Category": category,
    "Region": region
}

# compute realistic Revenue_per_Unit
if manual_price > 0.0:
    price_per_unit = manual_price
else:
    price_per_unit = median_price_by_cat.get(category, overall_median)
# show what price_per_unit is used
st.sidebar.markdown(f"**Price-per-unit used:** ₹{price_per_unit:.2f}")

# engineered features
demo["Revenue_per_Unit"] = price_per_unit
demo["CTR_Impact"] = demo["Clicks"] / (demo["Impressions"] + 1)
demo["CPC_Efficiency"] = demo["Ad_Spend"] / (demo["Clicks"] + 1)
demo["ROI"] = (demo["Revenue_per_Unit"] * demo["Units_Sold"] - demo["Ad_Spend"]) / (demo["Ad_Spend"] + 1)
demo["Discount_Effect"] = demo["Units_Sold"] * demo["Discount_Applied"]

# prepare dataframe and dummies (drop_first to match training)
row = pd.DataFrame([demo])
row = pd.get_dummies(row, columns=["Category","Region"], drop_first=True)

# ensure all expected features exist (same order used in training)
feature_cols = [
 'Units_Sold','Discount_Applied','Clicks','Impressions','Conversion_Rate',
 'Ad_CTR','Ad_CPC','Ad_Spend','Revenue_per_Unit','CTR_Impact',
 'CPC_Efficiency','ROI','Discount_Effect','Category_Clothing','Category_Electronics',
 'Category_Home Appliances','Category_Toys','Region_Europe','Region_North America'
]
for c in feature_cols:
    if c not in row.columns:
        row[c] = 0
X_row = row[feature_cols].copy()

# scale numeric cols then predict
numeric_cols = ['Units_Sold','Discount_Applied','Clicks','Impressions','Conversion_Rate','Ad_CTR','Ad_CPC','Ad_Spend','Revenue_per_Unit','CTR_Impact','CPC_Efficiency','ROI','Discount_Effect']
X_row[numeric_cols] = scaler.transform(X_row[numeric_cols])

pred = model.predict(X_row)[0]

# format & display
st.subheader("Prediction result (realistic)")
st.metric(label="Predicted Revenue (₹)", value=f"{pred:,.2f}")
st.write("Used inputs:")
st.json(demo)

# extra: show an example realistic actual revenue estimate = price_per_unit * units (for judges)
approx_actual = price_per_unit * units * (1 - discount)
st.caption(f"Approx. expected revenue (price_per_unit × units × (1-discount)) = ₹{approx_actual:,.2f}")

# allow download of the single-row prediction
out = row.copy()
out["Predicted_Revenue"] = pred
st.download_button("Download prediction CSV", out[["Units_Sold","Predicted_Revenue"]].to_csv(index=False).encode("utf-8"), file_name="single_prediction.csv")
