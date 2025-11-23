📦 E-commerce Revenue Prediction (ML + Streamlit App)

A complete end-to-end Machine Learning project that predicts E-commerce Revenue using real data, feature engineering, model comparison, and a fully functional Streamlit web app for live predictions.

This project performs:
Data Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Deployment

🚀 Key Highlights

Full ML Pipeline (EDA → FE → Modeling → Deployment)

Three models compared: Linear Regression, Decision Tree, Random Forest

Strong Feature Engineering (CTR Impact, ROI, CPC Efficiency, Discount Effect, etc.)

Random Forest selected as the best final model

Scaler + model saved for production

Streamlit UI for interactive predictions

Realistic input controls and downloadable results

🧠 ML Workflow Overview
1️⃣ Data Cleaning

Fixed date formats

Removed impossible values (CTR, clicks, impressions)

Removed duplicates

Converted data types safely

2️⃣ Exploratory Data Analysis

Revenue distributions

Boxplots for outlier understanding

Scatter plots: Units vs Revenue, Clicks vs Revenue

7-day smoothed revenue trend

Category-wise price-per-unit patterns

3️⃣ Feature Engineering

Created new business-impact features:

Revenue_per_Unit

CTR_Impact

CPC_Efficiency

ROI

Discount_Effect

4️⃣ Encoding & Scaling

One-Hot Encoding for Category & Region

StandardScaler for numeric data

Saved scaler as:

standard_scaler.pkl

5️⃣ Model Training & Selection

Trained 3 models:

Model	Metrics (MAE / RMSE / R²)
Linear Regression	Baseline
Decision Tree	Better but unstable
Random Forest	⭐ Best performing model

Saved final model:

RandomForest_model.pkl

💾 How to Create the Pickle Files (Model + Scaler)

You do not need to download anything.
Just run the training script, and it will automatically generate the files.

Step 1 — Run the training script
python model_training.py

Step 2 — This script will automatically create:

✔ RandomForest_model.pkl
✔ standard_scaler.pkl

Both will be saved in the project folder.

Step 3 — These files are then used by the Streamlit app.
🌐 How the Streamlit App Works

The Streamlit app (app.py) loads:

The trained .pkl model

The saved scaler

Median category-level metrics

User input from sidebar

It performs:

Feature engineering on the fly

Scaling with the saved scaler

Prediction using the saved model

Displays revenue prediction

Allows CSV download for the predicted row

Everything runs locally, nothing online.

▶️ How to Run the Streamlit App

After training your model:

Install required packages:

pip install -r requirements.txt


Run Streamlit:

streamlit run app.py


The interface will open at:

http://localhost:8501

📁 Project Structure
Ecommerce_retail_sales_revenue_prediction/
│── app.py                  # Streamlit App
│── model_training.py       # ML model training script
│── ecommerce_sales.csv     # Dataset
│── RandomForest_model.pkl  # Generated after training
│── standard_scaler.pkl     # Generated after training
│── requirements.txt
└── README.md

🎯 What This Project Demonstrates

Strong understanding of ML workflow

Practical EDA & visualization

Business-driven feature engineering

Model selection with metrics

Streamlit deployment skills

Handling real-world e-commerce data
