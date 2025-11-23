📦 E-commerce Revenue Prediction (ML + Streamlit App)

A complete end-to-end machine learning project that predicts E-commerce Revenue using advanced feature engineering, data cleaning, model comparison, and an interactive Streamlit web application.

This project performs full-stack ML workflow:
data analysis → visualization → feature engineering → model training → evaluation → deployment.

🚀 Key Highlights

Complete ML pipeline (EDA → Cleaning → FE → Scaling → Modeling)

Multiple algorithms tested:
Linear Regression, Decision Tree, Random Forest

Feature Engineering includes:
Revenue per Unit, CTR Impact, CPC Efficiency, ROI, Discount Effect

Best model saved and used in production (RandomForest)

Clean visualizations: histograms, boxplots, smoothed trends, feature importance plots

Fully functional Streamlit web app for live predictions

Accepts user input (Units, Discount, Clicks, CTR, CPC, Region, Category)

Auto-computes realistic revenue using category medians

Outputs final predicted revenue + downloadable CSV

🧠 Machine Learning Workflow (Short & Clear)
1️⃣ Data Cleaning & Processing

Converted date columns

Removed impossible values

Removed duplicates

Fixed bad CTR / Impressions cases

Identified numeric & categorical columns

2️⃣ Feature Engineering

Created business-driven features:

Revenue_per_Unit

CTR_Impact

CPC_Efficiency

ROI

Discount_Effect

3️⃣ Encoding & Scaling

One-hot encoding

StandardScaler (saved as .pkl)

4️⃣ Model Training

Trained three models and compared:

Linear Regression

Decision Tree

Random Forest

Random Forest gave the best RMSE and R² → selected as final model.

Saved using:

RandomForest_model.pkl
standard_scaler.pkl


(kept in Drive due to large size)

🌐 Streamlit App (Short Explanation)

The app.py file builds an interactive web interface where users can:

Adjust inputs (Units, Discount, Clicks, CTR, CPC, Category, Region…)

Auto-load saved model + scaler

Perform real-time revenue predictions

See metrics instantly

Download prediction CSV

Uses cached loading for performance

This allows non-technical users to interact with the trained ML model.

▶️ How to Run the Streamlit App

Install dependencies:

pip install -r requirements.txt


Run Streamlit:

streamlit run app.py


Browser opens automatically at:

http://localhost:8501

📁 Project Structure
Ecommerce_retail_sales_revenue_prediction/
│── app.py                  # Streamlit UI
│── model_training.py       # ML training script
│── ecommerce_sales.csv     # Dataset
│── standard_scaler.pkl     # Scaler (large → stored externally)
│── RandomForest_model.pkl  # Best model (large → stored externally)
│── requirements.txt

🔗 Model Files (Download)

(Large files excluded from GitHub)
Provide link here:

RandomForest_model.pkl → [Google Drive Link]
standard_scaler.pkl → [Google Drive Link]

📊 Use Cases

Forecasting daily revenue

Advertising budget optimization

Pricing strategy

Sales performance tracking

E-commerce dashboard integration
