🍷 Wine Quality Prediction — Modular Machine Learning Project

This project predicts wine quality using two ML models — Random Forest and SVM — wrapped inside a clean, modular Python package (wine_model).
A full Streamlit app is included to train models, tune hyperparameters, visualize evaluation results, and make single-sample predictions interactively.

🚀 Project Highlights

Modular architecture (data_utils, feature_engineering, model, training)

Two ML algorithms: Random Forest & SVM

Optional GridSearchCV for hyperparameter tuning

Automatic feature engineering

Clean evaluation: metrics, confusion matrix, feature importance

Interactive Streamlit UI

Supports CSV upload or built-in dataset

Handles missing values, scaling, and CV safety checks

📁 Project Structure
wine_quality/
│
├── data/
│   └── WineQT.csv
│
├── wine_model/
│   ├── data_utils.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── training.py
│   └── __init__.py
│
└── app.py   ← Streamlit Application

🔧 How It Works
1️⃣ Load & Inspect Data

Load dataset (server file, upload, or example CSV)

Show preview, shape, and distribution

Optional: drop ID column

2️⃣ Preprocess

Drop NULL rows (only when needed)

Automatically engineer additional features

Split data safely (stratification if possible)

3️⃣ Train Models

Choose:

Normal training

GridSearchCV with adjustable CV folds

Models trained:

RandomForestClassifier

SVC with probability=True

4️⃣ Evaluate

Accuracy, precision, recall, F1, confusion matrix

Random Forest feature importance

Store models + scaler in session state

5️⃣ Predict

Build a custom sample in the sidebar

Predict using:

Random Forest

SVM (scaled)

Show probability outputs if available

🖥️ Running the Streamlit App

Inside the project directory:

streamlit run app.py

📌 Notes for Use

No pickle files are included to keep the repo lightweight

Streamlit automatically loads trained models stored in session state

GridSearch runs only when class distribution allows a safe CV value

The project is fully modular and beginner-friendly yet production-style
