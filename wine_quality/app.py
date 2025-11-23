import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from wine_model.data_utils import load_data, basic_info
from wine_model.feature_engineering import engineer_features
from wine_model.model import WineQualityPredictor
from wine_model.training import (
    safe_train_test_split,
    choose_cv_for_grid,
    run_grid_search_rf,
    run_grid_search_svm,
    evaluate_model,
    compute_metrics
)

st.set_page_config(page_title="Wine Quality Trainer (modular)", layout="wide")
st.title("🍷 Wine Quality Trainer — Modular Streamlit App")

# --- Sidebar: Data selection ---
st.sidebar.header("Data")
use_server_file = st.sidebar.checkbox("Use server file (data/WineQT.csv)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])
example_data = st.sidebar.checkbox("Use small built-in example", value=False)

EXAMPLE_CSV = """fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0
7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,1
7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,2
11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,3
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4
7.4,0.66,0.0,1.8,0.075,13.0,40.0,0.9978,3.51,0.56,9.4,5,5
"""

df = None
if example_data:
    df = pd.read_csv(StringIO(EXAMPLE_CSV))
else:
    if use_server_file:
        try:
            df = load_data("data/WineQT.csv")
            st.sidebar.success("Loaded data/WineQT.csv")
        except Exception as e:
            st.sidebar.error(f"Could not load server file: {e}")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("Loaded uploaded file")
        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")

if df is None:
    st.info("Provide a dataset: upload a CSV, enable server file, or use example data.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10))
st.write("Shape:", df.shape)

# --- Sidebar: preprocessing options ---
st.sidebar.header("Preprocessing")
drop_id = st.sidebar.checkbox("Drop 'Id' column if present", value=True)
target_col = st.sidebar.text_input("Target column", value="quality")
add_engineered = st.sidebar.checkbox("Add engineered features", value=True)

if drop_id and 'Id' in df.columns:
    df = df.drop(columns=['Id'])

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")
    st.stop()

# handle missing values (simple demo behaviour)
if df.isnull().sum().sum() > 0:
    st.warning("Dataset contains missing values. For this demo rows with NA will be dropped.")
    if st.sidebar.button("Drop NA rows"):
        df = df.dropna()
        st.experimental_rerun()

X = df.drop(columns=[target_col])
y = df[target_col]

st.subheader("Target distribution")
st.bar_chart(y.value_counts().sort_index())

# feature engineering
if add_engineered:
    X = engineer_features(X)
    st.write("Features after engineering:", X.columns.tolist())

# --- Training controls ---
st.sidebar.header("Training")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42)
use_grid = st.sidebar.checkbox("Use GridSearchCV (both models)", value=True)

# Grid param options (small defaults for teaching)
st.sidebar.subheader("Grid options (small default)")
rf_n_estimators = st.sidebar.multiselect("RF n_estimators", [50, 100, 200], default=[100])
rf_max_depth = st.sidebar.multiselect("RF max_depth", [5, 10, None], default=[None])
rf_min_samples_split = st.sidebar.multiselect("RF min_samples_split", [2, 5], default=[2])
rf_min_samples_leaf = st.sidebar.multiselect("RF min_samples_leaf", [1, 2], default=[1])

svm_C = st.sidebar.multiselect("SVM C", [0.1, 1, 10], default=[1])
svm_kernel = st.sidebar.multiselect("SVM kernel", ["rbf", "linear"], default=["rbf"])

cv_max = st.sidebar.slider("Max CV folds for GridSearch", 2, 10, 5)

# Train button
if st.button("Train models"):
    # safe split (tries stratify where possible)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_size, random_state=random_state)
    st.write("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    st.write("Train target counts:", y_train.value_counts().to_dict())

    # decide cv for grid
    chosen_cv = choose_cv_for_grid(y_train, max_cv=cv_max)
    if use_grid and chosen_cv is None:
        st.warning("GridSearchCV not possible due to class counts; training without grid search.")
        do_grid = False
    else:
        do_grid = use_grid

    # instantiate predictor and prepare scaled arrays
    predictor = WineQualityPredictor(random_state=random_state)
    scaler = predictor.scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = None
    svm_model = None

    # RF Grid or fit
    if do_grid:
        rf_param_grid = {
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_split": rf_min_samples_split,
            "min_samples_leaf": rf_min_samples_leaf
        }
        st.info(f"Running RF GridSearch with cv={chosen_cv} ...")
        rf_gs = run_grid_search_rf(X_train, y_train, rf_param_grid, cv=chosen_cv)
        rf_model = rf_gs.best_estimator_
        st.write("RF best params:", rf_gs.best_params_, "best score:", rf_gs.best_score_)
    else:
        st.info("Fitting RF without GridSearch ...")
        rf_model = predictor.rf
        rf_model.fit(X_train, y_train)

    # SVM Grid or fit (on scaled data)
    if do_grid:
        svm_param_grid = {"C": svm_C, "kernel": svm_kernel}
        st.info(f"Running SVM GridSearch with cv={chosen_cv} ...")
        svm_gs = run_grid_search_svm(X_train_scaled, y_train, svm_param_grid, cv=chosen_cv)
        svm_model = svm_gs.best_estimator_
        st.write("SVM best params:", svm_gs.best_params_, "best score:", svm_gs.best_score_)
    else:
        st.info("Fitting SVM without GridSearch ...")
        svm_model = predictor.svm
        svm_model.fit(X_train_scaled, y_train)

    # Evaluate
    st.subheader("Evaluation results")

    # RF evaluate
    rf_eval = evaluate_model(rf_model, X_test, X_test_scaled, y_test, is_scaled=False)
    st.markdown("**Random Forest**")
    st.json(rf_eval['metrics'])
    st.text("Classification report:")
    st.text(rf_eval['report'])

    # plot confusion matrix
    labels_sorted = sorted(list(np.unique(y_test)))
    cm_rf = rf_eval['confusion_matrix']
    fig, ax = plt.subplots()
    im = ax.imshow(cm_rf, cmap=plt.cm.Blues)
    ax.set_title("RF Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels([str(l) for l in labels_sorted])
    ax.set_yticklabels([str(l) for l in labels_sorted])
    for i in range(cm_rf.shape[0]):
        for j in range(cm_rf.shape[1]):
            ax.text(j, i, format(cm_rf[i, j], 'd'), ha="center", va="center",
                    color="white" if cm_rf[i, j] > cm_rf.max()/2 else "black")
    st.pyplot(fig)

    # SVM evaluate
    svm_eval = evaluate_model(svm_model, X_test, X_test_scaled, y_test, is_scaled=True)
    st.markdown("**SVM**")
    st.json(svm_eval['metrics'])
    st.text("Classification report:")
    st.text(svm_eval['report'])

    cm_svm = svm_eval['confusion_matrix']
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(cm_svm, cmap=plt.cm.Blues)
    ax2.set_title("SVM Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_xticks(range(len(labels_sorted)))
    ax2.set_yticks(range(len(labels_sorted)))
    ax2.set_xticklabels([str(l) for l in labels_sorted])
    ax2.set_yticklabels([str(l) for l in labels_sorted])
    for i in range(cm_svm.shape[0]):
        for j in range(cm_svm.shape[1]):
            ax2.text(j, i, format(cm_svm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm_svm[i, j] > cm_svm.max()/2 else "black")
    st.pyplot(fig2)

    # feature importances (RF)
    if hasattr(rf_model, 'feature_importances_'):
        st.subheader("Random Forest feature importances")
        fi = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(fi.head(15))

    # store models + scaler + column order in session state
    st.session_state['rf_model'] = rf_model
    st.session_state['svm_model'] = svm_model
    st.session_state['scaler'] = scaler
    st.session_state['X_columns'] = X_train.columns.tolist()
    st.success("Training complete and models stored in session_state")

# --- Single-sample prediction UI ---
st.sidebar.header("Single-sample prediction")
if 'rf_model' in st.session_state and 'X_columns' in st.session_state:
    rf_model = st.session_state['rf_model']
    svm_model = st.session_state['svm_model']
    scaler = st.session_state['scaler']
    cols = st.session_state['X_columns']

    st.sidebar.info("Build a sample with sliders/inputs and click Predict")
    sample = {}
    for col in cols:
        if col in df.columns and df[col].dtype.kind in 'if':
            mn = float(df[col].min())
            mx = float(df[col].max())
            md = float(df[col].median())
            step = (mx - mn) / 100 if mx != mn else 0.1
            sample[col] = st.sidebar.slider(col, min_value=mn, max_value=mx, value=md, step=step)
        else:
            sample[col] = st.sidebar.text_input(col, value=str(df[col].iloc[0]) if col in df.columns else "")

    if st.sidebar.button("Predict this sample"):
        sample_df = pd.DataFrame([sample]).reindex(columns=cols)
        rf_pred = rf_model.predict(sample_df)[0]
        rf_proba = None
        try:
            rf_proba = rf_model.predict_proba(sample_df)
        except Exception:
            rf_proba = None

        sample_scaled = scaler.transform(sample_df)
        svm_pred = svm_model.predict(sample_scaled)[0]
        svm_proba = None
        try:
            svm_proba = svm_model.predict_proba(sample_scaled)
        except Exception:
            svm_proba = None

        st.write("Random Forest prediction:", rf_pred)
        if rf_proba is not None:
            st.write("RF probabilities (columns = classes):")
            st.dataframe(pd.DataFrame(rf_proba, columns=rf_model.classes_).round(4))

        st.write("SVM prediction:", svm_pred)
        if svm_proba is not None:
            st.write("SVM probabilities (columns = classes):")
            st.dataframe(pd.DataFrame(svm_proba, columns=svm_model.classes_).round(4))
else:
    st.sidebar.info("Train models to enable single-sample prediction")

st.write("---")
st.caption("Modular project: wine_model contains data, feature-engineering, model and training utilities. GridSearchCV runs only when class counts allow it.")