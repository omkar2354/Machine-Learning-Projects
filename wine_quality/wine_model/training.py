import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def safe_train_test_split(X: pd.DataFrame, y: pd.Series,
                          test_size: float = 0.2, random_state: int = 42):
    """
    Attempt stratified split if every class has >=2 samples and more than 1 class
    otherwise fall back to non-stratified split.
    """
    counts = y.value_counts()
    if (counts.min() >= 2) and (y.nunique() > 1):
        stratify = y
    else:
        stratify = None
    return train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=stratify)

def choose_cv_for_grid(y_train: pd.Series, max_cv: int = 5) -> Optional[int]:
    """
    Determine a safe cv value for GridSearchCV. Returns None if not feasible.
    """
    counts = y_train.value_counts()
    if len(counts) <= 1:
        return None
    min_count = counts.min()
    cv = min(max_cv, min_count)
    if cv < 2:
        return None
    return int(cv)

def run_grid_search_rf(X_train, y_train, param_grid: dict, cv: int):
    """
    Run GridSearchCV for RandomForest. Returns fitted GridSearchCV object.
    """
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs

def run_grid_search_svm(X_train_scaled, y_train, param_grid: dict, cv: int):
    """
    Run GridSearchCV for SVM (on scaled features). Returns fitted GridSearchCV object.
    """
    svm = SVC(probability=True, random_state=42)
    gs = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    return gs

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute basic classification metrics and (optionally) multiclass roc-auc (ovr).
    """
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    res['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    res['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    if y_proba is not None:
        try:
            res['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            res['roc_auc_ovr'] = None
    else:
        res['roc_auc_ovr'] = None
    return res

def evaluate_model(model, X_test, X_test_scaled, y_test, is_scaled=False):
    """
    Evaluate either sklearn model (RF expects original X, SVM expects scaled X).
    Returns dict with metrics + confusion matrix + report.
    """
    if is_scaled:
        y_pred = model.predict(X_test_scaled)
        try:
            y_proba = model.predict_proba(X_test_scaled)
        except Exception:
            y_proba = None
    else:
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }