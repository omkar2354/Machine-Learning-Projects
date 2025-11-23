from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

class WineQualityPredictor:
    """
    Encapsulates RF and SVM models and a scaler. Training may be done with or
    without GridSearchCV (use run_grid=True to run grid search provided param grids).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)
        self.rf = RandomForestClassifier(random_state=self.random_state)
        self.svm = SVC(probability=True, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            rf_params: Optional[Dict] = None,
            svm_params: Optional[Dict] = None,
            use_grid: bool = False,
            cv: int = 3):
        """
        Fit RF and SVM. If use_grid True, perform GridSearchCV with provided param dicts.
        rf_params example: {'n_estimators':[50,100], 'max_depth':[5,None]}
        svm_params example: {'C':[0.1,1], 'kernel':['rbf']}
        cv: number of folds for GridSearch (must be safe given class counts).
        """
        # Fit scaler on raw features (used by SVM)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # Random Forest
        if use_grid and rf_params:
            gs_rf = GridSearchCV(self.rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
            gs_rf.fit(X_train, y_train)
            self.rf = gs_rf.best_estimator_
            rf_cv_best = gs_rf.best_score_
        else:
            self.rf.fit(X_train, y_train)
            rf_cv_best = None

        # SVM (on scaled features)
        if use_grid and svm_params:
            gs_svm = GridSearchCV(self.svm, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
            gs_svm.fit(X_train_scaled, y_train)
            self.svm = gs_svm.best_estimator_
            svm_cv_best = gs_svm.best_score_
        else:
            self.svm.fit(X_train_scaled, y_train)
            svm_cv_best = None

        self.is_trained = True
        return {'rf_cv_best': rf_cv_best, 'svm_cv_best': svm_cv_best}

    def predict_rf(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(X)

    def predict_svm(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.svm.predict(Xs)

    def predict_proba_rf(self, X: pd.DataFrame):
        try:
            return self.rf.predict_proba(X)
        except Exception:
            return None

    def predict_proba_svm(self, X: pd.DataFrame):
        try:
            Xs = self.scaler.transform(X)
            return self.svm.predict_proba(Xs)
        except Exception:
            return None

    def feature_importances(self, feature_names: List[str]):
        if hasattr(self.rf, 'feature_importances_'):
            fi = list(zip(feature_names, self.rf.feature_importances_))
            return sorted(fi, key=lambda x: x[1], reverse=True)
        return []