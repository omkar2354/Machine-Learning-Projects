import pandas as pd
import numpy as np

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add a compact set of engineered features.
    Keeps original columns, appends new ones where possible.
    """
    Xf = X.copy()

    if 'fixed acidity' in Xf.columns and 'volatile acidity' in Xf.columns:
        Xf['acid_ratio'] = Xf['fixed acidity'] / (Xf['volatile acidity'] + 1e-8)
        Xf['acidity_interaction'] = Xf['fixed acidity'] * Xf['volatile acidity']

    if 'free sulfur dioxide' in Xf.columns and 'total sulfur dioxide' in Xf.columns:
        Xf['so2_ratio'] = Xf['free sulfur dioxide'] / (Xf['total sulfur dioxide'] + 1e-8)

    if 'residual sugar' in Xf.columns and 'alcohol' in Xf.columns:
        Xf['sugar_alcohol_ratio'] = Xf['residual sugar'] / (Xf['alcohol'] + 1e-8)

    if 'sulphates' in Xf.columns and 'alcohol' in Xf.columns:
        Xf['sulphates_alcohol'] = Xf['sulphates'] * Xf['alcohol']

    # Replace any infinities and fill NA produced by operations
    Xf.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xf = Xf.fillna(0)

    return Xf