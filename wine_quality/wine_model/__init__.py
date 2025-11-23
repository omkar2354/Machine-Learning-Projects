# make wine_model a package
from .data_utils import load_data, basic_info
from .feature_engineering import engineer_features
from .model import WineQualityPredictor
from .training import (
    safe_train_test_split,
    choose_cv_for_grid,
    run_grid_search_rf,
    run_grid_search_svm,
    evaluate_model,
    compute_metrics
)