"""
Model evaluation utilities for EnerCast.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze prediction errors."""
    errors = y_true - y_pred
    return {
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "max_error": np.max(np.abs(errors)),
        "percentile_90": np.percentile(np.abs(errors), 90)
    }


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Perform cross-validation on the model."""
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return {
        "cv_scores": -scores,
        "cv_mean": -np.mean(scores),
        "cv_std": np.std(scores)
    }
