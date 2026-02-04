"""
Machine learning models for EnerCast.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class EmissionPredictor:
    """Predict emissions using machine learning models."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the predictor with specified model type."""
        self.model_type = model_type
        self.model = self._create_model()
    
    def _create_model(self):
        """Create the specified model."""
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "linear": LinearRegression()
        }
        return models.get(self.model_type, RandomForestRegressor())
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (if available)."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
