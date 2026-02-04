"""
Scoring utilities for EnerCast.
"""

import numpy as np
import pandas as pd


class EmissionScorer:
    """Score and rank emission predictions."""
    
    def __init__(self, weights: dict = None):
        """Initialize scorer with custom weights."""
        self.weights = weights or {
            "accuracy": 0.4,
            "precision": 0.3,
            "coverage": 0.3
        }
    
    def calculate_score(self, metrics: dict) -> float:
        """Calculate weighted score from metrics."""
        score = 0.0
        for key, weight in self.weights.items():
            if key in metrics:
                score += weight * metrics[key]
        return score
    
    def rank_models(self, model_metrics: dict) -> list:
        """Rank models based on their scores."""
        scores = {
            model: self.calculate_score(metrics) 
            for model, metrics in model_metrics.items()
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def calculate_emission_score(predicted: float, actual: float) -> float:
    """Calculate emission prediction score."""
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0
    
    error_ratio = abs(predicted - actual) / actual
    return max(0, 1 - error_ratio)
