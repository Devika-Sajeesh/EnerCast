"""
Data preprocessing utilities for EnerCast.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    return df


def merge_datasets(energy_df: pd.DataFrame, emissions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge energy consumption and emissions datasets."""
    # Implement merging logic based on common keys
    pass


def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns using min-max scaling."""
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df
