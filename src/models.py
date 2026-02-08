"""
Machine Learning Models for EnerCast
====================================

This module contains ML models for CO2 emissions prediction.
Starting with Linear Regression as a diagnostic benchmark before
exploring non-linear models.

Design Philosophy:
-----------------
1. Time-based train/test split (no data leakage from future)
2. Feature safety enforced via filter_safe_columns()
3. Proper scaling for linear models
4. Compare against rule-based baseline to measure real improvement
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_safety import filter_safe_columns
from src.baseline import predict_baseline, calculate_baseline_error


def time_based_split(df: pd.DataFrame, split_year: int = 2015) -> tuple:
    """
    Split data by time to prevent information leakage.
    
    Why time-based split?
    ---------------------
    Random splits leak future information. A model trained on 2020 data
    shouldn't be evaluated on 2010 data - that's not how forecasting works.
    
    Args:
        df: DataFrame with 'year' column
        split_year: Cutoff year (inclusive for training)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train = df[df['year'] <= split_year].copy()
    test = df[df['year'] > split_year].copy()
    return train, test


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for ML training.
    
    Process:
    --------
    1. Use only safe columns (no leakage-prone features)
    2. Drop identifiers (country, year) - they're not features
    3. Handle missing values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    # Get only safe features (excludes leakage columns automatically)
    X = filter_safe_columns(df, include_target=False)
    
    # Drop identifiers - these are not predictive features
    X = X.drop(columns=['country', 'year'], errors='ignore')
    
    # Target variable
    y = df['co2_kilotons']
    
    return X, y


def train_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Train and evaluate a Linear Regression model.
    
    Why Linear Regression First?
    ----------------------------
    It provides a transparent benchmark to see whether linear relationships
    in energy consumption already explain emissions before moving to 
    non-linear models.
    
    Args:
        train_df: Training data
        test_df: Test data
        
    Returns:
        Dict with model, scaler, metrics, and predictions
    """
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    # Get common columns (some features may be missing in train or test)
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    common_cols.sort()  # Consistent ordering
    
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Handle missing values - fill with 0 for energy consumption
    # (missing usually means "no consumption of this fuel type")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Drop rows where target is missing
    valid_train = ~y_train.isna()
    valid_test = ~y_test.isna()
    
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]
    
    # Scale features (MANDATORY for Linear Regression)
    # Energy features have wildly different scales (TWh vs percentages)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': len(common_cols),
    }
    
    # Feature importance (coefficients in scaled space)
    feature_importance = pd.DataFrame({
        'feature': common_cols,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': feature_importance,
        'feature_names': common_cols,
    }


def run_linear_regression_pipeline(df: pd.DataFrame, split_year: int = 2015) -> dict:
    """
    Run the complete Linear Regression experiment pipeline.
    
    This is a DIAGNOSTIC experiment, not final model selection.
    We compare against the rule-based baseline to understand:
    - Does ML add value over transparent physics?
    - Where does ML succeed/fail?
    
    Args:
        df: Full dataset
        split_year: Year to split train/test
        
    Returns:
        Dict with all results for analysis
    """
    print("=" * 60)
    print("LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    # Time-based split
    train_df, test_df = time_based_split(df, split_year)
    
    print(f"\nData Split:")
    print(f"  Train: {train_df['year'].min()}–{train_df['year'].max()} ({len(train_df):,} rows)")
    print(f"  Test:  {test_df['year'].min()}–{test_df['year'].max()} ({len(test_df):,} rows)")
    
    # Train and evaluate model
    print("\nTraining Linear Regression...")
    lr_result = train_linear_regression(train_df, test_df)
    
    print(f"\nFeatures Used: {lr_result['metrics']['n_features']}")
    
    print("\n" + "-" * 40)
    print("ML MODEL PERFORMANCE (Test Set)")
    print("-" * 40)
    for key in ['mae', 'rmse', 'r2']:
        print(f"  {key.upper()}: {lr_result['metrics'][key]:.3f}")
    
    # Calculate baseline on test set for comparison
    print("\n" + "-" * 40)
    print("BASELINE COMPARISON (Test Set)")
    print("-" * 40)
    
    test_with_baseline = predict_baseline(test_df)
    baseline_metrics = calculate_baseline_error(test_with_baseline)
    
    print(f"  Baseline MAE:  {baseline_metrics['mae_kilotons']:.3f} kilotons")
    print(f"  Baseline RMSE: {baseline_metrics['rmse_kilotons']:.3f} kilotons")
    print(f"  Baseline R²:   {baseline_metrics['r2']:.4f}")
    
    # Head-to-head comparison
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    
    ml_mae = lr_result['metrics']['mae']
    bl_mae = baseline_metrics['mae_kilotons']
    ml_rmse = lr_result['metrics']['rmse']
    bl_rmse = baseline_metrics['rmse_kilotons']
    ml_r2 = lr_result['metrics']['r2']
    bl_r2 = baseline_metrics['r2']
    
    mae_improvement = (bl_mae - ml_mae) / bl_mae * 100
    rmse_improvement = (bl_rmse - ml_rmse) / bl_rmse * 100
    
    print(f"\n  {'Metric':<10} {'Baseline':>12} {'ML':>12} {'Improvement':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'MAE':<10} {bl_mae:>12.2f} {ml_mae:>12.2f} {mae_improvement:>+11.2f}%")
    print(f"  {'RMSE':<10} {bl_rmse:>12.2f} {ml_rmse:>12.2f} {rmse_improvement:>+11.2f}%")
    print(f"  {'R²':<10} {bl_r2:>12.4f} {ml_r2:>12.4f} {'':>12}")
    
    # Verdict
    print("\n" + "=" * 60)
    if ml_mae < bl_mae:
        print("✓ ML BEATS BASELINE")
        print(f"  Linear Regression reduces MAE by {mae_improvement:.1f}%")
    else:
        print("✗ BASELINE WINS")
        print(f"  Linear Regression performs {-mae_improvement:.1f}% worse on MAE")
    print("=" * 60)
    
    # Top features
    print("\nTop 10 Most Important Features (by coefficient magnitude):")
    print("-" * 40)
    top_features = lr_result['feature_importance'].head(10)
    for i, row in top_features.iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {sign} {row['feature']:<45} {row['coefficient']:>10.4f}")
    
    return {
        'lr_result': lr_result,
        'baseline_metrics': baseline_metrics,
        'train_df': train_df,
        'test_df': test_with_baseline,
        'comparison': {
            'ml_mae': ml_mae,
            'baseline_mae': bl_mae,
            'mae_improvement_pct': mae_improvement,
            'ml_rmse': ml_rmse,
            'baseline_rmse': bl_rmse,
            'rmse_improvement_pct': rmse_improvement,
            'ml_r2': ml_r2,
            'baseline_r2': bl_r2,
            'ml_wins': ml_mae < bl_mae,
        }
    }


if __name__ == "__main__":
    from pathlib import Path
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "merged_energy_emissions.csv"
    
    if data_path.exists():
        print(f"Loading data from: {data_path}\n")
        df = pd.read_csv(data_path)
        
        # Run the experiment
        results = run_linear_regression_pipeline(df, split_year=2015)
        
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_processing_pipeline.py first.")
