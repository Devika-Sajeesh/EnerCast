"""
Baseline CO₂ Emission Estimator for EnerCast
=============================================

A simple, transparent, rule-based estimator using global-average emission factors.
No machine learning. Just physics and chemistry.

Methodology:
-----------
1. Take fossil fuel consumption (coal, oil, gas) in TWh
2. Apply global-average emission factors (kg CO₂ per kWh)
3. Convert to kilotons for comparison with actual emissions

Emission Factors Source:
-----------------------
IPCC 2006 Guidelines for National Greenhouse Gas Inventories
Values are approximate global averages for electricity generation.
"""

import pandas as pd
import numpy as np


# =============================================================================
# EMISSION FACTORS (kg CO₂ per kWh of energy consumed)
# =============================================================================
# These are APPROXIMATE, CONSERVATIVE global averages.
# They are NOT country- or technology-specific.
#
# The baseline is intentionally coarse to highlight the limits of
# rule-based estimation vs. data-driven approaches.
#
# Sources: IPCC 2006 Guidelines, IEA, EPA estimates
# Actual values vary significantly by:
#   - Fuel quality (e.g., lignite vs anthracite coal)
#   - Combustion efficiency
#   - Power plant technology
#   - Country-specific fuel mix

EMISSION_FACTORS = {
    'coal': 0.82,    # kg CO₂ per kWh (hard coal average, conservative)
    'oil': 0.65,     # kg CO₂ per kWh (petroleum products average)
    'gas': 0.45,     # kg CO₂ per kWh (natural gas average)
}

# Unit conversion constants
TWH_TO_KWH = 1e9           # 1 TWh = 1 billion kWh
KG_TO_KILOTONS = 1e-6      # 1 million kg = 1 kiloton


def estimate_baseline_emissions(
    coal_twh: float,
    oil_twh: float, 
    gas_twh: float,
) -> float:
    """
    Estimate CO₂ emissions from fossil fuel consumption.
    
    Args:
        coal_twh: Coal consumption in TWh
        oil_twh: Oil consumption in TWh
        gas_twh: Natural gas consumption in TWh
        
    Returns:
        Estimated CO₂ emissions in kilotons
        
    Calculation:
        TWh → kWh → kg CO₂ → kilotons
        
        For each fuel:
        emissions_kg = consumption_TWh × 1e9 × emission_factor_kg_per_kwh
        emissions_kilotons = emissions_kg × 1e-6
        
        Simplified:
        emissions_kilotons = consumption_TWh × 1e9 × factor × 1e-6
                          = consumption_TWh × factor × 1e3
                          = consumption_TWh × factor × 1000
    """
    # Handle missing values
    coal_twh = 0.0 if pd.isna(coal_twh) else coal_twh
    oil_twh = 0.0 if pd.isna(oil_twh) else oil_twh
    gas_twh = 0.0 if pd.isna(gas_twh) else gas_twh
    
    # Calculate emissions for each fuel type
    # Formula: TWh × (kWh/TWh) × (kg CO₂/kWh) × (kilotons/kg)
    #        = TWh × 1e9 × factor × 1e-6
    #        = TWh × factor × 1000
    
    coal_emissions = coal_twh * EMISSION_FACTORS['coal'] * 1000
    oil_emissions = oil_twh * EMISSION_FACTORS['oil'] * 1000
    gas_emissions = gas_twh * EMISSION_FACTORS['gas'] * 1000
    
    total_emissions = coal_emissions + oil_emissions + gas_emissions
    
    return total_emissions


def predict_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline emission predictions to a DataFrame.
    
    Args:
        df: DataFrame with columns:
            - coal_consumption (TWh)
            - oil_consumption (TWh)
            - gas_consumption (TWh)
            
    Returns:
        DataFrame with new column 'baseline_co2_kilotons'
        
    Note:
        Uses row-wise apply() for clarity. For large-scale deployments,
        vectorized operations would be more efficient:
        
        df['baseline'] = (df['coal_consumption'].fillna(0) * 0.82 +
                          df['oil_consumption'].fillna(0) * 0.65 +
                          df['gas_consumption'].fillna(0) * 0.45) * 1000
    """
    df = df.copy()
    
    # Apply baseline estimation row by row (clarity > speed for this scale)
    df['baseline_co2_kilotons'] = df.apply(
        lambda row: estimate_baseline_emissions(
            coal_twh=row.get('coal_consumption', 0),
            oil_twh=row.get('oil_consumption', 0),
            gas_twh=row.get('gas_consumption', 0),
        ),
        axis=1
    )
    
    return df


def calculate_baseline_error(df: pd.DataFrame) -> dict:
    """
    Calculate error metrics for baseline predictions.
    
    Args:
        df: DataFrame with 'baseline_co2_kilotons' and 'co2_kilotons' columns
        
    Returns:
        Dict with error metrics (MAE, RMSE, MAPE, R²)
    """
    # Filter to rows with both actual and predicted values
    valid = df[['co2_kilotons', 'baseline_co2_kilotons']].dropna()
    
    if len(valid) == 0:
        return {'error': 'No valid rows for comparison'}
    
    actual = valid['co2_kilotons'].values
    predicted = valid['baseline_co2_kilotons'].values
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error (avoid division by zero)
    # Caveat: MAPE can be misleading for small actual values
    mask = actual > 100  # Exclude very small values where MAPE explodes
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
    
    # R² (coefficient of determination)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'n_samples': len(valid),
        'mae_kilotons': round(mae, 2),
        'rmse_kilotons': round(rmse, 2),
        'mape_percent': round(mape, 2),
        'r2': round(r2, 4),
    }


def run_baseline_pipeline(data_path: str) -> pd.DataFrame:
    """
    Run the complete baseline prediction pipeline.
    
    Args:
        data_path: Path to merged_energy_emissions.csv
        
    Returns:
        DataFrame with baseline predictions and error metrics printed
    """
    print("="*60)
    print("BASELINE CO₂ EMISSION ESTIMATOR")
    print("="*60)
    
    print("\nEmission Factors (kg CO₂ per kWh):")
    for fuel, factor in EMISSION_FACTORS.items():
        print(f"  {fuel.capitalize()}: {factor}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total rows: {len(df):,}")
    
    # Generate predictions
    print("\nGenerating baseline predictions...")
    df = predict_baseline(df)
    
    # Calculate errors
    print("\nCalculating error metrics...")
    metrics = calculate_baseline_error(df)
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # Run baseline pipeline
    data_path = Path(__file__).parent.parent / "data" / "processed" / "merged_energy_emissions.csv"
    
    if data_path.exists():
        result_df = run_baseline_pipeline(str(data_path))
        
        # Show sample predictions
        print("\nSample Predictions (first 10 with actual CO₂):")
        sample_cols = ['country', 'year', 'coal_consumption', 'oil_consumption', 
                       'gas_consumption', 'co2_kilotons', 'baseline_co2_kilotons']
        sample = result_df[sample_cols].dropna().head(10)
        print(sample.to_string(index=False))
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_processing_pipeline.py first.")
