"""
Feature Safety Configuration for EnerCast
==========================================

This module defines safe feature sets to prevent label leakage in ML models.
Leakage occurs when features contain information directly derived from the target variable.

CRITICAL: Before training any ML model, use SAFE_FEATURES or call get_safe_features()
"""

# =============================================================================
# DANGEROUS COLUMNS - DROP BEFORE TRAINING
# These columns are directly related to emissions and WILL cause leakage
# =============================================================================

LEAKAGE_COLUMNS = [
    # Direct emissions data
    'greenhouse_gas_emissions',    # Direct GHG metric - DEFINITE LEAKAGE
    'carbon_intensity_elec',       # CO2 per kWh electricity - DERIVED FROM EMISSIONS
    
    # Target columns (from CO2 emissions dataset)
    'co2_kilotons',               # TARGET VARIABLE - never use as feature!
    'co2_per_capita',             # Alternative target - exclude
]

# =============================================================================
# SUSPICIOUS COLUMNS - REVIEW CAREFULLY
# These may contain emission-derived information
# =============================================================================

SUSPICIOUS_COLUMNS = [
    # Fossil fuel columns that might encode emissions intensity
    'fossil_share_elec',           # Could correlate strongly with emissions
    'fossil_share_energy',         # Could correlate strongly with emissions
    'fossil_fuel_consumption',     # Direct fossil fuel use
    'fossil_energy_per_capita',    # Per capita fossil energy
]

# =============================================================================
# METADATA COLUMNS - NOT FEATURES
# These are identifiers, not predictive features
# =============================================================================

METADATA_COLUMNS = [
    'country',                     # Row identifier
    'year',                        # Time index
    'iso_code',                    # Country code
    'Region',                      # Geographic region (from emissions data)
]

# =============================================================================
# SAFE FEATURES - APPROVED FOR ML TRAINING
# Energy consumption metrics that don't directly encode emissions
# =============================================================================

SAFE_FEATURE_CATEGORIES = {
    'population_economy': [
        'population',
        'gdp',
    ],
    
    'biofuel': [
        'biofuel_consumption',
        'biofuel_cons_change_pct',
        'biofuel_cons_change_twh',
        'biofuel_cons_per_capita',
        'biofuel_elec_per_capita',
        'biofuel_electricity',
        'biofuel_share_elec',
        'biofuel_share_energy',
    ],
    
    'coal': [
        'coal_consumption',
        'coal_cons_change_pct',
        'coal_cons_change_twh',
        'coal_cons_per_capita',
        'coal_elec_per_capita',
        'coal_electricity',
        'coal_prod_change_pct',
        'coal_prod_change_twh',
        'coal_prod_per_capita',
        'coal_production',
        'coal_share_elec',
        'coal_share_energy',
    ],
    
    'gas': [
        'gas_consumption',
        'gas_cons_change_pct',
        'gas_cons_change_twh',
        'gas_elec_per_capita',
        'gas_electricity',
        'gas_energy_per_capita',
        'gas_prod_change_pct',
        'gas_prod_change_twh',
        'gas_prod_per_capita',
        'gas_production',
        'gas_share_elec',
        'gas_share_energy',
    ],
    
    'oil': [
        'oil_consumption',
        'oil_cons_change_pct',
        'oil_cons_change_twh',
        'oil_elec_per_capita',
        'oil_electricity',
        'oil_energy_per_capita',
        'oil_prod_change_pct',
        'oil_prod_change_twh',
        'oil_prod_per_capita',
        'oil_production',
        'oil_share_elec',
        'oil_share_energy',
    ],
    
    'electricity': [
        'electricity_demand',
        'electricity_generation',
        'electricity_share_energy',
        'per_capita_electricity',
        'net_elec_imports',
        'net_elec_imports_share_demand',
    ],
    
    'primary_energy': [
        'primary_energy_consumption',
        'energy_cons_change_pct',
        'energy_cons_change_twh',
        'energy_per_capita',
        'energy_per_gdp',
    ],
    
    'renewables': [
        'renewables_consumption',
        'renewables_cons_change_pct',
        'renewables_cons_change_twh',
        'renewables_elec_per_capita',
        'renewables_electricity',
        'renewables_energy_per_capita',
        'renewables_share_elec',
        'renewables_share_energy',
    ],
    
    'hydro': [
        'hydro_consumption',
        'hydro_cons_change_pct',
        'hydro_cons_change_twh',
        'hydro_elec_per_capita',
        'hydro_electricity',
        'hydro_energy_per_capita',
        'hydro_share_elec',
        'hydro_share_energy',
    ],
    
    'nuclear': [
        'nuclear_consumption',
        'nuclear_cons_change_pct',
        'nuclear_cons_change_twh',
        'nuclear_elec_per_capita',
        'nuclear_electricity',
        'nuclear_energy_per_capita',
        'nuclear_share_elec',
        'nuclear_share_energy',
    ],
    
    'solar': [
        'solar_consumption',
        'solar_cons_change_pct',
        'solar_cons_change_twh',
        'solar_elec_per_capita',
        'solar_electricity',
        'solar_energy_per_capita',
        'solar_share_elec',
        'solar_share_energy',
    ],
    
    'wind': [
        'wind_consumption',
        'wind_cons_change_pct',
        'wind_cons_change_twh',
        'wind_elec_per_capita',
        'wind_electricity',
        'wind_energy_per_capita',
        'wind_share_elec',
        'wind_share_energy',
    ],
    
    'low_carbon': [
        'low_carbon_consumption',
        'low_carbon_cons_change_pct',
        'low_carbon_cons_change_twh',
        'low_carbon_elec_per_capita',
        'low_carbon_electricity',
        'low_carbon_energy_per_capita',
        'low_carbon_share_elec',
        'low_carbon_share_energy',
    ],
    
    'other_renewables': [
        'other_renewable_consumption',
        'other_renewable_electricity',
        'other_renewable_exc_biofuel_electricity',
        'other_renewables_cons_change_pct',
        'other_renewables_cons_change_twh',
        'other_renewables_elec_per_capita',
        'other_renewables_elec_per_capita_exc_biofuel',
        'other_renewables_energy_per_capita',
        'other_renewables_share_elec',
        'other_renewables_share_elec_exc_biofuel',
        'other_renewables_share_energy',
    ],
}


def get_safe_features(categories: list = None) -> list:
    """
    Get list of safe features for ML training.
    
    Args:
        categories: Optional list of category names to include.
                   If None, returns all safe features.
    
    Returns:
        List of column names safe for ML training.
    """
    if categories is None:
        categories = list(SAFE_FEATURE_CATEGORIES.keys())
    
    features = []
    for cat in categories:
        if cat in SAFE_FEATURE_CATEGORIES:
            features.extend(SAFE_FEATURE_CATEGORIES[cat])
    
    return features


def filter_safe_columns(df, include_target: bool = False) -> 'pd.DataFrame':
    """
    Filter a DataFrame to only include safe features.
    
    Args:
        df: Input DataFrame with all columns
        include_target: If True, includes co2_kilotons as target column
        
    Returns:
        DataFrame with only safe columns
    """
    safe_cols = get_safe_features()
    available_cols = [c for c in safe_cols if c in df.columns]
    
    # Always include metadata for identification
    for col in ['country', 'year']:
        if col in df.columns and col not in available_cols:
            available_cols.insert(0, col)
    
    if include_target and 'co2_kilotons' in df.columns:
        available_cols.append('co2_kilotons')
    
    return df[available_cols].copy()


def validate_features(feature_list: list) -> dict:
    """
    Validate a list of features for leakage risks.
    
    Returns:
        Dict with 'safe', 'leakage', 'suspicious', 'unknown' lists
    """
    all_safe = get_safe_features()
    
    result = {
        'safe': [],
        'leakage': [],
        'suspicious': [],
        'unknown': [],
    }
    
    for feat in feature_list:
        if feat in LEAKAGE_COLUMNS:
            result['leakage'].append(feat)
        elif feat in SUSPICIOUS_COLUMNS:
            result['suspicious'].append(feat)
        elif feat in all_safe:
            result['safe'].append(feat)
        elif feat not in METADATA_COLUMNS:
            result['unknown'].append(feat)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("=== Feature Safety Check ===\n")
    print(f"Total safe features: {len(get_safe_features())}")
    print(f"Leakage columns to AVOID: {LEAKAGE_COLUMNS}")
    print(f"\nCategories: {list(SAFE_FEATURE_CATEGORIES.keys())}")
