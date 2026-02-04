"""
Data Processing Pipeline for EnerCast
======================================
Steps:
1. Load raw data files
2. Inspect columns and data types
3. Normalize country names (lowercase) and year columns
4. Merge datasets on Country + Year (left join)
5. Save processed dataset

Result: One row = one country-year
        Features = energy consumption metrics
        Target = CO2 emissions
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_energy_data(filepath: str) -> pd.DataFrame:
    """Load World Energy Consumption data."""
    df = pd.read_csv(filepath)
    print(f"Energy data shape: {df.shape}")
    print(f"Sample columns: {list(df.columns[:10])}...")
    print(f"Unique countries: {df['country'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    return df


def load_emissions_data(filepath: str) -> pd.DataFrame:
    """Load CO2 Emissions data."""
    df = pd.read_csv(filepath)
    print(f"\nEmissions data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique countries: {df['Country'].nunique()}")
    return df


def normalize_keys(energy_df: pd.DataFrame, emissions_df: pd.DataFrame):
    """
    STEP 2: Normalize keys for merging.
    - Standard country names (lowercase, stripped)
    - Integer year column
    """
    energy_df = energy_df.copy()
    emissions_df = emissions_df.copy()
    
    # Create normalized country key (lowercase for matching)
    energy_df['country_key'] = energy_df['country'].str.strip().str.lower()
    emissions_df['country_key'] = emissions_df['Country'].str.strip().str.lower()
    
    # Ensure year is integer in energy data
    energy_df['year'] = energy_df['year'].astype(int)
    
    # Extract year from Date column in emissions data (format: DD-MM-YYYY)
    emissions_df['year'] = pd.to_datetime(emissions_df['Date'], format='%d-%m-%Y').dt.year
    
    return energy_df, emissions_df


def analyze_country_overlap(energy_countries: set, emissions_countries: set):
    """Analyze overlap between country sets."""
    common = energy_countries & emissions_countries
    only_energy = energy_countries - emissions_countries
    only_emissions = emissions_countries - energy_countries
    
    print("\n" + "="*60)
    print("COUNTRY OVERLAP ANALYSIS")
    print("="*60)
    print(f"Countries in Energy data: {len(energy_countries)}")
    print(f"Countries in Emissions data: {len(emissions_countries)}")
    print(f"Common countries: {len(common)}")
    print(f"Only in Energy: {len(only_energy)}")
    print(f"Only in Emissions: {len(only_emissions)}")
    
    return common, only_energy, only_emissions


def merge_datasets(energy_df: pd.DataFrame, emissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 3: Merge datasets using left join on country_key + year.
    """
    # Select and rename emissions columns
    emissions_subset = emissions_df[[
        'country_key', 'year', 'Region', 
        'Kilotons of Co2', 'Metric Tons Per Capita'
    ]].copy()
    
    emissions_subset = emissions_subset.rename(columns={
        'Kilotons of Co2': 'co2_kilotons',
        'Metric Tons Per Capita': 'co2_per_capita'
    })
    
    # Left join: keep all energy data, add emissions where available
    merged = pd.merge(
        energy_df,
        emissions_subset,
        on=['country_key', 'year'],
        how='left'
    )
    
    # Drop the helper key column
    merged = merged.drop(columns=['country_key'])
    
    print("\n" + "="*60)
    print("MERGE RESULTS")
    print("="*60)
    print(f"Energy rows: {len(energy_df):,}")
    print(f"Emissions rows: {len(emissions_df):,}")
    print(f"Merged rows: {len(merged):,}")
    print(f"Rows with CO2 data: {merged['co2_kilotons'].notna().sum():,}")
    print(f"Countries with CO2 data: {merged[merged['co2_kilotons'].notna()]['country'].nunique()}")
    print(f"Match rate: {merged['co2_kilotons'].notna().sum() / len(merged) * 100:.1f}%")
    
    return merged


def save_processed_data(df: pd.DataFrame, filepath: str):
    """STEP 4: Save the merged dataset."""
    df.to_csv(filepath, index=False)
    print(f"\nSaved processed data to: {filepath}")
    print(f"Final shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")


def main():
    """Main pipeline execution."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    energy_file = raw_dir / "World Energy Consumption.csv"
    emissions_file = raw_dir / "Carbon_(CO2)_Emissions_by_Country.csv"
    output_file = processed_dir / "merged_energy_emissions.csv"
    
    # ===== STEP 1: LOAD RAW DATA =====
    print("="*60)
    print("STEP 1: LOADING RAW DATA")
    print("="*60)
    
    energy_df = load_energy_data(energy_file)
    emissions_df = load_emissions_data(emissions_file)
    
    # ===== STEP 2: NORMALIZE KEYS =====
    print("\n" + "="*60)
    print("STEP 2: NORMALIZING KEYS")
    print("="*60)
    
    energy_df, emissions_df = normalize_keys(energy_df, emissions_df)
    print("Created standardized country keys (lowercase)")
    print("Extracted integer year from Date column")
    print(f"Emissions year range: {emissions_df['year'].min()} - {emissions_df['year'].max()}")
    
    # Analyze country overlap
    energy_countries = set(energy_df['country_key'].unique())
    emissions_countries = set(emissions_df['country_key'].unique())
    analyze_country_overlap(energy_countries, emissions_countries)
    
    # ===== STEP 3: MERGE DATASETS =====
    print("\n" + "="*60)
    print("STEP 3: MERGING DATASETS (LEFT JOIN)")
    print("="*60)
    
    merged_df = merge_datasets(energy_df, emissions_df)
    
    # ===== STEP 4: SAVE PROCESSED DATA =====
    print("\n" + "="*60)
    print("STEP 4: SAVING PROCESSED DATA")
    print("="*60)
    
    save_processed_data(merged_df, output_file)
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (rows with CO2 data)")
    print("="*60)
    sample_cols = ['country', 'year', 'primary_energy_consumption', 'co2_kilotons', 'co2_per_capita']
    sample = merged_df[sample_cols].dropna()
    print(sample.head(15).to_string(index=False))
    
    print("\n Pipeline completed successfully!")
    return merged_df


if __name__ == "__main__":
    merged_data = main()
