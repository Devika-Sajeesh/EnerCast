"""
Data loading utilities for EnerCast.
"""

import pandas as pd
from pathlib import Path


def load_energy_consumption(filepath: str = None) -> pd.DataFrame:
    """Load energy consumption data from CSV."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "raw" / "energy_consumption.csv"
    return pd.read_csv(filepath)


def load_co2_emissions(filepath: str = None) -> pd.DataFrame:
    """Load CO2 emissions data from CSV."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "raw" / "co2_emissions.csv"
    return pd.read_csv(filepath)


def load_merged_data(filepath: str = None) -> pd.DataFrame:
    """Load merged energy and emissions data from CSV."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "processed" / "merged_energy_emissions.csv"
    return pd.read_csv(filepath)
