# EnerCast

Energy Consumption and CO2 Emissions Analysis & Forecasting

## Project Structure

```
EnerCast/
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── World Energy Consumption.csv
│   │   └── Carbon_(CO2)_Emissions_by_Country.csv
│   └── processed/              # Processed data files
│       └── merged_energy_emissions.csv
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_emission_rules.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_scenario_simulation.ipynb
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── data_processing_pipeline.py  # Merge pipeline
│   ├── feature_safety.py       # Feature leakage prevention
│   ├── preprocessing.py        # Data preprocessing
│   ├── baseline.py             # Baseline emission rules
│   ├── models.py               # ML models
│   ├── evaluation.py           # Model evaluation
│   └── scoring.py              # Scoring utilities
│
├── results/                    # Results and outputs
│   ├── figures/
│   └── metrics.json
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Pipeline

Run the data processing pipeline to merge energy + emissions data:

```bash
python src/data_processing_pipeline.py
```

This creates `data/processed/merged_energy_emissions.csv` with:
- 22,012 rows (country-year observations)
- 133 columns (energy features + CO2 target)
- 5,108 rows with matched CO2 emissions data

## Known Limitations

### 1. Country Name Matching (Partial)

Country names are matched using lowercase normalization. Some mismatches remain:

| Energy Dataset | Emissions Dataset | Status |
|----------------|-------------------|--------|
| United States | United States | ✅ Matched |
| Russia | Russia | ✅ Matched |
| South Korea | South Korea | ✅ Matched |
| Micronesia (country) | Micronesia | ❌ Not matched |
| USSR | USSR | ❌ Not matched |

**Impact:** ~172 countries matched out of ~230. Regional aggregates (e.g., "Africa", "ASEAN") don't have emissions data.

### 2. Region Column Conflict

The `Region` column comes from the emissions dataset only. The energy dataset encodes geography differently (via `iso_code`).

**For ML:** Use one region source only. Recommend using `iso_code` from energy data.

### 3. Feature Leakage Risk ⚠️

Some columns in the energy dataset are **emission-derived**. Using them as features will cause data leakage.

**MUST DROP before training:**
- `greenhouse_gas_emissions` - Direct GHG metric
- `carbon_intensity_elec` - CO2 per kWh (derived from emissions)

**Use `src/feature_safety.py` to filter safe features:**

```python
from src.feature_safety import filter_safe_columns, LEAKAGE_COLUMNS

df = pd.read_csv('data/processed/merged_energy_emissions.csv')
df_safe = filter_safe_columns(df, include_target=True)
```

## Usage

1. Run `python src/data_processing_pipeline.py` to merge data
2. Run notebooks in order (01 through 06)
3. Results will be saved in `results/`

## License

MIT License
