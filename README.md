# EnerCast

**Data-Driven Energy Emissions Estimation & Carbon Credit Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![Data](https://img.shields.io/badge/Data-Pandas-green.svg)](https://pandas.pydata.org)

---

## Overview

EnerCast is an **end-to-end machine learning pipeline** that estimates national CO₂ emissions from energy consumption data. The project demonstrates production-grade ML engineering practices including:

- **Rigorous data pipeline design** with validation and quality checks
- **Leakage-safe feature engineering** with centralized governance
- **Baseline-first modeling philosophy** for principled ML adoption
- **Time-series aware evaluation** to prevent temporal data leakage

> ⚠️ EnerCast is designed as a decision-support and analysis system, not a regulatory or policy-grade emissions calculator.

---

## Problem Statement

Accurately estimating carbon dioxide (CO₂) emissions is essential for understanding climate impact and evaluating emission-reduction strategies. Traditional emissions accounting relies on fixed emission factors and aggregated rules—transparent but often coarse.

### Key Research Questions

1. How well can CO₂ emissions be estimated from energy consumption patterns?
2. Where do simple rule-based (physics-inspired) models fail?
3. Can machine learning models learn systematic deviations beyond fixed emission factors?
4. How do different modeling approaches compare in accuracy and interpretability?

---

## Technical Highlights

| Component | Implementation | Skills Demonstrated |
|-----------|---------------|---------------------|
| **Data Pipeline** | Country-year normalization, overlap analysis | ETL, Data Validation, Pandas |
| **Feature Engineering** | Leakage-safe policy with banned/approved columns | Feature Governance, ML Best Practices |
| **Baseline Model** | Physics-inspired emission factor estimator | Domain Knowledge Integration |
| **ML Models** | Linear Regression with time-based splits | Scikit-Learn, Model Evaluation |
| **Evaluation** | MAE, RMSE, R² with baseline comparison | Metrics Design, Statistical Analysis |

---

## Project Scope

### In Scope
- Energy-related CO₂ emissions estimation
- Fossil fuel and energy mix effects analysis
- Historical data analysis and forecasting diagnostics

### Out of Scope
- Official greenhouse gas inventories replacement
- Non-energy emissions (land use, agriculture)
- Carbon credit registry or compliance systems

---

## Data Sources

The project uses publicly available global datasets:

| Dataset | Features |
|---------|----------|
| **Energy Consumption** | Country-level annual consumption by source |
| **Fossil Fuels** | Coal, oil, gas usage metrics |
| **Renewables** | Low-carbon and renewable energy indicators |
| **Emissions** | National CO₂ emissions (kilotons, per capita) |

All datasets are merged using **country–year composite keys** and validated for coverage and consistency.

---

## Methodology

### 1. Data Engineering Pipeline

```
Raw Data → Normalization → Overlap Analysis → Merged Dataset
                ↓
        Country/Year Keys
                ↓
    merged_energy_emissions.csv
```

- Country and year standardization with manual mapping for edge cases
- Explicit overlap and label-coverage analysis
- Reproducible merged dataset generation

### 2. Leakage-Safe Feature Selection

To prevent label leakage, EnerCast enforces a **centralized feature policy**:

```python
# Feature categories
BANNED_COLUMNS = ["emissions_derived_*", "co2_*"]  # Direct leakage
SUSPICIOUS_COLUMNS = ["gdp_emissions_*"]           # Indirect leakage
APPROVED_COLUMNS = ["coal_twh", "oil_twh", ...]    # Safe features
```

All ML experiments must pass through this feature filter before training.

### 3. Rule-Based Baseline

A transparent baseline estimator using:
- Fossil fuel consumption (coal, oil, gas)
- Global-average emission factors
- Unit-safe conversions: `TWh → kWh → kg → kilotons`

This baseline represents a **physics-inspired lower bound** that ML models must beat.

### 4. Machine Learning Models

**Linear Regression** serves as the diagnostic model to evaluate:
- Whether linear relationships already explain emissions
- Which features dominate predictions
- Where ML improves or fails relative to baseline

Models use **time-based train/test splits** to prevent temporal information leakage.

---

## Evaluation Framework

### Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** | Average prediction error magnitude |
| **RMSE** | Penalizes large errors more heavily |
| **R²** | Variance explained (secondary metric) |

### Evaluation Philosophy

> Primary emphasis is on **error reduction relative to baseline**, not absolute accuracy.

Outcomes where ML does not outperform the baseline are considered **valid and informative**.

---

## Results Philosophy

EnerCast prioritizes:

- ✅ **Explainability** over black-box accuracy
- ✅ **Honest failure analysis** over cherry-picked results
- ✅ **Robustness to feature changes** over overfitting
- ✅ **Engineering reasoning** over accuracy chasing

---

## Repository Structure

```
EnerCast/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Merged, validated dataset
│
├── notebooks/                # Exploration and analysis
│
├── src/
│   ├── feature_safety.py     # Leakage-safe feature policy
│   ├── baseline.py           # Rule-based emission estimator
│   └── models.py             # ML models and experiments
│
├── results/                  # Metrics and analysis outputs
│
├── README.md
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Baseline before ML** | ML must justify its computational cost |
| **Time-based splits** | Prevents temporal data leakage |
| **Centralized feature governance** | Leakage prevention is enforced, not assumed |
| **Transparency over complexity** | Simple models preferred until justified |

---

## Known Limitations

| Limitation | Impact |
|------------|--------|
| Global-average emission factors | Country-specific variations not captured |
| Non-energy emissions excluded | Underestimates total emissions |
| Annual aggregation | Seasonal dynamics ignored |
| Diagnostic models only | Not production-tuned |

These limitations are **acknowledged by design** to maintain project scope.

---

## Future Roadmap

- [ ] GDP and feature ablation studies
- [ ] Non-linear models (tree-based regressors, XGBoost)
- [ ] Regional error analysis and visualization
- [ ] Uncertainty estimation and confidence intervals
- [ ] Scenario simulations for energy transition analysis

---

## Tech Stack

- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebooks

---

## Disclaimer

> EnerCast is an educational and analytical project. It should **not** be used for regulatory reporting, compliance, or financial decision-making.

---

## Author

Developed as a machine learning and data engineering project demonstrating **end-to-end ML system design** with emphasis on:
- Production-grade data pipeline architecture
- Leakage-aware feature engineering
- Baseline-first modeling methodology
- Principled evaluation frameworks