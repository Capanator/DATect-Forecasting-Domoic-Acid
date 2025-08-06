# DATect: Domoic Acid Forecasting System

A machine learning system for predicting harmful algal bloom concentrations along the Pacific Coast using satellite oceanographic data, climate indices, and historical measurements.

## Quick Start after navigating to install location

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process data (30-60 min runtime)
python dataset-creation.py

# 3. Run forecasting system
python modular-forecast.py
```

## Core Components

### Main Pipeline
- `dataset-creation.py` - Complete data processing pipeline
- `modular-forecast.py` - Main forecasting application
- `config.py` - System configuration
- `forecasting/` - Core forecasting modules

### Data Organization
```
data/
├── raw/           # Original CSV files (da-input/, pn-input/)
├── intermediate/  # Cached satellite data
└── processed/     # Final combined dataset
```

### Analysis Tools (Optional)
```
analysis/
├── scientific-validation/  # Temporal integrity testing
└── data-visualization/     # Charts and analysis
```

## Configuration

Edit `config.py` to customize:
- **FORECAST_MODE**: "retrospective" or "realtime"
- **FORECAST_MODEL**: "xgboost" or "ridge" 
- **FORECAST_TASK**: "regression" or "classification"

## System Features

- **Temporal Integrity**: Strict data leakage prevention
- **Multi-Model Support**: XGBoost (primary), Ridge/Logistic (fallback)
- **Interactive Dashboards**: Real-time and retrospective analysis
- **Production Ready**: Direct Python deployment
- **Scientific Rigor**: Validated temporal safeguards

## Architecture

The system processes:
1. **Satellite Data**: MODIS chlorophyll, SST, PAR, fluorescence
2. **Climate Indices**: PDO, ONI, BEUTI upwelling
3. **Environmental Data**: USGS streamflow
4. **Toxin Measurements**: Historical DA/PN concentrations

Using XGBoost machine learning with comprehensive temporal safeguards to predict domoic acid levels at 10 Pacific Coast monitoring sites.

## Performance

- **Processing Speed**: 89,708 samples/second
- **Memory Usage**: <250MB
- **Model Accuracy**: R² > 0.5 for regression tasks
- **Temporal Coverage**: 2002-2023 (21 years)

## File Structure

```
DATect-Forecasting-Domoic-Acid/
├── dataset-creation.py           # Data processing pipeline
├── modular-forecast.py           # Main forecasting application
├── config.py                     # System configuration
├── requirements.txt              # Dependencies
├── README.md                     # This file
│
├── forecasting/                  # Core forecasting modules
│   ├── core/
│   │   ├── forecast_engine.py    # Main forecasting logic
│   │   ├── data_processor.py     # Data processing
│   │   ├── model_factory.py      # ML model creation
│   │   ├── env_config.py         # Environment configuration
│   │   ├── logging_config.py     # Logging system
│   │   └── exception_handling.py # Error handling
│   └── dashboard/
│       ├── realtime.py          # Interactive forecasting UI
│       └── retrospective.py     # Historical analysis UI
│
├── data/                         # Organized data storage
│   ├── raw/                     # Original CSV files
│   ├── intermediate/            # Cached satellite data
│   └── processed/               # Final combined dataset
│
├── analysis/                     # Analysis tools (optional)
│   ├── scientific-validation/   # Temporal integrity tests
│   │   ├── scientific_evidence/ # Validation results
│   │   ├── advanced_acf_pacf.py # Advanced autocorrelation analysis
│   │   ├── performance_profiler.py # Performance profiling
│   │   ├── run_scientific_validation.py # Validation runner
│   │   ├── scientific_validation.py # Main validation logic
│   │   └── test_temporal_integrity.py # Temporal integrity tests
│   └── data-visualization/      # Charts and plots
│       └── data-visualizations/ # Visualization scripts
│           ├── correlation heatmap.py # Correlation analysis
│           ├── sensitivity test.py # Sensitivity analysis
│           ├── time series comparison.py # Time series plots
│           ├── waterfall plot.py # Waterfall charts
│           └── xgboost_spectral_analysis.py # Spectral analysis
│
└── tools/                        # Development tools
    ├── testing/                 # Test scripts
    │   └── test_complete_pipeline.py
    └── documentation/           # Technical docs
```

## Documentation

- `tools/documentation/TESTING_DOCUMENTATION.md` - Testing explanations
- `tools/documentation/` - Technical documentation
- `analysis/scientific-validation/` - Validation reports

## Research Applications

- **Academic Research**: HAB prediction algorithm development
- **Operational Forecasting**: Real-time toxin level prediction
- **Public Health**: Early warning system for shellfish safety
- **Marine Management**: Fishery closure decision support

---

**Research Ready**: This system maintains scientific rigor with temporal safeguards suitable for peer review and operational deployment.