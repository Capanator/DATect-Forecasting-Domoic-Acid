# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DATect is a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. It's a scientific-grade application with comprehensive temporal safeguards to prevent data leakage, designed for both research and operational deployment.

## Key Commands

```bash
# Run the system locally (handles all setup automatically)
python run_datect.py

# Generate/update dataset (30-60 min first time, only when data changes)
python dataset-creation.py

# Deploy to Google Cloud (builds container automatically)
./deploy_gcloud.sh
```

## Architecture

### Core Components
- **backend/** - FastAPI server providing forecasting and visualization endpoints
- **frontend/** - React + Vite app with interactive dashboards
- **forecasting/core/** - ML engine with temporal integrity safeguards
- **data/** - Multi-year datasets (satellite, climate, toxin measurements)
- **cache/** - Pre-computed results for production optimization

### Tech Stack
- **Backend**: Python, FastAPI, XGBoost, scikit-learn, Pandas, Plotly
- **Frontend**: React 18, Vite, TailwindCSS, Plotly.js, Lucide Icons
- **ML Models**: XGBoost (primary), Linear/Logistic (interpretable)
- **Data**: Parquet, NetCDF, CSV formats with Pandas/XArray processing

### Critical Configuration (config.py)
- `FORECAST_MODE`: "retrospective" or "realtime"
- `FORECAST_TASK`: "regression" or "classification"  
- `FORECAST_MODEL`: "xgboost" or "linear"
- `USE_LAG_FEATURES`: Enable/disable time series lag features (default: True)
- `TEMPORAL_BUFFER_DAYS`: Minimum train/test gap (default: 1)
- `SATELLITE_BUFFER_DAYS`: Satellite processing delay (default: 7)

## Scientific Integrity Framework

This system implements strict temporal safeguards:

1. **Chronological splits only** - No random train/test splits
2. **Realistic data delays** - 7-day satellite buffer, 2-month climate delay
3. **Per-forecast DA categorization** - Categories created per prediction
4. **Temporal buffer enforcement** - Minimum gap between train/test
5. **No future information leakage** - Comprehensive validation tests

The system runs 7 critical temporal tests on startup and will refuse to operate if any fail.

## Key Development Patterns

### Adding New Features
1. Features must respect temporal ordering - check `forecasting/core/data_processor.py`
2. Control lag feature usage via `USE_LAG_FEATURES` in `config.py` (set to False to disable)
3. Update lag configurations in `LAG_FEATURES` if adding time-series features
4. Ensure satellite/climate data respects appropriate buffer delays

### Modifying Models
1. Models are in `forecasting/core/model_factory.py`
2. Any new model must pass all temporal validation tests
3. Update `FORECAST_MODEL` options in config.py if adding new model types

### Frontend Development
1. API endpoints are in `backend/api.py` (not main.py)
2. React components in `frontend/src/components/`
3. Pages in `frontend/src/pages/` (Dashboard.jsx, Historical.jsx, About.jsx)
4. Use existing Plotly patterns for new visualizations
5. Styling with TailwindCSS and HeadlessUI components

### Data Processing
1. Raw data ingestion: `data/raw/` directory (DA and PN measurements)
2. Feature engineering: `forecasting/core/feature_engineering.py`
3. Dataset creation: `dataset-creation.py` (regenerates all features)
4. Cache management: `backend/cache_manager.py` and `precompute_cache.py`

### Ports and Services
- **Backend API**: Port 8000 (FastAPI with auto-docs at /docs)
- **Frontend Dev Server**: Port 3000 (Vite)
- **Web Interface**: Integrated frontend/backend served together

## Performance Benchmarks
- R² ≈ 0.525 for regression
- 77.6% accuracy for classification
- MAE ≈ 4.57 μg/g
- Zero data leakage violations

## Important Notes
- System monitors 10 sites across Oregon/Washington coast
- 21 years of historical data (2002-2023)
- Pre-computed cache significantly improves production response times
- All random operations use fixed seeds for reproducibility