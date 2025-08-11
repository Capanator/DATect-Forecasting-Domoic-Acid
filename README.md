# DATect - Domoic Acid Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)

## Overview

Machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. Integrates satellite data, climate indices, and environmental measurements with strict temporal safeguards to prevent data leakage.

- **10 monitoring sites** from Oregon to Washington
- **21 years of data** (2002-2023)
- **R² ≈ 0.37** for regression, **77-82% accuracy** for classification
- **Zero data leakage** with 7 temporal validation tests

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Run locally (auto-installs dependencies)
python run_datect.py
```

Opens at http://localhost:3000

## Commands

```bash
# Run system locally
python run_datect.py

# Generate/update dataset (30-60 min, only when data changes)
python dataset-creation.py

# Deploy to Google Cloud
./deploy_gcloud.sh
```

## Google Cloud Deployment

```bash
# Set up Google Cloud CLI
gcloud auth login
gcloud config set project YOUR-PROJECT-ID

# Deploy (builds container automatically)
./deploy_gcloud.sh
```

## System Architecture

```
├── frontend/          # React + Vite interface
├── backend/           # FastAPI server
├── forecasting/core/  # ML engine with temporal safeguards
├── data/              # Datasets (satellite, climate, toxins)
└── cache/             # Pre-computed results
```

## Configuration

Edit `config.py`:

```python
FORECAST_MODE = "realtime"       # or "retrospective"
FORECAST_TASK = "classification" # or "regression"
FORECAST_MODEL = "xgboost"       # or "linear"
TEMPORAL_BUFFER_DAYS = 1         # Train/test gap
SATELLITE_BUFFER_DAYS = 7        # Data delay
```

## Using the System

### Dashboard
1. Select date (2008-2024), site, and model
2. Click "Forecast" for predictions
3. Risk categories:
   - **Low (≤5 μg/g)**: Safe
   - **Moderate (5-20 μg/g)**: Caution
   - **High (20-40 μg/g)**: Avoid
   - **Extreme (>40 μg/g)**: Hazard

### Historical Analysis
Access correlation heatmaps, sensitivity analysis, time series comparisons, and spectral analysis.

## API Endpoints

```
GET  /health                     # Status check
GET  /api/sites                  # Monitoring sites
POST /api/forecast               # Generate forecast
POST /api/forecast/enhanced      # Forecast with visualizations
GET  /api/visualizations/*       # Analysis plots
```

## Documentation

- [Forecast Pipeline](docs/FORECAST_PIPELINE.md) - Technical data flow
- [Scientific Validation](docs/SCIENTIFIC_VALIDATION.md) - Temporal safeguards
- [Visualizations Guide](docs/VISUALIZATIONS_GUIDE.md) - Chart interpretation

## Troubleshooting

**Port in use:**
```bash
python run_datect.py  # Kills existing processes automatically
```

**Missing dataset:**
```bash
python dataset-creation.py
```

**Node.js not found:**
Install from [nodejs.org](https://nodejs.org/)

## License

Scientific research project. Please cite if used in publications.

## Acknowledgments

- NOAA CoastWatch for satellite data
- USGS for streamflow data
- Olympic Region HAB Partnership, WA DOH, OR DFW for toxin measurements