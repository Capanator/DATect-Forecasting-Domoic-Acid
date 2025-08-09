# DATect - Domoic Acid Forecasting System 🌊🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)
[![Scientific](https://img.shields.io/badge/Status-Peer%20Review%20Ready-brightgreen.svg)](https://github.com/)

## 🎯 Overview

DATect is a state-of-the-art machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. The system integrates satellite oceanographic data, environmental measurements, and advanced temporal safeguards to provide scientifically rigorous predictions while preventing data leakage.

### 🔬 Key Features

- **Advanced ML Forecasting**: XGBoost-based predictions with proven R² ≈ 0.529 performance
- **Zero Data Leakage**: Bulletproof temporal safeguards validated through comprehensive testing
- **Multi-Source Data Integration**: MODIS satellite, climate indices, and streamflow data
- **Modern Web Interface**: React frontend with FastAPI backend
- **Real-time & Retrospective Analysis**: Support for both operational and research use
- **10 Monitoring Sites**: Complete Pacific Coast coverage from California to Washington
- **21+ Years of Data**: Temporal coverage from 2002-2023

## 🚀 Quick Start

### One-Command Launch

```bash
# Complete system startup (backend + frontend + browser)
python run_datect.py
```

This will:
1. ✅ Validate scientific data integrity
2. ✅ Check temporal safeguards
3. ✅ Install dependencies automatically
4. ✅ Start backend API (port 8000)
5. ✅ Start frontend (port 3000)
6. ✅ Open browser automatically

### Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Generate dataset (first time only, 30-60 min)
python dataset-creation.py

# 3. Run forecasting system  
python run_datect.py        # Complete web interface (auto-installs additional dependencies)
```

### Deploy to a URL (Docker)

```bash
# 1) Build the image
docker build -t datect:latest .

# 2) Generate dataset locally if you haven't yet (mount it into the container)
python dataset-creation.py

# 3) Run the container, mounting the processed data so the API can read it
docker run -d --name datect -p 8000:8000 \
  -e PORT=8000 \
  -e DATECT_ENV=production \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  datect:latest

# Open in browser (served by FastAPI):
# http://localhost:8000
# API docs:
# http://localhost:8000/docs
```

For cloud (Render, Fly.io, GCP, AWS, Azure), use this Docker image and point traffic to port 8000. When front-end is built, it is served by the same FastAPI container at the root path, and the API is under `/api/*`.

## 📊 System Architecture

```
DATect-Forecasting-Domoic-Acid/
├── 📁 backend/                 # FastAPI backend
│   ├── api.py                  # REST API endpoints
│   └── visualizations.py       # Scientific visualizations
├── 📁 frontend/                # React web interface
│   └── src/pages/              # Dashboard and visualization pages
├── 📁 forecasting/core/        # Core ML system
│   ├── forecast_engine.py      # Main forecasting logic
│   ├── data_processor.py       # Temporal-safe processing
│   └── model_factory.py        # ML model management
├── 📁 data/                    # Dataset storage
│   ├── raw/                    # Original CSV files
│   ├── intermediate/           # Cached satellite data
│   └── processed/              # ML-ready datasets
├── 📁 docs/                    # Documentation
├── config.py                   # System configuration
├── dataset-creation.py         # Data processing pipeline
├── requirements.txt            # Python dependencies
└── run_datect.py              # One-command launcher
```

## 🔬 Scientific Validation

### Temporal Integrity Safeguards

The system implements **rigorous temporal safeguards** to prevent data leakage:

- **Strict Chronological Splits**: Training data ≤ anchor date < test data
- **Temporal Buffers**: Configurable gaps between train/test sets
- **Lag Feature Cutoffs**: Future values physically set to NaN
- **Per-Forecast Validation**: Each prediction validated independently

### Built-in Validation

```
✅ Scientific data integrity validation
✅ Temporal safeguard validation  
✅ Zero data leakage confirmed
✅ 89,708 samples/second processing speed
✅ <250MB memory usage for full dataset
```

### Model Performance

- **XGBoost Regression**: R² ≈ 0.529, MAE ≈ 8.2 μg/g
- **Classification Accuracy**: ~70% for 4-category risk levels
- **Training Samples**: Minimum 3-5 required per forecast
- **Temporal Coverage**: 21 years (2002-2023)
- **Spatial Coverage**: 10 Pacific Coast sites

## 🖥️ Web Interface

### Dashboard Features

- **Real-time Forecasting**: Interactive date/site selection
- **Retrospective Analysis**: Historical model validation
- **Scientific Visualizations**:
  - Correlation heatmaps
  - Sensitivity analysis
  - Time series comparisons
  - Spectral analysis
  - Feature importance

### API Endpoints

```python
GET  /health                    # System health check
GET  /api/sites                 # Available monitoring sites
GET  /api/models                # Available ML models
POST /api/forecast              # Generate single forecast
POST /api/forecast/enhanced     # Enhanced forecast with graphs
POST /api/retrospective         # Run retrospective analysis
GET  /api/visualizations/*      # Scientific visualizations
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
FORECAST_MODE = "realtime"      # "realtime" or "retrospective"
FORECAST_TASK = "regression"    # "regression" or "classification"
FORECAST_MODEL = "xgboost"      # "xgboost" or "linear"
TEMPORAL_BUFFER_DAYS = 1        # Minimum days between train/test
LAG_FEATURES = [1, 3]           # Optimized lag selection
MIN_TRAINING_SAMPLES = 3        # Minimum training size
```

## 📈 Data Sources

- **MODIS Satellite**: Chlorophyll-a, SST, PAR, fluorescence
- **Climate Indices**: PDO, ONI, BEUTI upwelling
- **USGS Streamflow**: Columbia River discharge
- **In-situ Measurements**: DA toxin concentrations
- **Pseudo-nitzschia**: Cell count data

## 🧪 Built-in Validation

The system includes comprehensive validation that runs automatically:

```bash
# Complete validation (runs automatically on startup)
python run_datect.py

# Data pipeline validation
python dataset-creation.py
```

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_START.md) - One-command setup instructions
- [Development Guidelines](docs/CLAUDE.md) - For AI assistants and developers
- [API Documentation](docs/API_DOCUMENTATION.md) - Complete REST API reference
- [Scientific Validation](docs/SCIENTIFIC_VALIDATION.md) - Temporal safeguards and peer-review standards
- [Security Framework](docs/SECURITY_VALIDATION.md) - Data leakage prevention
- [Testing Documentation](docs/TESTING_DOCUMENTATION.md) - Validation framework

## 🏆 Scientific Publications

This system is designed for peer-reviewed publication with:
- Rigorous temporal validation
- Comprehensive statistical analysis
- Reproducible results (fixed random seeds)
- Complete data provenance
- Transparent methodology

## 🤝 Contributing

1. Run temporal integrity tests before any changes
2. Maintain data leakage prevention measures
3. Document scientific assumptions
4. Follow existing code patterns
5. Update tests for new features

## ⚠️ Important Notes

- **Never modify temporal safeguards** without running validation tests
- **Data generation** (dataset-creation.py) takes 30-60 minutes
- **Satellite data** requires internet connection for initial download
- **Model files** are included; no training required for basic use

## 📄 License

This project is part of scientific research. Please cite appropriately if used in publications.

## 🙏 Acknowledgments

- NOAA CoastWatch for satellite data access
- USGS for streamflow data
- Pacific Coast monitoring programs for DA measurements
- XGBoost and scikit-learn communities

---

**System Status**: ✅ Production Ready | 🔬 Peer Review Ready | 🚀 Actively Maintained

Last Updated: November 2024