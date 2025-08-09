# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Quick Start
```bash
# Complete system launch (recommended)
python run_datect.py

# Manual setup for first-time data generation
pip install -r requirements.txt
cd frontend && npm install && cd ..
python dataset-creation.py  # First time only (30-60 min)
```

### Development Servers
```bash
# Backend API server (FastAPI)
uvicorn backend.api:app --reload --port 8000

# Frontend development server  
cd frontend && npm run dev  # Port 3000
```

### Frontend Commands
```bash
cd frontend
npm run dev        # Development server with hot reload
npm run build      # Production build
npm run preview    # Preview production build  
npm run lint       # ESLint code quality check
npm run test       # Vitest testing
```

### Docker Deployment
```bash
docker build -t datect:latest .
docker run -d -p 8000:8000 -e PORT=8000 datect:latest
```

## High-Level Architecture

### System Overview
DATect is a scientific machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. The system has three main layers:

1. **Data Processing Layer** (`dataset-creation.py`, `forecasting/core/data_processor.py`)
   - Integrates MODIS satellite data, climate indices, streamflow, and in-situ measurements
   - Implements temporal safeguards to prevent data leakage
   - Produces ML-ready datasets with proper train/test splits

2. **ML Forecasting Engine** (`forecasting/core/`)
   - `forecast_engine.py`: Main prediction orchestration with temporal validation
   - `model_factory.py`: Manages XGBoost and linear models
   - `data_processor.py`: Temporal-safe feature engineering
   - Supports both regression (continuous values) and classification (risk levels)

3. **Web Interface** 
   - **Backend** (`backend/api.py`): FastAPI REST API with scientific visualizations
   - **Frontend** (`frontend/`): React SPA with Plotly.js visualizations
   - Real-time forecasting and retrospective analysis dashboards

### Key Scientific Safeguards
The system implements rigorous temporal integrity measures critical for peer-reviewed research:
- **Strict chronological splits**: Training data < anchor date < test data
- **Temporal buffers**: Configurable gaps between train/test sets (default: 1 day)
- **Lag feature cutoffs**: Future values physically set to NaN
- **Per-forecast validation**: Each prediction validated independently

### Configuration Management
All system behavior controlled via `config.py`:
- `FORECAST_MODE`: "realtime" or "retrospective"
- `FORECAST_TASK`: "regression" or "classification"  
- `FORECAST_MODEL`: "xgboost" or "linear"
- `LAG_FEATURES`: [1, 3] (optimized based on ACF/PACF analysis)
- `MIN_TRAINING_SAMPLES`: 3 (minimum for model training)
- `TEMPORAL_BUFFER_DAYS`: 1 (prevents data leakage)

### Data Flow
1. Raw CSV files (`data/raw/`) → 
2. Dataset creation pipeline downloads satellite/climate data →
3. Processed parquet files (`data/processed/`) →
4. Forecast engine loads data with temporal safeguards →
5. API serves predictions and visualizations →
6. React frontend displays results

### API Architecture
The FastAPI backend (`backend/api.py`) provides:
- `/api/forecast`: Single-point predictions
- `/api/forecast/enhanced`: Predictions with visualization data
- `/api/retrospective`: Historical validation analysis
- `/api/visualizations/*`: Scientific plots (correlation, sensitivity, etc.)
- Static file serving for production React build

### Frontend Architecture
React application (`frontend/src/`) with:
- `pages/ForecastDashboard.jsx`: Real-time forecasting interface
- `pages/RetrospectiveDashboard.jsx`: Historical analysis interface
- `components/`: Reusable UI components
- Plotly.js for scientific visualizations
- Tailwind CSS for styling

### Model Performance
- **XGBoost Regression**: R² ≈ 0.529, MAE ≈ 8.2 μg/g
- **Classification**: ~70% accuracy for 4-category risk levels
- **Data Coverage**: 21 years (2002-2023), 10 Pacific Coast sites
- **Processing Speed**: 89,708 samples/second

### Important Development Notes
- **Temporal safeguards are critical** - never modify without running validation
- **Dataset generation takes 30-60 minutes** first time (downloads satellite data)
- **Fixed random seed (42)** ensures reproducible results
- **All dates in UTC** for satellite data consistency
- **Parquet format** used for efficient data storage and access