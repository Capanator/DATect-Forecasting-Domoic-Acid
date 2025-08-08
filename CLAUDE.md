# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Development Commands

```bash
# One-command system launch (RECOMMENDED)
python run_datect.py             # Complete system startup with scientific validation

# Individual component commands
pip install -r requirements.txt  # Install dependencies
python dataset-creation.py       # Process data (30-60 min runtime)
python modular-forecast.py       # Run forecasting system (Dash interface)

# Development servers (advanced)
cd frontend && npm run dev       # Frontend development server only
cd backend && uvicorn api:app --reload  # Backend API server only

# Testing commands
python tools/testing/test_complete_pipeline.py                    # Integration tests
python analysis/scientific-validation/test_temporal_integrity.py  # Critical data leakage tests
python analysis/scientific-validation/run_scientific_validation.py # Peer-review validation
python analysis/scientific-validation/performance_profiler.py     # Performance analysis
```

## Repository Architecture

This is a scientific machine learning system for forecasting harmful algal bloom concentrations (domoic acid) using satellite oceanographic data and environmental measurements. The system processes 20+ years of data across 10 Pacific Coast monitoring sites, with both standalone Python dashboards and a modern React web application.

### Core Pipeline Architecture
```
Data Flow: Raw CSV → Satellite Processing → Feature Engineering → ML Forecasting → Dashboard/Analysis
Web Stack: React Frontend ↔ FastAPI Backend ↔ Python Forecasting Engine
```

### Key Components

#### Main Pipeline Files
- `dataset-creation.py` - Complete data processing pipeline (satellite + environmental data)
- `modular-forecast.py` - Main forecasting application with multiple operation modes
- `config.py` - System configuration with extensive satellite data URLs and settings
- `run_datect.py` - Full-stack web application launcher

#### Core Forecasting Module (`forecasting/core/`)
- `forecast_engine.py` - Main forecasting logic with strict temporal integrity protection
- `data_processor.py` - Data processing with temporal safeguards and lag feature creation
- `model_factory.py` - ML model creation (XGBoost primary, Ridge/Logistic fallback)
- `env_config.py` - Environment configuration management
- `logging_config.py` - Logging system
- `exception_handling.py` - Error handling utilities

#### Dashboard System (`forecasting/dashboard/`)
- `realtime.py` - Interactive forecasting UI (port 8066)
- `retrospective.py` - Historical analysis UI (port 8071)

#### Web Application (`frontend/` & `backend/`)
- `frontend/` - React application with Vite, TailwindCSS, and Plotly.js
- `backend/api.py` - FastAPI server providing REST API endpoints
- `backend/visualizations.py` - Scientific visualization logic
- Modern responsive UI with 5 interactive visualizations and forecasting tools

### Data Organization
```
data/
├── raw/           # Original CSV files (da-input/, pn-input/)
├── intermediate/  # Cached satellite data (satellite_data_intermediate.parquet)
└── processed/     # Final combined dataset (final_output.parquet)
```

### Configuration System
Edit `config.py` to customize key settings:
- `FORECAST_MODE`: "retrospective" or "realtime"
- `FORECAST_MODEL`: "xgboost" or "ridge"
- `FORECAST_TASK`: "regression" or "classification"
- `LAG_FEATURES`: Time series lag configuration (currently [1,3])

## Critical Technical Requirements

### Temporal Integrity (CRITICAL)
The system has strict temporal safeguards to prevent data leakage:
- All lag features use temporal cutoffs (`create_lag_features_safe`)
- Train/test splits maintain chronological order
- Satellite data respects buffer periods (7 days)
- Climate data respects reporting delays (2 months)

**Never modify temporal logic without running `test_temporal_integrity.py` - failure invalidates all scientific results.**

### Model Performance Requirements
- XGBoost is the primary model (superior performance)
- Ridge/Logistic regression as fallback methods
- Minimum 3-5 training samples required per forecast
- Temporal buffers: 1-7 days for data integrity

### Data Processing Pipeline
1. **Raw Data**: CSV files with DA/PN measurements
2. **Satellite Processing**: MODIS data (chlorophyll, SST, PAR, fluorescence)
3. **Environmental Data**: PDO, ONI, BEUTI climate indices, USGS streamflow
4. **Feature Engineering**: Lag features, anomaly detection, temporal alignment
5. **ML Pipeline**: Training, validation, prediction with temporal safeguards

## Testing Framework

### Critical Tests (Must Pass)
```bash
# These tests are AUTOMATICALLY run by run_datect.py
python analysis/scientific-validation/test_temporal_integrity.py  # 7/7 tests must pass
python tools/testing/test_complete_pipeline.py                   # Integration validation

# Manual execution for development
python -m pytest analysis/scientific-validation/test_temporal_integrity.py -v
```

### Scientific Validation
```bash
python analysis/scientific-validation/run_scientific_validation.py  # Peer-review standards
python analysis/scientific-validation/advanced_acf_pacf.py          # Statistical lag analysis
```

The system maintains a 100% test success rate across 21 test components. All temporal integrity tests (7/7) must pass for scientific validity.

## Development Guidelines

### Data Processing
- Always use the temporal-safe methods in `data_processor.py`
- Respect the minimum sample requirements (3-5 samples)
- Use parquet format for intermediate data storage
- Maintain strict chronological ordering in train/test splits

### Model Development
- Primary: XGBoost for superior performance
- Fallback: Ridge regression for linear relationships
- Classification: Logistic regression with 4 risk categories
- Always validate with temporal integrity tests

### Configuration Changes
- Modify `config.py` for system-wide settings
- Test configuration changes with complete pipeline
- Satellite data URLs are complex - validate before changing
- LAG_FEATURES impacts model performance significantly (current optimized: [1,3])

### Web Development
- Frontend uses React 18 with Vite build system and TailwindCSS styling
- Backend uses FastAPI with comprehensive scientific visualizations
- Scientific plots: Correlation heatmaps, sensitivity analysis, spectral analysis, waterfall plots
- Integration with XGBoost forecasting engine for real predictions
- All visualizations include proper temporal safeguards and NaN handling

### Performance Considerations
- Current system: 89,708 samples/second processing
- Memory usage: <250MB for full dataset
- Runtime: 30-60 minutes for complete data processing
- Dashboard ports: 8066 (realtime), 8071 (retrospective)
- Web app ports: 3000 (frontend), 8000 (backend API)

## Multi-Interface System

The system provides multiple interfaces:

### Python Dashboards (Dash/Plotly)
- Retrospective analysis: `python modular-forecast.py` (config: FORECAST_MODE="retrospective")
- Real-time forecasting: `python modular-forecast.py` (config: FORECAST_MODE="realtime")
- Scientific validation tools in `analysis/` directory

### Web Application (React/FastAPI)
- Modern responsive interface accessible via `python run_datect.py`
- REST API endpoints for programmatic access
- Interactive visualizations and forecasting tools
- Production-ready deployment architecture

## Scientific Context

This system is designed for peer-review publication and operational deployment. Key scientific features:
- 21 years of temporal coverage (2002-2023)
- 10 Pacific Coast monitoring sites
- Multi-source data integration (satellite, climate, environmental)
- Rigorous temporal validation to prevent data leakage
- Statistical lag selection analysis (ACF/PACF)

The system processes domoic acid concentrations from harmful algal blooms, providing early warning capabilities for public health and marine management applications.