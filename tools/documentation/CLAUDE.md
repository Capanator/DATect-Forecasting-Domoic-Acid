# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Domoic Acid (DA) forecasting system that predicts harmful algal bloom concentrations along the Pacific Coast. The system processes multiple data sources (satellite oceanographic data, climate indices, streamflow, and shellfish toxin measurements) to generate predictive models using XGBoost machine learning with comprehensive temporal safeguards.

## Core Architecture

The system follows a modular architecture with strict temporal integrity to prevent data leakage:

**Modular Forecasting Framework (`forecasting/` package)**
- `forecasting/core/forecast_engine.py`: Main forecasting logic with leak-free temporal safeguards
- `forecasting/core/data_processor.py`: Data cleaning and feature engineering with forward-only processing
- `forecasting/core/model_factory.py`: ML model creation (XGBoost primary, Ridge/Logistic fallbacks)
- `forecasting/dashboard/realtime.py`: Interactive forecasting dashboard for specific dates/sites
- `forecasting/dashboard/retrospective.py`: Historical validation and performance analysis

**Data Processing Pipeline (`dataset-creation.py`)**
- Downloads satellite oceanographic data from NOAA ERDDAP servers (MODIS chlorophyll, SST, PAR, etc.)
- Fetches climate indices (PDO, ONI, BEUTI) and USGS streamflow data
- Processes shellfish DA/PN measurements from CSV files in `da-input/` and `pn-input/`
- Combines all sources into unified weekly time series with temporal buffers
- Outputs processed data to `final_output.parquet` and caches satellite data in `satellite_data_intermediate.parquet`

**Entry Points**
- `modular-forecast.py`: Main application coordinating retrospective evaluation and real-time forecasting
- `dataset-creation.py`: Complete data processing pipeline (standalone)

## Critical Temporal Safeguards

The system implements multiple layers of data leakage prevention:
- **Temporal splitting BEFORE feature engineering**: Training/test split occurs before any preprocessing
- **Minimum temporal buffers**: 7-day separation between training end and prediction date
- **Forward-only interpolation**: Missing values filled using only historical data
- **Per-forecast category creation**: DA risk categories computed independently for each forecast
- **Lag feature temporal cutoffs**: Historical lag features use strict temporal boundaries

These safeguards are essential for scientific validity - never modify them without understanding the implications.

## Common Commands

**Full Data Processing Pipeline**
```bash
python dataset-creation.py
```
Downloads all external data sources and processes local CSV files. Runtime: 30-60 minutes depending on satellite data volume. Set `FORCE_SATELLITE_REPROCESSING = True` in the file to regenerate cached satellite data.

**Retrospective Model Evaluation**
```bash
python modular-forecast.py
```
Runs retrospective evaluation with random anchor points and launches interactive dashboard on port 8071. Configure evaluation parameters in `config.py`.

**Real-time Forecasting Dashboard**
```bash
python modular-forecast.py  # with FORECAST_MODE = "realtime" in config.py
```
Launches forecasting dashboard for specific date/site predictions on port 8065. Supports model selection between XGBoost and Ridge Regression.

**Scientific Validation Suite**
```bash
python run_scientific_validation.py
```
Runs comprehensive scientific validation tests including temporal integrity, model performance, statistical analysis, and feature validation. Essential for research validation and peer review.

**Custom Analysis**
```bash
python xgboost_spectral_analysis.py  # from spectral analysis folder
```
Runs comprehensive spectral analysis of XGBoost predictions including coherence and phase analysis.

## Configuration System

**Primary Configuration (`config.py`)**
- `FORECAST_MODEL`: Primary ML model ("xgboost" or "ridge")
- `FORECAST_MODE`: Operating mode ("retrospective" or "realtime") 
- `FORECAST_TASK`: Prediction type ("regression" or "classification")
- `TEMPORAL_BUFFER_DAYS`: Minimum separation between training and prediction (default: 1)
- `N_RANDOM_ANCHORS`: Number of evaluation points for retrospective analysis
- `SITES`: Dictionary mapping site names to [latitude, longitude] coordinates
- `ORIGINAL_DA_FILES` / `ORIGINAL_PN_FILES`: Local CSV file paths for toxin measurements
- `SATELLITE_DATA`: Complex nested configuration for ERDDAP satellite data URLs

**Key Configuration Patterns**
- Date ranges use pandas datetime format
- Site coordinates are [latitude, longitude] arrays
- URLs contain templated parameters `{start_date}`, `{end_date}` for dynamic date insertion
- All temporal buffers account for real-world data reporting delays

## ML Model Architecture

**Current Model Hierarchy (after comprehensive evaluation)**
- **Primary**: XGBoost (7.4% better R² than Random Forest baseline)
- **Fallback**: Ridge Regression (linear baseline for regression tasks)
- **Classification**: XGBoost for categorical risk prediction, Logistic Regression fallback

**Model Factory Pattern**
The `ModelFactory` class creates configured models based on task type. It validates model/task combinations and provides model descriptions. Only XGBoost and Ridge/Logistic models remain after streamlining - Random Forest and Stacking Ensemble were removed.

**Feature Engineering**
- Temporal features: sin/cos day-of-year transformations
- Lag features: 1, 2, 3-period historical DA values with temporal cutoffs
- Environmental features: Satellite oceanographic parameters, climate indices, streamflow
- Preprocessing: MedianImputer + MinMaxScaler fitted only on training data

## Data Flow and Dependencies

**Processing Sequence**
1. Raw CSV files in `da-input/` and `pn-input/` directories contain historical toxin measurements
2. `dataset-creation.py` downloads external data and processes all sources into `final_output.parquet`
3. Modular forecasting system loads processed data for training and prediction
4. Dashboards provide interactive interfaces for analysis and forecasting

**External Dependencies**
- NOAA ERDDAP servers for satellite oceanographic data (requires internet connectivity)
- NOAA climate indices (PDO/ONI) and BEUTI upwelling data
- USGS streamflow data (Columbia River station 14246900)
- All external downloads are cached - satellite data in `satellite_data_intermediate.parquet`

**Site Coverage**
10 Pacific Coast monitoring locations: Kalaloch, Quinault, Copalis, Twin Harbors, Long Beach, Clatsop Beach, Cannon Beach, Newport, Coos Bay, Gold Beach.

## Development Patterns

**When modifying temporal logic**: Always verify that training data cannot access future information. The system uses anchor dates, temporal buffers, and forward-only processing to maintain scientific integrity.

**When adding new features**: Follow the pattern in `data_processor.py` with temporal cutoffs. New features must be computable using only historical data available at prediction time.

**When updating ML models**: Use the `ModelFactory` pattern and ensure new models support the same interface (fit/predict). Maintain compatibility with both regression and classification tasks.

**Configuration changes**: Modify `config.py` for system-wide settings. The modular architecture loads configuration centrally, so changes propagate throughout the system.

**Dashboard customization**: Both dashboards use Plotly/Dash. The retrospective dashboard analyzes historical results, while the realtime dashboard generates new predictions interactively.

## Testing and Validation

**Scientific Validation Framework**
The system includes a comprehensive scientific validation suite (`forecasting/core/scientific_validation.py`) with:
- `ScientificValidator` class for autocorrelation analysis, residual diagnostics, and imputation method comparison
- Temporal data leakage prevention tests
- Statistical significance testing and model diagnostics
- Feature importance and selection validation

**Running Validation Tests**
```bash
# Full validation suite
python run_scientific_validation.py

# Specific test types
python run_scientific_validation.py --tests temporal,performance,statistical

# With custom output directory
python run_scientific_validation.py --output-dir ./validation_results/
```

**No Formal Unit Tests**: The codebase does not use pytest or unittest frameworks. Validation relies on the scientific validation suite and manual testing of components.