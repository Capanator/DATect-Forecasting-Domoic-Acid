# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Domoic Acid (DA) forecasting system that predicts harmful algal bloom concentrations along the Pacific Coast. The system processes multiple data sources (satellite oceanographic data, climate indices, streamflow, and shellfish toxin measurements) to generate predictive models using machine learning techniques.

## Core Architecture

**Data Processing Pipeline (`dataset-creation.py`)**
- Fetches and processes satellite oceanographic data (MODIS chlorophyll, SST, PAR, fluorescence, K490)
- Downloads climate indices (PDO, ONI, BEUTI) and streamflow data from NOAA/USGS APIs  
- Processes shellfish DA/PN measurements from CSV files
- Combines all data sources into a unified weekly time series
- Outputs processed data to `final_output.parquet`

**Forecasting Systems**
- `past-forecasts-final.py`: Retrospective model evaluation using random anchor points
- `future-forecasts.py`: Real-time forecasting dashboard for specific dates/sites

**Configuration Files**
- `config.json`: Main data sources, site coordinates, date ranges, API endpoints
- `satellite_config.json`: ERDDAP URLs for satellite data by location and parameter

## Key Data Flow

1. **Raw Data**: DA/PN measurements in `da-input/` and `pn-input/` directories
2. **Processing**: `data-preprocessing.py` creates `final_output.parquet` and `satellite_data_intermediate.parquet`
3. **Modeling**: Forecasting scripts use the processed parquet files for ML training/prediction

## Site Coverage

The system monitors 10 Pacific Coast locations from Washington to Oregon:
- Kalaloch, Quinault, Copalis, Twin Harbors, Long Beach
- Clatsop Beach, Cannon Beach, Newport, Coos Bay, Gold Beach

## Common Commands

**Data Processing**
```bash
python data-preprocessing.py
```
This downloads all external data sources, processes CSV files, and generates the main dataset. Runtime: 30-60 minutes depending on satellite data volume.

**Model Evaluation** 
```bash
python past-forecasts-final.py
```
Runs retrospective evaluation with random anchor forecasting. Launches interactive dashboard on port 8071.

**Real-time Forecasting**
```bash
python future-forecasts.py  
```
Launches forecasting dashboard for specific date/site predictions on port 8065.

## Data Dependencies

External data sources are automatically downloaded:
- NOAA ERDDAP servers for satellite oceanographic data
- NOAA climate indices (PDO/ONI) 
- USGS streamflow data
- BEUTI upwelling index

Local CSV files in `da-input/` and `pn-input/` contain historical shellfish toxin measurements.

## Model Architecture

**Regression Models**: Random Forest and Gradient Boosting for continuous DA level prediction
**Classification Models**: Random Forest for categorical risk levels (Low/Moderate/High/Extreme)
**Features**: Oceanographic variables, climate indices, lag features, seasonal components

The system uses time series cross-validation and supports both quantile regression (uncertainty estimation) and point predictions.

## Configuration Notes

Set `FORCE_SATELLITE_REPROCESSING = True` in `data-preprocessing.py` to regenerate satellite data cache. Satellite data processing is the most time-intensive step and results are cached in `satellite_data_intermediate.parquet`.

Date ranges and site coordinates are configured in `config.json`. The system expects weekly data resolution and handles missing values through interpolation.