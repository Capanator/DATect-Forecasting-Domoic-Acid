"""
DATect Web Application API
=========================

FastAPI backend API for the Domoic Acid forecasting web application.
Provides REST API endpoints for forecasting, data visualization, and analysis.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import sys
import os
import pandas as pd
import numpy as np
import json
import asyncio
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add parent directory to path to import forecasting modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.model_factory import ModelFactory
import config
from backend.visualizations import (
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis,
    generate_gradient_uncertainty_plot
)
from backend.cache_manager import cache_manager

# Fix path resolution - ensure we use absolute paths relative to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)

# Ensure spectral analysis XGBoost is enabled for local development
if os.getenv("NODE_ENV") != "production" and os.getenv("CACHE_DIR") != "/app/cache":
    # Clear any disabling flag from cache precomputation for local development
    if 'SPECTRAL_ENABLE_XGB' in os.environ:
        del os.environ['SPECTRAL_ENABLE_XGB']

# Simple on-disk cache for heavy computations (ephemeral per instance)
# Use /tmp by default for Cloud Run writeable filesystem
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/datect-cache")

# Disable legacy cache for local development
LEGACY_CACHE_ENABLED = os.getenv("NODE_ENV") == "production" or os.getenv("CACHE_DIR") == "/app/cache"

if LEGACY_CACHE_ENABLED:
    os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(*parts: str) -> str:
    if not LEGACY_CACHE_ENABLED:
        return "/dev/null"  # Disabled for local development
    safe_parts = [p.replace("/", "-") for p in parts]
    return os.path.join(CACHE_DIR, "_".join(safe_parts)) + ".json"

def _read_cache(path: str):
    if not LEGACY_CACHE_ENABLED:
        return None  # Always miss for local development
    try:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _write_cache(path: str, obj: dict):
    if not LEGACY_CACHE_ENABLED:
        return  # No-op for local development
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

def _list_cache():
    summary = {"dir": CACHE_DIR, "files": [], "total_size": 0, "writable": os.access(CACHE_DIR, os.W_OK)}
    try:
        if os.path.isdir(CACHE_DIR):
            for root, _, files in os.walk(CACHE_DIR):
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        size = os.path.getsize(fp)
                    except Exception:
                        size = 0
                    summary["files"].append({"path": fp.replace(project_root + os.sep, ""), "size": size})
                    summary["total_size"] += size
    except Exception:
        pass
    return summary

def _clear_cache_internal():
    result = {"dir": CACHE_DIR, "deleted_files": 0, "freed_bytes": 0, "writable": os.access(CACHE_DIR, os.W_OK)}
    if not os.path.isdir(CACHE_DIR):
        return result
    if not result["writable"]:
        # Cannot delete read-only baked cache (e.g., inside container image)
        return result
    for root, _, files in os.walk(CACHE_DIR):
        for name in files:
            fp = os.path.join(root, name)
            try:
                size = os.path.getsize(fp)
                os.remove(fp)
                result["deleted_files"] += 1
                result["freed_bytes"] += size
            except Exception:
                pass
    return result

app = FastAPI(
    title="DATect API",
    description="Domoic Acid Forecasting System REST API",
    version="1.0.0"
)

# CORS middleware for frontend connections
# CORS middleware for frontend connections (configurable for production)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Default allows common local dev servers; for production when serving same-origin, CORS won't be used
    origins = ["http://localhost:3000", "http://localhost:5173", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Lazy singletons to avoid heavy validation at startup on Cloud Run
forecast_engine = None
model_factory = None

def get_forecast_engine() -> ForecastEngine:
    global forecast_engine
    if forecast_engine is None:
        # Skip heavy validation on init; endpoints will validate as needed
        forecast_engine = ForecastEngine(validate_on_init=False)
    return forecast_engine

def get_model_factory() -> ModelFactory:
    global model_factory
    if model_factory is None:
        model_factory = ModelFactory()
    return model_factory

# Model mapping function
def get_actual_model_name(ui_model: str, task: str) -> str:
    """Map UI model selection to actual model names based on task."""
    if ui_model == "xgboost":
        return "xgboost"  # XGBoost works for both regression and classification
    elif ui_model == "linear":
        if task == "regression":
            return "linear"  # Linear regression
        else:
            return "logistic"  # Logistic regression for classification
    else:
        return ui_model  # Fallback to original name

def generate_quantile_predictions(data_file, forecast_date, site, model_type="xgboost"):
    """Generate quantile predictions using Gradient Boosting and point prediction using XGBoost."""
    try:
        # Load and prepare data
        data = pd.read_parquet(data_file)
        data['date'] = pd.to_datetime(data['date'])
        
        # Filter for site
        site_data = data[data['site'] == site].copy()
        site_data.sort_values('date', inplace=True)
        
        # Split training and forecast data with temporal integrity
        train_data = site_data[site_data['date'] < forecast_date].copy()
        
        # Remove rows with missing target values
        train_data = train_data.dropna(subset=['da'])
        
        if len(train_data) < 5:  # Minimum samples check
            return None
            
        # Prepare features (exclude target and metadata columns)
        feature_cols = [col for col in train_data.columns 
                       if col not in ['date', 'site', 'da', 'da-category']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['da']
        
        # Handle missing values in features
        X_train = X_train.dropna()
        if len(X_train) < 5:
            return None
            
        # Align y_train with cleaned X_train
        y_train = y_train.loc[X_train.index]
        
        # Create forecast row (use last available data point as template)
        last_row = train_data.iloc[-1].copy()
        last_row['date'] = forecast_date
        forecast_data = pd.DataFrame([last_row])
        X_forecast = forecast_data[feature_cols]
        
        # Preprocessing pipeline
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ]), numeric_cols)
        ], remainder='drop')
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_forecast_processed = preprocessor.transform(X_forecast)
        
        # Generate quantile predictions using Gradient Boosting
        quantiles = {'q05': 0.05, 'q50': 0.50, 'q95': 0.95}
        gb_predictions = {}
        
        for name, alpha in quantiles.items():
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                loss='quantile',
                alpha=alpha,
                random_state=42
            )
            gb_model.fit(X_train_processed, y_train)
            gb_predictions[name] = float(gb_model.predict(X_forecast_processed)[0])
        
        # Generate point prediction using existing XGBoost engine
        xgb_result = get_forecast_engine().generate_single_forecast(
            data_file, forecast_date, site, "regression", model_type
        )
        
        xgb_prediction = xgb_result.get('predicted_da') if xgb_result else gb_predictions['q50']
        
        return {
            'quantile_predictions': gb_predictions,
            'point_prediction': xgb_prediction,
            'training_samples': len(train_data)
        }
        
    except Exception as e:
        print(f"Error in quantile prediction: {str(e)}")
        return None

# Pydantic models
class ForecastRequest(BaseModel):
    date: date
    site: str
    task: str = "regression"  # "regression" or "classification"
    model: str = "xgboost"

class ConfigUpdateRequest(BaseModel):
    forecast_mode: str = "realtime"  # "realtime" or "retrospective" 
    forecast_task: str = "regression"  # "regression" or "classification"
    forecast_model: str = "xgboost"  # "xgboost" or "linear" (linear models)
    selected_sites: List[str] = []  # For retrospective site filtering

class ForecastResponse(BaseModel):
    success: bool
    forecast_date: date
    site: str
    task: str
    model: str
    prediction: Optional[float] = None
    predicted_category: Optional[int] = None
    training_samples: Optional[int] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class SiteInfo(BaseModel):
    sites: List[str]
    date_range: Dict[str, date]
    site_mapping: Dict[str, str]

class ModelInfo(BaseModel):
    available_models: Dict[str, List[str]]
    descriptions: Dict[str, str]

class CacheClearResponse(BaseModel):
    success: bool
    message: str
    details: Dict[str, Any]

class CacheDeleteOneResponse(BaseModel):
    success: bool
    message: str
    target: Dict[str, Any]

    

@app.get("/api")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DATect API - Domoic Acid Forecasting System",
        "version": "1.0.0",
            "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/sites", response_model=SiteInfo)
async def get_sites():
    """Get available sites and date range from the dataset."""
    try:
        # Load data to get site information
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        data['date'] = pd.to_datetime(data['date'])
        
        sites = sorted(data['site'].unique().tolist())
        # Also provide lowercase versions for easier API access
        site_mapping = {site.lower().replace(' ', '-'): site for site in sites}
        date_range = {
            "min": data['date'].min().date(),
            "max": data['date'].max().date()
        }
        
        return SiteInfo(sites=sites, date_range=date_range, site_mapping=site_mapping)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load site information: {str(e)}")

@app.get("/api/models", response_model=ModelInfo)
async def get_models():
    """Get available models and their descriptions."""
    try:
        mf = get_model_factory()
        available_models = {
            "regression": mf.get_supported_models('regression')['regression'],
            "classification": mf.get_supported_models('classification')['classification']
        }
        
        # Get descriptions for all models
        descriptions = {}
        all_models = set(available_models["regression"] + available_models["classification"])
        for model in all_models:
            descriptions[model] = mf.get_model_description(model)
        
        return ModelInfo(available_models=available_models, descriptions=descriptions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model information: {str(e)}")

@app.post("/api/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate a forecast for the specified parameters."""
    try:
        # Validate inputs
        if request.task not in ["regression", "classification"]:
            raise HTTPException(status_code=400, detail="Task must be 'regression' or 'classification'")
        
        # Handle site name mapping for forecast
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        
        actual_site = request.site
        if request.site.lower() in site_mapping:
            actual_site = site_mapping[request.site.lower()]
        elif request.site in site_mapping.values():
            actual_site = request.site
        
        # Generate forecast using the existing engine with proper model mapping
        actual_model = get_actual_model_name(request.model, request.task)
        result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            request.task,
            actual_model
        )
        
        if result is None:
            return ForecastResponse(
                success=False,
                forecast_date=request.date,
                site=request.site,
                task=request.task,
                model=request.model,
                error="Insufficient data for forecast"
            )
        
        # Format response based on task type
        response_data = {
            "success": True,
            "forecast_date": request.date,
            "site": request.site,
            "task": request.task,
            "model": request.model,
            "training_samples": result.get('training_samples')
        }
        
        if request.task == "regression":
            response_data["prediction"] = result.get('predicted_da')
        else:  # classification
            response_data["predicted_category"] = result.get('predicted_category')
        
        # Add feature importance if available
        if 'feature_importance' in result and result['feature_importance'] is not None:
            # Convert to list of dicts for JSON serialization
            importance_df = result['feature_importance']
            if hasattr(importance_df, 'to_dict'):
                response_data["feature_importance"] = importance_df.head(10).to_dict('records')
        
        return ForecastResponse(**response_data)
        
    except Exception as e:
        return ForecastResponse(
            success=False,
            forecast_date=request.date,
            site=request.site,
            task=request.task,
            model=request.model,
            error=str(e)
        )

@app.get("/api/historical/{site}")
async def get_historical_data(
    site: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 1000
):
    """Get historical DA measurements for a site."""
    try:
        # Load data
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        data['date'] = pd.to_datetime(data['date'])
        
        # Create site mapping for flexible site name handling
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        
        # Handle site name mapping
        actual_site = site
        if site.lower() in site_mapping:
            actual_site = site_mapping[site.lower()]
        elif site in site_mapping.values():
            actual_site = site
        
        # Filter by site
        site_data = data[data['site'] == actual_site].copy()
        
        if site_data.empty:
            raise HTTPException(status_code=404, detail=f"Site '{site}' not found")
        
        # Apply date filters
        if start_date:
            site_data = site_data[site_data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            site_data = site_data[site_data['date'] <= pd.to_datetime(end_date)]
        
        # Limit results and sort by date
        site_data = site_data.sort_values('date').tail(limit)
        
        # Select relevant columns
        result_columns = ['date', 'da']
        if 'da-category' in site_data.columns:
            result_columns.append('da-category')
        
        result_data = site_data[result_columns].copy()
        result_data['date'] = result_data['date'].dt.strftime('%Y-%m-%d')
        
        return {
            "site": site,
            "count": len(result_data),
            "data": result_data.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load historical data: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current system configuration."""
    return {
        "forecast_mode": getattr(config, 'FORECAST_MODE', 'realtime'),
        "forecast_task": getattr(config, 'FORECAST_TASK', 'regression'),
        "forecast_model": getattr(config, 'FORECAST_MODEL', 'xgboost')
    }

@app.post("/api/config")
async def update_config(config_request: ConfigUpdateRequest):
    """Update system configuration and write to config.py file."""
    try:
        # Update in-memory config values
        config.FORECAST_MODE = config_request.forecast_mode
        config.FORECAST_TASK = config_request.forecast_task  
        config.FORECAST_MODEL = config_request.forecast_model
        
        # Write changes to config.py file
        config_file_path = os.path.join(project_root, 'config.py')
        
        # Read current config.py
        with open(config_file_path, 'r') as f:
            config_content = f.read()
        
        # Update the specific lines
        import re
        config_content = re.sub(
            r'FORECAST_MODE = ".*?"',
            f'FORECAST_MODE = "{config_request.forecast_mode}"',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_TASK = ".*?"',
            f'FORECAST_TASK = "{config_request.forecast_task}"',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_MODEL = ".*?"',
            f'FORECAST_MODEL = "{config_request.forecast_model}"',
            config_content
        )
        
        # Write back to file
        with open(config_file_path, 'w') as f:
            f.write(config_content)
        
        return {
            "success": True,
            "message": "Configuration updated successfully in config.py",
            "config": {
                "forecast_mode": config.FORECAST_MODE,
                "forecast_task": config.FORECAST_TASK,
                "forecast_model": config.FORECAST_MODEL
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/historical/all")
async def get_all_sites_historical(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: Optional[int] = 1000
):
    """Get historical data for all sites."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        # Apply date filters
        if start_date:
            data = data[pd.to_datetime(data['date']) >= pd.to_datetime(start_date)]
        if end_date:
            data = data[pd.to_datetime(data['date']) <= pd.to_datetime(end_date)]
        
        # Sort by date and site
        data = data.sort_values(['site', 'date'])
        
        # Limit results
        if limit:
            data = data.head(limit)
        
        # Convert to dict for JSON response
        result = []
        for _, row in data.iterrows():
            result.append({
                'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                'da': float(row['da']) if pd.notna(row['da']) else None,
                'site': row['site'],
                'da-category': int(row['da-category']) if 'da-category' in row and pd.notna(row['da-category']) else None
            })
        
        return {
            "success": True,
            "data": result,
            "count": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical data: {str(e)}")

# Visualization endpoints
# NOTE: More specific routes must come before generic parameter routes
@app.get("/api/visualizations/correlation/all")
async def get_correlation_heatmap_all():
    """Generate correlation heatmap for all sites combined."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plot_data = generate_correlation_heatmap(data, site=None)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/correlation/{site}")
async def get_correlation_heatmap_single(site: str):
    """Generate correlation heatmap for a single site."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        # Handle site name mapping
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        actual_site = site_mapping.get(site.lower(), site)
        
        plot_data = generate_correlation_heatmap(data, actual_site)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/sensitivity")
async def get_sensitivity_analysis():
    """Generate sensitivity analysis plots."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plots = generate_sensitivity_analysis(data)
        return {"success": True, "plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/comparison/all")
async def get_time_series_comparison_all():
    """Generate time series comparison for all sites."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plot_data = generate_time_series_comparison(data, site=None)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate time series comparison: {str(e)}")

@app.get("/api/visualizations/comparison/{site}")
async def get_time_series_comparison_single(site: str):
    """Generate time series comparison for a single site."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        # Handle site name mapping
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        actual_site = site_mapping.get(site.lower(), site)
        
        plot_data = generate_time_series_comparison(data, actual_site)
        return {"success": True, "plot": plot_data}  # plot_data is already in correct format
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate time series comparison: {str(e)}")

@app.get("/api/visualizations/waterfall")
async def get_waterfall_plot():
    """Generate waterfall plot for all sites."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plot_data = generate_waterfall_plot(data)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate waterfall plot: {str(e)}")

@app.get("/api/visualizations/spectral/all")
async def get_spectral_analysis_all():
    """Generate spectral analysis for all sites combined (uses pre-computed cache)."""
    try:
        # First try pre-computed cache
        plots = cache_manager.get_spectral_analysis(site=None)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}
            
        # Fallback to legacy cache
        cache_file = _cache_path("spectral", "all")
        cached = _read_cache(cache_file)
        if cached and isinstance(cached.get("plots"), list):
            return {"success": True, "plots": cached["plots"], "cached": True, "source": "legacy"}

        # Last resort: compute on server (expensive)
        print("⚠️ WARNING: Computing spectral analysis on server - this is very expensive!")
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plots = generate_spectral_analysis(data, site=None)
        _write_cache(cache_file, {"plots": plots})
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

@app.get("/api/visualizations/spectral/{site}")
async def get_spectral_analysis_single(site: str):
    """Generate spectral analysis for a single site (uses pre-computed cache)."""
    try:
        # Handle site name mapping first
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        actual_site = site_mapping.get(site.lower(), site)

        # First try pre-computed cache
        plots = cache_manager.get_spectral_analysis(site=actual_site)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Fallback to legacy cache
        cache_key = actual_site.lower().replace(" ", "-")
        cache_file = _cache_path("spectral", cache_key)
        cached = _read_cache(cache_file)
        if cached and isinstance(cached.get("plots"), list):
            return {"success": True, "plots": cached["plots"], "cached": True, "source": "legacy"}

        # Last resort: compute on server (expensive)
        print(f"⚠️ WARNING: Computing spectral analysis for {actual_site} on server - this is very expensive!")
        plots = generate_spectral_analysis(data, actual_site)
        _write_cache(cache_file, {"plots": plots})
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

@app.get("/api/cache")
async def get_cache_status():
    """Return cache directory, file list, size, and writability."""
    legacy_cache = _list_cache()
    precomputed_cache = cache_manager.get_cache_status()
    
    return {
        "legacy_cache": legacy_cache,
        "precomputed_cache": precomputed_cache,
        "available_forecasts": cache_manager.list_available_forecasts(),
        "available_spectral": cache_manager.list_available_spectral()
    }

@app.delete("/api/cache", response_model=CacheClearResponse)
async def clear_cache():
    """Clear writable cache files (no-op if cache is read-only)."""
    details = _clear_cache_internal()
    if not details.get("writable", False):
        return CacheClearResponse(success=False, message="Cache directory is read-only; cannot clear.", details=details)
    return CacheClearResponse(success=True, message="Cache cleared.", details=details)

@app.delete("/api/cache/retrospective", response_model=CacheDeleteOneResponse)
async def delete_retrospective_cache(task: str, model: str):
    """Delete cached retrospective results for a specific (task, model)."""
    actual_model = get_actual_model_name(model, task)
    target_file = _cache_path("retrospective", task, actual_model)
    writable = os.access(CACHE_DIR, os.W_OK)
    if not writable:
        return CacheDeleteOneResponse(success=False, message="Cache directory is read-only; cannot delete.", target={"file": target_file})
    if os.path.isfile(target_file):
        try:
            os.remove(target_file)
            return CacheDeleteOneResponse(success=True, message="Retrospective cache deleted.", target={"file": target_file})
        except Exception as e:
            return CacheDeleteOneResponse(success=False, message=f"Failed to delete: {e}", target={"file": target_file})
    else:
        return CacheDeleteOneResponse(success=False, message="Cache file not found.", target={"file": target_file})

@app.delete("/api/cache/spectral", response_model=CacheDeleteOneResponse)
async def delete_spectral_cache(site: str = "all"):
    """Delete cached spectral analysis for a specific site or all-sites."""
    key = "all" if site.lower() in ("all", "*") else site.lower().replace(" ", "-")
    target_file = _cache_path("spectral", key)
    writable = os.access(CACHE_DIR, os.W_OK)
    if not writable:
        return CacheDeleteOneResponse(success=False, message="Cache directory is read-only; cannot delete.", target={"file": target_file})
    if os.path.isfile(target_file):
        try:
            os.remove(target_file)
            return CacheDeleteOneResponse(success=True, message="Spectral cache deleted.", target={"file": target_file})
        except Exception as e:
            return CacheDeleteOneResponse(success=False, message=f"Failed to delete: {e}", target={"file": target_file})
    else:
        return CacheDeleteOneResponse(success=False, message="Cache file not found.", target={"file": target_file})

@app.post("/api/visualizations/spectral/warm")
async def warm_spectral_caches_disabled():
    raise HTTPException(status_code=405, detail="Server-side spectral warm disabled. Precompute locally and bake into image.")

@app.post("/api/visualizations/gradient")
async def get_gradient_uncertainty_visualization(request: ForecastRequest):
    """Generate gradient uncertainty visualization with quantile regression."""
    try:
        # Handle site name mapping
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        
        actual_site = request.site
        if request.site.lower() in site_mapping:
            actual_site = site_mapping[request.site.lower()]
        elif request.site in site_mapping.values():
            actual_site = request.site
        
        # Generate quantile predictions
        quantile_result = generate_quantile_predictions(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            get_actual_model_name(request.model, "regression")
        )
        
        if not quantile_result:
            raise HTTPException(status_code=400, detail="Insufficient data for quantile prediction")
        
        gb_preds = quantile_result['quantile_predictions']
        xgb_pred = quantile_result['point_prediction']
        
        # Generate gradient visualization plot
        gradient_plot = generate_gradient_uncertainty_plot(
            gb_preds, xgb_pred, actual_da=None
        )
        
        return {
            "success": True,
            "plot": gradient_plot,
            "quantile_predictions": gb_preds,
            "xgboost_prediction": xgb_pred,
            "training_samples": quantile_result['training_samples']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate gradient visualization: {str(e)}")

@app.post("/api/forecast/enhanced")
async def generate_enhanced_forecast(request: ForecastRequest):
    """Generate enhanced forecast with both regression and classification, plus all graph data."""
    try:
        # Handle site name mapping for forecast
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = {s.lower().replace(' ', '-'): s for s in data['site'].unique()}
        
        actual_site = request.site
        if request.site.lower() in site_mapping:
            actual_site = site_mapping[request.site.lower()]
        elif request.site in site_mapping.values():
            actual_site = request.site
        
        # Generate both regression and classification forecasts with proper model mapping
        regression_model = get_actual_model_name(request.model, "regression")
        regression_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            "regression",
            regression_model
        )
        
        
        # For classification, use the mapped model
        classification_model = get_actual_model_name(request.model, "classification")
        classification_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            "classification",
            classification_model
        )
        
        # Helper function to make results JSON serializable
        def clean_result(result):
            if not result:
                return result
            cleaned = {}
            for key, value in result.items():
                if value is None:
                    cleaned[key] = None
                elif isinstance(value, np.ndarray):
                    cleaned[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    cleaned[key] = int(value)
                elif isinstance(value, (np.float64, np.float32, np.float16)):
                    cleaned[key] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    cleaned[key] = bool(value)
                elif isinstance(value, pd.DataFrame):
                    # Convert DataFrame to dict or limit to top entries
                    if key == 'feature_importance':
                        # For feature importance, convert to list of dicts (top 10)
                        cleaned[key] = value.head(10).to_dict('records') if not value.empty else None
                    else:
                        cleaned[key] = value.to_dict('records') if not value.empty else None
                elif hasattr(value, 'isoformat'):
                    cleaned[key] = value.isoformat()
                elif hasattr(value, 'item'):  # Handle numpy scalars
                    try:
                        cleaned[key] = value.item()
                    except:
                        cleaned[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    # Recursively clean lists/tuples
                    cleaned[key] = [clean_result({'item': item})['item'] if isinstance(item, (np.ndarray, np.integer, np.floating)) else item for item in value]
                elif isinstance(value, dict):
                    # Recursively clean nested dicts
                    cleaned[key] = clean_result(value)
                else:
                    cleaned[key] = value
            return cleaned
        
        # Create enhanced response with graph data
        response_data = {
            "success": True,
            "forecast_date": request.date,
            "site": actual_site,
            "regression": clean_result(regression_result),
            "classification": clean_result(classification_result),
            "graphs": {}
        }
        
        # Add advanced quantile-based uncertainty using Gradient Boosting + XGBoost
        if regression_result and 'predicted_da' in regression_result:
            # Generate quantile predictions using dual-model approach
            quantile_result = generate_quantile_predictions(
                config.FINAL_OUTPUT_PATH,
                pd.to_datetime(request.date),
                actual_site,
                get_actual_model_name(request.model, "regression")
            )
            
            if quantile_result:
                gb_preds = quantile_result['quantile_predictions']
                xgb_pred = quantile_result['point_prediction']
                
                # Generate advanced gradient visualization plot
                gradient_plot = generate_gradient_uncertainty_plot(
                    gb_preds, xgb_pred, actual_da=None
                )
                
                response_data["graphs"]["level_range"] = {
                    "gradient_quantiles": {
                        "q05": float(gb_preds['q05']),
                        "q50": float(gb_preds['q50']),
                        "q95": float(gb_preds['q95'])
                    },
                    "xgboost_prediction": float(xgb_pred),
                    "gradient_plot": gradient_plot,
                    "type": "gradient_uncertainty"
                }
            else:
                # Fallback to simple approach if quantile generation fails
                predicted_da = float(regression_result['predicted_da'])
                fallback_quantiles = {
                    "q05": predicted_da * 0.7,
                    "q50": predicted_da,
                    "q95": predicted_da * 1.3
                }
                
                # Generate gradient plot with fallback data
                gradient_plot = generate_gradient_uncertainty_plot(
                    fallback_quantiles, predicted_da, actual_da=None
                )
                
                response_data["graphs"]["level_range"] = {
                    "gradient_quantiles": fallback_quantiles,
                    "xgboost_prediction": float(predicted_da),
                    "gradient_plot": gradient_plot,
                    "type": "gradient_uncertainty"
                }
        
        # Add category graph data for classification  
        if classification_result and 'predicted_category' in classification_result:
            # Convert numpy arrays to Python lists for JSON serialization
            class_probs = classification_result.get('class_probabilities')
            if isinstance(class_probs, np.ndarray):
                class_probs = class_probs.tolist()
            
            response_data["graphs"]["category_range"] = {
                "predicted_category": int(classification_result['predicted_category']),
                "class_probabilities": class_probs,
                "category_labels": ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                "type": "category_range"
            }
        else:
            # Fallback: Create category graph from regression prediction if classification failed
            if regression_result and 'predicted_da' in regression_result:
                try:
                    from forecasting.core.data_processor import DataProcessor
                    
                    predicted_da = float(regression_result['predicted_da'])
                    data_processor = DataProcessor()
                    
                    # Create categories from the predicted DA value  
                    predicted_category_series = data_processor.create_da_categories_safe(pd.Series([predicted_da]))
                    if not predicted_category_series.isna().iloc[0]:
                        predicted_category = int(predicted_category_series.iloc[0])
                        
                        # Create fallback probabilities (high confidence for predicted category)
                        fallback_probs = [0.1, 0.1, 0.1, 0.1]  
                        fallback_probs[predicted_category] = 0.7
                        
                        response_data["graphs"]["category_range"] = {
                            "predicted_category": predicted_category,
                            "class_probabilities": fallback_probs,
                            "category_labels": ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                            "type": "category_range_fallback"
                        }
                except Exception as fallback_error:
                    # If fallback fails, at least provide basic category info
                    predicted_da = float(regression_result['predicted_da'])
                    if predicted_da <= 5:
                        predicted_category = 0
                    elif predicted_da <= 20:
                        predicted_category = 1
                    elif predicted_da <= 40:
                        predicted_category = 2
                    else:
                        predicted_category = 3
                        
                    fallback_probs = [0.25, 0.25, 0.25, 0.25]
                    fallback_probs[predicted_category] = 0.7
                    
                    response_data["graphs"]["category_range"] = {
                        "predicted_category": predicted_category,
                        "class_probabilities": fallback_probs,
                        "category_labels": ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                        "type": "category_range_manual"
                    }
        
        return response_data
        
    except Exception as e:
        return {
            "success": False,
            "forecast_date": request.date,
            "site": request.site,
            "error": str(e)
        }

class RetrospectiveRequest(BaseModel):
    selected_sites: List[str] = []  # Empty list means all sites

# Removed streaming progress functionality

@app.post("/api/retrospective")
async def run_retrospective_analysis(request: RetrospectiveRequest = RetrospectiveRequest()):
    """Run complete retrospective analysis based on current config (uses pre-computed cache for production)."""
    try:
        # Get actual model name for caching
        actual_model = get_actual_model_name(config.FORECAST_MODEL, config.FORECAST_TASK)
        
        # First try to get from pre-computed cache (for production)
        base_results = cache_manager.get_retrospective_forecast(config.FORECAST_TASK, actual_model)
        
        if base_results is None:
            # Fallback to old caching system and compute on-demand (for development)
            cache_file = _cache_path("retrospective", config.FORECAST_TASK, actual_model)
            cached = _read_cache(cache_file)
            
            if cached and isinstance(cached.get("results"), list):
                base_results = cached["results"]
            else:
                # Last resort: compute on server (expensive - not recommended for production)
                print(f"⚠️ WARNING: Computing retrospective analysis on server - this is expensive!")
                engine = get_forecast_engine()
                engine.data_file = config.FINAL_OUTPUT_PATH
                
                results_df = engine.run_retrospective_evaluation(
                    task=config.FORECAST_TASK,
                    model_type=actual_model,
                    n_anchors=getattr(config, 'N_RANDOM_ANCHORS', 30)
                )

                if results_df is None or results_df.empty:
                    return {"success": False, "error": "No results generated from retrospective analysis"}

                # Convert results to JSON format
                base_results = []
                for _, row in results_df.iterrows():
                    record = {
                        "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
                        "site": row['site'],
                        "actual_da": float(row['da']) if pd.notnull(row['da']) else None,
                        "predicted_da": float(row['Predicted_da']) if 'Predicted_da' in row and pd.notnull(row['Predicted_da']) else None,
                        "actual_category": int(row['da-category']) if 'da-category' in row and pd.notnull(row['da-category']) else None,
                        "predicted_category": int(row['Predicted_da-category']) if 'Predicted_da-category' in row and pd.notnull(row['Predicted_da-category']) else None
                    }
                    base_results.append(record)

                # Cache for future requests
                full_summary = _compute_summary(base_results)
                _write_cache(cache_file, {
                    "results": base_results,
                    "summary": full_summary,
                    "config": {
                        "forecast_task": config.FORECAST_TASK,
                        "forecast_model": config.FORECAST_MODEL
                    }
                })
        else:
            print(f"✅ Serving pre-computed retrospective analysis: {config.FORECAST_TASK}+{actual_model}")

        # Filter by selected sites if provided
        if request.selected_sites:
            filtered = [r for r in base_results if r['site'] in request.selected_sites]
        else:
            filtered = base_results

        summary = _compute_summary(filtered)
        return {
            "success": True,
            "config": {
                "forecast_mode": config.FORECAST_MODE,
                "forecast_task": config.FORECAST_TASK,
                "forecast_model": config.FORECAST_MODEL
            },
            "summary": summary,
            "results": filtered
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def _compute_summary(results_json: list) -> dict:
    """Compute summary metrics for retrospective results list."""
    summary = {"total_forecasts": len(results_json)}
    valid_regression = [(r['actual_da'], r['predicted_da']) for r in results_json 
                        if r.get('actual_da') is not None and r.get('predicted_da') is not None]
    valid_classification = [(r['actual_category'], r['predicted_category']) for r in results_json 
                            if r.get('actual_category') is not None and r.get('predicted_category') is not None]
    summary["regression_forecasts"] = len(valid_regression)
    summary["classification_forecasts"] = len(valid_classification)
    if valid_regression:
        from sklearn.metrics import r2_score, mean_absolute_error
        actual_vals = [r[0] for r in valid_regression]
        pred_vals = [r[1] for r in valid_regression]
        try:
            summary["r2_score"] = float(r2_score(actual_vals, pred_vals))
            summary["mae"] = float(mean_absolute_error(actual_vals, pred_vals))
        except Exception:
            pass
    if valid_classification:
        from sklearn.metrics import accuracy_score
        actual_cats = [r[0] for r in valid_classification]
        pred_cats = [r[1] for r in valid_classification]
        try:
            summary["accuracy"] = float(accuracy_score(actual_cats, pred_cats))
        except Exception:
            pass
    return summary

# Serve built frontend if present (single-origin deploy) even when running via `uvicorn backend.api:app`
try:
    frontend_dist = os.path.join(project_root, "frontend", "dist")
    if os.path.isdir(frontend_dist):
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))