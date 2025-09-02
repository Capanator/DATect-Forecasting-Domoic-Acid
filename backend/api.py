"""
DATect Web Application API
FastAPI backend providing forecasting, visualization, and analysis endpoints
"""

import logging
import math
import os
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel


from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
import config
from backend.visualizations import (
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis
)
from backend.cache_manager import cache_manager

# Configure logging
logger = logging.getLogger(__name__)

def clean_float_for_json(value):
    """Handle inf/nan values for JSON serialization"""
    if value is None or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)


app = FastAPI(
    title="DATect API",
    description="Domoic Acid Forecasting System REST API",
    version="1.0.0"
)

# CORS middleware (configurable for production)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Default for local dev; production uses same-origin
    origins = ["http://localhost:3000", "http://localhost:5173", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy singletons for Cloud Run optimization
forecast_engine = None
model_factory = None

def get_forecast_engine() -> ForecastEngine:
    global forecast_engine
    if forecast_engine is None:
        # Skip validation on init for faster startup
        forecast_engine = ForecastEngine(validate_on_init=False)
    return forecast_engine

def get_model_factory() -> ModelFactory:
    global model_factory
    if model_factory is None:
        model_factory = ModelFactory()
    return model_factory

def get_site_mapping(data):
    """Get site name mapping for flexible API access"""
    return {s.lower().replace(' ', '-'): s for s in data['site'].unique()}

def resolve_site_name(site: str, site_mapping: dict) -> str:
    """Resolve site name from mapping or return original"""
    return site_mapping.get(site.lower(), site)


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
    forecast_horizon_weeks: int = 1  # Weeks ahead to forecast from data cutoff

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


class RetrospectiveRequest(BaseModel):
    selected_sites: List[str] = []  # Empty list means all sites


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
        if request.task not in ["regression", "classification"]:
            raise HTTPException(status_code=400, detail="Task must be 'regression' or 'classification'")
        
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(request.site, site_mapping)
        
        forecast_date = pd.to_datetime(request.date)
        
        result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            forecast_date,
            actual_site,
            request.task,
            "xgboost"
        )
        
        if result is None:
            return ForecastResponse(
                success=False,
                forecast_date=forecast_date.date(),
                site=request.site,
                task=request.task,
                model=request.model,
                error="Insufficient data for forecast"
            )
        
        # Format response based on task type
        response_data = {
            "success": True,
            "forecast_date": forecast_date.date(),
            "site": request.site,
            "task": request.task,
            "model": request.model,
            "training_samples": result.get('training_samples')
        }
        
        if request.task == "regression":
            response_data["prediction"] = result.get('predicted_da')
        else:  # classification
            response_data["predicted_category"] = result.get('predicted_category')
        
        if 'feature_importance' in result and result['feature_importance'] is not None:
            importance_df = result['feature_importance']
            if hasattr(importance_df, 'to_dict'):
                response_data["feature_importance"] = importance_df.head(10).to_dict('records')
        
        return ForecastResponse(**response_data)
        
    except Exception as e:
        # Use request date if forecast_date wasn't calculated due to early error
        error_date = request.date
        if 'forecast_date' in locals():
            error_date = forecast_date.date()
        
        return ForecastResponse(
            success=False,
            forecast_date=error_date,
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
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
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
        "forecast_model": getattr(config, 'FORECAST_MODEL', 'xgboost'),
        "forecast_horizon_weeks": getattr(config, 'FORECAST_HORIZON_WEEKS', 1),
        "forecast_horizon_days": getattr(config, 'FORECAST_HORIZON_DAYS', 7)
    }

@app.post("/api/config")
async def update_config(config_request: ConfigUpdateRequest):
    """Update system configuration and write to config.py file."""
    try:
        # Update in-memory config values
        config.FORECAST_MODE = config_request.forecast_mode
        config.FORECAST_TASK = config_request.forecast_task  
        config.FORECAST_MODEL = config_request.forecast_model
        config.FORECAST_HORIZON_WEEKS = config_request.forecast_horizon_weeks
        config.FORECAST_HORIZON_DAYS = config_request.forecast_horizon_weeks * 7
        
        # Write changes to config.py file
        config_file_path = os.path.join(project_root, 'config.py')
        
        # Read current config.py
        with open(config_file_path, 'r') as f:
            config_content = f.read()
        
        # Update the specific lines
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
        config_content = re.sub(
            r'FORECAST_HORIZON_WEEKS = \d+',
            f'FORECAST_HORIZON_WEEKS = {config_request.forecast_horizon_weeks}',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_HORIZON_DAYS = .*',
            f'FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7  # Derived days value for internal calculations',
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
        
        # Convert to dict for JSON response with proper float cleaning
        result = []
        for _, row in data.iterrows():
            da_val = row['da'] if pd.notna(row['da']) else None
            da_category = row['da-category'] if 'da-category' in row and pd.notna(row['da-category']) else None
            
            result.append({
                'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                'da': clean_float_for_json(da_val),
                'site': row['site'],
                'da-category': clean_float_for_json(da_category)
            })
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "count": len(result)
        })
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
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
        plot_data = generate_correlation_heatmap(data, actual_site)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/sensitivity/all")
async def get_sensitivity_analysis_all():
    """Generate sensitivity analysis plots for all sites combined."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plots = generate_sensitivity_analysis(data, site=None)
        return {"success": True, "plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/sensitivity/{site}")
async def get_sensitivity_analysis_single(site: str):
    """Generate sensitivity analysis plots for a single site."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
        plots = generate_sensitivity_analysis(data, actual_site)
        return {"success": True, "plots": plots}
    except Exception as e:
        logging.error(f"Error in sensitivity analysis: {str(e)}")
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
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
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

        # Compute on server (expensive - only for local development)
        logging.warning("Computing spectral analysis on server - this is very expensive!")
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        plots = generate_spectral_analysis(data, site=None)
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

@app.get("/api/visualizations/spectral/{site}")
async def get_spectral_analysis_single(site: str):
    """Generate spectral analysis for a single site (uses pre-computed cache)."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)

        # First try pre-computed cache
        plots = cache_manager.get_spectral_analysis(site=actual_site)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Compute on server (expensive - only for local development)
        logging.warning(f"Computing spectral analysis for {actual_site} on server - this is very expensive!")
        plots = generate_spectral_analysis(data, actual_site)
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")


@app.post("/api/visualizations/spectral/warm")
async def warm_spectral_caches_disabled():
    raise HTTPException(status_code=405, detail="Server-side spectral warm disabled. Precompute locally and bake into image.")

@app.post("/api/forecast/enhanced")
async def generate_enhanced_forecast(request: ForecastRequest):
    """Generate enhanced forecast with both regression and classification for frontend."""
    try:
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(request.site, site_mapping)
        
        forecast_date = pd.to_datetime(request.date)
        
        # Generate both regression and classification forecasts
        regression_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH, forecast_date, actual_site, "regression", "xgboost"
        )
        
        classification_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH, forecast_date, actual_site, "classification", "xgboost"
        )
        
        # Clean numpy values for JSON serialization
        def clean_numpy_values(obj):
            if isinstance(obj, dict):
                return {k: clean_numpy_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_numpy_values(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, pd.DataFrame):
                return obj.head(10).to_dict('records') if not obj.empty else []
            elif obj is None:
                return None
            else:
                try:
                    if pd.isna(obj):
                        return None
                except (TypeError, ValueError):
                    pass
                return obj
        
        # Create response structure expected by frontend
        response_data = {
            "success": True,
            "forecast_date": forecast_date.strftime('%Y-%m-%d'),
            "site": actual_site,
            "regression": clean_numpy_values(regression_result),
            "classification": clean_numpy_values(classification_result),
            "graphs": {}
        }
        
        # Add level_range graph for regression
        if regression_result and 'predicted_da' in regression_result:
            predicted_da = float(regression_result['predicted_da'])
            
            # Simple uncertainty bounds (can be enhanced later)
            response_data["graphs"]["level_range"] = {
                "gradient_quantiles": {
                    "q05": predicted_da * 0.7,
                    "q50": predicted_da,
                    "q95": predicted_da * 1.3
                },
                "xgboost_prediction": predicted_da,
                "type": "simple_uncertainty"
            }
        
        # Add category_range graph for classification
        if classification_result and 'predicted_category' in classification_result:
            class_probs = classification_result.get('class_probabilities', [0.25, 0.25, 0.25, 0.25])
            if isinstance(class_probs, np.ndarray):
                class_probs = class_probs.tolist()
            
            response_data["graphs"]["category_range"] = {
                "predicted_category": int(classification_result['predicted_category']),
                "class_probabilities": class_probs,
                "category_labels": ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                "type": "category_range"
            }
        
        return response_data
        
    except Exception as e:
        return {
            "success": False,
            "forecast_date": request.date,
            "site": request.site,
            "error": str(e)
        }



@app.post("/api/retrospective")
async def run_retrospective_analysis(request: RetrospectiveRequest = RetrospectiveRequest()):
    """Run complete retrospective analysis based on current config (uses pre-computed cache for production)."""
    try:
        # Map model names for API compatibility
        if config.FORECAST_MODEL == "linear":
            actual_model = "linear" if config.FORECAST_TASK == "regression" else "logistic"
        else:
            actual_model = config.FORECAST_MODEL
        
        # First try to get from pre-computed cache (for production)
        base_results = cache_manager.get_retrospective_forecast(config.FORECAST_TASK, actual_model)
        
        if base_results is None:
            # Compute on server (expensive - only for local development)
            logging.warning("Computing retrospective analysis on server - this is expensive!")
            engine = get_forecast_engine()
            engine.data_file = config.FINAL_OUTPUT_PATH
            
            results_df = engine.run_retrospective_evaluation(
                task=config.FORECAST_TASK,
                model_type=actual_model,
                n_anchors=getattr(config, 'N_RANDOM_ANCHORS', 30)
            )

            if results_df is None or results_df.empty:
                return {"success": False, "error": "No results generated from retrospective analysis"}

            # Convert results to JSON format with proper float cleaning
            base_results = []
            for _, row in results_df.iterrows():
                record = {
                    "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
                    "site": row['site'],
                    "actual_da": clean_float_for_json(row['da']) if pd.notnull(row['da']) else None,
                    "predicted_da": clean_float_for_json(row['Predicted_da']) if 'Predicted_da' in row and pd.notnull(row['Predicted_da']) else None,
                    "actual_category": clean_float_for_json(row['da-category']) if 'da-category' in row and pd.notnull(row['da-category']) else None,
                    "predicted_category": clean_float_for_json(row['Predicted_da-category']) if 'Predicted_da-category' in row and pd.notnull(row['Predicted_da-category']) else None
                }
                base_results.append(record)

            # Results computed on-demand for local development
        else:
            logging.info(f"Serving pre-computed retrospective analysis: {config.FORECAST_TASK}+{actual_model}")

        # Normalize cached data format
        if base_results and isinstance(base_results, list):
            for result in base_results:
                if 'da' in result and 'actual_da' not in result:
                    result['actual_da'] = result.get('da')
                if 'Predicted_da' in result and 'predicted_da' not in result:
                    result['predicted_da'] = result.get('Predicted_da')
                if 'da-category' in result and 'actual_category' not in result:
                    result['actual_category'] = result.get('da-category')
                if 'Predicted_da-category' in result and 'predicted_category' not in result:
                    result['predicted_category'] = result.get('Predicted_da-category')

        # Filter by sites if specified
        filtered = [r for r in base_results if r['site'] in request.selected_sites] if request.selected_sites else base_results
        
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
    """Compute summary metrics for retrospective results."""
    summary = {"total_forecasts": len(results_json)}
    
    # Get valid pairs for regression and classification
    # Handle both API format (actual_da, predicted_da) and cached format (da, Predicted_da)
    valid_regression = []
    valid_classification = []
    
    for r in results_json:
        # Regression pairs - try both formats
        actual_da = r.get('actual_da') or r.get('da')
        predicted_da = r.get('predicted_da') or r.get('Predicted_da')
        if actual_da is not None and predicted_da is not None:
            valid_regression.append((actual_da, predicted_da))
        
        # Classification pairs - try both formats  
        actual_cat = r.get('actual_category') or r.get('da-category')
        predicted_cat = r.get('predicted_category') or r.get('Predicted_da-category')
        if actual_cat is not None and predicted_cat is not None:
            valid_classification.append((actual_cat, predicted_cat))
    
    summary["regression_forecasts"] = len(valid_regression)
    summary["classification_forecasts"] = len(valid_classification)
    
    # Regression metrics
    if valid_regression:
        from sklearn.metrics import r2_score, mean_absolute_error, f1_score
        actual_vals = [r[0] for r in valid_regression]
        pred_vals = [r[1] for r in valid_regression]
        try:
            summary["r2_score"] = float(r2_score(actual_vals, pred_vals))
            summary["mae"] = float(mean_absolute_error(actual_vals, pred_vals))
            
            # F1 score for spike detection (15 μg/g threshold)
            spike_threshold = 15.0
            actual_binary = [1 if val > spike_threshold else 0 for val in actual_vals]
            pred_binary = [1 if val > spike_threshold else 0 for val in pred_vals]
            summary["f1_score"] = float(f1_score(actual_binary, pred_binary, zero_division=0))
        except Exception:
            pass
    
    # Classification metrics
    if valid_classification:
        from sklearn.metrics import accuracy_score
        actual_cats = [r[0] for r in valid_classification]
        pred_cats = [r[1] for r in valid_classification]
        try:
            summary["accuracy"] = float(accuracy_score(actual_cats, pred_cats))
        except Exception as e:
            logging.error(f"Error calculating classification metrics: {e}")
    
    return summary

# Serve built frontend if present (single-origin deploy)
try:
    frontend_dist = os.path.join(project_root, "frontend", "dist")
    if os.path.isdir(frontend_dist):
        logging.info("Frontend dist found, mounting static files")
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    else:
        logging.warning(f"Frontend dist not found at: {frontend_dist}")
except Exception as e:
    logging.error(f"Error setting up static files: {e}")
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))