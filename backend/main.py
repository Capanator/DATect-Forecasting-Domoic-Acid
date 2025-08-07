"""
DATect Web Application Backend
============================

FastAPI backend for the Domoic Acid forecasting web application.
Provides REST API endpoints for forecasting, data management, and user operations.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import sys
import os
import pandas as pd

# Add parent directory to path to import forecasting modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.model_factory import ModelFactory
import config

# Fix path resolution - ensure we use absolute paths relative to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)

app = FastAPI(
    title="DATect API",
    description="Domoic Acid Forecasting System REST API",
    version="1.0.0"
)

# CORS middleware for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
forecast_engine = ForecastEngine()
model_factory = ModelFactory()

# Pydantic models
class ForecastRequest(BaseModel):
    date: date
    site: str
    task: str = "regression"  # "regression" or "classification"
    model: str = "xgboost"

class ConfigUpdateRequest(BaseModel):
    forecast_mode: str = "realtime"  # "realtime" or "retrospective" 
    forecast_task: str = "regression"  # "regression" or "classification"
    forecast_model: str = "xgboost"  # "xgboost" or "ridge"

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

@app.get("/")
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
        available_models = {
            "regression": model_factory.get_supported_models('regression')['regression'],
            "classification": model_factory.get_supported_models('classification')['classification']
        }
        
        # Get descriptions for all models
        descriptions = {}
        all_models = set(available_models["regression"] + available_models["classification"])
        for model in all_models:
            descriptions[model] = model_factory.get_model_description(model)
        
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
        
        # Generate forecast using the existing engine
        result = forecast_engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            request.task,
            request.model
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
        
        # Generate both regression and classification forecasts
        regression_result = forecast_engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            "regression",
            request.model
        )
        
        # For classification, use logistic if the current model doesn't support classification
        classification_models = model_factory.get_supported_models('classification')['classification']
        classification_model = "logistic" if request.model not in classification_models else request.model
        
        classification_result = forecast_engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            pd.to_datetime(request.date),
            actual_site,
            "classification",
            classification_model
        )
        
        # Create enhanced response with graph data
        response_data = {
            "success": True,
            "forecast_date": request.date,
            "site": actual_site,
            "regression": regression_result,
            "classification": classification_result,
            "graphs": {}
        }
        
        # Add level range graph data for regression
        if regression_result and 'predicted_da' in regression_result:
            predicted_da = regression_result['predicted_da']
            response_data["graphs"]["level_range"] = {
                "predicted_da": predicted_da,
                "q05": predicted_da * 0.7,
                "q50": predicted_da,
                "q95": predicted_da * 1.3,
                "type": "level_range"
            }
        
        # Add category graph data for classification  
        if classification_result and 'predicted_category' in classification_result:
            response_data["graphs"]["category_range"] = {
                "predicted_category": classification_result['predicted_category'],
                "class_probabilities": classification_result.get('class_probabilities'),
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
async def run_retrospective_analysis():
    """Run complete retrospective analysis based on current config."""
    try:
        # Run retrospective analysis using forecast engine directly
        # This avoids launching the dashboard but still gets the results
        engine = ForecastEngine(data_file=config.FINAL_OUTPUT_PATH)
        results_df = engine.run_retrospective_evaluation(
            task=config.FORECAST_TASK,
            model_type=config.FORECAST_MODEL,
            n_anchors=getattr(config, 'N_RANDOM_ANCHORS', 50)  # Use smaller number for API
        )
        
        if results_df is None or results_df.empty:
            return {
                "success": False,
                "error": "No results generated from retrospective analysis"
            }
        
        # Convert results to JSON format
        results_json = []
        for _, row in results_df.iterrows():
            record = {
                "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
                "site": row['site'],
                "actual_da": float(row['da']) if pd.notnull(row['da']) else None,
                "predicted_da": float(row['Predicted_da']) if 'Predicted_da' in row and pd.notnull(row['Predicted_da']) else None,
                "actual_category": int(row['da-category']) if 'da-category' in row and pd.notnull(row['da-category']) else None,
                "predicted_category": int(row['Predicted_da-category']) if 'Predicted_da-category' in row and pd.notnull(row['Predicted_da-category']) else None
            }
            results_json.append(record)
        
        # Calculate summary statistics
        valid_regression = [(r['actual_da'], r['predicted_da']) for r in results_json 
                          if r['actual_da'] is not None and r['predicted_da'] is not None]
        valid_classification = [(r['actual_category'], r['predicted_category']) for r in results_json 
                              if r['actual_category'] is not None and r['predicted_category'] is not None]
        
        summary = {
            "total_forecasts": len(results_json),
            "regression_forecasts": len(valid_regression),
            "classification_forecasts": len(valid_classification)
        }
        
        # Calculate R² and MAE if we have regression results
        if valid_regression:
            from sklearn.metrics import r2_score, mean_absolute_error
            actual_vals = [r[0] for r in valid_regression]
            pred_vals = [r[1] for r in valid_regression]
            summary["r2_score"] = float(r2_score(actual_vals, pred_vals))
            summary["mae"] = float(mean_absolute_error(actual_vals, pred_vals))
        
        # Calculate accuracy if we have classification results
        if valid_classification:
            from sklearn.metrics import accuracy_score
            actual_cats = [r[0] for r in valid_classification]
            pred_cats = [r[1] for r in valid_classification]
            summary["accuracy"] = float(accuracy_score(actual_cats, pred_cats))
        
        return {
            "success": True,
            "config": {
                "forecast_mode": config.FORECAST_MODE,
                "forecast_task": config.FORECAST_TASK,
                "forecast_model": config.FORECAST_MODEL
            },
            "summary": summary,
            "results": results_json
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)