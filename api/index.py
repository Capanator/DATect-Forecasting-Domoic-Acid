"""
Vercel Serverless Function Handler for DATect API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules
from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.model_factory import ModelFactory
import config

# Fix path resolution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)

app = FastAPI(
    title="DATect API",
    description="Domoic Acid Forecasting System REST API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
forecast_engine = ForecastEngine()
model_factory = ModelFactory()

# Request/Response Models
class ForecastRequest(BaseModel):
    date: str
    site: str
    model: str = "xgboost"
    task: str = "regression"

class RetrospectiveRequest(BaseModel):
    start_date: str
    end_date: str
    site: str
    model: str = "xgboost"
    task: str = "regression"

class ForecastResponse(BaseModel):
    prediction: float
    confidence: str
    metadata: Dict[str, Any]

# Model mapping
def get_actual_model_name(ui_model: str, task: str) -> str:
    """Map UI model selection to actual model names based on task."""
    if ui_model == "xgboost":
        return "xgboost"
    elif ui_model == "linear":
        if task == "regression":
            return "linear"
        else:
            return "logistic"
    else:
        return ui_model

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "DATect API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "DATect API"
    }

@app.get("/api/sites")
async def get_sites():
    """Get available monitoring sites."""
    sites = [
        {"value": "monterey_wharf", "label": "Monterey Wharf, CA"},
        {"value": "santa_cruz_wharf", "label": "Santa Cruz Wharf, CA"},
        {"value": "newport_beach_pier", "label": "Newport Beach Pier, CA"},
        {"value": "cal_poly_pier", "label": "Cal Poly Pier, CA"},
        {"value": "stearns_wharf", "label": "Stearns Wharf, CA"},
        {"value": "scripps_pier", "label": "Scripps Pier, CA"},
        {"value": "santa_monica_pier", "label": "Santa Monica Pier, CA"},
        {"value": "cape_disappointment", "label": "Cape Disappointment, WA"},
        {"value": "willapa_bay", "label": "Willapa Bay, WA"},
        {"value": "kalaloch", "label": "Kalaloch, WA"}
    ]
    return sites

@app.get("/api/models")
async def get_models():
    """Get available ML models."""
    models = [
        {"value": "xgboost", "label": "XGBoost", "description": "Gradient boosting model"},
        {"value": "linear", "label": "Linear/Logistic", "description": "Simple linear model"}
    ]
    return models

@app.post("/api/forecast")
async def generate_forecast(request: ForecastRequest):
    """Generate a single forecast."""
    try:
        # Parse date
        forecast_date = pd.to_datetime(request.date)
        
        # Get actual model name
        model_name = get_actual_model_name(request.model, request.task)
        
        # Update config
        config.FORECAST_TASK = request.task
        config.FORECAST_MODEL = model_name
        config.FORECAST_MODE = "realtime"
        
        # Generate forecast
        result = forecast_engine.generate_forecast(
            forecast_date=forecast_date,
            site=request.site,
            model_type=model_name,
            task=request.task
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate forecast")
        
        # Prepare response
        prediction_value = float(result.get('prediction', 0))
        
        # Determine confidence level
        if request.task == "regression":
            if prediction_value < 20:
                confidence = "Low Risk"
            elif prediction_value < 50:
                confidence = "Medium Risk"
            else:
                confidence = "High Risk"
        else:
            confidence = result.get('prediction_label', 'Unknown')
        
        return ForecastResponse(
            prediction=prediction_value,
            confidence=confidence,
            metadata={
                "model": model_name,
                "task": request.task,
                "site": request.site,
                "date": request.date,
                "features_used": result.get('features_used', []),
                "training_samples": result.get('training_samples', 0)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrospective")
async def run_retrospective(request: RetrospectiveRequest):
    """Run retrospective analysis."""
    try:
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Get actual model name
        model_name = get_actual_model_name(request.model, request.task)
        
        # Update config
        config.FORECAST_TASK = request.task
        config.FORECAST_MODEL = model_name
        config.FORECAST_MODE = "retrospective"
        
        # Run retrospective analysis
        results = forecast_engine.run_retrospective_analysis(
            start_date=start_date,
            end_date=end_date,
            site=request.site,
            model_type=model_name,
            task=request.task
        )
        
        # Calculate metrics
        if results and len(results) > 0:
            predictions = [r['prediction'] for r in results if r['prediction'] is not None]
            actuals = [r['actual'] for r in results if r['actual'] is not None]
            
            if len(predictions) > 0 and len(actuals) > 0:
                mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
                rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            else:
                mae = rmse = 0
            
            return {
                "results": results[:100],  # Limit to 100 results
                "metrics": {
                    "total_forecasts": len(results),
                    "successful_forecasts": len(predictions),
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2)
                },
                "metadata": {
                    "model": model_name,
                    "task": request.task,
                    "site": request.site,
                    "start_date": request.start_date,
                    "end_date": request.end_date
                }
            }
        else:
            return {
                "results": [],
                "metrics": {},
                "metadata": {
                    "model": model_name,
                    "task": request.task,
                    "site": request.site,
                    "start_date": request.start_date,
                    "end_date": request.end_date
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Vercel serverless handler
handler = app