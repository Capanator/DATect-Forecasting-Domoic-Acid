"""
API Data Models and Validation
==============================

Pydantic models for request/response validation and serialization.
Provides type safety and automatic API documentation.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, model_validator, confloat, conint
from pydantic.types import constr
import pandas as pd
import re


# Valid site names for Pacific Coast monitoring locations
VALID_SITES = {
    "Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach",
    "Clatsop Beach", "Cannon Beach", "Newport", "Coos Bay", "Gold Beach"
}

# Valid model names
VALID_MODELS = {
    "datect_xgboost_regression", "datect_xgboost_classification", 
    "datect_ridge_regression", "datect_logistic_classification",
    "xgboost", "ridge"  # Legacy names
}


class PredictionInputData(BaseModel):
    """Strict validation model for prediction input features."""
    
    # Core oceanographic features with scientific ranges
    sst: confloat(ge=5.0, le=25.0) = Field(
        ..., 
        description="Sea Surface Temperature in Celsius (5-25°C range)",
        example=12.5
    )
    
    chlorophyll: confloat(ge=0.1, le=100.0) = Field(
        ..., 
        description="Chlorophyll-a concentration in mg/m³ (0.1-100.0 range)",
        example=2.3
    )
    
    # Optional oceanographic features
    par: Optional[confloat(ge=10.0, le=70.0)] = Field(
        None,
        description="Photosynthetically Available Radiation (10-70 range)",
        example=45.2
    )
    
    fluorescence: Optional[confloat(ge=0.0, le=5.0)] = Field(
        None,
        description="Fluorescence Line Height (0-5 range)",
        example=0.8
    )
    
    k490: Optional[confloat(ge=0.01, le=2.0)] = Field(
        None,
        description="Diffuse Attenuation Coefficient (0.01-2.0 range)",
        example=0.15
    )
    
    # Climate indices with realistic ranges
    PDO: Optional[confloat(ge=-3.0, le=3.0)] = Field(
        None,
        description="Pacific Decadal Oscillation index (-3 to +3 range)",
        example=1.2
    )
    
    ONI: Optional[confloat(ge=-3.0, le=3.0)] = Field(
        None,
        description="Oceanic Niño Index (-3 to +3 range)",
        example=-0.5
    )
    
    BEUTI: Optional[confloat(ge=-500.0, le=500.0)] = Field(
        None,
        description="Biologically Effective Upwelling Transport Index (-500 to +500)",
        example=125.3
    )
    
    # Streamflow data
    streamflow: Optional[confloat(ge=1000.0, le=50000.0)] = Field(
        None,
        description="USGS streamflow in cubic feet per second (1000-50000 range)",
        example=15000.0
    )
    
    # Temporal features (auto-calculated if not provided)
    sin_day_of_year: Optional[confloat(ge=-1.0, le=1.0)] = Field(
        None,
        description="Sine of day of year (-1 to +1 range)",
        example=0.5
    )
    
    cos_day_of_year: Optional[confloat(ge=-1.0, le=1.0)] = Field(
        None,
        description="Cosine of day of year (-1 to +1 range)", 
        example=0.8
    )
    
    # Historical DA lag features with strict ranges
    da_lag_1: confloat(ge=0.0, le=500.0) = Field(
        ...,
        description="DA concentration 1 week ago in μg/g (0-500 range)",
        example=8.2
    )
    
    da_lag_2: Optional[confloat(ge=0.0, le=500.0)] = Field(
        None,
        description="DA concentration 2 weeks ago in μg/g (0-500 range)",
        example=6.1
    )
    
    da_lag_3: Optional[confloat(ge=0.0, le=500.0)] = Field(
        None,
        description="DA concentration 3 weeks ago in μg/g (0-500 range)", 
        example=7.8
    )
    
    # Pseudo-nitzschia cell count
    pn: Optional[confloat(ge=0.0, le=10000000.0)] = Field(
        None,
        description="Pseudo-nitzschia cell count (0-10M cells/L)",
        example=50000.0
    )
    
    # Site information (optional, can be provided in request)
    site: Optional[str] = Field(
        None,
        description="Monitoring site name",
        example="Newport"
    )
    
    @validator('site')
    def validate_site(cls, v):
        """Validate site name against known locations."""
        if v is not None and v not in VALID_SITES:
            raise ValueError(f"Site must be one of: {sorted(VALID_SITES)}")
        return v
    
    @model_validator(mode='after')
    def validate_data_completeness(self):
        """Ensure minimum required data for prediction."""
        required_fields = ['sst', 'chlorophyll', 'da_lag_1']
        missing_fields = [field for field in required_fields if getattr(self, field, None) is None]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return self


class PredictionRequest(BaseModel):
    """Enhanced request model for making predictions with strict validation."""
    
    # Model selection with validation
    model_name: constr(pattern=r'^[a-zA-Z0-9_-]+$') = Field(
        ..., 
        description="Name of the model to use for prediction",
        example="datect_xgboost_regression"
    )
    
    model_version: constr(pattern=r'^[a-zA-Z0-9_.-]+$') = Field(
        "latest", 
        description="Version of the model (default: latest)",
        example="latest"
    )
    
    # Strictly validated input data
    data: Union[List[PredictionInputData], PredictionInputData] = Field(
        ..., 
        description="Input data for prediction with strict validation"
    )
    
    # Optional site override (if not in data)
    site: Optional[str] = Field(
        None,
        description="Monitoring site name (overrides site in data)",
        example="Newport"
    )
    
    # Optional date context  
    prediction_date: Optional[date] = Field(
        None,
        description="Date for prediction context (ISO format)",
        example="2024-01-15"
    )
    
    # Advanced options
    return_probabilities: bool = Field(
        False, 
        description="Return class probabilities for classification tasks"
    )
    
    return_confidence: bool = Field(
        False,
        description="Return confidence intervals or uncertainty measures"
    )
    
    strict_validation: bool = Field(
        True,
        description="Enable strict input validation (recommended for production)"
    )
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name against known models."""
        # Note: We'll validate against actual available models in the service
        if not v or len(v) > 100:
            raise ValueError("Model name must be 1-100 characters")
        return v.lower()
    
    @validator('site')
    def validate_site_override(cls, v):
        """Validate site override."""
        if v is not None and v not in VALID_SITES:
            raise ValueError(f"Site must be one of: {sorted(VALID_SITES)}")
        return v
    
    @validator('data')
    def validate_data_format(cls, v):
        """Ensure data is in list format and validate completeness."""
        if isinstance(v, PredictionInputData):
            return [v]  # Convert single item to list
        elif isinstance(v, list):
            if not v:
                raise ValueError("Data list cannot be empty")
            if len(v) > 1000:  # Reasonable batch size limit
                raise ValueError("Maximum 1000 records per request")
            return v
        else:
            raise ValueError("Data must be a PredictionInputData object or list of objects")
    
    @model_validator(mode='after')
    def validate_complete_request(self):
        """Validate the complete request for consistency."""
        data = getattr(self, 'data', [])
        site_override = getattr(self, 'site', None)
        
        # If site is provided as override, ensure it's consistent
        if site_override:
            for item in data:
                if hasattr(item, 'site') and item.site and item.site != site_override:
                    raise ValueError(f"Site override '{site_override}' conflicts with data site '{item.site}'")
        
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "datect_xgboost_regression",
                "model_version": "latest",
                "data": {
                    "sst": 12.5,
                    "chlorophyll": 2.3,
                    "da_lag_1": 8.2,
                    "da_lag_2": 6.1,
                    "da_lag_3": 7.8,
                    "sin_day_of_year": 0.5,
                    "cos_day_of_year": 0.8,
                    "site": "Newport"
                },
                "site": "Newport",
                "prediction_date": "2024-01-15",
                "return_probabilities": False,
                "return_confidence": False,
                "strict_validation": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predictions: List[Union[float, int]] = Field(..., description="Model predictions")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    
    # Optional fields
    probabilities: Optional[List[List[float]]] = Field(
        None, 
        description="Class probabilities (for classification tasks)"
    )
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Confidence intervals or uncertainty measures"
    )
    
    # Metadata
    prediction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the predictions were generated"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to generate predictions (milliseconds)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [12.5, 8.3, 15.7],
                "model_info": {
                    "model_name": "datect_xgboost_regression",
                    "version": "20250106_120000",
                    "model_type": "XGBRegressor",
                    "task_type": "regression"
                },
                "probabilities": None,
                "confidence_intervals": None,
                "prediction_timestamp": "2025-01-06T12:00:00",
                "processing_time_ms": 45.2
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    
    # System information
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    models_available: int = Field(..., description="Number of available models")
    
    # Optional detailed information
    system_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed system information (debug mode only)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-06T12:00:00",
                "version": "1.0.0",
                "environment": "production",
                "uptime_seconds": 3600.5,
                "models_available": 3,
                "system_info": None
            }
        }


class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    
    models: Dict[str, List[str]] = Field(
        ..., 
        description="Dictionary mapping model names to available versions"
    )
    total_models: int = Field(..., description="Total number of model types")
    total_versions: int = Field(..., description="Total number of model versions")
    
    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "datect_xgboost_regression": ["20250106_120000", "20250105_100000"],
                    "datect_xgboost_classification": ["20250106_120000"],
                    "datect_ridge_regression": ["20250105_150000"]
                },
                "total_models": 3,
                "total_versions": 4
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_info: Dict[str, Any] = Field(..., description="Detailed model information")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "datect_xgboost_regression",
                "version": "20250106_120000",
                "model_info": {
                    "created_at": "2025-01-06T12:00:00",
                    "model_type": "XGBRegressor",
                    "task_type": "regression",
                    "training_samples": 1500,
                    "performance_metrics": {
                        "r2": 0.85,
                        "mae": 2.3,
                        "rmse": 3.1
                    },
                    "sites_trained": ["Newport", "Coos Bay", "Gold Beach"],
                    "artifact_path": "/model_artifacts/datect_xgboost_regression_v20250106_120000",
                    "size_mb": 12.5
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ModelNotFound",
                "message": "Model 'nonexistent_model' not found",
                "detail": {
                    "available_models": ["datect_xgboost_regression", "datect_ridge_regression"]
                },
                "timestamp": "2025-01-06T12:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Enhanced request model for batch predictions with strict validation."""
    
    # Model selection with validation
    model_name: constr(pattern=r'^[a-zA-Z0-9_-]+$') = Field(
        ..., 
        description="Name of the model to use",
        example="datect_xgboost_regression"
    )
    
    model_version: constr(pattern=r'^[a-zA-Z0-9_.-]+$') = Field(
        "latest", 
        description="Version of the model",
        example="latest"
    )
    
    # Strictly validated batch data
    data: List[PredictionInputData] = Field(
        ..., 
        description="List of input records with strict validation",
        min_items=1,
        max_items=10000
    )
    
    # Processing options with constraints
    chunk_size: conint(ge=1, le=1000) = Field(
        100, 
        description="Number of records to process in each chunk (1-1000)",
        example=100
    )
    
    return_individual_results: bool = Field(
        True,
        description="Return individual prediction results with metadata"
    )
    
    # Processing options
    fail_on_first_error: bool = Field(
        False,
        description="Stop processing on first validation error"
    )
    
    strict_validation: bool = Field(
        True,
        description="Enable strict input validation for all records"
    )
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name."""
        if not v or len(v) > 100:
            raise ValueError("Model name must be 1-100 characters")
        return v.lower()
    
    @validator('data')
    def validate_batch_consistency(cls, v):
        """Validate batch data consistency."""
        if not v:
            raise ValueError("Batch data cannot be empty")
        
        # Check for consistent site information across batch
        sites_in_data = {item.site for item in v if hasattr(item, 'site') and item.site}
        if len(sites_in_data) > 10:  # Reasonable limit
            raise ValueError("Too many different sites in batch (max 10)")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "datect_xgboost_regression",
                "model_version": "latest",
                "data": [
                    {"sst": 12.5, "chlorophyll": 2.3, "da_lag_1": 8.2},
                    {"sst": 13.1, "chlorophyll": 1.8, "da_lag_1": 7.5},
                    {"sst": 11.8, "chlorophyll": 3.1, "da_lag_1": 9.1}
                ],
                "chunk_size": 100,
                "return_individual_results": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[Union[float, int]] = Field(..., description="Batch predictions")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    
    # Batch processing metadata
    total_records: int = Field(..., description="Total number of records processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")
    average_time_per_record_ms: float = Field(..., description="Average time per record")
    
    # Optional detailed results
    individual_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Individual prediction results with metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [12.5, 8.3, 15.7],
                "model_info": {
                    "model_name": "datect_xgboost_regression",
                    "version": "20250106_120000"
                },
                "total_records": 3,
                "successful_predictions": 3,
                "failed_predictions": 0,
                "processing_time_ms": 156.8,
                "average_time_per_record_ms": 52.3,
                "individual_results": None
            }
        }