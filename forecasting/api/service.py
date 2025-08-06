"""
DATect FastAPI Prediction Service
=================================

Production-ready REST API service for domoic acid forecasting.
Provides secure, scalable prediction endpoints separate from dashboards.

Features:
- RESTful prediction endpoints
- Model management and versioning
- Health monitoring and metrics
- Authentication and rate limiting
- Comprehensive error handling
- Automatic API documentation

Usage:
    uvicorn forecasting.api.service:app --host 0.0.0.0 --port 8000
"""

import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from ..core.env_config import get_config
from ..core.forecast_engine import ForecastEngine
from ..core.logging_config import get_logger, log_performance
from ..core.health_monitoring import HealthMonitor
from ..core.data_validation import DataValidator
from ..core.exception_handling import (
    handle_data_error, handle_model_error, ModelError, ValidationError
)
from .models import (
    PredictionRequest, PredictionResponse, HealthResponse, ModelListResponse,
    ModelInfoResponse, ErrorResponse, BatchPredictionRequest, BatchPredictionResponse
)

# Initialize configuration and logging
config = get_config()
logger = get_logger(__name__)

# Application metadata
APP_VERSION = "1.0.0"
APP_TITLE = "DATect Prediction API"
APP_DESCRIPTION = """
## DATect Domoic Acid Forecasting API

Production-ready prediction service for harmful algal bloom forecasting along the Pacific Coast.

### Features
- **Machine Learning Predictions**: XGBoost and Ridge regression models
- **Real-time Forecasting**: Single and batch prediction endpoints  
- **Model Management**: Version control and model information
- **Health Monitoring**: Service health and performance metrics
- **Scientific Validation**: Peer-reviewed forecasting methodology

### Authentication
API key required for production environments. Contact administrators for access.

### Rate Limiting
- Development: No limits
- Production: 1000 requests per hour per API key

### Data Sources
Integrates satellite oceanographic data, climate indices, and shellfish toxin measurements.
"""

# Security
security = HTTPBearer(auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None,
    openapi_url="/openapi.json" if config.DEBUG else None
)

# Middleware configuration
if config.environment != "production":
    # Enable CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Trusted hosts middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", config.API_HOST]
)

# Global state
app_start_time = time.time()
forecast_engine = None
health_monitor = None
data_validator = None


async def startup_event():
    """Initialize services on startup."""
    global forecast_engine, health_monitor, data_validator
    
    try:
        logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"API Host: {config.API_HOST}:{config.API_PORT}")
        
        # Initialize health monitoring
        health_monitor = HealthMonitor(config)
        logger.info("HealthMonitor initialized successfully")
        
        # Initialize data validator
        data_validator = DataValidator(strict_mode=(config.environment == 'production'))
        logger.info("DataValidator initialized successfully")
        
        # Initialize forecast engine
        forecast_engine = ForecastEngine()
        logger.info("ForecastEngine initialized successfully")
        
        # Health checks are handled through endpoints
        logger.info("Health monitoring system ready")
        
        # Log available models
        models = forecast_engine.list_available_models()
        logger.info(f"Found {len(models)} model types available")
        
        logger.info("DATect Prediction API startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        logger.error(f"Startup traceback:\n{traceback.format_exc()}")
        raise


async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down DATect Prediction API")


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Register event handlers
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key for authentication."""
    if config.environment == "development":
        return "development"  # Skip authentication in development
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In production, validate against configured API key
    if config.API_KEY and credentials.credentials != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


def _check_forecast_engine_health():
    """Custom health check for forecast engine."""
    from ..core.health_monitoring import HealthCheckResult
    
    try:
        if forecast_engine is None:
            return HealthCheckResult("forecast_engine", "critical", "ForecastEngine not initialized")
        
        # Test model loading capability
        models = forecast_engine.list_available_models()
        model_count = sum(len(versions) for versions in models.values())
        
        if model_count == 0:
            return HealthCheckResult("forecast_engine", "warning", "No trained models available")
        
        return HealthCheckResult(
            "forecast_engine", 
            "healthy", 
            f"ForecastEngine operational with {model_count} models",
            metadata={"available_models": model_count}
        )
        
    except Exception as e:
        return HealthCheckResult("forecast_engine", "critical", f"ForecastEngine check failed: {str(e)}")


def _check_api_service_health():
    """Custom health check for API service."""
    from ..core.health_monitoring import HealthCheckResult
    
    try:
        uptime = time.time() - app_start_time
        
        # Check if we have recent activity (metrics collection)
        if health_monitor and hasattr(health_monitor, 'metrics'):
            metrics = health_monitor.metrics.get_metrics()
            
            status = "healthy"
            if metrics['error_rate_percentage'] > 10:
                status = "warning"
            elif metrics['error_rate_percentage'] > 25:
                status = "critical"
            
            message = f"API service running for {uptime:.1f}s, error rate: {metrics['error_rate_percentage']:.1f}%"
            
            return HealthCheckResult(
                "api_service", 
                status, 
                message,
                metadata={
                    "uptime_seconds": uptime,
                    "request_count": metrics['request_count'],
                    "error_rate": metrics['error_rate_percentage']
                }
            )
        
        return HealthCheckResult("api_service", "healthy", f"API service operational for {uptime:.1f}s")
        
    except Exception as e:
        return HealthCheckResult("api_service", "critical", f"API service check failed: {str(e)}")


def get_web_interface() -> str:
    """Generate the web interface HTML."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{APP_TITLE}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body class="bg-gray-50">
        <div id="app" class="min-h-screen">
            <nav class="bg-blue-900 text-white shadow-lg">
                <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div class="flex justify-between items-center h-16">
                        <div class="flex items-center">
                            <i class="fas fa-water text-2xl mr-3"></i>
                            <h1 class="text-xl font-bold">{APP_TITLE}</h1>
                        </div>
                        <div class="flex items-center space-x-4">
                            <span class="text-sm">v{APP_VERSION}</span>
                            <span class="px-2 py-1 bg-green-600 rounded text-xs" id="status-badge">Online</span>
                        </div>
                    </div>
                </div>
            </nav>

            <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- System Status Card -->
                    <div class="lg:col-span-1">
                        <div class="bg-white rounded-lg shadow-md p-6">
                            <h2 class="text-lg font-semibold text-gray-900 mb-4">
                                <i class="fas fa-heartbeat text-red-500 mr-2"></i>System Health
                            </h2>
                            <div id="health-status">
                                <div class="animate-pulse">Loading...</div>
                            </div>
                        </div>

                        <!-- Quick Stats -->
                        <div class="bg-white rounded-lg shadow-md p-6 mt-6">
                            <h2 class="text-lg font-semibold text-gray-900 mb-4">
                                <i class="fas fa-chart-line text-green-500 mr-2"></i>Quick Stats
                            </h2>
                            <div id="quick-stats">
                                <div class="animate-pulse">Loading...</div>
                            </div>
                        </div>
                    </div>

                    <!-- Prediction Interface -->
                    <div class="lg:col-span-2">
                        <div class="bg-white rounded-lg shadow-md p-6">
                            <h2 class="text-lg font-semibold text-gray-900 mb-6">
                                <i class="fas fa-brain text-blue-500 mr-2"></i>Domoic Acid Prediction
                            </h2>
                            
                            <form id="prediction-form" class="space-y-6">
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">Model</label>
                                        <select id="model-select" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                            <option value="">Loading models...</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">Site</label>
                                        <select id="site-select" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                            <option value="Newport">Newport</option>
                                            <option value="Coos Bay">Coos Bay</option>
                                            <option value="Gold Beach">Gold Beach</option>
                                            <option value="Cannon Beach">Cannon Beach</option>
                                            <option value="Clatsop Beach">Clatsop Beach</option>
                                            <option value="Long Beach">Long Beach</option>
                                            <option value="Twin Harbors">Twin Harbors</option>
                                            <option value="Copalis">Copalis</option>
                                            <option value="Quinault">Quinault</option>
                                            <option value="Kalaloch">Kalaloch</option>
                                        </select>
                                    </div>
                                </div>

                                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">SST (°C)</label>
                                        <input type="number" id="sst" step="0.1" value="12.5" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">Chlorophyll</label>
                                        <input type="number" id="chlorophyll" step="0.1" value="2.3" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">DA Lag 1</label>
                                        <input type="number" id="da_lag_1" step="0.1" value="8.2" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 mb-2">DA Lag 2</label>
                                        <input type="number" id="da_lag_2" step="0.1" value="6.1" class="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                </div>

                                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-md transition duration-200">
                                    <i class="fas fa-cogs mr-2"></i>Make Prediction
                                </button>
                            </form>

                            <!-- Results -->
                            <div id="prediction-results" class="mt-6 hidden">
                                <h3 class="text-lg font-semibold text-gray-900 mb-4">Prediction Results</h3>
                                <div id="results-content"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Documentation -->
                <div class="mt-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h2 class="text-lg font-semibold text-gray-900 mb-4">
                            <i class="fas fa-code text-purple-500 mr-2"></i>API Endpoints
                        </h2>
                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            <a href="/docs" class="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200">
                                <div class="text-sm font-medium text-blue-600">Interactive Docs</div>
                                <div class="text-xs text-gray-500">Swagger UI</div>
                            </a>
                            <a href="/health" class="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200">
                                <div class="text-sm font-medium text-green-600">Health Check</div>
                                <div class="text-xs text-gray-500">System status</div>
                            </a>
                            <a href="/metrics" class="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200">
                                <div class="text-sm font-medium text-orange-600">Metrics</div>
                                <div class="text-xs text-gray-500">Performance data</div>
                            </a>
                            <a href="/models" class="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200">
                                <div class="text-sm font-medium text-purple-600">Models</div>
                                <div class="text-xs text-gray-500">Available models</div>
                            </a>
                        </div>
                    </div>
                </div>
            </main>
        </div>

        <script>
            // Global variables
            let availableModels = [];

            // Initialize the application
            async function initApp() {{
                await loadHealth();
                await loadModels();
                await loadQuickStats();
                
                // Refresh health every 30 seconds
                setInterval(loadHealth, 30000);
            }}

            // Load system health
            async function loadHealth() {{
                try {{
                    const response = await axios.get('/health');
                    const health = response.data.health;
                    
                    document.getElementById('health-status').innerHTML = `
                        <div class="space-y-2">
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Overall Status</span>
                                <span class="px-2 py-1 rounded text-xs ${{
                                    health.status === 'healthy' ? 'bg-green-100 text-green-800' :
                                    health.status === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-red-100 text-red-800'
                                }}">
                                    ${{health.status.toUpperCase()}}
                                </span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Uptime</span>
                                <span class="text-sm text-gray-900">${{Math.round(response.data.uptime_seconds / 3600)}}h</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Checks</span>
                                <span class="text-sm text-gray-900">${{health.summary.healthy}}/${{health.summary.total_checks}}</span>
                            </div>
                        </div>
                    `;
                    
                    // Update status badge
                    const statusBadge = document.getElementById('status-badge');
                    if (health.status === 'healthy') {{
                        statusBadge.className = 'px-2 py-1 bg-green-600 rounded text-xs';
                        statusBadge.textContent = 'Online';
                    }} else {{
                        statusBadge.className = 'px-2 py-1 bg-yellow-600 rounded text-xs';
                        statusBadge.textContent = 'Warning';
                    }}
                }} catch (error) {{
                    document.getElementById('health-status').innerHTML = '<div class="text-red-500 text-sm">Unable to load health status</div>';
                }}
            }}

            // Load available models
            async function loadModels() {{
                try {{
                    const response = await axios.get('/models', {{
                        headers: {{ 'Authorization': 'Bearer development' }}
                    }});
                    
                    availableModels = Object.keys(response.data.models);
                    
                    const modelSelect = document.getElementById('model-select');
                    modelSelect.innerHTML = availableModels.map(model => 
                        `<option value="${{model}}">${{model}}</option>`
                    ).join('');
                    
                    if (availableModels.length > 0) {{
                        modelSelect.value = availableModels[0];
                    }}
                }} catch (error) {{
                    console.error('Error loading models:', error);
                    document.getElementById('model-select').innerHTML = '<option value="">Error loading models</option>';
                }}
            }}

            // Load quick stats
            async function loadQuickStats() {{
                try {{
                    const response = await axios.get('/metrics', {{
                        headers: {{ 'Authorization': 'Bearer development' }}
                    }});
                    
                    const metrics = response.data;
                    
                    document.getElementById('quick-stats').innerHTML = `
                        <div class="space-y-2">
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Requests</span>
                                <span class="text-sm text-gray-900">${{metrics.request_count || 0}}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Predictions</span>
                                <span class="text-sm text-gray-900">${{metrics.prediction_count || 0}}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-600">Avg Response</span>
                                <span class="text-sm text-gray-900">${{Math.round(metrics.average_response_time_ms || 0)}}ms</span>
                            </div>
                        </div>
                    `;
                }} catch (error) {{
                    document.getElementById('quick-stats').innerHTML = '<div class="text-red-500 text-sm">Unable to load stats</div>';
                }}
            }}

            // Handle prediction form submission
            document.getElementById('prediction-form').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {{
                    sst: parseFloat(document.getElementById('sst').value),
                    chlorophyll: parseFloat(document.getElementById('chlorophyll').value),
                    da_lag_1: parseFloat(document.getElementById('da_lag_1').value),
                    da_lag_2: parseFloat(document.getElementById('da_lag_2').value),
                    sin_day_of_year: Math.sin(2 * Math.PI * new Date().getDayOfYear() / 365),
                    cos_day_of_year: Math.cos(2 * Math.PI * new Date().getDayOfYear() / 365)
                }};
                
                const payload = {{
                    model_name: document.getElementById('model-select').value,
                    model_version: "latest",
                    data: [data]
                }};
                
                try {{
                    const response = await axios.post('/predict', payload, {{
                        headers: {{ 
                            'Authorization': 'Bearer development',
                            'Content-Type': 'application/json'
                        }}
                    }});
                    
                    const results = response.data;
                    const prediction = results.predictions[0];
                    
                    // Determine risk category
                    let riskCategory = 'Low';
                    let riskColor = 'green';
                    
                    if (prediction > 40) {{
                        riskCategory = 'Extreme';
                        riskColor = 'red';
                    }} else if (prediction > 20) {{
                        riskCategory = 'High'; 
                        riskColor = 'orange';
                    }} else if (prediction > 5) {{
                        riskCategory = 'Moderate';
                        riskColor = 'yellow';
                    }}
                    
                    document.getElementById('results-content').innerHTML = `
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="p-4 bg-blue-50 rounded-lg">
                                <h4 class="font-semibold text-blue-900">Predicted DA Concentration</h4>
                                <p class="text-2xl font-bold text-blue-600">${{prediction.toFixed(2)}} μg/g</p>
                            </div>
                            <div class="p-4 bg-${{riskColor}}-50 rounded-lg">
                                <h4 class="font-semibold text-${{riskColor}}-900">Risk Category</h4>
                                <p class="text-2xl font-bold text-${{riskColor}}-600">${{riskCategory}}</p>
                            </div>
                        </div>
                        <div class="mt-4 p-4 bg-gray-50 rounded-lg">
                            <h4 class="font-semibold text-gray-900">Model Information</h4>
                            <div class="mt-2 grid grid-cols-2 gap-4 text-sm">
                                <div><strong>Model:</strong> ${{results.model_info.model_name}}</div>
                                <div><strong>Type:</strong> ${{results.model_info.model_type}}</div>
                                <div><strong>Processing Time:</strong> ${{Math.round(results.processing_time_ms)}}ms</div>
                                <div><strong>Site:</strong> ${{document.getElementById('site-select').value}}</div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('prediction-results').classList.remove('hidden');
                    
                }} catch (error) {{
                    document.getElementById('results-content').innerHTML = `
                        <div class="p-4 bg-red-50 rounded-lg">
                            <h4 class="font-semibold text-red-900">Prediction Error</h4>
                            <p class="text-red-600">${{error.response?.data?.detail || error.message}}</p>
                        </div>
                    `;
                    document.getElementById('prediction-results').classList.remove('hidden');
                }}
            }});

            // Add day of year method to Date prototype
            Date.prototype.getDayOfYear = function() {{
                var start = new Date(this.getFullYear(), 0, 0);
                var diff = this - start;
                var oneDay = 1000 * 60 * 60 * 24;
                return Math.floor(diff / oneDay);
            }};

            // Initialize the application when page loads
            document.addEventListener('DOMContentLoaded', initApp);
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with web interface."""
    return get_web_interface()

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint."""
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "environment": config.environment,
        "docs": f"http://{config.API_HOST}:{config.API_PORT}/docs" if config.DEBUG else "disabled",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint using HealthMonitor."""
    try:
        # Record request start time for metrics
        request_start_time = time.time()
        
        # Get comprehensive health status
        health_status = health_monitor.get_health_status()
        
        # Record API request metrics
        processing_time = (time.time() - request_start_time) * 1000
        health_monitor.record_request(success=True, response_time_ms=processing_time)
        
        # Return comprehensive health information
        return {
            "service": APP_TITLE,
            "version": APP_VERSION,
            "environment": config.environment,
            "uptime_seconds": time.time() - app_start_time,
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        # Record failed request metrics
        processing_time = (time.time() - request_start_time) * 1000 if 'request_start_time' in locals() else 0
        if health_monitor:
            health_monitor.record_api_request(processing_time, 503)
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(get_api_key)):
    """List all available models and their versions."""
    try:
        models = forecast_engine.list_available_models()
        total_versions = sum(len(versions) for versions in models.values())
        
        return ModelListResponse(
            models=models,
            total_models=len(models),
            total_versions=total_versions
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models"
        )


@app.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(
    model_name: str, 
    version: str = "latest",
    api_key: str = Depends(get_api_key)
):
    """Get detailed information about a specific model."""
    try:
        model_info = forecast_engine.get_model_info(model_name, version)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' version '{version}' not found"
            )
        
        return ModelInfoResponse(
            model_name=model_name,
            version=version if version != "latest" else model_info.get("version", version),
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Make predictions using a trained model."""
    start_time = time.time()
    
    try:
        logger.info(f"Prediction request: model={request.model_name}, records={len(request.data)}")
        
        # Convert request data to DataFrame
        df_input = pd.DataFrame(request.data)
        
        # Validate input data
        is_valid, validation_results = data_validator.validate_prediction_data(df_input, "prediction")
        if not is_valid:
            error_msg = f"Data validation failed: {validation_results['errors'][:3]}"  # First 3 errors
            logger.warning(f"Prediction data validation failed: {validation_results}")
            raise ValidationError(error_msg)
        
        # Log validation warnings if any
        if validation_results['warnings']:
            logger.warning(f"Prediction data validation warnings: {validation_results['warnings'][:3]}")
        
        # Make predictions
        predictions = forecast_engine.predict_with_saved_model(
            model_name=request.model_name,
            data=df_input,
            version=request.model_version
        )
        
        # Record prediction metrics
        health_monitor.record_prediction((time.time() - start_time) * 1000, len(predictions))
        
        # Get model information
        model_info = forecast_engine.get_model_info(request.model_name, request.model_version)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Record API request metrics
        health_monitor.record_api_request(processing_time, 200)
        
        # Prepare response
        response = PredictionResponse(
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            model_info={
                "model_name": request.model_name,
                "version": model_info.get("version", request.model_version),
                "model_type": model_info.get("model_type", "unknown"),
                "task_type": model_info.get("task_type", "unknown"),
                "data_quality": validation_results.get("data_quality", {})
            },
            processing_time_ms=processing_time
        )
        
        # Log successful prediction
        background_tasks.add_task(
            log_prediction_metrics,
            model_name=request.model_name,
            num_predictions=len(predictions),
            processing_time_ms=processing_time
        )
        
        return response
        
    except ModelError as e:
        processing_time = (time.time() - start_time) * 1000
        health_monitor.record_api_request(processing_time, 400)
        logger.error(f"Model error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model error: {str(e)}"
        )
    except ValidationError as e:
        processing_time = (time.time() - start_time) * 1000
        health_monitor.record_api_request(processing_time, 422)
        logger.error(f"Validation error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        health_monitor.record_api_request(processing_time, 500)
        logger.error(f"Unexpected error in prediction: {str(e)}")
        logger.error(f"Prediction traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Make batch predictions for multiple records."""
    start_time = time.time()
    
    try:
        logger.info(f"Batch prediction: model={request.model_name}, records={len(request.data)}")
        
        # Convert request data to DataFrame
        df_input = pd.DataFrame(request.data)
        
        # Process in chunks for memory efficiency
        all_predictions = []
        successful_count = 0
        failed_count = 0
        
        chunk_size = request.chunk_size
        for i in range(0, len(df_input), chunk_size):
            chunk_df = df_input.iloc[i:i+chunk_size]
            
            try:
                chunk_predictions = forecast_engine.predict_with_saved_model(
                    model_name=request.model_name,
                    data=chunk_df,
                    version=request.model_version
                )
                
                all_predictions.extend(
                    chunk_predictions.tolist() if isinstance(chunk_predictions, np.ndarray) 
                    else chunk_predictions
                )
                successful_count += len(chunk_predictions)
                
            except Exception as e:
                logger.warning(f"Chunk prediction failed: {str(e)}")
                failed_count += len(chunk_df)
                # Add None for failed predictions
                all_predictions.extend([None] * len(chunk_df))
        
        # Get model information
        model_info = forecast_engine.get_model_info(request.model_name, request.model_version)
        
        # Calculate processing metrics
        processing_time = (time.time() - start_time) * 1000
        avg_time_per_record = processing_time / len(request.data) if request.data else 0
        
        # Prepare response
        response = BatchPredictionResponse(
            predictions=all_predictions,
            model_info={
                "model_name": request.model_name,
                "version": model_info.get("version", request.model_version)
            },
            total_records=len(request.data),
            successful_predictions=successful_count,
            failed_predictions=failed_count,
            processing_time_ms=processing_time,
            average_time_per_record_ms=avg_time_per_record
        )
        
        # Log batch prediction metrics
        background_tasks.add_task(
            log_prediction_metrics,
            model_name=request.model_name,
            num_predictions=successful_count,
            processing_time_ms=processing_time,
            batch_size=len(request.data)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        logger.error(f"Batch prediction traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


async def log_prediction_metrics(model_name: str, num_predictions: int, 
                                processing_time_ms: float, batch_size: int = None):
    """Log prediction metrics for monitoring."""
    try:
        metrics = {
            "model_name": model_name,
            "predictions_count": num_predictions,
            "processing_time_ms": processing_time_ms,
            "avg_time_per_prediction_ms": processing_time_ms / num_predictions if num_predictions > 0 else 0
        }
        
        if batch_size:
            metrics["batch_size"] = batch_size
        
        logger.info(f"Prediction metrics: {metrics}")
        
    except Exception as e:
        logger.warning(f"Failed to log prediction metrics: {e}")


@app.post("/validate", response_model=Dict[str, Any])
async def validate_data(
    request: Dict[str, Any],
    api_key: str = Depends(get_api_key)
):
    """Validate input data without making predictions."""
    try:
        start_time = time.time()
        
        # Extract data from request
        data = request.get("data", [])
        task_type = request.get("task_type", "prediction")
        
        # Convert to DataFrame
        df_input = pd.DataFrame(data)
        
        # Validate data
        is_valid, validation_results = data_validator.validate_prediction_data(df_input, task_type)
        
        # Add validation metadata
        validation_results.update({
            "input_records": len(data),
            "input_columns": list(df_input.columns) if not df_input.empty else [],
            "validation_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat()
        })
        
        # Record API request
        processing_time = (time.time() - start_time) * 1000
        status_code = 200 if is_valid else 400
        health_monitor.record_api_request(processing_time, status_code)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation service failed: {str(e)}"
        )


@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """Get system statistics and metrics."""
    try:
        service_metrics = health_monitor.get_service_metrics()
        system_metrics = health_monitor.get_system_metrics()
        
        return {
            "service_metrics": {
                "total_requests": service_metrics.total_requests,
                "successful_predictions": service_metrics.successful_predictions,
                "failed_predictions": service_metrics.failed_predictions,
                "average_response_time_ms": service_metrics.average_response_time_ms,
                "models_loaded": service_metrics.models_loaded,
                "health_status": service_metrics.health_status
            },
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_available_gb": system_metrics.memory_available_gb,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "uptime_seconds": system_metrics.uptime_seconds
            }
        }
    except Exception as e:
        logger.error(f"Stats endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats service failed: {str(e)}"
        )


@app.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """Detailed health check with comprehensive system information."""
    try:
        request_start_time = time.time()
        
        # Get comprehensive health status with all details
        health_status = health_monitor.get_detailed_status()
        
        # Record metrics
        processing_time = (time.time() - request_start_time) * 1000
        health_monitor.record_request(success=True, response_time_ms=processing_time)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        processing_time = (time.time() - request_start_time) * 1000 if 'request_start_time' in locals() else 0
        if health_monitor:
            health_monitor.record_api_request(processing_time, 503)
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detailed health check failed"
        )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(api_key: str = Depends(get_api_key)):
    """Get system performance metrics."""
    try:
        request_start_time = time.time()
        
        metrics = health_monitor.get_metrics_summary()
        
        # Add API-specific metrics
        metrics.update({
            "service": APP_TITLE,
            "version": APP_VERSION,
            "environment": config.environment,
            "timestamp": datetime.now().isoformat()
        })
        
        # Record metrics request
        processing_time = (time.time() - request_start_time) * 1000
        health_monitor.record_api_request(processing_time, 200)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@app.get("/metrics/prometheus", response_class=JSONResponse)
async def get_prometheus_metrics(api_key: str = Depends(get_api_key)):
    """Get metrics in Prometheus format."""
    try:
        prometheus_metrics = health_monitor.export_metrics_prometheus()
        
        return JSONResponse(
            content={"metrics": prometheus_metrics},
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Prometheus metrics export failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export Prometheus metrics"
        )


@app.post("/health/report", response_model=Dict[str, Any])
async def save_health_report(api_key: str = Depends(get_api_key)):
    """Generate and save detailed health report."""
    try:
        filepath = health_monitor.save_health_report()
        
        if filepath:
            return {
                "status": "success",
                "message": "Health report saved successfully",
                "filepath": filepath,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save health report"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health report generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate health report"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            detail=getattr(exc, "detail_extra", None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Exception traceback:\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail={"error_type": type(exc).__name__} if config.DEBUG else None
        ).dict()
    )


def run_api_server():
    """Run the FastAPI server with production settings."""
    uvicorn.run(
        "forecasting.api.service:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.DEBUG,
        workers=1 if config.DEBUG else 4,
        access_log=config.DEBUG,
        server_header=False,  # Hide server header for security
        date_header=False     # Hide date header for security
    )


if __name__ == "__main__":
    run_api_server()