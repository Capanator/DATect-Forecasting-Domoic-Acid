# DATect API Documentation

## 🌐 FastAPI Backend Documentation

The DATect system provides a comprehensive REST API for domoic acid forecasting and scientific data visualization.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs`  
**OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 📋 API Endpoints

### System Health & Information

#### `GET /`
Root endpoint providing API information.

**Response**: Basic API information and status

#### `GET /health`
Health check endpoint for system monitoring.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### `GET /api/sites`
Get available monitoring sites information.

**Response**:
```json
{
  "sites": [
    "Bodega Head",
    "Bolinas",
    "Cannon Beach",
    "Capitola Wharf",
    "Coos Bay",
    "Half Moon Bay",
    "Monterey Wharf #2",
    "Morro Bay",
    "Redondo",
    "Santa Cruz Wharf"
  ]
}
```

#### `GET /api/models`
Get available forecasting models information.

**Response**:
```json
{
  "models": ["xgboost", "linear"],
  "current_model": "xgboost",
  "performance": {
    "r_squared": 0.529,
    "description": "XGBoost with ~200 forecasts per site"
  }
}
```

### Configuration Management

#### `GET /api/config`
Get current system configuration.

**Response**:
```json
{
  "FORECAST_MODE": "realtime",
  "FORECAST_MODEL": "xgboost", 
  "FORECAST_TASK": "regression",
  "LAG_FEATURES": [1, 3],
  "FORECAST_HORIZON": 14,
  "MIN_TRAINING_SAMPLES": 5
}
```

#### `POST /api/config`
Update system configuration.

**Request Body**:
```json
{
  "FORECAST_MODE": "retrospective",
  "FORECAST_MODEL": "xgboost",
  "FORECAST_TASK": "classification"
}
```

**Response**: Updated configuration object

### Forecasting Endpoints

#### `POST /api/forecast`
Generate basic domoic acid forecast.

**Request Body**:
```json
{
  "date": "2015-06-15",
  "site": "Santa Cruz Wharf"
}
```

**Response**:
```json
{
  "predicted_da": 2.45,
  "prediction_date": "2015-06-15",
  "site": "Santa Cruz Wharf",
  "model_used": "xgboost",
  "risk_category": "Low",
  "confidence_interval": {
    "lower": 1.8,
    "upper": 3.2
  }
}
```

#### `POST /api/forecast/enhanced`
Generate enhanced forecast with uncertainty quantification and feature importance.

**Request Body**:
```json
{
  "date": "2015-06-15",
  "site": "Santa Cruz Wharf"
}
```

**Response**:
```json
{
  "predicted_da": 2.45,
  "prediction_date": "2015-06-15",
  "site": "Santa Cruz Wharf",
  "model_used": "xgboost",
  "risk_category": "Low",
  "quartiles": {
    "q05": 1.72,
    "q50": 2.45,
    "q95": 3.19
  },
  "feature_importance": [
    {
      "feature": "DA_lag_1",
      "importance": 0.342,
      "description": "Domoic acid 1-day lag"
    },
    {
      "feature": "chlor_a_lag_3",
      "importance": 0.187,
      "description": "Chlorophyll-a 3-day lag"
    }
  ]
}
```

#### `POST /api/retrospective`
Generate retrospective analysis for model validation.

**Request Body**:
```json
{
  "sites": ["Santa Cruz Wharf", "Monterey Wharf #2"],
  "start_date": "2015-01-01",
  "end_date": "2015-12-31"
}
```

**Response**: Server-Sent Events stream with progress updates and final results

### Historical Data

#### `GET /api/historical/{site}`
Get historical data for a specific site.

**Parameters**:
- `site`: Monitoring site name
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response**:
```json
{
  "site": "Santa Cruz Wharf",
  "data": [
    {
      "date": "2015-06-15",
      "da_measurement": 1.8,
      "pn_measurement": 45000,
      "chlor_a": 12.3,
      "sst": 16.2
    }
  ]
}
```

#### `GET /api/historical/all`
Get historical data for all sites.

**Parameters**:
- `start_date` (optional): Start date (YYYY-MM-DD)  
- `end_date` (optional): End date (YYYY-MM-DD)

**Response**: Combined historical data across all monitoring sites

### Data Visualization

#### `GET /api/visualizations/correlation/all`
Generate correlation heatmap for all features.

**Response**:
```json
{
  "plot_data": {
    "data": [
      {
        "z": [[1.0, 0.23, -0.15], ...],
        "x": ["DA", "Chlor_a", "SST"],
        "y": ["DA", "Chlor_a", "SST"],
        "type": "heatmap",
        "colorscale": "custom_correlation"
      }
    ],
    "layout": {
      "title": "Feature Correlation Matrix",
      "xaxis": {"title": "Features"},
      "yaxis": {"title": "Features"}
    }
  }
}
```

#### `GET /api/visualizations/correlation/{site}`
Generate correlation heatmap for specific site.

**Parameters**:
- `site`: Monitoring site name

#### `GET /api/visualizations/sensitivity`
Generate sensitivity analysis using Sobol indices.

**Response**: Plotly-formatted sensitivity analysis visualization

#### `GET /api/visualizations/comparison/all`
Generate DA vs Pseudo-nitzschia comparison for all sites.

**Response**: Time series comparison plot data

#### `GET /api/visualizations/comparison/{site}`  
Generate DA vs Pseudo-nitzschia comparison for specific site.

**Parameters**:
- `site`: Monitoring site name

#### `GET /api/visualizations/waterfall`
Generate waterfall plot showing DA levels by site and latitude.

**Response**: Waterfall visualization with regulatory reference bars

#### `GET /api/visualizations/spectral/all`
Generate spectral analysis for all sites with XGBoost comparison.

**Response**: Frequency domain analysis visualization

#### `GET /api/visualizations/spectral/{site}`
Generate spectral analysis for specific site.

**Parameters**:
- `site`: Monitoring site name

## 📊 Data Models

### Request Models

#### `ForecastRequest`
```python
class ForecastRequest(BaseModel):
    date: date
    site: str
    model_override: Optional[str] = None
```

#### `ConfigUpdateRequest` 
```python
class ConfigUpdateRequest(BaseModel):
    FORECAST_MODE: Optional[str] = None
    FORECAST_MODEL: Optional[str] = None
    FORECAST_TASK: Optional[str] = None
```

#### `RetrospectiveRequest`
```python
class RetrospectiveRequest(BaseModel):
    sites: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
```

### Response Models

#### `ForecastResponse`
```python
class ForecastResponse(BaseModel):
    predicted_da: float
    prediction_date: date
    site: str
    model_used: str
    risk_category: str
    confidence_interval: Optional[Dict[str, float]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    quartiles: Optional[Dict[str, float]] = None
```

## 🔧 Configuration Options

### Forecast Modes
- `realtime`: Live forecasting mode
- `retrospective`: Historical validation mode

### Forecast Models
- `xgboost`: Primary model (R² ≈ 0.529)
- `linear`: Fallback linear model

### Forecast Tasks  
- `regression`: Continuous DA concentration prediction
- `classification`: Risk category classification (4 levels)

## 🚨 Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (invalid site/endpoint)
- `422`: Validation Error (invalid request body)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error description",
  "error_type": "ValidationError|DataError|ModelError",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🔒 Security & Validation

### Temporal Integrity
All forecasting endpoints implement strict temporal validation:
- No future data leakage in training
- Proper lag feature cutoffs  
- Chronological data splits
- Buffer periods for data availability

### Input Validation
- Date range validation (2002-2023)
- Site name validation against available sites
- Model parameter validation
- Configuration parameter validation

### Data Quality Checks
- Missing value handling
- Outlier detection
- Feature completeness validation
- Temporal ordering verification

## 📈 Performance Specifications

### Response Times
- Simple forecast: <500ms
- Enhanced forecast: <1s
- Visualizations: <2s  
- Retrospective analysis: Variable (streaming)

### Data Processing
- Processing speed: >80,000 samples/second
- Memory usage: <250MB for full dataset
- Concurrent requests: Up to 10 simultaneous

### Model Performance
- XGBoost R²: ~0.529 (200 forecasts/site)
- Linear R²: ~0.35 (fallback performance)
- Prediction accuracy: Site-dependent

## 🚀 Development & Deployment

### Local Development
```bash
# Start API server
cd backend && uvicorn api:app --reload

# Access interactive docs
open http://localhost:8000/docs
```

### Production Deployment
```bash
# Start with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app

# Or use the integrated launcher
python run_datect.py
```

### Health Monitoring
Monitor the `/health` endpoint for system status and performance metrics.

---

## 📋 Quick Reference

### Most Common Endpoints
- **Health Check**: `GET /health`
- **Generate Forecast**: `POST /api/forecast/enhanced`
- **Get Sites**: `GET /api/sites`
- **Correlation Plot**: `GET /api/visualizations/correlation/all`
- **System Config**: `GET /api/config`

### Authentication
Currently no authentication required. System designed for internal scientific use.

### Rate Limiting
No explicit rate limiting implemented. Production deployment should consider rate limiting for resource protection.