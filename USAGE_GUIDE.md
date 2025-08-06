# DATect System Usage Guide

## 🚀 Quick Start - How to Use the System Now

The DATect system has been upgraded with production-ready logging, exception handling, and monitoring. Here's how to use it:

## 📋 Available Commands

### 1. **Data Processing** (Required First Step)
```bash
python dataset-creation.py
```
- **What it does**: Downloads satellite data, climate data, and processes all input CSV files
- **Runtime**: 30-60 minutes (depending on internet speed)
- **Output**: Creates `final_output.parquet` with all processed data
- **Requirements**: Internet connection for downloading satellite/climate data
- **Logging**: Full logging to `logs/datect_main.log`

### 2. **Interactive Forecasting Dashboard** (Default)
```bash
python modular-forecast.py
```
- **What it does**: Launches interactive forecasting dashboard
- **URL**: http://localhost:8065
- **Features**: 
  - Select any monitoring site
  - Choose prediction date
  - Pick model (XGBoost or Ridge)
  - View regression AND classification results
- **Config**: Uses settings from `config.py` (default: `FORECAST_MODE = "realtime"`)

### 3. **Research/Validation Mode** 
```bash
# Edit config.py: change FORECAST_MODE = "retrospective"
python modular-forecast.py
```
- **What it does**: Runs retrospective evaluation with random anchor points
- **Output**: Performance metrics and validation dashboard on port 8071
- **Purpose**: Scientific validation with temporal safeguards
- **Features**: Model performance analysis, cross-validation results

### 4. **API Service** (New!)
```bash
python run_api_service.py
```
- **What it does**: Starts production API service
- **URL**: http://localhost:8000
- **Features**: REST endpoints for predictions, health monitoring, metrics
- **Use cases**: Integration with external systems, production deployment
- **Documentation**: Available at http://localhost:8000/docs

### 5. **Health Monitoring** (New!)
```bash
curl http://localhost:8000/health
```
- **What it does**: Comprehensive system health check
- **Output**: System status, performance metrics, resource usage
- **Features**: Monitor CPU, memory, disk, model availability

## 🔧 Configuration

### Main Settings (`config.py`)

#### **Operating Mode**
```python
FORECAST_MODE = "realtime"     # "realtime" or "retrospective"
```
- `"realtime"`: Interactive dashboard for specific predictions
- `"retrospective"`: Historical validation with random anchor points

#### **Model Selection**
```python
FORECAST_MODEL = "xgboost"     # "xgboost" or "ridge"
```
- `"xgboost"`: Primary model (best performance)
- `"ridge"`: Linear fallback model

#### **Task Type**
```python
FORECAST_TASK = "regression"   # "regression" or "classification"
```
- `"regression"`: Predict DA concentration values (μg/g)
- `"classification"`: Predict risk categories (Low/Moderate/High/Extreme)

### Environment Configuration (Optional)

For production deployment, you can use environment variables instead of `config.py`:

```bash
# Set environment variables
export DATECT_ENVIRONMENT=production
export DATECT_FORECAST_MODE=realtime
export DATECT_FORECAST_MODEL=xgboost
export DATECT_API_PORT=8080
export DATECT_LOG_LEVEL=INFO

# Run with environment config
python modular-forecast.py
```

## 📊 Understanding the Output

### **Logging System**
All operations now use proper logging instead of print statements:
- **Main log**: `logs/datect_main.log` - All system operations
- **Error log**: `logs/datect_errors.log` - Errors and warnings only
- **Console output**: Important messages also shown in terminal

### **Dashboard Features**

#### **Realtime Dashboard** (Port 8065)
- **Model Selection**: XGBoost or Ridge Regression
- **Site Selection**: Choose from 10 Pacific Coast locations
- **Date Selection**: Pick any date for prediction
- **Dual Results**: Both regression (μg/g) and classification (risk level)
- **Feature Importance**: See which factors drive predictions

#### **Retrospective Dashboard** (Port 8071)  
- **Performance Metrics**: R², MAE, accuracy across all sites
- **Temporal Patterns**: Time series visualization of predictions vs actual
- **Site Comparison**: Performance variation by location
- **Validation Results**: Scientifically rigorous evaluation

### **API Endpoints** (Port 8000)
- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive system status  
- `GET /metrics` - Performance metrics
- `POST /predict` - Make predictions
- `POST /validate` - Validate input data
- `GET /models` - List available models

## 🔒 Scientific Integrity Features

### **Temporal Safeguards**
The system automatically prevents data leakage through:
- **Temporal splitting**: Data split BEFORE preprocessing
- **Buffer periods**: Minimum 7-day separation between training/test
- **Forward-only processing**: No future information used
- **Lag feature protection**: Historical data only

### **Model Validation**
- **Cross-validation**: Proper time series validation
- **Random anchors**: Unbiased evaluation points
- **Performance tracking**: Comprehensive metrics collection
- **Error handling**: Robust operation with detailed logging

## 🐛 Troubleshooting

### **Common Issues**

#### **"Cannot load data file"**
```bash
# Solution: Run data processing first
python dataset-creation.py
```

#### **"Model not supported for task"**
```bash
# Check config.py settings:
FORECAST_TASK = "regression"    # Use "regression" or "classification"  
FORECAST_MODEL = "xgboost"      # Use "xgboost" or "ridge"
```

#### **Dashboard won't load**
```bash
# Check if port is available
netstat -an | grep 8065

# Try different port in config.py:
DASHBOARD_PORT = 8066
```

#### **API service fails**
```bash
# Check logs
tail -f logs/datect_main.log

# Verify data file exists
ls -la final_output.parquet
```

### **Getting Help**

1. **Check logs first**: `tail -f logs/datect_main.log`
2. **Review configuration**: Ensure `config.py` settings are valid
3. **Verify data**: Make sure `final_output.parquet` exists
4. **Test components**: Run `python dataset-creation.py` first

## 📈 Performance Monitoring

### **System Health**
```bash
# Check overall health
curl http://localhost:8000/health

# Get detailed metrics  
curl http://localhost:8000/metrics

# View system resources
curl http://localhost:8000/health/detailed
```

### **Log Analysis**
```bash
# Follow main log in real-time
tail -f logs/datect_main.log

# Check for errors only
tail -f logs/datect_errors.log

# Search for specific events
grep "forecast" logs/datect_main.log
```

## 🔄 Typical Workflows

### **Research Workflow**
1. `python dataset-creation.py` (initial setup)
2. Edit `config.py`: `FORECAST_MODE = "retrospective"`  
3. `python modular-forecast.py` (validation)
4. Analyze results on port 8071

### **Operational Workflow**  
1. `python dataset-creation.py` (data refresh)
2. Keep `config.py`: `FORECAST_MODE = "realtime"`
3. `python modular-forecast.py` (forecasting)  
4. Use dashboard on port 8065

### **API Integration Workflow**
1. `python run_api_service.py` (start API)
2. Monitor via http://localhost:8000/health
3. Make predictions via POST requests
4. Integrate with external systems

## 📁 File Structure Guide

```
DATect-Forecasting-Domoic-Acid/
├── modular-forecast.py          # MAIN ENTRY POINT
├── dataset-creation.py          # Data processing pipeline
├── run_api_service.py           # API service launcher  
├── config.py                    # Main configuration
├── logs/                        # All log files
│   ├── datect_main.log         # Main operations log
│   └── datect_errors.log       # Errors only
├── forecasting/                 # Core system modules
│   ├── core/                   # Engine, logging, monitoring
│   ├── api/                    # REST API service
│   └── dashboard/              # Interactive dashboards
└── final_output.parquet        # Processed data (generated)
```

## 🎯 Key Improvements Made

1. **✅ Proper Logging**: All print statements replaced with structured logging
2. **✅ Exception Handling**: Comprehensive error handling with safe_execute
3. **✅ Health Monitoring**: System health checks and performance metrics  
4. **✅ API Service**: Production-ready REST API with validation
5. **✅ Data Validation**: Input validation with quality scoring
6. **✅ Environment Config**: Support for environment variables
7. **✅ Model Persistence**: Automatic model saving/loading
8. **✅ Temporal Safeguards**: Enhanced data leakage prevention

The system is now production-ready with enterprise-grade logging, monitoring, and error handling while maintaining the original scientific integrity and ease of use.