# 🌊 DATect System - Complete Beginner's Guide

**Welcome to DATect!** This is your complete guide to using the Domoic Acid forecasting system, written for absolute beginners. No technical background required! 

## 📋 Table of Contents
1. [What is DATect?](#what-is-datect)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Understanding the System](#understanding-the-system)
4. [Step-by-Step Instructions](#step-by-step-instructions)
5. [Different Ways to Use DATect](#different-ways-to-use-datect)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Options](#advanced-options)

---

## 🤔 What is DATect?

DATect is a **scientific forecasting system** that predicts dangerous levels of **Domoic Acid** in Pacific Coast waters. Domoic acid is a natural toxin produced by harmful algae that can contaminate shellfish and pose health risks.

**What it does:**
- 🔬 **Analyzes** satellite data, climate patterns, and water conditions
- 🤖 **Predicts** toxin levels using machine learning
- 📊 **Provides** easy-to-understand forecasts and risk levels
- 🖥️ **Offers** both web interface and scientific analysis tools

**Who uses it:**
- Marine biologists and researchers
- Public health officials
- Shellfish industry professionals  
- Anyone concerned about coastal water safety

---

## 🚀 Quick Start (5 minutes)

**Want to see it working right now? Follow these simple steps:**

### Option 1: Use the Web Interface (Easiest!)

```bash
# 1. Open your terminal/command prompt
# 2. Navigate to the DATect folder
cd /path/to/DATect-Forecasting-Domoic-Acid

# 3. Start the web service
python3 run_api_service.py
```

**Then open your web browser and go to:** `http://localhost:8000`

You'll see a beautiful web interface where you can:
- Make predictions by filling out a form
- View system health and status  
- Access API documentation
- See example predictions

### Option 2: Use Docker (Most Reliable!)

```bash
# 1. Navigate to the DATect folder
cd /path/to/DATect-Forecasting-Domoic-Acid

# 2. Start the containerized system
docker-compose up --build datect-api

# 3. Open your browser to: http://localhost:8000
```

**That's it!** The system is now running and ready to use.

---

## 🧠 Understanding the System

### The DATect System Has 4 Main Parts:

#### 1. **Data Processing** (`dataset-creation.py`)
- Downloads satellite and climate data from the internet
- Processes local toxin measurement files
- Creates a unified dataset for training models
- **You run this when:** You want to update the data with latest information

#### 2. **Prediction Models** (`modular-forecast.py`)  
- Trains machine learning models on the processed data
- Tests model accuracy and performance
- Creates interactive dashboards for analysis
- **You run this when:** You want to train new models or analyze performance

#### 3. **Web API Service** (`run_api_service.py`)
- Provides a user-friendly web interface
- Offers REST API endpoints for integration
- Handles real-time predictions
- **You run this when:** You want to make predictions through a web interface

#### 4. **Scientific Validation** (`run_scientific_validation.py`)
- Performs rigorous scientific testing
- Validates model accuracy and reliability
- Generates detailed research reports
- **You run this when:** You need scientific validation for research or publication

---

## 📝 Step-by-Step Instructions

### Step 1: Prepare Your System

**Check if you have Python:**
```bash
python3 --version
# Should show something like: Python 3.9.7
```

**If you don't have Python, install it:**
- **Mac:** Install from [python.org](https://python.org) or use `brew install python3`
- **Windows:** Install from [python.org](https://python.org)
- **Linux:** Use your package manager: `sudo apt install python3 python3-pip`

### Step 2: Get the Required Software

**Install Python packages:**
```bash
# Navigate to DATect folder first
cd /path/to/your/DATect-Forecasting-Domoic-Acid

# Install required packages
pip3 install -r requirements.txt

# If you get permission errors, try:
pip3 install --user -r requirements.txt
```

### Step 3: Choose Your Path

**For most users (just want to make predictions):**
```bash
python3 run_api_service.py
# Then open: http://localhost:8000
```

**For researchers (want to analyze and validate):**
```bash
# First, process the data (10-60 minutes)
python3 dataset-creation.py

# Then run analysis and get interactive dashboard
python3 modular-forecast.py

# Finally, validate scientifically (2-5 minutes)
python3 run_scientific_validation.py
```

**For scientific validation only:**
```bash
# Run all validation tests (recommended)
python3 run_scientific_validation.py

# Or run specific tests
python3 run_scientific_validation.py --tests temporal,performance

# With detailed logging
python3 run_scientific_validation.py --verbose --output-dir ./my_results/
```

### Step 4: Understanding What You See

**Web Interface (`http://localhost:8000`):**
- **Prediction Form:** Enter oceanographic data to get toxin predictions
- **Health Status:** Shows if the system is working properly
- **API Documentation:** Technical details for developers
- **Example Predictions:** Sample forecasts to understand outputs

**Dashboard (from `modular-forecast.py`):**
- **Interactive Charts:** Click and explore prediction data
- **Model Performance:** See how accurate the predictions are
- **Site Comparisons:** Compare different coastal locations

---

## 🛠️ Different Ways to Use DATect

### 🌐 Web Interface Mode (Easiest)
**Perfect for:** Making individual predictions, exploring the system

```bash
python3 run_api_service.py
```
**Access:** http://localhost:8000
**What you get:** User-friendly web forms and instant predictions

### 📊 Research Dashboard Mode
**Perfect for:** Scientists, researchers, detailed analysis

```bash
python3 modular-forecast.py
```
**Access:** http://localhost:8065 (if dashboard opens)  
**What you get:** Interactive charts, model comparisons, scientific plots

### 🔬 Scientific Validation Mode
**Perfect for:** Peer review, publication, rigorous testing

```bash
# Complete validation suite
python3 run_scientific_validation.py

# Specific validation tests
python3 run_scientific_validation.py --tests temporal,statistical

# Advanced options
python3 run_scientific_validation.py --model-type xgboost --task regression --verbose
```
**What you get:** 
- Temporal data integrity validation
- Statistical significance tests  
- Model performance metrics
- Feature importance analysis
- Research-grade validation reports
- Publication-ready results

### 🐳 Docker Mode (Most Reliable)
**Perfect for:** Production use, consistent environments

```bash
# Basic service
docker-compose up --build datect-api

# Full system with dashboard
docker-compose --profile dashboard up --build

# Production deployment
docker-compose --profile production up -d
```

### 📱 Command Line Mode (Advanced)
**Perfect for:** Automation, scripting, batch processing

```bash
# Update data
python3 dataset-creation.py

# Train models
python3 -c "from forecasting.core.forecast_engine import ForecastEngine; engine = ForecastEngine(); print('Models trained!')"
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### ❌ "Module not found" errors
```bash
# Solution: Install missing packages
pip3 install fastapi uvicorn pandas numpy scikit-learn xgboost

# Or install everything:
pip3 install -r requirements.txt
```

#### ❌ "Port already in use" errors
```bash
# Solution: Kill existing processes
# On Mac/Linux:
pkill -f "python3.*run_api_service"

# On Windows:
# Use Task Manager to end Python processes
```

#### ❌ "Permission denied" errors
```bash
# Solution: Use user installation
pip3 install --user package_name

# Or use virtual environment:
python3 -m venv datect_env
source datect_env/bin/activate  # On Mac/Linux
# datect_env\Scripts\activate   # On Windows
pip install -r requirements.txt
```

#### ❌ Data files missing errors
```bash
# Solution: Run data processing first
python3 dataset-creation.py
# Wait for completion (10-60 minutes)
```

#### ❌ Web interface won't load
```bash
# Check if service is running:
curl http://localhost:8000/health

# If no response, restart the service:
python3 run_api_service.py
```

### Getting Help

**Check log files:**
```bash
# Look in the logs/ directory
ls logs/
cat logs/datect_main.log
```

**Test system components:**
```bash
# Test configuration
python3 -c "import config; print('Config OK')"

# Test data loading  
python3 -c "import pandas as pd; df = pd.read_parquet('final_output.parquet'); print(f'Data: {len(df)} rows')"

# Test forecasting engine
python3 -c "from forecasting.core.forecast_engine import ForecastEngine; engine = ForecastEngine(); print('Engine OK')"
```

---

## 🎓 Advanced Options

### Customizing the System

#### Change Prediction Models
Edit `config.py`:
```python
# Use different model
FORECAST_MODEL = "ridge"  # or "xgboost"

# Change prediction type
FORECAST_TASK = "classification"  # or "regression"

# Adjust performance
PARALLEL_JOBS = 4  # Use more CPU cores
```

#### Configure for Production
Set environment variables:
```bash
# Production settings
export DATECT_ENVIRONMENT=production
export DATECT_API_PORT=8080
export DATECT_LOG_LEVEL=INFO

# Then run
python3 run_api_service.py
```

#### Scientific Validation Options
```bash
# Run specific tests only
python3 run_scientific_validation.py --tests temporal,performance

# Change output location
python3 run_scientific_validation.py --output-dir ./my_results/

# More detailed logging
python3 run_scientific_validation.py --verbose

# Different model validation
python3 run_scientific_validation.py --model-type ridge --task classification
```

### Docker Advanced Usage

#### Scale for High Traffic
```bash
# Run multiple API instances
docker-compose up --scale datect-api=3

# Production with monitoring
docker-compose --profile production --profile monitoring up -d
```

#### Custom Configuration
Create `.env` file:
```
DATECT_ENVIRONMENT=production
DATECT_API_PORT=8080
DATECT_LOG_LEVEL=INFO
DATECT_SECRET_KEY=your-secret-key
```

### Integration with Other Systems

#### Using the API Programmatically
```python
import requests

# Make a prediction
response = requests.post('http://localhost:8000/predict', json={
    'model_name': 'xgboost',
    'data': {
        'sst': 15.2,
        'chlorophyll': 2.1,
        'da_lag_1': 0.5,
        # ... other parameters
    }
})

prediction = response.json()
print(f"Predicted DA level: {prediction['prediction']}")
```

---

## 🎯 Summary

**You now know how to:**
- ✅ Start the DATect system in different modes
- ✅ Use the web interface for predictions  
- ✅ Run scientific validation tests
- ✅ Troubleshoot common issues
- ✅ Customize the system for your needs

**Remember:**
- Start with the **web interface** - it's the easiest way to explore
- Run **scientific validation** if you need research-grade results  
- Use **Docker** for reliable, consistent operation
- Check the **logs/** directory if anything goes wrong

**Happy forecasting! 🌊🔬**

---

*For technical support, check the GitHub issues page or review the detailed documentation files (DOCKER_DEPLOYMENT_GUIDE.md, WEB_INTERFACE_GUIDE.md, etc.)*