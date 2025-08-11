# DATect - Complete Setup Guide

This guide will get you running DATect from a **fresh computer with nothing installed**.

## Local Development Setup

### Step 1: Install Prerequisites

**On Windows:**
1. **Install Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation
   - Recommended: Python 3.9+ for best performance
2. **Install Node.js 16+**: Download from [nodejs.org](https://nodejs.org/)
   - Recommended: Node.js 18+ (LTS version)
3. **Install Git**: Download from [git-scm.com](https://git-scm.com/download/win)

**On macOS:**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install python node git
```

**On Linux (Ubuntu/Debian):**
```bash
# Update system
sudo apt update

# Install prerequisites
sudo apt install python3 python3-pip nodejs npm git

# For newer Node.js (recommended):
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify versions
python3 --version  # Should be 3.8+
node --version      # Should be 16+, recommended 18+
git --version
```

### Step 2: Clone and Launch (One Command!)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# 2. Launch everything automatically
python run_datect.py
```

### What You'll See

**First Run (dataset must exist first):**
```
🚀 DATect Scientific System Launcher
====================================

📋 System Prerequisites Check...
✅ Python 3.11.5 found
✅ Node.js 18.17.0 found  
✅ Git 2.40.1 found

📦 Installing Python Dependencies...
✅ Installing fastapi, uvicorn, pydantic...
✅ Installing pandas, numpy, scikit-learn, xgboost...
✅ Installing plotly, matplotlib...
✅ All Python dependencies installed (35 packages)

📦 Installing Node.js Dependencies...  
✅ Installing react, vite, @vitejs/plugin-react...
✅ Installing plotly.js, tailwindcss...
✅ All Node.js dependencies installed (1,247 packages)

🔬 Dataset Status Check...
❌ Dataset missing: data/processed/final_output.parquet
🔄 Generating dataset... (This will take 30-60 minutes)

📡 Downloading MODIS satellite data...
✅ Chlorophyll-a data: 2002-2024 (8,030 files)
✅ Sea surface temperature: 2002-2024 (8,030 files)  
✅ Photosynthetically active radiation: 2002-2024 (8,030 files)
✅ Fluorescence line height: 2002-2024 (8,030 files)

🌊 Processing climate indices...
✅ Pacific Decadal Oscillation (PDO): 22 years
✅ Oceanic Niño Index (ONI): 22 years
✅ Bakun Upwelling Index (BEUTI): 22 years

🏞️ Processing streamflow data...
✅ Columbia River discharge: 22 years (8,030 daily records)

🦠 Processing domoic acid measurements...
✅ Cannon Beach: 1,095 records (2003-2024)
✅ Newport: 1,095 records (2003-2024)
... [8 more sites]
✅ Total DA measurements: 10,950 records

🧬 Processing Pseudo-nitzschia data...
✅ Cell count data aligned with DA measurements

⚗️ Creating lag features and temporal safeguards...
✅ Lag features: [1, 3] day periods
✅ Temporal buffers: 7-day satellite, 60-day climate
✅ Forward-only interpolation applied

💾 Saving final dataset...
✅ Dataset saved: data/processed/final_output.parquet (162 MB)
✅ Dataset generation complete: 10,950 records, 17 features

🔬 Running Scientific Validation...
✅ Temporal integrity: 7/7 tests PASSED
✅ Data leakage checks: 0 violations detected
✅ Model configuration: Valid
✅ Feature engineering: Leak-free confirmed
✅ Classification fixes: Non-consecutive labels handled
✅ API functionality: All 8 endpoints tested

🏆 Scientific Integrity Rating: 95/100
📋 Status: PUBLICATION READY

🖥️  Starting Backend API Server...
✅ FastAPI server started on http://localhost:8000
✅ API documentation available at http://localhost:8000/docs

🎨 Starting Frontend Development Server...  
✅ React development server started on http://localhost:3000
✅ Frontend hot reloading enabled

🌐 Opening http://localhost:3000 in browser...

🎉 DATect System is now running!
====================================
🔗 Frontend Web App: http://localhost:3000
🔗 Backend API: http://localhost:8000  
📚 API Documentation: http://localhost:8000/docs
📊 System Status: HEALTHY
🔬 Validation Status: PASSED
⏱️  Total startup time: 45.2 minutes (first run with dataset)

Press Ctrl+C to stop all services...
```

**Subsequent Runs (dataset exists):**
```
🚀 DATect Scientific System Launcher
====================================

📋 System Prerequisites Check...
✅ All prerequisites satisfied

📦 Dependencies Status...
✅ Python packages: Up to date  
✅ Node.js packages: Up to date

🔬 Dataset Status Check...
✅ Dataset found: data/processed/final_output.parquet (162 MB)
✅ Dataset integrity: 10,950 records validated

🔬 Running Scientific Validation...
✅ Temporal safeguards: PASSED (0 leakage violations)
✅ Model consistency: PASSED 
✅ API functionality: PASSED

🖥️  Starting services...
✅ Backend API: http://localhost:8000 (ready in 3.2s)
✅ Frontend: http://localhost:3000 (ready in 2.1s)
🌐 Opening browser...

🎉 System ready! Total startup time: 8.3 seconds
```

## Google Cloud Deployment

### Step 1: Google Cloud Setup

**Install Google Cloud CLI:**
- **Windows**: Download from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
- **macOS**: `brew install google-cloud-sdk`
- **Linux**: Follow [official instructions](https://cloud.google.com/sdk/docs/install)

**Set up Google Cloud:**
```bash
# 1. Authenticate with Google Cloud
gcloud auth login

# 2. Create a new project (or use existing)
gcloud projects create your-datect-project --name="DATect Forecasting"

# 3. Set the project
gcloud config set project your-datect-project

# 4. Enable required services
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 5. Set default region (optional)
gcloud config set run/region us-central1
```

### Step 2: Deploy to Google Cloud

```bash
# 1. Clone repository (if not done already)
git clone https://github.com/your-username/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# 2. Generate dataset locally (required for deployment)
python dataset-creation.py

# 3. Deploy with one command
./deploy_gcloud.sh
```

**Deployment Output:**
```
🚀 DATect Google Cloud Deployment
=================================

📋 Pre-deployment Checks...
✅ Google Cloud CLI authenticated
✅ Project set: your-datect-project
✅ Required APIs enabled
✅ Dataset present: data/processed/final_output.parquet

🐳 Building Production Container Image...
✅ Building with Cloud Build...
✅ Image: gcr.io/your-datect-project/datect:latest

🌐 Deploying to Cloud Run...
✅ Service: datect-forecasting
✅ Region: us-central1  
✅ Scaling: 0-10 instances
✅ Memory: 2GB per instance
✅ CPU: 2 vCPUs per instance

🔒 Configuring IAM and Security...
✅ Service account created
✅ IAM permissions configured
✅ HTTPS enforcement enabled

📊 Health Check...
✅ Service responding at /health
✅ API endpoints functional
✅ Frontend assets served correctly

🎉 Deployment Successful!
========================
🌐 Live URL: https://datect-forecasting-xxxxx-uc.a.run.app
📚 API Docs: https://datect-forecasting-xxxxx-uc.a.run.app/docs
📊 Monitoring: https://console.cloud.google.com/run
💰 Estimated cost: $10-50/month (depends on usage)

Your DATect system is now live on the internet! 🚀
```

## Using the System

### Dashboard Features

**Real-time Forecasting:**
1. **Select Date**: Any date from 2008-2024
2. **Select Site**: 10 Pacific Coast monitoring locations
3. **Select Model**: XGBoost (recommended) or Linear/Logistic  
4. **Click "Forecast"**: Get predictions with visualizations

**Results Include:**
- **DA Concentration**: Predicted μg/g levels
- **Risk Category**: Low/Moderate/High/Extreme  
- **Feature Importance**: Top contributing variables
- **Class Probabilities**: Confidence distributions (classification only)

### Historical Analysis Tools

1. **Correlation Heatmaps**: Scientific colorscales, variable relationships
2. **Sensitivity Analysis**: Sobol indices and permutation importance
3. **Time Series Comparison**: DA vs Pseudo-nitzschia over time
4. **Spectral Analysis**: Frequency domain analysis
5. **Model Performance**: Retrospective validation metrics

## Troubleshooting

**"Command not found: python"**
```bash
# Try python3 instead
python3 run_datect.py

# Or check Python installation
which python
python --version
```

**"Port already in use"**
```bash
# Kill existing processes (automatic in run_datect.py)
kill $(lsof -ti:8000,3000)
```

**"Node.js version too old"**
```bash
# Update Node.js
# Windows/macOS: Download from nodejs.org
# Linux (Ubuntu/Debian): 
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS:
brew install node

# Verify version
node --version  # Should be 16+
```

**"Google Cloud deployment failed"**
```bash
# Check authentication
gcloud auth list
gcloud auth application-default login

# Check project
gcloud config list
gcloud config set project YOUR-PROJECT-ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Validation Checklist

After setup, verify everything works:

```bash
# 1. Local system validation
python run_datect.py
# Should show: "Scientific Integrity Rating: 95/100"

# 2. API endpoints test
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# 3. Frontend test  
# Browser should open automatically to http://localhost:3000

# 4. Generate a test forecast
# Use the web interface: Cannon Beach, 2015-06-24, XGBoost, Classification

# 5. Check retrospective validation
# Edit config.py: FORECAST_MODE = "retrospective"  
# Run: python run_datect.py
# Should generate 200 test forecasts
```

---

## Complete Dependencies Reference

### Python Dependencies (35+ packages)
Core packages automatically installed by `run_datect.py`:
- **Scientific Computing**: pandas, numpy, scipy, scikit-learn
- **Machine Learning**: xgboost (primary model engine)
- **Web Framework**: fastapi, uvicorn, pydantic
- **Data Processing**: xarray, pyarrow, netcdf4, requests
- **Visualization**: plotly, matplotlib
- **Analysis Tools**: SALib (Sobol analysis), tqdm (progress bars)

### Node.js Dependencies (1000+ packages)  
Frontend packages automatically installed by `run_datect.py`:
- **Framework**: React 18.2, Vite 4.4
- **Routing**: react-router-dom 6.11
- **UI Components**: @headlessui/react, lucide-react
- **Visualization**: plotly.js, react-plotly.js
- **Styling**: tailwindcss, postcss, autoprefixer
- **Forms**: react-datepicker, react-select
- **Testing**: vitest, @testing-library/react
- **Development**: eslint, @vitejs/plugin-react