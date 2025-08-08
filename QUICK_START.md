# DATect - One-Line Startup Guide

## 🚀 Run Everything with One Command

### Single Command Startup:
```bash
# Navigate to project directory
cd DATect-Forecasting-Domoic-Acid

# Launch complete system with scientific validation
python run_datect.py
```

That's it! This command will:

✅ **Validate scientific data integrity** (temporal safeguards, data leakage checks)  
✅ **Check all prerequisites** (Python, Node.js, data files)  
✅ **Install dependencies** (Python packages, npm packages)  
✅ **Start backend API** (http://localhost:8000)  
✅ **Start frontend** (http://localhost:3000)  
✅ **Auto-open browser** to the web application  

## What You'll See

The launcher will show scientific validation output like:
```
🚀 DATect Scientific System Launcher
====================================
🔬 Running Scientific Validation...
✅ Temporal integrity: 7/7 tests PASSED
✅ Data validation: All checks PASSED
✅ Model configuration: Valid
✅ Scientific safeguards: Active

📋 Checking system prerequisites...
✅ Data file found: data/processed/final_output.parquet
✅ Python 3.8+ available
✅ Node.js available
📦 Installing dependencies...
🖥️  Starting backend API server...
✅ Backend API is ready!
🎨 Starting frontend development server...
✅ Frontend is ready!
🌐 Opening http://localhost:3000 in browser...

🎉 DATect System is now running!
====================================
🔗 Frontend Web App: http://localhost:3000
🔗 Backend API: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
```

## Using the Application

Once the browser opens:

### 🏠 Dashboard Page
1. **Real-time Forecasting**: 
   - Select date and monitoring site
   - Generate enhanced forecasts with uncertainty ranges
   - View DA levels, risk categories, and feature importance
2. **Retrospective Analysis**:
   - Run historical model validation
   - Compare actual vs predicted values across sites
   - View performance metrics (R² ≈ 0.529 for XGBoost)

### 📊 Historical Analysis Page
1. **Correlation Heatmaps**: Variable relationships with scientific colorscales
2. **Sensitivity Analysis**: Feature importance using Sobol indices and permutation methods
3. **Time Series Comparison**: DA vs Pseudo-nitzschia over time
4. **Waterfall Plots**: Site-by-latitude visualization with reference bars
5. **Spectral Analysis**: Frequency domain analysis with XGBoost comparisons

## Stopping the System

Press **Ctrl+C** in the terminal where you ran the script. It will automatically:
- Stop the backend server
- Stop the frontend server
- Clean up all processes
- Show confirmation message

## Troubleshooting

**If ports are busy:**
The script automatically kills processes on ports 8000 and 3000

**If dependencies fail:**
Run manually first:
```bash
pip3 install fastapi uvicorn pydantic pandas numpy scikit-learn xgboost
cd frontend && npm install
```

**If browser doesn't open:**
Manually visit: http://localhost:3000

**If data file is missing:**
```bash
python dataset-creation.py  # Takes 30-60 minutes
```

## Alternative Commands

If you prefer manual control:

**Backend only:**
```bash
cd backend && uvicorn api:app --reload
```

**Frontend only:**
```bash
cd frontend && npm run dev
```

**Complete system with scientific validation:**
```bash
python run_datect.py
```

---

**Your DATect system is now a scientifically validated web application!** 🎉

- ✅ One-command startup with scientific validation
- ✅ Temporal integrity safeguards (zero data leakage)
- ✅ 5 interactive scientific visualizations
- ✅ XGBoost forecasting with uncertainty ranges
- ✅ Modern React interface with responsive design
- ✅ FastAPI backend with comprehensive error handling
- ✅ Peer-review ready validation framework