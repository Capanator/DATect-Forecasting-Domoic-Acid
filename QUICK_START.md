# DATect - One-Line Startup Guide

## 🚀 Run Everything with One Command

### Single Command Startup:
```bash
cd ~/Documents/GitHub/DATect-Forecasting-Domoic-Acid
./run-datect.sh
```

That's it! This command will:

✅ **Check all prerequisites** (Python, Node.js, data files)  
✅ **Install dependencies** (Python packages, npm packages)  
✅ **Start backend API** (http://localhost:8000)  
✅ **Start frontend** (http://localhost:3000)  
✅ **Auto-open browser** to the web application  

## What You'll See

The script will show colored output like:
```
🚀 DATect Complete System Launcher
====================================
📋 Checking prerequisites...
✅ Data file found
✅ Python 3 available  
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

1. **🌐 Main Interface**: The web app opens automatically
2. **⚙️ System Config**: Click "System Config" to modify:
   - Forecast Mode (realtime/retrospective)
   - Forecast Task (regression/classification)  
   - Forecast Model (xgboost/ridge)
3. **📊 Generate Forecasts**: 
   - Select date (2007-2018 range)
   - Choose monitoring site
   - Click "Generate Enhanced Forecast"
4. **📈 View Results**: See all original Dash graphs:
   - DA Level Forecast (gradient visualization)
   - Risk Category Distribution  
   - Feature Importance Charts

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
python3 dataset-creation.py
```

## Alternative Commands

If you prefer manual control:

**Backend only:**
```bash
python3 backend/main.py
```

**Frontend only:**
```bash
cd frontend && npm run dev
```

**Complete system with one command:**
```bash
./run-datect.sh
```

---

**Your DATect system is now a complete modern web application!** 🎉

- ✅ One-command startup
- ✅ Auto-opening browser
- ✅ All original Dash functionality
- ✅ Modern React interface
- ✅ Configuration management
- ✅ Enhanced visualizations