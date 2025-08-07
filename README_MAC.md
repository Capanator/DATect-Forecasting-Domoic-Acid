# DATect Web Application - macOS Setup

## Quick Start Guide

### Step 1: Navigate to Project Directory
```bash
cd ~/Documents/GitHub/DATect-Forecasting-Domoic-Acid
```

### Step 2: Start the Web Application
```bash
./start-webapp.sh
```

This will:
- ✅ Check prerequisites
- 📦 Install Python dependencies
- 🚀 Start the backend API server
- 🔗 Show you URLs to access

### Step 3: Test It Works
In another terminal:
```bash
./quick-test.sh
```

## What You'll See

When `start-webapp.sh` runs successfully, you'll see:

```
🚀 Starting DATect Web Application
Working directory: /Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid
==================================
✅ Data file found
✅ Python 3 available
📦 Installing Python dependencies...
🖥️  Starting backend API server...
Backend PID: 12345
⏳ Waiting for backend to start...
✅ Backend is ready!

🎉 Backend is running successfully!
==================================
🔗 Backend API: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
💚 Health Check: http://localhost:8000/health

🧪 Test the API with these sample calls:
curl http://localhost:8000/health
curl http://localhost:8000/api/sites

📱 To start the frontend (in another terminal):
cd frontend && npm install && npm run dev
Then visit: http://localhost:3000

Press Ctrl+C to stop the backend server
```

## URLs to Visit

- **API Documentation**: http://localhost:8000/docs (Interactive API docs)
- **Health Check**: http://localhost:8000/health
- **Frontend** (if running): http://localhost:3000

## Testing the API

### Basic Health Check
```bash
curl http://localhost:8000/health
```

### Get Available Sites
```bash
curl http://localhost:8000/api/sites
```

### Make a Forecast
```bash
curl -X POST http://localhost:8000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"date":"2010-06-15","site":"Newport","task":"regression","model":"xgboost"}'
```

### Get Historical Data
```bash
curl "http://localhost:8000/api/historical/Newport?limit=5"
```

## Optional: Start the Frontend

If you want the full web interface:

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

Then visit: http://localhost:3000

## Stopping the Application

Press **Ctrl+C** in the terminal where you ran `./start-webapp.sh`

## Troubleshooting

### Port Already in Use
```bash
# Kill processes on port 8000
lsof -ti:8000 | xargs kill -9
```

### Missing Dependencies
```bash
pip3 install fastapi uvicorn pydantic pandas numpy scikit-learn xgboost plotly
```

### Permission Denied
```bash
chmod +x start-webapp.sh
chmod +x quick-test.sh
```

### Data File Missing
If you see "Data file not found", run:
```bash
python3 dataset-creation.py
```

## File Structure

Your project should have:
```
DATect-Forecasting-Domoic-Acid/
├── backend/                 # FastAPI server
├── frontend/               # React application
├── data/processed/         # Data files
├── start-webapp.sh         # Main startup script
├── quick-test.sh          # Testing script
└── README_MAC.md          # This file
```

## Success Indicators

✅ **Backend Working**: You can visit http://localhost:8000/docs  
✅ **API Working**: `curl http://localhost:8000/health` returns JSON  
✅ **Forecasting Working**: API returns predictions for valid requests  
✅ **Frontend Working**: (Optional) Visit http://localhost:3000  

## Next Steps

1. **Test the API**: Use the interactive docs at http://localhost:8000/docs
2. **Try Forecasting**: Submit forecast requests with different sites and dates
3. **Explore Historical Data**: Query historical DA measurements
4. **Start Frontend**: Follow the frontend setup if you want the full web UI

Your DATect web application is now ready to use! 🎉