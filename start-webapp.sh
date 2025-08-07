#\!/bin/bash
# DATect Web Application Startup Script for macOS

echo "🚀 Starting DATect Web Application"
echo "=================================="

# Check if data file exists
if [ \! -f "data/processed/final_output.parquet" ]; then
    echo "❌ Data file not found"
    exit 1
fi
echo "✅ Data file found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --quiet fastapi uvicorn pydantic pandas numpy scikit-learn xgboost plotly requests

# Start backend
echo "🖥️  Starting backend..."
python3 backend/main.py &
BACKEND_PID=$\!
echo "Backend PID: $BACKEND_PID"

# Wait for backend
echo "⏳ Waiting for backend to start..."
sleep 5

# Test if it's running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is ready\!"
    echo ""
    echo "🎉 SUCCESS\! Your DATect API is running\!"
    echo "🔗 API Documentation: http://localhost:8000/docs"
    echo "💚 Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping backend..."
    kill $BACKEND_PID 2>/dev/null || true
    echo "✅ Stopped"
}

trap cleanup EXIT INT TERM
wait $BACKEND_PID
