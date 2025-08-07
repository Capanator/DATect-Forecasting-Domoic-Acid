#!/bin/bash
# DATect Complete System Launcher
# Single command to start backend, frontend, and open browser

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 DATect Complete System Launcher${NC}"
echo -e "${BLUE}====================================${NC}"
echo "Working directory: $(pwd)"

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port is already in use. Stopping existing process...${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=30
    local count=0
    
    echo -e "${YELLOW}⏳ Waiting for $name to be ready...${NC}"
    while [ $count -lt $max_wait ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is ready!${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        if [ $((count % 5)) -eq 0 ]; then
            echo "   Still waiting... ($count/$max_wait)"
        fi
    done
    echo -e "${RED}❌ $name failed to start within $max_wait seconds${NC}"
    return 1
}

# Function to open browser (cross-platform)
open_browser() {
    local url=$1
    echo -e "${GREEN}🌐 Opening $url in browser...${NC}"
    
    # Detect platform and open browser
    if command -v open >/dev/null 2>&1; then
        # macOS
        open "$url"
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open "$url"
    elif command -v start >/dev/null 2>&1; then
        # Windows
        start "$url"
    else
        echo -e "${YELLOW}⚠️  Could not auto-open browser. Please visit: $url${NC}"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down DATect system...${NC}"
    
    # Kill background processes
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Kill any remaining processes on our ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    
    echo -e "${GREEN}✅ DATect system stopped${NC}"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check data file
if [ ! -f "data/processed/final_output.parquet" ]; then
    echo -e "${RED}❌ Data file not found at data/processed/final_output.parquet${NC}"
    echo "Please run 'python3 dataset-creation.py' first to generate the data"
    exit 1
fi
echo -e "${GREEN}✅ Data file found${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 available${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js from https://nodejs.org or run: brew install node"
    exit 1
fi
echo -e "${GREEN}✅ Node.js available${NC}"

# Check ports
echo -e "${BLUE}🔍 Checking ports...${NC}"
check_port 8000
check_port 3000

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"

# Python dependencies
echo "Installing Python dependencies..."
pip3 install --quiet fastapi uvicorn pydantic pandas numpy scikit-learn xgboost plotly requests 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Some Python packages may already be installed${NC}"
}

# Node.js dependencies
echo "Installing Node.js dependencies..."
cd frontend
if [ ! -d "node_modules" ] || [ ! -f "package-lock.json" ]; then
    npm install > /dev/null 2>&1
fi
cd ..
echo -e "${GREEN}✅ All dependencies installed${NC}"

# Start backend
echo -e "${BLUE}🖥️  Starting backend API server...${NC}"
python3 backend/main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend
if ! wait_for_service "http://localhost:8000/health" "Backend API"; then
    echo -e "${RED}❌ Backend failed to start. Check backend.log for errors:${NC}"
    tail -10 backend.log 2>/dev/null || echo "No log file found"
    exit 1
fi

# Start frontend
echo -e "${BLUE}🎨 Starting frontend development server...${NC}"
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend
if ! wait_for_service "http://localhost:3000" "Frontend"; then
    echo -e "${RED}❌ Frontend failed to start. Check frontend.log for errors:${NC}"
    tail -10 frontend.log 2>/dev/null || echo "No log file found"
    exit 1
fi

# Open browser
sleep 1
open_browser "http://localhost:3000"

# Success message
echo ""
echo -e "${GREEN}🎉 DATect System is now running!${NC}"
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}🔗 Frontend Web App:${NC} http://localhost:3000"
echo -e "${GREEN}🔗 Backend API:${NC} http://localhost:8000"
echo -e "${GREEN}📚 API Documentation:${NC} http://localhost:8000/docs"
echo ""
echo -e "${BLUE}📖 How to use:${NC}"
echo "1. 🌐 Browser should open automatically to http://localhost:3000"
echo "2. ⚙️  Click 'System Config' to modify forecasting settings"
echo "3. 📊 Select date, site, and generate enhanced forecasts"
echo "4. 📈 View all original Dash graphs in the modern web interface"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the entire system${NC}"

# Keep script running and monitor processes
while kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; do
    sleep 2
done

echo -e "${RED}❌ One or more services have stopped unexpectedly${NC}"
echo "Check the log files for details:"
echo "- Backend: backend.log"
echo "- Frontend: frontend.log"