#!/bin/bash
# DATect Backend Startup Script

cd "$(dirname "$0")"  # Change to script directory (project root)

echo "Starting DATect Backend API..."
echo "Working directory: $(pwd)"

# Check if data file exists
if [ ! -f "data/processed/final_output.parquet" ]; then
    echo "Error: Data file not found at data/processed/final_output.parquet"
    echo "Please run 'python dataset-creation.py' first to generate the data"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip3 show fastapi > /dev/null 2>&1 || pip3 install fastapi uvicorn pydantic

# Start the backend server
echo "Starting server on http://localhost:8000"
python3 backend/main.py