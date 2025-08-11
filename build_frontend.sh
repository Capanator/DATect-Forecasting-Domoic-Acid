#!/bin/bash
# Build frontend for deployment

set -e

echo "🏗️ Building frontend locally..."

if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found"
    exit 1
fi

cd frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

if [ -d "dist" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf dist
fi

echo "🔨 Building frontend..."
NODE_OPTIONS="--max_old_space_size=4096" npm run build

if [ -d "dist" ]; then
    echo "✅ Frontend built successfully"
    echo "📁 Build output:"
    ls -la dist/
else
    echo "❌ Frontend build failed"
    exit 1
fi

cd ..
echo "🎉 Frontend build complete"