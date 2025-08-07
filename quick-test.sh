#!/bin/bash
# Quick test script for DATect API

echo "🧪 Quick DATect API Test"
echo "======================="

# Test if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "❌ Backend is not running. Start it first with:"
    echo "   ./start-webapp.sh"
    exit 1
fi

echo ""
echo "🔍 Testing API endpoints..."

# Test health
echo -n "Health check: "
curl -s http://localhost:8000/health | python3 -m json.tool || echo "Failed"

echo ""
echo -n "Sites available: "
curl -s http://localhost:8000/api/sites | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data['sites']))" 2>/dev/null || echo "Failed"

echo ""
echo -n "Models available: "
curl -s http://localhost:8000/api/models | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Regression: {len(data['available_models']['regression'])}, Classification: {len(data['available_models']['classification'])}\")" 2>/dev/null || echo "Failed"

echo ""
echo "🎯 Testing forecast..."
curl -s -X POST http://localhost:8000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"date":"2010-06-15","site":"Newport","task":"regression","model":"xgboost"}' | \
  python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Success: {data['success']}, Prediction: {data.get('prediction', 'N/A')}\")" 2>/dev/null || echo "Failed"

echo ""
echo "📊 Testing historical data..."
curl -s "http://localhost:8000/api/historical/Newport?limit=3" | \
  python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Records: {data['count']}\")" 2>/dev/null || echo "Failed"

echo ""
echo "✅ All tests completed!"
echo ""
echo "🌐 Open these URLs in your browser:"
echo "   API Docs: http://localhost:8000/docs"
echo "   Health:   http://localhost:8000/health"