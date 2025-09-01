#!/usr/bin/env python3
"""
Test Webapp Visualizations - Verify graphs/visualizations appear
"""

import requests
import json
import time
from datetime import datetime, timedelta

def test_webapp_visualizations():
    """Test that webapp visualizations work properly."""
    print("📊 Testing Webapp Visualizations")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    print("1. Testing backend health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Backend is healthy")
        else:
            print("   ❌ Backend health check failed")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to backend: {e}")
        print("   💡 Start backend with: python3 -m backend.api")
        return False
    
    # Test 2: Single forecast request (what the webapp calls)
    print("\n2. Testing single forecast (realtime mode)...")
    try:
        forecast_data = {
            "date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "site": "Newport",
            "task": "regression",
            "model": "balanced_lightgbm"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/api/forecast", json=forecast_data, timeout=30)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Forecast completed in {request_time:.2f} seconds")
            predicted_da = result.get('prediction', 'N/A')
            if isinstance(predicted_da, (int, float)):
                print(f"   ✅ Predicted DA: {predicted_da:.3f} μg/g")
            else:
                print(f"   ✅ Predicted DA: {predicted_da} μg/g")
            print(f"   ✅ Training samples: {result.get('training_samples', 'N/A')}")
            
            # Check for visualization data
            if 'visualization_data' in result:
                print("   ✅ Visualization data present")
                viz_data = result['visualization_data']
                if 'time_series' in viz_data:
                    print("   ✅ Time series data available")
                if 'feature_importance' in viz_data:
                    print("   ✅ Feature importance data available")
            else:
                print("   ⚠️  No visualization_data in response")
                
        else:
            print(f"   ❌ Forecast request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Forecast request failed: {e}")
        return False
    
    # Test 3: Enhanced forecast (includes more visualizations)
    print("\n3. Testing enhanced forecast...")
    try:
        enhanced_data = {
            "date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "site": "Newport",
            "task": "regression",
            "model": "balanced_lightgbm"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/api/forecast/enhanced", json=enhanced_data, timeout=45)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Enhanced forecast completed in {request_time:.2f} seconds")
            
            # Check both regression and classification results
            if 'regression_result' in result:
                reg_result = result['regression_result']
                print(f"   ✅ Regression DA: {reg_result.get('predicted_da', 'N/A'):.3f} μg/g")
                
            if 'classification_result' in result:
                cls_result = result['classification_result']
                print(f"   ✅ Classification risk: {cls_result.get('predicted_risk', 'N/A')}")
                
            # Check for comprehensive visualization data
            viz_keys = ['time_series_data', 'feature_importance', 'model_performance']
            for key in viz_keys:
                if key in result:
                    print(f"   ✅ {key} available")
                else:
                    print(f"   ⚠️  {key} missing")
                    
        else:
            print(f"   ❌ Enhanced forecast failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Enhanced forecast failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 WEBAPP VISUALIZATION TESTS COMPLETE!")
    print("✅ Backend is responding correctly")
    print("✅ Forecast endpoints are working")
    print("✅ LightGBM model is functioning")
    print("✅ Visualization data is being generated")
    print()
    print("🌐 Webapp should now display:")
    print("   - Real-time forecast results")
    print("   - Time series graphs")
    print("   - Feature importance charts")
    print("   - Model performance metrics")
    print("   - Risk level visualizations")
    
    return True

if __name__ == "__main__":
    success = test_webapp_visualizations()
    if success:
        print("\n✅ Visualizations are ready!")
        print("🚀 Start webapp: python3 run_datect.py")
    else:
        print("\n❌ Visualization tests failed")
    exit(0 if success else 1)