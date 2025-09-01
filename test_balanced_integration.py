#!/usr/bin/env python3
"""
Test Balanced Model Integration
===============================

Quick test to verify the balanced model works in the API pipeline.
"""

import requests
import json
from datetime import datetime, timedelta

def test_balanced_model_api():
    """Test the balanced_xgboost model through the API."""
    print("🧪 Testing Balanced Model API Integration")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: Check if balanced_xgboost is in supported models
        print("1. Checking supported models...")
        models_response = requests.get(f"{base_url}/api/models")
        
        if models_response.status_code == 200:
            models_data = models_response.json()
            regression_models = models_data['available_models']['regression']
            
            if 'balanced_xgboost' in regression_models:
                print("✅ balanced_xgboost found in supported models")
                print(f"   Description: {models_data['descriptions'].get('balanced_xgboost', 'N/A')}")
            else:
                print("❌ balanced_xgboost not found in supported models")
                print(f"   Available: {regression_models}")
                return False
        else:
            print(f"❌ Failed to get models: {models_response.status_code}")
            return False
        
        # Test 2: Update config to use balanced_xgboost
        print("\n2. Setting config to use balanced_xgboost...")
        config_data = {
            "forecast_mode": "realtime",
            "forecast_task": "regression",
            "forecast_model": "balanced_xgboost",
            "selected_sites": []
        }
        
        config_response = requests.post(f"{base_url}/api/config", json=config_data)
        
        if config_response.status_code == 200:
            result = config_response.json()
            if result.get('success'):
                print("✅ Config updated successfully")
                print(f"   Current model: {result['config']['forecast_model']}")
            else:
                print("❌ Failed to update config")
                return False
        else:
            print(f"❌ Config update failed: {config_response.status_code}")
            return False
        
        # Test 3: Get available sites
        print("\n3. Getting available sites...")
        sites_response = requests.get(f"{base_url}/api/sites")
        
        if sites_response.status_code == 200:
            sites_data = sites_response.json()
            sites = sites_data['sites']
            print(f"✅ Found {len(sites)} sites: {sites[0]}, {sites[1]}...")
        else:
            print(f"❌ Failed to get sites: {sites_response.status_code}")
            return False
        
        # Test 4: Generate a forecast with balanced_xgboost
        print("\n4. Generating forecast with balanced_xgboost...")
        
        # Use a recent date
        forecast_date = (datetime.now() - timedelta(days=60)).date()
        test_site = sites[0]  # Use first available site
        
        forecast_data = {
            "date": forecast_date.isoformat(),
            "site": test_site,
            "task": "regression",
            "model": "balanced_xgboost"
        }
        
        forecast_response = requests.post(f"{base_url}/api/forecast", json=forecast_data)
        
        if forecast_response.status_code == 200:
            forecast_result = forecast_response.json()
            
            if forecast_result.get('success'):
                prediction = forecast_result.get('prediction')
                training_samples = forecast_result.get('training_samples')
                
                print("✅ Forecast generated successfully!")
                print(f"   Site: {test_site}")
                print(f"   Date: {forecast_date}")
                print(f"   Predicted DA: {prediction:.2f} μg/g")
                print(f"   Training samples: {training_samples}")
                
                # Check if it's a reasonable prediction
                if prediction is not None and 0 <= prediction <= 100:
                    print("✅ Prediction value looks reasonable")
                else:
                    print(f"⚠️ Unusual prediction value: {prediction}")
                
            else:
                error = forecast_result.get('error', 'Unknown error')
                print(f"❌ Forecast failed: {error}")
                return False
        else:
            print(f"❌ Forecast request failed: {forecast_response.status_code}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Balanced model integration successful!")
        print("✅ Ready for production use!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("💡 Please start the server with: python run_datect.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_balanced_model_api()
    exit(0 if success else 1)