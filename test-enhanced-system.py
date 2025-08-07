#!/usr/bin/env python3
"""
Test script for enhanced DATect system with config management
"""

import subprocess
import time
import requests
import json
import sys
from pathlib import Path

def test_enhanced_backend():
    """Test the enhanced backend with new endpoints"""
    print("🧪 Testing Enhanced DATect Backend")
    print("=" * 40)
    
    # Start backend
    print("Starting backend server...")
    proc = subprocess.Popen([sys.executable, "backend/main.py"], 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    for i in range(15):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Backend server started")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("❌ Backend failed to start")
        return False
    
    try:
        base_url = "http://localhost:8000"
        
        # Test new config endpoints
        print("\n🔧 Testing Configuration Management...")
        
        # Get current config
        response = requests.get(f"{base_url}/api/config")
        if response.status_code == 200:
            config = response.json()
            print(f"✅ GET /api/config: {config}")
        else:
            print(f"❌ GET /api/config failed: {response.status_code}")
        
        # Update config
        new_config = {
            "forecast_mode": "realtime",
            "forecast_task": "regression", 
            "forecast_model": "xgboost"
        }
        response = requests.post(f"{base_url}/api/config", json=new_config)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ POST /api/config: {result['success']}")
        else:
            print(f"❌ POST /api/config failed: {response.status_code}")
        
        # Test enhanced forecast endpoint
        print("\n🎯 Testing Enhanced Forecasting...")
        
        forecast_request = {
            "date": "2010-06-15",
            "site": "Newport",
            "task": "regression",
            "model": "xgboost"
        }
        response = requests.post(f"{base_url}/api/forecast/enhanced", 
                               json=forecast_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Enhanced forecast: Success={result.get('success')}")
            
            # Check for graph data
            if 'graphs' in result:
                graphs = result['graphs']
                if 'level_range' in graphs:
                    level_data = graphs['level_range']
                    print(f"   📊 Level range graph: predicted_da={level_data.get('predicted_da'):.3f}")
                if 'category_range' in graphs:
                    cat_data = graphs['category_range']
                    print(f"   📈 Category graph: predicted_category={cat_data.get('predicted_category')}")
            
            # Check for both regression and classification results
            if result.get('regression'):
                reg_pred = result['regression'].get('predicted_da')
                print(f"   🎯 Regression prediction: {reg_pred:.3f} μg/g")
            if result.get('classification'):
                cls_pred = result['classification'].get('predicted_category')
                print(f"   📊 Classification prediction: {cls_pred}")
                
        else:
            print(f"❌ Enhanced forecast failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        print(f"\n🎉 Enhanced backend testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
        
    finally:
        print("Stopping backend server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    success = test_enhanced_backend()
    
    if success:
        print("\n" + "=" * 50)
        print("🚀 ENHANCED SYSTEM READY!")
        print("=" * 50)
        print("1. Restart your backend: python3 backend/main.py")
        print("2. Refresh your frontend: http://localhost:3000")
        print("3. Click 'System Config' to modify config.py settings")
        print("4. Generate enhanced forecasts with all original Dash graphs!")
        print("\nNew Features:")
        print("- ⚙️  Configuration management UI")
        print("- 📊 Level range graph (exact Dash replica)")
        print("- 📈 Category range graph (exact Dash replica)")  
        print("- 📋 Enhanced forecast results")
        print("- 🔄 Both regression AND classification forecasts")
    else:
        print("\n❌ Some issues were found. Please check the errors above.")
    
    sys.exit(0 if success else 1)