#!/usr/bin/env python3
"""
Test script for DATect backend API endpoints
"""

import subprocess
import time
import requests
import json
import sys
from threading import Thread
import signal
import os

class BackendTester:
    def __init__(self):
        self.process = None
        self.base_url = 'http://localhost:8000'
        
    def start_server(self):
        """Start the backend server"""
        print("Starting backend server...")
        self.process = subprocess.Popen(
            [sys.executable, 'backend/main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f'{self.base_url}/health', timeout=1)
                if response.status_code == 200:
                    print("✅ Server started successfully")
                    return True
            except:
                time.sleep(1)
        
        print("❌ Server failed to start")
        return False
    
    def stop_server(self):
        """Stop the backend server"""
        if self.process:
            print("Stopping server...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                pass
    
    def test_endpoints(self):
        """Test all API endpoints"""
        tests = []
        
        # Test health endpoint
        try:
            response = requests.get(f'{self.base_url}/health')
            tests.append(('Health', response.status_code == 200, response.json()))
        except Exception as e:
            tests.append(('Health', False, str(e)))
        
        # Test root endpoint
        try:
            response = requests.get(f'{self.base_url}/')
            tests.append(('Root', response.status_code == 200, response.json()))
        except Exception as e:
            tests.append(('Root', False, str(e)))
        
        # Test sites endpoint
        try:
            response = requests.get(f'{self.base_url}/api/sites')
            if response.status_code == 200:
                data = response.json()
                tests.append(('Sites', True, f"{len(data['sites'])} sites available"))
            else:
                tests.append(('Sites', False, f"Status {response.status_code}"))
        except Exception as e:
            tests.append(('Sites', False, str(e)))
        
        # Test models endpoint
        try:
            response = requests.get(f'{self.base_url}/api/models')
            if response.status_code == 200:
                data = response.json()
                reg_models = len(data['available_models']['regression'])
                cls_models = len(data['available_models']['classification'])
                tests.append(('Models', True, f"{reg_models} regression, {cls_models} classification"))
            else:
                tests.append(('Models', False, f"Status {response.status_code}"))
        except Exception as e:
            tests.append(('Models', False, str(e)))
        
        # Test forecast endpoint
        try:
            forecast_data = {
                'date': '2010-06-15',  # Use date within data range
                'site': 'Newport',     # Use proper capitalization
                'task': 'regression',
                'model': 'xgboost'
            }
            response = requests.post(f'{self.base_url}/api/forecast', json=forecast_data, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    prediction = data.get('prediction', 'N/A')
                    tests.append(('Forecast', True, f"Prediction: {prediction}"))
                else:
                    tests.append(('Forecast', False, data.get('error', 'Unknown error')))
            else:
                tests.append(('Forecast', False, f"Status {response.status_code}"))
        except Exception as e:
            tests.append(('Forecast', False, str(e)))
        
        # Test historical data endpoint
        try:
            response = requests.get(f'{self.base_url}/api/historical/Newport?limit=5')
            if response.status_code == 200:
                data = response.json()
                tests.append(('Historical', True, f"{data['count']} records"))
            else:
                tests.append(('Historical', False, f"Status {response.status_code}"))
        except Exception as e:
            tests.append(('Historical', False, str(e)))
        
        return tests
    
    def run_tests(self):
        """Run all tests"""
        if not self.start_server():
            return
        
        try:
            print(f"\n{'='*50}")
            print("Testing DATect Backend API Endpoints")
            print(f"{'='*50}")
            
            tests = self.test_endpoints()
            
            passed = 0
            for name, success, result in tests:
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{status:<10} {name:<12} {result}")
                if success:
                    passed += 1
            
            print(f"\n{'='*50}")
            print(f"Results: {passed}/{len(tests)} tests passed")
            print(f"{'='*50}")
            
        finally:
            self.stop_server()

if __name__ == "__main__":
    tester = BackendTester()
    tester.run_tests()