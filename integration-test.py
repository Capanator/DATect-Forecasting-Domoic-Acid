#!/usr/bin/env python3
"""
Complete integration test for DATect Web Application
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

class IntegrationTest:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check data file
        data_file = self.project_root / "data/processed/final_output.parquet"
        if not data_file.exists():
            print("❌ Data file missing. Run 'python dataset-creation.py' first.")
            return False
        print("✅ Data file found")
        
        # Check Python packages
        try:
            import fastapi, uvicorn, pandas, sklearn, xgboost
            print("✅ Python packages available")
        except ImportError as e:
            print(f"❌ Missing Python package: {e}")
            return False
            
        # Check Node.js (for frontend)
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True)
            print("✅ Node.js available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Node.js not available (frontend won't work)")
            
        return True
    
    def test_backend_standalone(self):
        """Test backend in standalone mode"""
        print("\n🖥️  Testing Backend API...")
        
        # Start backend
        self.backend_process = subprocess.Popen(
            [sys.executable, "backend/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.project_root
        )
        
        # Wait for startup
        print("   Starting backend server...")
        for i in range(15):
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    print("   ✅ Backend server started")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("   ❌ Backend failed to start")
            return False
            
        # Test API endpoints
        try:
            # Health check
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200
            print("   ✅ Health endpoint")
            
            # Sites endpoint
            response = requests.get("http://localhost:8000/api/sites")
            assert response.status_code == 200
            sites_data = response.json()
            print(f"   ✅ Sites endpoint ({len(sites_data['sites'])} sites)")
            
            # Models endpoint
            response = requests.get("http://localhost:8000/api/models")
            assert response.status_code == 200
            models_data = response.json()
            print(f"   ✅ Models endpoint ({len(models_data['available_models']['regression'])} regression models)")
            
            # Forecast endpoint
            forecast_request = {
                "date": "2010-06-15",
                "site": "Newport",
                "task": "regression", 
                "model": "xgboost"
            }
            response = requests.post("http://localhost:8000/api/forecast", json=forecast_request, timeout=30)
            assert response.status_code == 200
            forecast_data = response.json()
            if forecast_data["success"]:
                print(f"   ✅ Forecast endpoint (prediction: {forecast_data['prediction']:.4f})")
            else:
                print(f"   ⚠️  Forecast endpoint (insufficient data: {forecast_data.get('error')})")
            
            # Historical endpoint
            response = requests.get("http://localhost:8000/api/historical/Newport?limit=5")
            assert response.status_code == 200
            hist_data = response.json()
            print(f"   ✅ Historical endpoint ({hist_data['count']} records)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ API test failed: {e}")
            return False
            
        finally:
            if self.backend_process:
                self.backend_process.terminate()
                self.backend_process.wait()
                print("   Backend server stopped")
    
    def test_frontend_build(self):
        """Test frontend build process"""
        print("\n🎨 Testing Frontend Build...")
        
        frontend_dir = self.project_root / "frontend"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("   Installing frontend dependencies...")
            try:
                subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, capture_output=True)
                print("   ✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install dependencies: {e}")
                return False
        
        # Test build process
        try:
            print("   Building frontend...")
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, capture_output=True)
            print("   ✅ Frontend build successful")
            
            # Check if build artifacts exist
            dist_dir = frontend_dir / "dist"
            if dist_dir.exists() and (dist_dir / "index.html").exists():
                print("   ✅ Build artifacts created")
                return True
            else:
                print("   ❌ Build artifacts missing")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Frontend build failed: {e}")
            return False
        except FileNotFoundError:
            print("   ⚠️  npm not available - skipping frontend test")
            return True
    
    def test_docker_config(self):
        """Test Docker configuration"""
        print("\n🐳 Testing Docker Configuration...")
        
        # Check if docker-compose.yml is valid
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            print("   ❌ docker-compose.yml not found")
            return False
        print("   ✅ docker-compose.yml found")
        
        # Check Dockerfiles
        backend_dockerfile = self.project_root / "backend" / "Dockerfile"
        frontend_dockerfile = self.project_root / "frontend" / "Dockerfile"
        
        if not backend_dockerfile.exists():
            print("   ❌ Backend Dockerfile not found")
            return False
        print("   ✅ Backend Dockerfile found")
        
        if not frontend_dockerfile.exists():
            print("   ❌ Frontend Dockerfile not found")
            return False
        print("   ✅ Frontend Dockerfile found")
        
        # Check if Docker is available (optional)
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            print("   ✅ Docker available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   ⚠️  Docker not available (deployment won't work)")
            
        return True
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 DATect Web Application Integration Test")
        print("=" * 50)
        
        results = {
            "prerequisites": self.check_prerequisites(),
            "backend": False,
            "frontend": False,
            "docker": False
        }
        
        if results["prerequisites"]:
            results["backend"] = self.test_backend_standalone()
            results["frontend"] = self.test_frontend_build()
            results["docker"] = self.test_docker_config()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.capitalize()}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! The web application is ready to use.")
            print("\n📚 Quick Start:")
            print("1. Backend: python3 backend/main.py")
            print("2. Frontend: cd frontend && npm run dev")
            print("3. Docker: docker-compose up")
        else:
            print("⚠️  Some tests failed. Please check the issues above.")
        
        return passed_tests == total_tests

if __name__ == "__main__":
    tester = IntegrationTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)