#!/usr/bin/env python3
"""
DATect Complete System Launcher - Python Version
Single command to start backend, frontend, and open browser
"""

import subprocess
import time
import os
import sys
import signal
import webbrowser
import requests
from pathlib import Path
import pandas as pd
import importlib.util

class DATectLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent
        
    def print_colored(self, message, color='blue'):
        colors = {
            'red': '\033[0;31m',
            'green': '\033[0;32m',
            'blue': '\033[0;34m',
            'yellow': '\033[1;33m',
            'reset': '\033[0m'
        }
        print(f"{colors.get(color, '')}{message}{colors['reset']}")
    
    def check_port(self, port):
        """Check if port is available and kill process if needed"""
        try:
            # Try to find and kill process on port
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                self.print_colored(f"⚠️  Port {port} is busy. Stopping existing processes...", 'yellow')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        pass
                time.sleep(2)
        except:
            pass
    
    def wait_for_service(self, url, name, max_wait=30):
        """Wait for a service to become available"""
        self.print_colored(f"⏳ Waiting for {name} to be ready...", 'yellow')
        for i in range(max_wait):
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    self.print_colored(f"✅ {name} is ready!", 'green')
                    return True
            except:
                pass
            time.sleep(1)
            if i % 5 == 4:
                print(f"   Still waiting... ({i+1}/{max_wait})")
        
        self.print_colored(f"❌ {name} failed to start within {max_wait} seconds", 'red')
        return False
    
    def check_prerequisites(self):
        """Check all prerequisites including scientific data validation"""
        self.print_colored("📋 Checking prerequisites and scientific validity...", 'blue')
        
        # Check data file existence
        data_file = self.project_root / "data/processed/final_output.parquet"
        if not data_file.exists():
            self.print_colored("❌ Data file not found at data/processed/final_output.parquet", 'red')
            print("Please run 'python dataset-creation.py' first to generate the data")
            return False
        self.print_colored("✅ Data file found", 'green')
        
        # SCIENTIFIC VALIDATION: Check data integrity
        if not self._validate_scientific_data(data_file):
            return False
        
        # SCIENTIFIC VALIDATION: Check temporal integrity
        if not self._validate_temporal_integrity():
            return False
            
        # SCIENTIFIC VALIDATION: Check model configuration
        if not self._validate_model_config():
            return False
        
        # Check Python
        self.print_colored("✅ Python 3 available", 'green')
        
        # Check Node.js
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True)
            self.print_colored("✅ Node.js available", 'green')
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_colored("❌ Node.js is not installed", 'red')
            print("Please install Node.js from https://nodejs.org or run: brew install node")
            return False
            
        return True
    
    def _validate_scientific_data(self, data_file):
        """Validate scientific integrity of the dataset"""
        self.print_colored("🔬 Validating scientific data integrity...", 'blue')
        
        try:
            # Load and validate data structure
            data = pd.read_parquet(data_file)
            
            # Check basic required columns
            basic_columns = ['date', 'site', 'da']
            missing_cols = [col for col in basic_columns if col not in data.columns]
            if missing_cols:
                self.print_colored(f"❌ Missing required columns: {missing_cols}", 'red')
                print("Data must contain 'date', 'site', and 'da' columns")
                return False
            
            # Note: da-category should NOT be pre-created in final_output.parquet
            # It should only be created during individual forecast operations to maintain temporal integrity
            # The original scientific approach creates categories per-forecast to prevent data leakage
            
            # Check data quality
            if data.empty:
                self.print_colored("❌ Dataset is empty", 'red')
                return False
            
            if data['da'].isna().all():
                self.print_colored("❌ All DA values are missing", 'red')
                return False
            
            # Check temporal coverage
            try:
                data['date'] = pd.to_datetime(data['date'])
                date_range = data['date'].max() - data['date'].min()
                if date_range.days < 365:
                    self.print_colored("⚠️  Warning: Less than 1 year of data - may affect model performance", 'yellow')
            except Exception:
                self.print_colored("⚠️  Warning: Could not validate date range", 'yellow')
            
            # Check site coverage
            sites = data['site'].nunique()
            if sites < 3:
                self.print_colored("⚠️  Warning: Less than 3 sites - limited spatial coverage", 'yellow')
            
            self.print_colored(f"✅ Data validation passed: {len(data):,} records, {sites} sites", 'green')
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Data validation failed: {e}", 'red')
            return False
    
    def _validate_temporal_integrity(self):
        """Validate temporal integrity settings to prevent data leakage"""
        self.print_colored("⏱️  Validating temporal integrity safeguards...", 'blue')
        
        try:
            # Load config file dynamically
            config_path = self.project_root / "config.py"
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # Check critical temporal safeguards
            if not hasattr(config, 'TEMPORAL_BUFFER_DAYS'):
                self.print_colored("❌ TEMPORAL_BUFFER_DAYS not configured", 'red')
                return False
            
            if config.TEMPORAL_BUFFER_DAYS < 1:
                self.print_colored("❌ TEMPORAL_BUFFER_DAYS must be ≥ 1 to prevent data leakage", 'red')
                return False
                
            if not hasattr(config, 'SATELLITE_BUFFER_DAYS'):
                self.print_colored("❌ SATELLITE_BUFFER_DAYS not configured", 'red')
                return False
                
            if config.SATELLITE_BUFFER_DAYS < 7:
                self.print_colored("❌ SATELLITE_BUFFER_DAYS should be ≥ 7 days for realistic data availability", 'red')
                return False
            
            self.print_colored("✅ Temporal integrity safeguards validated", 'green')
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Temporal integrity validation failed: {e}", 'red')
            return False
    
    def _validate_model_config(self):
        """Validate model configuration for scientific validity"""
        self.print_colored("⚙️  Validating model configuration...", 'blue')
        
        try:
            # Load config file dynamically
            config_path = self.project_root / "config.py"
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # Validate model/task combinations
            valid_models = ['xgboost', 'ridge']
            if not hasattr(config, 'FORECAST_MODEL') or config.FORECAST_MODEL not in valid_models:
                self.print_colored(f"❌ Invalid FORECAST_MODEL. Must be one of: {valid_models}", 'red')
                return False
            
            valid_tasks = ['regression', 'classification']
            if not hasattr(config, 'FORECAST_TASK') or config.FORECAST_TASK not in valid_tasks:
                self.print_colored(f"❌ Invalid FORECAST_TASK. Must be one of: {valid_tasks}", 'red')
                return False
            
            # Check minimum training samples
            if not hasattr(config, 'MIN_TRAINING_SAMPLES') or config.MIN_TRAINING_SAMPLES < 3:
                self.print_colored("❌ MIN_TRAINING_SAMPLES must be ≥ 3 for reliable model training", 'red')
                return False
            
            # Check random seed for reproducibility
            if not hasattr(config, 'RANDOM_SEED'):
                self.print_colored("⚠️  Warning: RANDOM_SEED not set - results may not be reproducible", 'yellow')
            
            # Check lag features
            if hasattr(config, 'LAG_FEATURES'):
                if not isinstance(config.LAG_FEATURES, list) or len(config.LAG_FEATURES) == 0:
                    self.print_colored("❌ LAG_FEATURES must be a non-empty list", 'red')
                    return False
                
                if any(lag < 1 for lag in config.LAG_FEATURES):
                    self.print_colored("❌ All LAG_FEATURES must be ≥ 1", 'red')
                    return False
            
            self.print_colored("✅ Model configuration validated", 'green')
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Model configuration validation failed: {e}", 'red')
            return False
    
    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        self.print_colored("📦 Installing dependencies...", 'blue')
        
        # Python dependencies
        print("Installing Python dependencies...")
        python_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'pandas', 'numpy', 
            'scikit-learn', 'plotly', 'requests', 'xgboost', 'dash'
        ]
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--quiet'
            ] + python_packages, check=True, capture_output=True)
        except:
            self.print_colored("⚠️  Some Python packages may already be installed", 'yellow')
        
        # Node.js dependencies
        print("Installing Node.js dependencies...")
        frontend_dir = self.project_root / "frontend"
        if not (frontend_dir / "node_modules").exists():
            try:
                subprocess.run(['npm', 'install'], 
                             cwd=frontend_dir, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                self.print_colored("❌ Failed to install Node.js dependencies", 'red')
                return False
        
        self.print_colored("✅ All dependencies installed", 'green')
        return True
    
    def start_backend(self):
        """Start the backend server"""
        self.print_colored("🖥️  Starting backend API server...", 'blue')
        
        # Start backend process
        self.backend_process = subprocess.Popen([
            sys.executable, 'backend/api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.project_root)
        
        print(f"Backend PID: {self.backend_process.pid}")
        
        # Wait for backend to be ready
        if not self.wait_for_service("http://localhost:8000/health", "Backend API"):
            return False
            
        return True
    
    def start_frontend(self):
        """Start the frontend development server"""
        self.print_colored("🎨 Starting frontend development server...", 'blue')
        
        frontend_dir = self.project_root / "frontend"
        
        # Start frontend process
        self.frontend_process = subprocess.Popen([
            'npm', 'run', 'dev'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=frontend_dir)
        
        print(f"Frontend PID: {self.frontend_process.pid}")
        
        # Wait for frontend to be ready
        if not self.wait_for_service("http://localhost:3000", "Frontend", max_wait=45):
            return False
            
        return True
    
    def open_browser(self):
        """Open the web browser to the application"""
        self.print_colored("🌐 Opening http://localhost:3000 in browser...", 'green')
        time.sleep(1)
        try:
            webbrowser.open("http://localhost:3000")
        except:
            self.print_colored("⚠️  Could not auto-open browser. Please visit: http://localhost:3000", 'yellow')
    
    def cleanup(self):
        """Clean up processes"""
        self.print_colored("🛑 Shutting down DATect system...", 'yellow')
        
        # Terminate processes
        if self.backend_process:
            self.backend_process.terminate()
            time.sleep(2)
            if self.backend_process.poll() is None:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            time.sleep(2)
            if self.frontend_process.poll() is None:
                self.frontend_process.kill()
        
        # Kill any remaining processes on our ports
        for port in [8000, 3000]:
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except:
                            pass
            except:
                pass
        
        self.print_colored("✅ DATect system stopped", 'green')
    
    def run(self):
        """Main launcher function"""
        try:
            self.print_colored("🚀 DATect Complete System Launcher", 'blue')
            self.print_colored("====================================", 'blue')
            print(f"Working directory: {self.project_root}")
            
            # Check ports first
            self.print_colored("🔍 Checking ports...", 'blue')
            self.check_port(8000)
            self.check_port(3000)
            
            # Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Install dependencies
            if not self.install_dependencies():
                return False
            
            # Start backend
            if not self.start_backend():
                self.print_colored("❌ Backend failed to start", 'red')
                return False
            
            # Start frontend
            if not self.start_frontend():
                self.print_colored("❌ Frontend failed to start", 'red')
                return False
            
            # Open browser
            self.open_browser()
            
            # Success message
            print()
            self.print_colored("🎉 DATect System is now running!", 'green')
            self.print_colored("====================================", 'green')
            self.print_colored("🔗 Frontend Web App: http://localhost:3000", 'green')
            self.print_colored("🔗 Backend API: http://localhost:8000", 'green')
            self.print_colored("📚 API Documentation: http://localhost:8000/docs", 'green')
            print()
            self.print_colored("📖 How to use:", 'blue')
            print("1. 🌐 Browser should open automatically to http://localhost:3000")
            print("2. ⚙️  Click 'System Config' to modify forecasting settings")
            print("3. 📊 Select date, site, and generate enhanced forecasts")
            print("4. 📈 View all original Dash graphs in the modern web interface")
            print()
            self.print_colored("Press Ctrl+C to stop the entire system", 'yellow')
            
            # Keep running until interrupted
            try:
                while True:
                    # Check if processes are still running
                    if self.backend_process.poll() is not None:
                        self.print_colored("❌ Backend process stopped unexpectedly", 'red')
                        break
                    if self.frontend_process.poll() is not None:
                        self.print_colored("❌ Frontend process stopped unexpectedly", 'red')
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                self.print_colored("\n📝 Received shutdown signal...", 'yellow')
            
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Error: {e}", 'red')
            return False
        finally:
            self.cleanup()

if __name__ == "__main__":
    launcher = DATectLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)