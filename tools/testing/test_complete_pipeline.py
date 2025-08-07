#!/usr/bin/env python3
"""
Complete Pipeline Test Script
============================

Comprehensive test of the entire DATect forecasting pipeline.
This validates the complete end-to-end system and ensures
all system components work together seamlessly.

Usage:
    python test_complete_pipeline.py
    python test_complete_pipeline.py --quick  # Skip data creation
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class PipelineTester:
    """Complete pipeline testing framework."""
    
    def __init__(self, quick_mode=False):
        self.quick_mode = quick_mode
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, cmd, description, timeout=300):
        """Run a command and track results."""
        self.log(f"🔄 {description}...")
        start_time = time.time()
        
        try:
            # Run command
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"{description} completed successfully in {execution_time:.1f}s")
                self.log(f"✅ {description} completed in {execution_time:.1f}s")
                self.results[description] = {
                    'success': True,
                    'execution_time': execution_time,
                    'output_lines': len(result.stdout.split('\n')),
                    'command': cmd
                }
                return True
            else:
                logger.error(f"{description} failed in {execution_time:.1f}s: {result.stderr}")
                self.log(f"❌ {description} failed: {result.stderr}", "ERROR")
                self.results[description] = {
                    'success': False,
                    'execution_time': execution_time,
                    'error': result.stderr,
                    'command': cmd
                }
                return False
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.log(f"⏰ {description} timed out after {timeout}s", "ERROR")
            self.results[description] = {
                'success': False,
                'execution_time': execution_time,
                'error': f"Timeout after {timeout}s",
                'command': cmd
            }
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            self.log(f"💥 {description} crashed: {str(e)}", "ERROR")
            self.results[description] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'command': cmd
            }
            return False
    
    def check_file_exists(self, file_path, description):
        """Check if a required file exists."""
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
            self.log(f"✅ {description} exists ({file_size:.1f} MB)")
            return True
        else:
            self.log(f"❌ {description} missing: {file_path}", "ERROR")
            return False
    
    def test_data_creation(self):
        """Test data creation pipeline (Container 1)."""
        self.log("=" * 60)
        self.log("🗃️  TESTING DATA CREATION PIPELINE")
        self.log("=" * 60)
        
        if self.quick_mode:
            self.log("⚡ Quick mode: Skipping data creation (using existing data)")
            return self.check_file_exists("./data/processed/final_output.parquet", "Final output data")
        
        # Run data creation
        success = self.run_command(
            "python3 dataset-creation.py", 
            "Data creation pipeline",
            timeout=3600  # 1 hour timeout for data creation
        )
        
        if success:
            # Verify output files
            success &= self.check_file_exists("./data/processed/final_output.parquet", "Final output data")
            
        return success
    
    def test_scientific_validation(self):
        """Test scientific validation suite (Container 2)."""
        self.log("=" * 60)
        self.log("🔬 TESTING SCIENTIFIC VALIDATION")
        self.log("=" * 60)
        
        # Run comprehensive scientific validation
        success = self.run_command(
            "python3 analysis/scientific-validation/run_scientific_validation.py --tests all --output-dir ./tools/testing/results/validation/",
            "Scientific validation suite"
        )
        
        if success:
            # Check validation outputs
            success &= self.check_file_exists("./tools/testing/results/validation/comprehensive_validation_report.json", "Validation report")
            success &= self.check_file_exists("./tools/testing/results/validation/validation_summary.txt", "Validation summary")
        
        return success
    
    def test_unit_tests(self):
        """Test unit test suite."""
        self.log("=" * 60)
        self.log("🧪 TESTING UNIT TESTS")
        self.log("=" * 60)
        
        # Run temporal integrity tests
        success = self.run_command(
            "PYTHONPATH=. python3 analysis/scientific-validation/test_temporal_integrity.py",
            "Temporal integrity unit tests"
        )
        
        return success
    
    def test_performance_analysis(self):
        """Test performance profiling (Container 5)."""
        self.log("=" * 60)
        self.log("📊 TESTING PERFORMANCE ANALYSIS")
        self.log("=" * 60)
        
        # Run performance profiler
        success = self.run_command(
            "PYTHONPATH=. python3 analysis/scientific-validation/performance_profiler.py --data-path data/processed/final_output.parquet --output-dir ./tools/testing/results/performance/",
            "Performance profiling"
        )
        
        if success:
            # Check performance outputs
            perf_files = list(Path("./tools/testing/results/performance/").glob("performance_*.json"))
            if perf_files:
                self.log(f"✅ Performance report generated: {perf_files[0].name}")
            else:
                self.log("❌ No performance reports found", "ERROR")
                success = False
        
        return success
    
    def test_forecasting_engine(self):
        """Test core forecasting functionality."""
        self.log("=" * 60)
        self.log("🤖 TESTING FORECASTING ENGINE")
        self.log("=" * 60)
        
        # Test forecasting engine directly
        test_script = """
import pandas as pd
import sys
sys.path.append('.')
from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.model_factory import ModelFactory

try:
    # Test engine initialization
    engine = ForecastEngine('./data/processed/final_output.parquet')
    print("✅ ForecastEngine initialized")
    
    # Test model factory
    factory = ModelFactory()
    model = factory.get_model('regression', 'rf')
    print("✅ Random Forest model created")
    
    # Test data loading
    data = pd.read_parquet('./data/processed/final_output.parquet')
    print(f"✅ Data loaded: {len(data)} samples")
    
    print("SUCCESS: Core components working")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
        
        with open("test_forecasting.py", "w") as f:
            f.write(test_script)
        
        success = self.run_command(
            "python3 test_forecasting.py",
            "Core forecasting components"
        )
        
        # Clean up test file
        Path("test_forecasting.py").unlink(missing_ok=True)
        
        return success
    
    def test_dashboard_imports(self):
        """Test dashboard component imports."""
        self.log("=" * 60)
        self.log("📈 TESTING DASHBOARD COMPONENTS")
        self.log("=" * 60)
        
        # Test dashboard imports
        test_script = """
import sys
sys.path.append('.')

try:
    from forecasting.dashboard.realtime import RealtimeDashboard
    print("✅ RealtimeDashboard imported")
    
    from forecasting.dashboard.retrospective import RetrospectiveDashboard  
    print("✅ RetrospectiveDashboard imported")
    
    print("SUCCESS: Dashboard components importable")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""
        
        with open("test_dashboards.py", "w") as f:
            f.write(test_script)
        
        success = self.run_command(
            "python3 test_dashboards.py",
            "Dashboard component imports"
        )
        
        # Clean up test file
        Path("test_dashboards.py").unlink(missing_ok=True)
        
        return success
    
    def generate_final_report(self):
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        self.log("=" * 60)
        self.log("📋 FINAL PIPELINE TEST REPORT")
        self.log("=" * 60)
        
        # Count successes
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r['success'])
        
        self.log(f"Total tests: {total_tests}")
        self.log(f"Successful: {successful_tests}")
        self.log(f"Failed: {total_tests - successful_tests}")
        self.log(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        self.log(f"Total execution time: {total_time:.1f}s")
        
        # Detailed results
        self.log("\nDETAILED RESULTS:")
        self.log("-" * 40)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_str = f"{result['execution_time']:.1f}s"
            self.log(f"{status} {test_name} ({time_str})")
            
            if not result['success'] and 'error' in result:
                error_preview = result['error'][:100] + "..." if len(result['error']) > 100 else result['error']
                self.log(f"     Error: {error_preview}")
        
        # Overall assessment
        if successful_tests == total_tests:
            self.log("\n🎉 ALL TESTS PASSED - Pipeline is production ready!")
            return True
        elif successful_tests / total_tests >= 0.8:
            self.log(f"\n⚠️  Most tests passed ({successful_tests}/{total_tests}) - Pipeline mostly functional")
            return False
        else:
            self.log(f"\n💥 Multiple failures ({successful_tests}/{total_tests}) - Pipeline needs fixes")
            return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="DATect Complete Pipeline Tester")
    parser.add_argument('--quick', action='store_true',
                      help='Quick mode - skip data creation')
    
    args = parser.parse_args()
    
    print("🚀 DATect Complete Pipeline Test")
    print("=" * 50)
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Initialize tester
    tester = PipelineTester(quick_mode=args.quick)
    
    # Run all tests
    tests = [
        ('Data Creation', tester.test_data_creation),
        ('Scientific Validation', tester.test_scientific_validation),
        ('Unit Tests', tester.test_unit_tests),
        ('Forecasting Engine', tester.test_forecasting_engine),
        ('Dashboard Components', tester.test_dashboard_imports),
        ('Performance Analysis', tester.test_performance_analysis),
    ]
    
    overall_success = True
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            overall_success &= success
        except Exception as e:
            tester.log(f"💥 {test_name} crashed: {str(e)}", "ERROR")
            overall_success = False
    
    # Generate final report
    final_success = tester.generate_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if final_success else 1)


if __name__ == "__main__":
    main()