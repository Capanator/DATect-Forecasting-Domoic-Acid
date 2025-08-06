#!/usr/bin/env python3
"""
Streamlined Pipeline Test
========================

Tests the core DATect forecasting pipeline without scientific validation components.
Focuses on essential functionality: data processing, forecasting, and dashboards.
"""

import sys
import os
import pandas as pd
from datetime import datetime
import tempfile

sys.path.append('.')

def test_core_imports():
    """Test that all core components can be imported."""
    print("🔄 Testing Core Imports...")
    
    try:
        import config
        from forecasting.core.forecast_engine import ForecastEngine
        from forecasting.core.model_factory import ModelFactory
        from forecasting.core.data_processor import DataProcessor
        from forecasting.dashboard.realtime import RealtimeDashboard
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False


def test_model_factory():
    """Test model creation functionality."""
    print("🔄 Testing Model Factory...")
    
    try:
        from forecasting.core.model_factory import ModelFactory
        
        factory = ModelFactory()
        
        # Test XGBoost creation
        xgb_model = factory.get_model('regression', 'xgboost')
        print("✅ XGBoost model created")
        
        # Test Ridge creation
        ridge_model = factory.get_model('regression', 'ridge')
        print("✅ Ridge model created")
        
        # Test classification model
        class_model = factory.get_model('classification', 'xgboost')
        print("✅ Classification model created")
        
        return True
    except Exception as e:
        print(f"❌ Model factory failed: {e}")
        return False


def test_data_processor():
    """Test data processing functionality with mock data."""
    print("🔄 Testing Data Processor...")
    
    try:
        from forecasting.core.data_processor import DataProcessor
        import config
        
        processor = DataProcessor()
        
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='W'),
            'site': ['TestSite'] * 100,
            'da': [1.0 + i * 0.1 for i in range(100)]
        })
        
        # Test lag feature creation
        cutoff_date = pd.Timestamp('2020-06-01')
        lag_data = processor.create_lag_features_safe(
            test_data, 'site', 'da', config.LAG_FEATURES, cutoff_date
        )
        print("✅ Lag features created")
        
        # Test DA category creation
        categories = processor.create_da_categories_safe([1.0, 15.0, 25.0, 45.0])
        print("✅ DA categories created")
        
        # Test numeric transformer
        numeric_cols = ['da']
        transformer = processor.create_numeric_transformer(test_data, [])
        print("✅ Numeric transformer created")
        
        return True
    except Exception as e:
        print(f"❌ Data processor failed: {e}")
        return False


def test_forecast_engine():
    """Test forecast engine initialization."""
    print("🔄 Testing Forecast Engine...")
    
    try:
        from forecasting.core.forecast_engine import ForecastEngine
        
        # Create temporary test data file
        test_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50, freq='W'),
            'site': ['TestSite'] * 50,
            'da': [5.0 + i * 0.1 for i in range(50)],
            'chlorophyll': [2.0] * 50,
            'sst': [15.0] * 50,
            'sin_day_of_year': [0.5] * 50,
            'cos_day_of_year': [0.5] * 50
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            test_data.to_parquet(tmp.name)
            
            # Test engine initialization
            engine = ForecastEngine(tmp.name)
            print("✅ Forecast engine initialized")
            
            # Cleanup
            os.unlink(tmp.name)
        
        return True
    except Exception as e:
        print(f"❌ Forecast engine failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("🔄 Testing Configuration...")
    
    try:
        import config
        
        # Test key configuration values
        assert hasattr(config, 'SITES'), "SITES not configured"
        assert hasattr(config, 'FORECAST_MODE'), "FORECAST_MODE not configured"
        assert hasattr(config, 'FORECAST_MODEL'), "FORECAST_MODEL not configured"
        
        print(f"✅ Configuration valid - {len(config.SITES)} sites configured")
        print(f"   Mode: {config.FORECAST_MODE}, Model: {config.FORECAST_MODEL}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False


def test_dashboard_components():
    """Test dashboard component imports."""
    print("🔄 Testing Dashboard Components...")
    
    try:
        from forecasting.dashboard.realtime import RealtimeDashboard
        from forecasting.dashboard.retrospective import RetrospectiveDashboard
        
        print("✅ Dashboard imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard components failed: {e}")
        return False


def main():
    """Run streamlined pipeline tests."""
    print("🚀 DATect Streamlined Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Model Factory", test_model_factory),
        ("Data Processor", test_data_processor),
        ("Forecast Engine", test_forecast_engine),
        ("Configuration", test_configuration),
        ("Dashboard Components", test_dashboard_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append(success)
            print()
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print("📋 STREAMLINED PIPELINE TEST RESULTS")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Streamlined pipeline ready!")
        print("✅ Core functionality: Data processing, forecasting, dashboards")
        print("✅ Scientific validation: Available in analysis/ directory")
        print("✅ Clean architecture: Separated core from optional components")
        return 0
    else:
        print(f"❌ {total-passed} tests failed - Pipeline needs attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())