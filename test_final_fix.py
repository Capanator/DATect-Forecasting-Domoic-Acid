#!/usr/bin/env python3
"""
Final Test of LightGBM Fix
==========================

Test all the fixed functionality end-to-end.
"""

import time
import traceback
from datetime import datetime, timedelta

def test_model_creation():
    """Test basic model creation and fitting."""
    print("1. Testing LightGBM model creation and basic fitting...")
    
    try:
        from forecasting.model_factory import ModelFactory
        import pandas as pd
        import numpy as np
        
        factory = ModelFactory()
        model = factory.get_model('regression', 'balanced_lightgbm')
        
        # Create test data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.exponential(2, 100))  # Exponential distribution like DA data
        
        start_time = time.time()
        model.fit(X, y)
        predictions = model.predict(X[:10])
        elapsed = time.time() - start_time
        
        print(f"✅ Model training and prediction: {elapsed:.3f} seconds")
        print(f"✅ Sample predictions: {predictions[:3]}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_single_forecast():
    """Test single forecast functionality."""
    print("\n2. Testing single forecast with LightGBM...")
    
    try:
        import config
        from forecasting.forecast_engine import ForecastEngine
        
        engine = ForecastEngine(validate_on_init=False)
        
        # Test with a recent date
        forecast_date = datetime.now() - timedelta(days=30)
        site = "Newport"
        
        start_time = time.time()
        result = engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            forecast_date,
            site,
            'regression',
            'balanced_lightgbm'
        )
        elapsed = time.time() - start_time
        
        if result and 'predicted_da' in result:
            print(f"✅ Single forecast: {elapsed:.3f} seconds")
            print(f"✅ Predicted DA: {result['predicted_da']:.3f} μg/g")
            print(f"✅ Training samples: {result.get('training_samples', 'N/A')}")
            return True
        else:
            print("❌ Single forecast failed - no result")
            return False
            
    except Exception as e:
        print(f"❌ Single forecast failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_enhanced_forecast():
    """Test enhanced forecast with uncertainty."""
    print("\n3. Testing enhanced forecast with uncertainty...")
    
    try:
        import config
        from forecasting.forecast_engine import ForecastEngine
        
        engine = ForecastEngine(validate_on_init=False)
        
        forecast_date = datetime.now() - timedelta(days=30)
        site = "Newport"
        
        start_time = time.time()
        result = engine.generate_enhanced_forecast(
            config.FINAL_OUTPUT_PATH,
            forecast_date,
            site,
            'regression',
            'balanced_lightgbm',
            include_uncertainty=True,
            include_comparison=False
        )
        elapsed = time.time() - start_time
        
        if result and 'predicted_da' in result:
            print(f"✅ Enhanced forecast: {elapsed:.3f} seconds")
            print(f"✅ Predicted DA: {result['predicted_da']:.3f} μg/g")
            if 'uncertainty' in result:
                unc = result['uncertainty']
                print(f"✅ Uncertainty: {unc['lower_bound']:.3f} - {unc['upper_bound']:.3f} μg/g")
            return True
        else:
            print("❌ Enhanced forecast failed - no result")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced forecast failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_retrospective_small():
    """Test small retrospective analysis."""
    print("\n4. Testing small retrospective analysis...")
    
    try:
        import config
        from forecasting.forecast_engine import ForecastEngine
        
        # Use very small number of anchors
        config.N_RANDOM_ANCHORS = 2
        
        engine = ForecastEngine(validate_on_init=False)
        
        start_time = time.time()
        results_df = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=2
        )
        elapsed = time.time() - start_time
        
        if results_df is not None and not results_df.empty:
            print(f"✅ Retrospective analysis: {elapsed:.3f} seconds")
            print(f"✅ Results: {len(results_df)} forecasts generated")
            return True
        else:
            print("❌ Retrospective analysis failed - no results")
            return False
            
    except Exception as e:
        print(f"❌ Retrospective analysis failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_api_integration():
    """Test API integration (without starting server)."""
    print("\n5. Testing API model mapping...")
    
    try:
        from backend.api import get_realtime_model_name, get_actual_model_name
        
        # Test model name mapping
        regression_model = get_realtime_model_name("regression", "balanced_lightgbm")
        classification_model = get_realtime_model_name("classification", "balanced_lightgbm")
        
        print(f"✅ Regression model mapping: balanced_lightgbm -> {regression_model}")
        print(f"✅ Classification model mapping: balanced_lightgbm -> {classification_model}")
        
        # Should allow LightGBM for regression, fallback for classification
        if regression_model == "balanced_lightgbm":
            print("✅ LightGBM now allowed for realtime regression")
        else:
            print(f"❌ Unexpected regression model: {regression_model}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ API integration failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE LIGHTGBM FIX TEST")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_single_forecast,
        test_enhanced_forecast,
        test_retrospective_small,
        test_api_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            break  # Stop on first failure
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED!")
        print("✅ LightGBM hanging issue is RESOLVED")
        print("✅ LightGBM now works for both realtime and retrospective forecasting")
        print("✅ Ready for production use")
    else:
        print("❌ Some tests failed - issue may not be fully resolved")
    
    exit(0 if passed == len(tests) else 1)