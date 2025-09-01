#!/usr/bin/env python3
"""
Debug LightGBM Specific Issue
============================
"""

import numpy as np
import pandas as pd
import time
import traceback

def test_lgb_parameters():
    """Test the specific LightGBM parameters that might be causing issues."""
    print("🔍 TESTING LIGHTGBM PARAMETERS")
    print("=" * 50)
    
    try:
        import lightgbm as lgb
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Test 1: LightGBM with problematic class_weight parameter
        print("1. Testing with class_weight parameter...")
        start_time = time.time()
        
        problematic_params = {
            'n_estimators': 100,  # Reduced for testing
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': {0: 1, 1: 60.0},  # This might be the issue
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'objective': 'regression',
        }
        
        model1 = lgb.LGBMRegressor(**problematic_params)
        model1.fit(X, y)
        pred1 = model1.predict(X[:5])
        
        elapsed1 = time.time() - start_time
        print(f"✅ With class_weight: {elapsed1:.3f} seconds")
        print(f"   Sample predictions: {pred1[:3]}")
        
        # Test 2: LightGBM without class_weight parameter
        print("\n2. Testing without class_weight parameter...")
        start_time = time.time()
        
        clean_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'objective': 'regression',
        }
        
        model2 = lgb.LGBMRegressor(**clean_params)
        model2.fit(X, y)
        pred2 = model2.predict(X[:5])
        
        elapsed2 = time.time() - start_time
        print(f"✅ Without class_weight: {elapsed2:.3f} seconds")
        print(f"   Sample predictions: {pred2[:3]}")
        
        # Test 3: Sample weighting instead
        print("\n3. Testing with proper sample weighting...")
        start_time = time.time()
        
        # Create sample weights (simulate spike weighting)
        spike_threshold = np.percentile(y, 80)  # Top 20% as "spikes"
        sample_weights = np.where(y > spike_threshold, 60.0, 1.0)
        
        model3 = lgb.LGBMRegressor(**clean_params)
        model3.fit(X, y, sample_weight=sample_weights)
        pred3 = model3.predict(X[:5])
        
        elapsed3 = time.time() - start_time
        print(f"✅ With sample weighting: {elapsed3:.3f} seconds")
        print(f"   Sample predictions: {pred3[:3]}")
        
        # Test 4: Test with n_jobs=1 to rule out parallel processing issues
        print("\n4. Testing with n_jobs=1...")
        start_time = time.time()
        
        serial_params = clean_params.copy()
        serial_params['n_jobs'] = 1
        
        model4 = lgb.LGBMRegressor(**serial_params)
        model4.fit(X, y, sample_weight=sample_weights)
        pred4 = model4.predict(X[:5])
        
        elapsed4 = time.time() - start_time
        print(f"✅ With n_jobs=1: {elapsed4:.3f} seconds")
        print(f"   Sample predictions: {pred4[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_actual_model():
    """Test the actual BalancedSpikeLGBRegressor model."""
    print("\n🔍 TESTING ACTUAL MODEL")
    print("=" * 50)
    
    try:
        from forecasting.model_factory import ModelFactory
        
        factory = ModelFactory()
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.exponential(2, 100))  # More realistic DA-like distribution
        
        print("1. Testing BalancedSpikeLGBRegressor...")
        start_time = time.time()
        
        model = factory.get_model('regression', 'balanced_lightgbm')
        model.fit(X, y)
        predictions = model.predict(X[:5])
        
        elapsed = time.time() - start_time
        print(f"✅ Model fitting and prediction: {elapsed:.3f} seconds")
        print(f"   Predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Debugging LightGBM specific issues...")
    
    params_ok = test_lgb_parameters()
    model_ok = test_actual_model()
    
    print("\n" + "=" * 50)
    if params_ok and model_ok:
        print("✅ ALL PARAMETER TESTS PASSED")
        print("Issue might be in the retrospective evaluation logic")
    else:
        print("❌ PARAMETER ISSUES FOUND")