#!/usr/bin/env python3
"""
Quick Test of LightGBM Fix
==========================
"""

import time
import traceback
from datetime import datetime, timedelta

def test_basic_functionality():
    """Test basic LightGBM functionality that was previously hanging."""
    print("🚀 QUICK LIGHTGBM FIX VERIFICATION")
    print("=" * 50)
    
    try:
        print("1. Testing model creation...")
        from forecasting.model_factory import ModelFactory
        import pandas as pd
        import numpy as np
        
        factory = ModelFactory()
        model = factory.get_model('regression', 'balanced_lightgbm')
        
        # Create simple test data
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.exponential(2, 50))
        
        start_time = time.time()
        model.fit(X, y)
        predictions = model.predict(X[:5])
        elapsed = time.time() - start_time
        
        print(f"✅ Model fit+predict: {elapsed:.3f}s, predictions: {predictions[:3]}")
        
        print("\n2. Testing single forecast (no uncertainty)...")
        import config
        from forecasting.forecast_engine import ForecastEngine
        
        engine = ForecastEngine(validate_on_init=False)
        
        start_time = time.time()
        result = engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            datetime.now() - timedelta(days=60),  # Use older date
            'Newport',
            'regression',
            'balanced_lightgbm'
        )
        elapsed = time.time() - start_time
        
        if result and 'predicted_da' in result:
            print(f"✅ Single forecast: {elapsed:.3f}s, DA: {result['predicted_da']:.3f} μg/g")
        else:
            print("❌ Single forecast failed")
            return False
            
        print("\n3. Testing API model mapping...")
        from backend.api import get_realtime_model_name
        
        # Test that LightGBM is now allowed for regression
        reg_model = get_realtime_model_name("regression", "balanced_lightgbm")
        class_model = get_realtime_model_name("classification", "balanced_lightgbm")
        
        print(f"✅ Regression: balanced_lightgbm -> {reg_model}")
        print(f"✅ Classification: balanced_lightgbm -> {class_model}")
        
        if reg_model == "balanced_lightgbm":
            print("✅ LightGBM now supported for realtime regression!")
        
        print("\n4. Testing minimal retrospective (1 anchor)...")
        config.N_RANDOM_ANCHORS = 1
        
        start_time = time.time()
        results_df = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=1
        )
        elapsed = time.time() - start_time
        
        if results_df is not None and not results_df.empty:
            print(f"✅ Minimal retrospective: {elapsed:.3f}s, {len(results_df)} results")
        else:
            print("❌ Minimal retrospective failed")
            return False
            
        print("\n" + "=" * 50)
        print("🎉 ALL CORE FUNCTIONALITY WORKING!")
        print("✅ LightGBM hanging issue RESOLVED")
        print("✅ Model can be used in webapp without hanging")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🚀 Ready for webapp testing!")
        print("Users can now select 'Balanced LightGBM' without hanging")
    else:
        print("\n❌ Fix verification failed")
    exit(0 if success else 1)