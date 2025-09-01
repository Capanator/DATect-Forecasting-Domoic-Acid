#!/usr/bin/env python3
"""
Test Retrospective LightGBM Analysis - Debug hanging issue
"""

import time
import config
from datetime import datetime, timedelta
from forecasting.forecast_engine import ForecastEngine

def test_retrospective_lightgbm():
    """Test retrospective analysis to identify hanging issue."""
    print("🔍 Testing Retrospective LightGBM Analysis")
    print("=" * 60)
    
    print(f"Configuration:")
    print(f"- Mode: {config.FORECAST_MODE}")
    print(f"- Task: {config.FORECAST_TASK}")  
    print(f"- Model: {config.FORECAST_MODEL}")
    print(f"- Random anchors: {config.N_RANDOM_ANCHORS}")
    print(f"- Bootstrap iterations: {config.BOOTSTRAP_ITERATIONS}")
    print()
    
    # Test 1: Initialize engine
    print("1. Initializing forecast engine...")
    try:
        start_time = time.time()
        engine = ForecastEngine(validate_on_init=False)
        init_time = time.time() - start_time
        print(f"   ✅ Engine initialized in {init_time:.2f} seconds")
    except Exception as e:
        print(f"   ❌ Engine initialization failed: {e}")
        return False
    
    # Test 2: Single retrospective forecast
    print("\n2. Testing single retrospective forecast...")
    try:
        start_time = time.time()
        
        # Use a date from 2023 where we have data
        test_date = datetime(2023, 6, 15)  # Mid-2023
        
        result = engine.generate_single_forecast(
            'data/processed/final_output.parquet',
            test_date,
            'Newport',  # Known good site
            'regression',
            'balanced_lightgbm'
        )
        
        single_time = time.time() - start_time
        
        if result and 'predicted_da' in result:
            print(f"   ✅ Single forecast completed in {single_time:.2f} seconds")
            print(f"   ✅ Predicted DA: {result['predicted_da']:.3f} μg/g")
            print(f"   ✅ Training samples: {result.get('training_samples', 'N/A')}")
        else:
            print("   ❌ Single forecast failed - no result")
            print(f"   Result: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ Single forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Small retrospective batch (just 2 anchors)
    print(f"\n3. Testing small retrospective batch (2 anchors)...")
    try:
        # Temporarily reduce anchors for testing
        original_anchors = config.N_RANDOM_ANCHORS
        config.N_RANDOM_ANCHORS = 2
        
        start_time = time.time()
        
        # This should call the same retrospective endpoint that webapp uses
        results = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=2  # Just 2 anchors for testing
        )
        
        batch_time = time.time() - start_time
        config.N_RANDOM_ANCHORS = original_anchors  # Restore
        
        if results:
            print(f"   ✅ Batch analysis completed in {batch_time:.2f} seconds")
            print(f"   ✅ Results keys: {list(results.keys())}")
            
            if 'summary_stats' in results:
                stats = results['summary_stats']
                print(f"   ✅ R²: {stats.get('mean_r2', 'N/A'):.3f}")
                print(f"   ✅ RMSE: {stats.get('mean_rmse', 'N/A'):.3f}")
        else:
            print("   ❌ Batch analysis failed - no results")
            return False
            
    except Exception as e:
        print(f"   ❌ Batch analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check what webapp retrospective endpoint does
    print(f"\n4. Testing webapp retrospective endpoint logic...")
    try:
        # Simulate what the webapp calls
        from backend.api import app
        
        print("   ✅ Backend API imports work")
        
        # The webapp likely calls /api/retrospective or similar
        print("   💡 Check backend logs when webapp hangs to see exact endpoint")
        
    except Exception as e:
        print(f"   ❌ Backend API test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 RETROSPECTIVE TESTING COMPLETE")
    
    return True

if __name__ == "__main__":
    success = test_retrospective_lightgbm()
    if success:
        print("\n✅ Basic retrospective functionality works")
        print("💡 If webapp still hangs, check frontend JavaScript console")
        print("💡 Or check backend logs during webapp hang")
    else:
        print("\n❌ Retrospective functionality has issues")
    exit(0 if success else 1)