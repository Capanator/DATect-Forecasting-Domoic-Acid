#!/usr/bin/env python3
"""
Debug LightGBM Hanging Issue
============================

Test to reproduce the exact hanging scenario reported in the webapp.
"""

import sys
import traceback
import signal
import time
from datetime import datetime

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out after 30 seconds")

def debug_retrospective_lightgbm():
    """Debug retrospective analysis with LightGBM."""
    print("🔍 DEBUGGING LIGHTGBM HANGING ISSUE")
    print("=" * 60)
    
    # Set a 30-second timeout for this test
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        print("1. Setting up configuration...")
        import config
        config.FORECAST_MODEL = 'balanced_lightgbm'
        config.FORECAST_TASK = 'regression'
        config.FORECAST_MODE = 'retrospective'
        config.N_RANDOM_ANCHORS = 2  # Very small for testing
        
        print("2. Initializing engine...")
        from forecasting.forecast_engine import ForecastEngine
        engine = ForecastEngine(validate_on_init=False)
        
        print("3. Running retrospective evaluation (small sample)...")
        start_time = time.time()
        
        results_df = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=2
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Retrospective completed in {elapsed:.2f} seconds")
        
        if results_df is not None and not results_df.empty:
            print(f"✅ Generated {len(results_df)} results")
            print("✅ No hanging detected in core engine")
        else:
            print("❌ No results generated")
            
        # Now test the API route directly
        print("\n4. Testing API route...")
        from backend.api import get_forecast_engine, get_model_factory
        
        api_engine = get_forecast_engine()
        api_factory = get_model_factory()
        
        # Test model creation
        lgb_model = api_factory.get_model('regression', 'balanced_lightgbm')
        print(f"✅ API model created: {type(lgb_model).__name__}")
        
        # Test single forecast
        print("\n5. Testing single forecast...")
        result = api_engine.generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            datetime.now(),
            'Newport',
            'regression',  
            'balanced_lightgbm'
        )
        
        if result:
            print(f"✅ Single forecast: {result.get('predicted_da', 'N/A')} μg/g")
        else:
            print("❌ Single forecast failed")
            
        # Test enhanced forecast  
        print("\n6. Testing enhanced forecast...")
        enhanced_result = api_engine.generate_enhanced_forecast(
            config.FINAL_OUTPUT_PATH,
            datetime.now(),
            'Newport',
            'regression',
            'balanced_lightgbm',
            include_uncertainty=True,
            include_comparison=False
        )
        
        if enhanced_result:
            print(f"✅ Enhanced forecast: {enhanced_result.get('predicted_da', 'N/A')} μg/g")
            if 'uncertainty' in enhanced_result:
                print(f"✅ Uncertainty included: {enhanced_result['uncertainty']['method']}")
        else:
            print("❌ Enhanced forecast failed")
            
        signal.alarm(0)  # Cancel the alarm
        return True
        
    except TimeoutError:
        print("❌ OPERATION TIMED OUT - HANGING DETECTED!")
        print("The process was stuck for over 30 seconds")
        return False
        
    except KeyboardInterrupt:
        print("❌ INTERRUPTED BY USER")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        signal.alarm(0)  # Make sure to cancel any pending alarm

def debug_parallel_processing():
    """Debug parallel processing issues."""
    print("\n7. Testing parallel processing...")
    
    try:
        from forecasting.model_factory import ModelFactory
        factory = ModelFactory()
        
        # Create LightGBM model and check n_jobs setting
        lgb_model = factory.get_model('regression', 'balanced_lightgbm')
        
        # Check the actual LightGBM parameters
        if hasattr(lgb_model, 'lgb_params'):
            n_jobs = lgb_model.lgb_params.get('n_jobs', 'Not set')
            print(f"✅ LightGBM n_jobs setting: {n_jobs}")
            
        if hasattr(lgb_model, 'model') and lgb_model.model is not None:
            print(f"✅ Model initialized: {type(lgb_model.model).__name__}")
        else:
            print("ℹ️ Model not yet fitted")
            
        return True
        
    except Exception as e:
        print(f"❌ Parallel processing error: {e}")
        return False

if __name__ == "__main__":
    print("Starting LightGBM hanging debug...")
    
    success = debug_retrospective_lightgbm()
    parallel_ok = debug_parallel_processing()
    
    print("\n" + "=" * 60)
    if success and parallel_ok:
        print("🎉 NO HANGING DETECTED - ISSUE MAY BE IN WEBAPP LAYER")
    else:
        print("❌ HANGING OR ERROR DETECTED")
    
    exit(0 if success and parallel_ok else 1)