#!/usr/bin/env python3
"""
Debug Retrospective Analysis Step by Step
=========================================
"""

import sys
import time
import signal
import traceback
from datetime import datetime
import pandas as pd

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def debug_retrospective_steps():
    """Debug each step of retrospective analysis."""
    print("🔍 DEBUGGING RETROSPECTIVE ANALYSIS STEPS")
    print("=" * 60)
    
    # Set a reasonable timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    try:
        print("1. Setup and data loading...")
        import config
        config.FORECAST_MODEL = 'balanced_lightgbm'
        config.FORECAST_TASK = 'regression'
        config.N_RANDOM_ANCHORS = 3  # Very small for testing
        
        from forecasting.forecast_engine import ForecastEngine
        from forecasting.data_processor import DataProcessor
        
        engine = ForecastEngine(validate_on_init=False)
        print("✅ Engine initialized")
        
        print("2. Data loading...")
        data = engine.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        print(f"✅ Data loaded: {len(data)} records")
        
        print("3. Test single anchor processing...")
        # Get a single site and date for testing
        site = data['site'].iloc[0]
        dates = data[data['site'] == site]['date'].sort_values()
        if len(dates) < 10:
            print(f"❌ Not enough dates for {site}")
            return False
            
        anchor_date = dates.iloc[5]  # Pick a date in the middle
        min_target_date = pd.Timestamp("2008-01-01")
        
        print(f"   Testing with site: {site}, anchor: {anchor_date}")
        
        # Test the single anchor processing function directly
        print("4. Testing single anchor processing...")
        start_time = time.time()
        
        result = engine._forecast_single_anchor_leak_free(
            (site, anchor_date), 
            data, 
            min_target_date, 
            'regression', 
            'balanced_lightgbm'
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Single anchor processed in {elapsed:.3f} seconds")
        
        if result is not None:
            print(f"✅ Result shape: {result.shape}")
        else:
            print("ℹ️ No result (could be normal depending on data)")
            
        # Test model creation directly
        print("5. Testing model creation and fitting...")
        from forecasting.model_factory import ModelFactory
        factory = ModelFactory()
        
        # Create some test data
        import numpy as np
        X_test = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        y_test = pd.Series(np.random.exponential(2, 50))
        
        start_time = time.time()
        model = factory.get_model('regression', 'balanced_lightgbm')
        model.fit(X_test, y_test)
        predictions = model.predict(X_test[:5])
        elapsed = time.time() - start_time
        
        print(f"✅ Model fit and predict in {elapsed:.3f} seconds")
        print(f"✅ Sample predictions: {predictions[:3]}")
        
        # Test parallel processing
        print("6. Testing parallel processing configuration...")
        from joblib import Parallel, delayed
        
        def simple_task(x):
            import time
            time.sleep(0.1)  # Small delay
            return x * 2
        
        start_time = time.time()
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(simple_task)(i) for i in range(5)
        )
        elapsed = time.time() - start_time
        print(f"✅ Parallel processing test: {elapsed:.3f} seconds, results: {results}")
        
        signal.alarm(0)  # Cancel timeout
        return True
        
    except TimeoutError:
        print("❌ TIMEOUT - Operation hung")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        signal.alarm(0)  # Ensure timeout is cancelled

def test_minimal_retrospective():
    """Test minimal retrospective with just one anchor."""
    print("\n7. Testing minimal retrospective...")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        import config
        config.N_RANDOM_ANCHORS = 1  # Just one anchor
        
        from forecasting.forecast_engine import ForecastEngine
        engine = ForecastEngine(validate_on_init=False)
        
        start_time = time.time()
        
        results_df = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=1
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Minimal retrospective completed in {elapsed:.3f} seconds")
        
        if results_df is not None:
            print(f"✅ Results: {len(results_df)} rows")
        else:
            print("❌ No results generated")
            return False
            
        signal.alarm(0)
        return True
        
    except TimeoutError:
        print("❌ MINIMAL RETROSPECTIVE TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    print("Starting detailed retrospective debugging...")
    
    steps_ok = debug_retrospective_steps()
    minimal_ok = test_minimal_retrospective() if steps_ok else False
    
    print("\n" + "=" * 60)
    if steps_ok and minimal_ok:
        print("🎉 ALL TESTS PASSED - ISSUE RESOLVED!")
    else:
        print("❌ ISSUE STILL EXISTS")
        
    exit(0 if (steps_ok and minimal_ok) else 1)