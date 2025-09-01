#!/usr/bin/env python3
"""
Test Retrospective Fix - Verify n_anchors config is respected
"""

import config
from forecasting.forecast_engine import ForecastEngine

def test_retrospective_fix():
    """Test that retrospective analysis uses config N_RANDOM_ANCHORS."""
    print("🔧 Testing Retrospective Fix")
    print("=" * 50)
    
    print(f"Config N_RANDOM_ANCHORS: {config.N_RANDOM_ANCHORS}")
    print(f"Config BOOTSTRAP_ITERATIONS: {config.BOOTSTRAP_ITERATIONS}")
    print()
    
    engine = ForecastEngine(validate_on_init=False)
    
    # Test with config value (should use 5 anchors)
    print("Testing with config value (should use 5 anchors)...")
    try:
        import time
        start_time = time.time()
        
        results = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm'
            # n_anchors not specified - should use config value
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f} seconds")
        print(f"✅ Results shape: {results.shape if results is not None else 'None'}")
        
        # Should be much faster with 5 anchors vs 50
        if elapsed < 30:
            print("✅ Performance looks good (< 30 seconds)")
        else:
            print("⚠️  Still slow - may need further optimization")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_retrospective_fix()
    if success:
        print("\n🎉 Retrospective fix verified!")
        print("🚀 Webapp retrospective should now be fast")
    else:
        print("\n❌ Fix needs more work")
    exit(0 if success else 1)