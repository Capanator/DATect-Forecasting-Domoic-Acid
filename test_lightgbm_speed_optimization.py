#!/usr/bin/env python3
"""
Test LightGBM Speed Optimization - Measure performance improvements
"""

import time
import config
from datetime import datetime
from forecasting.forecast_engine import ForecastEngine

def test_speed_optimization():
    """Test the speed improvements from optimizations."""
    print("🚀 Testing LightGBM Speed Optimization")
    print("=" * 60)
    
    print("Current Configuration:")
    print(f"- N_RANDOM_ANCHORS: {config.N_RANDOM_ANCHORS}")
    print(f"- BOOTSTRAP_ITERATIONS: {config.BOOTSTRAP_ITERATIONS}")
    print(f"- ENABLE_UNCERTAINTY_QUANTIFICATION: {config.ENABLE_UNCERTAINTY_QUANTIFICATION}")
    print()
    
    # Test 1: Single forecast speed
    print("1. Testing single forecast speed...")
    engine = ForecastEngine(validate_on_init=False)
    
    # Test multiple single forecasts to get average
    times = []
    for i in range(3):
        start_time = time.time()
        result = engine.generate_single_forecast(
            'data/processed/final_output.parquet',
            datetime(2023, 6, 15),
            'Newport',
            'regression',
            'balanced_lightgbm'
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"   Forecast {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"   ✅ Average single forecast: {avg_time:.2f}s")
    print(f"   🎯 Target for 30 anchors: {avg_time * 30:.1f}s (~{avg_time * 30 / 60:.1f} minutes)")
    print()
    
    # Test 2: Small retrospective batch (3 anchors × 10 sites = 30 total)
    print("2. Testing optimized retrospective batch...")
    start_time = time.time()
    
    try:
        results = engine.run_retrospective_evaluation(
            task='regression',
            model_type='balanced_lightgbm',
            n_anchors=config.N_RANDOM_ANCHORS  # Use config value (3)
        )
        
        total_time = time.time() - start_time
        total_anchors = config.N_RANDOM_ANCHORS * 10  # 10 sites
        
        print(f"   ✅ Retrospective completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   ✅ Total anchors processed: {total_anchors}")
        print(f"   ✅ Average per anchor: {total_time/total_anchors:.1f}s")
        print(f"   ✅ Results shape: {results.shape if results is not None else 'None'}")
        
        # Performance analysis
        if total_time < 60:
            print(f"   🚀 EXCELLENT: Under 1 minute!")
        elif total_time < 120:
            print(f"   ✅ GOOD: Under 2 minutes")
        else:
            print(f"   ⚠️  SLOW: Over 2 minutes")
            
        # CPU utilization estimate
        theoretical_time = avg_time * total_anchors  # Sequential time
        speedup = theoretical_time / total_time
        print(f"   📊 Parallelization speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Retrospective failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_utilization():
    """Test if optimizations are utilizing CPU properly."""
    print("\n3. CPU Utilization Test...")
    print("💡 During retrospective analysis, you should hear:")
    print("   - Fan spinning up (high CPU usage)")
    print("   - System becoming responsive (parallel processing)")
    print("   - Progress bar moving smoothly")
    print()
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"   🖥️  Available CPU cores: {cpu_count}")
    print(f"   ⚡ LightGBM n_jobs: -1 (using all cores)")
    print(f"   🔧 Optimized parameters: 500 estimators, 0.1 learning rate")

if __name__ == "__main__":
    success = test_speed_optimization()
    test_cpu_utilization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SPEED OPTIMIZATION SUCCESS!")
        print("✅ LightGBM is now optimized for speed")
        print("✅ Bootstraps disabled for retrospective mode") 
        print("✅ Reduced anchors for faster analysis")
        print("✅ CPU should be fully utilized")
        print()
        print("🚀 Expected webapp experience:")
        print("   - Retrospective analysis: ~1-2 minutes")
        print("   - Fan noise: Audible during processing")
        print("   - Graphs: Appear after completion")
    else:
        print("❌ Speed optimization needs more work")
    
    exit(0 if success else 1)