#!/usr/bin/env python3
"""
Quick Test of New Forecasting Infrastructure
===========================================

Test the new spike detection models to ensure they work properly.
"""

import sys
import os
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spike_model_championship import SpikeModelChampionship
import config

warnings.filterwarnings('ignore')


def quick_test():
    """Run a quick test of a few models."""
    
    print("🧪 QUICK TEST OF NEW FORECASTING INFRASTRUCTURE")
    print("=" * 55)
    
    # Set configuration
    config.SPIKE_THRESHOLD = 20.0
    config.SPIKE_WEIGHT_MULTIPLIER = 5.0
    
    # Initialize championship
    championship = SpikeModelChampionship(spike_threshold=20.0)
    
    # Test a small set of models for quick validation
    test_models = [
        'xgboost',           # Original baseline
        'ensemble',          # New ensemble approach
        'rate_of_change',    # Rate-based detector
        'linear'             # Simple baseline
    ]
    
    print(f"Testing {len(test_models)} models with quick parameters...")
    print(f"Models: {', '.join(test_models)}")
    
    # Run quick test (reduced anchors, recent data only)
    results = championship.run_championship(
        models_to_test=test_models,
        n_anchors=15,  # Reduced for speed
        quick_test=True,
        compare_baselines=True
    )
    
    print("\n✅ Quick test completed!")
    print("If this worked, you can run the full championship with:")
    print("python spike_model_championship.py --anchors 100")
    
    return results


if __name__ == "__main__":
    try:
        results = quick_test()
        print("\n🎉 New forecasting infrastructure is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)