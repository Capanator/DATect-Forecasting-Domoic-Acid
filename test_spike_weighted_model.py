#!/usr/bin/env python3
"""
Test Spike-Weighted XGBoost Model
=================================

Quick test to see if the spike-weighted model performs better than standard XGBoost.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.spike_timing_optimizer import SpikeTimingOptimizer
from forecasting.logging_config import get_logger

logger = get_logger(__name__)


def test_spike_weighted_model():
    """Test the spike-weighted XGBoost model vs standard XGBoost."""
    
    print("🔬 Testing Spike-Weighted XGBoost Model")
    
    # Initialize components
    optimizer = SpikeTimingOptimizer(spike_threshold=20.0)
    engine = ForecastEngine(validate_on_init=False)
    
    print("   Generating spike-weighted XGBoost predictions...")
    
    # Generate spike-weighted XGBoost predictions
    spike_weighted_results = engine.run_retrospective_evaluation(
        task="regression",
        model_type="spike_xgboost",  # Use our new spike-weighted model
        n_anchors=30,  # Smaller sample for quick test
        min_test_date="2015-01-01"  # Recent data only
    )
    
    if spike_weighted_results is None:
        print("❌ Failed to generate spike-weighted predictions")
        return
    
    print(f"   Generated {len(spike_weighted_results)} spike-weighted predictions")
    
    # Load existing standard XGBoost data (filtered to same date range)
    xgboost_path = "cache/retrospective/regression_xgboost.json"
    with open(xgboost_path, 'r') as f:
        xgboost_data = json.load(f)
    
    xgboost_df = pd.DataFrame(xgboost_data)
    xgboost_df['date'] = pd.to_datetime(xgboost_df['date'])
    
    # Filter to same date range as spike-weighted test
    min_date = spike_weighted_results['date'].min()
    max_date = spike_weighted_results['date'].max()
    
    xgboost_filtered = xgboost_df[
        (xgboost_df['date'] >= min_date) &
        (xgboost_df['date'] <= max_date)
    ].copy()
    
    print(f"   Filtered standard XGBoost to {len(xgboost_filtered)} predictions for comparison")
    
    # Evaluate both models
    print("\n📊 COMPARISON RESULTS:")
    
    # Standard XGBoost
    standard_results = optimizer.evaluate_spike_timing_performance(xgboost_filtered)
    print(f"\n   Standard XGBoost:")
    print(f"      Spikes: {standard_results['n_actual_spikes']} actual, {standard_results['n_predicted_spikes']} predicted")
    if standard_results['spike_detection']:
        det = standard_results['spike_detection']
        print(f"      F1: {det['f1_score']:.3f}, Precision: {det['precision']:.3f}, Recall: {det['recall']:.3f}")
    if standard_results['spike_magnitude']:
        mag = standard_results['spike_magnitude']
        print(f"      Spike MAE: {mag.get('spike_mae', 0):.2f} ppm, Bias: {mag.get('spike_bias', 0):+.2f} ppm")
    
    # Spike-weighted XGBoost
    spike_weighted_eval = optimizer.evaluate_spike_timing_performance(spike_weighted_results)
    print(f"\n   Spike-Weighted XGBoost:")
    print(f"      Spikes: {spike_weighted_eval['n_actual_spikes']} actual, {spike_weighted_eval['n_predicted_spikes']} predicted")
    if spike_weighted_eval['spike_detection']:
        det = spike_weighted_eval['spike_detection']
        print(f"      F1: {det['f1_score']:.3f}, Precision: {det['precision']:.3f}, Recall: {det['recall']:.3f}")
    if spike_weighted_eval['spike_magnitude']:
        mag = spike_weighted_eval['spike_magnitude']
        print(f"      Spike MAE: {mag.get('spike_mae', 0):.2f} ppm, Bias: {mag.get('spike_bias', 0):+.2f} ppm")
    
    # Calculate improvement
    if (standard_results['spike_detection'] and spike_weighted_eval['spike_detection']):
        f1_improvement = spike_weighted_eval['spike_detection']['f1_score'] - standard_results['spike_detection']['f1_score']
        print(f"\n🎯 IMPROVEMENT:")
        print(f"   F1 Score: {f1_improvement:+.3f}")
        
        if f1_improvement > 0.02:
            print("   ✅ Spike-weighted model shows improvement!")
        elif f1_improvement < -0.02:
            print("   ❌ Standard model performs better")
        else:
            print("   ⚠️  Performance is similar")
    
    # Compare to naive baseline
    print(f"\n📋 BASELINE COMPARISON:")
    
    # Load naive baseline results from previous run
    baseline_path = "results/spike_timing_optimization/naive_lag_7d_baseline.parquet"
    if os.path.exists(baseline_path):
        baseline_df = pd.read_parquet(baseline_path)
        baseline_filtered = baseline_df[
            (baseline_df['date'] >= min_date) &
            (baseline_df['date'] <= max_date)
        ].copy()
        
        baseline_eval = optimizer.evaluate_spike_timing_performance(baseline_filtered)
        
        if baseline_eval['spike_detection']:
            baseline_f1 = baseline_eval['spike_detection']['f1_score']
            spike_f1 = spike_weighted_eval['spike_detection']['f1_score'] if spike_weighted_eval['spike_detection'] else 0
            
            print(f"   Naive 7-day lag F1: {baseline_f1:.3f}")
            print(f"   Spike-weighted F1: {spike_f1:.3f}")
            print(f"   Improvement vs baseline: {spike_f1 - baseline_f1:+.3f}")
            
            if spike_f1 > baseline_f1:
                print("   ✅ Spike-weighted model beats naive baseline!")
            else:
                print("   ❌ Naive baseline still outperforms spike-weighted model")
    
    print(f"\n✅ Spike-weighted model test completed!")
    
    return {
        'standard_xgboost': standard_results,
        'spike_weighted_xgboost': spike_weighted_eval
    }


if __name__ == "__main__":
    try:
        # Set spike optimization config
        config.SPIKE_THRESHOLD = 20.0
        config.SPIKE_WEIGHT_MULTIPLIER = 5.0
        
        results = test_spike_weighted_model()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)