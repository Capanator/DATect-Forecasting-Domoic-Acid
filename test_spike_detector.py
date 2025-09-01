#!/usr/bin/env python3
"""
Test Script for Precision Spike Detection System
===============================================

Evaluates the new spike detection system against naive baseline
with focus on 15 ppm threshold and false positive control.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add forecasting module to path
sys.path.insert(0, str(Path(__file__).parent / "forecasting"))

from forecasting.spike_detector import PrecisionSpikeDetector
import config


def main():
    """
    Run comprehensive spike detection evaluation.
    """
    print("=" * 60)
    print("PRECISION SPIKE DETECTION EVALUATION")
    print("=" * 60)
    print(f"Spike threshold: 15 ppm")
    print(f"Focus: Beat naive baseline with controlled false positives")
    print("")
    
    # Check data availability
    if not os.path.exists(config.FINAL_OUTPUT_PATH):
        print(f"ERROR: Data file not found: {config.FINAL_OUTPUT_PATH}")
        print("Please run dataset-creation.py first")
        return 1
    
    # Initialize spike detector
    print("Initializing precision spike detection system...")
    detector = PrecisionSpikeDetector()
    
    # Run evaluation with moderate sample size for thorough testing
    print("Running spike detection evaluation...")
    print("This will take several minutes...")
    print("")
    
    try:
        results_df, metrics = detector.run_precision_evaluation(
            n_anchors=150,  # Increased sample size for robust evaluation
            min_test_date="2010-01-01"  # Ensure sufficient historical context
        )
        
        if results_df is None or metrics is None:
            print("ERROR: Evaluation failed - no results generated")
            return 1
            
        print("")
        print("=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Save results for further analysis
        results_file = "cache/spike_detection_results.parquet"
        os.makedirs("cache", exist_ok=True)
        
        results_df.to_parquet(results_file, engine="pyarrow")
        print(f"Results saved to: {results_file}")
        
        # Quick summary statistics
        print("")
        print("QUICK SUMMARY:")
        total_predictions = len(results_df)
        actual_spikes = (results_df['da'] > 15.0).sum()
        spike_rate = actual_spikes / total_predictions * 100
        
        print(f"  Total predictions: {total_predictions}")
        print(f"  Actual spikes (>15 ppm): {actual_spikes} ({spike_rate:.1f}%)")
        print(f"  Date range: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
        print(f"  Sites: {results_df['site'].nunique()}")
        
        # Check if any model beat the naive baseline
        naive_metrics = metrics.get('naive_baseline', {})
        naive_f1 = naive_metrics.get('f1', 0)
        naive_r2 = naive_metrics.get('r2', 0)
        
        models_beating_naive = 0
        for model_name, model_metrics in metrics.items():
            if model_name != 'naive_baseline':
                model_f1 = model_metrics.get('f1', 0)
                model_r2 = model_metrics.get('r2', 0)
                if model_f1 > naive_f1 and model_r2 > naive_r2:
                    models_beating_naive += 1
        
        print("")
        if models_beating_naive > 0:
            print("🎉 SUCCESS: Found models that beat the naive baseline!")
            print(f"   {models_beating_naive} models outperform naive baseline")
        else:
            print("⚠️  CHALLENGE: No models consistently beat naive baseline")
            print("   Further model development recommended")
        
        # Practical utility assessment
        print("")
        print("PRACTICAL UTILITY:")
        for model_name, model_metrics in metrics.items():
            if model_name == 'naive_baseline':
                continue
            
            fps = model_metrics.get('n_false_positives', 0)
            precision = model_metrics.get('precision', 0)
            
            if precision > 0.7 and fps < total_predictions * 0.05:  # <5% false positives
                utility = "HIGH"
            elif precision > 0.5 and fps < total_predictions * 0.1:  # <10% false positives
                utility = "MODERATE"
            else:
                utility = "LOW"
            
            print(f"  {model_name}: {utility} utility (Precision: {precision:.3f}, FPs: {fps})")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)