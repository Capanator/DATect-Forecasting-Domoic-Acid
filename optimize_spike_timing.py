#!/usr/bin/env python3
"""
Optimize Domoic Acid Spike Timing Prediction
=============================================

Script to optimize the forecasting system for accurate initial spike timing prediction.
Implements naive baselines and spike-focused evaluation metrics.

Key objectives:
1. Focus on predicting when DA levels first exceed 20 ppm
2. Ensure forecasts rise simultaneously with actual DA increases
3. Validate against naive baseline (DA shifted forward by one week)
4. Prioritize spike timing accuracy over gradual decline accuracy

Usage:
    python optimize_spike_timing.py [--model xgboost] [--baseline-lag 7] [--spike-threshold 20]
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import warnings

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.spike_timing_optimizer import SpikeTimingOptimizer
from forecasting.data_processor import DataProcessor
from forecasting.logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


def load_existing_xgboost_data():
    """Load existing XGBoost retrospective data for comparison."""
    xgboost_path = "cache/retrospective/regression_xgboost.json"
    
    if not os.path.exists(xgboost_path):
        logger.error(f"XGBoost data file not found: {xgboost_path}")
        return None
    
    logger.info(f"Loading existing XGBoost data from {xgboost_path}")
    
    with open(xgboost_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['anchor_date'] = pd.to_datetime(df['anchor_date'])
    
    logger.info(f"Loaded {len(df)} XGBoost predictions from cache")
    return df


def create_weighted_loss_model_config():
    """Create model configuration that prioritizes spike timing accuracy."""
    
    # Modify config to prioritize early spike detection
    spike_focused_config = {
        'SPIKE_WEIGHT_MULTIPLIER': 5.0,  # Weight spike events 5x more
        'EARLY_PREDICTION_BONUS': 0.8,  # Slight bonus for early predictions vs late
        'TIMING_PENALTY_THRESHOLD': 7,  # Days beyond which timing penalty applies
        'USE_CUSTOM_LOSS': True,
        'SPIKE_THRESHOLD': 20.0
    }
    
    # Update config with spike-focused parameters
    for key, value in spike_focused_config.items():
        setattr(config, key, value)
    
    logger.info("Updated config with spike timing optimization parameters")
    return spike_focused_config


def run_spike_timing_optimization(model_type="xgboost", baseline_lag_days=7, spike_threshold=20.0):
    """
    Run the complete spike timing optimization pipeline.
    
    Args:
        model_type: Type of model to evaluate ("xgboost", "linear")
        baseline_lag_days: Days for naive baseline lag
        spike_threshold: DA threshold for spike events (ppm)
    """
    logger.info(f"Starting spike timing optimization with model={model_type}, lag={baseline_lag_days}d")
    
    # Initialize components
    optimizer = SpikeTimingOptimizer(spike_threshold=spike_threshold)
    data_processor = DataProcessor()
    
    # Create output directory
    output_dir = Path("results/spike_timing_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the full dataset for baseline creation
    logger.info("Loading full dataset for baseline generation")
    full_data = data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
    
    # Generate naive baselines
    logger.info("Generating naive baselines...")
    
    naive_lag_baseline = optimizer.create_naive_lag_baseline(full_data, lag_days=baseline_lag_days)
    persistence_baseline = optimizer.create_persistence_baseline(full_data)
    seasonal_baseline = optimizer.create_seasonal_baseline(full_data, seasonal_days=365)
    
    # Save baselines for later analysis
    naive_lag_baseline.to_parquet(output_dir / f"naive_lag_{baseline_lag_days}d_baseline.parquet")
    persistence_baseline.to_parquet(output_dir / "persistence_baseline.parquet")
    seasonal_baseline.to_parquet(output_dir / "seasonal_baseline.parquet")
    
    logger.info(f"Baseline predictions generated and saved to {output_dir}")
    
    # Load existing XGBoost data for comparison
    xgboost_predictions = load_existing_xgboost_data()
    
    if xgboost_predictions is None:
        logger.error("Could not load existing XGBoost data - generating new predictions")
        # Generate new XGBoost predictions
        engine = ForecastEngine(validate_on_init=False)
        xgboost_predictions = engine.run_retrospective_evaluation(
            task="regression", 
            model_type="xgboost",
            n_anchors=50,
            min_test_date="2008-01-01"
        )
        
        if xgboost_predictions is None:
            logger.error("Failed to generate XGBoost predictions")
            return None
    
    # Filter XGBoost data to match baseline date ranges for fair comparison
    baseline_date_range = naive_lag_baseline['date'].min(), naive_lag_baseline['date'].max()
    xgboost_filtered = xgboost_predictions[
        (xgboost_predictions['date'] >= baseline_date_range[0]) &
        (xgboost_predictions['date'] <= baseline_date_range[1])
    ].copy()
    
    logger.info(f"Filtered XGBoost data to {len(xgboost_filtered)} predictions for fair comparison")
    
    # Perform spike-focused evaluations
    logger.info("Evaluating spike timing performance...")
    
    results = {}
    
    # Evaluate XGBoost model
    logger.info("Evaluating XGBoost model performance")
    xgboost_results = optimizer.evaluate_spike_timing_performance(xgboost_filtered)
    results['xgboost'] = xgboost_results
    
    # Evaluate naive lag baseline
    logger.info(f"Evaluating naive {baseline_lag_days}-day lag baseline")
    naive_lag_results = optimizer.evaluate_spike_timing_performance(naive_lag_baseline)
    results['naive_lag'] = naive_lag_results
    
    # Evaluate persistence baseline
    logger.info("Evaluating persistence baseline")
    persistence_results = optimizer.evaluate_spike_timing_performance(persistence_baseline)
    results['persistence'] = persistence_results
    
    # Evaluate seasonal baseline
    logger.info("Evaluating seasonal baseline")
    seasonal_results = optimizer.evaluate_spike_timing_performance(seasonal_baseline)
    results['seasonal'] = seasonal_results
    
    # Compare XGBoost vs Naive Lag (key validation test)
    logger.info("Performing key validation: XGBoost vs Naive Lag Baseline")
    key_comparison = optimizer.compare_models_spike_focus(
        xgboost_filtered, 
        naive_lag_baseline,
        model_name=f"XGBoost",
        baseline_name=f"Naive {baseline_lag_days}-day Lag"
    )
    
    results['key_validation'] = key_comparison
    
    # Additional comparisons
    persistence_comparison = optimizer.compare_models_spike_focus(
        xgboost_filtered, 
        persistence_baseline,
        model_name="XGBoost",
        baseline_name="Persistence"
    )
    
    seasonal_comparison = optimizer.compare_models_spike_focus(
        xgboost_filtered, 
        seasonal_baseline,
        model_name="XGBoost", 
        baseline_name="Seasonal"
    )
    
    results['persistence_comparison'] = persistence_comparison
    results['seasonal_comparison'] = seasonal_comparison
    
    # Generate comprehensive report
    report = generate_spike_timing_report(results, spike_threshold, baseline_lag_days)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = output_dir / f"spike_timing_optimization_{timestamp}.json"
    optimizer.save_comparison_results(results, results_file)
    
    # Save human-readable report
    report_file = output_dir / f"spike_timing_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save performance summary CSV
    summary_file = output_dir / f"performance_summary_{timestamp}.csv"
    save_performance_summary(results, summary_file)
    
    logger.info(f"Spike timing optimization completed!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print key findings
    print_key_findings(results)
    
    return results


def generate_spike_timing_report(results: dict, spike_threshold: float, baseline_lag_days: int) -> str:
    """Generate a comprehensive markdown report of spike timing optimization results."""
    
    report = f"""# Domoic Acid Spike Timing Optimization Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report analyzes the forecasting system's performance in predicting initial domoic acid spike timing, with a focus on when DA levels first exceed {spike_threshold} ppm.

### Key Validation Test Results

The critical test compares our XGBoost model against a naive baseline where predicted DA = actual DA shifted forward by {baseline_lag_days} days.

"""
    
    # Key validation results
    key_val = results.get('key_validation', {})
    if key_val:
        improvements = key_val.get('improvements', {})
        spike_det_imp = improvements.get('spike_detection', {})
        
        f1_improvement = spike_det_imp.get('f1_improvement', 0)
        
        report += f"""#### XGBoost vs Naive {baseline_lag_days}-Day Lag Comparison

- **F1 Score Improvement**: {f1_improvement:.3f} ({"✅ Positive" if f1_improvement > 0 else "❌ Negative"})
- **Precision Improvement**: {spike_det_imp.get('precision_improvement', 0):.3f}
- **Recall Improvement**: {spike_det_imp.get('recall_improvement', 0):.3f}

**Interpretation**: {key_val.get('recommendation', 'No recommendation available')}

"""
    
    # Detailed performance metrics
    report += """## Detailed Performance Analysis

### Spike Detection Performance

| Model | Precision | Recall | F1-Score | Spikes Detected | True Positives |
|-------|-----------|--------|----------|-----------------|----------------|
"""
    
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
            
        if 'spike_detection' in model_results and model_results['spike_detection']:
            det = model_results['spike_detection']
            report += f"| {model_name.title()} | {det['precision']:.3f} | {det['recall']:.3f} | {det['f1_score']:.3f} | {model_results['n_predicted_spikes']} | {det['true_positives']} |\n"
    
    # Spike magnitude accuracy
    report += """
### Spike Magnitude Accuracy (for DA > {spike_threshold} ppm)

| Model | MAE (ppm) | RMSE (ppm) | Bias (ppm) | R² |
|-------|-----------|------------|------------|----|
"""
    
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
            
        if 'spike_magnitude' in model_results and model_results['spike_magnitude']:
            mag = model_results['spike_magnitude']
            report += f"| {model_name.title()} | {mag.get('spike_mae', 0):.2f} | {mag.get('spike_rmse', 0):.2f} | {mag.get('spike_bias', 0):.2f} | {mag.get('spike_r2', 0):.3f} |\n"
    
    # Overall performance
    report += """
### Overall Performance (All DA Levels)

| Model | MAE (ppm) | RMSE (ppm) | R² |
|-------|-----------|------------|----| 
"""
    
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
            
        if 'overall_performance' in model_results:
            overall = model_results['overall_performance']
            report += f"| {model_name.title()} | {overall['overall_mae']:.2f} | {overall['overall_rmse']:.2f} | {overall['overall_r2']:.3f} |\n"
    
    # Recommendations
    report += """
## Recommendations

Based on this analysis:

### Primary Findings

1. **Spike Timing Accuracy**: """
    
    if key_val and f1_improvement > 0.05:
        report += "XGBoost model shows significant improvement over naive lag baseline for spike timing prediction."
    elif key_val and f1_improvement < -0.05:
        report += "⚠️ **CRITICAL**: Naive lag baseline outperforms XGBoost model, indicating forecasts are not providing value for spike prediction."
    else:
        report += "XGBoost performance is similar to naive lag baseline, suggesting limited added value for spike timing prediction."
    
    report += """

2. **Model Performance**: The analysis reveals the current model's strengths and weaknesses in predicting initial spike timing vs gradual decline phases.

3. **Optimization Opportunities**: Identified specific areas for improvement in spike timing prediction accuracy.

### Next Steps

1. **If XGBoost outperforms baselines**: Focus on further optimizing the model architecture and feature engineering for spike detection.

2. **If baselines are competitive**: Consider implementing weighted loss functions that heavily penalize missed spikes and reward early detection.

3. **In all cases**: Implement real-time spike alert thresholds that prioritize sensitivity over specificity for public health protection.

---

*Report generated by DATect Spike Timing Optimizer v1.0*
"""
    
    return report


def save_performance_summary(results: dict, output_file: Path) -> None:
    """Save a CSV summary of key performance metrics."""
    
    summary_data = []
    
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
        
        row = {
            'model': model_name,
            'n_actual_spikes': model_results.get('n_actual_spikes', 0),
            'n_predicted_spikes': model_results.get('n_predicted_spikes', 0)
        }
        
        # Spike detection metrics
        if 'spike_detection' in model_results and model_results['spike_detection']:
            det = model_results['spike_detection']
            row.update({
                'spike_precision': det['precision'],
                'spike_recall': det['recall'],
                'spike_f1': det['f1_score'],
                'true_positives': det['true_positives'],
                'false_positives': det['false_positives'],
                'false_negatives': det['false_negatives']
            })
        
        # Spike magnitude metrics
        if 'spike_magnitude' in model_results and model_results['spike_magnitude']:
            mag = model_results['spike_magnitude']
            row.update({
                'spike_mae': mag.get('spike_mae', np.nan),
                'spike_rmse': mag.get('spike_rmse', np.nan),
                'spike_bias': mag.get('spike_bias', np.nan),
                'spike_r2': mag.get('spike_r2', np.nan)
            })
        
        # Overall performance
        if 'overall_performance' in model_results:
            overall = model_results['overall_performance']
            row.update({
                'overall_mae': overall['overall_mae'],
                'overall_rmse': overall['overall_rmse'],
                'overall_r2': overall['overall_r2']
            })
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    logger.info(f"Performance summary saved to {output_file}")


def print_key_findings(results: dict) -> None:
    """Print key findings to console."""
    
    print("\n" + "="*70)
    print("🎯 SPIKE TIMING OPTIMIZATION - KEY FINDINGS")
    print("="*70)
    
    # Key validation test
    key_val = results.get('key_validation', {})
    if key_val:
        improvements = key_val.get('improvements', {})
        spike_det_imp = improvements.get('spike_detection', {})
        f1_improvement = spike_det_imp.get('f1_improvement', 0)
        
        print(f"\n📊 CRITICAL VALIDATION TEST:")
        print(f"   XGBoost vs Naive 7-day Lag Baseline")
        print(f"   F1 Score Improvement: {f1_improvement:+.3f}")
        
        if f1_improvement > 0.05:
            print(f"   ✅ PASS: Model provides value for spike timing prediction")
        elif f1_improvement < -0.05:
            print(f"   ❌ FAIL: Naive baseline outperforms model - forecasts not adding value")
        else:
            print(f"   ⚠️  MARGINAL: Similar performance to baseline")
        
        print(f"\n   Recommendation: {key_val.get('recommendation', 'None')}")
    
    # Best performing model
    best_f1 = 0
    best_model = None
    
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
            
        if 'spike_detection' in model_results and model_results['spike_detection']:
            f1 = model_results['spike_detection']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
    
    if best_model:
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model.title()}")
        best_results = results[best_model]
        if 'spike_detection' in best_results:
            det = best_results['spike_detection']
            print(f"   Precision: {det['precision']:.3f}")
            print(f"   Recall: {det['recall']:.3f}")
            print(f"   F1-Score: {det['f1_score']:.3f}")
        
        if 'spike_magnitude' in best_results and best_results['spike_magnitude']:
            mag = best_results['spike_magnitude']
            print(f"   Spike MAE: {mag.get('spike_mae', 0):.2f} ppm")
            print(f"   Spike Bias: {mag.get('spike_bias', 0):+.2f} ppm")
    
    print(f"\n📈 SPIKE EVENTS SUMMARY:")
    for model_name, model_results in results.items():
        if model_name.endswith('_comparison') or model_name == 'key_validation':
            continue
        print(f"   {model_name.title()}: {model_results.get('n_actual_spikes', 0)} actual, " 
              f"{model_results.get('n_predicted_spikes', 0)} predicted")
    
    print("\n" + "="*70)


def main():
    """Main function to run spike timing optimization."""
    
    parser = argparse.ArgumentParser(description="Optimize domoic acid spike timing prediction")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "linear"],
                       help="Model type to evaluate")
    parser.add_argument("--baseline-lag", type=int, default=7,
                       help="Days for naive baseline lag")
    parser.add_argument("--spike-threshold", type=float, default=20.0,
                       help="DA threshold for spike events (ppm)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    print("🎯 Starting Domoic Acid Spike Timing Optimization")
    print(f"   Model: {args.model}")
    print(f"   Baseline lag: {args.baseline_lag} days") 
    print(f"   Spike threshold: {args.spike_threshold} ppm")
    
    try:
        results = run_spike_timing_optimization(
            model_type=args.model,
            baseline_lag_days=args.baseline_lag,
            spike_threshold=args.spike_threshold
        )
        
        if results:
            print("\n✅ Spike timing optimization completed successfully!")
            print("   Check the results/ directory for detailed outputs.")
        else:
            print("\n❌ Optimization failed - check logs for details")
            return 1
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())