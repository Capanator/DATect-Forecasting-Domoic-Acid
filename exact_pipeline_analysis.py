#!/usr/bin/env python3
"""
EXACT Pipeline Analysis - Matching Precompute Cache Methodology
================================================================

This replicates the EXACT same process as precompute_cache.py:
1. Load data from parquet (actual_da, predicted_da columns)
2. Convert to base_results format (actual_da, predicted_da)
3. Call _compute_summary() just like the pipeline does
4. Compare with naive baseline using same methodology
"""

import sys
import pandas as pd
import numpy as np
from datetime import timedelta
sys.path.append('.')

def main():
    print("🔬 EXACT PIPELINE ANALYSIS")
    print("="*50)
    print("Replicating precompute_cache.py methodology exactly...")
    
    # Step 1: Load XGBoost results from parquet (original column names)
    xgb_df = pd.read_parquet("./cache/retrospective/regression_xgboost.parquet")
    print(f"Loaded {len(xgb_df)} XGBoost predictions")
    
    # Step 2: Convert to base_results format (canonical keys)
    required_cols = {'date','site','actual_da','predicted_da','anchor_date'}
    missing = [c for c in ['actual_da','predicted_da'] if c not in xgb_df.columns]
    if missing:
        raise RuntimeError(f"XGBoost parquet is missing canonical columns {missing}. Regenerate cache via precompute_cache.py.")
    from backend.api import clean_float_for_json, _compute_summary
    
    base_results = []
    for _, row in xgb_df.iterrows():
        record = {
            "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
            "site": row['site'],
            "actual_da": clean_float_for_json(row['actual_da']) if 'actual_da' in row and pd.notnull(row['actual_da']) else None,
            "predicted_da": clean_float_for_json(row['predicted_da']) if 'predicted_da' in row and pd.notnull(row['predicted_da']) else None,
        }
        base_results.append(record)
    
    # Step 3: Calculate summary using EXACT pipeline method (correct zero handling)
    xgb_summary = _compute_summary(base_results)
    
    print(f"\n📊 XGBOOST METRICS (Exact Pipeline Method)")
    print(f"R² Score: {xgb_summary.get('r2_score', 0):.4f}")  # Should be ~0.3661
    print(f"MAE:      {xgb_summary.get('mae', 0):.4f}")       # Should be ~6.73
    print(f"F1 Score: {xgb_summary.get('f1_score', 0):.4f}") # Should be ~0.5924
    print(f"Total forecasts: {xgb_summary.get('total_forecasts', 0)}")
    
    # Step 4: Calculate naive baseline for comparison
    print(f"\n🔄 Calculating naive baseline...")
    
    # Load historical data
    historical_df = pd.read_parquet("./data/processed/final_output.parquet")
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    xgb_df['date'] = pd.to_datetime(xgb_df['date'])
    xgb_df['anchor_date'] = pd.to_datetime(xgb_df['anchor_date'])
    
    # Calculate naive predictions using same temporal safety
    naive_base_results = []
    
    for _, row in xgb_df.iterrows():
        site = row['site']
        anchor_date = row['anchor_date']
        actual_da = row['actual_da']
        
        # Get historical data before anchor date (exact pipeline temporal safety)
        site_history = historical_df[
            (historical_df['site'] == site) & 
            (historical_df['date'] < anchor_date) &
            (historical_df['da'].notna())
        ].sort_values('date')
        
        if site_history.empty:
            continue
            
        # Find naive prediction (7 days before anchor ±3 days)
        target_date = anchor_date - timedelta(days=7)
        candidates = site_history[
            (site_history['date'] >= target_date - timedelta(days=3)) &
            (site_history['date'] <= target_date + timedelta(days=3))
        ]
        
        if not candidates.empty:
            candidates = candidates.copy()
            candidates['diff'] = abs((candidates['date'] - target_date).dt.days)
            naive_pred = candidates.loc[candidates['diff'].idxmin(), 'da']
        else:
            naive_pred = site_history.iloc[-1]['da']
        
        # Format like pipeline base_results
        record = {
            "date": row['date'].strftime('%Y-%m-%d'),
            "site": site,
            "actual_da": clean_float_for_json(actual_da),
            "predicted_da": clean_float_for_json(naive_pred),  # Naive prediction
        }
        naive_base_results.append(record)
    
    # Step 5: Calculate naive summary using same pipeline method
    naive_summary = _compute_summary(naive_base_results)
    
    print(f"\n📊 NAIVE BASELINE METRICS (Exact Pipeline Method)")
    print(f"R² Score: {naive_summary.get('r2_score', 0):.4f}")
    print(f"MAE:      {naive_summary.get('mae', 0):.4f}")
    print(f"F1 Score: {naive_summary.get('f1_score', 0):.4f}")
    print(f"Total forecasts: {naive_summary.get('total_forecasts', 0)}")
    
    # Step 6: Comparison
    xgb_r2 = xgb_summary.get('r2_score', 0)
    naive_r2 = naive_summary.get('r2_score', 0)
    xgb_mae = xgb_summary.get('mae', 0)
    naive_mae = naive_summary.get('mae', 0)
    xgb_f1 = xgb_summary.get('f1_score', 0)
    naive_f1 = naive_summary.get('f1_score', 0)
    
    print(f"\n🏆 FINAL COMPARISON (Using Exact Pipeline Method)")
    print("="*60)
    print(f"| Metric   | XGBoost | Naive   | Winner    |")
    print(f"|----------|---------|---------|-----------|")
    print(f"| R² Score | {xgb_r2:.4f}  | {naive_r2:.4f}  | {'XGBoost' if xgb_r2 > naive_r2 else 'Naive':>9} |")
    print(f"| MAE      | {xgb_mae:.4f}  | {naive_mae:.4f}  | {'XGBoost' if xgb_mae < naive_mae else 'Naive':>9} |")
    print(f"| F1 Score | {xgb_f1:.4f}  | {naive_f1:.4f}  | {'XGBoost' if xgb_f1 > naive_f1 else 'Naive':>9} |")
    
    print(f"\n✅ VERIFICATION")
    print(f"Expected XGBoost R² ≈ 0.3661: Actual = {xgb_r2:.4f}")
    print(f"Expected XGBoost MAE ≈ 6.73:  Actual = {xgb_mae:.4f}")
    print(f"Expected XGBoost F1 ≈ 0.5924: Actual = {xgb_f1:.4f}")
    
    if abs(xgb_r2 - 0.3661) < 0.01:
        print("🎯 Match with pipeline output!")
    else:
        print(f"⚠️  Difference from expected (check data/cache)")
    
    return {
        'xgb_metrics': xgb_summary,
        'naive_metrics': naive_summary,
        'comparison_count': len(naive_base_results)
    }

if __name__ == "__main__":
    main()
