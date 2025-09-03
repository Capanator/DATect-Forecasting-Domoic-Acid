#!/usr/bin/env python3
"""
FINAL CORRECTED XGBoost vs Naive Baseline Analysis
==================================================

Uses the exact pipeline methodology and should match the expected R² ≈ 0.49
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, precision_score, recall_score
import os

def load_fresh_data():
    """Load the freshest XGBoost data and historical data."""
    print("📊 Loading fresh XGBoost results...")
    
    # Load XGBoost results (use parquet for speed)
    xgb_df = pd.read_parquet("./cache/retrospective/regression_xgboost.parquet")
    xgb_df['date'] = pd.to_datetime(xgb_df['date'])
    xgb_df['anchor_date'] = pd.to_datetime(xgb_df['anchor_date'])
    
    print(f"   Loaded {len(xgb_df)} XGBoost predictions")
    print(f"   R² sanity check on XGB data: {r2_score(xgb_df['da'], xgb_df['Predicted_da']):.4f}")
    
    # Load historical data
    historical_df = pd.read_parquet("./data/processed/final_output.parquet")
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df = historical_df.sort_values(['site', 'date'])
    
    print(f"   Loaded {len(historical_df)} historical records")
    
    return xgb_df, historical_df

def calculate_naive_predictions(xgb_df, historical_df):
    """Calculate naive baseline predictions with exact temporal safety."""
    print("🔄 Calculating naive baseline predictions...")
    
    results = []
    
    for _, row in xgb_df.iterrows():
        site = row['site']
        anchor_date = row['anchor_date']
        
        # Get historical data before anchor date
        site_history = historical_df[
            (historical_df['site'] == site) & 
            (historical_df['date'] < anchor_date) &
            (historical_df['da'].notna())
        ].sort_values('date')
        
        if site_history.empty:
            continue
            
        # Target: 7 days before anchor date
        target_date = anchor_date - timedelta(days=7)
        
        # Find closest match within ±3 days
        candidates = site_history[
            (site_history['date'] >= target_date - timedelta(days=3)) &
            (site_history['date'] <= target_date + timedelta(days=3))
        ]
        
        if not candidates.empty:
            # Use closest to target
            candidates = candidates.copy()
            candidates['diff'] = abs((candidates['date'] - target_date).dt.days)
            best_match = candidates.loc[candidates['diff'].idxmin()]
            naive_pred = best_match['da']
        else:
            # Fallback: most recent before anchor
            naive_pred = site_history.iloc[-1]['da']
        
        results.append({
            'site': row['site'],
            'date': row['date'],
            'anchor_date': row['anchor_date'],
            'actual_da': row['da'],  # Actual DA value
            'xgb_prediction': row['Predicted_da'],  # XGBoost prediction
            'naive_prediction': naive_pred  # Naive baseline
        })
    
    return pd.DataFrame(results)

def calculate_metrics(actual, predicted, threshold=15.0):
    """Calculate metrics using exact pipeline methodology."""
    # Regression metrics
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    
    # Spike detection (15 μg/g like pipeline)
    actual_binary = (actual > threshold).astype(int)
    pred_binary = (predicted > threshold).astype(int)
    
    f1 = f1_score(actual_binary, pred_binary, zero_division=0)
    precision = precision_score(actual_binary, pred_binary, zero_division=0)
    recall = recall_score(actual_binary, pred_binary, zero_division=0)
    
    return {
        'r2': r2,
        'mae': mae,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'spike_rate': actual_binary.mean()
    }

def main():
    """Run final corrected analysis."""
    
    # Load data
    xgb_df, historical_df = load_fresh_data()
    
    # Calculate naive predictions
    comparison_df = calculate_naive_predictions(xgb_df, historical_df)
    
    print(f"\n🔬 FINAL CORRECTED ANALYSIS RESULTS")
    print("="*50)
    print(f"Predictions analyzed: {len(comparison_df)}")
    print(f"Sites: {comparison_df['site'].nunique()}")
    print(f"Date range: {comparison_df['date'].min().date()} to {comparison_df['date'].max().date()}")
    
    # Calculate overall metrics
    xgb_metrics = calculate_metrics(comparison_df['actual_da'], comparison_df['xgb_prediction'])
    naive_metrics = calculate_metrics(comparison_df['actual_da'], comparison_df['naive_prediction'])
    
    print(f"\n📊 OVERALL PERFORMANCE RESULTS")
    print("-"*40)
    print(f"XGBoost R²:    {xgb_metrics['r2']:.4f}")
    print(f"Naive R²:      {naive_metrics['r2']:.4f}")
    print(f"R² Difference: {xgb_metrics['r2'] - naive_metrics['r2']:+.4f}")
    
    print(f"\nXGBoost MAE:   {xgb_metrics['mae']:.3f} μg/g")
    print(f"Naive MAE:     {naive_metrics['mae']:.3f} μg/g") 
    print(f"MAE Difference: {xgb_metrics['mae'] - naive_metrics['mae']:+.3f} μg/g")
    
    print(f"\nSpike Detection (15 μg/g threshold):")
    print(f"XGBoost F1:    {xgb_metrics['f1']:.4f}")
    print(f"Naive F1:      {naive_metrics['f1']:.4f}")
    print(f"F1 Difference: {xgb_metrics['f1'] - naive_metrics['f1']:+.4f}")
    
    # Winner summary
    print(f"\n🏆 WINNERS BY METRIC")
    print("-"*25)
    print(f"R² Score:      {'XGBoost' if xgb_metrics['r2'] > naive_metrics['r2'] else 'Naive'}")
    print(f"MAE:           {'XGBoost' if xgb_metrics['mae'] < naive_metrics['mae'] else 'Naive'}")
    print(f"F1 Score:      {'XGBoost' if xgb_metrics['f1'] > naive_metrics['f1'] else 'Naive'}")
    print(f"Precision:     {'XGBoost' if xgb_metrics['precision'] > naive_metrics['precision'] else 'Naive'}")
    print(f"Recall:        {'XGBoost' if xgb_metrics['recall'] > naive_metrics['recall'] else 'Naive'}")
    
    # Site breakdown (top 5 by number of predictions)
    print(f"\n🏖️  TOP SITE PERFORMANCE")
    print("-"*60)
    print(f"{'Site':<15} {'N':<4} {'XGB R²':<8} {'Naive R²':<8} {'Winner':<8}")
    print("-"*60)
    
    for site in comparison_df['site'].value_counts().head().index:
        site_data = comparison_df[comparison_df['site'] == site]
        site_xgb_r2 = r2_score(site_data['actual_da'], site_data['xgb_prediction'])
        site_naive_r2 = r2_score(site_data['actual_da'], site_data['naive_prediction'])
        winner = "XGBoost" if site_xgb_r2 > site_naive_r2 else "Naive"
        
        print(f"{site:<15} {len(site_data):<4} {site_xgb_r2:<8.3f} {site_naive_r2:<8.3f} {winner:<8}")
    
    # Save results
    comparison_df.to_parquet("./final_corrected_results.parquet", index=False)
    print(f"\n💾 Results saved to: final_corrected_results.parquet")
    
    print(f"\n✅ Analysis complete! XGBoost R² = {xgb_metrics['r2']:.4f}")
    print(f"   Expected ≈ 0.49, Actual = {xgb_metrics['r2']:.4f}")
    
    return xgb_metrics, naive_metrics, comparison_df

if __name__ == "__main__":
    main()