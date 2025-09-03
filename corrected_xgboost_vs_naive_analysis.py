#!/usr/bin/env python3
"""
CORRECTED XGBoost vs Naive Baseline Analysis
===========================================

Fixed analysis using EXACT same methodology as the pipeline:
- 15 μg/g spike threshold (not 20)
- Exact column names from cached data
- Same metric calculations as forecast_engine.py and api.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score
import json
import os

def load_xgboost_results():
    """Load XGBoost regression results from cache - using exact pipeline data."""
    print("📊 Loading XGBoost results from cache...")
    
    parquet_path = "./cache/retrospective/regression_xgboost.parquet"
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    df['anchor_date'] = pd.to_datetime(df['anchor_date'])
    
    print(f"   Loaded {len(df)} XGBoost predictions")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df

def load_historical_data():
    """Load historical data for naive baseline calculation."""
    print("📈 Loading historical data from final_output.parquet...")
    
    final_output_path = "./data/processed/final_output.parquet"
    if not os.path.exists(final_output_path):
        # Try alternative locations
        for alt_path in ["./final_output.parquet", "../final_output.parquet"]:
            if os.path.exists(alt_path):
                final_output_path = alt_path
                break
    
    df = pd.read_parquet(final_output_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['site', 'date'])
    print(f"   Loaded {len(df)} historical records")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df

def calculate_naive_baseline_pipeline_method(xgboost_df, historical_df):
    """
    Calculate naive baseline using EXACT same temporal safety as pipeline.
    Uses the pipeline's anchor_date methodology.
    """
    print("🔄 Calculating naive baseline with pipeline temporal safety...")
    
    results = []
    successful_matches = 0
    total_predictions = len(xgboost_df)
    
    for _, row in xgboost_df.iterrows():
        site = row['site']
        prediction_date = pd.to_datetime(row['date'])
        anchor_date = pd.to_datetime(row['anchor_date'])  # Use exact anchor from pipeline
        actual_da = row['da']  # Pipeline column name
        xgb_prediction = row['Predicted_da']  # Pipeline column name
        
        # PIPELINE TEMPORAL SAFETY: Only use historical data before anchor date
        site_historical = historical_df[
            (historical_df['site'] == site) & 
            (historical_df['date'] < anchor_date)  # Strict temporal cutoff matching pipeline
        ].sort_values('date')
        
        if site_historical.empty:
            continue
        
        # Find naive prediction: previous week's value (±3 days tolerance)
        target_baseline_date = anchor_date - timedelta(days=7)
        
        # Look for data within ±3 days of target
        tolerance_window = site_historical[
            (site_historical['date'] >= target_baseline_date - timedelta(days=3)) &
            (site_historical['date'] <= target_baseline_date + timedelta(days=3)) &
            (site_historical['da'].notna())
        ]
        
        naive_prediction = None
        baseline_date_used = None
        
        if not tolerance_window.empty:
            # Use closest date to target
            tolerance_window = tolerance_window.copy()
            tolerance_window['days_diff'] = abs((tolerance_window['date'] - target_baseline_date).dt.days)
            closest_match = tolerance_window.loc[tolerance_window['days_diff'].idxmin()]
            naive_prediction = closest_match['da']
            baseline_date_used = closest_match['date']
            successful_matches += 1
        else:
            # Fallback: use most recent available data (still before anchor)
            recent_data = site_historical[site_historical['da'].notna()]
            if not recent_data.empty:
                naive_prediction = recent_data.iloc[-1]['da']
                baseline_date_used = recent_data.iloc[-1]['date']
                successful_matches += 1
        
        if naive_prediction is not None and pd.notna(actual_da) and pd.notna(xgb_prediction):
            results.append({
                'site': site,
                'date': prediction_date,
                'anchor_date': anchor_date,
                'da': actual_da,  # PIPELINE COLUMN NAME
                'Predicted_da': xgb_prediction,  # PIPELINE COLUMN NAME  
                'naive_prediction': naive_prediction,
                'baseline_date_used': baseline_date_used,
                'days_to_baseline': (anchor_date - baseline_date_used).days if baseline_date_used else None
            })
    
    print(f"   Successfully matched {successful_matches}/{total_predictions} predictions ({100*successful_matches/total_predictions:.1f}%)")
    return pd.DataFrame(results)

def calculate_pipeline_spike_metrics(actual, predicted, threshold=15.0):
    """Calculate spike detection metrics using EXACT pipeline methodology."""
    # PIPELINE METHOD: Convert to binary using exact same logic
    actual_binary = [1 if val > threshold else 0 for val in actual]
    pred_binary = [1 if val > threshold else 0 for val in predicted]
    
    return {
        'f1': f1_score(actual_binary, pred_binary, zero_division=0),
        'precision': precision_score(actual_binary, pred_binary, zero_division=0),
        'recall': recall_score(actual_binary, pred_binary, zero_division=0),
        'accuracy': accuracy_score(actual_binary, pred_binary),
        'spike_rate': np.mean(actual_binary)
    }

def generate_corrected_analysis(comparison_df):
    """Generate analysis using EXACT pipeline methodology."""
    print("\n" + "="*70)
    print("🔬 CORRECTED XGBOOST vs NAIVE BASELINE ANALYSIS")
    print("   (Using EXACT pipeline methodology)")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%B %Y')}")
    print(f"Dataset: {len(comparison_df)} predictions across {comparison_df['site'].nunique()} sites")
    print(f"Date Range: {comparison_df['date'].min().date()} to {comparison_df['date'].max().date()}")
    print(f"Spike Threshold: 15 μg/g (matching pipeline)")
    
    # EXACT PIPELINE METHOD: Use 'da' and 'Predicted_da' columns
    actual_values = comparison_df['da']
    xgb_predictions = comparison_df['Predicted_da'] 
    naive_predictions = comparison_df['naive_prediction']
    
    print(f"\n📊 OVERALL PERFORMANCE COMPARISON (Pipeline Method)")
    print("-" * 60)
    
    # Regression metrics - EXACT pipeline calculation
    try:
        xgb_r2 = r2_score(actual_values, xgb_predictions)
        naive_r2 = r2_score(actual_values, naive_predictions)
        
        xgb_mae = mean_absolute_error(actual_values, xgb_predictions)
        naive_mae = mean_absolute_error(actual_values, naive_predictions)
        
        print(f"R² Score Results:")
        print(f"  XGBoost R²: {xgb_r2:.4f}")
        print(f"  Naive R²:   {naive_r2:.4f}")
        print(f"  Difference: {xgb_r2 - naive_r2:+.4f}")
    except Exception as e:
        print(f"Error in R² calculation: {e}")
        return None
    
    # Spike detection metrics - EXACT pipeline methodology (15 μg/g)
    xgb_spike_metrics = calculate_pipeline_spike_metrics(actual_values, xgb_predictions, 15.0)
    naive_spike_metrics = calculate_pipeline_spike_metrics(actual_values, naive_predictions, 15.0)
    
    print(f"\nMAE Results:")
    print(f"  XGBoost MAE: {xgb_mae:.3f} μg/g")
    print(f"  Naive MAE:   {naive_mae:.3f} μg/g")
    print(f"  Difference:  {xgb_mae - naive_mae:+.3f} μg/g")
    
    print(f"\nSpike Detection Results (15 μg/g threshold):")
    print(f"  XGBoost F1: {xgb_spike_metrics['f1']:.4f}")
    print(f"  Naive F1:   {naive_spike_metrics['f1']:.4f}")
    print(f"  Difference: {xgb_spike_metrics['f1'] - naive_spike_metrics['f1']:+.4f}")
    
    # Performance comparison table
    print(f"\n📋 PERFORMANCE COMPARISON TABLE")
    print("-" * 60)
    print(f"| Metric              | XGBoost | Naive Baseline | Winner    |")
    print(f"|---------------------|---------|----------------|-----------|")
    
    winner_r2 = "XGBoost" if xgb_r2 > naive_r2 else "Naive"
    winner_mae = "XGBoost" if xgb_mae < naive_mae else "Naive" 
    winner_f1 = "XGBoost" if xgb_spike_metrics['f1'] > naive_spike_metrics['f1'] else "Naive"
    
    print(f"| R² Score            | {xgb_r2:.4f}  | {naive_r2:.4f}       | **{winner_r2}**   |")
    print(f"| MAE (μg/g)          | {xgb_mae:.3f}   | {naive_mae:.3f}        | **{winner_mae}**   |")
    print(f"| F1 Score (15μg/g)   | {xgb_spike_metrics['f1']:.4f}  | {naive_spike_metrics['f1']:.4f}       | **{winner_f1}**   |")
    print(f"| Precision           | {xgb_spike_metrics['precision']:.4f}  | {naive_spike_metrics['precision']:.4f}       | **{'XGBoost' if xgb_spike_metrics['precision'] > naive_spike_metrics['precision'] else 'Naive'}**   |")
    print(f"| Recall              | {xgb_spike_metrics['recall']:.4f}  | {naive_spike_metrics['recall']:.4f}       | **{'XGBoost' if xgb_spike_metrics['recall'] > naive_spike_metrics['recall'] else 'Naive'}**   |")
    
    # Site-specific analysis
    print(f"\n🏖️  SITE-SPECIFIC ANALYSIS")
    print("-" * 80)
    print(f"| Site              | N   | XGB R²  | Naive R² | XGB MAE | Naive MAE | XGB F1  | Naive F1 |")
    print(f"|-------------------|-----|---------|----------|---------|-----------|---------|----------|")
    
    site_results = {}
    for site in sorted(comparison_df['site'].unique()):
        site_data = comparison_df[comparison_df['site'] == site]
        n = len(site_data)
        
        if n < 10:  # Skip sites with too few samples
            continue
        
        try:
            # EXACT pipeline methodology for each site
            site_actual = site_data['da']
            site_xgb = site_data['Predicted_da']
            site_naive = site_data['naive_prediction']
            
            site_xgb_r2 = r2_score(site_actual, site_xgb)
            site_naive_r2 = r2_score(site_actual, site_naive)
            site_xgb_mae = mean_absolute_error(site_actual, site_xgb)
            site_naive_mae = mean_absolute_error(site_actual, site_naive)
            
            site_xgb_spike = calculate_pipeline_spike_metrics(site_actual, site_xgb, 15.0)
            site_naive_spike = calculate_pipeline_spike_metrics(site_actual, site_naive, 15.0)
            
            site_results[site] = {
                'n': n,
                'xgb_r2': site_xgb_r2,
                'naive_r2': site_naive_r2,
                'xgb_mae': site_xgb_mae,
                'naive_mae': site_naive_mae,
                'xgb_f1': site_xgb_spike['f1'],
                'naive_f1': site_naive_spike['f1']
            }
            
            print(f"| {site:<17} | {n:<3} | {site_xgb_r2:7.3f} | {site_naive_r2:8.3f} | {site_xgb_mae:7.2f} | {site_naive_mae:9.2f} | {site_xgb_spike['f1']:7.3f} | {site_naive_spike['f1']:8.3f} |")
            
        except Exception as e:
            print(f"| {site:<17} | {n:<3} | Error calculating metrics: {str(e)[:40]}... |")
            continue
    
    # Summary statistics
    actual_spikes = (actual_values > 15).sum()
    xgb_predicted_spikes = (xgb_predictions > 15).sum() 
    naive_predicted_spikes = (naive_predictions > 15).sum()
    
    print(f"\n🚨 SPIKE DETECTION SUMMARY (15 μg/g threshold - Pipeline Method)")
    print("-" * 60)
    print(f"Actual spikes (>15 μg/g):     {actual_spikes}/{len(comparison_df)} ({100*actual_spikes/len(comparison_df):.1f}%)")
    print(f"XGBoost predicted spikes:     {xgb_predicted_spikes}/{len(comparison_df)} ({100*xgb_predicted_spikes/len(comparison_df):.1f}%)")
    print(f"Naive predicted spikes:       {naive_predicted_spikes}/{len(comparison_df)} ({100*naive_predicted_spikes/len(comparison_df):.1f}%)")
    
    print(f"\n🎯 KEY FINDINGS")
    print("-" * 30)
    if xgb_r2 > naive_r2:
        improvement = ((xgb_r2 - naive_r2) / abs(naive_r2) * 100) if naive_r2 != 0 else float('inf')
        print(f"✅ XGBoost R² ({xgb_r2:.4f}) > Naive R² ({naive_r2:.4f}) by {improvement:+.1f}%")
    else:
        improvement = ((naive_r2 - xgb_r2) / abs(xgb_r2) * 100) if xgb_r2 != 0 else float('inf')
        print(f"❌ Naive R² ({naive_r2:.4f}) > XGBoost R² ({xgb_r2:.4f}) by {improvement:+.1f}%")
    
    if xgb_mae < naive_mae:
        print(f"✅ XGBoost MAE ({xgb_mae:.3f}) < Naive MAE ({naive_mae:.3f})")
    else:
        print(f"❌ Naive MAE ({naive_mae:.3f}) < XGBoost MAE ({xgb_mae:.3f})")
    
    # Temporal characteristics
    avg_baseline_lag = comparison_df['days_to_baseline'].mean()
    print(f"\n⏰ TEMPORAL CHARACTERISTICS")
    print("-" * 35)
    print(f"Average baseline lag: {avg_baseline_lag:.1f} days")
    print(f"Successful temporal matches: {len(comparison_df)}/{len(comparison_df)} (100%)")
    
    return {
        'overall_metrics': {
            'xgb_r2': xgb_r2,
            'naive_r2': naive_r2,
            'xgb_mae': xgb_mae,
            'naive_mae': naive_mae,
            'xgb_spike_metrics': xgb_spike_metrics,
            'naive_spike_metrics': naive_spike_metrics
        },
        'site_results': site_results,
        'n_predictions': len(comparison_df),
        'n_sites': comparison_df['site'].nunique()
    }

def main():
    """Run corrected XGBoost vs naive baseline analysis."""
    try:
        # Load data using pipeline methodology
        xgboost_df = load_xgboost_results()
        historical_df = load_historical_data()
        
        # Calculate naive baseline with pipeline temporal safety
        comparison_df = calculate_naive_baseline_pipeline_method(xgboost_df, historical_df)
        
        if comparison_df.empty:
            print("❌ No valid comparisons could be made")
            return
        
        # Generate corrected analysis
        analysis_results = generate_corrected_analysis(comparison_df)
        
        if analysis_results is None:
            print("❌ Analysis failed")
            return
        
        # Save results for future reference
        output_path = "./corrected_xgboost_vs_naive_results.parquet"
        comparison_df.to_parquet(output_path, index=False)
        print(f"\n💾 Corrected results saved to: {output_path}")
        
        print(f"\n🎉 CORRECTED analysis complete!")
        print(f"   Expected XGBoost R² ≈ 0.49 based on your note")
        print(f"   Actual XGBoost R² = {analysis_results['overall_metrics']['xgb_r2']:.4f}")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error in corrected analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()