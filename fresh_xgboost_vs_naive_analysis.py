#!/usr/bin/env python3
"""
Fresh XGBoost vs Naive Baseline Analysis
========================================

Rerun the XGBoost vs naive baseline comparison using the latest cached data
from the cache folder and updated final_output.parquet file.

This analysis maintains temporal integrity by only using historical data
available before each prediction anchor date.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score
import json
import os

def load_fresh_xgboost_results():
    """Load fresh XGBoost regression results from cache."""
    print("📊 Loading fresh XGBoost results from cache...")
    
    # Check if parquet exists, otherwise use JSON
    parquet_path = "./cache/retrospective/regression_xgboost.parquet"
    json_path = "./cache/retrospective/regression_xgboost.json"
    
    if os.path.exists(parquet_path):
        print(f"   Loading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"   Loaded {len(df)} XGBoost predictions")
        return df
    elif os.path.exists(json_path):
        print(f"   Loading from JSON: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        records = []
        for record in data.get('results', []):
            records.append({
                'date': record['date'],
                'site': record['site'],
                'anchor_date': record.get('anchor_date'),
                'da': record['da'],
                'Predicted_da': record['Predicted_da']
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df['anchor_date'] = pd.to_datetime(df['anchor_date']) if 'anchor_date' in df.columns else None
        print(f"   Loaded {len(df)} XGBoost predictions from JSON")
        return df
    else:
        raise FileNotFoundError("No XGBoost results found in cache folder")

def load_historical_data():
    """Load historical data for naive baseline calculation."""
    print("📈 Loading historical data from final_output.parquet...")
    
    final_output_path = "./data/processed/final_output.parquet"
    if not os.path.exists(final_output_path):
        final_output_path = "./final_output.parquet"  # Alternative location
    
    if os.path.exists(final_output_path):
        df = pd.read_parquet(final_output_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['site', 'date'])
        print(f"   Loaded {len(df)} historical records")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   Sites: {sorted(df['site'].unique())}")
        return df
    else:
        raise FileNotFoundError(f"Historical data not found at {final_output_path}")

def calculate_naive_baseline_with_temporal_safety(xgboost_df, historical_df):
    """
    Calculate naive baseline predictions using only historical data available
    before each anchor date to maintain temporal integrity.
    """
    print("🔄 Calculating naive baseline with temporal safeguards...")
    
    results = []
    successful_matches = 0
    total_predictions = len(xgboost_df)
    
    for idx, row in xgboost_df.iterrows():
        site = row['site']
        prediction_date = pd.to_datetime(row['date'])
        anchor_date = pd.to_datetime(row['anchor_date']) if pd.notna(row.get('anchor_date')) else prediction_date - timedelta(days=7)
        actual_da = row['da']
        
        # TEMPORAL SAFETY: Only use historical data before anchor date
        site_historical = historical_df[
            (historical_df['site'] == site) & 
            (historical_df['date'] < anchor_date)  # Strict temporal cutoff
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
        
        if naive_prediction is not None and pd.notna(actual_da):
            results.append({
                'site': site,
                'date': prediction_date,
                'anchor_date': anchor_date,
                'actual_da': actual_da,
                'xgb_prediction': row['Predicted_da'],
                'naive_prediction': naive_prediction,
                'baseline_date_used': baseline_date_used,
                'days_to_baseline': (anchor_date - baseline_date_used).days if baseline_date_used else None
            })
    
    print(f"   Successfully matched {successful_matches}/{total_predictions} predictions ({100*successful_matches/total_predictions:.1f}%)")
    return pd.DataFrame(results)

def calculate_spike_detection_metrics(actual, predicted, threshold=20.0):
    """Calculate spike detection metrics for DA concentrations."""
    actual_binary = (actual > threshold).astype(int)
    pred_binary = (predicted > threshold).astype(int)
    
    return {
        'f1': f1_score(actual_binary, pred_binary, zero_division=0),
        'precision': precision_score(actual_binary, pred_binary, zero_division=0),
        'recall': recall_score(actual_binary, pred_binary, zero_division=0),
        'accuracy': accuracy_score(actual_binary, pred_binary),
        'spike_rate': actual_binary.mean()
    }

def generate_comprehensive_analysis(comparison_df):
    """Generate comprehensive analysis comparing XGBoost vs naive baseline."""
    print("\n" + "="*60)
    print("🔬 FRESH XGBOOST vs NAIVE BASELINE ANALYSIS")
    print("="*60)
    print(f"Analysis Date: {datetime.now().strftime('%B %Y')}")
    print(f"Dataset: {len(comparison_df)} predictions across {comparison_df['site'].nunique()} sites")
    print(f"Date Range: {comparison_df['date'].min().date()} to {comparison_df['date'].max().date()}")
    
    # Overall performance metrics
    print(f"\n📊 OVERALL PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Regression metrics
    xgb_r2 = r2_score(comparison_df['actual_da'], comparison_df['xgb_prediction'])
    naive_r2 = r2_score(comparison_df['actual_da'], comparison_df['naive_prediction'])
    
    xgb_mae = mean_absolute_error(comparison_df['actual_da'], comparison_df['xgb_prediction'])
    naive_mae = mean_absolute_error(comparison_df['actual_da'], comparison_df['naive_prediction'])
    
    # Spike detection metrics (20 μg/g threshold)
    xgb_spike_metrics = calculate_spike_detection_metrics(comparison_df['actual_da'], comparison_df['xgb_prediction'], 20.0)
    naive_spike_metrics = calculate_spike_detection_metrics(comparison_df['actual_da'], comparison_df['naive_prediction'], 20.0)
    
    print(f"| Metric              | XGBoost | Naive Baseline | Improvement |")
    print(f"|---------------------|---------|----------------|-------------|")
    print(f"| R² Score            | {xgb_r2:.4f}  | **{naive_r2:.4f}**     | {((naive_r2/xgb_r2-1)*100):+.1f}%      |")
    print(f"| MAE (μg/g)          | {xgb_mae:.2f}    | **{naive_mae:.2f}**       | {((xgb_mae/naive_mae-1)*100):+.1f}%      |")
    print(f"| F1 Score            | {xgb_spike_metrics['f1']:.4f}  | **{naive_spike_metrics['f1']:.4f}**     | {((naive_spike_metrics['f1']/xgb_spike_metrics['f1']-1)*100):+.1f}%      |")
    print(f"| Precision           | {xgb_spike_metrics['precision']:.4f}  | **{naive_spike_metrics['precision']:.4f}**     | {((naive_spike_metrics['precision']/xgb_spike_metrics['precision']-1)*100):+.1f}%      |")
    print(f"| Recall              | {xgb_spike_metrics['recall']:.4f}  | **{naive_spike_metrics['recall']:.4f}**     | {((naive_spike_metrics['recall']/xgb_spike_metrics['recall']-1)*100):+.1f}%      |")
    
    # Site-specific analysis
    print(f"\n🏖️  SITE-SPECIFIC PERFORMANCE")
    print("-" * 80)
    print(f"| Site              | N   | XGB R² | Naive R² | XGB MAE | Naive MAE | XGB F1 | Naive F1 | Spike Rate |")
    print(f"|-------------------|-----|--------|----------|---------|-----------|--------|----------|------------|")
    
    for site in sorted(comparison_df['site'].unique()):
        site_data = comparison_df[comparison_df['site'] == site]
        n = len(site_data)
        
        if n < 5:  # Skip sites with too few samples
            continue
        
        # Regression metrics
        try:
            site_xgb_r2 = r2_score(site_data['actual_da'], site_data['xgb_prediction'])
            site_naive_r2 = r2_score(site_data['actual_da'], site_data['naive_prediction'])
            site_xgb_mae = mean_absolute_error(site_data['actual_da'], site_data['xgb_prediction'])
            site_naive_mae = mean_absolute_error(site_data['actual_da'], site_data['naive_prediction'])
        except:
            continue
        
        # Spike detection metrics
        site_xgb_spike = calculate_spike_detection_metrics(site_data['actual_da'], site_data['xgb_prediction'], 20.0)
        site_naive_spike = calculate_spike_detection_metrics(site_data['actual_da'], site_data['naive_prediction'], 20.0)
        
        print(f"| {site:<17} | {n:<3} | {site_xgb_r2:.3f}  | **{site_naive_r2:.3f}**  | {site_xgb_mae:.2f}   | **{site_naive_mae:.2f}**    | {site_xgb_spike['f1']:.3f} | **{site_naive_spike['f1']:.3f}** | {site_naive_spike['spike_rate']*100:.1f}%     |")
    
    # Spike detection analysis
    print(f"\n🚨 SPIKE DETECTION ANALYSIS (>20 μg/g)")
    print("-" * 50)
    actual_spikes = (comparison_df['actual_da'] > 20).sum()
    xgb_predicted_spikes = (comparison_df['xgb_prediction'] > 20).sum()
    naive_predicted_spikes = (comparison_df['naive_prediction'] > 20).sum()
    
    print(f"Actual spikes (>20 μg/g):     {actual_spikes}/{len(comparison_df)} ({100*actual_spikes/len(comparison_df):.1f}%)")
    print(f"XGBoost predicted spikes:     {xgb_predicted_spikes}/{len(comparison_df)} ({100*xgb_predicted_spikes/len(comparison_df):.1f}%)")
    print(f"Naive predicted spikes:       {naive_predicted_spikes}/{len(comparison_df)} ({100*naive_predicted_spikes/len(comparison_df):.1f}%)")
    
    print(f"\nXGBoost Detection:  {xgb_spike_metrics['precision']:.1%} precision, {xgb_spike_metrics['recall']:.1%} recall")
    print(f"Naive Detection:    {naive_spike_metrics['precision']:.1%} precision, {naive_spike_metrics['recall']:.1%} recall")
    
    # Key insights
    print(f"\n💡 KEY FINDINGS")
    print("-" * 30)
    if naive_r2 > xgb_r2:
        print(f"✅ Naive baseline OUTPERFORMS XGBoost by {((naive_r2/xgb_r2-1)*100):.1f}% in R² score")
    else:
        print(f"❌ XGBoost outperforms naive baseline by {((xgb_r2/naive_r2-1)*100):.1f}% in R² score")
    
    if naive_mae < xgb_mae:
        print(f"✅ Naive baseline has {((xgb_mae/naive_mae-1)*100):.1f}% lower MAE than XGBoost")
    else:
        print(f"❌ XGBoost has {((naive_mae/xgb_mae-1)*100):.1f}% lower MAE than naive baseline")
    
    # Temporal persistence insights
    avg_baseline_lag = comparison_df['days_to_baseline'].mean()
    print(f"\n⏰ TEMPORAL CHARACTERISTICS")
    print("-" * 35)
    print(f"Average baseline lag: {avg_baseline_lag:.1f} days")
    print(f"Successful temporal matches: {len(comparison_df)} predictions")
    
    return {
        'overall_metrics': {
            'xgb_r2': xgb_r2,
            'naive_r2': naive_r2,
            'xgb_mae': xgb_mae,
            'naive_mae': naive_mae,
            'xgb_spike_metrics': xgb_spike_metrics,
            'naive_spike_metrics': naive_spike_metrics
        },
        'n_predictions': len(comparison_df),
        'n_sites': comparison_df['site'].nunique()
    }

def main():
    """Run fresh XGBoost vs naive baseline analysis."""
    try:
        # Load fresh data
        xgboost_df = load_fresh_xgboost_results()
        historical_df = load_historical_data()
        
        # Calculate naive baseline with temporal safety
        comparison_df = calculate_naive_baseline_with_temporal_safety(xgboost_df, historical_df)
        
        if comparison_df.empty:
            print("❌ No valid comparisons could be made")
            return
        
        # Generate comprehensive analysis
        analysis_results = generate_comprehensive_analysis(comparison_df)
        
        # Save results for future reference
        output_path = "./fresh_xgboost_vs_naive_results.parquet"
        comparison_df.to_parquet(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        print(f"\n🎉 Fresh analysis complete!")
        print(f"   Analyzed {analysis_results['n_predictions']} predictions across {analysis_results['n_sites']} sites")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()