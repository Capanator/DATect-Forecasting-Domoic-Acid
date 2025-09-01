#!/usr/bin/env python3
"""
Analyze Naive Baseline Failures
===============================

Study exactly where and why the naive baseline fails to create targeted improvements.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Add forecasting module to path
sys.path.insert(0, str(Path(__file__).parent / "forecasting"))

from forecasting.data_processor import DataProcessor
import config


class NaiveFailureAnalyzer:
    """
    Analyze exactly where naive baseline fails to find targeted improvements.
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.spike_threshold = 15.0
        
    def analyze_naive_failures(self):
        """
        Comprehensive analysis of naive baseline failures.
        """
        print("Loading data for failure analysis...")
        data = self.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        
        # Create naive predictions for all data
        all_results = []
        
        for site in data['site'].unique():
            print(f"Analyzing {site}...")
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date')
            
            for i in range(7, len(site_data) - 1):  # Need at least 7 days history
                current_row = site_data.iloc[i]
                future_row = site_data.iloc[i + 1]
                
                if pd.isna(current_row['da']) or pd.isna(future_row['da']):
                    continue
                
                # Naive prediction: use value from 7 days ago
                week_ago_idx = max(0, i - 7)
                naive_pred = site_data.iloc[week_ago_idx]['da']
                
                actual_da = future_row['da']
                prediction_error = abs(actual_da - naive_pred)
                
                result = {
                    'site': site,
                    'date': future_row['date'],
                    'actual_da': actual_da,
                    'naive_prediction': naive_pred,
                    'prediction_error': prediction_error,
                    'is_spike': actual_da > self.spike_threshold,
                    'predicted_spike': naive_pred > self.spike_threshold,
                    'spike_miss': (actual_da > self.spike_threshold) and (naive_pred <= self.spike_threshold),
                    'false_spike': (actual_da <= self.spike_threshold) and (naive_pred > self.spike_threshold),
                    'month': future_row['date'].month,
                    'season': self.get_season(future_row['date'].month),
                }
                
                # Add environmental context
                for env_var in ['sst', 'chlor_a', 'beuti', 'pdo', 'oni']:
                    if env_var in current_row:
                        result[f'current_{env_var}'] = current_row[env_var]
                
                all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        print(f"\nAnalysis completed: {len(results_df)} predictions")
        return self.identify_failure_patterns(results_df)
    
    def get_season(self, month):
        """Get season from month."""
        if month in [6, 7, 8]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
    
    def identify_failure_patterns(self, results_df):
        """
        Identify specific patterns where naive baseline fails.
        """
        print("\n" + "="*60)
        print("NAIVE BASELINE FAILURE ANALYSIS")
        print("="*60)
        
        # Overall performance
        overall_r2 = r2_score(results_df['actual_da'], results_df['naive_prediction'])
        overall_mae = mean_absolute_error(results_df['actual_da'], results_df['naive_prediction'])
        
        print(f"Overall Naive Performance: R² = {overall_r2:.4f}, MAE = {overall_mae:.2f}")
        
        # Identify high error cases
        high_error_threshold = results_df['prediction_error'].quantile(0.9)  # Top 10% errors
        high_error_cases = results_df[results_df['prediction_error'] > high_error_threshold]
        
        print(f"\nHigh Error Cases (>{high_error_threshold:.1f} error):")
        print(f"  Count: {len(high_error_cases)}/{len(results_df)} ({len(high_error_cases)/len(results_df)*100:.1f}%)")
        print(f"  Average error: {high_error_cases['prediction_error'].mean():.2f}")
        
        # Analyze spike detection failures
        spike_misses = results_df[results_df['spike_miss']]
        false_spikes = results_df[results_df['false_spike']]
        
        print(f"\nSpike Detection Analysis:")
        print(f"  Total actual spikes: {results_df['is_spike'].sum()}")
        print(f"  Missed spikes: {len(spike_misses)} ({len(spike_misses)/results_df['is_spike'].sum()*100:.1f}% of spikes)")
        print(f"  False spike alarms: {len(false_spikes)}")
        
        # Site-specific analysis
        print(f"\nSite-specific Failure Rates:")
        for site in results_df['site'].unique():
            site_data = results_df[results_df['site'] == site]
            site_high_error = len(site_data[site_data['prediction_error'] > high_error_threshold])
            site_spike_misses = len(site_data[site_data['spike_miss']])
            
            print(f"  {site}: {site_high_error/len(site_data)*100:.1f}% high errors, {site_spike_misses} spike misses")
        
        # Seasonal analysis
        print(f"\nSeasonal Failure Patterns:")
        for season in ['spring', 'summer', 'fall', 'winter']:
            season_data = results_df[results_df['season'] == season]
            if len(season_data) > 0:
                season_mae = mean_absolute_error(season_data['actual_da'], season_data['naive_prediction'])
                season_misses = len(season_data[season_data['spike_miss']])
                print(f"  {season}: MAE = {season_mae:.2f}, {season_misses} spike misses")
        
        # Environmental context analysis
        print(f"\nEnvironmental Context of Failures:")
        if len(high_error_cases) > 0:
            for env_var in ['current_sst', 'current_chlor_a', 'current_beuti']:
                if env_var in high_error_cases.columns:
                    env_vals = high_error_cases[env_var].dropna()
                    if len(env_vals) > 0:
                        print(f"  {env_var} during high errors: mean = {env_vals.mean():.3f}, std = {env_vals.std():.3f}")
        
        return {
            'all_results': results_df,
            'high_error_cases': high_error_cases,
            'spike_misses': spike_misses,
            'false_spikes': false_spikes,
            'high_error_threshold': high_error_threshold
        }


def main():
    analyzer = NaiveFailureAnalyzer()
    failure_analysis = analyzer.analyze_naive_failures()
    
    # Save results for next iteration
    os.makedirs("cache", exist_ok=True)
    failure_analysis['all_results'].to_parquet("cache/naive_failure_analysis.parquet")
    failure_analysis['high_error_cases'].to_parquet("cache/high_error_cases.parquet")
    
    print(f"\nFailure analysis saved to cache/")
    return failure_analysis


if __name__ == "__main__":
    results = main()