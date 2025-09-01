#!/usr/bin/env python3
"""
Simple Spike Detection Test - Focus on beating naive baseline
============================================================

Simplified approach to beat the naive baseline with robust models
and proper data type handling.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# Add forecasting module to path
sys.path.insert(0, str(Path(__file__).parent / "forecasting"))

from forecasting.data_processor import DataProcessor
import config


class SimpleSpikeTester:
    """
    Simple spike detection focused on beating naive baseline with robust methods.
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.spike_threshold = 15.0
        self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
        
    def create_simple_features(self, df, cutoff_date):
        """
        Create simple, robust features that work reliably.
        """
        df = df.copy()
        df = df.sort_values(['site', 'date'])
        
        # Basic lag features that we know work
        basic_lags = [1, 7, 14]
        df_with_lags = self.data_processor.create_lag_features_safe(
            df, 'site', 'da', basic_lags, cutoff_date
        )
        
        # Ensure all DA columns are float
        for lag in basic_lags:
            lag_col = f'da_lag_{lag}'
            if lag_col in df_with_lags.columns:
                df_with_lags[lag_col] = pd.to_numeric(df_with_lags[lag_col], errors='coerce')
        
        # Simple moving averages
        if 'da_lag_1' in df_with_lags.columns and 'da_lag_7' in df_with_lags.columns:
            df_with_lags['da_ma_7'] = (df_with_lags['da_lag_1'] + df_with_lags['da_lag_7']) / 2.0
        
        # Basic seasonal indicator
        df_with_lags['month'] = df_with_lags['date'].dt.month
        df_with_lags['is_summer'] = df_with_lags['month'].isin([6, 7, 8]).astype(int)
        
        # Clean up - keep only numeric features
        numeric_cols = df_with_lags.select_dtypes(include=[np.number]).columns
        essential_cols = ['date', 'site', 'da', 'month']
        
        keep_cols = list(numeric_cols) + [col for col in essential_cols if col in df_with_lags.columns]
        df_clean = df_with_lags[keep_cols].copy()
        
        return df_clean
    
    def create_naive_baseline(self, df, anchor_date):
        """
        Create naive baseline using 1-week lag.
        """
        site_data = df[df['date'] <= anchor_date].copy()
        if site_data.empty:
            return np.nan
            
        # Use 1 week ago value
        one_week_ago = anchor_date - pd.Timedelta(days=7)
        week_ago_data = site_data[site_data['date'] <= one_week_ago]
        
        if week_ago_data.empty:
            recent_data = site_data.sort_values('date').iloc[-1]
            da_val = recent_data['da']
        else:
            closest_data = week_ago_data.sort_values('date').iloc[-1]
            da_val = closest_data['da']
        
        return float(da_val) if not pd.isna(da_val) else np.nan
    
    def test_simple_models(self, X_train, y_train, X_test):
        """
        Test simple, robust models.
        """
        results = {}
        
        # Ensure all data is numeric and properly shaped
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        
        # Handle y_train properly
        if hasattr(y_train, 'values'):
            y_train_vals = y_train.values
        else:
            y_train_vals = y_train
        
        # Ensure y_train is 1D
        if isinstance(y_train_vals, np.ndarray) and y_train_vals.ndim > 1:
            y_train_vals = y_train_vals.flatten()
        
        y_train_series = pd.Series(y_train_vals).astype(float)
        
        # Fill NaN values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Simple Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
            rf_model.fit(X_train, y_train_series)
            rf_pred = rf_model.predict(X_test)[0]
            results['random_forest'] = max(0.0, float(rf_pred))
        except Exception as e:
            print(f"Random Forest failed: {e}")
            results['random_forest'] = np.nan
        
        # Simple Ridge Regression
        try:
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_train, y_train_series)
            ridge_pred = ridge_model.predict(X_test)[0]
            results['ridge'] = max(0.0, float(ridge_pred))
        except Exception as e:
            print(f"Ridge failed: {e}")
            results['ridge'] = np.nan
        
        # Simple average of lag features (enhanced naive)
        if 'da_lag_1' in X_test.columns and 'da_lag_7' in X_test.columns:
            lag1 = X_test['da_lag_1'].iloc[0]
            lag7 = X_test['da_lag_7'].iloc[0]
            if not pd.isna(lag1) and not pd.isna(lag7):
                results['enhanced_naive'] = float((lag1 * 0.7 + lag7 * 0.3))
            else:
                results['enhanced_naive'] = np.nan
        else:
            results['enhanced_naive'] = np.nan
        
        return results
    
    def run_simple_evaluation(self, n_anchors=50):
        """
        Run simple evaluation focused on beating naive baseline.
        """
        print("Loading data...")
        data = self.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        
        all_results = []
        
        # Sample sites and dates
        sites = data['site'].unique()
        
        for site in sites[:3]:  # Test on first 3 sites
            print(f"Processing {site}...")
            site_data = data[data['site'] == site].copy()
            site_dates = site_data['date'].sort_values().unique()
            
            if len(site_dates) > 50:
                # Sample some dates for testing
                test_dates = np.random.choice(site_dates[30:-10], min(n_anchors//len(sites[:3]), 20), replace=False)
                
                for test_date in test_dates:
                    anchor_date = test_date - pd.Timedelta(days=self.temporal_buffer_days)
                    
                    # Get training data
                    train_mask = site_data['date'] <= anchor_date
                    train_data = site_data[train_mask].copy()
                    
                    if len(train_data) < 20:  # Need sufficient training data
                        continue
                    
                    # Get test data
                    test_row = site_data[site_data['date'] == test_date]
                    if test_row.empty or pd.isna(test_row['da'].iloc[0]):
                        continue
                    
                    actual_da = float(test_row['da'].iloc[0])
                    
                    # Create features
                    site_data_enhanced = self.create_simple_features(site_data, anchor_date)
                    
                    train_enhanced = site_data_enhanced[site_data_enhanced['date'] <= anchor_date].copy()
                    test_enhanced = site_data_enhanced[site_data_enhanced['date'] == test_date].copy()
                    
                    if train_enhanced.empty or test_enhanced.empty:
                        continue
                    
                    train_clean = train_enhanced.dropna(subset=['da'])
                    if len(train_clean) < 10:
                        continue
                    
                    # Prepare features
                    drop_cols = ['date', 'site', 'da', 'month']
                    X_train = train_clean.drop(columns=[col for col in drop_cols if col in train_clean.columns])
                    y_train = train_clean['da']
                    X_test = test_enhanced.drop(columns=[col for col in drop_cols if col in test_enhanced.columns])
                    
                    # Align columns
                    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
                    
                    # Get naive baseline
                    naive_pred = self.create_naive_baseline(site_data, anchor_date)
                    
                    # Test models
                    model_results = self.test_simple_models(X_train, y_train, X_test)
                    
                    # Store results
                    result = {
                        'site': site,
                        'date': test_date,
                        'actual_da': actual_da,
                        'naive_baseline': naive_pred
                    }
                    result.update(model_results)
                    all_results.append(result)
        
        if not all_results:
            print("No results generated!")
            return None
        
        # Convert to DataFrame and evaluate
        results_df = pd.DataFrame(all_results)
        self.evaluate_results(results_df)
        
        return results_df
    
    def evaluate_results(self, results_df):
        """
        Evaluate results against naive baseline.
        """
        print("\n" + "="*50)
        print("SIMPLE SPIKE DETECTION RESULTS")
        print("="*50)
        
        # Get metrics for each model
        models = ['naive_baseline', 'random_forest', 'ridge', 'enhanced_naive']
        
        for model in models:
            if model not in results_df.columns:
                continue
                
            valid_mask = ~(results_df['actual_da'].isna() | results_df[model].isna())
            if not valid_mask.any():
                continue
                
            valid_data = results_df[valid_mask]
            
            # Regression metrics
            r2 = r2_score(valid_data['actual_da'], valid_data[model])
            mae = mean_absolute_error(valid_data['actual_da'], valid_data[model])
            
            # Spike detection metrics (15 ppm)
            y_true_spike = (valid_data['actual_da'] > self.spike_threshold).astype(int)
            y_pred_spike = (valid_data[model] > self.spike_threshold).astype(int)
            
            precision = precision_score(y_true_spike, y_pred_spike, zero_division=0)
            recall = recall_score(y_true_spike, y_pred_spike, zero_division=0)
            f1 = f1_score(y_true_spike, y_pred_spike, zero_division=0)
            
            n_false_positives = ((y_pred_spike == 1) & (y_true_spike == 0)).sum()
            
            print(f"\n{model.upper()}:")
            print(f"  R² = {r2:.4f}, MAE = {mae:.2f}")
            print(f"  Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
            print(f"  False Positives = {n_false_positives}/{len(valid_data)} ({n_false_positives/len(valid_data)*100:.1f}%)")
        
        # Compare to naive baseline
        naive_r2 = r2_score(results_df.dropna()['actual_da'], results_df.dropna()['naive_baseline'])
        naive_f1 = f1_score((results_df.dropna()['actual_da'] > self.spike_threshold).astype(int), 
                           (results_df.dropna()['naive_baseline'] > self.spike_threshold).astype(int), zero_division=0)
        
        print(f"\n" + "="*30)
        print("BASELINE COMPARISON:")
        print("="*30)
        
        for model in ['random_forest', 'ridge', 'enhanced_naive']:
            if model not in results_df.columns:
                continue
                
            valid = results_df.dropna(subset=['actual_da', model, 'naive_baseline'])
            if valid.empty:
                continue
                
            model_r2 = r2_score(valid['actual_da'], valid[model])
            model_f1 = f1_score((valid['actual_da'] > self.spike_threshold).astype(int),
                               (valid[model] > self.spike_threshold).astype(int), zero_division=0)
            
            r2_improvement = (model_r2 - naive_r2) / max(naive_r2, 0.001) * 100
            f1_improvement = (model_f1 - naive_f1) / max(naive_f1, 0.001) * 100
            
            beats_baseline = model_r2 > naive_r2 and model_f1 > naive_f1
            status = "✓ BEATS BASELINE" if beats_baseline else "✗ Below baseline"
            
            print(f"{model}: {status}")
            print(f"  R² improvement: {r2_improvement:+.1f}%")
            print(f"  F1 improvement: {f1_improvement:+.1f}%")


def main():
    print("Simple Spike Detection Test - Focused on beating naive baseline")
    print("Using 15 ppm threshold with robust data handling")
    print()
    
    tester = SimpleSpikeTester()
    results = tester.run_simple_evaluation(n_anchors=60)
    
    if results is not None:
        print(f"\nTest completed: {len(results)} predictions")
        return 0
    else:
        print("Test failed - no results")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)