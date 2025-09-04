#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Tuning Script for DATect Forecasting System
========================================================================

This script systematically tests various XGBoost hyperparameters, spike weighting factors,
and other optimization parameters to find the best settings for domoic acid forecasting.

It evaluates performance across multiple sites with 100 anchor points each, comparing
against linear and naive baselines to find significantly better configurations.
"""

import pandas as pd
import numpy as np
import itertools
import warnings
import time
import os
import json
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Import from the forecasting system
from forecasting.data_processor import DataProcessor
from forecasting.model_factory import ModelFactory
import config

warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Comprehensive hyperparameter tuning for XGBoost regression models."""
    
    def __init__(self, n_anchors_per_site=100):
        self.n_anchors_per_site = n_anchors_per_site
        self.data_processor = DataProcessor()
        self.results = []
        
        # Load data once
        print("Loading data...")
        data_path = "./data/processed/final_output.parquet"
        self.data = self.data_processor.load_and_prepare_base_data(data_path)
        print(f"Loaded {len(self.data)} rows of data")
        
        # Get available sites
        self.sites = sorted(self.data['site'].unique())
        print(f"Available sites: {self.sites}")
        
    def define_parameter_grid(self):
        """Define comprehensive parameter grid for testing."""
        
        # XGBoost hyperparameters - extensive range from basic to very high-end
        xgb_params = {
            'n_estimators': [50, 100, 200, 300, 500, 800, 1000, 1200, 1500],
            'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'reg_alpha': [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            'reg_lambda': [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6],
            'gamma': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        }
        
        # Spike weighting factors - REDUCED for speed
        spike_weights = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0]  # Reduced from 12 to 6 values
        
        # Spike thresholds - REDUCED for speed
        spike_thresholds = [15.0, 20.0, 25.0]  # Reduced from 4 to 3 values
        
        # Bootstrap parameters
        bootstrap_samples = [10, 20, 30, 50, 100]
        
        # Generate strategic parameter combinations
        # Instead of full grid search (which would be millions of combinations),
        # we'll use a strategic approach with different complexity levels
        
        param_combinations = []
        
        # 1. Basic configurations (fast models)
        basic_configs = [
            {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr}
            for n_est in [50, 100, 200]
            for depth in [3, 4, 5, 6]
            for lr in [0.1, 0.15, 0.2]
        ]
        
        # 2. Medium configurations (balanced performance/speed) - REDUCED
        medium_configs = [
            {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
             'subsample': sub, 'colsample_bytree': col}
            for n_est in [300, 500, 800]
            for depth in [6, 8]  # Reduced from [6, 7, 8]
            for lr in [0.05, 0.08, 0.1]
            for sub in [0.8, 0.9]
            for col in [0.8, 0.9]
        ]
        
        # 3. High-end configurations (maximum performance) - REDUCED
        high_configs = [
            {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
             'subsample': sub, 'colsample_bytree': col, 'reg_alpha': alpha, 'reg_lambda': lam}
            for n_est in [1000, 1500]  # Reduced from [1000, 1200, 1500]
            for depth in [8, 10]  # Reduced from [8, 10, 12]
            for lr in [0.03, 0.05, 0.08]
            for sub in [0.8, 0.9]  # Reduced from [0.8, 0.85, 0.9]
            for col in [0.8, 0.9]  # Reduced from [0.8, 0.85, 0.9]
            for alpha in [0.0, 0.5]  # Reduced from [0.0, 0.5, 1.0]
            for lam in [0.0, 0.5]  # Reduced from [0.0, 0.5, 1.0]
        ]
        
        # 4. Regularization-focused configurations - REDUCED
        reg_configs = [
            {'n_estimators': 800, 'max_depth': depth, 'learning_rate': 0.08,
             'reg_alpha': alpha, 'reg_lambda': lam, 'min_child_weight': mcw, 'gamma': gamma}
            for depth in [6, 8]  # Reduced from [6, 8, 10]
            for alpha in [0.5, 1.0, 1.5]  # Reduced from [0.5, 1.0, 1.5, 2.0]
            for lam in [0.5, 1.0, 1.5]  # Reduced from [0.5, 1.0, 1.5, 2.0]
            for mcw in [1, 3]  # Reduced from [1, 3, 5]
            for gamma in [0.0, 0.3]  # Reduced from [0.0, 0.3, 0.5]
        ]
        
        # Combine all configurations
        all_xgb_configs = basic_configs + medium_configs + high_configs + reg_configs
        
        # Ensure we don't have duplicates
        unique_configs = []
        seen = set()
        for config in all_xgb_configs:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)
        
        print(f"Generated {len(unique_configs)} unique XGBoost configurations")
        
        # Now combine with spike weighting and other parameters
        for xgb_config in unique_configs:
            for spike_weight in spike_weights:
                for spike_threshold in spike_thresholds:
                    param_combinations.append({
                        **xgb_config,
                        'spike_weight': spike_weight,
                        'spike_threshold': spike_threshold,
                        'bootstrap_samples': 20  # Fixed for consistency
                    })
        
        print(f"Total parameter combinations: {len(param_combinations)}")
        return param_combinations
        
    def get_current_xgboost_baseline(self):
        """Get current XGBoost performance using EXACT same method as main pipeline."""
        print("\nEvaluating current XGBoost baseline using main pipeline method...")
        
        # Use the EXACT same method as precompute_cache.py and main pipeline
        from forecasting.forecast_engine import ForecastEngine
        engine = ForecastEngine()
        
        # Use same parameters as main pipeline
        n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 100)  # Use config value like main pipeline
        
        results_df = engine.run_retrospective_evaluation(
            task="regression",
            model_type="xgboost", 
            n_anchors=n_anchors,
            min_test_date="2008-01-01"
        )
        
        if results_df is not None and not results_df.empty:
            from sklearn.metrics import r2_score, mean_absolute_error, f1_score
            
            valid_results = results_df.dropna(subset=['actual_da', 'predicted_da'])
            if not valid_results.empty:
                r2 = r2_score(valid_results['actual_da'], valid_results['predicted_da'])
                mae = mean_absolute_error(valid_results['actual_da'], valid_results['predicted_da'])
                
                # Spike F1 (same as main pipeline)
                y_true_binary = (valid_results['actual_da'] > 20.0).astype(int)
                y_pred_binary = (valid_results['predicted_da'] > 20.0).astype(int)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                baseline_result = {
                    'mean_r2': r2,
                    'mean_mae': mae,
                    'mean_spike_f1': f1,
                    'n_predictions': len(valid_results)
                }
                
                print(f"Current XGBoost Baseline (Main Pipeline Method):")
                print(f"  R² = {r2:.4f}")
                print(f"  MAE = {mae:.4f}")
                print(f"  Spike F1 = {f1:.4f}")
                print(f"  Predictions = {len(valid_results)}")
                return baseline_result
            else:
                print("No valid results from main pipeline method")
                return None
        else:
            print("Main pipeline method returned no results")
            return None
        
    def filter_better_configs(self, all_results, baseline_result):
        """Filter configurations that beat the current baseline."""
        if not baseline_result:
            print("No baseline to compare against - keeping all results")
            return all_results
        
        baseline_r2 = baseline_result['mean_r2']
        baseline_mae = baseline_result['mean_mae'] 
        baseline_f1 = baseline_result['mean_spike_f1']
        
        print(f"\nFiltering configurations that beat baseline:")
        print(f"  Must beat: R²>{baseline_r2:.4f}, MAE<{baseline_mae:.4f}, F1>{baseline_f1:.4f}")
        
        better_configs = []
        for result in all_results:
            # Must beat baseline in at least 2 out of 3 key metrics
            beats_r2 = result['mean_r2'] > baseline_r2
            beats_mae = result['mean_mae'] < baseline_mae
            beats_f1 = result['mean_spike_f1'] > baseline_f1
            
            score_improvements = sum([beats_r2, beats_mae, beats_f1])
            
            # Keep configs that improve at least 2 metrics, or have significant R² improvement
            if score_improvements >= 2 or (beats_r2 and result['mean_r2'] > baseline_r2 + 0.05):
                better_configs.append(result)
        
        print(f"  Found {len(better_configs)} configurations better than baseline")
        return better_configs
    
    def _evaluate_xgboost_config(self, params):
        """Evaluate a specific XGBoost configuration."""
        
        def evaluate_site(site):
            """Evaluate XGBoost config on a single site."""
            site_data = self.data[self.data['site'] == site].copy()
            site_data.sort_values('date', inplace=True)
            
            anchor_infos = self._generate_anchor_points(site_data, site, self.n_anchors_per_site)
            
            # Parallel evaluation of anchor points for this site
            def evaluate_anchor(anchor_date):
                try:
                    return self._single_forecast_xgboost(site_data, anchor_date, site, params)
                except Exception:
                    return None, None
            
            results = Parallel(n_jobs=2, verbose=0)(  # Use 2 jobs per site to avoid overloading
                delayed(evaluate_anchor)(anchor_date) for anchor_date in anchor_infos
            )
            
            predictions = []
            actuals = []
            for pred, actual in results:
                if pred is not None and actual is not None:
                    predictions.append(pred)
                    actuals.append(actual)
            
            if predictions:
                metrics = self._calculate_metrics(actuals, predictions)
                metrics['site'] = site
                metrics['n_predictions'] = len(predictions)
                return metrics
            else:
                return None
        
        # Evaluate all sites for this parameter configuration
        site_results = []
        for site in self.sites:
            result = evaluate_site(site)
            if result:
                site_results.append(result)
        
        if site_results:
            df_results = pd.DataFrame(site_results)
            return {
                'model_type': 'xgboost',
                'params': params,
                'mean_r2': df_results['r2'].mean(),
                'mean_mae': df_results['mae'].mean(),
                'mean_spike_precision': df_results['spike_precision'].mean(),
                'mean_spike_recall': df_results['spike_recall'].mean(),
                'mean_spike_f1': df_results['spike_f1'].mean(),
                'site_results': site_results
            }
        else:
            return None
    
    def _generate_anchor_points(self, site_data, site, n_anchors):
        """Generate anchor points for evaluation."""
        min_test_date = pd.Timestamp("2008-01-01")  # Ensure enough history
        
        site_dates = site_data["date"].sort_values().unique()
        
        if len(site_dates) <= 1:
            return []
        
        valid_anchors = []
        for i, date in enumerate(site_dates[:-1]):
            if date >= min_test_date:
                future_dates = site_dates[i+1:]
                valid_future = [d for d in future_dates if (d - date).days >= config.FORECAST_HORIZON_DAYS]
                if valid_future:
                    valid_anchors.append(date)
        
        if not valid_anchors:
            return []
        
        n_sample = min(len(valid_anchors), n_anchors)
        return np.random.choice(valid_anchors, n_sample, replace=False)
    
    def _optimize_xgboost_params(self, base_params):
        """Optimize XGBoost parameters for speed while maintaining accuracy."""
        optimized = base_params.copy()
        
        # Speed optimizations that don't hurt accuracy much
        optimized['tree_method'] = 'hist'  # Much faster tree construction
        optimized['max_bin'] = 256  # Reduce histogram bins for speed
        optimized['grow_policy'] = 'depthwise'  # More efficient growth
        
        # Use fewer iterations for very high n_estimators with early stopping simulation
        if optimized['n_estimators'] > 1000:
            # For very high n_estimators, we can often get similar results with fewer trees
            # This is a heuristic speedup
            optimized['n_estimators'] = min(optimized['n_estimators'], 1200)
        
        return optimized
    
    def _single_forecast_xgboost(self, site_data, anchor_date, site, params):
        """Single forecast using XGBoost with given parameters."""
        train_mask = site_data["date"] <= anchor_date
        test_candidates = site_data[site_data["date"] > anchor_date]
        
        train_df = site_data[train_mask].dropna(subset=['da']).copy()
        
        if len(train_df) < 5 or test_candidates.empty:
            return None, None
        
        # Create lag features if enabled
        if config.USE_LAG_FEATURES:
            site_data_with_lags = self.data_processor.create_lag_features_safe(
                site_data, "site", "da", config.LAG_FEATURES, anchor_date
            )
            train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].dropna(subset=['da']).copy()
        
        # Find test sample
        target_date = anchor_date + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        test_candidates = test_candidates.copy()
        test_candidates['date_diff'] = abs((test_candidates['date'] - target_date).dt.days)
        closest_idx = test_candidates['date_diff'].idxmin()
        test_df = test_candidates.loc[[closest_idx]].copy()
        
        actual_da = test_df['da'].iloc[0]
        if pd.isna(actual_da):
            return None, None
        
        # Prepare features
        drop_cols = ["date", "site", "da"]
        transformer, X_train = self.data_processor.create_numeric_transformer(train_df, drop_cols)
        
        if config.USE_LAG_FEATURES:
            # Apply lag features to test data too
            site_data_with_lags = self.data_processor.create_lag_features_safe(
                site_data, "site", "da", config.LAG_FEATURES, anchor_date
            )
            test_df = site_data_with_lags[site_data_with_lags["date"] == test_df['date'].iloc[0]].copy()
        
        X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Temporal safety validation
        self.data_processor.validate_transformer_temporal_safety(
            transformer, train_df, test_df, anchor_date
        )
        
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        # Prepare XGBoost parameters with speed optimizations
        base_xgb_params = {
            'n_estimators': params.get('n_estimators', 300),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.08),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'reg_lambda': params.get('reg_lambda', 0.0),
            'min_child_weight': params.get('min_child_weight', 1),
            'gamma': params.get('gamma', 0.0),
            'random_state': config.RANDOM_SEED,
            'n_jobs': 1  # Use 1 job per model since we're already parallelizing at higher level
        }
        
        # Apply speed optimizations
        xgb_params = self._optimize_xgboost_params(base_xgb_params)
        
        model = xgb.XGBRegressor(**xgb_params)
        
        # Apply spike weighting
        y_train = train_df["da"]
        spike_threshold = params.get('spike_threshold', 20.0)
        spike_weight = params.get('spike_weight', 8.0)
        
        spike_mask = y_train > spike_threshold
        sample_weights = np.ones(len(y_train))
        sample_weights[spike_mask] *= spike_weight
        
        model.fit(X_train_processed, y_train, sample_weight=sample_weights)
        
        prediction = model.predict(X_test_processed)[0]
        prediction = max(0.0, float(prediction))
        
        return prediction, actual_da
    
    def _calculate_metrics(self, actuals, predictions):
        """Calculate evaluation metrics."""
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        # Spike detection metrics
        y_true_binary = (np.array(actuals) > 20.0).astype(int)
        y_pred_binary = (np.array(predictions) > 20.0).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        return {
            'r2': r2,
            'mae': mae,
            'spike_precision': precision,
            'spike_recall': recall,
            'spike_f1': f1
        }
    
    def run_comprehensive_tuning(self):
        """Run XGBoost hyperparameter tuning to beat current baseline."""
        print("="*80)
        print("XGBOOST HYPERPARAMETER OPTIMIZATION FOR DATECT FORECASTING")
        print("="*80)
        print("Goal: Beat current XGBoost settings (R²=0.3203, MAE=6.84, F1=0.5796)")
        
        start_time = time.time()
        
        # Step 1: Evaluate current XGBoost baseline
        print("\n" + "="*50)
        print("STEP 1: CURRENT XGBOOST BASELINE")
        print("="*50)
        
        baseline_result = self.get_current_xgboost_baseline()
        if not baseline_result:
            print("ERROR: Could not establish baseline - continuing anyway")
            baseline_result = {
                'mean_r2': 0.3203,
                'mean_mae': 6.84, 
                'mean_spike_f1': 0.5796
            }
        
        # Step 2: Generate parameter combinations
        print("\n" + "="*50)
        print("STEP 2: PARAMETER GRID GENERATION")
        print("="*50)
        
        param_combinations = self.define_parameter_grid()
        
        # Step 3: Evaluate XGBoost configurations
        print("\n" + "="*50)
        print("STEP 3: XGBOOST HYPERPARAMETER EVALUATION")
        print("="*50)
        
        print(f"Testing {len(param_combinations)} XGBoost configurations...")
        print(f"Using {self.n_anchors_per_site} anchor points per site")
        print(f"Sites: {', '.join(self.sites)}")
        print(f"Using parallel processing with {os.cpu_count()} CPU cores")
        
        # Parallel evaluation with progress bar
        print(f"\nEvaluating {len(param_combinations)} parameter combinations...")
        
        def evaluate_single_config(params):
            """Wrapper function for parallel evaluation."""
            try:
                return self._evaluate_xgboost_config(params)
            except Exception as e:
                return None
        
        # Use parallel processing with progress bar
        all_results = []
        start_time = time.time()
        
        with tqdm(total=len(param_combinations), desc="Hyperparameter configs", 
                  unit="config", ncols=120, miniters=1) as pbar:
            
            # Process in batches for memory management
            batch_size = min(50, max(1, len(param_combinations) // 20))  # Adaptive batch size
            
            for i in range(0, len(param_combinations), batch_size):
                batch_start = time.time()
                batch = param_combinations[i:i+batch_size]
                
                # Parallel processing within batch
                batch_results = Parallel(n_jobs=-1, verbose=0)(
                    delayed(evaluate_single_config)(params) for params in batch
                )
                
                # Filter successful results
                successful_results = [r for r in batch_results if r is not None]
                all_results.extend(successful_results)
                
                # Calculate timing statistics
                batch_time = time.time() - batch_start
                total_processed = i + len(batch)
                success_rate = (len(all_results) / total_processed) * 100 if total_processed > 0 else 0
                
                # Estimate remaining time
                elapsed = time.time() - start_time
                configs_per_sec = total_processed / elapsed if elapsed > 0 else 0
                remaining_configs = len(param_combinations) - total_processed
                eta_seconds = remaining_configs / configs_per_sec if configs_per_sec > 0 else 0
                eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
                
                # Update progress bar with comprehensive info
                pbar.set_postfix({
                    'Batch': f"{len(successful_results)}/{len(batch)}",
                    'Total': len(all_results),
                    'Rate': f"{success_rate:.1f}%",
                    'Speed': f"{configs_per_sec:.1f}/s",
                    'ETA': eta_str
                }, refresh=True)
                pbar.update(len(batch))
        
        print(f"\nCompleted evaluation: {len(all_results)} successful configurations out of {len(param_combinations)} total")
        
        # Step 4: Filter configs that beat baseline
        print("\n" + "="*50)
        print("STEP 4: FILTERING BETTER CONFIGURATIONS")
        print("="*50)
        
        if not all_results:
            print("ERROR: No successful XGBoost configurations!")
            return
            
        # Filter configurations that beat the baseline
        better_configs = self.filter_better_configs(all_results, baseline_result)
        
        if not better_configs:
            print("❌ NO CONFIGURATIONS BEAT THE CURRENT BASELINE!")
            print("\nTop 5 configurations (even though they don't beat baseline):")
            all_results_sorted = sorted(all_results, key=lambda x: x['mean_r2'], reverse=True)
            for i, result in enumerate(all_results_sorted[:5]):
                params = result['params']
                print(f"{i+1}. R²={result['mean_r2']:.4f}, MAE={result['mean_mae']:.4f}, F1={result['mean_spike_f1']:.4f}")
                print(f"   n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']}")
            return None
        
        # Sort by performance metrics
        better_configs_sorted = sorted(better_configs, key=lambda x: x['mean_r2'], reverse=True)
        
        # Best by different metrics
        best_r2 = max(better_configs, key=lambda x: x['mean_r2'])
        best_mae = min(better_configs, key=lambda x: x['mean_mae'])
        best_spike_f1 = max(better_configs, key=lambda x: x['mean_spike_f1'])
        
        print(f"\n🎉 FOUND {len(better_configs)} CONFIGURATIONS THAT BEAT BASELINE!")
        print("="*60)
        
        # Compare with baseline
        baseline_r2 = baseline_result['mean_r2']
        baseline_mae = baseline_result['mean_mae'] 
        baseline_f1 = baseline_result['mean_spike_f1']
        
        print(f"\n🏆 BEST IMPROVEMENTS:")
        print("-" * 40)
        
        r2_improvement = ((best_r2['mean_r2'] - baseline_r2) / abs(baseline_r2)) * 100
        mae_improvement = ((baseline_mae - best_mae['mean_mae']) / baseline_mae) * 100
        f1_improvement = ((best_spike_f1['mean_spike_f1'] - baseline_f1) / baseline_f1) * 100
        
        print(f"\nBest R² Improvement: +{r2_improvement:.1f}% (from {baseline_r2:.4f} to {best_r2['mean_r2']:.4f})")
        print(f"Parameters: {best_r2['params']}")
        
        print(f"\nBest MAE Improvement: {mae_improvement:.1f}% (from {baseline_mae:.4f} to {best_mae['mean_mae']:.4f})")
        print(f"Parameters: {best_mae['params']}")
        
        print(f"\nBest Spike F1 Improvement: +{f1_improvement:.1f}% (from {baseline_f1:.4f} to {best_spike_f1['mean_spike_f1']:.4f})")
        print(f"Parameters: {best_spike_f1['params']}")
        
        # Top 10 configurations that beat baseline
        print(f"\nTOP 10 CONFIGURATIONS THAT BEAT BASELINE:")
        print("-" * 50)
        
        for i, result in enumerate(better_configs_sorted[:10]):
            params = result['params']
            r2_imp = ((result['mean_r2'] - baseline_r2) / abs(baseline_r2)) * 100
            mae_imp = ((baseline_mae - result['mean_mae']) / baseline_mae) * 100
            f1_imp = ((result['mean_spike_f1'] - baseline_f1) / baseline_f1) * 100
            
            print(f"{i+1:2d}. R²={result['mean_r2']:.4f}(+{r2_imp:+.1f}%) | MAE={result['mean_mae']:.4f}({mae_imp:+.1f}%) | F1={result['mean_spike_f1']:.4f}(+{f1_imp:+.1f}%)")
            print(f"    n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']}")
            print(f"    spike_w={params['spike_weight']}, spike_th={params['spike_threshold']}")
            
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"xgboost_optimization_results_{timestamp}.json"
        
        output_data = {
            'timestamp': timestamp,
            'n_anchors_per_site': self.n_anchors_per_site,
            'sites': self.sites,
            'baseline': baseline_result,
            'best_configs': {
                'best_r2': best_r2,
                'best_mae': best_mae,
                'best_spike_f1': best_spike_f1
            },
            'better_configs': better_configs_sorted,
            'total_configs_tested': len(all_results),
            'configs_better_than_baseline': len(better_configs),
            'runtime_seconds': time.time() - start_time
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        
        return output_data


def main():
    """Main execution function."""
    
    print("Initializing Hyperparameter Tuning System...")
    
    # Check if data file exists
    if not os.path.exists(config.FINAL_OUTPUT_PATH):
        print(f"ERROR: Data file not found at {config.FINAL_OUTPUT_PATH}")
        print("Please run dataset-creation.py first to generate the processed data.")
        return
    
    # Initialize tuner
    tuner = HyperparameterTuner(n_anchors_per_site=100)
    
    # Run comprehensive tuning
    results = tuner.run_comprehensive_tuning()
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    
    if results:
        best_config = results['best_configs']['best_r2']
        print(f"\nRECOMMENDED CONFIGURATION:")
        print(f"Model Factory Settings:")
        params = best_config['params']
        print(f"  n_estimators = {params['n_estimators']}")
        print(f"  max_depth = {params['max_depth']}")
        print(f"  learning_rate = {params['learning_rate']}")
        print(f"  subsample = {params.get('subsample', 0.8)}")
        print(f"  colsample_bytree = {params.get('colsample_bytree', 0.8)}")
        print(f"  reg_alpha = {params.get('reg_alpha', 0.0)}")
        print(f"  reg_lambda = {params.get('reg_lambda', 0.0)}")
        if 'min_child_weight' in params:
            print(f"  min_child_weight = {params['min_child_weight']}")
        if 'gamma' in params:
            print(f"  gamma = {params['gamma']}")
        
        print(f"\nSpike Weighting Settings:")
        print(f"  spike_threshold = {params['spike_threshold']}")
        print(f"  spike_weight = {params['spike_weight']}")
        
        print(f"\nExpected Performance:")
        print(f"  R² = {best_config['mean_r2']:.4f}")
        print(f"  MAE = {best_config['mean_mae']:.4f}")
        print(f"  Spike F1 = {best_config['mean_spike_f1']:.4f}")


if __name__ == "__main__":
    main()