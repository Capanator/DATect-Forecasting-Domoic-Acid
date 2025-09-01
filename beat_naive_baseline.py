#!/usr/bin/env python3
"""
Beat Naive Baseline Challenge
============================

Advanced approach to consistently beat the naive 1-week lag baseline
using sophisticated temporal modeling and environmental leading indicators.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# Add forecasting module to path
sys.path.insert(0, str(Path(__file__).parent / "forecasting"))

from forecasting.data_processor import DataProcessor
import config


class NaiveBaselineBeat:
    """
    Advanced system designed specifically to beat the naive 1-week lag baseline.
    
    Strategy:
    1. Multi-scale temporal features (1-28 day lags)
    2. Environmental leading indicators with proper lags
    3. Sophisticated ensemble with temporal weights
    4. Site-specific adaptations
    5. Two-stage prediction: trend + magnitude
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.spike_threshold = 15.0
        self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
    def create_advanced_temporal_features(self, df, cutoff_date):
        """
        Create advanced temporal features designed to capture patterns
        that simple 1-week lag misses.
        """
        df = df.copy()
        df = df.sort_values(['site', 'date'])
        
        # Multi-scale lag features (key to beating naive baseline)
        all_lags = [1, 2, 3, 7, 14, 21, 28]
        df_enhanced = self.data_processor.create_lag_features_safe(
            df, 'site', 'da', all_lags, cutoff_date
        )
        
        # Ensure all lag features are numeric
        for lag in all_lags:
            lag_col = f'da_lag_{lag}'
            if lag_col in df_enhanced.columns:
                df_enhanced[lag_col] = pd.to_numeric(df_enhanced[lag_col], errors='coerce')
        
        # Advanced temporal patterns that naive baseline can't capture
        
        # 1. Trend indicators (is DA increasing/decreasing?)
        if 'da_lag_1' in df_enhanced.columns and 'da_lag_7' in df_enhanced.columns:
            df_enhanced['trend_1_7'] = df_enhanced['da_lag_1'] - df_enhanced['da_lag_7']
            df_enhanced['trend_direction_1_7'] = (df_enhanced['trend_1_7'] > 0).astype(int)
        
        if 'da_lag_7' in df_enhanced.columns and 'da_lag_14' in df_enhanced.columns:
            df_enhanced['trend_7_14'] = df_enhanced['da_lag_7'] - df_enhanced['da_lag_14']
            df_enhanced['trend_direction_7_14'] = (df_enhanced['trend_7_14'] > 0).astype(int)
        
        # 2. Volatility indicators (how variable has DA been?)
        if len([col for col in df_enhanced.columns if 'da_lag_' in col]) >= 3:
            lag_cols = [f'da_lag_{lag}' for lag in [1, 2, 3, 7] 
                       if f'da_lag_{lag}' in df_enhanced.columns]
            if len(lag_cols) >= 3:
                df_enhanced['da_volatility_short'] = df_enhanced[lag_cols].std(axis=1)
        
        # 3. Acceleration indicators (rate of change in trend)
        if all(f'da_lag_{lag}' in df_enhanced.columns for lag in [1, 7, 14]):
            recent_trend = df_enhanced['da_lag_1'] - df_enhanced['da_lag_7']
            older_trend = df_enhanced['da_lag_7'] - df_enhanced['da_lag_14']
            df_enhanced['da_acceleration'] = recent_trend - older_trend
        
        # 4. Pattern recognition features
        # Weekly pattern: is this week similar to 1/2/3 weeks ago?
        for week_back in [2, 3, 4]:
            lag_col = f'da_lag_{week_back * 7}'
            if 'da_lag_7' in df_enhanced.columns and lag_col in df_enhanced.columns:
                pattern_col = f'weekly_pattern_{week_back}'
                df_enhanced[pattern_col] = np.abs(df_enhanced['da_lag_7'] - df_enhanced[lag_col])
        
        # 5. Environmental leading indicators (with proper temporal lags)
        env_features = ['sst', 'chlor_a', 'beuti', 'pdo', 'oni']
        for env_var in env_features:
            if env_var in df_enhanced.columns:
                env_vals = pd.to_numeric(df_enhanced[env_var], errors='coerce')
                
                # Create environmental lags (these might lead DA changes)
                for lag in [7, 14, 21]:
                    env_lag_col = f'{env_var}_lag_{lag}'
                    if env_lag_col not in df_enhanced.columns:
                        df_enhanced[env_lag_col] = df_enhanced.groupby('site')[env_var].shift(lag)
                    
                    # Environmental change indicators
                    env_lag_vals = pd.to_numeric(df_enhanced[env_lag_col], errors='coerce')
                    change_col = f'{env_var}_change_{lag}d'
                    df_enhanced[change_col] = env_vals - env_lag_vals
                    
                    # Significant environmental changes (potential spike triggers)
                    if df_enhanced[change_col].notna().any() and df_enhanced[change_col].std() > 0:
                        change_std = df_enhanced[change_col].std()
                        sig_change_col = f'{env_var}_significant_change_{lag}d'
                        df_enhanced[sig_change_col] = (
                            np.abs(df_enhanced[change_col]) > 1.5 * change_std
                        ).astype(int)
        
        # 6. Seasonal and contextual features
        df_enhanced['month'] = df_enhanced['date'].dt.month
        df_enhanced['day_of_year'] = df_enhanced['date'].dt.dayofyear
        
        # Peak DA season with more granular timing
        df_enhanced['is_peak_season'] = df_enhanced['month'].isin([6, 7, 8]).astype(int)
        df_enhanced['is_shoulder_season'] = df_enhanced['month'].isin([5, 9]).astype(int)
        
        # 7. Site-specific baseline adjustments
        # Each site may have different temporal patterns
        site_means = df_enhanced.groupby('site')['da'].transform('mean')
        if 'da_lag_7' in df_enhanced.columns:
            df_enhanced['site_adjusted_lag7'] = df_enhanced['da_lag_7'] - site_means
        
        # 8. Hybrid naive features (improvements on simple lag)
        lag_columns = [col for col in df_enhanced.columns if col.startswith('da_lag_')]
        if len(lag_columns) >= 2:
            # Weighted average of multiple lags (more sophisticated than simple lag)
            weights = np.array([0.4, 0.3, 0.2, 0.1])  # Recent lags weighted more
            weighted_cols = lag_columns[:len(weights)]
            if len(weighted_cols) >= 2:
                df_enhanced['weighted_lag_avg'] = 0
                for i, col in enumerate(weighted_cols):
                    df_enhanced['weighted_lag_avg'] += weights[i] * df_enhanced[col].fillna(0)
        
        print(f"Advanced features created: {len(df_enhanced.columns)} total columns")
        return df_enhanced
    
    def create_naive_baseline(self, df, anchor_date):
        """
        Create the exact naive baseline we need to beat (1-week lag).
        """
        site_data = df[df['date'] <= anchor_date].copy()
        if site_data.empty:
            return np.nan
            
        # Find value exactly 7 days ago
        one_week_ago = anchor_date - pd.Timedelta(days=7)
        
        # Get the closest date to 7 days ago
        available_dates = pd.to_datetime(site_data['date'])
        if len(available_dates) == 0:
            return np.nan
            
        time_diffs = np.abs((available_dates - one_week_ago).dt.days)
        closest_idx = np.argmin(time_diffs)
        closest_date = available_dates.iloc[closest_idx]
        
        # If closest date is more than 3 days away from target, use most recent
        if abs((closest_date - one_week_ago).days) > 3:
            recent_data = site_data.sort_values('date').iloc[-1]
            da_val = recent_data['da']
        else:
            week_data = site_data[site_data['date'] == closest_date]
            da_val = week_data['da'].iloc[0] if not week_data.empty else np.nan
        
        return float(da_val) if not pd.isna(da_val) else np.nan
    
    def create_baseline_beating_models(self):
        """
        Create ensemble of models specifically designed to beat naive baseline.
        """
        models = {
            # Temporal-focused Random Forest
            'temporal_rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features=0.7,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            
            # Gradient boosting optimized for temporal patterns
            'temporal_gbr': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_seed
            ),
            
            # Regularized linear model for stable predictions
            'regularized_linear': Ridge(
                alpha=2.0,
                random_state=self.random_seed
            )
        }
        
        return models
    
    def fit_baseline_beating_model(self, model, X_train, y_train, model_name):
        """
        Fit model with strategies to beat naive baseline.
        """
        # Ensure data is clean and numeric
        X_train_clean = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train_clean = pd.Series(y_train).astype(float)
        
        # Remove any remaining invalid data
        valid_mask = ~(y_train_clean.isna() | np.isinf(y_train_clean))
        if not valid_mask.all():
            X_train_clean = X_train_clean[valid_mask]
            y_train_clean = y_train_clean[valid_mask]
        
        if len(y_train_clean) < 10:
            return None
        
        # Fit model
        try:
            model.fit(X_train_clean, y_train_clean)
            return model
        except Exception as e:
            print(f"Model {model_name} fitting failed: {e}")
            return None
    
    def predict_with_ensemble(self, models, X_test):
        """
        Make ensemble prediction optimized to beat naive baseline.
        """
        X_test_clean = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce').fillna(0)
        
        predictions = []
        weights = []
        
        model_weights = {
            'temporal_rf': 0.4,         # Strong on complex patterns
            'temporal_gbr': 0.35,       # Good at temporal sequences
            'regularized_linear': 0.25   # Stable baseline
        }
        
        for model_name, model in models.items():
            if model is not None:
                try:
                    pred = model.predict(X_test_clean)[0]
                    if not np.isnan(pred) and not np.isinf(pred):
                        predictions.append(max(0.0, float(pred)))
                        weights.append(model_weights.get(model_name, 0.2))
                except Exception as e:
                    print(f"Prediction failed for {model_name}: {e}")
                    continue
        
        if not predictions:
            return np.nan
        
        # Weighted ensemble
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_pred = np.average(predictions, weights=weights)
        
        return max(0.0, float(ensemble_pred))
    
    def process_single_comparison(self, anchor_info, full_data, min_test_date):
        """
        Process single anchor point for naive baseline comparison.
        """
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data['site'] == site].copy()
        site_data = site_data.sort_values('date')
        
        # Find valid test date
        future_data = site_data[site_data['date'] > anchor_date]
        if future_data.empty:
            return None
            
        test_candidates = future_data[
            (future_data['date'] >= min_test_date) & 
            ((future_data['date'] - anchor_date).dt.days >= self.temporal_buffer_days)
        ]
        
        if test_candidates.empty:
            return None
            
        test_df = test_candidates.iloc[:1].copy()
        test_date = test_df['date'].iloc[0]
        actual_da = test_df['da'].iloc[0]
        
        if pd.isna(actual_da):
            return None
        
        # Create advanced features
        site_enhanced = self.create_advanced_temporal_features(site_data, anchor_date)
        
        # Prepare training data
        train_data = site_enhanced[site_enhanced['date'] <= anchor_date].copy()
        train_clean = train_data.dropna(subset=['da'])
        
        if len(train_clean) < 20:  # Need sufficient data
            return None
        
        # Prepare test data
        test_enhanced = site_enhanced[site_enhanced['date'] == test_date].copy()
        if test_enhanced.empty:
            return None
        
        # Features for modeling
        exclude_cols = ['date', 'site', 'da', 'month', 'day_of_year']
        feature_cols = [col for col in train_clean.columns 
                       if col not in exclude_cols and train_clean[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 5:  # Need enough features
            return None
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['da']
        X_test = test_enhanced[feature_cols].reindex(columns=feature_cols, fill_value=0)
        
        # Create naive baseline
        naive_pred = self.create_naive_baseline(site_data, anchor_date)
        
        # Train models
        models = self.create_baseline_beating_models()
        fitted_models = {}
        
        for model_name, model in models.items():
            fitted_model = self.fit_baseline_beating_model(model, X_train, y_train, model_name)
            if fitted_model is not None:
                fitted_models[model_name] = fitted_model
        
        if not fitted_models:
            return None
        
        # Make ensemble prediction
        ensemble_pred = self.predict_with_ensemble(fitted_models, X_test)
        
        return {
            'site': site,
            'date': test_date,
            'anchor_date': anchor_date,
            'actual_da': float(actual_da),
            'naive_baseline': naive_pred,
            'ensemble_prediction': ensemble_pred,
            'n_features_used': len(feature_cols),
            'n_models_fitted': len(fitted_models)
        }
    
    def run_baseline_beating_evaluation(self, n_anchors=80, min_test_date="2010-01-01"):
        """
        Run comprehensive evaluation to beat naive baseline.
        """
        print("Loading data for baseline beating challenge...")
        data = self.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        min_target_date = pd.Timestamp(min_test_date)
        
        # Generate strategic anchor points
        anchor_infos = []
        sites = data['site'].unique()[:5]  # Focus on first 5 sites for thorough testing
        
        for site in sites:
            site_data = data[data['site'] == site]
            site_dates = site_data['date'].sort_values().unique()
            
            if len(site_dates) > 60:  # Need sufficient history
                # Sample dates strategically
                valid_dates = [d for d in site_dates[30:-15] if d >= min_target_date]
                if len(valid_dates) > n_anchors // len(sites):
                    selected_dates = np.random.choice(
                        valid_dates, 
                        n_anchors // len(sites), 
                        replace=False
                    )
                    anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_dates])
        
        if not anchor_infos:
            print("ERROR: No valid anchor points generated")
            return None
            
        print(f"Generated {len(anchor_infos)} anchor points across {len(sites)} sites")
        print("Processing anchor points (this may take several minutes)...")
        
        # Process in parallel
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self.process_single_comparison)(ai, data, min_target_date)
            for ai in anchor_infos
        )
        
        # Filter valid results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            print("ERROR: No successful predictions generated")
            return None
            
        results_df = pd.DataFrame(valid_results)
        print(f"\nGenerated {len(results_df)} predictions")
        
        # Evaluate performance
        self.evaluate_against_baseline(results_df)
        
        return results_df
    
    def evaluate_against_baseline(self, results_df):
        """
        Comprehensive evaluation against naive baseline.
        """
        print("\n" + "="*60)
        print("NAIVE BASELINE BEATING CHALLENGE RESULTS")
        print("="*60)
        
        # Filter for valid predictions
        valid_mask = ~(results_df['actual_da'].isna() | 
                      results_df['naive_baseline'].isna() | 
                      results_df['ensemble_prediction'].isna())
        
        if not valid_mask.any():
            print("ERROR: No valid predictions for comparison")
            return
        
        valid_df = results_df[valid_mask]
        
        # Calculate metrics for both approaches
        naive_r2 = r2_score(valid_df['actual_da'], valid_df['naive_baseline'])
        naive_mae = mean_absolute_error(valid_df['actual_da'], valid_df['naive_baseline'])
        
        ensemble_r2 = r2_score(valid_df['actual_da'], valid_df['ensemble_prediction'])
        ensemble_mae = mean_absolute_error(valid_df['actual_da'], valid_df['ensemble_prediction'])
        
        # Spike detection metrics (15 ppm)
        y_true_spike = (valid_df['actual_da'] > self.spike_threshold).astype(int)
        
        naive_spike_pred = (valid_df['naive_baseline'] > self.spike_threshold).astype(int)
        ensemble_spike_pred = (valid_df['ensemble_prediction'] > self.spike_threshold).astype(int)
        
        naive_precision = precision_score(y_true_spike, naive_spike_pred, zero_division=0)
        naive_recall = recall_score(y_true_spike, naive_spike_pred, zero_division=0)
        naive_f1 = f1_score(y_true_spike, naive_spike_pred, zero_division=0)
        
        ensemble_precision = precision_score(y_true_spike, ensemble_spike_pred, zero_division=0)
        ensemble_recall = recall_score(y_true_spike, ensemble_spike_pred, zero_division=0)
        ensemble_f1 = f1_score(y_true_spike, ensemble_spike_pred, zero_division=0)
        
        # False positive analysis
        naive_fp = ((naive_spike_pred == 1) & (y_true_spike == 0)).sum()
        ensemble_fp = ((ensemble_spike_pred == 1) & (y_true_spike == 0)).sum()
        
        # Display results
        print(f"\nRESULTS SUMMARY ({len(valid_df)} predictions):")
        print("-" * 40)
        
        print("\nREGRESSION METRICS:")
        print(f"  Naive Baseline:     R² = {naive_r2:.4f}, MAE = {naive_mae:.2f}")
        print(f"  Advanced Ensemble:  R² = {ensemble_r2:.4f}, MAE = {ensemble_mae:.2f}")
        
        r2_improvement = (ensemble_r2 - naive_r2) / max(abs(naive_r2), 0.001) * 100
        mae_improvement = (naive_mae - ensemble_mae) / max(naive_mae, 0.001) * 100
        
        print(f"  R² Improvement:     {r2_improvement:+.1f}%")
        print(f"  MAE Improvement:    {mae_improvement:+.1f}%")
        
        print(f"\nSPIKE DETECTION (>{self.spike_threshold} ppm):")
        actual_spikes = y_true_spike.sum()
        print(f"  Actual spikes: {actual_spikes}/{len(valid_df)} ({actual_spikes/len(valid_df)*100:.1f}%)")
        
        print(f"\n  Naive Baseline:")
        print(f"    Precision = {naive_precision:.4f}, Recall = {naive_recall:.4f}, F1 = {naive_f1:.4f}")
        print(f"    False Positives = {naive_fp}")
        
        print(f"\n  Advanced Ensemble:")
        print(f"    Precision = {ensemble_precision:.4f}, Recall = {ensemble_recall:.4f}, F1 = {ensemble_f1:.4f}")
        print(f"    False Positives = {ensemble_fp}")
        
        f1_improvement = (ensemble_f1 - naive_f1) / max(naive_f1, 0.001) * 100
        precision_improvement = (ensemble_precision - naive_precision) / max(naive_precision, 0.001) * 100
        
        print(f"\n  F1 Improvement:     {f1_improvement:+.1f}%")
        print(f"  Precision Improvement: {precision_improvement:+.1f}%")
        
        # Success assessment
        print("\n" + "="*40)
        print("CHALLENGE OUTCOME:")
        print("="*40)
        
        beats_r2 = ensemble_r2 > naive_r2
        beats_f1 = ensemble_f1 > naive_f1
        maintains_precision = ensemble_precision >= naive_precision * 0.95  # Allow small precision drop
        
        if beats_r2 and beats_f1 and maintains_precision:
            print("🎉 SUCCESS: Advanced ensemble BEATS naive baseline!")
            print(f"   ✓ Better R² ({ensemble_r2:.4f} vs {naive_r2:.4f})")
            print(f"   ✓ Better F1 ({ensemble_f1:.4f} vs {naive_f1:.4f})")
            print(f"   ✓ Maintained precision ({ensemble_precision:.4f} vs {naive_precision:.4f})")
        elif beats_r2 or beats_f1:
            print("⚡ PARTIAL SUCCESS: Improvement in some metrics")
            if beats_r2:
                print(f"   ✓ Better R² ({ensemble_r2:.4f} vs {naive_r2:.4f})")
            if beats_f1:
                print(f"   ✓ Better F1 ({ensemble_f1:.4f} vs {naive_f1:.4f})")
        else:
            print("😅 CHALLENGE: Still working to beat naive baseline")
            print("   The naive baseline remains remarkably strong!")
        
        # Feature importance insights
        avg_features = valid_df['n_features_used'].mean()
        avg_models = valid_df['n_models_fitted'].mean()
        print(f"\nMODEL INSIGHTS:")
        print(f"  Average features used: {avg_features:.0f}")
        print(f"  Average models per prediction: {avg_models:.1f}")


def main():
    print("NAIVE BASELINE BEATING CHALLENGE")
    print("="*50)
    print("Goal: Beat naive 1-week lag baseline with advanced temporal modeling")
    print("Threshold: 15 ppm spike detection")
    print("Focus: Both regression accuracy and spike timing precision")
    print()
    
    challenger = NaiveBaselineBeat()
    
    try:
        results = challenger.run_baseline_beating_evaluation(n_anchors=100)
        
        if results is not None:
            # Save results for further analysis
            os.makedirs("cache", exist_ok=True)
            results.to_parquet("cache/baseline_challenge_results.parquet")
            print(f"\nResults saved to cache/baseline_challenge_results.parquet")
            return 0
        else:
            print("Challenge failed - no results generated")
            return 1
            
    except Exception as e:
        print(f"Challenge failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)