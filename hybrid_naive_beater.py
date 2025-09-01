#!/usr/bin/env python3
"""
Hybrid Naive Baseline Beater
============================

Targeted approach that uses naive baseline as foundation and adds corrections
for specific failure patterns identified in the analysis.

Strategy:
1. Start with naive baseline prediction
2. Add corrections for missed spikes using environmental triggers  
3. Site-specific adjustments for high-failure sites (Coos Bay, Newport)
4. Seasonal corrections for summer spike patterns
5. Conservative approach to minimize false positives
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# Add forecasting module to path
sys.path.insert(0, str(Path(__file__).parent / "forecasting"))

from forecasting.data_processor import DataProcessor
import config


class HybridNaiveBeater:
    """
    Hybrid model that improves on naive baseline by targeting specific failures.
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.spike_threshold = 15.0
        self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
        # Site-specific failure patterns from analysis
        self.high_failure_sites = ['Coos Bay', 'Newport', 'Gold Beach', 'Clatsop Beach']
        self.low_failure_sites = ['Cannon Beach', 'Copalis', 'Quinault']
        
    def create_targeted_features(self, df, cutoff_date):
        """
        Create features specifically targeting naive baseline failures.
        """
        df = df.copy()
        df = df.sort_values(['site', 'date'])
        
        # Basic lag features
        basic_lags = [1, 2, 7, 14]
        df_enhanced = self.data_processor.create_lag_features_safe(
            df, 'site', 'da', basic_lags, cutoff_date
        )
        
        # Ensure numeric
        for lag in basic_lags:
            lag_col = f'da_lag_{lag}'
            if lag_col in df_enhanced.columns:
                df_enhanced[lag_col] = pd.to_numeric(df_enhanced[lag_col], errors='coerce')
        
        # TARGETED CORRECTIONS FOR NAIVE FAILURES
        
        # 1. Spike momentum indicators (catches rapid increases naive misses)
        if 'da_lag_1' in df_enhanced.columns and 'da_lag_7' in df_enhanced.columns:
            # Recent acceleration
            df_enhanced['spike_momentum'] = df_enhanced['da_lag_1'] - df_enhanced['da_lag_7']
            df_enhanced['positive_momentum'] = (df_enhanced['spike_momentum'] > 0).astype(int)
            
            # Rapid increase indicator (catches what naive misses)
            df_enhanced['rapid_increase'] = (df_enhanced['spike_momentum'] > 5.0).astype(int)
        
        # 2. Environmental spike triggers (high BEUTI variability was key)
        env_features = ['beuti', 'sst', 'chlor_a']
        for env_var in env_features:
            if env_var in df_enhanced.columns:
                env_vals = pd.to_numeric(df_enhanced[env_var], errors='coerce')
                
                # Environmental volatility (spike trigger)
                env_lag_7 = f'{env_var}_lag_7'
                if env_lag_7 not in df_enhanced.columns:
                    df_enhanced[env_lag_7] = df_enhanced.groupby('site')[env_var].shift(7)
                
                env_lag_vals = pd.to_numeric(df_enhanced[env_lag_7], errors='coerce')
                change_col = f'{env_var}_change_7d'
                df_enhanced[change_col] = env_vals - env_lag_vals
                
                # Significant environmental shift (potential spike trigger)
                if df_enhanced[change_col].notna().any() and df_enhanced[change_col].std() > 0:
                    change_std = df_enhanced[change_col].std()
                    trigger_col = f'{env_var}_spike_trigger'
                    df_enhanced[trigger_col] = (
                        np.abs(df_enhanced[change_col]) > 2.0 * change_std
                    ).astype(int)
        
        # 3. Seasonal spike patterns (summer has highest MAE)
        df_enhanced['month'] = df_enhanced['date'].dt.month
        df_enhanced['is_summer_spike_season'] = df_enhanced['month'].isin([6, 7, 8]).astype(int)
        df_enhanced['is_fall_spike_season'] = df_enhanced['month'].isin([9, 10, 11]).astype(int)
        
        # 4. Site-specific risk indicators
        df_enhanced['is_high_risk_site'] = df_enhanced['site'].isin(self.high_failure_sites).astype(int)
        
        # 5. Pattern break detection (when naive assumptions fail)
        if 'da_lag_7' in df_enhanced.columns and 'da_lag_14' in df_enhanced.columns:
            # Trend reversal (naive assumes persistence, but trend is changing)
            recent_trend = df_enhanced['da_lag_1'] - df_enhanced['da_lag_7'] if 'da_lag_1' in df_enhanced.columns else 0
            older_trend = df_enhanced['da_lag_7'] - df_enhanced['da_lag_14']
            df_enhanced['trend_reversal'] = (
                (recent_trend > 0) & (older_trend < 0) |  # Turning upward
                (recent_trend < 0) & (older_trend > 0)    # Turning downward
            ).astype(int)
        
        # 6. Multiple time scale disagreement (when different lags disagree - sign of change)
        if all(f'da_lag_{lag}' in df_enhanced.columns for lag in [1, 7, 14]):
            lag1_signal = (df_enhanced['da_lag_1'] > self.spike_threshold).astype(int)
            lag7_signal = (df_enhanced['da_lag_7'] > self.spike_threshold).astype(int)
            lag14_signal = (df_enhanced['da_lag_14'] > self.spike_threshold).astype(int)
            
            # Disagreement suggests transition period (naive struggles here)
            df_enhanced['lag_disagreement'] = (
                (lag1_signal != lag7_signal) | (lag7_signal != lag14_signal)
            ).astype(int)
        
        print(f"Targeted features created: {len(df_enhanced.columns)} columns")
        return df_enhanced
    
    def create_naive_baseline(self, df, anchor_date):
        """
        Create exact naive baseline (7-day lag).
        """
        site_data = df[df['date'] <= anchor_date].copy()
        if site_data.empty:
            return np.nan
            
        one_week_ago = anchor_date - pd.Timedelta(days=7)
        available_dates = pd.to_datetime(site_data['date'])
        
        if len(available_dates) == 0:
            return np.nan
            
        time_diffs = np.abs((available_dates - one_week_ago).dt.days)
        closest_idx = np.argmin(time_diffs)
        closest_date = available_dates.iloc[closest_idx]
        
        if abs((closest_date - one_week_ago).days) > 3:
            recent_data = site_data.sort_values('date').iloc[-1]
            da_val = recent_data['da']
        else:
            week_data = site_data[site_data['date'] == closest_date]
            da_val = week_data['da'].iloc[0] if not week_data.empty else np.nan
        
        return float(da_val) if not pd.isna(da_val) else np.nan
    
    def create_spike_correction_model(self):
        """
        Model to detect when naive baseline will miss spikes.
        """
        return LogisticRegression(
            random_state=self.random_seed,
            class_weight='balanced',  # Handle imbalanced spike detection
            max_iter=1000
        )
    
    def create_magnitude_correction_model(self):
        """
        Model to correct magnitude when spike detected.
        """
        return Ridge(
            alpha=1.0,
            random_state=self.random_seed
        )
    
    def fit_hybrid_models(self, X_train, y_train, naive_predictions):
        """
        Fit models to correct naive baseline failures.
        """
        # Ensure clean data
        X_train_clean = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train_clean = pd.Series(y_train).astype(float)
        naive_pred_clean = pd.Series(naive_predictions).astype(float)
        
        valid_mask = ~(y_train_clean.isna() | naive_pred_clean.isna())
        if not valid_mask.all():
            X_train_clean = X_train_clean[valid_mask]
            y_train_clean = y_train_clean[valid_mask]
            naive_pred_clean = naive_pred_clean[valid_mask]
        
        if len(y_train_clean) < 20:
            return None, None
        
        # Create targets for correction models
        naive_spike_pred = (naive_pred_clean > self.spike_threshold).astype(int)
        actual_spike = (y_train_clean > self.spike_threshold).astype(int)
        
        # Spike correction: detect when naive misses spikes
        spike_miss_target = (actual_spike == 1) & (naive_spike_pred == 0)
        
        # Magnitude correction: how much to adjust naive prediction
        magnitude_correction_target = y_train_clean - naive_pred_clean
        
        models = {}
        
        # Train spike miss detection model
        if spike_miss_target.sum() > 5:  # Need some positive examples
            try:
                spike_model = self.create_spike_correction_model()
                spike_model.fit(X_train_clean, spike_miss_target)
                models['spike_correction'] = spike_model
            except Exception as e:
                print(f"Spike correction model failed: {e}")
        
        # Train magnitude correction model  
        try:
            magnitude_model = self.create_magnitude_correction_model()
            magnitude_model.fit(X_train_clean, magnitude_correction_target)
            models['magnitude_correction'] = magnitude_model
        except Exception as e:
            print(f"Magnitude correction model failed: {e}")
        
        return models
    
    def predict_hybrid(self, models, X_test, naive_prediction, site):
        """
        Make hybrid prediction: naive baseline + targeted corrections.
        """
        X_test_clean = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Start with naive baseline
        final_prediction = naive_prediction
        
        if pd.isna(naive_prediction) or not models:
            return final_prediction
        
        corrections_applied = []
        
        # Apply spike correction if available
        if 'spike_correction' in models:
            try:
                spike_miss_prob = models['spike_correction'].predict_proba(X_test_clean)[0][1]
                
                # If high probability of missed spike, boost prediction
                if spike_miss_prob > 0.6:  # Conservative threshold
                    spike_boost = min(10.0, spike_miss_prob * 20.0)  # Conservative boost
                    final_prediction += spike_boost
                    corrections_applied.append(f'spike_boost_{spike_boost:.1f}')
                    
            except Exception as e:
                pass
        
        # Apply magnitude correction if available
        if 'magnitude_correction' in models:
            try:
                magnitude_correction = models['magnitude_correction'].predict(X_test_clean)[0]
                
                # Apply conservative correction
                conservative_correction = magnitude_correction * 0.3  # Dampen to avoid overcorrection
                final_prediction += conservative_correction
                corrections_applied.append(f'mag_corr_{conservative_correction:.1f}')
                
            except Exception as e:
                pass
        
        # Site-specific adjustments for high-failure sites
        if site in self.high_failure_sites:
            # Slightly more aggressive for high-failure sites
            if final_prediction > 10.0:  # Near threshold
                final_prediction *= 1.05  # Small boost
                corrections_applied.append('site_boost')
        
        # Seasonal adjustment (summer boost)
        if hasattr(X_test_clean, 'iloc') and 'is_summer_spike_season' in X_test_clean.columns:
            if X_test_clean['is_summer_spike_season'].iloc[0] == 1:
                if final_prediction > 8.0:  # Near spike territory
                    final_prediction *= 1.03  # Small summer boost
                    corrections_applied.append('summer_boost')
        
        final_prediction = max(0.0, float(final_prediction))
        
        return final_prediction
    
    def process_hybrid_anchor(self, anchor_info, full_data, min_test_date):
        """
        Process single anchor with hybrid approach.
        """
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data['site'] == site].copy()
        site_data = site_data.sort_values('date')
        
        # Find test date
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
        
        # Create features
        site_enhanced = self.create_targeted_features(site_data, anchor_date)
        
        # Prepare training data
        train_data = site_enhanced[site_enhanced['date'] <= anchor_date].copy()
        train_clean = train_data.dropna(subset=['da'])
        
        if len(train_clean) < 30:
            return None
        
        # Prepare test data
        test_enhanced = site_enhanced[site_enhanced['date'] == test_date].copy()
        if test_enhanced.empty:
            return None
        
        # Create naive baseline predictions for training
        naive_train_predictions = []
        for _, row in train_clean.iterrows():
            naive_pred = self.create_naive_baseline(site_data, row['date'])
            naive_train_predictions.append(naive_pred)
        
        # Filter out invalid training examples
        valid_naive_mask = ~pd.Series(naive_train_predictions).isna()
        if not valid_naive_mask.any():
            return None
            
        train_clean = train_clean[valid_naive_mask]
        naive_train_predictions = [p for p, v in zip(naive_train_predictions, valid_naive_mask) if v]
        
        if len(train_clean) < 20:
            return None
        
        # Features for modeling
        exclude_cols = ['date', 'site', 'da', 'month']
        feature_cols = [col for col in train_clean.columns 
                       if col not in exclude_cols and train_clean[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 5:
            return None
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['da']
        X_test = test_enhanced[feature_cols].reindex(columns=feature_cols, fill_value=0)
        
        # Get naive prediction for test
        naive_test_pred = self.create_naive_baseline(site_data, anchor_date)
        
        # Fit hybrid models
        hybrid_models = self.fit_hybrid_models(X_train, y_train, naive_train_predictions)
        
        # Make hybrid prediction
        hybrid_pred = self.predict_hybrid(hybrid_models, X_test, naive_test_pred, site)
        
        return {
            'site': site,
            'date': test_date,
            'anchor_date': anchor_date,
            'actual_da': float(actual_da),
            'naive_prediction': naive_test_pred,
            'hybrid_prediction': hybrid_pred,
            'n_features': len(feature_cols),
            'models_fitted': len([k for k, v in hybrid_models.items() if v is not None]) if hybrid_models else 0
        }
    
    def run_hybrid_evaluation(self, n_anchors=100, min_test_date="2010-01-01"):
        """
        Run hybrid evaluation targeting naive baseline weaknesses.
        """
        print("HYBRID NAIVE BASELINE BEATER")
        print("="*50)
        print("Strategy: Naive baseline + targeted corrections for failures")
        print()
        
        data = self.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        min_target_date = pd.Timestamp(min_test_date)
        
        # Focus on high-failure sites first
        priority_sites = ['Coos Bay', 'Newport', 'Gold Beach', 'Clatsop Beach', 'Kalaloch']
        anchor_infos = []
        
        for site in priority_sites:
            if site not in data['site'].unique():
                continue
                
            site_data = data[data['site'] == site]
            site_dates = site_data['date'].sort_values().unique()
            
            if len(site_dates) > 50:
                valid_dates = [d for d in site_dates[30:-15] if d >= min_target_date]
                if len(valid_dates) > n_anchors // len(priority_sites):
                    selected_dates = np.random.choice(
                        valid_dates, 
                        n_anchors // len(priority_sites),
                        replace=False
                    )
                    anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_dates])
        
        print(f"Generated {len(anchor_infos)} anchor points focusing on high-failure sites")
        
        # Process anchors
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self.process_hybrid_anchor)(ai, data, min_target_date)
            for ai in anchor_infos
        )
        
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            print("ERROR: No valid results")
            return None
            
        results_df = pd.DataFrame(valid_results)
        print(f"\nGenerated {len(results_df)} hybrid predictions")
        
        # Evaluate
        self.evaluate_hybrid_performance(results_df)
        
        return results_df
    
    def evaluate_hybrid_performance(self, results_df):
        """
        Evaluate hybrid model against naive baseline.
        """
        print("\n" + "="*60)
        print("HYBRID MODEL EVALUATION RESULTS")
        print("="*60)
        
        valid_mask = ~(results_df['actual_da'].isna() | 
                      results_df['naive_prediction'].isna() |
                      results_df['hybrid_prediction'].isna())
        
        if not valid_mask.any():
            print("ERROR: No valid predictions")
            return
        
        valid_df = results_df[valid_mask]
        
        # Regression metrics
        naive_r2 = r2_score(valid_df['actual_da'], valid_df['naive_prediction'])
        naive_mae = mean_absolute_error(valid_df['actual_da'], valid_df['naive_prediction'])
        
        hybrid_r2 = r2_score(valid_df['actual_da'], valid_df['hybrid_prediction'])
        hybrid_mae = mean_absolute_error(valid_df['actual_da'], valid_df['hybrid_prediction'])
        
        # Spike detection metrics
        y_true_spike = (valid_df['actual_da'] > self.spike_threshold).astype(int)
        
        naive_spike_pred = (valid_df['naive_prediction'] > self.spike_threshold).astype(int)
        hybrid_spike_pred = (valid_df['hybrid_prediction'] > self.spike_threshold).astype(int)
        
        naive_precision = precision_score(y_true_spike, naive_spike_pred, zero_division=0)
        naive_recall = recall_score(y_true_spike, naive_spike_pred, zero_division=0)
        naive_f1 = f1_score(y_true_spike, naive_spike_pred, zero_division=0)
        
        hybrid_precision = precision_score(y_true_spike, hybrid_spike_pred, zero_division=0)
        hybrid_recall = recall_score(y_true_spike, hybrid_spike_pred, zero_division=0)
        hybrid_f1 = f1_score(y_true_spike, hybrid_spike_pred, zero_division=0)
        
        # False positives
        naive_fp = ((naive_spike_pred == 1) & (y_true_spike == 0)).sum()
        hybrid_fp = ((hybrid_spike_pred == 1) & (y_true_spike == 0)).sum()
        
        print(f"RESULTS ({len(valid_df)} predictions):")
        print("-" * 40)
        
        print(f"\nREGRESSION METRICS:")
        print(f"  Naive Baseline:  R² = {naive_r2:.4f}, MAE = {naive_mae:.2f}")
        print(f"  Hybrid Model:    R² = {hybrid_r2:.4f}, MAE = {hybrid_mae:.2f}")
        
        r2_improvement = (hybrid_r2 - naive_r2) / max(abs(naive_r2), 0.001) * 100
        mae_improvement = (naive_mae - hybrid_mae) / max(naive_mae, 0.001) * 100
        
        print(f"  R² Improvement:  {r2_improvement:+.1f}%")
        print(f"  MAE Improvement: {mae_improvement:+.1f}%")
        
        print(f"\nSPIKE DETECTION (>{self.spike_threshold} ppm):")
        actual_spikes = y_true_spike.sum()
        print(f"  Actual spikes: {actual_spikes}/{len(valid_df)} ({actual_spikes/len(valid_df)*100:.1f}%)")
        
        print(f"\n  Naive Baseline:")
        print(f"    Precision = {naive_precision:.4f}, Recall = {naive_recall:.4f}, F1 = {naive_f1:.4f}")
        print(f"    False Positives = {naive_fp}")
        
        print(f"\n  Hybrid Model:")
        print(f"    Precision = {hybrid_precision:.4f}, Recall = {hybrid_recall:.4f}, F1 = {hybrid_f1:.4f}")
        print(f"    False Positives = {hybrid_fp}")
        
        f1_improvement = (hybrid_f1 - naive_f1) / max(naive_f1, 0.001) * 100
        recall_improvement = (hybrid_recall - naive_recall) / max(naive_recall, 0.001) * 100
        
        print(f"\n  F1 Improvement:     {f1_improvement:+.1f}%")
        print(f"  Recall Improvement: {recall_improvement:+.1f}%")
        
        # SUCCESS CHECK
        print("\n" + "="*40)
        print("BASELINE BEATING ASSESSMENT:")
        print("="*40)
        
        beats_r2 = hybrid_r2 > naive_r2
        beats_f1 = hybrid_f1 > naive_f1
        maintains_precision = hybrid_precision >= naive_precision * 0.9
        
        improvements = []
        if beats_r2:
            improvements.append(f"R² (+{r2_improvement:.1f}%)")
        if beats_f1:
            improvements.append(f"F1 (+{f1_improvement:.1f}%)")
        if hybrid_recall > naive_recall:
            improvements.append(f"Recall (+{recall_improvement:.1f}%)")
        
        if beats_r2 and beats_f1 and maintains_precision:
            print("🎉 SUCCESS: HYBRID MODEL BEATS NAIVE BASELINE!")
            print(f"   ✓ Improvements: {', '.join(improvements)}")
        elif improvements:
            print("⚡ PARTIAL SUCCESS: Some improvements achieved")
            print(f"   ✓ Improvements: {', '.join(improvements)}")
            print("   → Continue refinement for full success")
        else:
            print("🔄 ITERATION: Refining approach...")
            print("   → Analyzing patterns for next iteration")
        
        avg_models = valid_df['models_fitted'].mean()
        print(f"\nModel complexity: {avg_models:.1f} correction models per prediction")


def main():
    hybrid = HybridNaiveBeater()
    results = hybrid.run_hybrid_evaluation(n_anchors=120)
    
    if results is not None:
        os.makedirs("cache", exist_ok=True)
        results.to_parquet("cache/hybrid_results.parquet")
        print(f"\nResults saved to cache/hybrid_results.parquet")
        return results
    else:
        print("Hybrid approach failed")
        return None


if __name__ == "__main__":
    results = main()