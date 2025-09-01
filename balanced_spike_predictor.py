#!/usr/bin/env python3
"""
BALANCED SPIKE PREDICTOR
=======================

Balance spike detection accuracy with overall R² performance.
Focus on high precision to minimize false positives while maintaining good regression performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class BalancedSpikePredictor:
    
    def __init__(self, spike_threshold=20.0):
        self.spike_threshold = spike_threshold
        
    def load_data(self):
        df = pd.read_parquet('data/processed/final_output.parquet')
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'site': 'site_id', 'da': 'DA_Levels'})
        return df.sort_values(['site_id', 'date']).reset_index(drop=True)
    
    def create_balanced_features(self, df):
        """Create features that help both regression and classification."""
        features = []
        
        for site in df['site_id'].unique():
            site_data = df[df['site_id'] == site].copy().reset_index(drop=True)
            
            # Core environmental features
            env_cols = ['discharge', 'beuti', 'modis-chla', 'modis-sst', 'pn', 'oni', 'pdo']
            
            # Essential lags for trend detection
            for col in env_cols:
                for lag in [1, 2, 3, 4]:
                    site_data[f'{col}_lag{lag}'] = site_data[col].shift(lag)
            
            # Rate features for both regression and spike detection
            for col in env_cols:
                site_data[f'{col}_change'] = site_data[col] - site_data[col].shift(1)
                site_data[f'{col}_trend2'] = site_data[col] - site_data[col].shift(2)
            
            # DA history (crucial for both tasks)
            for lag in [1, 2, 3, 4]:
                site_data[f'DA_lag{lag}'] = site_data['DA_Levels'].shift(lag)
            
            site_data['DA_change'] = site_data['DA_lag1'] - site_data['DA_lag2']
            site_data['DA_trend'] = site_data['DA_lag1'] - site_data['DA_lag3']
            
            # Spike target
            site_data['spike_target'] = (site_data['DA_Levels'] > self.spike_threshold).astype(int)
            
            features.append(site_data)
        
        return pd.concat(features, ignore_index=True)
    
    def prepare_data(self, df):
        df = df.dropna(subset=['DA_lag3']).copy()
        
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'site_id', 'DA_Levels', 'spike_target', 'lat', 'lon']]
        
        X = df[feature_cols].fillna(0)
        y_reg = df['DA_Levels']  # Regression target
        y_class = df['spike_target']  # Classification target
        dates = df['date']
        sites = df['site_id']
        
        return X, y_reg, y_class, dates, sites
    
    def create_baseline(self, dates_test, sites_test):
        full_df = self.load_data()
        baseline_preds = []
        
        for date, site in zip(dates_test, sites_test):
            lag_date = pd.to_datetime(date) - pd.Timedelta(days=7)
            site_data = full_df[full_df['site_id'] == site]
            
            if len(site_data) == 0:
                baseline_preds.append(0)
                continue
                
            closest_idx = (site_data['date'] - lag_date).abs().idxmin()
            lag_da = site_data.loc[closest_idx, 'DA_Levels']
            baseline_preds.append(1 if lag_da > self.spike_threshold else 0)
        
        return np.array(baseline_preds)
    
    def train_balanced_models(self, X_train, y_reg_train, y_class_train, X_test, y_reg_test, y_class_test, dates_test, sites_test):
        """Train models with different precision/recall trade-offs."""
        print("🎯 BALANCED SPIKE PREDICTOR - OPTIMIZE FOR BOTH METRICS")
        print("=" * 65)
        
        baseline_preds = self.create_baseline(dates_test, sites_test)
        baseline_f1 = f1_score(y_class_test, baseline_preds)
        
        print(f"Baseline F1: {baseline_f1:.3f}")
        print("-" * 40)
        
        # Test different configurations balancing spike detection and regression
        configs = [
            # (name, spike_weight, n_estimators, max_depth, learning_rate, description)
            ("balanced", 10, 500, 6, 0.1, "Moderate spike weight for balance"),
            ("precision_focus", 8, 800, 6, 0.08, "Lower weight, focus on precision"),
            ("high_precision", 6, 1000, 5, 0.08, "Even lower weight, higher precision"),
            ("regression_plus", 4, 800, 8, 0.1, "Regression-focused with spike boost"),
            ("moderate_boost", 12, 600, 7, 0.09, "Moderate spike boost"),
        ]
        
        results = []
        
        for name, weight, n_est, depth, lr, desc in configs:
            print(f"\n{name.upper()}: {desc}")
            print(f"Config: weight={weight}, trees={n_est}, depth={depth}, lr={lr}")
            
            # Train model
            model = xgb.XGBRegressor(  # Use regressor for better overall performance
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            # Create sample weights favoring spikes
            sample_weight = np.where(y_class_train == 1, weight, 1.0)
            model.fit(X_train, y_reg_train, sample_weight=sample_weight)
            
            # Predictions
            y_pred_reg = model.predict(X_test)
            
            # Convert regression predictions to classification
            y_pred_class = (y_pred_reg > self.spike_threshold).astype(int)
            
            # Evaluate both tasks
            r2 = r2_score(y_reg_test, y_pred_reg)
            mae = mean_absolute_error(y_reg_test, y_pred_reg)
            f1 = f1_score(y_class_test, y_pred_class)
            precision = precision_score(y_class_test, y_pred_class)
            recall = recall_score(y_class_test, y_pred_class)
            
            # Calculate balance score
            r2_norm = max(0, r2)  # Normalized R²
            balance_score = (r2_norm * 0.4) + (f1 * 0.6)  # Weight spike detection more
            
            beats_baseline = f1 > baseline_f1
            status = "✅" if beats_baseline else "❌"
            
            print(f"  Regression: R²={r2:.3f}, MAE={mae:.1f}")
            print(f"  Spikes:     F1={f1:.3f}, P={precision:.3f}, R={recall:.3f} {status}")
            print(f"  Balance Score: {balance_score:.3f}")
            
            results.append({
                'name': name,
                'model': model,
                'config': (weight, n_est, depth, lr),
                'r2': r2,
                'mae': mae,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'balance_score': balance_score,
                'beats_baseline': beats_baseline,
                'y_pred_reg': y_pred_reg,
                'y_pred_class': y_pred_class
            })
        
        # Find best balanced model
        valid_results = [r for r in results if r['r2'] > 0.3 and r['precision'] > 0.7]  # Reasonable thresholds
        
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['balance_score'])
            print(f"\n🏆 BEST BALANCED MODEL: {best_result['name'].upper()}")
            print(f"Balance Score: {best_result['balance_score']:.3f}")
            print(f"R²={best_result['r2']:.3f}, MAE={best_result['mae']:.1f}")
            print(f"F1={best_result['f1']:.3f}, P={best_result['precision']:.3f}, R={best_result['recall']:.3f}")
            
            if best_result['beats_baseline']:
                improvement = ((best_result['f1'] - baseline_f1) / baseline_f1) * 100
                print(f"🎉 Beats baseline by {improvement:+.1f}%!")
            
        else:
            print("\n⚠️ No models met quality thresholds (R²>0.3, Precision>0.7)")
            best_result = max(results, key=lambda x: x['balance_score'])
            print(f"Best available: {best_result['name']}")
        
        return results, best_result
    
    def run_balanced_experiment(self):
        print("🎯 BALANCED SPIKE PREDICTOR")
        print("Finding optimal trade-off between R² and spike detection")
        print("=" * 60)
        
        df = self.load_data()
        df = self.create_balanced_features(df)
        X, y_reg, y_class, dates, sites = self.prepare_data(df)
        
        print(f"Data: {X.shape[1]} features, {y_class.mean():.3f} spike ratio")
        
        # Split
        cutoff = dates.quantile(0.7)
        train_mask = dates <= cutoff
        
        X_train, X_test = X[train_mask], X[~train_mask]
        y_reg_train, y_reg_test = y_reg[train_mask], y_reg[~train_mask]
        y_class_train, y_class_test = y_class[train_mask], y_class[~train_mask]
        dates_test, sites_test = dates[~train_mask], sites[~train_mask]
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        
        # Train and evaluate
        results, best_model = self.train_balanced_models(
            X_train, y_reg_train, y_class_train, 
            X_test, y_reg_test, y_class_test, 
            dates_test, sites_test
        )
        
        return results, best_model


if __name__ == "__main__":
    predictor = BalancedSpikePredictor(spike_threshold=20.0)
    results, best_model = predictor.run_balanced_experiment()