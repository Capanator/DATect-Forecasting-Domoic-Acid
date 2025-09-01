#!/usr/bin/env python3
"""
EXTREME SPIKE PREDICTOR - PUSH THE LIMITS
=========================================

Aggressive approaches to beat F1=0.853 baseline.
Testing extreme weighting, ensemble methods, and spike-focused architectures.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ExtremeSpikePredictor:
    """Extreme approaches to beat the baseline."""
    
    def __init__(self, spike_threshold=20.0):
        self.spike_threshold = spike_threshold
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and prepare data."""
        print("📊 Loading data...")
        df = pd.read_parquet('data/processed/final_output.parquet')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['site', 'date']).reset_index(drop=True)
        df = df.rename(columns={'site': 'site_id', 'da': 'DA_Levels'})
        return df
    
    def create_extreme_features(self, df):
        """Create aggressive feature engineering for spike detection."""
        features = []
        
        for site in df['site_id'].unique():
            site_data = df[df['site_id'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Base features
            env_cols = ['oni', 'pdo', 'discharge', 'pn', 'beuti', 
                       'chla-anom', 'modis-chla', 'modis-flr', 'modis-k490', 
                       'modis-par', 'modis-sst', 'sst-anom']
            
            # Multi-lag features (1-8 weeks)
            for col in env_cols:
                if col in site_data.columns:
                    for lag in [1, 2, 3, 4, 6, 8]:
                        site_data[f'{col}_lag{lag}'] = site_data[col].shift(lag)
            
            # Rate of change over multiple periods
            for col in env_cols:
                if col in site_data.columns:
                    site_data[f'{col}_delta1'] = site_data[col] - site_data[col].shift(1)
                    site_data[f'{col}_delta2'] = site_data[col] - site_data[col].shift(2) 
                    site_data[f'{col}_delta4'] = site_data[col] - site_data[col].shift(4)
                    site_data[f'{col}_accel'] = site_data[f'{col}_delta1'] - site_data[f'{col}_delta1'].shift(1)
            
            # Rolling statistics
            for col in env_cols:
                if col in site_data.columns:
                    site_data[f'{col}_mean3'] = site_data[col].rolling(3, min_periods=1).mean()
                    site_data[f'{col}_std3'] = site_data[col].rolling(3, min_periods=1).std()
                    site_data[f'{col}_max4'] = site_data[col].rolling(4, min_periods=1).max()
                    site_data[f'{col}_min4'] = site_data[col].rolling(4, min_periods=1).min()
            
            # DA history without leakage
            for lag in [1, 2, 3, 4, 6, 8]:
                site_data[f'DA_lag{lag}'] = site_data['DA_Levels'].shift(lag)
            
            # DA trend analysis
            site_data['DA_trend_1w'] = site_data['DA_lag1'] - site_data['DA_lag2']
            site_data['DA_trend_2w'] = site_data['DA_lag1'] - site_data['DA_lag3']
            site_data['DA_trend_4w'] = site_data['DA_lag1'] - site_data['DA_lag4']
            site_data['DA_acceleration'] = site_data['DA_trend_1w'] - site_data['DA_trend_1w'].shift(1)
            
            # Peak detection features
            site_data['DA_is_local_max'] = ((site_data['DA_lag1'] > site_data['DA_lag2']) & 
                                           (site_data['DA_lag1'] > site_data['DA_lag3'])).astype(int)
            
            # Spike history
            site_data['spike_lag1'] = (site_data['DA_lag1'] > self.spike_threshold).astype(int)
            site_data['spike_lag2'] = (site_data['DA_lag2'] > self.spike_threshold).astype(int)
            site_data['days_since_spike'] = 0
            
            # Calculate days since last spike
            last_spike_idx = -999
            for i in range(len(site_data)):
                if site_data.loc[i, 'spike_lag1'] == 1:
                    last_spike_idx = i
                if last_spike_idx > -999:
                    site_data.loc[i, 'days_since_spike'] = (i - last_spike_idx) * 7  # weekly data
            
            # Target
            site_data['spike_target'] = (site_data['DA_Levels'] > self.spike_threshold).astype(int)
            
            features.append(site_data)
        
        result = pd.concat(features, ignore_index=True)
        print(f"Created aggressive features: {len(result)} rows")
        return result
    
    def prepare_data(self, df):
        """Prepare training data with aggressive cleaning."""
        # More aggressive history requirement
        df = df.dropna(subset=['DA_lag4']).copy()
        
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'site_id', 'DA_Levels', 'spike_target', 'lat', 'lon']]
        
        X = df[feature_cols]
        
        # Aggressive NaN handling - fill with rolling means
        for col in X.columns:
            if X[col].isna().sum() > 0:
                # Fill with rolling mean, then forward fill, then 0
                X[col] = X[col].fillna(X[col].rolling(12, min_periods=1).mean())
                X[col] = X[col].fillna(method='ffill')
                X[col] = X[col].fillna(0)
        
        y = df['spike_target']
        dates = df['date']
        sites = df['site_id']
        
        print(f"Aggressive training data: {len(X)} samples, {len(feature_cols)} features")
        print(f"Spike ratio: {y.mean():.3f}")
        
        return X, y, dates, sites, feature_cols
    
    def train_extreme_models(self, X_train, y_train):
        """Train with extreme configurations."""
        print("🚀 Training EXTREME models...")
        
        spike_ratio = y_train.mean()
        extreme_weight = min(100, 1 / spike_ratio)  # Up to 100x weight!
        sample_weight = np.where(y_train == 1, extreme_weight, 1.0)
        
        print(f"Using EXTREME spike weight: {extreme_weight:.1f}x")
        
        # Multiple scalers
        self.scalers['robust'] = RobustScaler()
        self.scalers['standard'] = StandardScaler()
        
        X_train_robust = self.scalers['robust'].fit_transform(X_train)
        X_train_standard = self.scalers['standard'].fit_transform(X_train)
        
        models = {
            'xgb_ultra': xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,
                reg_lambda=1,
                scale_pos_weight=extreme_weight,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'lgb_extreme': lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                class_weight={0: 1, 1: extreme_weight},
                random_state=42,
                verbose=-1
            ),
            
            'rf_ultra': RandomForestClassifier(
                n_estimators=1000,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={0: 1, 1: extreme_weight},
                random_state=42,
                n_jobs=-1
            ),
            
            'gb_extreme': GradientBoostingClassifier(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            
            'logistic_ultra': LogisticRegression(
                class_weight={0: 1, 1: extreme_weight},
                max_iter=2000,
                C=0.1,
                random_state=42
            )
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'logistic_ultra':
                model.fit(X_train_robust, y_train)
            elif name in ['gb_extreme']:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
            
            self.models[name] = model
        
        # Create ensemble
        print("Creating MEGA ensemble...")
        ensemble_models = [
            ('xgb', self.models['xgb_ultra']),
            ('lgb', self.models['lgb_extreme']),
            ('rf', self.models['rf_ultra'])
        ]
        
        self.models['mega_ensemble'] = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        self.models['mega_ensemble'].fit(X_train, y_train)
    
    def create_baseline(self, dates_test, sites_test):
        """Create naive 7-day lag baseline."""
        full_df = self.load_data()
        baseline_preds = []
        
        for date, site in zip(dates_test, sites_test):
            lag_date = pd.to_datetime(date) - pd.Timedelta(days=7)
            site_data = full_df[full_df['site_id'] == site].copy()
            site_data = site_data.sort_values('date')
            
            closest_idx = (site_data['date'] - lag_date).abs().idxmin()
            lag_da = site_data.loc[closest_idx, 'DA_Levels']
            
            spike_pred = 1 if lag_da > self.spike_threshold else 0
            baseline_preds.append(spike_pred)
        
        return np.array(baseline_preds)
    
    def evaluate_extreme(self, X_test, y_test, dates_test, sites_test):
        """Evaluate with extreme focus on beating baseline."""
        print("\n🎯 EXTREME EVALUATION - BEAT THE BASELINE!")
        print("=" * 60)
        
        # Baseline
        baseline_preds = self.create_baseline(dates_test, sites_test)
        baseline_f1 = f1_score(y_test, baseline_preds)
        
        print(f"🎯 TARGET TO BEAT: F1 = {baseline_f1:.3f}")
        print("-" * 40)
        
        # Test models
        X_test_robust = self.scalers['robust'].transform(X_test)
        X_test_standard = self.scalers['standard'].transform(X_test)
        
        winners = []
        best_f1 = 0
        best_model = None
        
        for name, model in self.models.items():
            if name == 'logistic_ultra':
                preds = model.predict(X_test_robust)
            else:
                preds = model.predict(X_test)
            
            f1 = f1_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            
            beat_baseline = f1 > baseline_f1
            improvement = ((f1 - baseline_f1) / baseline_f1) * 100
            
            status = "🏆 WINNER!" if beat_baseline else "❌"
            
            print(f"{name:20} F1={f1:.3f} P={precision:.3f} R={recall:.3f} {status}")
            
            if beat_baseline:
                print(f"                     🚀 {improvement:+.1f}% better than baseline!")
                winners.append((name, f1))
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = name
        
        print(f"\n🎖️  BEST MODEL: {best_model} (F1={best_f1:.3f})")
        
        if winners:
            print(f"🎉 SUCCESS! {len(winners)} models beat baseline!")
            winners.sort(key=lambda x: x[1], reverse=True)
            print(f"🥇 Champion: {winners[0][0]} with F1={winners[0][1]:.3f}")
        else:
            print("💔 No models beat baseline yet...")
            print(f"Best was {best_model} with F1={best_f1:.3f}")
            print(f"Need {baseline_f1 - best_f1:.3f} more F1 points!")
        
        return winners, best_model, best_f1, baseline_f1
    
    def run_extreme_experiment(self):
        """Run the extreme experiment."""
        print("🔥 EXTREME SPIKE PREDICTOR - NO LIMITS!")
        print("=" * 60)
        
        # Load and prepare
        df = self.load_data()
        df = self.create_extreme_features(df)
        X, y, dates, sites, feature_cols = self.prepare_data(df)
        
        # Split
        cutoff_date = dates.quantile(0.7)
        train_mask = dates <= cutoff_date
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        dates_test, sites_test = dates[~train_mask], sites[~train_mask]
        
        print(f"Extreme split: {len(X_train)} train, {len(X_test)} test")
        
        # Train
        self.train_extreme_models(X_train, y_train)
        
        # Evaluate
        winners, best_model, best_f1, baseline_f1 = self.evaluate_extreme(
            X_test, y_test, dates_test, sites_test
        )
        
        return winners, best_model, best_f1, baseline_f1


if __name__ == "__main__":
    predictor = ExtremeSpikePredictor(spike_threshold=20.0)
    winners, best_model, best_f1, baseline_f1 = predictor.run_extreme_experiment()