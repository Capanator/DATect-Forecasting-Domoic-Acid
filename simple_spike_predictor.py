#!/usr/bin/env python3
"""
SIMPLIFIED SPIKE PREDICTOR - BEAT THE BASELINE
==============================================

No overengineering. Direct path: data → features → model → results.
Goal: Beat naive baseline F1=0.853 for spike detection (DA > 20 ppm).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SimpleSpikePredictor:
    """Direct spike prediction without overengineering."""
    
    def __init__(self, spike_threshold=20.0):
        self.spike_threshold = spike_threshold
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load data directly from final_output.parquet."""
        print("📊 Loading final_output.parquet...")
        df = pd.read_parquet('data/processed/final_output.parquet')
        
        # Basic cleaning
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['site', 'date']).reset_index(drop=True)
        
        # Rename columns to match expected format
        df = df.rename(columns={'site': 'site_id', 'da': 'DA_Levels'})
        
        print(f"Loaded {len(df)} rows, {len(df['site_id'].unique())} sites")
        return df
    
    def create_spike_features(self, df):
        """Create minimal features focused on spike precursors."""
        features = []
        
        for site in df['site_id'].unique():
            site_data = df[df['site_id'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Core environmental features (using actual available columns)
            feature_cols = ['oni', 'pdo', 'discharge', 'pn', 'beuti', 
                           'chla-anom', 'modis-chla', 'modis-flr', 'modis-k490', 
                           'modis-par', 'modis-sst', 'sst-anom']
            
            # Lag features (1, 2, 3 weeks back)
            for col in feature_cols:
                if col in site_data.columns:
                    site_data[f'{col}_lag1'] = site_data[col].shift(1)
                    site_data[f'{col}_lag2'] = site_data[col].shift(2)
                    site_data[f'{col}_lag3'] = site_data[col].shift(3)
            
            # Rate of change features
            for col in feature_cols:
                if col in site_data.columns:
                    site_data[f'{col}_change'] = site_data[col] - site_data[col].shift(1)
                    site_data[f'{col}_change2'] = site_data[col] - site_data[col].shift(2)
            
            # DA history (without future leakage)
            site_data['DA_lag1'] = site_data['DA_Levels'].shift(1)
            site_data['DA_lag2'] = site_data['DA_Levels'].shift(2)
            site_data['DA_lag3'] = site_data['DA_Levels'].shift(3)
            site_data['DA_trend'] = site_data['DA_lag1'] - site_data['DA_lag2']
            
            # Spike target (binary)
            site_data['spike_target'] = (site_data['DA_Levels'] > self.spike_threshold).astype(int)
            
            features.append(site_data)
        
        result = pd.concat(features, ignore_index=True)
        print(f"Created features for {len(result)} rows")
        return result
    
    def prepare_training_data(self, df):
        """Prepare training data with extreme focus on spikes."""
        # Remove rows with insufficient history
        df = df.dropna(subset=['DA_lag3']).copy()
        
        # Feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'site_id', 'DA_Levels', 'spike_target']]
        
        X = df[feature_cols].fillna(0)  # Simple NaN handling
        y = df['spike_target']
        dates = df['date']
        sites = df['site_id']
        
        print(f"Training data: {len(X)} samples, {len(feature_cols)} features")
        print(f"Spike ratio: {y.mean():.3f}")
        
        return X, y, dates, sites, feature_cols
    
    def create_temporal_split(self, dates, train_ratio=0.7):
        """Simple temporal split - no anchoring complexity."""
        cutoff_date = dates.quantile(train_ratio)
        train_mask = dates <= cutoff_date
        return train_mask
    
    def train_models(self, X_train, y_train):
        """Train multiple spike-focused models."""
        print("🤖 Training spike detection models...")
        
        # Calculate extreme spike weights
        spike_ratio = y_train.mean()
        spike_weight = min(50, 1 / spike_ratio)  # Up to 50x weight
        sample_weight = np.where(y_train == 1, spike_weight, 1.0)
        
        print(f"Using spike weight: {spike_weight:.1f}x")
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        
        models_to_train = {
            'xgb_extreme': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=spike_weight
            ),
            'rf_weighted': RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={0: 1, 1: spike_weight},
                random_state=42
            ),
            'gb_extreme': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'logistic_balanced': LogisticRegression(
                class_weight={0: 1, 1: spike_weight},
                max_iter=1000,
                random_state=42
            )
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            if name == 'logistic_balanced':
                model.fit(X_train_scaled, y_train)
            elif name in ['gb_extreme']:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
            
            self.models[name] = model
    
    def evaluate_against_baseline(self, X_test, y_test, dates_test, sites_test):
        """Evaluate models against naive baseline."""
        print("\n📈 EVALUATING AGAINST NAIVE BASELINE")
        print("=" * 50)
        
        # Create naive baseline (7-day lag)
        baseline_preds = self.create_naive_baseline(dates_test, sites_test)
        baseline_f1 = f1_score(y_test, baseline_preds)
        
        print(f"🎯 NAIVE BASELINE F1: {baseline_f1:.3f}")
        print("-" * 30)
        
        # Test all models
        X_test_scaled = self.scalers['standard'].transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            if name == 'logistic_balanced':
                preds = model.predict(X_test_scaled)
            else:
                preds = model.predict(X_test)
            
            f1 = f1_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            
            beat_baseline = "✅ BEATS BASELINE!" if f1 > baseline_f1 else "❌"
            improvement = ((f1 - baseline_f1) / baseline_f1) * 100
            
            print(f"{name:20} F1={f1:.3f} P={precision:.3f} R={recall:.3f} {beat_baseline}")
            if f1 > baseline_f1:
                print(f"                     🚀 {improvement:+.1f}% better than baseline!")
            
            results[name] = {
                'f1': f1, 'precision': precision, 'recall': recall,
                'beats_baseline': f1 > baseline_f1,
                'improvement': improvement
            }
        
        return results, baseline_f1
    
    def create_naive_baseline(self, dates_test, sites_test):
        """Create naive 7-day lag baseline predictions."""
        # Load full dataset to get historical DA
        full_df = self.load_data()
        
        baseline_preds = []
        
        for i, (date, site) in enumerate(zip(dates_test, sites_test)):
            # Get 7-day lag date
            lag_date = pd.to_datetime(date) - pd.Timedelta(days=7)
            
            # Find DA value at lag date
            site_data = full_df[full_df['site_id'] == site].copy()
            site_data = site_data.sort_values('date')
            
            # Find closest date to lag_date
            closest_idx = (site_data['date'] - lag_date).abs().idxmin()
            lag_da = site_data.loc[closest_idx, 'DA_Levels']
            
            # Predict spike if lagged DA > threshold
            spike_pred = 1 if lag_da > self.spike_threshold else 0
            baseline_preds.append(spike_pred)
        
        return np.array(baseline_preds)
    
    def run_experiment(self):
        """Run the complete experiment."""
        print("🎯 SIMPLE SPIKE PREDICTOR - BEAT THE BASELINE")
        print("=" * 55)
        
        # Load and prepare data
        df = self.load_data()
        df = self.create_spike_features(df)
        X, y, dates, sites, feature_cols = self.prepare_training_data(df)
        
        # Simple temporal split
        train_mask = self.create_temporal_split(dates)
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        dates_test, sites_test = dates[~train_mask], sites[~train_mask]
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate
        results, baseline_f1 = self.evaluate_against_baseline(X_test, y_test, dates_test, sites_test)
        
        # Check if we beat the baseline
        winners = [name for name, r in results.items() if r['beats_baseline']]
        
        if winners:
            best_model = max(winners, key=lambda x: results[x]['f1'])
            print(f"\n🏆 SUCCESS! {best_model} beats baseline!")
            print(f"Best F1: {results[best_model]['f1']:.3f} vs Baseline: {baseline_f1:.3f}")
        else:
            print(f"\n💔 No model beat baseline yet. Best F1: {max(r['f1'] for r in results.values()):.3f}")
            print("Need to try more aggressive approaches...")
        
        return results, baseline_f1


if __name__ == "__main__":
    predictor = SimpleSpikePredictor(spike_threshold=20.0)
    results, baseline_f1 = predictor.run_experiment()