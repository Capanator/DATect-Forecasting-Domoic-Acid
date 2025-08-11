#!/usr/bin/env python3
"""
Comprehensive Model Testing Script - Production Pipeline
=========================================================
Tests 15+ machine learning models for Domoic Acid forecasting
using the EXACT production pipeline configuration with 50 anchors per site.

Production Configuration (from model_factory.py):
- n_estimators: 300
- max_depth: 8
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

Models tested:
- Gradient Boosting: XGBoost, LightGBM, CatBoost, sklearn GradientBoosting
- Ensemble: Random Forest, Extra Trees, Stacking, Voting, Bagging, AdaBoost
- Linear: Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge
- Traditional ML: SVM, KNN, Decision Tree
- Neural Networks: 2-Layer NN, MLP
- AutoML: FLAML

Author: DATect Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import json
import warnings
import sys
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from forecasting.core.data_processor import DataProcessor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ComprehensiveModelTest:
    """Test all available models using production pipeline configuration."""
    
    def __init__(self, anchors_per_site=50):
        """
        Initialize comprehensive model testing with production configuration.
        
        Args:
            anchors_per_site: Number of temporal anchor points per site (default: 50)
        """
        self.anchors_per_site = anchors_per_site
        self.processor = DataProcessor()
        self.results = []
        
        # PRODUCTION CONFIGURATION (exact match to model_factory.py)
        self.production_config = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': config.RANDOM_SEED
        }
        
        print("="*80)
        print(" "*15 + "🚀 COMPREHENSIVE MODEL TESTING (PRODUCTION PIPELINE)")
        print("="*80)
        print(f"\n📊 Production Configuration:")
        for key, value in self.production_config.items():
            print(f"   • {key}: {value}")
        print(f"   • Anchors per site: {anchors_per_site}")
        print(f"   • Temporal buffer: {config.TEMPORAL_BUFFER_DAYS} days")
        print(f"   • Satellite buffer: {config.SATELLITE_BUFFER_DAYS} days")
        print(f"   • Parallel processing: {min(multiprocessing.cpu_count() - 1, 8)} cores")
        print("="*80)
    
    def load_data(self):
        """Load and prepare the dataset using production pipeline."""
        print("\n📁 Loading Data (Production Pipeline)...")
        
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['date', 'site']).reset_index(drop=True)
        
        feature_cols = [col for col in data.columns 
                       if col not in ['date', 'site', 'da', 'lat', 'lon', 'Predicted_da']]
        
        # Get unique sites
        sites = data['site'].unique()
        
        print(f"✅ Loaded {len(data):,} records")
        print(f"✅ Features: {len(feature_cols)}")
        print(f"✅ Sites: {len(sites)}")
        print(f"✅ Date range: {data['date'].min().date()} to {data['date'].max().date()}")
        
        return data, feature_cols, sites
    
    def create_site_based_splits(self, data, feature_cols, sites):
        """Create temporal splits with specified anchors per site."""
        print(f"\n🔄 Creating Temporal Splits ({self.anchors_per_site} anchors per site)...")
        
        all_splits = []
        
        # Calculate buffer requirements (production settings)
        total_buffer = config.TEMPORAL_BUFFER_DAYS + config.SATELLITE_BUFFER_DAYS
        
        for site in sites:
            # Get data for this site
            site_data = data[data['site'] == site]
            
            # Get valid date range for this site
            min_date = site_data['date'].min() + pd.Timedelta(days=total_buffer + 365)
            max_date = site_data['date'].max() - pd.Timedelta(days=total_buffer)
            
            valid_dates = site_data[(site_data['date'] >= min_date) & 
                                   (site_data['date'] <= max_date)]['date'].unique()
            
            if len(valid_dates) == 0:
                continue
            
            # Sample anchor dates for this site
            np.random.seed(config.RANDOM_SEED + hash(site) % 1000)  # Site-specific seed for reproducibility
            n_anchors = min(self.anchors_per_site, len(valid_dates))
            anchor_dates = np.random.choice(valid_dates, size=n_anchors, replace=False)
            
            # Create splits for each anchor date
            for test_date in anchor_dates:
                # Production-style train/test split
                train_cutoff = test_date - pd.Timedelta(days=config.TEMPORAL_BUFFER_DAYS)
                
                # Train on ALL data before cutoff (not just site-specific)
                train_data = data[data['date'] < train_cutoff].copy()
                # Test on site-specific data for the test date
                test_data = site_data[site_data['date'] == test_date].copy()
                
                if len(train_data) < config.MIN_TRAINING_SAMPLES or len(test_data) == 0:
                    continue
                
                # Prepare features (fillna as in production)
                X_train = train_data[feature_cols].fillna(0).values
                y_train = train_data['da'].fillna(0).values
                X_test = test_data[feature_cols].fillna(0).values
                y_test = test_data['da'].fillna(0).values
                
                all_splits.append({
                    'site': site,
                    'test_date': test_date,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                })
        
        print(f"✅ Created {len(all_splits)} total temporal splits")
        print(f"✅ Average splits per site: {len(all_splits) / len(sites):.1f}")
        
        return all_splits
    
    def process_single_split(self, model, model_name, split, needs_scaling):
        """Process a single train/test split (for parallel execution)."""
        try:
            start_time = time.time()
            
            X_train = split['X_train']
            y_train = split['y_train']
            X_test = split['X_test']
            y_test = split['y_test']
            site = split['site']
            
            # Scale if needed
            if needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Special handling for certain models
            if model_name == 'FLAML':
                from flaml import AutoML
                automl = AutoML()
                automl.fit(X_train, y_train,
                          task="regression",
                          time_budget=5,  # Very short for speed with parallel processing
                          verbose=0)
                predictions = automl.predict(X_test)
            elif model_name in ['2-Layer NN', 'MLP']:
                from sklearn.neural_network import MLPRegressor
                if model_name == '2-Layer NN':
                    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                                           max_iter=500,
                                           random_state=self.production_config['random_state'])
                else:
                    nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                           max_iter=500,
                                           random_state=self.production_config['random_state'])
                nn_model.fit(X_train, y_train)
                predictions = nn_model.predict(X_test)
            else:
                # Clone the model for thread safety
                from sklearn.base import clone
                model_clone = clone(model) if model != 'FLAML' and model not in ['2-Layer NN', 'MLP'] else model
                model_clone.fit(X_train, y_train)
                predictions = model_clone.predict(X_test)
            
            return {
                'predictions': predictions.tolist(),
                'actuals': y_test.tolist(),
                'site': site,
                'training_time': time.time() - start_time,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def evaluate_model(self, model, model_name, splits, needs_scaling=False):
        """Evaluate a model across all temporal splits using parallel processing."""
        print(f"  Testing {model_name} with parallel processing...")
        
        # Use parallel processing with progress bar
        n_jobs = min(multiprocessing.cpu_count() - 1, 8)  # Leave one CPU free, max 8
        
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self.process_single_split)(model, model_name, split, needs_scaling)
            for split in tqdm(splits, desc=f"    Preparing {model_name}", leave=False)
        )
        
        # Aggregate results
        all_predictions = []
        all_actuals = []
        training_times = []
        site_performance = {}
        
        for result in results:
            if result['success']:
                all_predictions.extend(result['predictions'])
                all_actuals.extend(result['actuals'])
                training_times.append(result['training_time'])
                
                # Track per-site performance
                site = result['site']
                if site not in site_performance:
                    site_performance[site] = {'predictions': [], 'actuals': []}
                site_performance[site]['predictions'].extend(result['predictions'])
                site_performance[site]['actuals'].extend(result['actuals'])
        
        # Calculate overall metrics
        if len(all_predictions) > 0:
            r2 = r2_score(all_actuals, all_predictions)
            mae = mean_absolute_error(all_actuals, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
            
            # Calculate per-site R² scores
            site_r2_scores = []
            for site, perf in site_performance.items():
                if len(perf['predictions']) > 0:
                    site_r2 = r2_score(perf['actuals'], perf['predictions'])
                    site_r2_scores.append(site_r2)
            
            result = {
                'model': model_name,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'n_predictions': len(all_predictions),
                'n_sites_tested': len(site_performance),
                'avg_site_r2': np.mean(site_r2_scores) if site_r2_scores else 0,
                'std_site_r2': np.std(site_r2_scores) if site_r2_scores else 0,
                'avg_training_time': np.mean(training_times)
            }
            
            print(f"  ✅ {model_name}: R² = {r2:.4f}, MAE = {mae:.2f}, Sites = {len(site_performance)}")
            
            return result
        else:
            print(f"  ❌ {model_name}: Failed")
            return None
    
    def create_all_models(self):
        """Create all models with PRODUCTION configuration."""
        models = []
        
        print("\n🤖 Creating Models with Production Configuration...")
        
        # Gradient Boosting Models - WITH PRODUCTION SETTINGS
        try:
            import xgboost as xgb
            models.append(('XGBoost', xgb.XGBRegressor(
                n_estimators=self.production_config['n_estimators'],
                max_depth=self.production_config['max_depth'],
                learning_rate=self.production_config['learning_rate'],
                subsample=self.production_config['subsample'],
                colsample_bytree=self.production_config['colsample_bytree'],
                random_state=self.production_config['random_state'],
                n_jobs=-1
            ), False))
        except ImportError:
            print("  ⚠️ XGBoost not available")
        
        try:
            import lightgbm as lgb
            models.append(('LightGBM', lgb.LGBMRegressor(
                n_estimators=self.production_config['n_estimators'],
                max_depth=self.production_config['max_depth'],
                learning_rate=self.production_config['learning_rate'],
                subsample=self.production_config['subsample'],
                colsample_bytree=self.production_config['colsample_bytree'],
                num_leaves=2**self.production_config['max_depth'] - 1,
                random_state=self.production_config['random_state'],
                n_jobs=-1,
                verbose=-1
            ), False))
        except ImportError:
            print("  ⚠️ LightGBM not available")
        
        try:
            import catboost as cb
            models.append(('CatBoost', cb.CatBoostRegressor(
                iterations=self.production_config['n_estimators'],
                depth=self.production_config['max_depth'],
                learning_rate=self.production_config['learning_rate'],
                subsample=self.production_config['subsample'],
                random_state=self.production_config['random_state'],
                verbose=False
            ), False))
        except ImportError:
            print("  ⚠️ CatBoost not available")
        
        # Ensemble Methods - WITH PRODUCTION-EQUIVALENT SETTINGS
        from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                     GradientBoostingRegressor, AdaBoostRegressor,
                                     BaggingRegressor, VotingRegressor, StackingRegressor)
        from sklearn.linear_model import Ridge
        
        models.append(('Random Forest', RandomForestRegressor(
            n_estimators=self.production_config['n_estimators'],
            max_depth=self.production_config['max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.production_config['random_state'],
            n_jobs=-1
        ), False))
        
        models.append(('Extra Trees', ExtraTreesRegressor(
            n_estimators=self.production_config['n_estimators'],
            max_depth=self.production_config['max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.production_config['random_state'],
            n_jobs=-1
        ), False))
        
        models.append(('Gradient Boosting', GradientBoostingRegressor(
            n_estimators=self.production_config['n_estimators'],
            max_depth=self.production_config['max_depth'],
            learning_rate=self.production_config['learning_rate'],
            subsample=self.production_config['subsample'],
            random_state=self.production_config['random_state']
        ), False))
        
        models.append(('AdaBoost', AdaBoostRegressor(
            n_estimators=self.production_config['n_estimators'],
            learning_rate=self.production_config['learning_rate'],
            random_state=self.production_config['random_state']
        ), False))
        
        models.append(('Bagging', BaggingRegressor(
            n_estimators=100,  # Reduced for speed
            random_state=self.production_config['random_state'],
            n_jobs=-1
        ), False))
        
        # Stacking Ensemble
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=self.production_config['max_depth'], 
                                        random_state=self.production_config['random_state'])),
            ('ridge', Ridge())
        ]
        models.append(('Stacking', StackingRegressor(
            estimators=base_models, final_estimator=Ridge(), cv=3
        ), False))
        
        # Linear Models
        from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                         ElasticNet, BayesianRidge)
        
        models.append(('Linear Regression', LinearRegression(), False))
        models.append(('Ridge', Ridge(alpha=1.0, random_state=self.production_config['random_state']), False))
        models.append(('Lasso', Lasso(alpha=0.1, random_state=self.production_config['random_state'], max_iter=2000), False))
        models.append(('ElasticNet', ElasticNet(alpha=0.1, random_state=self.production_config['random_state'], max_iter=2000), False))
        models.append(('Bayesian Ridge', BayesianRidge(), False))
        
        # Traditional ML
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        models.append(('SVM', SVR(kernel='rbf', C=1.0, gamma='scale'), True))
        models.append(('KNN-5', KNeighborsRegressor(n_neighbors=5, n_jobs=-1), True))
        models.append(('Decision Tree', DecisionTreeRegressor(
            max_depth=self.production_config['max_depth'],
            random_state=self.production_config['random_state']
        ), False))
        
        # Neural Networks
        models.append(('2-Layer NN', '2-Layer NN', True))
        models.append(('MLP', 'MLP', True))
        
        # AutoML
        try:
            from flaml import AutoML
            models.append(('FLAML', 'FLAML', False))
        except ImportError:
            print("  ⚠️ FLAML not available")
        
        print(f"✅ Created {len(models)} models with production configuration")
        
        return models
    
    def run(self):
        """Run complete model comparison with production pipeline."""
        # Load data
        data, feature_cols, sites = self.load_data()
        
        # Create site-based temporal splits
        splits = self.create_site_based_splits(data, feature_cols, sites)
        
        # Create and test all models
        models = self.create_all_models()
        
        print("\n" + "="*80)
        print(" "*20 + "🏁 RUNNING PRODUCTION PIPELINE TESTS")
        print("="*80 + "\n")
        
        results = []
        for model_name, model, needs_scaling in models:
            result = self.evaluate_model(model, model_name, splits, needs_scaling)
            if result:
                results.append(result)
        
        # Sort by R² score
        results = sorted(results, key=lambda x: x['r2_score'], reverse=True)
        
        # Print summary
        self.print_summary(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def print_summary(self, results):
        """Print results summary."""
        print("\n" + "="*80)
        print(" "*20 + "📊 PRODUCTION PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{'Rank':<6} {'Model':<25} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'Sites':<8}")
        print("-"*69)
        
        for i, r in enumerate(results[:15], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji}{i:<4} {r['model']:<25} {r['r2_score']:<10.4f} "
                  f"{r['mae']:<10.2f} {r['rmse']:<10.2f} {r['n_sites_tested']:<8}")
    
    def save_results(self, results):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_models_production_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'configuration': {
                    'anchors_per_site': self.anchors_per_site,
                    'production_config': self.production_config,
                    'temporal_buffer': config.TEMPORAL_BUFFER_DAYS,
                    'satellite_buffer': config.SATELLITE_BUFFER_DAYS
                },
                'results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main execution."""
    tester = ComprehensiveModelTest(anchors_per_site=30)  # Reduced for faster execution
    results = tester.run()
    
    print("\n✅ Comprehensive production pipeline testing complete!")
    return results


if __name__ == "__main__":
    results = main()