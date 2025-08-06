#!/usr/bin/env python3
"""
Comprehensive Model Comparison for DA Forecasting
==================================================

Tests a wide variety of ML models to find one that significantly outperforms Random Forest.
Only integrates the best model if it shows substantial improvement.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import (
    ElasticNet, Lasso, HuberRegressor,
    SGDRegressor, PassiveAggressiveRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.tree import DecisionTreeRegressor
import time

# Try to import advanced libraries (will install if needed)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Install with: pip install catboost")

# Import our existing infrastructure
from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.data_processor import DataProcessor
import config


class ModelComparison:
    """Compare multiple models against RF baseline."""
    
    def __init__(self, n_test_anchors=20):
        """Initialize with test configuration."""
        self.n_test_anchors = n_test_anchors
        self.engine = ForecastEngine()
        self.processor = DataProcessor()
        self.results = {}
        
    def get_test_data(self):
        """Get a sample of data for testing."""
        # Load data
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        data['date'] = pd.to_datetime(data['date'])
        
        # Get a subset for faster testing
        sites = data['site'].unique()[:3]  # Test on 3 sites for speed
        test_data = data[data['site'].isin(sites)].copy()
        
        return test_data
    
    def create_regression_models(self):
        """Create all regression models to test."""
        models = {
            # BASELINE
            'Random Forest (Baseline)': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            
            # ENSEMBLE METHODS
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
            ),
            'Hist Gradient Boosting': HistGradientBoostingRegressor(
                max_iter=200, max_depth=10, random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100, random_state=42
            ),
            'Bagging': BaggingRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            
            # NEURAL NETWORKS
            'MLP (Deep)': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25), 
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'MLP (Wide)': MLPRegressor(
                hidden_layer_sizes=(200, 200), 
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            
            # SVM
            'SVR (RBF)': SVR(kernel='rbf', C=10, gamma='scale'),
            'SVR (Polynomial)': SVR(kernel='poly', C=10, degree=3),
            
            # NEIGHBORS
            'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
            
            # LINEAR MODELS
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'Huber': HuberRegressor(epsilon=1.35, alpha=0.001),
            
            # GAUSSIAN PROCESS (can be slow but powerful)
            'Gaussian Process': GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
                alpha=1e-6,
                normalize_y=True,
                random_state=42
            ),
        }
        
        # Add gradient boosting libraries if available
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
        if HAS_CATBOOST:
            models['CatBoost'] = cb.CatBoostRegressor(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
            
        return models
    
    def create_classification_models(self):
        """Create all classification models to test."""
        models = {
            # BASELINE
            'Random Forest (Baseline)': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            
            # ENSEMBLE METHODS
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
            ),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(
                max_iter=200, max_depth=10, random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, random_state=42
            ),
            
            # NEURAL NETWORKS
            'MLP (Deep)': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            
            # SVM
            'SVC (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
            
            # NEIGHBORS
            'KNN': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
        }
        
        # Add gradient boosting libraries if available
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
        if HAS_CATBOOST:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
            
        return models
    
    def test_model_on_sample(self, model, X_train, y_train, X_test, y_test, task='regression'):
        """Test a single model on sample data."""
        try:
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Evaluate
            if task == 'regression':
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                return {
                    'r2': r2,
                    'mae': mae,
                    'train_time': train_time,
                    'pred_time': pred_time,
                    'success': True
                }
            else:
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'pred_time': pred_time,
                    'success': True
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def run_comprehensive_comparison(self):
        """Run comprehensive model comparison."""
        print("="*60)
        print("COMPREHENSIVE MODEL COMPARISON FOR DA FORECASTING")
        print("="*60)
        print(f"Testing on MacBook Pro M4 with 18GB RAM")
        print(f"Looking for models that SIGNIFICANTLY outperform Random Forest")
        print("-"*60)
        
        # Load test data
        print("\nLoading test data...")
        test_data = self.get_test_data()
        
        # Prepare a sample for quick testing
        from sklearn.model_selection import train_test_split
        
        # Get features and target
        feature_cols = [col for col in test_data.columns 
                       if col not in ['date', 'site', 'da', 'pn', 'da-category']]
        
        # Clean data - remove rows with NaN in target
        test_data_clean = test_data.dropna(subset=['da'])
        X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
        y_reg = test_data_clean.loc[X.index, 'da']
        
        # Create classification target
        y_cls = pd.cut(y_reg, bins=config.DA_CATEGORY_BINS, 
                      labels=config.DA_CATEGORY_LABELS)
        # Handle any NaN values in categorical data
        y_cls = y_cls.fillna(0).astype(int)
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        _, _, y_cls_train, y_cls_test = train_test_split(
            X, y_cls, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_cols)}")
        
        # Test regression models
        print("\n" + "="*60)
        print("REGRESSION MODELS")
        print("="*60)
        
        reg_models = self.create_regression_models()
        reg_results = {}
        baseline_r2 = None
        
        for name, model in reg_models.items():
            print(f"\nTesting {name}...")
            result = self.test_model_on_sample(
                model, X_train, y_reg_train, X_test, y_reg_test, 'regression'
            )
            
            if result['success']:
                reg_results[name] = result
                print(f"  R² Score: {result['r2']:.4f}")
                print(f"  MAE: {result['mae']:.2f} μg/g")
                print(f"  Train time: {result['train_time']:.2f}s")
                
                if name == 'Random Forest (Baseline)':
                    baseline_r2 = result['r2']
                elif baseline_r2:
                    improvement = ((result['r2'] - baseline_r2) / abs(baseline_r2)) * 100
                    if improvement > 5:  # Significant = >5% improvement
                        print(f"  ⭐ SIGNIFICANT IMPROVEMENT: {improvement:.1f}% better than RF!")
            else:
                print(f"  ❌ Failed: {result['error']}")
        
        # Test classification models
        print("\n" + "="*60)
        print("CLASSIFICATION MODELS")
        print("="*60)
        
        cls_models = self.create_classification_models()
        cls_results = {}
        baseline_acc = None
        
        for name, model in cls_models.items():
            print(f"\nTesting {name}...")
            result = self.test_model_on_sample(
                model, X_train, y_cls_train, X_test, y_cls_test, 'classification'
            )
            
            if result['success']:
                cls_results[name] = result
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Train time: {result['train_time']:.2f}s")
                
                if name == 'Random Forest (Baseline)':
                    baseline_acc = result['accuracy']
                elif baseline_acc:
                    improvement = ((result['accuracy'] - baseline_acc) / baseline_acc) * 100
                    if improvement > 5:  # Significant = >5% improvement
                        print(f"  ⭐ SIGNIFICANT IMPROVEMENT: {improvement:.1f}% better than RF!")
            else:
                print(f"  ❌ Failed: {result['error']}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY - MODELS BEATING RANDOM FOREST")
        print("="*60)
        
        # Find best regression model
        if baseline_r2:
            print(f"\nRegression (Baseline RF R²: {baseline_r2:.4f}):")
            best_reg = None
            best_reg_score = baseline_r2
            
            for name, result in reg_results.items():
                if name != 'Random Forest (Baseline)' and result['r2'] > baseline_r2 * 1.05:  # >5% improvement
                    print(f"  ✅ {name}: R²={result['r2']:.4f} (+{((result['r2']-baseline_r2)/abs(baseline_r2)*100):.1f}%)")
                    if result['r2'] > best_reg_score:
                        best_reg = name
                        best_reg_score = result['r2']
            
            if best_reg:
                print(f"\n🏆 BEST REGRESSION MODEL: {best_reg} (R²={best_reg_score:.4f})")
            else:
                print("  No model significantly outperformed Random Forest")
        
        # Find best classification model
        if baseline_acc:
            print(f"\nClassification (Baseline RF Accuracy: {baseline_acc:.4f}):")
            best_cls = None
            best_cls_score = baseline_acc
            
            for name, result in cls_results.items():
                if name != 'Random Forest (Baseline)' and result['accuracy'] > baseline_acc * 1.05:
                    print(f"  ✅ {name}: Acc={result['accuracy']:.4f} (+{((result['accuracy']-baseline_acc)/baseline_acc*100):.1f}%)")
                    if result['accuracy'] > best_cls_score:
                        best_cls = name
                        best_cls_score = result['accuracy']
            
            if best_cls:
                print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_cls} (Accuracy={best_cls_score:.4f})")
            else:
                print("  No model significantly outperformed Random Forest")
        
        return reg_results, cls_results, reg_models, cls_models


def main():
    """Run comprehensive model comparison."""
    comparison = ModelComparison(n_test_anchors=20)
    reg_results, cls_results, reg_models, cls_models = comparison.run_comprehensive_comparison()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    # Check if any model significantly beat RF
    baseline_r2 = reg_results.get('Random Forest (Baseline)', {}).get('r2', 0)
    best_model = None
    best_improvement = 0
    
    for name, result in reg_results.items():
        if name != 'Random Forest (Baseline)' and result.get('r2', 0) > baseline_r2:
            improvement = ((result['r2'] - baseline_r2) / abs(baseline_r2)) * 100
            if improvement > best_improvement and improvement > 5:  # >5% is significant
                best_model = name
                best_improvement = improvement
    
    if best_model:
        print(f"\n✅ INTEGRATE {best_model.upper()} INTO PIPELINE")
        print(f"   - {best_improvement:.1f}% improvement over Random Forest")
        print(f"   - This is a SIGNIFICANT improvement worth the integration effort")
        print(f"\nNext steps:")
        print(f"1. Install required libraries if needed")
        print(f"2. Update model_factory.py to include {best_model}")
        print(f"3. Run full validation with more anchor points")
    else:
        print("\n❌ NO MODEL SIGNIFICANTLY OUTPERFORMED RANDOM FOREST")
        print("   - Random Forest remains the best choice")
        print("   - No integration needed at this time")
        print("\nPotential improvements to explore:")
        print("   - Feature engineering (more lag features, interactions)")
        print("   - Hyperparameter tuning of promising models")
        print("   - Ensemble of multiple models")


if __name__ == "__main__":
    main()