#!/usr/bin/env python3
"""
AutoML Testing for DA Forecasting
==================================

Tests AutoML frameworks that automatically find optimal models.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import config

# Try importing AutoML libraries
try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False
    print("AutoGluon not installed. Install with: pip install autogluon")

try:
    import h2o
    from h2o.automl import H2OAutoML
    HAS_H2O = True
except ImportError:
    HAS_H2O = False
    print("H2O not installed. Install with: pip install h2o")

try:
    from flaml import AutoML as FLAMLAutoML
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False
    print("FLAML not installed. Install with: pip install flaml")

try:
    from autosklearn.regression import AutoSklearnRegressor
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False
    print("Auto-sklearn not installed (Linux only)")

try:
    from tpot import TPOTRegressor
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False
    print("TPOT not installed. Install with: pip install tpot")


def test_autogluon():
    """Test AutoGluon AutoML."""
    if not HAS_AUTOGLUON:
        return None
        
    print("\n" + "="*60)
    print("TESTING AUTOGLUON")
    print("="*60)
    
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Prepare data
    sites = data['site'].unique()[:3]
    test_data = data[data['site'].isin(sites)].copy()
    
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'site', 'da', 'pn', 'da-category']]
    
    test_data_clean = test_data.dropna(subset=['da'])
    X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
    y = test_data_clean['da']
    
    # Create train/test with label column
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_data = X_train.copy()
    train_data['da'] = y_train
    test_data = X_test.copy()
    test_data['da'] = y_test
    
    # Train AutoGluon
    print("Training AutoGluon (this may take a few minutes)...")
    start_time = time.time()
    
    predictor = TabularPredictor(
        label='da',
        eval_metric='r2',
        problem_type='regression',
        verbosity=0
    )
    
    predictor.fit(
        train_data,
        time_limit=120,  # 2 minutes
        presets='best_quality',
        excluded_model_types=['NN_TORCH', 'FASTAI']  # Exclude deep learning for speed
    )
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = predictor.predict(test_data.drop('da', axis=1))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get best model info
    leaderboard = predictor.leaderboard(test_data, silent=True)
    best_model = leaderboard.iloc[0]['model']
    
    print(f"\nBest Model: {best_model}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f} μg/g")
    print(f"Train time: {train_time:.2f}s")
    
    print("\nTop 5 Models:")
    print(leaderboard[['model', 'score_test']].head())
    
    return {'r2': r2, 'mae': mae, 'best_model': best_model}


def test_flaml():
    """Test Microsoft FLAML AutoML."""
    if not HAS_FLAML:
        return None
        
    print("\n" + "="*60)
    print("TESTING FLAML (Microsoft AutoML)")
    print("="*60)
    
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Prepare data
    sites = data['site'].unique()[:3]
    test_data = data[data['site'].isin(sites)].copy()
    
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'site', 'da', 'pn', 'da-category']]
    
    test_data_clean = test_data.dropna(subset=['da'])
    X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
    y = test_data_clean['da']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train FLAML
    print("Training FLAML AutoML...")
    start_time = time.time()
    
    automl = FLAMLAutoML()
    automl.fit(
        X_train, y_train,
        task="regression",
        metric='r2',
        time_budget=120,  # 2 minutes
        early_stop=True,
        ensemble=True,
        verbose=0
    )
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = automl.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nBest Model: {automl.best_estimator}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f} μg/g")
    print(f"Train time: {train_time:.2f}s")
    print(f"Best config: {automl.best_config}")
    
    return {'r2': r2, 'mae': mae, 'best_model': str(automl.best_estimator)}


def test_tpot():
    """Test TPOT genetic programming AutoML."""
    if not HAS_TPOT:
        return None
        
    print("\n" + "="*60)
    print("TESTING TPOT (Genetic Programming AutoML)")
    print("="*60)
    
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Prepare data
    sites = data['site'].unique()[:3]
    test_data = data[data['site'].isin(sites)].copy()
    
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'site', 'da', 'pn', 'da-category']]
    
    test_data_clean = test_data.dropna(subset=['da'])
    X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
    y = test_data_clean['da']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train TPOT
    print("Training TPOT (genetic programming)...")
    start_time = time.time()
    
    tpot = TPOTRegressor(
        generations=5,  # Number of generations
        population_size=20,
        scoring='r2',
        cv=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        max_time_mins=2  # 2 minutes max
    )
    
    tpot.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = tpot.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nBest Pipeline Found:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f} μg/g")
    print(f"Train time: {train_time:.2f}s")
    
    # Export the best pipeline
    tpot.export('tpot_best_pipeline.py')
    print("Best pipeline exported to: tpot_best_pipeline.py")
    
    return {'r2': r2, 'mae': mae, 'best_model': 'TPOT Pipeline'}


def test_h2o():
    """Test H2O AutoML."""
    if not HAS_H2O:
        return None
        
    print("\n" + "="*60)
    print("TESTING H2O AUTOML")
    print("="*60)
    
    try:
        # Initialize H2O
        h2o.init(nthreads=-1, max_mem_size="4G")
        
        # Load data
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        data['date'] = pd.to_datetime(data['date'])
        
        # Prepare data
        sites = data['site'].unique()[:3]
        test_data = data[data['site'].isin(sites)].copy()
        
        feature_cols = [col for col in test_data.columns 
                       if col not in ['date', 'site', 'da', 'pn', 'da-category']]
        
        test_data_clean = test_data.dropna(subset=['da'])
        X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
        y = test_data_clean['da']
        
        # Create H2O frames
        train_data = X.copy()
        train_data['da'] = y
        h2o_data = h2o.H2OFrame(train_data)
        
        # Split data
        train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
        
        # Train H2O AutoML
        print("Training H2O AutoML...")
        start_time = time.time()
        
        aml = H2OAutoML(
            max_runtime_secs=120,  # 2 minutes
            seed=42,
            exclude_algos=['DeepLearning'],  # Exclude deep learning for speed
            sort_metric='RMSE'
        )
        
        aml.train(y='da', training_frame=train)
        train_time = time.time() - start_time
        
        # Get best model
        best_model = aml.leader
        
        # Evaluate
        perf = best_model.model_performance(test)
        
        print(f"\nBest Model: {best_model.__class__.__name__}")
        print(f"RMSE: {perf.rmse():.4f}")
        print(f"MAE: {perf.mae():.2f}")
        print(f"Train time: {train_time:.2f}s")
        
        # Leaderboard
        lb = aml.leaderboard
        print("\nTop Models:")
        print(lb.head())
        
        h2o.shutdown(prompt=False)
        
        return {'rmse': perf.rmse(), 'mae': perf.mae(), 'best_model': best_model.__class__.__name__}
        
    except Exception as e:
        print(f"H2O Error: {e}")
        try:
            h2o.shutdown(prompt=False)
        except:
            pass
        return None


def main():
    """Run AutoML tests."""
    print("="*60)
    print("AUTOML FRAMEWORK COMPARISON")
    print("="*60)
    print("Testing automated machine learning frameworks...")
    print("Baseline Random Forest R²: ~0.78")
    
    results = {}
    
    # Test AutoGluon
    if HAS_AUTOGLUON:
        try:
            results['AutoGluon'] = test_autogluon()
        except Exception as e:
            print(f"AutoGluon failed: {e}")
    
    # Test FLAML
    if HAS_FLAML:
        try:
            results['FLAML'] = test_flaml()
        except Exception as e:
            print(f"FLAML failed: {e}")
    
    # Test TPOT
    if HAS_TPOT:
        try:
            results['TPOT'] = test_tpot()
        except Exception as e:
            print(f"TPOT failed: {e}")
    
    # Test H2O
    if HAS_H2O:
        try:
            results['H2O'] = test_h2o()
        except Exception as e:
            print(f"H2O failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("AUTOML RESULTS SUMMARY")
    print("="*60)
    print("Baseline RF R²: 0.7818")
    print("\nAutoML Framework Results:")
    
    for framework, result in results.items():
        if result and 'r2' in result:
            improvement = ((result['r2'] - 0.7818) / 0.7818) * 100
            status = "✅" if improvement > 5 else "❌"
            print(f"{status} {framework}: R²={result['r2']:.4f} ({improvement:+.1f}%), Best: {result.get('best_model', 'N/A')}")
    
    # Find best AutoML
    best_framework = None
    best_r2 = 0.7818
    
    for framework, result in results.items():
        if result and result.get('r2', 0) > best_r2:
            best_framework = framework
            best_r2 = result['r2']
    
    if best_framework:
        print(f"\n🏆 BEST AUTOML: {best_framework} (R²={best_r2:.4f})")
        print("AutoML found a model better than our manual selection!")
    else:
        print("\n❌ No AutoML framework beat our manually selected XGBoost/Stacking")


if __name__ == "__main__":
    main()