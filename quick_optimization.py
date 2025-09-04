#!/usr/bin/env python3
"""
Quick XGBoost Optimization for DATect
====================================

Simple but comprehensive optimization script that actually works.
Tests key parameters that affect XGBoost performance.
"""

import pandas as pd
import numpy as np
import random
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
from sklearn.metrics import r2_score, mean_absolute_error, f1_score


@dataclass
class TestConfig:
    """Simple test configuration."""
    # XGBoost params
    n_estimators: int = 800
    max_depth: int = 6
    learning_rate: float = 0.08
    reg_alpha: float = 0.3
    reg_lambda: float = 0.5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    
    # Spike weighting
    spike_strategy: str = 'simple'
    spike_threshold: float = 20.0
    spike_weight: float = 8.0
    extreme_threshold: float = 40.0
    extreme_weight: float = 15.0
    
    # Other settings
    use_lag_features: bool = False
    min_training_samples: int = 3
    forecast_horizon_days: int = 7


class TestModelFactory(ModelFactory):
    """Test model factory with configurable parameters."""
    
    def __init__(self, test_config: TestConfig):
        super().__init__()
        self.test_config = test_config
    
    def _get_regression_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed")
            
            return xgb.XGBRegressor(
                n_estimators=self.test_config.n_estimators,
                max_depth=self.test_config.max_depth,
                learning_rate=self.test_config.learning_rate,
                reg_alpha=self.test_config.reg_alpha,
                reg_lambda=self.test_config.reg_lambda,
                subsample=self.test_config.subsample,
                colsample_bytree=self.test_config.colsample_bytree,
                min_child_weight=self.test_config.min_child_weight,
                gamma=self.test_config.gamma,
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method='hist',
                verbosity=0
            )
        else:
            return super()._get_regression_model(model_type)
    
    def get_test_spike_weights(self, y_values):
        """Generate spike weights for testing."""
        y_array = np.array(y_values)
        weights = np.ones_like(y_array, dtype=float)
        
        if self.test_config.spike_strategy == 'none':
            return weights
        elif self.test_config.spike_strategy == 'simple':
            weights[y_array > self.test_config.spike_threshold] *= self.test_config.spike_weight
        elif self.test_config.spike_strategy == 'dual':
            weights[y_array > self.test_config.spike_threshold] *= self.test_config.spike_weight
            weights[y_array > self.test_config.extreme_threshold] *= self.test_config.extreme_weight
        elif self.test_config.spike_strategy == 'progressive':
            weights[y_array > 5.0] *= 2.0
            weights[y_array > 10.0] *= 3.0
            weights[y_array > self.test_config.spike_threshold] *= self.test_config.spike_weight
            weights[y_array > self.test_config.extreme_threshold] *= self.test_config.extreme_weight
        
        return weights


def test_single_config(test_config: TestConfig, n_anchors_per_site: int = 75) -> Dict[str, Any]:
    """Test a single configuration."""
    print(f"Testing: n_est={test_config.n_estimators}, depth={test_config.max_depth}, "
          f"lr={test_config.learning_rate}, spike={test_config.spike_strategy}({test_config.spike_weight}), "
          f"lags={test_config.use_lag_features}")
    
    start_time = time.time()
    
    try:
        # Create test engine
        engine = ForecastEngine(validate_on_init=False)
        engine.model_factory = TestModelFactory(test_config)
        engine.min_training_samples = test_config.min_training_samples
        
        # Load data
        engine.data = engine.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        min_target_date = pd.Timestamp("2008-01-01")
        
        # Override config for testing
        original_lag_features = config.LAG_FEATURES
        original_use_lags = config.USE_LAG_FEATURES
        original_horizon = config.FORECAST_HORIZON_DAYS
        
        config.USE_LAG_FEATURES = test_config.use_lag_features
        config.LAG_FEATURES = [1, 2, 3] if test_config.use_lag_features else []
        config.FORECAST_HORIZON_DAYS = test_config.forecast_horizon_days
        
        # Generate anchor points
        anchor_infos = []
        for site in engine.data["site"].unique():
            site_dates = engine.data[engine.data["site"] == site]["date"].sort_values().unique()
            if len(site_dates) > 1:
                date_span_days = (site_dates[-1] - site_dates[0]).days
                if date_span_days >= test_config.forecast_horizon_days * 2:
                    valid_anchors = []
                    for i, date in enumerate(site_dates[:-1]):
                        if date >= min_target_date:
                            future_dates = site_dates[i+1:]
                            valid_future = [d for d in future_dates if (d - date).days >= test_config.forecast_horizon_days]
                            if valid_future:
                                valid_anchors.append(date)
                    
                    if valid_anchors:
                        n_sample = min(len(valid_anchors), n_anchors_per_site)
                        selected_anchors = random.sample(list(valid_anchors), n_sample)
                        anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
        
        if not anchor_infos:
            # Restore config
            config.USE_LAG_FEATURES = original_use_lags
            config.LAG_FEATURES = original_lag_features
            config.FORECAST_HORIZON_DAYS = original_horizon
            return {'r2_score': -999, 'error': 'No valid anchors'}
        
        # Run forecasts with custom spike weighting
        results = []
        for anchor_info in anchor_infos:
            site, anchor_date = anchor_info
            
            site_data = engine.data[engine.data["site"] == site].copy()
            site_data.sort_values("date", inplace=True)
            
            train_mask = site_data["date"] <= anchor_date
            target_forecast_date = anchor_date + pd.Timedelta(days=test_config.forecast_horizon_days)
            test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
            
            train_df = site_data[train_mask].copy()
            test_candidates = site_data[test_mask]
            
            if train_df.empty or test_candidates.empty:
                continue
            
            # Find closest test sample
            test_candidates = test_candidates.copy()
            test_candidates['date_diff'] = abs((test_candidates['date'] - target_forecast_date).dt.days)
            closest_idx = test_candidates['date_diff'].idxmin()
            test_df = test_candidates.loc[[closest_idx]].copy()
            test_date = test_df["date"].iloc[0]
            
            # Create lag features if enabled
            if test_config.use_lag_features:
                site_data_with_lags = engine.data_processor.create_lag_features_safe(
                    site_data, "site", "da", config.LAG_FEATURES, anchor_date
                )
            else:
                site_data_with_lags = site_data
            
            train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].copy()
            test_df = site_data_with_lags[site_data_with_lags["date"] == test_date].copy()
            
            if train_df.empty or test_df.empty:
                continue
            
            train_df_clean = train_df.dropna(subset=["da"]).copy()
            if train_df_clean.empty or len(train_df_clean) < test_config.min_training_samples:
                continue
            
            # Prepare features
            drop_cols = ["date", "site", "da"]
            transformer, X_train = engine.data_processor.create_numeric_transformer(train_df_clean, drop_cols)
            X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            
            X_train_processed = transformer.fit_transform(X_train)
            X_test_processed = transformer.transform(X_test)
            
            # Train model with test spike weighting
            model = engine.model_factory.get_model("regression", "xgboost")
            y_train = train_df_clean["da"]
            sample_weights = engine.model_factory.get_test_spike_weights(y_train.values)
            
            model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            pred_da = model.predict(X_test_processed)[0]
            pred_da = max(0.0, float(pred_da))
            
            actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
            
            if actual_da is not None:
                results.append({
                    'actual_da': actual_da,
                    'predicted_da': pred_da,
                    'date': test_date,
                    'site': site
                })
        
        # Restore config
        config.USE_LAG_FEATURES = original_use_lags
        config.LAG_FEATURES = original_lag_features
        config.FORECAST_HORIZON_DAYS = original_horizon
        
        if not results:
            return {'r2_score': -999, 'error': 'No successful forecasts'}
        
        # Calculate metrics
        df = pd.DataFrame(results)
        r2 = r2_score(df['actual_da'], df['predicted_da'])
        mae = mean_absolute_error(df['actual_da'], df['predicted_da'])
        
        # Spike F1
        y_true_binary = (df['actual_da'] > test_config.spike_threshold).astype(int)
        y_pred_binary = (df['predicted_da'] > test_config.spike_threshold).astype(int)
        spike_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        training_time = time.time() - start_time
        
        return {
            'config': {
                'n_estimators': test_config.n_estimators,
                'max_depth': test_config.max_depth,
                'learning_rate': test_config.learning_rate,
                'reg_alpha': test_config.reg_alpha,
                'reg_lambda': test_config.reg_lambda,
                'subsample': test_config.subsample,
                'colsample_bytree': test_config.colsample_bytree,
                'min_child_weight': test_config.min_child_weight,
                'gamma': test_config.gamma,
                'spike_strategy': test_config.spike_strategy,
                'spike_weight': test_config.spike_weight,
                'extreme_weight': test_config.extreme_weight,
                'use_lag_features': test_config.use_lag_features,
                'min_training_samples': test_config.min_training_samples,
                'forecast_horizon_days': test_config.forecast_horizon_days
            },
            'r2_score': r2,
            'mae': mae,
            'spike_f1': spike_f1,
            'training_time': training_time,
            'total_forecasts': len(anchor_infos),
            'successful_forecasts': len(df)
        }
    
    except Exception as e:
        # Restore config on error
        try:
            config.USE_LAG_FEATURES = original_use_lags
            config.LAG_FEATURES = original_lag_features  
            config.FORECAST_HORIZON_DAYS = original_horizon
        except:
            pass
        return {'r2_score': -999, 'error': str(e)}


def generate_test_configurations() -> List[TestConfig]:
    """Generate test configurations."""
    configs = []
    
    # 1. Baseline/current
    configs.append(TestConfig())  # Default config
    
    # 2. Original minimal
    configs.append(TestConfig(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        reg_alpha=0.0, reg_lambda=0.0
    ))
    
    # 3. Test tree counts
    for n_est in [300, 500, 800, 1000, 1200, 1500]:
        configs.append(TestConfig(n_estimators=n_est))
    
    # 4. Test depths
    for depth in [3, 4, 5, 6, 7, 8, 9, 10]:
        configs.append(TestConfig(max_depth=depth))
    
    # 5. Test learning rates
    for lr in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        configs.append(TestConfig(learning_rate=lr))
    
    # 6. Test regularization
    reg_combos = [(0.0, 0.0), (0.1, 0.1), (0.3, 0.5), (0.5, 1.0), (0.8, 1.2), (1.0, 1.5)]
    for alpha, lam in reg_combos:
        configs.append(TestConfig(reg_alpha=alpha, reg_lambda=lam))
    
    # 7. Test spike strategies
    spike_configs = [
        ('none', 1.0, 1.0),
        ('simple', 4.0, 8.0),
        ('simple', 8.0, 15.0), 
        ('simple', 12.0, 20.0),
        ('simple', 15.0, 25.0),
        ('dual', 8.0, 15.0),
        ('dual', 12.0, 20.0),
        ('progressive', 8.0, 15.0),
    ]
    
    for strategy, weight, ext_weight in spike_configs:
        configs.append(TestConfig(
            spike_strategy=strategy, spike_weight=weight, extreme_weight=ext_weight
        ))
    
    # 8. Test lag features
    configs.append(TestConfig(use_lag_features=True))
    
    # 9. Test min training samples
    for min_samples in [3, 5, 8, 10, 15, 20]:
        configs.append(TestConfig(min_training_samples=min_samples))
    
    # 10. Test forecast horizons
    for horizon in [3, 7, 14, 21]:
        configs.append(TestConfig(forecast_horizon_days=horizon))
    
    # 11. High-performance combinations
    configs.append(TestConfig(
        n_estimators=1200, max_depth=8, learning_rate=0.05,
        reg_alpha=0.3, reg_lambda=0.8, spike_strategy='dual',
        spike_weight=12.0, extreme_weight=20.0, use_lag_features=True
    ))
    
    configs.append(TestConfig(
        n_estimators=1000, max_depth=6, learning_rate=0.06,
        reg_alpha=0.2, reg_lambda=0.6, spike_strategy='progressive',
        spike_weight=10.0, extreme_weight=18.0, use_lag_features=True,
        min_training_samples=5
    ))
    
    print(f"Generated {len(configs)} test configurations")
    return configs


def run_quick_optimization():
    """Run quick optimization."""
    print("Starting Quick DATect XGBoost Optimization")
    print("=" * 60)
    
    configs = generate_test_configurations()
    results = []
    
    total = len(configs)
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{total}] ", end="")
        result = test_single_config(config, n_anchors_per_site=75)
        results.append(result)
        
        if result['r2_score'] > -900:
            print(f"✓ R²={result['r2_score']:.4f}, MAE={result['mae']:.2f}, F1={result['spike_f1']:.4f}")
        else:
            print(f"✗ {result.get('error', 'Failed')}")
    
    # Save results
    results_file = Path("quick_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {results_file}")
    
    # Find best configurations
    valid_results = [r for r in results if r['r2_score'] > -900]
    if not valid_results:
        print("No valid results found!")
        return
    
    valid_results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS:")
    print("=" * 60)
    
    for i, result in enumerate(valid_results[:10]):
        config = result['config']
        print(f"\n{i+1}. R²={result['r2_score']:.4f}, MAE={result['mae']:.2f}, F1={result['spike_f1']:.4f}")
        print(f"   XGBoost: n_est={config['n_estimators']}, depth={config['max_depth']}, lr={config['learning_rate']}")
        print(f"   Regularization: alpha={config['reg_alpha']}, lambda={config['reg_lambda']}")
        print(f"   Spike: {config['spike_strategy']} weight={config['spike_weight']}")
        print(f"   Lags: {config['use_lag_features']}, Min samples: {config['min_training_samples']}")
        print(f"   Time: {result['training_time']:.1f}s, Forecasts: {result['successful_forecasts']}")
    
    print(f"\n\nBest R² score: {valid_results[0]['r2_score']:.4f}")
    print("\nTo implement the best config, update your model_factory.py with the parameters shown above!")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    random.seed(42)
    np.random.seed(42)
    
    run_quick_optimization()