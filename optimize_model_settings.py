#!/usr/bin/env python3
"""
XGBoost Hyperparameter Optimization for DATect
===============================================

Tests a comprehensive grid of XGBoost settings, spike weighting strategies,
and other parameters to find optimal configuration for algal bloom forecasting.

Usage: python optimize_model_settings.py
"""

import pandas as pd
import numpy as np
import random
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time
import json
from pathlib import Path

from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
from forecasting.data_processor import DataProcessor
import config

from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score


@dataclass
class ModelConfig:
    """Configuration for a single model test."""
    # XGBoost hyperparameters
    n_estimators: int
    max_depth: int
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    gamma: float
    max_delta_step: int
    
    # Spike weighting strategy
    spike_strategy: str
    spike_threshold: float
    spike_weight: float
    extreme_threshold: Optional[float] = None
    extreme_weight: Optional[float] = None
    
    # Other settings
    use_early_stopping: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'max_delta_step': self.max_delta_step,
            'spike_strategy': self.spike_strategy,
            'spike_threshold': self.spike_threshold,
            'spike_weight': self.spike_weight,
            'extreme_threshold': self.extreme_threshold,
            'extreme_weight': self.extreme_weight,
            'use_early_stopping': self.use_early_stopping
        }


@dataclass
class TestResult:
    """Result from testing a single configuration."""
    config: ModelConfig
    r2_score: float
    mae: float
    spike_f1: float
    accuracy: float  # For classification
    training_time: float
    total_forecasts: int
    successful_forecasts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': self.config.to_dict(),
            'r2_score': self.r2_score,
            'mae': self.mae,
            'spike_f1': self.spike_f1,
            'accuracy': self.accuracy,
            'training_time': self.training_time,
            'total_forecasts': self.total_forecasts,
            'successful_forecasts': self.successful_forecasts
        }


class OptimizedModelFactory(ModelFactory):
    """Extended ModelFactory that can be configured with test parameters."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.test_config = config
    
    def _get_regression_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed")
            
            import xgboost as xgb
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
                max_delta_step=self.test_config.max_delta_step,
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method='hist',
                verbosity=0
            )
        else:
            return super()._get_regression_model(model_type)
    
    def _get_classification_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed")
            
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=self.test_config.n_estimators,
                max_depth=self.test_config.max_depth,
                learning_rate=self.test_config.learning_rate,
                reg_alpha=self.test_config.reg_alpha,
                reg_lambda=self.test_config.reg_lambda,
                subsample=self.test_config.subsample,
                colsample_bytree=self.test_config.colsample_bytree,
                min_child_weight=self.test_config.min_child_weight,
                gamma=self.test_config.gamma,
                max_delta_step=self.test_config.max_delta_step,
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method='hist',
                verbosity=0,
                eval_metric='logloss'
            )
        else:
            return super()._get_classification_model(model_type)
    
    def get_test_spike_weights(self, y_values):
        """Generate spike weights according to test configuration."""
        y_array = np.array(y_values)
        weights = np.ones_like(y_array, dtype=float)
        
        if self.test_config.spike_strategy == 'none':
            return weights
        
        elif self.test_config.spike_strategy == 'simple':
            spike_mask = y_array > self.test_config.spike_threshold
            weights[spike_mask] *= self.test_config.spike_weight
        
        elif self.test_config.spike_strategy == 'dual_threshold':
            spike_mask = y_array > self.test_config.spike_threshold
            weights[spike_mask] *= self.test_config.spike_weight
            
            if self.test_config.extreme_threshold is not None:
                extreme_mask = y_array > self.test_config.extreme_threshold
                weights[extreme_mask] *= self.test_config.extreme_weight
        
        elif self.test_config.spike_strategy == 'percentile':
            p90 = np.percentile(y_array, 90)
            p95 = np.percentile(y_array, 95)
            p99 = np.percentile(y_array, 99)
            
            weights[y_array > p90] *= 2.0
            weights[y_array > p95] *= self.test_config.spike_weight
            weights[y_array > p99] *= self.test_config.spike_weight * 1.5
        
        elif self.test_config.spike_strategy == 'progressive':
            # Progressive weighting based on DA levels
            weights[y_array > 5.0] *= 1.5
            weights[y_array > 10.0] *= 2.0
            weights[y_array > 20.0] *= self.test_config.spike_weight
            weights[y_array > 40.0] *= self.test_config.spike_weight * 1.5
        
        return weights


class OptimizedForecastEngine(ForecastEngine):
    """Extended ForecastEngine that uses optimized model factory."""
    
    def __init__(self, test_config: ModelConfig):
        super().__init__(validate_on_init=False)
        self.model_factory = OptimizedModelFactory(test_config)
        self.test_config = test_config
    
    def _forecast_single_anchor_optimized(self, anchor_info, full_data, min_target_date, task, model_type):
        """Optimized version of single anchor forecast with custom spike weighting."""
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data["site"] == site].copy()
        site_data.sort_values("date", inplace=True)
        
        train_mask = site_data["date"] <= anchor_date
        target_forecast_date = anchor_date + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
        
        train_df = site_data[train_mask].copy()
        test_candidates = site_data[test_mask]
        
        if train_df.empty or test_candidates.empty:
            return None
        
        # Find closest test sample
        test_candidates = test_candidates.copy()
        test_candidates['date_diff'] = abs((test_candidates['date'] - target_forecast_date).dt.days)
        closest_idx = test_candidates['date_diff'].idxmin()
        test_df = test_candidates.loc[[closest_idx]].copy()
        test_date = test_df["date"].iloc[0]
        
        # Create lag features
        site_data_with_lags = self.data_processor.create_lag_features_safe(
            site_data, "site", "da", config.LAG_FEATURES, anchor_date
        )
        
        train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].copy()
        test_df = site_data_with_lags[site_data_with_lags["date"] == test_date].copy()
        
        if train_df.empty or test_df.empty:
            return None
        
        train_df_clean = train_df.dropna(subset=["da"]).copy()
        if train_df_clean.empty or len(train_df_clean) < self.min_training_samples:
            return None
        
        train_df_clean["da-category"] = self.data_processor.create_da_categories_safe(train_df_clean["da"])
        
        base_drop_cols = ["date", "site", "da"]
        train_drop_cols = base_drop_cols + ["da-category"]
        test_drop_cols = base_drop_cols
        
        transformer, X_train = self.data_processor.create_numeric_transformer(train_df_clean, train_drop_cols)
        X_test = test_df.drop(columns=[col for col in test_drop_cols if col in test_df.columns])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Validate temporal safety
        self.data_processor.validate_transformer_temporal_safety(
            transformer, train_df_clean, test_df, anchor_date
        )
        
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        if pd.isna(train_df_clean["da"]).any():
            return None
        
        actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
        actual_category = self.data_processor.create_da_categories_safe(pd.Series([actual_da]))[0] if actual_da is not None else None
        
        result = {
            'date': test_date,
            'site': site,
            'anchor_date': anchor_date,
            'actual_da': actual_da,
            'actual_category': actual_category
        }
        
        if task == "regression":
            reg_model = self.model_factory.get_model("regression", model_type)
            y_train = train_df_clean["da"]
            
            # Use test-specific spike weighting
            sample_weights = self.model_factory.get_test_spike_weights(y_train.values)
            
            if model_type in ["xgboost", "xgb"]:
                reg_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                reg_model.fit(X_train_processed, y_train)
            
            pred_da = reg_model.predict(X_test_processed)[0]
            pred_da = max(0.0, float(pred_da))
            result['predicted_da'] = pred_da
        
        elif task == "classification":
            unique_classes = train_df_clean["da-category"].nunique()
            if unique_classes > 1:
                unique_cats = sorted(train_df_clean["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                y_train_encoded = train_df_clean["da-category"].map(cat_mapping)
                cls_model = self.model_factory.get_model("classification", model_type)
                
                if model_type in ["xgboost", "xgb"]:
                    # Use simple class weights for classification
                    class_counts = train_df_clean["da-category"].value_counts()
                    total_samples = len(train_df_clean)
                    sample_weights = np.array([total_samples / (len(class_counts) * class_counts[cat]) 
                                             for cat in train_df_clean["da-category"].values])
                    cls_model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights)
                else:
                    cls_model.fit(X_train_processed, y_train_encoded)
                
                pred_encoded = cls_model.predict(X_test_processed)[0]
                pred_category = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(pred_category)
            else:
                dominant_class = train_df_clean["da-category"].mode()[0]
                result['predicted_category'] = int(dominant_class)
        
        return pd.DataFrame([result])


def generate_test_configurations() -> List[ModelConfig]:
    """Generate comprehensive test configurations from minimal to maximal settings."""
    
    # Define parameter ranges
    n_estimators_range = [300, 500, 800, 1000, 1200, 1500, 2000]  # From minimal to maximal
    max_depth_range = [3, 4, 6, 8, 10, 12]  # From conservative to aggressive
    learning_rate_range = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]  # From slow to fast
    reg_alpha_range = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]  # From none to high
    reg_lambda_range = [0.0, 0.1, 0.5, 1.0, 1.2, 1.5, 2.0]  # From none to high
    subsample_range = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]  # From aggressive to conservative
    colsample_range = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]  # From aggressive to conservative
    min_child_weight_range = [1, 2, 3, 5, 8, 10]  # From permissive to restrictive
    gamma_range = [0.0, 0.05, 0.1, 0.2, 0.5]  # From none to high
    max_delta_step_range = [0, 1, 2, 5]  # From none to high constraint
    
    # Spike weighting strategies
    spike_strategies = [
        ('none', 20.0, 1.0, None, None),  # No weighting
        ('simple', 20.0, 4.0, None, None),  # Light spike emphasis
        ('simple', 20.0, 8.0, None, None),  # Original spike emphasis
        ('simple', 20.0, 12.0, None, None),  # Strong spike emphasis
        ('simple', 20.0, 20.0, None, None),  # Very strong spike emphasis
        ('dual_threshold', 20.0, 8.0, 40.0, 15.0),  # Dual threshold moderate
        ('dual_threshold', 20.0, 12.0, 40.0, 25.0),  # Dual threshold strong
        ('percentile', 20.0, 8.0, None, None),  # Percentile-based
        ('progressive', 20.0, 8.0, None, None),  # Progressive weighting
    ]
    
    configurations = []
    
    # Generate a strategic sample rather than full grid (would be millions of combinations)
    
    # 1. Original/baseline configurations
    configurations.extend([
        # Original minimal settings
        ModelConfig(300, 8, 0.1, 0.0, 0.0, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0),
        # Current settings  
        ModelConfig(800, 6, 0.08, 0.3, 0.5, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0),
        # Failed "optimized" settings
        ModelConfig(1500, 8, 0.05, 0.8, 1.2, 0.85, 0.9, 3, 0.1, 2, 'dual_threshold', 20.0, 12.0, 40.0, 25.0),
    ])
    
    # 2. Systematic exploration of key parameters
    # Test different tree counts with fixed other params
    for n_est in [300, 500, 800, 1000, 1500]:
        configurations.append(ModelConfig(n_est, 6, 0.08, 0.3, 0.5, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0))
    
    # Test different depths
    for depth in [3, 4, 6, 8, 10]:
        configurations.append(ModelConfig(800, depth, 0.08, 0.3, 0.5, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0))
    
    # Test different learning rates
    for lr in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]:
        configurations.append(ModelConfig(800, 6, lr, 0.3, 0.5, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0))
    
    # Test different regularization levels
    for reg_alpha, reg_lambda in [(0.0, 0.0), (0.1, 0.1), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5)]:
        configurations.append(ModelConfig(800, 6, 0.08, reg_alpha, reg_lambda, 0.8, 0.8, 1, 0.0, 0, 'simple', 20.0, 8.0))
    
    # Test different spike weighting strategies
    base_config_params = (800, 6, 0.08, 0.3, 0.5, 0.8, 0.8, 1, 0.0, 0)
    for strategy, threshold, weight, ext_threshold, ext_weight in spike_strategies:
        configurations.append(ModelConfig(*base_config_params, strategy, threshold, weight, ext_threshold, ext_weight))
    
    # 3. Random combinations for broader exploration
    random.seed(42)
    for _ in range(20):  # Add 20 random combinations
        config = ModelConfig(
            n_estimators=random.choice(n_estimators_range),
            max_depth=random.choice(max_depth_range),
            learning_rate=random.choice(learning_rate_range),
            reg_alpha=random.choice(reg_alpha_range),
            reg_lambda=random.choice(reg_lambda_range),
            subsample=random.choice(subsample_range),
            colsample_bytree=random.choice(colsample_range),
            min_child_weight=random.choice(min_child_weight_range),
            gamma=random.choice(gamma_range),
            max_delta_step=random.choice(max_delta_step_range),
            spike_strategy=random.choice(spike_strategies)[0],
            spike_threshold=random.choice(spike_strategies)[1],
            spike_weight=random.choice(spike_strategies)[2],
            extreme_threshold=random.choice(spike_strategies)[3],
            extreme_weight=random.choice(spike_strategies)[4]
        )
        configurations.append(config)
    
    print(f"Generated {len(configurations)} test configurations")
    return configurations


def test_single_configuration(test_config: ModelConfig, n_anchors_per_site: int = 100) -> TestResult:
    """Test a single configuration and return performance metrics."""
    print(f"Testing: n_est={test_config.n_estimators}, depth={test_config.max_depth}, "
          f"lr={test_config.learning_rate}, spike={test_config.spike_strategy}({test_config.spike_weight})")
    
    start_time = time.time()
    
    try:
        # Create optimized engine with test configuration
        engine = OptimizedForecastEngine(test_config)
        
        # Load data
        engine.data = engine.data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
        min_target_date = pd.Timestamp("2008-01-01")
        
        # Generate anchor points (reduced for speed)
        anchor_infos = []
        for site in engine.data["site"].unique():
            site_dates = engine.data[engine.data["site"] == site]["date"].sort_values().unique()
            if len(site_dates) > 1:
                date_span_days = (site_dates[-1] - site_dates[0]).days
                if date_span_days >= config.FORECAST_HORIZON_DAYS * 2:
                    valid_anchors = []
                    for i, date in enumerate(site_dates[:-1]):
                        if date >= min_target_date:
                            future_dates = site_dates[i+1:]
                            valid_future = [d for d in future_dates if (d - date).days >= config.FORECAST_HORIZON_DAYS]
                            if valid_future:
                                valid_anchors.append(date)
                    
                    if valid_anchors:
                        n_sample = min(len(valid_anchors), n_anchors_per_site)
                        selected_anchors = random.sample(list(valid_anchors), n_sample)
                        anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
        
        if not anchor_infos:
            return TestResult(test_config, -999, 999, 0, 0, time.time() - start_time, 0, 0)
        
        # Run forecasts
        results = []
        for anchor_info in anchor_infos:
            result = engine._forecast_single_anchor_optimized(
                anchor_info, engine.data, min_target_date, "regression", "xgboost"
            )
            if result is not None:
                results.append(result)
        
        if not results:
            return TestResult(test_config, -999, 999, 0, 0, time.time() - start_time, len(anchor_infos), 0)
        
        # Combine results
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.dropna(subset=['actual_da', 'predicted_da'])
        
        if final_df.empty:
            return TestResult(test_config, -999, 999, 0, 0, time.time() - start_time, len(anchor_infos), 0)
        
        # Calculate metrics
        r2 = r2_score(final_df['actual_da'], final_df['predicted_da'])
        mae = mean_absolute_error(final_df['actual_da'], final_df['predicted_da'])
        
        # Spike detection F1 (20 μg/g threshold)
        spike_threshold = 20.0
        y_true_binary = (final_df['actual_da'] > spike_threshold).astype(int)
        y_pred_binary = (final_df['predicted_da'] > spike_threshold).astype(int)
        spike_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # For classification-like accuracy, use category-based accuracy
        actual_cats = pd.cut(final_df['actual_da'], bins=[-np.inf, 5, 20, 40, np.inf], labels=[0, 1, 2, 3])
        pred_cats = pd.cut(final_df['predicted_da'], bins=[-np.inf, 5, 20, 40, np.inf], labels=[0, 1, 2, 3])
        accuracy = accuracy_score(actual_cats, pred_cats)
        
        training_time = time.time() - start_time
        
        return TestResult(
            test_config, r2, mae, spike_f1, accuracy, training_time, 
            len(anchor_infos), len(final_df)
        )
    
    except Exception as e:
        print(f"Error testing configuration: {e}")
        return TestResult(test_config, -999, 999, 0, 0, time.time() - start_time, 0, 0)


def run_optimization():
    """Run the complete hyperparameter optimization."""
    print("Starting DATect XGBoost Hyperparameter Optimization")
    print("=" * 60)
    
    # Generate test configurations
    configurations = generate_test_configurations()
    
    # Test each configuration
    results = []
    total_configs = len(configurations)
    
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{total_configs}] ", end="")
        result = test_single_configuration(config, n_anchors_per_site=100)
        results.append(result)
        
        if result.r2_score > -900:  # Valid result
            print(f"✓ R²={result.r2_score:.4f}, MAE={result.mae:.2f}, "
                  f"F1={result.spike_f1:.4f}, Time={result.training_time:.1f}s")
        else:
            print("✗ Failed")
    
    # Save results
    results_file = Path("optimization_results.json")
    results_data = [result.to_dict() for result in results]
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n\nResults saved to {results_file}")
    
    # Find best configurations
    valid_results = [r for r in results if r.r2_score > -900]
    if not valid_results:
        print("No valid results found!")
        return
    
    # Sort by R²
    valid_results.sort(key=lambda x: x.r2_score, reverse=True)
    
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS BY R²:")
    print("=" * 60)
    
    for i, result in enumerate(valid_results[:10]):
        config = result.config
        print(f"\n{i+1}. R²={result.r2_score:.4f}, MAE={result.mae:.2f}, F1={result.spike_f1:.4f}")
        print(f"   n_estimators={config.n_estimators}, max_depth={config.max_depth}, "
              f"learning_rate={config.learning_rate}")
        print(f"   reg_alpha={config.reg_alpha}, reg_lambda={config.reg_lambda}")
        print(f"   subsample={config.subsample}, colsample_bytree={config.colsample_bytree}")
        print(f"   spike_strategy={config.spike_strategy}, spike_weight={config.spike_weight}")
        print(f"   Time: {result.training_time:.1f}s, Forecasts: {result.successful_forecasts}")
    
    # Best by spike F1
    valid_results.sort(key=lambda x: x.spike_f1, reverse=True)
    best_spike = valid_results[0]
    
    print(f"\n\nBEST FOR SPIKE DETECTION (F1={best_spike.spike_f1:.4f}):")
    print(f"R²={best_spike.r2_score:.4f}, MAE={best_spike.mae:.2f}")
    config = best_spike.config
    print(f"n_estimators={config.n_estimators}, max_depth={config.max_depth}, lr={config.learning_rate}")
    print(f"reg_alpha={config.reg_alpha}, reg_lambda={config.reg_lambda}")
    print(f"spike_strategy={config.spike_strategy}, spike_weight={config.spike_weight}")
    
    print(f"\n\nOptimization complete! Check {results_file} for full results.")


if __name__ == "__main__":
    # Fix issues with the import
    import sys
    sys.path.append('.')
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    run_optimization()