#!/usr/bin/env python3
"""
Comprehensive DATect Optimization
=================================

Tests ALL configurable parameters across the entire codebase that could affect XGBoost performance:
- XGBoost hyperparameters
- Spike weighting strategies  
- Data preprocessing settings
- Feature engineering options
- Temporal parameters
- Training data requirements
- Category thresholds
- Forecast horizons

This goes far beyond just model settings to optimize the entire pipeline.
"""

import pandas as pd
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from pathlib import Path
import itertools
import copy

# Import all the modules we need to modify
import config as config_module
from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
from forecasting.data_processor import DataProcessor

from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer


@dataclass
class ComprehensiveConfig:
    """Complete configuration covering ALL tunable parameters."""
    
    # === XGBoost Model Parameters ===
    n_estimators: int = 800
    max_depth: int = 6
    learning_rate: float = 0.08
    reg_alpha: float = 0.3
    reg_lambda: float = 0.5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    max_delta_step: int = 0
    
    # === Spike Weighting Strategy ===
    spike_strategy: str = 'simple'  # 'none', 'simple', 'dual', 'progressive', 'exponential'
    spike_threshold: float = 20.0
    spike_weight: float = 8.0
    extreme_threshold: Optional[float] = 40.0
    extreme_weight: Optional[float] = 15.0
    
    # === Data Preprocessing ===
    scaler_type: str = 'minmax'  # 'minmax', 'standard', 'robust', 'quantile', 'none'
    imputation_strategy: str = 'median'  # 'mean', 'median', 'most_frequent', 'constant'
    outlier_removal: bool = False
    outlier_threshold: float = 3.0  # Standard deviations
    
    # === Feature Engineering ===
    use_lag_features: bool = False
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3])
    use_derived_features: bool = False  # Moving averages, differences, etc.
    use_seasonal_features: bool = False  # Month, season indicators
    use_trend_features: bool = False  # Linear trend over training window
    
    # === Temporal Parameters ===
    forecast_horizon_days: int = 7  # 1-28 days
    min_training_samples: int = 3  # 3-20 minimum samples
    temporal_buffer_days: int = 0  # Extra temporal safety buffer
    
    # === Category Thresholds ===
    category_bins: List[float] = field(default_factory=lambda: [-float('inf'), 5, 20, 40, float('inf')])
    
    # === Training Data Selection ===
    max_training_window_days: Optional[int] = None  # Limit training window
    training_data_sampling: str = 'all'  # 'all', 'balanced', 'recent_weighted'
    
    # === Cross-Validation ===
    use_temporal_cv: bool = False
    cv_folds: int = 3
    
    # === Early Stopping ===
    use_early_stopping: bool = False
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
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
            'scaler_type': self.scaler_type,
            'imputation_strategy': self.imputation_strategy,
            'outlier_removal': self.outlier_removal,
            'outlier_threshold': self.outlier_threshold,
            'use_lag_features': self.use_lag_features,
            'lag_features': self.lag_features,
            'use_derived_features': self.use_derived_features,
            'use_seasonal_features': self.use_seasonal_features,
            'use_trend_features': self.use_trend_features,
            'forecast_horizon_days': self.forecast_horizon_days,
            'min_training_samples': self.min_training_samples,
            'temporal_buffer_days': self.temporal_buffer_days,
            'category_bins': self.category_bins,
            'max_training_window_days': self.max_training_window_days,
            'training_data_sampling': self.training_data_sampling,
            'use_temporal_cv': self.use_temporal_cv,
            'cv_folds': self.cv_folds,
            'use_early_stopping': self.use_early_stopping,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction
        }


class EnhancedDataProcessor(DataProcessor):
    """Extended DataProcessor with additional preprocessing options."""
    
    def __init__(self, config: ComprehensiveConfig):
        super().__init__()
        self.config = config
        
    def create_enhanced_features(self, df, group_cols, target_col, anchor_date=None):
        """Create enhanced features based on configuration."""
        result_df = df.copy()
        
        # Ensure temporal safety
        if anchor_date is not None:
            result_df = result_df[result_df['date'] <= anchor_date]
        
        # Lag features
        if self.config.use_lag_features and self.config.lag_features:
            result_df = self.create_lag_features_safe(
                result_df, group_cols[0], target_col, self.config.lag_features, anchor_date
            )
        
        # Derived features from lags
        if self.config.use_derived_features and self.config.use_lag_features:
            lag_cols = [f'{target_col}_lag_{lag}' for lag in self.config.lag_features if f'{target_col}_lag_{lag}' in result_df.columns]
            
            if len(lag_cols) >= 2:
                # Moving average of available lags
                result_df[f'{target_col}_lag_mean'] = result_df[lag_cols].mean(axis=1)
                
                # Week-over-week change
                if f'{target_col}_lag_1' in lag_cols and f'{target_col}_lag_2' in lag_cols:
                    result_df[f'{target_col}_weekly_change'] = result_df[f'{target_col}_lag_1'] - result_df[f'{target_col}_lag_2']
                
                # Trend indicator
                if len(lag_cols) >= 3:
                    result_df[f'{target_col}_trending_up'] = (
                        (result_df[f'{target_col}_lag_1'] > result_df[f'{target_col}_lag_2']) & 
                        (result_df[f'{target_col}_lag_2'] > result_df[f'{target_col}_lag_3'])
                    ).astype(int)
        
        # Seasonal features
        if self.config.use_seasonal_features:
            result_df['month'] = pd.to_datetime(result_df['date']).dt.month
            result_df['season'] = pd.to_datetime(result_df['date']).dt.month % 12 // 3
            
            # Cyclical encoding for month
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        
        # Trend features
        if self.config.use_trend_features:
            # Create time trend feature (days since first observation)
            min_date = result_df['date'].min()
            result_df['days_since_start'] = (pd.to_datetime(result_df['date']) - min_date).dt.days
        
        return result_df
    
    def create_enhanced_transformer(self, train_df, drop_cols):
        """Create transformer with enhanced preprocessing options."""
        from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in drop_cols]
        numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            return None, train_df[feature_cols]
        
        # Choose scaler based on configuration
        if self.config.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            scaler = RobustScaler()
        elif self.config.scaler_type == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.config.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:  # 'none'
            scaler = None
        
        # Build pipeline
        numeric_pipeline_steps = [
            ('imputer', SimpleImputer(strategy=self.config.imputation_strategy))
        ]
        
        if scaler is not None:
            numeric_pipeline_steps.append(('scaler', scaler))
        
        numeric_pipeline = Pipeline(numeric_pipeline_steps)
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features)
            ],
            remainder='drop'
        )
        
        X_features = train_df[feature_cols]
        
        # Outlier removal if configured
        if self.config.outlier_removal:
            # Remove outliers based on Z-score
            z_scores = np.abs((X_features.select_dtypes(include=[np.number]) - 
                             X_features.select_dtypes(include=[np.number]).mean()) / 
                             X_features.select_dtypes(include=[np.number]).std())
            outlier_mask = (z_scores < self.config.outlier_threshold).all(axis=1)
            X_features = X_features[outlier_mask]
        
        return preprocessor, X_features


class EnhancedModelFactory(ModelFactory):
    """Enhanced ModelFactory with comprehensive configuration options."""
    
    def __init__(self, config: ComprehensiveConfig):
        super().__init__()
        self.config = config
    
    def _get_regression_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed")
            
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'min_child_weight': self.config.min_child_weight,
                'gamma': self.config.gamma,
                'max_delta_step': self.config.max_delta_step,
                'random_state': self.random_seed,
                'n_jobs': -1,
                'tree_method': 'hist',
                'verbosity': 0
            }
            
            # Early stopping configuration
            if self.config.use_early_stopping:
                params['early_stopping_rounds'] = self.config.early_stopping_rounds
            
            return xgb.XGBRegressor(**params)
        else:
            return super()._get_regression_model(model_type)
    
    def get_enhanced_spike_weights(self, y_values):
        """Generate spike weights according to comprehensive configuration."""
        y_array = np.array(y_values)
        weights = np.ones_like(y_array, dtype=float)
        
        if self.config.spike_strategy == 'none':
            return weights
        
        elif self.config.spike_strategy == 'simple':
            spike_mask = y_array > self.config.spike_threshold
            weights[spike_mask] *= self.config.spike_weight
        
        elif self.config.spike_strategy == 'dual':
            spike_mask = y_array > self.config.spike_threshold
            weights[spike_mask] *= self.config.spike_weight
            
            if self.config.extreme_threshold is not None:
                extreme_mask = y_array > self.config.extreme_threshold
                weights[extreme_mask] *= self.config.extreme_weight
        
        elif self.config.spike_strategy == 'progressive':
            # Progressive weighting based on DA levels
            weights[y_array > 5.0] *= 1.5
            weights[y_array > 10.0] *= 2.0
            weights[y_array > self.config.spike_threshold] *= self.config.spike_weight
            if self.config.extreme_threshold:
                weights[y_array > self.config.extreme_threshold] *= self.config.extreme_weight
        
        elif self.config.spike_strategy == 'exponential':
            # Exponential weighting for extreme values
            median_val = np.median(y_array)
            if median_val > 0:
                normalized = y_array / median_val
                weights *= np.exp(np.maximum(0, normalized - 1) * 0.5)
        
        return weights


class ComprehensiveForecastEngine(ForecastEngine):
    """Comprehensive forecast engine with all optimization options."""
    
    def __init__(self, config: ComprehensiveConfig):
        # Initialize without validation for speed
        super().__init__(validate_on_init=False)
        
        # Override with enhanced components
        self.config = config
        self.data_processor = EnhancedDataProcessor(config)
        self.model_factory = EnhancedModelFactory(config)
        
        # Update training parameters
        self.min_training_samples = config.min_training_samples
        self.forecast_horizon_days = config.forecast_horizon_days
    
    def _forecast_single_anchor_comprehensive(self, anchor_info, full_data, min_target_date, task, model_type):
        """Enhanced single anchor forecast with all optimizations."""
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data["site"] == site].copy()
        site_data.sort_values("date", inplace=True)
        
        # Apply temporal buffer if configured
        effective_anchor_date = anchor_date - pd.Timedelta(days=self.config.temporal_buffer_days)
        train_mask = site_data["date"] <= effective_anchor_date
        
        # Calculate target forecast date with configured horizon
        target_forecast_date = anchor_date + pd.Timedelta(days=self.config.forecast_horizon_days)
        test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
        
        train_df = site_data[train_mask].copy()
        test_candidates = site_data[test_mask]
        
        if train_df.empty or test_candidates.empty:
            return None
        
        # Limit training window if configured
        if self.config.max_training_window_days is not None:
            cutoff_date = effective_anchor_date - pd.Timedelta(days=self.config.max_training_window_days)
            train_df = train_df[train_df["date"] >= cutoff_date]
        
        # Find closest test sample
        test_candidates = test_candidates.copy()
        test_candidates['date_diff'] = abs((test_candidates['date'] - target_forecast_date).dt.days)
        closest_idx = test_candidates['date_diff'].idxmin()
        test_df = test_candidates.loc[[closest_idx]].copy()
        test_date = test_df["date"].iloc[0]
        
        # Enhanced feature engineering
        site_data_enhanced = self.data_processor.create_enhanced_features(
            site_data, ["site"], "da", anchor_date
        )
        
        train_df = site_data_enhanced[site_data_enhanced["date"] <= effective_anchor_date].copy()
        test_df = site_data_enhanced[site_data_enhanced["date"] == test_date].copy()
        
        if train_df.empty or test_df.empty:
            return None
        
        train_df_clean = train_df.dropna(subset=["da"]).copy()
        if train_df_clean.empty or len(train_df_clean) < self.config.min_training_samples:
            return None
        
        # Create categories with configured bins
        train_df_clean["da-category"] = pd.cut(
            train_df_clean["da"], 
            bins=self.config.category_bins, 
            labels=range(len(self.config.category_bins)-1)
        ).astype(int)
        
        # Training data sampling strategy
        if self.config.training_data_sampling == 'balanced':
            # Balance classes by undersampling majority class
            min_class_size = train_df_clean["da-category"].value_counts().min()
            if min_class_size > 0:
                balanced_dfs = []
                for category in train_df_clean["da-category"].unique():
                    cat_df = train_df_clean[train_df_clean["da-category"] == category]
                    if len(cat_df) > min_class_size:
                        cat_df = cat_df.sample(n=min_class_size, random_state=42)
                    balanced_dfs.append(cat_df)
                train_df_clean = pd.concat(balanced_dfs, ignore_index=True)
        
        elif self.config.training_data_sampling == 'recent_weighted':
            # Weight more recent samples higher
            train_df_clean = train_df_clean.copy()
            max_date = train_df_clean['date'].max()
            days_from_max = (max_date - train_df_clean['date']).dt.days
            recency_weights = np.exp(-days_from_max / 30.0)  # Exponential decay over ~30 days
            train_df_clean['recency_weight'] = recency_weights
        
        # Enhanced preprocessing
        base_drop_cols = ["date", "site", "da", "da-category"]
        if 'recency_weight' in train_df_clean.columns:
            base_drop_cols.append('recency_weight')
        
        transformer, X_train = self.data_processor.create_enhanced_transformer(train_df_clean, base_drop_cols)
        
        if transformer is None:
            return None
        
        test_drop_cols = ["date", "site", "da"]
        X_test = test_df.drop(columns=[col for col in test_drop_cols if col in test_df.columns])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
        actual_category = pd.cut([actual_da], bins=self.config.category_bins, 
                               labels=range(len(self.config.category_bins)-1))[0] if actual_da is not None else None
        
        result = {
            'date': test_date,
            'site': site,
            'anchor_date': anchor_date,
            'actual_da': actual_da,
            'actual_category': int(actual_category) if actual_category is not None else None
        }
        
        if task == "regression":
            model = self.model_factory.get_model("regression", model_type)
            y_train = train_df_clean["da"]
            
            # Enhanced spike weighting
            sample_weights = self.model_factory.get_enhanced_spike_weights(y_train.values)
            
            # Apply recency weights if configured
            if self.config.training_data_sampling == 'recent_weighted' and 'recency_weight' in train_df_clean.columns:
                sample_weights *= train_df_clean['recency_weight'].values
            
            if model_type in ["xgboost", "xgb"]:
                # Use validation set for early stopping if configured
                if self.config.use_early_stopping:
                    val_size = int(len(X_train_processed) * self.config.validation_fraction)
                    if val_size > 0:
                        val_indices = np.random.choice(len(X_train_processed), val_size, replace=False)
                        train_indices = np.setdiff1d(range(len(X_train_processed)), val_indices)
                        
                        X_val = X_train_processed[val_indices]
                        y_val = y_train.iloc[val_indices]
                        X_train_es = X_train_processed[train_indices]
                        y_train_es = y_train.iloc[train_indices]
                        weights_es = sample_weights[train_indices]
                        
                        model.fit(X_train_es, y_train_es, sample_weight=weights_es,
                                eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        model.fit(X_train_processed, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train)
            
            pred_da = model.predict(X_test_processed)[0]
            pred_da = max(0.0, float(pred_da))
            result['predicted_da'] = pred_da
        
        return pd.DataFrame([result])


def generate_comprehensive_configurations() -> List[ComprehensiveConfig]:
    """Generate comprehensive test configurations covering all parameters."""
    
    configurations = []
    
    # 1. Baseline configurations
    baseline = ComprehensiveConfig()  # Default configuration
    configurations.append(baseline)
    
    # Original minimal settings
    minimal = ComprehensiveConfig(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        reg_alpha=0.0, reg_lambda=0.0, spike_weight=8.0
    )
    configurations.append(minimal)
    
    # 2. XGBoost parameter exploration
    for n_est in [300, 500, 800, 1200, 1500]:
        config = copy.deepcopy(baseline)
        config.n_estimators = n_est
        configurations.append(config)
    
    for depth in [3, 4, 6, 8, 10]:
        config = copy.deepcopy(baseline)
        config.max_depth = depth
        configurations.append(config)
    
    for lr in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]:
        config = copy.deepcopy(baseline)
        config.learning_rate = lr
        configurations.append(config)
    
    # 3. Spike weighting strategies
    spike_configs = [
        ('none', 20.0, 1.0, None, None),
        ('simple', 20.0, 4.0, None, None),
        ('simple', 20.0, 8.0, None, None),
        ('simple', 20.0, 15.0, None, None),
        ('dual', 20.0, 8.0, 40.0, 15.0),
        ('dual', 20.0, 12.0, 40.0, 20.0),
        ('progressive', 20.0, 8.0, 40.0, 15.0),
        ('exponential', 20.0, 8.0, None, None),
    ]
    
    for strategy, threshold, weight, ext_threshold, ext_weight in spike_configs:
        config = copy.deepcopy(baseline)
        config.spike_strategy = strategy
        config.spike_threshold = threshold
        config.spike_weight = weight
        config.extreme_threshold = ext_threshold
        config.extreme_weight = ext_weight
        configurations.append(config)
    
    # 4. Data preprocessing options
    for scaler in ['minmax', 'standard', 'robust', 'quantile', 'none']:
        config = copy.deepcopy(baseline)
        config.scaler_type = scaler
        configurations.append(config)
    
    for imputation in ['mean', 'median', 'most_frequent']:
        config = copy.deepcopy(baseline)
        config.imputation_strategy = imputation
        configurations.append(config)
    
    # 5. Feature engineering options
    # Lag features
    for use_lags in [True, False]:
        for lags in [[1, 2], [1, 2, 3], [1, 2, 3, 4]]:
            config = copy.deepcopy(baseline)
            config.use_lag_features = use_lags
            config.lag_features = lags if use_lags else []
            configurations.append(config)
    
    # Derived features
    config = copy.deepcopy(baseline)
    config.use_lag_features = True
    config.use_derived_features = True
    configurations.append(config)
    
    # Seasonal features
    config = copy.deepcopy(baseline)
    config.use_seasonal_features = True
    configurations.append(config)
    
    # 6. Temporal parameters
    for horizon in [3, 7, 14, 21]:
        config = copy.deepcopy(baseline)
        config.forecast_horizon_days = horizon
        configurations.append(config)
    
    for min_samples in [3, 5, 8, 10, 15]:
        config = copy.deepcopy(baseline)
        config.min_training_samples = min_samples
        configurations.append(config)
    
    # 7. Training data strategies
    for sampling in ['all', 'balanced', 'recent_weighted']:
        config = copy.deepcopy(baseline)
        config.training_data_sampling = sampling
        configurations.append(config)
    
    # 8. Category threshold variations
    alt_bins = [
        [-float('inf'), 3, 15, 30, float('inf')],  # More sensitive
        [-float('inf'), 10, 25, 50, float('inf')], # Less sensitive
        [-float('inf'), 5, 15, 25, 40, float('inf')],  # 5 categories
    ]
    
    for bins in alt_bins:
        config = copy.deepcopy(baseline)
        config.category_bins = bins
        configurations.append(config)
    
    # 9. Advanced combinations
    # High-performance configuration
    high_perf = ComprehensiveConfig(
        n_estimators=1200, max_depth=8, learning_rate=0.05,
        reg_alpha=0.3, reg_lambda=0.8, subsample=0.9,
        spike_strategy='dual', spike_weight=12.0, extreme_weight=20.0,
        use_lag_features=True, lag_features=[1, 2, 3, 4],
        use_derived_features=True, use_seasonal_features=True,
        scaler_type='robust', min_training_samples=5,
        training_data_sampling='recent_weighted'
    )
    configurations.append(high_perf)
    
    # Feature-rich configuration
    feature_rich = ComprehensiveConfig(
        use_lag_features=True, lag_features=[1, 2, 3, 4],
        use_derived_features=True, use_seasonal_features=True,
        use_trend_features=True, scaler_type='quantile',
        spike_strategy='progressive'
    )
    configurations.append(feature_rich)
    
    print(f"Generated {len(configurations)} comprehensive configurations")
    return configurations


def test_comprehensive_configuration(config: ComprehensiveConfig, n_anchors_per_site: int = 50) -> Dict[str, Any]:
    """Test a comprehensive configuration."""
    print(f"Testing comprehensive config: n_est={config.n_estimators}, "
          f"scaler={config.scaler_type}, spike={config.spike_strategy}, "
          f"lags={config.use_lag_features}")
    
    start_time = time.time()
    
    try:
        # Create comprehensive engine
        engine = ComprehensiveForecastEngine(config)
        engine.data = engine.data_processor.load_and_prepare_base_data(config_module.FINAL_OUTPUT_PATH)
        min_target_date = pd.Timestamp("2008-01-01")
        
        # Generate anchor points
        anchor_infos = []
        for site in engine.data["site"].unique():
            site_dates = engine.data[engine.data["site"] == site]["date"].sort_values().unique()
            if len(site_dates) > 1:
                date_span_days = (site_dates[-1] - site_dates[0]).days
                if date_span_days >= config.forecast_horizon_days * 2:
                    valid_anchors = []
                    for i, date in enumerate(site_dates[:-1]):
                        if date >= min_target_date:
                            future_dates = site_dates[i+1:]
                            valid_future = [d for d in future_dates if (d - date).days >= config.forecast_horizon_days]
                            if valid_future:
                                valid_anchors.append(date)
                    
                    if valid_anchors:
                        n_sample = min(len(valid_anchors), n_anchors_per_site)
                        selected_anchors = random.sample(list(valid_anchors), n_sample)
                        anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
        
        if not anchor_infos:
            return {'config': config.to_dict(), 'r2_score': -999, 'error': 'No valid anchors'}
        
        # Run forecasts
        results = []
        for anchor_info in anchor_infos:
            result = engine._forecast_single_anchor_comprehensive(
                anchor_info, engine.data, min_target_date, "regression", "xgboost"
            )
            if result is not None:
                results.append(result)
        
        if not results:
            return {'config': config.to_dict(), 'r2_score': -999, 'error': 'No successful forecasts'}
        
        # Combine and evaluate
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.dropna(subset=['actual_da', 'predicted_da'])
        
        if final_df.empty:
            return {'config': config.to_dict(), 'r2_score': -999, 'error': 'No valid predictions'}
        
        # Calculate metrics
        r2 = r2_score(final_df['actual_da'], final_df['predicted_da'])
        mae = mean_absolute_error(final_df['actual_da'], final_df['predicted_da'])
        
        # Spike detection F1
        y_true_binary = (final_df['actual_da'] > config.spike_threshold).astype(int)
        y_pred_binary = (final_df['predicted_da'] > config.spike_threshold).astype(int)
        spike_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        training_time = time.time() - start_time
        
        return {
            'config': config.to_dict(),
            'r2_score': r2,
            'mae': mae,
            'spike_f1': spike_f1,
            'training_time': training_time,
            'total_forecasts': len(anchor_infos),
            'successful_forecasts': len(final_df)
        }
    
    except Exception as e:
        return {
            'config': config.to_dict(), 
            'r2_score': -999, 
            'error': str(e),
            'training_time': time.time() - start_time
        }


def run_comprehensive_optimization():
    """Run comprehensive optimization of all parameters."""
    print("Starting Comprehensive DATect Optimization")
    print("Testing XGBoost + Data Processing + Feature Engineering + Temporal Parameters")
    print("=" * 80)
    
    configurations = generate_comprehensive_configurations()
    results = []
    
    total_configs = len(configurations)
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{total_configs}] ", end="")
        result = test_comprehensive_configuration(config, n_anchors_per_site=50)
        results.append(result)
        
        if result['r2_score'] > -900:
            print(f"✓ R²={result['r2_score']:.4f}, MAE={result['mae']:.2f}, F1={result['spike_f1']:.4f}")
        else:
            print(f"✗ {result.get('error', 'Failed')}")
    
    # Save comprehensive results
    results_file = Path("comprehensive_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {results_file}")
    
    # Analyze results
    valid_results = [r for r in results if r['r2_score'] > -900]
    if not valid_results:
        print("No valid results found!")
        return
    
    valid_results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    print("\n" + "=" * 80)
    print("TOP 5 COMPREHENSIVE CONFIGURATIONS:")
    print("=" * 80)
    
    for i, result in enumerate(valid_results[:5]):
        config = result['config']
        print(f"\n{i+1}. R²={result['r2_score']:.4f}, MAE={result['mae']:.2f}, F1={result['spike_f1']:.4f}")
        print(f"   XGBoost: n_est={config['n_estimators']}, depth={config['max_depth']}, lr={config['learning_rate']}")
        print(f"   Preprocessing: scaler={config['scaler_type']}, impute={config['imputation_strategy']}")
        print(f"   Features: lags={config['use_lag_features']}, derived={config['use_derived_features']}, seasonal={config['use_seasonal_features']}")
        print(f"   Spike: strategy={config['spike_strategy']}, weight={config['spike_weight']}")
        print(f"   Temporal: horizon={config['forecast_horizon_days']}d, min_samples={config['min_training_samples']}")
        print(f"   Training: sampling={config['training_data_sampling']}")
        print(f"   Time: {result['training_time']:.1f}s")
    
    print(f"\n\nComprehensive optimization complete! Best R²: {valid_results[0]['r2_score']:.4f}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    random.seed(42)
    np.random.seed(42)
    
    run_comprehensive_optimization()