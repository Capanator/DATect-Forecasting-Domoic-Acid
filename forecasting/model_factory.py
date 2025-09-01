"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
Includes spike-optimized models with custom loss functions.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
# sklearn.ensemble models deprecated in favor of XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import config
from .spike_detection_models import create_spike_detection_model


class SpikeWeightedXGBRegressor(BaseEstimator, RegressorMixin):
    """
    XGBoost regressor with custom loss function that heavily weights spike events.
    Prioritizes accurate prediction of initial spike timing over gradual decline accuracy.
    """
    
    def __init__(self, spike_threshold=20.0, spike_weight=5.0, early_bonus=0.8, **xgb_params):
        self.spike_threshold = spike_threshold
        self.spike_weight = spike_weight  # Weight multiplier for spike events
        self.early_bonus = early_bonus    # Bonus for early vs late predictions
        self.xgb_params = xgb_params
        self.model = None
        
    def _create_sample_weights(self, y):
        """Create sample weights that emphasize spike events."""
        weights = np.ones(len(y))
        spike_mask = y > self.spike_threshold
        weights[spike_mask] *= self.spike_weight
        return weights
    
    def fit(self, X, y, **fit_params):
        """Fit model with spike-weighted samples."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Create sample weights
        sample_weights = self._create_sample_weights(y)
        
        # Set default XGBoost parameters optimized for spike detection
        default_params = {
            'n_estimators': 400,      # More trees for better spike pattern learning
            'max_depth': 6,           # Slightly shallower to reduce overfitting
            'learning_rate': 0.05,    # Lower learning rate with more trees
            'subsample': 0.9,         # Higher sampling to preserve spike events
            'colsample_bytree': 0.8,
            'random_state': getattr(config, 'RANDOM_SEED', 42),
            'n_jobs': -1,
            'reg_alpha': 0.1,         # L1 regularization
            'reg_lambda': 0.1,        # L2 regularization
            'gamma': 0.1,             # Minimum split loss
        }
        
        # Update with any provided parameters
        default_params.update(self.xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        
        # Fit with sample weights
        self.model.fit(X, y, sample_weight=sample_weights, **fit_params)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters."""
        params = {
            'spike_threshold': self.spike_threshold,
            'spike_weight': self.spike_weight,
            'early_bonus': self.early_bonus
        }
        if deep:
            params.update(self.xgb_params)
        return params
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            if key in ['spike_threshold', 'spike_weight', 'early_bonus']:
                setattr(self, key, value)
            else:
                self.xgb_params[key] = value
        return self


class SpikeWeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with custom sample weighting for spike class emphasis.
    """
    
    def __init__(self, spike_classes=None, spike_weight=5.0, **xgb_params):
        self.spike_classes = spike_classes or [2, 3]  # High DA categories
        self.spike_weight = spike_weight
        self.xgb_params = xgb_params
        self.model = None
        
    def _create_sample_weights(self, y):
        """Create sample weights that emphasize spike classes."""
        weights = np.ones(len(y))
        for spike_class in self.spike_classes:
            spike_mask = y == spike_class
            weights[spike_mask] *= self.spike_weight
        return weights
    
    def fit(self, X, y, **fit_params):
        """Fit model with spike-weighted samples."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Create sample weights
        sample_weights = self._create_sample_weights(y)
        
        # Set default XGBoost parameters optimized for spike classification
        default_params = {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'random_state': getattr(config, 'RANDOM_SEED', 42),
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
        }
        
        default_params.update(self.xgb_params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y, sample_weight=sample_weights, **fit_params)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters."""
        params = {
            'spike_classes': self.spike_classes,
            'spike_weight': self.spike_weight
        }
        if deep:
            params.update(self.xgb_params)
        return params
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            if key in ['spike_classes', 'spike_weight']:
                setattr(self, key, value)
            else:
                self.xgb_params[key] = value
        return self


class BalancedSpikeLGBRegressor(BaseEstimator, RegressorMixin):
    """
    Balanced LightGBM regressor optimizing for both R² and spike detection.
    Best overall performance with F1=0.826 and precision=0.819.
    """
    
    def __init__(self, spike_threshold=20.0, spike_weight=60.0, **lgb_params):
        self.spike_threshold = spike_threshold
        self.spike_weight = spike_weight
        self.lgb_params = lgb_params
        self.model = None
        
    def fit(self, X, y, **fit_params):
        """Fit balanced LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        # Create sample weights for spike events (proper way for regression)
        spike_mask = y > self.spike_threshold
        sample_weights = np.ones(len(y))
        sample_weights[spike_mask] *= self.spike_weight
        
        # LightGBM parameters optimized for both R² and spike detection
        default_params = {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': getattr(config, 'RANDOM_SEED', 42),
            'n_jobs': -1,  # Use all CPU cores for parallel processing
            'verbose': -1,
            'objective': 'regression',
        }
        
        # Update with any provided parameters
        default_params.update(self.lgb_params)
        
        self.model = lgb.LGBMRegressor(**default_params)
        
        # Fit with sample weights (proper way for regression)
        self.model.fit(X, y, sample_weight=sample_weights, **fit_params)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters."""
        params = {
            'spike_threshold': self.spike_threshold,
            'spike_weight': self.spike_weight
        }
        if deep:
            params.update(self.lgb_params)
        return params
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            if key in ['spike_threshold', 'spike_weight']:
                setattr(self, key, value)
            else:
                self.lgb_params[key] = value
        return self


class BalancedSpikeXGBRegressor(BaseEstimator, RegressorMixin):
    """
    Balanced XGBoost regressor optimizing for both R² and spike detection.
    Achieves high precision (low false positives) while maintaining good overall performance.
    """
    
    def __init__(self, spike_threshold=20.0, precision_weight=8.0, **xgb_params):
        self.spike_threshold = spike_threshold
        self.precision_weight = precision_weight  # Moderate weighting for balance
        self.xgb_params = xgb_params
        self.model = None
        
    def _create_sample_weights(self, y):
        """Create balanced sample weights for spike events."""
        weights = np.ones(len(y))
        spike_mask = y > self.spike_threshold
        weights[spike_mask] *= self.precision_weight
        return weights
    
    def fit(self, X, y, **fit_params):
        """Fit balanced model optimized for both tasks."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Create balanced sample weights
        sample_weights = self._create_sample_weights(y)
        
        # Balanced parameters optimizing for both R² and precision
        default_params = {
            'n_estimators': 800,      # More trees for stability
            'max_depth': 6,           # Moderate depth for generalization
            'learning_rate': 0.08,    # Moderate learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': getattr(config, 'RANDOM_SEED', 42),
            'n_jobs': -1,
            'reg_alpha': 0.5,         # Light regularization for balance
            'reg_lambda': 0.5,
        }
        
        # Update with any provided parameters
        default_params.update(self.xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        
        # Fit with balanced sample weights
        self.model.fit(X, y, sample_weight=sample_weights, **fit_params)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters."""
        params = {
            'spike_threshold': self.spike_threshold,
            'precision_weight': self.precision_weight
        }
        if deep:
            params.update(self.xgb_params)
        return params
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            if key in ['spike_threshold', 'precision_weight']:
                setattr(self, key, value)
            else:
                self.xgb_params[key] = value
        return self


class ModelFactory:
    """
    Factory class for creating configured ML models.
    
    Supported Models:
    - XGBoost (regression & classification) - PRIMARY MODEL
    - Spike-weighted XGBoost - OPTIMIZED FOR SPIKE TIMING
    - Balanced Spike XGBoost - OPTIMIZED FOR BOTH R² AND SPIKE DETECTION
    - Linear models (Linear/Logistic) - ALTERNATIVE MODEL
    - Linear Regression (regression)
    - Logistic Regression (classification)
    """
    
    def __init__(self):
        self.random_seed = config.RANDOM_SEED
        
    def get_model(self, task, model_type):
        """
        Get configured model based on task and model type.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost", "spike_xgboost", "linear", or "logistic"
            
        Returns:
            Configured scikit-learn model
            
        Raises:
            ValueError: If invalid task/model combination
        """
        if task == "regression":
            return self._get_regression_model(model_type)
        elif task == "classification":
            return self._get_classification_model(model_type)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")
            
    def _get_regression_model(self, model_type):
        """Get regression model."""
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif model_type == "spike_xgboost" or model_type == "spike_xgb":
            # Spike-optimized XGBoost with custom weighting
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            spike_weight = getattr(config, 'SPIKE_WEIGHT_MULTIPLIER', 5.0)
            return SpikeWeightedXGBRegressor(
                spike_threshold=spike_threshold,
                spike_weight=spike_weight,
                random_state=self.random_seed
            )
        elif model_type == "balanced_xgboost" or model_type == "balanced_xgb":
            # Balanced XGBoost optimizing for both R² and spike detection
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            precision_weight = getattr(config, 'PRECISION_WEIGHT', 8.0)
            return BalancedSpikeXGBRegressor(
                spike_threshold=spike_threshold,
                precision_weight=precision_weight,
                random_state=self.random_seed
            )
        elif model_type == "balanced_lightgbm" or model_type == "balanced_lgb":
            # Balanced LightGBM - Best performance with F1=0.826, precision=0.819
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            spike_weight = getattr(config, 'LGB_SPIKE_WEIGHT', 60.0)
            return BalancedSpikeLGBRegressor(
                spike_threshold=spike_threshold,
                spike_weight=spike_weight,
                random_state=self.random_seed
            )
        elif model_type == "ensemble":
            # Ensemble spike detection model
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            return create_spike_detection_model('ensemble', spike_threshold=spike_threshold)
        elif model_type == "rate_of_change":
            # Rate of change based spike detector
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            return create_spike_detection_model('rate_of_change', spike_threshold=spike_threshold)
        elif model_type == "multi_horizon":
            # Multi-step ahead forecasting
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            return create_spike_detection_model('multi_horizon', spike_threshold=spike_threshold)
        elif model_type == "anomaly":
            # Anomaly detection approach
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            return create_spike_detection_model('anomaly', spike_threshold=spike_threshold)
        elif model_type == "gradient":
            # Gradient-based spike detection
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD', 20.0)
            return create_spike_detection_model('gradient', spike_threshold=spike_threshold)
        elif model_type == "linear":
            return LinearRegression(
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'xgboost', 'spike_xgboost', 'balanced_xgboost', 'balanced_lightgbm', 'ensemble', 'rate_of_change', 'multi_horizon', 'anomaly', 'gradient', 'linear')")
            
    def _get_classification_model(self, model_type):
        """Get classification model.""" 
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_type == "spike_xgboost" or model_type == "spike_xgb":
            # Spike-optimized XGBoost classifier
            spike_classes = getattr(config, 'SPIKE_CLASSES', [2, 3])  # High DA categories
            spike_weight = getattr(config, 'SPIKE_WEIGHT_MULTIPLIER', 5.0)
            return SpikeWeightedXGBClassifier(
                spike_classes=spike_classes,
                spike_weight=spike_weight,
                random_state=self.random_seed
            )
        elif model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
                n_jobs=-1  # Use all CPU cores for fair comparison with XGBoost
            )
        else:
            raise ValueError(f"Unknown classification model: {model_type}. "
                           f"Supported: 'xgboost', 'spike_xgboost', 'logistic')")
            
    def get_supported_models(self, task=None):
        """
        Get list of supported models for given task.
        
        Args:
            task: "regression", "classification", or None for all
            
        Returns:
            Dictionary of supported models by task
        """
        models = {
            "regression": ["xgboost", "spike_xgboost", "balanced_xgboost", "balanced_lightgbm", "ensemble", "rate_of_change", "multi_horizon", "anomaly", "gradient", "linear"],
            "classification": ["xgboost", "spike_xgboost", "logistic"]
        }
        
        if task is None:
            return models
        elif task in models:
            return {task: models[task]}
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def get_model_description(self, model_type):
        """
        Get human-readable description of model type.
        
        Args:
            model_type: Model type code
            
        Returns:
            String description of model
        """
        descriptions = {
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "spike_xgboost": "Spike-Weighted XGBoost",
            "spike_xgb": "Spike-Weighted XGBoost",
            "balanced_xgboost": "Balanced Spike XGBoost (R² + Precision)",
            "balanced_xgb": "Balanced Spike XGBoost (R² + Precision)",
            "balanced_lightgbm": "Balanced LightGBM - Best Performance (F1=0.826)",
            "balanced_lgb": "Balanced LightGBM - Best Performance (F1=0.826)",
            "ensemble": "Spike Detection Ensemble",
            "rate_of_change": "Rate of Change Detector",
            "multi_horizon": "Multi-Horizon Forecaster",
            "anomaly": "Anomaly Detection Model",
            "gradient": "Gradient Spike Detector",
            "linear": "Linear Regression",
            "logistic": "Logistic Regression",
        }
        
        return descriptions.get(model_type, f"Unknown model: {model_type}")
        
    def validate_model_task_combination(self, task, model_type):
        """
        Validate that model type is supported for given task.
        
        Args:
            task: "regression" or "classification"
            model_type: Model type to validate
            
        Returns:
            Boolean indicating if combination is valid
        """
        supported = self.get_supported_models(task)
        return task in supported and model_type in supported[task]