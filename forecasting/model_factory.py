"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
Optimized for harmful algal bloom spike detection with advanced XGBoost tuning.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import numpy as np
import config


class ModelFactory:
    """
    Factory class for creating configured ML models.
    
    Supported Models:
    - XGBoost (regression & classification) - PRIMARY MODEL
    - Linear models (Linear/Logistic) - ALTERNATIVE MODEL
    - Linear Regression (regression)
    - Logistic Regression (classification)
    """
    
    def __init__(self):
        self.random_seed = config.RANDOM_SEED
        # XGBoost performance optimization settings
        self.xgb_common_params = {
            'random_state': self.random_seed,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'grow_policy': 'lossguide',  # Better for imbalanced data
            'max_leaves': 128,  # Optimal for spike detection
            'verbosity': 0
        }
        
    def get_model(self, task, model_type):
        if task == "regression":
            return self._get_regression_model(model_type)
        elif task == "classification":
            return self._get_classification_model(model_type)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")
            
    def _get_regression_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(
                # Conservative settings that actually work
                n_estimators=800,               # Original working value
                max_depth=6,                    # Original working depth
                learning_rate=0.08,             # Original working rate
                
                # Moderate regularization
                reg_alpha=0.3,                  # Light L1 regularization
                reg_lambda=0.5,                 # Light L2 regularization
                gamma=0.0,                      # No minimum split loss
                
                # Standard sampling
                subsample=0.8,                  # Original working value
                colsample_bytree=0.8,           # Original working value
                
                # Standard settings
                min_child_weight=1,             # Default
                max_delta_step=0,               # Default (no constraint)
                
                # Performance optimization
                **self.xgb_common_params
            )
        elif model_type == "linear":
            return LinearRegression(
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'xgboost', 'linear')")
            
    def _get_classification_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBClassifier(
                # Conservative settings that actually work
                n_estimators=800,               # Original working value
                max_depth=6,                    # Original working depth  
                learning_rate=0.08,             # Original working rate
                
                # Moderate regularization
                reg_alpha=0.3,                  # Light L1 regularization
                reg_lambda=0.5,                 # Light L2 regularization
                gamma=0.0,                      # No minimum split loss
                
                # Standard sampling
                subsample=0.8,                  # Original working value
                colsample_bytree=0.8,           # Original working value
                
                # Standard settings
                min_child_weight=1,             # Default
                max_delta_step=0,               # Default
                scale_pos_weight=None,          # Default
                
                # Performance settings
                eval_metric='logloss',          # Standard metric
                **self.xgb_common_params
            )
        elif model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classification model: {model_type}. "
                           f"Supported: 'xgboost', 'logistic')")
            
    def get_supported_models(self, task=None):
        models = {
            "regression": ["xgboost", "linear"],
            "classification": ["xgboost", "logistic"]
        }
        
        if task is None:
            return models
        elif task in models:
            return {task: models[task]}
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def get_model_description(self, model_type):
        descriptions = {
            "xgboost": "XGBoost (Optimized for Spike Detection)",
            "xgb": "XGBoost (Optimized for Spike Detection)",
            "linear": "Linear Regression",
            "logistic": "Logistic Regression",
        }
        
        return descriptions.get(model_type, f"Unknown model: {model_type}")
        
    def validate_model_task_combination(self, task, model_type):
        supported = self.get_supported_models(task)
        return task in supported and model_type in supported[task]
    
    def get_advanced_spike_weights(self, y_values, strategy='conservative'):
        """Generate advanced sample weights optimized for XGBoost spike detection."""
        y_array = np.array(y_values)
        weights = np.ones_like(y_array, dtype=float)
        
        if strategy == 'conservative':
            # Simple, effective spike weighting that actually works
            spike_20 = y_array > 20.0  # Federal limit (most important)
            spike_40 = y_array > 40.0  # Extreme risk
            
            weights[spike_20] *= 6.0   # Reasonable emphasis on spikes
            weights[spike_40] *= 12.0  # Strong focus on extreme spikes
            
        elif strategy == 'original':
            # Revert to exactly the original simple approach
            spike_mask = y_array > 20.0
            weights[spike_mask] *= 8.0
            
        elif strategy == 'moderate':
            # Slightly better than original but not crazy
            moderate = y_array > 5.0   # Moderate risk
            high = y_array > 20.0      # Federal limit  
            extreme = y_array > 40.0   # Extreme risk
            
            weights[moderate] *= 2.0   # Light emphasis
            weights[high] *= 8.0       # Original spike weight
            weights[extreme] *= 15.0   # Stronger for extreme
            
        return weights
    
    def get_class_weights_for_imbalanced_classification(self, y_categories):
        """Calculate conservative class weights for XGBoost spike classification."""
        from collections import Counter
        
        # Count class frequencies
        class_counts = Counter(y_categories)
        total_samples = len(y_categories)
        n_classes = len(class_counts)
        
        # Calculate balanced class weights (not too extreme)
        class_weights = {}
        for class_label, count in class_counts.items():
            # Simple inverse frequency weighting
            base_weight = total_samples / (n_classes * count)
            
            # Moderate importance multipliers (much more conservative)
            if class_label == 0:    # Low (≤5 μg/g): Safe
                multiplier = 1.0
            elif class_label == 1:  # Moderate (5-20 μg/g): Caution
                multiplier = 1.2    # Slight increase
            elif class_label == 2:  # High (20-40 μg/g): Federal limit exceeded
                multiplier = 1.5    # Moderate increase
            elif class_label == 3:  # Extreme (>40 μg/g): Emergency level
                multiplier = 2.0    # Double weight (not 8x!)
            else:
                multiplier = 1.0
                
            class_weights[class_label] = base_weight * multiplier
        
        return class_weights