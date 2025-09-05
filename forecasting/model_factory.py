"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import config
from .logging_config import get_logger

logger = get_logger(__name__)


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
        
    def get_model(self, task, model_type, class_weights=None):
        if task == "regression":
            return self._get_regression_model(model_type)
        elif task == "classification":
            return self._get_classification_model(model_type, class_weights)
        elif task == "spike_detection":
            return self._get_spike_detection_model(model_type)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression', 'classification', or 'spike_detection'")
            
    def _get_regression_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=400,           # Increased for better accuracy
                max_depth=6,                # Deeper trees for complex patterns  
                learning_rate=0.05,         # Lower learning rate for stability
                subsample=0.85,             # Slightly higher for more data
                colsample_bytree=0.85,      # More features per tree
                colsample_bylevel=0.8,      # Column sampling by level
                reg_alpha=0.1,              # L1 regularization  
                reg_lambda=1.0,             # L2 regularization
                gamma=0.1,                  # Minimum split loss
                min_child_weight=3,         # More conservative splits
                tree_method='hist',
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif model_type == "linear":
            return LinearRegression(
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'xgboost', 'linear')")
            
    def _get_classification_model(self, model_type, class_weights=None):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            
            # Configure XGBoost with optimized hyperparameters for accuracy
            xgb_params = {
                'n_estimators': 500,        # More trees for better accuracy
                'max_depth': 7,             # Deeper for complex DA patterns
                'learning_rate': 0.03,      # Lower for stable convergence
                'subsample': 0.9,           # More data per tree
                'colsample_bytree': 0.9,    # More features per tree
                'colsample_bylevel': 0.8,   # Column sampling by level
                'reg_alpha': 0.1,           # L1 regularization
                'reg_lambda': 2.0,          # Strong L2 regularization
                'gamma': 0.2,               # Minimum split loss
                'min_child_weight': 5,      # Conservative splits for imbalanced data
                'tree_method': 'hist',
                'random_state': self.random_seed,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            
            # Add class balancing if weights provided
            if class_weights is not None:
                # Convert class weights to XGBoost scale_pos_weight format
                # For multi-class, we use sample weights during training instead
                pass
            else:
                # Use balanced mode for automatic class weighting
                xgb_params['class_weight'] = 'balanced' if hasattr(xgb.XGBClassifier(), 'class_weight') else None
                
            return xgb.XGBClassifier(**xgb_params)
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
                           
    def _get_spike_detection_model(self, model_type):
        """
        Get model optimized specifically for binary spike detection.
        Focus on high recall (not missing spikes) over precision.
        """
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            
            # Optimized for spike detection with high recall
            return xgb.XGBClassifier(
                n_estimators=600,        # More trees for better spike detection
                max_depth=8,             # Deeper for complex spike patterns
                learning_rate=0.02,      # Very conservative learning for stability
                subsample=0.9,           
                colsample_bytree=0.9,    
                colsample_bylevel=0.8,   
                reg_alpha=0.01,          # Light L1 regularization
                reg_lambda=1.5,          # Moderate L2 regularization
                gamma=0.1,               # Conservative minimum split loss
                min_child_weight=1,      # Allow small splits for rare spikes
                tree_method='hist',
                random_state=self.random_seed,
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic'  # Binary classification
            )
        elif model_type == "logistic":
            return LogisticRegression(
                solver="liblinear",      # Better for binary classification
                max_iter=2000,
                C=10.0,                  # Less regularization for spike detection
                class_weight='balanced', # Handle class imbalance
                random_state=self.random_seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown spike detection model: {model_type}. "
                           f"Supported: 'xgboost', 'logistic')")
            
    def get_supported_models(self, task=None):
        models = {
            "regression": ["xgboost", "linear"],
            "classification": ["xgboost", "logistic"],
            "spike_detection": ["xgboost", "logistic"]
        }
        
        if task is None:
            return models
        elif task in models:
            return {task: models[task]}
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def get_model_description(self, model_type):
        descriptions = {
            "xgboost": "XGBoost",
            "xgb": "XGBoost", 
            "linear": "Linear Regression",
            "logistic": "Logistic Regression"
        }
        
        return descriptions.get(model_type, f"Unknown model: {model_type}")
        
    def compute_sample_weights_for_classification(self, y_train):
        """
        Compute sample weights to handle class imbalance in classification.
        Returns weights that emphasize minority classes (especially extreme events).
        """
        import numpy as np
        
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        
        # Create mapping of class to weight
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Apply weights to each sample
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        # Extra emphasis on extreme events (class 3) for better detection
        extreme_mask = y_train == 3
        if extreme_mask.any():
            sample_weights[extreme_mask] *= 2.0  # Double weight for extreme events
            
        return sample_weights
        
    def compute_spike_focused_weights(self, y_actual, y_predicted_proba=None):
        """
        Compute sample weights specifically for spike detection timing.
        Heavily penalizes false negatives (missed spikes) and moderately penalizes false positives.
        """
        import numpy as np
        
        # Convert to binary spike labels if not already
        if hasattr(config, 'USE_BINARY_SPIKE_DETECTION') and config.USE_BINARY_SPIKE_DETECTION:
            # Assume y_actual is already binary (0 = no spike, 1 = spike)
            actual_spikes = y_actual
        else:
            # Convert from DA values to binary spike indicators
            actual_spikes = (y_actual > config.SPIKE_THRESHOLD).astype(int)
        
        # Initialize weights
        sample_weights = np.ones(len(actual_spikes))
        
        # Spike events get massive weight (focus on not missing these)
        spike_mask = actual_spikes == 1
        sample_weights[spike_mask] = config.SPIKE_FALSE_NEGATIVE_WEIGHT
        
        # Non-spike events get minimal weight (most of the year)
        non_spike_mask = actual_spikes == 0
        sample_weights[non_spike_mask] = config.SPIKE_TRUE_NEGATIVE_WEIGHT
        
        spike_count = spike_mask.sum()
        total_count = len(actual_spikes)
        
        logger.debug(f"Spike-focused weights: {spike_count}/{total_count} spikes with weight {config.SPIKE_FALSE_NEGATIVE_WEIGHT}")
        logger.debug(f"Non-spike samples: {total_count-spike_count} with weight {config.SPIKE_TRUE_NEGATIVE_WEIGHT}")
        
        return sample_weights
        
    def validate_model_task_combination(self, task, model_type):
        supported = self.get_supported_models(task)
        return task in supported and model_type in supported[task]