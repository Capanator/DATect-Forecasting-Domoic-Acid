"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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
            # Allow overrides from config
            cfg = getattr(config, 'XGB_REGRESSION_PARAMS', None) or {}
            params = {
                # Defaults chosen for responsiveness to sharp onsets
                "n_estimators": 800,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                # Speed/robustness
                "tree_method": "hist",
                "max_bin": 256,
            }
            params.update(cfg)
            # Optional Tweedie objective for heavy-tailed nonnegative targets
            if getattr(config, 'USE_TWEEDIE_REGRESSION', False):
                params["objective"] = "reg:tweedie"
                params["tweedie_variance_power"] = float(getattr(config, 'TWEEDIE_VARIANCE_POWER', 1.3))

            params.update({
                "random_state": self.random_seed,
                "n_jobs": -1,
            })
            return xgb.XGBRegressor(**params)
        elif model_type == "linear":
            # Use Ridge for improved numerical stability on collinear, scaled features
            use_ridge = getattr(config, 'LINEAR_REGRESSION_TYPE', 'ridge') == 'ridge'
            if use_ridge:
                alpha = float(getattr(config, 'RIDGE_ALPHA', 1.0))
                return Ridge(alpha=alpha, random_state=self.random_seed)
            else:
                # Fallback to plain LinearRegression
                try:
                    return LinearRegression(n_jobs=-1)
                except TypeError:
                    return LinearRegression()
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'xgboost', 'linear')")
            
    def _get_classification_model(self, model_type):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            cfg = getattr(config, 'XGB_CLASSIFICATION_PARAMS', None) or {}
            params = {
                "n_estimators": 800,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "eval_metric": 'logloss',
                # Speed/robustness
                "tree_method": "hist",
                "max_bin": 256,
            }
            params.update(cfg)
            params.update({
                "random_state": self.random_seed,
                "n_jobs": -1,
            })
            return xgb.XGBClassifier(**params)
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
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "linear": "Linear Regression",
            "logistic": "Logistic Regression",
        }
        
        return descriptions.get(model_type, f"Unknown model: {model_type}")
        
    def validate_model_task_combination(self, task, model_type):
        supported = self.get_supported_models(task)
        return task in supported and model_type in supported[task]
