"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
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
        """
        Get configured model based on task and model type.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost", "linear", or "logistic"
            
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
                n_estimators=800,        # was 300
                max_depth=6,             # was 8  
                learning_rate=0.08,      # was 0.1
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,           # NEW
                reg_lambda=0.5,          # NEW
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
                           f"Supported: 'xgboost', 'logistic')")
            
    def get_supported_models(self, task=None):
        """
        Get list of supported models for given task.
        
        Args:
            task: "regression", "classification", or None for all
            
        Returns:
            Dictionary of supported models by task
        """
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