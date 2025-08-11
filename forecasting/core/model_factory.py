"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import config


class ModelFactory:
    """
    Factory class for creating configured ML models.
    
    Supported Models:
    - Random Forest (regression & classification) - PRIMARY MODEL
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
            model_type: "rf", "linear", or "logistic"
            
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
        if model_type == "rf" or model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                max_features='sqrt',
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif model_type == "xgboost" or model_type == "xgb":
            # XGBoost deprecated in favor of Random Forest
            raise ValueError("XGBoost deprecated. Use 'rf' or 'linear' instead.")
        elif model_type == "ridge":
            # Ridge deprecated in favor of LinearRegression
            raise ValueError("Ridge deprecated. Use 'linear' for Linear Regression instead.")
        elif model_type == "linear":
            return LinearRegression(
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'rf', 'linear')")
            
    def _get_classification_model(self, model_type):
        """Get classification model.""" 
        if model_type == "rf" or model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                max_features='sqrt',
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif model_type == "xgboost" or model_type == "xgb":
            # XGBoost deprecated in favor of Random Forest
            raise ValueError("XGBoost deprecated. Use 'rf' or 'logistic' instead.")
        elif model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
                n_jobs=-1  # Use all CPU cores for fair comparison with Random Forest
            )
        else:
            raise ValueError(f"Unknown classification model: {model_type}. "
                           f"Supported: 'rf', 'logistic')")
            
    def get_supported_models(self, task=None):
        """
        Get list of supported models for given task.
        
        Args:
            task: "regression", "classification", or None for all
            
        Returns:
            Dictionary of supported models by task
        """
        models = {
            "regression": ["rf", "linear"],
            "classification": ["rf", "logistic"]
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
            "rf": "Random Forest",
            "random_forest": "Random Forest",
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