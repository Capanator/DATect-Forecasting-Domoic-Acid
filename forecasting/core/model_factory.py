"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LogisticRegression
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


class ModelFactory:
    """
    Factory class for creating configured ML models.
    
    Supported Models:
    - Random Forest (regression & classification) 
    - XGBoost (regression & classification) - BEST PERFORMER
    - Stacking Ensemble (regression) - HIGHEST ACCURACY
    - Ridge Regression (regression)
    - Logistic Regression (classification)
    """
    
    def __init__(self):
        self.random_seed = config.RANDOM_SEED
        
    def get_model(self, task, model_type):
        """
        Get configured model based on task and model type.
        
        Args:
            task: "regression" or "classification"
            model_type: "rf", "xgboost", "stacking", "ridge", or "logistic"
            
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
        if model_type == "rf":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_seed,
                n_jobs=1
            )
        elif model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif model_type == "stacking":
            # Stacking ensemble - BEST PERFORMER (8.1% better than RF)
            if not HAS_XGBOOST or not HAS_LIGHTGBM:
                raise ImportError("Stacking requires XGBoost and LightGBM. Run: pip install xgboost lightgbm")
            
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_seed, n_jobs=-1)),
                ('et', ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=self.random_seed, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=self.random_seed, n_jobs=-1)),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=8, random_state=self.random_seed, n_jobs=-1, verbose=-1))
            ]
            
            return StackingRegressor(
                estimators=base_models,
                final_estimator=xgb.XGBRegressor(n_estimators=50, random_state=self.random_seed, n_jobs=-1),
                cv=5,
                n_jobs=-1
            )
        elif model_type == "ridge":
            return Ridge(
                alpha=1.0,
                random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'rf', 'xgboost', 'stacking', 'ridge'")
            
    def _get_classification_model(self, model_type):
        """Get classification model."""
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_seed,
                n_jobs=1
            )
        elif model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown classification model: {model_type}. "
                           f"Supported: 'rf', 'xgboost', 'logistic'")
            
    def get_supported_models(self, task=None):
        """
        Get list of supported models for given task.
        
        Args:
            task: "regression", "classification", or None for all
            
        Returns:
            Dictionary of supported models by task
        """
        models = {
            "regression": ["rf", "xgboost", "stacking", "ridge"],
            "classification": ["rf", "xgboost", "logistic"]
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
            "xgboost": "XGBoost (7.4% better than RF)",
            "xgb": "XGBoost (7.4% better than RF)",
            "stacking": "Stacking Ensemble (8.1% better than RF)",
            "ridge": "Ridge Regression",
            "logistic": "Logistic Regression"
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