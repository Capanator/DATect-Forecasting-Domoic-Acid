import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, log_loss
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to create lag features within pipeline."""
    
    def __init__(self, lags=[1, 2, 3], target_col='da'):
        self.lags = lags
        self.target_col = target_col
        self.lag_values_ = None
    
    def fit(self, X, y=None):
        """Fit by storing the last few values for lag creation."""
        if self.target_col in X.columns:
            # Store last values for creating lags in transform
            self.lag_values_ = X[self.target_col].tail(max(self.lags)).values
        return self
    
    def transform(self, X):
        """Transform by adding lag features."""
        X_transformed = X.copy()
        
        if self.lag_values_ is not None and len(self.lag_values_) > 0:
            for i, lag in enumerate(self.lags):
                lag_idx = len(self.lag_values_) - lag
                if lag_idx >= 0:
                    X_transformed[f'{self.target_col}_lag_{lag}'] = self.lag_values_[lag_idx]
                else:
                    X_transformed[f'{self.target_col}_lag_{lag}'] = np.nan
        else:
            for lag in self.lags:
                X_transformed[f'{self.target_col}_lag_{lag}'] = np.nan
                
        return X_transformed


class DACategorizerTransformer(BaseEstimator, TransformerMixin):
    """Transform continuous DA values to categories using training data distribution."""
    
    def __init__(self):
        self.bins_ = None
        self.labels_ = [0, 1, 2, 3]
    
    def fit(self, X, y=None):
        """Fit using fixed bins to prevent leakage."""
        self.bins_ = [-float('inf'), 5, 20, 40, float('inf')]
        return self
    
    def transform(self, X):
        """Transform DA values to categories."""
        X_transformed = X.copy()
        if 'da' in X_transformed.columns:
            X_transformed['da_category'] = pd.cut(
                X_transformed['da'],
                bins=self.bins_,
                labels=self.labels_,
                right=True
            ).astype('Int64')
        return X_transformed


class TimeSeriesForecaster:
    """Complete forecasting pipeline that prevents data leakage."""
    
    def __init__(self, model_type='random_forest', task='regression', 
                 include_lags=True, random_state=42):
        self.model_type = model_type
        self.task = task
        self.include_lags = include_lags
        self.random_state = random_state
        self.pipeline = None
        self.feature_columns = None
        
    def _get_base_model(self):
        """Get base model based on type and task."""
        models = {
            'random_forest': {
                'regression': RandomForestRegressor(random_state=self.random_state, n_jobs=1),
                'classification': RandomForestClassifier(random_state=self.random_state, n_jobs=1)
            },
            'gradient_boosting': {
                'regression': GradientBoostingRegressor(random_state=self.random_state),
                'classification': None  # Not typically used for classification
            },
            'linear': {
                'regression': LinearRegression(n_jobs=1),
                'classification': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
        }
        
        model = models.get(self.model_type, {}).get(self.task)
        if model is None:
            raise ValueError(f"Invalid model_type={self.model_type} for task={self.task}")
        return model
    
    def _create_preprocessing_pipeline(self):
        """Create preprocessing pipeline."""
        steps = []
        
        # Add temporal features
        steps.append(('temporal', TimeFeatureTransformer()))
        
        # Add lag features if enabled
        if self.include_lags and self.task == 'regression':
            steps.append(('lags', LagFeatureTransformer()))
        
        # Add categorizer for classification tasks
        if self.task == 'classification':
            steps.append(('categorizer', DACategorizerTransformer()))
        
        # Feature selection and scaling
        steps.append(('preprocessor', FeaturePreprocessor()))
        
        return Pipeline(steps)
    
    def create_pipeline(self, **model_params):
        """Create complete modeling pipeline."""
        preprocessing = self._create_preprocessing_pipeline()
        model = self._get_base_model()
        
        if model_params:
            model.set_params(**model_params)
        
        self.pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', model)
        ])
        
        return self.pipeline
    
    def fit(self, X, y):
        """Fit the complete pipeline."""
        if self.pipeline is None:
            self.create_pipeline()
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict_proba(X)


class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add temporal features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if 'date' in X_transformed.columns:
            day_of_year = pd.to_datetime(X_transformed['date']).dt.dayofyear
            X_transformed['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
            X_transformed['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        return X_transformed


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Feature selection and preprocessing."""
    
    def __init__(self):
        self.feature_columns = None
        self.numeric_pipeline = None
        
    def fit(self, X, y=None):
        # Identify feature columns (exclude site, date, targets)
        exclude_cols = ['site', 'date', 'da', 'da_category']
        self.feature_columns = [col for col in X.columns if col not in exclude_cols]
        
        # Create numeric preprocessing pipeline
        self.numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        # Fit on numeric features
        if self.feature_columns:
            numeric_data = X[self.feature_columns].select_dtypes(include=[np.number])
            if not numeric_data.empty:
                self.numeric_pipeline.fit(numeric_data)
        
        return self
    
    def transform(self, X):
        if not self.feature_columns:
            return pd.DataFrame(index=X.index)
        
        # Select and process numeric features
        numeric_data = X[self.feature_columns].select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return pd.DataFrame(index=X.index)
        
        # Transform numeric features
        processed = self.numeric_pipeline.transform(numeric_data)
        
        # Convert back to DataFrame with proper column names
        return pd.DataFrame(
            processed, 
            columns=numeric_data.columns,
            index=X.index
        )


class ModelOptimizer:
    """Hyperparameter optimization with proper time series validation."""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def get_param_grids(self):
        """Get parameter grids for different models."""
        return {
            'random_forest_regression': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 3]
            },
            'random_forest_classification': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 3],
                'model__class_weight': ['balanced', None]
            }
        }
    
    def optimize(self, forecaster: TimeSeriesForecaster, X, y, 
                scoring='r2', param_grid=None):
        """Optimize hyperparameters using time series cross-validation."""
        if param_grid is None:
            grids = self.get_param_grids()
            key = f"{forecaster.model_type}_{forecaster.task}"
            param_grid = grids.get(key, {})
        
        if not param_grid:
            print(f"No parameter grid for {forecaster.model_type}_{forecaster.task}")
            return {}
        
        # Create pipeline
        pipeline = forecaster.create_pipeline()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best {scoring}: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        # Return params without 'model__' prefix
        return {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}


class ForecastEvaluator:
    """Evaluate forecasting performance."""
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """Evaluate regression performance."""
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_proba=None):
        """Evaluate classification performance."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except ValueError:
                metrics['log_loss'] = np.nan
        
        return metrics
    
    @staticmethod
    def coverage_probability(y_true, y_lower, y_upper):
        """Calculate coverage probability for interval predictions."""
        return np.mean((y_true >= y_lower) & (y_true <= y_upper))


class QuantileForecaster:
    """Quantile regression for uncertainty estimation."""
    
    def __init__(self, quantiles=[0.05, 0.5, 0.95], random_state=42):
        self.quantiles = quantiles
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
    
    def fit(self, X, y):
        """Fit quantile models."""
        # Create preprocessor
        self.preprocessor = FeaturePreprocessor()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Fit a model for each quantile
        for quantile in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                random_state=self.random_state,
                n_estimators=100
            )
            model.fit(X_processed, y)
            self.models[quantile] = model
        
        return self
    
    def predict(self, X):
        """Predict quantiles."""
        X_processed = self.preprocessor.transform(X)
        
        predictions = {}
        for quantile, model in self.models.items():
            predictions[f'q{int(quantile*100):02d}'] = model.predict(X_processed)
        
        return predictions