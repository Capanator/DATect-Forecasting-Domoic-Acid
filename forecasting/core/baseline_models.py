"""
Baseline Models for DATect Forecasting System

This module implements various baseline models for comparison:
- Persistence (naive) forecast
- Climatological average
- Seasonal climatology
- Linear regression
- Simple moving average
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging

logger = logging.getLogger(__name__)


class PersistenceModel:
    """
    Persistence (naive) forecast model
    Predicts the last known value
    """
    
    def __init__(self, lag: int = 1):
        """
        Initialize persistence model
        
        Args:
            lag: Number of time steps to look back (default 1 = last value)
        """
        self.lag = lag
        self.last_values = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Fit the persistence model (just stores last values)
        
        Args:
            X: Feature matrix (not used, kept for consistency)
            y: Target values
            dates: Optional dates for the data
        """
        self.last_values = y[-self.lag:] if len(y) >= self.lag else y
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make persistence predictions
        
        Args:
            X: Feature matrix (used only for shape)
            
        Returns:
            Array of predictions
        """
        n_samples = X.shape[0] if len(X.shape) > 1 else len(X)
        
        if self.last_values is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the last known value for all predictions
        if len(self.last_values) > 0:
            return np.full(n_samples, self.last_values[-1])
        else:
            return np.zeros(n_samples)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters"""
        return {'lag': self.lag}
    
    def set_params(self, **params) -> 'PersistenceModel':
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClimatologyModel:
    """
    Climatological average model
    Predicts the long-term average
    """
    
    def __init__(self, use_seasonal: bool = False, seasonal_period: int = 52):
        """
        Initialize climatology model
        
        Args:
            use_seasonal: Whether to use seasonal climatology
            seasonal_period: Period for seasonal patterns (e.g., 52 for weekly data)
        """
        self.use_seasonal = use_seasonal
        self.seasonal_period = seasonal_period
        self.climatology = None
        self.seasonal_climatology = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Fit the climatology model
        
        Args:
            X: Feature matrix (not used for basic climatology)
            y: Target values
            dates: Optional dates for seasonal climatology
        """
        # Calculate overall mean
        self.climatology = np.mean(y)
        
        # Calculate seasonal climatology if requested
        if self.use_seasonal and dates is not None:
            df = pd.DataFrame({'y': y, 'date': dates})
            
            # Extract seasonal component (week of year for weekly data)
            df['week'] = df['date'].dt.isocalendar().week
            
            # Calculate mean for each week
            self.seasonal_climatology = df.groupby('week')['y'].mean().to_dict()
            
            # Fill any missing weeks with overall mean
            for week in range(1, 54):
                if week not in self.seasonal_climatology:
                    self.seasonal_climatology[week] = self.climatology
        
        return self
    
    def predict(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Make climatological predictions
        
        Args:
            X: Feature matrix (used only for shape)
            dates: Optional dates for seasonal predictions
            
        Returns:
            Array of predictions
        """
        if self.climatology is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = X.shape[0] if len(X.shape) > 1 else len(X)
        
        if self.use_seasonal and self.seasonal_climatology is not None and dates is not None:
            # Use seasonal climatology
            predictions = []
            for date in dates:
                week = date.isocalendar().week
                # Handle week 53 (map to week 52)
                if week == 53:
                    week = 52
                predictions.append(self.seasonal_climatology.get(week, self.climatology))
            return np.array(predictions)
        else:
            # Use overall climatology
            return np.full(n_samples, self.climatology)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters"""
        return {
            'use_seasonal': self.use_seasonal,
            'seasonal_period': self.seasonal_period
        }
    
    def set_params(self, **params) -> 'ClimatologyModel':
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class MovingAverageModel:
    """
    Simple and exponential moving average models
    """
    
    def __init__(self, window: int = 4, method: str = 'simple', alpha: float = 0.3):
        """
        Initialize moving average model
        
        Args:
            window: Window size for moving average
            method: 'simple' or 'exponential'
            alpha: Smoothing parameter for exponential MA (0 < alpha <= 1)
        """
        self.window = window
        self.method = method
        self.alpha = alpha
        self.history = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Fit the moving average model
        
        Args:
            X: Feature matrix (not used)
            y: Target values
            dates: Optional dates
        """
        # Store recent history for prediction
        self.history = y[-self.window:] if len(y) >= self.window else y
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make moving average predictions
        
        Args:
            X: Feature matrix (used only for shape)
            
        Returns:
            Array of predictions
        """
        if self.history is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = X.shape[0] if len(X.shape) > 1 else len(X)
        
        if self.method == 'simple':
            # Simple moving average
            prediction = np.mean(self.history)
        elif self.method == 'exponential':
            # Exponential moving average
            weights = np.array([(1 - self.alpha) ** i for i in range(len(self.history) - 1, -1, -1)])
            weights = weights / weights.sum()
            prediction = np.sum(self.history * weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return np.full(n_samples, prediction)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters"""
        return {
            'window': self.window,
            'method': self.method,
            'alpha': self.alpha
        }
    
    def set_params(self, **params) -> 'MovingAverageModel':
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LinearTrendModel:
    """
    Linear regression model with optional polynomial features
    """
    
    def __init__(self, degree: int = 1, use_time_features: bool = True, 
                 regularization: Optional[str] = None, alpha: float = 1.0):
        """
        Initialize linear trend model
        
        Args:
            degree: Polynomial degree (1 for linear, 2 for quadratic, etc.)
            use_time_features: Whether to use time-based features
            regularization: None, 'ridge', or 'lasso'
            alpha: Regularization strength
        """
        self.degree = degree
        self.use_time_features = use_time_features
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.feature_means = None
        self.feature_stds = None
        
    def _create_features(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """Create features for linear model"""
        features = []
        
        # Use provided features
        if X is not None and len(X.shape) > 1:
            features.append(X)
        
        # Add time features if requested
        if self.use_time_features and dates is not None:
            time_features = []
            
            # Time index (normalized)
            time_index = np.arange(len(dates)) / len(dates)
            time_features.append(time_index.reshape(-1, 1))
            
            # Polynomial features
            for d in range(2, self.degree + 1):
                time_features.append((time_index ** d).reshape(-1, 1))
            
            # Seasonal features
            day_of_year = pd.Series(dates).dt.dayofyear.values
            time_features.append(np.sin(2 * np.pi * day_of_year / 365.25).reshape(-1, 1))
            time_features.append(np.cos(2 * np.pi * day_of_year / 365.25).reshape(-1, 1))
            
            if len(time_features) > 0:
                features.append(np.hstack(time_features))
        
        if len(features) > 0:
            return np.hstack(features) if len(features) > 1 else features[0]
        else:
            # Fallback to simple time index
            return np.arange(len(X)).reshape(-1, 1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Fit the linear model
        
        Args:
            X: Feature matrix
            y: Target values
            dates: Optional dates for time features
        """
        # Create features
        features = self._create_features(X, dates)
        
        # Normalize features
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1  # Avoid division by zero
        features_normalized = (features - self.feature_means) / self.feature_stds
        
        # Select model based on regularization
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        
        # Fit model
        self.model.fit(features_normalized, y)
        
        return self
    
    def predict(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Make predictions with linear model
        
        Args:
            X: Feature matrix
            dates: Optional dates for time features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Create features
        features = self._create_features(X, dates)
        
        # Normalize features
        features_normalized = (features - self.feature_means) / self.feature_stds
        
        # Make predictions
        return self.model.predict(features_normalized)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters"""
        return {
            'degree': self.degree,
            'use_time_features': self.use_time_features,
            'regularization': self.regularization,
            'alpha': self.alpha
        }
    
    def set_params(self, **params) -> 'LinearTrendModel':
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BaselineEnsemble:
    """
    Ensemble of baseline models
    """
    
    def __init__(self, models: Optional[List] = None, weights: Optional[np.ndarray] = None):
        """
        Initialize baseline ensemble
        
        Args:
            models: List of baseline models to ensemble
            weights: Weights for each model (if None, uses equal weights)
        """
        if models is None:
            # Default ensemble
            self.models = [
                PersistenceModel(),
                ClimatologyModel(use_seasonal=True),
                MovingAverageModel(window=4),
                LinearTrendModel(degree=1)
            ]
        else:
            self.models = models
        
        if weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = weights / np.sum(weights)
    
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Fit all models in the ensemble
        
        Args:
            X: Feature matrix
            y: Target values
            dates: Optional dates
        """
        for model in self.models:
            try:
                model.fit(X, y, dates)
            except TypeError:
                # Some models might not accept dates parameter
                model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Feature matrix
            dates: Optional dates
            
        Returns:
            Weighted average of all model predictions
        """
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(X, dates)
            except TypeError:
                # Some models might not accept dates parameter
                pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        return np.sum(predictions * self.weights.reshape(-1, 1), axis=0)
    
    def get_individual_predictions(self, X: np.ndarray, 
                                  dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, np.ndarray]:
        """
        Get predictions from each model separately
        
        Returns:
            Dictionary with model names and their predictions
        """
        results = {}
        
        for i, model in enumerate(self.models):
            model_name = model.__class__.__name__
            try:
                results[f"{model_name}_{i}"] = model.predict(X, dates)
            except TypeError:
                results[f"{model_name}_{i}"] = model.predict(X)
        
        return results


def evaluate_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            dates_train: Optional[pd.DatetimeIndex] = None,
                            dates_test: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    Evaluate all baseline models
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        dates_train: Optional training dates
        dates_test: Optional test dates
        
    Returns:
        DataFrame with evaluation results
    """
    models = {
        'Persistence': PersistenceModel(),
        'Climatology': ClimatologyModel(use_seasonal=False),
        'Seasonal_Climatology': ClimatologyModel(use_seasonal=True),
        'Moving_Average_4': MovingAverageModel(window=4),
        'Moving_Average_8': MovingAverageModel(window=8),
        'Exponential_MA': MovingAverageModel(method='exponential'),
        'Linear_Trend': LinearTrendModel(degree=1),
        'Quadratic_Trend': LinearTrendModel(degree=2),
        'Ridge_Regression': LinearTrendModel(regularization='ridge'),
        'Ensemble': BaselineEnsemble()
    }
    
    results = []
    
    for name, model in models.items():
        try:
            # Fit model
            try:
                model.fit(X_train, y_train, dates_train)
            except TypeError:
                model.fit(X_train, y_train)
            
            # Make predictions
            try:
                y_pred = model.predict(X_test, dates_test)
            except TypeError:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate relative improvement over persistence
            persistence = PersistenceModel()
            persistence.fit(X_train, y_train)
            y_persistence = persistence.predict(X_test)
            persistence_mae = mean_absolute_error(y_test, y_persistence)
            
            improvement = ((persistence_mae - mae) / persistence_mae) * 100 if persistence_mae > 0 else 0
            
            results.append({
                'Model': name,
                'R2': r2,
                'MAE': mae,
                'RMSE': rmse,
                'Improvement_over_persistence': improvement
            })
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")
            results.append({
                'Model': name,
                'R2': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan,
                'Improvement_over_persistence': np.nan
            })
    
    return pd.DataFrame(results).sort_values('MAE')


# Example usage
if __name__ == "__main__":
    # Generate sample time series data
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    
    # Create synthetic time series with trend and seasonality
    trend = 0.01 * time
    seasonal = 5 * np.sin(2 * np.pi * time / 52)  # Weekly seasonality
    noise = np.random.randn(n_samples) * 2
    y = 20 + trend + seasonal + noise
    
    # Create dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='W')
    
    # Simple features (could be external variables)
    X = np.random.randn(n_samples, 3)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    # Evaluate all baseline models
    print("Evaluating baseline models...")
    results = evaluate_baseline_models(X_train, y_train, X_test, y_test, 
                                      dates_train, dates_test)
    
    print("\nBaseline Model Performance:")
    print(results.to_string(index=False))
    
    # Test individual models
    print("\n\nTesting Seasonal Climatology Model:")
    model = ClimatologyModel(use_seasonal=True)
    model.fit(X_train, y_train, dates_train)
    predictions = model.predict(X_test, dates_test)
    
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae:.2f}")