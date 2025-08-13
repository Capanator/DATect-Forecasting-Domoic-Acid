"""
Hyperparameter Optimization with Nested Cross-Validation for DATect

This module implements:
- Nested cross-validation for unbiased performance estimation
- Grid search and random search for hyperparameter tuning
- Bayesian optimization
- Time series specific cross-validation
- Spatial cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, clone
import warnings
from datetime import datetime, timedelta
import logging
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict
    best_score: float
    cv_scores: List[float]
    all_results: pd.DataFrame
    test_score: Optional[float] = None
    optimization_time: Optional[float] = None


class TimeSeriesCrossValidator:
    """
    Time series aware cross-validation splitter
    """
    
    def __init__(self, n_splits: int = 5, 
                 min_train_size: Optional[int] = None,
                 max_train_size: Optional[int] = None,
                 gap: int = 0,
                 strategy: str = 'expanding'):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of splits
            min_train_size: Minimum training set size
            max_train_size: Maximum training set size (for sliding window)
            gap: Gap between train and test sets
            strategy: 'expanding' or 'sliding' window
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.gap = gap
        self.strategy = strategy
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices for time series CV
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = X.shape[0]
        
        # Calculate test size
        if self.min_train_size is None:
            min_train_size = n_samples // (self.n_splits + 1)
        else:
            min_train_size = self.min_train_size
        
        test_size = (n_samples - min_train_size - self.gap) // self.n_splits
        
        for i in range(self.n_splits):
            if self.strategy == 'expanding':
                # Expanding window
                train_end = min_train_size + i * test_size
                test_start = train_end + self.gap
                test_end = test_start + test_size
                
                train_indices = np.arange(0, train_end)
                
            else:  # sliding
                # Sliding window
                if self.max_train_size is None:
                    train_size = min_train_size + i * test_size
                else:
                    train_size = min(self.max_train_size, min_train_size + i * test_size)
                
                train_end = min_train_size + i * test_size
                train_start = max(0, train_end - train_size)
                test_start = train_end + self.gap
                test_end = test_start + test_size
                
                train_indices = np.arange(train_start, train_end)
            
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class SpatialCrossValidator:
    """
    Spatial cross-validation for geographic data
    """
    
    def __init__(self, coordinates: np.ndarray, 
                 n_splits: int = 5,
                 buffer_distance: float = 50.0,
                 strategy: str = 'random'):
        """
        Initialize spatial cross-validator
        
        Args:
            coordinates: Array of (lat, lon) coordinates for each sample
            n_splits: Number of spatial folds
            buffer_distance: Buffer distance (km) to exclude nearby points
            strategy: 'random', 'systematic', or 'clustered'
        """
        self.coordinates = coordinates
        self.n_splits = n_splits
        self.buffer_distance = buffer_distance
        self.strategy = strategy
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distances()
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate pairwise distances between locations"""
        # Convert to radians
        coords_rad = np.radians(self.coordinates)
        
        # Haversine distance
        distances = np.zeros((len(self.coordinates), len(self.coordinates)))
        
        for i in range(len(self.coordinates)):
            for j in range(i+1, len(self.coordinates)):
                lat1, lon1 = coords_rad[i]
                lat2, lon2 = coords_rad[j]
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                
                # Earth radius in km
                r = 6371
                distance = r * c
                
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices for spatial CV
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.strategy == 'random':
            # Random spatial blocks
            np.random.shuffle(indices)
            fold_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
                test_indices = indices[test_start:test_end]
                
                # Exclude nearby points from training
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[test_indices] = False
                
                # Apply spatial buffer
                for test_idx in test_indices:
                    nearby = self.distance_matrix[test_idx] < self.buffer_distance
                    train_mask[nearby] = False
                
                train_indices = indices[train_mask]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
        
        elif self.strategy == 'systematic':
            # Systematic spatial sampling
            unique_coords = np.unique(self.coordinates, axis=0)
            
            if len(unique_coords) >= self.n_splits:
                # Split unique locations
                location_folds = np.array_split(unique_coords, self.n_splits)
                
                for fold_coords in location_folds:
                    test_mask = np.zeros(n_samples, dtype=bool)
                    
                    for coord in fold_coords:
                        coord_matches = np.all(self.coordinates == coord, axis=1)
                        test_mask |= coord_matches
                    
                    test_indices = indices[test_mask]
                    
                    # Apply spatial buffer
                    train_mask = ~test_mask
                    for test_idx in test_indices:
                        nearby = self.distance_matrix[test_idx] < self.buffer_distance
                        train_mask[nearby] = False
                    
                    train_indices = indices[train_mask]
                    
                    if len(train_indices) > 0 and len(test_indices) > 0:
                        yield train_indices, test_indices


class NestedCrossValidation:
    """
    Nested cross-validation for unbiased hyperparameter optimization
    """
    
    def __init__(self, 
                 estimator: BaseEstimator,
                 param_grid: Union[Dict, List[Dict]],
                 outer_cv: Optional[Any] = None,
                 inner_cv: Optional[Any] = None,
                 scoring: str = 'neg_mean_squared_error',
                 n_jobs: int = 1,
                 verbose: int = 0):
        """
        Initialize nested cross-validation
        
        Args:
            estimator: Base estimator to optimize
            param_grid: Parameter grid for search
            outer_cv: Outer cross-validation splitter
            inner_cv: Inner cross-validation splitter
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.outer_cv = outer_cv or TimeSeriesCrossValidator(n_splits=5)
        self.inner_cv = inner_cv or TimeSeriesCrossValidator(n_splits=3)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = None
    
    def _score(self, estimator, X, y):
        """Calculate score for given estimator"""
        predictions = estimator.predict(X)
        
        if self.scoring == 'neg_mean_squared_error':
            return -mean_squared_error(y, predictions)
        elif self.scoring == 'neg_mean_absolute_error':
            return -mean_absolute_error(y, predictions)
        elif self.scoring == 'r2':
            return r2_score(y, predictions)
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NestedCrossValidation':
        """
        Perform nested cross-validation
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self
        """
        outer_scores = []
        best_params_list = []
        
        # Outer loop
        for fold_idx, (train_outer, test_outer) in enumerate(self.outer_cv.split(X, y)):
            if self.verbose > 0:
                print(f"Outer fold {fold_idx + 1}/{self.outer_cv.n_splits}")
            
            X_train_outer = X[train_outer]
            y_train_outer = y[train_outer]
            X_test_outer = X[test_outer]
            y_test_outer = y[test_outer]
            
            # Inner loop - hyperparameter optimization
            best_score_inner = -np.inf
            best_params_inner = None
            
            # Grid search in inner loop
            param_list = list(ParameterGrid(self.param_grid))
            
            for params in param_list:
                scores_inner = []
                
                # Inner cross-validation
                for train_inner, val_inner in self.inner_cv.split(X_train_outer, y_train_outer):
                    X_train_inner = X_train_outer[train_inner]
                    y_train_inner = y_train_outer[train_inner]
                    X_val_inner = X_train_outer[val_inner]
                    y_val_inner = y_train_outer[val_inner]
                    
                    # Train model with current parameters
                    model = clone(self.estimator)
                    model.set_params(**params)
                    model.fit(X_train_inner, y_train_inner)
                    
                    # Evaluate on validation set
                    score = self._score(model, X_val_inner, y_val_inner)
                    scores_inner.append(score)
                
                # Average score across inner folds
                mean_score_inner = np.mean(scores_inner)
                
                if mean_score_inner > best_score_inner:
                    best_score_inner = mean_score_inner
                    best_params_inner = params
            
            # Train on full outer training set with best parameters
            best_model = clone(self.estimator)
            best_model.set_params(**best_params_inner)
            best_model.fit(X_train_outer, y_train_outer)
            
            # Evaluate on outer test set
            score_outer = self._score(best_model, X_test_outer, y_test_outer)
            outer_scores.append(score_outer)
            best_params_list.append(best_params_inner)
            
            if self.verbose > 0:
                print(f"  Best params: {best_params_inner}")
                print(f"  Outer score: {score_outer:.4f}")
        
        # Store results
        self.cv_results_ = {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_per_fold': best_params_list
        }
        
        # Select most common best parameters
        from collections import Counter
        param_strings = [str(params) for params in best_params_list]
        most_common = Counter(param_strings).most_common(1)[0][0]
        self.best_params_ = eval(most_common)
        self.best_score_ = np.mean(outer_scores)
        
        return self
    
    def get_results(self) -> OptimizationResult:
        """Get optimization results"""
        results_df = pd.DataFrame({
            'fold': range(len(self.cv_results_['outer_scores'])),
            'score': self.cv_results_['outer_scores'],
            'params': [str(p) for p in self.cv_results_['best_params_per_fold']]
        })
        
        return OptimizationResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            cv_scores=self.cv_results_['outer_scores'],
            all_results=results_df
        )


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning
    """
    
    def __init__(self,
                 estimator: BaseEstimator,
                 param_bounds: Dict[str, Tuple[float, float]],
                 cv: Optional[Any] = None,
                 n_iter: int = 50,
                 init_points: int = 5,
                 scoring: str = 'neg_mean_squared_error',
                 verbose: int = 0):
        """
        Initialize Bayesian optimizer
        
        Args:
            estimator: Base estimator
            param_bounds: Parameter bounds for optimization
            cv: Cross-validation splitter
            n_iter: Number of optimization iterations
            init_points: Number of random initialization points
            scoring: Scoring metric
            verbose: Verbosity level
        """
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.cv = cv or TimeSeriesCrossValidator(n_splits=3)
        self.n_iter = n_iter
        self.init_points = init_points
        self.scoring = scoring
        self.verbose = verbose
        
        self.optimization_history = []
        self.best_params_ = None
        self.best_score_ = None
    
    def _objective(self, X: np.ndarray, y: np.ndarray, **params):
        """Objective function for Bayesian optimization"""
        # Convert parameters to appropriate types
        processed_params = {}
        for key, value in params.items():
            if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                processed_params[key] = int(value)
            else:
                processed_params[key] = value
        
        # Cross-validation
        scores = []
        for train_idx, val_idx in self.cv.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Train model
            model = clone(self.estimator)
            model.set_params(**processed_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_val)
            
            if self.scoring == 'neg_mean_squared_error':
                score = -mean_squared_error(y_val, predictions)
            elif self.scoring == 'neg_mean_absolute_error':
                score = -mean_absolute_error(y_val, predictions)
            elif self.scoring == 'r2':
                score = r2_score(y_val, predictions)
            else:
                raise ValueError(f"Unknown scoring metric: {self.scoring}")
            
            scores.append(score)
        
        return np.mean(scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianOptimizer':
        """
        Perform Bayesian optimization
        
        Note: This is a simplified version. For production, use libraries like
        scikit-optimize or optuna
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
            
            # Define search space
            dimensions = []
            param_names = []
            
            for param_name, (low, high) in self.param_bounds.items():
                param_names.append(param_name)
                
                if param_name in ['n_estimators', 'max_depth', 'min_samples_split']:
                    dimensions.append(Integer(int(low), int(high), name=param_name))
                else:
                    dimensions.append(Real(low, high, name=param_name))
            
            # Objective function wrapper
            @use_named_args(dimensions)
            def objective(**params):
                return -self._objective(X, y, **params)  # Minimize negative score
            
            # Run optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.n_iter,
                n_initial_points=self.init_points,
                verbose=self.verbose > 0
            )
            
            # Store results
            self.best_params_ = dict(zip(param_names, result.x))
            self.best_score_ = -result.fun
            
            # Process integer parameters
            for key in ['n_estimators', 'max_depth', 'min_samples_split']:
                if key in self.best_params_:
                    self.best_params_[key] = int(self.best_params_[key])
            
        except ImportError:
            # Fallback to random search if scikit-optimize not available
            logger.warning("scikit-optimize not available, falling back to random search")
            
            best_score = -np.inf
            best_params = None
            
            for i in range(self.n_iter):
                # Random sample parameters
                params = {}
                for param_name, (low, high) in self.param_bounds.items():
                    if param_name in ['n_estimators', 'max_depth', 'min_samples_split']:
                        params[param_name] = np.random.randint(int(low), int(high) + 1)
                    else:
                        params[param_name] = np.random.uniform(low, high)
                
                # Evaluate
                score = self._objective(X, y, **params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if self.verbose > 0 and (i + 1) % 10 == 0:
                    print(f"Iteration {i + 1}/{self.n_iter}, Best score: {best_score:.4f}")
            
            self.best_params_ = best_params
            self.best_score_ = best_score
        
        return self


def optimize_xgboost_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    method: str = 'nested_cv') -> OptimizationResult:
    """
    Optimize XGBoost hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        method: 'nested_cv', 'bayesian', or 'grid_search'
        
    Returns:
        OptimizationResult with best parameters and scores
    """
    from xgboost import XGBRegressor
    
    # Define parameter search space
    if method == 'bayesian':
        param_bounds = {
            'n_estimators': (50, 500),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 10)
        }
        
        optimizer = BayesianOptimizer(
            estimator=XGBRegressor(random_state=42),
            param_bounds=param_bounds,
            n_iter=30,
            verbose=1
        )
        
        optimizer.fit(X_train, y_train)
        best_params = optimizer.best_params_
        
    else:  # nested_cv or grid_search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        if method == 'nested_cv':
            optimizer = NestedCrossValidation(
                estimator=XGBRegressor(random_state=42),
                param_grid=param_grid,
                verbose=1
            )
        else:  # grid_search
            from sklearn.model_selection import GridSearchCV
            
            optimizer = GridSearchCV(
                estimator=XGBRegressor(random_state=42),
                param_grid=param_grid,
                cv=TimeSeriesCrossValidator(n_splits=3),
                scoring='neg_mean_squared_error',
                verbose=1
            )
        
        optimizer.fit(X_train, y_train)
        best_params = optimizer.best_params_ if hasattr(optimizer, 'best_params_') else optimizer.best_params
    
    # Train final model with best parameters
    best_model = XGBRegressor(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_predictions = best_model.predict(X_test)
    test_score = r2_score(y_test, test_predictions)
    
    # Create result object
    if isinstance(optimizer, NestedCrossValidation):
        return optimizer.get_results()
    else:
        return OptimizationResult(
            best_params=best_params,
            best_score=optimizer.best_score_ if hasattr(optimizer, 'best_score_') else None,
            cv_scores=[],
            all_results=pd.DataFrame(),
            test_score=test_score
        )


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test nested cross-validation
    print("Testing Nested Cross-Validation...")
    result = optimize_xgboost_hyperparameters(
        X_train, y_train, X_test, y_test, method='nested_cv'
    )
    
    print(f"\nBest parameters: {result.best_params}")
    print(f"CV score: {result.best_score:.4f}")
    print(f"Test score: {result.test_score:.4f}")