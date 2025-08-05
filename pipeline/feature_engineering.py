"""
Feature engineering pipeline without data leakage.
All feature creation happens within proper sklearn transformers.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Creates temporal features from date column."""
    
    def __init__(self, date_col='date'):
        self.date_col = date_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        dates = pd.to_datetime(X_copy[self.date_col])
        day_of_year = dates.dt.dayofyear
        
        X_copy['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
        X_copy['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        X_copy['month'] = dates.dt.month
        X_copy['quarter'] = dates.dt.quarter
        
        return X_copy


class LagFeatures(BaseEstimator, TransformerMixin):
    """Creates lag features for time series data."""
    
    def __init__(self, target_col='da', group_col='site', lags=[1, 2, 3]):
        self.target_col = target_col
        self.group_col = group_col
        self.lags = lags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for lag in self.lags:
            X_copy[f'{self.target_col}_lag_{lag}'] = (
                X_copy.groupby(self.group_col)[self.target_col].shift(lag)
            )
        return X_copy


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Creates category labels based on training data distribution."""
    
    def __init__(self, target_col='da', bins=None):
        self.target_col = target_col
        self.bins = bins or [-float('inf'), 5, 20, 40, float('inf')]
        self.fitted_bins_ = None
        
    def fit(self, X, y=None):
        # Use training data to determine category boundaries
        target_data = X[self.target_col].dropna()
        if len(target_data) == 0:
            raise ValueError("No valid target data for category encoding")
            
        # Can adjust bins based on training data distribution if needed
        self.fitted_bins_ = self.bins
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['da_category'] = pd.cut(
            X_copy[self.target_col],
            bins=self.fitted_bins_,
            labels=[0, 1, 2, 3],
            right=True
        ).astype('Int64')
        return X_copy


class DataCleaner(BaseEstimator, TransformerMixin):
    """Handles data cleaning and validation."""
    
    def __init__(self, required_cols=None):
        self.required_cols = required_cols or []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Sort by site and date for consistency
        if 'site' in X_copy.columns and 'date' in X_copy.columns:
            X_copy = X_copy.sort_values(['site', 'date']).reset_index(drop=True)
        
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects relevant features and drops unnecessary columns."""
    
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or ['date', 'site']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Drop specified columns
        cols_to_drop = [col for col in self.drop_cols if col in X_copy.columns]
        if cols_to_drop:
            X_copy = X_copy.drop(columns=cols_to_drop)
        
        # Select only numeric columns
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
        return X_copy[numeric_cols]