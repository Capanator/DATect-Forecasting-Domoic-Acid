"""
Domoic Acid Forecasting Pipeline

A modular, leak-free forecasting system for predicting harmful algal bloom concentrations.
"""

from .forecast_pipeline import DAForecastPipeline, DAForecastConfig
from .feature_engineering import (
    TemporalFeatures, LagFeatures, CategoryEncoder, 
    DataCleaner, FeatureSelector
)
from .models import ModelFactory, ModelTrainer, ModelPredictor
from .data_splitter import TimeSeriesSplitter, RandomAnchorGenerator, DataValidator
from .evaluation import ForecastEvaluator, RegressionEvaluator, ClassificationEvaluator

__all__ = [
    'DAForecastPipeline',
    'DAForecastConfig',
    'TemporalFeatures',
    'LagFeatures', 
    'CategoryEncoder',
    'DataCleaner',
    'FeatureSelector',
    'ModelFactory',
    'ModelTrainer',
    'ModelPredictor',
    'TimeSeriesSplitter',
    'RandomAnchorGenerator',
    'DataValidator',
    'ForecastEvaluator',
    'RegressionEvaluator',
    'ClassificationEvaluator'
]

__version__ = "1.0.0"