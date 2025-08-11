"""
Domoic Acid Forecasting System
========================================

A modular, temporally-safe forecasting system for predicting harmful algal bloom
concentrations along the Pacific Coast.

Key Features:
- Complete elimination of data leakage
- Temporal validation for all features
- Per-forecast DA category creation
- Strict train/test split ordering
- Configurable ML models and evaluation metrics

Usage:
    from forecasting import ForecastEngine
    
    # Create forecasting engine
    engine = ForecastEngine(data_file="final_output.parquet")
    
    # Generate single forecast
    result = engine.generate_single_forecast(
        data_path="final_output.parquet",
        forecast_date="2015-06-15", 
        site="Santa Cruz Wharf",
        task="regression",
        model_type="rf"
    )
"""

__version__ = "2.0.0"
__author__ = "DATect Team"
__date__ = "2025-01-08"

# Import main classes for easy access
from .core.forecast_engine import ForecastEngine
from .core.model_factory import ModelFactory
# Dashboard components removed - using web interface only

__all__ = [
    'ForecastEngine',
    'ModelFactory'
]