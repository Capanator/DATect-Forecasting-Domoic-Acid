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
    from forecasting import ForecastEngine, RetrospectiveDashboard, RealtimeDashboard
    
    # Create forecasting engine
    engine = ForecastEngine(data_file="final_output.parquet")
    
    # Run retrospective evaluation
    results = engine.run_retrospective_evaluation(
        task="regression", 
        model_type="rf", 
        n_anchors=50
    )
    
    # Launch dashboard
    dashboard = RetrospectiveDashboard(results)
    dashboard.run(port=8071)
"""

__version__ = "2.0.0"
__author__ = "Claude Code Analysis & Modularization"
__date__ = "2025-01-08"

# Import main classes for easy access
from .core.forecast_engine import ForecastEngine
from .core.model_factory import ModelFactory
from .dashboard.retrospective import RetrospectiveDashboard
from .dashboard.realtime import RealtimeDashboard

__all__ = [
    'ForecastEngine',
    'ModelFactory', 
    'RetrospectiveDashboard',
    'RealtimeDashboard'
]