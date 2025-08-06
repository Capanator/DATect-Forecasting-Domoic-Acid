"""
Data Pipeline Package
====================

This package contains the core data integration and processing pipeline
components for the DATect forecasting system:

- data_integrator: Main data integration and coordination logic
- temporal_safeguards: Temporal integrity and leakage prevention

The pipeline orchestrates the various data sources and applies scientific
safeguards to ensure data quality and prevent temporal leakage.
"""

from .data_integrator import DataIntegrator
from .temporal_safeguards import TemporalSafeguards

__all__ = [
    'DataIntegrator',
    'TemporalSafeguards'
]