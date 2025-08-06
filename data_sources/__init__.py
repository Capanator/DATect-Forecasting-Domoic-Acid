"""
Data Sources Package
===================

This package contains modules for processing different data sources used in the
DATect forecasting system:

- satellite_processor: MODIS satellite oceanographic data
- climate_processor: Climate indices (PDO, ONI, BEUTI)  
- streamflow_processor: USGS streamflow data
- toxin_processor: DA/PN toxin measurement data

Each processor follows a consistent interface pattern for data loading,
validation, and processing with comprehensive error handling.
"""

from .satellite_processor import SatelliteProcessor
from .climate_processor import ClimateProcessor
from .streamflow_processor import StreamflowProcessor
from .toxin_processor import ToxinProcessor

__all__ = [
    'SatelliteProcessor',
    'ClimateProcessor', 
    'StreamflowProcessor',
    'ToxinProcessor'
]