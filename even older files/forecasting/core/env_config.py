"""
Environment Configuration Module
===============================

Handles environment-specific configuration loading and management.
"""

import os
from pathlib import Path
import config as main_config


def get_config():
    """
    Get configuration object with environment-specific overrides.
    
    Returns:
        Configuration object
    """
    return main_config


def get_data_path():
    """
    Get the path to the main data file.
    
    Returns:
        Path to final_output.parquet
    """
    return getattr(main_config, 'FINAL_OUTPUT_PATH', 'final_output.parquet')


def get_environment():
    """
    Detect current environment (development, production, testing).
    
    Returns:
        String indicating environment
    """
    return os.getenv('DATECT_ENV', 'development')


def is_production():
    """
    Check if running in production environment.
    
    Returns:
        Boolean indicating production environment
    """
    return get_environment().lower() == 'production'