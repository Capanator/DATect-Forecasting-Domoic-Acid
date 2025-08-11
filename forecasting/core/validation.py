"""
Configuration and System Validation Module
==========================================

Provides validation functions for system configuration and startup checks.
Ensures the forecasting system is properly configured before operation.
"""

import os
import pandas as pd
from pathlib import Path
import config


def validate_configuration():
    """
    Validate system configuration parameters.
    
    Returns:
        bool: True if all configuration is valid
        
    Raises:
        ValueError: If critical configuration issues found
        FileNotFoundError: If required files are missing
    """
    print("[INFO] Validating system configuration...")
    
    # Validate temporal buffer settings
    if config.TEMPORAL_BUFFER_DAYS < 0:
        raise ValueError(f"Invalid TEMPORAL_BUFFER_DAYS: {config.TEMPORAL_BUFFER_DAYS} (must be >= 0)")
    if config.SATELLITE_BUFFER_DAYS < config.TEMPORAL_BUFFER_DAYS:
        raise ValueError("SATELLITE_BUFFER_DAYS should be >= TEMPORAL_BUFFER_DAYS for consistency")
    if config.CLIMATE_BUFFER_MONTHS < 0:
        raise ValueError(f"Invalid CLIMATE_BUFFER_MONTHS: {config.CLIMATE_BUFFER_MONTHS} (must be >= 0)")
        
    # Validate model configuration
    valid_modes = ["retrospective", "realtime"]
    if config.FORECAST_MODE not in valid_modes:
        raise ValueError(f"Invalid FORECAST_MODE: {config.FORECAST_MODE}. Must be one of: {valid_modes}")
        
    valid_models = ["rf", "linear"]
    if config.FORECAST_MODEL not in valid_models:
        raise ValueError(f"Invalid FORECAST_MODEL: {config.FORECAST_MODEL}. Must be one of: {valid_models}")
        
    valid_tasks = ["regression", "classification"]
    if config.FORECAST_TASK not in valid_tasks:
        raise ValueError(f"Invalid FORECAST_TASK: {config.FORECAST_TASK}. Must be one of: {valid_tasks}")
        
    # Validate lag features
    if not config.LAG_FEATURES or not isinstance(config.LAG_FEATURES, list):
        raise ValueError("LAG_FEATURES must be a non-empty list")
    if any(lag <= 0 for lag in config.LAG_FEATURES):
        raise ValueError("All LAG_FEATURES must be positive integers")
    if max(config.LAG_FEATURES) > 100:  # Reasonable upper bound
        raise ValueError("LAG_FEATURES values too large (max recommended: 100)")
        
    # Validate DA categories
    if len(config.DA_CATEGORY_BINS) != len(config.DA_CATEGORY_LABELS) + 1:
        raise ValueError("DA_CATEGORY_BINS must have one more element than DA_CATEGORY_LABELS")
    if not all(config.DA_CATEGORY_BINS[i] <= config.DA_CATEGORY_BINS[i+1] 
               for i in range(len(config.DA_CATEGORY_BINS)-1)):
        raise ValueError("DA_CATEGORY_BINS must be in ascending order")
        
    # Validate minimum samples
    if config.MIN_TRAINING_SAMPLES < 1:
        raise ValueError(f"MIN_TRAINING_SAMPLES must be >= 1, got: {config.MIN_TRAINING_SAMPLES}")
    if config.MIN_TRAINING_SAMPLES > 1000:  # Reasonable upper bound
        print(f"Warning: MIN_TRAINING_SAMPLES is very high ({config.MIN_TRAINING_SAMPLES})")
        
    # Validate date ranges
    try:
        start_date = pd.Timestamp(config.START_DATE)
        end_date = pd.Timestamp(config.END_DATE)
        if start_date >= end_date:
            raise ValueError(f"START_DATE ({config.START_DATE}) must be before END_DATE ({config.END_DATE})")
        if start_date < pd.Timestamp('1990-01-01'):
            raise ValueError(f"START_DATE too early ({config.START_DATE}) - satellite data not available")
        if end_date > pd.Timestamp.now() + pd.Timedelta(days=30):
            print(f"Warning: END_DATE ({config.END_DATE}) is in the future")
    except Exception as e:
        raise ValueError(f"Invalid date format in START_DATE or END_DATE: {e}")
        
    print("[INFO] Configuration validation passed")
    return True


def validate_data_files():
    """
    Validate required data files exist and are accessible.
    
    Returns:
        bool: True if all required files exist
        
    Raises:
        FileNotFoundError: If critical files are missing
    """
    print("[INFO] Validating data files...")
    
    # Check main data file
    if not os.path.exists(config.FINAL_OUTPUT_PATH):
        raise FileNotFoundError(f"Main data file not found: {config.FINAL_OUTPUT_PATH}")
    
    # Validate file size and basic properties
    file_size = os.path.getsize(config.FINAL_OUTPUT_PATH)
    if file_size < 1024:  # Less than 1KB
        raise ValueError(f"Data file appears to be empty or corrupted: {file_size} bytes")
    
    # Try to read the file to ensure it's valid parquet
    try:
        import pandas as pd
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH, engine="pyarrow")
        if data.empty:
            raise ValueError("Data file is empty")
        print(f"[INFO] Main data file validated: {len(data)} records, {file_size:,} bytes")
    except Exception as e:
        raise ValueError(f"Cannot read main data file: {e}")
    
    # Check for intermediate data directory
    intermediate_dir = os.path.dirname(config.SATELLITE_CACHE_PATH)
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir, exist_ok=True)
        print(f"[INFO] Created intermediate data directory: {intermediate_dir}")
        
    # Check output log directory
    if hasattr(config, 'LOG_OUTPUT_DIR') and config.LOG_OUTPUT_DIR:
        if not os.path.exists(config.LOG_OUTPUT_DIR):
            os.makedirs(config.LOG_OUTPUT_DIR, exist_ok=True)
            print(f"[INFO] Created log output directory: {config.LOG_OUTPUT_DIR}")
    
    print("[INFO] Data files validation passed")
    return True


def validate_sites():
    """
    Validate site configuration.
    
    Returns:
        bool: True if site configuration is valid
        
    Raises:
        ValueError: If site configuration issues found
    """
    print("[INFO] Validating site configuration...")
    
    if not config.SITES or not isinstance(config.SITES, dict):
        raise ValueError("SITES must be a non-empty dictionary")
    
    for site_name, coordinates in config.SITES.items():
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError(f"Site '{site_name}' coordinates must be [lat, lon] list")
        
        lat, lon = coordinates
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError(f"Site '{site_name}' coordinates must be numeric")
        
        # Validate coordinate ranges (Pacific Northwest region)
        if not (40 <= lat <= 50):
            raise ValueError(f"Site '{site_name}' latitude ({lat}) outside expected range (40-50°N)")
        if not (-130 <= lon <= -120):
            raise ValueError(f"Site '{site_name}' longitude ({lon}) outside expected range (130-120°W)")
    
    print(f"[INFO] Site validation passed: {len(config.SITES)} sites configured")
    return True


def validate_system_startup():
    """
    Complete system validation for startup.
    Runs all validation checks in proper order.
    
    Returns:
        bool: True if entire system validation passes
        
    Raises:
        Various exceptions if validation fails
    """
    print("[INFO] Starting comprehensive system validation...")
    
    try:
        validate_configuration()
        validate_sites()
        validate_data_files()
        print("[INFO] ✅ All system validation checks passed - ready for forecasting")
        return True
        
    except Exception as e:
        print(f"[ERROR] ❌ System validation failed: {e}")
        raise


def validate_runtime_parameters(n_anchors=None, min_test_date=None):
    """
    Validate runtime parameters for forecasting operations.
    
    Args:
        n_anchors: Number of anchor points for evaluation
        min_test_date: Minimum test date for evaluation
        
    Returns:
        bool: True if parameters are valid
        
    Raises:
        ValueError: If invalid parameters provided
    """
    if n_anchors is not None:
        if not isinstance(n_anchors, int) or n_anchors <= 0:
            raise ValueError(f"n_anchors must be positive integer, got: {n_anchors}")
        if n_anchors > 10000:  # Reasonable upper bound
            print(f"Warning: Very large n_anchors ({n_anchors}) may be slow")
    
    if min_test_date is not None:
        try:
            min_date = pd.Timestamp(min_test_date)
            if min_date < pd.Timestamp('2000-01-01'):
                raise ValueError(f"min_test_date too early: {min_test_date}")
            if min_date > pd.Timestamp.now():
                raise ValueError(f"min_test_date cannot be in future: {min_test_date}")
        except Exception as e:
            raise ValueError(f"Invalid min_test_date format: {e}")
    
    return True