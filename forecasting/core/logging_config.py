"""
Centralized Logging Configuration
=================================

Production-ready logging system for DATect forecasting system.
Provides structured logging with appropriate levels, formatting, and output destinations.

Features:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output with rotation
- Structured formatting for production monitoring
- Performance logging for operational insights
- Error tracking for debugging

Usage:
    from forecasting.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("This is an informational message")
    logger.error("This is an error message")
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_dir="./logs/", enable_file_logging=True):
    """
    Configure centralized logging for the DATect system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to write logs to files
        
    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    if enable_file_logging:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (always enabled for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Main application log file with rotation
        main_log_file = os.path.join(log_dir, 'datect_main.log')
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error-only log file for critical issues
        error_log_file = os.path.join(log_dir, 'datect_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    return root_logger


def get_logger(name):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(func):
    """
    Decorator to log function execution time and performance metrics.
    
    Usage:
        @log_performance
        def my_function():
            # function code here
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_data_pipeline_stage(stage_name, data_shape=None, success=True, details=None):
    """
    Log data pipeline processing stages with consistent formatting.
    
    Args:
        stage_name: Name of the processing stage
        data_shape: Shape of data being processed (rows, columns)
        success: Whether the stage completed successfully
        details: Additional details to log
    """
    logger = get_logger("datect.pipeline")
    
    status = "✓" if success else "✗"
    message = f"{status} {stage_name}"
    
    if data_shape:
        if isinstance(data_shape, tuple) and len(data_shape) == 2:
            message += f" | Data: {data_shape[0]:,} rows × {data_shape[1]} cols"
        else:
            message += f" | Data size: {data_shape}"
    
    if details:
        message += f" | {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_model_performance(model_name, metrics_dict, data_points=None):
    """
    Log model performance metrics with consistent formatting.
    
    Args:
        model_name: Name of the model
        metrics_dict: Dictionary of metric names and values
        data_points: Number of data points evaluated
    """
    logger = get_logger("datect.model")
    
    message = f"Model Performance: {model_name}"
    if data_points:
        message += f" (n={data_points:,})"
    
    logger.info(message)
    
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def log_forecast_result(site, forecast_date, predicted_value, confidence=None, model_used=None):
    """
    Log individual forecast results with structured format for monitoring.
    
    Args:
        site: Monitoring site name
        forecast_date: Date of forecast
        predicted_value: Predicted DA concentration
        confidence: Confidence interval or uncertainty measure
        model_used: Model type used for prediction
    """
    logger = get_logger("datect.forecast")
    
    message = f"Forecast: {site} | {forecast_date} | DA={predicted_value:.2f} μg/g"
    
    if confidence:
        message += f" | CI={confidence}"
    if model_used:
        message += f" | Model={model_used}"
    
    logger.info(message)


# Initialize default logging on import
# This ensures logging is available throughout the application
_default_logger = setup_logging(
    log_level=logging.INFO,
    log_dir="./logs/",
    enable_file_logging=True
)

# Create module logger for this configuration module
_config_logger = get_logger(__name__)
_config_logger.info("DATect logging system initialized")