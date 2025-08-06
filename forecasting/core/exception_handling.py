"""
Comprehensive Exception Handling
================================

Production-ready exception handling for DATect forecasting system.
Provides structured error handling, recovery mechanisms, and detailed error reporting.

Features:
- Custom exception classes for different error types
- Graceful degradation strategies  
- Detailed error logging with context
- Recovery and retry mechanisms
- Data validation error handling
- Model failure handling

Usage:
    from forecasting.core.exception_handling import handle_data_error, DataProcessingError
    
    try:
        # risky operation
        result = process_data(data)
    except Exception as e:
        handle_data_error(e, context="processing satellite data")
"""

import logging
import traceback
import functools
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from .logging_config import get_logger


class DATectError(Exception):
    """Base exception class for DATect forecasting system."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "DATECT_UNKNOWN"
        self.context = context or {}
        self.timestamp = time.time()
    
    def to_dict(self):
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp
        }


class DataProcessingError(DATectError):
    """Exception for data processing and loading errors."""
    pass


class ModelError(DATectError):
    """Exception for machine learning model errors."""
    pass


class ValidationError(DATectError):
    """Exception for data validation errors."""
    pass


class ConfigurationError(DATectError):
    """Exception for configuration and setup errors."""
    pass


class ExternalServiceError(DATectError):
    """Exception for external service (NOAA, USGS) errors."""
    pass


def safe_execute(func: Callable, *args, fallback_value=None, max_retries=3, 
                retry_delay=1.0, context: str = None, **kwargs):
    """
    Safely execute a function with automatic retry and fallback.
    
    Args:
        func: Function to execute
        *args: Function positional arguments
        fallback_value: Value to return if all retries fail
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        context: Context description for logging
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or fallback_value if all retries fail
    """
    logger = get_logger(__name__)
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"Function succeeded on retry {attempt}: {context or func.__name__}")
            return result
            
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {context or func.__name__}: {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {context or func.__name__}: {str(e)}")
    
    # Log final failure with full traceback
    logger.error(f"Returning fallback value after {max_retries + 1} failures")
    logger.debug(f"Final exception traceback:\n{traceback.format_exc()}")
    
    return fallback_value


def handle_data_error(exception: Exception, context: str = None, 
                     data_shape: tuple = None, recovery_action: str = None):
    """
    Handle data processing errors with detailed logging and context.
    
    Args:
        exception: The exception that occurred
        context: Context description (e.g., "loading satellite data")
        data_shape: Shape of data being processed when error occurred
        recovery_action: Description of recovery action taken
    """
    logger = get_logger("datect.error_handler")
    
    error_details = {
        'exception_type': type(exception).__name__,
        'message': str(exception),
        'context': context,
        'data_shape': data_shape,
        'recovery_action': recovery_action
    }
    
    logger.error(f"Data processing error in {context}: {str(exception)}")
    logger.error(f"Error details: {error_details}")
    
    # Log stack trace for debugging
    logger.debug(f"Stack trace:\n{traceback.format_exc()}")
    
    # Raise appropriate DATect exception
    if "connection" in str(exception).lower() or "network" in str(exception).lower():
        raise ExternalServiceError(
            f"Network/connection error during {context}",
            error_code="NETWORK_ERROR",
            context=error_details
        )
    elif "file" in str(exception).lower() or "path" in str(exception).lower():
        raise DataProcessingError(
            f"File system error during {context}",
            error_code="FILE_ERROR", 
            context=error_details
        )
    else:
        raise DataProcessingError(
            f"Data processing error during {context}",
            error_code="DATA_ERROR",
            context=error_details
        )


def handle_model_error(exception: Exception, model_name: str = None, 
                      data_points: int = None, features: list = None):
    """
    Handle machine learning model errors with model-specific context.
    
    Args:
        exception: The exception that occurred
        model_name: Name of the model that failed
        data_points: Number of data points being processed
        features: List of feature names being used
    """
    logger = get_logger("datect.model_error")
    
    error_details = {
        'exception_type': type(exception).__name__,
        'message': str(exception),
        'model_name': model_name,
        'data_points': data_points,
        'features': features
    }
    
    logger.error(f"Model error with {model_name}: {str(exception)}")
    logger.error(f"Model details: {error_details}")
    logger.debug(f"Stack trace:\n{traceback.format_exc()}")
    
    raise ModelError(
        f"Machine learning model error with {model_name}",
        error_code="MODEL_ERROR",
        context=error_details
    )


def validate_data_integrity(data, required_columns: list = None, 
                          min_rows: int = None, max_missing_rate: float = 0.5):
    """
    Validate data integrity and raise ValidationError if checks fail.
    
    Args:
        data: DataFrame to validate
        required_columns: List of columns that must exist
        min_rows: Minimum number of rows required
        max_missing_rate: Maximum allowed missing data rate (0.0-1.0)
        
    Raises:
        ValidationError: If validation fails
    """
    logger = get_logger("datect.validation")
    
    try:
        # Check if data exists
        if data is None or data.empty:
            raise ValidationError(
                "Data is empty or None",
                error_code="EMPTY_DATA",
                context={'data_shape': None}
            )
        
        # Check minimum rows
        if min_rows and len(data) < min_rows:
            raise ValidationError(
                f"Insufficient data rows: {len(data)} < {min_rows}",
                error_code="INSUFFICIENT_ROWS",
                context={'actual_rows': len(data), 'required_rows': min_rows}
            )
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValidationError(
                    f"Missing required columns: {missing_columns}",
                    error_code="MISSING_COLUMNS",
                    context={'missing_columns': list(missing_columns), 'available_columns': list(data.columns)}
                )
        
        # Check missing data rate
        if max_missing_rate is not None:
            total_cells = len(data) * len(data.columns)
            missing_cells = data.isnull().sum().sum()
            missing_rate = missing_cells / total_cells if total_cells > 0 else 0
            
            if missing_rate > max_missing_rate:
                raise ValidationError(
                    f"Excessive missing data: {missing_rate:.2%} > {max_missing_rate:.2%}",
                    error_code="EXCESSIVE_MISSING_DATA",
                    context={'missing_rate': missing_rate, 'max_allowed': max_missing_rate}
                )
        
        logger.info(f"Data validation passed: {data.shape[0]:,} rows × {data.shape[1]} columns")
        
    except ValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Unexpected error during data validation: {str(e)}")
        raise ValidationError(
            f"Validation process failed: {str(e)}",
            error_code="VALIDATION_PROCESS_ERROR",
            context={'original_error': str(e)}
        )


def robust_decorator(fallback_value=None, max_retries=1, handle_errors=True):
    """
    Decorator for robust function execution with error handling.
    
    Args:
        fallback_value: Value to return if function fails
        max_retries: Number of retry attempts
        handle_errors: Whether to handle and log errors gracefully
        
    Usage:
        @robust_decorator(fallback_value=[], max_retries=2)
        def risky_function():
            # potentially failing code
            return result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if handle_errors:
                return safe_execute(
                    func, *args, **kwargs,
                    fallback_value=fallback_value,
                    max_retries=max_retries,
                    context=f"{func.__module__}.{func.__name__}"
                )
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def create_error_report(error: Exception, context: str = None, 
                       additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized error report for logging and debugging.
    
    Args:
        error: The exception that occurred
        context: Context where error occurred
        additional_info: Additional information to include
        
    Returns:
        Dictionary containing error report
    """
    import sys
    import platform
    
    report = {
        'timestamp': time.time(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'additional_info': additional_info or {},
        'system_info': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'machine': platform.machine()
        },
        'stack_trace': traceback.format_exc()
    }
    
    return report


def log_error_report(error: Exception, context: str = None, 
                    additional_info: Dict[str, Any] = None):
    """
    Generate and log a comprehensive error report.
    
    Args:
        error: The exception that occurred
        context: Context where error occurred  
        additional_info: Additional information to include
    """
    logger = get_logger("datect.error_reporter")
    
    report = create_error_report(error, context, additional_info)
    
    logger.error(f"ERROR REPORT - {context or 'Unknown Context'}")
    logger.error(f"Error Type: {report['error_type']}")
    logger.error(f"Error Message: {report['error_message']}")
    
    if report['additional_info']:
        logger.error(f"Additional Info: {report['additional_info']}")
    
    logger.debug(f"Full Error Report: {report}")
    
    return report