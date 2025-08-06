"""
Exception Handling Utilities
============================

Provides robust exception handling and error reporting for scientific operations.
"""

import functools
import traceback
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


def safe_execute(func: Callable, error_message: str = "Operation failed", 
                default_return: Any = None, raise_on_error: bool = False) -> Any:
    """
    Execute a function safely with comprehensive error handling.
    
    Args:
        func: Function to execute
        error_message: Custom error message for logging
        default_return: Value to return on error
        raise_on_error: Whether to re-raise exceptions
        
    Returns:
        Function result or default_return on error
    """
    try:
        result = func()
        return result
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if raise_on_error:
            raise
        
        return default_return


def handle_data_errors(func: Callable) -> Callable:
    """
    Decorator for handling data processing errors gracefully.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"Data file not found in {func.__name__}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid data in {func.__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    return wrapper


def validate_data_integrity(data, required_columns: list, 
                          min_rows: int = 1) -> tuple[bool, str]:
    """
    Validate data integrity for scientific operations.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum required number of rows
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if data is None:
            return False, "Data is None"
        
        if len(data) < min_rows:
            return False, f"Insufficient data: {len(data)} rows < {min_rows} required"
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}"


class ScientificValidationError(Exception):
    """Custom exception for scientific validation failures."""
    pass


class TemporalLeakageError(Exception):
    """Custom exception for temporal data leakage detection."""
    pass


class ModelValidationError(Exception):
    """Custom exception for model validation failures."""
    pass