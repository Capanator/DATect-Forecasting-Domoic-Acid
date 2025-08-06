"""
Logging Configuration Module
============================

Provides centralized logging configuration for the DATect forecasting system.
Supports both console and file logging with configurable levels.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level='INFO', enable_file_logging=False, log_dir='./logs/'):
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        enable_file_logging: Whether to enable file logging
        log_dir: Directory to store log files
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'datect_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"File logging enabled: {log_file}")


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)