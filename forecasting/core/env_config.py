"""
Environment Configuration Management
===================================

Production-ready environment variable configuration for DATect forecasting system.
Provides secure, flexible configuration management with validation and defaults.

Features:
- Environment variable loading with type conversion
- Configuration validation and sanitization  
- Secure handling of sensitive values
- Development/staging/production environment support
- Configuration override hierarchy

Usage:
    from forecasting.core.env_config import EnvironmentConfig
    
    config = EnvironmentConfig()
    
    # Access configuration values
    forecast_model = config.FORECAST_MODEL
    database_url = config.DATABASE_URL
    log_level = config.LOG_LEVEL
"""

import os
import logging
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

from .logging_config import get_logger


class EnvironmentConfig:
    """
    Environment-based configuration management with validation and defaults.
    
    Supports multiple configuration sources in priority order:
    1. Environment variables (highest priority)
    2. .env file 
    3. Default values (lowest priority)
    
    Features:
    - Type conversion and validation
    - Required vs optional configuration
    - Environment-specific overrides
    - Secure handling of sensitive data
    """
    
    def __init__(self, env_file: Optional[str] = None, environment: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.environment = environment or self._detect_environment()
        
        # Load configuration from multiple sources
        self._load_dotenv(env_file)
        self._load_configuration()
        self._validate_configuration()
        
        self.logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _detect_environment(self) -> str:
        """Detect current environment from environment variables."""
        return os.getenv('DATECT_ENVIRONMENT', os.getenv('ENVIRONMENT', 'development')).lower()
    
    def _load_dotenv(self, env_file: Optional[str]):
        """Load environment variables from .env file if available."""
        try:
            # Try to import python-dotenv (optional dependency)
            from dotenv import load_dotenv
            
            if env_file:
                env_path = Path(env_file)
            else:
                # Look for environment-specific .env files
                possible_files = [
                    f'.env.{self.environment}',
                    '.env.local', 
                    '.env'
                ]
                env_path = None
                for filename in possible_files:
                    if Path(filename).exists():
                        env_path = Path(filename)
                        break
            
            if env_path and env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment variables from: {env_path}")
            else:
                self.logger.info("No .env file found, using system environment variables only")
                
        except ImportError:
            self.logger.info("python-dotenv not installed, using system environment variables only")
        except Exception as e:
            self.logger.warning(f"Error loading .env file: {e}")
    
    def _load_configuration(self):
        """Load all configuration values with type conversion and defaults."""
        
        # =================================================================
        # CORE APPLICATION SETTINGS
        # =================================================================
        
        self.FORECAST_MODE = self._get_env_choice(
            'DATECT_FORECAST_MODE', 
            default='retrospective',
            choices=['retrospective', 'realtime'],
            description="Operating mode for forecasting system"
        )
        
        self.FORECAST_TASK = self._get_env_choice(
            'DATECT_FORECAST_TASK',
            default='regression', 
            choices=['regression', 'classification'],
            description="Type of prediction task"
        )
        
        self.FORECAST_MODEL = self._get_env_choice(
            'DATECT_FORECAST_MODEL',
            default='xgboost',
            choices=['xgboost', 'ridge'],
            description="Machine learning algorithm to use"
        )
        
        # =================================================================
        # DATA AND FILE PATHS
        # =================================================================
        
        self.FINAL_OUTPUT_PATH = self._get_env_str(
            'DATECT_FINAL_OUTPUT_PATH',
            default='final_output.parquet',
            description="Path to processed data file"
        )
        
        self.MODEL_ARTIFACTS_DIR = self._get_env_str(
            'DATECT_MODEL_ARTIFACTS_DIR',
            default='./model_artifacts/',
            description="Directory for saved model artifacts"
        )
        
        self.LOGS_DIR = self._get_env_str(
            'DATECT_LOGS_DIR',
            default='./logs/',
            description="Directory for log files"
        )
        
        # =================================================================
        # PERFORMANCE AND SCALING
        # =================================================================
        
        self.N_RANDOM_ANCHORS = self._get_env_int(
            'DATECT_N_RANDOM_ANCHORS',
            default=200,
            min_value=1,
            max_value=1000,
            description="Number of random anchor points for evaluation"
        )
        
        self.MIN_TRAINING_SAMPLES = self._get_env_int(
            'DATECT_MIN_TRAINING_SAMPLES',
            default=3,
            min_value=1,
            max_value=100,
            description="Minimum samples required to train a model"
        )
        
        self.TEMPORAL_BUFFER_DAYS = self._get_env_int(
            'DATECT_TEMPORAL_BUFFER_DAYS',
            default=1,
            min_value=0,
            max_value=30,
            description="Minimum days between training and prediction"
        )
        
        self.PARALLEL_JOBS = self._get_env_int(
            'DATECT_PARALLEL_JOBS',
            default=-1,  # Use all available cores
            min_value=-1,
            max_value=64,
            description="Number of parallel jobs (-1 for all cores)"
        )
        
        # =================================================================
        # NETWORK AND SERVICES
        # =================================================================
        
        self.DASHBOARD_PORT = self._get_env_int(
            'DATECT_DASHBOARD_PORT',
            default=8065,
            min_value=1024,
            max_value=65535,
            description="Port for web dashboard"
        )
        
        self.API_PORT = self._get_env_int(
            'DATECT_API_PORT', 
            default=8000,
            min_value=1024,
            max_value=65535,
            description="Port for prediction API service"
        )
        
        self.API_HOST = self._get_env_str(
            'DATECT_API_HOST',
            default='localhost',
            description="Host address for API service"
        )
        
        # =================================================================
        # LOGGING AND MONITORING
        # =================================================================
        
        self.LOG_LEVEL = self._get_env_choice(
            'DATECT_LOG_LEVEL',
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            description="Logging level"
        )
        
        self.ENABLE_FILE_LOGGING = self._get_env_bool(
            'DATECT_ENABLE_FILE_LOGGING',
            default=True,
            description="Enable logging to files"
        )
        
        self.MAX_LOG_FILE_SIZE_MB = self._get_env_int(
            'DATECT_MAX_LOG_FILE_SIZE_MB',
            default=10,
            min_value=1,
            max_value=100,
            description="Maximum log file size in MB"
        )
        
        # =================================================================
        # EXTERNAL SERVICES (Optional - may be None in development)
        # =================================================================
        
        self.DATABASE_URL = self._get_env_str(
            'DATABASE_URL',
            default=None,
            required=False,
            description="Database connection URL (optional)"
        )
        
        self.REDIS_URL = self._get_env_str(
            'REDIS_URL', 
            default=None,
            required=False,
            description="Redis connection URL for caching (optional)"
        )
        
        # =================================================================
        # SECURITY AND AUTHENTICATION
        # =================================================================
        
        self.SECRET_KEY = self._get_env_str(
            'DATECT_SECRET_KEY',
            default=None,
            required=(self.environment == 'production'),
            description="Secret key for API authentication"
        )
        
        self.API_KEY = self._get_env_str(
            'DATECT_API_KEY',
            default=None,
            required=False,
            description="API key for external service authentication"
        )
        
        # =================================================================
        # DEVELOPMENT AND DEBUG SETTINGS
        # =================================================================
        
        self.DEBUG = self._get_env_bool(
            'DATECT_DEBUG',
            default=(self.environment == 'development'),
            description="Enable debug mode"
        )
        
        self.ENABLE_PROFILING = self._get_env_bool(
            'DATECT_ENABLE_PROFILING',
            default=False,
            description="Enable performance profiling"
        )
        
        self.SAVE_INTERMEDIATE_RESULTS = self._get_env_bool(
            'DATECT_SAVE_INTERMEDIATE_RESULTS',
            default=(self.environment != 'production'),
            description="Save intermediate processing results"
        )
    
    def _get_env_str(self, key: str, default: Optional[str] = None, 
                     required: bool = True, description: str = "") -> Optional[str]:
        """Get string environment variable with validation."""
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set. {description}")
        
        if value is not None:
            value = value.strip()
            if value == "":
                value = None
        
        self.logger.debug(f"Config {key}: {value} (default: {default})")
        return value
    
    def _get_env_int(self, key: str, default: int, min_value: int = None, 
                     max_value: int = None, description: str = "") -> int:
        """Get integer environment variable with validation."""
        value_str = os.getenv(key)
        
        if value_str is None:
            value = default
        else:
            try:
                value = int(value_str.strip())
            except ValueError:
                self.logger.warning(f"Invalid integer for {key}: {value_str}, using default: {default}")
                value = default
        
        # Validate range
        if min_value is not None and value < min_value:
            self.logger.warning(f"{key} value {value} below minimum {min_value}, using minimum")
            value = min_value
        
        if max_value is not None and value > max_value:
            self.logger.warning(f"{key} value {value} above maximum {max_value}, using maximum")  
            value = max_value
        
        self.logger.debug(f"Config {key}: {value} (default: {default})")
        return value
    
    def _get_env_bool(self, key: str, default: bool, description: str = "") -> bool:
        """Get boolean environment variable with validation."""
        value_str = os.getenv(key)
        
        if value_str is None:
            value = default
        else:
            value_str = value_str.strip().lower()
            if value_str in ('true', '1', 'yes', 'on', 'enabled'):
                value = True
            elif value_str in ('false', '0', 'no', 'off', 'disabled'):
                value = False
            else:
                self.logger.warning(f"Invalid boolean for {key}: {value_str}, using default: {default}")
                value = default
        
        self.logger.debug(f"Config {key}: {value} (default: {default})")
        return value
    
    def _get_env_choice(self, key: str, default: str, choices: List[str], 
                       description: str = "") -> str:
        """Get choice environment variable with validation."""
        value = os.getenv(key, default).strip().lower()
        
        if value not in [choice.lower() for choice in choices]:
            self.logger.warning(f"Invalid choice for {key}: {value}, using default: {default}")
            value = default
        
        self.logger.debug(f"Config {key}: {value} (default: {default}, choices: {choices})")
        return value
    
    def _validate_configuration(self):
        """Validate configuration consistency and requirements."""
        errors = []
        warnings = []
        
        # Production environment validations
        if self.environment == 'production':
            if self.SECRET_KEY is None:
                errors.append("SECRET_KEY is required for production environment")
            
            if self.DEBUG:
                warnings.append("DEBUG mode enabled in production environment")
        
        # Network port conflicts
        if self.DASHBOARD_PORT == self.API_PORT:
            errors.append(f"DASHBOARD_PORT and API_PORT cannot be the same: {self.DASHBOARD_PORT}")
        
        # Performance validations
        if self.N_RANDOM_ANCHORS > 500 and self.environment == 'production':
            warnings.append(f"High N_RANDOM_ANCHORS ({self.N_RANDOM_ANCHORS}) may impact production performance")
        
        # Log validation errors and warnings
        for error in errors:
            self.logger.error(f"Configuration Error: {error}")
        
        for warning in warnings:
            self.logger.warning(f"Configuration Warning: {warning}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get sanitized configuration summary for logging/debugging."""
        config_dict = {}
        
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name.isupper():
                value = getattr(self, attr_name)
                
                # Sanitize sensitive values
                if any(sensitive in attr_name.lower() for sensitive in ['key', 'password', 'secret', 'token']):
                    if value:
                        config_dict[attr_name] = '***REDACTED***'
                    else:
                        config_dict[attr_name] = None
                else:
                    config_dict[attr_name] = value
        
        return config_dict
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding sensitive values)."""
        return self.get_config_summary()


# Global configuration instance
# Initialize once and reuse throughout the application
_config_instance = None

def get_config(reload: bool = False) -> EnvironmentConfig:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        reload: If True, reload configuration from environment
        
    Returns:
        EnvironmentConfig instance
    """
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = EnvironmentConfig()
    
    return _config_instance