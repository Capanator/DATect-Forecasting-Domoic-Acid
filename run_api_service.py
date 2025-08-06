#!/usr/bin/env python3
"""
DATect API Service Runner
=========================

Production-ready launcher for the DATect prediction API service.
Handles configuration, logging setup, and graceful service management.

Usage:
    python run_api_service.py
    
    # With environment overrides
    DATECT_API_PORT=8080 DATECT_LOG_LEVEL=DEBUG python run_api_service.py
    
    # Production deployment
    DATECT_ENVIRONMENT=production DATECT_API_KEY=your-key python run_api_service.py
"""

import sys
import signal
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from forecasting.core.env_config import get_config
from forecasting.core.logging_config import setup_logging, get_logger


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for API service."""
    try:
        # Load configuration
        config = get_config()
        
        # Setup logging with configuration
        log_level = getattr(logging, config.LOG_LEVEL.upper())
        setup_logging(
            log_level=log_level,
            log_dir=config.LOGS_DIR,
            enable_file_logging=config.ENABLE_FILE_LOGGING
        )
        
        global logger
        logger = get_logger(__name__)
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Log startup information
        logger.info("="*60)
        logger.info("DATect Prediction API Service")
        logger.info("="*60)
        logger.info(f"Environment: {config.environment}")
        logger.info(f"API Host: {config.API_HOST}:{config.API_PORT}")
        logger.info(f"Debug Mode: {config.DEBUG}")
        logger.info(f"Log Level: {config.LOG_LEVEL}")
        logger.info(f"Model Artifacts: {config.MODEL_ARTIFACTS_DIR}")
        
        # Import and run API service
        from forecasting.api.service import run_api_server
        
        logger.info("Starting FastAPI server...")
        run_api_server()
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start API service: {str(e)}")
        logger.error(f"Startup error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()