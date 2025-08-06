#!/usr/bin/env python3
"""
Domoic Acid Forecasting System (Modular)
==================================================

Modular version of the DA forecasting system with clean architecture.
Provides both retrospective evaluation and real-time forecasting capabilities
with strict temporal safeguards to prevent data leakage.

This system is designed for scientific publication and maintains complete
temporal integrity throughout all forecasting operations.

Usage:
    python modular-forecast.py

Configuration is managed through config.py - see that file for all options.
"""

import sys
import pandas as pd
from datetime import datetime

# Import configuration and modular components
import config
from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.model_factory import ModelFactory
from forecasting.core.logging_config import setup_logging, get_logger
from forecasting.core.exception_handling import safe_execute
from forecasting.dashboard.retrospective import RetrospectiveDashboard
from forecasting.dashboard.realtime import RealtimeDashboard


class LeakFreeForecastApp:
    """
    Main application class for the DA forecasting system.
    
    Coordinates between different components while maintaining temporal safeguards.
    """
    
    def __init__(self, data_path):
        """
        Initialize the forecasting application.
        
        Args:
            data_path: Path to processed data file
        """
        # Setup logging
        setup_logging(log_level="INFO", log_dir="./logs/", enable_file_logging=True)
        self.logger = get_logger(__name__)
        
        self.data_path = data_path
        self.forecast_engine = ForecastEngine(data_file=data_path)
        self.model_factory = ModelFactory()
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration settings."""
        # Check file exists
        try:
            data = pd.read_parquet(self.data_path)
            self.logger.info(f"Loaded data: {len(data)} records across {data['site'].nunique()} sites")
        except Exception as e:
            self.logger.error(f"Cannot load data file {self.data_path}: {e}")
            sys.exit(1)
            
        # Validate model/task combination
        if not self.model_factory.validate_model_task_combination(config.FORECAST_TASK, config.FORECAST_MODEL):
            supported = self.model_factory.get_supported_models(config.FORECAST_TASK)
            self.logger.error(f"Model '{config.FORECAST_MODEL}' not supported for task '{config.FORECAST_TASK}'")
            self.logger.error(f"Supported models for {config.FORECAST_TASK}: {supported[config.FORECAST_TASK]}")
            sys.exit(1)
            
        self.logger.info("Configuration validated successfully")
        self.logger.info(f"Mode: {config.FORECAST_MODE}")
        self.logger.info(f"Task: {config.FORECAST_TASK}")
        self.logger.info(f"Model: {config.FORECAST_MODEL} ({self.model_factory.get_model_description(config.FORECAST_MODEL)})")
        
    def run_retrospective_evaluation(self):
        """
        Run retrospective forecasting evaluation with random anchors.
        
        Returns:
            DataFrame with evaluation results
        """
        self.logger.info(f"Starting retrospective evaluation with {config.N_RANDOM_ANCHORS} random anchors")
        self.logger.info(f"Temporal buffer: {config.TEMPORAL_BUFFER_DAYS} days")
        self.logger.info(f"Minimum training samples: {config.MIN_TRAINING_SAMPLES}")
        
        # Run evaluation with error handling
        results_df = safe_execute(
            self.forecast_engine.run_retrospective_evaluation,
            task=config.FORECAST_TASK,
            model_type=config.FORECAST_MODEL,
            n_anchors=config.N_RANDOM_ANCHORS,
            fallback_value=None,
            context="retrospective_evaluation"
        )
        
        if results_df is not None and not results_df.empty:
            self.logger.info(f"Generated {len(results_df)} forecasts successfully")
            
            # Calculate and display metrics
            self._display_evaluation_metrics(results_df)
            
            # Launch dashboard
            self.logger.info("Launching retrospective dashboard on port 8071...")
            dashboard = RetrospectiveDashboard(results_df)
            dashboard.run(port=8071, debug=False)
            
        else:
            self.logger.error("No forecasts generated. Check data and configuration.")
            
        return results_df
        
    def run_realtime_forecasting(self):
        """Launch interactive real-time forecasting dashboard."""
        self.logger.info("Launching real-time forecasting dashboard on port 8065...")
        self.logger.info(f"Available models: {config.FORECAST_MODEL} and Ridge Regression")
        self.logger.info("Tasks: Both regression and classification available")
        
        dashboard = RealtimeDashboard(self.data_path)
        dashboard.run(port=8065, debug=False)
        
    def _display_evaluation_metrics(self, results_df):
        """Display evaluation metrics summary."""
        self.logger.info("="*60)
        self.logger.info("FORECASTING EVALUATION RESULTS")
        self.logger.info("="*60)
        
        # Regression metrics
        if 'predicted_da' in results_df.columns:
            valid_reg = results_df.dropna(subset=['actual_da', 'predicted_da'])
            if not valid_reg.empty:
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(valid_reg['actual_da'], valid_reg['predicted_da'])
                mae = mean_absolute_error(valid_reg['actual_da'], valid_reg['predicted_da'])
                
                self.logger.info("REGRESSION PERFORMANCE:")
                self.logger.info(f"  R² Score: {r2:.4f}")
                self.logger.info(f"  MAE: {mae:.2f} μg/g")
                self.logger.info(f"  Valid forecasts: {len(valid_reg)}")
                
        # Classification metrics
        if 'predicted_category' in results_df.columns:
            valid_cls = results_df.dropna(subset=['actual_category', 'predicted_category'])
            if not valid_cls.empty:
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(valid_cls['actual_category'], valid_cls['predicted_category'])
                
                self.logger.info("CLASSIFICATION PERFORMANCE:")
                self.logger.info(f"  Accuracy: {accuracy:.4f}")
                self.logger.info(f"  Valid forecasts: {len(valid_cls)}")
                
        # Site breakdown
        site_counts = results_df['site'].value_counts()
        self.logger.info("FORECASTS BY SITE:")
        for site, count in site_counts.items():
            self.logger.info(f"  {site}: {count}")
            
        self.logger.info("="*60)
        self.logger.info("All forecasts generated with temporal safeguards")
        self.logger.info("="*60)


def main():
    """Main entry point for the application."""
    try:
        # Setup basic logging first
        setup_logging(log_level="INFO", log_dir="./logs/", enable_file_logging=True)
        logger = get_logger("main")
        
        logger.info("Domoic Acid Forecasting System (Modular)")
        logger.info("=" * 50)
        logger.info("Version: Modular Architecture with Production Logging")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        # Initialize application with error handling
        app = safe_execute(
            LeakFreeForecastApp,
            config.FINAL_OUTPUT_PATH,
            fallback_value=None,
            context="application_initialization"
        )
        
        if app is None:
            logger.error("Failed to initialize forecasting application")
            sys.exit(1)
        
        # Run based on configuration mode
        if config.FORECAST_MODE == "retrospective":
            app.run_retrospective_evaluation()
        elif config.FORECAST_MODE == "realtime":
            app.run_realtime_forecasting()
        else:
            logger.error(f"Unknown forecast mode: {config.FORECAST_MODE}")
            logger.error("Supported modes: 'retrospective', 'realtime'")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()