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
from forecasting.dashboard.retrospective import RetrospectiveDashboard
from forecasting.dashboard.realtime import RealtimeDashboard

# Import logging and exception handling
from forecasting.core.logging_config import setup_logging, get_logger
from forecasting.core.exception_handling import ScientificValidationError

# Setup logging with less verbose temporal warnings for retrospective mode  
setup_logging(log_level='INFO', enable_file_logging=True)
logger = get_logger(__name__)


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
        self.data_path = data_path
        self.forecast_engine = ForecastEngine(data_file=data_path)
        self.model_factory = ModelFactory()
        
        # Validate configuration
        self._validate_config()
        
    def _get_actual_model_name(self, ui_model, task):
        """Map UI model selection to actual model names based on task."""
        if ui_model == "xgboost":
            return "xgboost"  # XGBoost works for both regression and classification
        elif ui_model == "ridge":
            if task == "regression":
                return "ridge"  # Ridge regression
            else:
                return "logistic"  # Logistic regression for classification
        else:
            return ui_model  # Fallback to original name
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Check file exists
        try:
            data = pd.read_parquet(self.data_path)
            print(f"[INFO] Loaded data: {len(data)} records across {data['site'].nunique()} sites")
        except Exception as e:
            print(f"[ERROR] Cannot load data file {self.data_path}: {e}")
            sys.exit(1)
            
        # Map model name for validation
        actual_model = self._get_actual_model_name(config.FORECAST_MODEL, config.FORECAST_TASK)
        
        # Validate model/task combination
        if not self.model_factory.validate_model_task_combination(config.FORECAST_TASK, actual_model):
            supported = self.model_factory.get_supported_models(config.FORECAST_TASK)
            print(f"[ERROR] Model '{actual_model}' not supported for task '{config.FORECAST_TASK}'")
            print(f"[ERROR] Supported models for {config.FORECAST_TASK}: {supported[config.FORECAST_TASK]}")
            sys.exit(1)
            
        print(f"[INFO] Configuration validated successfully")
        print(f"[INFO] Mode: {config.FORECAST_MODE}")
        print(f"[INFO] Task: {config.FORECAST_TASK}")
        print(f"[INFO] Model: {config.FORECAST_MODEL} → {actual_model} ({self.model_factory.get_model_description(actual_model)})")
        
    def run_retrospective_evaluation(self):
        """
        Run retrospective forecasting evaluation with random anchors.
        
        Returns:
            DataFrame with evaluation results
        """
        try:
            logger.info(f"Starting retrospective evaluation with {config.N_RANDOM_ANCHORS} anchors")
            print(f"\n[INFO] Starting retrospective evaluation with {config.N_RANDOM_ANCHORS} random anchors")
            print(f"[INFO] Temporal buffer: {config.TEMPORAL_BUFFER_DAYS} days")
            print(f"[INFO] Minimum training samples: {config.MIN_TRAINING_SAMPLES}")
            
            # Run evaluation with mapped model name
            actual_model = self._get_actual_model_name(config.FORECAST_MODEL, config.FORECAST_TASK)
            results_df = self.forecast_engine.run_retrospective_evaluation(
                task=config.FORECAST_TASK,
                model_type=actual_model,
                n_anchors=config.N_RANDOM_ANCHORS
            )
            
            if results_df is not None and not results_df.empty:
                logger.info(f"Generated {len(results_df)} forecasts successfully")
                print(f"\n[SUCCESS] Generated {len(results_df)} forecasts")
                
                # Calculate and display metrics
                self._display_evaluation_metrics(results_df)
                
                # Launch dashboard
                print(f"\n[INFO] Launching retrospective dashboard...")
                dashboard = RetrospectiveDashboard(results_df)
                dashboard.run(port=config.RETROSPECTIVE_PORT, debug=False)
                logger.info("Retrospective evaluation completed successfully")
                return results_df
            else:
                error_msg = "No valid forecasts generated - check data quality and parameters"
                logger.error(error_msg)
                print("[ERROR] No forecasts generated. Check data and configuration.")
                raise ScientificValidationError(error_msg)
                
        except Exception as e:
            logger.error(f"Retrospective evaluation failed: {str(e)}")
            raise
        
    def run_realtime_forecasting(self):
        """Launch interactive real-time forecasting dashboard."""
        try:
            logger.info("Starting real-time forecasting dashboard")
            logger.info("Using XGBoost model for real-time forecasting")
            
            print(f"\n[INFO] Launching real-time forecasting dashboard...")
            print(f"[INFO] Model: XGBoost (from config)")
            print(f"[INFO] Tasks: Both regression (primary) & classification displayed")
            
            dashboard = RealtimeDashboard(self.data_path)
            logger.info(f"Launching dashboard on port {config.DASHBOARD_PORT}")
            dashboard.run(port=config.DASHBOARD_PORT, debug=False)
            
        except Exception as e:
            logger.error(f"Real-time forecasting dashboard failed: {str(e)}")
            raise ScientificValidationError(f"Dashboard launch failed: {str(e)}")
        
    def _display_evaluation_metrics(self, results_df):
        """Display evaluation metrics summary."""
        try:
            logger.info("Computing and displaying evaluation metrics")
            
            print("\n" + "="*60)
            print("FORECASTING EVALUATION RESULTS")
            print("="*60)
        
            # Regression metrics
            if 'predicted_da' in results_df.columns:
                valid_reg = results_df.dropna(subset=['actual_da', 'predicted_da'])
                if not valid_reg.empty:
                    from sklearn.metrics import r2_score, mean_absolute_error
                    r2 = r2_score(valid_reg['actual_da'], valid_reg['predicted_da'])
                    mae = mean_absolute_error(valid_reg['actual_da'], valid_reg['predicted_da'])
                    
                    logger.info(f"Regression metrics: R²={r2:.4f}, MAE={mae:.2f}, n={len(valid_reg)}")
                    
                    print(f"REGRESSION PERFORMANCE:")
                    print(f"  R² Score: {r2:.4f}")
                    print(f"  MAE: {mae:.2f} μg/g")
                    print(f"  Valid forecasts: {len(valid_reg)}")
                
            # Classification metrics
            if 'predicted_category' in results_df.columns:
                valid_cls = results_df.dropna(subset=['actual_category', 'predicted_category'])
                if not valid_cls.empty:
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(valid_cls['actual_category'], valid_cls['predicted_category'])
                    
                    logger.info(f"Classification metrics: Accuracy={accuracy:.4f}, n={len(valid_cls)}")
                    
                    print(f"CLASSIFICATION PERFORMANCE:")
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  Valid forecasts: {len(valid_cls)}")
                
            # Site breakdown
            site_counts = results_df['site'].value_counts()
            logger.info(f"Site distribution: {dict(site_counts)}")
            print(f"\nFORECASTS BY SITE:")
            
        except Exception as e:
            logger.error(f"Failed to compute evaluation metrics: {str(e)}")
            print(f"[ERROR] Failed to compute metrics: {str(e)}")
            raise
        for site, count in site_counts.items():
            print(f"  {site}: {count}")
            
        print("="*60)
        print("All forecasts generated with temporal safeguards")
        print("="*60)


def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Domoic Acid Forecasting System")
        
        print("Domoic Acid Forecasting System (Modular)")
        print("=" * 50)
        print(f"Version: Modular Architecture")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        logger.info(f"System started in {config.FORECAST_MODE} mode")
        
        # Initialize application
        logger.info("Initializing forecast application")
        app = LeakFreeForecastApp(config.FINAL_OUTPUT_PATH)
        
        # Run based on configuration mode
        if config.FORECAST_MODE == "retrospective":
            logger.info("Running retrospective evaluation mode")
            app.run_retrospective_evaluation()
        elif config.FORECAST_MODE == "realtime":
            logger.info("Running realtime forecasting mode")
            app.run_realtime_forecasting()
        else:
            error_msg = f"Unknown forecast mode: {config.FORECAST_MODE}. Supported modes: 'retrospective', 'realtime'"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            raise ScientificValidationError(error_msg)
            
        logger.info("Domoic Acid Forecasting System completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        print(f"[ERROR] Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()