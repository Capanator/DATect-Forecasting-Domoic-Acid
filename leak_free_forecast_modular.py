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
    python leak_free_forecast_modular.py

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
        
    def _validate_config(self):
        """Validate configuration settings."""
        # Check file exists
        try:
            data = pd.read_parquet(self.data_path)
            print(f"[INFO] Loaded data: {len(data)} records across {data['site'].nunique()} sites")
        except Exception as e:
            print(f"[ERROR] Cannot load data file {self.data_path}: {e}")
            sys.exit(1)
            
        # Validate model/task combination
        if not self.model_factory.validate_model_task_combination(config.FORECAST_TASK, config.FORECAST_MODEL):
            supported = self.model_factory.get_supported_models(config.FORECAST_TASK)
            print(f"[ERROR] Model '{config.FORECAST_MODEL}' not supported for task '{config.FORECAST_TASK}'")
            print(f"[ERROR] Supported models for {config.FORECAST_TASK}: {supported[config.FORECAST_TASK]}")
            sys.exit(1)
            
        print(f"[INFO] Configuration validated successfully")
        print(f"[INFO] Mode: {config.FORECAST_MODE}")
        print(f"[INFO] Task: {config.FORECAST_TASK}")
        print(f"[INFO] Model: {config.FORECAST_MODEL} ({self.model_factory.get_model_description(config.FORECAST_MODEL)})")
        
    def run_retrospective_evaluation(self):
        """
        Run retrospective forecasting evaluation with random anchors.
        
        Returns:
            DataFrame with evaluation results
        """
        print(f"\n[INFO] Starting retrospective evaluation with {config.N_RANDOM_ANCHORS} random anchors")
        print(f"[INFO] Temporal buffer: {config.TEMPORAL_BUFFER_DAYS} days")
        print(f"[INFO] Minimum training samples: {config.MIN_TRAINING_SAMPLES}")
        
        # Run evaluation
        results_df = self.forecast_engine.run_retrospective_evaluation(
            task=config.FORECAST_TASK,
            model_type=config.FORECAST_MODEL,
            n_anchors=config.N_RANDOM_ANCHORS
        )
        
        if results_df is not None and not results_df.empty:
            print(f"\n[SUCCESS] Generated {len(results_df)} forecasts")
            
            # Calculate and display metrics
            self._display_evaluation_metrics(results_df)
            
            # Launch dashboard
            print(f"\n[INFO] Launching retrospective dashboard...")
            dashboard = RetrospectiveDashboard(results_df)
            dashboard.run(port=8071, debug=False)
            
        else:
            print("[ERROR] No forecasts generated. Check data and configuration.")
            
        return results_df
        
    def run_realtime_forecasting(self):
        """Launch interactive real-time forecasting dashboard."""
        print(f"\n[INFO] Launching real-time forecasting dashboard...")
        print(f"[INFO] Model: Random Forest (fixed for realtime)")
        print(f"[INFO] Tasks: Both regression (primary) & classification displayed")
        
        dashboard = RealtimeDashboard(self.data_path)
        dashboard.run(port=8065, debug=False)
        
    def _display_evaluation_metrics(self, results_df):
        """Display evaluation metrics summary."""
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
                
                print(f"CLASSIFICATION PERFORMANCE:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Valid forecasts: {len(valid_cls)}")
                
        # Site breakdown
        site_counts = results_df['site'].value_counts()
        print(f"\nFORECASTS BY SITE:")
        for site, count in site_counts.items():
            print(f"  {site}: {count}")
            
        print("="*60)
        print("All forecasts generated with temporal safeguards")
        print("="*60)


def main():
    """Main entry point for the application."""
    print("Domoic Acid Forecasting System (Modular)")
    print("=" * 50)
    print(f"Version: Modular Architecture")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Initialize application
    app = LeakFreeForecastApp(config.FINAL_OUTPUT_PATH)
    
    # Run based on configuration mode
    if config.FORECAST_MODE == "retrospective":
        app.run_retrospective_evaluation()
    elif config.FORECAST_MODE == "realtime":
        app.run_realtime_forecasting()
    else:
        print(f"[ERROR] Unknown forecast mode: {config.FORECAST_MODE}")
        print(f"[ERROR] Supported modes: 'retrospective', 'realtime'")
        sys.exit(1)


if __name__ == "__main__":
    main()