#!/usr/bin/env python3
"""
XGBoost Hyperparameter Optimization for DATect
==============================================

Optimizes XGBoost hyperparameters for both regression and classification tasks
using grid search with cross-validation and temporal integrity validation.
"""

import pandas as pd
import numpy as np
import itertools
import json
import time
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import xgboost as xgb

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.data_processor import DataProcessor
from forecasting.validation import validate_runtime_parameters


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for XGBoost models with temporal integrity.
    """
    
    def __init__(self):
        self.engine = ForecastEngine(validate_on_init=False)
        self.data_processor = DataProcessor()
        self.results = []
        
    def get_parameter_grid(self, task="regression"):
        """
        Define hyperparameter search space.
        Focus on most impactful parameters to keep search manageable.
        """
        if task == "regression":
            return {
                'n_estimators': [300, 400, 500, 600],
                'max_depth': [4, 5, 6, 7, 8],
                'learning_rate': [0.03, 0.05, 0.07, 0.1],
                'subsample': [0.8, 0.85, 0.9],
                'colsample_bytree': [0.8, 0.85, 0.9],
                'reg_alpha': [0.0, 0.1, 0.2],
                'reg_lambda': [0.5, 1.0, 2.0],
                'gamma': [0.0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
        else:  # classification
            return {
                'n_estimators': [400, 500, 600, 700],
                'max_depth': [5, 6, 7, 8, 9],
                'learning_rate': [0.02, 0.03, 0.05, 0.07],
                'subsample': [0.85, 0.9, 0.95],
                'colsample_bytree': [0.85, 0.9, 0.95],
                'reg_alpha': [0.0, 0.1, 0.2],
                'reg_lambda': [1.0, 2.0, 3.0],
                'gamma': [0.1, 0.2, 0.3],
                'min_child_weight': [3, 5, 7]
            }
    
    def create_parameter_combinations(self, param_grid, max_combinations=50):
        """
        Create parameter combinations, limiting total to keep optimization tractable.
        Use smart sampling to cover parameter space effectively.
        """
        # Get all parameter names and their possible values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Calculate total possible combinations
        total_combinations = np.prod([len(vals) for vals in param_values])
        print(f"Total possible combinations: {total_combinations}")
        
        if total_combinations <= max_combinations:
            # Use all combinations if small enough
            combinations = list(itertools.product(*param_values))
        else:
            # Sample random combinations
            print(f"Sampling {max_combinations} combinations from {total_combinations} total")
            np.random.seed(config.RANDOM_SEED)
            combinations = []
            
            for _ in range(max_combinations):
                combo = []
                for param_vals in param_values:
                    combo.append(np.random.choice(param_vals))
                combinations.append(tuple(combo))
        
        # Convert to list of parameter dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def evaluate_parameters(self, params, task="regression", n_splits=2):
        """
        Evaluate a single parameter combination using the existing forecasting infrastructure.
        """
        print(f"Evaluating: {params}")
        
        try:
            # Use the existing forecast engine method for evaluation
            # Create multiple forecast scenarios for cross-validation
            data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
            
            # Use Newport as representative site (good data coverage)
            site = 'Newport'
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date')
            
            if len(site_data) < 100:
                print(f"Insufficient data for {site}: {len(site_data)} samples")
                return None
            
            # Create temporal splits for validation
            # Use dates with sufficient training history
            site_data = site_data.dropna(subset=['da'])
            available_dates = site_data['date'].tolist()
            
            if len(available_dates) < 50:
                print(f"Insufficient training dates for {site}")
                return None
                
            # Select validation dates (avoid very early dates with insufficient history)
            start_idx = max(30, len(available_dates) // 4)  # Skip early dates
            end_idx = len(available_dates) - 10  # Leave some buffer at the end
            val_dates = available_dates[start_idx:end_idx:max(1, (end_idx - start_idx) // n_splits)]
            
            if len(val_dates) < n_splits:
                val_dates = available_dates[start_idx:end_idx:max(1, (end_idx - start_idx) // max(1, len(val_dates)))]
            
            scores = []
            
            # Temporarily update model factory with test parameters
            original_method = self.engine.model_factory._get_regression_model if task == "regression" else self.engine.model_factory._get_classification_model
            
            def create_test_model(model_type):
                if model_type == "xgboost" or model_type == "xgb":
                    if not xgb:
                        raise ImportError("XGBoost not available")
                    
                    model_params = {
                        **params,
                        'random_state': config.RANDOM_SEED,
                        'n_jobs': 1,  # Single thread for parallel optimization
                        'tree_method': 'hist'
                    }
                    
                    if task == "regression":
                        return xgb.XGBRegressor(**model_params)
                    else:
                        model_params['eval_metric'] = 'logloss'
                        return xgb.XGBClassifier(**model_params)
                else:
                    return original_method(model_type)
            
            # Monkey patch for testing
            if task == "regression":
                self.engine.model_factory._get_regression_model = create_test_model
            else:
                self.engine.model_factory._get_classification_model = create_test_model
            
            try:
                for val_date in val_dates[:n_splits]:  # Limit to n_splits evaluations
                    try:
                        # Use the existing forecast method
                        result = self.engine.generate_single_forecast(
                            config.FINAL_OUTPUT_PATH,
                            val_date,
                            site,
                            task,
                            "xgboost"
                        )
                        
                        if result is None or 'actual_da' not in result or 'predicted_value' not in result:
                            continue
                            
                        actual = result['actual_da']
                        predicted = result['predicted_value']
                        
                        if actual is None or predicted is None:
                            continue
                        
                        # Calculate score based on task
                        if task == "regression":
                            # Use MAE as proxy (single point R² not meaningful)
                            score = -abs(actual - predicted)  # Negative MAE (higher is better)
                        else:  # classification
                            # Convert to categories
                            actual_cat = self.data_processor.create_da_categories_safe([actual])[0]
                            pred_cat = self.data_processor.create_da_categories_safe([predicted])[0]
                            score = 1.0 if actual_cat == pred_cat else 0.0
                        
                        scores.append(score)
                        
                    except Exception as e:
                        print(f"    Error on {val_date}: {str(e)}")
                        continue
                        
            finally:
                # Restore original method
                if task == "regression":
                    self.engine.model_factory._get_regression_model = original_method
                else:
                    self.engine.model_factory._get_classification_model = original_method
            
            if not scores:
                print("  No successful evaluations")
                return None
                
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            metric_name = "Neg MAE" if task == "regression" else "Accuracy"
            print(f"  {metric_name}: {mean_score:.4f} (±{std_score:.4f}) [{len(scores)} evals]")
            
            return {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'task': task,
                'metric': metric_name,
                'individual_scores': scores,
                'n_evaluations': len(scores)
            }
            
        except Exception as e:
            print(f"  Error evaluating parameters: {str(e)}")
            return None
    
    def optimize_task(self, task="regression", max_combinations=50):
        """
        Optimize hyperparameters for a specific task.
        """
        print(f"\n{'='*60}")
        print(f"Optimizing XGBoost hyperparameters for {task}")
        print(f"{'='*60}")
        
        # Get parameter grid
        param_grid = self.get_parameter_grid(task)
        print(f"Parameter space: {param_grid}")
        
        # Create parameter combinations
        param_combinations = self.create_parameter_combinations(param_grid, max_combinations)
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        # Evaluate each combination
        task_results = []
        start_time = time.time()
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}]", end=" ")
            
            result = self.evaluate_parameters(params, task)
            if result is not None:
                task_results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            if i % 10 == 0:
                avg_time = elapsed / i
                remaining = (len(param_combinations) - i) * avg_time
                print(f"\nProgress: {i}/{len(param_combinations)} "
                      f"({elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining)")
        
        if not task_results:
            print(f"No successful evaluations for {task}")
            return None
        
        # Find best parameters
        best_result = max(task_results, key=lambda x: x['mean_score'])
        
        print(f"\n{'='*40}")
        print(f"Best {task} results:")
        print(f"{'='*40}")
        print(f"Best {best_result['metric']}: {best_result['mean_score']:.4f} (±{best_result['std_score']:.4f})")
        print(f"Best parameters:")
        for param, value in best_result['params'].items():
            print(f"  {param}: {value}")
        
        return {
            'task': task,
            'best_result': best_result,
            'all_results': task_results
        }
    
    def run_optimization(self, tasks=["regression", "classification"], max_combinations=50):
        """
        Run hyperparameter optimization for specified tasks.
        """
        print("DATect XGBoost Hyperparameter Optimization")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data file: {config.FINAL_OUTPUT_PATH}")
        
        all_results = {}
        
        for task in tasks:
            result = self.optimize_task(task, max_combinations)
            if result:
                all_results[task] = result
        
        # Save results
        output_file = f"hyperparameter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for task, result in all_results.items():
            json_results[task] = {
                'best_params': result['best_result']['params'],
                'best_score': result['best_result']['mean_score'],
                'best_std': result['best_result']['std_score'],
                'metric': result['best_result']['metric'],
                'total_combinations_tested': len(result['all_results'])
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_file}")
        
        for task, result in all_results.items():
            best = result['best_result']
            print(f"\n{task.upper()} - Best {best['metric']}: {best['mean_score']:.4f}")
            print("Recommended parameters:")
            for param, value in best['params'].items():
                print(f"  '{param}': {value},")
        
        return all_results


def main():
    """
    Run hyperparameter optimization.
    """
    print("Starting XGBoost hyperparameter optimization...")
    
    # Validate system before optimization
    try:
        validate_runtime_parameters()
        print("✓ Runtime validation passed")
    except Exception as e:
        print(f"✗ Runtime validation failed: {e}")
        return
    
    optimizer = HyperparameterOptimizer()
    
    # Run optimization for both tasks
    # Use smaller max_combinations for faster testing (increase for thorough search)
    results = optimizer.run_optimization(
        tasks=["regression", "classification"],
        max_combinations=10  # Small number for quick testing
    )
    
    print("\nOptimization completed!")
    print("Review the results and update model_factory.py with the best parameters.")


if __name__ == "__main__":
    main()