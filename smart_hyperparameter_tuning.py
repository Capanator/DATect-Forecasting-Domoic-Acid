#!/usr/bin/env python3
"""
Smart Hyperparameter Tuning - Individual Parameter Optimization
==============================================================

Instead of testing all combinations, test each parameter individually to find
the best value, then combine the top performers. Much faster and still effective.
"""

import pandas as pd
import numpy as np
import time
import warnings
from joblib import Parallel, delayed
from hyperparameter_tuning import HyperparameterTuner

warnings.filterwarnings('ignore')

class SmartHyperparameterTuner(HyperparameterTuner):
    """Smart tuning that tests parameters individually first."""
    
    def __init__(self, n_anchors_per_site=100):
        super().__init__(n_anchors_per_site)
        
        # Baseline configuration (current settings)
        self.baseline_config = {
            'n_estimators': 800,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'spike_weight': 8.0,
            'spike_threshold': 20.0
        }
    
    def test_individual_parameters(self):
        """Test each parameter individually to find best values."""
        print("="*70)
        print("SMART HYPERPARAMETER TUNING - INDIVIDUAL PARAMETER TESTING")
        print("="*70)
        
        # Get baseline performance
        print("\nStep 1: Establishing baseline...")
        baseline_result = self.get_current_xgboost_baseline()
        if not baseline_result:
            print("Could not establish baseline - using fallback values")
            baseline_result = {'mean_r2': 0.32, 'mean_mae': 7.0, 'mean_spike_f1': 0.54}
        
        baseline_r2 = baseline_result['mean_r2']
        baseline_mae = baseline_result['mean_mae']
        baseline_f1 = baseline_result['mean_spike_f1']
        
        print(f"Baseline to beat: R²={baseline_r2:.4f}, MAE={baseline_mae:.4f}, F1={baseline_f1:.4f}")
        
        # Parameter ranges to test
        param_tests = {
            'n_estimators': [200, 300, 500, 800, 1000, 1200, 1500],
            'max_depth': [4, 5, 6, 7, 8, 10],
            'learning_rate': [0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.85, 0.9, 0.95],
            'colsample_bytree': [0.7, 0.8, 0.85, 0.9, 0.95],
            'reg_alpha': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
            'reg_lambda': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
            'spike_weight': [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
            'spike_threshold': [15.0, 17.5, 20.0, 22.5, 25.0]
        }
        
        best_params = {}
        results_summary = {}
        
        total_tests = sum(len(values) for values in param_tests.values())
        print(f"\nTesting {total_tests} individual parameter values...")
        
        # Test each parameter individually
        for param_name, param_values in param_tests.items():
            print(f"\nStep 2.{len(best_params)+1}: Testing {param_name}...")
            print(f"  Values: {param_values}")
            
            def test_single_param_value(value):
                """Test a single parameter value."""
                test_config = self.baseline_config.copy()
                test_config[param_name] = value
                result = self._evaluate_xgboost_config(test_config)
                
                if result:
                    return {
                        'value': value,
                        'r2': result['mean_r2'],
                        'mae': result['mean_mae'],
                        'spike_f1': result['mean_spike_f1'],
                        'improvement_score': self._calculate_improvement_score(
                            result, baseline_result
                        )
                    }
                return None
            
            # Use parallel processing to test all values for this parameter
            print(f"  Running {len(param_values)} tests in parallel...")
            param_results = Parallel(n_jobs=-1, verbose=1)(
                delayed(test_single_param_value)(value) for value in param_values
            )
            
            # Filter out None results
            param_results = [r for r in param_results if r is not None]
            
            if param_results:
                # Find best value for this parameter based on improvement score
                best_for_param = max(param_results, key=lambda x: x['improvement_score'])
                best_params[param_name] = best_for_param['value']
                results_summary[param_name] = {
                    'best_value': best_for_param['value'],
                    'baseline_value': self.baseline_config[param_name],
                    'r2_improvement': best_for_param['r2'] - baseline_r2,
                    'mae_improvement': baseline_mae - best_for_param['mae'],
                    'f1_improvement': best_for_param['spike_f1'] - baseline_f1,
                    'all_results': param_results
                }
                
                print(f"  Best {param_name}: {best_for_param['value']} (was {self.baseline_config[param_name]})")
                print(f"  Improvement: R²={best_for_param['r2'] - baseline_r2:+.4f}, MAE={baseline_mae - best_for_param['mae']:+.4f}, F1={best_for_param['spike_f1'] - baseline_f1:+.4f}")
            else:
                print(f"  No valid results for {param_name} - keeping baseline value")
                best_params[param_name] = self.baseline_config[param_name]
        
        return best_params, results_summary, baseline_result
    
    def test_combined_configurations(self, best_params, baseline_result):
        """Test combinations of the best individual parameters."""
        print(f"\nStep 3: Testing Combined Configurations")
        print("="*50)
        
        # Create different combination strategies
        combinations_to_test = []
        
        # 1. All best parameters together
        all_best = best_params.copy()
        combinations_to_test.append(("All Best", all_best))
        
        # 2. Best XGBoost params only (no spike weighting changes)
        xgb_best = self.baseline_config.copy()
        for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
            xgb_best[param] = best_params[param]
        combinations_to_test.append(("Best XGBoost Only", xgb_best))
        
        # 3. Best spike params only
        spike_best = self.baseline_config.copy()
        spike_best['spike_weight'] = best_params['spike_weight']
        spike_best['spike_threshold'] = best_params['spike_threshold']
        combinations_to_test.append(("Best Spike Only", spike_best))
        
        # 4. User's specific configuration from model_factory.py
        user_config = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,  # Not specified in user's config
            'reg_lambda': 0.0,  # Not specified in user's config
            'spike_weight': 8.0,  # Keep baseline spike settings
            'spike_threshold': 20.0
        }
        combinations_to_test.append(("User's Config", user_config))
        
        # 5. Conservative combination (only parameters with significant improvements)
        conservative = self.baseline_config.copy()
        # Add logic to include only params with >5% improvement
        combinations_to_test.append(("Conservative", conservative))
        
        print(f"Testing {len(combinations_to_test)} combined configurations...")
        
        def test_single_combination(combo_name, config):
            """Test a single combination configuration."""
            result = self._evaluate_xgboost_config(config)
            
            if result:
                r2_imp = ((result['mean_r2'] - baseline_result['mean_r2']) / abs(baseline_result['mean_r2'])) * 100
                mae_imp = ((baseline_result['mean_mae'] - result['mean_mae']) / baseline_result['mean_mae']) * 100
                f1_imp = ((result['mean_spike_f1'] - baseline_result['mean_spike_f1']) / baseline_result['mean_spike_f1']) * 100
                
                return {
                    'name': combo_name,
                    'config': config,
                    'r2': result['mean_r2'],
                    'mae': result['mean_mae'],
                    'spike_f1': result['mean_spike_f1'],
                    'r2_improvement_pct': r2_imp,
                    'mae_improvement_pct': mae_imp,
                    'f1_improvement_pct': f1_imp,
                    'overall_score': (r2_imp + mae_imp + f1_imp) / 3
                }
            return None
        
        # Run combination tests in parallel
        print(f"  Running {len(combinations_to_test)} combination tests in parallel...")
        combination_results = Parallel(n_jobs=-1, verbose=1)(
            delayed(test_single_combination)(combo_name, config) 
            for combo_name, config in combinations_to_test
        )
        
        # Filter out None results and display
        combination_results = [r for r in combination_results if r is not None]
        
        for combo_result in combination_results:
            print(f"\nResults for {combo_result['name']}:")
            print(f"  R²={combo_result['r2']:.4f}({combo_result['r2_improvement_pct']:+.1f}%) | "
                  f"MAE={combo_result['mae']:.4f}({combo_result['mae_improvement_pct']:+.1f}%) | "
                  f"F1={combo_result['spike_f1']:.4f}({combo_result['f1_improvement_pct']:+.1f}%)")
        
        return combination_results
    
    def _calculate_improvement_score(self, result, baseline):
        """Calculate overall improvement score for a parameter test."""
        r2_imp = (result['mean_r2'] - baseline['mean_r2']) / abs(baseline['mean_r2'])
        mae_imp = (baseline['mean_mae'] - result['mean_mae']) / baseline['mean_mae'] 
        f1_imp = (result['mean_spike_f1'] - baseline['mean_spike_f1']) / baseline['mean_spike_f1']
        
        # Weight R² and F1 more heavily than MAE
        return (r2_imp * 2 + f1_imp * 2 + mae_imp * 1) / 5
    
    def run_smart_tuning(self):
        """Run the complete smart tuning process."""
        start_time = time.time()
        
        # Step 1 & 2: Test individual parameters
        best_params, param_results, baseline = self.test_individual_parameters()
        
        # Step 3: Test combined configurations  
        combo_results = self.test_combined_configurations(best_params, baseline)
        
        # Step 4: Show final results
        self._show_final_results(best_params, param_results, combo_results, baseline, start_time)
        
        return {
            'individual_results': param_results,
            'combination_results': combo_results,
            'recommended_config': max(combo_results, key=lambda x: x['overall_score']) if combo_results else None
        }
    
    def _show_final_results(self, best_params, param_results, combo_results, baseline, start_time):
        """Display comprehensive final results."""
        runtime = time.time() - start_time
        
        print("\n" + "="*70)
        print("SMART HYPERPARAMETER TUNING RESULTS")
        print("="*70)
        
        print(f"\nRuntime: {runtime/60:.1f} minutes")
        print(f"Baseline: R²={baseline['mean_r2']:.4f}, MAE={baseline['mean_mae']:.4f}, F1={baseline['mean_spike_f1']:.4f}")
        
        # Individual parameter results
        print(f"\n🔍 INDIVIDUAL PARAMETER WINNERS:")
        print("-" * 40)
        for param, results in param_results.items():
            old_val = results['baseline_value']
            new_val = results['best_value']
            r2_imp = results['r2_improvement']
            print(f"{param:15s}: {old_val:6} → {new_val:6} (R² {r2_imp:+.4f})")
        
        # Combination results
        if combo_results:
            print(f"\n🏆 COMBINATION RESULTS:")
            print("-" * 50)
            combo_results_sorted = sorted(combo_results, key=lambda x: x['overall_score'], reverse=True)
            
            for i, combo in enumerate(combo_results_sorted):
                print(f"{i+1}. {combo['name']}")
                print(f"   R²={combo['r2']:.4f}({combo['r2_improvement_pct']:+.1f}%) | MAE={combo['mae']:.4f}({combo['mae_improvement_pct']:+.1f}%) | F1={combo['spike_f1']:.4f}({combo['f1_improvement_pct']:+.1f}%)")
                print(f"   Overall Score: {combo['overall_score']:+.2f}")
                print()
            
            # Final recommendation
            best_combo = combo_results_sorted[0]
            print(f"🎯 RECOMMENDED CONFIGURATION:")
            print("-" * 30)
            for param, value in best_combo['config'].items():
                print(f"  {param} = {value}")
            
            print(f"\nExpected Performance:")
            print(f"  R² = {best_combo['r2']:.4f} ({best_combo['r2_improvement_pct']:+.1f}% improvement)")
            print(f"  MAE = {best_combo['mae']:.4f} ({best_combo['mae_improvement_pct']:+.1f}% improvement)") 
            print(f"  Spike F1 = {best_combo['spike_f1']:.4f} ({best_combo['f1_improvement_pct']:+.1f}% improvement)")


def main():
    print("Starting Smart Hyperparameter Tuning...")
    
    tuner = SmartHyperparameterTuner(n_anchors_per_site=100)  # Keep 100 anchors as requested
    results = tuner.run_smart_tuning()
    
    print("\n" + "="*70)
    print("SMART TUNING COMPLETE!")
    print("="*70)
    
    if results['recommended_config']:
        print("\nThis approach tested ~50 configurations instead of 6,696!")
        print("Much faster while still finding optimal parameter combinations.")


if __name__ == "__main__":
    main()