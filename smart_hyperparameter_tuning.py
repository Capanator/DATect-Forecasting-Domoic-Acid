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
import sys
import warnings
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
        best_r2_params = {}  # Best for R² specifically
        best_f1_params = {}  # Best for F1 specifically
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
            
            # Use sequential processing to avoid import issues
            print(f"  Running {len(param_values)} tests sequentially...")
            param_results = []
            for i, value in enumerate(param_values):
                print(f"    Testing {param_name}={value} ({i+1}/{len(param_values)})...")
                result = test_single_param_value(value)
                if result:
                    param_results.append(result)
                    print(f"    → R²={result['r2']:.4f}, F1={result['spike_f1']:.4f}")
                else:
                    print(f"    → Failed")
            
            if param_results:
                # Find best values for different metrics
                best_for_param = max(param_results, key=lambda x: x['improvement_score'])
                best_r2_param = max(param_results, key=lambda x: x['r2'])
                best_f1_param = max(param_results, key=lambda x: x['spike_f1'])
                
                best_params[param_name] = best_for_param['value']
                best_r2_params[param_name] = best_r2_param['value'] 
                best_f1_params[param_name] = best_f1_param['value']
                
                results_summary[param_name] = {
                    'best_combined': best_for_param['value'],
                    'best_r2': best_r2_param['value'],
                    'best_f1': best_f1_param['value'],
                    'baseline_value': self.baseline_config[param_name],
                    'r2_improvement': best_for_param['r2'] - baseline_r2,
                    'mae_improvement': baseline_mae - best_for_param['mae'],
                    'f1_improvement': best_for_param['spike_f1'] - baseline_f1,
                    'all_results': param_results
                }
                
                print(f"  Best {param_name}: {best_for_param['value']} (was {self.baseline_config[param_name]})")
                print(f"  Best R²: {best_r2_param['value']} (R²={best_r2_param['r2']:.4f})")
                print(f"  Best F1: {best_f1_param['value']} (F1={best_f1_param['spike_f1']:.4f})")
                print(f"  Combined improvement: R²={best_for_param['r2'] - baseline_r2:+.4f}, F1={best_for_param['spike_f1'] - baseline_f1:+.4f}")
            else:
                print(f"  No valid results for {param_name} - keeping baseline value")
                best_params[param_name] = self.baseline_config[param_name]
                best_r2_params[param_name] = self.baseline_config[param_name]
                best_f1_params[param_name] = self.baseline_config[param_name]
        
        return best_params, best_r2_params, best_f1_params, results_summary, baseline_result
    
    def test_combined_configurations(self, best_params, best_r2_params, best_f1_params, baseline_result):
        """Test combinations of the best individual parameters."""
        print(f"\nStep 3: Testing Combined Configurations")
        print("="*50)
        
        # Create different combination strategies
        combinations_to_test = []
        
        # 1. All best combined parameters
        all_best = best_params.copy()
        combinations_to_test.append(("Best Combined", all_best))
        
        # 2. Best R² focused parameters
        all_best_r2 = best_r2_params.copy()
        combinations_to_test.append(("Best R² Focus", all_best_r2))
        
        # 3. Best F1 focused parameters  
        all_best_f1 = best_f1_params.copy()
        combinations_to_test.append(("Best F1 Focus", all_best_f1))
        
        # 4. Best XGBoost params only (no spike weighting changes)
        xgb_best = self.baseline_config.copy()
        for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
            xgb_best[param] = best_params[param]
        combinations_to_test.append(("Best XGBoost Only", xgb_best))
        
        # 5. Best spike params only
        spike_best = self.baseline_config.copy()
        spike_best['spike_weight'] = best_params['spike_weight']
        spike_best['spike_threshold'] = best_params['spike_threshold']
        combinations_to_test.append(("Best Spike Only", spike_best))
        
        # 6. User's specific configuration from model_factory.py
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
                    'overall_score': (r2_imp * 2 + f1_imp * 2 + mae_imp * 1) / 5  # Weight R²+F1 more than MAE
                }
            return None
        
        # Run combination tests sequentially to avoid import issues  
        print(f"  Running {len(combinations_to_test)} combination tests sequentially...")
        combination_results = []
        for i, (combo_name, config) in enumerate(combinations_to_test):
            print(f"    Testing {combo_name} ({i+1}/{len(combinations_to_test)})...")
            result = test_single_combination(combo_name, config)
            if result:
                combination_results.append(result)
                print(f"    → R²={result['r2']:.4f}({result['r2_improvement_pct']:+.1f}%), F1={result['spike_f1']:.4f}({result['f1_improvement_pct']:+.1f}%)")
            else:
                print(f"    → Failed")
        
        # Results already filtered during sequential processing
        
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
        
        # Weight R² and F1 more heavily than MAE for best combined
        return (r2_imp * 2 + f1_imp * 2 + mae_imp * 1) / 5
    
    def _evaluate_xgboost_config(self, params):
        """Evaluate XGBoost configuration by modifying model_factory.py directly."""
        import shutil
        import re
        import sys
        
        try:
            original_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory.py"
            backup_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory_backup.py"
            
            # Create backup
            shutil.copy2(original_file, backup_file)
            
            # Read the original file
            with open(original_file, 'r') as f:
                content = f.read()
            
            # Replace the hardcoded values with our test parameters
            modified_content = content
            for param_name, value in params.items():
                if param_name in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                    # Find the line with this parameter and replace it
                    pattern = f"{param_name}=[\\d.]+,?"
                    replacement = f"{param_name}={value},"
                    modified_content = re.sub(pattern, replacement, modified_content)
            
            # Write the modified file
            with open(original_file, 'w') as f:
                f.write(modified_content)
            
            # Clear the module cache completely to force reload
            modules_to_remove = [k for k in sys.modules.keys() if 'forecasting' in k or 'model_factory' in k]
            for module in modules_to_remove:
                if module in sys.modules:
                    del sys.modules[module]
            
            # Also clear any __pycache__ files
            import os
            pycache_path = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/__pycache__"
            if os.path.exists(pycache_path):
                import glob
                for cache_file in glob.glob(os.path.join(pycache_path, "model_factory*.pyc")):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            
            # Run evaluation using subprocess to avoid module caching issues
            result = self._run_subprocess_evaluation()
            
            return result
            
        except Exception as e:
            print(f"Error evaluating config: {e}")
            return None
        finally:
            # Always restore the original file
            try:
                if backup_file and original_file:
                    shutil.copy2(backup_file, original_file)
                    # Clean up backup
                    import os
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                        
                    # Clear module cache again to restore original
                    modules_to_remove = [k for k in sys.modules.keys() if 'forecasting' in k or 'model_factory' in k]
                    for module in modules_to_remove:
                        if module in sys.modules:
                            del sys.modules[module]
            except Exception as e:
                print(f"Warning: Could not restore original model_factory.py: {e}")
    
    def _run_subprocess_evaluation(self):
        """Run evaluation in a subprocess to avoid module caching issues."""
        import subprocess
        import json
        
        try:
            # Run the evaluation in a separate Python process
            cmd = [
                sys.executable,
                "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/run_single_evaluation.py",
                "--n_anchors", str(self.n_anchors_per_site)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid")
            
            if result.returncode == 0:
                try:
                    # Parse the JSON output
                    output_data = json.loads(result.stdout.strip().split('\n')[-1])  # Get last line as JSON
                    
                    if output_data.get('success'):
                        return {
                            'mean_r2': output_data['mean_r2'],
                            'mean_mae': output_data['mean_mae'],
                            'mean_spike_f1': output_data['mean_spike_f1'],
                            'n_predictions': output_data['n_predictions']
                        }
                    else:
                        print(f"  Subprocess error: {output_data.get('error')}")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"  JSON decode error: {e}")
                    print(f"  Subprocess stdout: {result.stdout}")
                    print(f"  Subprocess stderr: {result.stderr}")
                    return None
            else:
                print(f"  Subprocess failed with code {result.returncode}")
                print(f"  Stderr: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"  Subprocess execution error: {e}")
            return None
    
    def _run_forecast_engine_evaluation(self):
        """Run evaluation using the same method as the main pipeline."""
        try:
            from forecasting.forecast_engine import ForecastEngine
            engine = ForecastEngine()
            
            # Run retrospective evaluation
            results_df = engine.run_retrospective_evaluation(
                task="regression",
                model_type="xgboost", 
                n_anchors=self.n_anchors_per_site,
                min_test_date="2008-01-01"
            )
            
            if results_df is not None and not results_df.empty:
                from sklearn.metrics import r2_score, mean_absolute_error, f1_score
                
                valid_results = results_df.dropna(subset=['actual_da', 'predicted_da'])
                if not valid_results.empty:
                    r2 = r2_score(valid_results['actual_da'], valid_results['predicted_da'])
                    mae = mean_absolute_error(valid_results['actual_da'], valid_results['predicted_da'])
                    
                    # Spike F1 (same as main pipeline)
                    y_true_binary = (valid_results['actual_da'] > 20.0).astype(int)
                    y_pred_binary = (valid_results['predicted_da'] > 20.0).astype(int)
                    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    
                    return {
                        'mean_r2': r2,
                        'mean_mae': mae,
                        'mean_spike_f1': f1,
                        'n_predictions': len(valid_results)
                    }
            
            return None
        except Exception as e:
            print(f"Error in forecast engine evaluation: {e}")
            return None
    
    def run_smart_tuning(self):
        """Run the complete smart tuning process."""
        start_time = time.time()
        
        # Step 1 & 2: Test individual parameters
        best_params, best_r2_params, best_f1_params, param_results, baseline = self.test_individual_parameters()
        
        # Step 3: Test combined configurations  
        combo_results = self.test_combined_configurations(best_params, best_r2_params, best_f1_params, baseline)
        
        # Step 4: Show final results
        self._show_final_results(best_params, param_results, combo_results, baseline, start_time)
        
        # Find the top 3 configurations
        winners = {}
        if combo_results:
            # Best combined (overall score with weighted metrics)
            winners['best_combined'] = max(combo_results, key=lambda x: x['overall_score'])
            
            # Best R² focused
            winners['best_r2'] = max(combo_results, key=lambda x: x['r2'])
            
            # Best F1 focused  
            winners['best_f1'] = max(combo_results, key=lambda x: x['spike_f1'])
        
        return {
            'individual_results': param_results,
            'combination_results': combo_results,
            'winners': winners,
            'recommended_config': winners.get('best_combined')  # Keep backward compatibility
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
            new_val = results['best_combined']
            best_r2_val = results['best_r2'] 
            best_f1_val = results['best_f1']
            r2_imp = results['r2_improvement']
            print(f"{param:15s}: Combined={new_val} | R²={best_r2_val} | F1={best_f1_val} (was {old_val})")
        
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
            
            # Top 3 Winners
            print(f"🏆 TOP 3 WINNERS:")
            print("-" * 40)
            
            # Best combined
            best_combined = max(combo_results, key=lambda x: x['overall_score'])
            print(f"🎯 BEST COMBINED (Weighted Score):")
            print(f"   {best_combined['name']} - Overall Score: {best_combined['overall_score']:+.2f}")
            print(f"   R²={best_combined['r2']:.4f}({best_combined['r2_improvement_pct']:+.1f}%) | MAE={best_combined['mae']:.4f}({best_combined['mae_improvement_pct']:+.1f}%) | F1={best_combined['spike_f1']:.4f}({best_combined['f1_improvement_pct']:+.1f}%)")
            print()
            
            # Best R²
            best_r2 = max(combo_results, key=lambda x: x['r2'])
            print(f"📈 BEST R² PERFORMANCE:")
            print(f"   {best_r2['name']} - R²={best_r2['r2']:.4f}({best_r2['r2_improvement_pct']:+.1f}%)")
            print(f"   MAE={best_r2['mae']:.4f}({best_r2['mae_improvement_pct']:+.1f}%) | F1={best_r2['spike_f1']:.4f}({best_r2['f1_improvement_pct']:+.1f}%)")
            print()
            
            # Best F1
            best_f1 = max(combo_results, key=lambda x: x['spike_f1'])
            print(f"🚨 BEST SPIKE DETECTION (F1):")
            print(f"   {best_f1['name']} - F1={best_f1['spike_f1']:.4f}({best_f1['f1_improvement_pct']:+.1f}%)")
            print(f"   R²={best_f1['r2']:.4f}({best_f1['r2_improvement_pct']:+.1f}%) | MAE={best_f1['mae']:.4f}({best_f1['mae_improvement_pct']:+.1f}%)")
            print()
            
            print(f"🎯 RECOMMENDED CONFIG (Best Combined):")
            print("-" * 35)
            for param, value in best_combined['config'].items():
                print(f"  {param} = {value}")


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