#!/usr/bin/env python3
"""
Test script to verify that parameter patching is working correctly.
Tests two very different max_depth values to see if they produce different results.
"""

import shutil
import re
import sys
import os

def test_parameter_patching():
    """Test if modifying model_factory.py actually changes model parameters."""
    
    original_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory.py"
    backup_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory_backup.py"
    
    # Test configs: Very different max_depth values
    test_configs = [
        {'max_depth': 1},   # Very shallow
        {'max_depth': 20}   # Very deep
    ]
    
    results = []
    
    try:
        # Create backup
        shutil.copy2(original_file, backup_file)
        print("Created backup of model_factory.py")
        
        for i, params in enumerate(test_configs):
            print(f"\n=== Testing config {i+1}: {params} ===")
            
            # Read the original file
            with open(backup_file, 'r') as f:
                content = f.read()
            
            # Replace the hardcoded values with our test parameters
            modified_content = content
            for param_name, value in params.items():
                if param_name in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                    # Find the line with this parameter and replace it
                    pattern = f"{param_name}=[\\d.]+,?"
                    replacement = f"{param_name}={value},"
                    modified_content = re.sub(pattern, replacement, modified_content)
                    print(f"  Replaced {param_name} with {value}")
            
            # Write the modified file
            with open(original_file, 'w') as f:
                f.write(modified_content)
            print("  Modified model_factory.py")
            
            # Clear the module cache completely to force reload
            modules_to_remove = [k for k in sys.modules.keys() if 'forecasting' in k or 'model_factory' in k]
            for module in modules_to_remove:
                if module in sys.modules:
                    del sys.modules[module]
            
            # Also clear any __pycache__ files
            pycache_path = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/__pycache__"
            if os.path.exists(pycache_path):
                import glob
                for cache_file in glob.glob(os.path.join(pycache_path, "model_factory*.pyc")):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            print("  Cleared module cache and pycache")
            
            # Run a simple evaluation with just 10 anchors to test quickly
            try:
                from smart_hyperparameter_tuning import SmartHyperparameterTuner
                tuner = SmartHyperparameterTuner(n_anchors_per_site=10)  # Very small for fast testing
                result = tuner._run_forecast_engine_evaluation()
                
                if result:
                    print(f"  Result: R²={result['mean_r2']:.6f}, F1={result['mean_spike_f1']:.6f}")
                    results.append({
                        'config': params,
                        'r2': result['mean_r2'],
                        'f1': result['mean_spike_f1']
                    })
                else:
                    print("  No result returned")
                    
            except Exception as e:
                print(f"  Error running evaluation: {e}")
                
    finally:
        # Always restore the original file
        try:
            shutil.copy2(backup_file, original_file)
            print("\nRestored original model_factory.py")
            
            # Clean up backup
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print("Cleaned up backup file")
                
            # Clear module cache again to restore original
            modules_to_remove = [k for k in sys.modules.keys() if k.startswith('forecasting')]
            for module in modules_to_remove:
                if module in sys.modules:
                    del sys.modules[module]
            print("Cleared module cache (restored)")
            
        except Exception as e:
            print(f"Warning: Could not restore original model_factory.py: {e}")
    
    # Analyze results
    print("\n" + "="*50)
    print("PARAMETER PATCHING TEST RESULTS")
    print("="*50)
    
    if len(results) >= 2:
        config1, config2 = results[0], results[1]
        
        print(f"Config 1 ({config1['config']}): R²={config1['r2']:.6f}, F1={config1['f1']:.6f}")
        print(f"Config 2 ({config2['config']}): R²={config2['r2']:.6f}, F1={config2['f1']:.6f}")
        
        r2_diff = abs(config1['r2'] - config2['r2'])
        f1_diff = abs(config1['f1'] - config2['f1'])
        
        print(f"\nDifferences: R²={r2_diff:.6f}, F1={f1_diff:.6f}")
        
        if r2_diff > 0.001 or f1_diff > 0.001:
            print("✅ SUCCESS: Parameters are actually changing! Different configs produce different results.")
        else:
            print("❌ FAILURE: Parameters are NOT changing. Identical results suggest patching isn't working.")
    else:
        print("❌ Not enough results to compare")
        
if __name__ == "__main__":
    test_parameter_patching()