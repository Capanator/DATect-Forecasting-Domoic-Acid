#!/usr/bin/env python3
"""
Subprocess script to run a single evaluation with modified model_factory.py.
This avoids module caching issues by running in a fresh Python process.
"""

import sys
import argparse
import json

def run_evaluation(n_anchors_per_site):
    """Run a single evaluation and return results."""
    try:
        from smart_hyperparameter_tuning import SmartHyperparameterTuner
        
        tuner = SmartHyperparameterTuner(n_anchors_per_site=n_anchors_per_site)
        result = tuner._run_forecast_engine_evaluation()
        
        if result:
            return {
                'success': True,
                'mean_r2': result['mean_r2'],
                'mean_mae': result['mean_mae'],
                'mean_spike_f1': result['mean_spike_f1'],
                'n_predictions': result['n_predictions']
            }
        else:
            return {'success': False, 'error': 'No result returned'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_anchors', type=int, required=True)
    args = parser.parse_args()
    
    result = run_evaluation(args.n_anchors)
    print(json.dumps(result))