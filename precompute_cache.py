#!/usr/bin/env python3
"""
DATect Pre-computation Cache Generator

Pre-computes expensive operations for deployment to ensure EXACT match with fresh runs:
- Retrospective forecasts
- Spectral analysis 
- Visualization data
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import random
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import config
from forecasting.forecast_engine import ForecastEngine
from backend.visualizations import generate_spectral_analysis

class DATectCacheGenerator:
    """Generates pre-computed cache for all expensive operations."""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "retrospective").mkdir(exist_ok=True)
        (self.cache_dir / "spectral").mkdir(exist_ok=True)
        (self.cache_dir / "visualizations").mkdir(exist_ok=True)
        
    def precompute_retrospective_forecasts(self):
        """Pre-compute all retrospective forecast combinations - EXACT match with API."""
        print("Pre-computing retrospective forecasts...")
        
        # These combinations match exactly what the API expects
        # The API maps "linear" to either "linear" (regression) or "logistic" (classification)
        combinations = [
            ("regression", "xgboost"),
            ("regression", "linear"),  
            ("classification", "xgboost"),
            ("classification", "logistic")
        ]
        
        for task, model_type in combinations:
            print(f"  {task} + {model_type}...")
            
            try:
                # CRITICAL: Reset random seeds for each model to ensure reproducibility
                # This matches what happens in the ForecastEngine.__init__
                random.seed(config.RANDOM_SEED)
                np.random.seed(config.RANDOM_SEED)
                
                # Create fresh engine instance for each model to avoid state contamination
                # Pass validate_on_init=False to match API behavior
                engine = ForecastEngine(validate_on_init=False)
                engine.data_file = config.FINAL_OUTPUT_PATH
                
                # Use exact same parameters as API would use
                n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 500)
                
                # Run evaluation exactly as the API does
                results_df = engine.run_retrospective_evaluation(
                    task=task,
                    model_type=model_type,
                    n_anchors=n_anchors,
                    min_test_date="2008-01-01"
                )
                
                if results_df is not None and not results_df.empty:
                    cache_file = self.cache_dir / "retrospective" / f"{task}_{model_type}"
                    
                    # Save parquet exactly as engine produced
                    results_df.to_parquet(f"{cache_file}.parquet", index=False)
                    
                    # Convert to JSON format expected by cache_manager
                    # This matches the exact format the API would produce
                    results_json = []
                    for _, row in results_df.iterrows():
                        record = {}
                        # Copy all fields from the DataFrame
                        for col in results_df.columns:
                            value = row[col]
                            # Handle datetime columns
                            if pd.api.types.is_datetime64_any_dtype(results_df[col]):
                                record[col] = value.strftime('%Y-%m-%d') if pd.notnull(value) else None
                            # Handle numeric columns with NaN/inf
                            elif pd.api.types.is_numeric_dtype(results_df[col]):
                                if pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
                                    record[col] = None
                                else:
                                    record[col] = float(value) if pd.api.types.is_float_dtype(results_df[col]) else int(value)
                            else:
                                record[col] = value
                        results_json.append(record)
                    
                    # Save JSON in the exact format expected by the cache_manager
                    with open(f"{cache_file}.json", 'w') as f:
                        json.dump(results_json, f, default=str, indent=2)
                    
                    # Calculate and display metrics to verify
                    if task == "regression":
                        from sklearn.metrics import r2_score, mean_absolute_error, f1_score
                        valid_mask = results_df['da'].notna() & results_df['Predicted_da'].notna()
                        valid_df = results_df[valid_mask]
                        
                        if len(valid_df) > 0:
                            r2 = r2_score(valid_df['da'], valid_df['Predicted_da'])
                            mae = mean_absolute_error(valid_df['da'], valid_df['Predicted_da'])
                            
                            spike_threshold = 15.0
                            actual_binary = [1 if val > spike_threshold else 0 for val in valid_df['da']]
                            pred_binary = [1 if val > spike_threshold else 0 for val in valid_df['Predicted_da']]
                            f1 = f1_score(actual_binary, pred_binary, zero_division=0)
                            
                            print(f"    Saved {len(results_df)} predictions")
                            print(f"    Metrics: R²={r2:.4f}, MAE={mae:.2f}, F1={f1:.4f}")
                    else:
                        print(f"    Saved {len(results_df)} predictions")
                        
                else:
                    print("    No results generated")
                    
            except Exception as e:
                print(f"    Error: {str(e)}")
                import traceback
                traceback.print_exc()
                
    def precompute_spectral_analysis(self):
        """Pre-compute spectral analysis for all sites."""
        print("Pre-computing spectral analysis...")
        
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        os.environ['SPECTRAL_ENABLE_XGB'] = '1'
        
        sites = list(data['site'].unique()) + [None]
        
        for site in sites:
            site_name = site or "all_sites"
            print(f"  {site_name}...")
            
            try:
                spectral_plots = generate_spectral_analysis(data, site=site)
                
                if spectral_plots:
                    cache_file = self.cache_dir / "spectral" / f"{site_name}.json"
                    with open(cache_file, 'w') as f:
                        json.dump(spectral_plots, f, indent=2)
                    print(f"    Saved {len(spectral_plots)} plots")
                else:
                    print("    No spectral data")
                    
            except Exception as e:
                print(f"    Error: {str(e)}")
                
        os.environ.pop('SPECTRAL_ENABLE_XGB', None)
            
    def precompute_visualization_data(self):
        """Pre-compute other visualization data."""
        print("Pre-computing visualization data...")
        
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site]
            numeric_cols = site_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'lat', 'lon']]
            
            if len(numeric_cols) > 1:
                corr_matrix = site_data[numeric_cols].corr()
                cache_file = self.cache_dir / "visualizations" / f"{site}_correlation.json"
                
                corr_data = {
                    'matrix': corr_matrix.to_dict(),
                    'columns': corr_matrix.columns.tolist(),
                    'site': site
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(corr_data, f, default=str, indent=2)
        
        print("  Correlation matrices cached")
        
    def generate_cache_manifest(self):
        """Generate manifest of all cached files."""
        print("Generating manifest...")
        
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'cache_version': '2.0',  # Updated version for new cache format
            'config': {
                'N_RANDOM_ANCHORS': getattr(config, 'N_RANDOM_ANCHORS', 500),
                'RANDOM_SEED': config.RANDOM_SEED,
                'FORECAST_HORIZON_DAYS': config.FORECAST_HORIZON_DAYS,
                'MIN_TRAINING_SAMPLES': getattr(config, 'MIN_TRAINING_SAMPLES', 5)
            },
            'files': {}
        }
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file() and cache_file.suffix in ['.json', '.parquet']:
                relative_path = cache_file.relative_to(self.cache_dir)
                manifest['files'][str(relative_path)] = {
                    'size_bytes': cache_file.stat().st_size
                }
                
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"  {len(manifest['files'])} files cached")
        
    def run_full_precomputation(self):
        """Run all pre-computation tasks."""
        print("Starting DATect cache pre-computation")
        print("=====================================")
        print(f"Configuration:")
        print(f"  N_RANDOM_ANCHORS: {getattr(config, 'N_RANDOM_ANCHORS', 500)}")
        print(f"  RANDOM_SEED: {config.RANDOM_SEED}")
        print(f"  FORECAST_HORIZON_DAYS: {config.FORECAST_HORIZON_DAYS}")
        print("=====================================")
        
        start_time = datetime.now()
        
        self.precompute_retrospective_forecasts()
        self.precompute_spectral_analysis()
        self.precompute_visualization_data()
        self.generate_cache_manifest()
        
        elapsed = datetime.now() - start_time
        
        print("=====================================")
        print(f"Pre-computation complete! ({elapsed})")
        print(f"Cache saved to: {self.cache_dir.absolute()}")
        
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        print(f"Total cache size: {total_size / (1024*1024):.1f} MB")
        
        print("\nIMPORTANT: Cache has been regenerated to match fresh run behavior.")
        print("All cached results should now match what the API produces during fresh runs.")


if __name__ == "__main__":
    generator = DATectCacheGenerator()
    generator.run_full_precomputation()