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
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import config
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
        """Pre-compute all retrospective forecast combinations."""
        print("Pre-computing retrospective forecasts...")
        
        combinations = [
            ("regression", "xgboost"),
            ("regression", "linear"),  
            ("classification", "xgboost"),
            ("classification", "logistic")
        ]
        
        for task, model_type in combinations:
            print(f"  {task} + {model_type}...")
            
            try:
                from backend.api import get_forecast_engine, clean_float_for_json, _compute_summary
                
                engine = get_forecast_engine()
                n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 500)
                
                results_df = engine.run_retrospective_evaluation(
                    task=task,
                    model_type=model_type,
                    n_anchors=n_anchors,
                    min_test_date="2008-01-01"
                )
                
                if results_df is not None and not results_df.empty:
                    base_results = []
                    for _, row in results_df.iterrows():
                        record = {
                            "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
                            "site": row['site'],
                            "da": clean_float_for_json(row['da']) if pd.notnull(row['da']) else None,
                            "Predicted_da": clean_float_for_json(row['Predicted_da']) if 'Predicted_da' in row and pd.notnull(row['Predicted_da']) else None,
                            "da-category": clean_float_for_json(row['da-category']) if 'da-category' in row and pd.notnull(row['da-category']) else None,
                            "Predicted_da-category": clean_float_for_json(row['Predicted_da-category']) if 'Predicted_da-category' in row and pd.notnull(row['Predicted_da-category']) else None
                        }
                        base_results.append(record)
                    

                    results_json = []
                    for result in base_results:
                        record = {
                            'date': result['date'],
                            'site': result['site'],
                            'da': result['da'],
                            'da-category': result['da-category'],
                            'Predicted_da': result['Predicted_da'],
                            'Predicted_da-category': result['Predicted_da-category']
                        }
                        if 'anchor_date' in results_df.columns:
                            anchor_row = results_df[results_df['site'] == result['site']].iloc[0]
                            if pd.notnull(anchor_row['anchor_date']):
                                record['anchor_date'] = anchor_row['anchor_date'].strftime('%Y-%m-%d')
                        results_json.append(record)
                        
                    cache_file = self.cache_dir / "retrospective" / f"{task}_{model_type}"
                    
                    results_df.to_parquet(f"{cache_file}.parquet", index=False)
                    
                    with open(f"{cache_file}.json", 'w') as f:
                        json.dump(results_json, f, default=str, indent=2)
                    
                    try:
                        summary = _compute_summary(base_results)
                        if 'r2_score' in summary:
                            r2 = summary['r2_score']
                            mae = summary.get('mae', 0)
                            f1 = summary.get('f1_score', 0)
                            print(f"    Saved {len(results_df)} predictions")
                            print(f"    Metrics: R²={r2:.4f}, MAE={mae:.2f}, F1={f1:.4f}")
                        else:
                            print(f"    Saved {len(results_df)} predictions")
                    except Exception as e:
                        print(f"    Saved {len(results_df)} predictions (metrics error: {e})")
                        
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
        


if __name__ == "__main__":
    generator = DATectCacheGenerator()
    generator.run_full_precomputation()