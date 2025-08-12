#!/usr/bin/env python3
"""
DATect Pre-computation Cache Generator
=====================================

Pre-computes all expensive operations for Google Cloud deployment:
1. Retrospective forecasts (all task/model combinations)
2. Spectral analysis (all sites + aggregate)
3. Other compute-heavy visualizations

Run this locally before deployment to avoid expensive compute on the server.
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

# Ensure imports work
import sys
sys.path.append('.')

import config
from forecasting.core.forecast_engine import ForecastEngine
from backend.visualizations import generate_spectral_analysis

class DATectCacheGenerator:
    """Generates pre-computed cache for all expensive operations."""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "retrospective").mkdir(exist_ok=True)
        (self.cache_dir / "spectral").mkdir(exist_ok=True)
        (self.cache_dir / "visualizations").mkdir(exist_ok=True)
        
        # Cache directory initialized
        
    def print_status(self, message, color='blue'):
        """Print colored status message"""
        colors = {
            'blue': '\033[94m',
            'green': '\033[92m', 
            'yellow': '\033[93m',
            'red': '\033[91m',
            'end': '\033[0m'
        }
        print(f"{colors.get(color, '')}{message}{colors['end']}")
        
    def precompute_retrospective_forecasts(self):
        """Pre-compute all retrospective forecast combinations."""
        
        self.print_status("🔮 Pre-computing retrospective forecasts...", 'blue')
        
        # All combinations to cache
        combinations = [
            ("classification", "xgboost"),
            ("classification", "logistic"),
            ("regression", "xgboost"), 
            ("regression", "linear")
        ]
        
        engine = ForecastEngine()
        
        for task, model_type in combinations:
            self.print_status(f"  Computing {task} + {model_type}...", 'yellow')
            
            try:
                # Use configured N_RANDOM_ANCHORS
                n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 200)
                
                results = engine.run_retrospective_evaluation(
                    task=task,
                    model_type=model_type,
                    n_anchors=n_anchors,
                    min_test_date="2008-01-01"
                )
                
                if results is not None and not results.empty:
                    # Save as both parquet (efficient) and JSON (API-ready)
                    cache_file = self.cache_dir / "retrospective" / f"{task}_{model_type}"
                    
                    # Parquet for efficient storage
                    results.to_parquet(f"{cache_file}.parquet", index=False)
                    
                    # JSON for API responses
                    results_json = results.to_dict('records')
                    with open(f"{cache_file}.json", 'w') as f:
                        json.dump(results_json, f, default=str, indent=2)
                    
                    self.print_status(f"    ✅ Saved {len(results)} predictions", 'green')
                    
                    # Save summary stats
                    summary = {
                        'total_predictions': len(results),
                        'sites': results['site'].nunique(),
                        'date_range': {
                            'min': str(results['date'].min()),
                            'max': str(results['date'].max())
                        },
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    with open(f"{cache_file}_summary.json", 'w') as f:
                        json.dump(summary, f, indent=2)
                        
                else:
                    self.print_status(f"    ❌ No results generated", 'red')
                    
            except Exception as e:
                self.print_status(f"    ❌ Error: {str(e)}", 'red')
                
    def precompute_spectral_analysis(self):
        """Pre-compute spectral analysis for all sites."""
        
        self.print_status("📊 Pre-computing spectral analysis...", 'blue')
        
        # Load data
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        # Enable XGBoost for spectral analysis to include comparisons in cache
        # This will use the already-cached retrospective results
        os.environ['SPECTRAL_ENABLE_XGB'] = '1'
        
        # Get all sites plus aggregate
        sites = list(data['site'].unique()) + [None]  # None = aggregate
        
        for site in sites:
            site_name = site or "all_sites"
            self.print_status(f"  Computing spectral analysis for {site_name}...", 'yellow')
            
            try:
                spectral_plots = generate_spectral_analysis(data, site=site)
                
                if spectral_plots:
                    # Save spectral analysis
                    cache_file = self.cache_dir / "spectral" / f"{site_name}.json"
                    
                    with open(cache_file, 'w') as f:
                        json.dump(spectral_plots, f, indent=2)
                        
                    self.print_status(f"    ✅ Saved {len(spectral_plots)} spectral plots", 'green')
                else:
                    self.print_status(f"    ⚠️  No spectral data generated", 'yellow')
                    
            except Exception as e:
                self.print_status(f"    ❌ Error: {str(e)}", 'red')
                
        # Restore environment
        if 'SPECTRAL_ENABLE_XGB' in os.environ:
            del os.environ['SPECTRAL_ENABLE_XGB']
            
    def precompute_visualization_data(self):
        """Pre-compute other visualization data that might be expensive."""
        
        self.print_status("📈 Pre-computing visualization data...", 'blue')
        
        # Load data
        data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
        
        # Pre-compute correlation matrices for all sites
        sites = data['site'].unique()
        
        for site in sites:
            site_data = data[data['site'] == site].copy()
            
            # Compute correlation matrix for numerical columns
            numeric_cols = site_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'lat', 'lon']]  # Exclude non-meaningful cols
            
            if len(numeric_cols) > 1:
                corr_matrix = site_data[numeric_cols].corr()
                
                # Save correlation data
                cache_file = self.cache_dir / "visualizations" / f"{site}_correlation.json"
                
                corr_data = {
                    'matrix': corr_matrix.to_dict(),
                    'columns': corr_matrix.columns.tolist(),
                    'site': site,
                    'generated_at': datetime.now().isoformat()
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(corr_data, f, default=str, indent=2)
                    
        self.print_status("    ✅ Correlation matrices cached", 'green')
        
    def generate_cache_manifest(self):
        """Generate manifest of all cached files."""
        
        self.print_status("📋 Generating cache manifest...", 'blue')
        
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'cache_version': '1.0',
            'files': {}
        }
        
        # Scan all cache files
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file() and cache_file.suffix in ['.json', '.parquet']:
                relative_path = cache_file.relative_to(self.cache_dir)
                
                manifest['files'][str(relative_path)] = {
                    'size_bytes': cache_file.stat().st_size,
                    'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
                }
                
        # Save manifest
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        self.print_status(f"    ✅ Manifest generated: {len(manifest['files'])} files", 'green')
        
    def run_full_precomputation(self):
        """Run all pre-computation tasks."""
        
        self.print_status("🚀 Starting DATect cache pre-computation", 'blue')
        self.print_status("=" * 50, 'blue')
        
        start_time = datetime.now()
        
        # 1. Retrospective forecasts (most expensive)
        self.precompute_retrospective_forecasts()
        
        # 2. Spectral analysis (also expensive)
        self.precompute_spectral_analysis()
        
        # 3. Other visualizations
        self.precompute_visualization_data()
        
        # 4. Generate manifest
        self.generate_cache_manifest()
        
        elapsed = datetime.now() - start_time
        
        self.print_status("=" * 50, 'green')
        self.print_status(f"✅ Pre-computation complete! ({elapsed})", 'green')
        self.print_status(f"📁 Cache saved to: {self.cache_dir.absolute()}", 'green')
        
        # Print cache size
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        self.print_status(f"💾 Total cache size: {total_size / (1024*1024):.1f} MB", 'green')


if __name__ == "__main__":
    generator = DATectCacheGenerator()
    generator.run_full_precomputation()