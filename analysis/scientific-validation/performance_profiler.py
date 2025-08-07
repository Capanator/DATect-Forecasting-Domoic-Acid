#!/usr/bin/env python3
"""
Performance Profiler for DATect System
======================================

Comprehensive performance profiling and benchmarking tool for the 
DATect forecasting system. Provides detailed timing analysis,
memory usage monitoring, and computational requirements documentation.

Usage:
    python performance_profiler.py
    python performance_profiler.py --full-benchmark
"""

import time
import psutil
import os
import sys
from functools import wraps
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

# Import system components
from forecasting.core.forecast_engine import ForecastEngine
from forecasting.core.data_processor import DataProcessor
from forecasting.core.model_factory import ModelFactory


class PerformanceProfiler:
    """
    Performance profiler for DATect forecasting system.
    
    Provides detailed analysis of:
    - Execution time for each component
    - Memory usage patterns
    - CPU utilization
    - I/O operations
    - Scalability metrics
    """
    
    def __init__(self, output_dir: str = "./performance_analysis/"):
        """Initialize performance profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.peak_memory = 0
        
    def profile_function(self, func_name: str):
        """Decorator to profile function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Memory before execution
                mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
                
                # Time execution
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                end_time = time.perf_counter()
                
                # Memory after execution
                mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = mem_after - mem_before
                
                # Store results
                self.results[func_name] = {
                    'execution_time_seconds': end_time - start_time,
                    'memory_before_mb': mem_before,
                    'memory_after_mb': mem_after,
                    'memory_delta_mb': memory_delta,
                    'success': success,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update peak memory
                self.peak_memory = max(self.peak_memory, mem_after)
                
                if not success:
                    raise Exception(error)
                    
                return result
            return wrapper
        return decorator
    
    def start_system_monitoring(self):
        """Start system-wide performance monitoring."""
        self.start_time = time.perf_counter()
        self.system_start_stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
    
    def end_system_monitoring(self):
        """End system monitoring and calculate deltas."""
        if not self.start_time:
            return
            
        total_time = time.perf_counter() - self.start_time
        end_stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        self.system_performance = {
            'total_execution_time': total_time,
            'average_cpu_percent': (self.system_start_stats['cpu_percent'] + end_stats['cpu_percent']) / 2,
            'memory_percent_delta': end_stats['memory_percent'] - self.system_start_stats['memory_percent'],
            'peak_memory_mb': self.peak_memory,
            'start_stats': self.system_start_stats,
            'end_stats': end_stats
        }
    
    def benchmark_data_loading(self, data_path: str) -> Dict[str, Any]:
        """Benchmark data loading performance."""
        print("🔄 Benchmarking data loading...")
        
        @self.profile_function('data_loading')
        def load_data():
            processor = DataProcessor()
            return processor.load_and_prepare_base_data(data_path)
        
        data = load_data()
        
        # Add data-specific metrics
        if data is not None:
            self.results['data_loading'].update({
                'rows_loaded': len(data),
                'columns_loaded': len(data.columns),
                'memory_per_row_kb': (self.results['data_loading']['memory_delta_mb'] * 1024) / len(data),
                'loading_rate_rows_per_second': len(data) / self.results['data_loading']['execution_time_seconds']
            })
        
        return data
    
    def benchmark_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Benchmark feature engineering performance."""
        print("🔄 Benchmarking feature engineering...")
        
        @self.profile_function('feature_engineering')
        def create_features():
            processor = DataProcessor()
            # Use first site and middle date as cutoff
            test_site = data['site'].iloc[0]
            site_data = data[data['site'] == test_site]
            middle_idx = len(site_data) // 2
            cutoff_date = site_data['date'].iloc[middle_idx]
            
            return processor.create_lag_features_safe(
                site_data, 'site', 'da', [1, 2, 3], cutoff_date
            )
        
        processed_data = create_features()
        
        # Add feature engineering metrics
        if processed_data is not None:
            original_cols = len(data.columns)
            new_cols = len(processed_data.columns)
            features_added = new_cols - original_cols
            
            self.results['feature_engineering'].update({
                'features_added': features_added,
                'processing_rate_rows_per_second': len(data) / self.results['feature_engineering']['execution_time_seconds'],
                'time_per_feature_ms': (self.results['feature_engineering']['execution_time_seconds'] * 1000) / max(features_added, 1)
            })
        
        return processed_data
    
    def benchmark_model_training(self, data: pd.DataFrame) -> Any:
        """Benchmark model training performance."""
        print("🔄 Benchmarking model training...")
        
        @self.profile_function('model_training')
        def train_model():
            # Prepare data for training
            processor = DataProcessor()
            model_factory = ModelFactory()
            
            # Use subset of data for training benchmark
            train_data = data.dropna(subset=['da']).head(1000)  # Limit for consistent benchmarking
            
            if len(train_data) < 10:
                return None
            
            # Create DA categories
            train_data = train_data.copy()
            train_data['da-category'] = processor.create_da_categories_safe(train_data['da'])
            
            # Prepare features
            drop_cols = ['date', 'site', 'da', 'da-category']
            transformer, X_train = processor.create_numeric_transformer(train_data, drop_cols)
            X_train_processed = transformer.fit_transform(X_train)
            
            # Train XGBoost model
            model = model_factory.get_model('regression', 'xgboost')
            model.fit(X_train_processed, train_data['da'])
            
            return model, X_train_processed, train_data
        
        model_result = train_model()
        
        # Add model training metrics
        if model_result:
            model, X_train_processed, train_data = model_result
            training_samples = len(train_data)
            
            self.results['model_training'].update({
                'training_samples': training_samples,
                'features_count': X_train_processed.shape[1],
                'training_time_per_sample_ms': (self.results['model_training']['execution_time_seconds'] * 1000) / training_samples,
                'samples_per_second': training_samples / self.results['model_training']['execution_time_seconds']
            })
        
        return model_result
    
    def benchmark_prediction(self, model_result: Any, data: pd.DataFrame):
        """Benchmark prediction performance."""
        if not model_result:
            return
            
        print("🔄 Benchmarking prediction...")
        
        @self.profile_function('prediction')
        def make_predictions():
            model, X_train_processed, train_data = model_result
            processor = DataProcessor()
            
            # Use a small subset for prediction benchmarking
            test_data = data.dropna(subset=['da']).tail(100).head(50)
            
            if len(test_data) < 5:
                return None
            
            # Prepare test features
            drop_cols = ['date', 'site', 'da']
            transformer, X_test = processor.create_numeric_transformer(test_data, drop_cols)
            
            # Ensure same columns as training
            X_test = X_test.reindex(columns=X_train_processed.columns, fill_value=0)
            X_test_processed = transformer.fit_transform(X_test)
            
            # Make predictions
            predictions = model.predict(X_test_processed)
            return predictions
        
        predictions = make_predictions()
        
        # Add prediction metrics
        if predictions is not None:
            prediction_count = len(predictions)
            
            self.results['prediction'].update({
                'predictions_made': prediction_count,
                'prediction_time_per_sample_ms': (self.results['prediction']['execution_time_seconds'] * 1000) / prediction_count,
                'predictions_per_second': prediction_count / self.results['prediction']['execution_time_seconds']
            })
    
    def run_full_benchmark(self, data_path: str):
        """Run comprehensive performance benchmark."""
        print("🚀 Starting comprehensive DATect performance benchmark")
        print("=" * 60)
        
        self.start_system_monitoring()
        
        try:
            # Benchmark each component
            data = self.benchmark_data_loading(data_path)
            if data is not None:
                processed_data = self.benchmark_feature_engineering(data)
                model_result = self.benchmark_model_training(processed_data or data)
                self.benchmark_prediction(model_result, data)
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            
        finally:
            self.end_system_monitoring()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform
            },
            'component_performance': self.results,
            'system_performance': getattr(self, 'system_performance', {}),
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            'total_components_tested': len(self.results),
            'successful_components': sum(1 for r in self.results.values() if r.get('success', False)),
            'total_execution_time': sum(r.get('execution_time_seconds', 0) for r in self.results.values()),
            'peak_memory_usage_mb': self.peak_memory,
            'performance_classification': 'Unknown'
        }
        
        # Classify overall performance
        total_time = summary['total_execution_time']
        if total_time < 10:
            summary['performance_classification'] = 'Excellent (<10s)'
        elif total_time < 30:
            summary['performance_classification'] = 'Good (10-30s)'
        elif total_time < 60:
            summary['performance_classification'] = 'Acceptable (30-60s)'
        else:
            summary['performance_classification'] = 'Slow (>60s)'
        
        return summary
    
    def save_report(self, report: Dict[str, Any]):
        """Save performance report to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f'performance_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable report
        text_file = self.output_dir / f'performance_summary_{timestamp}.txt'
        with open(text_file, 'w') as f:
            self._write_text_report(f, report)
        
        print(f"\n📊 Performance report saved:")
        print(f"  📄 JSON: {json_file}")
        print(f"  📄 Summary: {text_file}")
        
    def _write_text_report(self, f, report):
        """Write human-readable performance report."""
        f.write("DATect Performance Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # System info
        f.write("System Information:\n")
        f.write("-" * 20 + "\n")
        sys_info = report['system_info']
        f.write(f"Python Version: {sys_info['python_version'].split()[0]}\n")
        f.write(f"CPU Cores: {sys_info['cpu_count']}\n")
        f.write(f"Total Memory: {sys_info['total_memory_gb']:.1f} GB\n")
        f.write(f"Platform: {sys_info['platform']}\n\n")
        
        # Performance summary
        f.write("Performance Summary:\n")
        f.write("-" * 20 + "\n")
        summary = report['summary']
        f.write(f"Overall Classification: {summary['performance_classification']}\n")
        f.write(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds\n")
        f.write(f"Peak Memory Usage: {summary['peak_memory_usage_mb']:.1f} MB\n")
        f.write(f"Components Tested: {summary['total_components_tested']}\n")
        f.write(f"Successful Components: {summary['successful_components']}\n\n")
        
        # Component details
        f.write("Component Performance:\n")
        f.write("-" * 20 + "\n")
        for component, metrics in report['component_performance'].items():
            f.write(f"\n{component.replace('_', ' ').title()}:\n")
            f.write(f"  Execution Time: {metrics.get('execution_time_seconds', 0):.3f} seconds\n")
            f.write(f"  Memory Usage: {metrics.get('memory_delta_mb', 0):.1f} MB\n")
            f.write(f"  Status: {'✅ Success' if metrics.get('success') else '❌ Failed'}\n")
            
            # Additional metrics
            for key, value in metrics.items():
                if key not in ['execution_time_seconds', 'memory_delta_mb', 'success', 'error', 'timestamp', 'memory_before_mb', 'memory_after_mb']:
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")


def main():
    """Main entry point for performance profiling."""
    parser = argparse.ArgumentParser(description="DATect Performance Profiler")
    parser.add_argument('--data-path', default='final_output.parquet',
                      help='Path to data file for benchmarking')
    parser.add_argument('--output-dir', default='./performance_analysis/',
                      help='Output directory for reports')
    parser.add_argument('--full-benchmark', action='store_true',
                      help='Run comprehensive benchmark suite')
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = PerformanceProfiler(args.output_dir)
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Data file not found: {args.data_path}")
        print("ℹ️  Please run dataset-creation.py first to generate data")
        return
    
    try:
        if args.full_benchmark:
            # Run full benchmark suite
            profiler.run_full_benchmark(args.data_path)
        else:
            # Run basic benchmark
            print("🔄 Running basic performance check...")
            profiler.start_system_monitoring()
            data = profiler.benchmark_data_loading(args.data_path)
            profiler.end_system_monitoring()
        
        # Generate and save report
        report = profiler.generate_report()
        profiler.save_report(report)
        
        # Print summary
        summary = report['summary']
        print(f"\n🏁 Performance Analysis Complete!")
        print(f"Overall Performance: {summary['performance_classification']}")
        print(f"Total Time: {summary['total_execution_time']:.2f}s")
        print(f"Peak Memory: {summary['peak_memory_usage_mb']:.1f} MB")
        
    except Exception as e:
        print(f"❌ Performance profiling failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())