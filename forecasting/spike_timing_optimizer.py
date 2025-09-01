"""
Spike Timing Optimization Module
================================

Specialized module for optimizing domoic acid spike timing prediction.
Implements naive baselines and spike-focused evaluation metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from typing import Dict, List, Tuple, Optional, Any
import json

from .logging_config import get_logger
from .data_processor import DataProcessor

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class SpikeTimingOptimizer:
    """
    Optimizer focused on improving domoic acid spike timing prediction.
    
    Key features:
    - Naive baseline implementations (lag, persistence, seasonal)
    - Spike-focused evaluation metrics
    - Timing accuracy assessment
    - Performance comparison tools
    """
    
    def __init__(self, spike_threshold: float = 20.0, temporal_window: int = 7):
        """
        Initialize spike timing optimizer.
        
        Args:
            spike_threshold: DA concentration threshold for spike events (ppm)
            temporal_window: Window for timing accuracy evaluation (days)
        """
        self.spike_threshold = spike_threshold
        self.temporal_window = temporal_window
        self.data_processor = DataProcessor()
        
        logger.info(f"SpikeTimingOptimizer initialized with spike_threshold={spike_threshold}, window={temporal_window}")
        
    def create_naive_lag_baseline(self, data: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
        """
        Create naive baseline: actual DA values shifted forward by lag_days.
        This represents the baseline we want to beat for spike timing prediction.
        
        Args:
            data: Input DataFrame with columns ['date', 'site', 'da']
            lag_days: Number of days to shift actual values forward
            
        Returns:
            DataFrame with naive lag predictions
        """
        logger.info(f"Creating naive lag baseline with {lag_days}-day lag")
        
        result_rows = []
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Create lagged predictions: predict today's value = value from lag_days ago
            for i in range(len(site_data)):
                current_date = site_data.iloc[i]['date']
                actual_da = site_data.iloc[i]['da']
                
                # Find the lag date
                lag_date = current_date - pd.Timedelta(days=lag_days)
                
                # Find actual DA value at lag date (or nearest available)
                lag_candidates = site_data[site_data['date'] <= lag_date]
                
                if not lag_candidates.empty:
                    # Use the most recent value within lag period
                    lag_value = lag_candidates.iloc[-1]['da']
                    anchor_date = lag_candidates.iloc[-1]['date']
                else:
                    # If no historical data, use 0 as prediction
                    lag_value = 0.0
                    anchor_date = current_date - pd.Timedelta(days=lag_days)
                
                # Only add row if we have valid data (no NaNs)
                if not pd.isna(actual_da) and not pd.isna(lag_value):
                    result_rows.append({
                        'date': current_date,
                        'site': site,
                        'anchor_date': anchor_date,
                        'da': float(actual_da),
                        'Predicted_da': float(lag_value),
                        'baseline_type': f'naive_lag_{lag_days}d'
                    })
        
        result_df = pd.DataFrame(result_rows)
        logger.info(f"Generated {len(result_df)} naive lag baseline predictions")
        
        return result_df
        
    def create_persistence_baseline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create persistence baseline: tomorrow's DA = today's DA.
        
        Args:
            data: Input DataFrame with columns ['date', 'site', 'da']
            
        Returns:
            DataFrame with persistence predictions
        """
        logger.info("Creating persistence baseline")
        
        result_rows = []
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Create persistence predictions
            for i in range(1, len(site_data)):  # Start from index 1
                current_date = site_data.iloc[i]['date']
                actual_da = site_data.iloc[i]['da']
                prev_da = site_data.iloc[i-1]['da']  # Previous observation
                anchor_date = site_data.iloc[i-1]['date']
                
                # Only add row if we have valid data (no NaNs)
                if not pd.isna(actual_da) and not pd.isna(prev_da):
                    result_rows.append({
                        'date': current_date,
                        'site': site,
                        'anchor_date': anchor_date,
                        'da': float(actual_da),
                        'Predicted_da': float(prev_da),
                        'baseline_type': 'persistence'
                    })
        
        result_df = pd.DataFrame(result_rows)
        logger.info(f"Generated {len(result_df)} persistence baseline predictions")
        
        return result_df
        
    def create_seasonal_baseline(self, data: pd.DataFrame, seasonal_days: int = 365) -> pd.DataFrame:
        """
        Create seasonal baseline: this year's DA = same day last year's DA.
        
        Args:
            data: Input DataFrame with columns ['date', 'site', 'da']
            seasonal_days: Number of days for seasonal cycle
            
        Returns:
            DataFrame with seasonal predictions
        """
        logger.info(f"Creating seasonal baseline with {seasonal_days}-day cycle")
        
        result_rows = []
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Create seasonal predictions
            for i in range(len(site_data)):
                current_date = site_data.iloc[i]['date']
                actual_da = site_data.iloc[i]['da']
                
                # Find the seasonal reference date
                seasonal_date = current_date - pd.Timedelta(days=seasonal_days)
                
                # Find actual DA value at seasonal date (or nearest available)
                seasonal_candidates = site_data[
                    (site_data['date'] >= seasonal_date - pd.Timedelta(days=7)) &
                    (site_data['date'] <= seasonal_date + pd.Timedelta(days=7))
                ]
                
                if not seasonal_candidates.empty:
                    # Use the closest date to seasonal reference
                    closest_idx = np.argmin(np.abs(seasonal_candidates['date'] - seasonal_date))
                    seasonal_value = seasonal_candidates.iloc[closest_idx]['da']
                    anchor_date = seasonal_candidates.iloc[closest_idx]['date']
                else:
                    # If no seasonal data, use historical mean
                    historical_data = site_data[site_data['date'] < current_date]
                    seasonal_value = historical_data['da'].mean() if not historical_data.empty else 0.0
                    anchor_date = current_date - pd.Timedelta(days=seasonal_days)
                
                # Only add row if we have valid data (no NaNs)
                if not pd.isna(actual_da) and not pd.isna(seasonal_value):
                    result_rows.append({
                        'date': current_date,
                        'site': site,
                        'anchor_date': anchor_date,
                        'da': float(actual_da),
                        'Predicted_da': float(seasonal_value),
                        'baseline_type': f'seasonal_{seasonal_days}d'
                    })
        
        result_df = pd.DataFrame(result_rows)
        logger.info(f"Generated {len(result_df)} seasonal baseline predictions")
        
        return result_df
    
    def evaluate_spike_timing_performance(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate performance specifically for spike timing prediction.
        
        Args:
            predictions_df: DataFrame with actual and predicted DA values
            
        Returns:
            Dictionary with spike timing metrics
        """
        logger.info("Evaluating spike timing performance")
        
        # Clean data - remove rows with NaN values
        clean_df = predictions_df.dropna(subset=['da', 'Predicted_da']).copy()
        
        if clean_df.empty:
            logger.warning("No valid data after NaN removal")
            return {
                'n_actual_spikes': 0,
                'n_predicted_spikes': 0,
                'spike_metrics': None
            }
        
        logger.info(f"Cleaned data: {len(clean_df)} valid records (removed {len(predictions_df) - len(clean_df)} NaN records)")
        
        # Identify spike events
        actual_spikes = clean_df['da'] > self.spike_threshold
        predicted_spikes = clean_df['Predicted_da'] > self.spike_threshold
        
        # Basic spike detection metrics
        n_actual_spikes = actual_spikes.sum()
        n_predicted_spikes = predicted_spikes.sum()
        
        if n_actual_spikes == 0:
            logger.warning("No actual spikes found in data")
            return {
                'n_actual_spikes': 0,
                'n_predicted_spikes': n_predicted_spikes,
                'spike_metrics': None
            }
        
        # Calculate spike detection performance
        true_positives = (actual_spikes & predicted_spikes).sum()
        false_positives = (~actual_spikes & predicted_spikes).sum()
        false_negatives = (actual_spikes & ~predicted_spikes).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate timing accuracy for detected spikes
        timing_metrics = self._calculate_spike_timing_accuracy(clean_df)
        
        # Calculate magnitude accuracy for spikes
        spike_data = clean_df[actual_spikes].copy()
        magnitude_metrics = {}
        if not spike_data.empty:
            magnitude_metrics = {
                'spike_mae': mean_absolute_error(spike_data['da'], spike_data['Predicted_da']),
                'spike_rmse': np.sqrt(np.mean((spike_data['da'] - spike_data['Predicted_da'])**2)),
                'spike_bias': np.mean(spike_data['Predicted_da'] - spike_data['da']),
                'spike_r2': r2_score(spike_data['da'], spike_data['Predicted_da'])
            }
        
        # Overall metrics
        overall_metrics = {
            'overall_mae': mean_absolute_error(clean_df['da'], clean_df['Predicted_da']),
            'overall_rmse': np.sqrt(np.mean((clean_df['da'] - clean_df['Predicted_da'])**2)),
            'overall_r2': r2_score(clean_df['da'], clean_df['Predicted_da'])
        }
        
        results = {
            'n_actual_spikes': int(n_actual_spikes),
            'n_predicted_spikes': int(n_predicted_spikes),
            'spike_detection': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            },
            'spike_magnitude': magnitude_metrics,
            'spike_timing': timing_metrics,
            'overall_performance': overall_metrics,
            'spike_threshold_ppm': self.spike_threshold
        }
        
        logger.info(f"Spike timing evaluation completed: {n_actual_spikes} actual spikes, "
                   f"F1={f1_score:.3f}, Spike MAE={magnitude_metrics.get('spike_mae', 0):.2f}")
        
        return results
    
    def _calculate_spike_timing_accuracy(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate timing accuracy metrics for spike events.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with timing metrics
        """
        timing_results = {
            'early_predictions': 0,
            'on_time_predictions': 0,
            'late_predictions': 0,
            'missed_predictions': 0,
            'average_timing_error_days': 0,
            'timing_accuracy_within_window': 0,
            'site_specific_timing': {}
        }
        
        # Group by site for timing analysis
        for site in predictions_df['site'].unique():
            site_data = predictions_df[predictions_df['site'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            site_timing = self._analyze_site_spike_timing(site_data)
            timing_results['site_specific_timing'][site] = site_timing
            
            # Aggregate site results
            timing_results['early_predictions'] += site_timing['early_predictions']
            timing_results['on_time_predictions'] += site_timing['on_time_predictions']
            timing_results['late_predictions'] += site_timing['late_predictions']
            timing_results['missed_predictions'] += site_timing['missed_predictions']
        
        # Calculate overall timing metrics
        total_spikes = (timing_results['early_predictions'] + 
                       timing_results['on_time_predictions'] + 
                       timing_results['late_predictions'] + 
                       timing_results['missed_predictions'])
        
        if total_spikes > 0:
            timing_results['timing_accuracy_within_window'] = (
                timing_results['on_time_predictions'] / total_spikes
            )
        
        return timing_results
    
    def _analyze_site_spike_timing(self, site_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze timing accuracy for a specific site.
        
        Args:
            site_data: Site-specific DataFrame with predictions
            
        Returns:
            Dictionary with site timing metrics
        """
        # Identify actual spike events
        actual_spikes = site_data['da'] > self.spike_threshold
        spike_indices = site_data[actual_spikes].index.tolist()
        
        timing_results = {
            'early_predictions': 0,
            'on_time_predictions': 0,
            'late_predictions': 0,
            'missed_predictions': 0,
            'timing_errors': [],
            'spike_events_analyzed': len(spike_indices)
        }
        
        # Analyze timing for each spike event
        for spike_idx in spike_indices:
            spike_date = pd.to_datetime(site_data.loc[spike_idx, 'date'])
            
            # Look for predictions in the window around this spike
            window_start = spike_date - pd.Timedelta(days=self.temporal_window)
            window_end = spike_date + pd.Timedelta(days=self.temporal_window)
            
            # Convert dates to datetime for comparison
            site_data_dates = pd.to_datetime(site_data['date'])
            window_data = site_data[
                (site_data_dates >= window_start) & 
                (site_data_dates <= window_end)
            ]
            
            # Check if any predictions in the window predicted a spike
            predicted_spikes_in_window = window_data['Predicted_da'] > self.spike_threshold
            
            if not predicted_spikes_in_window.any():
                timing_results['missed_predictions'] += 1
            else:
                # Find the first predicted spike in the window
                first_predicted_spike_idx = window_data[predicted_spikes_in_window].index[0]
                predicted_spike_date = pd.to_datetime(site_data.loc[first_predicted_spike_idx, 'date'])
                
                # Calculate timing error
                timing_error_days = (predicted_spike_date - spike_date).days
                timing_results['timing_errors'].append(timing_error_days)
                
                if timing_error_days < -1:  # More than 1 day early
                    timing_results['early_predictions'] += 1
                elif timing_error_days > 1:  # More than 1 day late
                    timing_results['late_predictions'] += 1
                else:  # Within 1 day
                    timing_results['on_time_predictions'] += 1
        
        return timing_results
    
    def compare_models_spike_focus(self, model_predictions: pd.DataFrame, 
                                 baseline_predictions: pd.DataFrame, 
                                 model_name: str = "Primary Model", 
                                 baseline_name: str = "Baseline") -> Dict[str, Any]:
        """
        Compare two models with focus on spike timing performance.
        
        Args:
            model_predictions: Primary model predictions DataFrame
            baseline_predictions: Baseline model predictions DataFrame
            model_name: Name of primary model
            baseline_name: Name of baseline model
            
        Returns:
            Detailed comparison results
        """
        logger.info(f"Comparing {model_name} vs {baseline_name} for spike timing performance")
        
        # Evaluate both models
        model_results = self.evaluate_spike_timing_performance(model_predictions)
        baseline_results = self.evaluate_spike_timing_performance(baseline_predictions)
        
        # Calculate improvements
        comparison = {
            'model_name': model_name,
            'baseline_name': baseline_name,
            'model_performance': model_results,
            'baseline_performance': baseline_results,
            'improvements': self._calculate_improvements(model_results, baseline_results),
            'recommendation': self._generate_recommendation(model_results, baseline_results, model_name, baseline_name)
        }
        
        logger.info(f"Model comparison completed. F1 improvement: "
                   f"{comparison['improvements']['spike_detection']['f1_improvement']:.3f}")
        
        return comparison
    
    def _calculate_improvements(self, model_results: Dict, baseline_results: Dict) -> Dict[str, Any]:
        """Calculate improvement metrics between model and baseline."""
        improvements = {
            'spike_detection': {},
            'spike_magnitude': {},
            'overall_performance': {}
        }
        
        # Spike detection improvements
        if model_results['spike_detection'] and baseline_results['spike_detection']:
            model_det = model_results['spike_detection']
            baseline_det = baseline_results['spike_detection']
            
            improvements['spike_detection'] = {
                'precision_improvement': model_det['precision'] - baseline_det['precision'],
                'recall_improvement': model_det['recall'] - baseline_det['recall'],
                'f1_improvement': model_det['f1_score'] - baseline_det['f1_score']
            }
        
        # Spike magnitude improvements
        if model_results['spike_magnitude'] and baseline_results['spike_magnitude']:
            model_mag = model_results['spike_magnitude']
            baseline_mag = baseline_results['spike_magnitude']
            
            improvements['spike_magnitude'] = {
                'mae_improvement': baseline_mag['spike_mae'] - model_mag['spike_mae'],
                'rmse_improvement': baseline_mag['spike_rmse'] - model_mag['spike_rmse'],
                'bias_improvement': abs(baseline_mag['spike_bias']) - abs(model_mag['spike_bias']),
                'r2_improvement': model_mag['spike_r2'] - baseline_mag['spike_r2']
            }
        
        # Overall performance improvements
        model_overall = model_results['overall_performance']
        baseline_overall = baseline_results['overall_performance']
        
        improvements['overall_performance'] = {
            'mae_improvement': baseline_overall['overall_mae'] - model_overall['overall_mae'],
            'rmse_improvement': baseline_overall['overall_rmse'] - model_overall['overall_rmse'],
            'r2_improvement': model_overall['overall_r2'] - baseline_overall['overall_r2']
        }
        
        return improvements
    
    def _generate_recommendation(self, model_results: Dict, baseline_results: Dict, 
                               model_name: str, baseline_name: str) -> str:
        """Generate recommendation based on comparison results."""
        
        if not model_results['spike_detection'] or not baseline_results['spike_detection']:
            return "Cannot compare models - insufficient spike data"
        
        model_f1 = model_results['spike_detection']['f1_score']
        baseline_f1 = baseline_results['spike_detection']['f1_score']
        
        f1_improvement = model_f1 - baseline_f1
        
        if f1_improvement > 0.05:  # 5% improvement threshold
            return f"{model_name} shows significant improvement over {baseline_name} " \
                   f"(F1: {model_f1:.3f} vs {baseline_f1:.3f}). Recommend using {model_name}."
        elif f1_improvement < -0.05:
            return f"{baseline_name} outperforms {model_name} " \
                   f"(F1: {baseline_f1:.3f} vs {model_f1:.3f}). Consider using {baseline_name} " \
                   f"or improving {model_name}."
        else:
            return f"Performance is similar between {model_name} and {baseline_name} " \
                   f"(F1 difference: {f1_improvement:.3f}). Consider other factors for selection."
    
    def save_comparison_results(self, comparison_results: Dict[str, Any], 
                              output_path: str) -> None:
        """
        Save model comparison results to JSON file.
        
        Args:
            comparison_results: Results from compare_models_spike_focus
            output_path: Path to save results
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(comparison_results)
        
        # Add metadata
        serializable_results['evaluation_metadata'] = {
            'spike_threshold_ppm': self.spike_threshold,
            'temporal_window_days': self.temporal_window,
            'generated_at': datetime.now().isoformat(),
            'optimizer_version': '1.0'
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to {output_path}")