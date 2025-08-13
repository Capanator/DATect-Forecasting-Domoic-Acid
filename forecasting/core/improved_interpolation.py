"""
Improved Data Interpolation with Scientific Constraints

This module implements improved interpolation methods with:
- Maximum gap length limits (4-6 weeks)
- Forward-only interpolation to prevent temporal leakage
- Quality tracking and uncertainty estimation
- Multiple interpolation strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class ConstrainedInterpolator:
    """
    Interpolator with scientific constraints for time series data
    """
    
    def __init__(self, 
                 max_gap_weeks: int = 6,
                 method: str = 'linear',
                 limit_direction: str = 'forward',
                 track_quality: bool = True):
        """
        Initialize constrained interpolator
        
        Args:
            max_gap_weeks: Maximum gap length to interpolate (in weeks)
            method: Interpolation method ('linear', 'polynomial', 'spline')
            limit_direction: 'forward', 'backward', or 'both'
            track_quality: Whether to track interpolation quality metrics
        """
        self.max_gap_weeks = max_gap_weeks
        self.method = method
        self.limit_direction = limit_direction
        self.track_quality = track_quality
        
        self.interpolation_stats = {}
        
    def interpolate_series(self, 
                          series: pd.Series, 
                          dates: pd.DatetimeIndex,
                          series_name: str = "unknown") -> Tuple[pd.Series, Dict]:
        """
        Interpolate a time series with gap length constraints
        
        Args:
            series: Time series to interpolate
            dates: Corresponding dates
            series_name: Name of the series for logging
            
        Returns:
            Tuple of (interpolated_series, quality_metrics)
        """
        if len(series) != len(dates):
            raise ValueError("Series and dates must have the same length")
        
        # Create a copy to work with
        interpolated = series.copy()
        
        # Find missing value gaps
        gaps = self._find_gaps(series, dates)
        
        # Filter gaps by maximum allowed length
        valid_gaps = []
        rejected_gaps = []
        
        for gap_start, gap_end, gap_length_weeks in gaps:
            if gap_length_weeks <= self.max_gap_weeks:
                valid_gaps.append((gap_start, gap_end, gap_length_weeks))
            else:
                rejected_gaps.append((gap_start, gap_end, gap_length_weeks))
                if self.track_quality:
                    logger.info(f"Rejecting {gap_length_weeks:.1f} week gap in {series_name} "
                              f"(exceeds {self.max_gap_weeks} week limit)")
        
        # Interpolate valid gaps
        interpolated_points = 0
        
        for gap_start, gap_end, gap_length in valid_gaps:
            # Get the gap indices
            gap_mask = (dates >= gap_start) & (dates <= gap_end)
            gap_indices = np.where(gap_mask)[0]
            
            if len(gap_indices) > 0:
                # Interpolate this gap
                interpolated.iloc[gap_indices] = self._interpolate_gap(
                    series, gap_indices, method=self.method
                )
                interpolated_points += len(gap_indices)
        
        # Calculate quality metrics
        quality_metrics = {}
        if self.track_quality:
            quality_metrics = self._calculate_quality_metrics(
                series, interpolated, valid_gaps, rejected_gaps, series_name
            )
        
        return interpolated, quality_metrics
    
    def _find_gaps(self, series: pd.Series, dates: pd.DatetimeIndex) -> List[Tuple]:
        """
        Find gaps (consecutive missing values) in the time series
        
        Returns:
            List of (gap_start_date, gap_end_date, gap_length_weeks)
        """
        gaps = []
        is_missing = series.isna()
        
        if not is_missing.any():
            return gaps
        
        # Find consecutive runs of missing values
        missing_runs = []
        start_idx = None
        
        for i, missing in enumerate(is_missing):
            if missing and start_idx is None:
                # Start of a gap
                start_idx = i
            elif not missing and start_idx is not None:
                # End of a gap
                missing_runs.append((start_idx, i - 1))
                start_idx = None
        
        # Handle case where series ends with missing values
        if start_idx is not None:
            missing_runs.append((start_idx, len(is_missing) - 1))
        
        # Calculate gap lengths in weeks
        for start_idx, end_idx in missing_runs:
            gap_start_date = dates[start_idx]
            gap_end_date = dates[end_idx]
            gap_length_days = (gap_end_date - gap_start_date).days + 1
            gap_length_weeks = gap_length_days / 7.0
            
            gaps.append((gap_start_date, gap_end_date, gap_length_weeks))
        
        return gaps
    
    def _interpolate_gap(self, 
                        series: pd.Series, 
                        gap_indices: np.ndarray,
                        method: str = 'linear') -> np.ndarray:
        """
        Interpolate a specific gap using the specified method
        
        Args:
            series: Original series
            gap_indices: Indices of the gap to interpolate
            method: Interpolation method
            
        Returns:
            Interpolated values for the gap
        """
        if method == 'linear':
            return self._linear_interpolation(series, gap_indices)
        elif method == 'polynomial':
            return self._polynomial_interpolation(series, gap_indices)
        elif method == 'spline':
            return self._spline_interpolation(series, gap_indices)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _linear_interpolation(self, 
                             series: pd.Series, 
                             gap_indices: np.ndarray) -> np.ndarray:
        """Linear interpolation for a gap"""
        # Find nearest non-missing values
        start_idx = gap_indices[0] - 1
        end_idx = gap_indices[-1] + 1
        
        # Handle edge cases
        if start_idx < 0:
            if self.limit_direction == 'backward':
                # Use first available value after the gap
                while end_idx < len(series) and pd.isna(series.iloc[end_idx]):
                    end_idx += 1
                if end_idx < len(series):
                    return np.full(len(gap_indices), series.iloc[end_idx])
            return np.full(len(gap_indices), np.nan)
        
        if end_idx >= len(series):
            if self.limit_direction == 'forward':
                # Use last available value before the gap
                return np.full(len(gap_indices), series.iloc[start_idx])
            return np.full(len(gap_indices), np.nan)
        
        # Check if boundary values are available
        start_val = series.iloc[start_idx]
        end_val = series.iloc[end_idx]
        
        if pd.isna(start_val) or pd.isna(end_val):
            return np.full(len(gap_indices), np.nan)
        
        # Direction constraints
        if self.limit_direction == 'forward':
            # Only use start value (persistence)
            return np.full(len(gap_indices), start_val)
        elif self.limit_direction == 'backward':
            # Only use end value
            return np.full(len(gap_indices), end_val)
        
        # Both directions allowed - linear interpolation
        n_points = len(gap_indices)
        interpolated = np.linspace(start_val, end_val, n_points + 2)[1:-1]
        
        return interpolated
    
    def _polynomial_interpolation(self, 
                                 series: pd.Series, 
                                 gap_indices: np.ndarray,
                                 degree: int = 2) -> np.ndarray:
        """Polynomial interpolation for a gap"""
        # Get more context points for polynomial fitting
        context_size = max(degree + 1, 4)
        
        start_context = max(0, gap_indices[0] - context_size)
        end_context = min(len(series), gap_indices[-1] + context_size + 1)
        
        # Extract non-missing values in context
        context_indices = []
        context_values = []
        
        for i in range(start_context, end_context):
            if i not in gap_indices and not pd.isna(series.iloc[i]):
                context_indices.append(i)
                context_values.append(series.iloc[i])
        
        if len(context_values) < degree + 1:
            # Fall back to linear interpolation
            return self._linear_interpolation(series, gap_indices)
        
        # Fit polynomial
        try:
            coeffs = np.polyfit(context_indices, context_values, degree)
            interpolated = np.polyval(coeffs, gap_indices)
            return interpolated
        except np.linalg.LinAlgError:
            # Fall back to linear interpolation
            return self._linear_interpolation(series, gap_indices)
    
    def _spline_interpolation(self, 
                             series: pd.Series, 
                             gap_indices: np.ndarray) -> np.ndarray:
        """Spline interpolation for a gap"""
        try:
            from scipy.interpolate import UnivariateSpline
            
            # Get context for spline fitting
            context_size = 6
            start_context = max(0, gap_indices[0] - context_size)
            end_context = min(len(series), gap_indices[-1] + context_size + 1)
            
            # Extract non-missing values
            context_indices = []
            context_values = []
            
            for i in range(start_context, end_context):
                if i not in gap_indices and not pd.isna(series.iloc[i]):
                    context_indices.append(i)
                    context_values.append(series.iloc[i])
            
            if len(context_values) < 4:
                # Fall back to linear interpolation
                return self._linear_interpolation(series, gap_indices)
            
            # Fit spline
            spline = UnivariateSpline(context_indices, context_values, s=0.1)
            interpolated = spline(gap_indices)
            
            return interpolated
            
        except ImportError:
            logger.warning("scipy not available, falling back to linear interpolation")
            return self._linear_interpolation(series, gap_indices)
        except Exception:
            # Fall back to linear interpolation
            return self._linear_interpolation(series, gap_indices)
    
    def _calculate_quality_metrics(self, 
                                  original: pd.Series,
                                  interpolated: pd.Series,
                                  valid_gaps: List,
                                  rejected_gaps: List,
                                  series_name: str) -> Dict:
        """Calculate quality metrics for interpolation"""
        total_points = len(original)
        original_missing = original.isna().sum()
        interpolated_missing = interpolated.isna().sum()
        
        interpolated_points = original_missing - interpolated_missing
        interpolated_ratio = interpolated_points / total_points if total_points > 0 else 0
        
        # Gap statistics
        valid_gap_lengths = [length for _, _, length in valid_gaps]
        rejected_gap_lengths = [length for _, _, length in rejected_gaps]
        
        max_interpolated_gap = max(valid_gap_lengths) if valid_gap_lengths else 0
        total_rejected_weeks = sum(rejected_gap_lengths)
        
        metrics = {
            'series_name': series_name,
            'total_points': total_points,
            'original_missing': original_missing,
            'interpolated_points': interpolated_points,
            'remaining_missing': interpolated_missing,
            'interpolation_ratio': interpolated_ratio,
            'data_completeness': 1 - (interpolated_missing / total_points),
            'n_valid_gaps': len(valid_gaps),
            'n_rejected_gaps': len(rejected_gaps),
            'max_interpolated_gap_weeks': max_interpolated_gap,
            'total_rejected_weeks': total_rejected_weeks,
            'avg_valid_gap_length': np.mean(valid_gap_lengths) if valid_gap_lengths else 0,
            'avg_rejected_gap_length': np.mean(rejected_gap_lengths) if rejected_gap_lengths else 0
        }
        
        return metrics


def improve_dataset_interpolation(df: pd.DataFrame,
                                 value_columns: List[str],
                                 date_column: str = 'Date',
                                 site_column: str = 'Site',
                                 max_gap_weeks: int = 6) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply improved interpolation to dataset with gap constraints
    
    Args:
        df: Input dataframe
        value_columns: Columns to interpolate
        date_column: Name of date column
        site_column: Name of site column
        max_gap_weeks: Maximum gap length to interpolate
        
    Returns:
        Tuple of (improved_dataframe, quality_report)
    """
    df_improved = df.copy()
    quality_report = {
        'interpolation_settings': {
            'max_gap_weeks': max_gap_weeks,
            'method': 'linear',
            'direction': 'forward'
        },
        'series_metrics': []
    }
    
    interpolator = ConstrainedInterpolator(
        max_gap_weeks=max_gap_weeks,
        method='linear',
        limit_direction='forward'
    )
    
    # Process each site separately
    sites = df[site_column].unique()
    
    for site in sites:
        site_mask = df[site_column] == site
        site_data = df[site_mask].copy().sort_values(date_column)
        
        dates = pd.to_datetime(site_data[date_column])
        
        for column in value_columns:
            if column in site_data.columns:
                series = site_data[column]
                series_name = f"{site}_{column}"
                
                # Apply constrained interpolation
                interpolated_series, metrics = interpolator.interpolate_series(
                    series, dates, series_name
                )
                
                # Update the dataframe
                df_improved.loc[site_mask, column] = interpolated_series.values
                
                # Store metrics
                quality_report['series_metrics'].append(metrics)
                
                # Log results
                logger.info(f"Interpolated {series_name}: "
                          f"{metrics['interpolated_points']} points, "
                          f"{metrics['n_rejected_gaps']} gaps rejected")
    
    # Calculate overall statistics
    quality_report['overall_stats'] = _calculate_overall_stats(
        quality_report['series_metrics']
    )
    
    return df_improved, quality_report


def _calculate_overall_stats(series_metrics: List[Dict]) -> Dict:
    """Calculate overall interpolation statistics"""
    if not series_metrics:
        return {}
    
    total_points = sum(m['total_points'] for m in series_metrics)
    total_interpolated = sum(m['interpolated_points'] for m in series_metrics)
    total_rejected_gaps = sum(m['n_rejected_gaps'] for m in series_metrics)
    total_rejected_weeks = sum(m['total_rejected_weeks'] for m in series_metrics)
    
    avg_completeness = np.mean([m['data_completeness'] for m in series_metrics])
    max_gap_interpolated = max(m['max_interpolated_gap_weeks'] for m in series_metrics)
    
    return {
        'total_data_points': total_points,
        'total_interpolated_points': total_interpolated,
        'overall_interpolation_ratio': total_interpolated / total_points if total_points > 0 else 0,
        'average_data_completeness': avg_completeness,
        'total_rejected_gaps': total_rejected_gaps,
        'total_rejected_weeks': total_rejected_weeks,
        'max_gap_interpolated_weeks': max_gap_interpolated,
        'series_processed': len(series_metrics)
    }


def analyze_interpolation_impact(original_df: pd.DataFrame,
                               improved_df: pd.DataFrame,
                               value_columns: List[str],
                               quality_report: Dict) -> Dict:
    """
    Analyze the impact of improved interpolation
    
    Returns:
        Dictionary with comparison metrics
    """
    analysis = {
        'data_quality_improvement': {},
        'interpolation_constraints': {},
        'recommendations': []
    }
    
    # Compare data completeness
    for column in value_columns:
        orig_missing = original_df[column].isna().sum()
        improved_missing = improved_df[column].isna().sum()
        total_points = len(original_df)
        
        orig_completeness = 1 - (orig_missing / total_points)
        improved_completeness = 1 - (improved_missing / total_points)
        
        analysis['data_quality_improvement'][column] = {
            'original_completeness': orig_completeness,
            'improved_completeness': improved_completeness,
            'improvement': improved_completeness - orig_completeness,
            'points_interpolated': orig_missing - improved_missing,
            'remaining_missing': improved_missing
        }
    
    # Analyze constraint effectiveness
    overall_stats = quality_report['overall_stats']
    
    analysis['interpolation_constraints'] = {
        'max_gap_limit_weeks': quality_report['interpolation_settings']['max_gap_weeks'],
        'gaps_rejected': overall_stats.get('total_rejected_gaps', 0),
        'weeks_not_interpolated': overall_stats.get('total_rejected_weeks', 0),
        'max_gap_actually_interpolated': overall_stats.get('max_gap_interpolated_weeks', 0),
        'constraint_effectiveness': 'Good' if overall_stats.get('total_rejected_gaps', 0) > 0 else 'No long gaps found'
    }
    
    # Generate recommendations
    if overall_stats.get('total_rejected_gaps', 0) > 10:
        analysis['recommendations'].append(
            "Consider increasing max_gap_weeks if biological processes support longer interpolation"
        )
    
    if overall_stats.get('average_data_completeness', 0) < 0.8:
        analysis['recommendations'].append(
            "Data completeness is low - consider alternative gap-filling methods"
        )
    
    if overall_stats.get('max_gap_interpolated_weeks', 0) > 4:
        analysis['recommendations'].append(
            "Some long gaps were interpolated - verify biological plausibility"
        )
    
    return analysis


# Example usage and testing
if __name__ == "__main__":
    # Generate test data with gaps
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    
    # Create synthetic data with missing values
    values = np.random.randn(200) * 5 + 20
    
    # Introduce gaps of various lengths
    values[10:15] = np.nan   # 5-week gap (should be interpolated)
    values[50:60] = np.nan   # 10-week gap (should be rejected)
    values[100:103] = np.nan # 3-week gap (should be interpolated)
    values[150:165] = np.nan # 15-week gap (should be rejected)
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'Date': dates,
        'Site': 'TestSite',
        'DA_Levels': values,
        'PN_Levels': values + np.random.randn(200)
    })
    
    print("Testing improved interpolation...")
    print(f"Original missing values: {test_df['DA_Levels'].isna().sum()}")
    
    # Apply improved interpolation
    improved_df, quality_report = improve_dataset_interpolation(
        test_df, 
        ['DA_Levels', 'PN_Levels'],
        max_gap_weeks=6
    )
    
    print(f"After interpolation missing values: {improved_df['DA_Levels'].isna().sum()}")
    print(f"Gaps rejected: {quality_report['overall_stats']['total_rejected_gaps']}")
    print(f"Weeks not interpolated: {quality_report['overall_stats']['total_rejected_weeks']}")
    
    # Analyze impact
    analysis = analyze_interpolation_impact(test_df, improved_df, ['DA_Levels'], quality_report)
    print(f"\nData completeness improvement: {analysis['data_quality_improvement']['DA_Levels']['improvement']:.3f}")
    print("Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")