"""
Seasonal Modeling with Time-Varying Parameters

This module implements advanced seasonal models including:
- Time-varying coefficient models
- Seasonal decomposition and modeling
- Adaptive seasonal patterns
- Environmental seasonality detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import logging

logger = logging.getLogger(__name__)


class SeasonalDecomposer:
    """
    Decompose time series into trend, seasonal, and residual components
    """
    
    def __init__(self, 
                 seasonal_periods: List[int] = [52],  # Weekly data: 52 weeks = 1 year
                 method: str = 'additive',
                 robust: bool = True):
        """
        Initialize seasonal decomposer
        
        Args:
            seasonal_periods: List of seasonal periods to detect
            method: 'additive' or 'multiplicative'
            robust: Whether to use robust seasonal estimation
        """
        self.seasonal_periods = seasonal_periods
        self.method = method
        self.robust = robust
        
        self.trend_ = None
        self.seasonal_ = {}
        self.residual_ = None
        self.seasonality_strength_ = {}
        
    def fit_transform(self, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, np.ndarray]:
        """
        Decompose time series into components
        
        Args:
            y: Time series values
            dates: Optional dates for the time series
            
        Returns:
            Dictionary with trend, seasonal components, and residuals
        """
        n = len(y)
        
        # Estimate trend using moving average
        self.trend_ = self._estimate_trend(y)
        
        # Detrend the series
        if self.method == 'additive':
            detrended = y - self.trend_
        else:  # multiplicative
            detrended = y / (self.trend_ + 1e-10)
        
        # Estimate seasonal components
        total_seasonal = np.zeros(n)
        
        for period in self.seasonal_periods:
            if period < n:
                seasonal_component = self._estimate_seasonal_component(detrended, period)
                self.seasonal_[period] = seasonal_component
                
                if self.method == 'additive':
                    total_seasonal += seasonal_component
                else:
                    total_seasonal *= seasonal_component
                
                # Calculate seasonality strength
                self.seasonality_strength_[period] = self._calculate_seasonality_strength(
                    detrended, seasonal_component
                )
        
        # Calculate residuals
        if self.method == 'additive':
            self.residual_ = y - self.trend_ - total_seasonal
        else:
            self.residual_ = y / ((self.trend_ + 1e-10) * (total_seasonal + 1e-10))
        
        return {
            'trend': self.trend_,
            'seasonal': total_seasonal,
            'seasonal_components': self.seasonal_,
            'residual': self.residual_,
            'seasonality_strength': self.seasonality_strength_
        }
    
    def _estimate_trend(self, y: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """Estimate trend component using centered moving average"""
        n = len(y)
        
        if window_size is None:
            # Use largest seasonal period for trend window, or default to 24
            window_size = max(self.seasonal_periods) if self.seasonal_periods else 24
            window_size = min(window_size, n // 4)  # Don't exceed 1/4 of series length
        
        # Centered moving average
        trend = np.full(n, np.nan)
        half_window = window_size // 2
        
        for i in range(half_window, n - half_window):
            trend[i] = np.nanmean(y[i - half_window:i + half_window + 1])
        
        # Extrapolate to edges
        trend[:half_window] = trend[half_window]
        trend[-half_window:] = trend[-half_window - 1]
        
        return trend
    
    def _estimate_seasonal_component(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """Estimate seasonal component for given period"""
        n = len(detrended)
        seasonal = np.zeros(n)
        
        # Calculate seasonal pattern by averaging over cycles
        seasonal_pattern = np.full(period, np.nan)
        
        for phase in range(period):
            # Get all observations at this seasonal phase
            indices = np.arange(phase, n, period)
            values = detrended[indices]
            
            if self.robust:
                # Use median for robustness
                seasonal_pattern[phase] = np.nanmedian(values)
            else:
                seasonal_pattern[phase] = np.nanmean(values)
        
        # Remove any remaining NaN values
        seasonal_pattern = self._interpolate_seasonal_pattern(seasonal_pattern)
        
        # Center the seasonal pattern (make it sum to zero for additive)
        if self.method == 'additive':
            seasonal_pattern -= np.mean(seasonal_pattern)
        else:
            seasonal_pattern /= np.mean(seasonal_pattern)
        
        # Repeat pattern for full series
        for i in range(n):
            seasonal[i] = seasonal_pattern[i % period]
        
        return seasonal
    
    def _interpolate_seasonal_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Interpolate missing values in seasonal pattern"""
        if not np.isnan(pattern).any():
            return pattern
        
        # Simple linear interpolation for missing values
        indices = np.arange(len(pattern))
        valid_mask = ~np.isnan(pattern)
        
        if np.sum(valid_mask) < 2:
            # Not enough valid values, use overall mean
            return np.full_like(pattern, np.nanmean(pattern))
        
        pattern_interpolated = np.interp(indices, indices[valid_mask], pattern[valid_mask])
        return pattern_interpolated
    
    def _calculate_seasonality_strength(self, detrended: np.ndarray, seasonal: np.ndarray) -> float:
        """Calculate strength of seasonal component"""
        # Variance explained by seasonal component
        total_var = np.var(detrended[~np.isnan(detrended)])
        seasonal_var = np.var(seasonal)
        
        if total_var > 0:
            return min(1.0, seasonal_var / total_var)
        else:
            return 0.0


class TimeVaryingCoefficientModel(BaseEstimator, RegressorMixin):
    """
    Model with time-varying coefficients using local linear regression
    """
    
    def __init__(self, 
                 bandwidth: float = 0.1,
                 kernel: str = 'gaussian',
                 seasonal_features: bool = True,
                 trend_features: bool = True,
                 regularization: float = 0.01):
        """
        Initialize time-varying coefficient model
        
        Args:
            bandwidth: Bandwidth for local regression (fraction of data)
            kernel: Kernel function ('gaussian', 'epanechnikov', 'uniform')
            seasonal_features: Whether to include seasonal features
            trend_features: Whether to include trend features
            regularization: L2 regularization parameter
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.seasonal_features = seasonal_features
        self.trend_features = trend_features
        self.regularization = regularization
        
        self.X_train_ = None
        self.y_train_ = None
        self.time_train_ = None
        self.feature_scaler_ = None
        
    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        """Calculate kernel weights"""
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * (distances ** 2))
        elif self.kernel == 'epanechnikov':
            weights = 1 - distances ** 2
            weights[distances > 1] = 0
            return np.maximum(0, weights)
        elif self.kernel == 'uniform':
            return (distances <= 1).astype(float)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _create_time_features(self, time_values: np.ndarray) -> np.ndarray:
        """Create time-based features"""
        features = []
        
        if self.trend_features:
            # Linear and polynomial trends
            normalized_time = (time_values - time_values.min()) / (time_values.max() - time_values.min())
            features.append(normalized_time)
            features.append(normalized_time ** 2)
        
        if self.seasonal_features and len(time_values) > 52:
            # Seasonal features (assuming weekly data)
            week_of_year = np.arange(len(time_values)) % 52
            features.append(np.sin(2 * np.pi * week_of_year / 52))
            features.append(np.cos(2 * np.pi * week_of_year / 52))
            
            # Semi-annual cycle
            features.append(np.sin(4 * np.pi * week_of_year / 52))
            features.append(np.cos(4 * np.pi * week_of_year / 52))
        
        return np.column_stack(features) if features else np.zeros((len(time_values), 1))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            time_values: Optional[np.ndarray] = None) -> 'TimeVaryingCoefficientModel':
        """
        Fit the time-varying coefficient model
        
        Args:
            X: Feature matrix
            y: Target values
            time_values: Time values (if None, uses indices)
        """
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        if time_values is None:
            self.time_train_ = np.arange(len(y)).astype(float)
        else:
            self.time_train_ = time_values.copy()
        
        # Create time features
        time_features = self._create_time_features(self.time_train_)
        
        # Combine spatial and time features
        if X.shape[1] > 0:
            self.X_train_ = np.hstack([X, time_features])
        else:
            self.X_train_ = time_features
        
        # Scale features
        self.feature_scaler_ = StandardScaler()
        self.X_train_ = self.feature_scaler_.fit_transform(self.X_train_)
        
        return self
    
    def predict(self, X: np.ndarray, time_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using local linear regression"""
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_pred = X.shape[0]
        predictions = np.zeros(n_pred)
        
        if time_values is None:
            # Assume prediction times follow training times
            time_pred = self.time_train_[-1] + np.arange(1, n_pred + 1)
        else:
            time_pred = time_values.copy()
        
        # Create time features for prediction
        time_features_pred = self._create_time_features(time_pred)
        
        # Combine features
        if X.shape[1] > 0:
            X_pred = np.hstack([X, time_features_pred])
        else:
            X_pred = time_features_pred
        
        # Scale features
        X_pred = self.feature_scaler_.transform(X_pred)
        
        # Local regression for each prediction point
        bandwidth_points = max(1, int(self.bandwidth * len(self.X_train_)))
        
        for i in range(n_pred):
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train_ - X_pred[i]) ** 2, axis=1))
            
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances)[:bandwidth_points]
            
            # Calculate kernel weights
            local_distances = distances[nearest_indices]
            if local_distances.max() > 0:
                normalized_distances = local_distances / local_distances.max()
            else:
                normalized_distances = local_distances
            
            weights = self._kernel_function(normalized_distances)
            
            # Weighted local regression
            X_local = self.X_train_[nearest_indices]
            y_local = self.y_train_[nearest_indices]
            
            if np.sum(weights) > 0:
                # Weighted ridge regression
                try:
                    # Add regularization
                    XtW = X_local.T * weights
                    XtWX = XtW @ X_local + self.regularization * np.eye(X_local.shape[1])
                    XtWy = XtW @ y_local
                    
                    coeffs = np.linalg.solve(XtWX, XtWy)
                    predictions[i] = X_pred[i] @ coeffs
                    
                except np.linalg.LinAlgError:
                    # Fallback to weighted mean
                    predictions[i] = np.average(y_local, weights=weights)
            else:
                predictions[i] = np.mean(self.y_train_)
        
        return predictions


class AdaptiveSeasonalModel(BaseEstimator, RegressorMixin):
    """
    Adaptive seasonal model that adjusts to changing seasonal patterns
    """
    
    def __init__(self,
                 base_periods: List[int] = [52],  # Annual cycle for weekly data
                 adaptation_rate: float = 0.1,
                 min_cycles: int = 2,
                 use_fourier: bool = True,
                 n_harmonics: int = 3):
        """
        Initialize adaptive seasonal model
        
        Args:
            base_periods: Base seasonal periods to model
            adaptation_rate: Rate of adaptation to new patterns
            min_cycles: Minimum number of cycles needed for seasonal estimation
            use_fourier: Whether to use Fourier series for seasonal modeling
            n_harmonics: Number of harmonics in Fourier series
        """
        self.base_periods = base_periods
        self.adaptation_rate = adaptation_rate
        self.min_cycles = min_cycles
        self.use_fourier = use_fourier
        self.n_harmonics = n_harmonics
        
        self.seasonal_models_ = {}
        self.trend_model_ = None
        self.decomposer_ = None
        
    def _create_fourier_features(self, time_values: np.ndarray, period: int) -> np.ndarray:
        """Create Fourier series features for seasonal modeling"""
        features = []
        
        for k in range(1, self.n_harmonics + 1):
            # Sine and cosine terms
            angle = 2 * np.pi * k * time_values / period
            features.append(np.sin(angle))
            features.append(np.cos(angle))
        
        return np.column_stack(features)
    
    def _create_seasonal_features(self, time_values: np.ndarray) -> np.ndarray:
        """Create all seasonal features"""
        all_features = []
        
        for period in self.base_periods:
            if len(time_values) >= self.min_cycles * period:
                if self.use_fourier:
                    fourier_features = self._create_fourier_features(time_values, period)
                    all_features.append(fourier_features)
                else:
                    # Dummy variables for each season
                    seasonal_dummies = np.zeros((len(time_values), period))
                    for i, t in enumerate(time_values):
                        phase = int(t) % period
                        seasonal_dummies[i, phase] = 1
                    all_features.append(seasonal_dummies)
        
        if all_features:
            return np.hstack(all_features)
        else:
            return np.zeros((len(time_values), 1))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            time_values: Optional[np.ndarray] = None) -> 'AdaptiveSeasonalModel':
        """
        Fit the adaptive seasonal model
        
        Args:
            X: Feature matrix (external variables)
            y: Target values
            time_values: Time values
        """
        if time_values is None:
            time_values = np.arange(len(y)).astype(float)
        
        # Decompose time series
        self.decomposer_ = SeasonalDecomposer(
            seasonal_periods=self.base_periods,
            method='additive'
        )
        
        components = self.decomposer_.fit_transform(y, time_values)
        
        # Create seasonal features
        seasonal_features = self._create_seasonal_features(time_values)
        
        # Combine with external features
        if X.shape[1] > 0:
            all_features = np.hstack([X, seasonal_features])
        else:
            all_features = seasonal_features
        
        # Fit seasonal models for each period
        for period in self.base_periods:
            if len(time_values) >= self.min_cycles * period:
                period_features = self._create_fourier_features(time_values, period)
                
                # Fit Ridge regression for this seasonal component
                model = Ridge(alpha=0.1)
                
                if period in components['seasonal_components']:
                    seasonal_target = components['seasonal_components'][period]
                    model.fit(period_features, seasonal_target)
                    self.seasonal_models_[period] = model
        
        # Fit trend model
        trend_features = np.column_stack([
            time_values,
            time_values ** 2
        ])
        
        self.trend_model_ = Ridge(alpha=0.1)
        self.trend_model_.fit(trend_features, components['trend'])
        
        return self
    
    def predict(self, X: np.ndarray, time_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Make adaptive seasonal predictions"""
        if not self.seasonal_models_ or self.trend_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_pred = X.shape[0]
        
        if time_values is None:
            # Extrapolate from training data
            time_values = np.arange(n_pred).astype(float)
        
        predictions = np.zeros(n_pred)
        
        # Predict trend
        trend_features = np.column_stack([
            time_values,
            time_values ** 2
        ])
        trend_pred = self.trend_model_.predict(trend_features)
        predictions += trend_pred
        
        # Predict seasonal components
        for period, model in self.seasonal_models_.items():
            seasonal_features = self._create_fourier_features(time_values, period)
            seasonal_pred = model.predict(seasonal_features)
            predictions += seasonal_pred
        
        return predictions
    
    def get_seasonal_strength(self) -> Dict[int, float]:
        """Get seasonal strength for each period"""
        if self.decomposer_:
            return self.decomposer_.seasonality_strength_
        else:
            return {}


def detect_seasonal_patterns(y: np.ndarray, 
                           dates: Optional[pd.DatetimeIndex] = None,
                           max_period: int = 104) -> Dict[str, Any]:
    """
    Detect seasonal patterns in time series using spectral analysis
    
    Args:
        y: Time series values
        dates: Optional dates
        max_period: Maximum period to consider
        
    Returns:
        Dictionary with detected periods and their strength
    """
    n = len(y)
    
    # Remove trend
    detrended = y - np.linspace(y[0], y[-1], n)
    
    # FFT-based periodogram
    fft = np.fft.fft(detrended)
    frequencies = np.fft.fftfreq(n)
    power = np.abs(fft) ** 2
    
    # Find peaks in power spectrum
    periods = []
    strengths = []
    
    for i in range(1, n // 2):
        if frequencies[i] > 0:
            period = 1 / frequencies[i]
            if 2 <= period <= max_period:
                periods.append(period)
                strengths.append(power[i])
    
    # Sort by strength
    if periods:
        sorted_indices = np.argsort(strengths)[::-1]
        sorted_periods = [periods[i] for i in sorted_indices[:10]]  # Top 10
        sorted_strengths = [strengths[i] for i in sorted_indices[:10]]
    else:
        sorted_periods = []
        sorted_strengths = []
    
    # Detect common environmental periods
    environmental_periods = {
        'annual': 52,  # 52 weeks
        'semi_annual': 26,  # 6 months
        'seasonal': 13,  # 3 months
        'monthly': 4.3,  # ~4.3 weeks per month
        'fortnightly': 2,  # 2 weeks
        'el_nino': 156  # ~3 years (ENSO cycle)
    }
    
    detected_environmental = {}
    
    for name, expected_period in environmental_periods.items():
        # Find closest detected period
        if sorted_periods:
            distances = [abs(p - expected_period) for p in sorted_periods]
            min_distance = min(distances)
            
            if min_distance < expected_period * 0.2:  # Within 20% of expected
                closest_idx = distances.index(min_distance)
                detected_environmental[name] = {
                    'expected_period': expected_period,
                    'detected_period': sorted_periods[closest_idx],
                    'strength': sorted_strengths[closest_idx],
                    'relative_strength': sorted_strengths[closest_idx] / sum(sorted_strengths)
                }
    
    return {
        'detected_periods': sorted_periods,
        'period_strengths': sorted_strengths,
        'environmental_patterns': detected_environmental,
        'dominant_period': sorted_periods[0] if sorted_periods else None,
        'seasonality_score': max(sorted_strengths) / np.var(detrended) if sorted_strengths else 0
    }


# Example usage
if __name__ == "__main__":
    # Generate synthetic seasonal time series
    np.random.seed(42)
    n = 300  # ~6 years of weekly data
    time = np.arange(n)
    
    # Create complex seasonal pattern
    trend = 0.01 * time
    annual_cycle = 5 * np.sin(2 * np.pi * time / 52)
    semi_annual = 2 * np.cos(4 * np.pi * time / 52)
    noise = np.random.randn(n) * 2
    
    y = 20 + trend + annual_cycle + semi_annual + noise
    
    # Add some time-varying amplitude
    varying_amplitude = 1 + 0.5 * np.sin(2 * np.pi * time / 104)  # 2-year cycle
    y += annual_cycle * varying_amplitude
    
    # Create external features
    X = np.random.randn(n, 3)
    
    print("Testing Seasonal Models...")
    
    # Test seasonal decomposition
    print("\n1. Seasonal Decomposition:")
    decomposer = SeasonalDecomposer(seasonal_periods=[52, 26])
    components = decomposer.fit_transform(y)
    
    print(f"Seasonality strengths: {decomposer.seasonality_strength_}")
    
    # Test time-varying coefficient model
    print("\n2. Time-Varying Coefficient Model:")
    tvc_model = TimeVaryingCoefficientModel(bandwidth=0.2)
    
    # Split data
    split_idx = int(0.8 * n)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    time_train, time_test = time[:split_idx], time[split_idx:]
    
    tvc_model.fit(X_train, y_train, time_train)
    tvc_pred = tvc_model.predict(X_test, time_test)
    tvc_r2 = r2_score(y_test, tvc_pred)
    
    print(f"Time-varying model R²: {tvc_r2:.3f}")
    
    # Test adaptive seasonal model
    print("\n3. Adaptive Seasonal Model:")
    adaptive_model = AdaptiveSeasonalModel(base_periods=[52, 26])
    adaptive_model.fit(X_train, y_train, time_train)
    adaptive_pred = adaptive_model.predict(X_test, time_test)
    adaptive_r2 = r2_score(y_test, adaptive_pred)
    
    print(f"Adaptive seasonal model R²: {adaptive_r2:.3f}")
    print(f"Seasonal strengths: {adaptive_model.get_seasonal_strength()}")
    
    # Test pattern detection
    print("\n4. Seasonal Pattern Detection:")
    patterns = detect_seasonal_patterns(y)
    print(f"Dominant period: {patterns['dominant_period']:.1f}")
    print(f"Environmental patterns detected: {list(patterns['environmental_patterns'].keys())}")
    
    for name, info in patterns['environmental_patterns'].items():
        print(f"  {name}: detected period = {info['detected_period']:.1f}, "
              f"expected = {info['expected_period']:.1f}")