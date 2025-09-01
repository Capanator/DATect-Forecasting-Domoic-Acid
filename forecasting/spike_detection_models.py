"""
Advanced Spike Detection Models for Domoic Acid Forecasting
============================================================

Specialized models designed specifically for detecting initial spike timing
rather than overall regression accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.neural_network import MLPRegressor
    HAS_MLP = True
except ImportError:
    HAS_MLP = False

from .logging_config import get_logger

logger = get_logger(__name__)


class SpikeDetectionEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble model combining baseline prediction with spike detection.
    
    Strategy:
    1. Baseline model predicts gradual changes
    2. Spike detector identifies when spikes are likely
    3. Combination logic merges predictions
    """
    
    def __init__(self, spike_threshold=20.0, baseline_weight=0.3, spike_weight=0.7):
        self.spike_threshold = spike_threshold
        self.baseline_weight = baseline_weight
        self.spike_weight = spike_weight
        self.baseline_model = None
        self.spike_model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Fit both baseline and spike detection models."""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_baseline = y.copy()
        y_spike = (y > self.spike_threshold).astype(int)
        
        # Fit baseline model (predicts actual values, good for gradual changes)
        self.baseline_model = LinearRegression()
        self.baseline_model.fit(X_scaled, y_baseline)
        
        # Fit spike detection model (binary classification for spike events)
        if HAS_XGBOOST:
            self.spike_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=10,  # Heavy weight for positive class
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.spike_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        
        self.spike_model.fit(X_scaled, y_spike)
        
        return self
    
    def predict(self, X):
        """Combine baseline and spike predictions."""
        X_scaled = self.scaler.transform(X)
        
        # Get baseline predictions
        baseline_pred = self.baseline_model.predict(X_scaled)
        
        # Get spike probabilities
        if hasattr(self.spike_model, 'predict_proba'):
            spike_probs = self.spike_model.predict_proba(X_scaled)[:, 1]  # Probability of spike
        else:
            spike_probs = self.spike_model.predict(X_scaled)
        
        # Combination logic: if spike probability is high, predict elevated DA
        final_pred = baseline_pred.copy()
        
        # Where spike probability > 0.5, adjust prediction upward
        spike_mask = spike_probs > 0.5
        
        # For detected spikes, blend baseline with spike threshold
        final_pred[spike_mask] = (
            self.baseline_weight * baseline_pred[spike_mask] + 
            self.spike_weight * (self.spike_threshold + spike_probs[spike_mask] * 30)
        )
        
        return np.maximum(final_pred, 0)  # Ensure non-negative
    
    def get_spike_probabilities(self, X):
        """Get spike detection probabilities for analysis."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.spike_model, 'predict_proba'):
            return self.spike_model.predict_proba(X_scaled)[:, 1]
        else:
            return self.spike_model.predict(X_scaled)


class RateOfChangeModel(BaseEstimator, RegressorMixin):
    """
    Model focusing on rate of change and derivative features to detect spikes.
    """
    
    def __init__(self, spike_threshold=20.0, lookback_window=3):
        self.spike_threshold = spike_threshold
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = StandardScaler()
        
    def _create_rate_features(self, X, y_lag=None):
        """Create rate of change and derivative features."""
        
        # Convert to DataFrame for easier manipulation
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Only use environmental features for rate calculations to avoid data leakage
        # Don't use y_lag during prediction as it won't be available
        
        # Create rate-of-change features from environmental variables only
        feature_cols = [col for col in X_df.columns if not col.startswith('da_')]
        
        # Create rolling statistics and rate features for environmental variables
        for col in feature_cols[:8]:  # Limit to avoid too many features
            if X_df[col].dtype in ['float64', 'int64']:
                # Rolling statistics (window=2 for responsiveness)
                X_df[f'{col}_roll_mean'] = X_df[col].rolling(window=2, min_periods=1).mean()
                X_df[f'{col}_roll_std'] = X_df[col].rolling(window=2, min_periods=1).std()
                
                # Rate of change (difference from previous)
                X_df[f'{col}_rate'] = X_df[col].diff(1)
                
                # Trend indicator
                X_df[f'{col}_trend'] = np.sign(X_df[f'{col}_rate'])
        
        # Create interaction features between key environmental variables
        if len(feature_cols) >= 2:
            # Temperature * Chlorophyll interactions (common bloom trigger)
            temp_cols = [col for col in feature_cols if 'sst' in col.lower() or 'temp' in col.lower()]
            chl_cols = [col for col in feature_cols if 'chla' in col.lower() or 'chlor' in col.lower()]
            
            for temp_col in temp_cols[:2]:  # Limit to avoid too many features
                for chl_col in chl_cols[:2]:
                    if temp_col in X_df.columns and chl_col in X_df.columns:
                        X_df[f'{temp_col}_x_{chl_col}'] = X_df[temp_col] * X_df[chl_col]
        
        return X_df.fillna(0)  # Fill NaN from rolling operations
    
    def fit(self, X, y):
        """Fit model with rate-of-change features."""
        
        # Create enhanced features (only using environmental variables)
        X_enhanced = self._create_rate_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        # Use Random Forest for better handling of feature interactions
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                reg_alpha=0.1,
                reg_lambda=0.1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        # Weight samples to emphasize spike events
        sample_weights = np.ones(len(y))
        spike_mask = y > self.spike_threshold
        sample_weights[spike_mask] *= 8  # 8x weight for spikes
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X):
        """Predict using rate-of-change enhanced features."""
        X_enhanced = self._create_rate_features(X)
        X_scaled = self.scaler.transform(X_enhanced)
        return np.maximum(self.model.predict(X_scaled), 0)


class MultiHorizonModel(BaseEstimator, RegressorMixin):
    """
    Multi-step ahead forecasting to capture spike development patterns.
    """
    
    def __init__(self, horizons=[1, 3, 7], spike_threshold=20.0):
        self.horizons = horizons
        self.spike_threshold = spike_threshold
        self.models = {}
        self.scalers = {}
        
    def fit(self, X, y):
        """Fit models for multiple forecast horizons."""
        
        for horizon in self.horizons:
            # Create shifted targets for this horizon
            y_shifted = pd.Series(y).shift(-horizon).dropna()
            X_horizon = X.iloc[:-horizon] if len(y_shifted) < len(X) else X
            
            if len(y_shifted) == 0:
                continue
                
            # Scale features for this horizon
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_horizon)
            self.scalers[horizon] = scaler
            
            # Fit model for this horizon with spike emphasis
            if HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Weight samples for spike detection
            sample_weights = np.ones(len(y_shifted))
            spike_mask = y_shifted > self.spike_threshold
            sample_weights[spike_mask] *= 6
            
            model.fit(X_scaled, y_shifted, sample_weight=sample_weights)
            self.models[horizon] = model
            
        return self
    
    def predict(self, X):
        """Combine predictions from multiple horizons."""
        if not self.models:
            return np.zeros(len(X))
            
        predictions = []
        weights = []
        
        for horizon in self.horizons:
            if horizon in self.models:
                X_scaled = self.scalers[horizon].transform(X)
                pred = self.models[horizon].predict(X_scaled)
                predictions.append(pred)
                
                # Weight shorter horizons more heavily
                weight = 1.0 / horizon
                weights.append(weight)
        
        if not predictions:
            return np.zeros(len(X))
            
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_pred = np.average(predictions, axis=0, weights=weights)
        
        return np.maximum(final_pred, 0)


class AnomalyDetectionModel(BaseEstimator, RegressorMixin):
    """
    Treat spike detection as an anomaly detection problem.
    """
    
    def __init__(self, spike_threshold=20.0, contamination=0.1):
        self.spike_threshold = spike_threshold
        self.contamination = contamination
        self.anomaly_detector = None
        self.baseline_model = None
        self.scaler = StandardScaler()
        self.historical_mean = 0
        
    def fit(self, X, y):
        """Fit anomaly detector and baseline model."""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store historical statistics
        self.historical_mean = np.mean(y)
        
        # Fit anomaly detector on features (unsupervised)
        self.anomaly_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_detector.fit(X_scaled)
        
        # Fit baseline model for normal periods
        normal_mask = y <= self.spike_threshold
        if normal_mask.sum() > 10:  # Need sufficient normal data
            self.baseline_model = LinearRegression()
            self.baseline_model.fit(X_scaled[normal_mask], y[normal_mask])
        else:
            self.baseline_model = None
            
        return self
    
    def predict(self, X):
        """Predict using anomaly detection + baseline."""
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        is_anomaly = self.anomaly_detector.predict(X_scaled) == -1
        
        # Get baseline predictions
        if self.baseline_model:
            baseline_pred = self.baseline_model.predict(X_scaled)
        else:
            baseline_pred = np.full(len(X), self.historical_mean)
        
        # Combine: use baseline for normal, elevated prediction for anomalies
        final_pred = baseline_pred.copy()
        
        # For anomalies, predict elevated values based on anomaly score
        anomaly_intensity = np.abs(anomaly_scores[is_anomaly])
        if len(anomaly_intensity) > 0:
            # Normalize anomaly scores to reasonable DA range
            max_intensity = np.percentile(anomaly_intensity, 95)
            scaled_intensity = anomaly_intensity / max_intensity if max_intensity > 0 else anomaly_intensity
            
            final_pred[is_anomaly] = (
                0.3 * baseline_pred[is_anomaly] + 
                0.7 * (self.spike_threshold + scaled_intensity * 40)
            )
        
        return np.maximum(final_pred, 0)


class GradientSpikeDetector(BaseEstimator, RegressorMixin):
    """
    Focus on gradient/rate-of-change patterns that precede spikes.
    """
    
    def __init__(self, spike_threshold=20.0, gradient_window=2):
        self.spike_threshold = spike_threshold
        self.gradient_window = gradient_window
        self.model = None
        self.scaler = StandardScaler()
        
    def _create_gradient_features(self, X):
        """Create gradient-based features."""
        
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Create rolling statistics to capture trend changes
        for window in [2, 3, 5]:
            for col in X_df.columns:
                if X_df[col].dtype in ['float64', 'int64']:
                    # Rolling mean and std
                    X_df[f'{col}_roll_mean_{window}'] = X_df[col].rolling(window).mean()
                    X_df[f'{col}_roll_std_{window}'] = X_df[col].rolling(window).std()
                    
                    # Rate of change
                    X_df[f'{col}_diff_{window}'] = X_df[col].diff(window)
                    
                    # Trend detection (positive/negative gradient)
                    X_df[f'{col}_trend_{window}'] = np.sign(X_df[f'{col}_diff_{window}'])
        
        return X_df.fillna(method='ffill').fillna(0)
    
    def fit(self, X, y):
        """Fit gradient-based spike detector."""
        
        # Create gradient features
        X_gradient = self._create_gradient_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_gradient)
        
        # Create targets focusing on spike transitions
        y_spike_transition = np.zeros_like(y)
        
        # Mark points that lead to spikes in next few periods
        for i in range(len(y) - self.gradient_window):
            future_vals = y[i+1:i+1+self.gradient_window]
            if any(val > self.spike_threshold for val in future_vals):
                y_spike_transition[i] = np.max(future_vals)
        
        # Use ensemble of models for robustness
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.95,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        
        # Heavily weight spike transition points
        sample_weights = np.ones(len(y))
        transition_mask = y_spike_transition > self.spike_threshold
        sample_weights[transition_mask] *= 15  # Very high weight for transitions
        
        self.model.fit(X_scaled, y_spike_transition, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X):
        """Predict using gradient-based features."""
        X_gradient = self._create_gradient_features(X)
        X_scaled = self.scaler.transform(X_gradient)
        predictions = self.model.predict(X_scaled)
        
        # Apply smoothing to avoid erratic predictions
        if len(predictions) > 3:
            smoothed = np.convolve(predictions, [0.2, 0.6, 0.2], mode='same')
            # Use original for edges
            smoothed[0] = predictions[0]
            smoothed[-1] = predictions[-1]
            predictions = smoothed
        
        return np.maximum(predictions, 0)


def create_spike_detection_model(model_type, **kwargs):
    """Factory function to create spike detection models."""
    
    models = {
        'ensemble': SpikeDetectionEnsemble,
        'rate_of_change': RateOfChangeModel, 
        'multi_horizon': MultiHorizonModel,
        'anomaly': AnomalyDetectionModel,
        'gradient': GradientSpikeDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown spike detection model: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)