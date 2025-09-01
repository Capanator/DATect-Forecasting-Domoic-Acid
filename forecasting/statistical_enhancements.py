"""
Statistical Enhancements for DATect Forecasting System
=====================================================

Implements bootstrap confidence intervals, uncertainty quantification,
and statistical validation for scientific rigor.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from typing import Dict, Any, Tuple, Optional, Union

warnings.filterwarnings('ignore')

class StatisticalEnhancer:
    """Provides statistical enhancements for forecasting models."""
    
    def __init__(self, n_bootstrap_iterations: int = 1000, confidence_level: float = 0.95):
        """
        Initialize statistical enhancer.
        
        Args:
            n_bootstrap_iterations: Number of bootstrap samples for CI calculation
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        """
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def bootstrap_prediction_interval(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_pred: pd.DataFrame,
        task_type: str = 'regression'
    ) -> Dict[str, Any]:
        """
        Generate bootstrap prediction intervals for uncertainty quantification.
        
        Args:
            model: Trained sklearn-compatible model
            X_train: Training features
            y_train: Training targets
            X_pred: Features for prediction
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with prediction statistics and confidence intervals
        """
        predictions = []
        
        # Handle single prediction case
        if len(X_pred.shape) == 1 or (len(X_pred.shape) == 2 and X_pred.shape[0] == 1):
            if len(X_pred.shape) == 1:
                X_pred = X_pred.reshape(1, -1)
            single_prediction = True
        else:
            single_prediction = False
            
        for i in range(self.n_bootstrap_iterations):
            try:
                # Bootstrap sample with replacement
                n_samples = len(X_train)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                X_boot = X_train.iloc[bootstrap_indices].copy()
                y_boot = y_train.iloc[bootstrap_indices].copy()
                
                # Clone and train model on bootstrap sample
                model_boot = clone(model)
                model_boot.fit(X_boot, y_boot)
                
                # Make prediction
                if task_type == 'regression':
                    pred = model_boot.predict(X_pred)
                else:  # classification
                    pred = model_boot.predict_proba(X_pred)[:, 1] if hasattr(model_boot, 'predict_proba') else model_boot.predict(X_pred)
                
                predictions.append(pred[0] if single_prediction else pred)
                
            except Exception as e:
                # Skip failed bootstrap iterations
                continue
        
        if not predictions:
            # Fallback if all bootstrap iterations failed
            base_pred = model.predict(X_pred)
            if single_prediction:
                base_pred = base_pred[0]
            return {
                'mean': base_pred,
                'std': np.nan,
                'lower_bound': base_pred,
                'upper_bound': base_pred,
                'n_successful_iterations': 0
            }
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        result = {
            'mean': np.mean(predictions, axis=0),
            'median': np.median(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_bound': np.percentile(predictions, lower_percentile, axis=0),
            'upper_bound': np.percentile(predictions, upper_percentile, axis=0),
            'n_successful_iterations': len(predictions),
            'confidence_level': self.confidence_level
        }
        
        # Handle single prediction case
        if single_prediction:
            for key in ['mean', 'median', 'std', 'lower_bound', 'upper_bound']:
                if isinstance(result[key], np.ndarray):
                    result[key] = float(result[key])
        
        return result
    
    def model_comparison_test(
        self, 
        y_true: np.ndarray, 
        y_pred1: np.ndarray, 
        y_pred2: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Statistical test comparing two models' predictive performance.
        
        Args:
            y_true: True values
            y_pred1: Predictions from first model
            y_pred2: Predictions from second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary with comparison statistics and test results
        """
        # Calculate residuals
        residuals1 = y_true - y_pred1
        residuals2 = y_true - y_pred2
        
        # Calculate metrics for both models
        mae1, mae2 = mean_absolute_error(y_true, y_pred1), mean_absolute_error(y_true, y_pred2)
        mse1, mse2 = mean_squared_error(y_true, y_pred1), mean_squared_error(y_true, y_pred2)
        r2_1, r2_2 = r2_score(y_true, y_pred1), r2_score(y_true, y_pred2)
        
        # Paired t-test on squared residuals (for comparing MSE)
        squared_residuals1 = residuals1 ** 2
        squared_residuals2 = residuals2 ** 2
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(squared_residuals1, squared_residuals2)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(squared_residuals1, squared_residuals2)
        except:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        
        # Effect size (Cohen's d for paired data)
        diff = squared_residuals1 - squared_residuals2
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        return {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_metrics': {'mae': mae1, 'rmse': np.sqrt(mse1), 'r2': r2_1},
            'model2_metrics': {'mae': mae2, 'rmse': np.sqrt(mse2), 'r2': r2_2},
            'improvement': {
                'mae_improvement_pct': ((mae1 - mae2) / mae1) * 100,
                'rmse_improvement_pct': ((np.sqrt(mse1) - np.sqrt(mse2)) / np.sqrt(mse1)) * 100,
                'r2_improvement': r2_2 - r2_1
            },
            'statistical_tests': {
                'paired_t_test': {'statistic': t_stat, 'p_value': p_value},
                'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
                'cohens_d': cohens_d
            },
            'significant_improvement': p_value < 0.05 and mae2 < mae1,
            'interpretation': self._interpret_comparison(mae1, mae2, r2_1, r2_2, p_value)
        }
    
    def _interpret_comparison(self, mae1: float, mae2: float, r2_1: float, r2_2: float, p_value: float) -> str:
        """Generate human-readable interpretation of model comparison."""
        mae_improvement = ((mae1 - mae2) / mae1) * 100
        r2_improvement = r2_2 - r2_1
        
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        if mae2 < mae1:
            return f"Model 2 shows {mae_improvement:.1f}% improvement in MAE and {r2_improvement:.3f} improvement in R² ({significance}, p={p_value:.4f})"
        else:
            return f"Model 1 performs better than Model 2 ({significance}, p={p_value:.4f})"
    
    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive residual analysis for model diagnostics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual analysis results
        """
        residuals = y_true - y_pred
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan)
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        
        # Homoscedasticity (constant variance) tests
        # Split residuals into groups based on predicted values
        n_groups = min(3, len(residuals) // 10)  # Ensure sufficient samples per group
        if n_groups >= 2:
            pred_groups = pd.qcut(y_pred, q=n_groups, duplicates='drop')
            grouped_residuals = [residuals[pred_groups == group] for group in pred_groups.unique()]
            
            try:
                levene_stat, levene_p = stats.levene(*grouped_residuals)
            except:
                levene_stat, levene_p = np.nan, np.nan
        else:
            levene_stat, levene_p = np.nan, np.nan
        
        # Autocorrelation test (Durbin-Watson approximation)
        if len(residuals) > 2:
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        else:
            dw_stat = np.nan
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }
        
        return {
            'residual_statistics': residual_stats,
            'normality_tests': {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p}
            },
            'homoscedasticity_tests': {
                'levene': {'statistic': levene_stat, 'p_value': levene_p}
            },
            'autocorrelation_tests': {
                'durbin_watson': dw_stat
            },
            'diagnostic_flags': {
                'non_normal_residuals': shapiro_p < 0.05 if not np.isnan(shapiro_p) else False,
                'heteroscedasticity': levene_p < 0.05 if not np.isnan(levene_p) else False,
                'autocorrelation': dw_stat < 1.5 or dw_stat > 2.5 if not np.isnan(dw_stat) else False
            }
        }

    def calculate_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   uncertainty_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive prediction metrics including uncertainty validation.
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            uncertainty_bounds: Optional tuple of (lower_bounds, upper_bounds) for interval validation
            
        Returns:
            Dictionary with comprehensive prediction metrics
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        metrics = {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_absolute_error': median_ae,
            'n_samples': len(y_true)
        }
        
        # Uncertainty validation if bounds provided
        if uncertainty_bounds is not None:
            lower_bounds, upper_bounds = uncertainty_bounds
            
            # Coverage probability (percentage of true values within prediction intervals)
            within_bounds = (y_true >= lower_bounds) & (y_true <= upper_bounds)
            coverage = np.mean(within_bounds) * 100
            
            # Mean interval width
            interval_width = np.mean(upper_bounds - lower_bounds)
            
            # Interval score (lower is better)
            alpha = 1 - self.confidence_level
            interval_score = np.mean(
                (upper_bounds - lower_bounds) + 
                (2/alpha) * (lower_bounds - y_true) * (y_true < lower_bounds) +
                (2/alpha) * (y_true - upper_bounds) * (y_true > upper_bounds)
            )
            
            metrics.update({
                'uncertainty_validation': {
                    'coverage_probability': coverage,
                    'target_coverage': self.confidence_level * 100,
                    'coverage_error': coverage - (self.confidence_level * 100),
                    'mean_interval_width': interval_width,
                    'interval_score': interval_score,
                    'well_calibrated': abs(coverage - (self.confidence_level * 100)) < 5  # Within 5% of target
                }
            })
        
        return metrics