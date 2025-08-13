"""
Statistical Enhancements for DATect Forecasting System

This module implements advanced statistical methods including:
- Bootstrap confidence intervals
- Uncertainty quantification
- Statistical significance testing
- Residual analysis
- Model comparison metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)


class BootstrapConfidenceIntervals:
    """Calculate bootstrap confidence intervals for predictions"""
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Initialize bootstrap confidence interval calculator
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_prediction_intervals(self, 
                                      X_train: np.ndarray, 
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      model_class,
                                      model_params: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using bootstrap
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            model_class: Model class to use (e.g., XGBRegressor)
            model_params: Model parameters
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if model_params is None:
            model_params = {}
            
        n_samples = len(X_train)
        bootstrap_predictions = []
        
        print(f"Calculating bootstrap confidence intervals ({self.n_bootstrap} iterations)...")
        
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = resample(range(n_samples), n_samples=n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train model on bootstrap sample
            model = model_class(**model_params)
            model.fit(X_boot, y_boot)
            
            # Make predictions
            predictions = model.predict(X_test)
            bootstrap_predictions.append(predictions)
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{self.n_bootstrap} bootstrap iterations")
        
        # Calculate percentiles
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Point predictions (mean of bootstrap predictions)
        predictions = np.mean(bootstrap_predictions, axis=0)
        
        # Confidence intervals
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        # Calculate prediction uncertainty (standard deviation)
        uncertainty = np.std(bootstrap_predictions, axis=0)
        
        return predictions, lower_bound, upper_bound, uncertainty
    
    def calculate_metric_confidence_intervals(self,
                                             y_true: np.ndarray,
                                             y_pred: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Calculate confidence intervals for performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metric names and (value, lower_ci, upper_ci)
        """
        n_samples = len(y_true)
        
        # Bootstrap metrics
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        
        for _ in range(self.n_bootstrap):
            indices = resample(range(n_samples), n_samples=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            r2_scores.append(r2_score(y_true_boot, y_pred_boot))
            mae_scores.append(mean_absolute_error(y_true_boot, y_pred_boot))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        
        # Calculate confidence intervals
        results = {}
        
        for metric_name, scores in [('R2', r2_scores), ('MAE', mae_scores), ('RMSE', rmse_scores)]:
            value = np.mean(scores)
            lower = np.percentile(scores, (self.alpha / 2) * 100)
            upper = np.percentile(scores, (1 - self.alpha / 2) * 100)
            results[metric_name] = (value, lower, upper)
        
        return results


class StatisticalSignificanceTesting:
    """Perform statistical significance tests for model comparisons"""
    
    @staticmethod
    def paired_t_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
        """
        Perform paired t-test on prediction errors
        
        Args:
            errors1: Errors from model 1
            errors2: Errors from model 2
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        differences = errors1 - errors2
        t_stat, p_value = stats.ttest_rel(errors1, errors2)
        return t_stat, p_value
    
    @staticmethod
    def wilcoxon_signed_rank_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative)
        
        Args:
            errors1: Errors from model 1
            errors2: Errors from model 2
            
        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.wilcoxon(errors1, errors2)
        return statistic, p_value
    
    @staticmethod
    def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray, h: int = 1) -> Tuple[float, float]:
        """
        Diebold-Mariano test for predictive accuracy
        
        Args:
            errors1: Errors from model 1
            errors2: Errors from model 2
            h: Forecast horizon
            
        Returns:
            Tuple of (dm_statistic, p_value)
        """
        d = errors1 - errors2
        mean_d = np.mean(d)
        
        # Calculate autocorrelation-robust variance
        gamma_0 = np.var(d)
        if h > 1:
            gamma_sum = 0
            for i in range(1, h):
                gamma_i = np.cov(d[:-i], d[i:])[0, 1]
                gamma_sum += 2 * gamma_i
            var_d = (gamma_0 + gamma_sum) / len(d)
        else:
            var_d = gamma_0 / len(d)
        
        dm_stat = mean_d / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return dm_stat, p_value
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
        """
        Apply multiple comparison correction
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            Corrected p-values
        """
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            return [min(p * n_tests, 1.0) for p in p_values]
        
        elif method == 'holm':
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            corrected = []
            
            for i, p in enumerate(sorted_p):
                corrected_p = min(p * (n_tests - i), 1.0)
                if i > 0:
                    corrected_p = max(corrected_p, corrected[i-1])
                corrected.append(corrected_p)
            
            # Restore original order
            result = np.zeros(n_tests)
            result[sorted_indices] = corrected
            return result.tolist()
        
        elif method == 'fdr':  # Benjamini-Hochberg
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            corrected = []
            
            for i in range(n_tests - 1, -1, -1):
                if i == n_tests - 1:
                    corrected_p = sorted_p[i]
                else:
                    corrected_p = min(sorted_p[i] * n_tests / (i + 1), corrected[-1])
                corrected.append(corrected_p)
            
            corrected.reverse()
            
            # Restore original order
            result = np.zeros(n_tests)
            result[sorted_indices] = corrected
            return result.tolist()
        
        else:
            raise ValueError(f"Unknown correction method: {method}")


class ResidualAnalysis:
    """Perform comprehensive residual analysis"""
    
    @staticmethod
    def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate residuals"""
        return y_true - y_pred
    
    @staticmethod
    def test_normality(residuals: np.ndarray) -> Dict[str, Any]:
        """
        Test residuals for normality
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Shapiro-Wilk test
        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            results['shapiro_wilk'] = {'statistic': stat, 'p_value': p_value}
        
        # Kolmogorov-Smirnov test
        stat, p_value = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p_value}
        
        # Anderson-Darling test
        result = stats.anderson(residuals, dist='norm')
        results['anderson_darling'] = {
            'statistic': result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], result.critical_values))
        }
        
        # Skewness and Kurtosis
        results['skewness'] = stats.skew(residuals)
        results['kurtosis'] = stats.kurtosis(residuals)
        
        # Q-Q plot statistics
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sorted_residuals = np.sort(residuals)
        qq_correlation = np.corrcoef(theoretical_quantiles, sorted_residuals)[0, 1]
        results['qq_correlation'] = qq_correlation
        
        return results
    
    @staticmethod
    def test_homoscedasticity(residuals: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Test for homoscedasticity (constant variance)
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Breusch-Pagan test
        # Regress squared residuals on predicted values
        from sklearn.linear_model import LinearRegression
        
        X = y_pred.reshape(-1, 1)
        y = residuals ** 2
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        n = len(residuals)
        rss = np.sum((y - predictions) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss
        
        lm_statistic = n * r2
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        results['breusch_pagan'] = {'statistic': lm_statistic, 'p_value': p_value}
        
        # Levene's test (split data into groups)
        n_groups = 3
        group_size = len(residuals) // n_groups
        groups = [residuals[i*group_size:(i+1)*group_size] for i in range(n_groups)]
        
        if all(len(g) > 0 for g in groups):
            stat, p_value = stats.levene(*groups)
            results['levene'] = {'statistic': stat, 'p_value': p_value}
        
        return results
    
    @staticmethod
    def test_autocorrelation(residuals: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """
        Test for autocorrelation in residuals
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Durbin-Watson test
        diff = np.diff(residuals)
        dw_statistic = np.sum(diff ** 2) / np.sum(residuals ** 2)
        results['durbin_watson'] = dw_statistic
        
        # Ljung-Box test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        lb_result = acorr_ljungbox(residuals, lags=min(max_lag, len(residuals)//4), return_df=True)
        results['ljung_box'] = {
            'statistics': lb_result['lb_stat'].tolist(),
            'p_values': lb_result['lb_pvalue'].tolist()
        }
        
        # ACF values
        acf_values = []
        for lag in range(1, min(max_lag + 1, len(residuals)//4)):
            if lag < len(residuals):
                acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                acf_values.append(acf)
        results['acf_values'] = acf_values
        
        return results
    
    @staticmethod
    def calculate_influence_statistics(X: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate influence statistics (leverage, Cook's distance)
        
        Returns:
            Dictionary with influence statistics
        """
        n = len(y)
        p = X.shape[1]
        residuals = y - predictions
        
        # Calculate hat matrix diagonal (leverage)
        try:
            H_diag = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            H_diag = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
        
        # Standardized residuals
        mse = np.mean(residuals ** 2)
        std_residuals = residuals / (np.sqrt(mse * (1 - H_diag)))
        
        # Cook's distance
        cooks_d = (std_residuals ** 2 / p) * (H_diag / (1 - H_diag))
        
        # DFFITS
        dffits = std_residuals * np.sqrt(H_diag / (1 - H_diag))
        
        return {
            'leverage': H_diag,
            'standardized_residuals': std_residuals,
            'cooks_distance': cooks_d,
            'dffits': dffits
        }


class UncertaintyQuantification:
    """Quantify prediction uncertainty"""
    
    @staticmethod
    def calculate_epistemic_uncertainty(predictions: np.ndarray) -> float:
        """
        Calculate epistemic (model) uncertainty
        
        Args:
            predictions: Array of predictions from multiple models/bootstraps
            
        Returns:
            Epistemic uncertainty estimate
        """
        return np.std(predictions, axis=0)
    
    @staticmethod
    def calculate_aleatoric_uncertainty(residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculate aleatoric (data) uncertainty
        
        Args:
            residuals: Model residuals
            X: Feature matrix
            
        Returns:
            Aleatoric uncertainty for each sample
        """
        # Estimate heteroscedastic noise
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a model to predict squared residuals
        squared_residuals = residuals ** 2
        noise_model = RandomForestRegressor(n_estimators=100, random_state=42)
        noise_model.fit(X, squared_residuals)
        
        # Predict variance
        predicted_variance = noise_model.predict(X)
        aleatoric_uncertainty = np.sqrt(predicted_variance)
        
        return aleatoric_uncertainty
    
    @staticmethod
    def calculate_total_uncertainty(epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
        """
        Calculate total uncertainty
        
        Args:
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
            
        Returns:
            Total uncertainty
        """
        return np.sqrt(epistemic ** 2 + aleatoric ** 2)
    
    @staticmethod
    def calibrate_uncertainty(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             uncertainty: np.ndarray,
                             n_bins: int = 10) -> Dict[str, Any]:
        """
        Calibrate and evaluate uncertainty estimates
        
        Returns:
            Dictionary with calibration metrics
        """
        # Calculate z-scores
        z_scores = np.abs(y_true - y_pred) / uncertainty
        
        # Expected vs observed coverage
        coverage_levels = np.linspace(0.1, 0.9, 9)
        expected_coverage = []
        observed_coverage = []
        
        for level in coverage_levels:
            z_critical = stats.norm.ppf((1 + level) / 2)
            expected_coverage.append(level)
            observed_coverage.append(np.mean(z_scores <= z_critical))
        
        # Calibration error
        calibration_error = np.mean(np.abs(np.array(expected_coverage) - np.array(observed_coverage)))
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainty)
        
        # Interval score
        alpha = 0.05
        z_critical = stats.norm.ppf(1 - alpha/2)
        lower = y_pred - z_critical * uncertainty
        upper = y_pred + z_critical * uncertainty
        
        interval_score = (upper - lower) + (2/alpha) * (lower - y_true) * (y_true < lower) + \
                        (2/alpha) * (y_true - upper) * (y_true > upper)
        mean_interval_score = np.mean(interval_score)
        
        return {
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'mean_interval_score': mean_interval_score,
            'expected_coverage': expected_coverage,
            'observed_coverage': observed_coverage
        }


class ModelComparisonFramework:
    """Framework for comparing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, predictions: np.ndarray, y_true: np.ndarray):
        """Add a model's predictions for comparison"""
        self.models[name] = {
            'predictions': predictions,
            'y_true': y_true,
            'errors': np.abs(y_true - predictions)
        }
    
    def compare_models(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Compare all models with statistical tests
        
        Returns:
            DataFrame with comparison results
        """
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # Calculate metrics for each model
        metrics_data = []
        for name in model_names:
            model_data = self.models[name]
            y_true = model_data['y_true']
            y_pred = model_data['predictions']
            
            metrics = {
                'Model': name,
                'R2': r2_score(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'Mean_Error': np.mean(model_data['errors'])
            }
            metrics_data.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Pairwise comparisons
        comparison_results = []
        sig_tester = StatisticalSignificanceTesting()
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1 = model_names[i]
                model2 = model_names[j]
                
                errors1 = self.models[model1]['errors']
                errors2 = self.models[model2]['errors']
                
                # Paired t-test
                t_stat, t_pval = sig_tester.paired_t_test(errors1, errors2)
                
                # Wilcoxon test
                w_stat, w_pval = sig_tester.wilcoxon_signed_rank_test(errors1, errors2)
                
                # Diebold-Mariano test
                dm_stat, dm_pval = sig_tester.diebold_mariano_test(errors1, errors2)
                
                comparison_results.append({
                    'Model1': model1,
                    'Model2': model2,
                    'T-test_pval': t_pval,
                    'Wilcoxon_pval': w_pval,
                    'DM_pval': dm_pval,
                    'Significant_diff': (t_pval < significance_level) or (dm_pval < significance_level)
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Apply multiple comparison correction
        p_values = comparison_df['T-test_pval'].tolist()
        corrected_p = sig_tester.multiple_comparison_correction(p_values, method='fdr')
        comparison_df['T-test_pval_corrected'] = corrected_p
        
        return metrics_df, comparison_df
    
    def rank_models(self) -> pd.DataFrame:
        """Rank models based on multiple criteria"""
        rankings = []
        
        for name in self.models:
            y_true = self.models[name]['y_true']
            y_pred = self.models[name]['predictions']
            
            # Calculate various metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate AIC and BIC approximations
            n = len(y_true)
            mse = mean_squared_error(y_true, y_pred)
            
            # Approximate number of parameters (would need actual model for exact count)
            k_approx = 10  # Placeholder
            
            aic = n * np.log(mse) + 2 * k_approx
            bic = n * np.log(mse) + k_approx * np.log(n)
            
            rankings.append({
                'Model': name,
                'R2_rank': -r2,  # Negative for sorting (higher is better)
                'MAE_rank': mae,  # Lower is better
                'RMSE_rank': rmse,  # Lower is better
                'AIC': aic,
                'BIC': bic
            })
        
        ranking_df = pd.DataFrame(rankings)
        
        # Add overall rank
        rank_columns = ['R2_rank', 'MAE_rank', 'RMSE_rank', 'AIC', 'BIC']
        for col in rank_columns:
            ranking_df[f'{col}_position'] = ranking_df[col].rank()
        
        # Calculate mean rank
        position_columns = [col for col in ranking_df.columns if '_position' in col]
        ranking_df['Mean_Rank'] = ranking_df[position_columns].mean(axis=1)
        ranking_df['Overall_Rank'] = ranking_df['Mean_Rank'].rank()
        
        return ranking_df.sort_values('Overall_Rank')


def generate_statistical_report(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               X: Optional[np.ndarray] = None,
                               model_name: str = "Model") -> Dict[str, Any]:
    """
    Generate comprehensive statistical report
    
    Args:
        y_true: True values
        y_pred: Predictions
        X: Feature matrix (optional)
        model_name: Name of the model
        
    Returns:
        Dictionary with comprehensive statistical analysis
    """
    report = {
        'model_name': model_name,
        'n_samples': len(y_true)
    }
    
    # Basic metrics
    report['metrics'] = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }
    
    # Residual analysis
    residual_analyzer = ResidualAnalysis()
    residuals = residual_analyzer.calculate_residuals(y_true, y_pred)
    
    report['residual_analysis'] = {
        'normality': residual_analyzer.test_normality(residuals),
        'homoscedasticity': residual_analyzer.test_homoscedasticity(residuals, y_pred),
        'autocorrelation': residual_analyzer.test_autocorrelation(residuals)
    }
    
    # Influence statistics if X is provided
    if X is not None:
        report['influence'] = residual_analyzer.calculate_influence_statistics(X, y_true, y_pred)
    
    return report


# Example usage function
def example_usage():
    """Example of how to use the statistical enhancements"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y_true = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    y_pred = y_true + np.random.randn(n_samples) * 0.3
    
    # Bootstrap confidence intervals
    print("Calculating bootstrap confidence intervals...")
    bootstrap = BootstrapConfidenceIntervals(n_bootstrap=100)
    metric_ci = bootstrap.calculate_metric_confidence_intervals(y_true, y_pred)
    
    for metric, (value, lower, upper) in metric_ci.items():
        print(f"{metric}: {value:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
    
    # Statistical significance testing
    print("\nStatistical significance testing...")
    sig_tester = StatisticalSignificanceTesting()
    
    # Compare two models
    errors1 = np.abs(y_true - y_pred)
    errors2 = np.abs(y_true - (y_pred + np.random.randn(n_samples) * 0.1))
    
    t_stat, p_val = sig_tester.paired_t_test(errors1, errors2)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.3f}")
    
    # Residual analysis
    print("\nResidual analysis...")
    residual_analyzer = ResidualAnalysis()
    residuals = residual_analyzer.calculate_residuals(y_true, y_pred)
    
    normality_results = residual_analyzer.test_normality(residuals)
    print(f"Shapiro-Wilk p-value: {normality_results.get('shapiro_wilk', {}).get('p_value', 'N/A'):.3f}")
    print(f"Skewness: {normality_results['skewness']:.3f}")
    print(f"Kurtosis: {normality_results['kurtosis']:.3f}")
    
    # Generate statistical report
    print("\nGenerating statistical report...")
    report = generate_statistical_report(y_true, y_pred, X, "Example Model")
    print(f"R²: {report['metrics']['r2']:.3f}")
    print(f"RMSE: {report['metrics']['rmse']:.3f}")


if __name__ == "__main__":
    example_usage()