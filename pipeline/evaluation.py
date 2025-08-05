"""
Model evaluation metrics and validation without leakage.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, 
    classification_report, confusion_matrix
)
from typing import Dict, List, Any, Optional


class RegressionEvaluator:
    """Evaluates regression model performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           prefix: str = "") -> Dict[str, float]:
        """Calculate regression metrics."""
        # Remove NaN values for evaluation
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return {f"{prefix}r2": np.nan, f"{prefix}mae": np.nan, f"{prefix}n_samples": 0}
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        metrics = {
            f"{prefix}r2": r2_score(y_true_clean, y_pred_clean),
            f"{prefix}mae": mean_absolute_error(y_true_clean, y_pred_clean),
            f"{prefix}n_samples": len(y_true_clean)
        }
        
        return metrics
    
    def evaluate_quantile_coverage(self, y_true: np.ndarray, 
                                 quantile_preds: Dict[str, np.ndarray],
                                 coverage_level: float = 0.9) -> Dict[str, float]:
        """Evaluate quantile prediction coverage."""
        if 'q05' not in quantile_preds or 'q95' not in quantile_preds:
            return {"coverage": np.nan, "n_samples": 0}
        
        lower = quantile_preds['q05']
        upper = quantile_preds['q95']
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
        if mask.sum() == 0:
            return {"coverage": np.nan, "n_samples": 0}
        
        y_true_clean = y_true[mask]
        lower_clean = lower[mask]
        upper_clean = upper[mask]
        
        # Calculate coverage
        within_interval = (y_true_clean >= lower_clean) & (y_true_clean <= upper_clean)
        coverage = within_interval.mean()
        
        return {
            "coverage": coverage,
            "n_samples": len(y_true_clean),
            "expected_coverage": coverage_level
        }


class ClassificationEvaluator:
    """Evaluates classification model performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           prefix: str = "") -> Dict[str, float]:
        """Calculate classification metrics."""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return {f"{prefix}accuracy": np.nan, f"{prefix}n_samples": 0}
        
        y_true_clean = y_true[mask].astype(int)
        y_pred_clean = y_pred[mask].astype(int)
        
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true_clean, y_pred_clean),
            f"{prefix}n_samples": len(y_true_clean)
        }
        
        return metrics
    
    def detailed_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return "No valid predictions for detailed report"
        
        y_true_clean = y_true[mask].astype(int)
        y_pred_clean = y_pred[mask].astype(int)
        
        return classification_report(y_true_clean, y_pred_clean)


class ForecastEvaluator:
    """Comprehensive forecast evaluation."""
    
    def __init__(self):
        self.regression_evaluator = RegressionEvaluator()
        self.classification_evaluator = ClassificationEvaluator()
    
    def evaluate_forecast_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate comprehensive forecast results."""
        evaluation = {}
        
        # Regression evaluation
        if 'Actual_da' in results_df.columns and 'Predicted_da' in results_df.columns:
            reg_metrics = self.regression_evaluator.evaluate_predictions(
                results_df['Actual_da'].values,
                results_df['Predicted_da'].values,
                prefix="regression_"
            )
            evaluation.update(reg_metrics)
        
        # Classification evaluation
        if 'Actual_da_category' in results_df.columns and 'Predicted_da_category' in results_df.columns:
            cls_metrics = self.classification_evaluator.evaluate_predictions(
                results_df['Actual_da_category'].values,
                results_df['Predicted_da_category'].values,
                prefix="classification_"
            )
            evaluation.update(cls_metrics)
        
        # Quantile evaluation
        quantile_cols = [col for col in results_df.columns if col.startswith('Predicted_da_q')]
        if len(quantile_cols) >= 2 and 'Actual_da' in results_df.columns:
            quantile_preds = {}
            for col in quantile_cols:
                quantile_name = col.replace('Predicted_da_', '')
                quantile_preds[quantile_name] = results_df[col].values
            
            coverage_metrics = self.regression_evaluator.evaluate_quantile_coverage(
                results_df['Actual_da'].values,
                quantile_preds
            )
            evaluation.update({f"quantile_{k}": v for k, v in coverage_metrics.items()})
        
        return evaluation
    
    def evaluate_by_site(self, results_df: pd.DataFrame, site_col: str = 'site') -> pd.DataFrame:
        """Evaluate metrics by site."""
        site_metrics = []
        
        for site in results_df[site_col].unique():
            site_data = results_df[results_df[site_col] == site]
            site_eval = self.evaluate_forecast_results(site_data)
            site_eval['site'] = site
            site_metrics.append(site_eval)
        
        return pd.DataFrame(site_metrics)
    
    def print_evaluation_summary(self, evaluation: Dict[str, Any], title: str = "Evaluation Summary"):
        """Print formatted evaluation summary."""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        # Regression metrics
        if 'regression_r2' in evaluation:
            print(f"Regression R²: {evaluation['regression_r2']:.4f}")
            print(f"Regression MAE: {evaluation['regression_mae']:.4f}")
            print(f"Regression Samples: {evaluation['regression_n_samples']}")
        
        # Classification metrics
        if 'classification_accuracy' in evaluation:
            print(f"Classification Accuracy: {evaluation['classification_accuracy']:.4f}")
            print(f"Classification Samples: {evaluation['classification_n_samples']}")
        
        # Quantile metrics
        if 'quantile_coverage' in evaluation:
            print(f"Quantile Coverage: {evaluation['quantile_coverage']:.4f}")
            print(f"Expected Coverage: {evaluation.get('quantile_expected_coverage', 0.9):.2f}")
            print(f"Quantile Samples: {evaluation['quantile_n_samples']}")
        
        print(f"{'='*50}\n")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_evaluation_data(results_df: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate that evaluation data contains required columns."""
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns for evaluation: {missing_cols}")
    
    if results_df.empty:
        raise ValidationError("Results dataframe is empty")
    
    print(f"Validation passed: {len(results_df)} samples ready for evaluation")