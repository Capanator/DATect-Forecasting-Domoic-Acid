"""
Model Comparison Module
======================

Provides comprehensive model performance comparison between XGBoost and traditional statistical methods.
Shows the scientific value of modern ML approaches over conventional statistical forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
from typing import Dict, List, Any, Tuple
import logging

from .statistical_enhancements import StatisticalEnhancer
from .logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

class ModelComparator:
    """Compares XGBoost performance against traditional statistical baselines."""
    
    def __init__(self):
        self.statistical_enhancer = StatisticalEnhancer()
    
    def run_comprehensive_comparison(self, results_df: pd.DataFrame, 
                                   include_xgb: bool = True, 
                                   include_linear: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive model comparison analysis on retrospective results.
        
        Args:
            results_df: DataFrame with predictions from multiple models
            include_xgb: Whether to include XGBoost results
            include_linear: Whether to include linear/logistic results
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        logger.info("Running comprehensive model comparison analysis")
        
        comparison_results = {
            'overall_summary': {},
            'model_performance': {},
            'statistical_tests': {},
            'site_level_analysis': {},
            'temporal_analysis': {},
            'improvement_metrics': {}
        }
        
        try:
            # Filter to valid predictions only
            valid_data = results_df.dropna(subset=['da']).copy()
            
            if len(valid_data) == 0:
                logger.warning("No valid data for model comparison")
                return comparison_results
            
            # Identify available model columns
            model_columns = self._identify_model_columns(valid_data)
            logger.info(f"Found model columns: {model_columns}")
            
            # Overall performance comparison
            comparison_results['overall_summary'] = self._compute_overall_performance(
                valid_data, model_columns
            )
            
            # Statistical significance tests
            comparison_results['statistical_tests'] = self._run_statistical_tests(
                valid_data, model_columns
            )
            
            # Site-level breakdown
            comparison_results['site_level_analysis'] = self._analyze_by_site(
                valid_data, model_columns
            )
            
            # Temporal analysis
            comparison_results['temporal_analysis'] = self._analyze_temporal_performance(
                valid_data, model_columns
            )
            
            # Improvement metrics
            comparison_results['improvement_metrics'] = self._calculate_improvement_metrics(
                valid_data, model_columns
            )
            
            # Generate summary insights
            comparison_results['insights'] = self._generate_insights(comparison_results)
            
            logger.info("Model comparison analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive comparison: {e}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def _identify_model_columns(self, data: pd.DataFrame) -> Dict[str, str]:
        """Identify columns containing model predictions."""
        model_cols = {}
        
        # Common prediction column patterns
        if 'Predicted_da' in data.columns:
            model_cols['xgboost_regression'] = 'Predicted_da'
        if 'linear_predicted_da' in data.columns:
            model_cols['linear_regression'] = 'linear_predicted_da'
        if 'Predicted_da-category' in data.columns:
            model_cols['xgboost_classification'] = 'Predicted_da-category'
        if 'logistic_predicted_category' in data.columns:
            model_cols['logistic_classification'] = 'logistic_predicted_category'
            
        return model_cols
    
    def _compute_overall_performance(self, data: pd.DataFrame, 
                                   model_columns: Dict[str, str]) -> Dict[str, Any]:
        """Compute overall performance metrics for each model."""
        performance = {}
        
        for model_name, pred_col in model_columns.items():
            if pred_col not in data.columns:
                continue
                
            valid_preds = data.dropna(subset=[pred_col, 'da'])
            
            if len(valid_preds) == 0:
                continue
            
            if 'regression' in model_name:
                # Regression metrics
                y_true = valid_preds['da']
                y_pred = valid_preds[pred_col]
                
                performance[model_name] = {
                    'n_predictions': len(valid_preds),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2': r2_score(y_true, y_pred),
                    'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                    'median_ae': np.median(np.abs(y_true - y_pred)),
                    'mean_prediction': np.mean(y_pred),
                    'std_prediction': np.std(y_pred)
                }
                
            elif 'classification' in model_name:
                # Classification metrics
                if 'da-category' in data.columns:
                    y_true = valid_preds['da-category']
                    y_pred = valid_preds[pred_col]
                    
                    performance[model_name] = {
                        'n_predictions': len(valid_preds),
                        'accuracy': accuracy_score(y_true, y_pred),
                        'class_distribution': y_pred.value_counts().to_dict()
                    }
        
        return performance
    
    def _run_statistical_tests(self, data: pd.DataFrame, 
                             model_columns: Dict[str, str]) -> Dict[str, Any]:
        """Run statistical significance tests between models."""
        tests = {}
        
        # Compare XGBoost vs Linear for regression
        if ('xgboost_regression' in model_columns and 
            'linear_regression' in model_columns):
            
            xgb_col = model_columns['xgboost_regression']
            lin_col = model_columns['linear_regression']
            
            valid_data = data.dropna(subset=['da', xgb_col, lin_col])
            
            if len(valid_data) > 10:  # Minimum for statistical testing
                y_true = valid_data['da'].values
                y_xgb = valid_data[xgb_col].values
                y_lin = valid_data[lin_col].values
                
                test_result = self.statistical_enhancer.model_comparison_test(
                    y_true, y_xgb, y_lin, "XGBoost", "Linear Regression"
                )
                tests['xgboost_vs_linear_regression'] = test_result
        
        # Compare XGBoost vs Logistic for classification
        if ('xgboost_classification' in model_columns and 
            'logistic_classification' in model_columns):
            
            xgb_col = model_columns['xgboost_classification']
            log_col = model_columns['logistic_classification']
            
            valid_data = data.dropna(subset=['da-category', xgb_col, log_col])
            
            if len(valid_data) > 10:
                # For classification, compare accuracy directly
                y_true = valid_data['da-category'].values
                xgb_acc = accuracy_score(y_true, valid_data[xgb_col].values)
                log_acc = accuracy_score(y_true, valid_data[log_col].values)
                
                tests['xgboost_vs_logistic_classification'] = {
                    'xgboost_accuracy': xgb_acc,
                    'logistic_accuracy': log_acc,
                    'improvement': (xgb_acc - log_acc) / log_acc * 100 if log_acc > 0 else 0,
                    'n_samples': len(valid_data)
                }
        
        return tests
    
    def _analyze_by_site(self, data: pd.DataFrame, 
                        model_columns: Dict[str, str]) -> Dict[str, Any]:
        """Analyze model performance by site."""
        site_analysis = {}
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site]
            site_analysis[site] = {}
            
            for model_name, pred_col in model_columns.items():
                if pred_col not in site_data.columns:
                    continue
                    
                valid_site_data = site_data.dropna(subset=[pred_col, 'da'])
                
                if len(valid_site_data) < 3:  # Minimum for meaningful metrics
                    continue
                
                if 'regression' in model_name:
                    y_true = valid_site_data['da']
                    y_pred = valid_site_data[pred_col]
                    
                    site_analysis[site][model_name] = {
                        'n_predictions': len(valid_site_data),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'r2': r2_score(y_true, y_pred)
                    }
        
        return site_analysis
    
    def _analyze_temporal_performance(self, data: pd.DataFrame, 
                                    model_columns: Dict[str, str]) -> Dict[str, Any]:
        """Analyze how model performance changes over time."""
        temporal_analysis = {}
        
        if 'date' not in data.columns:
            return temporal_analysis
        
        # Convert date column to datetime if needed
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Analyze by year
        data['year'] = data['date'].dt.year
        
        for year in sorted(data['year'].unique()):
            year_data = data[data['year'] == year]
            temporal_analysis[str(year)] = {}
            
            for model_name, pred_col in model_columns.items():
                if pred_col not in year_data.columns:
                    continue
                    
                valid_year_data = year_data.dropna(subset=[pred_col, 'da'])
                
                if len(valid_year_data) < 3:
                    continue
                
                if 'regression' in model_name:
                    y_true = valid_year_data['da']
                    y_pred = valid_year_data[pred_col]
                    
                    temporal_analysis[str(year)][model_name] = {
                        'n_predictions': len(valid_year_data),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred)
                    }
        
        return temporal_analysis
    
    def _calculate_improvement_metrics(self, data: pd.DataFrame, 
                                     model_columns: Dict[str, str]) -> Dict[str, Any]:
        """Calculate improvement of XGBoost over traditional methods."""
        improvements = {}
        
        # XGBoost vs Linear Regression
        if ('xgboost_regression' in model_columns and 
            'linear_regression' in model_columns):
            
            xgb_col = model_columns['xgboost_regression']
            lin_col = model_columns['linear_regression']
            
            valid_data = data.dropna(subset=['da', xgb_col, lin_col])
            
            if len(valid_data) > 0:
                y_true = valid_data['da']
                
                xgb_mae = mean_absolute_error(y_true, valid_data[xgb_col])
                lin_mae = mean_absolute_error(y_true, valid_data[lin_col])
                
                xgb_r2 = r2_score(y_true, valid_data[xgb_col])
                lin_r2 = r2_score(y_true, valid_data[lin_col])
                
                improvements['xgboost_vs_linear'] = {
                    'mae_improvement_pct': ((lin_mae - xgb_mae) / lin_mae) * 100 if lin_mae > 0 else 0,
                    'r2_improvement': xgb_r2 - lin_r2,
                    'xgboost_mae': xgb_mae,
                    'linear_mae': lin_mae,
                    'xgboost_r2': xgb_r2,
                    'linear_r2': lin_r2
                }
        
        return improvements
    
    def _generate_insights(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from comparison results."""
        insights = []
        
        try:
            # Overall performance insights
            performance = comparison_results.get('overall_summary', {})
            improvements = comparison_results.get('improvement_metrics', {})
            
            if 'xgboost_vs_linear' in improvements:
                mae_improvement = improvements['xgboost_vs_linear']['mae_improvement_pct']
                r2_improvement = improvements['xgboost_vs_linear']['r2_improvement']
                
                if mae_improvement > 0:
                    insights.append(f"XGBoost reduces prediction error by {mae_improvement:.1f}% compared to linear regression")
                
                if r2_improvement > 0:
                    insights.append(f"XGBoost achieves {r2_improvement:.3f} higher R² score than linear regression")
            
            # Statistical significance insights
            tests = comparison_results.get('statistical_tests', {})
            if 'xgboost_vs_linear_regression' in tests:
                test_result = tests['xgboost_vs_linear_regression']
                if test_result.get('significant_improvement', False):
                    insights.append("XGBoost shows statistically significant improvement over linear regression")
            
            # Site-level insights
            site_analysis = comparison_results.get('site_level_analysis', {})
            if site_analysis:
                best_sites = []
                for site, site_data in site_analysis.items():
                    if 'xgboost_regression' in site_data and 'linear_regression' in site_data:
                        xgb_r2 = site_data['xgboost_regression'].get('r2', 0)
                        if xgb_r2 > 0.5:  # Good performance threshold
                            best_sites.append(site)
                
                if best_sites:
                    insights.append(f"XGBoost performs particularly well at {', '.join(best_sites[:3])}")
            
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
            insights.append("Model comparison completed - see detailed results for analysis")
        
        return insights
    
    def create_comparison_visualization_data(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create data structure for frontend visualization of model comparison."""
        
        viz_data = {
            'summary_chart': {},
            'site_comparison': {},
            'temporal_trends': {},
            'improvement_metrics': {},
            'insights': comparison_results.get('insights', [])
        }
        
        try:
            # Summary performance chart data
            performance = comparison_results.get('overall_summary', {})
            
            models = []
            mae_values = []
            r2_values = []
            
            for model_name, metrics in performance.items():
                if 'regression' in model_name:
                    models.append(model_name.replace('_', ' ').title())
                    mae_values.append(metrics.get('mae', 0))
                    r2_values.append(metrics.get('r2', 0))
            
            viz_data['summary_chart'] = {
                'models': models,
                'mae_values': mae_values,
                'r2_values': r2_values
            }
            
            # Site-level comparison data
            site_analysis = comparison_results.get('site_level_analysis', {})
            viz_data['site_comparison'] = site_analysis
            
            # Improvement metrics for display
            improvements = comparison_results.get('improvement_metrics', {})
            viz_data['improvement_metrics'] = improvements
            
        except Exception as e:
            logger.error(f"Error creating visualization data: {e}")
        
        return viz_data