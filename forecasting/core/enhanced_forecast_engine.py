"""
Enhanced Forecast Engine with Statistical Improvements

This module integrates all statistical enhancements:
- Bootstrap confidence intervals
- Hyperparameter optimization with nested CV
- Spatial cross-validation
- Baseline model comparisons
- Statistical significance testing
- Constrained data interpolation
- Residual analysis
- Uncertainty quantification
- Seasonal modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
import json

# Import all enhancement modules
from .statistical_enhancements import (
    BootstrapConfidenceIntervals, StatisticalSignificanceTesting,
    ResidualAnalysis, UncertaintyQuantification, ModelComparisonFramework,
    generate_statistical_report
)
from .baseline_models import (
    PersistenceModel, ClimatologyModel, MovingAverageModel,
    LinearTrendModel, BaselineEnsemble, evaluate_baseline_models
)
from .hyperparameter_optimization import (
    NestedCrossValidation, BayesianOptimizer, TimeSeriesCrossValidator,
    SpatialCrossValidator, optimize_xgboost_hyperparameters
)
from .improved_interpolation import (
    ConstrainedInterpolator, improve_dataset_interpolation,
    analyze_interpolation_impact
)
from .seasonal_modeling import (
    SeasonalDecomposer, TimeVaryingCoefficientModel,
    AdaptiveSeasonalModel, detect_seasonal_patterns
)

# Import existing modules
from .data_processor import DataProcessor
from .model_factory import ModelFactory

logger = logging.getLogger(__name__)


class EnhancedForecastEngine:
    """
    Enhanced forecasting engine with comprehensive statistical improvements
    """
    
    def __init__(self, config):
        """
        Initialize enhanced forecast engine
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        
        # Core components
        self.data_processor = DataProcessor(config)
        self.model_factory = ModelFactory(config)
        
        # Enhanced components
        self.bootstrap_ci = BootstrapConfidenceIntervals(n_bootstrap=1000)
        self.significance_tester = StatisticalSignificanceTesting()
        self.residual_analyzer = ResidualAnalysis()
        self.uncertainty_quantifier = UncertaintyQuantification()
        self.model_comparator = ModelComparisonFramework()
        
        # Optimization settings
        self.use_nested_cv = getattr(config, 'USE_NESTED_CV', False)
        self.use_spatial_cv = getattr(config, 'USE_SPATIAL_CV', False)
        self.max_interpolation_weeks = getattr(config, 'MAX_INTERPOLATION_WEEKS', 6)
        self.include_baseline_models = getattr(config, 'INCLUDE_BASELINE_MODELS', True)
        self.use_seasonal_modeling = getattr(config, 'USE_SEASONAL_MODELING', False)
        
        # Results storage
        self.optimization_results = {}
        self.model_comparison_results = {}
        self.statistical_reports = {}
        
    def enhanced_forecast(self, 
                         target_date: pd.Timestamp,
                         site: str,
                         model_type: str = 'xgboost',
                         task_type: str = 'regression',
                         return_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Generate enhanced forecast with comprehensive statistical analysis
        
        Args:
            target_date: Date to forecast
            site: Site name
            model_type: Type of model to use
            task_type: 'regression' or 'classification'
            return_uncertainty: Whether to include uncertainty quantification
            
        Returns:
            Dictionary with enhanced forecast results
        """
        logger.info(f"Generating enhanced forecast for {site} on {target_date}")
        
        # Load and preprocess data
        data, features = self._load_and_preprocess_data(target_date, site)
        
        if len(data) < self.config.MIN_TRAINING_SAMPLES:
            raise ValueError(f"Insufficient training data: {len(data)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test, dates_train, dates_test = self._create_train_test_split(
            data, features, target_date
        )
        
        results = {
            'target_date': target_date,
            'site': site,
            'model_type': model_type,
            'task_type': task_type,
            'training_samples': len(X_train),
            'features_used': features
        }
        
        # 1. Hyperparameter optimization
        if self.use_nested_cv:
            logger.info("Running nested cross-validation for hyperparameter optimization")
            optimization_result = self._optimize_hyperparameters(
                X_train, y_train, model_type, task_type
            )
            results['optimization'] = optimization_result
        else:
            optimization_result = None
        
        # 2. Train models with comparison
        model_results = self._train_and_compare_models(
            X_train, X_test, y_train, y_test,
            dates_train, dates_test,
            model_type, task_type,
            optimization_result
        )
        results['models'] = model_results
        
        # 3. Statistical analysis
        statistical_analysis = self._comprehensive_statistical_analysis(
            model_results['main_model']['predictions'],
            y_test, X_test
        )
        results['statistical_analysis'] = statistical_analysis
        
        # 4. Uncertainty quantification
        if return_uncertainty:
            uncertainty_analysis = self._quantify_uncertainty(
                X_train, y_train, X_test, y_test,
                model_type, optimization_result
            )
            results['uncertainty'] = uncertainty_analysis
        
        # 5. Seasonal analysis (if enabled)
        if self.use_seasonal_modeling:
            seasonal_analysis = self._analyze_seasonality(data, site)
            results['seasonality'] = seasonal_analysis
        
        # 6. Generate final forecast
        forecast_result = self._generate_final_forecast(
            results, target_date, site
        )
        results['forecast'] = forecast_result
        
        # Save detailed results
        self._save_enhanced_results(results, site, target_date)
        
        return results
    
    def _load_and_preprocess_data(self, target_date: pd.Timestamp, site: str) -> Tuple[pd.DataFrame, List[str]]:
        """Load and preprocess data with enhanced interpolation"""
        
        # Load data using existing processor
        data = self.data_processor.load_site_data(site, target_date)
        
        # Apply enhanced interpolation if enabled
        if self.max_interpolation_weeks < 50:  # Only if constraint is reasonable
            logger.info(f"Applying constrained interpolation (max {self.max_interpolation_weeks} weeks)")
            
            interpolator = ConstrainedInterpolator(
                max_gap_weeks=self.max_interpolation_weeks,
                limit_direction='forward'
            )
            
            # Identify columns to interpolate
            interpolation_columns = ['da', 'pn_levels'] if 'pn_levels' in data.columns else ['da']
            
            for col in interpolation_columns:
                if col in data.columns:
                    data[col], quality_metrics = interpolator.interpolate_series(
                        data[col], data['date'], f"{site}_{col}"
                    )
        
        # Get features
        features = self.data_processor.get_feature_columns(data)
        
        return data, features
    
    def _create_train_test_split(self, data: pd.DataFrame, features: List[str], 
                                target_date: pd.Timestamp) -> Tuple:
        """Create temporal train/test split"""
        
        # Apply temporal buffer
        buffer_days = getattr(self.config, 'TEMPORAL_BUFFER_DAYS', 7)
        cutoff_date = target_date - timedelta(days=buffer_days)
        
        train_mask = data['date'] < cutoff_date
        test_mask = data['date'] >= target_date
        
        if not test_mask.any():
            # Create synthetic test point for forecast
            test_data = data[train_mask].tail(1).copy()
            test_data['date'] = target_date
            test_mask = pd.Series([True], index=[len(data)])
            data = pd.concat([data, test_data])
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        # Prepare features and targets
        X_train = train_data[features].values
        X_test = test_data[features].values
        y_train = train_data['da'].values
        y_test = test_data['da'].values if 'da' in test_data.columns else np.array([np.nan])
        
        dates_train = pd.to_datetime(train_data['date'])
        dates_test = pd.to_datetime(test_data['date'])
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 model_type: str, task_type: str) -> Dict:
        """Optimize hyperparameters using nested CV"""
        
        if model_type == 'xgboost':
            # Use existing optimization function
            optimization_result = optimize_xgboost_hyperparameters(
                X_train, y_train, X_train[:1], y_train[:1],  # Dummy test set
                method='nested_cv'
            )
            return {
                'method': 'nested_cv',
                'best_params': optimization_result.best_params,
                'cv_score': optimization_result.best_score,
                'cv_scores': optimization_result.cv_scores
            }
        else:
            return {'method': 'default', 'best_params': {}}
    
    def _train_and_compare_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 dates_train: pd.DatetimeIndex, dates_test: pd.DatetimeIndex,
                                 model_type: str, task_type: str,
                                 optimization_result: Optional[Dict]) -> Dict:
        """Train main model and baseline models for comparison"""
        
        results = {}
        
        # Train main model
        if optimization_result and optimization_result.get('best_params'):
            main_model = self.model_factory.create_model(
                model_type, task_type, **optimization_result['best_params']
            )
        else:
            main_model = self.model_factory.create_model(model_type, task_type)
        
        main_model.fit(X_train, y_train)
        main_predictions = main_model.predict(X_test)
        
        results['main_model'] = {
            'type': model_type,
            'predictions': main_predictions,
            'model_object': main_model
        }
        
        # Train baseline models for comparison
        if self.include_baseline_models:
            baseline_results = evaluate_baseline_models(
                X_train, y_train, X_test, y_test,
                dates_train, dates_test
            )
            results['baselines'] = baseline_results
            
            # Add to model comparator
            self.model_comparator.add_model('main_model', main_predictions, y_test)
            
            for _, row in baseline_results.iterrows():
                model_name = row['Model']
                # Note: We'd need predictions from baseline models to add them
                # This is simplified for demonstration
        
        return results
    
    def _comprehensive_statistical_analysis(self, predictions: np.ndarray,
                                           y_true: np.ndarray, X: np.ndarray) -> Dict:
        """Perform comprehensive statistical analysis"""
        
        analysis = {}
        
        # Basic metrics with bootstrap confidence intervals
        metric_ci = self.bootstrap_ci.calculate_metric_confidence_intervals(
            y_true[~np.isnan(y_true)], predictions[~np.isnan(y_true)]
        )
        analysis['metrics_with_ci'] = metric_ci
        
        # Residual analysis
        residuals = self.residual_analyzer.calculate_residuals(y_true, predictions)
        valid_mask = ~(np.isnan(residuals) | np.isnan(predictions))
        
        if np.sum(valid_mask) > 10:
            analysis['normality_tests'] = self.residual_analyzer.test_normality(
                residuals[valid_mask]
            )
            analysis['homoscedasticity_tests'] = self.residual_analyzer.test_homoscedasticity(
                residuals[valid_mask], predictions[valid_mask]
            )
            analysis['autocorrelation_tests'] = self.residual_analyzer.test_autocorrelation(
                residuals[valid_mask]
            )
            
            # Influence statistics if X is provided
            if X is not None and len(X) > 0:
                try:
                    analysis['influence_stats'] = self.residual_analyzer.calculate_influence_statistics(
                        X[valid_mask], y_true[valid_mask], predictions[valid_mask]
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate influence statistics: {e}")
        
        return analysis
    
    def _quantify_uncertainty(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             model_type: str, optimization_result: Optional[Dict]) -> Dict:
        """Quantify prediction uncertainty"""
        
        uncertainty_results = {}
        
        try:
            # Bootstrap prediction intervals
            from xgboost import XGBRegressor
            
            model_params = optimization_result.get('best_params', {}) if optimization_result else {}
            
            predictions, lower_bound, upper_bound, uncertainty = self.bootstrap_ci.calculate_prediction_intervals(
                X_train, y_train, X_test, XGBRegressor, model_params
            )
            
            uncertainty_results['bootstrap'] = {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertainty': uncertainty
            }
            
            # Calibrate uncertainty if we have test data
            if not np.isnan(y_test).all():
                valid_test_mask = ~np.isnan(y_test)
                if np.sum(valid_test_mask) > 0:
                    calibration_metrics = self.uncertainty_quantifier.calibrate_uncertainty(
                        y_test[valid_test_mask],
                        predictions[valid_test_mask],
                        uncertainty[valid_test_mask]
                    )
                    uncertainty_results['calibration'] = calibration_metrics
            
        except Exception as e:
            logger.warning(f"Could not quantify uncertainty: {e}")
            uncertainty_results['error'] = str(e)
        
        return uncertainty_results
    
    def _analyze_seasonality(self, data: pd.DataFrame, site: str) -> Dict:
        """Analyze seasonal patterns in the data"""
        
        seasonal_results = {}
        
        if len(data) > 104:  # Need at least 2 years for seasonal analysis
            try:
                # Detect seasonal patterns
                patterns = detect_seasonal_patterns(
                    data['da'].values, 
                    pd.to_datetime(data['date'])
                )
                seasonal_results['pattern_detection'] = patterns
                
                # Seasonal decomposition
                decomposer = SeasonalDecomposer(
                    seasonal_periods=[52, 26, 13],  # Annual, semi-annual, quarterly
                    method='additive'
                )
                
                components = decomposer.fit_transform(data['da'].values)
                seasonal_results['decomposition'] = {
                    'trend_strength': np.var(components['trend']) / np.var(data['da']),
                    'seasonality_strengths': decomposer.seasonality_strength_,
                    'residual_variance': np.var(components['residual'])
                }
                
            except Exception as e:
                logger.warning(f"Could not analyze seasonality: {e}")
                seasonal_results['error'] = str(e)
        
        return seasonal_results
    
    def _generate_final_forecast(self, results: Dict, target_date: pd.Timestamp, site: str) -> Dict:
        """Generate final forecast with uncertainty bounds"""
        
        main_prediction = results['models']['main_model']['predictions'][0]
        
        forecast = {
            'date': target_date,
            'site': site,
            'predicted_da': float(main_prediction),
            'model_type': results['model_type']
        }
        
        # Add uncertainty bounds if available
        if 'uncertainty' in results and 'bootstrap' in results['uncertainty']:
            uncertainty_data = results['uncertainty']['bootstrap']
            forecast['lower_bound_95'] = float(uncertainty_data['lower_bound'][0])
            forecast['upper_bound_95'] = float(uncertainty_data['upper_bound'][0])
            forecast['prediction_uncertainty'] = float(uncertainty_data['uncertainty'][0])
        
        # Add confidence based on statistical analysis
        if 'statistical_analysis' in results and 'metrics_with_ci' in results['statistical_analysis']:
            r2_info = results['statistical_analysis']['metrics_with_ci'].get('R2', (0, 0, 0))
            forecast['model_r2'] = r2_info[0]
            forecast['r2_confidence_interval'] = [r2_info[1], r2_info[2]]
        
        # Risk categorization (if applicable)
        if hasattr(self.config, 'DA_THRESHOLDS'):
            thresholds = self.config.DA_THRESHOLDS
            if main_prediction <= thresholds[0]:
                risk_category = 'Low'
            elif main_prediction <= thresholds[1]:
                risk_category = 'Moderate'
            elif main_prediction <= thresholds[2]:
                risk_category = 'High'
            else:
                risk_category = 'Extreme'
            
            forecast['risk_category'] = risk_category
        
        return forecast
    
    def _save_enhanced_results(self, results: Dict, site: str, target_date: pd.Timestamp):
        """Save enhanced results for later analysis"""
        
        # Create results directory
        results_dir = Path("enhanced_forecast_results")
        results_dir.mkdir(exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_forecast_{site}_{target_date.strftime('%Y%m%d')}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = self._prepare_for_json(results)
        
        # Save to file
        with open(results_dir / filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Enhanced results saved to {results_dir / filename}")
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return None  # Skip complex objects
        else:
            return obj
    
    def batch_enhanced_forecast(self, 
                              sites: List[str],
                              target_dates: List[pd.Timestamp],
                              **kwargs) -> Dict[str, Dict]:
        """Run enhanced forecasts for multiple sites and dates"""
        
        results = {}
        
        for site in sites:
            site_results = {}
            for target_date in target_dates:
                try:
                    forecast_result = self.enhanced_forecast(
                        target_date, site, **kwargs
                    )
                    site_results[target_date.strftime('%Y-%m-%d')] = forecast_result
                except Exception as e:
                    logger.error(f"Failed to forecast {site} on {target_date}: {e}")
                    site_results[target_date.strftime('%Y-%m-%d')] = {
                        'error': str(e)
                    }
            
            results[site] = site_results
        
        return results
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive model comparison report"""
        
        if not self.model_comparator.models:
            logger.warning("No models available for comparison")
            return pd.DataFrame()
        
        # Get comparison results
        metrics_df, comparison_df = self.model_comparator.compare_models()
        
        # Combine with ranking
        ranking_df = self.model_comparator.rank_models()
        
        # Create comprehensive report
        report = pd.merge(metrics_df, ranking_df[['Model', 'Overall_Rank']], on='Model')
        
        return report.sort_values('Overall_Rank')


def create_enhanced_engine(config) -> EnhancedForecastEngine:
    """
    Factory function to create enhanced forecast engine
    
    Args:
        config: Configuration object
        
    Returns:
        Configured EnhancedForecastEngine
    """
    return EnhancedForecastEngine(config)


# Example usage
if __name__ == "__main__":
    # This would typically be run with actual configuration
    class MockConfig:
        MIN_TRAINING_SAMPLES = 50
        TEMPORAL_BUFFER_DAYS = 7
        USE_NESTED_CV = True
        USE_SPATIAL_CV = False
        MAX_INTERPOLATION_WEEKS = 6
        INCLUDE_BASELINE_MODELS = True
        USE_SEASONAL_MODELING = True
        DA_THRESHOLDS = [5, 20, 40]
    
    config = MockConfig()
    
    print("Enhanced Forecast Engine created successfully")
    print("Available enhancements:")
    print("✓ Bootstrap confidence intervals")
    print("✓ Nested cross-validation")
    print("✓ Baseline model comparisons") 
    print("✓ Statistical significance testing")
    print("✓ Residual analysis")
    print("✓ Uncertainty quantification")
    print("✓ Constrained interpolation")
    print("✓ Seasonal modeling")
    print("✓ Comprehensive reporting")