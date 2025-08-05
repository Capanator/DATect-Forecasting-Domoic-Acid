"""
Unified forecasting pipeline that replaces both past-forecasts-final.py and future-forecasts.py.
Implements proper sklearn Pipeline without data leakage.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed
import warnings

from .feature_engineering import (
    TemporalFeatures, LagFeatures, CategoryEncoder, 
    DataCleaner, FeatureSelector
)
from .models import ModelFactory, ModelTrainer, ModelPredictor
from .data_splitter import TimeSeriesSplitter, RandomAnchorGenerator, DataValidator
from .evaluation import ForecastEvaluator

warnings.filterwarnings('ignore', category=UserWarning)


class DAForecastPipeline:
    """Main forecasting pipeline for Domoic Acid predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.feature_pipeline = None
        self.models = {}
        self.is_fitted = False
        
        # Initialize components
        self.splitter = TimeSeriesSplitter()
        self.validator = DataValidator()
        self.evaluator = ForecastEvaluator()
        self.trainer = ModelTrainer(random_state=self.config['random_state'])
        self.predictor = ModelPredictor()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'random_state': 42,
            'enable_lag_features': True,
            'target_col': 'da',
            'site_col': 'site',
            'date_col': 'date',
            'drop_cols': ['date', 'site', 'da', 'da_category', 'da-category'],
            'quantiles': [0.05, 0.5, 0.95],
            'n_jobs': -1
        }
    
    def _create_feature_pipeline(self) -> Pipeline:
        """Create the feature engineering pipeline."""
        steps = [
            ('cleaner', DataCleaner()),
            ('temporal', TemporalFeatures(date_col=self.config['date_col'])),
            ('category_encoder', CategoryEncoder(target_col=self.config['target_col']))
        ]
        
        if self.config['enable_lag_features']:
            steps.append(('lag_features', LagFeatures(
                target_col=self.config['target_col'],
                group_col=self.config['site_col']
            )))
        
        # Don't drop target columns in the feature pipeline - we need them for training
        non_target_drop_cols = [col for col in self.config['drop_cols'] if col not in ['da', 'da_category']]
        steps.append(('feature_selector', FeatureSelector(
            drop_cols=non_target_drop_cols
        )))
        
        return Pipeline(steps)
    
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare data."""
        print(f"Loading data from {file_path}")
        data = pd.read_parquet(file_path)
        data[self.config['date_col']] = pd.to_datetime(data[self.config['date_col']])
        return data
    
    def fit(self, data: pd.DataFrame) -> 'DAForecastPipeline':
        """Fit the pipeline on training data."""
        print("Fitting forecasting pipeline...")
        
        # Create feature pipeline
        self.feature_pipeline = self._create_feature_pipeline()
        
        # Fit feature pipeline on full data to learn transformations
        # Note: This only learns scalers, encoders etc., not using future target info
        processed_data = self.feature_pipeline.fit_transform(data)
        
        # Create models (they will be trained per forecast, not globally)
        self.models = {
            'regression': ModelFactory.create_regression_models(self.config['random_state']),
            'classification': ModelFactory.create_classification_models(self.config['random_state']),
            'quantile': ModelFactory.create_quantile_models(self.config['quantiles'], self.config['random_state'])
        }
        
        self.is_fitted = True
        print("Pipeline fitted successfully")
        return self
    
    def _fit_and_predict_single(self, train_data: pd.DataFrame, 
                               forecast_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit models and make predictions for a single forecast."""
        # Process features - this creates da_category column
        train_processed = self.feature_pipeline.transform(train_data)
        forecast_processed = self.feature_pipeline.transform(forecast_data)
        
        # Get training features (excluding target columns)
        X_train = train_processed.drop(columns=[col for col in self.config['drop_cols'] if col in train_processed.columns], errors='ignore')
        X_forecast = forecast_processed.drop(columns=[col for col in self.config['drop_cols'] if col in forecast_processed.columns], errors='ignore')
        
        # Get targets from processed data
        y_reg = train_processed[self.config['target_col']].values if self.config['target_col'] in train_processed.columns else train_data[self.config['target_col']].values
        y_cls = train_processed['da_category'].values if 'da_category' in train_processed.columns else np.full(len(train_processed), np.nan)
        
        # Remove rows with NaN targets from training
        reg_mask = ~np.isnan(y_reg)
        cls_mask = ~np.isnan(y_cls)
        
        results = {}
        
        # Regression predictions
        if reg_mask.sum() > 0:
            X_train_reg = X_train[reg_mask]
            y_train_reg = y_reg[reg_mask]
            
            # Random Forest regression
            rf_reg = self.trainer.train_single_model(
                self.models['regression']['random_forest'], 
                X_train_reg, y_train_reg
            )
            rf_pred = self.predictor.predict_regression(rf_reg, X_forecast)[0]
            results['Predicted_da'] = rf_pred
            
            # Quantile regression
            quantile_models = self.trainer.train_quantile_models(
                X_train_reg, y_train_reg, self.config['quantiles']
            )
            quantile_preds = self.predictor.predict_quantiles(quantile_models, X_forecast)
            for name, pred in quantile_preds.items():
                results[f'Predicted_da_{name}'] = pred[0]
        
        # Classification predictions
        if cls_mask.sum() > 1:  # Need at least 2 classes
            X_train_cls = X_train[cls_mask]
            y_train_cls = y_cls[cls_mask]
            
            # Check if we have multiple classes
            unique_classes = np.unique(y_train_cls[~np.isnan(y_train_cls)])
            if len(unique_classes) > 1:
                rf_cls = self.trainer.train_single_model(
                    self.models['classification']['random_forest'],
                    X_train_cls, y_train_cls
                )
                cls_pred, cls_proba = self.predictor.predict_classification(rf_cls, X_forecast)
                results['Predicted_da_category'] = cls_pred[0]
                results['Prediction_probabilities'] = cls_proba[0]
        
        return results
    
    def forecast_single(self, data: pd.DataFrame, forecast_date: pd.Timestamp, 
                       site: str) -> Dict[str, Any]:
        """Make a single forecast for specific date and site."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Split data
        train_data, forecast_data, test_date = self.splitter.create_forecast_split(
            data, forecast_date, site
        )
        
        if train_data.empty:
            raise ValueError(f"No training data available for site {site} before {forecast_date}")
        
        # Validate split
        if not self.validator.validate_split(train_data, forecast_data):
            raise ValueError("Data leakage detected in train/test split")
        
        # Make predictions
        predictions = self._fit_and_predict_single(train_data, forecast_data)
        
        # Prepare result
        result = {
            'site': site,
            'forecast_date': forecast_date,
            'anchor_date': train_data[self.config['date_col']].max(),
            'test_date': test_date,
            **predictions
        }
        
        # Add actual values if available
        if not forecast_data[self.config['target_col']].isna().all():
            result['Actual_da'] = forecast_data[self.config['target_col']].iloc[0]
        
        # Handle both possible category column names
        cat_col = 'da_category' if 'da_category' in forecast_data.columns else 'da-category'
        if cat_col in forecast_data.columns and not forecast_data[cat_col].isna().all():
            result['Actual_da_category'] = forecast_data[cat_col].iloc[0]
        
        return result
    
    def evaluate_retrospective(self, data: pd.DataFrame, 
                             n_anchors_per_site: int = 100,
                             min_test_date: str = "2008-01-01") -> pd.DataFrame:
        """Perform retrospective evaluation using random anchors."""
        print("Starting retrospective evaluation...")
        
        # Generate random anchors
        anchor_generator = RandomAnchorGenerator(
            date_col=self.config['date_col'],
            site_col=self.config['site_col'],
            random_state=self.config['random_state']
        )
        
        min_date = pd.to_datetime(min_test_date) if min_test_date else None
        anchors = anchor_generator.generate_anchors(
            data, n_anchors_per_site, min_date
        )
        
        print(f"Generated {len(anchors)} anchor points for evaluation")
        
        # Process anchors in parallel
        def process_anchor(anchor_info):
            site, anchor_date = anchor_info
            try:
                return self.forecast_single(data, anchor_date, site)
            except Exception as e:
                print(f"Error processing anchor {site} at {anchor_date}: {e}")
                return None
        
        results = Parallel(n_jobs=self.config['n_jobs'], verbose=1)(
            delayed(process_anchor)(anchor) for anchor in anchors
        )
        
        # Filter successful results
        valid_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(valid_results)} forecasts")
        
        return pd.DataFrame(valid_results)
    
    def evaluate_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate overall performance."""
        return self.evaluator.evaluate_forecast_results(results_df)
    
    def get_site_performance(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Get performance metrics by site."""
        return self.evaluator.evaluate_by_site(results_df, self.config['site_col'])


class DAForecastConfig:
    """Configuration class for the forecasting pipeline."""
    
    @staticmethod
    def create_config(enable_lag_features: bool = True,
                     random_state: int = 42,
                     n_jobs: int = -1,
                     quantiles: List[float] = None) -> Dict[str, Any]:
        """Create configuration dictionary."""
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
            
        return {
            'random_state': random_state,
            'enable_lag_features': enable_lag_features,
            'target_col': 'da',
            'site_col': 'site', 
            'date_col': 'date',
            'drop_cols': ['date', 'site', 'da', 'da_category', 'da-category'],
            'quantiles': quantiles,
            'n_jobs': n_jobs
        }