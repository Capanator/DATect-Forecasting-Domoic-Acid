"""
Leak-Free Forecasting Engine
Core forecasting with complete temporal integrity protection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

import config
from .data_processor import DataProcessor
from .model_factory import ModelFactory
from .validation import validate_system_startup, validate_runtime_parameters
from .logging_config import get_logger
from .statistical_enhancements import StatisticalEnhancer

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class ForecastEngine:
    """
    Leak-free domoic acid forecasting engine with temporal integrity.
    
    Features: Per-forecast DA categories, strict temporal ordering,
    temporal buffers for all features, no future data leakage.
    """
    
    def __init__(self, data_file=None, validate_on_init=True):
        logger.info("Initializing ForecastEngine")
        self.data_file = data_file or config.FINAL_OUTPUT_PATH
        self.data = None
        self.results_df = None
        
        logger.info(f"Using data file: {self.data_file}")
        
        # Validate system configuration on initialization - CRITICAL for temporal integrity
        if validate_on_init:
            logger.info("Validating system startup configuration")
            validate_system_startup()
            logger.info("System startup validation completed successfully")
        
        # Initialize sub-components
        logger.info("Initializing data processor and model factory")
        self.data_processor = DataProcessor()
        self.model_factory = ModelFactory()
        
        # Initialize statistical enhancements
        n_bootstrap = getattr(config, 'BOOTSTRAP_ITERATIONS', 1000)
        confidence_level = getattr(config, 'CONFIDENCE_LEVEL', 0.95)
        self.statistical_enhancer = StatisticalEnhancer(n_bootstrap, confidence_level)
        
        # Configuration matching original
        self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
        self.min_training_samples = max(1, int(getattr(config, 'MIN_TRAINING_SAMPLES', 5)))
        self.random_seed = config.RANDOM_SEED
        
        logger.info(f"Configuration: buffer_days={self.temporal_buffer_days}, min_samples={self.min_training_samples}, seed={self.random_seed}")
        
        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info("ForecastEngine initialization completed successfully")
        
    def run_retrospective_evaluation(self, task="regression", model_type="xgboost", 
                                   n_anchors=50, min_test_date="2008-01-01"):
        """
        Run leak-free retrospective evaluation matching original behavior.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost", "linear", or "logistic" 
            n_anchors: Number of random anchor points per site
            min_test_date: Earliest date for test anchors
            
        Returns:
            DataFrame with evaluation results matching original format
        """
        # Validate runtime parameters
        validate_runtime_parameters(n_anchors, min_test_date)
        
        logger.info(f"Running LEAK-FREE {task} evaluation with {model_type}")
        
        # Load data using original method
        self.data = self.data_processor.load_and_prepare_base_data(self.data_file)
        min_target_date = pd.Timestamp(min_test_date)
        
        # Diagnostics
        self.last_diagnostics = {
            "task": task,
            "model_type": model_type,
            "min_test_date": str(min_test_date),
            "per_site": {}
        }

        # Generate anchor points using original algorithm
        anchor_infos = []
        for site in self.data["site"].unique():
            self.last_diagnostics["per_site"][site] = {
                "candidate_dates": 0,
                "valid_future": 0,
                "selected": 0,
                "earliest_selected_date": None
            }
            site_dates = self.data[self.data["site"] == site]["date"].sort_values().unique()
            if len(site_dates) > self.temporal_buffer_days:  # Need enough history
                # Only use dates that have sufficient history and future data
                valid_anchors = []
                for i, date in enumerate(site_dates[:-1]):  # Exclude last date
                    self.last_diagnostics["per_site"][site]["candidate_dates"] += 1
                    if date >= min_target_date:
                        # Check if there's a future date with sufficient buffer
                        future_dates = site_dates[i+1:]
                        valid_future = [d for d in future_dates if (d - date).days >= self.temporal_buffer_days]
                        if valid_future:
                            self.last_diagnostics["per_site"][site]["valid_future"] += 1
                            valid_anchors.append(date)
                
                if valid_anchors:
                    n_sample = min(len(valid_anchors), n_anchors)
                    # Restore random sampling of anchors to avoid selection bias
                    selected_anchors = random.sample(list(valid_anchors), n_sample)
                    anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
                    sel_sorted = sorted(selected_anchors)
                    self.last_diagnostics["per_site"][site]["selected"] = len(selected_anchors)
                    self.last_diagnostics["per_site"][site]["earliest_selected_date"] = str(sel_sorted[0].date()) if sel_sorted else None
        
        if not anchor_infos:
            logger.warning("No valid anchor points generated")
            return None
        
        logger.info(f"Generated {len(anchor_infos)} leak-free anchor points")
        
        # Process anchors in parallel using original method
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self._forecast_single_anchor_leak_free)(ai, self.data, min_target_date, task, model_type) 
            for ai in tqdm(anchor_infos, desc="Processing Leak-Free Anchors")
        )
        
        # Filter successful results and combine using original method
        forecast_dfs = [df for df in results if df is not None]
        if not forecast_dfs:
            logger.warning("No successful forecasts")
            return None
            
        final_df = pd.concat(forecast_dfs, ignore_index=True)
        final_df = final_df.sort_values(["date", "site"]).drop_duplicates(["date", "site"])
        logger.info(f"Successfully processed {len(forecast_dfs)} leak-free forecasts")
        
        # Store results for dashboard
        self.results_df = final_df
        
        # Display metrics using original format
        self._display_evaluation_metrics(task)
        
        return final_df
        
    def _forecast_single_anchor_leak_free(self, anchor_info, full_data, min_target_date, task, model_type):
        """Process single anchor forecast with ZERO data leakage - original algorithm."""
        site, anchor_date = anchor_info
        
        # Get site data
        site_data = full_data[full_data["site"] == site].copy()
        site_data.sort_values("date", inplace=True)
        
        # CRITICAL: Split data FIRST, before any feature engineering (original method)
        train_mask = site_data["date"] <= anchor_date
        test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
        
        train_df = site_data[train_mask].copy()
        test_candidates = site_data[test_mask]
        
        if train_df.empty or test_candidates.empty:
            return None
            
        # Get the first test point (next available date after anchor)
        test_df = test_candidates.iloc[:1].copy()
        test_date = test_df["date"].iloc[0]
        
        # Ensure minimum temporal buffer
        if (test_date - anchor_date).days < self.temporal_buffer_days:
            return None
        
        # NOW create lag features with strict temporal cutoff (original method)
        site_data_with_lags = self.data_processor.create_lag_features_safe(
            site_data, "site", "da", config.LAG_FEATURES, anchor_date
        )
        
        # Re-extract training and test data with lag features
        train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].copy()
        test_df = site_data_with_lags[site_data_with_lags["date"] == test_date].copy()
        
        if train_df.empty or test_df.empty:
            return None
        
        # Remove rows with missing target values from training FIRST
        train_df_clean = train_df.dropna(subset=["da"]).copy()
        if train_df_clean.empty or len(train_df_clean) < self.min_training_samples:
            return None
        
        # Create DA categories ONLY from clean training data
        train_df_clean["da-category"] = self.data_processor.create_da_categories_safe(train_df_clean["da"])
        train_df = train_df_clean
        
        # Define columns to drop (original method)
        base_drop_cols = ["date", "site", "da"]
        train_drop_cols = base_drop_cols + ["da-category"]
        test_drop_cols = base_drop_cols  # Test data doesn't have categories yet
        
        # Prepare features using original method
        transformer, X_train = self.data_processor.create_numeric_transformer(train_df, train_drop_cols)
        X_test = test_df.drop(columns=[col for col in test_drop_cols if col in test_df.columns])
        
        # Ensure test features match training features
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Transform features
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        # Check for NaN in targets
        if pd.isna(train_df["da"]).any():
            return None
        
        # Get actual values
        actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
        actual_category = self.data_processor.create_da_categories_safe(pd.Series([actual_da]))[0] if actual_da is not None else None
        
        # Create result using original format
        result = {
            'date': test_date,
            'site': site,
            'anchor_date': anchor_date,
            'da': actual_da,
            'da-category': actual_category
        }
        
        # Make predictions based on task (original method)
        if task == "regression" or task == "both":
            reg_model = self.model_factory.get_model("regression", model_type)
            
            # Create sample weights for spike events (enhance spike detection)
            y_train = train_df["da"]
            spike_mask = y_train > 20.0  # spike threshold
            sample_weights = np.ones(len(y_train))
            sample_weights[spike_mask] *= 8.0  # precision weight for spikes
            
            # Fit with sample weights for XGBoost models
            if model_type in ["xgboost", "xgb"]:
                reg_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                reg_model.fit(X_train_processed, y_train)
            
            pred_da = reg_model.predict(X_test_processed)[0]
            # Ensure DA predictions cannot be negative (biological constraint)
            pred_da = max(0.0, float(pred_da))
            result['Predicted_da'] = pred_da
        
        if task == "classification" or task == "both":
            # Check if we have multiple classes in training data
            unique_classes = train_df["da-category"].nunique()
            if unique_classes > 1:
                # Handle non-consecutive class labels for XGBoost
                unique_cats = sorted(train_df["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                # Convert to consecutive labels for XGBoost
                y_train_encoded = train_df["da-category"].map(cat_mapping)
                
                cls_model = self.model_factory.get_model("classification", model_type)
                cls_model.fit(X_train_processed, y_train_encoded)
                pred_encoded = cls_model.predict(X_test_processed)[0]
                
                # Convert back to original category
                pred_category = reverse_mapping[pred_encoded]
                result['Predicted_da-category'] = pred_category
            else:
                # Single class scenario - predict the dominant class
                # This allows sites like Cannon Beach with limited toxin diversity to still generate predictions
                dominant_class = train_df["da-category"].mode()[0]
                result['Predicted_da-category'] = dominant_class
                result['single_class_prediction'] = True
                # Note: This is a naive baseline but maintains temporal integrity
        
        return pd.DataFrame([result])
    
    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        """
        Generate a single forecast for a specific date and site using original algorithm.
        
        Args:
            data_path: Path to data file
            forecast_date: Date to forecast for
            site: Site to forecast for
            task: "regression" or "classification"
            model_type: Model type to use
            
        Returns:
            Dictionary with forecast results or None if insufficient data
        """
        # Load data using original method
        data = self.data_processor.load_and_prepare_base_data(data_path)
        forecast_date = pd.Timestamp(forecast_date)
        
        # Validate forecast inputs - CRITICAL for temporal integrity
        self.data_processor.validate_forecast_inputs(data, site, forecast_date)
        
        # Get site data
        df_site = data[data['site'] == site].copy()
        df_site.sort_values('date', inplace=True)
        
        # Find the latest available data before forecast date
        available_before = df_site[df_site['date'] < forecast_date]
        if available_before.empty:
            return None
            
        anchor_date = available_before['date'].max()
        
        # Ensure minimum temporal buffer
        if (forecast_date - anchor_date).days < self.temporal_buffer_days:
            return None
        
        # Create lag features with strict cutoff (original method)
        df_site_with_lags = self.data_processor.create_lag_features_safe(
            df_site, "site", "da", config.LAG_FEATURES, anchor_date
        )
        
        # Get training data (everything up to and including anchor date)
        df_train = df_site_with_lags[df_site_with_lags['date'] <= anchor_date].copy()
        df_train_clean = df_train.dropna(subset=['da']).copy()
        
        if df_train_clean.empty or len(df_train_clean) < self.min_training_samples:
            return None
        
        # Create DA categories from training data only
        df_train_clean["da-category"] = self.data_processor.create_da_categories_safe(df_train_clean["da"])
        
        # Prepare features (original method)
        drop_cols = ["date", "site", "da", "da-category"]
        transformer, X_train = self.data_processor.create_numeric_transformer(df_train_clean, drop_cols)
        
        # Transform features
        X_train_processed = transformer.fit_transform(X_train)
        
        # Create forecast point using latest available data
        latest_data = df_train_clean.iloc[-1:].copy()
        X_forecast = transformer.transform(latest_data.drop(columns=drop_cols, errors='ignore'))
        
        # Initialize result
        result = {
            'forecast_date': forecast_date,
            'site': site,
            'task': task,
            'model_type': model_type,
            'training_samples': len(df_train_clean)
        }
        
        # Make predictions based on task
        if task == "regression":
            model = self.model_factory.get_model("regression", model_type)
            
            # Create sample weights for spike events (enhance spike detection)
            y_train = df_train_clean["da"]
            spike_mask = y_train > 20.0  # spike threshold
            sample_weights = np.ones(len(y_train))
            sample_weights[spike_mask] *= 8.0  # precision weight for spikes
            
            # Fit with sample weights for XGBoost models
            if model_type in ["xgboost", "xgb"]:
                model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train)
            
            prediction = model.predict(X_forecast)[0]
            # Ensure DA predictions cannot be negative (biological constraint)
            prediction = max(0.0, float(prediction))
            result['predicted_da'] = prediction
            result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
            logger.debug(f"Regression prediction completed for {site}: {prediction:.4f}")
            
        elif task == "classification":
            # Check if we have multiple classes
            unique_classes = df_train_clean["da-category"].nunique()
            if unique_classes > 1:
                # Handle non-consecutive class labels for XGBoost
                unique_cats = sorted(df_train_clean["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                # Convert to consecutive labels for XGBoost
                y_train_encoded = df_train_clean["da-category"].map(cat_mapping)
                
                model = self.model_factory.get_model("classification", model_type)
                model.fit(X_train_processed, y_train_encoded)
                pred_encoded = model.predict(X_forecast)[0]
                
                # Convert back to original category
                prediction = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(prediction)
                result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
                logger.debug(f"Classification prediction completed for {site}: {prediction}")
                
                # Add class probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_forecast)[0]
                    # Convert to 4-element array format [cat0, cat1, cat2, cat3] for frontend
                    prob_array = [0.0, 0.0, 0.0, 0.0]  # Initialize all categories
                    for i, prob in enumerate(probabilities):
                        original_cat = reverse_mapping[i]
                        prob_array[original_cat] = float(prob)
                    result['class_probabilities'] = prob_array
                        
            else:
                # Single class scenario - predict the dominant class
                # This allows sites like Cannon Beach with limited toxin diversity to still generate predictions
                dominant_class = df_train_clean["da-category"].mode()[0]
                result['predicted_category'] = int(dominant_class)
                result['single_class_prediction'] = True
                logger.debug(f"Single-class prediction for {site}: {dominant_class} (only class in training data)")
                
        return result
            
    def _display_evaluation_metrics(self, task):
        """Display evaluation metrics using original format."""
        if self.results_df is None or self.results_df.empty:
            logger.warning("No results for evaluation")
            return
            
        logger.info(f"Successfully processed {len(self.results_df)} leak-free forecasts")
        
        if task == "regression" or task == "both":
            # Calculate regression metrics
            valid_results = self.results_df.dropna(subset=['da', 'Predicted_da'])
            if not valid_results.empty:
                r2 = r2_score(valid_results['da'], valid_results['Predicted_da'])
                mae = mean_absolute_error(valid_results['da'], valid_results['Predicted_da'])
                
                # Convert regression to binary classification for F1, precision, recall (spike detection)
                spike_threshold = 20.0
                y_true_binary = (valid_results['da'] > spike_threshold).astype(int)
                y_pred_binary = (valid_results['Predicted_da'] > spike_threshold).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                logger.info(f"LEAK-FREE Regression Metrics:")
                logger.info(f"  R2: {r2:.4f}, MAE: {mae:.4f}")
                logger.info(f"  Spike Detection (>{spike_threshold}): Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            else:
                logger.warning("No valid regression results for evaluation")
                
        if task == "classification" or task == "both":
            # Calculate classification metrics
            valid_results = self.results_df.dropna(subset=['da-category', 'Predicted_da-category'])
            if not valid_results.empty:
                y_true = valid_results['da-category']
                y_pred = valid_results['Predicted_da-category']
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                logger.info(f"LEAK-FREE Classification Metrics:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
            else:
                logger.warning("No valid classification results for evaluation")
    
    def generate_enhanced_forecast(self, data_path, forecast_date, site, task, model_type, 
                                 include_uncertainty=True, include_comparison=False):
        """
        Generate enhanced forecast with bootstrap uncertainty quantification and optional baseline comparison.
        
        Args:
            data_path: Path to data file
            forecast_date: Date to forecast for
            site: Site to forecast for
            task: "regression" or "classification"  
            model_type: Model type to use
            include_uncertainty: Whether to include bootstrap confidence intervals
            include_comparison: Whether to include baseline model comparison
            
        Returns:
            Dictionary with enhanced forecast results including uncertainty bounds
        """
        logger.info(f"Generating enhanced forecast for {site} on {forecast_date} using {model_type}")
        
        # Get base forecast
        base_result = self.generate_single_forecast(data_path, forecast_date, site, task, model_type)
        
        if base_result is None:
            return None
            
        # If uncertainty not requested, return base result
        if not include_uncertainty and not include_comparison:
            return base_result
            
        try:
            # Load and prepare data for uncertainty estimation
            data = self.data_processor.load_and_prepare_base_data(data_path)
            forecast_date = pd.Timestamp(forecast_date)
            
            df_site = data[data['site'] == site].copy()
            df_site.sort_values('date', inplace=True)
            
            # Get training data (same logic as generate_single_forecast)
            available_before = df_site[df_site['date'] < forecast_date]
            if available_before.empty:
                logger.warning("No training data available for uncertainty estimation")
                return base_result
            
            anchor_date = available_before['date'].max()
            temporal_cutoff = forecast_date - pd.Timedelta(days=self.temporal_buffer_days)
            
            # Training data: all available data before temporal cutoff 
            training_cutoff = min(anchor_date, temporal_cutoff)
            train_data = df_site[df_site['date'] <= training_cutoff].copy()
            
            if len(train_data) < self.min_training_samples:
                logger.warning("Insufficient training data for uncertainty estimation")
                return base_result
            
            # Use same data preparation logic as generate_single_forecast
            df_site_with_lags = self.data_processor.create_lag_features_safe(
                df_site, "site", "da", config.LAG_FEATURES, anchor_date
            )
            
            # Get training data (everything up to and including anchor date)
            df_train = df_site_with_lags[df_site_with_lags['date'] <= anchor_date].copy()
            df_train_clean = df_train.dropna(subset=['da']).copy()
            
            if df_train_clean.empty or len(df_train_clean) < self.min_training_samples:
                logger.warning("Insufficient clean training data for uncertainty estimation")
                return base_result
            
            # Create DA categories from training data only
            df_train_clean["da-category"] = self.data_processor.create_da_categories_safe(df_train_clean["da"])
            
            # Prepare features using same method as original
            drop_cols = ["date", "site", "da", "da-category"]
            transformer, X_train = self.data_processor.create_numeric_transformer(df_train_clean, drop_cols)
            
            # Transform features
            X_train_processed = transformer.fit_transform(X_train)
            y_train = df_train_clean["da"]
            
            # Create forecast point using latest available data
            latest_data = df_train_clean.iloc[-1:].copy()
            X_pred_processed = transformer.transform(latest_data.drop(columns=drop_cols, errors='ignore'))
            
            if len(X_train_processed) < self.min_training_samples:
                logger.warning("Insufficient processed training data for uncertainty estimation")
                return base_result
            
            # Create and train model using processed features
            model = self.model_factory.get_model(task, model_type)
            
            # Create sample weights for spike events if task is regression
            if task == "regression":
                spike_mask = y_train > 20.0  # spike threshold
                sample_weights = np.ones(len(y_train))
                sample_weights[spike_mask] *= 8.0  # precision weight for spikes
                
                # Fit with sample weights for XGBoost models
                if model_type in ["xgboost", "xgb"]:
                    model.fit(X_train_processed, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train_processed, y_train)
            else:
                model.fit(X_train_processed, y_train)
            
            enhanced_result = base_result.copy()
            
            # Add uncertainty estimation
            if include_uncertainty:
                logger.info("Computing bootstrap confidence intervals...")
                
                # Convert to DataFrames for bootstrap method
                X_train_df = pd.DataFrame(X_train_processed) if not isinstance(X_train_processed, pd.DataFrame) else X_train_processed
                X_pred_df = pd.DataFrame(X_pred_processed) if not isinstance(X_pred_processed, pd.DataFrame) else X_pred_processed
                
                uncertainty = self.statistical_enhancer.bootstrap_prediction_interval(
                    model, X_train_df, y_train, X_pred_df, task
                )
                
                enhanced_result['uncertainty'] = {
                    'method': 'bootstrap',
                    'confidence_level': self.statistical_enhancer.confidence_level,
                    'n_iterations': uncertainty['n_successful_iterations'],
                    'prediction_mean': uncertainty['mean'],
                    'prediction_std': uncertainty['std'],
                    'lower_bound': uncertainty['lower_bound'],
                    'upper_bound': uncertainty['upper_bound']
                }
                
                logger.info(f"Uncertainty bounds: [{uncertainty['lower_bound']:.3f}, {uncertainty['upper_bound']:.3f}]")
            
            # Add baseline comparison
            if include_comparison:
                logger.info("Computing baseline model comparison...")
                baseline_result = self._compute_baseline_comparison(
                    X_train_df, y_train, X_pred_df, model, task, model_type
                )
                enhanced_result['baseline_comparison'] = baseline_result
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced forecast: {e}")
            logger.warning("Falling back to base forecast result")
            return base_result
    
    def _compute_baseline_comparison(self, X_train, y_train, X_pred, primary_model, task, primary_model_type):
        """Compute comparison between primary model and baseline models."""
        comparisons = {}
        
        try:
            # Get primary model prediction
            primary_pred = primary_model.predict(X_pred)[0]
            
            # Linear baseline (existing in the system)
            baseline_model_type = 'linear' if task == 'regression' else 'logistic'
            baseline_model = self.model_factory.get_model(task, baseline_model_type)
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_pred)[0]
            
            # Simple persistence baseline (last known value)
            persistence_pred = y_train.iloc[-1] if task == 'regression' else y_train.mode().iloc[0]
            
            # Historical mean baseline
            mean_pred = y_train.mean() if task == 'regression' else y_train.mode().iloc[0]
            
            comparisons = {
                'primary_model': {
                    'name': primary_model_type,
                    'prediction': primary_pred
                },
                'baselines': {
                    'statistical_model': {
                        'name': baseline_model_type,
                        'prediction': baseline_pred,
                        'improvement_vs_primary': ((baseline_pred - primary_pred) / abs(baseline_pred) * 100) if baseline_pred != 0 else 0
                    },
                    'persistence': {
                        'name': 'persistence',
                        'prediction': persistence_pred,
                        'improvement_vs_primary': ((persistence_pred - primary_pred) / abs(persistence_pred) * 100) if persistence_pred != 0 else 0
                    },
                    'historical_mean': {
                        'name': 'historical_mean',
                        'prediction': mean_pred,
                        'improvement_vs_primary': ((mean_pred - primary_pred) / abs(mean_pred) * 100) if mean_pred != 0 else 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error computing baseline comparison: {e}")
            comparisons = {'error': str(e)}
        
        return comparisons