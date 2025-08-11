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

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

import config
from .data_processor import DataProcessor
from .model_factory import ModelFactory
from .validation import validate_system_startup, validate_runtime_parameters
from .logging_config import get_logger
from .exception_handling import safe_execute, handle_data_errors, ScientificValidationError, TemporalLeakageError

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class ForecastEngine:
    """
    Leak-free domoic acid forecasting engine with temporal integrity.
    
    Features: Per-forecast DA categories, strict temporal ordering,
    temporal buffers for all features, no future data leakage.
    """
    
    def __init__(self, data_file=None, validate_on_init=True):
        try:
            logger.info("Initializing ForecastEngine")
            self.data_file = data_file or config.FINAL_OUTPUT_PATH
            self.data = None
            self.results_df = None
            
            logger.info(f"Using data file: {self.data_file}")
            
            # Validate system configuration on initialization
            if validate_on_init:
                logger.info("Validating system startup configuration")
                validate_system_startup()
                logger.info("System startup validation completed successfully")
            
            # Initialize sub-components
            logger.info("Initializing data processor and model factory")
            self.data_processor = DataProcessor()
            self.model_factory = ModelFactory()
            
            # Configuration matching original
            self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
            # Honor configurable minimum training samples
            try:
                self.min_training_samples = max(1, int(getattr(config, 'MIN_TRAINING_SAMPLES', 5)))
            except Exception:
                self.min_training_samples = 5
            self.random_seed = config.RANDOM_SEED
            
            logger.info(f"Configuration: buffer_days={self.temporal_buffer_days}, min_samples={self.min_training_samples}, seed={self.random_seed}")
            
            # Set random seeds
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            
            logger.info("ForecastEngine initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ForecastEngine: {str(e)}")
            raise ScientificValidationError(f"ForecastEngine initialization failed: {str(e)}")
        
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
        
        print(f"\n[INFO] Running LEAK-FREE {task} evaluation with {model_type}")
        
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
            print("[ERROR] No valid anchor points generated")
            return None
        
        print(f"[INFO] Generated {len(anchor_infos)} leak-free anchor points")
        
        # Process anchors in parallel using original method
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self._forecast_single_anchor_leak_free)(ai, self.data, min_target_date, task, model_type) 
            for ai in tqdm(anchor_infos, desc="Processing Leak-Free Anchors")
        )
        
        # Filter successful results and combine using original method
        forecast_dfs = [df for df in results if df is not None]
        if not forecast_dfs:
            print("[ERROR] No successful forecasts")
            return None
            
        final_df = pd.concat(forecast_dfs, ignore_index=True)
        final_df = final_df.sort_values(["date", "site"]).drop_duplicates(["date", "site"])
        print(f"[INFO] Successfully processed {len(forecast_dfs)} leak-free forecasts")
        
        # Store results for dashboard
        self.results_df = final_df
        
        # Display metrics using original format
        self._display_evaluation_metrics(task)
        
        return final_df
        
    def _forecast_single_anchor_leak_free(self, anchor_info, full_data, min_target_date, task, model_type):
        """Process single anchor forecast with ZERO data leakage - original algorithm."""
        site, anchor_date = anchor_info
        
        try:
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
            try:
                X_train_processed = transformer.fit_transform(X_train)
                X_test_processed = transformer.transform(X_test)
                
                # Additional validation - check for NaN in targets
                if pd.isna(train_df["da"]).any():
                    return None
                    
            except Exception as e:
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
                reg_model.fit(X_train_processed, train_df["da"])
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
            
        except Exception as e:
            return None
    
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
        try:
            # Load data using original method
            data = self.data_processor.load_and_prepare_base_data(data_path)
            forecast_date = pd.Timestamp(forecast_date)
            
            # Validate forecast inputs
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
                model.fit(X_train_processed, df_train_clean["da"])
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
                        try:
                            probabilities = model.predict_proba(X_forecast)[0]
                            # Convert to 4-element array format [cat0, cat1, cat2, cat3] for frontend
                            prob_array = [0.0, 0.0, 0.0, 0.0]  # Initialize all categories
                            for i, prob in enumerate(probabilities):
                                original_cat = reverse_mapping[i]
                                prob_array[original_cat] = float(prob)
                            result['class_probabilities'] = prob_array
                        except Exception:
                            # Silently skip probability calculation if not supported
                            pass
                            
                else:
                    # Single class scenario - predict the dominant class
                    # This allows sites like Cannon Beach with limited toxin diversity to still generate predictions
                    dominant_class = df_train_clean["da-category"].mode()[0]
                    result['predicted_category'] = int(dominant_class)
                    result['single_class_prediction'] = True
                    logger.debug(f"Single-class prediction for {site}: {dominant_class} (only class in training data)")
                    
            return result
            
        except Exception as e:
            return None
            
    def _display_evaluation_metrics(self, task):
        """Display evaluation metrics using original format."""
        if self.results_df is None or self.results_df.empty:
            print("No results for evaluation")
            return
            
        print(f"[INFO] Successfully processed {len(self.results_df)} leak-free forecasts")
        
        if task == "regression" or task == "both":
            # Calculate regression metrics
            valid_results = self.results_df.dropna(subset=['da', 'Predicted_da'])
            if not valid_results.empty:
                r2 = r2_score(valid_results['da'], valid_results['Predicted_da'])
                mae = mean_absolute_error(valid_results['da'], valid_results['Predicted_da'])
                print(f"[INFO] LEAK-FREE Regression R2: {r2:.4f}, MAE: {mae:.4f}")
            else:
                print("[WARNING] No valid regression results for evaluation")
                
        if task == "classification" or task == "both":
            # Calculate classification metrics
            valid_results = self.results_df.dropna(subset=['da-category', 'Predicted_da-category'])
            if not valid_results.empty:
                accuracy = accuracy_score(valid_results['da-category'], valid_results['Predicted_da-category'])
                print(f"[INFO] LEAK-FREE Classification Accuracy: {accuracy:.4f}")
            else:
                print("[WARNING] No valid classification results for evaluation")