"""
Leak-Free Forecasting Engine
============================

Core forecasting logic matching the original leak_free_forecast.py behavior
while maintaining modular architecture and complete temporal integrity protection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import random
import traceback
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, Dict, List, Any

from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    precision_recall_fscore_support, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

import config
from .data_processor import DataProcessor
from .model_factory import ModelFactory
from .scientific_validation import ScientificValidator
from .logging_config import get_logger, log_performance, log_model_performance, log_data_pipeline_stage
from .exception_handling import (
    safe_execute, handle_data_error, handle_model_error, validate_data_integrity,
    robust_decorator, DataProcessingError, ModelError, ValidationError
)
from .model_persistence import ModelArtifactManager

warnings.filterwarnings('ignore')


class ForecastEngine:
    """
    Leak-free domoic acid forecasting engine matching original behavior.
    
    Features:
    - Complete temporal integrity protection
    - Per-forecast DA category creation
    - Strict train/test split ordering
    - Temporal buffers for all features
    - Original algorithm performance
    """
    
    def __init__(self, data_file=None):
        self.data_file = data_file or config.FINAL_OUTPUT_PATH
        self.data = None
        self.results_df = None
        
        # Initialize logging
        self.logger = get_logger(__name__)
        
        # Initialize sub-components
        self.data_processor = DataProcessor()
        self.model_factory = ModelFactory()
        self.scientific_validator = ScientificValidator(save_plots=True, plot_dir="./evaluation_plots/")
        self.model_manager = ModelArtifactManager()
        
        # Configuration matching original
        self.temporal_buffer_days = config.TEMPORAL_BUFFER_DAYS
        self.min_training_samples = 5  # Restore original minimum
        self.random_seed = config.RANDOM_SEED
        
        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    @robust_decorator(fallback_value=None, max_retries=1, handle_errors=True)
    def run_retrospective_evaluation(self, task="regression", model_type="rf", 
                                   n_anchors=50, min_test_date="2008-01-01"):
        """
        Run leak-free retrospective evaluation with comprehensive error handling.
        
        Args:
            task: "regression" or "classification"
            model_type: "rf", "gb", "ridge", or "logistic" 
            n_anchors: Number of random anchor points per site
            min_test_date: Earliest date for test anchors
            
        Returns:
            DataFrame with evaluation results matching original format
            
        Raises:
            DataProcessingError: If data loading/processing fails
            ValidationError: If data validation fails
            ModelError: If model training/prediction fails
        """
        try:
            self.logger.info(f"Starting LEAK-FREE {task} evaluation with {model_type} model")
            
            # Load and validate data
            try:
                self.data = self.data_processor.load_and_prepare_base_data(self.data_file)
                
                # Validate loaded data
                validate_data_integrity(
                    self.data,
                    required_columns=['date', 'site', 'da'],
                    min_rows=10,
                    max_missing_rate=0.8
                )
                
            except Exception as e:
                handle_data_error(e, context="loading retrospective evaluation data", 
                                data_shape=getattr(self.data, 'shape', None))
            
            min_target_date = pd.Timestamp(min_test_date)
            
            # Generate anchor points using original algorithm with error handling
            anchor_infos = []
            
            try:
                for site in self.data["site"].unique():
                    site_dates = self.data[self.data["site"] == site]["date"].sort_values().unique()
                    if len(site_dates) > self.temporal_buffer_days:  # Need enough history
                        # Only use dates that have sufficient history and future data
                        valid_anchors = []
                        for i, date in enumerate(site_dates[:-1]):  # Exclude last date
                            if date >= min_target_date:
                                # Check if there's a future date with sufficient buffer
                                future_dates = site_dates[i+1:]
                                valid_future = [d for d in future_dates if (d - date).days >= self.temporal_buffer_days]
                                if valid_future:
                                    valid_anchors.append(date)
                        
                        if valid_anchors:
                            n_sample = min(len(valid_anchors), n_anchors)
                            selected_anchors = random.sample(list(valid_anchors), n_sample)
                            anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
                            
            except Exception as e:
                handle_data_error(e, context="generating temporal anchor points",
                                data_shape=self.data.shape)
        
            if not anchor_infos:
                self.logger.error("No valid anchor points generated - insufficient data for evaluation")
                return None
            
            log_data_pipeline_stage(
                "Anchor Point Generation", 
                data_shape=len(anchor_infos), 
                success=True,
                details=f"{len(anchor_infos)} temporal anchor points across {len(self.data['site'].unique())} sites"
            )
            
            # Process anchors in parallel with error handling
            try:
                self.logger.info(f"Processing {len(anchor_infos)} anchor points in parallel")
                results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(self._forecast_single_anchor_leak_free)(ai, self.data, min_target_date, task, model_type) 
                    for ai in tqdm(anchor_infos, desc="Processing Leak-Free Anchors")
                )
            except Exception as e:
                handle_model_error(e, model_name=model_type, 
                                 data_points=len(anchor_infos),
                                 features=list(self.data.columns))
        
            # Filter successful results and combine using original method
            try:
                forecast_dfs = [df for df in results if df is not None]
                if not forecast_dfs:
                    self.logger.error("No successful forecasts generated - check data quality and anchor points")
                    return None
                    
                final_df = pd.concat(forecast_dfs, ignore_index=True)
                final_df = final_df.sort_values(["date", "site"]).drop_duplicates(["date", "site"])
                
                log_data_pipeline_stage(
                    "Forecast Generation",
                    data_shape=final_df.shape,
                    success=True,
                    details=f"Generated {len(forecast_dfs)} successful forecasts from {len(anchor_infos)} attempts"
                )
                
                # Store results for dashboard
                self.results_df = final_df
                
                # Display metrics using original format with error handling
                try:
                    self._display_evaluation_metrics(task)
                except Exception as e:
                    self.logger.warning(f"Error displaying evaluation metrics: {str(e)}")
                
                return final_df
                
            except Exception as e:
                handle_data_error(e, context="combining forecast results",
                                data_shape=(len(forecast_dfs) if 'forecast_dfs' in locals() else 0, 0))
                
        except Exception as e:
            self.logger.error(f"Unexpected error in retrospective evaluation: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
        
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
                site_data, "site", "da", [1, 2, 3], anchor_date
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
                result['Predicted_da'] = pred_da
            
            if task == "classification" or task == "both":
                # Check if we have multiple classes in training data
                unique_classes = train_df["da-category"].nunique()
                if unique_classes > 1:
                    cls_model = self.model_factory.get_model("classification", model_type)
                    cls_model.fit(X_train_processed, train_df["da-category"])
                    pred_category = cls_model.predict(X_test_processed)[0]
                    result['Predicted_da-category'] = pred_category
                else:
                    result['Predicted_da-category'] = np.nan
            
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
                df_site, "site", "da", [1, 2, 3], anchor_date
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
                result['predicted_da'] = prediction
                result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
                
            elif task == "classification":
                # Check if we have multiple classes
                unique_classes = df_train_clean["da-category"].nunique()
                if unique_classes > 1:
                    model = self.model_factory.get_model("classification", model_type)
                    model.fit(X_train_processed, df_train_clean["da-category"])
                    prediction = model.predict(X_forecast)[0]
                    result['predicted_category'] = prediction
                    result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
                    
                    # Add class probabilities if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            probabilities = model.predict_proba(X_forecast)[0]
                            result['class_probabilities'] = probabilities
                        except Exception:
                            # Silently skip probability calculation if not supported
                            pass
                else:
                    return None
                    
            return result
            
        except Exception as e:
            return None
            
    def _display_evaluation_metrics(self, task):
        """Display evaluation metrics with proper logging."""
        if self.results_df is None or self.results_df.empty:
            self.logger.warning("No results available for evaluation")
            return
            
        self.logger.info(f"Evaluating performance on {len(self.results_df)} forecasts")
        
        if task == "regression" or task == "both":
            # Calculate comprehensive regression metrics with residual analysis
            valid_results = self.results_df.dropna(subset=['da', 'Predicted_da'])
            if not valid_results.empty:
                y_true = valid_results['da'].values
                y_pred = valid_results['Predicted_da'].values
                
                # Basic metrics
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                # Additional regression metrics
                residuals = y_true - y_pred
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                
                # Log regression performance
                regression_metrics = {
                    'R²': r2,
                    'MAE': mae, 
                    'RMSE': rmse,
                    'Mean Residual': mean_residual,
                    'Residual Std': std_residual
                }
                
                log_model_performance(
                    f"{config.FORECAST_MODEL} Regression",
                    regression_metrics,
                    data_points=len(valid_results)
                )
                
                # Comprehensive residual analysis
                self.logger.info("Performing comprehensive residual analysis for model validation")
                residual_results = self.scientific_validator.analyze_residuals(
                    y_true=y_true,
                    y_pred=y_pred,
                    title="DATect Regression Model"
                )
                
                # Additional diagnostic information
                self._print_regression_diagnostics(residual_results)
                
            else:
                self.logger.warning("No valid regression results for evaluation - insufficient prediction data")
                
        if task == "classification" or task == "both":
            # Calculate comprehensive classification metrics
            valid_results = self.results_df.dropna(subset=['da-category', 'Predicted_da-category'])
            if not valid_results.empty:
                y_true = valid_results['da-category'].astype(int)
                y_pred = valid_results['Predicted_da-category'].astype(int)
                
                # Basic metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Comprehensive metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
                
                # Per-class metrics
                precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                    y_true, y_pred, average=None, zero_division=0
                )
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Log classification performance
                classification_metrics = {
                    'Accuracy': accuracy,
                    'Precision (weighted)': precision,
                    'Recall (weighted)': recall,
                    'F1-Score (weighted)': f1
                }
                
                log_model_performance(
                    f"{config.FORECAST_MODEL} Classification",
                    classification_metrics,
                    data_points=len(valid_results)
                )
                
                # Log per-class breakdown 
                class_names = ['Low', 'Moderate', 'High', 'Extreme']
                self.logger.info("Per-class classification performance:")
                for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class)):
                    if i < len(class_names) and s > 0:
                        self.logger.info(f"  {class_names[i]}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
                
                # Log confusion matrix summary
                self.logger.info("Confusion matrix generated and saved to evaluation plots")
                
                # ROC-AUC for multiclass (if possible)
                try:
                    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                    if len(unique_classes) >= 2:
                        # For multiclass, use macro average
                        auc_score = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
                        self.logger.info(f"ROC-AUC (macro average): {auc_score:.4f}")
                except Exception as e:
                    self.logger.info("ROC-AUC not computed (requires probability scores for multiclass)")
                
                # Save confusion matrix plot
                self._plot_confusion_matrix(cm, class_names[:len(np.unique(y_true))])
                
            else:
                self.logger.warning("No valid classification results for evaluation - insufficient prediction data")
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Create and save confusion matrix visualization."""
        try:
            import os
            os.makedirs("./evaluation_plots", exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            
            # Set ticks and labels
            ax.set(xticks=np.arange(cm.shape[1]),
                  yticks=np.arange(cm.shape[0]),
                  xticklabels=class_names,
                  yticklabels=class_names,
                  title='Confusion Matrix - DA Risk Categories',
                  ylabel='True Category',
                  xlabel='Predicted Category')
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            fmt = 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=12)
            
            fig.tight_layout()
            plt.savefig("./evaluation_plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Confusion matrix plot saved to: ./evaluation_plots/confusion_matrix.png")
            
        except Exception as e:
            self.logger.warning(f"Could not save confusion matrix plot: {e}")
    
    def _print_regression_diagnostics(self, residual_results):
        """Log interpretation of residual analysis results."""
        self.logger.info("Regression Model Diagnostics Summary:")
        
        # Normality assessment
        if 'shapiro_wilk' in residual_results:
            sw_p = residual_results['shapiro_wilk']['p_value']
            if sw_p > 0.05:
                self.logger.info(f"✓ Residuals are normally distributed (Shapiro-Wilk p={sw_p:.4f})")
            else:
                self.logger.warning(f"⚠ Residuals deviate from normality (Shapiro-Wilk p={sw_p:.4f})")
        
        # Heteroscedasticity assessment  
        if 'white_test' in residual_results and 'error' not in residual_results['white_test']:
            white_p = residual_results['white_test']['lm_p_value']
            if white_p > 0.05:
                self.logger.info(f"✓ Homoscedasticity assumption satisfied (White's test p={white_p:.4f})")
            else:
                self.logger.warning(f"⚠ Heteroscedasticity detected (White's test p={white_p:.4f})")
        
        # Residual statistics interpretation
        basic_stats = residual_results['basic_stats']
        mean_res = abs(basic_stats['mean_residual'])
        if mean_res < 0.1:
            self.logger.info(f"✓ Model shows minimal bias (mean residual = {mean_res:.4f})")
        else:
            self.logger.warning(f"⚠ Model shows some bias (mean residual = {mean_res:.4f})")
        
        # Skewness and kurtosis assessment
        skewness = abs(basic_stats['skewness'])
        kurtosis = abs(basic_stats['kurtosis'])
        
        if skewness < 0.5:
            self.logger.info(f"✓ Residuals are approximately symmetric (skewness = {basic_stats['skewness']:.3f})")
        else:
            self.logger.warning(f"⚠ Residuals show skewness (skewness = {basic_stats['skewness']:.3f})")
            
        if kurtosis < 3:
            self.logger.info(f"✓ Normal residual distribution shape (kurtosis = {kurtosis:.3f})")
        else:
            self.logger.warning(f"⚠ Heavy-tailed residual distribution (kurtosis = {kurtosis:.3f})")
        
        self.logger.info("Detailed residual plots saved to: ./evaluation_plots/")
        self.logger.info("Publication-ready model validation plots generated")
    
    def save_trained_model(self, model, preprocessor, performance_metrics: dict, 
                          model_name: str = None, version: str = None) -> str:
        """
        Save trained model with preprocessing pipeline and performance metrics.
        
        Args:
            model: Trained ML model
            preprocessor: Data preprocessing pipeline
            performance_metrics: Dictionary of model performance metrics
            model_name: Name for the model (auto-generated if None)
            version: Version string (auto-generated if None)
            
        Returns:
            Path to saved model artifact
        """
        try:
            # Generate model name if not provided
            if model_name is None:
                model_name = f"datect_{config.FORECAST_MODEL}_{config.FORECAST_TASK}"
            
            # Prepare metadata
            metadata = {
                'performance_metrics': performance_metrics,
                'task_type': config.FORECAST_TASK,
                'forecast_model': config.FORECAST_MODEL,
                'temporal_buffer_days': config.TEMPORAL_BUFFER_DAYS,
                'training_data_path': str(self.data_file),
                'training_samples': len(self.data) if self.data is not None else None,
                'sites_trained': list(self.data['site'].unique()) if self.data is not None else None,
                'date_range': {
                    'start': str(self.data['date'].min()) if self.data is not None else None,
                    'end': str(self.data['date'].max()) if self.data is not None else None
                }
            }
            
            # Save model artifact
            artifact_path = self.model_manager.save_model(
                model=model,
                preprocessor=preprocessor,
                metadata=metadata,
                model_name=model_name,
                version=version
            )
            
            self.logger.info(f"Model saved successfully: {model_name} -> {artifact_path}")
            return artifact_path
            
        except Exception as e:
            self.logger.error(f"Failed to save trained model: {str(e)}")
            raise ModelError(f"Model saving failed: {str(e)}", 
                           error_code="MODEL_SAVE_FAILED")
    
    def load_trained_model(self, model_name: str, version: str = "latest") -> Tuple[Any, Any, Dict]:
        """
        Load trained model with preprocessing pipeline and metadata.
        
        Args:
            model_name: Name of the model to load
            version: Version string or "latest"
            
        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        try:
            model, preprocessor, metadata = self.model_manager.load_model(model_name, version)
            
            self.logger.info(f"Model loaded successfully: {model_name} v{version}")
            self.logger.info(f"Model trained on {metadata.get('training_samples', 'unknown')} samples")
            
            # Log performance metrics if available
            if 'performance_metrics' in metadata:
                for metric, value in metadata['performance_metrics'].items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {metric}: {value:.4f}")
            
            return model, preprocessor, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {str(e)}")
            raise ModelError(f"Model loading failed: {str(e)}", 
                           error_code="MODEL_LOAD_FAILED")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available trained models and their versions.
        
        Returns:
            Dictionary mapping model names to list of versions
        """
        try:
            models = self.model_manager.list_models()
            
            self.logger.info(f"Found {len(models)} model types:")
            for model_name, versions in models.items():
                self.logger.info(f"  {model_name}: {len(versions)} versions")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return {}
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get detailed information about a saved model.
        
        Args:
            model_name: Name of the model
            version: Version string or "latest"
            
        Returns:
            Dictionary containing model information
        """
        try:
            info = self.model_manager.get_model_info(model_name, version)
            
            self.logger.info(f"Retrieved info for {model_name} v{version}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def predict_with_saved_model(self, model_name: str, data: pd.DataFrame, 
                                version: str = "latest") -> np.ndarray:
        """
        Make predictions using a saved model.
        
        Args:
            model_name: Name of the model to use
            data: Input data for prediction
            version: Version string or "latest"
            
        Returns:
            Array of predictions
        """
        try:
            # Load model
            model, preprocessor, metadata = self.load_trained_model(model_name, version)
            
            # Validate input data
            validate_data_integrity(
                data,
                required_columns=None,  # Will be validated by preprocessor
                min_rows=1,
                max_missing_rate=0.9
            )
            
            # Preprocess data if preprocessor available
            if preprocessor is not None:
                processed_data = preprocessor.transform(data)
            else:
                processed_data = data
            
            # Make predictions
            predictions = model.predict(processed_data)
            
            self.logger.info(f"Generated {len(predictions)} predictions using {model_name} v{version}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed with saved model: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}", 
                           error_code="PREDICTION_FAILED")