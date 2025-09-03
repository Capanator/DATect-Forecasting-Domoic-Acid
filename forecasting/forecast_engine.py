"""
Leak-Free Forecasting Engine
Core forecasting with complete temporal integrity protection
"""

import pandas as pd
import numpy as np
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
        
        if validate_on_init:
            logger.info("Validating system startup configuration")
            validate_system_startup()
            logger.info("System startup validation completed successfully")
        
        logger.info("Initializing data processor and model factory")
        self.data_processor = DataProcessor()
        self.model_factory = ModelFactory()
        
        
        self.min_training_samples = max(1, int(getattr(config, 'MIN_TRAINING_SAMPLES', 5)))
        self.random_seed = config.RANDOM_SEED
        
        logger.info(f"Configuration: min_samples={self.min_training_samples}, seed={self.random_seed}")
        
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
        validate_runtime_parameters(n_anchors, min_test_date)
        
        logger.info(f"Running LEAK-FREE {task} evaluation with {model_type}")
        
        self.data = self.data_processor.load_and_prepare_base_data(self.data_file)
        min_target_date = pd.Timestamp(min_test_date)
        
        self.last_diagnostics = {
            "task": task,
            "model_type": model_type,
            "min_test_date": str(min_test_date),
            "per_site": {}
        }

        anchor_infos = []
        for site in self.data["site"].unique():
            self.last_diagnostics["per_site"][site] = {
                "candidate_dates": 0,
                "valid_future": 0,
                "selected": 0,
                "earliest_selected_date": None
            }
            site_dates = self.data[self.data["site"] == site]["date"].sort_values().unique()
            # Need enough data span to support the forecast horizon
            if len(site_dates) > 1:
                date_span_days = (site_dates[-1] - site_dates[0]).days
                if date_span_days >= config.FORECAST_HORIZON_DAYS * 2:  # At least 2x horizon for meaningful evaluation
                    # Only use dates that have sufficient history and future data
                    valid_anchors = []
                    for i, date in enumerate(site_dates[:-1]):  # Exclude last date
                        self.last_diagnostics["per_site"][site]["candidate_dates"] += 1
                        if date >= min_target_date:
                            # Check if there's a future date at the required forecast horizon
                            future_dates = site_dates[i+1:]
                            valid_future = [d for d in future_dates if (d - date).days >= config.FORECAST_HORIZON_DAYS]
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
        
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self._forecast_single_anchor_leak_free)(ai, self.data, min_target_date, task, model_type) 
            for ai in tqdm(anchor_infos, desc="Processing Leak-Free Anchors")
        )
        
        forecast_dfs = [df for df in results if df is not None]
        if not forecast_dfs:
            logger.warning("No successful forecasts")
            return None
            
        final_df = pd.concat(forecast_dfs, ignore_index=True)
        final_df = final_df.sort_values(["date", "site"]).drop_duplicates(["date", "site"])
        logger.info(f"Successfully processed {len(forecast_dfs)} leak-free forecasts")
        
        self.results_df = final_df
        
        self._display_evaluation_metrics(task)
        
        return final_df
        
    def _forecast_single_anchor_leak_free(self, anchor_info, full_data, min_target_date, task, model_type):
        """Process single anchor forecast with ZERO data leakage - original algorithm."""
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data["site"] == site].copy()
        site_data.sort_values("date", inplace=True)
        
        train_mask = site_data["date"] <= anchor_date
        
        # Calculate target forecast date based on configured horizon
        target_forecast_date = anchor_date + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        
        # Find test samples within reasonable range of target forecast date
        test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
        
        train_df = site_data[train_mask].copy()
        test_candidates = site_data[test_mask]
        
        if train_df.empty or test_candidates.empty:
            return None
        
        # Find the test sample closest to the target forecast date
        test_candidates = test_candidates.copy()
        test_candidates['date_diff'] = abs((test_candidates['date'] - target_forecast_date).dt.days)
        closest_idx = test_candidates['date_diff'].idxmin()
        test_df = test_candidates.loc[[closest_idx]].copy()
        test_date = test_df["date"].iloc[0]
        
        site_data_with_lags = self.data_processor.create_lag_features_safe(
            site_data, "site", "da", config.LAG_FEATURES, anchor_date
        )

        train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].copy()
        test_df = site_data_with_lags[site_data_with_lags["date"] == test_date].copy()

        # Add derived persistence features (leak-safe)
        try:
            self._add_safe_persistence_features(train_df, test_df, site_data, anchor_date)
        except Exception:
            pass
        
        if train_df.empty or test_df.empty:
            return None
        
        train_df_clean = train_df.dropna(subset=["da"]).copy()
        if train_df_clean.empty or len(train_df_clean) < self.min_training_samples:
            return None
        
        train_df_clean["da-category"] = self.data_processor.create_da_categories_safe(train_df_clean["da"])
        train_df = train_df_clean
        
        base_drop_cols = ["date", "site", "da"]
        train_drop_cols = base_drop_cols + ["da-category"]
        test_drop_cols = base_drop_cols
        
        transformer, X_train = self.data_processor.create_numeric_transformer(train_df, train_drop_cols)
        X_test = test_df.drop(columns=[col for col in test_drop_cols if col in test_df.columns])
        
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # CRITICAL: Validate temporal safety before fitting transformer
        self.data_processor.validate_transformer_temporal_safety(
            transformer, train_df, test_df, anchor_date
        )
        
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        if pd.isna(train_df["da"]).any():
            return None
        
        actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
        actual_category = self.data_processor.create_da_categories_safe(pd.Series([actual_da]))[0] if actual_da is not None else None
        
        result = {
            'date': test_date,
            'site': site,
            'anchor_date': anchor_date,
            'actual_da': actual_da,
            'actual_category': actual_category
        }
        
        if task == "regression" or task == "both":
            reg_model = self.model_factory.get_model("regression", model_type)
            
            y_train = train_df["da"]
            # Spike weighting (emphasize high DA levels)
            spike_threshold = getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0)
            spike_mult = float(getattr(config, 'SPIKE_WEIGHT_MULT', 12.0))
            pre_spike_mult = float(getattr(config, 'PRE_SPIKE_WEIGHT_MULT', 10.0))
            
            sample_weights = np.ones(len(y_train), dtype=float)
            spike_mask = y_train >= spike_threshold
            sample_weights[spike_mask] *= spike_mult

            # Pre-spike onset weighting: below-threshold points that precede a spike within the
            # configured window (computed strictly within training data to avoid leakage)
            try:
                pre_spike_mask = self._compute_pre_spike_mask(train_df, anchor_date)
                sample_weights[pre_spike_mask] *= pre_spike_mult
            except Exception:
                # Be robust if any edge case occurs; fall back to spike-only weighting
                pass
            
            if model_type in ["xgboost", "xgb"]:
                reg_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                reg_model.fit(X_train_processed, y_train)
            
            pred_da = reg_model.predict(X_test_processed)[0]
            pred_da = max(0.0, float(pred_da))
            result['predicted_da'] = pred_da

            # Optional: train an onset classifier (imminent spike in next N days)
            try:
                onset_prob = self._train_and_predict_onset_probability(
                    train_df, X_train_processed, X_test_processed, site_data, anchor_date, model_type
                )
                result['onset_prob'] = float(onset_prob)
            except Exception:
                # Be robust: if we cannot compute onset probability, skip silently
                pass
        
        if task == "classification" or task == "both":
            unique_classes = train_df["da-category"].nunique()
            if unique_classes > 1:
                unique_cats = sorted(train_df["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                y_train_encoded = train_df["da-category"].map(cat_mapping)
                
                cls_model = self.model_factory.get_model("classification", model_type)
                cls_model.fit(X_train_processed, y_train_encoded)
                pred_encoded = cls_model.predict(X_test_processed)[0]
                
                pred_category = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(pred_category)
            else:
                dominant_class = train_df["da-category"].mode()[0]
                result['predicted_category'] = int(dominant_class)
                result['single_class_prediction'] = True
        
        return pd.DataFrame([result])

    def _compute_pre_spike_mask(self, train_df: pd.DataFrame, anchor_date: pd.Timestamp) -> np.ndarray:
        """
        Identify below-threshold training points that precede a spike within a
        configured window. Uses ONLY data up to the anchor_date to avoid leakage.

        Returns a boolean mask aligned to train_df index order.
        """
        if train_df.empty:
            return np.zeros(0, dtype=bool)

        threshold = float(getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0))
        window_days = int(getattr(config, 'PRE_SPIKE_WINDOW_DAYS', 7))

        df = train_df[["date", "da"]].copy()
        df = df.dropna(subset=["date", "da"]).copy()
        # Ensure strictly within training period
        df = df[df["date"] <= anchor_date]
        if df.empty:
            return np.zeros(len(train_df), dtype=bool)

        df_sorted = df.sort_values("date").reset_index()  # preserve original indices
        n = len(df_sorted)
        pre_spike_local = np.zeros(n, dtype=bool)

        for i in range(n):
            di = df_sorted.loc[i, "date"]
            yi = df_sorted.loc[i, "da"]
            if not pd.notna(yi) or yi >= threshold:
                continue
            j = i + 1
            # Scan forward within window, bounded by anchor_date
            while j < n:
                dj = df_sorted.loc[j, "date"]
                if (dj - di).days > window_days or dj > anchor_date:
                    break
                yj = df_sorted.loc[j, "da"]
                if pd.notna(yj) and yj >= threshold:
                    pre_spike_local[i] = True
                    break
                j += 1

        # Map back to train_df order
        mask_series = pd.Series(False, index=train_df.index)
        for k in range(n):
            if pre_spike_local[k]:
                orig_idx = df_sorted.loc[k, "index"]
                if orig_idx in mask_series.index:
                    mask_series.loc[orig_idx] = True

        return mask_series.values

    def _add_safe_persistence_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, site_data: pd.DataFrame, anchor_date: pd.Timestamp) -> None:
        """
        Add leak-safe persistence features that improve R²/MAE while preserving temporal integrity.
        - For training rows (<= anchor_date), use standard lag features present on each row
        - For the single test row (> anchor_date), substitute with last known values up to anchor_date
        """
        if not getattr(config, 'USE_DERIVED_LAG_FEATURES', True):
            return
        settings = getattr(config, 'DERIVED_LAG_SETTINGS', {})
        lags = getattr(config, 'LAG_FEATURES', [])

        def has(col, df):
            return col in df.columns

        # Determine last known DA at anchor
        hist = site_data[(site_data['date'] <= anchor_date) & (site_data['da'].notna())].sort_values('date')
        last_da = float(hist.iloc[-1]['da']) if not hist.empty else np.nan

        # Build training features from available lags on each row
        if settings.get('use_last3_mean', True):
            cols = [c for c in [
                'da_lag_1' if 1 in lags else None,
                'da_lag_2' if 2 in lags else None,
                'da_lag_3' if 3 in lags else None,
            ] if c is not None and has(c, train_df)]
            if cols:
                train_df['last3_mean_safe'] = train_df[cols].mean(axis=1)
                test_df['last3_mean_safe'] = np.nanmean([last_da, last_da, last_da]) if not np.isnan(last_da) else np.nan

        if settings.get('use_weekly_change', True):
            if 1 in lags and 7 in lags and has('da_lag_1', train_df) and has('da_lag_7', train_df):
                train_df['weekly_change_safe'] = train_df['da_lag_1'] - train_df['da_lag_7']
                test_df['weekly_change_safe'] = 0.0  # no future; assume persistence baseline

        if settings.get('use_biweekly_change', True):
            if 1 in lags and 14 in lags and has('da_lag_1', train_df) and has('da_lag_14', train_df):
                train_df['biweekly_change_safe'] = train_df['da_lag_1'] - train_df['da_lag_14']
                test_df['biweekly_change_safe'] = 0.0

        if settings.get('use_rising_flag', True):
            cond_cols = [c for c in ['da_lag_1', 'da_lag_2', 'da_lag_3'] if has(c, train_df)]
            if set(cond_cols) == set(['da_lag_1', 'da_lag_2', 'da_lag_3']):
                train_df['rising_flag_safe'] = ((train_df['da_lag_1'] > train_df['da_lag_2']) & (train_df['da_lag_2'] > train_df['da_lag_3'])).astype(int)
                # For test, we cannot know; use 1 if last known da is positive trend vs its own prior
                test_df['rising_flag_safe'] = 0


    def _train_and_predict_onset_probability(
        self,
        train_df: pd.DataFrame,
        X_train_processed,
        X_test_processed,
        site_data: pd.DataFrame,
        anchor_date: pd.Timestamp,
        model_type: str,
    ) -> float:
        """
        Train a leak-free classifier for imminent spike (>= threshold in next ONSET_WINDOW_DAYS)
        using only data up to the anchor_date. Returns predicted probability for the test row.
        """
        threshold = float(getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0))
        window_days = int(getattr(config, 'ONSET_WINDOW_DAYS', 7))
        pos_weight = float(getattr(config, 'ONSET_POS_WEIGHT', 6.0))

        # Build labels for each training row t: does a spike occur within (t, t+window] and <= anchor_date?
        df = train_df[['date', 'da']].copy().dropna()
        df.sort_values('date', inplace=True)

        labels = []
        for _, r in df.iterrows():
            t = r['date']
            # Only consider up to anchor_date
            end = min(anchor_date, t + pd.Timedelta(days=window_days))
            future = site_data[(site_data['date'] > t) & (site_data['date'] <= end)][['date', 'da']].dropna()
            labels.append(1 if ((not future.empty) and (future['da'].max() >= threshold)) else 0)

        if len(labels) == 0 or sum(labels) == 0:
            # No positive examples; return low probability
            return 0.0

        # Align labels with X_train_processed rows length
        # X_train_processed corresponds to train_df rows after transformer creation
        # Ensure we only use rows present in X_train_processed index
        X_tr = X_train_processed
        # Recompute labels aligned to X_tr index positions
        # We will map train_df indices to labels
        label_map = {}
        df_labeled = df.reset_index()
        for i, lab in enumerate(labels):
            label_map[df_labeled.loc[i, 'index']] = lab
        y_labels = pd.Series(0, index=X_tr.index)
        for idx in X_tr.index:
            y_labels.loc[idx] = label_map.get(idx, 0)

        # Train classifier (use XGBoost if available)
        cls_model = self.model_factory.get_model('classification', model_type)
        sw = np.ones(len(y_labels), dtype=float)
        sw[y_labels.values.astype(int) == 1] *= pos_weight
        if hasattr(cls_model, 'fit'):
            try:
                cls_model.fit(X_tr, y_labels.values.astype(int), sample_weight=sw)
            except TypeError:
                cls_model.fit(X_tr, y_labels.values.astype(int))

        if hasattr(cls_model, 'predict_proba'):
            prob = float(cls_model.predict_proba(X_test_processed)[0, 1])
        else:
            # Fallback using decision_function or predicted class
            pred = cls_model.predict(X_test_processed)[0]
            prob = float(pred)

        return prob
    
    def generate_bootstrap_confidence_intervals(self, X_train_processed, y_train, X_forecast, model_type, n_bootstrap=100):
        """
        Generate bootstrap confidence intervals using resampling.
        
        Args:
            X_train_processed: Processed training features
            y_train: Training targets
            X_forecast: Processed forecast features
            model_type: Type of model to use
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with quantile predictions
        """
        predictions = []
        
        # Generate bootstrap predictions
        for _ in range(n_bootstrap):
            # Bootstrap resample with replacement
            n_samples = len(X_train_processed)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Handle both DataFrame and numpy array cases
            if hasattr(X_train_processed, 'iloc'):
                X_bootstrap = X_train_processed.iloc[bootstrap_indices]
                y_bootstrap = y_train.iloc[bootstrap_indices]
            else:
                X_bootstrap = X_train_processed[bootstrap_indices]
                y_bootstrap = y_train[bootstrap_indices] if hasattr(y_train, '__getitem__') else y_train.iloc[bootstrap_indices]
            
            # Train model on bootstrap sample
            bootstrap_model = self.model_factory.get_model("regression", model_type)
            
            # Apply spike weighting if XGBoost
            if model_type in ["xgboost", "xgb"]:
                spike_threshold = getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0)
                spike_mult = float(getattr(config, 'SPIKE_WEIGHT_MULT', 12.0))
                sample_weights = np.ones(len(y_bootstrap), dtype=float)
                spike_mask = y_bootstrap >= spike_threshold
                sample_weights[spike_mask] *= spike_mult
                bootstrap_model.fit(X_bootstrap, y_bootstrap, sample_weight=sample_weights)
            else:
                bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Make prediction
            pred = bootstrap_model.predict(X_forecast)[0]
            pred = max(0.0, float(pred))
            predictions.append(pred)
        
        # Calculate quantiles
        predictions = np.array(predictions)
        return {
            "q05": float(np.percentile(predictions, 5)),
            "q50": float(np.percentile(predictions, 50)),
            "q95": float(np.percentile(predictions, 95)),
            "bootstrap_predictions": predictions.tolist()
        }

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
        data = self.data_processor.load_and_prepare_base_data(data_path)
        forecast_date = pd.Timestamp(forecast_date)
        
        self.data_processor.validate_forecast_inputs(data, site, forecast_date)
        
        df_site = data[data['site'] == site].copy()
        df_site.sort_values('date', inplace=True)
        
        # Calculate target anchor date based on forecast horizon
        target_anchor_date = forecast_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        
        available_before = df_site[df_site['date'] < forecast_date]
        if available_before.empty:
            return None
        
        # Find the available data point closest to our target anchor date    
        available_before = available_before.copy()
        available_before['anchor_diff'] = abs((available_before['date'] - target_anchor_date).dt.days)
        closest_idx = available_before['anchor_diff'].idxmin()
        anchor_date = available_before.loc[closest_idx, 'date']
        
        df_site_with_lags = self.data_processor.create_lag_features_safe(
            df_site, "site", "da", config.LAG_FEATURES, anchor_date
        )
        
        df_train = df_site_with_lags[df_site_with_lags['date'] <= anchor_date].copy()
        df_train_clean = df_train.dropna(subset=['da']).copy()
        
        if df_train_clean.empty or len(df_train_clean) < self.min_training_samples:
            return None
        
        df_train_clean["da-category"] = self.data_processor.create_da_categories_safe(df_train_clean["da"])
        
        drop_cols = ["date", "site", "da", "da-category"]
        transformer, X_train = self.data_processor.create_numeric_transformer(df_train_clean, drop_cols)
        
        # CRITICAL: Validate temporal safety (single forecast uses all training data)
        if 'date' in df_train_clean.columns:
            future_data = df_train_clean[df_train_clean['date'] > anchor_date]
            if not future_data.empty:
                logger.error(f"TEMPORAL LEAKAGE in single forecast: {len(future_data)} records after anchor")
                raise ValueError(f"Training data contains future data after {anchor_date}")
            logger.debug(f"Single forecast temporal safety: all training data ≤ {anchor_date}")
        
        X_train_processed = transformer.fit_transform(X_train)
        
        latest_data = df_train_clean.iloc[-1:].copy()
        X_forecast = transformer.transform(latest_data.drop(columns=drop_cols, errors='ignore'))
        
        result = {
            'forecast_date': forecast_date,
            'anchor_date': anchor_date,
            'site': site,
            'task': task,
            'model_type': model_type,
            'training_samples': len(df_train_clean)
        }
        
        if task == "regression":
            model = self.model_factory.get_model("regression", model_type)
            
            y_train = df_train_clean["da"]
            spike_mask = y_train > 15.0  # spike threshold
            sample_weights = np.ones(len(y_train))
            sample_weights[spike_mask] *= 8.0  # precision weight for spikes
            
            if model_type in ["xgboost", "xgb"]:
                model.fit(X_train_processed, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train)
            
            prediction = model.predict(X_forecast)[0]
            prediction = max(0.0, float(prediction))
            result['predicted_da'] = prediction
            result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
            
            # Generate bootstrap confidence intervals for regression tasks
            if len(df_train_clean) >= 10:  # Only if we have enough data for meaningful bootstrap
                bootstrap_quantiles = self.generate_bootstrap_confidence_intervals(
                    X_train_processed, y_train, X_forecast, model_type, n_bootstrap=20
                )
                result['bootstrap_quantiles'] = bootstrap_quantiles
                logger.debug(f"Bootstrap confidence intervals: q05={bootstrap_quantiles['q05']:.3f}, q50={bootstrap_quantiles['q50']:.3f}, q95={bootstrap_quantiles['q95']:.3f}")
            
            logger.debug(f"Regression prediction completed for {site}: {prediction:.4f}")
            
        elif task == "classification":
            unique_classes = df_train_clean["da-category"].nunique()
            if unique_classes > 1:
                unique_cats = sorted(df_train_clean["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                y_train_encoded = df_train_clean["da-category"].map(cat_mapping)
                
                model = self.model_factory.get_model("classification", model_type)
                model.fit(X_train_processed, y_train_encoded)
                pred_encoded = model.predict(X_forecast)[0]
                
                prediction = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(prediction)
                result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
                logger.debug(f"Classification prediction completed for {site}: {prediction}")
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_forecast)[0]
                    prob_array = [0.0, 0.0, 0.0, 0.0]
                    for i, prob in enumerate(probabilities):
                        original_cat = reverse_mapping[i]
                        prob_array[original_cat] = float(prob)
                    result['class_probabilities'] = prob_array
                        
            else:
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
            
        logger.info(f"Successfully processed {len(self.results_df)} forecasts")
        
        if task == "regression" or task == "both":
            valid_results = self.results_df.dropna(subset=['actual_da', 'predicted_da'])
            if not valid_results.empty:
                r2 = r2_score(valid_results['actual_da'], valid_results['predicted_da'])
                mae = mean_absolute_error(valid_results['actual_da'], valid_results['predicted_da'])
                
                spike_threshold = float(getattr(config, 'SPIKE_THRESHOLD_PPM', 20.0))
                y_true_binary = (valid_results['actual_da'] > spike_threshold).astype(int)
                y_pred_binary = (valid_results['predicted_da'] > spike_threshold).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                logger.info(f"Regression Metrics:")
                logger.info(f"  R2: {r2:.4f}, MAE: {mae:.4f}")
                logger.info(f"  Spike Detection (>{spike_threshold}): Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            else:
                logger.warning("No valid regression results for evaluation")
                
        if task == "classification" or task == "both":
            valid_results = self.results_df.dropna(subset=['actual_category', 'predicted_category'])
            if not valid_results.empty:
                y_true = valid_results['actual_category']
                y_pred = valid_results['predicted_category']
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                logger.info(f"Classification Metrics:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
            else:
                logger.warning("No valid classification results for evaluation")
    
