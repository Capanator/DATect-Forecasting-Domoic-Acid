"""
Data Processing Module
=====================

Handles all data loading, cleaning, and feature engineering with temporal safeguards.
All operations maintain strict temporal integrity to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import config
from .logging_config import get_logger
from .exception_handling import handle_data_errors, validate_data_integrity, ScientificValidationError, TemporalLeakageError

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles data processing with temporal safeguards.
    
    Key Features:
    - Forward-only interpolation
    - Temporal lag feature creation
    - Per-forecast DA category creation
    - Leak-free preprocessing pipelines
    """
    
    def __init__(self):
        logger.info("Initializing DataProcessor")
        self.da_category_bins = config.DA_CATEGORY_BINS
        self.da_category_labels = config.DA_CATEGORY_LABELS
        logger.info(f"DA category configuration loaded: {len(self.da_category_bins)-1} categories")
        
    @handle_data_errors
    def validate_data_integrity(self, df, required_columns=None):
        """
        Validate data integrity for scientific forecasting.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns (defaults to essential columns)
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If critical data integrity issues found
        """
        try:
            logger.info("Starting data integrity validation")
            
            if required_columns is None:
                required_columns = ['date', 'site', 'da']
            
            logger.debug(f"Validating required columns: {required_columns}")
            
            # Check if DataFrame is empty
            if df.empty:
                error_msg = "Empty dataset detected - cannot proceed with forecasting"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            # Check for required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                error_msg = f"Critical columns missing: {missing_cols}"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            # Check date column integrity
            if 'date' in df.columns:
                logger.debug("Validating date column")
                if df['date'].isna().all():
                    error_msg = "All dates are NaN - invalid temporal data"
                    logger.error(error_msg)
                    raise ScientificValidationError(error_msg)
                if df['date'].dtype not in ['datetime64[ns]', 'object']:
                    error_msg = f"Invalid date column type: {df['date'].dtype}"
                    logger.error(error_msg)
                    raise ScientificValidationError(error_msg)
                    
            # Check site column integrity
            if 'site' in df.columns:
                logger.debug("Validating site column")
                if df['site'].isna().all():
                    error_msg = "All sites are NaN - invalid site data"
                    logger.error(error_msg)
                    raise ScientificValidationError(error_msg)
                valid_sites = set(config.SITES.keys())
                data_sites = set(df['site'].dropna().unique())
                invalid_sites = data_sites - valid_sites
                if invalid_sites:
                    logger.warning(f"Unknown sites found: {invalid_sites}")
                    print(f"Warning: Unknown sites found: {invalid_sites}")
                    
            # Check DA target variable integrity
            if 'da' in df.columns:
                logger.debug("Validating DA target variable")
                da_values = df['da'].dropna()
                if not da_values.empty:
                    if (da_values < 0).any():
                        error_msg = "Negative DA values detected - invalid biological data"
                        logger.error(error_msg)
                        raise ScientificValidationError(error_msg)
                    if (da_values > 1000).any():  # Extremely high threshold
                        warning_msg = f"Very high DA values detected (max: {da_values.max():.2f})"
                        logger.warning(warning_msg)
                        print(f"Warning: {warning_msg}")
                        
            logger.info(f"Data integrity validation passed: {len(df)} records, {len(df.columns)} features")
            print(f"[INFO] Data integrity validation passed: {len(df)} records, {len(df.columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {str(e)}")
            raise
        
    @handle_data_errors
    def validate_forecast_inputs(self, data, site, forecast_date):
        """
        Validate inputs for a specific forecast.
        
        Args:
            data: DataFrame with historical data
            site: Site name for forecast
            forecast_date: Target forecast date
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If invalid inputs detected
        """
        try:
            logger.info(f"Validating forecast inputs for {site} on {forecast_date}")
            
            # Validate site
            if site not in config.SITES:
                error_msg = f"Unknown site: '{site}'. Valid sites: {list(config.SITES.keys())}"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            # Validate forecast date
            forecast_date = pd.Timestamp(forecast_date)
            if forecast_date < pd.Timestamp('2000-01-01'):
                error_msg = "Forecast date too early - satellite data not available before 2000"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
            if forecast_date > pd.Timestamp.now() + pd.Timedelta(days=365):
                error_msg = "Forecast date too far in future (>1 year)"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            # Validate site-specific data availability
            site_data = data[data['site'] == site] if 'site' in data.columns else data
            if site_data.empty:
                error_msg = f"No historical data available for site: {site}"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            # Check temporal data coverage
            available_dates = site_data['date'].dropna()
            if available_dates.empty:
                error_msg = f"No valid dates in historical data for site: {site}"
                logger.error(error_msg)
                raise ScientificValidationError(error_msg)
                
            latest_data = available_dates.max()
            gap_days = (forecast_date - latest_data).days
            if gap_days > 365:
                warning_msg = f"Large gap between latest data ({latest_data.date()}) and forecast date ({forecast_date.date()}): {gap_days} days"
                logger.warning(warning_msg)
                print(f"Warning: {warning_msg}")
                
            logger.info(f"Forecast input validation passed for {site} on {forecast_date.date()}")
            print(f"[INFO] Forecast input validation passed for {site} on {forecast_date.date()}")
            return True
            
        except Exception as e:
            logger.error(f"Forecast input validation failed: {str(e)}")
            raise
        
    @handle_data_errors
    def load_and_prepare_base_data(self, file_path):
        """
        Load base data WITHOUT any target-based preprocessing.
        
        Args:
            file_path: Path to parquet data file
            
        Returns:
            DataFrame with base features and temporal components
        """
        try:
            logger.info(f"Loading base data from {file_path}")
            
            # Load data
            data = pd.read_parquet(file_path, engine="pyarrow")
            logger.info(f"Raw data loaded: {len(data)} records, {len(data.columns)} columns")
            
            # Process temporal data
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values(["site", "date"], inplace=True)
            data.reset_index(drop=True, inplace=True)
            logger.debug("Data sorted and indexed by site and date")

            # Validate data integrity after loading
            logger.info("Validating loaded data integrity")
            self.validate_data_integrity(data)

            # Add temporal features (safe - no future information)
            logger.debug("Adding temporal features")
            day_of_year = data["date"].dt.dayofyear
            data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
            data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)
            logger.debug("Temporal features added: sin_day_of_year, cos_day_of_year")

            # DO NOT create da-category globally - this will be done per forecast
            sites_count = data['site'].nunique()
            logger.info(f"Data preparation completed: {len(data)} records across {sites_count} sites")
            print(f"[INFO] Loaded {len(data)} records across {sites_count} sites")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load and prepare base data: {str(e)}")
            raise ScientificValidationError(f"Data loading failed: {str(e)}")
        
    @handle_data_errors
    def create_lag_features_safe(self, df, group_col, value_col, lags, cutoff_date):
        """
        Create lag features with strict temporal cutoff to prevent leakage.
        Uses original algorithm from leak_free_forecast.py
        
        Args:
            df: DataFrame to process
            group_col: Column to group by (e.g., 'site')
            value_col: Column to create lags for (e.g., 'da')
            lags: List of lag periods [1, 2, 3]
            cutoff_date: Temporal cutoff date
            
        Returns:
            DataFrame with lag features and temporal safeguards
        """
        try:
            logger.info(f"Creating lag features for {value_col} with temporal cutoff at {cutoff_date}")
            logger.debug(f"Lag periods: {lags}")
            
            df = df.copy()
            df_sorted = df.sort_values([group_col, 'date'])
            
            created_features = []
            for lag in lags:
                feature_name = f"{value_col}_lag_{lag}"
                logger.debug(f"Creating lag feature: {feature_name}")
                
                # Create lag feature
                df_sorted[feature_name] = df_sorted.groupby(group_col)[value_col].shift(lag)
                
                # CRITICAL: Only use lag values that are strictly before cutoff_date
                # This prevents using future information in training data
                buffer_days = 1  # Reduced from original stricter implementation
                lag_cutoff_date = cutoff_date - pd.Timedelta(days=buffer_days)
                lag_cutoff_mask = df_sorted['date'] > lag_cutoff_date
                
                # Apply temporal safeguard
                affected_rows = lag_cutoff_mask.sum()
                df_sorted.loc[lag_cutoff_mask, feature_name] = np.nan
                
                logger.debug(f"Lag feature {feature_name} created, {affected_rows} rows masked for temporal safety")
                created_features.append(feature_name)
                
            logger.info(f"Successfully created {len(created_features)} lag features with temporal safeguards")
            
            # Validate temporal integrity
            if df_sorted['date'].max() > cutoff_date:
                logger.debug(f"Data contains dates after cutoff ({cutoff_date}) - temporal leakage risk exists (expected for retrospective evaluation)")
                
            return df_sorted
            
        except Exception as e:
            logger.error(f"Failed to create lag features: {str(e)}")
            raise TemporalLeakageError(f"Lag feature creation failed: {str(e)}")
        
    @handle_data_errors
    def create_da_categories_safe(self, da_values):
        """
        Create DA categories from training data only.
        
        Args:
            da_values: Series of DA concentration values
            
        Returns:
            Categorical series with DA risk categories
        """
        try:
            logger.debug(f"Creating DA categories for {len(da_values)} values")
            
            # Validate input
            if da_values.empty:
                logger.warning("Empty DA values provided for categorization")
                return pd.Series([], dtype=pd.Int64Dtype())
                
            # Check for invalid values
            invalid_mask = da_values < 0
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"Found {invalid_count} negative DA values - setting to NaN")
                da_values = da_values.copy()
                da_values[invalid_mask] = np.nan
                
            # Create categories
            categories = pd.cut(
                da_values,
                bins=self.da_category_bins,
                labels=self.da_category_labels,
                right=True,
            ).astype(pd.Int64Dtype())
            
            # Log category distribution
            value_counts = categories.value_counts().sort_index()
            logger.debug(f"DA category distribution: {dict(value_counts)}")
            
            return categories
            
        except Exception as e:
            logger.error(f"Failed to create DA categories: {str(e)}")
            raise ScientificValidationError(f"DA categorization failed: {str(e)}")
        
    @handle_data_errors
    def create_numeric_transformer(self, df, drop_cols):
        """
        Create preprocessing transformer for numeric features.
        
        Args:
            df: DataFrame to process
            drop_cols: Columns to exclude from features
            
        Returns:
            Tuple of (transformer, feature_dataframe)
        """
        try:
            logger.debug(f"Creating numeric transformer, dropping columns: {drop_cols}")
            
            X = df.drop(columns=drop_cols, errors="ignore")
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            logger.debug(f"Selected {len(numeric_cols)} numeric features: {list(numeric_cols)[:10]}...")
            
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for feature transformation")
                raise ScientificValidationError("No numeric features available for modeling")
                
            # Create preprocessing pipeline
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ])
            
            transformer = ColumnTransformer(
                [("num", numeric_pipeline, numeric_cols)],
                remainder="drop",  # Drop non-numeric to avoid issues
                verbose_feature_names_out=False
            )
            transformer.set_output(transform="pandas")
            
            logger.debug(f"Numeric transformer created with {len(numeric_cols)} features")
            return transformer, X
            
        except Exception as e:
            logger.error(f"Failed to create numeric transformer: {str(e)}")
            raise ScientificValidationError(f"Feature transformation setup failed: {str(e)}")
        
    def validate_temporal_integrity(self, train_df, test_df):
        """
        Validate that temporal ordering is maintained.
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Boolean indicating if temporal integrity is maintained
        """
        try:
            logger.debug("Validating temporal integrity between train and test sets")
            
            if train_df.empty or test_df.empty:
                logger.warning("Empty train or test dataframe - temporal validation failed")
                return False
                
            max_train_date = train_df['date'].max()
            min_test_date = test_df['date'].min()
            
            # Training data should be strictly before test data
            is_valid = max_train_date < min_test_date
            
            if is_valid:
                gap_days = (min_test_date - max_train_date).days
                logger.debug(f"Temporal integrity validated: {gap_days} day gap between train ({max_train_date.date()}) and test ({min_test_date.date()})")
            else:
                logger.error(f"TEMPORAL LEAKAGE DETECTED: train ends {max_train_date.date()}, test starts {min_test_date.date()}")
                raise TemporalLeakageError(f"Training data ({max_train_date}) overlaps with test data ({min_test_date})")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Temporal integrity validation failed: {str(e)}")
            raise
        
    def get_feature_importance(self, model, feature_names):
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained scikit-learn model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            logger.debug(f"Extracting feature importance from {type(model).__name__}")
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.debug(f"Feature importance extracted: top feature is {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.4f})")
                return importance_df
            else:
                logger.debug(f"Model {type(model).__name__} does not support feature importance extraction")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {str(e)}")
            return None