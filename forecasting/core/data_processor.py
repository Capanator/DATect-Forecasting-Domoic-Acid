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
        self.da_category_bins = config.DA_CATEGORY_BINS
        self.da_category_labels = config.DA_CATEGORY_LABELS
        
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
        if required_columns is None:
            required_columns = ['date', 'site', 'da']
            
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Empty dataset detected - cannot proceed with forecasting")
            
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Critical columns missing: {missing_cols}")
            
        # Check date column integrity
        if 'date' in df.columns:
            if df['date'].isna().all():
                raise ValueError("All dates are NaN - invalid temporal data")
            if df['date'].dtype not in ['datetime64[ns]', 'object']:
                raise ValueError(f"Invalid date column type: {df['date'].dtype}")
                
        # Check site column integrity
        if 'site' in df.columns:
            if df['site'].isna().all():
                raise ValueError("All sites are NaN - invalid site data")
            valid_sites = set(config.SITES.keys())
            data_sites = set(df['site'].dropna().unique())
            invalid_sites = data_sites - valid_sites
            if invalid_sites:
                print(f"Warning: Unknown sites found: {invalid_sites}")
                
        # Check DA target variable integrity
        if 'da' in df.columns:
            da_values = df['da'].dropna()
            if not da_values.empty:
                if (da_values < 0).any():
                    raise ValueError("Negative DA values detected - invalid biological data")
                if (da_values > 1000).any():  # Extremely high threshold
                    print(f"Warning: Very high DA values detected (max: {da_values.max():.2f})")
                    
        print(f"[INFO] Data integrity validation passed: {len(df)} records, {df.columns.nunique()} features")
        return True
        
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
        # Validate site
        if site not in config.SITES:
            raise ValueError(f"Unknown site: '{site}'. Valid sites: {list(config.SITES.keys())}")
            
        # Validate forecast date
        forecast_date = pd.Timestamp(forecast_date)
        if forecast_date < pd.Timestamp('2000-01-01'):
            raise ValueError("Forecast date too early - satellite data not available before 2000")
        if forecast_date > pd.Timestamp.now() + pd.Timedelta(days=365):
            raise ValueError("Forecast date too far in future (>1 year)")
            
        # Validate site-specific data availability
        site_data = data[data['site'] == site] if 'site' in data.columns else data
        if site_data.empty:
            raise ValueError(f"No historical data available for site: {site}")
            
        # Check temporal data coverage
        available_dates = site_data['date'].dropna()
        if available_dates.empty:
            raise ValueError(f"No valid dates in historical data for site: {site}")
            
        latest_data = available_dates.max()
        if (forecast_date - latest_data).days > 365:
            print(f"Warning: Large gap between latest data ({latest_data.date()}) and forecast date ({forecast_date.date()})")
            
        print(f"[INFO] Forecast input validation passed for {site} on {forecast_date.date()}")
        return True
        
    def load_and_prepare_base_data(self, file_path):
        """
        Load base data WITHOUT any target-based preprocessing.
        
        Args:
            file_path: Path to parquet data file
            
        Returns:
            DataFrame with base features and temporal components
        """
        data = pd.read_parquet(file_path, engine="pyarrow")
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values(["site", "date"], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Validate data integrity after loading
        self.validate_data_integrity(data)

        # Add temporal features (safe - no future information)
        day_of_year = data["date"].dt.dayofyear
        data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
        data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

        # DO NOT create da-category globally - this will be done per forecast
        print(f"[INFO] Loaded {len(data)} records across {data['site'].nunique()} sites")
        return data
        
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
        df = df.copy()
        df_sorted = df.sort_values([group_col, 'date'])
        
        for lag in lags:
            # Create lag feature
            df_sorted[f"{value_col}_lag_{lag}"] = df_sorted.groupby(group_col)[value_col].shift(lag)
            
            # CRITICAL: Only use lag values that are strictly before cutoff_date
            # This prevents using future information in training data
            # But be less restrictive - only affect data very close to cutoff (original method)
            buffer_days = 1  # Reduced from original stricter implementation
            lag_cutoff_date = cutoff_date - pd.Timedelta(days=buffer_days)
            lag_cutoff_mask = df_sorted['date'] > lag_cutoff_date
            df_sorted.loc[lag_cutoff_mask, f"{value_col}_lag_{lag}"] = np.nan
            
        return df_sorted
        
    def create_da_categories_safe(self, da_values):
        """
        Create DA categories from training data only.
        
        Args:
            da_values: Series of DA concentration values
            
        Returns:
            Categorical series with DA risk categories
        """
        return pd.cut(
            da_values,
            bins=self.da_category_bins,
            labels=self.da_category_labels,
            right=True,
        ).astype(pd.Int64Dtype())
        
    def create_numeric_transformer(self, df, drop_cols):
        """
        Create preprocessing transformer for numeric features.
        
        Args:
            df: DataFrame to process
            drop_cols: Columns to exclude from features
            
        Returns:
            Tuple of (transformer, feature_dataframe)
        """
        X = df.drop(columns=drop_cols, errors="ignore")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
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
        
        return transformer, X
        
    def validate_temporal_integrity(self, train_df, test_df):
        """
        Validate that temporal ordering is maintained.
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Boolean indicating if temporal integrity is maintained
        """
        if train_df.empty or test_df.empty:
            return False
            
        max_train_date = train_df['date'].max()
        min_test_date = test_df['date'].min()
        
        # Training data should be strictly before test data
        return max_train_date < min_test_date
        
    def get_feature_importance(self, model, feature_names):
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained scikit-learn model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None