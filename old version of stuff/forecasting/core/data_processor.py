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