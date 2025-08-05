import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataProcessor:
    """Clean data processing that prevents data leakage."""
    
    def __init__(self):
        self.da_category_bins = [-float('inf'), 5, 20, 40, float('inf')]
        self.da_category_labels = [0, 1, 2, 3]
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and basic preparation of data."""
        data = pd.read_parquet(file_path)
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(['site', 'date'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features that don't cause leakage."""
        df = df.copy()
        day_of_year = df['date'].dt.dayofyear
        df['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'da', 
                          lags: list = [1, 2, 3]) -> pd.DataFrame:
        """Create lag features within training data only."""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('site')[target_col].shift(lag)
        return df
    
    def create_da_categories(self, da_values: pd.Series) -> pd.Series:
        """Create DA categories from continuous values."""
        return pd.cut(
            da_values,
            bins=self.da_category_bins,
            labels=self.da_category_labels,
            right=True
        ).astype('Int64')
    
    def get_train_test_split(self, df: pd.DataFrame, site: str, 
                           anchor_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data for a specific site and anchor date."""
        site_data = df[df['site'] == site].copy()
        
        train_data = site_data[site_data['date'] <= anchor_date].copy()
        test_data = site_data[site_data['date'] > anchor_date].copy()
        
        return train_data, test_data
    
    def prepare_training_data(self, train_df: pd.DataFrame, 
                            include_lags: bool = True) -> pd.DataFrame:
        """Prepare training data with proper feature engineering."""
        df = self.add_temporal_features(train_df)
        
        if include_lags:
            df = self.create_lag_features(df)
        
        # Create categories ONLY from training data
        df['da_category'] = self.create_da_categories(df['da'])
        
        # Drop rows with missing target or lag values
        required_cols = ['da', 'da_category']
        if include_lags:
            required_cols.extend([f'da_lag_{i}' for i in [1, 2, 3]])
        
        df = df.dropna(subset=required_cols)
        return df
    
    def prepare_forecast_data(self, test_df: pd.DataFrame, train_df: pd.DataFrame,
                            include_lags: bool = True) -> pd.DataFrame:
        """Prepare forecast data using only information available at forecast time."""
        df = self.add_temporal_features(test_df)
        
        if include_lags and not train_df.empty:
            # Create lag features using only training data history
            site = df['site'].iloc[0]
            train_site = train_df[train_df['site'] == site].sort_values('date')
            
            if len(train_site) >= 3:
                # Use last 3 values from training data for lags
                last_values = train_site['da'].tail(3).values
                df['da_lag_1'] = last_values[-1] if len(last_values) >= 1 else np.nan
                df['da_lag_2'] = last_values[-2] if len(last_values) >= 2 else np.nan  
                df['da_lag_3'] = last_values[-3] if len(last_values) >= 3 else np.nan
            else:
                df['da_lag_1'] = df['da_lag_2'] = df['da_lag_3'] = np.nan
        
        return df
    
    def get_feature_columns(self, include_lags: bool = True) -> list:
        """Get list of feature columns for modeling."""
        base_features = ['sin_day_of_year', 'cos_day_of_year']
        
        # Add any other oceanographic/climate features that exist
        # These would be features like SST, chlorophyll, etc.
        
        if include_lags:
            base_features.extend([f'da_lag_{i}' for i in [1, 2, 3]])
            
        return base_features
    
    def filter_features(self, df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
        """Filter to only modeling features plus site/date/targets."""
        feature_cols = self.get_feature_columns(include_lags)
        
        # Keep essential columns
        keep_cols = ['site', 'date', 'da', 'da_category'] + feature_cols
        
        # Add any oceanographic features that exist in the data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ocean_features = [col for col in numeric_cols 
                         if col not in keep_cols and 
                         col not in ['da', 'da_category'] and
                         not col.startswith('da_lag_')]
        
        keep_cols.extend(ocean_features)
        
        # Only keep columns that actually exist in the dataframe
        available_cols = [col for col in keep_cols if col in df.columns]
        
        return df[available_cols]