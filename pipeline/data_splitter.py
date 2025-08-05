"""
Time series data splitting without leakage.
Handles train/test splits properly for forecasting scenarios.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class TimeSeriesSplitter:
    """Handles time series data splitting without leakage."""
    
    def __init__(self, date_col='date', site_col='site'):
        self.date_col = date_col
        self.site_col = site_col
    
    def split_by_date(self, data: pd.DataFrame, split_date: pd.Timestamp, 
                     site: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by date, optionally filtering by site."""
        if site is not None:
            data = data[data[self.site_col] == site].copy()
        
        data = data.sort_values([self.site_col, self.date_col]).reset_index(drop=True)
        
        train_data = data[data[self.date_col] < split_date].copy()
        test_data = data[data[self.date_col] >= split_date].copy()
        
        return train_data, test_data
    
    def create_forecast_split(self, data: pd.DataFrame, forecast_date: pd.Timestamp, 
                            site: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Timestamp]]:
        """Create proper train/forecast split for a specific site and date."""
        site_data = data[data[self.site_col] == site].copy()
        site_data = site_data.sort_values(self.date_col).reset_index(drop=True)
        
        # Training data: everything before forecast date
        train_data = site_data[site_data[self.date_col] < forecast_date].copy()
        
        # Find the actual next available date for testing
        future_data = site_data[site_data[self.date_col] >= forecast_date]
        
        if future_data.empty:
            # No future data available - create synthetic forecast row
            if train_data.empty:
                raise ValueError(f"No training data available for site {site} before {forecast_date}")
            
            last_row = train_data.iloc[-1].copy()
            forecast_row = pd.DataFrame([last_row])
            forecast_row[self.date_col] = forecast_date
            
            # Clear target variables for true forecast
            target_cols = ['da', 'da_category']
            for col in target_cols:
                if col in forecast_row.columns:
                    forecast_row[col] = np.nan
            
            test_date = None
        else:
            # Use actual next available data point
            forecast_row = future_data.iloc[:1].copy()
            test_date = forecast_row[self.date_col].iloc[0]
        
        return train_data, forecast_row, test_date


class RandomAnchorGenerator:
    """Generates random anchor points for evaluation."""
    
    def __init__(self, date_col='date', site_col='site', random_state=42):
        self.date_col = date_col
        self.site_col = site_col
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_anchors(self, data: pd.DataFrame, n_anchors_per_site: int = 100,
                        min_test_date: Optional[pd.Timestamp] = None) -> List[Tuple[str, pd.Timestamp]]:
        """Generate random anchor points for each site."""
        anchors = []
        
        for site in data[self.site_col].unique():
            site_data = data[data[self.site_col] == site]
            dates = sorted(site_data[self.date_col].unique())
            
            if len(dates) < 2:
                continue
            
            # Only use dates that have future data for testing
            valid_dates = dates[:-1]  # Exclude last date as it has no future
            
            if min_test_date is not None:
                valid_dates = [d for d in valid_dates 
                             if any(future_d >= min_test_date for future_d in dates if future_d > d)]
            
            if not valid_dates:
                continue
            
            # Randomly sample anchor dates
            n_samples = min(len(valid_dates), n_anchors_per_site)
            selected_dates = np.random.choice(valid_dates, size=n_samples, replace=False)
            
            anchors.extend([(site, pd.Timestamp(date)) for date in selected_dates])
        
        return anchors


class DataValidator:
    """Validates data splits to prevent leakage."""
    
    def __init__(self, date_col='date'):
        self.date_col = date_col
    
    def validate_split(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        """Validate that test data doesn't leak into training."""
        if train_data.empty or test_data.empty:
            return True
        
        max_train_date = train_data[self.date_col].max()
        min_test_date = test_data[self.date_col].min()
        
        if max_train_date >= min_test_date:
            print(f"WARNING: Data leakage detected! Max train date: {max_train_date}, "
                  f"Min test date: {min_test_date}")
            return False
        
        return True
    
    def check_feature_leakage(self, train_features: pd.DataFrame, 
                            test_features: pd.DataFrame, 
                            suspicious_cols: List[str] = None) -> bool:
        """Check for potential feature leakage."""
        if suspicious_cols is None:
            suspicious_cols = ['da', 'da_category']
        
        for col in suspicious_cols:
            if col in test_features.columns:
                if not test_features[col].isna().all():
                    print(f"WARNING: Potential feature leakage in column {col}")
                    return False
        
        return True