#!/usr/bin/env python3
"""
Enhanced Documentation Example
==============================

Example of enhanced documentation with type hints, detailed docstrings,
and comprehensive error handling for scientific computing.

This shows how to improve the existing codebase for publication readiness.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedDataProcessor:
    """
    Enhanced data processor with comprehensive type hints and documentation.
    
    This class demonstrates best practices for scientific computing code
    that will undergo peer review and publication.
    
    Attributes:
        da_category_bins (List[float]): Bin edges for DA risk categorization
        da_category_labels (List[int]): Numeric labels for DA categories
        logger (logging.Logger): Logger instance for this class
        
    Example:
        >>> processor = EnhancedDataProcessor()
        >>> data = processor.load_data("data.parquet")
        >>> processed = processor.create_lag_features(data, "site", "da", [1, 2, 3])
    """
    
    def __init__(self, 
                 da_category_bins: Optional[List[float]] = None,
                 da_category_labels: Optional[List[int]] = None) -> None:
        """
        Initialize enhanced data processor.
        
        Args:
            da_category_bins: Custom bin edges for DA categorization.
                            If None, uses default risk thresholds [0, 5, 20, 40, inf].
            da_category_labels: Custom numeric labels for categories.
                              If None, uses default [0, 1, 2, 3].
                              
        Raises:
            ValueError: If bins and labels have incompatible lengths.
        """
        # Default DA risk thresholds (μg/g)
        self.da_category_bins = da_category_bins or [-float("inf"), 5, 20, 40, float("inf")]
        self.da_category_labels = da_category_labels or [0, 1, 2, 3]
        
        # Validate configuration
        if len(self.da_category_labels) != len(self.da_category_bins) - 1:
            raise ValueError(
                f"Number of labels ({len(self.da_category_labels)}) must be "
                f"one less than number of bins ({len(self.da_category_bins)})"
            )
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Enhanced data processor initialized")
        
    def load_data(self, 
                  file_path: Union[str, Path],
                  required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and validate data from parquet file.
        
        Args:
            file_path: Path to parquet data file
            required_columns: List of columns that must be present.
                            If None, no column validation is performed.
                            
        Returns:
            Loaded and validated DataFrame with proper datetime conversion
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If required columns are missing
            pd.errors.EmptyDataError: If the file is empty or corrupted
            
        Example:
            >>> processor = EnhancedDataProcessor()
            >>> data = processor.load_data("data.parquet", ["date", "site", "da"])
        """
        file_path = Path(file_path)
        
        # Validate file existence
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data
            data = pd.read_parquet(file_path, engine="pyarrow")
            
            # Validate data is not empty
            if data.empty:
                raise pd.errors.EmptyDataError(f"Data file is empty: {file_path}")
                
            # Convert date column
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                
            # Validate required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    raise ValueError(
                        f"Required columns missing from {file_path}: {missing_cols}"
                    )
                    
            # Sort by site and date for temporal consistency
            if 'site' in data.columns and 'date' in data.columns:
                data = data.sort_values(['site', 'date']).reset_index(drop=True)
                
            self.logger.info(
                f"Loaded {len(data)} records across {data.get('site', pd.Series()).nunique()} sites"
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
            
    def create_lag_features_safe(self,
                                df: pd.DataFrame,
                                group_col: str,
                                value_col: str,
                                lags: List[int],
                                cutoff_date: pd.Timestamp,
                                buffer_days: int = 1) -> pd.DataFrame:
        """
        Create temporal lag features with strict data leakage prevention.
        
        This function implements critical temporal safeguards for scientific validity.
        It ensures that lag features cannot access future information that would
        not be available at prediction time.
        
        Args:
            df: Input DataFrame containing time series data
            group_col: Column name for grouping (e.g., 'site')  
            value_col: Column to create lag features from (e.g., 'da')
            lags: List of lag periods to create (e.g., [1, 2, 3])
            cutoff_date: Temporal cutoff for training data
            buffer_days: Additional buffer days near cutoff to prevent leakage
            
        Returns:
            DataFrame with lag features and temporal safeguards applied
            
        Raises:
            KeyError: If required columns are missing from DataFrame
            ValueError: If lags contains non-positive integers
            
        Note:
            This function implements the temporal safeguards described in:
            - Section 3.2 of the methodology paper
            - Temporal integrity requirements for peer review
            
        Example:
            >>> cutoff = pd.Timestamp('2020-06-01')
            >>> data_with_lags = processor.create_lag_features_safe(
            ...     data, 'site', 'da', [1, 2, 3], cutoff
            ... )
        """
        # Input validation
        required_cols = [group_col, value_col, 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Required columns missing: {missing_cols}")
            
        if not all(isinstance(lag, int) and lag > 0 for lag in lags):
            raise ValueError("All lag values must be positive integers")
            
        self.logger.info(
            f"Creating lag features {lags} for {value_col} with cutoff {cutoff_date}"
        )
        
        # Create working copy
        df_result = df.copy()
        df_sorted = df_result.sort_values([group_col, 'date'])
        
        # Create lag features with temporal safeguards
        for lag in lags:
            lag_col = f"{value_col}_lag_{lag}"
            
            # Create lag feature using pandas shift
            df_sorted[lag_col] = df_sorted.groupby(group_col)[value_col].shift(lag)
            
            # CRITICAL: Apply temporal cutoff to prevent data leakage
            # Set lag values to NaN for data points near the cutoff date
            lag_cutoff_date = cutoff_date - pd.Timedelta(days=buffer_days)
            temporal_mask = df_sorted['date'] > lag_cutoff_date
            
            # Count affected points for logging
            affected_points = temporal_mask.sum()
            
            # Apply temporal safeguard
            df_sorted.loc[temporal_mask, lag_col] = np.nan
            
            self.logger.debug(
                f"Applied temporal cutoff to lag_{lag}: {affected_points} points set to NaN"
            )
            
        self.logger.info(
            f"Created {len(lags)} lag features with temporal safeguards"
        )
        
        return df_sorted
        
    def validate_temporal_integrity(self,
                                  train_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  date_col: str = 'date') -> Tuple[bool, str]:
        """
        Validate that temporal ordering prevents data leakage.
        
        This validation is CRITICAL for peer review and scientific publication.
        It ensures that the train/test split maintains proper temporal ordering.
        
        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame  
            date_col: Name of the date column
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if temporal integrity is maintained
            - error_message: Description of any integrity violations
            
        Example:
            >>> is_valid, message = processor.validate_temporal_integrity(train_df, test_df)
            >>> if not is_valid:
            ...     raise TemporalLeakageError(message)
        """
        try:
            if train_df.empty:
                return False, "Training data is empty"
                
            if test_df.empty:
                return False, "Test data is empty"
                
            if date_col not in train_df.columns:
                return False, f"Date column '{date_col}' not found in training data"
                
            if date_col not in test_df.columns:
                return False, f"Date column '{date_col}' not found in test data"
                
            # Check temporal ordering
            max_train_date = train_df[date_col].max()
            min_test_date = test_df[date_col].min()
            
            if max_train_date >= min_test_date:
                return False, (
                    f"Temporal integrity violation: training data extends to {max_train_date} "
                    f"but test data starts at {min_test_date}"
                )
                
            # Calculate temporal gap
            temporal_gap = (min_test_date - max_train_date).days
            
            self.logger.info(
                f"Temporal integrity validated: {temporal_gap} day gap between train/test"
            )
            
            return True, f"Temporal integrity maintained with {temporal_gap} day gap"
            
        except Exception as e:
            error_msg = f"Error validating temporal integrity: {e}"
            self.logger.error(error_msg)
            return False, error_msg


# Example usage and demonstration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Demonstrate enhanced data processor
    processor = EnhancedDataProcessor()
    
    # Example of how type hints and documentation improve code quality
    print("✅ Enhanced documentation with type hints demonstrates:")
    print("  • Clear function signatures with expected types")
    print("  • Comprehensive docstrings with examples")
    print("  • Proper error handling and validation")
    print("  • Scientific methodology documentation")
    print("  • Peer review ready code structure")