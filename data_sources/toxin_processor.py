"""
Toxin Data Processor
===================

Handles processing of toxin measurement data including:
- Domoic Acid (DA) concentrations
- Pseudo-nitzschia (PN) cell counts

This module provides:
- Automated processing of Parquet toxin files
- Weekly temporal aggregation using ISO week format
- Flexible column detection and data validation
- Site name normalization and mapping
- Comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from forecasting.core.logging_config import get_logger
from forecasting.core.exception_handling import handle_data_errors, validate_data_integrity
import config

logger = get_logger(__name__)


class ToxinProcessor:
    """
    Processes toxin measurement data for the DATect forecasting system.
    
    Handles both Domoic Acid (DA) concentrations and Pseudo-nitzschia (PN) 
    cell counts with flexible column detection and temporal aggregation.
    """
    
    def __init__(self):
        """Initialize toxin processor."""
        logger.info("Initialized ToxinProcessor")
    
    @handle_data_errors
    def process_da(self, da_files_dict: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Process DA data from Parquet files, returns weekly aggregated DataFrame.
        
        Args:
            da_files_dict: Dictionary mapping site keys to DA Parquet file paths
            
        Returns:
            DataFrame with Year-Week, DA_Levels, Site columns or None on failure
        """
        logger.info("Starting DA data processing")
        print("\\n--- Processing DA Data ---")
        
        data_frames = []
        processed_sites = 0
        failed_sites = 0
        
        for name, path in da_files_dict.items():
            try:
                # Normalize site name from dict key
                site_name_guess = self._normalize_site_name(name, 'da')
                logger.debug(f"Processing DA file: {name} -> {site_name_guess}")
                
                # Process individual DA file
                weekly_da = self._process_single_da_file(path, name, site_name_guess)
                
                if weekly_da is not None and not weekly_da.empty:
                    data_frames.append(weekly_da[['Year-Week', 'DA_Levels', 'Site']])
                    processed_sites += 1
                    print(f"    Successfully processed {len(weekly_da)} weekly DA records for {name}.")
                    logger.info(f"Processed DA data for {name}: {len(weekly_da)} records")
                else:
                    failed_sites += 1
                    logger.warning(f"No valid DA data processed for {name}")
                    
            except Exception as e:
                failed_sites += 1
                error_msg = f"Error processing DA file {name} ({os.path.basename(path)}): {e}"
                print(f"  {error_msg}")
                logger.error(error_msg)
        
        # Combine all processed data
        if not data_frames:
            logger.error("No DA data was successfully processed")
            return None
        
        print("Combining all processed DA data...")
        final_da_df = pd.concat(data_frames, ignore_index=True)
        
        # Final aggregation to handle potential duplicates from different files
        if not final_da_df.empty:
            final_da_df = final_da_df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()
        
        logger.info(f"DA processing complete: {processed_sites} successful, {failed_sites} failed")
        print(f"Combined DA data shape: {final_da_df.shape}")
        
        return final_da_df
    
    def _process_single_da_file(self, path: str, name: str, site_name: str) -> Optional[pd.DataFrame]:
        """
        Process a single DA Parquet file.
        
        Args:
            path: File path to DA Parquet file
            name: Site identifier from configuration
            site_name: Normalized site name
            
        Returns:
            Weekly aggregated DA DataFrame or None on failure
        """
        try:
            # Validate file exists
            if not os.path.exists(path):
                logger.error(f"DA file not found: {path}")
                return None
            
            # Load Parquet file
            df = pd.read_parquet(path)
            logger.debug(f"Loaded DA file {name} with shape {df.shape}")
            
            # Validate basic structure
            is_valid, error_msg = validate_data_integrity(df, [], min_rows=1)
            if not is_valid:
                logger.error(f"DA file validation failed for {name}: {error_msg}")
                return None
            
            # Identify Date and DA columns
            date_col, da_col = self._identify_da_columns(df, name)
            if date_col is None or da_col is None:
                return None
            
            # Process date and DA values
            df = self._process_da_columns(df, date_col, da_col, site_name)
            
            if df.empty:
                logger.warning(f"No valid DA records after processing for {name}")
                return None
            
            # Aggregate to weekly data
            weekly_da = self._aggregate_weekly_da(df)
            
            return weekly_da
            
        except Exception as e:
            logger.error(f"Error processing DA file {name}: {e}")
            return None
    
    def _identify_da_columns(self, df: pd.DataFrame, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify date and DA columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            name: Site name for logging
            
        Returns:
            Tuple of (date_column, da_column) names or (None, None)
        """
        date_col = None
        da_col = None
        
        # Identify date column
        if 'CollectDate' in df.columns:
            date_col = 'CollectDate'
        elif all(c in df.columns for c in ['Harvest Month', 'Harvest Date', 'Harvest Year']):
            # Create combined date from separate components
            try:
                df['CombinedDateStr'] = (df['Harvest Month'].astype(str) + " " + 
                                       df['Harvest Date'].astype(str) + ", " + 
                                       df['Harvest Year'].astype(str))
                df['Date'] = pd.to_datetime(df['CombinedDateStr'], format='%B %d, %Y', errors='coerce')
                date_col = 'Date'
                logger.debug(f"Created date column from harvest components for {name}")
            except Exception as e:
                logger.error(f"Error creating date column for {name}: {e}")
        
        # Identify DA column
        if 'Domoic Result' in df.columns:
            da_col = 'Domoic Result'
        elif 'Domoic Acid' in df.columns:
            da_col = 'Domoic Acid'
        
        # Log results
        if date_col is None:
            logger.error(f"No date column found for {name}. Available columns: {list(df.columns)}")
        else:
            logger.debug(f"Using date column '{date_col}' for {name}")
            
        if da_col is None:
            logger.error(f"No DA column found for {name}. Available columns: {list(df.columns)}")
        else:
            logger.debug(f"Using DA column '{da_col}' for {name}")
        
        return date_col, da_col
    
    def _process_da_columns(self, df: pd.DataFrame, date_col: str, 
                           da_col: str, site_name: str) -> pd.DataFrame:
        """
        Process date and DA columns with validation.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            da_col: DA column name
            site_name: Site name for the data
            
        Returns:
            Processed DataFrame with standardized columns
        """
        # Parse date column
        df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Parse DA levels (numeric)
        df['DA_Levels'] = pd.to_numeric(df[da_col], errors='coerce')
        
        # Add site information
        df['Site'] = site_name
        
        # Remove invalid records
        initial_count = len(df)
        df.dropna(subset=['Parsed_Date', 'DA_Levels', 'Site'], inplace=True)
        
        final_count = len(df)
        if initial_count > final_count:
            dropped = initial_count - final_count
            logger.debug(f"Dropped {dropped} invalid records for {site_name}")
        
        return df
    
    def _aggregate_weekly_da(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate DA data to weekly resolution using ISO week format.
        
        Args:
            df: DataFrame with Parsed_Date, DA_Levels, Site columns
            
        Returns:
            Weekly aggregated DataFrame
        """
        # Create ISO week identifier (YYYY-WW format)
        df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
        
        # Group by week and site, calculate mean DA levels
        weekly_da = df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()
        
        logger.debug(f"Aggregated {len(df)} daily records to {len(weekly_da)} weekly records")
        
        return weekly_da
    
    @handle_data_errors
    def process_pn(self, pn_files_dict: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Process Pseudo-nitzschia (PN) cell count data from Parquet files.
        
        Args:
            pn_files_dict: Dictionary mapping site keys to PN Parquet file paths
            
        Returns:
            DataFrame with Year-Week, PN_Levels, Site columns or None on failure
        """
        logger.info("Starting PN data processing")
        print("\\n--- Processing PN Data ---")
        
        data_frames = []
        processed_sites = 0
        failed_sites = 0
        
        for name, path in pn_files_dict.items():
            try:
                # Normalize site name from dict key
                site_name_guess = self._normalize_site_name(name, 'pn')
                logger.debug(f"Processing PN file: {name} -> {site_name_guess}")
                
                # Process individual PN file
                weekly_pn = self._process_single_pn_file(path, name, site_name_guess)
                
                if weekly_pn is not None and not weekly_pn.empty:
                    data_frames.append(weekly_pn[['Year-Week', 'PN_Levels', 'Site']])
                    processed_sites += 1
                    print(f"  Successfully processed {len(weekly_pn)} weekly PN records for {name}.")
                    logger.info(f"Processed PN data for {name}: {len(weekly_pn)} records")
                else:
                    failed_sites += 1
                    logger.warning(f"No valid PN data processed for {name}")
                    
            except Exception as e:
                failed_sites += 1
                error_msg = f"Error processing PN file {name} ({os.path.basename(path)}): {e}"
                print(f"  {error_msg}")
                logger.error(error_msg)
        
        # Combine all processed data
        if not data_frames:
            logger.error("No PN data was successfully processed")
            return None
        
        final_pn_df = pd.concat(data_frames, ignore_index=True)
        
        # Final aggregation to handle potential duplicates
        if not final_pn_df.empty:
            final_pn_df = final_pn_df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()
        
        logger.info(f"PN processing complete: {processed_sites} successful, {failed_sites} failed")
        print(f"Combined PN data shape: {final_pn_df.shape}")
        
        return final_pn_df
    
    def _process_single_pn_file(self, path: str, name: str, site_name: str) -> Optional[pd.DataFrame]:
        """
        Process a single PN Parquet file.
        
        Args:
            path: File path to PN Parquet file
            name: Site identifier from configuration
            site_name: Normalized site name
            
        Returns:
            Weekly aggregated PN DataFrame or None on failure
        """
        try:
            # Validate file exists
            if not os.path.exists(path):
                logger.error(f"PN file not found: {path}")
                return None
            
            # Load Parquet file
            df = pd.read_parquet(path)
            logger.debug(f"Loaded PN file {name} with shape {df.shape}")
            
            # Validate basic structure
            is_valid, error_msg = validate_data_integrity(df, [], min_rows=1)
            if not is_valid:
                logger.error(f"PN file validation failed for {name}: {error_msg}")
                return None
            
            # Identify Date and PN columns
            date_col, pn_col = self._identify_pn_columns(df, name)
            if date_col is None or pn_col is None:
                return None
            
            # Process date and PN values
            df = self._process_pn_columns(df, date_col, pn_col, site_name)
            
            if df.empty:
                logger.warning(f"No valid PN records after processing for {name}")
                return None
            
            # Aggregate to weekly data
            weekly_pn = self._aggregate_weekly_pn(df)
            
            return weekly_pn
            
        except Exception as e:
            logger.error(f"Error processing PN file {name}: {e}")
            return None
    
    def _identify_pn_columns(self, df: pd.DataFrame, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify date and PN columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            name: Site name for logging
            
        Returns:
            Tuple of (date_column, pn_column) names or (None, None)
        """
        # Find date column
        date_col_candidates = ['Date']
        date_col = next((c for c in date_col_candidates if c in df.columns), None)
        
        # Find PN column (look for columns containing "pseudo" and "nitzschia")
        pn_col_candidates = [c for c in df.columns 
                            if "pseudo" in str(c).lower() and "nitzschia" in str(c).lower()]
        
        pn_col = None
        if len(pn_col_candidates) == 1:
            pn_col = pn_col_candidates[0]
        elif len(pn_col_candidates) > 1:
            # If multiple candidates, prefer the first one but log warning
            pn_col = pn_col_candidates[0]
            logger.warning(f"Multiple PN column candidates for {name}: {pn_col_candidates}. Using: {pn_col}")
        
        # Log results
        if date_col is None:
            logger.error(f"No date column found for {name}. Available columns: {list(df.columns)}")
        else:
            logger.debug(f"Using date column '{date_col}' for {name}")
            
        if pn_col is None:
            logger.error(f"No PN column found for {name}. Available columns: {list(df.columns)}")
        else:
            logger.debug(f"Using PN column '{pn_col}' for {name}")
        
        return date_col, pn_col
    
    def _process_pn_columns(self, df: pd.DataFrame, date_col: str, 
                           pn_col: str, site_name: str) -> pd.DataFrame:
        """
        Process date and PN columns with validation.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            pn_col: PN column name
            site_name: Site name for the data
            
        Returns:
            Processed DataFrame with standardized columns
        """
        # Parse date column
        df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Parse PN levels (numeric cell counts)
        df['PN_Levels'] = pd.to_numeric(df[pn_col], errors='coerce')
        
        # Add site information
        df['Site'] = site_name
        
        # Remove invalid records
        initial_count = len(df)
        df.dropna(subset=['Parsed_Date', 'PN_Levels', 'Site'], inplace=True)
        
        final_count = len(df)
        if initial_count > final_count:
            dropped = initial_count - final_count
            logger.debug(f"Dropped {dropped} invalid PN records for {site_name}")
        
        return df
    
    def _aggregate_weekly_pn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate PN data to weekly resolution using ISO week format.
        
        Args:
            df: DataFrame with Parsed_Date, PN_Levels, Site columns
            
        Returns:
            Weekly aggregated DataFrame
        """
        # Create ISO week identifier (YYYY-WW format)
        df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
        
        # Group by week and site, calculate mean PN levels
        weekly_pn = df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()
        
        logger.debug(f"Aggregated {len(df)} daily PN records to {len(weekly_pn)} weekly records")
        
        return weekly_pn
    
    def _normalize_site_name(self, key_name: str, data_type: str) -> str:
        """
        Normalize site name from configuration key.
        
        Args:
            key_name: Original key name from configuration
            data_type: Data type ('da' or 'pn')
            
        Returns:
            Normalized site name in Title Case
        """
        # Remove data type suffixes and normalize
        normalized = (key_name
                     .replace(f'-{data_type}', '')
                     .replace(f'_{data_type}', '')
                     .replace('-', ' ')
                     .replace('_', ' ')
                     .title())
        
        return normalized
    
    def get_toxin_statistics(self, da_df: Optional[pd.DataFrame], 
                            pn_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate statistics for toxin data quality assessment.
        
        Args:
            da_df: DA DataFrame
            pn_df: PN DataFrame
            
        Returns:
            Dictionary with statistics for both data types
        """
        stats = {}
        
        # DA statistics
        if da_df is not None and not da_df.empty:
            stats['da'] = {
                'total_records': len(da_df),
                'unique_sites': da_df['Site'].nunique(),
                'unique_weeks': da_df['Year-Week'].nunique(),
                'mean_da_level': da_df['DA_Levels'].mean(),
                'median_da_level': da_df['DA_Levels'].median(),
                'max_da_level': da_df['DA_Levels'].max(),
                'sites': list(da_df['Site'].unique())
            }
        else:
            stats['da'] = {'total_records': 0, 'error': 'No DA data available'}
        
        # PN statistics
        if pn_df is not None and not pn_df.empty:
            stats['pn'] = {
                'total_records': len(pn_df),
                'unique_sites': pn_df['Site'].nunique(),
                'unique_weeks': pn_df['Year-Week'].nunique(),
                'mean_pn_level': pn_df['PN_Levels'].mean(),
                'median_pn_level': pn_df['PN_Levels'].median(),
                'max_pn_level': pn_df['PN_Levels'].max(),
                'sites': list(pn_df['Site'].unique())
            }
        else:
            stats['pn'] = {'total_records': 0, 'error': 'No PN data available'}
        
        logger.info(f"Toxin data statistics: DA={stats['da'].get('total_records', 0)} records, "
                   f"PN={stats['pn'].get('total_records', 0)} records")
        
        return stats
    
    def validate_toxin_data(self, da_df: Optional[pd.DataFrame], 
                           pn_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate toxin data quality and completeness.
        
        Args:
            da_df: DA DataFrame to validate
            pn_df: PN DataFrame to validate
            
        Returns:
            True if data passes validation, False otherwise
        """
        validation_passed = True
        
        # Validate DA data
        if da_df is not None and not da_df.empty:
            required_da_cols = ['Year-Week', 'DA_Levels', 'Site']
            is_valid, error_msg = validate_data_integrity(da_df, required_da_cols, min_rows=1)
            if not is_valid:
                logger.error(f"DA data validation failed: {error_msg}")
                validation_passed = False
            else:
                logger.info("DA data validation passed")
        else:
            logger.warning("No DA data available for validation")
        
        # Validate PN data
        if pn_df is not None and not pn_df.empty:
            required_pn_cols = ['Year-Week', 'PN_Levels', 'Site']
            is_valid, error_msg = validate_data_integrity(pn_df, required_pn_cols, min_rows=1)
            if not is_valid:
                logger.error(f"PN data validation failed: {error_msg}")
                validation_passed = False
            else:
                logger.info("PN data validation passed")
        else:
            logger.warning("No PN data available for validation")
        
        return validation_passed