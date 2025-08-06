"""
Climate Data Processor
======================

Handles downloading and processing of climate indices including:
- Pacific Decadal Oscillation (PDO)
- Oceanic Niño Index (ONI)  
- Biologically Effective Upwelling Transport Index (BEUTI)

This module provides:
- Automated climate data downloading with validation
- Temporal aggregation and processing
- Site-specific interpolation for BEUTI data
- Comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from forecasting.core.secure_download import secure_download_file
from forecasting.core.validation import validate_url, sanitize_filename
from forecasting.core.logging_config import get_logger
from forecasting.core.exception_handling import handle_data_errors, safe_execute
import config

logger = get_logger(__name__)


class ClimateProcessor:
    """
    Processes climate index data for the DATect forecasting system.
    
    Handles PDO, ONI, and BEUTI data with temporal aggregation and
    site-specific interpolation capabilities.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize climate processor.
        
        Args:
            temp_dir: Optional temporary directory for downloads
        """
        self.temp_dir = temp_dir or config.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        self.downloaded_files = []
        
        logger.info("Initialized ClimateProcessor")
    
    def _generate_filename(self, url: str, extension: str) -> str:
        """Generate secure filename for download."""
        from forecasting.core.secure_download import generate_secure_filename
        return generate_secure_filename(url, extension, self.temp_dir)
    
    @handle_data_errors
    def fetch_climate_index(self, url: str, var_name: str) -> Optional[pd.DataFrame]:
        """
        Process climate index data (PDO, ONI) with comprehensive error handling.
        
        Args:
            url: URL to download climate index data from
            var_name: Variable name to extract (e.g., 'PDO', 'ONI')
            
        Returns:
            DataFrame with Month and index columns, or None on failure
        """
        logger.info(f"Starting climate index processing: {var_name}")
        
        try:
            # Validate URL
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                logger.error(f"Invalid URL for {var_name}: {error_msg}")
                return None
            
            # Download file
            fname = self._generate_filename(url, '.nc')
            logger.debug(f"Generated filename for {var_name}: {fname}")
            
            downloaded_path = secure_download_file(url, fname)
            if downloaded_path is None:
                logger.error(f"Failed to download {var_name} data")
                return None
                
            self.downloaded_files.append(downloaded_path)
            logger.info(f"Successfully downloaded {var_name} data to {downloaded_path}")
                    
            # Open and process dataset
            logger.debug(f"Opening dataset: {downloaded_path}")
            ds = xr.open_dataset(downloaded_path)
            df = ds.to_dataframe().reset_index()
            logger.debug(f"Dataset shape for {var_name}: {df.shape}")
            
            # Find time column
            time_col = self._find_time_column(df, var_name)
            if time_col is None:
                ds.close()
                return None
            
            # Find variable column
            actual_var_name = self._find_variable_column(df, var_name)
            if actual_var_name is None:
                ds.close()
                return None
            
            # Process and aggregate data
            result = self._process_climate_data(df, time_col, actual_var_name, var_name)
            
            ds.close()
            logger.info(f"Successfully processed {var_name}: {len(result) if result is not None else 0} monthly records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing climate index {var_name}: {e}")
            return None
    
    def _find_time_column(self, df: pd.DataFrame, var_name: str) -> Optional[str]:
        """
        Find the time column in the dataset.
        
        Args:
            df: Input DataFrame
            var_name: Variable name for error messages
            
        Returns:
            Time column name or None if not found
        """
        time_cols = ['time', 'datetime', 'Date', 'T']
        time_col = next((c for c in time_cols if c in df.columns), None)
        
        if time_col is None:
            logger.error(f"No time column found in {var_name} dataset. Available columns: {list(df.columns)}")
            return None
            
        logger.debug(f"Using time column '{time_col}' for {var_name}")
        return time_col
    
    def _find_variable_column(self, df: pd.DataFrame, var_name: str) -> Optional[str]:
        """
        Find the variable column in the dataset (case-insensitive).
        
        Args:
            df: Input DataFrame
            var_name: Variable name to find
            
        Returns:
            Actual variable column name or None if not found
        """
        actual_var_name = var_name
        
        # First try exact match
        if actual_var_name not in df.columns:
            # Try case-insensitive match
            var_name_lower = var_name.lower()
            found_var = next((c for c in df.columns if c.lower() == var_name_lower), None)
            actual_var_name = found_var or var_name
            
        if actual_var_name not in df.columns:
            logger.error(f"Variable '{var_name}' not found in dataset. Available columns: {list(df.columns)}")
            return None
            
        logger.debug(f"Using variable column '{actual_var_name}' for {var_name}")
        return actual_var_name
    
    def _process_climate_data(self, df: pd.DataFrame, time_col: str, 
                             var_col: str, var_name: str) -> pd.DataFrame:
        """
        Process climate data with temporal aggregation.
        
        Args:
            df: Input DataFrame
            time_col: Time column name
            var_col: Variable column name  
            var_name: Variable name for logging
            
        Returns:
            Processed DataFrame with monthly aggregation
        """
        # Convert datetime and clean data
        df['datetime'] = pd.to_datetime(df[time_col])
        df_clean = df[['datetime', var_col]].dropna().rename(columns={var_col: 'index'})
        
        # Aggregate monthly (mean values)
        df_clean['Month'] = df_clean['datetime'].dt.to_period('M')
        result = df_clean.groupby('Month')['index'].mean().reset_index()
        
        return result[['Month', 'index']].sort_values('Month')
    
    @handle_data_errors
    def fetch_beuti_data(self, url: str, sites_dict: Dict[str, List[float]], 
                        power: float = 2.0) -> Optional[pd.DataFrame]:
        """
        Process BEUTI data with site-specific interpolation.
        
        Args:
            url: URL to download BEUTI data from
            sites_dict: Dictionary of site names to [lat, lon] coordinates
            power: Power parameter for inverse distance weighting
            
        Returns:
            DataFrame with Date, Site, and beuti columns, or None on failure
        """
        logger.info("Starting BEUTI data processing")
        
        try:
            # Validate URL
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                logger.error(f"Invalid BEUTI URL: {error_msg}")
                return None
            
            # Download file
            fname = self._generate_filename(url, '.nc')
            logger.debug(f"Generated filename for BEUTI: {fname}")
            
            downloaded_path = secure_download_file(url, fname)
            if downloaded_path is None:
                logger.error("Failed to download BEUTI data")
                return None
                
            self.downloaded_files.append(downloaded_path)
            logger.info(f"Successfully downloaded BEUTI data to {downloaded_path}")
            
            # Process data
            ds = xr.open_dataset(downloaded_path)
            df = ds.to_dataframe().reset_index()
            logger.debug(f"BEUTI dataset shape: {df.shape}")
            
            # Find required columns
            column_mapping = self._identify_beuti_columns(df, ds)
            if not all(column_mapping.values()):
                ds.close()
                return None
            
            # Prepare DataFrame for interpolation
            df_subset = self._prepare_beuti_data(df, column_mapping)
            
            # Interpolate for each site
            results_list = self._interpolate_beuti_for_sites(df_subset, sites_dict, power)
            
            # Create final DataFrame
            beuti_final_df = pd.DataFrame(results_list)
            if not beuti_final_df.empty:
                beuti_final_df['Date'] = pd.to_datetime(beuti_final_df['Date'])
                result = beuti_final_df[['Date', 'Site', 'beuti']].sort_values(['Site', 'Date'])
            else:
                result = pd.DataFrame(columns=['Date', 'Site', 'beuti'])
            
            ds.close()
            logger.info(f"Successfully processed BEUTI data: {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing BEUTI data: {e}")
            return None
    
    def _identify_beuti_columns(self, df: pd.DataFrame, 
                               ds: xr.Dataset) -> Dict[str, Optional[str]]:
        """
        Identify required columns in BEUTI dataset.
        
        Args:
            df: Input DataFrame
            ds: xarray Dataset
            
        Returns:
            Dictionary mapping column types to actual column names
        """
        # Find time column
        time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] 
                        if c in df.columns), None)
        
        # Find latitude column
        lat_col = next((c for c in ['latitude', 'lat'] 
                       if c in df.columns), None)
        
        # Find BEUTI variable
        beuti_var = next((c for c in ['BEUTI', 'beuti'] 
                         if c in df.columns or c in ds.data_vars), None)
        
        column_mapping = {
            'time': time_col,
            'latitude': lat_col,
            'beuti': beuti_var
        }
        
        # Log missing columns
        for col_type, col_name in column_mapping.items():
            if col_name is None:
                logger.error(f"Could not find {col_type} column in BEUTI dataset")
            else:
                logger.debug(f"Using {col_type} column: {col_name}")
        
        return column_mapping
    
    def _prepare_beuti_data(self, df: pd.DataFrame, 
                           column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Prepare BEUTI data for interpolation.
        
        Args:
            df: Input DataFrame
            column_mapping: Column name mapping
            
        Returns:
            Prepared DataFrame with standardized columns
        """
        time_col, lat_col, beuti_var = (
            column_mapping['time'],
            column_mapping['latitude'], 
            column_mapping['beuti']
        )
        
        # Create subset with required columns
        df_subset = df[[time_col, lat_col, beuti_var]].copy()
        df_subset.rename(columns={
            time_col: 'Date', 
            lat_col: 'lat', 
            beuti_var: 'beuti'
        }, inplace=True)
        
        # Clean and process data
        df_subset['Date'] = pd.to_datetime(df_subset['Date']).dt.date
        df_subset = df_subset.dropna(subset=['Date', 'lat', 'beuti'])
        
        # Sort for efficient processing
        df_sorted = df_subset.sort_values(by=['Date', 'lat'])
        
        logger.debug(f"Prepared BEUTI data with {len(df_sorted)} records")
        return df_sorted
    
    def _interpolate_beuti_for_sites(self, df_sorted: pd.DataFrame, 
                                    sites_dict: Dict[str, List[float]], 
                                    power: float) -> List[Dict[str, Any]]:
        """
        Interpolate BEUTI values for each monitoring site.
        
        Args:
            df_sorted: Sorted BEUTI data
            sites_dict: Site coordinates dictionary
            power: Power parameter for inverse distance weighting
            
        Returns:
            List of records with interpolated BEUTI values
        """
        results_list = []
        
        for site, coords in sites_dict.items():
            logger.debug(f"Interpolating BEUTI for site: {site}")
            
            # Get site latitude
            site_lat = coords[0] if isinstance(coords, (list, tuple)) and coords else np.nan
            
            if pd.isna(site_lat):
                logger.warning(f"Invalid latitude for site {site}: {coords}")
                continue
            
            site_results = []
            
            # Group by date to interpolate for each day
            for date, group in df_sorted.groupby('Date'):
                lats = group['lat'].values
                beuti_vals = group['beuti'].values
                
                # Interpolate BEUTI value for this site and date
                interpolated_beuti = self._interpolate_single_point(
                    lats, beuti_vals, site_lat, power
                )
                
                if pd.notna(interpolated_beuti):
                    site_results.append({
                        'Date': date, 
                        'Site': site, 
                        'beuti': interpolated_beuti
                    })
            
            if site_results:
                results_list.extend(site_results)
                logger.debug(f"Interpolated {len(site_results)} BEUTI records for {site}")
        
        return results_list
    
    def _interpolate_single_point(self, lats: np.ndarray, beuti_vals: np.ndarray, 
                                 site_lat: float, power: float) -> float:
        """
        Interpolate BEUTI value for a single site and date.
        
        Args:
            lats: Array of latitude values
            beuti_vals: Array of BEUTI values
            site_lat: Target site latitude
            power: Power parameter for inverse distance weighting
            
        Returns:
            Interpolated BEUTI value
        """
        # Check for exact match first
        exact_match_indices = np.where(np.isclose(lats, site_lat))[0]
        if exact_match_indices.size > 0:
            return np.mean(beuti_vals[exact_match_indices])
        
        # Inverse distance weighting
        distances = np.abs(lats - site_lat)
        weights = 1.0 / (distances ** power + 1e-9)  # Small epsilon to avoid division by zero
        
        # Filter valid values
        valid_indices = ~np.isnan(weights) & ~np.isnan(beuti_vals)
        if np.any(valid_indices):
            valid_weights = weights[valid_indices]
            valid_beuti = beuti_vals[valid_indices]
            return np.sum(valid_beuti * valid_weights) / np.sum(valid_weights)
        else:
            return np.nan
    
    def process_all_climate_data(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Process all climate indices (PDO, ONI, BEUTI) from configuration.
        
        Returns:
            Dictionary with processed climate data
        """
        logger.info("Processing all climate data sources")
        
        results = {}
        
        # Process PDO
        if hasattr(config, 'PDO_URL'):
            results['PDO'] = self.fetch_climate_index(config.PDO_URL, 'PDO')
        
        # Process ONI
        if hasattr(config, 'ONI_URL'):
            results['ONI'] = self.fetch_climate_index(config.ONI_URL, 'ONI')
        
        # Process BEUTI
        if hasattr(config, 'BEUTI_URL') and hasattr(config, 'SITES'):
            results['BEUTI'] = self.fetch_beuti_data(config.BEUTI_URL, config.SITES)
        
        # Log results
        for name, data in results.items():
            if data is not None:
                logger.info(f"Successfully processed {name}: {len(data)} records")
            else:
                logger.warning(f"Failed to process {name}")
        
        return results
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        for file_path in self.downloaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up file {file_path}: {e}")
        
        self.downloaded_files.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_temp_files()

