"""
Streamflow Data Processor
=========================

Handles downloading and processing of USGS streamflow data for the Columbia River.
Provides daily discharge measurements that are used as environmental predictors
in the DATect forecasting system.

This module provides:
- Automated USGS streamflow data downloading
- JSON data parsing and validation  
- Daily discharge measurement processing
- Comprehensive error handling and logging
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from forecasting.core.secure_download import secure_download_file
from forecasting.core.validation import validate_url
from forecasting.core.logging_config import get_logger
from forecasting.core.exception_handling import handle_data_errors
import config

logger = get_logger(__name__)


class StreamflowProcessor:
    """
    Processes USGS streamflow data for the DATect forecasting system.
    
    Handles downloading and processing of daily streamflow measurements
    from USGS water services with comprehensive validation and error handling.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize streamflow processor.
        
        Args:
            temp_dir: Optional temporary directory for downloads
        """
        self.temp_dir = temp_dir or config.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        self.downloaded_files = []
        
        logger.info("Initialized StreamflowProcessor")
    
    def _generate_filename(self, url: str, extension: str) -> str:
        """Generate secure filename for download."""
        from forecasting.core.secure_download import generate_secure_filename
        return generate_secure_filename(url, extension, self.temp_dir)
    
    @handle_data_errors
    def process_streamflow(self, url: str) -> Optional[pd.DataFrame]:
        """
        Process USGS streamflow data (daily) with comprehensive error handling.
        
        Args:
            url: URL to download streamflow JSON data from
            
        Returns:
            DataFrame with Date and Flow columns, or None on failure
        """
        logger.info("Starting streamflow data processing")
        
        try:
            # Validate URL
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                logger.error(f"Invalid streamflow URL: {error_msg}")
                return None
            
            # Download file
            fname = self._generate_filename(url, '.json')
            logger.debug(f"Generated filename for streamflow: {fname}")
            
            downloaded_path = secure_download_file(url, fname)
            if downloaded_path is None:
                logger.error("Failed to download streamflow data")
                return None
                
            self.downloaded_files.append(downloaded_path)
            logger.info(f"Successfully downloaded streamflow data to {downloaded_path}")
            
            # Load and parse JSON data
            with open(downloaded_path) as f:
                data = json.load(f)
            
            logger.debug("Parsing USGS JSON data structure")
            
            # Extract time series values
            values = self._extract_discharge_values(data)
            
            # Parse records into DataFrame
            records = self._parse_streamflow_records(values)
            
            # Create and clean DataFrame
            df = pd.DataFrame(records)
            if df.empty:
                logger.warning("No valid streamflow records found")
                return pd.DataFrame(columns=['Date', 'Flow'])
            
            df = df.dropna(subset=['Date', 'Flow'])  # Remove invalid entries
            result = df[['Date', 'Flow']].sort_values('Date')
            
            logger.info(f"Successfully processed streamflow data: {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing streamflow data: {e}")
            return None
    
    def _extract_discharge_values(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract discharge values from USGS JSON structure.
        
        Args:
            data: Loaded JSON data from USGS water services
            
        Returns:
            List of value dictionaries containing discharge measurements
        """
        values = []
        ts_data = data.get('value', {}).get('timeSeries', [])
        
        if not ts_data:
            logger.warning("No timeSeries data found in USGS JSON")
            return values
        
        logger.debug(f"Found {len(ts_data)} time series in USGS data")
        
        # Find discharge time series (parameter code 00060 = discharge)
        discharge_ts = None
        for ts in ts_data:
            var_code = ts.get('variable', {}).get('variableCode', [{}])
            if var_code and var_code[0].get('value') == '00060':
                discharge_ts = ts
                logger.debug("Found discharge time series (parameter 00060)")
                break
        
        # Fall back to first time series if discharge not found specifically
        if discharge_ts is None and len(ts_data) == 1:
            discharge_ts = ts_data[0]
            logger.warning("Discharge parameter 00060 not found, using first time series")
        
        if discharge_ts:
            # Extract values array from time series
            values_data = discharge_ts.get('values', [{}])
            if values_data:
                values = values_data[0].get('value', [])
                logger.debug(f"Found {len(values)} streamflow measurements")
            else:
                logger.warning("No values array found in discharge time series")
        else:
            logger.error("Could not find any suitable discharge time series in USGS data")
        
        return values
    
    def _parse_streamflow_records(self, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse streamflow records from USGS values array.
        
        Args:
            values: List of value dictionaries from USGS API
            
        Returns:
            List of parsed records with Date and Flow columns
        """
        records = []
        invalid_count = 0
        negative_flow_count = 0
        
        for item in values:
            if not isinstance(item, dict) or 'dateTime' not in item or 'value' not in item:
                invalid_count += 1
                continue
            
            try:
                # Parse datetime (USGS provides UTC timestamps)
                dt = pd.to_datetime(item['dateTime'], utc=True)
                
                # Parse flow value (cubic feet per second)
                flow = pd.to_numeric(item['value'], errors='coerce')
                
                # Validate data quality
                if pd.isna(dt):
                    invalid_count += 1
                    continue
                
                if pd.isna(flow):
                    invalid_count += 1
                    continue
                
                if flow < 0:
                    negative_flow_count += 1
                    continue  # Skip negative flow values (measurement errors)
                
                # Add valid record
                records.append({
                    'Date': dt.tz_localize(None),  # Remove timezone for consistency
                    'Flow': flow
                })
                    
            except Exception as e:
                logger.debug(f"Error parsing streamflow record {item}: {e}")
                invalid_count += 1
        
        # Log data quality statistics
        if invalid_count > 0:
            logger.debug(f"Skipped {invalid_count} invalid streamflow records")
        if negative_flow_count > 0:
            logger.debug(f"Skipped {negative_flow_count} negative flow measurements")
        
        total_processed = len(records) + invalid_count + negative_flow_count
        if total_processed > 0:
            quality_pct = (len(records) / total_processed) * 100
            logger.info(f"Streamflow data quality: {quality_pct:.1f}% valid records")
        
        return records
    
    def get_streamflow_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate streamflow statistics for quality assessment.
        
        Args:
            df: Streamflow DataFrame with Date and Flow columns
            
        Returns:
            Dictionary with statistical metrics
        """
        if df.empty:
            return {}
        
        stats = {
            'count': len(df),
            'mean_flow': df['Flow'].mean(),
            'median_flow': df['Flow'].median(),
            'std_flow': df['Flow'].std(),
            'min_flow': df['Flow'].min(),
            'max_flow': df['Flow'].max(),
            'date_range_days': (df['Date'].max() - df['Date'].min()).days
        }
        
        # Add percentiles
        stats['flow_25th_pct'] = df['Flow'].quantile(0.25)
        stats['flow_75th_pct'] = df['Flow'].quantile(0.75)
        
        logger.info(f"Streamflow statistics: {stats['count']} records, "
                   f"mean flow: {stats['mean_flow']:.0f} cfs, "
                   f"date range: {stats['date_range_days']} days")
        
        return stats
    
    def validate_streamflow_data(self, df: pd.DataFrame) -> bool:
        """
        Validate streamflow data quality and completeness.
        
        Args:
            df: Streamflow DataFrame to validate
            
        Returns:
            True if data passes validation, False otherwise
        """
        if df.empty:
            logger.error("Streamflow data is empty")
            return False
        
        # Check required columns
        required_cols = ['Date', 'Flow']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values
        null_dates = df['Date'].isna().sum()
        null_flows = df['Flow'].isna().sum()
        
        if null_dates > 0:
            logger.error(f"Found {null_dates} null date values")
            return False
        
        if null_flows > 0:
            logger.error(f"Found {null_flows} null flow values")
            return False
        
        # Check date range continuity (warn but don't fail)
        date_gaps = df['Date'].diff().dt.days
        large_gaps = date_gaps[date_gaps > 7].count()  # More than 1 week gaps
        
        if large_gaps > 0:
            logger.warning(f"Found {large_gaps} date gaps larger than 7 days")
        
        # Check for reasonable flow values (0 to 1,000,000 cfs)
        unreasonable_flows = ((df['Flow'] < 0) | (df['Flow'] > 1_000_000)).sum()
        
        if unreasonable_flows > 0:
            logger.error(f"Found {unreasonable_flows} unreasonable flow values")
            return False
        
        logger.info("Streamflow data validation passed")
        return True
    
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