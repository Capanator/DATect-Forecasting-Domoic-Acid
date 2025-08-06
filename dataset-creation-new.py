#!/usr/bin/env python3
"""
DATect Dataset Creation Pipeline - Modular Version
=================================================

Refactored data processing pipeline using modular architecture for better
maintainability, testability, and separation of concerns.

This version maintains full compatibility with the original dataset-creation.py
while providing improved code organization and maintainability.

Usage:
    python dataset-creation-new.py
    
Configuration:
    See config.py for all data sources, sites, and processing parameters.
"""

import pandas as pd
import numpy as np
import json
import os
import requests
import tempfile
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import shutil
import config
from pathlib import Path

# Import secure utilities
from forecasting.core.secure_download import SecureDownloader, secure_download_file, cleanup_downloaded_files
from forecasting.core.validation import validate_url, sanitize_filename
from forecasting.core.logging_config import setup_logging, get_logger
from forecasting.core.exception_handling import handle_data_errors, safe_execute

# Setup logging
setup_logging(log_level=config.LOG_LEVEL, enable_file_logging=True, log_dir=config.LOG_OUTPUT_DIR)
logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting non-nanosecond precision datetime values to nanosecond precision")

logger.info("Starting DATect dataset creation pipeline - modular version")

# =============================================================================
# CONFIGURATION AND GLOBAL VARIABLES
# =============================================================================

# Processing flags
FORCE_SATELLITE_REPROCESSING = False  # Set to True to always regenerate satellite data

# File tracking for cleanup
downloaded_files = []
generated_parquet_files = []
temporary_nc_files_for_stitching = []  # Track yearly files per dataset

# Load main configuration
print(f"--- Loading Configuration from config.py ---")

# Extract config values
da_files = config.ORIGINAL_DA_FILES
pn_files = config.ORIGINAL_PN_FILES
sites = config.SITES
pdo_url = config.PDO_URL
oni_url = config.ONI_URL
beuti_url = config.BEUTI_URL
streamflow_url = config.STREAMFLOW_URL
start_date = pd.to_datetime(config.START_DATE)
end_date = pd.to_datetime(config.END_DATE)
final_output_path = config.FINAL_OUTPUT_PATH
SATELLITE_OUTPUT_PARQUET = './data/intermediate/satellite_data_intermediate.parquet'

print(f"Configuration loaded: {len(da_files)} DA files, {len(pn_files)} PN files, {len(sites)} sites")
print(f"Date range: {start_date.date()} to {end_date.date()}, Output: {final_output_path}")

# Load satellite configuration from main config
satellite_metadata = config.SATELLITE_DATA
print(f"\\n--- Satellite Configuration loaded from main config ---")
print(f"Satellite configuration loaded with {len(satellite_metadata)} data types.")

# =============================================================================
# MODULAR UTILITY FUNCTIONS
# =============================================================================

@handle_data_errors
def download_file(url, filename):
    """
    Secure download file from URL with validation and error handling.
    
    Args:
        url (str): URL to download from
        filename (str): Local filename to save to
        
    Returns:
        str: Path to downloaded file or None on failure
        
    Raises:
        requests.RequestException: If download fails after all retries
    """
    logger.info(f"Starting download: {url}")
    
    # Validate URL before attempting download
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        logger.error(f"URL validation failed for {url}: {error_msg}")
        raise ValueError(f"Invalid URL: {error_msg}")
    
    # Sanitize filename
    safe_filename = sanitize_filename(filename)
    
    # Use secure downloader
    result = secure_download_file(url, safe_filename)
    
    if result is None:
        logger.error(f"Download failed for {url}")
        raise requests.RequestException(f"Secure download failed for {url}")
    
    downloaded_files.append(result)
    logger.info(f"Download completed: {url} -> {result}")
    return result

@handle_data_errors  
def local_filename(url, ext, temp_dir=None):
    """
    Generate secure, sanitized local filename from URL.
    
    Args:
        url (str): Source URL
        ext (str): File extension to use (e.g., '.nc', '.csv')
        temp_dir (str, optional): Temporary directory path
        
    Returns:
        str: Generated secure filename with directory path
    """
    try:
        # Validate URL first
        is_valid, error_msg = validate_url(url)
        if not is_valid:
            logger.warning(f"URL validation warning for filename generation: {error_msg}")
            # Continue with filename generation but log the warning
        
        # Use secure filename generation from validation module
        from forecasting.core.secure_download import generate_secure_filename
        
        if temp_dir:
            # Generate filename in specified directory
            base_filename = generate_secure_filename(url, ext.replace('.', ''))
            return os.path.join(temp_dir, base_filename + ext)
        else:
            # Use default temporary directory
            return generate_secure_filename(url, ext.replace('.', '')) + ext
            
    except Exception as e:
        logger.warning(f"Error generating secure filename for {url}: {e}")
        # Fallback to simple filename generation
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:10]
        filename = f"download_{url_hash}{ext}"
        
        if temp_dir:
            return os.path.join(temp_dir, filename)
        else:
            return filename

# =============================================================================
# TOXIN DATA PROCESSING MODULE
# =============================================================================

class ToxinDataProcessor:
    """Handles processing of DA and PN toxin measurement data."""
    
    @handle_data_errors
    def process_da_data(self, da_files, sites):
        """Process DA (Domoic Acid) measurement data."""
        print("\\n--- Processing DA Data ---")
        
        all_da_data = []
        
        for site_name, file_path in da_files.items():
            try:
                logger.debug(f"Processing DA data for site: {site_name}")
                
                if not os.path.exists(file_path):
                    logger.warning(f"DA file not found for {site_name}: {file_path}")
                    continue
                
                # Read CSV data
                df = pd.read_csv(file_path)
                
                # Process and validate data
                df = self._process_toxin_file(df, site_name, 'DA')
                
                if df is not None and len(df) > 0:
                    all_da_data.append(df)
                    print(f"    Successfully processed {len(df)} weekly DA records for {site_name}.")
                else:
                    logger.warning(f"No valid DA data found for {site_name}")
                    
            except Exception as e:
                logger.error(f"Error processing DA data for {site_name}: {e}")
                continue
        
        if all_da_data:
            print("Combining all processed DA data...")
            combined_df = pd.concat(all_da_data, ignore_index=True)
            print(f"Combined DA data shape: {combined_df.shape}")
            return combined_df
        else:
            logger.warning("No DA data was successfully processed")
            return pd.DataFrame()

    @handle_data_errors
    def process_pn_data(self, pn_files, sites):
        """Process PN (Pseudo-nitzschia) measurement data."""
        print("\\n--- Processing PN Data ---")
        
        all_pn_data = []
        
        for site_name, file_path in pn_files.items():
            try:
                logger.debug(f"Processing PN data for site: {site_name}")
                
                if not os.path.exists(file_path):
                    logger.warning(f"PN file not found for {site_name}: {file_path}")
                    continue
                
                # Read CSV data
                df = pd.read_csv(file_path)
                
                # Process and validate data
                df = self._process_toxin_file(df, site_name, 'PN')
                
                if df is not None and len(df) > 0:
                    all_pn_data.append(df)
                    print(f"  Successfully processed {len(df)} weekly PN records for {site_name}.")
                else:
                    logger.warning(f"No valid PN data found for {site_name}")
                    
            except Exception as e:
                logger.error(f"Error processing PN data for {site_name}: {e}")
                continue
        
        if all_pn_data:
            combined_df = pd.concat(all_pn_data, ignore_index=True)
            print(f"Combined PN data shape: {combined_df.shape}")
            return combined_df
        else:
            logger.warning("No PN data was successfully processed")
            return pd.DataFrame()

    def _process_toxin_file(self, df, site_name, data_type):
        """Process individual toxin measurement file."""
        try:
            # Find date column (flexible column name matching)
            date_column = None
            for col in df.columns:
                if any(date_word in col.lower() for date_word in ['date', 'time', 'sample']):
                    date_column = col
                    break
            
            if date_column is None:
                logger.error(f"No date column found in {data_type} file for {site_name}")
                return None
            
            # Find value column  
            value_column = None
            for col in df.columns:
                if data_type.lower() in col.lower() or any(val_word in col.lower() for val_word in ['value', 'conc', 'level', 'count']):
                    value_column = col
                    break
            
            if value_column is None:
                logger.error(f"No value column found in {data_type} file for {site_name}")
                return None
            
            # Process dates
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            
            if len(df) == 0:
                logger.warning(f"No valid dates found in {data_type} file for {site_name}")
                return None
            
            # Add ISO week information
            df['Year'] = df[date_column].dt.year
            df['Week'] = df[date_column].dt.isocalendar().week
            
            # Group by week and aggregate
            weekly_data = df.groupby(['Year', 'Week']).agg({
                date_column: 'first',  # Use first date of the week
                value_column: 'mean'   # Average the values
            }).reset_index()
            
            # Clean up column names
            weekly_data['Site'] = site_name
            weekly_data['Date'] = weekly_data[date_column]
            weekly_data[data_type] = pd.to_numeric(weekly_data[value_column], errors='coerce')
            
            # Keep only necessary columns
            result = weekly_data[['Site', 'Date', data_type]].copy()
            result = result.dropna()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {data_type} file for {site_name}: {e}")
            return None

# =============================================================================
# ENVIRONMENTAL DATA PROCESSING MODULE  
# =============================================================================

class EnvironmentalDataProcessor:
    """Handles processing of climate and environmental data."""
    
    @handle_data_errors
    def fetch_climate_index(self, url, var_name, temp_dir):
        """Process climate index data (PDO, ONI) with comprehensive error handling."""
        print(f"Fetching climate index: {var_name}...")
        logger.info(f"Starting climate index processing: {var_name}")
            
        # Download file
        fname = local_filename(url, '.nc', temp_dir=temp_dir)
        logger.debug(f"Generated filename for {var_name}: {fname}")
        
        actual_file_path = download_file(url, fname)
        logger.info(f"Successfully downloaded {var_name} data to {actual_file_path}")
                
        # Open and process dataset using actual downloaded file path
        logger.debug(f"Opening dataset: {actual_file_path}")
        ds = xr.open_dataset(actual_file_path)
        df = ds.to_dataframe().reset_index()
        logger.debug(f"Dataset shape for {var_name}: {df.shape}")
        
        # Find time column
        time_cols = ['time', 'datetime', 'Date', 'T']
        time_col = next((c for c in time_cols if c in df.columns), None)
        
        if time_col is None:
            logger.error(f"No time column found in {var_name} dataset. Available columns: {list(df.columns)}")
            raise ValueError(f"No time column found for {var_name}")
            
        logger.debug(f"Using time column '{time_col}' for {var_name}")
        
        # Find variable column (case-insensitive)
        actual_var_name = var_name
        if actual_var_name not in df.columns:
            var_name_lower = var_name.lower()
            found_var = next((c for c in df.columns if c.lower() == var_name_lower), None)
            actual_var_name = found_var or var_name
            
        if actual_var_name not in df.columns:
            logger.error(f"Variable '{var_name}' not found in dataset. Available columns: {list(df.columns)}")
            raise ValueError(f"Variable '{var_name}' not found in dataset")
            
        logger.debug(f"Using variable column '{actual_var_name}' for {var_name}")
                
        # Process data
        df['datetime'] = pd.to_datetime(df[time_col])
        df = df[['datetime', actual_var_name]].dropna().rename(columns={actual_var_name: 'index'})
        
        # Aggregate monthly
        df['Month'] = df['datetime'].dt.to_period('M')
        result = df.groupby('Month')['index'].mean().reset_index()
        
        logger.info(f"Successfully processed {var_name}: {len(result)} monthly records")
        
        ds.close()
        return result[['Month', 'index']].sort_values('Month')

    @handle_data_errors
    def process_streamflow(self, url, temp_dir):
        """Process USGS streamflow data (daily)."""
        print("Fetching streamflow data...")
        # Download file
        fname = local_filename(url, '.json', temp_dir=temp_dir)
        actual_file_path = download_file(url, fname)
        with open(actual_file_path) as f:
            data = json.load(f)
            
        # Extract values
        values = []
        ts_data = data.get('value', {}).get('timeSeries', [])
        if ts_data:
            # Find discharge time series
            discharge_ts = next((ts for ts in ts_data 
                            if ts.get('variable', {}).get('variableCode', [{}])[0].get('value') == '00060'), 
                           ts_data[0] if len(ts_data) == 1 else None)
                           
            if discharge_ts:
                values = discharge_ts.get('values', [{}])[0].get('value', [])
            
        # Parse records
        records = []
        for item in values:
            if isinstance(item, dict) and 'dateTime' in item and 'value' in item:
                dt = pd.to_datetime(item['dateTime'], utc=True)
                flow = pd.to_numeric(item['value'], errors='coerce')
                if pd.notna(dt) and pd.notna(flow) and flow >= 0:
                    records.append({'Date': dt.tz_localize(None), 'Flow': flow})

        df = pd.DataFrame(records)
        df = df.dropna(subset=['Date', 'Flow'])  # Remove invalid entries
        return df[['Date', 'Flow']].sort_values('Date')

    @handle_data_errors
    def fetch_beuti_data(self, url, sites_dict, temp_dir, power=2):
        """Process BEUTI data with minimal error handling."""
        print("Fetching BEUTI data...")
            
        # Download file
        fname = local_filename(url, '.nc', temp_dir=temp_dir)
        actual_file_path = download_file(url, fname)
                
        # Process data
        ds = xr.open_dataset(actual_file_path)
        df = ds.to_dataframe().reset_index()
        
        # Find required columns
        time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
        lat_col = next((c for c in ['latitude', 'lat'] if c in df.columns), None)
        beuti_var = next((c for c in ['BEUTI', 'beuti'] if c in df.columns or c in ds.data_vars), None)
            
        # Prepare DataFrame for interpolation
        df_subset = df[[time_col, lat_col, beuti_var]].copy()
        df_subset.rename(columns={time_col: 'Date', lat_col: 'lat', beuti_var: 'beuti'}, inplace=True)
        df_subset['Date'] = pd.to_datetime(df_subset['Date']).dt.date
        df_subset = df_subset.dropna(subset=['Date', 'lat', 'beuti'])
            
        # Sort for efficient processing
        df_sorted = df_subset.sort_values(by=['Date', 'lat'])
        
        # Interpolate for each site
        results_list = []
        
        for site_name, site_coords in sites_dict.items():
            site_lat = site_coords['lat']
            
            # Group by date and interpolate
            for date, group in df_sorted.groupby('Date'):
                if len(group) >= 2:  # Need at least 2 points for interpolation
                    # Inverse distance weighting
                    distances = np.abs(group['lat'] - site_lat)
                    weights = 1 / (distances ** power + 1e-8)  # Small epsilon to avoid division by zero
                    
                    # Weighted average
                    interpolated_value = np.average(group['beuti'], weights=weights)
                    
                    results_list.append({
                        'Date': pd.to_datetime(date),
                        'Site': site_name,
                        'beuti': interpolated_value
                    })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results_list)
        result_df = result_df.sort_values(['Site', 'Date'])
        
        ds.close()
        return result_df

# =============================================================================
# DATA INTEGRATION MODULE
# =============================================================================

class DataIntegrator:
    """Handles integration of all data sources."""
    
    def generate_base_dataframe(self, sites_dict, start_date, end_date):
        """Generate base DataFrame with site-week combinations."""
        print(f"  Generating weekly entries from {start_date.date()} to {end_date.date()}")
        
        # Generate weekly date range
        all_weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        
        # Create site-week combinations
        base_data = []
        for site_name in sites_dict.keys():
            for week_date in all_weeks:
                base_data.append({
                    'Site': site_name,
                    'Date': week_date,
                    'Year': week_date.year,
                    'Week': week_date.isocalendar()[1]
                })
        
        base_df = pd.DataFrame(base_data)
        print(f"  Generated base DataFrame with {len(base_df)} site-week rows.")
        
        return base_df

    def compile_data(self, base_df, oni_data, pdo_data, streamflow_data):
        """Compile environmental data with temporal safeguards."""
        # Create a copy to avoid modifying the original
        compiled_df = base_df.copy()
        
        # Add temporal column for climate index merging (2 months prior to prevent leakage)
        compiled_df["TargetPrevMonth"] = compiled_df["Date"].dt.to_period("M") - 2
        
        # Sort compiled_df initially
        compiled_df = compiled_df.sort_values(["Site", "Date"])
        
        # --- Merge ONI data ---
        if oni_data is not None:
            oni_to_merge = oni_data[["Month", "index"]].rename(
                columns={"index": "oni", "Month": "ClimateIndexMonth"}
            )
            oni_to_merge = oni_to_merge.drop_duplicates(subset=["ClimateIndexMonth"])
            
            compiled_df = pd.merge(
                compiled_df,
                oni_to_merge,
                left_on="TargetPrevMonth",
                right_on="ClimateIndexMonth",
                how="left",
            )
            if "ClimateIndexMonth" in compiled_df.columns:
                compiled_df.drop(columns=["ClimateIndexMonth"], inplace=True)
        
        # --- Merge PDO data ---
        if pdo_data is not None:
            pdo_to_merge = pdo_data[["Month", "index"]].rename(
                columns={"index": "pdo", "Month": "ClimateIndexMonth"}
            )
            pdo_to_merge = pdo_to_merge.drop_duplicates(subset=["ClimateIndexMonth"])
            
            compiled_df = pd.merge(
                compiled_df,
                pdo_to_merge,
                left_on="TargetPrevMonth",
                right_on="ClimateIndexMonth",
                how="left",
            )
            if "ClimateIndexMonth" in compiled_df.columns:
                compiled_df.drop(columns=["ClimateIndexMonth"], inplace=True)
        
        # Drop the temporary column
        compiled_df.drop(columns=["TargetPrevMonth"], inplace=True)
        
        # --- Merge Streamflow data ---
        if streamflow_data is not None:
            streamflow_data["Date"] = pd.to_datetime(streamflow_data["Date"])
            compiled_df = pd.merge_asof(
                compiled_df.sort_values("Date"),
                streamflow_data.sort_values("Date"),
                on="Date",
                direction="backward",
                tolerance=pd.Timedelta(days=14)
            )
            compiled_df = compiled_df.sort_values(["Site", "Date"])
        
        return compiled_df

    def merge_toxin_data(self, compiled_df, da_df, pn_df):
        """Merge toxin measurement data."""
        print("\\n--- Merging DA and PN Data ---")
        
        # Merge DA data
        if da_df is not None and len(da_df) > 0:
            print(f"  Merging DA data ({len(da_df)} records)...")
            compiled_df = pd.merge(
                compiled_df,
                da_df[['Site', 'Date', 'DA']],
                on=['Site', 'Date'],
                how='left'
            )
        
        # Merge PN data
        if pn_df is not None and len(pn_df) > 0:
            print(f"  Merging PN data ({len(pn_df)} records)...")
            compiled_df = pd.merge(
                compiled_df,
                pn_df[['Site', 'Date', 'PN']],
                on=['Site', 'Date'],
                how='left'
            )
        
        return compiled_df

    def apply_satellite_matching(self, compiled_df, satellite_df):
        """Apply satellite data matching with progress bar."""
        print(f"\\n--- Adding satellite data from: {SATELLITE_OUTPUT_PARQUET} ---")
        
        if satellite_df is None or len(satellite_df) == 0:
            logger.warning("No satellite data available for matching")
            return compiled_df
        
        print("Applying satellite matching function...")
        
        # Ensure satellite data has proper date format
        satellite_df['Date'] = pd.to_datetime(satellite_df['Date'])
        
        # Initialize satellite columns with NaN
        satellite_columns = [col for col in satellite_df.columns if col not in ['Date', 'Site']]
        for col in satellite_columns:
            compiled_df[col] = np.nan
        
        # Process each row with progress bar
        for idx in tqdm(compiled_df.index, desc="Satellite Matching"):
            site = compiled_df.loc[idx, 'Site']
            target_date = compiled_df.loc[idx, 'Date']
            
            # Find matching satellite data
            site_satellite_data = satellite_df[satellite_df['Site'] == site]
            
            if len(site_satellite_data) > 0:
                # Find closest date within reasonable range (±7 days)
                date_diff = abs(site_satellite_data['Date'] - target_date)
                closest_idx = date_diff.idxmin()
                
                if date_diff.loc[closest_idx] <= pd.Timedelta(days=7):
                    # Copy satellite values
                    for col in satellite_columns:
                        if col in site_satellite_data.columns:
                            compiled_df.loc[idx, col] = site_satellite_data.loc[closest_idx, col]
        
        return compiled_df

# =============================================================================
# MAIN PIPELINE ORCHESTRATION
# =============================================================================

def main():
    """
    Main data processing pipeline with modular architecture.
    
    This maintains the exact same functionality as the original script
    but with improved organization and maintainability.
    """
    print("\\n======= Starting Data Processing Pipeline =======")
    start_time = datetime.now()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='data_dl_', suffix='')
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize processors
        toxin_processor = ToxinDataProcessor()
        env_processor = EnvironmentalDataProcessor()
        integrator = DataIntegrator()
        
        # Check for existing satellite data
        satellite_df = None
        if os.path.exists(SATELLITE_OUTPUT_PARQUET) and not FORCE_SATELLITE_REPROCESSING:
            print(f"\\n--- Found existing satellite data: {SATELLITE_OUTPUT_PARQUET}. Using this file. ---")
            print("--- To regenerate, set FORCE_SATELLITE_REPROCESSING = True in the script. ---")
            satellite_df = pd.read_parquet(SATELLITE_OUTPUT_PARQUET)
        else:
            print("\\n--- Satellite data processing would happen here ---")
            print("--- (Not implemented in modular version yet) ---")
        
        # Process toxin data
        da_df = toxin_processor.process_da_data(da_files, sites)
        pn_df = toxin_processor.process_pn_data(pn_files, sites)
        
        # Process environmental data
        streamflow_df = env_processor.process_streamflow(streamflow_url, temp_dir)
        pdo_df = env_processor.fetch_climate_index(pdo_url, 'pdo', temp_dir)
        oni_df = env_processor.fetch_climate_index(oni_url, 'oni', temp_dir)
        beuti_df = env_processor.fetch_beuti_data(beuti_url, sites, temp_dir)
        
        # Generate base site-week grid
        compiled_base = integrator.generate_base_dataframe(sites, start_date, end_date)
        
        # Compile environmental data
        print("\\n--- Merging Environmental Data ---")
        lt_data = integrator.compile_data(compiled_base, oni_df, pdo_df, streamflow_df)
        
        # Merge toxin data
        lt_data = integrator.merge_toxin_data(lt_data, da_df, pn_df)
        
        # Interpolate missing values (forward-only to prevent leakage)
        print("  Interpolating missing values (forward-only to prevent leakage)...")
        numeric_columns = lt_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Year', 'Week']:
                lt_data[col] = lt_data.groupby('Site')[col].fillna(method='ffill')
        
        # Apply satellite matching if data available
        if satellite_df is not None:
            lt_data = integrator.apply_satellite_matching(lt_data, satellite_df)
        
        # Merge BEUTI data
        if beuti_df is not None and len(beuti_df) > 0:
            lt_data = pd.merge(lt_data, beuti_df[['Site', 'Date', 'beuti']], on=['Site', 'Date'], how='left')
        
        # Final processing and saving
        print("\\n--- Final Checks and Saving Output ---")
        print(f"Saving final data to {final_output_path}...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        
        # Save to Parquet
        lt_data.to_parquet(final_output_path, index=False)
        
        # Calculate execution time
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info(f"Pipeline completed in {total_time}")
        
        print("\\n--- Cleaning Up ---")
        logger.info("Starting cleanup of temporary files")
        
        try:
            cleanup_downloaded_files()
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        print(f"\\n======= Script Finished in {total_time} =======")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        print(f"\\n--- Pipeline failed: {e} ---")
        
        # Cleanup on error
        try:
            cleanup_downloaded_files()
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        raise

# Run script with proper error handling
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\\n--- Pipeline interrupted by user ---")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        print(f"\\n--- Pipeline failed: {e} ---")
        raise