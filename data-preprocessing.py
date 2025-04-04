import pandas as pd
import numpy as np
import json
import os
import requests
import tempfile
import xarray as xr
from datetime import datetime
import traceback
from tqdm import tqdm
import warnings
import shutil # Added for rmtree

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'
SATELLITE_CONFIG_FILE = 'satellite_config.json'

# Initialize variables with defaults or None
config = {}
satellite_metadata = {}
da_files = {}
pn_files = {}
sites = {}
pdo_url = None
oni_url = None
beuti_url = None
streamflow_url = None
start_date = None
end_date = None
year_cutoff = 2000 # Default, will be overwritten
week_cutoff = 1    # Default, will be overwritten
include_satellite = False # Default to False
final_output_path = 'default_final_output.parquet' # Default
SATELLITE_OUTPUT_PARQUET = 'satellite_data_intermediate.parquet' # Default intermediate name

print(f"--- Loading Configuration from {CONFIG_FILE} ---")
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Extract values, providing defaults where critical if missing
    da_files = config.get('original_da_files', {})
    pn_files = config.get('original_pn_files', {})
    sites = config.get('sites', {})
    pdo_url = config.get('pdo_url')
    oni_url = config.get('oni_url')
    beuti_url = config.get('beuti_url')
    streamflow_url = config.get('streamflow_url')
    start_date = pd.to_datetime(config.get('start_date', '2000-01-01')) # Default start if missing
    end_date = pd.to_datetime(config.get('end_date', datetime.now().strftime('%Y-%m-%d'))) # Default end if missing
    year_cutoff = config.get('year_cutoff', 2000)
    week_cutoff = config.get('week_cutoff', 1)
    include_satellite = config.get('include_satellite', False)
    final_output_path = config.get('final_output_path', 'config_final_output.parquet')

    print("Configuration loaded successfully:")
    print(f"  DA Files: {len(da_files)} entries")
    print(f"  PN Files: {len(pn_files)} entries")
    print(f"  Sites: {len(sites)} entries")
    print(f"  Date Range: {start_date.date()} to {end_date.date()}")
    print(f"  Cutoff: {year_cutoff}-W{week_cutoff:02d}")
    print(f"  Include Satellite: {include_satellite}")
    print(f"  Output Path: {final_output_path}")

except FileNotFoundError:
    print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Cannot proceed.")
    print("Please ensure the configuration file exists in the same directory as the script.")
    exit() # Exit if main config is missing
except json.JSONDecodeError as e:
    print(f"ERROR: Could not decode JSON from '{CONFIG_FILE}': {e}")
    exit()
except Exception as e:
    print(f"ERROR: An unexpected error occurred loading configuration from '{CONFIG_FILE}': {e}")
    exit()

# Load satellite configuration ONLY if include_satellite is True
if include_satellite:
    print(f"\n--- Loading Satellite Configuration from {SATELLITE_CONFIG_FILE} ---")
    try:
        with open(SATELLITE_CONFIG_FILE, 'r') as f:
            satellite_metadata = json.load(f)
        print(f"Satellite configuration loaded successfully. Found {len(satellite_metadata)-1} data types (excluding end_date).")
        # Add the main end_date to the satellite metadata if satellite end_date is not present
        if 'end_date' not in satellite_metadata and end_date is not None:
             satellite_metadata['end_date'] = end_date.strftime('%Y-%m-%dT%H:%M:%SZ') # Format for URL replacement
             print(f"  Using main end_date ({satellite_metadata['end_date']}) for satellite URL formatting.")

    except FileNotFoundError:
        print(f"Warning: Satellite configuration file '{SATELLITE_CONFIG_FILE}' not found, but include_satellite=True.")
        print("Satellite processing will be skipped.")
        satellite_metadata = {}
        include_satellite = False # Force skip if file is missing
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from '{SATELLITE_CONFIG_FILE}': {e}")
        print("Satellite processing will be skipped.")
        satellite_metadata = {}
        include_satellite = False # Force skip on decode error
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading configuration from '{SATELLITE_CONFIG_FILE}': {e}")
        print("Satellite processing will be skipped.")
        satellite_metadata = {}
        include_satellite = False # Force skip on other errors
else:
    print("\nSatellite processing is disabled in the configuration (include_satellite=False).")
    satellite_metadata = {} # Ensure it's empty if skipped

# Lists to track temporary files for cleanup
downloaded_files = []          # Files downloaded via download_file helper
generated_parquet_files = []   # Parquet files generated from CSVs or intermediates

# =============================================================================
# Basic Helper Functions (Keep functions as they were in the previous version)
# =============================================================================
def download_file(url, filename):
    """Download a file from URL, save locally, track for cleanup."""
    if not url:
        print("  Download failed: No URL provided.")
        return None
    try:
        print(f"Downloading {url.split('/')[-1]} to {os.path.basename(filename)}...")
        response = requests.get(url, timeout=120, stream=True) # Increased timeout
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        downloaded_files.append(filename) # Track for cleanup
        print(f"  Successfully downloaded.")
        return filename
    except requests.exceptions.Timeout:
        print(f"  Error downloading {url}: Timeout occurred.")
        if os.path.exists(filename):
            try: os.remove(filename)
            except Exception: pass
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        if os.path.exists(filename):
            try: os.remove(filename)
            except Exception: pass
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during download of {url}: {e}")
        if os.path.exists(filename):
            try: os.remove(filename)
            except Exception: pass
        return None


def local_filename(url, ext, temp_dir=None):
    """Derive local filename, ensure correct extension, optionally use temp dir."""
    if not url: base_name = f"unknown_file_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"
    else:
        try:
            base = url.split('?')[0].split('/')[-1]
            if not base: # Handle cases like '.../latest?'
                 base = url.split('?')[0].split('/')[-2] # Try second to last part
            sanitized_base = "".join(c for c in base if c.isalnum() or c in ('-', '_', '.'))
            if not sanitized_base: sanitized_base = f"downloaded_file_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            root, existing_ext = os.path.splitext(sanitized_base)
            # Use provided ext unless the original URL already had a valid one (e.g. .nc)
            base_name = root + (ext if not existing_ext or existing_ext == '.' else existing_ext)
        except Exception: # Fallback for weird URLs
             base_name = f"download_error_name_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"

    if temp_dir:
        return os.path.join(temp_dir, base_name)
    else:
        return base_name


def csv_to_parquet(csv_path):
    """Convert CSV to Parquet, track for cleanup."""
    if not isinstance(csv_path, str) or not csv_path.lower().endswith('.csv'):
        return csv_path # Return original if not a csv string path
    if not os.path.exists(csv_path):
        print(f"  Warning: CSV file not found, cannot convert: {csv_path}")
        return csv_path # Return original path if file doesn't exist

    parquet_path = csv_path[:-4] + '.parquet'
    try:
        print(f"Converting {os.path.basename(csv_path)} to Parquet...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_parquet(parquet_path, index=False)
        if parquet_path not in generated_parquet_files:
            generated_parquet_files.append(parquet_path) # Track for cleanup
        print(f"  Successfully converted to {os.path.basename(parquet_path)}")
        return parquet_path
    except Exception as e:
        print(f"  Error converting {csv_path} to Parquet: {e}")
        return csv_path # Return original path on failure


def convert_files_to_parquet(files_dict):
    """Convert CSVs in file dictionary to Parquet, returns updated dict."""
    if not isinstance(files_dict, dict): return {}
    new_files = {}
    print("\n--- Converting Input CSV Files to Parquet (if any) ---")
    count = 0
    for name, path in files_dict.items():
        if isinstance(path, str) and path.lower().endswith('.csv'):
            parquet_path = csv_to_parquet(path)
            new_files[name] = parquet_path
            if parquet_path != path: count += 1
        else:
            new_files[name] = path # Keep non-csv paths as they are
    if count == 0: print("  No CSV files found to convert in this set.")
    return new_files

# =============================================================================
# Satellite Data Processing Functions (Keep functions as they were)
# =============================================================================
def process_dataset(url, data_type, site, temp_dir):
    """
    Downloads satellite data (.nc) to a temp dir, averages values per timestamp,
    returns DataFrame. Cleans up its own temporary NetCDF file download.
    """
    # (Function content is the same as the previous version)
    # print(f"\nProcessing satellite: {data_type} for {site}...") # Verbose

    # --- Determine data variable name ---
    data_var = None
    url_lower = url.lower() if url else ""
    dtype_lower = data_type.lower() if data_type else ""

    if 'chla_anomaly' in url_lower or 'chlorophyll-anom' in dtype_lower: data_var = 'chla_anomaly'
    elif 'sst_anomaly' in url_lower or 'temperature-anom' in dtype_lower: data_var = 'sst_anomaly'
    elif 'chlorophyll' in url_lower or 'chlorophyll' in dtype_lower: data_var = 'chlorophyll'
    elif 'sst' in url_lower or 'temperature' in dtype_lower: data_var = 'sst'
    elif 'par' in url_lower or 'par' in dtype_lower: data_var = 'par'
    elif 'fluorescence' in url_lower or 'fluorescence' in dtype_lower: data_var = 'fluorescence'
    else:
        print(f"  Warning: Could not determine satellite variable for {site}/{data_type} from URL/type. Skipping.")
        return None

    tmp_nc_path = None # Path for the temporary NetCDF file
    ds = None # Initialize dataset variable

    # Create a temporary file path within the provided temp_dir
    try:
        # Use mkstemp for unique name and manual deletion control within this function
        fd, tmp_nc_path = tempfile.mkstemp(suffix='.nc', prefix=f"{site}_{data_type}_", dir=temp_dir)
        os.close(fd) # Close file descriptor, we just need the path
    except Exception as e:
        print(f"  Error creating temporary file path for {site}/{data_type}: {e}")
        return None

    try:
        # --- Download ---
        try:
            # print(f"  Downloading satellite data to {tmp_nc_path}...") # Verbose
            response = requests.get(url, timeout=300, stream=True) # Use stream for potentially large files
            response.raise_for_status()
            with open(tmp_nc_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # print("  Download successful.") # Verbose
        except requests.exceptions.RequestException as e:
            print(f"  Failed to download satellite data for {site}/{data_type}: {e}")
            return None # Error occurred, tmp_nc_path will be cleaned in finally

        # --- Open & Process ---
        ds = xr.open_dataset(tmp_nc_path)

        if data_var not in ds.data_vars:
            available_vars = list(ds.data_vars)
            print(f"  Error: Variable '{data_var}' not found in satellite dataset for {site}/{data_type}.")
            print(f"    Available data variables: {available_vars}")
            # Attempt fallback if only one variable exists (common case)
            if len(available_vars) == 1:
                 fallback_var = available_vars[0]
                 print(f"    Attempting to use fallback variable: '{fallback_var}'")
                 data_var = fallback_var
            else:
                 if data_var in ds.coords: print(f"  Note: '{data_var}' exists as a coordinate.")
                 return None # Error occurred, tmp_nc_path will be cleaned in finally

        # Select the specific data variable
        data_array = ds[data_var]

        # Identify time coordinate (handle variations like 'time', 't')
        time_coord_name = None
        for coord_name in ['time', 't', 'datetime']:
             if coord_name in data_array.coords:
                  time_coord_name = coord_name
                  break
        if not time_coord_name:
             print(f"  Error: Could not find a suitable time coordinate in satellite data for {site}/{data_type}. Coords: {list(data_array.coords)}")
             return None

        # Average over spatial dimensions (assuming lat/lon or x/y)
        spatial_dims = [dim for dim in data_array.dims if dim != time_coord_name]
        if spatial_dims:
            # print(f"  Averaging over spatial dimensions: {spatial_dims}") # Verbose
            # Use skipna=True for robustness
            averaged_array = data_array.mean(dim=spatial_dims, skipna=True)
        else:
            # print("  No spatial dimensions found to average over.") # Verbose
            averaged_array = data_array # Already a time series

        # Convert to DataFrame
        df = averaged_array.to_dataframe(name='value').reset_index()

        # Check and rename time column standardly to 'timestamp'
        if time_coord_name not in df.columns:
            print(f"  Error: Expected time column '{time_coord_name}' not found after converting to DataFrame for {site}/{data_type}.")
            return None
        df = df.rename(columns={time_coord_name: 'timestamp'})

        df = df.dropna(subset=['timestamp', 'value'])
        if df.empty:
            # print(f"  No valid satellite data points after dropping NaNs for {site}/{data_type}.") # Verbose
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop if time conversion failed

        if df.empty:
            # print(f"  No valid satellite data points after timestamp conversion for {site}/{data_type}.") # Verbose
            return None

        df['site'] = site
        df['data_type'] = data_type # Store the original requested data_type
        df = df.rename(columns={'value': data_var}) # Rename value to specific variable name

        # print(f"  Successfully processed {data_type} for {site}.") # Verbose
        return df[['timestamp', 'site', 'data_type', data_var]] # Return specific value column

    except FileNotFoundError:
        print(f"  Error opening satellite dataset for {site}/{data_type}: File not found at {tmp_nc_path} (Download likely failed).")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred processing satellite {url} for {site}/{data_type}: {e}")
        traceback.print_exc() # Keep for satellite processing as it's complex
        return None
    finally:
        # --- Clean up the temporary NetCDF file downloaded by *this* function ---
        if ds:
            try:
                ds.close()
            except Exception as e_close:
                print(f"  Warning: Error closing satellite dataset for {tmp_nc_path}: {e_close}")
        if tmp_nc_path and os.path.exists(tmp_nc_path):
             try:
                 os.unlink(tmp_nc_path)
                 # print(f"  Cleaned up satellite temp file: {tmp_nc_path}") # Verbose
             except Exception as e_clean:
                 print(f"  Warning: Could not remove satellite temp file {tmp_nc_path}: {e_clean}")


def generate_satellite_parquet(satellite_metadata_dict, main_sites_list, output_path):
    """Processes satellite data for relevant sites and saves results to a Parquet file."""
    # (Function content is the same as the previous version)
    print("\n--- Processing Satellite Data ---")
    if not satellite_metadata_dict:
        print("Satellite metadata is empty. Skipping satellite data generation.")
        return None
    # Allow processing even if end_date isn't top-level, might be in URL directly
    # if 'end_date' not in satellite_metadata_dict:
    #     print("Warning: 'end_date' key not found in top level of satellite metadata.")
    sat_end_date_global = satellite_metadata_dict.get('end_date', None) # Get if exists

    # Create a dedicated temporary directory for satellite downloads
    sat_temp_dir = None # Initialize
    try:
        sat_temp_dir = tempfile.mkdtemp(prefix="sat_downloads_")
        print(f"Using temporary directory for satellite downloads: {sat_temp_dir}")
    except Exception as e:
        print(f"Error creating temporary directory for satellite downloads: {e}. Aborting satellite processing.")
        return None

    satellite_tasks = []
    processed_sites = set()
    for data_type, sat_sites_dict in satellite_metadata_dict.items():
        if data_type == 'end_date': continue # Skip metadata key
        if not isinstance(sat_sites_dict, dict):
            print(f"Warning: Expected a dictionary of sites for satellite data type '{data_type}', got {type(sat_sites_dict)}. Skipping.")
            continue

        for site, url_or_list in sat_sites_dict.items():
             # Allow site name variations (e.g., case, space vs underscore)
            normalized_site_name = site.lower().replace('_', ' ').replace('-', ' ')
            relevant_main_site = None
            for main_site in main_sites_list:
                 # Normalize main site name for comparison as well
                 if main_site.lower().replace('_', ' ').replace('-', ' ') == normalized_site_name:
                      relevant_main_site = main_site # Use the exact name from main_sites_list
                      break

            if relevant_main_site: # Only process sites relevant to the main analysis
                urls_to_process = []
                if isinstance(url_or_list, str):
                     urls_to_process.append(url_or_list)
                elif isinstance(url_or_list, list):
                     urls_to_process.extend(url_or_list)
                else:
                     print(f"Warning: Satellite URL entry for {relevant_main_site}/{data_type} is not a string or list. Skipping.")
                     continue

                for url in urls_to_process:
                     if isinstance(url, str) and url.strip():
                         try:
                             # Replace end_date placeholder if present and available
                             processed_url = url
                             if '{end_date}' in url:
                                  if sat_end_date_global:
                                       processed_url = url.replace('{end_date}', sat_end_date_global)
                                  else:
                                       print(f"Warning: URL for {relevant_main_site}/{data_type} contains '{{end_date}}' but no end_date found in satellite metadata. Using URL as is.")
                                       # Decide: skip or try URL as is? Let's try as is.
                                       # continue # Optional: skip if end_date needed but missing

                             satellite_tasks.append((data_type, relevant_main_site, processed_url))
                             processed_sites.add(relevant_main_site)
                         except Exception as e:
                             print(f"Error processing satellite URL for {data_type}/{relevant_main_site}: {e}")
                     else:
                         print(f"Warning: Invalid satellite URL '{url}' for {relevant_main_site}/{data_type}. Skipping.")
            # else: # Verbose debug
            #      # Check if site name exists with different spacing/case only if no match found yet
            #      site_exists_any_case = any(s.lower().replace('_', ' ').replace('-', ' ') == normalized_site_name for s in main_sites_list)
            #      if not site_exists_any_case:
            #           print(f"Debug: Site '{site}' from satellite metadata not found in main sites list: {main_sites_list}")


    if not satellite_tasks:
        print("No satellite processing tasks defined for relevant sites found in the main site list.")
        if sat_temp_dir and os.path.exists(sat_temp_dir):
             try: shutil.rmtree(sat_temp_dir)
             except Exception: pass
        return None

    print(f"Processing {len(satellite_tasks)} satellite datasets for {len(processed_sites)} sites...")
    satellite_results_list = [] # Store results from process_dataset

    # Use tqdm for progress bar
    for data_type, site, url in tqdm(satellite_tasks, desc="Satellite Data"):
        result_df = process_dataset(url, data_type, site, sat_temp_dir)
        if result_df is not None and not result_df.empty:
            satellite_results_list.append(result_df)
        # else: # Optional: Log skips more explicitly
        #     print(f"  Skipping results for {site}/{data_type} (empty or error).")


    # Clean up the temporary directory for satellite downloads AFTER processing all files
    if sat_temp_dir and os.path.exists(sat_temp_dir):
        try:
            shutil.rmtree(sat_temp_dir)
            print(f"Removed temporary satellite download directory: {sat_temp_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary satellite download directory {sat_temp_dir}: {e}")

    if not satellite_results_list:
        print("No valid satellite data generated after processing all tasks.")
        return None

    # --- Combine and Pivot ---
    try:
        print("Combining individual satellite results...")
        combined_satellite_df = pd.concat(satellite_results_list, ignore_index=True)

        # Pivot the combined data
        print("Pivoting combined satellite data...")
        # Determine value columns (dynamic based on processing results)
        value_cols = [col for col in combined_satellite_df.columns if col not in ['timestamp', 'site', 'data_type']]
        if not value_cols:
             print("Error: No value columns found in combined satellite data after processing. Cannot pivot.")
             return None

        # Use pivot_table which handles potential duplicate timestamp/site/type entries by averaging
        processed_satellite_pivot = combined_satellite_df.pivot_table(
            index=['site', 'timestamp'],
            columns='data_type', # Use original requested type for column names
            values=value_cols, # Use the actual data columns found
            aggfunc='mean' # Average if multiple entries exist (e.g., different URLs for same type)
        )

        # Flatten MultiIndex columns if they exist (e.g., if value_cols had multiple items)
        if isinstance(processed_satellite_pivot.columns, pd.MultiIndex):
             processed_satellite_pivot.columns = ['sat_' + '_'.join(col).strip() for col in processed_satellite_pivot.columns.values]
        else:
             processed_satellite_pivot.columns = [f'sat_{col}' for col in processed_satellite_pivot.columns]

        processed_satellite_pivot = processed_satellite_pivot.reset_index()

        # --- IMPORTANT: Ensure timestamp is datetime object BEFORE saving ---
        processed_satellite_pivot['timestamp'] = pd.to_datetime(processed_satellite_pivot['timestamp'], errors='coerce')
        processed_satellite_pivot.dropna(subset=['timestamp'], inplace=True) # Should be redundant but safe

        print(f"Successfully processed and pivoted satellite data. Shape: {processed_satellite_pivot.shape}")
        if processed_satellite_pivot.empty:
             print("Pivoted satellite data is empty. Cannot save.")
             return None

        print(f"Saving processed satellite data to {output_path}...")
        processed_satellite_pivot.to_parquet(output_path, index=False)
        print("Satellite data saved.")
        # Track this intermediate file for cleanup LATER in main
        if output_path not in generated_parquet_files:
            generated_parquet_files.append(output_path)
        return output_path # Return path on success

    except Exception as e:
        print(f"ERROR during final satellite data combination, pivoting, or saving to {output_path}: {e}")
        traceback.print_exc()
        return None # Return None on failure

# =============================================================================
# Satellite Data Merging Functions (Keep functions as they were)
# =============================================================================
def find_best_satellite_match(target_row, sat_pivot_indexed):
    """
    Finds best satellite match (prefer same month, then overall closest).
    Uses argmin() on TimedeltaIndex to fix the 'no attribute idxmin' error.
    """
    target_site = target_row.get('Site') # Use 'Site' column from target_df (as used in main)
    target_ts = target_row.get('timestamp_dt') # Use the temporary datetime col

    if pd.isna(target_ts) or target_site is None:
        # print(f"Debug: Skipping match - Target ts or site is null ({target_ts}, {target_site})") # Verbose Debug
        return pd.Series(dtype='float64')

    # Normalize site name from target row for matching index
    target_site_normalized = target_site.lower().replace('_', ' ').replace('-', ' ')
    # Get unique site names from index - handle potential non-string values gracefully
    index_sites = sat_pivot_indexed.index.get_level_values('site')
    unique_original_index_sites = index_sites.unique()
    # Normalize index site names, ensuring they are strings first
    unique_normalized_index_sites = [str(s).lower().replace('_', ' ').replace('-', ' ') for s in unique_original_index_sites]


    if target_site_normalized not in unique_normalized_index_sites:
        # print(f"Debug: Skipping match - Normalized target site '{target_site_normalized}' not in normalized satellite index sites.") # Verbose Debug
        return pd.Series(dtype='float64')

    # Find the original case site name from the index that matches the normalized target
    original_index_site = None
    for i, norm_site in enumerate(unique_normalized_index_sites):
        if norm_site == target_site_normalized:
            original_index_site = unique_original_index_sites[i]
            break

    if original_index_site is None: # Should not happen if check above passed, but safety first
         print(f"Debug: Could not find original index site name for normalized '{target_site_normalized}'")
         return pd.Series(dtype='float64')

    try:
        # Select data for the specific site using the original index name
        # Use .loc for potentially non-unique index levels if xs causes issues
        # site_data = sat_pivot_indexed.loc[(original_index_site, slice(None)), :] # Alternative if xs fails
        site_data = sat_pivot_indexed.xs(original_index_site, level='site')


        if site_data.empty:
            # print(f"Debug: Skipping match - No satellite data found for site '{original_index_site}' after xs.") # Verbose Debug
            return pd.Series(dtype='float64')

        # Ensure the index is datetime (should be from pivot, but double check)
        if not isinstance(site_data.index, pd.DatetimeIndex):
            original_index = site_data.index
            site_data.index = pd.to_datetime(site_data.index, errors='coerce')
            # Check which indices failed conversion
            failed_indices = original_index[site_data.index.isna()]
            if not failed_indices.empty:
                # print(f"Debug: Failed to convert index to datetime for site '{original_index_site}': {failed_indices}") # Verbose
                pass
            site_data = site_data.dropna(axis=0, subset=[site_data.index.name]) # Drop rows where conversion failed


        if site_data.empty:
            # print(f"Debug: Skipping match - No satellite data after index conversion for site '{original_index_site}'.") # Verbose Debug
            return pd.Series(dtype='float64')

        # --- Matching Logic ---
        target_year = target_ts.year
        target_month = target_ts.month

        # 1. Look for matches within the same month and year
        month_matches = site_data[(site_data.index.year == target_year) & (site_data.index.month == target_month)]

        best_match_timestamp = None
        if not month_matches.empty:
            # Find the closest timestamp *within* that month
            # --- FIX APPLIED HERE ---
            time_diff_in_month = np.abs(month_matches.index - target_ts) # This creates a TimedeltaIndex
            if not time_diff_in_month.empty: # Check if TimedeltaIndex is not empty
                # Find the integer position of the minimum timedelta
                min_idx_pos = time_diff_in_month.argmin()
                # Use that position to get the corresponding timestamp from the original index
                best_match_timestamp = month_matches.index[min_idx_pos]
            # print(f"Debug: Found match in same month for {target_site} @ {target_ts}: {best_match_timestamp}") # Verbose
        #else: # Only check overall if no month match FOUND
        #    # print(f"Debug: No match in same month for {target_site} @ {target_ts}. Checking overall.") # Verbose
        #    pass

        # 2. If no match in the same month, find the absolute closest timestamp overall for that site
        if best_match_timestamp is None: # Check if a match was already found in the month
            # --- FIX APPLIED HERE ---
            time_diff_overall = np.abs(site_data.index - target_ts) # This creates a TimedeltaIndex
            if not time_diff_overall.empty:
                # Find the integer position of the minimum timedelta
                min_overall_pos = time_diff_overall.argmin()
                # Use that position to get the corresponding timestamp from the original index
                best_match_timestamp = site_data.index[min_overall_pos]
            # print(f"Debug: Found closest overall match for {target_site} @ {target_ts}: {best_match_timestamp}") # Verbose

        # Retrieve the data for the best matching timestamp
        if best_match_timestamp is not None:
            # Use .loc which is robust for DatetimeIndex
            return site_data.loc[best_match_timestamp]
        else:
            # print(f"Debug: No suitable match found for {target_site} @ {target_ts}") # Verbose
            return pd.Series(dtype='float64') # No match found

    except KeyError:
        # This can happen if xs fails unexpectedly, though the check above should prevent it
        # print(f"Warning: KeyError during satellite matching for site {target_site} (matched as {original_index_site}) at {target_ts}.") # Verbose
        return pd.Series(dtype='float64')
    except Exception as e:
        # Catch other potential errors during matching
        print(f"Error during satellite matching for site {target_site} (matched as {original_index_site}) at {target_ts}: {e}") # Keep this error visible
        traceback.print_exc() # Print traceback for this specific error as it's tricky
        return pd.Series(dtype='float64')


def add_satellite_data(target_df, satellite_parquet_path):
    """
    Adds satellite data from a pre-pivoted parquet file using site and specific
    timestamp matching rules (calling the corrected find_best_satellite_match).
    """
    print(f"\n--- Adding Satellite Data from {satellite_parquet_path} ---")
    # Input validation
    if not os.path.exists(satellite_parquet_path):
        print(f"Satellite data file not found: {satellite_parquet_path}. Skipping satellite merge.")
        return target_df.copy()
    if target_df is None or target_df.empty:
        print("Target DataFrame is empty. Skipping satellite merge.")
        return target_df.copy()
    # Use 'Site' and 'Date' as these are the columns used/created in the main script flow
    required_target_cols = ['Site', 'Date']
    if not all(c in target_df.columns for c in required_target_cols):
        print(f"Error: target_df needs columns {required_target_cols} for satellite merge. Found: {target_df.columns}. Skipping.")
        return target_df.copy()

    satellite_df = None
    try:
        print(f"Reading satellite data from {satellite_parquet_path}...")
        satellite_df = pd.read_parquet(satellite_parquet_path)
        print(f"Satellite data loaded. Shape: {satellite_df.shape}")
    except Exception as e:
        print(f"Error reading satellite data parquet file {satellite_parquet_path}: {e}")
        return target_df.copy() # Return original df if satellite data can't be read

    if satellite_df.empty:
        print("Satellite DataFrame loaded from Parquet is empty. Skipping merge.")
        return target_df.copy()

    # Required cols after pivoting in generate_satellite_parquet
    required_sat_cols = ['site', 'timestamp']
    if not all(c in satellite_df.columns for c in required_sat_cols):
        print(f"Error: satellite parquet missing required columns {required_sat_cols}. Found: {satellite_df.columns}. Skipping.")
        return target_df.copy()
    # Check if there are any actual data columns (prefixed with sat_)
    sat_data_cols = [col for col in satellite_df.columns if col.startswith('sat_')]
    if not sat_data_cols:
        print(f"Error: satellite parquet missing actual data columns (expected prefix 'sat_'). Found: {satellite_df.columns}. Skipping.")
        return target_df.copy()

    print("Preparing data for satellite matching...")
    target_df_proc = target_df.copy()

    try:
        # Ensure target 'Date' is datetime for matching, create temp 'timestamp_dt'
        target_df_proc['timestamp_dt'] = pd.to_datetime(target_df_proc['Date'], errors='coerce')

        # Ensure satellite 'timestamp' is datetime (should be from parquet, but double check)
        satellite_df['timestamp'] = pd.to_datetime(satellite_df['timestamp'], errors='coerce')

        # Drop rows where crucial keys are missing AFTER conversion attempts
        target_df_proc.dropna(subset=['timestamp_dt', 'Site'], inplace=True)
        satellite_df.dropna(subset=['timestamp', 'site'], inplace=True) # Use 'site' column from satellite data

        if target_df_proc.empty:
            print("Target DF is empty after timestamp conversion/dropna. Cannot merge satellite data.")
            return target_df.copy() # Return original target_df
        if satellite_df.empty:
            print("Satellite DF is empty after timestamp conversion/dropna. Cannot merge satellite data.")
            return target_df.copy() # Return original target_df

        print(f"Target DF has {len(target_df_proc)} rows for matching.")
        print(f"Satellite DF has {len(satellite_df)} rows available.")

    except Exception as e:
        print(f"Error preparing timestamp columns for merge: {e}")
        return target_df.copy()

    # Set index on satellite data for efficient lookup using xs
    try:
        print("Setting index on satellite data for matching...")
        # Ensure site names in index match case used in satellite_df consistently
        satellite_pivot_indexed = satellite_df.set_index(['site', 'timestamp']).sort_index()
        if satellite_pivot_indexed.empty:
            print("Indexed satellite data is empty. Cannot merge.")
            return target_df.copy()
        print(f"Satellite index set. Shape: {satellite_pivot_indexed.shape}")
    except Exception as e:
        print(f"Error setting index on satellite data: {e}")
        traceback.print_exc()
        return target_df.copy()

    # Apply matching function
    print(f"Applying satellite matching function to {len(target_df_proc)} target rows...")
    matched_data = None
    try: # Use progress_apply if tqdm is available and integrated
        tqdm.pandas(desc="Satellite Matching")
        matched_data = target_df_proc.progress_apply(
            find_best_satellite_match, axis=1, sat_pivot_indexed=satellite_pivot_indexed
        )
    except AttributeError:
        print("tqdm.progress_apply not available, using standard apply (may take time)...")
        matched_data = target_df_proc.apply(
            find_best_satellite_match, axis=1, sat_pivot_indexed=satellite_pivot_indexed
        )
    except Exception as e:
        print(f"An unexpected error occurred during the apply step for satellite matching: {e}")
        traceback.print_exc()
        return target_df.copy() # Return original target_df if apply fails

    # Join results
    print("Joining matched satellite data to target DataFrame...")
    # Ensure index alignment before joining
    target_df_proc.reset_index(drop=True, inplace=True)
    if matched_data is not None:
        matched_data.reset_index(drop=True, inplace=True)
        result_df = target_df_proc.join(matched_data)
    else: # Handle case where apply step failed or returned None
        print("Warning: matched_data is None after apply step. Returning target DataFrame without satellite data.")
        result_df = target_df_proc # Return the dataframe before the join attempt

    # Clean up temporary column
    result_df.drop(columns=['timestamp_dt'], inplace=True, errors='ignore')

    # Fill NaNs *only* for the newly added satellite columns with 0
    # Identify columns that were added from matched_data
    if matched_data is not None:
        sat_cols_added = [col for col in matched_data.columns if col in result_df.columns and col.startswith('sat_')]
        if sat_cols_added:
            print(f"Filling NaNs with 0 for satellite columns: {sat_cols_added}")
            result_df[sat_cols_added] = result_df[sat_cols_added].fillna(0)
        else:
            print("Warning: No satellite columns seem to have been added during the join.")
    else:
        print("Skipping NaN fill for satellite columns as matched_data was None.")


    print(f"Satellite data joining complete. Result shape: {result_df.shape}")
    return result_df

# =============================================================================
# Core Data Processing Functions (Keep functions as they were)
# =============================================================================
def fetch_climate_index(url, var_name, temp_dir):
    """
    Downloads, processes climate index NetCDF, aggregates MONTHLY,
    and returns a DataFrame with 'Month' (Period) and 'index' columns.
    """
    print(f"Fetching climate index: {var_name}...")
    if not url:
        print(f"   Skipping {var_name}: No URL provided.")
        return pd.DataFrame(columns=['Month', 'index']) # Return correct columns

    # Determine local filename using helper, place in temp_dir
    fname = local_filename(url, '.nc', temp_dir=temp_dir)

    # Download using helper (handles tracking and errors)
    if not os.path.exists(fname):
        download_result = download_file(url, fname)
        if download_result is None:
            print(f"   WARNING: Could not download climate index {var_name} from {url}. Skipping.")
            return pd.DataFrame(columns=['Month', 'index']) # Return correct columns
    else:
        print(f"   Using existing file: {os.path.basename(fname)}")
        if fname not in downloaded_files: downloaded_files.append(fname)


    ds = None # Initialize
    try:
        # Open dataset
        ds = xr.open_dataset(fname)

        # Process data
        df = ds.to_dataframe().reset_index()

        # Find time and variable columns robustly
        time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
        if not time_col:
            print(f"   ERROR: No suitable time column found in {os.path.basename(fname)} for {var_name}. Columns: {df.columns}. Skipping.")
            return pd.DataFrame(columns=['Month', 'index']) # Return correct columns

        # Allow for case variations in var_name if needed, but prefer exact match
        actual_var_name = var_name
        if actual_var_name not in df.columns:
             # Simple case-insensitive check as fallback
             var_name_lower = var_name.lower()
             found_var = next((c for c in df.columns if c.lower() == var_name_lower), None)
             if found_var:
                 print(f"   Warning: Variable '{var_name}' not found, using case-insensitive match '{found_var}'.")
                 actual_var_name = found_var
             else:
                 print(f"   ERROR: Variable '{var_name}' not found in {os.path.basename(fname)}. Columns: {df.columns}. Skipping.")
                 return pd.DataFrame(columns=['Month', 'index']) # Return correct columns

        df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df[['datetime', actual_var_name]].dropna().rename(columns={actual_var_name: 'index'})

        if df.empty:
            print(f"   Warning: No valid data after processing for {var_name}. Skipping.")
            return pd.DataFrame(columns=['Month', 'index']) # Return correct columns

        # --- CHANGE: Aggregate MONTHLY ---
        # Create a Period object representing the month (YYYY-MM)
        # Ensure NaT dates are handled before creating period
        df.dropna(subset=['datetime'], inplace=True)
        if df.empty:
             print(f"   Warning: No valid datetime data after dropna for {var_name}. Skipping.")
             return pd.DataFrame(columns=['Month', 'index'])

        df['Month'] = df['datetime'].dt.to_period('M')
        result = df.groupby('Month')['index'].mean().reset_index()
        # Ensure sorted by Month for merge_asof later
        result = result.sort_values('Month')

        print(f"   Successfully processed {len(result)} MONTHLY {var_name} records.")
        return result[['Month', 'index']] # Return Month and index

    except FileNotFoundError:
         print(f"   ERROR: File not found when trying to open {fname} for {var_name}. Should have been downloaded.")
         return pd.DataFrame(columns=['Month', 'index']) # Return correct columns
    except Exception as e:
        print(f"   ERROR processing climate index {os.path.basename(fname)} for {var_name}: {e}")
        # traceback.print_exc() # Optional for detailed debug
        return pd.DataFrame(columns=['Month', 'index']) # Return correct columns
    finally:
        if ds:
            try: ds.close()
            except Exception: pass # Ignore errors on close

def process_streamflow(url, temp_dir):
    """Downloads, processes USGS streamflow JSON, returns weekly avg DataFrame."""
    # (Function content is the same as the previous version)
    print("Fetching streamflow data...")
    if not url:
        print("  Skipping streamflow: No URL provided.")
        return pd.DataFrame(columns=['Date', 'Flow']) # Match expected output cols

    # Determine local filename using helper, place in temp_dir
    fname = local_filename(url, '.json', temp_dir=temp_dir)

    # Download using helper
    if not os.path.exists(fname):
        download_result = download_file(url, fname)
        if download_result is None:
            print("  WARNING: Could not download streamflow data. Skipping.")
            return pd.DataFrame(columns=['Date', 'Flow'])
    else:
        print(f"  Using existing file: {os.path.basename(fname)}")
        if fname not in downloaded_files: downloaded_files.append(fname)

    try:
        # Load JSON
        with open(fname) as f:
            data = json.load(f)

        # Extract values safely
        values = []
        try:
            ts_data = data.get('value', {}).get('timeSeries', [])
            if ts_data and isinstance(ts_data, list) and len(ts_data) > 0:
                 # Find the time series with the desired parameter code (e.g., 00060 for Discharge) or take the first one
                 # This might need adjustment based on the specific service format
                 discharge_ts = next((ts for ts in ts_data if ts.get('variable', {}).get('variableCode', [{}])[0].get('value') == '00060'), None)
                 if discharge_ts is None and len(ts_data) == 1: # Fallback to first if only one exists
                      discharge_ts = ts_data[0]

                 if discharge_ts and 'values' in discharge_ts and isinstance(discharge_ts['values'], list) and len(discharge_ts['values']) > 0:
                    values = discharge_ts['values'][0].get('value', [])
        except Exception as e:
            print(f"  Error parsing structure of streamflow JSON: {e}")
            values = [] # Ensure it's an empty list on error

        if not values or not isinstance(values, list):
            print("  ERROR: No streamflow values found or format unexpected in JSON. Skipping.")
            return pd.DataFrame(columns=['Date', 'Flow'])

        # Parse records
        records = []
        for item in values:
            if isinstance(item, dict) and 'dateTime' in item and 'value' in item:
                try:
                    # Be robust to timezone info if present (common in dateTime)
                    dt = pd.to_datetime(item['dateTime'], errors='coerce', utc=True) # Assume UTC if timezone specified, otherwise naive
                    flow = pd.to_numeric(item['value'], errors='coerce')
                    # Ensure flow is non-negative (physical constraint)
                    if pd.notna(dt) and pd.notna(flow) and flow >= 0:
                        # Store as naive datetime in UTC reference for consistency before weekly grouping
                        records.append({'Date': dt.tz_convert(None) if dt.tzinfo else dt, 'Flow': flow})
                except Exception:
                    continue # Skip malformed records

        if not records:
            print("  WARNING: No valid records parsed from streamflow data. Skipping.")
            return pd.DataFrame(columns=['Date', 'Flow'])

        df = pd.DataFrame(records)

        # Aggregate weekly - Use ISO week ending on Sunday (format %Y-%U-%w with 0 for Sunday)
        # Or use ISO week ending on Monday (%G-%V) if preferred for consistency with climate indices
        df['week_key'] = df['Date'].dt.strftime('%G-%V') # ISO Year-Week
        weekly_flow = df.groupby('week_key')['Flow'].mean().reset_index()

        # Recreate a representative 'Date' for the week (e.g., start of the ISO week - Monday)
        # This requires parsing '%G-%V' back to a date.
        weekly_flow['Date'] = pd.to_datetime(weekly_flow['week_key'] + '-1', format='%G-%V-%w', errors='coerce') # '-1' for Monday
        weekly_flow.dropna(subset=['Date'], inplace=True) # Drop if date parsing failed

        print(f"  Successfully processed {len(weekly_flow)} weekly streamflow records.")
        return weekly_flow[['Date', 'Flow']].sort_values('Date') # Return Date and Flow

    except FileNotFoundError:
         print(f"  ERROR: File not found when trying to open {fname} for streamflow.")
         return pd.DataFrame(columns=['Date', 'Flow'])
    except json.JSONDecodeError as e:
         print(f"  ERROR: Could not decode streamflow JSON file {os.path.basename(fname)}: {e}")
         return pd.DataFrame(columns=['Date', 'Flow'])
    except Exception as e:
        print(f"  ERROR processing streamflow data from {os.path.basename(fname)}: {e}")
        # traceback.print_exc() # Optional
        return pd.DataFrame(columns=['Date', 'Flow'])


def fetch_beuti_data(url, sites_dict, temp_dir, power=2):
    """Downloads BEUTI NetCDF, interpolates to site latitudes, returns daily DataFrame."""
    # (Function content is the same as the previous version)
    print("Fetching BEUTI data...")
    if not url:
        print("  Skipping BEUTI: No URL provided.")
        return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])
    if not sites_dict:
        print("  Skipping BEUTI: No site dictionary provided for interpolation.")
        return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])

    # Determine local filename using helper, place in temp_dir
    fname = local_filename(url, '.nc', temp_dir=temp_dir)

    # Download using helper
    if not os.path.exists(fname):
        download_result = download_file(url, fname)
        if download_result is None:
            print("  WARNING: Could not download BEUTI data. Skipping.")
            return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])
    else:
        print(f"  Using existing file: {os.path.basename(fname)}")
        if fname not in downloaded_files: downloaded_files.append(fname)

    ds = None # Initialize
    try:
        # Open dataset
        ds = xr.open_dataset(fname)

        # Process data
        df = ds.to_dataframe().reset_index()

        # Find required columns robustly
        time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
        lat_col = next((c for c in ['latitude', 'lat'] if c in df.columns), None)
        beuti_var = next((c for c in ['BEUTI', 'beuti'] if c in df.columns), None) # Check columns first
        if beuti_var is None: # Fallback to check data_vars
             beuti_var = next((c for c in ['BEUTI', 'beuti'] if c in ds.data_vars), None)


        if not all([time_col, lat_col, beuti_var]):
             print(f"  ERROR: Missing required columns/variables (time, latitude, BEUTI) in {os.path.basename(fname)}.")
             print(f"    Found Columns: {list(df.columns)}, Found Vars: {list(ds.data_vars)}")
             return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])

        # Prepare DataFrame for interpolation
        df_subset = df[[time_col, lat_col, beuti_var]].copy() # Select only needed cols
        df_subset.rename(columns={time_col: 'Date', lat_col: 'latitude', beuti_var: 'BEUTI'}, inplace=True)

        df_subset['Date'] = pd.to_datetime(df_subset['Date'], errors='coerce')
        # Keep only the date part for daily grouping/interpolation
        df_subset['Date'] = df_subset['Date'].dt.date
        df_subset['latitude'] = pd.to_numeric(df_subset['latitude'], errors='coerce')
        df_subset['BEUTI'] = pd.to_numeric(df_subset['BEUTI'], errors='coerce')

        df_subset.dropna(subset=['Date', 'latitude', 'BEUTI'], inplace=True)

        if df_subset.empty:
            print("  BEUTI data empty after initial processing and NaN removal. Skipping.")
            return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])

        # Sort for efficient processing
        df_sorted = df_subset.sort_values(by=['Date', 'latitude'])

        # --- Interpolation for each site ---
        print(f"  Interpolating BEUTI for {len(sites_dict)} sites...")
        results_list = []
        sites_processed_count = 0
        for site, coords in sites_dict.items():
            # Ensure coords is a list/tuple with at least one value (latitude)
            if isinstance(coords, (list, tuple)) and len(coords) > 0:
                 site_lat = pd.to_numeric(coords[0], errors='coerce')
            else: site_lat = np.nan

            if pd.isna(site_lat):
                print(f"    Warning: Skipping site '{site}' - Invalid or missing latitude in site dictionary.")
                continue

            site_results = []
            # Group by date to interpolate for each day separately
            for date, group in df_sorted.groupby('Date'):
                 if group.empty: continue

                 # Extract latitudes and BEUTI values for the current day
                 lats = group['latitude'].values
                 beuti_vals = group['BEUTI'].values

                 # Simple check for exact match first (optimization)
                 exact_match_indices = np.where(np.isclose(lats, site_lat))[0]
                 if len(exact_match_indices) > 0:
                     # If multiple exact matches (unlikely), average them
                     interpolated_beuti = np.mean(beuti_vals[exact_match_indices])
                 else:
                     # Inverse distance weighting if no exact match
                     distances = np.abs(lats - site_lat)
                     # Avoid division by zero and handle cases with zero distance correctly
                     weights = 1.0 / (distances ** power + 1e-9) # Add epsilon for stability

                     # Ensure no NaNs in weights or values used for calculation
                     valid_indices = ~np.isnan(weights) & ~np.isnan(beuti_vals)
                     if np.any(valid_indices):
                          valid_weights = weights[valid_indices]
                          valid_beuti = beuti_vals[valid_indices]
                          if np.sum(valid_weights) > 1e-9: # Check if sum of weights is non-zero
                              interpolated_beuti = np.sum(valid_beuti * valid_weights) / np.sum(valid_weights)
                          else: # Handle case where all valid points are too far / weights are zero
                              interpolated_beuti = np.nan # Or choose another default, like nearest?
                     else: # No valid points found for interpolation
                         interpolated_beuti = np.nan

                 # Only append if interpolation was successful
                 if pd.notna(interpolated_beuti):
                     site_results.append({'Date': date, 'Site': site, 'BEUTI': interpolated_beuti})

            if site_results:
                results_list.extend(site_results)
                sites_processed_count += 1
            # else: # Verbose
            #      print(f"    No BEUTI results generated for site '{site}'.")


        if not results_list:
            print("  No BEUTI results generated after interpolation for any site. Skipping.")
            return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])

        beuti_final_df = pd.DataFrame(results_list)
        # Convert Date back to datetime objects (start of day)
        beuti_final_df['Date'] = pd.to_datetime(beuti_final_df['Date'])

        print(f"  Successfully processed BEUTI data for {sites_processed_count} sites.")
        return beuti_final_df[['Date', 'Site', 'BEUTI']].sort_values(['Site', 'Date'])

    except FileNotFoundError:
         print(f"  ERROR: File not found when trying to open {fname} for BEUTI.")
         return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])
    except Exception as e:
        print(f"  ERROR processing BEUTI data {os.path.basename(fname)}: {e}")
        traceback.print_exc() # BEUTI interpolation can be complex
        return pd.DataFrame(columns=['Date', 'Site', 'BEUTI'])
    finally:
        if ds:
            try: ds.close()
            except Exception: pass


def process_da(da_files_dict):
    """Processes DA data from Parquet files, returns weekly aggregated DataFrame."""
    # (Function content is the same as the previous version)
    print("\n--- Processing DA Data ---")
    data_frames = []
    if not da_files_dict:
         print("  No DA files provided. Skipping DA processing.")
         return pd.DataFrame(columns=['Year-Week', 'DA_Levels', 'Site']) # Match expected cols

    for name, path in da_files_dict.items():
        # Normalize name from dict key for site name guess
        site_name_guess = name.replace('-da', '').replace('_da', '').replace('-', ' ').replace('_', ' ').title()
        print(f"  Processing DA for site: '{site_name_guess}' from {os.path.basename(path)}")
        if not isinstance(path, str) or not path.lower().endswith('.parquet'):
             print(f"    Warning: Skipping {name} - Path is not a Parquet file string: {path}")
             continue
        if not os.path.exists(path):
             print(f"    Warning: Skipping {name} - Parquet file not found: {path}")
             continue

        try:
            df = pd.read_parquet(path)
            # Identify Date and DA columns (handle variations)
            date_col, da_col = None, None
            if 'CollectDate' in df.columns: date_col = 'CollectDate'
            elif 'Date' in df.columns: date_col = 'Date'
            elif all(c in df.columns for c in ['Harvest Month', 'Harvest Date', 'Harvest Year']):
                # Combine date parts only if individual date column not found
                 try:
                     # Ensure parts are strings before combining robustly
                     df['CombinedDateStr'] = df['Harvest Month'].astype(str) + " " + df['Harvest Date'].astype(str) + ", " + df['Harvest Year'].astype(str)
                     df['Date'] = pd.to_datetime(df['CombinedDateStr'], format='%B %d, %Y', errors='coerce')
                     date_col = 'Date' # Now use the created 'Date' column
                 except Exception as e_date:
                     print(f"    Warning: Could not combine date columns for {name}: {e_date}")
                     # Don't continue, maybe another date col exists
            # Add other common date column names if needed

            if 'Domoic Result' in df.columns: da_col = 'Domoic Result'
            elif 'Domoic Acid' in df.columns: da_col = 'Domoic Acid'
            elif 'DA' in df.columns: da_col = 'DA'
            # Add other common DA column names

            if not date_col: # Try finding 'Date' again if combining failed but 'Date' existed
                 if 'Date' in df.columns: date_col = 'Date'

            if not date_col or not da_col:
                 print(f"    Warning: Skipping {name} - Could not find required Date ({date_col}) or DA ({da_col}) columns. Found: {list(df.columns)}")
                 continue

            # Process valid columns
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['DA_Levels'] = pd.to_numeric(df[da_col], errors='coerce')

            # Use site column if it exists, otherwise use the name derived from filename
            # Normalize site column values for consistency
            site_col = next((c for c in df.columns if c.lower() in ['site', 'location', 'area', 'site name', 'beach name']), None)
            if site_col:
                df['Site'] = df[site_col].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title() # Use existing site info, normalized
                print(f"    Using site information from column '{site_col}'.")
            else:
                df['Site'] = site_name_guess # Fallback to filename derived site (already normalized)


            df.dropna(subset=['Parsed_Date', 'DA_Levels', 'Site'], inplace=True)
            if df.empty:
                 print(f"    Warning: No valid DA data after cleaning for {name}.")
                 continue

            # Aggregate weekly - Use ISO week for consistency
            df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
            # Group by week AND site (using the determined Site column)
            weekly_da = df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()

            data_frames.append(weekly_da[['Year-Week', 'DA_Levels', 'Site']])
            print(f"    Successfully processed {len(weekly_da)} weekly DA records for {name}.")

        except Exception as e:
            print(f"  Error processing DA file {name} ({os.path.basename(path)}): {e}")
            # traceback.print_exc() # Optional

    if not data_frames:
        print("  No DA dataframes were successfully processed.")
        return pd.DataFrame(columns=['Year-Week', 'DA_Levels', 'Site'])

    print("Combining all processed DA data...")
    final_da_df = pd.concat(data_frames, ignore_index=True)
    # Add a final group-by after concat to handle cases where different files might represent the same site-week
    if not final_da_df.empty:
         final_da_df = final_da_df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()
    print(f"Combined DA data shape: {final_da_df.shape}")
    return final_da_df


def process_pn(pn_files_dict):
    """Processes PN data from Parquet files, returns weekly aggregated DataFrame."""
    # (Function content is the same as the previous version)
    print("\n--- Processing PN Data ---")
    data_frames = []
    if not pn_files_dict:
         print("  No PN files provided. Skipping PN processing.")
         return pd.DataFrame(columns=['Year-Week', 'PN_Levels', 'Site']) # Match expected cols

    for name, path in pn_files_dict.items():
        # Normalize name from dict key for site name guess
        site_name_guess = name.replace('-pn', '').replace('_pn', '').replace('-', ' ').replace('_', ' ').title()
        print(f"  Processing PN for site: '{site_name_guess}' from {os.path.basename(path)}")
        if not isinstance(path, str) or not path.lower().endswith('.parquet'):
             print(f"    Warning: Skipping {name} - Path is not a Parquet file string: {path}")
             continue
        if not os.path.exists(path):
             print(f"    Warning: Skipping {name} - Parquet file not found: {path}")
             continue

        try:
            df = pd.read_parquet(path)
            # Identify Date and PN columns (handle variations)
            date_col, pn_col = None, None
            # Try various common date column names
            date_col_candidates = ['Date', 'SampleDate', 'CollectDate', 'Sample Date']
            date_col = next((c for c in date_col_candidates if c in df.columns), None)

            # Try various common PN column names/patterns - be more specific
            pn_col_candidates = [c for c in df.columns if "pseudo" in str(c).lower() and "nitzschia" in str(c).lower()]
            # Fallback if specific name not found
            if not pn_col_candidates:
                 pn_col_candidates = [c for c in df.columns if "pn" in str(c).lower() and "level" in str(c).lower()] # e.g., 'pn_level'
                 if not pn_col_candidates: # Broader fallback
                      pn_col_candidates = [c for c in df.columns if "pn" in str(c).lower()]

            if len(pn_col_candidates) == 1:
                pn_col = pn_col_candidates[0]
            elif len(pn_col_candidates) > 1:
                 # Try to find the most likely candidate (e.g., shortest name, or one ending in 'cells/L'?)
                 # Simple approach: use the first one found
                 print(f"    Warning: Multiple possible PN columns found in {name}: {pn_col_candidates}. Using the first one: '{pn_col_candidates[0]}'.")
                 pn_col = pn_col_candidates[0]

            if not date_col or not pn_col:
                 print(f"    Warning: Skipping {name} - Could not find required Date ({date_col}) or PN ({pn_col}) columns. Found: {list(df.columns)}")
                 continue

            # Process valid columns
            # Try multiple date formats robustly
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            # Example: Add more formats if needed based on actual data
            if df['Parsed_Date'].isna().all():
                 try: # Try common M/D/Y format
                      df['Parsed_Date'] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
                 except ValueError: pass # Ignore if format fails
            if df['Parsed_Date'].isna().all():
                 try: # Try common M/D/YY format
                      df['Parsed_Date'] = pd.to_datetime(df[date_col], format='%m/%d/%y', errors='coerce')
                 except ValueError: pass # Ignore if format fails

            df['PN_Levels'] = pd.to_numeric(df[pn_col], errors='coerce')

            # Use site column if it exists, otherwise use the name derived from filename
            # Normalize site column values
            site_col = next((c for c in df.columns if c.lower() in ['site', 'location', 'area', 'site name', 'beach name']), None)
            if site_col:
                df['Site'] = df[site_col].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title() # Use existing site info, normalized
                print(f"    Using site information from column '{site_col}'.")
            else:
                df['Site'] = site_name_guess # Fallback to filename derived site (already normalized)

            df.dropna(subset=['Parsed_Date', 'PN_Levels', 'Site'], inplace=True)
            if df.empty:
                 print(f"    Warning: No valid PN data after cleaning for {name}.")
                 continue

            # Aggregate weekly - Use ISO week for consistency
            df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
            # Group by week AND site
            weekly_pn = df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()

            data_frames.append(weekly_pn[['Year-Week', 'PN_Levels', 'Site']])
            print(f"    Successfully processed {len(weekly_pn)} weekly PN records for {name}.")

        except Exception as e:
            print(f"  Error processing PN file {name} ({os.path.basename(path)}): {e}")
            # traceback.print_exc() # Optional

    if not data_frames:
        print("  No PN dataframes were successfully processed.")
        return pd.DataFrame(columns=['Year-Week', 'PN_Levels', 'Site'])

    print("Combining all processed PN data...")
    final_pn_df = pd.concat(data_frames, ignore_index=True)
    # Add a final group-by after concat
    if not final_pn_df.empty:
        final_pn_df = final_pn_df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()
    print(f"Combined PN data shape: {final_pn_df.shape}")
    return final_pn_df


def generate_compiled_data(sites_dict, start_dt, end_dt):
    """Generates a base DataFrame with all Site-Week combinations."""
    # (Function content is the same as the previous version)
    print("\n--- Generating Base Site-Week DataFrame ---")
    if not sites_dict:
        print("  Error: Site dictionary is empty. Cannot generate base data.")
        return pd.DataFrame()
    if pd.isna(start_dt) or pd.isna(end_dt):
         print("  Error: Start or end date is invalid. Cannot generate base data.")
         return pd.DataFrame()

    print(f"  Generating weekly entries from {start_dt.date()} to {end_dt.date()}.")
    # Generate weekly dates - Use Monday as the start of the week for consistency ('W-MON')
    # or use '%G-%V' week definitions if preferred. Let's stick to W-MON for now.
    weeks = pd.date_range(start_dt, end_dt, freq='W-MON', name='Date') # W-MON = Weekly, starting Monday
    if weeks.empty:
         print("  Error: No weeks generated in the specified date range.")
         return pd.DataFrame()

    df_list = []
    print(f"  Using site definitions:")
    for site, coords in sites_dict.items():
         # Ensure coords are valid lat/lon
         lat, lon = (coords[0], coords[1]) if isinstance(coords, (list, tuple)) and len(coords) == 2 else (np.nan, np.nan)
         if pd.isna(lat) or pd.isna(lon):
              print(f"    Warning: Invalid coordinates for site '{site}'. Lat/Lon will be NaN.")
         else: # Print valid sites being used
             print(f"    - {site}: ({lat}, {lon})")
         # Normalize site name when creating the dataframe
         normalized_site = site.replace('_', ' ').replace('-', ' ').title()
         site_df = pd.DataFrame({'Date': weeks, 'Site': normalized_site, 'latitude': lat, 'longitude': lon})
         df_list.append(site_df)

    if not df_list:
        print("  Error: No site dataframes created.")
        return pd.DataFrame()

    compiled_df = pd.concat(df_list, ignore_index=True)
    print(f"  Generated base DataFrame with {len(compiled_df)} site-week rows.")
    # Ensure Date is datetime
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])
    return compiled_df.sort_values(['Site', 'Date'])


def compile_data(compiled_df, oni_df, pdo_df, streamflow_df):
    """
    Merges climate indices (monthly nearest) and streamflow data (weekly backward)
    into the base DataFrame.
    """
    print("\n--- Merging Environmental Data (Climate Indices, Streamflow) ---")
    if compiled_df is None or compiled_df.empty:
        print("   Base compiled DataFrame is empty. Cannot merge environmental data.")
        return compiled_df

    # Ensure base 'Date' is datetime
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'], errors='coerce')
    compiled_df.dropna(subset=['Date'], inplace=True)
    if compiled_df.empty:
        print("   Base DataFrame empty after Date conversion/dropna. Aborting merge.")
        return compiled_df

    # --- Prepare for Monthly Merge ---
    # Create Month Period column for merging ONI/PDO
    compiled_df.dropna(subset=['Date'], inplace=True) # Re-check after potential NaT coercion
    if compiled_df.empty:
         print("   Base DataFrame empty after Date dropna before month conversion. Aborting merge.")
         return compiled_df
    compiled_df['Month'] = compiled_df['Date'].dt.to_period('M')

    # --- FIX: Sort LEFT DataFrame by 'Month' right before merge_asof ---
    print("   Sorting base data by 'Month' for climate index merge...")
    # Ensure no NaNs in Month column before sorting
    compiled_df.dropna(subset=['Month'], inplace=True)
    if compiled_df.empty:
        print("   Base DataFrame empty after Month dropna. Aborting climate merge.")
    else:
        compiled_df = compiled_df.sort_values('Month') # <<< Sort strictly by Month here

        # --- Merge ONI using merge_asof (nearest month) ---
        if oni_df is not None and not oni_df.empty and 'Month' in oni_df.columns and 'index' in oni_df.columns:
            print(f"   Merging ONI data ({len(oni_df)} monthly records) using nearest month...")
            oni_df = oni_df.sort_values('Month')
            oni_df['Month'] = pd.to_datetime(oni_df['Month'].astype(str), errors='coerce').dt.to_period('M')
            oni_df.dropna(subset=['Month'], inplace=True)

            if not oni_df.empty:
                compiled_df = pd.merge_asof( # Now compiled_df IS sorted by 'Month'
                    compiled_df,
                    oni_df[['Month', 'index']],
                    on='Month',
                    direction='nearest'
                )
                compiled_df.rename(columns={'index': 'ONI'}, inplace=True)
                print(f"     ONI merge complete. {compiled_df.get('ONI', pd.Series(dtype=float)).notna().sum()} values assigned.") # Safely check count
            else:
                 print("   ONI data became empty after month conversion. Filling ONI column with NaN.")
                 compiled_df['ONI'] = np.nan
        else:
            print("   ONI data not available or invalid. Filling ONI column with NaN.")
            compiled_df['ONI'] = np.nan # Ensure column exists


        # --- Merge PDO using merge_asof (nearest month) ---
        # No need to re-sort compiled_df by 'Month' if it wasn't modified in between ONI/PDO merge
        if pdo_df is not None and not pdo_df.empty and 'Month' in pdo_df.columns and 'index' in pdo_df.columns:
            print(f"   Merging PDO data ({len(pdo_df)} monthly records) using nearest month...")
            pdo_df = pdo_df.sort_values('Month')
            pdo_df['Month'] = pd.to_datetime(pdo_df['Month'].astype(str), errors='coerce').dt.to_period('M')
            pdo_df.dropna(subset=['Month'], inplace=True)

            if not pdo_df.empty:
                # Ensure compiled_df still exists and has 'Month' before merging
                if 'Month' in compiled_df.columns:
                    compiled_df = pd.merge_asof( # compiled_df should still be sorted by 'Month'
                        compiled_df,
                        pdo_df[['Month', 'index']],
                        on='Month',
                        direction='nearest'
                    )
                    compiled_df.rename(columns={'index': 'PDO'}, inplace=True)
                    print(f"     PDO merge complete. {compiled_df.get('PDO', pd.Series(dtype=float)).notna().sum()} values assigned.") # Safely check count
                else:
                    print("   PDO merge skipped: 'Month' column missing from base data at this stage.")
                    compiled_df['PDO'] = np.nan

            else:
                 print("   PDO data became empty after month conversion. Filling PDO column with NaN.")
                 compiled_df['PDO'] = np.nan
        else:
            print("   PDO data not available or invalid. Filling PDO column with NaN.")
            compiled_df['PDO'] = np.nan # Ensure column exists

    # --- Drop the temporary Month key ---
    compiled_df.drop(columns=['Month'], inplace=True, errors='ignore')


    # --- Streamflow Merge (remains the same weekly logic) ---
    if streamflow_df is not None and not streamflow_df.empty and 'Date' in streamflow_df.columns and 'Flow' in streamflow_df.columns:
        print(f"   Merging Streamflow data ({len(streamflow_df)} weekly records) using merge_asof...")
        streamflow_df['Date'] = pd.to_datetime(streamflow_df['Date'], errors='coerce')
        streamflow_df.dropna(subset=['Date'], inplace=True)
        streamflow_df = streamflow_df.sort_values('Date')

        # --- FIX: Sort LEFT DataFrame by 'Date' right before merge_asof ---
        print("   Sorting base data by 'Date' for streamflow merge...")
        # Ensure 'Date' exists and has no NaNs before sorting/merging
        compiled_df.dropna(subset=['Date'], inplace=True)
        if compiled_df.empty:
             print("   Base DataFrame empty after Date dropna before streamflow merge. Skipping streamflow merge.")
             compiled_df['Streamflow'] = np.nan
        elif not streamflow_df.empty:
            compiled_df = compiled_df.sort_values('Date') # <<< Sort strictly by Date here
            compiled_df = pd.merge_asof( # Now compiled_df IS sorted by 'Date'
                compiled_df,
                streamflow_df[['Date', 'Flow']],
                on='Date',
                direction='backward',
                tolerance=pd.Timedelta('7days')
            )
            compiled_df.rename(columns={'Flow': 'Streamflow'}, inplace=True)
            print(f"     Streamflow merge complete. {compiled_df.get('Streamflow', pd.Series(dtype=float)).notna().sum()} matches found.") # Safely check count
        else:
            print("    Skipping streamflow merge due to empty streamflow DataFrame. Filling Streamflow column with NaN.")
            compiled_df['Streamflow'] = np.nan

    else:
        print("   Streamflow data not available or invalid. Filling Streamflow column with NaN.")
        compiled_df['Streamflow'] = np.nan # Ensure column exists


    print(f"Environmental data merge complete. Shape: {compiled_df.shape}")
    # Return sorted by the original primary keys
    return compiled_df.sort_values(['Site', 'Date'])


def compile_da_pn(lt_df, da_df, pn_df):
    """Merges weekly DA and PN data, interpolates, and applies threshold/fill."""
    # (Function content is the same as the previous version)
    print("\n--- Merging DA and PN Data ---")
    if lt_df is None or lt_df.empty:
        print("  Base DataFrame is empty. Cannot merge DA/PN data.")
        return lt_df

    lt_df_merged = lt_df.copy() # Work on a copy
    # Ensure base 'Date' is datetime (should be already, but check)
    lt_df_merged['Date'] = pd.to_datetime(lt_df_merged['Date'], errors='coerce')

    # --- Merge DA Data ---
    if da_df is not None and not da_df.empty and 'Year-Week' in da_df.columns and 'DA_Levels' in da_df.columns and 'Site' in da_df.columns:
        print(f"  Merging DA data ({len(da_df)} weekly records)...")
        da_df_copy = da_df.copy()
        # Convert Year-Week ('%G-%V') to the Monday date of that week for merging
        da_df_copy['Date'] = pd.to_datetime(da_df_copy['Year-Week'] + '-1', format='%G-%V-%w', errors='coerce')
        da_df_copy.dropna(subset=['Date', 'Site', 'DA_Levels'], inplace=True)

        if not da_df_copy.empty:
             # Normalize Site names in DA data before merging
             da_df_copy['Site'] = da_df_copy['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()

             # Merge based on the calculated weekly 'Date' and 'Site'
             # Ensure 'Site' in lt_df_merged is also normalized if it wasn't already in generate_compiled_data
             lt_df_merged['Site'] = lt_df_merged['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()

             lt_df_merged = pd.merge(lt_df_merged, da_df_copy[['Date', 'Site', 'DA_Levels']], on=['Date', 'Site'], how='left')
             # Rename immediately to avoid confusion before interpolation
             lt_df_merged.rename(columns={'DA_Levels': 'DA_Levels_orig'}, inplace=True)
             print(f"    DA merge complete. {lt_df_merged['DA_Levels_orig'].notna().sum()} matches found.")
        else:
             print("    DA data became empty after date conversion/dropna. Filling DA column with NaN.")
             lt_df_merged['DA_Levels_orig'] = np.nan
    else:
        print("  DA data not available or invalid. Filling DA column with NaN.")
        lt_df_merged['DA_Levels_orig'] = np.nan

    # --- Merge PN Data ---
    if pn_df is not None and not pn_df.empty and 'Year-Week' in pn_df.columns and 'PN_Levels' in pn_df.columns and 'Site' in pn_df.columns:
        print(f"  Merging PN data ({len(pn_df)} weekly records)...")
        pn_df_copy = pn_df.copy()
        # Convert Year-Week ('%G-%V') to the Monday date of that week
        pn_df_copy['Date'] = pd.to_datetime(pn_df_copy['Year-Week'] + '-1', format='%G-%V-%w', errors='coerce')
        pn_df_copy.dropna(subset=['Date', 'Site', 'PN_Levels'], inplace=True)

        if not pn_df_copy.empty:
             # Normalize Site names in PN data
             pn_df_copy['Site'] = pn_df_copy['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()

             # Ensure 'Site' in lt_df_merged is normalized (should be already)
             lt_df_merged['Site'] = lt_df_merged['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()

             # Merge based on the calculated weekly 'Date' and 'Site'
             lt_df_merged = pd.merge(lt_df_merged, pn_df_copy[['Date', 'Site', 'PN_Levels']], on=['Date', 'Site'], how='left')
             # PN_Levels column name is already distinct
             print(f"    PN merge complete. {lt_df_merged['PN_Levels'].notna().sum()} matches found.")
        else:
             print("    PN data became empty after date conversion/dropna. Filling PN column with NaN.")
             if 'PN_Levels' not in lt_df_merged.columns: lt_df_merged['PN_Levels'] = np.nan # Ensure column exists even if empty merge
    else:
        print("  PN data not available or invalid. Filling PN column with NaN.")
        if 'PN_Levels' not in lt_df_merged.columns: lt_df_merged['PN_Levels'] = np.nan # Ensure column exists

    # --- Interpolation and Filling ---
    print("  Interpolating missing DA/PN values linearly within each site group...")
    lt_df_merged = lt_df_merged.sort_values(by=['Site', 'Date'])

    # Interpolate DA
    if 'DA_Levels_orig' in lt_df_merged.columns:
        lt_df_merged['DA_Levels'] = lt_df_merged.groupby('Site')['DA_Levels_orig'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both', limit_area=None) # Interpolate over NaNs
        )
        # Optional: Apply thresholding *after* interpolation
        # lt_df['DA_Levels'] = lt_df['DA_Levels'].apply(lambda x: 0 if pd.notna(x) and x < 1 else x)
        lt_df_merged.drop(columns=['DA_Levels_orig'], inplace=True, errors='ignore')
        print(f"    DA interpolation complete. Non-NaN count: {lt_df_merged['DA_Levels'].notna().sum()}")
    elif 'DA_Levels' not in lt_df_merged.columns: # Ensure column exists if original was missing
         lt_df_merged['DA_Levels'] = np.nan

    # Interpolate PN
    if 'PN_Levels' in lt_df_merged.columns:
        lt_df_merged['PN_Levels'] = lt_df_merged.groupby('Site')['PN_Levels'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both', limit_area=None) # Interpolate over NaNs
        )
        print(f"    PN interpolation complete. Non-NaN count: {lt_df_merged['PN_Levels'].notna().sum()}")
    # else: PN_Levels column should exist from merge step, even if all NaN

    # Final Fill and Thresholding
    # Apply DA threshold (set values < 1 to 0 AFTER interpolation)
    if 'DA_Levels' in lt_df_merged.columns:
         lt_df_merged['DA_Levels'] = lt_df_merged['DA_Levels'].apply(lambda x: 0 if pd.notna(x) and x < 1 else x)
         print("    Applied DA threshold (<1 set to 0).")

    # Fill remaining NaNs (likely at ends where interpolation didn't reach, or if all orig were NaN) with 0
    fill_cols = ['DA_Levels', 'PN_Levels']
    for col in fill_cols:
        if col in lt_df_merged.columns:
             lt_df_merged[col] = lt_df_merged[col].fillna(0)
    print("  Filled remaining NaN values in DA/PN columns with 0.")

    # Remove potential duplicate columns if merges somehow created them
    lt_df_final = lt_df_merged.loc[:, ~lt_df_merged.columns.duplicated()]
    print(f"DA/PN merge and processing complete. Shape: {lt_df_final.shape}")
    return lt_df_final


def filter_data(data_df, cutoff_yr, cutoff_wk):
    """Filters data to include only records from cutoff week/year onwards."""
    # (Function content is the same as the previous version)
    print(f"\n--- Filtering Data by Date Cutoff ({cutoff_yr}-W{cutoff_wk:02d}) ---")
    if data_df is None or data_df.empty:
        print("  Dataframe is empty. No filtering applied.")
        return data_df
    if 'Date' not in data_df.columns:
        print("  Error: 'Date' column not found. Cannot filter by date.")
        return data_df

    data_df_copy = data_df.copy() # Work on copy
    data_df_copy['Date'] = pd.to_datetime(data_df_copy['Date'], errors='coerce')
    data_df_copy.dropna(subset=['Date'], inplace=True)
    if data_df_copy.empty:
        print("  Dataframe empty after Date conversion/dropna. No filtering applied.")
        return data_df_copy

    # Use ISO year and week for filtering comparison
    try:
        isocal = data_df_copy['Date'].dt.isocalendar()
        data_df_copy['Year'] = isocal.year
        data_df_copy['Week'] = isocal.week
    except Exception as e:
        print(f"  Error calculating ISO calendar year/week: {e}. Cannot filter.")
        return data_df # Return original if calendar fails

    original_rows = len(data_df_copy)
    # Apply filter: Keep rows where (Year > cutoff_yr) OR (Year == cutoff_yr AND Week >= cutoff_wk)
    filtered_df = data_df_copy[
        (data_df_copy['Year'] > cutoff_yr) |
        ((data_df_copy['Year'] == cutoff_yr) & (data_df_copy['Week'] >= cutoff_wk))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    filtered_rows = len(filtered_df)
    print(f"  Filtering complete. Kept {filtered_rows} rows out of {original_rows}.")

    # Drop temporary Year/Week columns
    return filtered_df.drop(columns=['Year', 'Week'])


def process_duplicates(data_df):
    """Handles duplicate rows based on Date and Site by averaging numeric columns."""
    # (Function content is the same as the previous version)
    print("\n--- Processing Duplicates (Group by Date, Site) ---")
    if data_df is None or data_df.empty:
        print("  Dataframe is empty. No duplicates to process.")
        return data_df

    key_cols = ['Date', 'Site']
    if not all(c in data_df.columns for c in key_cols):
        print(f"  Error: Missing key columns {key_cols} for duplicate processing.")
        return data_df

    # Check if duplicates exist before processing
    duplicates_exist = data_df.duplicated(subset=key_cols).any()
    if not duplicates_exist:
        print("  No duplicate Date-Site pairs found.")
        return data_df

    print(f"  Found {data_df.duplicated(subset=key_cols).sum()} duplicate Date-Site pairs. Aggregating...")
    original_rows = len(data_df)

    # Define aggregation logic
    numeric_cols = data_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude lat/lon from averaging, keep the first value encountered
    agg_dict = {}
    for col in numeric_cols:
        if col.lower() in ['latitude', 'longitude']:
            agg_dict[col] = 'first' # Keep first lat/lon
        else:
            agg_dict[col] = 'mean' # Average other numeric columns like ONI, PDO, DA, PN etc.

    # Add non-numeric columns we want to keep (take the first occurrence)
    non_numeric_cols = data_df.select_dtypes(exclude=np.number).columns.tolist()
    for col in non_numeric_cols:
        if col not in key_cols: # Don't need to aggregate the grouping keys
            # Check if column exists before adding to dict
            if col in data_df.columns:
                agg_dict[col] = 'first'


    # Perform the aggregation
    try:
        aggregated_df = data_df.groupby(key_cols, as_index=False).agg(agg_dict)
        processed_rows = len(aggregated_df)
        print(f"  Aggregation complete. Reduced rows from {original_rows} to {processed_rows}.")
        return aggregated_df.sort_values(key_cols) # Return sorted
    except Exception as e:
        print(f"  Error during duplicate aggregation: {e}")
        traceback.print_exc()
        return data_df # Return original df on error


def convert_and_fill(data_df):
    """Converts columns to numeric where possible and fills remaining NaNs in numeric cols with 0."""
    # (Function content is the same as the previous version)
    print("\n--- Converting Data Types and Filling NaNs ---")
    if data_df is None or data_df.empty:
        print("  Dataframe is empty. Skipping conversion and filling.")
        return data_df

    df_processed = data_df.copy()
    cols_to_process = df_processed.columns.difference(['Date', 'Site']) # Exclude key identifiers

    print("  Attempting to convert columns to numeric...")
    converted_count = 0
    for col in cols_to_process:
        # Check if column exists and is not already numeric
        if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
             original_dtype = df_processed[col].dtype
             # Apply conversion
             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
             new_dtype = df_processed[col].dtype
             if str(original_dtype) != str(new_dtype):
                 converted_count += 1
                 # print(f"    Converted column '{col}' from {original_dtype} to {new_dtype}.") # Verbose
    if converted_count > 0: print(f"    Converted {converted_count} columns to numeric (or attempted).")
    else: print("    No non-numeric columns found needing conversion.")


    # Fill NaNs with 0 *only* for columns that are now numeric
    print("  Filling NaNs with 0 in numeric columns...")
    num_cols = df_processed.select_dtypes(include=np.number).columns
    nan_before = df_processed[num_cols].isna().sum().sum()
    if nan_before > 0:
         df_processed[num_cols] = df_processed[num_cols].fillna(0)
         nan_after = df_processed[num_cols].isna().sum().sum()
         print(f"    Filled {nan_before - nan_after} NaN values.")
    else:
         print("    No NaNs found in numeric columns to fill.")

    # Ensure Date is still datetime
    if 'Date' in df_processed.columns:
         df_processed['Date'] = pd.to_datetime(df_processed['Date'])

    print("Conversion and filling complete.")
    return df_processed


# =============================================================================
# Main Execution Logic
# =============================================================================
def main(
    run_satellite: bool, # Flag from config
    sat_meta: dict,
    site_dict: dict,
    da_f: dict,
    pn_f: dict,
    start_dt: datetime,
    end_dt: datetime,
    stream_url: str,
    pdo_url: str,
    oni_url: str,
    beuti_url: str,
    yr_cutoff: int,
    wk_cutoff: int,
    output_path: str,
    sat_intermediate_path: str
    ):
    """Main data processing pipeline."""
    print("\n======= Starting Data Processing Pipeline =======")
    start_time = datetime.now()

    # --- Setup Temporary Directory for Downloads ---
    download_temp_dir = None
    try:
        download_temp_dir = tempfile.mkdtemp(prefix="data_dl_")
        print(f"Using temporary directory for general downloads: {download_temp_dir}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create temporary download directory: {e}")
        return # Cannot proceed without a temp directory

    # --- 1. Convert Input CSVs to Parquet (if necessary) ---
    # Assumes da_f and pn_f are dictionaries like {"siteA": "path/to/siteA.csv", ...}
    da_files_parquet = convert_files_to_parquet(da_f)
    pn_files_parquet = convert_files_to_parquet(pn_f)

    # --- 2. Generate Satellite Parquet File (Pre-processing) ---
    satellite_parquet_file_path = None # Initialize
    if run_satellite:
        satellite_parquet_file_path = generate_satellite_parquet(
            sat_meta,                 # Satellite metadata dictionary
            list(site_dict.keys()),   # List of main sites to process for
            sat_intermediate_path     # Output path for the intermediate file
        )
        # Note: generate_satellite_parquet now adds sat_intermediate_path to generated_parquet_files on success.
    else:
        print("\nSkipping satellite data generation step as per configuration.")


    # --- 3. Run Core Data Processing Pipeline ---
    # Process DA/PN data from Parquet files
    da_data = process_da(da_files_parquet)
    pn_data = process_pn(pn_files_parquet)

    # Fetch and process environmental data (pass the general temp dir)
    streamflow_data = process_streamflow(stream_url, download_temp_dir)
    pdo_data = fetch_climate_index(pdo_url, 'PDO', download_temp_dir)
    oni_data = fetch_climate_index(oni_url, 'ONI', download_temp_dir)
    beuti_data = fetch_beuti_data(beuti_url, site_dict, download_temp_dir)

    # Compile base dataframe (Site-Week combinations)
    compiled_base = generate_compiled_data(site_dict, start_dt, end_dt)

    # Merge Environmental Data
    lt_data = compile_data(compiled_base, oni_data, pdo_data, streamflow_data)

    # Merge DA/PN Data (includes interpolation and filling)
    lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)

    # Filter by Date Cutoff
    filtered_data = filter_data(lt_da_pn, yr_cutoff, wk_cutoff)

    # Process Duplicates (Aggregate after most merges)
    aggregated_data = process_duplicates(filtered_data)

    # Final Type Conversion and NaN Filling (before satellite and BEUTI merge)
    base_final_data = convert_and_fill(aggregated_data)

    # Merge BEUTI Data (after main processing, before satellite)
    print("\n--- Merging BEUTI Data ---")
    if beuti_data is not None and not beuti_data.empty:
        if 'Date' not in base_final_data.columns or 'Site' not in base_final_data.columns:
             print("  Error: Cannot merge BEUTI - Base data missing 'Date' or 'Site' columns.")
             if 'BEUTI' not in base_final_data.columns: base_final_data['BEUTI'] = 0 # Add column but fill with 0
        else:
             # Ensure Date columns are compatible (both datetime)
             base_final_data['Date'] = pd.to_datetime(base_final_data['Date'])
             beuti_data['Date'] = pd.to_datetime(beuti_data['Date'])
             original_rows = len(base_final_data)
             # Ensure Site names are normalized before merge
             base_final_data['Site'] = base_final_data['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()
             beuti_data['Site'] = beuti_data['Site'].astype(str).str.replace('_', ' ').str.replace('-', ' ').str.title()

             base_final_data = pd.merge(base_final_data, beuti_data, on=['Date', 'Site'], how='left')
             # Fill NaNs introduced by the merge with 0 for BEUTI
             base_final_data['BEUTI'] = base_final_data['BEUTI'].fillna(0)
             print(f"  BEUTI merge complete. Shape changed from {original_rows} to {len(base_final_data)} rows (should be same if merge='left').")
    else:
        print("  BEUTI data not available or invalid. Filling BEUTI column with 0.")
        if 'BEUTI' not in base_final_data.columns: base_final_data['BEUTI'] = 0

    # Ensure core columns exist before satellite merge
    core_cols = ['Date', 'Site', 'latitude', 'longitude', 'ONI', 'PDO', 'Streamflow', 'DA_Levels', 'PN_Levels', 'BEUTI']
    for col in core_cols:
        if col not in base_final_data.columns:
            print(f"Warning: Column '{col}' missing before satellite merge. Adding with NaNs/0.")
            # Determine appropriate fill value based on likely type
            if col in ['Date']: fill_val = pd.NaT
            elif col in ['Site']: fill_val = 'Unknown'
            else: fill_val = 0 # Assume numeric otherwise
            base_final_data[col] = fill_val


    # Final check on 'Date' type
    if 'Date' in base_final_data.columns:
         base_final_data['Date'] = pd.to_datetime(base_final_data['Date'])
         base_final_data.dropna(subset=['Date'], inplace=True) # Crucial before satellite merge


    print(f"\n--- Core Data Pipeline Complete. Shape before satellite merge: {base_final_data.shape} ---")
    if not base_final_data.empty: print(f"Columns: {base_final_data.columns.tolist()}")


    # --- 4. Add Satellite Data (Final Step) ---
    if run_satellite and satellite_parquet_file_path and os.path.exists(satellite_parquet_file_path):
        final_data_with_satellite = add_satellite_data(base_final_data, satellite_parquet_file_path)
    else:
        if run_satellite: # Only print warning if it was supposed to run but file is missing
             print("\nSkipping satellite data merge: Intermediate file was not generated or found.")
        else: # Otherwise, just confirm it was skipped by config
             print("\nSkipping satellite data merge as per configuration.")

        final_data_with_satellite = base_final_data # Use data without satellite info
        # Ensure satellite columns don't exist if merge skipped
        sat_cols_to_remove = [col for col in final_data_with_satellite.columns if col.startswith('sat_')]
        if sat_cols_to_remove:
             print(f"  Removing potentially existing satellite columns: {sat_cols_to_remove}")
             final_data_with_satellite = final_data_with_satellite.drop(columns=sat_cols_to_remove)


   # --- 5. Final Checks and Save Output ---
    print("\n--- Final Checks and Saving Output ---")
    if final_data_with_satellite is None or final_data_with_satellite.empty:
        print("ERROR: Final DataFrame is empty or None after all processing steps. Cannot save output.")
    else:
        try:
            final_data_to_save = final_data_with_satellite.copy()

            # Define expected final columns (core + potential satellite)
            final_core_cols = ["Date", "Site", "latitude", "longitude", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels", "BEUTI"]
            sat_cols_present = sorted([col for col in final_data_to_save.columns if col.startswith('sat_')])
            final_expected_cols = final_core_cols + sat_cols_present

            # Ensure all expected columns exist, fill with 0/NaN if somehow missing
            cols_added = []
            for col in final_expected_cols:
                    if col not in final_data_to_save.columns:
                            cols_added.append(col)
                            # Determine appropriate fill value
                            if col in ['Date']: fill_val = pd.NaT
                            elif col in ['Site']: fill_val = 'Unknown'
                            else: fill_val = 0 # Assume numeric otherwise
                            final_data_to_save[col] = fill_val
            if cols_added: print(f"Warning: Added missing final columns and filled: {cols_added}")


            # Reorder columns for consistency
            # Ensure only columns that ACTUALLY exist are selected
            final_ordered_cols = [col for col in final_expected_cols if col in final_data_to_save.columns]
            other_cols = [col for col in final_data_to_save.columns if col not in final_ordered_cols]
            final_data_to_save = final_data_to_save[final_ordered_cols + other_cols]


            # Final sort
            if 'Site' in final_data_to_save.columns and 'Date' in final_data_to_save.columns:
                # Ensure Date is datetime before sorting if it wasn't already
                final_data_to_save['Date'] = pd.to_datetime(final_data_to_save['Date'], errors='coerce')
                final_data_to_save = final_data_to_save.sort_values(['Site', 'Date'])

            # <<< --- ADD THIS LINE --- >>>
            # Convert the 'Date' column to the desired string format "MM/DD/YYYY" BEFORE saving
            if 'Date' in final_data_to_save.columns:
                 print("Converting 'Date' column to string format 'MM/DD/YYYY' for output...")
                 final_data_to_save['Date'] = final_data_to_save['Date'].dt.strftime('%m/%d/%Y')
            # <<< --- END OF ADDED LINE --- >>>

            print(f"Saving final merged data ({final_data_to_save.shape[0]} rows, {final_data_to_save.shape[1]} cols) to {output_path}...")
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                    print(f"  Creating output directory: {output_dir}")
                    os.makedirs(output_dir)

            final_data_to_save.to_parquet(output_path, index=False) # Now 'Date' column will be saved as strings
            print(f"\nFinal output saved successfully to '{output_path}'")
            print(f"Final DataFrame shape: {final_data_to_save.shape}")
            print(f"Final Columns: {final_data_to_save.columns.tolist()}")

        except Exception as e:
            print(f"\nERROR saving final output to {output_path}: {e}")
            traceback.print_exc()


    # --- 6. Clean Up Temporary Files ---
    print("\n--- Cleaning Up Temporary Files ---")
    # Combine files downloaded directly and parquet files generated
    cleanup_files_list = list(set(downloaded_files + generated_parquet_files)) # Use set to avoid duplicates
    deleted_count = 0
    for f in cleanup_files_list:
        if f and os.path.exists(f): # Check if f is not None or empty string
            try:
                os.remove(f)
                # print(f"  Deleted: {os.path.basename(f)}") # Verbose
                deleted_count += 1
            except Exception as e:
                print(f"  Warning: Error deleting temp file {f}: {e}")
        # else: # Verbose
        #     print(f"  Skipping cleanup for non-existent or invalid path: {f}")

    print(f"Attempted to delete {deleted_count} tracked temporary files.")

    # Remove the main temporary download directory
    if download_temp_dir and os.path.exists(download_temp_dir):
        try:
            shutil.rmtree(download_temp_dir)
            print(f"Removed temporary download directory: {download_temp_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary download directory {download_temp_dir}: {e}")

    end_time = datetime.now()
    print(f"\n======= Script Finished in {end_time - start_time} =======")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    # Configuration is loaded at the top now.
    # Check if critical variables were loaded successfully before calling main.
    critical_vars = [
        da_files, pn_files, sites, pdo_url, oni_url, beuti_url, streamflow_url,
        start_date, end_date, final_output_path, SATELLITE_OUTPUT_PARQUET
        ]

    # Check if sites dict is not empty, as it's crucial
    if not sites:
        print("CRITICAL ERROR: 'sites' dictionary is empty or missing in config.json. Cannot proceed.")
    # Check if essential URLs are present (can add more checks)
    elif not all([pdo_url, oni_url, beuti_url, streamflow_url]):
         print("CRITICAL ERROR: One or more essential URLs (PDO, ONI, BEUTI, Streamflow) are missing in config.json.")
    else:
        # Pass all required config variables loaded at the top to main
        main(
            run_satellite=include_satellite, # Pass the flag
            sat_meta=satellite_metadata,
            site_dict=sites,
            da_f=da_files,
            pn_f=pn_files,
            start_dt=start_date,
            end_dt=end_date,
            stream_url=streamflow_url,
            pdo_url=pdo_url,
            oni_url=oni_url,
            beuti_url=beuti_url,
            yr_cutoff=year_cutoff,
            wk_cutoff=week_cutoff,
            output_path=final_output_path,
            sat_intermediate_path=SATELLITE_OUTPUT_PARQUET
        )