import pandas as pd
import numpy as np
import json
import os
import requests
import tempfile
import xarray as xr
from datetime import datetime
from tqdm import tqdm
import warnings
import shutil
import config

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting non-nanosecond precision datetime values to nanosecond precision") # Add this for xarray date handling

# --- Configuration Loading ---
FORCE_SATELLITE_REPROCESSING = False # New flag: Set to True to always regenerate

# Lists to track temporary files for cleanup
downloaded_files = []
generated_parquet_files = []
temporary_nc_files_for_stitching = [] # Track yearly files per dataset


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
SATELLITE_OUTPUT_PARQUET = 'satellite_data_intermediate.parquet'

print(f"Configuration loaded: {len(da_files)} DA files, {len(pn_files)} PN files, {len(sites)} sites")
print(f"Date range: {start_date.date()} to {end_date.date()}, Output: {final_output_path}")

# Load satellite configuration from main config
satellite_metadata = config.SATELLITE_DATA
print(f"\n--- Satellite Configuration loaded from main config ---")
print(f"Satellite configuration loaded with {len(satellite_metadata)} data types.")

# --- Helper Functions ---
def download_file(url, filename):
    response = requests.get(url, timeout=5000, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    downloaded_files.append(filename)
    return filename

def local_filename(url, ext, temp_dir=None):
    """Generate appropriate local filename for download"""
    base = url.split('?')[0].split('/')[-1] or url.split('?')[0].split('/')[-2]
    sanitized_base = "".join(c for c in base if c.isalnum() or c in ('-', '_', '.'))
    root, existing_ext = os.path.splitext(sanitized_base or f"downloaded_file")
    base_name = root + (ext if not existing_ext or existing_ext == '.' else existing_ext)
    return os.path.join(temp_dir, base_name) if temp_dir else base_name

def csv_to_parquet(csv_path):
    parquet_path = csv_path[:-4] + '.parquet'
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    generated_parquet_files.append(parquet_path)
    return parquet_path

def convert_files_to_parquet(files_dict):
    """Convert multiple CSV files to Parquet format"""
    new_files = {}
    for name, path in files_dict.items():
        new_files[name] = csv_to_parquet(path)
    return new_files

def process_stitched_dataset(yearly_nc_files, data_type, site):
    """
    Process a stitched satellite NetCDF dataset from multiple yearly files.
    (Core logic for stitching)
    """
    ds = xr.open_mfdataset(yearly_nc_files, combine='nested', concat_dim='time', engine='netcdf4', decode_times=True, parallel=False)
    ds = ds.sortby('time')

    # Determine data variable name
    data_var = None
    dtype_lower = data_type.lower() if data_type else ""
    var_mapping = {
        'chla': ['chla', 'chlorophyll'],
        'sst': ['sst', 'temperature'],
        'par': ['par'],
        'fluorescence': ['fluorescence', 'flr'],
        'diffuse attenuation': ['diffuse attenuation', 'kd', 'k490'],
        'chla_anomaly': ['chla_anomaly', 'chlorophyll-anom'],
        'sst_anomaly': ['sst_anomaly', 'temperature-anom'],
    }
    possible_data_vars = list(ds.data_vars)
    found_match = False
    for var_key, keywords in var_mapping.items():
        if any(kw in dtype_lower for kw in keywords):
            if var_key in possible_data_vars:
                data_var = var_key
                found_match = True
                break
            for kw in keywords:
                if kw in possible_data_vars:
                    data_var = kw
                    found_match = True
                    break
        if found_match:
            break
    if not found_match and len(possible_data_vars) == 1:
        data_var = possible_data_vars[0]
        found_match = True

    data_array = ds[data_var]

    # Find time coordinate
    time_coord_name = None
    time_coords_to_check = ['time', 't', 'datetime']
    time_coord_name = next((c for c in time_coords_to_check if c in data_array.coords), None)

    # Average over spatial dimensions
    averaged_array = data_array
    spatial_dims = [dim for dim in data_array.dims if dim != time_coord_name]
    if spatial_dims:
        averaged_array = data_array.mean(dim=spatial_dims, skipna=True)

    # Convert to DataFrame and format
    df_final = None
    try:
        df = averaged_array.to_dataframe(name='value').reset_index()
        df = df.rename(columns={time_coord_name: 'timestamp'})
        df = df.dropna(subset=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['site'] = site
        df['data_type'] = data_type
        df = df.rename(columns={'value': data_var})
        final_cols = ['timestamp', 'site', 'data_type', data_var]
        df_final = df[[col for col in final_cols if col in df.columns]]
    except Exception as df_err:
        print(f"      ERROR: Failed during DataFrame conversion/formatting for {site} - {data_type}: {df_err}")
        df_final = None
    finally:
        if ds:
            ds.close()

    return df_final

def generate_satellite_parquet(satellite_metadata_dict, main_sites_list, output_path):
    """
    Downloads monthly satellite data, processes via stitching, combines, pivots,
    and saves to Parquet. Includes an inner progress bar for monthly downloads.
    """

    # --- Date Range Setup ---
    global_start_str = satellite_metadata_dict.get("satellite_start_date")
    global_anom_start_str = satellite_metadata_dict.get("satellite_anom_start_date")
    main_end_date_str = config.END_DATE

    main_end_dt = pd.to_datetime(main_end_date_str)
    # Ensure the global_end_dt represents the very end of the specified day
    global_end_str = main_end_dt.strftime('%Y-%m-%dT23:59:59Z')
    global_start_dt = pd.to_datetime(global_start_str) if global_start_str else None
    global_anom_start_dt = pd.to_datetime(global_anom_start_str) if global_anom_start_str else None
    global_end_dt = pd.to_datetime(global_end_str) if global_end_str else None

    # --- Temporary Directory ---
    sat_temp_dir = tempfile.mkdtemp(prefix="sat_monthly_dl_") # Changed prefix for clarity
    satellite_results_list = []
    path_to_return = None

    # --- Build Task List ---
    tasks = []
    processed_site_datatype_pairs = set()
    for data_type, sat_sites_dict in satellite_metadata_dict.items():
        if data_type in ["satellite_end_date", "satellite_start_date", "satellite_anom_start_date"] or not isinstance(sat_sites_dict, dict):
            continue
        for site, url_template in sat_sites_dict.items():
            normalized_site_name = site.lower().replace("_", " ").replace("-", " ")
            relevant_main_site = next((s for s in main_sites_list if s.lower().replace("_", " ").replace("-", " ") == normalized_site_name), None)
            if relevant_main_site and isinstance(url_template, str) and url_template.strip():
                if (relevant_main_site, data_type) not in processed_site_datatype_pairs:
                     tasks.append({
                         "site": relevant_main_site,
                         "data_type": data_type,
                         "url_template": url_template
                     })
                     processed_site_datatype_pairs.add((relevant_main_site, data_type))

    print(f"Prepared {len(tasks)} satellite processing tasks.")

    # --- Main Loop over Tasks (Outer progress bar) ---
    try:
        for task in tqdm(tasks, desc="Satellite Tasks", unit="task", position=0, leave=True):
            site = task["site"]
            data_type = task["data_type"]
            url_template = task["url_template"]

            # Determine overall start and end for this specific task
            is_anomaly_type = 'anom' in data_type.lower()
            current_overall_start_dt = global_anom_start_dt if is_anomaly_type and global_anom_start_dt else global_start_dt
            current_overall_end_dt = global_end_dt

            if not current_overall_start_dt or not current_overall_end_dt or current_overall_start_dt > current_overall_end_dt:
                print(f"\n          Skipping {site}-{data_type} due to invalid overall date range.")
                continue

            # --- Monthly Download Loop (With Inner progress bar) ---
            # Generate monthly periods based on the task's specific overall start and end dates
            # Ensure loop_start_date is the first of its month and loop_end_date is the end of its month
            # to correctly generate all intervening months with pd.date_range.
            loop_start_for_range = current_overall_start_dt.normalize().replace(day=1)
            loop_end_for_range = (current_overall_end_dt + pd.offsets.MonthEnd(0)).normalize()

            monthly_periods = pd.date_range(start=loop_start_for_range, end=loop_end_for_range, freq='MS') # MS for Month Start

            monthly_files_for_dataset = [] # Changed from yearly_files_for_dataset

            # Inner progress bar: Use position=1, leave=False
            for month_iterator_start_dt in tqdm(monthly_periods, desc=f"Download {site}-{data_type}", unit="month", position=1, leave=False):
                # Define the start and end of the current month for iteration
                current_month_loop_start_dt = month_iterator_start_dt
                current_month_loop_end_dt = (month_iterator_start_dt + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59, microsecond=999999) # End of the month

                # Clamp these loop dates with the actual overall start/end dates for the data request.
                # This handles partial first and last months correctly.
                effective_chunk_start_dt = max(current_overall_start_dt, current_month_loop_start_dt)
                effective_chunk_end_dt = min(current_overall_end_dt, current_month_loop_end_dt)

                if effective_chunk_start_dt > effective_chunk_end_dt:
                    # This can happen if current_overall_start_dt is after current_month_loop_end_dt
                    # or current_overall_end_dt is before current_month_loop_start_dt.
                    # Essentially, the generated month is outside the effective range.
                    continue

                month_start_str_url = effective_chunk_start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                month_end_str_url = effective_chunk_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

                # Replace placeholders in URL
                # Using a more generic "chunk_url" as it's now monthly
                chunk_url = url_template.replace("{start_date}", month_start_str_url)\
                                         .replace("{end_date}", month_end_str_url)
                if "{anom_start_date}" in chunk_url:
                    # Assuming anom_start_date in the template should also use the chunk's start date
                    chunk_url = chunk_url.replace("{anom_start_date}", month_start_str_url)

                # Use year and month in the temporary file name for clarity
                year_month_str = month_iterator_start_dt.strftime('%Y-%m')
                fd, tmp_nc_path = tempfile.mkstemp(suffix=f'_{year_month_str}.nc', prefix=f"{site}_{data_type}_", dir=sat_temp_dir)
                os.close(fd)

                # Download Logic
                try:
                    response = requests.get(chunk_url, timeout = 5000, stream=True)
                    response.raise_for_status()
                    with open(tmp_nc_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if os.path.getsize(tmp_nc_path) > 100: # Basic content check
                        monthly_files_for_dataset.append(tmp_nc_path)
                    else:
                        print(f"\n          Warning: Downloaded file for month {year_month_str} ({site}-{data_type}) seems empty. Skipping.")
                        if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path) # Ensure removal if empty
                except requests.exceptions.RequestException as req_err:
                    print(f"\n          ERROR downloading month {year_month_str} ({site}-{data_type}): {req_err}. Skipping month.")
                    if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path)
                except Exception as e:
                    print(f"\n          ERROR processing download for month {year_month_str} ({site}-{data_type}): {e}. Skipping.")
                    if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path)
            # --- End of Monthly Download Loop (Inner progress bar finishes here) ---

            # Process the collected monthly files
            if monthly_files_for_dataset:
                result_df = process_stitched_dataset(monthly_files_for_dataset, data_type, site)
                if result_df is not None and not result_df.empty:
                    satellite_results_list.append(result_df)

            # Clean up monthly files immediately after processing attempt
            for f_path in monthly_files_for_dataset:
                if os.path.exists(f_path):
                    os.unlink(f_path)
        # --- End Main Task Loop (Outer progress bar finishes here) ---

        # --- Combine and Pivot Results (Same as before) ---
        # ... (rest of the function remains the same from this point)
        processed_satellite_pivot = None
        if not satellite_results_list:
            processed_satellite_pivot = pd.DataFrame(columns=['site', 'timestamp'])
            processed_satellite_pivot['timestamp'] = pd.to_datetime([])
        else:
            combined_satellite_df = pd.concat(satellite_results_list, ignore_index=True)
            index_cols = ['site', 'timestamp']
            columns_col = 'data_type'
            all_cols = combined_satellite_df.columns.tolist()
            value_cols = [c for c in all_cols if c not in index_cols and c != columns_col]

            try:
                processed_satellite_pivot = combined_satellite_df.pivot_table(
                    index=index_cols, columns=columns_col, values=value_cols, aggfunc='mean'
                )
                # Flatten MultiIndex columns
                if isinstance(processed_satellite_pivot.columns, pd.MultiIndex):
                        processed_satellite_pivot.columns = [
                        f"sat_{level1.replace('-', '_')}_{level0.replace('-', '_')}" if len(value_cols) > 1 else f"sat_{level1.replace('-', '_')}"
                        for level0, level1 in processed_satellite_pivot.columns.values
                        ]
                else:
                        processed_satellite_pivot.columns = [f"sat_{col.replace('-', '_')}" for col in processed_satellite_pivot.columns]

                processed_satellite_pivot = processed_satellite_pivot.reset_index()

                if 'timestamp' in processed_satellite_pivot.columns:
                    processed_satellite_pivot['timestamp'] = pd.to_datetime(processed_satellite_pivot['timestamp'])
                else:
                    print("WARNING: 'timestamp' column missing after pivot. Adding empty NaT column.")
                    processed_satellite_pivot['timestamp'] = pd.NaT
            except Exception as pivot_err:
                    print(f"ERROR during satellite pivot: {pivot_err}") # Added error log
                    processed_satellite_pivot = pd.DataFrame(columns=['site', 'timestamp'])
                    processed_satellite_pivot['timestamp'] = pd.to_datetime([])

        # --- Save to Parquet ---
        if processed_satellite_pivot is None: # Should be initialized above, but as a safeguard
             processed_satellite_pivot = pd.DataFrame(columns=['site', 'timestamp'])
             processed_satellite_pivot['timestamp'] = pd.to_datetime([])

        processed_satellite_pivot.to_parquet(output_path, index=False)
        print(f"Satellite Parquet file write operation completed for path: {output_path}")
        path_to_return = output_path

    except Exception as main_err:
         print(f"\nFATAL ERROR during satellite data generation: {main_err}")
         path_to_return = None

    finally:
        # --- Final Cleanup ---
        print(f"\nCleaning up main temporary directory: {sat_temp_dir}")
        if os.path.exists(sat_temp_dir):
             try:
                 shutil.rmtree(sat_temp_dir)
             except OSError as e:
                 print(f"  Warning: Could not remove temp directory {sat_temp_dir}: {e}")

    if path_to_return:
        print(f"generate_satellite_parquet is returning path: {path_to_return}")
    else:
        print(f"generate_satellite_parquet is returning None (error or no file generated).")
    return path_to_return

def find_best_satellite_match(target_row, sat_pivot_indexed):
    target_site = target_row.get('Site')
    target_ts = target_row.get('timestamp_dt') # Weekly timestamp from target_df

    expected_cols = sat_pivot_indexed.columns if not sat_pivot_indexed.empty else pd.Index([])
    result_series = pd.Series(index=expected_cols, dtype=float)

    if pd.isna(target_ts):
        return result_series

    target_site_normalized = str(target_site).lower().replace('_', ' ').replace('-', ' ')

    unique_original_index_sites = sat_pivot_indexed.index.get_level_values('site').unique()
    original_index_site = None
    for s_val in unique_original_index_sites:
        if str(s_val).lower().replace('_', ' ').replace('-', ' ') == target_site_normalized:
            original_index_site = s_val
            break

    if original_index_site is None:
        return result_series

    try:
        site_data = sat_pivot_indexed.xs(original_index_site, level='site')
    except KeyError:
        return result_series

    if site_data.empty:
        return result_series

    if not isinstance(site_data.index, pd.DatetimeIndex):
        site_data.index = pd.to_datetime(site_data.index)
    site_data = site_data[pd.notna(site_data.index)]

    if site_data.empty:
        return result_series

    for var_name in expected_cols:
        if var_name not in site_data.columns:
            continue

        var_series_at_site = site_data[var_name]
        non_nan_var_series = var_series_at_site.dropna()

        if non_nan_var_series.empty:
            continue

        is_anomaly_var = "anom" in var_name.lower()

        if is_anomaly_var:
            # FIXED: Use data from at least 2 months before target to avoid leakage
            current_month_period = target_ts.to_period('M')
            # Use data from 2 months prior to ensure no temporal overlap
            safe_month_period = current_month_period - 2
            safe_month_start_time = safe_month_period.start_time
            safe_month_end_time = safe_month_period.end_time

            data_in_safe_month = non_nan_var_series[
                (non_nan_var_series.index >= safe_month_start_time) &
                (non_nan_var_series.index <= safe_month_end_time)
            ]

            if not data_in_safe_month.empty:
                chosen_ts = data_in_safe_month.index.max()
                result_series[var_name] = data_in_safe_month.loc[chosen_ts]
            else:
                # Fallback: Use data from at least 1 month before target
                cutoff_date = target_ts - pd.DateOffset(months=1)
                data_before_cutoff = non_nan_var_series[non_nan_var_series.index <= cutoff_date]
                if not data_before_cutoff.empty:
                    chosen_ts = data_before_cutoff.index.max()
                    result_series[var_name] = data_before_cutoff.loc[chosen_ts]
        else:
            # FIXED: For regular satellite data, use strict temporal cutoff (1 week minimum)
            cutoff_date = target_ts - pd.Timedelta(days=7)
            data_on_or_before = non_nan_var_series[non_nan_var_series.index <= cutoff_date]

            if not data_on_or_before.empty:
                chosen_ts = data_on_or_before.index.max()
                result_series[var_name] = data_on_or_before.loc[chosen_ts]
            else:
                # No fallback - if no data available with safe temporal distance, leave as NaN
                pass
    return result_series

def add_satellite_data(target_df, satellite_parquet_path):
    """Add satellite data to the target DataFrame"""        
    # Load satellite data
    satellite_df = pd.read_parquet(satellite_parquet_path)
        
    # Prepare data for matching
    target_df_proc = target_df.copy()
    target_df_proc['timestamp_dt'] = pd.to_datetime(target_df_proc['Date'])
    satellite_df['timestamp'] = pd.to_datetime(satellite_df['timestamp'])

    # Drop rows with missing keys essential for matching
    target_df_proc = target_df_proc.dropna(subset=['timestamp_dt', 'Site'])
    satellite_df = satellite_df.dropna(subset=['timestamp', 'site'])

    # Index satellite data for efficient lookup
    satellite_pivot_indexed = satellite_df.set_index(['site', 'timestamp']).sort_index()

    # Apply matching function
    print(f"Applying satellite matching function...")
    tqdm.pandas(desc="Satellite Matching")
    matched_data = target_df_proc.progress_apply(
        find_best_satellite_match, axis=1, sat_pivot_indexed=satellite_pivot_indexed
    )

    # Join results
    target_df_proc.reset_index(drop=True, inplace=True)
    matched_data.reset_index(drop=True, inplace=True)
    result_df = target_df_proc.join(matched_data)

    # Clean up by dropping the temporary timestamp column
    result_df = result_df.drop(columns=['timestamp_dt'], errors='ignore')
    return result_df

# --- Environmental Data Processing ---
def fetch_climate_index(url, var_name, temp_dir):
    """Process climate index data (PDO, ONI)"""
    print(f"Fetching climate index: {var_name}...")
        
    # Download file
    fname = local_filename(url, '.nc', temp_dir=temp_dir)
    download_file(url, fname)
            
    # Open and process dataset
    ds = xr.open_dataset(fname)
    df = ds.to_dataframe().reset_index()
    
    # Find time column
    time_cols = ['time', 'datetime', 'Date', 'T']
    time_col = next((c for c in time_cols if c in df.columns), None)
    
        
    # Find variable column (case-insensitive)
    actual_var_name = var_name
    if actual_var_name not in df.columns:
        var_name_lower = var_name.lower()
        found_var = next((c for c in df.columns if c.lower() == var_name_lower), None)
        actual_var_name = found_var or var_name
            
    # Process data
    df['datetime'] = pd.to_datetime(df[time_col])
    df = df[['datetime', actual_var_name]].dropna().rename(columns={actual_var_name: 'index'})
    
    # Aggregate monthly
    df['Month'] = df['datetime'].dt.to_period('M')
    result = df.groupby('Month')['index'].mean().reset_index()
    
    ds.close()
    return result[['Month', 'index']].sort_values('Month')

def process_streamflow(url, temp_dir):
    """Process USGS streamflow data (daily)"""
    print("Fetching streamflow data...")
    # Download file
    fname = local_filename(url, '.json', temp_dir=temp_dir)
    download_file(url, fname)
    with open(fname) as f:
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


def fetch_beuti_data(url, sites_dict, temp_dir, power=2):
    """Process BEUTI data with minimal error handling"""
    print("Fetching BEUTI data...")
        
    # Download file
    fname = local_filename(url, '.nc', temp_dir=temp_dir)
    download_file(url, fname)
            
    # Process data
    ds = xr.open_dataset(fname)
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
    
    for site, coords in sites_dict.items():
        # Get site lat
        site_lat = coords[0] if isinstance(coords, (list, tuple)) and coords else np.nan
        
        if pd.isna(site_lat):
            continue
            
        site_results = []
        # Group by date to interpolate for each day
        for date, group in df_sorted.groupby('Date'):
            lats = group['lat'].values
            beuti_vals = group['beuti'].values
            
            # Check for exact match first
            exact_match_indices = np.where(np.isclose(lats, site_lat))[0]
            if exact_match_indices.size > 0:
                interpolated_beuti = np.mean(beuti_vals[exact_match_indices])
            else:
                # Inverse distance weighting
                distances = np.abs(lats - site_lat)
                weights = 1.0 / (distances ** power + 1e-9)
                
                valid_indices = ~np.isnan(weights) & ~np.isnan(beuti_vals)
                if np.any(valid_indices):
                    valid_weights = weights[valid_indices]
                    valid_beuti = beuti_vals[valid_indices]
                    interpolated_beuti = np.sum(valid_beuti * valid_weights) / np.sum(valid_weights)
                else:
                    interpolated_beuti = np.nan
                    
            if pd.notna(interpolated_beuti):
                site_results.append({'Date': date, 'Site': site, 'beuti': interpolated_beuti})
                
        if site_results:
            results_list.extend(site_results)
            
        
    beuti_final_df = pd.DataFrame(results_list)
    beuti_final_df['Date'] = pd.to_datetime(beuti_final_df['Date'])
    
    ds.close()
    return beuti_final_df[['Date', 'Site', 'beuti']].sort_values(['Site', 'Date'])

# --- Core Data Processing ---
def process_da(da_files_dict):
    """Processes DA data from Parquet files, returns weekly aggregated DataFrame."""
    print("\n--- Processing DA Data ---")
    data_frames = []

    for name, path in da_files_dict.items():
        # Normalize name from dict key for site name guess
        site_name_guess = name.replace('-da', '').replace('_da', '').replace('-', ' ').replace('_', ' ').title()
        try:
            df = pd.read_parquet(path)
            # Identify Date and DA columns (handle variations)
            date_col, da_col = None, None
            if 'CollectDate' in df.columns: date_col = 'CollectDate'
            elif all(c in df.columns for c in ['Harvest Month', 'Harvest Date', 'Harvest Year']):
                df['CombinedDateStr'] = df['Harvest Month'].astype(str) + " " + df['Harvest Date'].astype(str) + ", " + df['Harvest Year'].astype(str)
                df['Date'] = pd.to_datetime(df['CombinedDateStr'], format='%B %d, %Y', errors='coerce')
                date_col = 'Date' # Now use the created 'Date' column

            if 'Domoic Result' in df.columns: da_col = 'Domoic Result'
            elif 'Domoic Acid' in df.columns: da_col = 'Domoic Acid'

            # Process valid columns
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['DA_Levels'] = pd.to_numeric(df[da_col], errors='coerce')
            df['Site'] = site_name_guess

            df.dropna(subset=['Parsed_Date', 'DA_Levels', 'Site'], inplace=True)

            # Aggregate weekly - Use ISO week for consistency
            df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
            # Group by week AND site (using the determined Site column)
            weekly_da = df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()

            data_frames.append(weekly_da[['Year-Week', 'DA_Levels', 'Site']])
            print(f"    Successfully processed {len(weekly_da)} weekly DA records for {name}.")

        except Exception as e:
            print(f"  Error processing DA file {name} ({os.path.basename(path)}): {e}")

    print("Combining all processed DA data...")
    final_da_df = pd.concat(data_frames, ignore_index=True)
    # Add a final group-by after concat to handle cases where different files might represent the same site-week
    if not final_da_df.empty:
         final_da_df = final_da_df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()
    print(f"Combined DA data shape: {final_da_df.shape}")
    return final_da_df

def process_pn(pn_files_dict):
    """Processes PN data from Parquet files, returns weekly aggregated DataFrame."""
    print("\n--- Processing PN Data ---")
    data_frames = []

    for name, path in pn_files_dict.items():
        # Normalize name from dict key for site name guess
        site_name_guess = name.replace('-pn', '').replace('_pn', '').replace('-', ' ').replace('_', ' ').title()

        df = pd.read_parquet(path)
        # Identify Date and PN columns (handle variations)
        date_col, pn_col = None, None
        # Try various common date column names
        date_col_candidates = ['Date']
        date_col = next((c for c in date_col_candidates if c in df.columns), None)
        pn_col_candidates = [c for c in df.columns if "pseudo" in str(c).lower() and "nitzschia" in str(c).lower()]
        if len(pn_col_candidates) == 1:
            pn_col = pn_col_candidates[0]

        # Process valid columns
        df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['PN_Levels'] = pd.to_numeric(df[pn_col], errors='coerce')
        df['Site'] = site_name_guess
        df.dropna(subset=['Parsed_Date', 'PN_Levels', 'Site'], inplace=True)

        # Aggregate weekly - Use ISO week for consistency
        df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
        # Group by week AND site
        weekly_pn = df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()

        data_frames.append(weekly_pn[['Year-Week', 'PN_Levels', 'Site']])
        print(f"  Successfully processed {len(weekly_pn)} weekly PN records for {name}.")

    final_pn_df = pd.concat(data_frames, ignore_index=True)
    # Add a final group-by after concat
    if not final_pn_df.empty:
        final_pn_df = final_pn_df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()
    print(f"Combined PN data shape: {final_pn_df.shape}")
    return final_pn_df

def generate_compiled_data(sites_dict, start_dt, end_dt):
    """Generate base DataFrame with all Site-Week combinations"""
    print(f"  Generating weekly entries from {start_dt.date()} to {end_dt.date()}")
    weeks = pd.date_range(start_dt, end_dt, freq='W-MON', name='Date')
    
    df_list = []
    for site, coords in sites_dict.items():
        lat, lon = (coords[0], coords[1]) if isinstance(coords, (list, tuple)) and len(coords) == 2 else (np.nan, np.nan)
        normalized_site = site.replace('_', ' ').replace('-', ' ').title()
        site_df = pd.DataFrame({'Date': weeks, 'Site': normalized_site, 'lat': lat, 'lon': lon})
        df_list.append(site_df)
        
    compiled_df = pd.concat(df_list, ignore_index=True)
    print(f"  Generated base DataFrame with {len(compiled_df)} site-week rows.")
    return compiled_df.sort_values(['Site', 'Date'])

def compile_data(compiled_df, oni_df, pdo_df, streamflow_df):
    """Merge climate indices and streamflow data into base DataFrame.
    ONI and PDO are merged based on the previous month's value with temporal buffer.
    Streamflow is merged using backward fill within a 7-day tolerance.
    """
    print("\n--- Merging Environmental Data ---")

    # Ensure Date is datetime type
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])

    # FIXED: Add temporal buffer for climate indices to account for reporting delays
    # Use data from 2 months prior to ensure it was available at prediction time
    compiled_df['TargetPrevMonth'] = compiled_df['Date'].dt.to_period(
        "M"
    ) - 2  # Use 2 months prior to account for reporting delays

    # Sort compiled_df initially (optional for merge, but good for consistency)
    compiled_df = compiled_df.sort_values(["Site", "Date"])

    # --- Merge ONI data ---
    # oni_df['Month'] is already a Period[M] object from fetch_climate_index.
    # We'll merge compiled_df['TargetPrevMonth'] with oni_df['Month'].
    # Prepare oni_df for merge: select columns, rename for clarity, and ensure uniqueness.
    oni_to_merge = oni_df[["Month", "index"]].rename(
        columns={"index": "oni", "Month": "ClimateIndexMonth"}
    )
    # fetch_climate_index should already provide unique months, but drop_duplicates is a safeguard.
    oni_to_merge = oni_to_merge.drop_duplicates(subset=["ClimateIndexMonth"])

    compiled_df = pd.merge(
        compiled_df,
        oni_to_merge,
        left_on="TargetPrevMonth",
        right_on="ClimateIndexMonth",
        how="left",
    )
    # Clean up columns: remove the merge key from the right table.
    if "ClimateIndexMonth" in compiled_df.columns:
        compiled_df.drop(columns=["ClimateIndexMonth"], inplace=True)

    # --- Merge PDO data ---
    # pdo_df['Month'] is already a Period[M] object.
    pdo_to_merge = pdo_df[["Month", "index"]].rename(
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

    # Drop the temporary 'TargetPrevMonth' column from compiled_df
    compiled_df.drop(columns=["TargetPrevMonth"], inplace=True)

    # --- Merge Streamflow data ---
    # This part remains the same: daily data, backward fill with tolerance.
    streamflow_df["Date"] = pd.to_datetime(streamflow_df["Date"])
    streamflow_df = streamflow_df.sort_values("Date")

    # Ensure compiled_df is sorted by 'Date' for merge_asof.
    # This might temporarily disrupt 'Site' order, so a final sort is crucial.
    compiled_df = compiled_df.sort_values("Date")

    compiled_df = pd.merge_asof(
        compiled_df,
        streamflow_df[["Date", "Flow"]],
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta("7days"),
    )
    compiled_df.rename(columns={"Flow": "discharge"}, inplace=True)

    # Final sort to ensure consistent output order
    return compiled_df.sort_values(["Site", "Date"])

def compile_da_pn(lt_df, da_df, pn_df):
    """Merge DA and PN data with interpolation"""
    print("\n--- Merging DA and PN Data ---")
    lt_df_merged = lt_df.copy()
    
    # Merge DA Data
    print(f"  Merging DA data ({len(da_df)} records)...")
    da_df_copy = da_df.copy()
    da_df_copy['Date'] = pd.to_datetime(da_df_copy['Year-Week'] + '-1', format='%G-%V-%w')
    da_df_copy = da_df_copy.dropna(subset=['Date', 'Site', 'DA_Levels'])
    lt_df_merged['Site'] = lt_df_merged['Site'].astype(str).str.replace('_', ' ').str.title()
    da_df_copy['Site'] = da_df_copy['Site'].astype(str).str.replace('_', ' ').str.title()
    lt_df_merged = pd.merge(lt_df_merged, da_df_copy[['Date', 'Site', 'DA_Levels']], 
                            on=['Date', 'Site'], how='left')
    lt_df_merged.rename(columns={'DA_Levels': 'DA_Levels_orig'}, inplace=True)
        
    # Merge PN Data
    print(f"  Merging PN data ({len(pn_df)} records)...")
    pn_df_copy = pn_df.copy()
    pn_df_copy['Date'] = pd.to_datetime(pn_df_copy['Year-Week'] + '-1', format='%G-%V-%w')
    pn_df_copy = pn_df_copy.dropna(subset=['Date', 'Site', 'PN_Levels'])
    pn_df_copy['Site'] = pn_df_copy['Site'].astype(str).str.replace('_', ' ').str.title()
    lt_df_merged = pd.merge(lt_df_merged, pn_df_copy[['Date', 'Site', 'PN_Levels']], 
                            on=['Date', 'Site'], how='left')
        
    # FIXED: Interpolate missing values using only forward direction to prevent future data leakage
    print("  Interpolating missing values (forward-only to prevent leakage)...")
    lt_df_merged = lt_df_merged.sort_values(by=['Site', 'Date'])
    
    # Interpolate DA - ONLY forward direction
    lt_df_merged['DA_Levels'] = lt_df_merged.groupby('Site')['DA_Levels_orig'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='forward')
    )
    lt_df_merged.drop(columns=['DA_Levels_orig'], inplace=True)
        
    # Interpolate PN - ONLY forward direction
    lt_df_merged['PN_Levels'] = lt_df_merged.groupby('Site')['PN_Levels'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='forward')
    )
    
    return lt_df_merged

def convert_and_fill(data_df):
    df_processed = data_df.copy()
    cols_to_process = df_processed.columns.difference(['Date', 'Site'])

    # Convert to numeric
    for col in cols_to_process:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Ensure Date is datetime
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        
    return df_processed

def main():
    """Main data processing pipeline"""
    print("\n======= Starting Data Processing Pipeline =======")
    start_time = datetime.now()

    # Create temp directory for downloads
    download_temp_dir = tempfile.mkdtemp(prefix="data_dl_")
    print(f"Using temporary directory: {download_temp_dir}")

    # Convert input CSVs to Parquet
    da_files_parquet = convert_files_to_parquet(da_files)
    pn_files_parquet = convert_files_to_parquet(pn_files)

    # --- Satellite Data Handling ---
    satellite_parquet_file_path = None  # Initialize path

    # Determine if we need to generate a new satellite file
    should_generate_new_satellite_file = False
    if FORCE_SATELLITE_REPROCESSING:
        print(
            f"\n--- FORCE_SATELLITE_REPROCESSING is True. Satellite data will be regenerated. ---"
        )
        should_generate_new_satellite_file = True
    elif not os.path.exists(SATELLITE_OUTPUT_PARQUET):
        print(
            f"\n--- Intermediate satellite data file '{SATELLITE_OUTPUT_PARQUET}' not found. Will attempt to generate. ---"
        )
        should_generate_new_satellite_file = True
    else:
        print(
            f"\n--- Found existing satellite data: {SATELLITE_OUTPUT_PARQUET}. Using this file. ---"
        )
        print(
            f"--- To regenerate, set FORCE_SATELLITE_REPROCESSING = True in the script. ---"
        )
        satellite_parquet_file_path = (
            SATELLITE_OUTPUT_PARQUET  # Use existing file
        )

    if should_generate_new_satellite_file:
        print(
            f"--- Generating satellite data. This may take a while... ---"
        )
        if FORCE_SATELLITE_REPROCESSING and os.path.exists(SATELLITE_OUTPUT_PARQUET):
            try:
                os.remove(SATELLITE_OUTPUT_PARQUET)
                print(
                    f"--- Removed old intermediate file due to force reprocessing: {SATELLITE_OUTPUT_PARQUET} ---"
                )
            except OSError as e:
                print(
                    f"--- Warning: Could not remove old intermediate file {SATELLITE_OUTPUT_PARQUET}: {e} ---"
                )

        generated_path = generate_satellite_parquet(
            satellite_metadata,
            list(sites.keys()),
            SATELLITE_OUTPUT_PARQUET,
        )
        if generated_path and os.path.exists(generated_path):
            print(
                f"--- Satellite data successfully generated and saved to: {generated_path} ---"
            )
            satellite_parquet_file_path = generated_path
        else:
            satellite_parquet_file_path = None

    # Process core data
    da_data = process_da(da_files_parquet)
    pn_data = process_pn(pn_files_parquet)

    # Process environmental data
    streamflow_data = process_streamflow(streamflow_url, download_temp_dir)
    pdo_data = fetch_climate_index(pdo_url, "pdo", download_temp_dir)
    oni_data = fetch_climate_index(oni_url, "oni", download_temp_dir)
    beuti_data = fetch_beuti_data(beuti_url, sites, download_temp_dir)

    # Generate and merge data
    compiled_base = generate_compiled_data(sites, start_date, end_date)
    lt_data = compile_data(compiled_base, oni_data, pdo_data, streamflow_data)
    lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)

    # Filter, process duplicates, and final processing
    base_final_data = convert_and_fill(lt_da_pn)

    # Merge BEUTI data
    beuti_data["Date"] = pd.to_datetime(beuti_data["Date"])
    beuti_data["Site"] = (
        beuti_data["Site"].astype(str).str.replace("_", " ").str.title()
    )
    base_final_data = pd.merge(
        base_final_data, beuti_data, on=["Date", "Site"], how="left"
    )
    base_final_data["beuti"] = base_final_data["beuti"].fillna(0)

    # Add satellite data if a valid path was determined and the file exists
    final_data = base_final_data
    if satellite_parquet_file_path and os.path.exists(
        satellite_parquet_file_path
    ):
        print(
            f"\n--- Adding satellite data from: {satellite_parquet_file_path} ---"
        )
        final_data = add_satellite_data(
            base_final_data, satellite_parquet_file_path
        )

    # Final processing and save
    print("\n--- Final Checks and Saving Output ---")

    # Sort columns
    final_core_cols = [
        "Date",
        "Site",
        "lat",
        "lon",
        "oni",
        "pdo",
        "discharge",
        "DA_Levels",
        "PN_Levels",
        "beuti",
    ]
    sat_cols = sorted(
        [col for col in final_data.columns if col.startswith("sat_")]
    )

    final_cols = [
        col for col in final_core_cols if col in final_data.columns
    ] + sat_cols
    final_data = final_data[final_cols]

    # Convert Date to string format
    final_data["Date"] = final_data["Date"].dt.strftime("%m/%d/%Y")

    # Rename columns if needed
    col_mapping = {
        "Date": "date",
        "Site": "site",
        "DA_Levels": "da",
        "PN_Levels": "pn",
    }

    if len(sat_cols) >= 7:
        sat_mapping = {}
        if len(sat_cols) > 0: sat_mapping[sat_cols[0]] = "chla-anom"
        if len(sat_cols) > 1: sat_mapping[sat_cols[1]] = "modis-chla"
        if len(sat_cols) > 2: sat_mapping[sat_cols[2]] = "modis-flr"
        if len(sat_cols) > 3: sat_mapping[sat_cols[3]] = "modis-k490"
        if len(sat_cols) > 4: sat_mapping[sat_cols[4]] = "modis-par"
        if len(sat_cols) > 5: sat_mapping[sat_cols[5]] = "modis-sst"
        if len(sat_cols) > 6: sat_mapping[sat_cols[6]] = "sst-anom"
        col_mapping.update(sat_mapping)

    final_data = final_data.rename(columns=col_mapping)

    # Save output
    print(f"Saving final data to {final_output_path}...")

    # Check if final_output_path has a directory component
    output_dir = os.path.dirname(final_output_path)
    if output_dir: # Only create directories if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)

    final_data.to_parquet(final_output_path, index=False)

    # Clean up
    print("\n--- Cleaning Up ---")
    for f in set(downloaded_files + generated_parquet_files):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                print(f"  Warning: Could not remove temporary file {f}: {e}")

    try:
        shutil.rmtree(download_temp_dir)
    except OSError as e:
        print(f"  Warning: Could not remove temp directory {download_temp_dir}: {e}")


    end_time = datetime.now()
    print(f"\n======= Script Finished in {end_time - start_time} =======")

# Run script
if __name__ == "__main__":
    main()
