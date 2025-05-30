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

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting non-nanosecond precision datetime values to nanosecond precision")

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'
SATELLITE_CONFIG_FILE = 'satellite_config.json'
FORCE_SATELLITE_REPROCESSING = False # New flag: Set to True to always regenerate

# Lists to track temporary files for cleanup
downloaded_files = []
generated_parquet_files = []
# temporary_nc_files_for_stitching = [] # This was per dataset, managed locally in generate_satellite_parquet

# Load main configuration
print(f"--- Loading Configuration from {CONFIG_FILE} ---")
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Extract config values
da_files = config.get('original_da_files', {})
pn_files = config.get('original_pn_files', {})
sites = config.get('sites', {})
pdo_url = config.get('pdo_url')
oni_url = config.get('oni_url')
beuti_url = config.get('beuti_url')
streamflow_url = config.get('streamflow_url')
start_date = pd.to_datetime(config.get('start_date', '2000-01-01'))
end_date = pd.to_datetime(config.get('end_date', datetime.now().strftime('%Y-%m-%d')))
final_output_path = config.get('final_output_path', 'config_final_output.parquet')
SATELLITE_OUTPUT_PARQUET = 'satellite_data_intermediate.parquet'

print(f"Configuration loaded: {len(da_files)} DA files, {len(pn_files)} PN files, {len(sites)} sites")
print(f"Date range: {start_date.date()} to {end_date.date()}, Output: {final_output_path}")

# Load satellite configuration
satellite_metadata = {}
print(f"\n--- Loading Satellite Configuration from {SATELLITE_CONFIG_FILE} ---")
with open(SATELLITE_CONFIG_FILE, 'r') as f:
    satellite_metadata = json.load(f)
print(f"Satellite configuration loaded with {len(satellite_metadata)-1} data types.")

# --- Helper Functions ---
def download_file(url, filename):
    response = requests.get(url, timeout=500, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    downloaded_files.append(filename)
    return filename

def local_filename(url, ext, temp_dir=None):
    base_url_part = url.split('?')[0]
    filename_part = base_url_part.split('/')[-1] or base_url_part.split('/')[-2]
    sanitized_filename = "".join(c for c in filename_part if c.isalnum() or c in ('-', '_', '.')) or "downloaded_file"
    root, existing_ext = os.path.splitext(sanitized_filename)
    final_filename = root + (ext if not existing_ext or existing_ext == '.' else existing_ext)
    return os.path.join(temp_dir, final_filename) if temp_dir else final_filename

def csv_to_parquet(csv_path):
    parquet_path = csv_path.replace('.csv', '.parquet') # More robust than [:-4]
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    generated_parquet_files.append(parquet_path)
    return parquet_path

def convert_files_to_parquet(files_dict):
    return {name: csv_to_parquet(path) for name, path in files_dict.items()}

def process_stitched_dataset(yearly_nc_files, data_type, site):
    ds = xr.open_mfdataset(yearly_nc_files, combine='nested', concat_dim='time', engine='netcdf4', decode_times=True, parallel=False)
    ds = ds.sortby('time')

    data_var = None
    dtype_lower = data_type.lower() if data_type else ""
    var_mapping = {
        'chla': ['chla', 'chlorophyll'], 'sst': ['sst', 'temperature'], 'par': ['par'],
        'fluorescence': ['fluorescence', 'flr'], 'diffuse attenuation': ['diffuse attenuation', 'kd', 'k490'],
        'chla_anomaly': ['chla_anomaly', 'chlorophyll-anom'], 'sst_anomaly': ['sst_anomaly', 'temperature-anom'],
    }
    possible_data_vars = list(ds.data_vars)
    found_match = False
    for var_key, keywords in var_mapping.items():
        if any(kw in dtype_lower for kw in keywords):
            if var_key in possible_data_vars: data_var = var_key; found_match = True; break
            for kw in keywords:
                if kw in possible_data_vars: data_var = kw; found_match = True; break
        if found_match: break
    if not found_match and len(possible_data_vars) == 1: data_var = possible_data_vars[0]
    # If no data_var found, subsequent ds[data_var] will fail. This matches original behavior.

    data_array = ds[data_var]
    time_coord_name = next((c for c in ['time', 't', 'datetime'] if c in data_array.coords), None)
    
    averaged_array = data_array
    spatial_dims = [dim for dim in data_array.dims if dim != time_coord_name]
    if spatial_dims:
        averaged_array = data_array.mean(dim=spatial_dims, skipna=True)

    df_final = None
    try:
        df = (averaged_array.to_dataframe(name='value')
              .reset_index()
              .rename(columns={time_coord_name: 'timestamp', 'value': data_var}))
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', data_var])
        if not df.empty:
            df['site'] = site
            df['data_type'] = data_type
            final_cols = ['timestamp', 'site', 'data_type', data_var]
            df_final = df[[col for col in final_cols if col in df.columns]]
    except Exception as df_err:
        print(f"      ERROR: Failed during DataFrame conversion/formatting for {site} - {data_type}: {df_err}")
    finally:
        if ds: ds.close()
    return df_final

def generate_satellite_parquet(satellite_metadata_dict, main_sites_list, output_path):
    global_start_str = satellite_metadata_dict.get("start_date")
    global_anom_start_str = satellite_metadata_dict.get("anom_start_date")
    with open(CONFIG_FILE, 'r') as f: main_config = json.load(f) # Inlined
    main_end_dt = pd.to_datetime(main_config.get('end_date', datetime.now().strftime('%Y-%m-%d')))
    global_end_str = main_end_dt.strftime('%Y-%m-%dT23:59:59Z')
    global_start_dt = pd.to_datetime(global_start_str) if global_start_str else None
    global_anom_start_dt = pd.to_datetime(global_anom_start_str) if global_anom_start_str else None
    global_end_dt = pd.to_datetime(global_end_str) if global_end_str else None

    sat_temp_dir = tempfile.mkdtemp(prefix="sat_monthly_dl_")
    satellite_results_list = []
    path_to_return = None

    tasks = []
    processed_site_datatype_pairs = set()
    # Pre-normalize main sites for slightly faster lookup if many sites
    # normalized_main_sites = {s.lower().replace("_", " ").replace("-", " "): s for s in main_sites_list}
    for data_type, sat_sites_dict in satellite_metadata_dict.items():
        if data_type in {"end_date", "start_date", "anom_start_date"} or not isinstance(sat_sites_dict, dict):
            continue
        for site_key, url_template in sat_sites_dict.items():
            if not (isinstance(url_template, str) and url_template.strip()): continue
            
            normalized_site_key = site_key.lower().replace("_", " ").replace("-", " ")
            # relevant_main_site = normalized_main_sites.get(normalized_site_key) # Use if pre-normalized
            relevant_main_site = next((s_main for s_main in main_sites_list if s_main.lower().replace("_", " ").replace("-", " ") == normalized_site_key), None)

            if relevant_main_site and (relevant_main_site, data_type) not in processed_site_datatype_pairs:
                 tasks.append({"site": relevant_main_site, "data_type": data_type, "url_template": url_template})
                 processed_site_datatype_pairs.add((relevant_main_site, data_type))
    print(f"Prepared {len(tasks)} satellite processing tasks.")

    try:
        for task in tqdm(tasks, desc="Satellite Tasks", unit="task", position=0, leave=True):
            site, data_type, url_template = task["site"], task["data_type"], task["url_template"]
            is_anomaly_type = 'anom' in data_type.lower()
            current_overall_start_dt = global_anom_start_dt if is_anomaly_type and global_anom_start_dt else global_start_dt
            if not current_overall_start_dt or not global_end_dt or current_overall_start_dt > global_end_dt:
                print(f"\n          Skipping {site}-{data_type} due to invalid overall date range.")
                continue

            loop_start_for_range = current_overall_start_dt.normalize().replace(day=1)
            loop_end_for_range = (global_end_dt + pd.offsets.MonthEnd(0)).normalize()
            monthly_periods = pd.date_range(start=loop_start_for_range, end=loop_end_for_range, freq='MS')
            monthly_files_for_dataset = []

            for month_iterator_start_dt in tqdm(monthly_periods, desc=f"Download {site}-{data_type}", unit="month", position=1, leave=False):
                current_month_loop_end_dt = (month_iterator_start_dt + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59, microsecond=999999)
                effective_chunk_start_dt = max(current_overall_start_dt, month_iterator_start_dt)
                effective_chunk_end_dt = min(global_end_dt, current_month_loop_end_dt)

                if effective_chunk_start_dt > effective_chunk_end_dt: continue

                month_start_str_url = effective_chunk_start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                month_end_str_url = effective_chunk_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                chunk_url = url_template.replace("{start_date}", month_start_str_url).replace("{end_date}", month_end_str_url)
                if "{anom_start_date}" in chunk_url: chunk_url = chunk_url.replace("{anom_start_date}", month_start_str_url)

                year_month_str = month_iterator_start_dt.strftime('%Y-%m')
                fd, tmp_nc_path = tempfile.mkstemp(suffix=f'_{year_month_str}.nc', prefix=f"{site}_{data_type}_", dir=sat_temp_dir); os.close(fd)

                try:
                    response = requests.get(chunk_url, timeout=1200, stream=True)
                    response.raise_for_status()
                    with open(tmp_nc_path, 'wb') as f: shutil.copyfileobj(response.raw, f) # Potentially more efficient for binary
                    if os.path.getsize(tmp_nc_path) > 100: monthly_files_for_dataset.append(tmp_nc_path)
                    else:
                        print(f"\n          Warning: Downloaded file for month {year_month_str} ({site}-{data_type}) seems empty. Skipping.")
                        if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path)
                except requests.exceptions.RequestException as req_err:
                    print(f"\n          ERROR downloading month {year_month_str} ({site}-{data_type}): {req_err}. Skipping month.")
                    if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path)
                except Exception as e: # Catch broader errors during download/write
                    print(f"\n          ERROR processing download for month {year_month_str} ({site}-{data_type}): {e}. Skipping.")
                    if os.path.exists(tmp_nc_path): os.unlink(tmp_nc_path)
            
            if monthly_files_for_dataset:
                result_df = process_stitched_dataset(monthly_files_for_dataset, data_type, site)
                if result_df is not None and not result_df.empty: satellite_results_list.append(result_df)
            for f_path in monthly_files_for_dataset: # Cleanup monthly files
                if os.path.exists(f_path): os.unlink(f_path)
        
        processed_satellite_pivot = pd.DataFrame(columns=['site', 'timestamp'], data={'timestamp': pd.to_datetime([])}) # Init with dtype
        if satellite_results_list:
            combined_satellite_df = pd.concat(satellite_results_list, ignore_index=True)
            index_cols, columns_col = ['site', 'timestamp'], 'data_type'
            value_cols = [c for c in combined_satellite_df.columns if c not in index_cols + [columns_col]]
            try:
                processed_satellite_pivot = combined_satellite_df.pivot_table(index=index_cols, columns=columns_col, values=value_cols, aggfunc='mean')
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
                else: # Should not happen if pivot_table includes timestamp in index and then reset_index
                    print("WARNING: 'timestamp' column missing after pivot. Adding empty NaT column.")
                    processed_satellite_pivot['timestamp'] = pd.NaT
            except Exception as pivot_err:
                print(f"ERROR during satellite pivot: {pivot_err}")
                # Reset to empty DF with expected structure
                processed_satellite_pivot = pd.DataFrame(columns=['site', 'timestamp'], data={'timestamp': pd.to_datetime([])})
        
        processed_satellite_pivot.to_parquet(output_path, index=False)
        print(f"Satellite Parquet file write operation completed for path: {output_path}")
        path_to_return = output_path
    except Exception as main_err:
         print(f"\nFATAL ERROR during satellite data generation: {main_err}")
    finally:
        print(f"\nCleaning up main temporary directory: {sat_temp_dir}")
        if os.path.exists(sat_temp_dir):
             try: shutil.rmtree(sat_temp_dir)
             except OSError as e: print(f"  Warning: Could not remove temp directory {sat_temp_dir}: {e}")
    print(f"generate_satellite_parquet is returning path: {path_to_return if path_to_return else 'None'}")
    return path_to_return

def _get_closest_fallback(series, target_ts):
    # Helper for find_best_satellite_match
    time_deltas = pd.Series(np.abs(series.index - target_ts), index=series.index)
    return series.loc[time_deltas.idxmin()]

def find_best_satellite_match(target_row, sat_pivot_indexed):
    target_site, target_ts = target_row.get('Site'), target_row.get('timestamp_dt')
    expected_cols = sat_pivot_indexed.columns if not sat_pivot_indexed.empty else pd.Index([])
    result_series = pd.Series(index=expected_cols, dtype=float)
    if pd.isna(target_ts) or pd.isna(target_site): return result_series

    target_site_normalized = str(target_site).lower().replace('_', ' ').replace('-', ' ')
    # Efficiently find matching site name from index
    original_index_site = next((s_val for s_val in sat_pivot_indexed.index.get_level_values('site').unique() 
                                if str(s_val).lower().replace('_', ' ').replace('-', ' ') == target_site_normalized), None)
    if original_index_site is None: return result_series

    try: site_data = sat_pivot_indexed.xs(original_index_site, level='site')
    except KeyError: return result_series
    if site_data.empty: return result_series
    if not isinstance(site_data.index, pd.DatetimeIndex): site_data.index = pd.to_datetime(site_data.index)
    site_data = site_data[pd.notna(site_data.index)]
    if site_data.empty: return result_series

    for var_name in expected_cols:
        if var_name not in site_data.columns: continue
        non_nan_var_series = site_data[var_name].dropna()
        if non_nan_var_series.empty: continue

        is_anomaly_var = "anom" in var_name.lower()
        if is_anomaly_var:
            prev_month_period = target_ts.to_period('M') - 1
            data_in_prev_month = non_nan_var_series[
                (non_nan_var_series.index >= prev_month_period.start_time) &
                (non_nan_var_series.index < prev_month_period.end_time) # Corrected to < end_time
            ]
            if not data_in_prev_month.empty:
                result_series[var_name] = data_in_prev_month.loc[data_in_prev_month.index.max()]
            else: result_series[var_name] = _get_closest_fallback(non_nan_var_series, target_ts)
        else: # Non-Anomaly
            data_on_or_before = non_nan_var_series[non_nan_var_series.index <= target_ts]
            if not data_on_or_before.empty:
                result_series[var_name] = data_on_or_before.loc[data_on_or_before.index.max()]
            else: result_series[var_name] = _get_closest_fallback(non_nan_var_series, target_ts)
    return result_series

def add_satellite_data(target_df, satellite_parquet_path):        
    satellite_df = pd.read_parquet(satellite_parquet_path)
    target_df_proc = target_df.copy()
    target_df_proc['timestamp_dt'] = pd.to_datetime(target_df_proc['Date'])
    satellite_df['timestamp'] = pd.to_datetime(satellite_df['timestamp'])

    target_df_proc.dropna(subset=['timestamp_dt', 'Site'], inplace=True) # Inplace
    satellite_df.dropna(subset=['timestamp', 'site'], inplace=True) # Inplace

    if satellite_df.empty: # If no satellite data after cleaning, return original df
        return target_df_proc.drop(columns=['timestamp_dt'], errors='ignore')

    satellite_pivot_indexed = satellite_df.set_index(['site', 'timestamp']).sort_index()
    
    print(f"Applying satellite matching function...")
    tqdm.pandas(desc="Satellite Matching")
    matched_data = target_df_proc.progress_apply(find_best_satellite_match, axis=1, sat_pivot_indexed=satellite_pivot_indexed)
    
    # Ensure indices are aligned for join; progress_apply should preserve index.
    result_df = target_df_proc.join(matched_data)
    return result_df.drop(columns=['timestamp_dt'], errors='ignore')

# --- Environmental Data Processing ---
def fetch_climate_index(url, var_name, temp_dir):
    print(f"Fetching climate index: {var_name}...")
    fname = local_filename(url, '.nc', temp_dir=temp_dir)
    download_file(url, fname)
    
    with xr.open_dataset(fname) as ds: # Use context manager for ds
        df = ds.to_dataframe().reset_index()
    
    time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
    actual_var_name = var_name
    if actual_var_name not in df.columns: # Try to find it case-insensitively
        actual_var_name = next((c for c in df.columns if c.lower() == var_name.lower()), var_name)
            
    df['datetime'] = pd.to_datetime(df[time_col])
    processed_df = df[['datetime', actual_var_name]].dropna().rename(columns={actual_var_name: 'index'})
    processed_df['Month'] = processed_df['datetime'].dt.to_period('M')
    result = (processed_df.groupby('Month')['index'].mean()
              .reset_index()
              .sort_values('Month'))
    return result[['Month', 'index']]

def process_streamflow(url, temp_dir):
    print("Fetching streamflow data...")
    fname = local_filename(url, '.json', temp_dir=temp_dir)
    download_file(url, fname)
    with open(fname) as f: data = json.load(f)
        
    values = []
    ts_data = data.get('value', {}).get('timeSeries', [])
    if ts_data:
        # Prioritize official discharge code, fallback to first series if only one, else None.
        discharge_ts = next((ts for ts in ts_data if ts.get('variable', {}).get('variableCode', [{}])[0].get('value') == '00060'),
                            ts_data[0] if len(ts_data) == 1 else None)
        if discharge_ts: values = discharge_ts.get('values', [{}])[0].get('value', [])
        
    records = []
    for item in values:
        if isinstance(item, dict) and 'dateTime' in item and 'value' in item:
            flow = pd.to_numeric(item['value'], errors='coerce')
            # Ensure flow is valid before processing dateTime (minor optimization)
            if pd.notna(flow) and flow >= 0:
                dt = pd.to_datetime(item['dateTime'], errors='coerce', utc=True) # Coerce date
                if pd.notna(dt): # Check dt after coercion
                    records.append({'Date': dt.tz_localize(None), 'Flow': flow})

    if not records: return pd.DataFrame(columns=['Date', 'Flow'])
    return pd.DataFrame(records).sort_values('Date')


def fetch_beuti_data(url, sites_dict, temp_dir, power=2):
    print("Fetching BEUTI data...")
    fname = local_filename(url, '.nc', temp_dir=temp_dir)
    download_file(url, fname)
            
    with xr.open_dataset(fname) as ds: # Context manager
        df = ds.to_dataframe().reset_index()
    
    time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
    lat_col = next((c for c in ['latitude', 'lat'] if c in df.columns), None)
    beuti_var = next((c for c in ['BEUTI', 'beuti'] if c in df.columns or c in ds.data_vars), None)
    
    # Chained processing for df_sorted
    df_sorted = (df[[time_col, lat_col, beuti_var]].copy()
                 .rename(columns={time_col: 'Date', lat_col: 'lat', beuti_var: 'beuti'})
                 .assign(Date=lambda x: pd.to_datetime(x['Date']).dt.date)
                 .dropna(subset=['Date', 'lat', 'beuti'])
                 .sort_values(by=['Date', 'lat']))
        
    results_list = []
    for site, coords in sites_dict.items():
        site_lat = coords[0] if isinstance(coords, (list, tuple)) and len(coords)>0 else np.nan # Check len
        if pd.isna(site_lat): continue
            
        site_results = []
        for date_val, group in df_sorted.groupby('Date'): # Use different name for date variable
            lats, beuti_vals = group['lat'].values, group['beuti'].values
            exact_match_indices = np.where(np.isclose(lats, site_lat))[0]
            if exact_match_indices.size > 0:
                interpolated_beuti = np.mean(beuti_vals[exact_match_indices])
            else:
                distances = np.abs(lats - site_lat)
                weights = 1.0 / (distances ** power + 1e-9) # Add epsilon for stability
                valid_indices = ~np.isnan(weights) & ~np.isnan(beuti_vals)
                if np.any(valid_indices):
                    interpolated_beuti = np.sum(beuti_vals[valid_indices] * weights[valid_indices]) / np.sum(weights[valid_indices])
                else: interpolated_beuti = np.nan
            if pd.notna(interpolated_beuti):
                site_results.append({'Date': date_val, 'Site': site, 'beuti': interpolated_beuti})
        results_list.extend(site_results)
            
    if not results_list: return pd.DataFrame(columns=['Date', 'Site', 'beuti'])
    return (pd.DataFrame(results_list)
            .assign(Date=lambda x: pd.to_datetime(x['Date']))
            .sort_values(['Site', 'Date']))

# --- Refactored Core Analytical Data Processing ---
def _determine_site_name(name, suffixes):
    for sfx in suffixes: name = name.replace(sfx, '')
    return name.replace('-', ' ').replace('_', ' ').title()

def _process_single_analytical_file(df, name, data_type_label, col_config, site_name_suffixes):
    site_name_guess = _determine_site_name(name, site_name_suffixes)
    date_col, value_col = None, None

    # Date column identification
    if col_config['date_col_primary'] in df.columns:
        date_col = col_config['date_col_primary']
    elif col_config.get('date_col_fallback_fields') and \
         all(c in df.columns for c in col_config['date_col_fallback_fields']):
        df['CombinedDateStr'] = df[col_config['date_col_fallback_fields'][0]].astype(str) + " " + \
                                df[col_config['date_col_fallback_fields'][1]].astype(str) + ", " + \
                                df[col_config['date_col_fallback_fields'][2]].astype(str)
        df['DateFromCombined'] = pd.to_datetime(df['CombinedDateStr'], format=col_config['date_col_fallback_format'], errors='coerce')
        date_col = 'DateFromCombined'

    # Value column identification
    if col_config['value_col_primary'] in df.columns:
        value_col = col_config['value_col_primary']
    elif col_config.get('value_col_secondary') and col_config['value_col_secondary'] in df.columns:
        value_col = col_config['value_col_secondary']
    elif col_config.get('value_col_candidates_logic'): # Special for PN
        pn_candidates = [c for c in df.columns if "pseudo" in str(c).lower() and "nitzschia" in str(c).lower()]
        if len(pn_candidates) == 1: value_col = pn_candidates[0]
    
    if not date_col or not value_col:
        print(f"    Could not find date or value columns for {name} ({data_type_label}). Skipping.")
        return pd.DataFrame()

    processed_df = pd.DataFrame({
        'Parsed_Date': pd.to_datetime(df[date_col], errors='coerce'),
        'Levels': pd.to_numeric(df[value_col], errors='coerce'),
        'Site': site_name_guess
    }).dropna(subset=['Parsed_Date', 'Levels', 'Site'])

    if processed_df.empty: return pd.DataFrame()
    
    processed_df['Year-Week'] = processed_df['Parsed_Date'].dt.strftime('%G-%V')
    weekly_data = processed_df.groupby(['Year-Week', 'Site'])['Levels'].mean().reset_index()
    print(f"    Successfully processed {len(weekly_data)} weekly {data_type_label} records for {name}.")
    return weekly_data

def process_analytical_data(files_dict, data_type_label, col_config, site_name_suffixes):
    print(f"\n--- Processing {data_type_label} Data ---")
    all_weekly_data = [
        _process_single_analytical_file(pd.read_parquet(path), name, data_type_label, col_config, site_name_suffixes)
        for name, path in files_dict.items()
    ]
    valid_dataframes = [df for df in all_weekly_data if not df.empty]
    if not valid_dataframes:
        return pd.DataFrame(columns=['Year-Week', 'Site', f'{data_type_label}_Levels'])

    final_df = pd.concat(valid_dataframes, ignore_index=True)
    if not final_df.empty: # Ensure grouping only if DF is not empty
         final_df = final_df.groupby(['Year-Week', 'Site'])['Levels'].mean().reset_index()
    final_df.rename(columns={'Levels': f'{data_type_label}_Levels'}, inplace=True)
    print(f"Combined {data_type_label} data shape: {final_df.shape}")
    return final_df

da_config = {
    'date_col_primary': 'CollectDate', 
    'date_col_fallback_fields': ['Harvest Month', 'Harvest Date', 'Harvest Year'], 
    'date_col_fallback_format': '%B %d, %Y',
    'value_col_primary': 'Domoic Result',
    'value_col_secondary': 'Domoic Acid'}
pn_config = {
    'date_col_primary': 'Date', # Assumes 'Date' is primary for PN
    'value_col_candidates_logic': True} # Special flag for PN's value column logic

# --- Core Data Processing (cont.) ---
def generate_compiled_data(sites_dict, start_dt, end_dt):
    print(f"  Generating weekly entries from {start_dt.date()} to {end_dt.date()}")
    weeks = pd.date_range(start_dt, end_dt, freq='W-MON', name='Date')
    
    df_list = [pd.DataFrame({
        'Date': weeks, 
        'Site': site.replace('_', ' ').replace('-', ' ').title(), 
        'lat': coords[0] if isinstance(coords, (list, tuple)) and len(coords) == 2 else np.nan, 
        'lon': coords[1] if isinstance(coords, (list, tuple)) and len(coords) == 2 else np.nan
    }) for site, coords in sites_dict.items()]
        
    compiled_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    print(f"  Generated base DataFrame with {len(compiled_df)} site-week rows.")
    return compiled_df.sort_values(['Site', 'Date'])

def _merge_climate_data_internal(compiled_df, climate_df, col_name):
    # Helper for compile_data
    climate_to_merge = (climate_df[["Month", "index"]]
                        .rename(columns={"index": col_name, "Month": "ClimateIndexMonth"})
                        .drop_duplicates(subset=["ClimateIndexMonth"]))
    merged_df = pd.merge(compiled_df, climate_to_merge,
                         left_on="TargetPrevMonth", right_on="ClimateIndexMonth", how="left")
    return merged_df.drop(columns=["ClimateIndexMonth"], errors='ignore')

def compile_data(compiled_df, oni_df, pdo_df, streamflow_df):
    print("\n--- Merging Environmental Data ---")
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])
    compiled_df['TargetPrevMonth'] = compiled_df['Date'].dt.to_period("M") - 1
    compiled_df.sort_values(["Site", "Date"], inplace=True) # Sort before climate merges

    compiled_df = _merge_climate_data_internal(compiled_df, oni_df, "oni")
    compiled_df = _merge_climate_data_internal(compiled_df, pdo_df, "pdo")
    compiled_df.drop(columns=["TargetPrevMonth"], inplace=True)

    streamflow_df["Date"] = pd.to_datetime(streamflow_df["Date"])
    compiled_df = pd.merge_asof(compiled_df.sort_values("Date"), # Sort by Date for merge_asof
                                streamflow_df[["Date", "Flow"]].sort_values("Date"), 
                                on="Date", direction="backward", tolerance=pd.Timedelta("7days"))
    return compiled_df.rename(columns={"Flow": "discharge"}).sort_values(["Site", "Date"]) # Final sort

def compile_da_pn(lt_df, da_df, pn_df):
    print("\n--- Merging DA and PN Data ---")
    lt_df_merged = lt_df.copy()
    
    # Prepare site columns for consistent merging
    lt_df_merged['Site'] = lt_df_merged['Site'].astype(str).str.replace('_', ' ').str.title()
    
    for df_to_merge, col_name_suffix, label in [(da_df, "DA_Levels", "DA"), (pn_df, "PN_Levels", "PN")]:
        print(f"  Merging {label} data ({len(df_to_merge)} records)...")
        if df_to_merge.empty: # Handle empty input df
            lt_df_merged[col_name_suffix] = np.nan 
            if label == "DA": lt_df_merged.rename(columns={col_name_suffix: 'DA_Levels_orig'}, inplace=True)
            continue

        df_copy = df_to_merge.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Year-Week'] + '-1', format='%G-%V-%w')
        df_copy['Site'] = df_copy['Site'].astype(str).str.replace('_', ' ').str.title()
        
        # Select only necessary columns to avoid potential clashes during merge
        merge_cols = ['Date', 'Site', col_name_suffix]
        lt_df_merged = pd.merge(lt_df_merged, df_copy[merge_cols], on=['Date', 'Site'], how='left')
        if label == "DA": # Specific rename for DA before interpolation
             lt_df_merged.rename(columns={col_name_suffix: 'DA_Levels_orig'}, inplace=True)
        
    print("  Interpolating missing values...")
    lt_df_merged = lt_df_merged.sort_values(by=['Site', 'Date'])
    # Interpolate DA (handle if DA_Levels_orig doesn't exist due to empty da_df)
    if 'DA_Levels_orig' in lt_df_merged.columns:
        lt_df_merged['DA_Levels'] = lt_df_merged.groupby('Site')['DA_Levels_orig'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
        lt_df_merged.drop(columns=['DA_Levels_orig'], inplace=True)
    else: lt_df_merged['DA_Levels'] = np.nan

    # Interpolate PN (handle if PN_Levels doesn't exist)
    if 'PN_Levels' in lt_df_merged.columns:
        lt_df_merged['PN_Levels'] = lt_df_merged.groupby('Site')['PN_Levels'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
    else: lt_df_merged['PN_Levels'] = np.nan
    return lt_df_merged

def convert_and_fill(data_df):
    df_processed = data_df.copy()
    # Convert all columns except 'Date' and 'Site' to numeric, coercing errors
    for col in df_processed.columns.difference(['Date', 'Site']):
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    return df_processed

def main():
    print("\n======= Starting Data Processing Pipeline ======="); start_time = datetime.now()
    download_temp_dir = tempfile.mkdtemp(prefix="data_dl_")
    print(f"Using temporary directory: {download_temp_dir}")

    da_files_parquet = convert_files_to_parquet(da_files)
    pn_files_parquet = convert_files_to_parquet(pn_files)

    # --- Satellite Data Handling ---
    satellite_parquet_file_path = None
    gen_reason = ""
    if FORCE_SATELLITE_REPROCESSING: gen_reason = "FORCE_SATELLITE_REPROCESSING is True."
    elif not os.path.exists(SATELLITE_OUTPUT_PARQUET): gen_reason = f"Intermediate file '{SATELLITE_OUTPUT_PARQUET}' not found."
    
    if gen_reason:
        print(f"\n--- {gen_reason} Satellite data will be (re)generated. ---")
        if FORCE_SATELLITE_REPROCESSING and os.path.exists(SATELLITE_OUTPUT_PARQUET):
            try: os.remove(SATELLITE_OUTPUT_PARQUET); print(f"--- Removed old intermediate file: {SATELLITE_OUTPUT_PARQUET} ---")
            except OSError as e: print(f"--- Warning: Could not remove old file {SATELLITE_OUTPUT_PARQUET}: {e} ---")
        
        print(f"--- Generating satellite data. This may take a while... ---")
        generated_path = generate_satellite_parquet(satellite_metadata, list(sites.keys()), SATELLITE_OUTPUT_PARQUET)
        if generated_path and os.path.exists(generated_path):
            print(f"--- Satellite data successfully generated: {generated_path} ---")
            satellite_parquet_file_path = generated_path
        # else: satellite_parquet_file_path remains None if generation fails
    else:
        print(f"\n--- Found existing satellite data: {SATELLITE_OUTPUT_PARQUET}. Using this file. ---")
        print(f"--- To regenerate, set FORCE_SATELLITE_REPROCESSING = True or remove the file. ---")
        satellite_parquet_file_path = SATELLITE_OUTPUT_PARQUET
    
    # Process core and environmental data
    da_data = process_analytical_data(da_files_parquet, "DA", da_config, ['-da', '_da'])
    pn_data = process_analytical_data(pn_files_parquet, "PN", pn_config, ['-pn', '_pn'])
    streamflow_data = process_streamflow(streamflow_url, download_temp_dir)
    pdo_data = fetch_climate_index(pdo_url, "pdo", download_temp_dir)
    oni_data = fetch_climate_index(oni_url, "oni", download_temp_dir)
    beuti_data = fetch_beuti_data(beuti_url, sites, download_temp_dir)

    compiled_base = generate_compiled_data(sites, start_date, end_date)
    lt_data = compile_data(compiled_base, oni_data, pdo_data, streamflow_data)
    lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)
    base_final_data = convert_and_fill(lt_da_pn)

    # Merge BEUTI data
    if not beuti_data.empty:
        beuti_data["Date"] = pd.to_datetime(beuti_data["Date"])
        beuti_data["Site"] = beuti_data["Site"].astype(str).str.replace("_", " ").str.title()
        base_final_data = pd.merge(base_final_data, beuti_data, on=["Date", "Site"], how="left")
    else: base_final_data["beuti"] = np.nan # Ensure column exists if beuti_data is empty
    base_final_data["beuti"] = base_final_data["beuti"].fillna(0) # Original behavior

    final_data = base_final_data
    if satellite_parquet_file_path and os.path.exists(satellite_parquet_file_path):
        print(f"\n--- Adding satellite data from: {satellite_parquet_file_path} ---")
        final_data = add_satellite_data(base_final_data, satellite_parquet_file_path)

    print("\n--- Final Checks and Saving Output ---")
    final_core_cols = ["Date", "Site", "lat", "lon", "oni", "pdo", "discharge", "DA_Levels", "PN_Levels", "beuti"]
    sat_cols = sorted([col for col in final_data.columns if col.startswith("sat_")])
    final_cols_ordered = [col for col in final_core_cols if col in final_data.columns] + sat_cols
    final_data = final_data[final_cols_ordered]
    final_data["Date"] = pd.to_datetime(final_data["Date"]).dt.strftime("%m/%d/%Y") # Ensure Date is datetime before strftime

    col_mapping = {"Date": "date", "Site": "site", "DA_Levels": "da", "PN_Levels": "pn"}
    sat_target_names = ["chla-anom", "modis-chla", "modis-flr", "modis-k490", "modis-par", "modis-sst", "sst-anom"]
    col_mapping.update({sat_cols[i]: sat_target_names[i] for i in range(min(len(sat_cols), len(sat_target_names)))})
    final_data.rename(columns=col_mapping, inplace=True)

    print(f"Saving final data to {final_output_path}...")
    output_dir = os.path.dirname(final_output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    final_data.to_parquet(final_output_path, index=False)

    print("\n--- Cleaning Up ---")
    for f_path in set(downloaded_files + generated_parquet_files): # Use set to avoid duplicates
        if os.path.exists(f_path):
            try: os.remove(f_path)
            except OSError as e: print(f"  Warning: Could not remove temporary file {f_path}: {e}")
    try: shutil.rmtree(download_temp_dir)
    except OSError as e: print(f"  Warning: Could not remove temp directory {download_temp_dir}: {e}")

    print(f"\n======= Script Finished in {datetime.now() - start_time} =======")

if __name__ == "__main__":
    main()