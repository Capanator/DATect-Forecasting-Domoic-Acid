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

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'
SATELLITE_CONFIG_FILE = 'satellite_config.json'

# Lists to track temporary files for cleanup
downloaded_files = []
generated_parquet_files = []

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

# Load satellite configuration if needed
satellite_metadata = {}
print(f"\n--- Loading Satellite Configuration from {SATELLITE_CONFIG_FILE} ---")
with open(SATELLITE_CONFIG_FILE, 'r') as f:
    satellite_metadata = json.load(f)
print(f"Satellite configuration loaded with {len(satellite_metadata)-1} data types.")

# --- Helper Functions ---
def download_file(url, filename):
    """Download file from URL"""
    if not url:
        return None
    response = requests.get(url, timeout=180, stream=True)
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

# --- Satellite Data Processing ---
def process_dataset(url, data_type, site, temp_dir):
    """Process satellite NetCDF data"""
    # Determine data variable name
    data_var = None
    url_lower = url.lower() if url else ""
    dtype_lower = data_type.lower() if data_type else ""
    
    var_mapping = {
        'chla-anom': ['chla_anomaly', 'chlorophyll-anom'],
        'sst-anom': ['sst_anomaly', 'temperature-anom'],
        'modis-chla': ['chlorophyll'],
        'modis-sst': ['sst', 'temperature'],
        'modis-par': ['par'],
        'modis-flr': ['fluorescence']
    }
    
    for var_name, keywords in var_mapping.items():
        if any(kw in url_lower or kw in dtype_lower for kw in keywords):
            data_var = var_name
            break

    # Create temporary file path and download data
    fd, tmp_nc_path = tempfile.mkstemp(suffix='.nc', prefix=f"{site}_{data_type}_", dir=temp_dir)
    os.close(fd)
    
    response = requests.get(url, timeout=300, stream=True)
    with open(tmp_nc_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Process data
    ds = xr.open_dataset(tmp_nc_path)
    
    # Handle case when data variable isn't found
    if data_var not in ds.data_vars:
        available_vars = list(ds.data_vars)
        if len(available_vars) == 1:
            data_var = available_vars[0]
        else:
            os.unlink(tmp_nc_path)
            ds.close()
            return None

    # Select data variable and process
    data_array = ds[data_var]
    
    # Find time coordinate
    time_coords = ['time', 't', 'datetime']
    time_coord_name = next((c for c in time_coords if c in data_array.coords), None)
    if not time_coord_name:
        os.unlink(tmp_nc_path)
        ds.close()
        return None

    # Average over spatial dimensions
    spatial_dims = [dim for dim in data_array.dims if dim != time_coord_name]
    averaged_array = data_array.mean(dim=spatial_dims, skipna=True) if spatial_dims else data_array
    
    # Convert to DataFrame and format
    df = averaged_array.to_dataframe(name='value').reset_index()
    df = df.rename(columns={time_coord_name: 'timestamp'})
    df = df.dropna(subset=['timestamp', 'value'])
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
        
    df['site'] = site
    df['data_type'] = data_type
    df = df.rename(columns={'value': data_var})
    
    # Cleanup and return
    os.unlink(tmp_nc_path)
    ds.close()
    
    return df[['timestamp', 'site', 'data_type', data_var]]

def generate_satellite_parquet(satellite_metadata_dict, main_sites_list, output_path):
    """Process satellite data and save to Parquet"""
    print("\n--- Processing Satellite Data ---")
        
    sat_end_date_global = satellite_metadata_dict.get('end_date', None)
    
    # Create temp directory
    sat_temp_dir = tempfile.mkdtemp(prefix="sat_downloads_")
    print(f"Using temporary directory: {sat_temp_dir}")
    
    # Build task list
    satellite_tasks = []
    processed_sites = set()
    
    for data_type, sat_sites_dict in satellite_metadata_dict.items():
        if data_type == 'end_date' or not isinstance(sat_sites_dict, dict):
            continue
            
        for site, url_or_list in sat_sites_dict.items():
            # Match site names with main site list (case-insensitive)
            normalized_site_name = site.lower().replace('_', ' ').replace('-', ' ')
            relevant_main_site = next((s for s in main_sites_list 
                                     if s.lower().replace('_', ' ').replace('-', ' ') == normalized_site_name), None)
            
            if relevant_main_site:
                urls_to_process = [url_or_list] if isinstance(url_or_list, str) else url_or_list
                
                for url in urls_to_process:
                    if isinstance(url, str) and url.strip():
                        processed_url = url
                        if '{end_date}' in url and sat_end_date_global:
                            processed_url = url.replace('{end_date}', sat_end_date_global)
                            
                        satellite_tasks.append((data_type, relevant_main_site, processed_url))
                        processed_sites.add(relevant_main_site)
        
    # Process satellite datasets
    print(f"Processing {len(satellite_tasks)} satellite datasets for {len(processed_sites)} sites...")
    satellite_results_list = []
    
    for data_type, site, url in tqdm(satellite_tasks, desc="Satellite Data"):
        result_df = process_dataset(url, data_type, site, sat_temp_dir)
        if result_df is not None and not result_df.empty:
            satellite_results_list.append(result_df)
            
    # Clean up temp directory
    shutil.rmtree(sat_temp_dir)
        
    # Combine and pivot results
    combined_satellite_df = pd.concat(satellite_results_list, ignore_index=True)
    
    # Determine value columns
    value_cols = [col for col in combined_satellite_df.columns if col not in ['timestamp', 'site', 'data_type']]
    
    # Pivot table
    processed_satellite_pivot = combined_satellite_df.pivot_table(
        index=['site', 'timestamp'],
        columns='data_type',
        values=value_cols,
        aggfunc='mean'
    )
    
    # Flatten MultiIndex columns
    if isinstance(processed_satellite_pivot.columns, pd.MultiIndex):
        processed_satellite_pivot.columns = ['sat_' + '_'.join(col).strip() 
                                          for col in processed_satellite_pivot.columns.values]
    else:
        processed_satellite_pivot.columns = [f'sat_{col}' for col in processed_satellite_pivot.columns]
        
    processed_satellite_pivot = processed_satellite_pivot.reset_index()
    processed_satellite_pivot['timestamp'] = pd.to_datetime(processed_satellite_pivot['timestamp'])
    
    # Save to parquet
    processed_satellite_pivot.to_parquet(output_path, index=False)
    generated_parquet_files.append(output_path)
    
    return output_path

def find_best_satellite_match(target_row, sat_pivot_indexed):
    """Find best satellite data match for a given row"""
    target_site = target_row.get('Site')
    target_ts = target_row.get('timestamp_dt')
        
    # Normalize site names for matching
    target_site_normalized = target_site.lower().replace('_', ' ').replace('-', ' ')
    index_sites = sat_pivot_indexed.index.get_level_values('site')
    unique_original_index_sites = index_sites.unique()
    unique_normalized_index_sites = [str(s).lower().replace('_', ' ').replace('-', ' ') 
                                   for s in unique_original_index_sites]
        
    # Find original case site name
    original_index_site = None
    for i, norm_site in enumerate(unique_normalized_index_sites):
        if norm_site == target_site_normalized:
            original_index_site = unique_original_index_sites[i]
            break
        
    # Get data for the specific site
    site_data = sat_pivot_indexed.xs(original_index_site, level='site')
        
    # Ensure index is datetime
    if not isinstance(site_data.index, pd.DatetimeIndex):
        site_data.index = pd.to_datetime(site_data.index)
        site_data = site_data.dropna(axis=0)
        
    # Look for matches in same month and year
    target_year = target_ts.year
    target_month = target_ts.month
    month_matches = site_data[(site_data.index.year == target_year) & (site_data.index.month == target_month)]
    
    if not month_matches.empty:
        # Find closest timestamp in same month
        time_diff_in_month = np.abs(month_matches.index - target_ts)
        min_idx_pos = time_diff_in_month.argmin()
        return month_matches.iloc[min_idx_pos]
    else:
        # Find closest timestamp overall
        time_diff_overall = np.abs(site_data.index - target_ts)
        min_overall_pos = time_diff_overall.argmin()
        return site_data.iloc[min_overall_pos]

def add_satellite_data(target_df, satellite_parquet_path):
    """Add satellite data to the target DataFrame"""        
    # Load satellite data
    satellite_df = pd.read_parquet(satellite_parquet_path)
        
    # Prepare data for matching
    target_df_proc = target_df.copy()
    target_df_proc['timestamp_dt'] = pd.to_datetime(target_df_proc['Date'])
    satellite_df['timestamp'] = pd.to_datetime(satellite_df['timestamp'])
    
    # Drop rows with missing keys
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
    
    # Clean up and fill NaNs
    result_df = result_df.drop(columns=['timestamp_dt'], errors='ignore')
    
    sat_cols_added = [col for col in matched_data.columns if col in result_df.columns and col.startswith('sat_')]
    if sat_cols_added:
        result_df[sat_cols_added] = result_df[sat_cols_added].fillna(0)
        
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
    """Process USGS streamflow data"""
    print("Fetching streamflow data...")
        
    # Download file
    fname = local_filename(url, '.json', temp_dir=temp_dir)
    download_file(url, fname)
            
    # Load JSON
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

    # Aggregate weekly
    df['week_key'] = df['Date'].dt.strftime('%G-%V')
    weekly_flow = df.groupby('week_key')['Flow'].mean().reset_index()
    weekly_flow['Date'] = pd.to_datetime(weekly_flow['week_key'] + '-1', format='%G-%V-%w')
    weekly_flow = weekly_flow.dropna(subset=['Date'])

    return weekly_flow[['Date', 'Flow']].sort_values('Date')

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
    """Merge climate indices and streamflow data into base DataFrame"""
    print("\n--- Merging Environmental Data ---")
    
    # Prepare for Monthly Merge
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])
    compiled_df['Month'] = compiled_df['Date'].dt.to_period('M')
    compiled_df = compiled_df.sort_values('Month')
    
    # Merge ONI using nearest month
    oni_df = oni_df.sort_values('Month')
    oni_df['Month'] = pd.to_datetime(oni_df['Month'].astype(str)).dt.to_period('M')
    compiled_df = pd.merge_asof(
        compiled_df,
        oni_df[['Month', 'index']],
        on='Month',
        direction='nearest'
    )
    compiled_df.rename(columns={'index': 'oni'}, inplace=True)
        
    # Merge PDO using nearest month
    pdo_df = pdo_df.sort_values('Month')
    pdo_df['Month'] = pd.to_datetime(pdo_df['Month'].astype(str)).dt.to_period('M')
    compiled_df = pd.merge_asof(
        compiled_df,
        pdo_df[['Month', 'index']],
        on='Month',
        direction='nearest'
    )
    compiled_df.rename(columns={'index': 'pdo'}, inplace=True)
        
    # Drop temporary Month column
    compiled_df.drop(columns=['Month'], inplace=True)
    
    # Merge Streamflow data
    streamflow_df['Date'] = pd.to_datetime(streamflow_df['Date'])
    streamflow_df = streamflow_df.sort_values('Date')
    compiled_df = compiled_df.sort_values('Date')
    compiled_df = pd.merge_asof(
        compiled_df,
        streamflow_df[['Date', 'Flow']],
        on='Date',
        direction='backward',
        tolerance=pd.Timedelta('7days')
    )
    compiled_df.rename(columns={'Flow': 'discharge'}, inplace=True)
        
    return compiled_df.sort_values(['Site', 'Date'])

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
        
    # Interpolate missing values
    print("  Interpolating missing values...")
    lt_df_merged = lt_df_merged.sort_values(by=['Site', 'Date'])
    
    # Interpolate DA
    lt_df_merged['DA_Levels'] = lt_df_merged.groupby('Site')['DA_Levels_orig'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    lt_df_merged.drop(columns=['DA_Levels_orig'], inplace=True)
        
    # Interpolate PN
    lt_df_merged['PN_Levels'] = lt_df_merged.groupby('Site')['PN_Levels'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
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
    
    # Generate satellite data if needed
    satellite_parquet_file_path = generate_satellite_parquet(
        satellite_metadata,
        list(sites.keys()),
        SATELLITE_OUTPUT_PARQUET
    )
    
    # Process core data
    da_data = process_da(da_files_parquet)
    pn_data = process_pn(pn_files_parquet)
    
    # Process environmental data
    streamflow_data = process_streamflow(streamflow_url, download_temp_dir)
    pdo_data = fetch_climate_index(pdo_url, 'pdo', download_temp_dir)
    oni_data = fetch_climate_index(oni_url, 'oni', download_temp_dir)
    beuti_data = fetch_beuti_data(beuti_url, sites, download_temp_dir)
    
    # Generate and merge data
    compiled_base = generate_compiled_data(sites, start_date, end_date)
    lt_data = compile_data(compiled_base, oni_data, pdo_data, streamflow_data)
    lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)
    
    # Filter, process duplicates, and final processing
    base_final_data = convert_and_fill(lt_da_pn)
    
    # Merge BEUTI data
    print("\n--- Merging BEUTI Data ---")
    beuti_data['Date'] = pd.to_datetime(beuti_data['Date'])
    beuti_data['Site'] = beuti_data['Site'].astype(str).str.replace('_', ' ').str.title()
    base_final_data = pd.merge(base_final_data, beuti_data, on=['Date', 'Site'], how='left')
    base_final_data['beuti'] = base_final_data['beuti'].fillna(0)
                
    # Add satellite data if available
    final_data = add_satellite_data(base_final_data, satellite_parquet_file_path)
        
    # Final processing and save
    print("\n--- Final Checks and Saving Output ---")
    
    # Sort columns
    final_core_cols = ["Date", "Site", "lat", "lon", "oni", "pdo", 
                     "discharge", "DA_Levels", "PN_Levels", "beuti"]
    sat_cols = sorted([col for col in final_data.columns if col.startswith('sat_')])
    
    final_cols = [col for col in final_core_cols if col in final_data.columns] + sat_cols
    final_data = final_data[final_cols]
    
    # Convert Date to string format
    final_data['Date'] = final_data['Date'].dt.strftime('%m/%d/%Y')
    
    # Rename columns if needed
    col_mapping = {
        "Date": "date", 
        "Site": "site", 
        "DA_Levels": "da", 
        "PN_Levels": "pn", 
    }
    
    # Add satellite column mappings
    if len(sat_cols) >= 6:
        sat_mapping = {
            sat_cols[0]: "chla-anom",
            sat_cols[1]: "modis-chla",
            sat_cols[2]: "modis-flr", 
            sat_cols[3]: "modis-par",
            sat_cols[4]: "sst-anom", 
            sat_cols[5]: "modis-sst"
        }
        col_mapping.update(sat_mapping)
        
    final_data = final_data.rename(columns=col_mapping)
    
    # Save output
    print(f"Saving final data to {final_output_path}...")
    
    # Check if final_output_path has a directory component
    output_dir = os.path.dirname(final_output_path)
    if output_dir:  # Only create directories if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)
        
    final_data.to_parquet(final_output_path, index=False)
    
    # Clean up
    print("\n--- Cleaning Up ---")
    for f in set(downloaded_files + generated_parquet_files):
        if os.path.exists(f):
            os.remove(f)
            
    shutil.rmtree(download_temp_dir)
    
    end_time = datetime.now()
    print(f"\n======= Script Finished in {end_time - start_time} =======")

# Run script
if __name__ == "__main__":
    main()