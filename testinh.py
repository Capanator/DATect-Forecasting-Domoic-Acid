import pandas as pd, numpy as np, json, os, requests, tempfile, xarray as xr, shutil
from datetime import datetime
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message=("Could not infer format, so each element will be parsed individually, "
                                                               "falling back to `dateutil`|Converting non-nanosecond precision datetime values to nanosecond precision"))

# Configuration
CONFIG_FILE = 'config.json'
SATELLITE_CONFIG_FILE = 'satellite_config.json'
FORCE_SATELLITE_REPROCESSING = False

# Track temporary files
downloaded_files, generated_parquet_files, temporary_nc_files_for_stitching = [], [], []

# Load configurations
print(f"--- Loading Configurations ---")
with open(CONFIG_FILE) as f: config = json.load(f)
da_files = config.get('original_da_files', {})
pn_files = config.get('original_pn_files', {})
sites = config.get('sites', {})
start_date = pd.to_datetime(config.get('start_date', '2000-01-01'))
end_date = pd.to_datetime(config.get('end_date', datetime.now().strftime('%Y-%m-%d')))
final_output_path = config.get('final_output_path', 'config_final_output.parquet')
SATELLITE_OUTPUT_PARQUET = 'satellite_data_intermediate.parquet'

print(f"Config loaded: {len(da_files)} DA, {len(pn_files)} PN, {len(sites)} sites")
print(f"Date range: {start_date.date()} to {end_date.date()}, Output: {final_output_path}")

with open(SATELLITE_CONFIG_FILE) as f: satellite_metadata = json.load(f)
print(f"Satellite config loaded with {len(satellite_metadata)-1} data types")

# Helper functions
def download_file(url, filename):
    with requests.get(url, timeout=500, stream=True) as response:
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
    downloaded_files.append(filename)
    return filename

def local_filename(url, ext, temp_dir=None):
    base = url.split('?')[0].split('/')[-1] or url.split('?')[0].split('/')[-2]
    sanitized = "".join(c for c in base if c.isalnum() or c in ('-', '_', '.'))
    root, ext_exist = os.path.splitext(sanitized or "downloaded_file")
    return os.path.join(temp_dir, root + (ext if not ext_exist or ext_exist == '.' else ext_exist))

def csv_to_parquet(csv_path):
    parquet_path = csv_path[:-4] + '.parquet'
    pd.read_csv(csv_path, low_memory=False).to_parquet(parquet_path, index=False)
    generated_parquet_files.append(parquet_path)
    return parquet_path

def convert_files_to_parquet(files_dict):
    return {name: csv_to_parquet(path) for name, path in files_dict.items()}

def process_stitched_dataset(yearly_nc_files, data_type, site):
    ds = xr.open_mfdataset(yearly_nc_files, combine='nested', concat_dim='time', engine='netcdf4', decode_times=True)
    ds = ds.sortby('time')
    
    # Identify data variable
    dtype_lower = data_type.lower()
    var_mapping = {
        'chla': ['chla', 'chlorophyll'],
        'sst': ['sst', 'temperature'],
        'par': ['par'],
        'fluorescence': ['fluorescence', 'flr'],
        'diffuse attenuation': ['diffuse attenuation', 'kd', 'k490'],
        'chla_anomaly': ['chla_anomaly', 'chlorophyll-anom'],
        'sst_anomaly': ['sst_anomaly', 'temperature-anom'],
    }
    possible_vars = list(ds.data_vars)
    data_var = next((v for key, kws in var_mapping.items() 
                    if any(kw in dtype_lower for kw in kws) 
                    for v in ([key] if key in possible_vars else [])
                    or next((v for kw in kws if kw in possible_vars), None)), None) or possible_vars[0] if len(possible_vars)==1 else None

    time_coord = next((c for c in ['time', 't', 'datetime'] if c in ds[data_var].coords), None)
    spatial_dims = [dim for dim in ds[data_var].dims if dim != time_coord]
    averaged = ds[data_var].mean(dim=spatial_dims, skipna=True) if spatial_dims else ds[data_var]

    try:
        df = averaged.to_dataframe(name='value').reset_index().rename(columns={time_coord: 'timestamp'})
        df = df.dropna(subset=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['site'], df['data_type'] = site, data_type
        return df.rename(columns={'value': data_var})[['timestamp', 'site', 'data_type', data_var]]
    except Exception as e:
        print(f"      ERROR: DataFrame conversion failed for {site}-{data_type}: {e}")
        return None
    finally: ds.close()

def generate_satellite_parquet(satellite_metadata_dict, main_sites_list, output_path):
    # Load configuration once
    with open(CONFIG_FILE, 'r') as f:
        main_config = json.load(f)
    main_end_date_str = main_config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    main_end_dt = pd.to_datetime(main_end_date_str)
    global_end_dt = pd.to_datetime(main_end_dt.strftime('%Y-%m-%dT23:59:59Z'))
    
    # Get global dates
    global_start_dt = pd.to_datetime(satellite_metadata_dict.get("start_date"))
    global_anom_start_dt = pd.to_datetime(satellite_metadata_dict.get("anom_start_date"))
    
    sat_temp_dir = tempfile.mkdtemp(prefix="sat_monthly_dl_")
    satellite_results_list = []
    path_to_return = None

    # Build task list
    tasks = []
    processed_pairs = set()
    for data_type, sat_sites_dict in satellite_metadata_dict.items():
        if data_type in ["end_date", "start_date", "anom_start_date"] or not isinstance(sat_sites_dict, dict):
            continue
        for site, url_template in sat_sites_dict.items():
            norm_site = site.lower().replace("_", " ").replace("-", " ")
            main_site = next((s for s in main_sites_list if s.lower().replace("_", " ").replace("-", " ") == norm_site), None)
            if main_site and url_template.strip() and (main_site, data_type) not in processed_pairs:
                tasks.append({"site": main_site, "data_type": data_type, "url_template": url_template})
                processed_pairs.add((main_site, data_type))

    print(f"Prepared {len(tasks)} satellite tasks")

    try:
        for task in tqdm(tasks, desc="Satellite Tasks", leave=True):
            site, data_type, url_template = task["site"], task["data_type"], task["url_template"]
            is_anomaly = 'anom' in data_type.lower()
            current_start = global_anom_start_dt if is_anomaly and global_anom_start_dt else global_start_dt
            
            # Skip invalid date ranges
            if not current_start or not global_end_dt or current_start > global_end_dt:
                print(f"          Skipping {site}-{data_type} (invalid date range)")
                continue

            # Calculate monthly periods
            monthly_periods = pd.date_range(
                start=current_start.normalize().replace(day=1),
                end=(global_end_dt + pd.offsets.MonthEnd(0)).normalize(),
                freq='MS'
            )
            monthly_files = []

            for month_start in tqdm(monthly_periods, desc=f"Download {site}-{data_type}", leave=False):
                month_end = (month_start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
                eff_start = max(current_start, month_start)
                eff_end = min(global_end_dt, month_end)
                if eff_start > eff_end: 
                    continue

                # Build URL
                start_str = eff_start.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = eff_end.strftime('%Y-%m-%dT%H:%M:%SZ')
                chunk_url = url_template.replace("{start_date}", start_str).replace("{end_date}", end_str)
                if "{anom_start_date}" in chunk_url: 
                    chunk_url = chunk_url.replace("{anom_start_date}", start_str)

                # Create temp file
                fd, tmp_path = tempfile.mkstemp(
                    suffix=f'_{month_start.strftime("%Y-%m")}.nc', 
                    prefix=f"{site}_{data_type}_", 
                    dir=sat_temp_dir
                )
                os.close(fd)

                # Download and process
                try:
                    with requests.get(chunk_url, timeout=1200, stream=True) as response:
                        response.raise_for_status()
                        with open(tmp_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    # Validate file content
                    if os.path.getsize(tmp_path) > 100:
                        monthly_files.append(tmp_path)
                    else:
                        print(f"          Warning: Empty file for {month_start.strftime('%Y-%m')} ({site}-{data_type})")
                        os.unlink(tmp_path)
                except Exception as e:
                    print(f"          ERROR processing {month_start.strftime('%Y-%m')} ({site}-{data_type}): {e}")
                    if os.path.exists(tmp_path): 
                        os.unlink(tmp_path)

            # Process downloaded files
            if monthly_files:
                result_df = process_stitched_dataset(monthly_files, data_type, site)
                if result_df is not None and not result_df.empty:
                    satellite_results_list.append(result_df)
            
            # Cleanup monthly files
            for f_path in monthly_files:
                if os.path.exists(f_path):
                    os.unlink(f_path)

        # Combine and pivot results
        if not satellite_results_list:
            final_df = pd.DataFrame(columns=['site', 'timestamp'])
            final_df['timestamp'] = pd.to_datetime([])
        else:
            combined = pd.concat(satellite_results_list, ignore_index=True)
            value_cols = [c for c in combined.columns if c not in ['site', 'timestamp', 'data_type']]
            
            # Pivot and flatten columns
            pivot_df = combined.pivot_table(index=['site', 'timestamp'], columns='data_type', values=value_cols, aggfunc='mean')
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [
                    f"sat_{v.replace('-', '_')}_{k.replace('-', '_')}" if len(value_cols) > 1 
                    else f"sat_{v.replace('-', '_')}" 
                    for k, v in pivot_df.columns.values
                ]
            else:
                pivot_df.columns = [f"sat_{col.replace('-', '_')}" for col in pivot_df.columns]
            
            final_df = pivot_df.reset_index()
            if 'timestamp' not in final_df.columns:
                print("WARNING: 'timestamp' column missing after pivot. Adding empty NaT column.")
                final_df['timestamp'] = pd.NaT

        # Save to Parquet
        final_df.to_parquet(output_path, index=False)
        path_to_return = output_path
        print(f"Satellite data saved to: {output_path}")

    except Exception as main_err:
        print(f"\nFATAL ERROR during satellite processing: {main_err}")
        path_to_return = None
    finally:
        # Cleanup temp directory
        if os.path.exists(sat_temp_dir):
            try:
                shutil.rmtree(sat_temp_dir)
                print(f"Cleaned temp directory: {sat_temp_dir}")
            except OSError as e:
                print(f"  Warning: Could not remove temp directory: {e}")

    return path_to_return

def find_best_satellite_match(target_row, sat_pivot_indexed):
    site, target_ts = target_row.get('Site'), target_row.get('timestamp_dt')
    if pd.isna(target_ts): return pd.Series(dtype=float)
    
    site_norm = str(site).lower().replace('_', ' ').replace('-', ' ')
    orig_site = next((s for s in sat_pivot_indexed.index.get_level_values('site').unique() 
                     if str(s).lower().replace('_', ' ').replace('-', ' ') == site_norm), None)
    if not orig_site: return pd.Series(dtype=float)
    
    try: site_data = sat_pivot_indexed.xs(orig_site, level='site')
    except KeyError: return pd.Series(dtype=float)
    if site_data.empty: return pd.Series(dtype=float)
    
    site_data.index = pd.to_datetime(site_data.index)
    site_data = site_data[pd.notna(site_data.index)]
    result_series = pd.Series(index=sat_pivot_indexed.columns, dtype=float)
    
    for var in sat_pivot_indexed.columns:
        if var not in site_data.columns: continue
        var_series = site_data[var].dropna()
        if var_series.empty: continue
        
        if "anom" in var.lower():
            month_period = target_ts.to_period('M') - 1
            prev_month_data = var_series[(var_series.index >= month_period.start_time) & 
                                        (var_series.index < month_period.end_time)]
            if not prev_month_data.empty: 
                result_series[var] = prev_month_data.loc[prev_month_data.index.max()]
            else:
                closest_idx = np.abs(var_series.index - target_ts).argmin()
                result_series[var] = var_series.iloc[closest_idx]
        else:
            data_before = var_series[var_series.index <= target_ts]
            if not data_before.empty: 
                result_series[var] = data_before.loc[data_before.index.max()]
            else:
                closest_idx = np.abs(var_series.index - target_ts).argmin()
                result_series[var] = var_series.iloc[closest_idx]
    return result_series

def add_satellite_data(target_df, sat_path):
    sat_df = pd.read_parquet(sat_path)
    target_proc = target_df.copy()
    target_proc['timestamp_dt'] = pd.to_datetime(target_proc['Date'])
    sat_df['timestamp'] = pd.to_datetime(sat_df['timestamp'])
    
    for df in [target_proc, sat_df]: 
        df.dropna(subset=['timestamp_dt', 'Site'] if df is target_proc else ['timestamp', 'site'], inplace=True)
    
    sat_indexed = sat_df.set_index(['site', 'timestamp']).sort_index()
    print("Applying satellite matching...")
    tqdm.pandas(desc="Satellite Matching")
    matched = target_proc.progress_apply(find_best_satellite_match, axis=1, sat_pivot_indexed=sat_indexed)
    return target_proc.join(matched).drop(columns=['timestamp_dt'], errors='ignore')

# Environmental data processing
def fetch_climate_index(url, var_name, temp_dir):
    print(f"Fetching {var_name}...")
    fname = local_filename(url, '.nc', temp_dir)
    download_file(url, fname)
    ds = xr.open_dataset(fname)
    df = ds.to_dataframe().reset_index()
    
    time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
    var_col = next((c for c in df.columns if c.lower() == var_name.lower()), var_name)
    
    df['datetime'] = pd.to_datetime(df[time_col])
    df = df[['datetime', var_col]].dropna().rename(columns={var_col: 'index'})
    df['Month'] = df['datetime'].dt.to_period('M')
    result = df.groupby('Month')['index'].mean().reset_index()
    ds.close()
    return result[['Month', 'index']].sort_values('Month')

def process_streamflow(url, temp_dir):
    print("Fetching streamflow...")
    fname = local_filename(url, '.json', temp_dir)
    download_file(url, fname)
    with open(fname) as f: data = json.load(f)
    
    values = next((ts['values'][0]['value'] for ts in data.get('value', {}).get('timeSeries', []) 
                 if ts.get('variable', {}).get('variableCode', [{}])[0].get('value') == '00060'), [])
    
    records = [{'Date': pd.to_datetime(item['dateTime'], utc=True).tz_localize(None), 'Flow': float(item['value'])} 
              for item in values if 'dateTime' in item and 'value' in item and float(item['value']) >= 0]
    
    return pd.DataFrame(records).dropna().sort_values('Date')

def fetch_beuti_data(url, sites_dict, temp_dir, power=2):
    print("Fetching BEUTI...")
    fname = local_filename(url, '.nc', temp_dir)
    download_file(url, fname)
    ds = xr.open_dataset(fname)
    df = ds.to_dataframe().reset_index()
    
    time_col = next((c for c in ['time', 'datetime', 'Date', 'T'] if c in df.columns), None)
    lat_col = next((c for c in ['latitude', 'lat'] if c in df.columns), None)
    beuti_var = next((c for c in ['BEUTI', 'beuti'] if c in df.columns or c in ds.data_vars), None)
    
    df_sub = df[[time_col, lat_col, beuti_var]].copy().rename(columns={time_col: 'Date', lat_col: 'lat', beuti_var: 'beuti'})
    df_sub['Date'] = pd.to_datetime(df_sub['Date']).dt.date
    df_sub = df_sub.dropna().sort_values(['Date', 'lat'])
    
    results = []
    for site, coords in sites_dict.items():
        site_lat = coords[0] if isinstance(coords, (list, tuple)) and coords else np.nan
        if pd.isna(site_lat): continue
            
        for date, group in df_sub.groupby('Date'):
            lats, beuti_vals = group['lat'].values, group['beuti'].values
            exact_idx = np.where(np.isclose(lats, site_lat))[0]
            if exact_idx.size > 0:
                beuti_val = np.mean(beuti_vals[exact_idx])
            else:
                dist = np.abs(lats - site_lat)
                weights = 1.0 / (dist ** power + 1e-9)
                valid = ~np.isnan(weights) & ~np.isnan(beuti_vals)
                beuti_val = np.sum(beuti_vals[valid] * weights[valid]) / np.sum(weights[valid]) if np.any(valid) else np.nan
            if pd.notna(beuti_val): results.append({'Date': date, 'Site': site, 'beuti': beuti_val})
                
    result_df = pd.DataFrame(results)
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    ds.close()
    return result_df.sort_values(['Site', 'Date'])

# Core processing
def process_da(da_files_dict):
    print("\n--- Processing DA Data ---")
    results = []
    for name, path in da_files_dict.items():
        site_name = name.replace('-da', '').replace('_da', '').replace('-', ' ').replace('_', ' ').title()
        try:
            df = pd.read_parquet(path)
            if 'Harvest Month' in df.columns and 'Harvest Date' in df.columns and 'Harvest Year' in df.columns:
                df['Date'] = pd.to_datetime(
                    df['Harvest Month'].astype(str) + " " + df['Harvest Date'].astype(str) + ", " + df['Harvest Year'].astype(str),
                    format='%B %d, %Y', errors='coerce'
                )
                date_col = 'Date'
            else:
                date_col = 'CollectDate' if 'CollectDate' in df.columns else None
            da_col = next((c for c in ['Domoic Result', 'Domoic Acid'] if c in df.columns), None)
            
            if not date_col or not da_col: continue
                
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['DA_Levels'] = pd.to_numeric(df[da_col], errors='coerce')
            df['Site'] = site_name
            df = df.dropna(subset=['Parsed_Date', 'DA_Levels', 'Site'])
            
            df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
            weekly = df.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index()
            results.append(weekly[['Year-Week', 'DA_Levels', 'Site']])
            print(f"    Processed {len(weekly)} DA records for {name}")
        except Exception as e: print(f"  Error processing {name}: {e}")
    
    final = pd.concat(results, ignore_index=True)
    return final.groupby(['Year-Week', 'Site'])['DA_Levels'].mean().reset_index() if not final.empty else final

def process_pn(pn_files_dict):
    print("\n--- Processing PN Data ---")
    results = []
    for name, path in pn_files_dict.items():
        site_name = name.replace('-pn', '').replace('_pn', '').replace('-', ' ').replace('_', ' ').title()
        try:
            df = pd.read_parquet(path)
            date_col = 'Date' if 'Date' in df.columns else None
            pn_col = next((c for c in df.columns if "pseudo" in c.lower() and "nitzschia" in c.lower()), None)
            
            if not date_col or not pn_col: continue
                
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['PN_Levels'] = pd.to_numeric(df[pn_col], errors='coerce')
            df['Site'] = site_name
            df = df.dropna(subset=['Parsed_Date', 'PN_Levels', 'Site'])
            
            df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
            weekly = df.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index()
            results.append(weekly[['Year-Week', 'PN_Levels', 'Site']])
            print(f"    Processed {len(weekly)} PN records for {name}")
        except Exception as e: print(f"  Error processing {name}: {e}")
    
    final = pd.concat(results, ignore_index=True)
    return final.groupby(['Year-Week', 'Site'])['PN_Levels'].mean().reset_index() if not final.empty else final

def generate_compiled_data(sites_dict, start_dt, end_dt):
    weeks = pd.date_range(start_dt, end_dt, freq='W-MON', name='Date')
    df_list = [pd.DataFrame({'Date': weeks, 'Site': site.replace('_', ' ').title(), 
                            'lat': coords[0], 'lon': coords[1]}) 
              for site, coords in sites_dict.items() 
              if isinstance(coords, (list, tuple)) and len(coords)==2]
    compiled = pd.concat(df_list, ignore_index=True)
    print(f"  Generated {len(compiled)} site-week entries")
    return compiled.sort_values(['Site', 'Date'])

def compile_data(compiled_df, oni_df, pdo_df, streamflow_df):
    compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])
    compiled_df['TargetPrevMonth'] = compiled_df['Date'].dt.to_period("M") - 1
    
    # Merge climate indices
    for df, col in [(oni_df, 'oni'), (pdo_df, 'pdo')]:
        merge_df = df[["Month", "index"]].rename(columns={"index": col, "Month": "ClimateIndexMonth"}).drop_duplicates()
        compiled_df = pd.merge(compiled_df, merge_df, left_on="TargetPrevMonth", right_on="ClimateIndexMonth", how='left')
        compiled_df.drop(columns=['ClimateIndexMonth'], inplace=True, errors='ignore')
    
    compiled_df.drop(columns=["TargetPrevMonth"], inplace=True)
    
    # Merge streamflow
    streamflow_df["Date"] = pd.to_datetime(streamflow_df["Date"])
    compiled_df = pd.merge_asof(
        compiled_df.sort_values("Date"), 
        streamflow_df[["Date", "Flow"]].sort_values("Date"), 
        on="Date", 
        direction="backward", 
        tolerance=pd.Timedelta("7days")
    ).rename(columns={"Flow": "discharge"}).sort_values(["Site", "Date"])
    
    return compiled_df

def compile_da_pn(lt_df, da_df, pn_df):
    lt_df['Site'] = lt_df['Site'].str.replace('_', ' ').str.title()
    
    for df, col in [(da_df, 'DA_Levels'), (pn_df, 'PN_Levels')]:
        proc_df = df.copy()
        proc_df['Date'] = pd.to_datetime(proc_df['Year-Week'] + '-1', format='%G-%V-%w')
        proc_df['Site'] = proc_df['Site'].str.replace('_', ' ').str.title()
        lt_df = pd.merge(lt_df, proc_df[['Date', 'Site', col]], on=['Date', 'Site'], how='left')
        if col == 'DA_Levels': 
            lt_df.rename(columns={col: f'{col}_orig'}, inplace=True)
            lt_df[col] = lt_df.groupby('Site')[f'{col}_orig'].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
            lt_df.drop(columns=[f'{col}_orig'], inplace=True)
        else:
            lt_df[col] = lt_df.groupby('Site')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    return lt_df

def convert_and_fill(df):
    for col in df.columns.difference(['Date', 'Site']):
        if not pd.api.types.is_numeric_dtype(df[col]): 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Date' in df.columns: 
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# Main pipeline
def main():
    start_time = datetime.now()
    print("\n======= Starting Pipeline =======")
    download_temp_dir = tempfile.mkdtemp(prefix="data_dl_")
    
    # Convert CSVs to Parquet
    da_files_parquet = convert_files_to_parquet(da_files)
    pn_files_parquet = convert_files_to_parquet(pn_files)
    
    # Satellite processing
    satellite_path = None
    if FORCE_SATELLITE_REPROCESSING and os.path.exists(SATELLITE_OUTPUT_PARQUET):
        try: os.remove(SATELLITE_OUTPUT_PARQUET)
        except OSError as e: print(f"Warning: Couldn't remove old satellite file: {e}")
    
    if FORCE_SATELLITE_REPROCESSING or not os.path.exists(SATELLITE_OUTPUT_PARQUET):
        print("--- Generating satellite data ---")
        satellite_path = generate_satellite_parquet(satellite_metadata, list(sites.keys()), SATELLITE_OUTPUT_PARQUET)
    else:
        print("--- Using existing satellite data ---")
        satellite_path = SATELLITE_OUTPUT_PARQUET
    
    # Process core data
    da_data = process_da(da_files_parquet)
    pn_data = process_pn(pn_files_parquet)
    
    # Environmental data
    streamflow_data = process_streamflow(config['streamflow_url'], download_temp_dir)
    pdo_data = fetch_climate_index(config['pdo_url'], "pdo", download_temp_dir)
    oni_data = fetch_climate_index(config['oni_url'], "oni", download_temp_dir)
    beuti_data = fetch_beuti_data(config['beuti_url'], sites, download_temp_dir)
    
    # Compile data
    compiled = generate_compiled_data(sites, start_date, end_date)
    lt_data = compile_data(compiled, oni_data, pdo_data, streamflow_data)
    lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)
    final_data = convert_and_fill(lt_da_pn)
    
    # Merge BEUTI
    beuti_data["Site"] = beuti_data["Site"].str.replace('_', ' ').str.title()
    final_data = pd.merge(final_data, beuti_data, on=["Date", "Site"], how="left").fillna({"beuti": 0})
    
    # Add satellite data
    if satellite_path and os.path.exists(satellite_path):
        final_data = add_satellite_data(final_data, satellite_path)
    
    # Final formatting
    sat_cols = sorted([col for col in final_data.columns if col.startswith("sat_")])
    core_cols = ['Date','Site','lat','lon','oni','pdo','discharge','DA_Levels','PN_Levels','beuti']
    final_data = final_data[[col for col in core_cols if col in final_data.columns] + sat_cols]
    final_data['Date'] = final_data['Date'].dt.strftime("%m/%d/%Y")
    
    # Rename columns
    rename_map = {'Date': 'date', 'Site': 'site', 'DA_Levels': 'da', 'PN_Levels': 'pn'}
    if len(sat_cols) >= 7:
        sat_map = dict(zip(sat_cols[:7], ["chla-anom", "modis-chla", "modis-flr", "modis-k490", "modis-par", "modis-sst", "sst-anom"]))
        rename_map.update(sat_map)
    final_data = final_data.rename(columns=rename_map)
    
    # Save output
    output_dir = os.path.dirname(final_output_path)
    if output_dir:  # Only create directories if path contains directories
        os.makedirs(output_dir, exist_ok=True)
    final_data.to_parquet(final_output_path, index=False)
    
    # Cleanup
    for f in set(downloaded_files + generated_parquet_files):
        if os.path.exists(f): 
            try: os.remove(f)
            except OSError: pass
    try: shutil.rmtree(download_temp_dir)
    except OSError: pass
    
    print(f"\n======= Finished in {datetime.now()-start_time} =======")

if __name__ == "__main__":
    main()