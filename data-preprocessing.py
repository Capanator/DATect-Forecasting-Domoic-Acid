import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import xarray as xr
import requests

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# File paths for original CSVs (DA and PN) from metadata
original_da_files = metadata["original_da_files"]
original_pn_files = metadata["original_pn_files"]

# URLs and satellite information from metadata
satellite_info = metadata["satellite_info"]
pdo_url = metadata["pdo_url"]
oni_url = metadata["oni_url"]
beuti_url = metadata["beuti_url"]
streamflow_url = metadata["streamflow_url"]

# Sites and date range from metadata
sites = metadata["sites"]
start_date = datetime.strptime(metadata["start_date"], "%Y-%m-%d")
end_date = datetime.strptime(metadata["end_date"], "%Y-%m-%d")
year_cutoff = metadata["year_cutoff"]
week_cutoff = metadata["week_cutoff"]

# Flags and additional satellite metadata
include_satellite = metadata["include_satellite"]
satellite_data_source = metadata.get("satellite_data_source", "url")
measurement_paths = metadata.get("measurement_paths", {})

# Final output file path
final_output_path = metadata.get("final_output_path", "final_output.parquet")

# Global list to track downloaded files
downloaded_files = []

# ------------------------------
# Helper Functions
# ------------------------------

def download_file(url, local_filename):
    """Download a file from the given URL to a local filename.
    If an HTTP error occurs, print an error and return None.
    """
    global downloaded_files
    try:
        print(f"Downloading {url} to {local_filename}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        downloaded_files.append(local_filename)
        return local_filename
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}")
        return None

def get_local_filename(url, default_ext):
    """Generate a local filename based on the URL."""
    base = url.split('?')[0].split('/')[-1]
    if not base.endswith(default_ext):
        base += default_ext
    return base

def convert_csv_to_parquet(csv_path, parquet_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    return parquet_path

# ------------------------------
# File Paths and Conversion
# ------------------------------

# Convert DA CSVs to Parquet and update file paths
da_files = {}
for name, csv_path in original_da_files.items():
    parquet_path = csv_path.replace('.csv', '.parquet')
    convert_csv_to_parquet(csv_path, parquet_path)
    da_files[name] = parquet_path

# Convert PN CSVs to Parquet and update file paths
pn_files = {}
for name, csv_path in original_pn_files.items():
    parquet_path = csv_path.replace('.csv', '.parquet')
    convert_csv_to_parquet(csv_path, parquet_path)
    pn_files[name] = parquet_path

# ------------------------------
# Data Processing Functions
# ------------------------------

def process_da(da_files):
    da_dfs = {name: pd.read_parquet(path) for name, path in da_files.items()}
    for name, df in da_dfs.items():
        if 'CollectDate' in df.columns:
            df['Year-Week'] = pd.to_datetime(df['CollectDate']).dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Result']
        else:
            df['CollectDate'] = df.apply(lambda x: f"{x['Harvest Month']} {x['Harvest Date']}, {x['Harvest Year']}", axis=1)
            df['Year-Week'] = pd.to_datetime(df['CollectDate'], format='%B %d, %Y').dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Acid']
        df['Location'] = name.replace('-', ' ').title()
    return pd.concat([df[['Year-Week', 'DA', 'Location']] for df in da_dfs.values()], ignore_index=True)

def process_pn(pn_files):
    pn_dfs = []
    for name, path in pn_files.items():
        df = pd.read_parquet(path)
        df['Date'] = df['Date'].astype(str)
        pn_column = [col for col in df.columns if "Pseudo-nitzschia" in col][0]
        date_format = '%m/%d/%Y' if df.loc[df['Date'] != 'nan', 'Date'].iloc[0].count('/') == 2 and \
            len(df.loc[df['Date'] != 'nan', 'Date'].iloc[0].split('/')[-1]) == 4 else '%m/%d/%y'
        df['Year-Week'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce').dt.strftime('%Y-%U')
        df['PN'] = df[pn_column]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        pn_dfs.append(df[['Year-Week', 'PN', 'Location']].dropna(subset=['Year-Week']))
    return pd.concat(pn_dfs, ignore_index=True)

def process_streamflow_json(url):
    local_filename = get_local_filename(url, '.json')
    if not os.path.exists(local_filename):
        if download_file(url, local_filename) is None:
            print(f"Skipping streamflow data from {url}")
            return pd.DataFrame(columns=['Date', 'Flow'])
    try:
        with open(local_filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {local_filename}: {e}")
        return pd.DataFrame(columns=['Date', 'Flow'])
    time_series = data.get('value', {}).get('timeSeries', [])
    if not time_series:
        return pd.DataFrame(columns=['Date', 'Flow'])
    ts = time_series[0]
    values_list = ts.get('values', [])
    if values_list:
        values_list = values_list[0].get('value', [])
    else:
        values_list = []
    records = []
    for item in values_list:
        dt_str = item.get('dateTime')
        try:
            flow = float(item.get('value'))
        except (ValueError, TypeError):
            flow = np.nan
        records.append((dt_str, flow))
    df = pd.DataFrame(records, columns=['Date', 'Flow'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year-Week'] = df['Date'].dt.strftime('%Y-W%W')
    weekly = df.groupby('Year-Week')['Flow'].mean().reset_index()
    weekly['Date'] = weekly['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w'))
    return weekly[['Date', 'Flow']]

# ------------------------------
# NetCDF Climate Index Functions
# ------------------------------

def fetch_climate_index_netcdf(url, var_name):
    local_filename = get_local_filename(url, '.nc')
    if not os.path.exists(local_filename):
        if download_file(url, local_filename) is None:
            print(f"Skipping climate index data from {url}")
            return pd.DataFrame()
    try:
        ds = xr.open_dataset(local_filename)
    except Exception as e:
        print(f"Error opening {local_filename}: {e}")
        return pd.DataFrame()
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', var_name]].dropna()
    df = df.rename(columns={'time': 'datetime', var_name: 'index'})
    df['week'] = df['datetime'].dt.strftime('%Y-W%W')
    weekly = df.groupby('week')['index'].mean().reset_index()
    return weekly

def fetch_beuti_netcdf(url):
    local_filename = get_local_filename(url, '.nc')
    if not os.path.exists(local_filename):
        if download_file(url, local_filename) is None:
            print(f"Skipping BEUTI data from {url}")
            return pd.DataFrame()
    try:
        ds = xr.open_dataset(local_filename)
    except Exception as e:
        print(f"Error opening {local_filename}: {e}")
        return pd.DataFrame()
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    if 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    df['Year-Week'] = df['time'].dt.strftime('%Y-W%W')
    df = df[['Year-Week', 'latitude', 'BEUTI']].dropna(subset=['BEUTI'])
    return df.groupby(['latitude', 'Year-Week'])['BEUTI'].mean().reset_index()

def fetch_satellite_data(url, var_name):
    local_filename = get_local_filename(url, '.nc')
    if not os.path.exists(local_filename):
        if download_file(url, local_filename) is None:
            print(f"Skipping satellite data from {url}")
            return pd.DataFrame()
    try:
        ds = xr.open_dataset(local_filename)
    except Exception as e:
        print(f"Error opening {local_filename}: {e}")
        return pd.DataFrame()
    df = ds.to_dataframe().reset_index()
    if 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    if 'lon' in df.columns:
        df = df.rename(columns={'lon': 'longitude'})
    df['time'] = pd.to_datetime(df['time'])
    df['Year-Week'] = df['time'].dt.strftime('%Y-W%W')
    df = df.rename(columns={var_name: 'value'})
    df = df[['Year-Week', 'latitude', 'longitude', 'value']]
    df = df.dropna(subset=['value'])
    return df

# ------------------------------
# Data Compilation Functions
# ------------------------------

def generate_compiled_data(sites, start_date, end_date):
    weeks = [current_week.strftime('%Y-W%W') for current_week in pd.date_range(start_date, end_date, freq='W')]
    return pd.DataFrame([{'Date': week, 'Site': site, 'Latitude': lat, 'Longitude': lon} 
                         for week in weeks for site, (lat, lon) in sites.items()])

def compile_lt_data(compiled_data, beuti_data, oni_data, pdo_data, streamflow_data):
    compiled_data['Date Float'] = compiled_data['Date'].apply(lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    beuti_data['Date Float'] = beuti_data['Year-Week'].apply(lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    points = beuti_data[['latitude', 'Date Float']].values
    values = beuti_data['BEUTI'].values
    interpolated_values = griddata(points, values, compiled_data[['Latitude', 'Date Float']].values, method='linear')
    compiled_data['BEUTI'] = interpolated_values
    kd_tree = KDTree(points)
    nan_indices = compiled_data[pd.isna(compiled_data['BEUTI'])].index
    for index in nan_indices:
        row = compiled_data.loc[index]
        distance, nearest_index = kd_tree.query([row['Latitude'], row['Date Float']])
        compiled_data.at[index, 'BEUTI'] = values[nearest_index]
    compiled_data = compiled_data.merge(oni_data, left_on='Date', right_on='week', how='left')\
                                 .drop('week', axis=1)\
                                 .rename(columns={'index': 'ONI'})
    compiled_data = compiled_data.merge(pdo_data, left_on='Date', right_on='week', how='left')\
                                 .drop('week', axis=1)\
                                 .rename(columns={'index': 'PDO'})
    compiled_data['Date'] = compiled_data['Date'].apply(lambda x: f"{x}-1")
    compiled_data['Date'] = pd.to_datetime(compiled_data['Date'], format='%Y-W%W-%w')
    streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'], format='%Y-W%W')
    compiled_data = compiled_data.merge(streamflow_data, on='Date', how='left')\
                                 .rename(columns={'Flow': 'Streamflow'})
    return compiled_data.drop_duplicates(subset=['Date', 'Latitude', 'Longitude'])

def compile_da_pn_data(lt_data, da_data, pn_data):
    da_data.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn_data.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    da_data['Date'] = da_data['Date'].apply(lambda x: f"{x}-1")
    da_data['Date'] = pd.to_datetime(da_data['Date'], format='%Y-%U-%w')
    pn_data['Date'] = pn_data['Date'].apply(lambda x: f"{x}-1")
    pn_data['Date'] = pd.to_datetime(pn_data['Date'], format='%Y-%U-%w')
    da_data['DA_Levels'] = pd.to_numeric(da_data['DA_Levels'], errors='coerce')
    compiled_with_da = pd.merge(lt_data, da_data, how='left', on=['Date', 'Site'])
    compiled_full = pd.merge(compiled_with_da, pn_data, how='left', on=['Date', 'Site'])
    compiled_full['DA_Levels'] = compiled_full['DA_Levels'].interpolate(method='linear')
    compiled_full['PN_Levels'] = compiled_full['PN_Levels'].interpolate(method='linear')
    compiled_full = compiled_full.fillna({'DA_Levels': 0, 'PN_Levels': 0})
    compiled_full['DA_Levels'] = compiled_full['DA_Levels'].apply(lambda x: 0 if x < 1 else x)
    return compiled_full.loc[:, ~compiled_full.columns.duplicated()]

def filter_data(data, year_cutoff, week_cutoff):
    data['Year'] = data['Date'].dt.year
    data['Week'] = data['Date'].dt.isocalendar().week
    mask = (data['Year'] > year_cutoff) | ((data['Year'] == year_cutoff) & (data['Week'] >= week_cutoff))
    return data[mask]

def process_duplicates(data):
    # Build the aggregation dictionary dynamically based on available columns
    agg_dict = {'BEUTI': 'mean',
                'ONI': 'mean',
                'PDO': 'mean',
                'Streamflow': 'mean',
                'DA_Levels': 'mean',
                'PN_Levels': lambda x: x.iloc[0]}
    # Add satellite columns if they exist
    for col in ['chlorophyll_value', 'fluorescence_value', 'temperature_value', 'radiation_value']:
        if col in data.columns:
            agg_dict[col] = 'mean'
    # Add latitude and longitude if they exist (we want lowercase in final output)
    for col in ['Latitude', 'Longitude', 'latitude', 'longitude']:
        if col in data.columns:
            agg_dict[col] = 'first'
    return data.groupby(['Date', 'Site']).agg(agg_dict).reset_index()

def convert_and_fill(data):
    columns_to_convert = data.columns.difference(['Date', 'Site'])
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    return data.fillna(0)

# ------------------------------
# New Satellite Data Functions
# ------------------------------

def add_satellite_measurements(data, satellite_info, measurement_paths, satellite_data_source="url"):
    """
    Add satellite measurements to the main dataset using monthly matching.
    Two options are available:
      - "url": fetch data from remote NetCDF endpoints.
      - "local": load pre-cleaned CSV files from local paths.
    """
    # Create a Year-Month column for monthly matching
    data['Year-Month'] = data['Date'].dt.strftime('%Y-%m')
    
    if satellite_data_source.lower() == "url":
        for meas_type, (url, var_name, out_col) in satellite_info.items():
            sat_df = fetch_satellite_data(url, var_name)
            # Create a Year-Month column for satellite data.
            # If the 'time' column exists, use it; otherwise, derive it from Year-Week.
            if 'time' in sat_df.columns:
                sat_df['Year-Month'] = pd.to_datetime(sat_df['time']).dt.strftime('%Y-%m')
            else:
                # If 'time' is not available, assume 'Year-Week' exists and convert a representative day.
                sat_df['Year-Month'] = sat_df['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w').strftime('%Y-%m'))
            
            values = []
            # For each row in the main dataset, get the corresponding satellite measurement for the same month.
            for _, row in data.iterrows():
                month = row['Year-Month']
                subset = sat_df[sat_df['Year-Month'] == month]
                if subset.empty:
                    values.append(np.nan)
                else:
                    # Use spatial matching: find the nearest satellite measurement using a BallTree
                    coords_sat = np.radians(subset[['latitude', 'longitude']].values)
                    tree = BallTree(coords_sat, leaf_size=40, metric='haversine')
                    dist, ind = tree.query(np.radians([[row['latitude'], row['longitude']]]), k=1)
                    value = subset.iloc[ind[0][0]]['value']
                    values.append(value)
            data[out_col] = values

    elif satellite_data_source.lower() == "local":
        for meas_type, csv_path in measurement_paths.items():
            # Use the output column name from satellite_info if available
            if meas_type in satellite_info:
                _, _, out_col = satellite_info[meas_type]
            else:
                out_col = meas_type
            try:
                sat_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue
            
            # Create Year-Month column for local satellite data
            if 'time' in sat_df.columns:
                sat_df['Year-Month'] = pd.to_datetime(sat_df['time']).dt.strftime('%Y-%m')
            elif 'Year-Week' in sat_df.columns:
                sat_df['Year-Month'] = sat_df['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w').strftime('%Y-%m'))
            else:
                sat_df['Year-Month'] = np.nan  # or handle differently if necessary
            
            values = []
            for _, row in data.iterrows():
                month = row['Year-Month']
                subset = sat_df[sat_df['Year-Month'] == month]
                if subset.empty:
                    values.append(np.nan)
                else:
                    coords_sat = np.radians(subset[['latitude', 'longitude']].values)
                    tree = BallTree(coords_sat, leaf_size=40, metric='haversine')
                    dist, ind = tree.query(np.radians([[row['latitude'], row['longitude']]]), k=1)
                    if out_col in subset.columns:
                        value = subset.iloc[ind[0][0]][out_col]
                    else:
                        # If the specified column isn't found, use the last column as a fallback.
                        value = subset.iloc[ind[0][0]].iloc[-1]
                    values.append(value)
            data[out_col] = values

    # Optionally, if you prefer not to drop rows missing satellite data, comment out the next line.
    data.dropna(inplace=True)
    return data

# ------------------------------
# Data Processing Pipeline
# ------------------------------

da_data = process_da(da_files)
pn_data = process_pn(pn_files)
streamflow_data = process_streamflow_json(streamflow_url)
pdo_data = fetch_climate_index_netcdf(pdo_url, 'PDO')
oni_data = fetch_climate_index_netcdf(oni_url, 'ONI')
beuti_data = fetch_beuti_netcdf(beuti_url)

compiled_data = generate_compiled_data(sites, start_date, end_date)
lt_data = compile_lt_data(compiled_data, beuti_data, oni_data, pdo_data, streamflow_data)
lt_da_pn_data = compile_da_pn_data(lt_data, da_data, pn_data)
filtered_data = filter_data(lt_da_pn_data, year_cutoff, week_cutoff)

if include_satellite:
    filtered_data = filtered_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    data_with_satellite = add_satellite_measurements(filtered_data, satellite_info, measurement_paths, satellite_data_source)
    data_without_date_float = data_with_satellite.drop('Date Float', axis=1)
else:
    data_without_date_float = filtered_data.copy()

processed_data = process_duplicates(data_without_date_float)
final_data = convert_and_fill(processed_data)

# ------------------------------
# Standardize and Reorder Columns
# ------------------------------

# Rename coordinate columns to lowercase if needed
if 'Latitude' in final_data.columns:
    final_data = final_data.rename(columns={'Latitude': 'latitude'})
if 'Longitude' in final_data.columns:
    final_data = final_data.rename(columns={'Longitude': 'longitude'})

# Define desired columns based on the include_satellite flag.
desired_cols = ["Date", "Site", "latitude", "longitude", "BEUTI", "ONI", "PDO", "Streamflow",
                "DA_Levels", "PN_Levels"]
if include_satellite:
    desired_cols += ["chlorophyll_value", "temperature_value", "radiation_value", "fluorescence_value"]

# Ensure all expected columns exist; if a column is missing, add it with NaN.
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan

# Reorder (and drop any extra columns) so that only the desired columns remain.
final_data = final_data[desired_cols]

# Format date as m/d/yyyy (note: %-m/%-d works on Unix; on Windows use %#m/%#d/%Y)
final_data['Date'] = pd.to_datetime(final_data['Date']).dt.strftime("%-m/%-d/%Y")

# ------------------------------
# Save Final Output and Cleanup
# ------------------------------

final_data.to_parquet(final_output_path, index=False)
print(f"Final output saved to '{final_output_path}'")

def cleanup_downloaded_files():
    global downloaded_files
    for file in downloaded_files:
        try:
            os.remove(file)
            print(f"Deleted downloaded file: {file}")
        except Exception as e:
            print(f"Could not delete file {file}: {e}")

cleanup_downloaded_files()