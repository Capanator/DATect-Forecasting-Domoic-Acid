import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import xarray as xr
import requests
from scipy.linalg import svd  # Use SciPy's SVD

# ------------------------------
# Load Metadata
# ------------------------------
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

# Boolean flag to include satellite processing from metadata
include_satellite = metadata["include_satellite"]

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
# DINEOF Helper Function
# ------------------------------
def dineof_reconstruction(matrix, num_modes=1, tol=1e-5, max_iter=100):
    """
    Reconstruct missing values in a DataFrame using a basic DINEOF approach.
    
    Parameters:
        matrix (pd.DataFrame): DataFrame with index as time (numeric) and columns as space (e.g., latitude).
        num_modes (int): Number of SVD modes to retain.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        pd.DataFrame: The filled DataFrame.
    """
    M = matrix.copy()
    # Create a boolean mask for missing values
    mask = M.isna()
    # Initial fill with the column means; if a column is all missing, fill with 0.
    M = M.fillna(M.mean()).fillna(0)
    
    # Also replace any inf values that might be present.
    M.replace([np.inf, -np.inf], 0, inplace=True)
    
    for iteration in range(max_iter):
        M_old = M.copy()
        # Before SVD, ensure there are no NaNs/infs.
        A = M.values.copy()
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        try:
            U, s, Vt = svd(A, full_matrices=False, lapack_driver='gesvd')
        except Exception as e:
            # If SVD fails, add a very small noise and try again
            A += np.random.normal(0, 1e-8, A.shape)
            U, s, Vt = svd(A, full_matrices=False, lapack_driver='gesvd')
        
        # Reconstruct using the first num_modes modes
        S = np.diag(s[:num_modes])
        M_reconstructed = np.dot(U[:, :num_modes], np.dot(S, Vt[:num_modes, :]))
        # Only update the missing entries
        M.values[mask.values] = M_reconstructed[mask.values]
        # Check for convergence
        if np.linalg.norm(M - M_old) < tol:
            break
    return M

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
    # Create a numeric representation of the week for time axis
    compiled_data['Date Float'] = compiled_data['Date'].apply(
        lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    beuti_data['Date Float'] = beuti_data['Year-Week'].apply(
        lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    
    # Build a grid for DINEOF using the unique Date Float and Latitude values in the compiled data
    unique_dates = np.array(sorted(compiled_data['Date Float'].unique()))
    unique_lats = np.array(sorted(compiled_data['Latitude'].unique()))
    
    # Create an empty DataFrame for the grid
    grid_df = pd.DataFrame(np.nan, index=unique_dates, columns=unique_lats)
    
    # Group BEUTI data by Date Float and latitude and fill the grid with observed BEUTI values
    group = beuti_data.groupby(['Date Float', 'latitude'])['BEUTI'].mean().reset_index()
    for _, r in group.iterrows():
        dt = r['Date Float']
        lat = r['latitude']
        value = r['BEUTI']
        if dt in grid_df.index and lat in grid_df.columns:
            grid_df.at[dt, lat] = value
    
    # Apply the DINEOF reconstruction to fill missing values in the BEUTI grid
    grid_filled = dineof_reconstruction(grid_df, num_modes=1)
    
    # Helper function to find nearest grid value
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    # Assign reconstructed BEUTI values to compiled_data based on nearest Date Float and Latitude
    compiled_data['BEUTI'] = compiled_data.apply(
        lambda row: grid_filled.at[find_nearest(unique_dates, row['Date Float']),
                                    find_nearest(unique_lats, row['Latitude'])],
        axis=1
    )
    
    # Merge in ONI and PDO indices
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
    # Rename columns for consistency
    da_data.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn_data.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    da_data['Date'] = da_data['Date'].apply(lambda x: f"{x}-1")
    da_data['Date'] = pd.to_datetime(da_data['Date'], format='%Y-%U-%w')
    pn_data['Date'] = pn_data['Date'].apply(lambda x: f"{x}-1")
    pn_data['Date'] = pd.to_datetime(pn_data['Date'], format='%Y-%U-%w')
    da_data['DA_Levels'] = pd.to_numeric(da_data['DA_Levels'], errors='coerce')
    
    # Merge the DA and PN data with lt_data
    compiled_with_da = pd.merge(lt_data, da_data, how='left', on=['Date', 'Site'])
    compiled_full = pd.merge(compiled_with_da, pn_data, how='left', on=['Date', 'Site'])
    
    # Apply cubic spline interpolation per site for both DA and PN levels
    compiled_full['DA_Levels'] = compiled_full.groupby('Site')['DA_Levels']\
        .transform(lambda x: x.interpolate(method='spline', order=3))
    compiled_full['PN_Levels'] = compiled_full.groupby('Site')['PN_Levels']\
        .transform(lambda x: x.interpolate(method='spline', order=3))
    
    # Fill any remaining missing values with 0
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
    for col in ['chlorophyll_value', 'fluorescence_value', 'temperature_value', 'radiation_value']:
        if col in data.columns:
            agg_dict[col] = 'mean'
    for col in ['Latitude', 'Longitude', 'latitude', 'longitude']:
        if col in data.columns:
            agg_dict[col] = 'first'
    return data.groupby(['Date', 'Site']).agg(agg_dict).reset_index()

def convert_and_fill(data):
    columns_to_convert = data.columns.difference(['Date', 'Site'])
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    return data.fillna(0)

# ------------------------------
# Satellite Data Functions
# ------------------------------
def add_satellite_measurements(data, satellite_info):
    data['Year-Week'] = data['Date'].dt.strftime('%Y-W%W')
    for meas_type, (url, var_name, out_col) in satellite_info.items():
        sat_df = fetch_satellite_data(url, var_name)
        values = []
        for _, row in data.iterrows():
            week = row['Year-Week']
            subset = sat_df[sat_df['Year-Week'] == week]
            if subset.empty:
                values.append(np.nan)
            else:
                coords_sat = np.radians(subset[['latitude', 'longitude']].values)
                tree = BallTree(coords_sat, leaf_size=40, metric='haversine')
                dist, ind = tree.query(np.radians([[row['latitude'], row['longitude']]]), k=1)
                value = subset.iloc[ind[0][0]]['value']
                values.append(value)
        data[out_col] = values
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
    data_with_satellite = add_satellite_measurements(filtered_data, satellite_info)
    data_without_date_float = data_with_satellite.drop('Date Float', axis=1)
else:
    data_without_date_float = filtered_data.copy()

processed_data = process_duplicates(data_without_date_float)
final_data = convert_and_fill(processed_data)

# ------------------------------
# Standardize and Reorder Columns
# ------------------------------
if 'Latitude' in final_data.columns:
    final_data = final_data.rename(columns={'Latitude': 'latitude'})
if 'Longitude' in final_data.columns:
    final_data = final_data.rename(columns={'Longitude': 'longitude'})

desired_cols = ["Date", "Site", "latitude", "longitude", "BEUTI", "ONI", "PDO", "Streamflow",
                "DA_Levels", "PN_Levels", "chlorophyll_value", "temperature_value", "radiation_value", "fluorescence_value"]
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[desired_cols]

# Format date as m/d/yyyy (Unix formatting; adjust for Windows if needed)
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