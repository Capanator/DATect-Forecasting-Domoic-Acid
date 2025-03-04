import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import xarray as xr
import requests

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# File paths for original CSVs (DA and PN) from metadata
original_da_files = metadata["original_da_files"]
original_pn_files = metadata["original_pn_files"]

# URLs from metadata
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

# Final output file path
final_output_path = metadata.get("final_output_path", "final_output.parquet")

# Global list to track downloaded files
downloaded_files = []

# ------------------------------
# Helper Functions
# ------------------------------

def download_file(url, local_filename):
    """Download a file from the given URL to a local filename."""
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
da_files = {name: convert_csv_to_parquet(csv_path, csv_path.replace('.csv', '.parquet'))
            for name, csv_path in original_da_files.items()}

# Convert PN CSVs to Parquet and update file paths
pn_files = {name: convert_csv_to_parquet(csv_path, csv_path.replace('.csv', '.parquet'))
            for name, csv_path in original_pn_files.items()}

# ------------------------------
# Data Processing Functions
# ------------------------------

def process_da(da_files):
    da_dfs = {name: pd.read_parquet(path) for name, path in da_files.items()}
    for name, df in da_dfs.items():
        df['Year-Week'] = pd.to_datetime(df['CollectDate']).dt.strftime('%Y-%U')
        df['DA'] = df['Domoic Result']
        df['Location'] = name.replace('-', ' ').title()
    return pd.concat([df[['Year-Week', 'DA', 'Location']] for df in da_dfs.values()], ignore_index=True)

def process_pn(pn_files):
    pn_dfs = []
    for name, path in pn_files.items():
        df = pd.read_parquet(path)
        df['Year-Week'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%U')
        df['PN'] = df[[col for col in df.columns if "Pseudo-nitzschia" in col][0]]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        pn_dfs.append(df[['Year-Week', 'PN', 'Location']].dropna(subset=['Year-Week']))
    return pd.concat(pn_dfs, ignore_index=True)

def fetch_climate_index_netcdf(url, var_name):
    local_filename = get_local_filename(url, '.nc')
    if not os.path.exists(local_filename):
        if download_file(url, local_filename) is None:
            print(f"Skipping climate index data from {url}")
            return pd.DataFrame()
    try:
        ds = xr.open_dataset(local_filename)
        df = ds.to_dataframe().reset_index()
        df['week'] = pd.to_datetime(df['time']).dt.strftime('%Y-W%W')
        return df[['week', var_name]].rename(columns={var_name: 'index'})
    except Exception as e:
        print(f"Error opening {local_filename}: {e}")
        return pd.DataFrame()

def generate_compiled_data(sites, start_date, end_date):
    weeks = [current_week.strftime('%Y-W%W') for current_week in pd.date_range(start_date, end_date, freq='W')]
    return pd.DataFrame([{'Date': week, 'Site': site, 'Latitude': lat, 'Longitude': lon} 
                         for week in weeks for site, (lat, lon) in sites.items()])

def compile_da_pn_data(compiled_data, da_data, pn_data):
    da_data.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn_data.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    compiled_full = pd.merge(pd.merge(compiled_data, da_data, how='left', on=['Date', 'Site']),
                             pn_data, how='left', on=['Date', 'Site'])
    return compiled_full.fillna({'DA_Levels': 0, 'PN_Levels': 0})

def process_duplicates(data):
    return data.groupby(['Date', 'Site']).agg({'BEUTI': 'mean', 'ONI': 'mean', 'PDO': 'mean',
                                               'DA_Levels': 'mean', 'PN_Levels': 'first'}).reset_index()

def convert_and_fill(data):
    return data.fillna(0)

# ------------------------------
# Data Processing Pipeline
# ------------------------------

da_data = process_da(da_files)
pn_data = process_pn(pn_files)
pdo_data = fetch_climate_index_netcdf(pdo_url, 'PDO')
oni_data = fetch_climate_index_netcdf(oni_url, 'ONI')

compiled_data = generate_compiled_data(sites, start_date, end_date)
lt_da_pn_data = compile_da_pn_data(compiled_data, da_data, pn_data)
processed_data = process_duplicates(lt_da_pn_data)
final_data = convert_and_fill(processed_data)

# ------------------------------
# Save Final Output and Cleanup
# ------------------------------

final_data.to_parquet(final_output_path, index=False)
print(f"Final output saved to '{final_output_path}'")

def cleanup_downloaded_files():
    global downloaded_files

    # Delete downloaded climate index files
    for file in downloaded_files:
        try:
            os.remove(file)
            print(f"Deleted downloaded file: {file}")
        except Exception as e:
            print(f"Could not delete file {file}: {e}")

    # Delete DA and PN parquet files
    parquet_files = list(da_files.values()) + list(pn_files.values())
    for file in parquet_files:
        try:
            os.remove(file)
            print(f"Deleted intermediate parquet file: {file}")
        except Exception as e:
            print(f"Could not delete parquet file {file}: {e}")

cleanup_downloaded_files()