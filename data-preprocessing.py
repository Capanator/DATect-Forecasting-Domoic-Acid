import os
import json
import requests
import pandas as pd
from datetime import datetime
import xarray as xr
import numpy as np

# =============================================================================
# Load Metadata from Configuration
# =============================================================================
with open('metadata.json') as f:
    metadata = json.load(f)

# Metadata elements
original_da_files = metadata["original_da_files"]
original_pn_files = metadata["original_pn_files"]
pdo_url = metadata["pdo_url"]
oni_url = metadata["oni_url"]
beuti_url = metadata["beuti_url"]
streamflow_url = metadata["streamflow_url"]
sites = metadata["sites"]
start_date = datetime.strptime(metadata["start_date"], "%Y-%m-%d")
end_date = datetime.strptime(metadata["end_date"], "%Y-%m-%d")
year_cutoff = metadata["year_cutoff"]
week_cutoff = metadata["week_cutoff"]
final_output_path = metadata.get("final_output_path", "final_output.parquet")

# Lists to track temporary downloads and conversions
downloaded_files = []          # Temporary downloaded files
generated_parquet_files = []   # Parquet files generated from CSVs

# =============================================================================
# Helper Functions
# =============================================================================
def download_file(url, filename):
    """
    Download a file from the provided URL and save it to a local filename.
    """
    try:
        print(f"Downloading {url} to {filename}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        downloaded_files.append(filename)
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def local_filename(url, ext):
    """
    Derive a local filename from the URL ensuring it ends with the specified extension.
    """
    base = url.split('?')[0].split('/')[-1]
    return base if base.endswith(ext) else base + ext

def csv_to_parquet(csv_path):
    """
    Convert a CSV file to a Parquet file.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    parquet_path = csv_path.replace('.csv', '.parquet')
    df.to_parquet(parquet_path, index=False)
    return parquet_path

def convert_files_to_parquet(files):
    """
    Convert CSV files in the given dictionary to Parquet files.
    Returns a new dictionary with updated file paths.
    """
    new_files = {}
    for name, path in files.items():
        if path.lower().endswith('.csv'):
            parquet_path = csv_to_parquet(path)
            new_files[name] = parquet_path
            generated_parquet_files.append(parquet_path)
        else:
            new_files[name] = path
    return new_files

# Convert DA and PN files if they are in CSV format
da_files = convert_files_to_parquet(original_da_files)
pn_files = convert_files_to_parquet(original_pn_files)

# =============================================================================
# Data Processing Functions
# =============================================================================
def process_da(files):
    """
    Process Domoic Acid (DA) data from the given files.
    Returns a concatenated DataFrame with 'Year-Week', 'DA', and 'Location'.
    """
    data_frames = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        if 'CollectDate' in df.columns:
            df['Year-Week'] = pd.to_datetime(df['CollectDate']).dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Result']
        else:
            # Build date from separate Harvest columns
            df['CollectDate'] = df.apply(
                lambda x: f"{x['Harvest Month']} {x['Harvest Date']}, {x['Harvest Year']}", axis=1
            )
            df['Year-Week'] = pd.to_datetime(df['CollectDate'], format='%B %d, %Y').dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Acid']
        df['Location'] = name.replace('-', ' ').title()
        data_frames.append(df[['Year-Week', 'DA', 'Location']])
    return pd.concat(data_frames, ignore_index=True)

def process_pn(files):
    """
    Process Pseudo-nitzschia (PN) data from the given files.
    Returns a concatenated DataFrame with 'Year-Week', 'PN', and 'Location'.
    """
    data_frames = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        df['Date'] = df['Date'].astype(str)
        # Identify the PN column (first occurrence containing the key phrase)
        pn_col = [c for c in df.columns if "Pseudo-nitzschia" in c][0]
        sample_date = df.loc[df['Date'] != 'nan', 'Date'].iloc[0]
        fmt = '%m/%d/%Y' if sample_date.count('/') == 2 and len(sample_date.split('/')[-1]) == 4 else '%m/%d/%y'
        df['Year-Week'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce').dt.strftime('%Y-%U')
        df['PN'] = df[pn_col]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        data_frames.append(df[['Year-Week', 'PN', 'Location']].dropna(subset=['Year-Week']))
    return pd.concat(data_frames, ignore_index=True)

def process_streamflow(url):
    """
    Download and process streamflow data from the given URL.
    Returns a DataFrame with weekly average streamflow.
    """
    fname = local_filename(url, '.json')
    if not os.path.exists(fname) and download_file(url, fname) is None:
        return pd.DataFrame(columns=['Date', 'Flow'])
    try:
        with open(fname) as f:
            data = json.load(f)
    except Exception as e:
        print(e)
        return pd.DataFrame(columns=['Date', 'Flow'])
    
    ts = data.get('value', {}).get('timeSeries', [{}])[0]
    values = ts.get('values', [{}])[0].get('value', [])
    
    records = []
    for item in values:
        dt_str = item.get('dateTime')
        try:
            flow = float(item.get('value'))
        except Exception:
            flow = np.nan
        records.append((dt_str, flow))
    
    df = pd.DataFrame(records, columns=['Date', 'Flow'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year-Week'] = df['Date'].dt.strftime('%Y-W%W')
    
    weekly = df.groupby('Year-Week')['Flow'].mean().reset_index()
    weekly['Date'] = weekly['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w'))
    return weekly[['Date', 'Flow']]

def fetch_climate_index(url, var_name):
    """
    Download and process a climate index (e.g., ONI or PDO) from the given URL.
    Returns a DataFrame with weekly average index values.
    """
    fname = local_filename(url, '.nc')
    if not os.path.exists(fname) and download_file(url, fname) is None:
        return pd.DataFrame()
    try:
        ds = xr.open_dataset(fname)
    except Exception as e:
        print(e)
        return pd.DataFrame()
    
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', var_name]].dropna().rename(columns={'time': 'datetime', var_name: 'index'})
    df['week'] = df['datetime'].dt.strftime('%Y-W%W')
    return df.groupby('week')['index'].mean().reset_index()

def generate_compiled_data(sites, start, end):
    """
    Generate a DataFrame with all week and site combinations within the specified date range.
    """
    weeks = [d.strftime('%Y-W%W') for d in pd.date_range(start, end, freq='W')]
    return pd.DataFrame([
        {'Date': week, 'Site': site, 'Latitude': lat, 'Longitude': lon}
        for week in weeks for site, (lat, lon) in sites.items()
    ])

def compile_data(compiled, oni, pdo, streamflow):
    """
    Merge the compiled site-week data with climate indices (ONI, PDO) and streamflow data.
    """
    compiled = compiled.merge(oni, left_on='Date', right_on='week', how='left') \
                       .drop('week', axis=1) \
                       .rename(columns={'index': 'ONI'})
    compiled = compiled.merge(pdo, left_on='Date', right_on='week', how='left') \
                       .drop('week', axis=1) \
                       .rename(columns={'index': 'PDO'})
    
    # Convert week string to actual date
    compiled['Date'] = pd.to_datetime(compiled['Date'] + '-1', format='%Y-W%W-%w')
    streamflow['Date'] = pd.to_datetime(streamflow['Date'], format='%Y-W%W')
    compiled = compiled.merge(streamflow, on='Date', how='left') \
                       .rename(columns={'Flow': 'Streamflow'})
    return compiled.drop_duplicates(subset=['Date', 'Latitude', 'Longitude'])

def compile_da_pn(lt, da, pn):
    """
    Merge DA and PN data with the main dataset.
    """
    da.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    
    da['Date'] = pd.to_datetime(da['Date'] + '-1', format='%Y-%U-%w')
    pn['Date'] = pd.to_datetime(pn['Date'] + '-1', format='%Y-%U-%w')
    da['DA_Levels'] = pd.to_numeric(da['DA_Levels'], errors='coerce')
    
    merged = lt.merge(da, on=['Date', 'Site'], how='left') \
               .merge(pn, on=['Date', 'Site'], how='left')
    merged['DA_Levels'] = merged['DA_Levels'].interpolate(method='linear')
    merged['PN_Levels'] = merged['PN_Levels'].interpolate(method='linear')
    merged.fillna({'DA_Levels': 0, 'PN_Levels': 0}, inplace=True)
    merged['DA_Levels'] = merged['DA_Levels'].apply(lambda x: 0 if x < 1 else x)
    return merged.loc[:, ~merged.columns.duplicated()]

def filter_data(data, cutoff_year, cutoff_week):
    """
    Filter the data to include only records after the specified cutoff year and week.
    """
    data['Year'] = data['Date'].dt.year
    data['Week'] = data['Date'].dt.isocalendar().week
    return data[(data['Year'] > cutoff_year) | ((data['Year'] == cutoff_year) & (data['Week'] >= cutoff_week))]

def process_duplicates(data):
    """
    Aggregate duplicate entries by averaging numeric values and taking the first value for coordinates.
    """
    agg = {
        'ONI': 'mean',
        'PDO': 'mean',
        'Streamflow': 'mean',
        'DA_Levels': 'mean',
        'PN_Levels': lambda x: x.iloc[0]
    }
    for col in ['Latitude', 'Longitude']:
        if col in data.columns:
            agg[col] = 'first'
    return data.groupby(['Date', 'Site']).agg(agg).reset_index()

def convert_and_fill(data):
    """
    Convert all columns (except Date and Site) to numeric and fill missing values with zeros.
    """
    cols = data.columns.difference(['Date', 'Site'])
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    return data.fillna(0)

def fetch_beuti_data(url, sites, power=2):
    """
    Download and process BEUTI data using IDW spatial interpolation.
    
    Parameters:
        url (str): URL to the netCDF file containing BEUTI data.
        sites (dict): Dictionary of sites with (latitude, longitude) tuples.
        power (float): Power parameter for the IDW interpolation (default is 2).
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Date', 'Site', 'BEUTI', 'beuti_latitude']
    """
    fname = local_filename(url, '.nc')
    if not os.path.exists(fname) and download_file(url, fname) is None:
        return pd.DataFrame()
    try:
        ds = xr.open_dataset(fname)
    except Exception as e:
        print(e)
        return pd.DataFrame()
    
    df = ds.to_dataframe().reset_index()
    if 'time' in df.columns:
        df.rename(columns={'time': 'datetime'}, inplace=True)
    
    # Create a Date column from datetime values
    df['Date'] = pd.to_datetime(df['datetime']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    
    beuti_values = []
    unique_dates = np.sort(df['Date'].unique())
    
    # Iterate over each site and perform IDW interpolation by date
    for site, (site_lat, _) in sites.items():
        site_beuti = []
        for date in unique_dates:
            date_df = df[df['Date'] == date]
            if date_df.empty:
                site_beuti.append(np.nan)
                continue
            
            # If an observation matches the site latitude, use it directly
            exact_match = date_df[np.isclose(date_df['latitude'], site_lat)]
            if not exact_match.empty:
                site_beuti.append(exact_match['BEUTI'].iloc[0])
            else:
                distances = np.abs(date_df['latitude'] - site_lat)
                if np.any(distances == 0):
                    site_beuti.append(date_df.loc[distances.idxmin(), 'BEUTI'])
                else:
                    weights = 1 / (distances ** power)
                    weighted_value = np.sum(date_df['BEUTI'] * weights) / np.sum(weights)
                    site_beuti.append(weighted_value)
                    
        site_data = pd.DataFrame({
            'Date': unique_dates,
            'Site': site,
            'BEUTI': site_beuti,
            'beuti_latitude': site_lat
        })
        beuti_values.append(site_data)
    
    return pd.concat(beuti_values, ignore_index=True)

# =============================================================================
# Data Processing Pipeline
# =============================================================================
da_data = process_da(da_files)
pn_data = process_pn(pn_files)
streamflow_data = process_streamflow(streamflow_url)
pdo_data = fetch_climate_index(pdo_url, 'PDO')
oni_data = fetch_climate_index(oni_url, 'ONI')
beuti_data = fetch_beuti_data(beuti_url, sites)

compiled = generate_compiled_data(sites, start_date, end_date)
lt_data = compile_data(compiled, oni_data, pdo_data, streamflow_data)
lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)
filtered_data = filter_data(lt_da_pn, year_cutoff, week_cutoff)
aggregated_data = process_duplicates(filtered_data)
final_data = convert_and_fill(aggregated_data)

# Standardize column names and order
if 'Latitude' in final_data.columns:
    final_data.rename(columns={'Latitude': 'latitude'}, inplace=True)
if 'Longitude' in final_data.columns:
    final_data.rename(columns={'Longitude': 'longitude'}, inplace=True)

desired_cols = ["Date", "Site", "latitude", "longitude", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels"]
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[desired_cols]
final_data['Date'] = final_data['Date'].dt.strftime("%-m/%-d/%Y")

# =============================================================================
# Integrate BEUTI Data
# =============================================================================
final_data['Date'] = pd.to_datetime(final_data['Date'])  # Ensure Date is datetime
final_data = pd.merge(final_data, beuti_data, on=['Date', 'Site'], how='left')

# Optionally remove the beuti_latitude column if not needed
if 'beuti_latitude' in final_data.columns:
    final_data.drop(columns=['beuti_latitude'], inplace=True)

final_data['BEUTI'] = final_data['BEUTI'].fillna(0)

# Standardize columns and order again
if 'Latitude' in final_data.columns:
    final_data.rename(columns={'Latitude': 'latitude'}, inplace=True)
if 'Longitude' in final_data.columns:
    final_data.rename(columns={'Longitude': 'longitude'}, inplace=True)

desired_cols = ["Date", "Site", "latitude", "longitude", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels", "BEUTI"]
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[desired_cols]
final_data['Date'] = final_data['Date'].dt.strftime("%-m/%-d/%Y")

# Save final output to Parquet
final_data.to_parquet(final_output_path, index=False)
print(f"Final output saved to '{final_output_path}'")

# =============================================================================
# Clean Up Temporary Files
# =============================================================================
# Delete downloaded temporary files
for f in downloaded_files:
    try:
        os.remove(f)
        print(f"Deleted downloaded file: {f}")
    except Exception as e:
        print(f"Error deleting {f}: {e}")

# Delete generated Parquet files from CSV conversion
for f in generated_parquet_files:
    try:
        os.remove(f)
    except Exception as e:
        print(f"Error deleting generated parquet file {f}: {e}")