import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
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
    Returns a concatenated DataFrame with 'Biweek', 'DA', and 'Location'.
    """
    data_frames = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        if 'CollectDate' in df.columns:
            df['Date'] = pd.to_datetime(df['CollectDate'])
        else:
            # Build date from separate Harvest columns
            df['CollectDate'] = df.apply(
                lambda x: f"{x['Harvest Month']} {x['Harvest Date']}, {x['Harvest Year']}", axis=1
            )
            df['Date'] = pd.to_datetime(df['CollectDate'], format='%B %d, %Y')
        
        # Calculate biweekly period - every two weeks
        df['Year'] = df['Date'].dt.year
        df['BiweekNum'] = ((df['Date'].dt.dayofyear - 1) // 14) + 1
        df['Biweek'] = df['Year'].astype(str) + '-B' + df['BiweekNum'].astype(str).str.zfill(2)
        
        # Set DA column
        if 'Domoic Result' in df.columns:
            df['DA'] = df['Domoic Result']
        else:
            df['DA'] = df['Domoic Acid']
            
        df['Location'] = name.replace('-', ' ').title()
        data_frames.append(df[['Biweek', 'DA', 'Location', 'Date']])
    
    return pd.concat(data_frames, ignore_index=True)

def process_pn(files):
    """
    Process Pseudo-nitzschia (PN) data from the given files.
    Returns a concatenated DataFrame with 'Biweek', 'PN', and 'Location'.
    """
    data_frames = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        df['Date'] = df['Date'].astype(str)
        # Identify the PN column (first occurrence containing the key phrase)
        pn_col = [c for c in df.columns if "Pseudo-nitzschia" in c][0]
        sample_date = df.loc[df['Date'] != 'nan', 'Date'].iloc[0]
        fmt = '%m/%d/%Y' if sample_date.count('/') == 2 and len(sample_date.split('/')[-1]) == 4 else '%m/%d/%y'
        df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
        
        # Calculate biweekly period - every two weeks
        df['Year'] = df['Date'].dt.year
        df['BiweekNum'] = ((df['Date'].dt.dayofyear - 1) // 14) + 1
        df['Biweek'] = df['Year'].astype(str) + '-B' + df['BiweekNum'].astype(str).str.zfill(2)
        
        df['PN'] = df[pn_col]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        data_frames.append(df[['Biweek', 'PN', 'Location', 'Date']].dropna(subset=['Biweek']))
    
    return pd.concat(data_frames, ignore_index=True)

def process_streamflow(url):
    """
    Download and process streamflow data from the given URL.
    Returns a DataFrame with biweekly average streamflow.
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
    
    # Calculate biweekly period
    df['Year'] = df['Date'].dt.year
    df['BiweekNum'] = ((df['Date'].dt.dayofyear - 1) // 14) + 1
    df['Biweek'] = df['Year'].astype(str) + '-B' + df['BiweekNum'].astype(str).str.zfill(2)
    
    biweekly = df.groupby('Biweek')['Flow'].mean().reset_index()
    
    # Create a representative date for each biweek (first day of the biweek)
    biweekly['Date'] = biweekly['Biweek'].apply(
        lambda x: datetime(int(x.split('-B')[0]), 1, 1) + 
                 timedelta(days=(int(x.split('-B')[1]) - 1) * 14)
    )
    
    return biweekly[['Date', 'Flow', 'Biweek']]

def fetch_climate_index(url, var_name):
    """
    Download and process a climate index (e.g., ONI or PDO) from the given URL.
    Returns a DataFrame with biweekly average index values.
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
    
    # Calculate biweekly period
    df['Year'] = df['datetime'].dt.year
    df['BiweekNum'] = ((df['datetime'].dt.dayofyear - 1) // 14) + 1
    df['Biweek'] = df['Year'].astype(str) + '-B' + df['BiweekNum'].astype(str).str.zfill(2)
    
    return df.groupby('Biweek')['index'].mean().reset_index()

def generate_compiled_data(sites, start, end):
    """
    Generate a DataFrame with all biweek and site combinations within the specified date range.
    """
    # Create biweekly periods
    date_range = pd.date_range(start, end, freq='2W')
    biweeks = []
    
    for d in date_range:
        year = d.year
        biweek_num = ((d.dayofyear - 1) // 14) + 1
        biweek = f"{year}-B{biweek_num:02d}"
        biweeks.append((biweek, d))
    
    return pd.DataFrame([
        {'Date': date, 'Biweek': biweek, 'Site': site, 'Latitude': lat, 'Longitude': lon}
        for biweek, date in biweeks for site, (lat, lon) in sites.items()
    ])

def compile_data(compiled, oni, pdo, streamflow):
    """
    Merge the compiled site-biweek data with climate indices (ONI, PDO) and streamflow data.
    """
    compiled = compiled.merge(oni, on='Biweek', how='left') \
                       .rename(columns={'index': 'ONI'})
    compiled = compiled.merge(pdo, on='Biweek', how='left') \
                       .rename(columns={'index': 'PDO'})
    
    compiled = compiled.merge(streamflow[['Biweek', 'Flow']], on='Biweek', how='left') \
                       .rename(columns={'Flow': 'Streamflow'})
    
    return compiled.drop_duplicates(subset=['Date', 'Latitude', 'Longitude'])

def compile_da_pn(lt, da, pn):
    """
    Merge DA and PN data with the main dataset.
    """
    da.rename(columns={'Biweek': 'Biweek', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn.rename(columns={'Biweek': 'Biweek', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    
    da['DA_Levels'] = pd.to_numeric(da['DA_Levels'], errors='coerce')
    
    # Merge using Biweek instead of Date to minimize interpolation needs
    merged = lt.merge(da[['Biweek', 'Site', 'DA_Levels']], on=['Biweek', 'Site'], how='left') \
               .merge(pn[['Biweek', 'Site', 'PN_Levels']], on=['Biweek', 'Site'], how='left')
    
    # Much less interpolation should be needed now since we're on biweekly schedule
    # matching DA collection frequency
    merged['DA_Levels'] = merged['DA_Levels'].fillna(0)
    merged['PN_Levels'] = merged['PN_Levels'].fillna(0)
    merged['DA_Levels'] = merged['DA_Levels'].apply(lambda x: 0 if x < 1 else x)
    
    return merged.loc[:, ~merged.columns.duplicated()]

def filter_data(data, cutoff_year, cutoff_week):
    """
    Filter the data to include only records after the specified cutoff year and week.
    For biweekly data, we'll convert the week cutoff to a biweek cutoff.
    """
    data['Year'] = data['Date'].dt.year
    biweek_cutoff = (cutoff_week + 1) // 2  # Convert week to biweek (approximate)
    
    return data[(data['Year'] > cutoff_year) | 
                ((data['Year'] == cutoff_year) & 
                 (data['Biweek'].str.split('-B').str[1].astype(int) >= biweek_cutoff))]

def process_duplicates(data):
    """
    Aggregate duplicate entries by averaging numeric values and taking the first value for coordinates.
    """
    # Create a new DataFrame for the aggregated results
    result = data.groupby(['Biweek', 'Site'])['Date'].first().reset_index()
    
    # Add each numeric column one by one
    for col in ['ONI', 'PDO', 'Streamflow', 'DA_Levels', 'PN_Levels']:
        if col in data.columns:
            try:
                # Convert to numeric and calculate mean
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                aggregated = data.groupby(['Biweek', 'Site'])[col].mean()
                result = result.merge(aggregated.reset_index(), on=['Biweek', 'Site'], how='left')
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                # Use first value as fallback
                aggregated = data.groupby(['Biweek', 'Site'])[col].first()
                result = result.merge(aggregated.reset_index(), on=['Biweek', 'Site'], how='left')
    
    # Add coordinate columns
    for col in ['Latitude', 'Longitude']:
        if col in data.columns:
            aggregated = data.groupby(['Biweek', 'Site'])[col].first()
            result = result.merge(aggregated.reset_index(), on=['Biweek', 'Site'], how='left')
            
    return result


def convert_and_fill(data):
    """
    Convert all columns (except Date, Biweek, and Site) to numeric and fill missing values with zeros.
    """
    cols = data.columns.difference(['Date', 'Biweek', 'Site'])
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
        pd.DataFrame: DataFrame with columns ['Date', 'Biweek', 'Site', 'BEUTI', 'beuti_latitude']
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
    
    # Calculate biweekly period
    df['Year'] = df['Date'].dt.year
    df['BiweekNum'] = ((df['Date'].dt.dayofyear - 1) // 14) + 1
    df['Biweek'] = df['Year'].astype(str) + '-B' + df['BiweekNum'].astype(str).str.zfill(2)
    
    beuti_values = []
    
    # Group by biweek for the interpolation
    biweek_groups = df.groupby('Biweek')
    unique_biweeks = sorted(df['Biweek'].unique())
    
    # Iterate over each site and perform IDW interpolation by biweek
    for site, (site_lat, _) in sites.items():
        site_beuti = []
        for biweek in unique_biweeks:
            biweek_df = biweek_groups.get_group(biweek) if biweek in biweek_groups.groups else pd.DataFrame()
            if biweek_df.empty:
                site_beuti.append((biweek, np.nan))
                continue
            
            # If an observation matches the site latitude, use it directly
            exact_match = biweek_df[np.isclose(biweek_df['latitude'], site_lat)]
            if not exact_match.empty:
                site_beuti.append((biweek, exact_match['BEUTI'].iloc[0]))
            else:
                distances = np.abs(biweek_df['latitude'] - site_lat)
                if np.any(distances == 0):
                    site_beuti.append((biweek, biweek_df.loc[distances.idxmin(), 'BEUTI']))
                else:
                    weights = 1 / (distances ** power)
                    weighted_value = np.sum(biweek_df['BEUTI'] * weights) / np.sum(weights)
                    site_beuti.append((biweek, weighted_value))
        
        # Get a representative date for each biweek
        biweek_dates = {biweek: df[df['Biweek'] == biweek]['Date'].min() 
                       for biweek in unique_biweeks if biweek in df['Biweek'].values}
        
        site_data = pd.DataFrame({
            'Biweek': [b for b, _ in site_beuti],
            'Date': [biweek_dates.get(b, pd.NaT) for b, _ in site_beuti],
            'Site': site,
            'BEUTI': [v for _, v in site_beuti],
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

desired_cols = ["Date", "Biweek", "Site", "latitude", "longitude", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels"]
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[desired_cols]
final_data['Date'] = final_data['Date'].dt.strftime("%-m/%-d/%Y")

# =============================================================================
# Integrate BEUTI Data
# =============================================================================
final_data['Date'] = pd.to_datetime(final_data['Date'])  # Ensure Date is datetime
final_data = pd.merge(final_data, beuti_data[['Biweek', 'Site', 'BEUTI']], on=['Biweek', 'Site'], how='left')

final_data['BEUTI'] = final_data['BEUTI'].fillna(0)

# Standardize columns and order again
# Standardize columns and order again
if 'Latitude' in final_data.columns:
    final_data.rename(columns={'Latitude': 'latitude'}, inplace=True)
if 'Longitude' in final_data.columns:
    final_data.rename(columns={'Longitude': 'longitude'}, inplace=True)

# Include Biweek during processing
processing_cols = ["Date", "Biweek", "Site", "latitude", "longitude", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels", "BEUTI"]
for col in processing_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[processing_cols]
final_data['Date'] = final_data['Date'].dt.strftime("%-m/%-d/%Y")

# Before saving, drop the Biweek column
final_data = final_data.drop(columns=['Biweek'])

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
