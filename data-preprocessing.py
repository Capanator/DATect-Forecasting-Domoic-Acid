import os, json, requests
import pandas as pd, numpy as np
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import xarray as xr

# ------------------------------
# Load Metadata
# ------------------------------
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

downloaded_files = []  # To track temporary downloads

# ------------------------------
# Helper Functions
# ------------------------------
def download_file(url, filename):
    try:
        print(f"Downloading {url} to {filename}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
        downloaded_files.append(filename)
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def local_filename(url, ext):
    base = url.split('?')[0].split('/')[-1]
    return base if base.endswith(ext) else base + ext

def csv_to_parquet(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    parquet_path = csv_path.replace('.csv', '.parquet')
    df.to_parquet(parquet_path, index=False)
    return parquet_path

# ------------------------------
# File Conversion
# ------------------------------
# Convert DA and PN CSVs to Parquet
da_files = {k: csv_to_parquet(v) for k, v in original_da_files.items()}
pn_files = {k: csv_to_parquet(v) for k, v in original_pn_files.items()}

# ------------------------------
# Data Processing Functions
# ------------------------------
def process_da(files):
    dfs = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        if 'CollectDate' in df.columns:
            df['Year-Week'] = pd.to_datetime(df['CollectDate']).dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Result']
        else:
            df['CollectDate'] = df.apply(
                lambda x: f"{x['Harvest Month']} {x['Harvest Date']}, {x['Harvest Year']}", axis=1
            )
            df['Year-Week'] = pd.to_datetime(df['CollectDate'], format='%B %d, %Y').dt.strftime('%Y-%U')
            df['DA'] = df['Domoic Acid']
        df['Location'] = name.replace('-', ' ').title()
        dfs.append(df[['Year-Week', 'DA', 'Location']])
    return pd.concat(dfs, ignore_index=True)

def process_pn(files):
    dfs = []
    for name, path in files.items():
        df = pd.read_parquet(path)
        df['Date'] = df['Date'].astype(str)
        pn_col = [c for c in df.columns if "Pseudo-nitzschia" in c][0]
        sample_date = df.loc[df['Date'] != 'nan', 'Date'].iloc[0]
        fmt = '%m/%d/%Y' if sample_date.count('/') == 2 and len(sample_date.split('/')[-1]) == 4 else '%m/%d/%y'
        df['Year-Week'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce').dt.strftime('%Y-%U')
        df['PN'] = df[pn_col]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        dfs.append(df[['Year-Week', 'PN', 'Location']].dropna(subset=['Year-Week']))
    return pd.concat(dfs, ignore_index=True)

def process_streamflow(url):
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
        except:
            flow = np.nan
        records.append((dt_str, flow))
    df = pd.DataFrame(records, columns=['Date', 'Flow'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year-Week'] = df['Date'].dt.strftime('%Y-W%W')
    weekly = df.groupby('Year-Week')['Flow'].mean().reset_index()
    weekly['Date'] = weekly['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w'))
    return weekly[['Date', 'Flow']]

def fetch_climate_index(url, var_name):
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

def fetch_beuti(url):
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
    if 'lat' in df.columns:
        df.rename(columns={'lat': 'latitude'}, inplace=True)
    df['Year-Week'] = df['time'].dt.strftime('%Y-W%W')
    df = df[['Year-Week', 'latitude', 'BEUTI']].dropna(subset=['BEUTI'])
    return df.groupby(['latitude', 'Year-Week'])['BEUTI'].mean().reset_index()

def generate_compiled_data(sites, start, end):
    weeks = [d.strftime('%Y-W%W') for d in pd.date_range(start, end, freq='W')]
    return pd.DataFrame([
        {'Date': week, 'Site': site, 'Latitude': lat, 'Longitude': lon}
        for week in weeks for site, (lat, lon) in sites.items()
    ])

def compile_lt(compiled, beuti, oni, pdo, streamflow):
    # Convert week strings to a float for interpolation
    compiled['Date Float'] = compiled['Date'].apply(lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    beuti['Date Float'] = beuti['Year-Week'].apply(lambda x: float(x.split('-')[0]) + float(x.split('-')[1][1:]) / 52)
    
    pts = beuti[['latitude', 'Date Float']].values
    vals = beuti['BEUTI'].values
    compiled['BEUTI'] = griddata(pts, vals, compiled[['Latitude', 'Date Float']].values, method='linear')
    
    # Replace any NaNs with nearest neighbor from a KDTree
    tree = KDTree(pts)
    for idx in compiled[compiled['BEUTI'].isna()].index:
        row = compiled.loc[idx]
        _, nearest = tree.query([row['Latitude'], row['Date Float']])
        compiled.at[idx, 'BEUTI'] = vals[nearest]
    
    compiled = compiled.merge(oni, left_on='Date', right_on='week', how='left')\
                       .drop('week', axis=1)\
                       .rename(columns={'index': 'ONI'})
    compiled = compiled.merge(pdo, left_on='Date', right_on='week', how='left')\
                       .drop('week', axis=1)\
                       .rename(columns={'index': 'PDO'})
    
    compiled['Date'] = pd.to_datetime(compiled['Date'] + '-1', format='%Y-W%W-%w')
    streamflow['Date'] = pd.to_datetime(streamflow['Date'], format='%Y-W%W')
    compiled = compiled.merge(streamflow, on='Date', how='left')\
                       .rename(columns={'Flow': 'Streamflow'})
    return compiled.drop_duplicates(subset=['Date', 'Latitude', 'Longitude'])

def compile_da_pn(lt, da, pn):
    da.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    da['Date'] = pd.to_datetime(da['Date'] + '-1', format='%Y-%U-%w')
    pn['Date'] = pd.to_datetime(pn['Date'] + '-1', format='%Y-%U-%w')
    da['DA_Levels'] = pd.to_numeric(da['DA_Levels'], errors='coerce')
    
    merged = lt.merge(da, on=['Date', 'Site'], how='left')\
               .merge(pn, on=['Date', 'Site'], how='left')
    merged['DA_Levels'] = merged['DA_Levels'].interpolate(method='linear')
    merged['PN_Levels'] = merged['PN_Levels'].interpolate(method='linear')
    merged.fillna({'DA_Levels': 0, 'PN_Levels': 0}, inplace=True)
    merged['DA_Levels'] = merged['DA_Levels'].apply(lambda x: 0 if x < 1 else x)
    return merged.loc[:, ~merged.columns.duplicated()]

def filter_data(data, cutoff_year, cutoff_week):
    data['Year'] = data['Date'].dt.year
    data['Week'] = data['Date'].dt.isocalendar().week
    return data[(data['Year'] > cutoff_year) | ((data['Year'] == cutoff_year) & (data['Week'] >= cutoff_week))]

def process_duplicates(data):
    agg = {
        'BEUTI': 'mean',
        'ONI': 'mean',
        'PDO': 'mean',
        'Streamflow': 'mean',
        'DA_Levels': 'mean',
        'PN_Levels': lambda x: x.iloc[0]
    }
    for col in ['Latitude', 'Longitude', 'latitude', 'longitude']:
        if col in data.columns:
            agg[col] = 'first'
    return data.groupby(['Date', 'Site']).agg(agg).reset_index()

def convert_and_fill(data):
    cols = data.columns.difference(['Date', 'Site'])
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    return data.fillna(0)

# ------------------------------
# Data Processing Pipeline
# ------------------------------
da_data = process_da(da_files)
pn_data = process_pn(pn_files)
streamflow_data = process_streamflow(streamflow_url)
pdo_data = fetch_climate_index(pdo_url, 'PDO')
oni_data = fetch_climate_index(oni_url, 'ONI')
beuti_data = fetch_beuti(beuti_url)

compiled = generate_compiled_data(sites, start_date, end_date)
lt_data = compile_lt(compiled, beuti_data, oni_data, pdo_data, streamflow_data)
lt_da_pn = compile_da_pn(lt_data, da_data, pn_data)
filtered = filter_data(lt_da_pn, year_cutoff, week_cutoff)
processed = process_duplicates(filtered)
final_data = convert_and_fill(processed)

# Standardize column names and order
if 'Latitude' in final_data.columns:
    final_data.rename(columns={'Latitude': 'latitude'}, inplace=True)
if 'Longitude' in final_data.columns:
    final_data.rename(columns={'Longitude': 'longitude'}, inplace=True)
desired_cols = ["Date", "Site", "latitude", "longitude", "BEUTI", "ONI", "PDO", "Streamflow", "DA_Levels", "PN_Levels"]
for col in desired_cols:
    if col not in final_data.columns:
        final_data[col] = np.nan
final_data = final_data[desired_cols]
final_data['Date'] = pd.to_datetime(final_data['Date']).dt.strftime("%-m/%-d/%Y")

# Save the final output
final_data.to_parquet(final_output_path, index=False)
print(f"Final output saved to '{final_output_path}'")

# Clean up downloaded temporary files
for f in downloaded_files:
    try:
        os.remove(f)
        print(f"Deleted {f}")
    except Exception as e:
        print(f"Error deleting {f}: {e}")
