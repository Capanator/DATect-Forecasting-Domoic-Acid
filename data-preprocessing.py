import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import xarray as xr
import requests

# ------------------------------
# Data Processing Functions
# ------------------------------

def process_da(da_files):
    da_dfs = {name: pd.read_csv(path, low_memory=False) for name, path in da_files.items()}
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
        df = pd.read_csv(path, low_memory=False)
        df['Date'] = df['Date'].astype(str)
        pn_column = [col for col in df.columns if "Pseudo-nitzschia" in col][0]
        date_format = '%m/%d/%Y' if df.loc[df['Date'] != 'nan', 'Date'].iloc[0].count('/') == 2 and len(df.loc[df['Date'] != 'nan', 'Date'].iloc[0].split('/')[-1]) == 4 else '%m/%d/%y'
        df['Year-Week'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce').dt.strftime('%Y-%U')
        df['PN'] = df[pn_column]
        df['Location'] = name.replace('-pn', '').replace('-', ' ').title()
        pn_dfs.append(df[['Year-Week', 'PN', 'Location']].dropna(subset=['Year-Week']))
    return pd.concat(pn_dfs, ignore_index=True)

def process_streamflow_json(url):
    """
    Fetches USGS streamflow data from the provided JSON URL.
    Parses the JSON structure (as shown in your sample) to extract daily flow values,
    groups the data by week (using the 'Year-Week' format), and returns a DataFrame
    with weekly average flows.
    """
    response = requests.get(url)
    data = response.json()
    # Navigate into the JSON structure to get the timeSeries values.
    time_series = data.get('value', {}).get('timeSeries', [])
    if not time_series:
        return pd.DataFrame(columns=['Date', 'Flow'])
    ts = time_series[0]
    values_list = ts.get('values', [])
    if values_list:
        # values_list is a list; we use the first element’s 'value' key
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
    # Create a representative datetime for each week (using the first day of the week)
    weekly['Date'] = weekly['Year-Week'].apply(lambda x: pd.to_datetime(x + '-1', format='%Y-W%W-%w'))
    return weekly[['Date', 'Flow']]

# ------------------------------
# NetCDF Climate Index Functions
# ------------------------------

def fetch_climate_index_netcdf(url, var_name):
    """
    Opens a netCDF file for a climate index (PDO or ONI) and returns a DataFrame 
    with weekly averaged values. The output DataFrame has columns 'week' (formatted as 'YYYY-Www')
    and 'index'.
    """
    ds = xr.open_dataset(url)
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', var_name]].dropna()
    df = df.rename(columns={'time': 'datetime', var_name: 'index'})
    df['week'] = df['datetime'].dt.strftime('%Y-W%W')
    weekly = df.groupby('week')['index'].mean().reset_index()
    return weekly

def fetch_beuti_netcdf(url):
    """
    Opens the BEUTI netCDF file and converts it into a DataFrame.
    Renames the 'lat' coordinate to 'latitude' (if needed), creates a weekly column,
    and returns the data grouped by latitude and week.
    """
    ds = xr.open_dataset(url)
    df = ds.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    if 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    df['Year-Week'] = df['time'].dt.strftime('%Y-W%W')
    df = df[['Year-Week', 'latitude', 'BEUTI']].dropna(subset=['BEUTI'])
    return df.groupby(['latitude', 'Year-Week'])['BEUTI'].mean().reset_index()

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
    compiled_data = compiled_data.merge(oni_data, left_on='Date', right_on='week', how='left').drop('week', axis=1).rename(columns={'index': 'ONI'})
    compiled_data = compiled_data.merge(pdo_data, left_on='Date', right_on='week', how='left').drop('week', axis=1).rename(columns={'index': 'PDO'})
    
    # Convert 'Date' column to datetime format
    compiled_data['Date'] = compiled_data['Date'].apply(lambda x: f"{x}-1")
    compiled_data['Date'] = pd.to_datetime(compiled_data['Date'], format='%Y-W%W-%w')
    
    streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'], format='%Y-W%W')
    
    compiled_data = compiled_data.merge(streamflow_data, on='Date', how='left').rename(columns={'Flow': 'Streamflow'})
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
    return data.groupby(['Date', 'Site']).agg({
        'latitude': 'first',
        'longitude': 'first',
        'BEUTI': 'mean',
        'ONI': 'mean',
        'PDO': 'mean',
        'Streamflow': 'mean',
        'DA_Levels': 'mean',
        'PN_Levels': lambda x: x.iloc[0],
        'chlorophyll_value': 'mean',
        'temperature_value': 'mean',
        'radiation_value': 'mean',
        'fluorescence_value': 'mean'
    }).reset_index()

def convert_and_fill(data):
    columns_to_convert = data.columns.difference(['Date', 'Site'])
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    return data.fillna(0)

# ------------------------------
# New Satellite Data Functions
# ------------------------------

def fetch_satellite_data(url, var_name):
    """
    Opens a netCDF file from the given URL and converts the variable of interest to a DataFrame.
    Renames lat/lon to 'latitude' and 'longitude', creates a Year-Week column, and drops NaN values.
    """
    ds = xr.open_dataset(url)
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

def add_satellite_measurements(data, satellite_info):
    """
    For each satellite measurement type, fetch the netCDF data,
    then assign a measurement value for each row in 'data' by matching the Year-Week
    and using a nearest-neighbor (BallTree) search based on latitude and longitude.
    After processing, any rows with NaN values are dropped.
    """
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
# File Paths and Configuration
# ------------------------------

da_files = {
    'twin-harbors': './da-input/twin-harbors-da.csv',
    'long-beach': './da-input/long-beach-da.csv',
    'quinault': './da-input/quinault-da.csv',
    'kalaloch': './da-input/kalaloch-da.csv',
    'copalis': './da-input/copalis-da.csv',
    'newport': './da-input/newport-da.csv',
    'gold-beach': './da-input/gold-beach-da.csv',
    'coos-bay': './da-input/coos-bay-da.csv',
    'clatsop-beach': './da-input/clatsop-beach-da.csv',
    'cannon-beach': './da-input/cannon-beach-da.csv'
}

pn_files = {
    "gold-beach-pn": "./pn-input/gold-beach-pn.csv",
    "coos-bay-pn": "./pn-input/coos-bay-pn.csv",
    "newport-pn": "./pn-input/newport-pn.csv",
    "clatsop-beach-pn": "./pn-input/clatsop-beach-pn.csv",
    "cannon-beach-pn": "./pn-input/cannon-beach-pn.csv",
    "kalaloch-pn": "./pn-input/kalaloch-pn.csv",
    "copalis-pn": "./pn-input/copalis-pn.csv",
    "long-beach-pn": "./pn-input/long-beach-pn.csv",
    "twin-harbors-pn": "./pn-input/twin-harbors-pn.csv",
    "quinault-pn": "./pn-input/quinault-pn.csv"
}

sites = {
    'Kalaloch': (47.604444, -124.370833),
    'Quinault': (47.466944, -123.845278),
    'Copalis': (47.117778, -124.178333),
    'Twin Harbors': (46.856667, -124.106944),
    'Long Beach': (46.350833, -124.053611),
    'Clatsop Beach': (46.028889, -123.917222),
    'Cannon Beach': (45.881944, -123.959444),
    'Newport': (44.6, -124.05),
    'Coos Bay': (43.376389, -124.237222),
    'Gold Beach': (42.377222, -124.414167)
}

start_date = datetime(2002, 1, 1)
end_date = datetime(2023, 12, 31)

year_cutoff = 2002
week_cutoff = 26

# Satellite netCDF links and corresponding variable info
# (Each tuple is: (netCDF URL, variable name in the netCDF file, output column name))
satellite_info = {
    'fluorescence': (
        'https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B(2025-02-19T00:00:00Z):1:(2025-02-19T00:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B(42):1:(48.75)%5D%5B(-125.5):1:(-123.5)%5D',
        'fluorescence', 'fluorescence_value'
    ),
    'temperature': (
        'https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B(2002-07-05):1:(2025-02-19T00:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B(42):1:(48.75)%5D%5B(-125.5):1:(-123.5)%5D',
        'sst', 'temperature_value'
    ),
    'chlorophyll': (
        'https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B(2025-02-19T00:00:00Z):1:(2025-02-19T00:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B(42):1:(48.75)%5D%5B(-125.5):1:(-123.5)%5D',
        'chlorophyll', 'chlorophyll_value'
    ),
    'radiation': (
        'https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B(2002-07-05):1:(2025-02-19T00:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B(42):1:(48.75)%5D%5B(-125.5):1:(-123.5)%5D',
        'par', 'radiation_value'
    )
}

# NetCDF URLs for climate indices (replacing local CSVs)
pdo_url = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.nc?time%2CPDO&time%3E=2002-06-01&time%3C=2025-01-01T00%3A00%3A00Z"
oni_url = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_ONI.nc?time%2CONI&time%3E=2002-06-01&time%3C=2024-12-01T00%3A00%3A00Z"
beuti_url = "https://oceanview.pfeg.noaa.gov/erddap/griddap/erdBEUTIdaily.nc?BEUTI%5B(2002-06-01):1:(2024-11-28T00:00:00Z)%5D%5B(42):1:(47.0)%5D"

# USGS JSON URL for streamflow
streamflow_url = "https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site=14246900&agencyCd=USGS&statCd=00003&parameterCd=00060&startDT=2002-06-01&endDT=2025-02-22"

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

# Add satellite measurements from netCDF sources
data_with_satellite = add_satellite_measurements(filtered_data, satellite_info)
data_without_date_float = data_with_satellite.drop('Date Float', axis=1)
processed_data = process_duplicates(data_without_date_float)
final_data = convert_and_fill(processed_data)

final_data.to_csv('final_output.csv', index=False)
print("Final output saved to 'final_output.csv'")
