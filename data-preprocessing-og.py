import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

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

def process_streamflow(streamflow_data):
    output_df = pd.DataFrame(columns=['Date', 'Flow'])
    for column in streamflow_data.columns[1:]:
        year = int(column.split(":")[0])
        temp_df = pd.DataFrame()
        temp_df['Date'] = pd.to_datetime(streamflow_data['mm/dd'] + f"-{year}", format='%d-%b-%Y', errors='coerce')
        temp_df['Flow'] = streamflow_data[column]
        temp_df = temp_df.dropna(subset=['Date'])
        temp_df['Year-Week'] = temp_df['Date'].dt.strftime('%Y-W%U')
        weekly_avg = temp_df.groupby('Year-Week')[['Date', 'Flow']].agg({'Date': 'first', 'Flow': 'mean'}).reset_index()
        output_df = pd.concat([output_df, weekly_avg], ignore_index=True)
    return output_df.groupby('Year-Week').agg({'Date': 'first', 'Flow': 'mean'}).reset_index()

def process_climate_index(index_data, index_name):
    index_df = index_data.drop(index=0).reset_index(drop=True)
    index_df = index_df.iloc[:, :2]  # Keep only the first two columns
    index_df.columns = ['datetime', 'index']
    index_df['datetime'] = pd.to_datetime(index_df['datetime'])
    index_df['index'] = pd.to_numeric(index_df['index'])
    weeks_df = pd.DataFrame({'week': [f"{year}-W{week:02d}" for year in range(2002, 2024) for week in range(1, 54)]})
    weeks_df['year'] = weeks_df['week'].apply(lambda x: int(x.split('-W')[0]))
    weeks_df['week_number'] = weeks_df['week'].apply(lambda x: int(x.split('-W')[1]))
    index_df['year'] = index_df['datetime'].dt.year
    index_df['month'] = index_df['datetime'].dt.month
    index_values = []
    for _, row in weeks_df.iterrows():
        year, week_number = row['year'], row['week_number']
        week_start = datetime.strptime(f"{year}-W{week_number}-1", "%Y-W%W-%w")
        week_end = week_start + timedelta(days=6)
        start_month_index = index_df[(index_df['year'] == week_start.year) & (index_df['month'] == week_start.month)]['index'].values
        end_month_index = index_df[(index_df['year'] == week_end.year) & (index_df['month'] == week_end.month)]['index'].values
        if len(start_month_index) > 0 and len(end_month_index) > 0:
            index_value = (start_month_index[0] + end_month_index[0]) / 2 if week_start.month != week_end.month else start_month_index[0]
        else:
            index_value = None
        index_values.append(index_value)
    weeks_df['index'] = index_values
    return weeks_df[['week', 'index']]

def process_beuti(beuti_data):
    beuti_df = beuti_data.copy()
    beuti_df = beuti_df[beuti_df['time'] != 'UTC']  # Remove rows where 'time' is 'UTC'
    beuti_df['time'] = pd.to_datetime(beuti_df['time'])
    beuti_df['Year-Week'] = beuti_df['time'].dt.strftime('%Y-W%W')
    beuti_df['latitude'] = beuti_df['latitude'].astype(float)
    beuti_df['BEUTI'] = beuti_df['BEUTI'].astype(float)
    return beuti_df.groupby(['latitude', 'Year-Week'])['BEUTI'].mean().reset_index()

def generate_compiled_data(sites, start_date, end_date):
    weeks = [current_week.strftime('%Y-W%W') for current_week in pd.date_range(start_date, end_date, freq='W')]
    return pd.DataFrame([{'Date': week, 'Site': site, 'Latitude': lat, 'Longitude': lon} for week in weeks for site, (lat, lon) in sites.items()])

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
    
    # Convert 'Date' column to datetime format in both DataFrames
    compiled_data['Date'] = compiled_data['Date'].apply(lambda x: f"{x}-1") # Add "-1" to represent the first day of the week
    compiled_data['Date'] = pd.to_datetime(compiled_data['Date'], format='%Y-W%W-%w')
    
    streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'], format='%Y-W%W')
    
    compiled_data = compiled_data.merge(streamflow_data, on='Date', how='left').rename(columns={'Flow': 'Streamflow'})
    return compiled_data.drop_duplicates(subset=['Date', 'Latitude', 'Longitude'])

def compile_da_pn_data(lt_data, da_data, pn_data):
    da_data.rename(columns={'Year-Week': 'Date', 'DA': 'DA_Levels', 'Location': 'Site'}, inplace=True)
    pn_data.rename(columns={'Year-Week': 'Date', 'PN': 'PN_Levels', 'Location': 'Site'}, inplace=True)
    
    # Convert 'Date' column to datetime format in da_data and pn_data
    da_data['Date'] = da_data['Date'].apply(lambda x: f"{x}-1") # Add "-1" to represent the first day of the week
    da_data['Date'] = pd.to_datetime(da_data['Date'], format='%Y-%U-%w')
    pn_data['Date'] = pn_data['Date'].apply(lambda x: f"{x}-1") # Add "-1" to represent the first day of the week
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

def add_measurements(data, measurement_paths):
    data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    for measurement_type, path in measurement_paths.items():
        measurement_df = pd.read_csv(path)
        measurement_df['Year-Week'] = measurement_df['Year-Week'].apply(lambda x: f"{x.split('-')[0]}-W{x.split('-')[1]}")
        measurement_df['latitude'] = pd.to_numeric(measurement_df['latitude'], errors='coerce')
        measurement_df['longitude'] = pd.to_numeric(measurement_df['longitude'], errors='coerce')
        measurement_coords = np.radians(measurement_df[['latitude', 'longitude']])
        tree = BallTree(measurement_coords, leaf_size=40, metric='haversine')
        distances, indices = tree.query(np.radians(data[['latitude', 'longitude']]), k=1)
        data[f'{measurement_type}_value'] = [measurement_df.iloc[i]['value'] for i in indices.flatten()]
    return data

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

measurement_paths = {
    'chlorophyll': './short-term/chlorophyll_cleaned.csv',
    'temperature': './short-term/temperature_cleaned.csv',
    'radiation': './short-term/radiation_cleaned.csv',
    'fluorescence': './short-term/fluorescence_cleaned.csv',
}

year_cutoff = 2002
week_cutoff = 26

da_data = process_da(da_files)
pn_data = process_pn(pn_files)
streamflow_data = process_streamflow(pd.read_csv('./long-term/streamflow.csv', low_memory=False))
pdo_data = process_climate_index(pd.read_csv('./long-term/pdo.csv', low_memory=False), 'pdo') 
oni_data = process_climate_index(pd.read_csv('./long-term/oni.csv', low_memory=False), 'oni')
beuti_data = process_beuti(pd.read_csv('./long-term/beuti.csv', low_memory=False))

compiled_data = generate_compiled_data(sites, start_date, end_date)
lt_data = compile_lt_data(compiled_data, beuti_data, oni_data, pdo_data, streamflow_data)
lt_da_pn_data = compile_da_pn_data(lt_data, da_data, pn_data)
filtered_data = filter_data(lt_da_pn_data, year_cutoff, week_cutoff)
data_with_measurements = add_measurements(filtered_data, measurement_paths)
data_without_date_float = data_with_measurements.drop('Date Float', axis=1)
processed_data = process_duplicates(data_without_date_float)
final_data = convert_and_fill(processed_data)

final_data.to_csv('final_output.csv', index=False)
print("Final output saved to 'final_output.csv'")