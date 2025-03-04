# first line: 68
@memory.cache
def load_and_prepare_data(file_path, season=None):
    print(f"Loading data from {file_path}...")
    # Use Parquet with pyarrow for faster IO
    data = pd.read_parquet(file_path, engine='pyarrow')
    print("Converting 'Date' column to datetime...")
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Seasonal filtering if season-specific trainer is enabled.
    if season is not None and ENABLE_SEASON_SPECIFIC_TRAINER:
        print(f"Applying seasonal filtering for season: {season}...")
        if season == 'spring':
            data = data[data['Date'].dt.month.isin([3, 4, 5, 6, 7])]
        elif season == 'fall':
            data = data[data['Date'].dt.month.isin([9, 10, 11, 12])]
    
    print("Sorting data by 'Site' and 'Date'...")
    data = data.sort_values(['Site', 'Date']).copy()
    
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        
    for lag in [1, 2, 3]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    def categorize_da_levels(x):
        if x <= 5:
            return 0
        elif x <= 20:
            return 1
        elif x <= 40:
            return 2
        else:
            return 3
    data['DA_Category'] = data['DA_Levels'].apply(categorize_da_levels)  
    
    print(f"Data loading and preparation complete for season: {season}")
    return data
