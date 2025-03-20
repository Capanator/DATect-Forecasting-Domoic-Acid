# first line: 76
@memory.cache
def load_and_prepare_data(file_path: str, season: str = None) -> pd.DataFrame:
    """
    Loads data from a Parquet file, applies optional seasonal filtering,
    adds time-based features, and creates lag features & categories.
    """
    print(f"[INFO] Loading data from {file_path}")
    data = pd.read_parquet(file_path, engine='pyarrow')
    
    # Convert and sort by date
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)
    
    # Optional seasonal subset
    if season and ENABLE_SEASON_SPECIFIC_TRAINER:
        print(f"[INFO] Applying seasonal filter for '{season}'")
        if season == 'spring':
            data = data[data['Date'].dt.month.isin([3, 4, 5, 6, 7])]
        elif season == 'fall':
            data = data[data['Date'].dt.month.isin([9, 10, 11, 12])]
    
    # Cyclical day-of-year transformations
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # Create lag features
    for lag in [1, 2, 3]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    # Create DA_Level categories
    def categorize_da_levels(x):
        if x <= 5:
            return 0
        elif x <= 20:
            return 1
        elif x <= 40:
            return 2
        return 3

    data['DA_Category'] = data['DA_Levels'].apply(categorize_da_levels)
    return data
