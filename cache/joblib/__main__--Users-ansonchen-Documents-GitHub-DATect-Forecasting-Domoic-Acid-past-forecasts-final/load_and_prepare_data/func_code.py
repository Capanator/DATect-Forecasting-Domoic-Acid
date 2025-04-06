# first line: 68
@memory.cache
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data with feature engineering"""
    print(f"[INFO] Loading data from {file_path}")
    data = pd.read_parquet(file_path, engine='pyarrow')
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(['site', 'date'], inplace=True)
    
    # Create cyclical features for day-of-year
    day_of_year = data['date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # Create lag features for da
    for lag in [1, 2, 3]:
        data[f'da_lag_{lag}'] = data.groupby('site')['da'].shift(lag)
    
    # Categorize da
    data['da-category'] = pd.cut(
        data['da'], 
        bins=[-float('inf'), 5, 20, 40, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return data
