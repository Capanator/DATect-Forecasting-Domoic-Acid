# first line: 68
@memory.cache
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data with feature engineering"""
    print(f"[INFO] Loading data from {file_path}")
    data = pd.read_parquet(file_path, engine='pyarrow')
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)
    
    # Create cyclical features for day-of-year
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # Create lag features for DA_Levels
    for lag in [1, 2, 3]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    # Categorize DA_Levels
    data['DA_Category'] = pd.cut(
        data['DA_Levels'], 
        bins=[-float('inf'), 5, 20, 40, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return data
