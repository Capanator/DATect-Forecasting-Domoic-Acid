# first line: 76
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
    
    # --- Spatial Clustering ---
    if ENABLE_SPATIAL_CLUSTERING:
        print("Performing spatial clustering...")
        kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
        data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
        data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')
    
    # --- Seasonal Features (sin/cos) ---
    if ENABLE_SEASONAL_FEATURES:
        print("Creating seasonal features (sin/cos)...")
        day_of_year = data['Date'].dt.dayofyear
        data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
        data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # --- Month (one-hot) ---
    if ENABLE_MONTH_ONE_HOT:
        print("Creating one-hot encoded month feature...")
        data['Month'] = data['Date'].dt.month
        data = pd.get_dummies(data, columns=['Month'], prefix='Month')
    
    # --- Year ---
    if ENABLE_YEAR_FEATURE:
        print("Adding Year feature...")
        data['Year'] = data['Date'].dt.year
    
    # --- Lag Features ---
    if ENABLE_LAG_FEATURES:
        print("Generating lag features...")
        for lag in [1, 2, 3, 7, 14]:
            data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    # --- Interaction: cluster * cyclical ---
    if ENABLE_INTERACTION_FEATURES:
        print("Creating interaction features between clusters and seasonal features...")
        if ENABLE_SPATIAL_CLUSTERING and ENABLE_SEASONAL_FEATURES:
            cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
            for col in cluster_cols:
                data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
                data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']
    
    # --- DA_Category Based on DA_Levels ---
    if ENABLE_DA_CATEGORY:
        print("Categorizing DA_Levels into DA_Category...")
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
