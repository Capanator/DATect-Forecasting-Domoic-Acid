# first line: 58
@memory.cache
def load_and_prepare_data(file_path, season=None):
    # Use Parquet with pyarrow for faster IO
    data = pd.read_parquet(file_path, engine='pyarrow')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Seasonal filtering
    if season == 'spring':
        data = data[data['Date'].dt.month.isin([3, 4, 5, 6, 7])]
    elif season == 'fall':
        data = data[data['Date'].dt.month.isin([9, 10, 11, 12])]
    
    data = data.sort_values(['Site', 'Date']).copy()

    # --- Spatial Clustering using scikit-learn KMeans ---
    kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # --- Seasonal Features (sin/cos) ---
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # --- Month (one-hot) ---
    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')

    # --- Year ---
    data['Year'] = data['Date'].dt.year

    # --- Lag Features ---
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    # --- Interaction: cluster * cyclical ---
    cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
        data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']

    # --- DA_Category Based on DA_Levels ---
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
    return data
