import pandas as pd
import numpy as np
from joblib import Memory
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ---------------------------------------------------------
# Global flags for feature engineering steps
ENABLE_SPATIAL_CLUSTERING    = False
ENABLE_SEASONAL_FEATURES     = True
ENABLE_MONTH_ONE_HOT         = False
ENABLE_YEAR_FEATURE          = False
ENABLE_LAG_FEATURES          = True
ENABLE_INTERACTION_FEATURES  = False
ENABLE_DA_CATEGORY           = True

# Global flag for season-specific trainer (i.e. seasonal filtering)
ENABLE_SEASON_SPECIFIC_TRAINER = False

# Global flag to enable/disable linear/logistic regression code.
ENABLE_LINEAR_LOGISTIC = False  # Set to False to disable LR code

# Global flag to enable/disable random anchor forecasts (and related dashboard components)
ENABLE_RANDOM_ANCHOR_FORECASTS = True  # Set to False to disable random anchor forecasts entirely

# New global flag to enable/disable GridSearchCV in model training.
ENABLE_GRIDSEARCHCV = False  # Set to False to bypass GridSearchCV

# ---------------------------------------------------------
# Setup caching directory for intermediate data (using joblib)
print("Setting up caching directory...")
cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)
print("Caching directory is ready.")

# ---------------------------------------------------------
# Helper: Create a numeric preprocessing pipeline
def create_numeric_pipeline():
    print("Creating numeric preprocessing pipeline...")
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

# Helper: Create a ColumnTransformer for numeric features
def create_numeric_transformer(df, drop_cols):
    print("Creating numeric transformer for given dataframe...")
    X = df.drop(columns=drop_cols, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    pipeline = create_numeric_pipeline()
    transformer = ColumnTransformer(
        transformers=[('num', pipeline, numeric_cols)],
        remainder='passthrough'
    )
    return transformer, X

# ---------------------------------------------------------
# 1) Data Loading & Feature Engineering (cached)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2) Common Model Training Functions
# ---------------------------------------------------------
def train_model(model, X_train, y_train, X_test, model_type='regression', cv=None, param_grid=None):
    print("Training model...")
    # Only perform GridSearchCV if enabled AND a parameter grid is provided.
    if param_grid is not None and ENABLE_GRIDSEARCHCV:
        print("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='r2' if model_type == 'regression' else 'accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("GridSearchCV complete. Best parameters used:", grid_search.best_params_)
    else:
        # Otherwise, just train the model directly.
        model.fit(X_train, y_train)
        best_model = model
        if param_grid is not None:
            print("GridSearchCV is disabled. Model trained with default parameters.")
        else:
            print("Model training complete without GridSearchCV.")
    predictions = best_model.predict(X_test)
    return best_model, predictions

# ---------------------------------------------------------
# 3a) Train & Predict Function (Machine Learning Approach)
# ---------------------------------------------------------
def train_and_predict(data):
    print("Starting ML-based train and predict...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Regression Setup
    print("Preparing data for regression...")
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.2, random_state=42)
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(train_set_reg, drop_cols_reg)
    X_test_reg = test_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = transformer_reg.transform(test_set_reg.drop(columns=drop_cols_reg, errors='ignore'))
    
    print("Training RandomForestRegressor...")
    param_grid_reg = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_reg, y_pred_reg = train_model(RandomForestRegressor(random_state=42),
                                       X_train_reg_processed, train_set_reg['DA_Levels'],
                                       X_test_reg_processed,
                                       model_type='regression',
                                       cv=tscv,
                                       param_grid=param_grid_reg)
    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(test_set_reg['DA_Levels'], y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(test_set_reg['DA_Levels'], y_pred_reg))
    print("Regression model training and prediction complete.")
    
    site_stats_reg = test_set_reg[['DA_Levels', 'Predicted_DA_Levels']].groupby(test_set_reg['Site']).apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )
    
    # Classification Setup
    print("Preparing data for classification...")
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.2, random_state=42)
    drop_cols_cls = ['DA_Category', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(train_set_cls, drop_cols_cls)
    X_test_cls = test_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = transformer_cls.transform(test_set_cls.drop(columns=drop_cols_cls, errors='ignore'))
    
    print("Training RandomForestClassifier...")
    best_cls, y_pred_cls = train_model(RandomForestClassifier(random_state=42),
                                       X_train_cls_processed, train_set_cls['DA_Category'],
                                       X_test_cls_processed,
                                       model_type='classification',
                                       cv=tscv,
                                       param_grid={
                                           'n_estimators': [200, 300],
                                           'max_depth': [10, 15],
                                           'min_samples_split': [2, 5],
                                           'min_samples_leaf': [1, 2]
                                       })
    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(test_set_cls['DA_Category'], y_pred_cls)
    print("Classification model training and prediction complete.")
    
    site_stats_cls = test_set_cls[['DA_Category', 'Predicted_DA_Category']].groupby(test_set_cls['Site']).apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )
    
    print("ML train and predict function complete.")
    return {
        "DA_Level": {
            "test_df": test_set_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_rmse": overall_rmse_reg
        },
        "DA_Category": {
            "test_df": test_set_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls
        }
    }

# ---------------------------------------------------------
# 3b) Train & Predict Function (Linear & Logistic Regression)
# ---------------------------------------------------------
def train_and_predict_lr(data):
    print("Starting LR-based train and predict...")
    # Regression Setup
    print("Preparing regression data for LinearRegression...")
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.25, random_state=42)
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(train_set_reg, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(test_set_reg.drop(columns=drop_cols_reg, errors='ignore'))
    
    print("Training LinearRegression model...")
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train_reg_processed, train_set_reg['DA_Levels'])
    y_pred_reg = lin_regressor.predict(X_test_reg_processed)
    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(test_set_reg['DA_Levels'], y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(test_set_reg['DA_Levels'], y_pred_reg))
    print("Linear regression complete.")
    
    site_stats_reg = test_set_reg[['DA_Levels', 'Predicted_DA_Levels']].groupby(test_set_reg['Site']).apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )
    
    # Classification Setup
    print("Preparing classification data for LogisticRegression...")
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.25, random_state=42)
    drop_cols_cls = ['DA_Category', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(train_set_cls, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = transformer_cls.transform(test_set_cls.drop(columns=drop_cols_cls, errors='ignore'))
    
    print("Training LogisticRegression model...")
    log_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_classifier.fit(X_train_cls_processed, train_set_cls['DA_Category'])
    y_pred_cls = log_classifier.predict(X_test_cls_processed)
    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(test_set_cls['DA_Category'], y_pred_cls)
    print("Logistic regression complete.")
    
    site_stats_cls = test_set_cls[['DA_Category', 'Predicted_DA_Category']].groupby(test_set_cls['Site']).apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )
    
    print("LR train and predict function complete.")
    return {
        "DA_Level": {
            "test_df": test_set_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_rmse": overall_rmse_reg
        },
        "DA_Category": {
            "test_df": test_set_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls
        }
    }

# ---------------------------------------------------------
# 4a) Time-Based Forecast Function for Random Anchor Dates (ML)
# ---------------------------------------------------------
def forecast_next_date(df, anchor_date, site):
    print(f"Forecasting next date for site: {site} after anchor date: {anchor_date} (ML)...")
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        print("No future data available for forecasting.")
        return None
    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        print("No training data available before anchor date.")
        return None
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        print("No test data available for the next date.")
        return None

    # Regression forecast
    print("Forecasting DA Levels using RandomForestRegressor...")
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(df_train, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = transformer_reg.transform(df_test.drop(columns=drop_cols_reg, errors='ignore'))
    reg_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None)
    reg_model.fit(X_train_reg_processed, df_train['DA_Levels'])
    y_pred_reg = reg_model.predict(X_test_reg_processed)

    # Classification forecast
    print("Forecasting DA Category using RandomForestClassifier...")
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(df_train, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = transformer_cls.transform(df_test.drop(columns=drop_cols_cls, errors='ignore'))
    cls_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
    cls_model.fit(X_train_cls_processed, df_train['DA_Category'])
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    print("Forecast complete for ML method.")
    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]) if not df_test.empty else None,
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0]) if not df_test.empty else None
    }

# ---------------------------------------------------------
# 4b) Time-Based Forecast Function for Random Anchor Dates (LR)
# ---------------------------------------------------------
def forecast_next_date_lr(df, anchor_date, site):
    print(f"Forecasting next date for site: {site} after anchor date: {anchor_date} (LR)...")
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        print("No future data available for forecasting.")
        return None
    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        print("No training data available before anchor date.")
        return None
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        print("No test data available for the next date.")
        return None

    # Regression using LinearRegression
    print("Forecasting DA Levels using LinearRegression...")
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(df_train, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = transformer_reg.transform(df_test.drop(columns=drop_cols_reg, errors='ignore'))
    lin_model = LinearRegression()
    lin_model.fit(X_train_reg_processed, df_train['DA_Levels'])
    y_pred_reg = lin_model.predict(X_test_reg_processed)

    # Classification using LogisticRegression
    print("Forecasting DA Category using LogisticRegression...")
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(df_train, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = transformer_cls.transform(df_test.drop(columns=drop_cols_cls, errors='ignore'))
    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_model.fit(X_train_cls_processed, df_train['DA_Category'])
    y_pred_cls = log_model.predict(X_test_cls_processed)

    print("Forecast complete for LR method.")
    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]) if not df_test.empty else None,
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0]) if not df_test.empty else None
    }

# ---------------------------------------------------------
# 5) Modified Random Anchor Forecast Helper Function
# ---------------------------------------------------------
def get_random_anchor_forecasts(data, forecast_func):
    print("Generating random anchor forecasts...")
    NUM_RANDOM_ANCHORS = 50
    df_after_2010 = data[data['Date'].dt.year >= 2010].copy().sort_values(['Site', 'Date'])
    pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()
    df_random_anchors = pd.concat([
        group.sample(n=min(NUM_RANDOM_ANCHORS, len(group)), random_state=42)
        for _, group in pairs_after_2010.groupby('Site')
    ]).reset_index(drop=True)

    results_list = []
    for _, row in df_random_anchors.iterrows():
        site_ = row['Site']
        anchor_date_ = row['Date']
        print(f"Forecasting for Site: {site_} at Anchor Date: {anchor_date_}...")
        result = forecast_func(data, anchor_date_, site_)
        if result is not None:
            results_list.append(result)
    df_results_anchors = pd.DataFrame(results_list)

    if not df_results_anchors.empty:
        rmse_anchors = np.sqrt(mean_squared_error(
            df_results_anchors['Actual_DA_Levels'],
            df_results_anchors['Predicted_DA_Levels']
        ))
        acc_anchors = (df_results_anchors['Actual_DA_Category'] == df_results_anchors['Predicted_DA_Category']).mean()
    else:
        rmse_anchors = None
        acc_anchors = None

    figs_random_site = {}
    if not df_results_anchors.empty:
        for site, df_site in df_results_anchors.groupby("Site"):
            df_line = df_site.copy().sort_values("NextDate")
            df_plot_melt = df_line.melt(
                id_vars=["NextDate"],
                value_vars=["Actual_DA_Levels", "Predicted_DA_Levels"],
                var_name="Type", value_name="DA_Level"
            )
            fig = px.line(
                df_plot_melt,
                x="NextDate",
                y="DA_Level",
                color="Type",
                title=f"Random Anchor Forecast for Site {site}"
            )
            fig.update_layout(xaxis_title="Next Date", yaxis_title="DA Level")
            figs_random_site[site] = fig
    print("Random anchor forecasts generated.")
    return df_results_anchors, figs_random_site, rmse_anchors, acc_anchors

# ---------------------------------------------------------
# 6) Prepare Data, Predictions, and Random Anchors for Each Season
# ---------------------------------------------------------
print("Starting data preparation...")
file_path = 'final_output.parquet'
raw_data_annual = load_and_prepare_data(file_path, season=None)
print("Annual data loaded.")

if ENABLE_SEASON_SPECIFIC_TRAINER:
    print("Loading season-specific data for spring and fall...")
    raw_data_spring = load_and_prepare_data(file_path, season='spring')
    raw_data_fall   = load_and_prepare_data(file_path, season='fall')
    raw_data_dict = {
        'annual': raw_data_annual,
        'spring': raw_data_spring,
        'fall': raw_data_fall
    }
else:
    raw_data_dict = {'annual': raw_data_annual}
    print("Season-specific trainer is disabled. Using annual data only.")

print("Computing ML-based predictions/forecasts...")
predictions_ml = {
    'annual': train_and_predict(raw_data_annual),
    'spring': train_and_predict(raw_data_spring) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
    'fall': train_and_predict(raw_data_fall) if ENABLE_SEASON_SPECIFIC_TRAINER else None
}

if ENABLE_RANDOM_ANCHOR_FORECASTS:
    random_anchors_ml = {
        'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date),
        'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
        'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date) if ENABLE_SEASON_SPECIFIC_TRAINER else None
    }
else:
    random_anchors_ml = None

if ENABLE_LINEAR_LOGISTIC:
    print("Computing LR-based predictions/forecasts...")
    predictions_lr = {
        'annual': train_and_predict_lr(raw_data_annual),
        'spring': train_and_predict_lr(raw_data_spring),
        'fall': train_and_predict_lr(raw_data_fall)
    }
    if ENABLE_RANDOM_ANCHOR_FORECASTS:
        random_anchors_lr = {
            'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date_lr),
            'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date_lr),
            'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date_lr)
        }
    else:
        random_anchors_lr = None
else:
    predictions_lr = None
    random_anchors_lr = None
    print("LR-based predictions/forecasts are disabled.")

# ---------------------------------------------------------
# 7) Build the Dash App Layout & Callbacks
# ---------------------------------------------------------
print("Setting up Dash app layout...")
app = dash.Dash(__name__)

# Define the season dropdown conditionally.
if ENABLE_SEASON_SPECIFIC_TRAINER:
    season_dropdown = dcc.Dropdown(
        id='season-dropdown',
        options=[
            {'label': 'Annual', 'value': 'annual'},
            {'label': 'Spring Bloom', 'value': 'spring'},
            {'label': 'Fall Bloom', 'value': 'fall'}
        ],
        value='annual',
        style={'width': '30%'}
    )
else:
    season_dropdown = None

# Add a dummy store to trigger callbacks when season dropdown is not used.
dummy_store = dcc.Store(id='dummy-store', data='initial')

# Analysis layout – includes forecast type and site selection.
analysis_layout = html.Div([
    html.H3("Overall Analysis (Time-Series)"),
    dcc.Dropdown(
        id='forecast-type-dropdown',
        options=[
            {'label': 'DA Levels', 'value': 'DA_Level'},
            {'label': 'DA Category', 'value': 'DA_Category'}
        ],
        value='DA_Level',
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        placeholder='Select Site',
        style={'width': '50%'}
    ),
    dcc.Graph(id='analysis-graph')
])

# Random Anchor Forecast layout.
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    random_anchors_layout = html.Div([
        html.H3("Random Anchor Dates Forecast -> Next Date by Site"),
        html.Div(id='random-anchor-container')
    ])
else:
    random_anchors_layout = html.Div([
        html.H3("Random Anchor Forecasts are disabled.")
    ])

# Build the global layout.
tabs_children = [dcc.Tab(label='Analysis', children=[analysis_layout])]
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    tabs_children.append(dcc.Tab(label='Random Anchor Forecast', children=[random_anchors_layout]))

layout_children = [
    html.H1("Domoic Acid Forecast Dashboard"),
    html.Div([
        season_dropdown if season_dropdown is not None else html.Div(),
        html.Label("Select Forecast Method"),
        dcc.Dropdown(
            id='forecast-method-dropdown',
            options=[{'label': 'Machine Learning Forecasts', 'value': 'ml'}] +
                    ([{'label': 'Linear/Logistic Regression Forecasts', 'value': 'lr'}] if ENABLE_LINEAR_LOGISTIC else []),
            value='ml',
            style={'width': '30%', 'marginLeft': '20px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
    dcc.Tabs(id="tabs", children=tabs_children),
    dummy_store  # Add dummy store so callbacks have an Input even if season dropdown is absent.
]
app.layout = html.Div(layout_children)
print("Dash app layout is set.")

# ---------------------------------------------------------------------
# Callback to update the site dropdown options.
if ENABLE_SEASON_SPECIFIC_TRAINER:
    @app.callback(
        Output('site-dropdown', 'options'),
        [Input('season-dropdown', 'value')]
    )
    def update_site_dropdown(selected_season):
        print("Updating site dropdown options for season:", selected_season)
        data = raw_data_dict[selected_season]
        options = [{'label': 'All Sites', 'value': 'All Sites'}] + [
            {'label': site, 'value': site} for site in data['Site'].unique()
        ]
        return options
else:
    @app.callback(
        Output('site-dropdown', 'options'),
        [Input('dummy-store', 'data')]
    )
    def update_site_dropdown(dummy_data):
        print("Updating site dropdown options with dummy data input...")
        selected_season = 'annual'
        data = raw_data_dict[selected_season]
        options = [{'label': 'All Sites', 'value': 'All Sites'}] + [
            {'label': site, 'value': site} for site in data['Site'].unique()
        ]
        print(f"Site options updated for season: {selected_season}")
        return options

# ---------------------------------------------------------------------
# Callback to update the analysis graph.
@app.callback(
    Output('analysis-graph', 'figure'),
    (
        [Input('forecast-type-dropdown', 'value'),
         Input('site-dropdown', 'value'),
         Input('forecast-method-dropdown', 'value')]
        + ([Input('season-dropdown', 'value')] if ENABLE_SEASON_SPECIFIC_TRAINER else [])
    )
)
def update_graph(forecast_type, selected_site, forecast_method, *season):
    print("Updating analysis graph...")
    if ENABLE_SEASON_SPECIFIC_TRAINER:
        selected_season = season[0] if season else 'annual'
    else:
        selected_season = 'annual'
    
    if forecast_method == 'lr' and not ENABLE_LINEAR_LOGISTIC:
        print("LR forecast requested but disabled. Defaulting to ML.")
        forecast_method = 'ml'
    
    if forecast_method == 'ml':
        pred = predictions_ml[selected_season]
    else:
        pred = predictions_lr[selected_season] if predictions_lr is not None else predictions_ml[selected_season]
    
    if forecast_type == 'DA_Level':
        df_plot = pred['DA_Level']['test_df'].copy()
        site_stats = pred['DA_Level']['site_stats']
        overall_r2 = pred['DA_Level']['overall_r2']
        overall_rmse = pred['DA_Level']['overall_rmse']
        y_axis_title = 'Domoic Acid Levels'
        df_plot_melted = pd.melt(
            df_plot,
            id_vars=['Date', 'Site'],
            value_vars=['DA_Levels', 'Predicted_DA_Levels'],
            var_name='Metric', value_name='Value'
        )
        if selected_site is None or selected_site == 'All Sites':
            performance_text = f"Overall R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f}"
        else:
            if selected_site in site_stats.index:
                site_r2 = site_stats.loc[selected_site, 'r2']
                site_rmse = site_stats.loc[selected_site, 'rmse']
                performance_text = f"R² = {site_r2:.2f}, RMSE = {site_rmse:.2f}"
            else:
                performance_text = "No data for selected site."
    else:
        df_plot = pred['DA_Category']['test_df'].copy()
        site_stats = pred['DA_Category']['site_stats']
        overall_accuracy = pred['DA_Category']['overall_accuracy']
        y_axis_title = 'Domoic Acid Category'
        df_plot_melted = pd.melt(
            df_plot,
            id_vars=['Date', 'Site'],
            value_vars=['DA_Category', 'Predicted_DA_Category'],
            var_name='Metric', value_name='Value'
        )
        if selected_site is None or selected_site == 'All Sites':
            performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
        else:
            if selected_site in site_stats.index:
                site_accuracy = site_stats.loc[selected_site]
                performance_text = f"Accuracy = {site_accuracy:.2f}"
            else:
                performance_text = "No data for selected site."
    
    if selected_site and selected_site != 'All Sites':
        df_plot_melted = df_plot_melted[df_plot_melted['Site'] == selected_site]
    df_plot_melted = df_plot_melted.sort_values('Date')
    
    if selected_site == 'All Sites' or not selected_site:
        fig = px.line(
            df_plot_melted,
            x='Date',
            y='Value',
            color='Site',
            line_dash='Metric',
            title=f"{y_axis_title} Forecast - All Sites"
        )
    else:
        fig = px.line(
            df_plot_melted,
            x='Date',
            y='Value',
            color='Metric',
            title=f"{y_axis_title} Forecast - {selected_site}"
        )
    fig.update_layout(
        yaxis_title=y_axis_title,
        xaxis_title='Date',
        annotations=[{
            'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': -0.2,
            'xanchor': 'center', 'yanchor': 'top',
            'text': performance_text,
            'showarrow': False
        }]
    )
    print("Analysis graph updated.")
    return fig

# ---------------------------------------------------------------------
# Callback to update the Random Anchor Forecast layout.
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    @app.callback(
        Output('random-anchor-container', 'children'),
        [Input('forecast-method-dropdown', 'value')] + ([Input('season-dropdown', 'value')] if ENABLE_SEASON_SPECIFIC_TRAINER else [])
    )
    def update_random_anchor_layout(forecast_method, *season):
        print("Updating Random Anchor Forecast layout...")
        if ENABLE_SEASON_SPECIFIC_TRAINER:
            selected_season = season[0] if season else 'annual'
        else:
            selected_season = 'annual'
        
        if forecast_method == 'lr' and not ENABLE_LINEAR_LOGISTIC:
            print("LR forecast requested but disabled. Defaulting to ML.")
            forecast_method = 'ml'
            
        if forecast_method == 'ml':
            df_results, figs_random_site, rmse, acc = random_anchors_ml[selected_season]
        else:
            df_results, figs_random_site, rmse, acc = random_anchors_lr[selected_season]
        graphs = [dcc.Graph(figure=fig) for site, fig in figs_random_site.items()]
        metrics = html.Div([
            html.H4("Overall Performance on These Forecasts"),
            html.Ul([
                html.Li(f"RMSE (DA Levels): {rmse:.3f}") if rmse is not None else html.Li("No RMSE (no data)"),
                html.Li(f"Accuracy (DA Category): {acc:.3f}") if acc is not None else html.Li("No Accuracy (no data)")
            ])
        ], style={'marginTop': 20})
        print("Random Anchor Forecast layout updated.")
        return html.Div(graphs + [metrics])

# ---------------------------------------------------------
# 8) Run the Dash App
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Starting Dash app...")
    app.run_server(debug=True, port=8071)