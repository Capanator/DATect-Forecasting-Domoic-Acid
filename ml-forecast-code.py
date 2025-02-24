import pandas as pd
import numpy as np

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
# 1) Data Loading & Feature Engineering
# ---------------------------------------------------------
def load_and_prepare_data(file_path, season=None):
    # Change from CSV to Parquet
    data = pd.read_parquet(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data for seasonal analyses if requested.
    if season == 'spring':
        data = data[data['Date'].dt.month.isin([3, 4, 5, 6, 7])]
    elif season == 'fall':
        data = data[data['Date'].dt.month.isin([9, 10, 11, 12])]
    
    data = data.sort_values(['Site', 'Date']).copy()

    # --- Spatial Clustering ---
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

# ---------------------------------------------------------
# 2a) Train & Predict Function (Machine Learning - RF with GridSearchCV & TimeSeriesSplit)
# ---------------------------------------------------------
def train_and_predict(data):
    # --- Regression Setup ---
    data_reg = data.drop(['DA_Category'], axis=1)
    # Use a hold-out test set (20% of data)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.2, random_state=42)

    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    X_train_reg = train_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = train_set_reg['DA_Levels']
    X_test_reg = test_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = test_set_reg['DA_Levels']

    numeric_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    numeric_transformer_reg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor_reg = ColumnTransformer(
        transformers=[('num', numeric_transformer_reg, numeric_cols_reg)],
        remainder='passthrough'
    )
    X_train_reg_processed = preprocessor_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = preprocessor_reg.transform(X_test_reg)

    # Set up time series cross-validation and grid search for regression
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid_reg = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search_reg = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid_reg,
        cv=tscv,
        scoring='r2',
        n_jobs=-1
    )
    grid_search_reg.fit(X_train_reg_processed, y_train_reg)
    best_reg = grid_search_reg.best_estimator_
    y_pred_reg = best_reg.predict(X_test_reg_processed)

    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg

    overall_r2_reg = r2_score(y_test_reg, y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    site_stats_reg = test_set_reg[['DA_Levels', 'Predicted_DA_Levels']].groupby(test_set_reg['Site']).apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )

    # --- Classification Setup ---
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.2, random_state=42)

    drop_cols_cls = ['DA_Category', 'Date', 'Site']
    X_train_cls = train_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = train_set_cls['DA_Category']
    X_test_cls = test_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_test_cls = test_set_cls['DA_Category']

    numeric_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    numeric_transformer_cls = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor_cls = ColumnTransformer(
        transformers=[('num', numeric_transformer_cls, numeric_cols_cls)],
        remainder='passthrough'
    )
    X_train_cls_processed = preprocessor_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = preprocessor_cls.transform(X_test_cls)

    # Grid search for classification using TimeSeriesSplit
    grid_search_cls = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            'n_estimators': [200, 300],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_cls.fit(X_train_cls_processed, y_train_cls)
    best_cls = grid_search_cls.best_estimator_
    y_pred_cls = best_cls.predict(X_test_cls_processed)

    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls

    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    site_stats_cls = test_set_cls[['DA_Category', 'Predicted_DA_Category']].groupby(test_set_cls['Site']).apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )

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
# 2b) Train & Predict Function (Linear & Logistic Regression)
# ---------------------------------------------------------
def train_and_predict_lr(data):
    # --- Regression Setup ---
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.25, random_state=42)

    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    X_train_reg = train_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = train_set_reg['DA_Levels']
    X_test_reg = test_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = test_set_reg['DA_Levels']

    numeric_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    numeric_transformer_reg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor_reg = ColumnTransformer(
        transformers=[('num', numeric_transformer_reg, numeric_cols_reg)],
        remainder='passthrough'
    )
    X_train_reg_processed = preprocessor_reg.fit_transform(X_train_reg)
    X_test_reg_processed = preprocessor_reg.transform(X_test_reg)

    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train_reg_processed, y_train_reg)
    y_pred_reg = lin_regressor.predict(X_test_reg_processed)

    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg

    overall_r2_reg = r2_score(y_test_reg, y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    site_stats_reg = test_set_reg[['DA_Levels', 'Predicted_DA_Levels']].groupby(test_set_reg['Site']).apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )

    # --- Classification Setup ---
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.25, random_state=42)

    drop_cols_cls = ['DA_Category', 'Date', 'Site']
    X_train_cls = train_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = train_set_cls['DA_Category']
    X_test_cls = test_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_test_cls = test_set_cls['DA_Category']

    numeric_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    numeric_transformer_cls = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor_cls = ColumnTransformer(
        transformers=[('num', numeric_transformer_cls, numeric_cols_cls)],
        remainder='passthrough'
    )
    X_train_cls_processed = preprocessor_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = preprocessor_cls.transform(X_test_cls)

    log_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_classifier.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = log_classifier.predict(X_test_cls_processed)

    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls

    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    site_stats_cls = test_set_cls[['DA_Category', 'Predicted_DA_Category']].groupby(test_set_cls['Site']).apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )

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
# 3a) Time-Based Forecast Function for Random Anchor Dates (ML)
# ---------------------------------------------------------
def forecast_next_date(df, anchor_date, site):
    df_site = df[df['Site'] == site].copy()
    df_site = df_site.sort_values('Date').copy()

    # Identify the next date strictly greater than anchor_date
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None
    next_date = df_future['Date'].iloc[0]

    # Training set: data up to (and including) anchor_date
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        return None

    # Test set: exactly the next date
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        return None

    # --- Train Regression (DA_Levels) ---
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    X_train_reg = df_train.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = df_train['DA_Levels']
    X_test_reg = df_test.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = df_test['DA_Levels']

    num_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    reg_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    col_trans_reg = ColumnTransformer(
        transformers=[('num', reg_preproc, num_cols_reg)],
        remainder='passthrough'
    )
    X_train_reg_processed = col_trans_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = col_trans_reg.transform(X_test_reg)

    # For forecasting a single step, we use a simple model (grid search is skipped)
    reg_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None)
    reg_model.fit(X_train_reg_processed, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg_processed)

    # --- Train Classification (DA_Category) ---
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    X_train_cls = df_train.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = df_train['DA_Category']
    X_test_cls = df_test.drop(columns=drop_cols_cls, errors='ignore')
    y_test_cls = df_test['DA_Category']

    num_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    cls_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    col_trans_cls = ColumnTransformer(
        transformers=[('num', cls_preproc, num_cols_cls)],
        remainder='passthrough'
    )
    X_train_cls_processed = col_trans_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = col_trans_cls.transform(X_test_cls)

    cls_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
    cls_model.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(y_test_reg.iloc[0]) if len(y_test_reg) else None,
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(y_test_cls.iloc[0]) if len(y_test_cls) else None
    }

# ---------------------------------------------------------
# 3b) Time-Based Forecast Function for Random Anchor Dates (LR)
# ---------------------------------------------------------
def forecast_next_date_lr(df, anchor_date, site):
    df_site = df[df['Site'] == site].copy()
    df_site = df_site.sort_values('Date').copy()

    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None
    next_date = df_future['Date'].iloc[0]

    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        return None

    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        return None

    # --- Regression using LinearRegression ---
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    X_train_reg = df_train.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = df_train['DA_Levels']
    X_test_reg = df_test.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = df_test['DA_Levels']

    num_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    reg_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    col_trans_reg = ColumnTransformer(
        transformers=[('num', reg_preproc, num_cols_reg)],
        remainder='passthrough'
    )
    X_train_reg_processed = col_trans_reg.fit_transform(X_train_reg)
    X_test_reg_processed  = col_trans_reg.transform(X_test_reg)

    lin_model = LinearRegression()
    lin_model.fit(X_train_reg_processed, y_train_reg)
    y_pred_reg = lin_model.predict(X_test_reg_processed)

    # --- Classification using LogisticRegression ---
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    X_train_cls = df_train.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = df_train['DA_Category']
    X_test_cls = df_test.drop(columns=drop_cols_cls, errors='ignore')
    y_test_cls = df_test['DA_Category']

    num_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    cls_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    col_trans_cls = ColumnTransformer(
        transformers=[('num', cls_preproc, num_cols_cls)],
        remainder='passthrough'
    )
    X_train_cls_processed = col_trans_cls.fit_transform(X_train_cls)
    X_test_cls_processed  = col_trans_cls.transform(X_test_cls)

    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_model.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = log_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(y_test_reg.iloc[0]) if len(y_test_reg) else None,
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(y_test_cls.iloc[0]) if len(y_test_cls) else None
    }

# ---------------------------------------------------------
# 4) Modified Random Anchor Forecast Helper Function
# ---------------------------------------------------------
def get_random_anchor_forecasts(data, forecast_func):
    NUM_RANDOM_ANCHORS = 2
    df_after_2010 = data[data['Date'].dt.year >= 2010].copy()
    df_after_2010 = df_after_2010.sort_values(['Site', 'Date']).copy()

    pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()

    # Group by site and sample up to NUM_RANDOM_ANCHORS for each site
    df_random_anchors = pd.concat([
        group.sample(n=min(NUM_RANDOM_ANCHORS, len(group)), random_state=42)
        for _, group in pairs_after_2010.groupby('Site')
    ]).reset_index(drop=True)

    results_list = []
    for _, row in df_random_anchors.iterrows():
        site_ = row['Site']
        anchor_date_ = row['Date']
        result = forecast_func(data, anchor_date_, site_)
        if result is not None:
            results_list.append(result)

    df_results_anchors = pd.DataFrame(results_list)

    if not df_results_anchors.empty:
        rmse_anchors = np.sqrt(mean_squared_error(
            df_results_anchors['Actual_DA_Levels'],
            df_results_anchors['Predicted_DA_Levels']
        ))
        acc_anchors = (
            df_results_anchors['Actual_DA_Category'] ==
            df_results_anchors['Predicted_DA_Category']
        ).mean()
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
    return df_results_anchors, figs_random_site, rmse_anchors, acc_anchors

# ---------------------------------------------------------
# 5) Prepare Data, Predictions, and Random Anchors for Each Season
# ---------------------------------------------------------
# Update the file path to a Parquet file
file_path = 'final_output.parquet'

# Create datasets for annual, spring, and fall.
raw_data_annual = load_and_prepare_data(file_path, season=None)
raw_data_spring = load_and_prepare_data(file_path, season='spring')
raw_data_fall   = load_and_prepare_data(file_path, season='fall')

# Store raw data in a dict (used later for dropdown update)
raw_data_dict = {
    'annual': raw_data_annual,
    'spring': raw_data_spring,
    'fall': raw_data_fall
}

# Generate predictions for each season.
predictions_ml = {
    'annual': train_and_predict(raw_data_annual),
    'spring': train_and_predict(raw_data_spring),
    'fall': train_and_predict(raw_data_fall)
}

predictions_lr = {
    'annual': train_and_predict_lr(raw_data_annual),
    'spring': train_and_predict_lr(raw_data_spring),
    'fall': train_and_predict_lr(raw_data_fall)
}

# Generate random anchor forecasts for each season.
random_anchors_ml = {
    'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date),
    'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date),
    'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date)
}

random_anchors_lr = {
    'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date_lr),
    'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date_lr),
    'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date_lr)
}

# ---------------------------------------------------------
# 6) Build the Dash App Layout & Callbacks
# ---------------------------------------------------------
app = dash.Dash(__name__)

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

# Random Anchor Forecast layout – a container that will update based on season.
random_anchors_layout = html.Div([
    html.H3("Random Anchor Dates Forecast -> Next Date by Site"),
    html.Div(id='random-anchor-container')
])

# Global layout including season and forecast method selectors at the top.
app.layout = html.Div([
    html.H1("Domoic Acid Forecast Dashboard"),
    html.Div([
        html.Label("Select Season"),
        dcc.Dropdown(
            id='season-dropdown',
            options=[
                {'label': 'Annual', 'value': 'annual'},
                {'label': 'Spring Bloom', 'value': 'spring'},
                {'label': 'Fall Bloom', 'value': 'fall'}
            ],
            value='annual',
            style={'width': '30%'}
        ),
        html.Label("Select Forecast Method"),
        dcc.Dropdown(
            id='forecast-method-dropdown',
            options=[
                {'label': 'Machine Learning Forecasts', 'value': 'ml'},
                {'label': 'Linear/Logistic Regression Forecasts', 'value': 'lr'}
            ],
            value='ml',
            style={'width': '30%', 'marginLeft': '20px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Analysis', children=[analysis_layout]),
        dcc.Tab(label='Random Anchor Forecast', children=[random_anchors_layout])
    ])
])

# Callback to update site dropdown options based on season.
@app.callback(
    Output('site-dropdown', 'options'),
    Input('season-dropdown', 'value')
)
def update_site_dropdown(selected_season):
    data = raw_data_dict[selected_season]
    options = [{'label': 'All Sites', 'value': 'All Sites'}] + \
              [{'label': site, 'value': site} for site in data['Site'].unique()]
    return options

# Callback for updating the Analysis graph.
@app.callback(
    Output('analysis-graph', 'figure'),
    [
        Input('season-dropdown', 'value'),
        Input('forecast-type-dropdown', 'value'),
        Input('site-dropdown', 'value'),
        Input('forecast-method-dropdown', 'value')
    ]
)
def update_graph(selected_season, selected_forecast_type, selected_site, forecast_method):
    # Choose the predictions based on forecast method.
    if forecast_method == 'ml':
        pred = predictions_ml[selected_season]
    else:
        pred = predictions_lr[selected_season]

    if selected_forecast_type == 'DA_Level':
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
    return fig

# Callback for updating the Random Anchor Forecast layout based on season and forecast method.
@app.callback(
    Output('random-anchor-container', 'children'),
    [
        Input('season-dropdown', 'value'),
        Input('forecast-method-dropdown', 'value')
    ]
)
def update_random_anchor_layout(selected_season, forecast_method):
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
    return html.Div(graphs + [metrics])

# ---------------------------------------------------------
# 7) Run the Dash App
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
