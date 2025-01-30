import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# -------------------------------------------
# Data Loading and Feature Engineering
# -------------------------------------------
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # 1) Spatial Clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # 2) Seasonal Features
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')

    data['Year'] = data['Date'].dt.year

    # 3) Lag Features
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    # 4) Interaction: cluster * cyclical
    cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
        data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']

    # 5) Categorize DA_Levels
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

# -------------------------------------------
# Single forecast for "next date" after anchor
# -------------------------------------------
def forecast_next_date(df, anchor_date, site):
    """
    1. Train on data up to and including anchor_date (no future leakage).
    2. Forecast the next available date strictly > anchor_date for that site.
    3. Return dictionary of actual/predicted for the 'next date' row (if it exists).
    """

    # Filter data for just this site
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)

    # Next available date
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        # No future date for this site
        return None
    next_date = df_future['Date'].iloc[0]

    # Train on everything up to (and including) anchor_date
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        # Not enough data to train
        return None

    # We'll predict on that 'next date' row
    df_test = df_site[df_site['Date'] == next_date].copy()

    # If there's no row exactly for next_date, we can't do a direct comparison.
    if df_test.empty:
        return None

    # ---- Train Regression (DA_Levels) ----
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

    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X_train_reg_processed, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg_processed)

    # ---- Train Classification (DA_Category) ----
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

    cls_model = RandomForestClassifier(random_state=42)
    cls_model.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,  # the chosen anchor
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(y_test_reg.iloc[0]) if len(y_test_reg) else None,
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(y_test_cls.iloc[0]) if len(y_test_cls) else None
    }

# -------------------------------------------
# Dash App
# -------------------------------------------
app = dash.Dash(__name__)

file_path = 'final_output.csv'  # Update with your CSV
df = load_and_prepare_data(file_path)

# 1) Filter data to "year >= 2010"
df_after_2010 = df[df['Date'].dt.year >= 2010]
df_after_2010.sort_values(['Site', 'Date'], inplace=True)

# 2) Create unique (Site, Date) pairs
pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()

# 3) Randomly pick up to 200 anchor dates
if len(pairs_after_2010) > 200:
    df_random_200 = pairs_after_2010.sample(n=200, random_state=42)
else:
    df_random_200 = pairs_after_2010.copy()  # if fewer than 200 available

# 4) For each anchor, forecast the *next* date
results_list = []
for _, row in df_random_200.iterrows():
    site_ = row['Site']
    anchor_date_ = row['Date']
    result = forecast_next_date(df, anchor_date_, site_)
    if result is not None:
        results_list.append(result)

df_results_200 = pd.DataFrame(results_list)

# Create a Plotly scatter: actual vs. predicted levels
# We'll color by site, and hover with AnchorDate, NextDate
if not df_results_200.empty:
    fig_200_scatter = px.scatter(
        df_results_200,
        x='Actual_DA_Levels',
        y='Predicted_DA_Levels',
        color='Site',
        hover_data=['AnchorDate', 'NextDate'],
        title="Random 200 (Anchor-Date) Forecasts for Next Available Date (>= 2010)"
    )
else:
    # If no data, create an empty figure
    fig_200_scatter = px.scatter(
        title="No Data Available for Random 200 Selections"
    )

app.layout = html.Div([
    html.H1("Forecast Next Available Date (After 2010)"),
    html.Div([
        html.P("We sampled 200 random anchor dates (Site, Date) from data after 2010."),
        html.P("For each anchor, we trained on data up to that date, then forecasted the next date.")
    ]),
    dcc.Graph(figure=fig_200_scatter),

    html.Div(id='explanation', style={'whiteSpace': 'pre-wrap', 'marginTop': 20}),
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8057)
