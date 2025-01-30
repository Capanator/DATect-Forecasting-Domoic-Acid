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

# ---------------------------------------------------------
# 1) Data Loading & Feature Engineering
# ---------------------------------------------------------
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # Spatial Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # Seasonal (sin/cos)
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # Month (one-hot)
    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')

    # Year
    data['Year'] = data['Date'].dt.year

    # Lag Features
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    # Interaction: cluster * cyclical
    cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
        data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']

    # DA_Category
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
# 2) Original Analysis: Train & Predict (Full Split)
# ---------------------------------------------------------
def train_and_predict(data):
    """
    Splits data into train/test sets, applies
    imputation + scaling to the training set,
    and uses RandomForest for both regression and classification.
    Returns dictionaries for each task with:
      - test DataFrame containing predictions
      - site-level metrics
      - overall metrics
    """
    # 2A) Regression
    data_reg = data.drop(['DA_Category'], axis=1)
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

    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train_reg_processed, y_train_reg)
    y_pred_reg = rf_regressor.predict(X_test_reg_processed)

    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg

    overall_r2_reg = r2_score(y_test_reg, y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    site_stats_reg = test_set_reg.groupby('Site').apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )

    # 2B) Classification
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

    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = rf_classifier.predict(X_test_cls_processed)

    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls

    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    site_stats_cls = test_set_cls.groupby('Site').apply(
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
# 3) Time-Based Forecast Function
#    Train on data <= anchor date, predict next date > anchor
# ---------------------------------------------------------
def forecast_next_date(df, anchor_date, site):
    """
    1. Train on data up to and including anchor_date (no future leakage).
    2. Forecast the next available date strictly > anchor_date for that site.
    3. Return dictionary of actual/predicted for that 'next date' row (if it exists).
    """
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)

    # Next available date
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        # No future date for this site
        return None
    next_date = df_future['Date'].iloc[0]

    # Training set: everything up to anchor_date
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        # No training data
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

    reg_model = RandomForestRegressor(random_state=42)
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

    cls_model = RandomForestClassifier(random_state=42)
    cls_model.fit(X_train_cls_processed, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    # Return single-row results
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
# Build the Dash App
# ---------------------------------------------------------
app = dash.Dash(__name__)

file_path = 'final_output.csv'  # your CSV
raw_data = load_and_prepare_data(file_path)

# 1) Run your original "train_and_predict" for the overall analysis
predictions = train_and_predict(raw_data)

# ----------------------------------------------------------
# 2) Prepare Data for Random 200 Approach (Dates >= 2010)
# ----------------------------------------------------------
df_after_2010 = raw_data[raw_data['Date'].dt.year >= 2010].copy()
df_after_2010.sort_values(['Site', 'Date'], inplace=True)

pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()
# Take 200 random or all if <200
if len(pairs_after_2010) > 200:
    df_random_200 = pairs_after_2010.sample(n=200, random_state=42)
else:
    df_random_200 = pairs_after_2010

results_list = []
for _, row in df_random_200.iterrows():
    site_ = row['Site']
    anchor_date_ = row['Date']
    result = forecast_next_date(raw_data, anchor_date_, site_)
    if result is not None:
        results_list.append(result)

df_results_200 = pd.DataFrame(results_list)

# Overall metrics on these 200 forecasts
if not df_results_200.empty:
    # For regression (levels)
    rmse_200 = np.sqrt(mean_squared_error(
        df_results_200['Actual_DA_Levels'], df_results_200['Predicted_DA_Levels']
    ))
    # For classification (category)
    acc_200 = (df_results_200['Actual_DA_Category'] == df_results_200['Predicted_DA_Category']).mean()
else:
    rmse_200 = None
    acc_200 = None

# Create a line chart with time on x-axis, DA level on y-axis, 2 lines: actual & predicted
if not df_results_200.empty:
    df_line = df_results_200.copy()
    df_line = df_line.sort_values('NextDate')
    df_plot_melt = df_line.melt(
        id_vars=['Site','NextDate'],
        value_vars=['Actual_DA_Levels','Predicted_DA_Levels'],
        var_name='Type', value_name='DA_Level'
    )
    fig_random_line = px.line(
        df_plot_melt,
        x='NextDate',
        y='DA_Level',
        color='Type',
        line_group='Site',
        hover_data=['Site'],
        title="Random 200 Next-Date Forecast (After 2010)"
    )
    fig_random_line.update_layout(
        xaxis_title='Next Date',
        yaxis_title='DA Level'
    )
else:
    fig_random_line = px.line(title="No valid data for random 200 approach")

# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------

# ---------- Tab 1: Original Analysis -----------
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
        options=[{'label': 'All Sites', 'value': 'All Sites'}] +
                [{'label': site, 'value': site} for site in raw_data['Site'].unique()],
        value='All Sites',
        style={'width': '50%'}
    ),
    dcc.Graph(id='analysis-graph')
])

# ---------- Tab 2: Forecast by Date & Site (Strictly Partial Training) ----------

# 2A) Build disabled_days for DatePickerSingle
valid_dates = sorted(raw_data['Date'].unique())
all_dates_range = pd.date_range(valid_dates[0], valid_dates[-1], freq='D')
disabled_days = [d for d in all_dates_range if d not in valid_dates]

forecast_layout = html.Div([
    html.H3("Forecast by Specific Date & Site (Partial Training up to Date)"),

    html.Label("Choose a Site:"),
    dcc.Dropdown(
        id='site-dropdown-forecast',
        options=[{'label': s, 'value': s} for s in raw_data['Site'].unique()],
        value=raw_data['Site'].unique()[0],
        style={'width': '50%'}
    ),

    html.Label("Pick an Anchor Date:"),
    dcc.DatePickerSingle(
        id='forecast-date-picker',
        min_date_allowed=valid_dates[0],
        max_date_allowed=valid_dates[-1],
        initial_visible_month=valid_dates[0],
        date=valid_dates[0],
        disabled_days=disabled_days  # gray out invalid dates
    ),

    html.Div(id='forecast-output-partial', style={'whiteSpace': 'pre-wrap', 'marginTop': 20})
])

# ---------- Tab 3: Random 200 Next-Date ----------

random_200_layout = html.Div([
    html.H3("Random 200 Anchor Dates (Post-2010) Forecast -> Next Date"),
    dcc.Graph(figure=fig_random_line),
    html.Div([
        html.H4("Overall Performance on These 200 Forecasts"),
        html.Ul([
            html.Li(f"RMSE (DA Levels): {rmse_200:.3f}") if rmse_200 is not None else html.Li("No RMSE (no data)"),
            html.Li(f"Accuracy (DA Category): {acc_200:.3f}") if acc_200 is not None else html.Li("No Accuracy (no data)")
        ])
    ], style={'marginTop': 20})
])

# ---------- Combine Tabs ----------
app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Analysis', children=[analysis_layout]),
        dcc.Tab(label='Forecast by Date & Site', children=[forecast_layout]),
        dcc.Tab(label='Random 200 Next-Date Forecast', children=[random_200_layout]),
    ])
])

# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

# --- Tab 1 Callback: Original Analysis Graph ---
@app.callback(
    Output('analysis-graph', 'figure'),
    [Input('forecast-type-dropdown', 'value'),
     Input('site-dropdown', 'value')]
)
def update_graph(selected_forecast_type, selected_site):
    if selected_forecast_type == 'DA_Level':
        df_plot = predictions['DA_Level']['test_df'].copy()
        site_stats = predictions['DA_Level']['site_stats']
        overall_r2 = predictions['DA_Level']['overall_r2']
        overall_rmse = predictions['DA_Level']['overall_rmse']
        y_axis_title = 'Domoic Acid Levels'
        y_columns = ['DA_Levels', 'Predicted_DA_Levels']

        if selected_site == 'All Sites':
            performance_text = f"Overall R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f}"
        elif selected_site in site_stats.index:
            site_r2 = site_stats.loc[selected_site, 'r2']
            site_rmse = site_stats.loc[selected_site, 'rmse']
            performance_text = f"R² = {site_r2:.2f}, RMSE = {site_rmse:.2f}"
        else:
            performance_text = "No data for selected site."

    else:  # DA_Category
        df_plot = predictions['DA_Category']['test_df'].copy()
        site_stats = predictions['DA_Category']['site_stats']
        overall_accuracy = predictions['DA_Category']['overall_accuracy']
        y_axis_title = 'Domoic Acid Category'
        y_columns = ['DA_Category', 'Predicted_DA_Category']

        if selected_site == 'All Sites':
            performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
        elif selected_site in site_stats.index:
            site_accuracy = site_stats.loc[selected_site]
            performance_text = f"Accuracy = {site_accuracy:.2f}"
        else:
            performance_text = "No data for selected site."

    if selected_site != 'All Sites':
        df_plot = df_plot[df_plot['Site'] == selected_site]

    df_plot = df_plot.sort_values('Date')

    fig = px.line(
        df_plot,
        x='Date',
        y=y_columns,
        color='Site' if selected_site == 'All Sites' else None,
        title=f"{y_axis_title} Forecast - {selected_site}"
    )
    fig.update_layout(
        yaxis_title=y_axis_title,
        xaxis_title='Date',
        annotations=[
            dict(
                xref='paper', yref='paper', x=0.5, y=-0.2,
                xanchor='center', yanchor='top',
                text=performance_text,
                showarrow=False
            )
        ]
    )
    return fig

# --- Tab 2 Callback: Partial Training up to Date & Forecast Next Date ---
@app.callback(
    Output('forecast-output-partial', 'children'),
    [Input('forecast-date-picker', 'date'),
     Input('site-dropdown-forecast', 'value')]
)
def partial_forecast_callback(anchor_date_str, site):
    if not anchor_date_str or not site:
        return "Please select a site and a valid date."

    anchor_date = pd.to_datetime(anchor_date_str)

    # Use the same function as random 200 approach
    result = forecast_next_date(raw_data, anchor_date, site)

    if result is None:
        return (
            f"No forecast possible for Site={site} after {anchor_date.date()}.\n"
            "Possibly no future date or no training data up to that date."
        )

    lines = [
        f"Selected Anchor Date (training cut-off): {result['AnchorDate'].date()}",
        f"Next Date (forecast target): {result['NextDate'].date()}",
        "",
        f"Predicted DA Level: {result['Predicted_DA_Levels']:.2f}",
        f"Actual   DA Level: {result['Actual_DA_Levels']:.2f} (error = {abs(result['Predicted_DA_Levels'] - result['Actual_DA_Levels']):.2f})",
        "",
        f"Predicted DA Category: {result['Predicted_DA_Category']}",
        f"Actual   DA Category: {result['Actual_DA_Category']} "
        f"({'MATCH' if result['Predicted_DA_Category'] == result['Actual_DA_Category'] else 'MISMATCH'})"
    ]

    return "\n".join(lines)

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8065)
