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
# Model Training and Prediction
# -------------------------------------------
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
    # -----------------------------
    # 1) DA_Levels Regression
    # -----------------------------
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.2, random_state=42)

    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    X_train_reg = train_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = train_set_reg['DA_Levels']

    X_test_reg = test_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = test_set_reg['DA_Levels']

    numeric_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    numeric_transformer_reg = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor_reg = ColumnTransformer(
        transformers=[('num', numeric_transformer_reg, numeric_cols_reg)],
        remainder='passthrough'
    )

    X_train_reg_processed = preprocessor_reg.fit_transform(X_train_reg)
    X_test_reg_processed = preprocessor_reg.transform(X_test_reg)

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

    # -----------------------------
    # 2) DA_Category Classification
    # -----------------------------
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.2, random_state=42)

    drop_cols_cls = ['DA_Category', 'Date', 'Site']
    X_train_cls = train_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = train_set_cls['DA_Category']

    X_test_cls = test_set_cls.drop(columns=drop_cols_cls, errors='ignore')
    y_test_cls = test_set_cls['DA_Category']

    numeric_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    numeric_transformer_cls = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor_cls = ColumnTransformer(
        transformers=[('num', numeric_transformer_cls, numeric_cols_cls)],
        remainder='passthrough'
    )

    X_train_cls_processed = preprocessor_cls.fit_transform(X_train_cls)
    X_test_cls_processed = preprocessor_cls.transform(X_test_cls)

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

# -------------------------------------------
# Dash App
# -------------------------------------------
app = dash.Dash(__name__)

file_path = 'final_output.csv'  # Update as needed
raw_data = load_and_prepare_data(file_path)
predictions = train_and_predict(raw_data)

# 1. Pull out test sets for each task
test_reg = predictions["DA_Level"]["test_df"]
test_cls = predictions["DA_Category"]["test_df"]

# 2. Merge them so we can reference predictions together
#    Some rows might not overlap if the random splits differ.
#    We'll do an outer join on [Site, Date]:
test_merged = pd.merge(
    test_reg[['Site', 'Date', 'DA_Levels', 'Predicted_DA_Levels']],
    test_cls[['Site', 'Date', 'DA_Category', 'Predicted_DA_Category']],
    on=['Site', 'Date'],
    how='outer'
).sort_values(['Site', 'Date'])

# 3. Prepare a 200-sample comparison
test_merged_no_na = test_merged.dropna(
    subset=['DA_Levels', 'Predicted_DA_Levels', 'DA_Category', 'Predicted_DA_Category']
)

if len(test_merged_no_na) >= 200:
    random_200 = test_merged_no_na.sample(n=200, random_state=42)
else:
    # If data is smaller, just use all
    random_200 = test_merged_no_na.copy()

rmse_200 = np.sqrt(mean_squared_error(random_200["DA_Levels"], random_200["Predicted_DA_Levels"])) if not random_200.empty else None
acc_200 = accuracy_score(random_200["DA_Category"], random_200["Predicted_DA_Category"]) if not random_200.empty else None

random_sample_text = (
    "Analysis on 200 (or fewer if limited data) randomly selected date-site combinations:\n"
)
if rmse_200 is not None and acc_200 is not None:
    random_sample_text += f"- RMSE (Domoic Acid Level) = {rmse_200:.2f}\n"
    random_sample_text += f"- Accuracy (Domoic Acid Category) = {acc_200:.2f}\n"
else:
    random_sample_text += "Not enough data to sample 200 rows."

# -------------------------------------------
# Tab 1 Layout: Overall Analysis
# -------------------------------------------
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

# -------------------------------------------
# Tab 2 Layout: Forecast by Date & Site
# -------------------------------------------
forecast_layout = html.Div([
    html.H3("Forecast by Specific Date & Site"),

    html.Div([
        html.Label("Choose a Site:"),
        dcc.Dropdown(
            id='site-dropdown-forecast',
            options=[{'label': site, 'value': site} for site in raw_data['Site'].unique()],
            value=raw_data['Site'].unique()[0],
            style={'width': '50%'}
        )
    ], style={'marginBottom': 20}),

    html.Div([
        html.Label("Pick a Date:"),
        dcc.DatePickerSingle(
            id='forecast-date-picker',
            min_date_allowed=raw_data['Date'].min(),
            max_date_allowed=raw_data['Date'].max(),
            initial_visible_month=raw_data['Date'].min(),
            date=raw_data['Date'].min()
        )
    ], style={'marginBottom': 20}),

    html.Div(id='forecast-output', style={'whiteSpace': 'pre-wrap'}),

    html.Hr(),
    html.Div([
        html.H4("Random 200-Sample Analysis"),
        html.Div(random_sample_text, style={'whiteSpace': 'pre-wrap', 'border': '1px solid #ccc', 'padding': '10px'})
    ])
])

# -------------------------------------------
# Combine Tabs into Single Layout
# -------------------------------------------
app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Analysis', children=[analysis_layout]),
        dcc.Tab(label='Forecast by Date & Site', children=[forecast_layout]),
    ])
])

# -------------------------------------------
# Callbacks
# -------------------------------------------

# --- Callback for Tab 1 (Analysis Graph) ---
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

    else:
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

# --- Callback for Tab 2 (Date & Site Forecast) ---
@app.callback(
    Output('forecast-output', 'children'),
    [Input('forecast-date-picker', 'date'),
     Input('site-dropdown-forecast', 'value')]
)
def update_forecast(selected_date, selected_site):
    if not selected_date or not selected_site:
        return "Please select a date and site."

    selected_date_obj = pd.to_datetime(selected_date)

    # Filter by selected site
    df_site = test_merged[test_merged['Site'] == selected_site].copy()

    # Find the *next* date in the dataset after the chosen date
    df_future = df_site[df_site['Date'] > selected_date_obj].sort_values('Date')
    if df_future.empty:
        return "No future forecast available for this Site after the selected date."

    # Take the first row as the "next available" forecast
    row = df_future.iloc[0]
    next_date_str = row['Date'].strftime('%Y-%m-%d')

    # Extract predicted values
    pred_level = row.get('Predicted_DA_Levels', None)
    pred_cat   = row.get('Predicted_DA_Category', None)

    # Extract actual values (if present)
    actual_level = row.get('DA_Levels', None)
    actual_cat   = row.get('DA_Category', None)

    # Build text
    lines = [
        f"Next available forecast date for {selected_site} after {selected_date_obj.date()}: {next_date_str}",
        f"Predicted DA Level: {pred_level:.2f}" if pred_level is not None else "No predicted level.",
        f"Predicted DA Category: {pred_cat}" if pred_cat is not None else "No predicted category."
    ]

    # If actual data is available, compute difference and match
    if pd.notnull(actual_level):
        diff = abs(pred_level - actual_level)
        lines.append(f"Actual DA Level: {actual_level:.2f} (error = {diff:.2f})")

    if pd.notnull(actual_cat):
        match_text = "MATCH" if (pred_cat == actual_cat) else "MISMATCH"
        lines.append(f"Actual DA Category: {actual_cat} ({match_text})")
    else:
        lines.append("No actual category available.")

    return "\n".join(lines)

# -------------------------------------------
# Run Server
# -------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
