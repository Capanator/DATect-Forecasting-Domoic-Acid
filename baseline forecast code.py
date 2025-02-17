import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    """
    Reads CSV and applies feature engineering:
      - KMeans clustering for spatial grouping
      - Cyclical encoding of day_of_year
      - One-hot encoding of Month
      - Year as a separate feature
      - Lag features for DA_Levels
      - Categorize DA_Levels into DA_Category
    No final imputation/scaling is done here.
    """
    data = pd.read_csv(file_path)

    # Convert date and sort
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # Spatial Feature Engineering (KMeans on entire dataset)
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # Temporal Feature Engineering
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # Add Month as Categorical Variable (one-hot)
    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')

    # Include Year as a Feature
    data['Year'] = data['Date'].dt.year

    # Lag Features (group by Site to avoid mixing sites)
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    # Interaction Terms: cluster dummies * sin/cos day_of_year
    cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
        data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']

    # Categorize DA Levels
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
    imputation + scaling only to the training set
    and uses LinearRegression + LogisticRegression
    as baseline models.
    Returns test data with predictions and metrics.
    """
    # -----------------------------
    # 1) DA_Levels Regression
    # -----------------------------
    data_reg = data.drop(['DA_Category'], axis=1)

    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.2, random_state=42)

    # Identify columns not used as numeric features
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']
    X_train_reg = train_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = train_set_reg['DA_Levels']

    X_test_reg = test_set_reg.drop(columns=drop_cols_reg, errors='ignore')
    y_test_reg = test_set_reg['DA_Levels']

    # Build a pipeline for numeric columns
    numeric_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns
    numeric_transformer_reg = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_reg, numeric_cols_reg)
        ],
        remainder='passthrough'  # pass other (dummy) columns as is
    )

    # Fit the pipeline on training data
    X_train_reg_processed = preprocessor_reg.fit_transform(X_train_reg)
    X_test_reg_processed = preprocessor_reg.transform(X_test_reg)

    # Train LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train_reg_processed, y_train_reg)

    # Predict
    y_pred_reg = linear_regressor.predict(X_test_reg_processed)

    # Attach predictions to the test set
    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg

    # Metrics: overall R² and RMSE
    overall_r2_reg = r2_score(y_test_reg, y_pred_reg)
    overall_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    # Per-site metrics
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

    # Build a pipeline for numeric columns
    numeric_cols_cls = X_train_cls.select_dtypes(include=[np.number]).columns
    numeric_transformer_cls = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor_cls = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_cls, numeric_cols_cls)
        ],
        remainder='passthrough'
    )

    # Fit the pipeline on training data
    X_train_cls_processed = preprocessor_cls.fit_transform(X_train_cls)
    X_test_cls_processed = preprocessor_cls.transform(X_test_cls)

    # Train LogisticRegression
    logistic_classifier = LogisticRegression(max_iter=200, random_state=42)
    logistic_classifier.fit(X_train_cls_processed, y_train_cls)

    # Predict
    y_pred_cls = logistic_classifier.predict(X_test_cls_processed)

    # Attach predictions
    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls

    # Calculate accuracy
    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    site_stats_cls = test_set_cls.groupby('Site').apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )

    return {
        "DA_Level": (test_set_reg, site_stats_reg, overall_r2_reg, overall_rmse_reg),
        "DA_Category": (test_set_cls, site_stats_cls, overall_accuracy_cls)
    }

# -------------------------------------------
# Dash App
# -------------------------------------------
app = dash.Dash(__name__)

# Load data (feature engineering only)
file_path = 'final_output.csv'  # Update as needed
raw_data = load_and_prepare_data(file_path)

# Train models and get predictions
predictions = train_and_predict(raw_data)

app.layout = html.Div([
    html.H1("Domoic Acid Analysis Dashboard"),
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

@app.callback(
    Output('analysis-graph', 'figure'),
    [Input('forecast-type-dropdown', 'value'),
     Input('site-dropdown', 'value')]
)
def update_graph(selected_forecast_type, selected_site):
    if selected_forecast_type == 'DA_Level':
        df_plot, site_stats, overall_r2, overall_rmse = predictions['DA_Level']
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
        df_plot, site_stats, overall_accuracy = predictions['DA_Category']
        y_axis_title = 'Domoic Acid Category'
        y_columns = ['DA_Category', 'Predicted_DA_Category']
        
        if selected_site == 'All Sites':
            performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
        elif selected_site in site_stats.index:
            site_accuracy = site_stats.loc[selected_site]
            performance_text = f"Accuracy = {site_accuracy:.2f}"
        else:
            performance_text = "No data for selected site."
    
    # Filter based on selected site if necessary.
    if selected_site != 'All Sites':
        df_plot = df_plot[df_plot['Site'] == selected_site]
    
    # Sort by date for a cleaner plot.
    df_plot = df_plot.sort_values('Date')
    
    # --- Create a forecast date ---
    # If your forecast should be made for the next day, then:
    df_plot = df_plot.copy()  # avoid modifying the original DataFrame
    df_plot['Forecast_Date'] = df_plot['Date'] + pd.DateOffset(days=1)
    
    # --- Build the line plot using Forecast_Date as the x-axis ---
    fig = px.line(
        df_plot,
        x='Forecast_Date',
        y=y_columns,
        color='Site' if selected_site == 'All Sites' else None,
        title=f"{y_axis_title} Forecast - {selected_site}"
    )
    
    fig.update_layout(
        xaxis_title='Forecast Date',
        yaxis_title=y_axis_title,
        annotations=[{
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.2,
            'xanchor': 'center', 'yanchor': 'top',
            'text': performance_text,
            'showarrow': False
        }]
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8054)
