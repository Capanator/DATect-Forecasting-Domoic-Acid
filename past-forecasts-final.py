import os
import numpy as np
import pandas as pd
from joblib import Memory
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "ENABLE_LINEAR_LOGISTIC": False,
    "ENABLE_RANDOM_ANCHOR_FORECASTS": False,
    "CACHE_DIR": './cache',
    "DATA_FILE": 'final_output.parquet',
    "PORT": 8071,
    "NUM_RANDOM_ANCHORS": 20,
    "TEST_SIZE": 0.2
}

# Setup cache
os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
memory = Memory(CONFIG["CACHE_DIR"], verbose=0)
print(f"[INFO] Caching directory is ready at: {CONFIG['CACHE_DIR']}")

# ---------------------------------------------------------
# Data Processing Functions
# ---------------------------------------------------------
def create_numeric_transformer(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """Creates a transformer for numeric features and returns it with the processed dataframe"""
    X = df.drop(columns=drop_cols, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    transformer = ColumnTransformer(
        transformers=[('num', pipeline, numeric_cols)],
        remainder='passthrough'
    )
    return transformer, X

def transform_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                  drop_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform training and test data using a numeric transformer"""
    transformer, X_train = create_numeric_transformer(train_df, drop_cols)
    X_train_proc = transformer.fit_transform(X_train)
    X_test_proc = transformer.transform(test_df.drop(columns=drop_cols, errors='ignore'))
    return X_train_proc, X_test_proc

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

# ---------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------
def run_model(model, X_train, y_train, X_test, model_type='regression'):
    """Generic model training function"""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def create_model_configs():
    """Create model configurations for different methods"""
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_reg_config = {
        'model': RandomForestRegressor(random_state=42),
        'cv': tscv
    }
    
    rf_cls_config = {
        'model': RandomForestClassifier(random_state=42),
        'cv': tscv
    }
    
    lr_reg_config = {
        'model': LinearRegression(),
        'cv': None
    }
    
    lr_cls_config = {
        'model': LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
        'cv': None
    }
    
    return {
        'ml': {'reg': rf_reg_config, 'cls': rf_cls_config},
        'lr': {'reg': lr_reg_config, 'cls': lr_cls_config}
    }

def train_and_evaluate(data: pd.DataFrame, method='ml', test_size=None):
    """
    Generic training function that handles both regression and classification
    for any of the specified methods (ml, lr).
    """
    if test_size is None:
        test_size = CONFIG["TEST_SIZE"]
        
    print(f"[INFO] Training models using method: {method}")
    
    # Get model configurations
    model_configs = create_model_configs()
    if method not in model_configs:
        raise ValueError(f"Method '{method}' not supported")
    
    reg_config = model_configs[method]['reg']
    cls_config = model_configs[method]['cls']
    
    # Common column lists
    common_cols = ['Date', 'Site']
    reg_drop_cols = common_cols + ['DA_Levels', 'DA_Category']
    cls_drop_cols = common_cols + ['DA_Levels', 'DA_Category']
    
    # Results dictionary
    results = {}
    
    # Train and evaluate regression model
    print("[INFO] Training regression model...")
    data_reg = data.copy()
    train_reg, test_reg = train_test_split(data_reg, test_size=test_size, random_state=42)
    X_train_reg, X_test_reg = transform_data(train_reg, test_reg, reg_drop_cols)
    y_train_reg, y_test_reg = train_reg['DA_Levels'], test_reg['DA_Levels']
    
    best_reg, y_pred_reg = run_model(
        reg_config['model'], X_train_reg, y_train_reg, X_test_reg,
        model_type='regression'
    )
    
    test_reg_with_pred = test_reg.copy()
    test_reg_with_pred['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(y_test_reg, y_pred_reg)
    overall_mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
    
    site_stats_reg = test_reg_with_pred.groupby('Site')[['DA_Levels', 'Predicted_DA_Levels']].apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'mae': mean_absolute_error(x['DA_Levels'], x['Predicted_DA_Levels'])
        })
    )
    
    results["DA_Level"] = {
        "test_df": test_reg_with_pred,
        "site_stats": site_stats_reg,
        "overall_r2": overall_r2_reg,
        "overall_mae": overall_mae_reg,
        "model": best_reg
    }
    
    # Train and evaluate classification model
    print("[INFO] Training classification model...")
    data_cls = data.copy()
    train_cls, test_cls = train_test_split(data_cls, test_size=test_size, random_state=42)
    X_train_cls, X_test_cls = transform_data(train_cls, test_cls, cls_drop_cols)
    y_train_cls, y_test_cls = train_cls['DA_Category'], test_cls['DA_Category']
    
    best_cls, y_pred_cls = run_model(
        cls_config['model'], X_train_cls, y_train_cls, X_test_cls,
        model_type='classification'
    )
    
    test_cls_with_pred = test_cls.copy()
    test_cls_with_pred['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    
    site_stats_cls = test_cls_with_pred.groupby('Site')[['DA_Category', 'Predicted_DA_Category']].apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )
    
    results["DA_Category"] = {
        "test_df": test_cls_with_pred,
        "site_stats": site_stats_cls,
        "overall_accuracy": overall_accuracy_cls,
        "model": best_cls
    }
    
    print(f"[INFO] {method.upper()} training complete.")
    return results

# ---------------------------------------------------------
# Forecasting Functions
# ---------------------------------------------------------
def forecast_future(df: pd.DataFrame, anchor_date, site, method='ml'):
    """
    Unified forecasting function that supports different methods
    """
    print(f"[INFO] Forecasting for site '{site}' after '{anchor_date}' using {method}")
    
    # Filter data for the specific site and prepare train/test sets
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    
    if df_future.empty:
        return None
        
    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    df_test = df_site[df_site['Date'] == next_date].copy()
    
    if df_train.empty or df_test.empty:
        return None
    
    # Common columns to drop
    drop_cols = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    
    # Get model configurations
    model_configs = create_model_configs()
    if method not in model_configs:
        raise ValueError(f"Method '{method}' not supported")
    
    reg_config = model_configs[method]['reg']
    cls_config = model_configs[method]['cls']
    
    # Process data
    X_train, X_test = transform_data(df_train, df_test, drop_cols)
    
    # Train and predict with regression model
    reg_model = reg_config['model']
    reg_model.fit(X_train, df_train['DA_Levels'])
    y_pred_reg = reg_model.predict(X_test)
    
    # Train and predict with classification model
    cls_model = cls_config['model']
    cls_model.fit(X_train, df_train['DA_Category'])
    y_pred_cls = cls_model.predict(X_test)
    
    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]),
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0])
    }

def get_random_anchor_forecasts(data: pd.DataFrame, method='ml'):
    """Generate random anchor forecasts using the specified method"""
    print(f"[INFO] Generating random anchor forecasts using {method}...")
    
    df_after_2010 = data[data['Date'].dt.year >= 2010].copy().sort_values(['Site', 'Date'])
    pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()
    
    # Sample random anchor dates for each site
    df_random_anchors = pd.concat([
        group.sample(n=min(CONFIG["NUM_RANDOM_ANCHORS"], len(group)), random_state=42)
        for _, group in pairs_after_2010.groupby('Site')
    ]).reset_index(drop=True)
    
    # Generate forecasts
    results_list = []
    for _, row in df_random_anchors.iterrows():
        site_ = row['Site']
        anchor_date_ = row['Date']
        result = forecast_future(data, anchor_date_, site_, method)
        if result is not None:
            results_list.append(result)
    
    df_results_anchors = pd.DataFrame(results_list)
    if df_results_anchors.empty:
        return df_results_anchors, {}, None, None
    
    # Calculate performance metrics
    mae_anchors = mean_absolute_error(
        df_results_anchors['Actual_DA_Levels'],
        df_results_anchors['Predicted_DA_Levels']
    )
    acc_anchors = (
        df_results_anchors['Actual_DA_Category'] == df_results_anchors['Predicted_DA_Category']
    ).mean()
    
    # Create figures for each site
    figs_random_site = {}
    for site, df_site in df_results_anchors.groupby("Site"):
        df_line = df_site.sort_values("NextDate")
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
    
    return df_results_anchors, figs_random_site, mae_anchors, acc_anchors

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def prepare_all_predictions(data):
    """Prepare all predictions based on configuration"""
    predictions = {}
    random_anchors = {}
    
    # ML predictions
    predictions['ml'] = train_and_evaluate(data, method='ml')
    
    # Linear/Logistic predictions if enabled
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        predictions['lr'] = train_and_evaluate(data, method='lr')
    
    # Random anchor forecasts if enabled
    if CONFIG["ENABLE_RANDOM_ANCHOR_FORECASTS"]:
        random_anchors['ml'] = get_random_anchor_forecasts(data, method='ml')
        
        if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
            random_anchors['lr'] = get_random_anchor_forecasts(data, method='lr')
    
    return predictions, random_anchors

# ---------------------------------------------------------
# Dash App Setup
# ---------------------------------------------------------
def create_dash_app(predictions, random_anchors, data):
    """Create and configure the Dash app"""
    app = dash.Dash(__name__)
    
    # Create layout based on configuration
    analysis_layout = html.Div([
        html.H3("Overall Analysis (Time-Series)"),
        dcc.Dropdown(
            id='forecast-type-dropdown',
            options=[
                {'label': 'DA Levels', 'value': 'DA_Level'},
                {'label': 'DA Category', 'value': 'DA_Category'}
            ],
            value='DA_Level',
            style={'width': '50%', 'marginBottom': '15px'}
        ),
        dcc.Dropdown(
            id='site-dropdown',
            placeholder='Select Site',
            style={'width': '50%', 'marginBottom': '15px'}
        ),
        dcc.Graph(id='analysis-graph')
    ])
    
    if CONFIG["ENABLE_RANDOM_ANCHOR_FORECASTS"]:
        random_anchors_layout = html.Div([
            html.H3("Random Anchor Dates Forecast -> Next Date by Site"),
            html.Div(id='random-anchor-container')
        ])
    else:
        random_anchors_layout = html.Div([
            html.H3("Random Anchor Forecasts are disabled.")
        ])
    
    # Create tabs
    tabs_children = [dcc.Tab(label='Analysis', children=[analysis_layout])]
    if CONFIG["ENABLE_RANDOM_ANCHOR_FORECASTS"]:
        tabs_children.append(dcc.Tab(label='Random Anchor Forecast', children=[random_anchors_layout]))
    
    # Setup forecast method dropdown options
    forecast_methods = [
        {'label': 'Machine Learning Forecasts', 'value': 'ml'}
    ]
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        forecast_methods.append({'label': 'Linear/Logistic Regression Forecasts', 'value': 'lr'})
    
    # Create layout
    app.layout = html.Div([
        html.H1("Domoic Acid Forecast Dashboard"),
        html.Div([
            html.Label("Select Forecast Method"),
            dcc.Dropdown(
                id='forecast-method-dropdown',
                options=forecast_methods,
                value='ml',
                style={'width': '30%', 'marginLeft': '20px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
        dcc.Tabs(id="tabs", children=tabs_children),
        dcc.Store(id='data-store', data={'sites': data['Site'].unique().tolist()})
    ])
    
    # Define callbacks
    @app.callback(
        Output('site-dropdown', 'options'),
        [Input('data-store', 'data')]
    )
    def update_site_dropdown(data_store):
        sites = data_store['sites']
        options = [{'label': 'All Sites', 'value': 'All Sites'}] + [
            {'label': site, 'value': site} for site in sites
        ]
        return options
    
    @app.callback(
        Output('analysis-graph', 'figure'),
        [Input('forecast-type-dropdown', 'value'),
         Input('site-dropdown', 'value'),
         Input('forecast-method-dropdown', 'value')]
    )
    def update_graph(forecast_type, selected_site, forecast_method):
        # Validate inputs
        if forecast_method not in predictions:
            forecast_method = 'ml'
        
        pred = predictions[forecast_method]
        
        # Get appropriate data and stats based on forecast type
        if forecast_type == 'DA_Level':
            df_plot = pred['DA_Level']['test_df'].copy()
            site_stats = pred['DA_Level']['site_stats']
            overall_r2 = pred['DA_Level']['overall_r2']
            overall_mae = pred['DA_Level']['overall_mae']
            y_axis_title = 'Domoic Acid Levels'
            
            # Prepare data for plotting
            df_plot_melted = pd.melt(
                df_plot,
                id_vars=['Date', 'Site'],
                value_vars=['DA_Levels', 'Predicted_DA_Levels'],
                var_name='Metric',
                value_name='Value'
            )
            
            # Get performance text
            if selected_site is None or selected_site == 'All Sites':
                performance_text = f"Overall R² = {overall_r2:.2f}, MAE = {overall_mae:.2f}"
            else:
                if selected_site in site_stats.index:
                    site_r2 = site_stats.loc[selected_site, 'r2']
                    site_mae = site_stats.loc[selected_site, 'mae']
                    performance_text = f"R² = {site_r2:.2f}, MAE = {site_mae:.2f}"
                else:
                    performance_text = "No data for selected site."
        else:
            df_plot = pred['DA_Category']['test_df'].copy()
            site_stats = pred['DA_Category']['site_stats']
            overall_accuracy = pred['DA_Category']['overall_accuracy']
            y_axis_title = 'Domoic Acid Category'
            
            # Prepare data for plotting
            df_plot_melted = pd.melt(
                df_plot,
                id_vars=['Date', 'Site'],
                value_vars=['DA_Category', 'Predicted_DA_Category'],
                var_name='Metric',
                value_name='Value'
            )
            
            # Get performance text
            if selected_site is None or selected_site == 'All Sites':
                performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
            else:
                if selected_site in site_stats.index:
                    site_accuracy = site_stats.loc[selected_site]
                    performance_text = f"Accuracy = {site_accuracy:.2f}"
                else:
                    performance_text = "No data for selected site."
        
        # Filter data by site if specified
        if selected_site and selected_site != 'All Sites':
            df_plot_melted = df_plot_melted[df_plot_melted['Site'] == selected_site]
        
        df_plot_melted.sort_values('Date', inplace=True)
        
        # Create figure
        if selected_site == 'All Sites' or not selected_site:
            fig = px.line(
                df_plot_melted, x='Date', y='Value',
                color='Site', line_dash='Metric',
                title=f"{y_axis_title} Forecast - All Sites"
            )
        else:
            fig = px.line(
                df_plot_melted, x='Date', y='Value',
                color='Metric',
                category_orders={'Metric': ['DA_Levels', 'Predicted_DA_Levels'] 
                               if forecast_type == 'DA_Level' 
                               else ['DA_Category', 'Predicted_DA_Category']},
                title=f"{y_axis_title} Forecast - {selected_site}"
            )
        
        # Update layout
        fig.update_layout(
            yaxis_title=y_axis_title,
            xaxis_title='Date',
            annotations=[{
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': -0.2,
                'xanchor': 'center', 'yanchor': 'top',
                'text': performance_text,
                'showarrow': False
            }]
        )
        
        return fig
    
    if CONFIG["ENABLE_RANDOM_ANCHOR_FORECASTS"]:
        @app.callback(
            Output('random-anchor-container', 'children'),
            [Input('forecast-method-dropdown', 'value')]
        )
        def update_random_anchor_layout(forecast_method):
            # Validate input
            if forecast_method not in random_anchors:
                forecast_method = 'ml'
            
            df_results, figs_random_site, mae, acc = random_anchors[forecast_method]
            
            # Create graph components
            graphs = [dcc.Graph(figure=fig) for _, fig in figs_random_site.items()]
            
            # Create metrics display
            metrics = html.Div([
                html.H4("Overall Performance on Random Anchor Forecasts"),
                html.Ul([
                    html.Li(f"MAE (DA Levels): {mae:.3f}") if mae is not None else html.Li("No MAE (no data)"),
                    html.Li(f"Accuracy (DA Category): {acc:.3f}") if acc is not None else html.Li("No Accuracy (no data)")
                ])
            ], style={'marginTop': 20})
            
            return html.Div(graphs + [metrics])
    
    return app

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    # Load and prepare data
    print("[INFO] Loading and preparing data...")
    data = load_and_prepare_data(CONFIG["DATA_FILE"])
    
    # Generate predictions
    predictions, random_anchors = prepare_all_predictions(data)
    
    # Create and run Dash app
    app = create_dash_app(predictions, random_anchors, data)
    print(f"[INFO] Starting Dash app on port {CONFIG['PORT']}")
    app.run_server(debug=True, port=CONFIG["PORT"])