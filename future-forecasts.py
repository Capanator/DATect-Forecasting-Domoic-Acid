import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ================================
# DATA HANDLING
# ================================
def load_and_prepare_data(file_path):
    """Load and prepare data with essential features."""
    data = pd.read_parquet(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(['site', 'date'], inplace=True)

    # Create lag features and seasonal components
    for lag in [1, 2, 3]:
        data[f'da_lag_{lag}'] = data.groupby('site')['da'].shift(lag)

    day_of_year = data['date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # Create category feature
    data['da-category'] = pd.cut(
        data['da'],
        bins=[-float('inf'), 5, 20, 40, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Drop rows with NaNs in critical columns (especially lags and target itself)
    # This is important because if 'da' is NaN, 'da-category' will also be problematic.
    # Lag features inherently create NaNs at the beginning of each site's series.
    critical_cols_for_dropna = ['da', 'da-category'] + [f'da_lag_{lag}' for lag in [1,2,3]]
    data.dropna(subset=critical_cols_for_dropna, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

# ================================
# FORECASTING
# ================================
def get_training_forecast_data(df_all_sites: pd.DataFrame, forecast_date: pd.Timestamp, site_for_forecast: str):
    """
    Extract training data from ALL sites up to an anchor date,
    and forecast data for the specified date and site.

    Args:
        df_all_sites: DataFrame containing data for all sites.
        forecast_date: The date for which a forecast is desired.
        site_for_forecast: The specific site for which the forecast is being made.

    Returns:
        Tuple: (df_train, df_forecast, anchor_date, test_date)
               df_train: Training data from all sites up to anchor_date.
               df_forecast: The row(s) for the site_for_forecast at the forecast_date/test_date.
               anchor_date: The latest date in data for site_for_forecast strictly before forecast_date.
               test_date: The actual date of the forecast point (might be >= forecast_date).
    """
    # Work with data for the specific site to determine anchor_date and test_date
    df_target_site = df_all_sites[df_all_sites['site'] == site_for_forecast].copy()
    df_target_site.sort_values('date', inplace=True)

    # Determine anchor_date: latest date strictly before forecast_date for the target site
    df_before_forecast_target_site = df_target_site[df_target_site['date'] < forecast_date]

    if df_before_forecast_target_site.empty:
        # Not enough historical data for this site before the forecast_date to determine an anchor
        # Or the forecast_date is too early for this site.
        print(f"Warning: No data found for site '{site_for_forecast}' before {forecast_date.date()} to set an anchor.")
        return pd.DataFrame(), pd.DataFrame(), None, None

    anchor_date = df_before_forecast_target_site['date'].max()

    # Determine test_date and df_forecast for the target site
    # test_date is the earliest date >= forecast_date for the target site
    df_after_or_on_forecast_target_site = df_target_site[df_target_site['date'] >= forecast_date]
    test_date = df_after_or_on_forecast_target_site['date'].min() if not df_after_or_on_forecast_target_site.empty else None

    # Get forecast row (actual data if exists at forecast_date or test_date, otherwise synthetic)
    if forecast_date in df_target_site['date'].values:
        df_forecast = df_target_site[df_target_site['date'] == forecast_date].copy()
    elif test_date is not None: # Actual data exists on or after forecast_date
        df_forecast = df_target_site[df_target_site['date'] == test_date].copy()
    else:
        # Create a synthetic forecast row if no future data point exists for the target site.
        # This scenario implies forecasting beyond the known data for that site.
        last_known_row_target_site = df_target_site[df_target_site['date'] == anchor_date].iloc[[0]].copy() # Use iloc[[0]] to ensure DataFrame
        
        # Update date and clear target values for the synthetic row
        last_known_row_target_site['date'] = forecast_date # Use the requested forecast_date
        last_known_row_target_site[['da', 'da-category']] = np.nan
        
        # Re-create lag features for this synthetic row based on the *actual* last known row
        # This is a simplification; true multi-step forecasting would update lags iteratively.
        # For a one-step-ahead, the lags should ideally come from the anchor_date row.
        # If 'da' from anchor_date is 'X', then da_lag_1 for forecast_date should be 'X'.
        if not df_before_forecast_target_site.empty:
            actual_anchor_row_for_lags = df_before_forecast_target_site[df_before_forecast_target_site['date'] == anchor_date].iloc[0]
            last_known_row_target_site['da_lag_1'] = actual_anchor_row_for_lags['da']
            if 'da_lag_1' in actual_anchor_row_for_lags: # if lag 1 existed
                 last_known_row_target_site['da_lag_2'] = actual_anchor_row_for_lags['da_lag_1']
            if 'da_lag_2' in actual_anchor_row_for_lags: # if lag 2 existed
                 last_known_row_target_site['da_lag_3'] = actual_anchor_row_for_lags['da_lag_2']
        
        # Update seasonal features for the synthetic row for the new forecast_date
        day_of_year_forecast = forecast_date.dayofyear
        last_known_row_target_site['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year_forecast / 365)
        last_known_row_target_site['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year_forecast / 365)

        df_forecast = last_known_row_target_site
        test_date = forecast_date # The test_date is the forecast_date itself for synthetic rows

    # --- Training data: Use data from ALL sites up to and including the anchor_date ---
    df_train = df_all_sites[df_all_sites['date'] <= anchor_date].copy()
    
    # Ensure df_train and df_forecast are not empty before proceeding
    if df_train.empty:
        print(f"Warning: Training data is empty for anchor_date {anchor_date}.")
        return pd.DataFrame(), df_forecast, anchor_date, test_date
    if df_forecast.empty:
        # This case should be rare due to synthetic row creation, but good to check.
        print(f"Warning: Forecast data frame is empty for site '{site_for_forecast}' at {forecast_date.date()}.")
        return df_train, pd.DataFrame(), anchor_date, test_date


    return df_train, df_forecast, anchor_date, test_date


def forecast_for_date(df_all_sites, forecast_date, site_for_forecast):
    """Generate complete forecast for a specific date and site, using all past site data for training."""
    # Get data splits: df_train now contains data from ALL sites up to anchor_date
    result = get_training_forecast_data(df_all_sites, forecast_date, site_for_forecast)
    df_train, df_forecast, anchor_date, test_date = result

    if df_train.empty or df_forecast.empty:
        error_message = "Not enough data to generate forecast. "
        if df_train.empty:
            error_message += f"Training data empty (anchor: {anchor_date}). "
        if df_forecast.empty:
            error_message += f"Forecast row could not be constructed for site {site_for_forecast} on {forecast_date.date()}."
        print(error_message) # Also print for server logs
        # Return a dictionary with Nones or raise an error, depending on how caller handles it
        return { # Match structure of successful return but with Nones
            'ForecastPoint': forecast_date, 'Anchordate': anchor_date, 'Testdate': test_date,
            'Predicted_da_Q05': None, 'Predicted_da_Q50': None, 'Predicted_da_Q95': None,
            'Predicted_da_RF': None, 'Actual_da': None, 'SingledateCoverage': None,
            'Predicted_da-category': None, 'Probabilities': [0.25]*4, # Placeholder
            'Actual_da-category': None, 'SingledateLogLoss': None,
            'Error': error_message # Add an error field for clarity
        }


    # Common feature processing
    # 'site' is dropped here. If you want the model to use 'site' as a feature,
    # remove 'site' from drop_cols and ensure it's handled (e.g., one-hot encoded)
    # by the preprocessor.
    drop_cols = ['date', 'site', 'da', 'da-category']
    numeric_processor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Prepare features and targets
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train_reg = df_train['da']
    y_train_cls = df_train['da-category']

    X_forecast = df_forecast.drop(columns=drop_cols, errors='ignore')
    
    # Ensure X_forecast has the same columns as X_train (after dropping)
    # This is important if some features were entirely NaN in X_forecast and got dropped,
    # or if new columns appeared due to different processing (less likely with fixed drop_cols).
    X_forecast = X_forecast.reindex(columns=X_train.columns, fill_value=0) # Use 0 or np.nan, imputer handles np.nan


    # Check if actual target exists in forecast row
    y_test_reg = None if df_forecast['da'].isnull().all() else df_forecast['da']
    y_test_cls = None if df_forecast['da-category'].isnull().all() else df_forecast['da-category']


    # Preprocess features
    # Ensure consistent feature set for preprocessor based on X_train
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    # If no numeric columns, preprocessor might fail or act unexpectedly.
    if not num_cols.tolist(): # Check if num_cols is empty
        print(f"Warning: No numeric columns found in training data for site {site_for_forecast}, date {forecast_date}.")
        # Handle this scenario: maybe return Nones or an error structure
        return { # Match structure but indicate error
            'ForecastPoint': forecast_date, 'Anchordate': anchor_date, 'Testdate': test_date,
             'Error': "No numeric features for training."
        }

    preprocessor = ColumnTransformer([('num', numeric_processor, num_cols)], remainder='drop', verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas") # Ensures output is DataFrame

    X_train_processed = preprocessor.fit_transform(X_train)
    X_forecast_processed = preprocessor.transform(X_forecast)
    
    # Ensure X_forecast_processed has columns if X_train_processed did (e.g. if all values were imputed to a constant then scaled)
    if hasattr(X_train_processed, 'columns') and not X_forecast_processed.columns.equals(X_train_processed.columns):
        X_forecast_processed = X_forecast_processed.reindex(columns=X_train_processed.columns, fill_value=0)


    # REGRESSION FORECAST
    quantiles = {'q05': 0.05, 'q50': 0.50, 'q95': 0.95}
    gb_preds = {}

    for name, alpha in quantiles.items():
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            loss='quantile',
            alpha=alpha,
            random_state=42
        )
        model.fit(X_train_processed, y_train_reg)
        gb_preds[name] = float(model.predict(X_forecast_processed)[0])

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_processed, y_train_reg)
    rf_pred = float(rf_model.predict(X_forecast_processed)[0])

    single_coverage = None
    actual_levels = float(y_test_reg.iloc[0]) if y_test_reg is not None and not y_test_reg.empty else None
    if actual_levels is not None:
        single_coverage = 1.0 if gb_preds['q05'] <= actual_levels <= gb_preds['q95'] else 0.0

    # CLASSIFICATION FORECAST
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_processed, y_train_cls)

    pred_cat_proba = clf.predict_proba(X_forecast_processed)[0]
    pred_cat = int(np.argmax(pred_cat_proba)) # Get class with highest probability
    prob_list = list(pred_cat_proba)


    actual_cat = int(y_test_cls.iloc[0]) if y_test_cls is not None and not y_test_cls.empty else None
    single_logloss = None

    if actual_cat is not None:
        # Ensure clf.classes_ aligns with the labels [0, 1, 2, 3] if not all are present in y_train_cls
        # For log_loss, labels parameter should cover all possible true labels.
        # If da-category is always 0,1,2,3 then this is fine.
        possible_labels = sorted(df_all_sites['da-category'].unique()) # Use all possible known labels
        single_logloss = log_loss([actual_cat], [prob_list], labels=possible_labels)


    return {
        'ForecastPoint': forecast_date,
        'Anchordate': anchor_date,
        'Testdate': test_date,
        'Predicted_da_Q05': gb_preds['q05'],
        'Predicted_da_Q50': gb_preds['q50'],
        'Predicted_da_Q95': gb_preds['q95'],
        'Predicted_da_RF': rf_pred,
        'Actual_da': actual_levels,
        'SingledateCoverage': single_coverage,
        'Predicted_da-category': pred_cat,
        'Probabilities': prob_list,
        'Actual_da-category': actual_cat,
        'SingledateLogLoss': single_logloss,
        'Error': None # Indicate success
    }

# ================================
# VISUALIZATION
# ================================
def create_level_range_graph(q05, q50, q95, actual_levels=None, rf_prediction=None):
    """Create gradient visualization for DA level forecast."""
    fig = go.Figure()
    n_segments = 30
    range_width = q95 - q05
    max_distance = max(q50 - q05, q95 - q50) if range_width > 1e-6 else 1
    if max_distance <= 1e-6: 
        max_distance = 1  # Avoid division by zero if q05=q50=q95

    base_color = (70, 130, 180)  # Steel blue

    # Gradient confidence area
    for i in range(n_segments):
        x0 = q05 + (i / n_segments) * (range_width)
        x1 = q05 + ((i + 1) / n_segments) * (range_width)
        midpoint = (x0 + x1) / 2
        # Calculate opacity - handle case where max_distance is very small
        opacity = 1 - (abs(midpoint - q50) / max_distance) ** 0.5 if max_distance > 1e-6 else (0.8 if abs(midpoint - q50) < 1e-6 else 0.2)
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.4, y1=0.6,
            line=dict(width=0),
            fillcolor=f'rgba{(*base_color, max(0, min(1, opacity)))}',  # Ensure opacity is valid
            layer='below'
        )

    # Add median line
    fig.add_trace(go.Scatter(
        x=[q50, q50], y=[0.4, 0.6],
        mode='lines',
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='Median (Q50 - GB)'
    ))

    # Add range endpoints
    fig.add_trace(go.Scatter(
        x=[q05, q95], y=[0.5, 0.5],
        mode='markers',
        marker=dict(size=15, color='rgba(70, 130, 180, 0.3)', symbol='line-ns-open'),
        name='Prediction Range (GB Q05-Q95)'
    ))

    if rf_prediction is not None:
        fig.add_trace(go.Scatter(
            x=[rf_prediction], y=[0.5],  # Plot at the same y-level
            mode='markers',
            marker=dict(
                size=14,
                color='darkorange',
                symbol='diamond-tall',
                line=dict(width=1, color='black')  # Add outline for visibility
            ),
            name='Random Forest Pred.'
        ))

    # Add actual value if available
    if actual_levels is not None:
        fig.add_trace(go.Scatter(
            x=[actual_levels], y=[0.5],
            mode='markers',
            marker=dict(size=18, color='red', symbol='x-thin', line=dict(width=2)),
            name='Actual Value'
        ))

    fig.update_layout(
        title="DA Level Forecast: Gradient (GB) & Point (RF)",
        xaxis_title="DA Level",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=True,
        height=300,
        plot_bgcolor='white'
    )

    return fig

def create_category_graph(probs, pred_cat, actual_cat=None):
    """Create bar chart for category probabilities."""
    CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
    # Ensure pred_cat is within bounds before highlighting
    if 0 <= pred_cat < len(colors):
        colors[pred_cat] = '#2ca02c'  # Highlight predicted category

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=CATEGORY_LABELS,
        y=probs,
        marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition='auto'
    ))

    if actual_cat is not None and 0 <= actual_cat < len(probs):
        fig.add_annotation(
            x=actual_cat,
            y=probs[actual_cat] + 0.05,
            text="Actual",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            font=dict(color='red')
        )

    fig.update_layout(
        title="Category Probability Distribution (Random Forest)",
        yaxis=dict(title="Probability", range=[0, 1.1]),
        xaxis=dict(title="Category"),
        showlegend=False,
        height=400
    )

    return fig

def format_forecast_output(result):
    """Format forecast results as text for display."""
    CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']

    q05 = result['Predicted_da_Q05']
    q50 = result['Predicted_da_Q50']
    q95 = result['Predicted_da_Q95']
    rf_pred = result['Predicted_da_RF']  # Get RF prediction
    actual_levels = result['Actual_da']
    prob_list = result['Probabilities']

    lines = [
        f"Forecast date (target): {result['ForecastPoint'].date()}",
        f"Anchor date (training cutoff): {result['Anchordate'].date()}",
    ]

    if result['Testdate'] is not None:
        lines.append(f"Test date (for accuracy): {result['Testdate'].date()}")
    else:
        lines.append("Test date (for accuracy): N/A")

    lines += [
        "",
        "--- Regression (da) ---",
        f"Predicted Range (GB): {q05:.2f} (Q05) – {q50:.2f} (Q50) – {q95:.2f} (Q95)",
        f"Predicted Value (RF): {rf_pred:.2f}",
    ]

    if actual_levels is not None:
        within_range = result['SingledateCoverage']
        status = 'Within GB Range ✅' if within_range else 'Outside GB Range ❌'
        lines.append(f"Actual Value: {actual_levels:.2f} ({status})")
    else:
        lines.append("Actual Value: N/A (forecast beyond available data)")

    lines += [
        "",
        "--- Classification (da-category, Random Forest) ---",
        f"Predicted: {CATEGORY_LABELS[result['Predicted_da-category']]}",
        "Probabilities: " + ", ".join([
            f"{label}: {prob * 100:.1f}%"
            for label, prob in zip(CATEGORY_LABELS, prob_list)
        ])
    ]

    if result['Actual_da-category'] is not None:
        actual_cat = result['Actual_da-category']
        match_status = "✅ MATCH" if result['Predicted_da-category'] == actual_cat else "❌ MISMATCH"
        lines.append(f"Actual: {CATEGORY_LABELS[actual_cat]} {match_status}")
    else:
        lines.append("Actual Category: N/A")

    return "\n".join(lines)

# ================================
# DASH APP
# ================================
# Load data
file_path = 'final_output.parquet'  # Make sure this file exists or change the path
try:
    raw_data = load_and_prepare_data(file_path)
    min_forecast_date = pd.to_datetime("2010-01-01")  # Or set based on data min date + buffer
    available_sites = raw_data['site'].unique()
    initial_site = available_sites[0] if len(available_sites) > 0 else None
except FileNotFoundError:
    print(f"Error: Parquet file not found at {file_path}")
    print("Please ensure the data file is in the correct location.")
    # Provide dummy data or exit if essential
    raw_data = pd.DataFrame({'site': ['dummy'], 'date': [pd.Timestamp('2023-01-01')], 'da': [10]})
    raw_data = load_and_prepare_data(raw_data)  # Process dummy data
    min_forecast_date = pd.to_datetime("2023-01-01")
    available_sites = ['dummy']
    initial_site = 'dummy'
except Exception as e:
    print(f"An error occurred loading or preparing data: {e}")
    raw_data = pd.DataFrame({'site': ['error'], 'date': [pd.Timestamp('2023-01-01')], 'da': [0]})
    raw_data = load_and_prepare_data(raw_data)
    min_forecast_date = pd.to_datetime("2023-01-01")
    available_sites = ['error']
    initial_site = 'error'


app = dash.Dash(__name__)

# Original UI Layout
app.layout = html.Div([
    html.H3("Forecast by Specific date & site"),

    html.Label("Choose a site:"),
    dcc.Dropdown(
        id='site-dropdown-forecast',
        options=[{'label': s, 'value': s} for s in available_sites],
        value=initial_site,
        style={'width': '50%'}
    ),

    html.Label("Pick a Forecast date (≥ 2010 or data start):"),
    dcc.DatePickerSingle(
        id='forecast-date-picker',
        min_date_allowed=min_forecast_date,
        max_date_allowed='2099-12-31',
        initial_visible_month=min_forecast_date,
        date=min_forecast_date,
    ),

    html.Div(
        children=[
            # Textual output
            html.Div(id='forecast-output-partial', style={'whiteSpace': 'pre-wrap', 'marginTop': 20, 'border': '1px solid lightgrey', 'padding': '10px'}),

            # Graphs for forecast ranges
            html.Div([
                dcc.Graph(id='level-range-graph', style={'display': 'inline-block', 'width': '49%'}),
                dcc.Graph(id='category-range-graph', style={'display': 'inline-block', 'width': '49%'})
            ])
        ],
        style={'marginTop': 30}
    )
])

# ================================
# CALLBACK
# ================================
@app.callback(
    [
        Output('forecast-output-partial', 'children'),
        Output('level-range-graph', 'figure'),
        Output('category-range-graph', 'figure')
    ],
    [
        Input('forecast-date-picker', 'date'),
        Input('site-dropdown-forecast', 'value')
    ]
)
def update_forecast(forecast_date_str, site):
    # Add basic input validation
    if not forecast_date_str:
        return ("Please select a forecast date.", go.Figure(), go.Figure())
    if not site:
         return ("Please select a site.", go.Figure(), go.Figure())

    # Ensure raw_data is available (in case of loading errors)
    if raw_data is None or raw_data.empty:
         return ("Error: Data not loaded. Cannot generate forecast.", go.Figure(), go.Figure())

    try:
        forecast_date = pd.to_datetime(forecast_date_str)
    except ValueError:
        return (f"Invalid date format: {forecast_date_str}", go.Figure(), go.Figure())

    # Run the forecast calculation
    try:
        result = forecast_for_date(raw_data, forecast_date, site)
    except Exception as e:
        print(f"Error during forecasting for {site} on {forecast_date}: {e}")
        return (f"An error occurred during forecast calculation: {e}", go.Figure(), go.Figure())

    # Handle cases where forecasting is not possible (e.g., insufficient data)
    if not result:
        msg = (f"No forecast possible for site={site} using Forecast date={forecast_date.date()}.\n"
               "Possibly not enough historical data before this date.")
        return (msg, go.Figure(), go.Figure())

    # Format output and generate figures if successful
    text_output = format_forecast_output(result)

    level_fig = create_level_range_graph(
        result['Predicted_da_Q05'],
        result['Predicted_da_Q50'],
        result['Predicted_da_Q95'],
        result['Actual_da'],
        result['Predicted_da_RF']
    )

    category_fig = create_category_graph(
        result['Probabilities'],
        result['Predicted_da-category'],
        result['Actual_da-category']
    )

    return (text_output, level_fig, category_fig)

# ================================
# MAIN
# ================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8065)