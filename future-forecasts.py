import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # Create lag features and seasonal components
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # Create category feature
    data['DA_Category'] = pd.cut(
        data['DA_Levels'], 
        bins=[-float('inf'), 5, 20, 40, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return data

# ================================
# FORECASTING
# ================================
def get_training_forecast_data(df, forecast_date, site):
    """Extract training and forecast data for the specified date and site."""
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)
    
    # Must have historical data
    df_before = df_site[df_site['Date'] < forecast_date]
    if df_before.empty:
        return None, None, None, None
        
    # Training anchor and test dates
    anchor_date = df_before['Date'].max()
    df_after = df_site[df_site['Date'] >= forecast_date]
    test_date = df_after['Date'].min() if not df_after.empty else None
    
    # Get forecast row (real or synthetic)
    if forecast_date in df_site['Date'].values:
        df_forecast = df_site[df_site['Date'] == forecast_date].copy()
    elif test_date is not None:
        df_forecast = df_site[df_site['Date'] == test_date].copy()
    else:
        # Create synthetic forecast row
        last_row = df_site[df_site['Date'] == anchor_date].iloc[0]
        new_row = last_row.copy()
        new_row['Date'] = forecast_date
        new_row['DA_Levels'] = new_row['DA_Category'] = np.nan
        df_forecast = pd.DataFrame([new_row])
    
    # Training data
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    
    return df_train, df_forecast, anchor_date, test_date

def forecast_for_date(df, forecast_date, site):
    """Generate complete forecast for a specific date and site."""
    # Get data splits
    result = get_training_forecast_data(df, forecast_date, site)
    if result is None:
        return None
    df_train, df_forecast, anchor_date, test_date = result
    
    # Common feature processing
    drop_cols = ['Date', 'Site', 'DA_Levels', 'DA_Category']
    numeric_processor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    # REGRESSION FORECAST
    X_train_reg = df_train.drop(columns=drop_cols)
    y_train_reg = df_train['DA_Levels']
    X_forecast_reg = df_forecast.drop(columns=drop_cols)
    
    # Check if actual target exists in forecast row
    y_test_reg = None if df_forecast['DA_Levels'].isnull().all() else df_forecast['DA_Levels']
    
    # Preprocess features
    num_cols = X_train_reg.select_dtypes(include=[np.number]).columns
    preprocessor = ColumnTransformer([('num', numeric_processor, num_cols)], remainder='drop')
    
    X_train_processed = preprocessor.fit_transform(X_train_reg)
    X_forecast_processed = preprocessor.transform(X_forecast_reg)
    
    # Train quantile regressors
    quantiles = {'q05': 0.05, 'q50': 0.50, 'q95': 0.95}
    preds = {}
    
    for name, alpha in quantiles.items():
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            loss='quantile', 
            alpha=alpha, 
            random_state=42
        )
        model.fit(X_train_processed, y_train_reg)
        preds[name] = float(model.predict(X_forecast_processed)[0])
    
    # Check if actual value is within prediction interval
    single_coverage = None
    actual_levels = float(y_test_reg.iloc[0]) if y_test_reg is not None else None
    if actual_levels is not None:
        single_coverage = 1.0 if preds['q05'] <= actual_levels <= preds['q95'] else 0.0
    
    # CLASSIFICATION FORECAST
    X_train_cls = X_train_reg  # Reuse the same features
    y_train_cls = df_train['DA_Category']
    X_forecast_cls = X_forecast_reg
    y_test_cls = None if df_forecast['DA_Category'].isnull().all() else df_forecast['DA_Category']
    
    # Use a simpler classifier (no need for stacking)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf.fit(X_train_processed, y_train_cls)
    
    pred_cat = int(clf.predict(X_forecast_processed)[0])
    prob_list = list(clf.predict_proba(X_forecast_processed)[0])
    
    # Calculate metrics if actual value exists
    actual_cat = int(y_test_cls.iloc[0]) if y_test_cls is not None else None
    single_logloss = None
    
    if actual_cat is not None:
        single_logloss = log_loss([actual_cat], [prob_list], labels=clf.classes_)
    
    return {
        'ForecastPoint': forecast_date,
        'AnchorDate': anchor_date,
        'TestDate': test_date,
        # Regression results
        'Predicted_DA_Levels_Q05': preds['q05'],
        'Predicted_DA_Levels_Q50': preds['q50'],
        'Predicted_DA_Levels_Q95': preds['q95'],
        'Actual_DA_Levels': actual_levels,
        'SingleDateCoverage': single_coverage,
        # Classification results
        'Predicted_DA_Category': pred_cat,
        'Probabilities': prob_list,
        'Actual_DA_Category': actual_cat,
        'SingleDateLogLoss': single_logloss,
    }

# ================================
# VISUALIZATION
# ================================
def create_level_range_graph(q05, q50, q95, actual_levels=None):
    """Create gradient visualization for DA level forecast."""
    fig = go.Figure()
    n_segments = 30
    max_distance = max(q50 - q05, q95 - q50) if max(q50 - q05, q95 - q50) > 0 else 1
    base_color = (70, 130, 180)  # Steel blue

    # Gradient confidence area
    for i in range(n_segments):
        x0 = q05 + (i/n_segments)*(q95 - q05)
        x1 = q05 + ((i+1)/n_segments)*(q95 - q05)
        midpoint = (x0 + x1) / 2
        opacity = 1 - (abs(midpoint - q50) / max_distance)**0.5
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.4, y1=0.6,
            line=dict(width=0),
            fillcolor=f'rgba{(*base_color, opacity)}',
            layer='below'
        )

    # Add median line
    fig.add_trace(go.Scatter(
        x=[q50, q50], y=[0.4, 0.6],
        mode='lines', 
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='Median (Q50)'
    ))
    
    # Add range endpoints
    fig.add_trace(go.Scatter(
        x=[q05, q95], y=[0.5, 0.5],
        mode='markers',
        marker=dict(size=15, color='rgba(70, 130, 180, 0.3)', symbol='line-ns-open'),
        name='Prediction Range'
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
        title="DA Level Forecast Range with Gradient Confidence",
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
    colors[pred_cat] = '#2ca02c'  # Highlight predicted category
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=CATEGORY_LABELS,
        y=probs,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto'
    ))
    
    if actual_cat is not None:
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
        title="Category Probability Distribution",
        yaxis=dict(title="Probability", range=[0, 1.1]),
        xaxis=dict(title="Category"),
        showlegend=False,
        height=400
    )
    
    return fig

def format_forecast_output(result):
    """Format forecast results as text for display."""
    CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
    
    q05 = result['Predicted_DA_Levels_Q05']
    q50 = result['Predicted_DA_Levels_Q50']
    q95 = result['Predicted_DA_Levels_Q95']
    actual_levels = result['Actual_DA_Levels']
    prob_list = result['Probabilities']
    
    lines = [
        f"Forecast Date (target): {result['ForecastPoint'].date()}",
        f"Anchor Date (training cutoff): {result['AnchorDate'].date()}",
    ]
    
    if result['TestDate'] is not None:
        lines.append(f"Test Date (for accuracy): {result['TestDate'].date()}")
    else:
        lines.append("Test Date (for accuracy): N/A")
    
    lines += [
        "",
        "--- Regression (DA_Levels) ---",
        f"Predicted Range: {q05:.2f} (Q05) – {q50:.2f} (Q50) – {q95:.2f} (Q95)",
    ]
    
    if actual_levels is not None:
        within_range = result['SingleDateCoverage']
        status = 'Within Range ✅' if within_range else 'Outside Range ❌'
        lines.append(f"Actual Value: {actual_levels:.2f} ({status})")
    else:
        lines.append("Actual Value: N/A (forecast beyond available data)")
    
    lines += [
        "",
        "--- Classification (DA_Category) ---",
        f"Predicted: {CATEGORY_LABELS[result['Predicted_DA_Category']]}",
        "Probabilities: " + ", ".join([
            f"{label}: {prob*100:.1f}%" 
            for label, prob in zip(CATEGORY_LABELS, prob_list)
        ])
    ]
    
    if result['Actual_DA_Category'] is not None:
        actual_cat = result['Actual_DA_Category']
        match_status = "✅ MATCH" if result['Predicted_DA_Category'] == actual_cat else "❌ MISMATCH"
        lines.append(f"Actual: {CATEGORY_LABELS[actual_cat]} {match_status}")
    else:
        lines.append("Actual Category: N/A")
    
    return "\n".join(lines)

# ================================
# DASH APP
# ================================
# Load data
file_path = 'final_output_og.parquet'
raw_data = load_and_prepare_data(file_path)
min_forecast_date = pd.to_datetime("2010-01-01")

app = dash.Dash(__name__)

# Original UI Layout
app.layout = html.Div([
    html.H3("Forecast by Specific Date & Site"),
    
    html.Label("Choose a Site:"),
    dcc.Dropdown(
        id='site-dropdown-forecast',
        options=[{'label': s, 'value': s} for s in raw_data['Site'].unique()],
        value=raw_data['Site'].unique()[0],
        style={'width': '50%'}
    ),

    html.Label("Pick a Forecast Date (≥ 2010):"),
    dcc.DatePickerSingle(
        id='forecast-date-picker',
        min_date_allowed=min_forecast_date,
        max_date_allowed='2099-12-31',  # Allow future dates
        initial_visible_month=min_forecast_date,
        date=min_forecast_date,
    ),

    html.Div(
        children=[
            # Textual output
            html.Div(id='forecast-output-partial', style={'whiteSpace': 'pre-wrap', 'marginTop': 20}),
            
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
    if not forecast_date_str or not site:
        return ("Please select a site and a valid date.", go.Figure(), go.Figure())
    
    forecast_date = pd.to_datetime(forecast_date_str)
    result = forecast_for_date(raw_data, forecast_date, site)
    
    if not result:
        msg = (f"No forecast possible for Site={site} using Forecast Date={forecast_date.date()}.\n"
               "Possibly not enough training data.")
        return (msg, go.Figure(), go.Figure())
    
    text_output = format_forecast_output(result)
    
    level_fig = create_level_range_graph(
        result['Predicted_DA_Levels_Q05'],
        result['Predicted_DA_Levels_Q50'],
        result['Predicted_DA_Levels_Q95'],
        result['Actual_DA_Levels']
    )
    
    category_fig = create_category_graph(
        result['Probabilities'],
        result['Predicted_DA_Category'],
        result['Actual_DA_Category']
    )
    
    return (text_output, level_fig, category_fig)

# ================================
# MAIN
# ================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8065)
