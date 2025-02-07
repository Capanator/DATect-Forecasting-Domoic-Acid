import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ================================
# 1) UTILITY FUNCTIONS
# ================================
def pinball_loss(y_true, y_pred, alpha):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    loss = np.where(diff > 0, alpha * diff, (alpha - 1) * diff)
    return np.mean(loss)

def coverage(y_true, lower, upper):
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)
    return np.mean((y_true >= lower) & (y_true <= upper))

# ================================
# 2) LOAD & PREPARE DATA
# ================================
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')
    data['Year'] = data['Date'].dt.year

    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    cluster_cols = [col for col in data.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        data[f'{col}_sin_day_of_year'] = data[col] * data['sin_day_of_year']
        data[f'{col}_cos_day_of_year'] = data[col] * data['cos_day_of_year']

    def categorize_da_levels(x):
        if x <= 5: return 0
        elif x <= 20: return 1
        elif x <= 40: return 2
        else: return 3
    data['DA_Category'] = data['DA_Levels'].apply(categorize_da_levels)

    return data

# ================================
# 3) MODIFIED FORECAST FUNCTION
# ================================
def forecast_next_date(df, anchor_date, next_date, site):
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)

    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        return None

    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        return None

    # Regression setup
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
        [('num', reg_preproc, num_cols_reg)],
        remainder='passthrough'
    )

    X_train_reg_processed = col_trans_reg.fit_transform(X_train_reg)
    X_test_reg_processed = col_trans_reg.transform(X_test_reg)

    gb_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gb_q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50, random_state=42)
    gb_q95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)

    gb_q05.fit(X_train_reg_processed, y_train_reg)
    gb_q50.fit(X_train_reg_processed, y_train_reg)
    gb_q95.fit(X_train_reg_processed, y_train_reg)

    y_pred_reg_q05 = gb_q05.predict(X_test_reg_processed)
    y_pred_reg_q50 = gb_q50.predict(X_test_reg_processed)
    y_pred_reg_q95 = gb_q95.predict(X_test_reg_processed)

    actual_levels = float(y_test_reg.iloc[0]) if len(y_test_reg) > 0 else None
    single_coverage = None
    if actual_levels is not None:
        covered = (actual_levels >= y_pred_reg_q05[0]) and (actual_levels <= y_pred_reg_q95[0])
        single_coverage = 1.0 if covered else 0.0

    # Classification setup
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
        [('num', cls_preproc, num_cols_cls)],
        remainder='passthrough'
    )

    X_train_cls_processed = col_trans_cls.fit_transform(X_train_cls)
    X_test_cls_processed = col_trans_cls.transform(X_test_cls)

    cls_model = RandomForestClassifier(random_state=42)
    cls_model.fit(X_train_cls_processed, y_train_cls)

    y_pred_cls = cls_model.predict(X_test_cls_processed)
    y_pred_cls_proba = cls_model.predict_proba(X_test_cls_processed)

    actual_cat = int(y_test_cls.iloc[0]) if len(y_test_cls) > 0 else None
    pred_cat = int(y_pred_cls[0])
    prob_list = list(y_pred_cls_proba[0])

    single_logloss = None
    if len(cls_model.classes_) > 1 and actual_cat is not None:
        single_logloss = log_loss(
            [actual_cat],
            [prob_list],
            labels=cls_model.classes_
        )

    return {
        'AnchorDate': anchor_date,
        'SelectedDate': next_date,
        'NextDate': next_date,
        'Predicted_DA_Levels_Q05': float(y_pred_reg_q05[0]),
        'Predicted_DA_Levels_Q50': float(y_pred_reg_q50[0]),
        'Predicted_DA_Levels_Q95': float(y_pred_reg_q95[0]),
        'Actual_DA_Levels': actual_levels,
        'SingleDateCoverage': single_coverage,
        'Predicted_DA_Category': pred_cat,
        'Probabilities': prob_list,
        'Actual_DA_Category': actual_cat,
        'SingleDateLogLoss': single_logloss,
    }

# ================================
# 4) DASH APP SETUP
# ================================
app = dash.Dash(__name__)
file_path = 'final_output.csv'  # Update this path if needed
raw_data = load_and_prepare_data(file_path)

app.layout = html.Div([
    html.H3("DA Forecast Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Site:"),
        dcc.Dropdown(
            id='site-dropdown',
            options=[{'label': s, 'value': s} for s in raw_data['Site'].unique()],
            value=raw_data['Site'].unique()[0],
            clearable=False
        )
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div([
        html.Label("Select Forecast Date:"),
        dcc.DatePickerSingle(
            id='date-picker',
            min_date_allowed=raw_data['Date'].min(),
            max_date_allowed=raw_data['Date'].max(),
            initial_visible_month=raw_data['Date'].min(),
            date=raw_data['Date'].min()
        )
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div(id='output-container', style={'marginTop': 20}),
    dcc.Graph(id='da-level-plot'),
    dcc.Graph(id='da-category-plot')
])

# ================================
# 5) MODIFIED CALLBACK
# ================================
@app.callback(
    [Output('output-container', 'children'),
     Output('da-level-plot', 'figure'),
     Output('da-category-plot', 'figure')],
    [Input('date-picker', 'date'),
     Input('site-dropdown', 'value')]
)
def update_output(selected_date, site):
    if not selected_date or not site:
        return ("Please select a date and site.", {}, {})
    
    selected_date = pd.to_datetime(selected_date)
    site_data = raw_data[raw_data['Site'] == site].copy()
    
    # Find anchor date (last date <= selected_date)
    mask = site_data['Date'] <= selected_date
    if not mask.any():
        return (f"No historical data for {site} before {selected_date.date()}", {}, {})
    anchor_date = site_data[mask]['Date'].max()
    
    # Find forecast target date (first date >= selected_date)
    future_dates = site_data[site_data['Date'] >= selected_date]['Date']
    if future_dates.empty:
        return (f"No future data for {site} after {selected_date.date()}", {}, {})
    next_date = future_dates.min()
    
    result = forecast_next_date(raw_data, anchor_date, next_date, site)
    if not result:
        return ("Forecast failed for selected parameters.", {}, {})

    # Text output
    CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
    q05 = result['Predicted_DA_Levels_Q05']
    q50 = result['Predicted_DA_Levels_Q50']
    q95 = result['Predicted_DA_Levels_Q95']
    
    text_output = [
        html.H4(f"Forecast for {result['NextDate'].strftime('%Y-%m-%d')}"),
        html.P(f"Training cutoff: {result['AnchorDate'].strftime('%Y-%m-%d')}"),
        html.Hr(),
        html.H5("DA Levels Forecast:"),
        html.Ul([
            html.Li(f"Q05: {q05:.2f}"),
            html.Li(f"Q50 (Median): {q50:.2f}"),
            html.Li(f"Q95: {q95:.2f}")
        ]),
        html.H5("DA Category Forecast:"),
        html.Ul([
            html.Li(f"Predicted: {CATEGORY_LABELS[result['Predicted_DA_Category']]}"),
            html.Li("Probabilities: " + ", ".join(
                [f"{label}: {p*100:.1f}%" 
                 for label, p in zip(CATEGORY_LABELS, result['Probabilities'])]
            ))
        ])
    ]

    # DA Level Plot with gradient
    fig_level = go.Figure()
    n_segments = 50
    max_distance = max(q50 - q05, q95 - q50)
    base_color = (70, 130, 180)  # Steel blue
    
    for i in range(n_segments):
        x0 = q05 + (i/n_segments)*(q95 - q05)
        x1 = q05 + ((i+1)/n_segments)*(q95 - q05)
        midpoint = (x0 + x1)/2
        distance = abs(midpoint - q50)
        opacity = 1 - (distance/max_distance)**0.5
        
        fig_level.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.4, y1=0.6,
            line=dict(width=0),
            fillcolor=f'rgba{(*base_color, opacity)}',
            layer='below'
        )

    fig_level.add_trace(go.Scatter(
        x=[q50, q50],
        y=[0.4, 0.6],
        mode='lines',
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='Median'
    ))

    if result['Actual_DA_Levels'] is not None:
        fig_level.add_trace(go.Scatter(
            x=[result['Actual_DA_Levels']],
            y=[0.5],
            mode='markers',
            marker=dict(size=18, color='red', symbol='x-thin', line=dict(width=2)),
            name='Actual Value'
        ))

    fig_level.update_layout(
        title=f"DA Level Forecast Range for {result['NextDate'].strftime('%Y-%m-%d')}",
        xaxis_title="DA Level",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
        height=300,
        plot_bgcolor='white'
    )

    # DA Category Plot
    fig_category = go.Figure()
    colors = ['#1f77b4'] * 4
    colors[result['Predicted_DA_Category']] = '#2ca02c'
    
    fig_category.add_trace(go.Bar(
        x=CATEGORY_LABELS,
        y=result['Probabilities'],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in result['Probabilities']],
        textposition='auto'
    ))

    if result['Actual_DA_Category'] is not None:
        fig_category.add_annotation(
            x=result['Actual_DA_Category'],
            y=result['Probabilities'][result['Actual_DA_Category']] + 0.05,
            text="Actual",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            font=dict(color='red')
        )

    fig_category.update_layout(
        title=f"Category Probabilities for {result['NextDate'].strftime('%Y-%m-%d')}",
        yaxis=dict(title="Probability", range=[0, 1.1]),
        xaxis_title="Category",
        showlegend=False,
        height=400
    )

    return (text_output, fig_level, fig_category)

# ================================
# 6) RUN SERVER
# ================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8067)