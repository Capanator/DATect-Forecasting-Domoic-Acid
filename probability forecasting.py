import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                             log_loss, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ================================
# 1) UTILITY FUNCTIONS
# ================================
def pinball_loss(y_true, y_pred, alpha):
    """
    Compute the pinball (quantile) loss for a given quantile alpha.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    # If diff > 0 => alpha*diff, else => (alpha-1)*diff
    loss = np.where(diff > 0, alpha * diff, (alpha - 1) * diff)
    return np.mean(loss)

def coverage(y_true, lower, upper):
    """
    Fraction of times y_true is within [lower, upper].
    """
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

    # Spatial Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # Seasonal features (sin/cos)
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

    # Categorize DA_Levels -> DA_Category
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

# ================================
# 3) FORECAST FUNCTION (PARTIAL)
# ================================
def forecast_next_date(df, anchor_date, site):
    """
    Train on data up to anchor_date (for a given site),
    then forecast the next date > anchor_date.
    Returns a dict with Q05/Q50/Q95, classification probabilities, 
    plus single-date coverage/logloss if feasible.
    """
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)

    # Next date (strictly > anchor_date)
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None
    next_date = df_future['Date'].iloc[0]

    # Training set: up to anchor_date
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        return None

    # Test set: exactly the next date
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_test.empty:
        return None

    # =======================
    # 1) REGRESSION (DA_Levels)
    # =======================
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
    if actual_levels is not None:
        covered = (actual_levels >= y_pred_reg_q05[0]) and (actual_levels <= y_pred_reg_q95[0])
        single_coverage = 1.0 if covered else 0.0
    else:
        single_coverage = None

    # =======================
    # 2) CLASSIFICATION (DA_Category)
    # =======================
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

    # Single-date log loss: only compute if actual is known and >1 class
    single_logloss = None
    if len(cls_model.classes_) > 1 and actual_cat is not None:
        single_logloss = log_loss(
            [actual_cat],
            [prob_list],
            labels=cls_model.classes_
        )

    # Single-date Brier Score (only if binary)
    single_brier = None
    if len(cls_model.classes_) == 2 and actual_cat is not None:
        if 1 in cls_model.classes_:
            idx_class1 = list(cls_model.classes_).index(1)
            prob_class1 = prob_list[idx_class1]
            y_true_binary = 1 if actual_cat == 1 else 0
            single_brier = (prob_class1 - y_true_binary) ** 2

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,

        # Regression predictions
        'Predicted_DA_Levels_Q05': float(y_pred_reg_q05[0]),
        'Predicted_DA_Levels_Q50': float(y_pred_reg_q50[0]),
        'Predicted_DA_Levels_Q95': float(y_pred_reg_q95[0]),
        'Actual_DA_Levels': actual_levels,
        'SingleDateCoverage': single_coverage,

        # Classification predictions
        'Predicted_DA_Category': pred_cat,
        'Probabilities': prob_list,
        'Actual_DA_Category': actual_cat,
        'SingleDateLogLoss': single_logloss,
        'SingleDateBrier': single_brier
    }

# ================================
# 4) BUILD DASH APP (ONLY TAB 2)
# ================================
app = dash.Dash(__name__)

# Load and prepare data
file_path = 'final_output.csv'  # Update if needed
raw_data = load_and_prepare_data(file_path)

# Filter out all dates before 2010 for forecasting
raw_data_after_2010 = raw_data[raw_data['Date'].dt.year >= 2010]
valid_dates = sorted(raw_data_after_2010['Date'].unique())
if not valid_dates:
    raise ValueError("No valid data on or after 2010 found in the dataset.")

forecast_layout = html.Div([
    html.H3("Forecast by Specific Date & Site (Training Up to Date ≥ 2010)"),
    
    html.Label("Choose a Site:"),
    dcc.Dropdown(
        id='site-dropdown-forecast',
        options=[{'label': s, 'value': s} for s in raw_data['Site'].unique()],
        value=raw_data['Site'].unique()[0],
        style={'width': '50%'}
    ),

    html.Label("Pick an Anchor Date (≥ 2010):"),
    dcc.DatePickerSingle(
        id='forecast-date-picker',
        min_date_allowed=valid_dates[0],  # First valid date on/after 2010
        max_date_allowed=valid_dates[-1],
        initial_visible_month=valid_dates[0],
        date=valid_dates[0],
        # We won't disable out-of-data days here, but you could 
        # if you want to replicate the entire "disabled_days" logic.
    ),

    html.Div(
        children=[
            # Textual output
            html.Div(id='forecast-output-partial', style={'whiteSpace': 'pre-wrap', 'marginTop': 20}),
            
            # New graphs for forecast ranges
            html.Div([
                dcc.Graph(id='level-range-graph', style={'display': 'inline-block', 'width': '49%'}),
                dcc.Graph(id='category-range-graph', style={'display': 'inline-block', 'width': '49%'})
            ])
        ],
        style={'marginTop': 30}
    )
])

# Make the layout just this single “Tab 2” content.
app.layout = html.Div([
    forecast_layout
])

# ================================
# 5) CALLBACK
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
def partial_forecast_callback(anchor_date_str, site):
    if not anchor_date_str or not site:
        return ("Please select a site and a valid date.", go.Figure(), go.Figure())
    
    anchor_date = pd.to_datetime(anchor_date_str)
    result = forecast_next_date(raw_data, anchor_date, site)
    
    if not result:
        msg = (f"No forecast possible for Site={site} after {anchor_date.date()}.\n"
               "Possibly no future date or no training data up to that date.")
        return (msg, go.Figure(), go.Figure())

    # Define category labels
    CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
    
    # Prepare formatted text output
    q05 = result['Predicted_DA_Levels_Q05']
    q50 = result['Predicted_DA_Levels_Q50']
    q95 = result['Predicted_DA_Levels_Q95']
    actual_levels = result['Actual_DA_Levels']
    prob_list = [round(p, 3) for p in result['Probabilities']]
    formatted_probs = [f"{p*100:.1f}%" for p in prob_list]

    # Build textual output
    lines = [
        f"Anchor Date (training cut-off): {result['AnchorDate'].date()}",
        f"Next Date (forecast target): {result['NextDate'].date()}",
        "",
        "--- Regression (DA_Levels) ---",
        f"Predicted Range: {q05:.2f} (Q05) – {q50:.2f} (Q50) – {q95:.2f} (Q95)",
    ]
    
    if actual_levels is not None:
        lines.append(f"Actual Value: {actual_levels:.2f} "
                     f"({'Within Range ✅' if result['SingleDateCoverage'] else 'Outside Range ❌'})")

    lines += [
        "",
        "--- Classification (DA_Category) ---",
        f"Predicted: {CATEGORY_LABELS[result['Predicted_DA_Category']]}",
        "Probabilities: " + ", ".join([
            f"{label}: {prob}" 
            for label, prob in zip(CATEGORY_LABELS, formatted_probs)
        ])
    ]
    
    if result['Actual_DA_Category'] is not None:
        match_status = "✅ MATCH" if result['Predicted_DA_Category'] == result['Actual_DA_Category'] else "❌ MISMATCH"
        lines.append(f"Actual: {CATEGORY_LABELS[result['Actual_DA_Category']]} {match_status}")

    # Create DA_Levels visualization with gradient effect
    fig_level = go.Figure()
    
    # Create gradient effect using multiple semi-transparent rectangles
    q05 = result['Predicted_DA_Levels_Q05']
    q50 = result['Predicted_DA_Levels_Q50']
    q95 = result['Predicted_DA_Levels_Q95']
    n_segments = 50  # More segments = smoother gradient
    max_distance = max(q50 - q05, q95 - q50)
    base_color = (70, 130, 180)  # Steel blue
    
    # Add gradient background
    for i in range(n_segments):
        x0 = q05 + (i/n_segments)*(q95 - q05)
        x1 = q05 + ((i+1)/n_segments)*(q95 - q05)
        midpoint = (x0 + x1) / 2
        distance = abs(midpoint - q50)
        opacity = 1 - (distance / max_distance)**0.5  # Square root for smoother falloff
        
        fig_level.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.4, y1=0.6,  # Vertical position of the range bar
            line=dict(width=0),
            fillcolor=f'rgba{(*base_color, opacity)}',
            layer='below'
        )

    # Add median line
    fig_level.add_trace(go.Scatter(
        x=[q50, q50],
        y=[0.4, 0.6],
        mode='lines',
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='Median (Q50)'
    ))

    # Add range boundaries
    fig_level.add_trace(go.Scatter(
        x=[q05, q95],
        y=[0.5, 0.5],
        mode='markers',
        marker=dict(
            size=15,
            color=['rgba(70, 130, 180, 0.3)', 'rgba(70, 130, 180, 0.3)'],
            symbol='line-ns-open'
        ),
        name='Prediction Range'
    ))

    # Add actual value if available
    if actual_levels is not None:
        fig_level.add_trace(go.Scatter(
            x=[actual_levels],
            y=[0.5],
            mode='markers',
            marker=dict(
                size=18,
                color='red',
                symbol='x-thin',
                line=dict(width=2)
            ),
            name='Actual Value'
        ))

    fig_level.update_layout(
        title="DA Level Forecast Range with Gradient Confidence",
        xaxis_title="DA Level",
        yaxis=dict(
            visible=False,
            range=[0, 1]
        ),
        showlegend=True,
        height=300,
        plot_bgcolor='white'
    )

    # Create DA_Category visualization
    pred_cat = result['Predicted_DA_Category']
    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
    if 0 <= pred_cat < len(colors):
        colors[pred_cat] = '#2ca02c'  # Highlight predicted category
    
    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        x=CATEGORY_LABELS,
        y=prob_list,
        marker_color=colors,
        text=formatted_probs,
        textposition='auto'
    ))
    
    if result['Actual_DA_Category'] is not None:
        actual_cat = result['Actual_DA_Category']
        fig_cat.add_annotation(
            x=actual_cat,
            y=prob_list[actual_cat] + 0.05,
            text="Actual",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            font=dict(color='red')
        )

    fig_cat.update_layout(
        title="Category Probability Distribution",
        yaxis=dict(title="Probability", range=[0, 1.1]),
        xaxis=dict(title="Category"),
        showlegend=False,
        height=400
    )

    return ("\n".join(lines), fig_level, fig_cat)

# ================================
# 6) MAIN
# ================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8065)
