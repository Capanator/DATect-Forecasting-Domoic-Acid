import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
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
# NEW: Function to create a synthetic forecast row
# ================================
def create_forecast_row(last_row, forecast_date):
    """
    Create a synthetic forecast row based on the last available row.
    Only the lag features and DA_Category are preserved.
    The target values are removed.
    """
    new_row = last_row.copy()
    new_row['Date'] = forecast_date
    new_row['DA_Levels'] = np.nan
    new_row['DA_Category'] = np.nan
    return new_row

# ================================
# 2) LOAD & PREPARE DATA
# ================================
def load_and_prepare_data(file_path):
    # Load data from a Parquet file instead of CSV
    data = pd.read_parquet(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # Only keep Lag Features for DA_Levels.
    for lag in [1, 2, 3, 7, 14, 28, 56]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

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
# 3) FORECAST FUNCTION (UPDATED)
# ================================
def forecast_for_date(df, forecast_date, site):
    """
    Given a forecast_date (the date you enter) and a site,
    determine the training anchor (last date before forecast_date)
    and the test date (the date that is on or immediately after forecast_date).
    If forecast_date is beyond the available data, create a synthetic forecast row.
    
    Train models on data up to the anchor date and forecast for the forecast point.
    Accuracy metrics are computed only if a real test row exists.
    """
    df_site = df[df['Site'] == site].copy()
    df_site.sort_values('Date', inplace=True)

    # Must have at least one historical date before forecast_date.
    df_before = df_site[df_site['Date'] < forecast_date]
    if df_before.empty:
        return None
    anchor_date = df_before['Date'].max()

    # Determine test date: earliest date on or after forecast_date (if any)
    df_after = df_site[df_site['Date'] >= forecast_date]
    if not df_after.empty:
        test_date = df_after['Date'].min()
    else:
        test_date = None

    # Determine the forecast row:
    if forecast_date in df_site['Date'].values:
        df_forecast = df_site[df_site['Date'] == forecast_date].copy()
    else:
        if test_date is not None:
            # If forecast_date is not present but a later date exists, use that row.
            df_forecast = df_site[df_site['Date'] == test_date].copy()
        else:
            # Forecast date is beyond available data—create a synthetic forecast row.
            last_row = df_site[df_site['Date'] == anchor_date].iloc[0]
            forecast_row = create_forecast_row(last_row, forecast_date)
            df_forecast = pd.DataFrame([forecast_row])

    # Training set: all data up to (and including) the anchor date.
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    if df_train.empty:
        return None

    # =======================
    # 1) REGRESSION (DA_Levels)
    # =======================
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    X_train_reg = df_train.drop(columns=drop_cols_reg, errors='ignore')
    y_train_reg = df_train['DA_Levels']

    X_forecast_reg = df_forecast.drop(columns=drop_cols_reg, errors='ignore')
    # If the forecast row is synthetic (or the target is missing), no actual value is available.
    if df_forecast['DA_Levels'].isnull().all():
        y_test_reg = None
    else:
        y_test_reg = df_forecast['DA_Levels']

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
    X_forecast_reg_processed = col_trans_reg.transform(X_forecast_reg)

    gb_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gb_q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50, random_state=42)
    gb_q95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)

    gb_q05.fit(X_train_reg_processed, y_train_reg)
    gb_q50.fit(X_train_reg_processed, y_train_reg)
    gb_q95.fit(X_train_reg_processed, y_train_reg)

    y_pred_reg_q05 = gb_q05.predict(X_forecast_reg_processed)
    y_pred_reg_q50 = gb_q50.predict(X_forecast_reg_processed)
    y_pred_reg_q95 = gb_q95.predict(X_forecast_reg_processed)

    if y_test_reg is not None:
        actual_levels = float(y_test_reg.iloc[0])
        covered = (actual_levels >= y_pred_reg_q05[0]) and (actual_levels <= y_pred_reg_q95[0])
        single_coverage = 1.0 if covered else 0.0
    else:
        actual_levels = None
        single_coverage = None

    # =======================
    # 2) CLASSIFICATION (DA_Category)
    # =======================
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    X_train_cls = df_train.drop(columns=drop_cols_cls, errors='ignore')
    y_train_cls = df_train['DA_Category']

    X_forecast_cls = df_forecast.drop(columns=drop_cols_cls, errors='ignore')
    if df_forecast['DA_Category'].isnull().all():
        y_test_cls = None
    else:
        y_test_cls = df_forecast['DA_Category']

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
    X_forecast_cls_processed = col_trans_cls.transform(X_forecast_cls)

    cls_model = RandomForestClassifier(random_state=42)
    cls_model.fit(X_train_cls_processed, y_train_cls)

    y_pred_cls = cls_model.predict(X_forecast_cls_processed)
    y_pred_cls_proba = cls_model.predict_proba(X_forecast_cls_processed)

    if y_test_cls is not None:
        actual_cat = int(y_test_cls.iloc[0])
    else:
        actual_cat = None

    pred_cat = int(y_pred_cls[0])
    prob_list = list(y_pred_cls_proba[0])

    single_logloss = None
    single_brier = None
    if y_test_cls is not None:
        if len(cls_model.classes_) > 1 and actual_cat is not None:
            single_logloss = log_loss(
                [actual_cat],
                [prob_list],
                labels=cls_model.classes_
            )
        if len(cls_model.classes_) == 2 and actual_cat is not None:
            if 1 in cls_model.classes_:
                idx_class1 = list(cls_model.classes_).index(1)
                prob_class1 = prob_list[idx_class1]
                y_true_binary = 1 if actual_cat == 1 else 0
                single_brier = (prob_class1 - y_true_binary) ** 2

    return {
        'ForecastPoint': forecast_date,
        'AnchorDate': anchor_date,
        'TestDate': test_date,
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

# Load and prepare data from a Parquet file
file_path = 'final_output_og.parquet'  # Updated path to Parquet file
raw_data = load_and_prepare_data(file_path)

# Allow selection of dates from the dataset—but restrict forecast dates to be on or after 2010.
min_forecast_date = pd.to_datetime("2010-01-01")
valid_dates = sorted(raw_data['Date'].unique())

forecast_layout = html.Div([
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

app.layout = html.Div([ forecast_layout ])

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
def partial_forecast_callback(forecast_date_str, site):
    if not forecast_date_str or not site:
        return ("Please select a site and a valid date.", go.Figure(), go.Figure())
    
    forecast_date = pd.to_datetime(forecast_date_str)
    result = forecast_for_date(raw_data, forecast_date, site)
    
    if not result:
        msg = (f"No forecast possible for Site={site} using Forecast Date={forecast_date.date()}.\n"
               "Possibly not enough training data.")
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
        lines.append(f"Actual Value: {actual_levels:.2f} "
                     f"({'Within Range ✅' if result['SingleDateCoverage'] else 'Outside Range ❌'})")
    else:
        lines.append("Actual Value: N/A (forecast beyond available data)")

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
    else:
        lines.append("Actual Category: N/A")

    # ---------------------------
    # Create DA_Levels Visualization
    # ---------------------------
    fig_level = go.Figure()
    n_segments = 50
    max_distance = max(q50 - q05, q95 - q50) if max(q50 - q05, q95 - q50) > 0 else 1
    base_color = (70, 130, 180)  # Steel blue

    # Add gradient background using multiple semi-transparent rectangles
    for i in range(n_segments):
        x0 = q05 + (i/n_segments)*(q95 - q05)
        x1 = q05 + ((i+1)/n_segments)*(q95 - q05)
        midpoint = (x0 + x1) / 2
        distance = abs(midpoint - q50)
        opacity = 1 - (distance / max_distance)**0.5
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
        name='Median (Q50)'
    ))
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
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=True,
        height=300,
        plot_bgcolor='white'
    )

    # ---------------------------
    # Create DA_Category Visualization
    # ---------------------------
    pred_cat = result['Predicted_DA_Category']
    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
    if 0 <= pred_cat < len(colors):
        colors[pred_cat] = '#2ca02c'
    
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