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
# 3) TRAIN & PREDICT
# ================================
def train_and_predict(data):
    """
    Splits data into train/test sets, applies transformations, and fits:
      - Regression models (quantile GBM: Q05, Q50, Q95)
      - Classification model (RandomForestClassifier)

    Returns a dictionary with metrics + predictions for each task.
    """
    # A) REGRESSION PART
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(
        data_reg, test_size=0.2, random_state=42
    )

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
    X_test_reg_processed = preprocessor_reg.transform(X_test_reg)

    # Train 3 quantile regressors
    gb_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gb_q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50, random_state=42)
    gb_q95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)

    gb_q05.fit(X_train_reg_processed, y_train_reg)
    gb_q50.fit(X_train_reg_processed, y_train_reg)
    gb_q95.fit(X_train_reg_processed, y_train_reg)

    # Predict
    pred_q05 = gb_q05.predict(X_test_reg_processed)
    pred_q50 = gb_q50.predict(X_test_reg_processed)
    pred_q95 = gb_q95.predict(X_test_reg_processed)

    test_set_reg = test_set_reg.copy()
    test_set_reg['Predicted_DA_Levels_Q05'] = pred_q05
    test_set_reg['Predicted_DA_Levels_Q50'] = pred_q50
    test_set_reg['Predicted_DA_Levels_Q95'] = pred_q95

    # Evaluate (point forecast on Q50)
    overall_r2_reg = r2_score(y_test_reg, pred_q50)
    overall_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, pred_q50))

    # Coverage (actual within Q05-Q95)
    overall_coverage_reg = coverage(y_test_reg, pred_q05, pred_q95)

    # Pinball Losses
    pb_loss_q05 = pinball_loss(y_test_reg, pred_q05, alpha=0.05)
    pb_loss_q50 = pinball_loss(y_test_reg, pred_q50, alpha=0.50)
    pb_loss_q95 = pinball_loss(y_test_reg, pred_q95, alpha=0.95)

    # Site-level metrics
    def site_metrics_reg(grp):
        actual = grp['DA_Levels']
        q05_ = grp['Predicted_DA_Levels_Q05']
        q50_ = grp['Predicted_DA_Levels_Q50']
        q95_ = grp['Predicted_DA_Levels_Q95']
        return pd.Series({
            'r2': r2_score(actual, q50_),
            'rmse': np.sqrt(mean_squared_error(actual, q50_)),
            'coverage_90': coverage(actual, q05_, q95_),
        })
    site_stats_reg = test_set_reg.groupby('Site').apply(site_metrics_reg)

    # B) CLASSIFICATION PART
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(
        data_cls, test_size=0.2, random_state=42
    )

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
    X_test_cls_processed = preprocessor_cls.transform(X_test_cls)

    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_cls_processed, y_train_cls)

    y_pred_cls = rf_classifier.predict(X_test_cls_processed)
    y_pred_cls_proba = rf_classifier.predict_proba(X_test_cls_processed)

    test_set_cls = test_set_cls.copy()
    test_set_cls['Predicted_DA_Category'] = y_pred_cls

    # Probability columns
    class_labels = rf_classifier.classes_  # e.g., [0,1,2,3]
    for i, lbl in enumerate(class_labels):
        test_set_cls[f'Prob_Category_{lbl}'] = y_pred_cls_proba[:, i]

    # Classification metrics
    overall_accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
    overall_logloss_cls = log_loss(y_test_cls, y_pred_cls_proba)

    # If you have a binary scenario for Brier, you can do this for class=1:
    # but for multi-class you can adapt or skip
    if len(class_labels) == 2 and 1 in class_labels:
        # Probability of class=1:
        proba_class_1 = y_pred_cls_proba[:, list(class_labels).index(1)]
        overall_brier_cls = brier_score_loss(y_test_cls, proba_class_1, pos_label=1)
    else:
        # For multi-class, Brier score can be computed in a loop or just skip
        overall_brier_cls = None

    site_stats_cls = test_set_cls.groupby('Site').apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )

    return {
        "DA_Level": {
            "test_df": test_set_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_rmse": overall_rmse_reg,
            "overall_coverage": overall_coverage_reg,
            "pinball_losses": {
                "q05": pb_loss_q05,
                "q50": pb_loss_q50,
                "q95": pb_loss_q95
            }
        },
        "DA_Category": {
            "test_df": test_set_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls,
            "overall_logloss": overall_logloss_cls,
            "overall_brier": overall_brier_cls
        }
    }


# ================================
# 4) FORECAST FUNCTION (PARTIAL)
# ================================
def forecast_next_date(df, anchor_date, site):
    """
    Train on data up to anchor_date (per site), forecast the next date > anchor_date.
    Returns a dict with Q05/Q50/Q95, classification probabilities, plus
    single-date coverage/logloss/brier if feasible.
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

    # Pipeline for numeric columns
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

    # Train quantile regressors
    gb_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42)
    gb_q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50, random_state=42)
    gb_q95 = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42)

    gb_q05.fit(X_train_reg_processed, y_train_reg)
    gb_q50.fit(X_train_reg_processed, y_train_reg)
    gb_q95.fit(X_train_reg_processed, y_train_reg)

    y_pred_reg_q05 = gb_q05.predict(X_test_reg_processed)
    y_pred_reg_q50 = gb_q50.predict(X_test_reg_processed)
    y_pred_reg_q95 = gb_q95.predict(X_test_reg_processed)

    # Single-date coverage: 1 if actual is in [Q05, Q95], else 0
    actual_levels = float(y_test_reg.iloc[0]) if len(y_test_reg) > 0 else None
    if actual_levels is not None:
        covered = (actual_levels >= y_pred_reg_q05[0]) and (actual_levels <= y_pred_reg_q95[0])
        single_coverage = 1.0 if covered else 0.0
    else:
        single_coverage = None

    # Optionally compute single-date pinball losses
    def pinball_loss_single(y_true, y_pred, alpha):
        diff = y_true - y_pred
        return alpha * diff if diff > 0 else (alpha - 1) * diff

    single_pinball_q05 = (pinball_loss_single(actual_levels, y_pred_reg_q05[0], 0.05)
                          if actual_levels is not None else None)
    single_pinball_q50 = (pinball_loss_single(actual_levels, y_pred_reg_q50[0], 0.50)
                          if actual_levels is not None else None)
    single_pinball_q95 = (pinball_loss_single(actual_levels, y_pred_reg_q95[0], 0.95)
                          if actual_levels is not None else None)

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

    # Single-date log loss: only compute if the classifier has >1 class
    single_logloss = None
    if len(cls_model.classes_) > 1 and actual_cat is not None:
        from sklearn.metrics import log_loss
        single_logloss = log_loss(
            [actual_cat],
            [prob_list],
            labels=cls_model.classes_
        )

    # Single-date Brier Score (only if binary)
    single_brier = None
    if len(cls_model.classes_) == 2 and actual_cat is not None:
        # Assume classes_ might be [0,1] or [1,2]—check if '1' is present
        if 1 in cls_model.classes_:
            idx_class1 = list(cls_model.classes_).index(1)
            prob_class1 = prob_list[idx_class1]
            y_true_binary = 1 if actual_cat == 1 else 0
            single_brier = (prob_class1 - y_true_binary) ** 2

    # Return info
    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,

        # Regression predictions
        'Predicted_DA_Levels_Q05': float(y_pred_reg_q05[0]),
        'Predicted_DA_Levels_Q50': float(y_pred_reg_q50[0]),
        'Predicted_DA_Levels_Q95': float(y_pred_reg_q95[0]),
        'Actual_DA_Levels': actual_levels,
        'SingleDateCoverage': single_coverage,    # 1 or 0 if actual_levels is known
        'Pinball_Q05': single_pinball_q05,
        'Pinball_Q50': single_pinball_q50,
        'Pinball_Q95': single_pinball_q95,

        # Classification predictions
        'Predicted_DA_Category': pred_cat,
        'Probabilities': prob_list,
        'Actual_DA_Category': actual_cat,
        'SingleDateLogLoss': single_logloss,
        'SingleDateBrier': single_brier
    }


# ================================
# 5) BUILD DASH APP
# ================================
app = dash.Dash(__name__)

file_path = 'final_output.csv'  # Or your data file
raw_data = load_and_prepare_data(file_path)

# -- Run the main train_and_predict for analysis
predictions = train_and_predict(raw_data)

# ================================
# 6) RANDOM 200 APPROACH
# ================================

df_after_2010 = raw_data[raw_data['Date'].dt.year >= 2010].copy()
df_after_2010.sort_values(['Site', 'Date'], inplace=True)

pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()
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

# Compute performance on these single-step forecasts:
if not df_results_200.empty:
    # -- Classification accuracy (DA_Category)
    acc_200 = (
        df_results_200['Actual_DA_Category'] == df_results_200['Predicted_DA_Category']
    ).mean()
    
    # (Below lines are for the regression metrics, if you still want them)
    rmse_200 = np.sqrt(mean_squared_error(
        df_results_200['Actual_DA_Levels'], df_results_200['Predicted_DA_Levels_Q50']
    ))
    coverage_values = []
    for idx, row in df_results_200.iterrows():
        act = row['Actual_DA_Levels']
        lo = row['Predicted_DA_Levels_Q05']
        hi = row['Predicted_DA_Levels_Q95']
        coverage_values.append(1 if (act >= lo) and (act <= hi) else 0)
    coverage_200 = np.mean(coverage_values)
else:
    rmse_200 = None
    acc_200 = None
    coverage_200 = None

# -------------------------------
# CHANGE HERE: plot Actual vs. Predicted DA_Category
# -------------------------------
if not df_results_200.empty:
    df_line = df_results_200.copy().sort_values('NextDate')
    
    # Melt actual vs. predicted categories for easy plotting
    df_plot_melt = df_line.melt(
        id_vars=['Site', 'NextDate'],
        value_vars=['Actual_DA_Category', 'Predicted_DA_Category'],
        var_name='Type',
        value_name='DA_Category'
    )
    
    # Use a scatter plot (or line plot) to compare categories
    fig_random_line = px.scatter(
        df_plot_melt,
        x='NextDate',
        y='DA_Category',
        color='Type',
        hover_data=['Site'],
        title="Random 200 Next-Date Forecast (After 2010) - DA Category"
    )
    fig_random_line.update_layout(xaxis_title='Next Date', yaxis_title='DA Category')
else:
    fig_random_line = px.line(title="No valid data for random 200 approach")


# ================================
# 7) DASH LAYOUT
# ================================
analysis_layout = html.Div([
    html.H3("Overall Analysis (Time-Series)"),
    dcc.Dropdown(
        id='forecast-type-dropdown',
        options=[
            {'label': 'DA Levels (Regression)', 'value': 'DA_Level'},
            {'label': 'DA Category (Classification)', 'value': 'DA_Category'}
        ],
        value='DA_Level',
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': 'All Sites', 'value': 'All Sites'}] +
                [{'label': s, 'value': s} for s in raw_data['Site'].unique()],
        value='All Sites',
        style={'width': '50%'}
    ),
    dcc.Graph(id='analysis-graph')
])

valid_dates = sorted(raw_data['Date'].unique())
all_dates_range = pd.date_range(valid_dates[0], valid_dates[-1], freq='D')
# For Dash >= 2.18, pass 'disabled_days' as datetime.date objects:
disabled_days = [d.date() for d in all_dates_range if d not in valid_dates]

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
        disabled_days=disabled_days,
        date=valid_dates[0]
    ),
    html.Div(id='forecast-output-partial', style={'whiteSpace': 'pre-wrap', 'marginTop': 20})
])

random_200_layout = html.Div([
    html.H3("Random 200 Anchor Dates (Post-2010) Forecast -> Next Date"),

    # The updated graph now shows categories instead of levels:
    dcc.Graph(figure=fig_random_line),

    html.Div([
        html.H4("Overall Performance on These 200 Single-Step Forecasts"),
        html.Ul([
            # Keep or remove regression metrics as needed:
            html.Li(f"RMSE (DA Levels, Q50): {rmse_200:.3f}") if rmse_200 is not None else html.Li("No RMSE"),
            html.Li(f"Coverage (Q05-Q95): {coverage_200:.3f}") if coverage_200 is not None else html.Li("No coverage"),
            html.Li(f"Accuracy (DA Category): {acc_200:.3f}") if acc_200 is not None else html.Li("No Accuracy")
        ])
    ], style={'marginTop': 20})
])

app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Analysis', children=[analysis_layout]),
        dcc.Tab(label='Forecast by Date & Site', children=[forecast_layout]),
        dcc.Tab(label='Random 200 Next-Date Forecast', children=[random_200_layout]),
    ])
])


# ================================
# 8) CALLBACKS
# ================================
@app.callback(
    Output('analysis-graph', 'figure'),
    [Input('forecast-type-dropdown', 'value'),
     Input('site-dropdown', 'value')]
)
def update_analysis_graph(selected_forecast_type, selected_site):
    if selected_forecast_type == 'DA_Level':
        df_plot = predictions['DA_Level']['test_df'].copy()
        site_stats = predictions['DA_Level']['site_stats']
        overall_r2 = predictions['DA_Level']['overall_r2']
        overall_rmse = predictions['DA_Level']['overall_rmse']
        overall_cov = predictions['DA_Level']['overall_coverage']
        pb_losses = predictions['DA_Level']['pinball_losses']

        if selected_site == 'All Sites':
            performance_text = (f"Overall R²={overall_r2:.2f}, RMSE={overall_rmse:.2f}, "
                                f"Coverage(90%)={overall_cov:.2f}, "
                                f"Pinball(Q05)={pb_losses['q05']:.3f}, "
                                f"Q50={pb_losses['q50']:.3f}, "
                                f"Q95={pb_losses['q95']:.3f}")
        else:
            df_plot = df_plot[df_plot['Site'] == selected_site]
            if selected_site in site_stats.index:
                site_r2 = site_stats.loc[selected_site, 'r2']
                site_rmse = site_stats.loc[selected_site, 'rmse']
                site_cov = site_stats.loc[selected_site, 'coverage_90']
                performance_text = (f"Site R²={site_r2:.2f}, RMSE={site_rmse:.2f}, "
                                    f"Coverage(90%)={site_cov:.2f}")
            else:
                performance_text = "No data for that site."

        df_plot = df_plot.sort_values('Date')

        # Build a go.Figure with actual, plus Q05-Q95 ribbon and Q50 line
        fig = go.Figure()

        # Actual
        fig.add_trace(
            go.Scatter(
                x=df_plot['Date'],
                y=df_plot['DA_Levels'],
                mode='lines',
                name='Actual DA_Levels',
                line=dict(color='blue')
            )
        )

        # Q05
        fig.add_trace(
            go.Scatter(
                x=df_plot['Date'],
                y=df_plot['Predicted_DA_Levels_Q05'],
                mode='lines',
                line=dict(width=0),
                name='Q05',
                showlegend=False
            )
        )

        # Q95 (fill to previous trace -> ribbon)
        fig.add_trace(
            go.Scatter(
                x=df_plot['Date'],
                y=df_plot['Predicted_DA_Levels_Q95'],
                mode='lines',
                fill='tonexty',
                line=dict(width=0, color='lightgrey'),
                name='Q05-Q95 band'
            )
        )

        # Q50
        fig.add_trace(
            go.Scatter(
                x=df_plot['Date'],
                y=df_plot['Predicted_DA_Levels_Q50'],
                mode='lines',
                name='Predicted (Q50)',
                line=dict(color='red')
            )
        )

        fig.update_layout(
            title=f"DA Levels Forecast - {selected_site}",
            xaxis_title='Date',
            yaxis_title='DA Level',
            annotations=[
                dict(
                    xref='paper', yref='paper',
                    x=0.5, y=-0.2,
                    xanchor='center', yanchor='top',
                    text=performance_text,
                    showarrow=False
                )
            ]
        )
        return fig

    else:
        # Classification
        df_plot = predictions['DA_Category']['test_df'].copy()
        site_stats = predictions['DA_Category']['site_stats']
        overall_accuracy = predictions['DA_Category']['overall_accuracy']
        overall_logloss = predictions['DA_Category']['overall_logloss']
        overall_brier = predictions['DA_Category']['overall_brier']

        if selected_site == 'All Sites':
            performance_text = (f"Overall Accuracy={overall_accuracy:.2f}, "
                                f"LogLoss={overall_logloss:.3f}, "
                                f"Brier={overall_brier if overall_brier else 'N/A'}")
        else:
            df_plot = df_plot[df_plot['Site'] == selected_site]
            if selected_site in site_stats.index:
                site_acc = site_stats.loc[selected_site]
                performance_text = f"Site Accuracy={site_acc:.2f}"
            else:
                performance_text = "No data for that site."

        df_plot = df_plot.sort_values('Date')

        # Simple line for actual vs predicted category
        fig = px.line(
            df_plot,
            x='Date',
            y=['DA_Category', 'Predicted_DA_Category'],
            title=f"DA Category Forecast - {selected_site}"
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Category',
            annotations=[
                dict(
                    xref='paper', yref='paper',
                    x=0.5, y=-0.2,
                    xanchor='center', yanchor='top',
                    text=performance_text,
                    showarrow=False
                )
            ]
        )
        return fig


@app.callback(
    Output('forecast-output-partial', 'children'),
    [Input('forecast-date-picker', 'date'),
     Input('site-dropdown-forecast', 'value')]
)
def partial_forecast_callback(anchor_date_str, site):
    if not anchor_date_str or not site:
        return "Please select a site and a valid date."

    anchor_date = pd.to_datetime(anchor_date_str)
    result = forecast_next_date(raw_data, anchor_date, site)
    if result is None:
        return (f"No forecast possible for Site={site} after {anchor_date.date()}. "
                "Possibly no future date or no training data up to that date.")

    # Extract data
    q05 = result['Predicted_DA_Levels_Q05']
    q50 = result['Predicted_DA_Levels_Q50']
    q95 = result['Predicted_DA_Levels_Q95']
    actual_levels = result['Actual_DA_Levels']
    single_cov = result['SingleDateCoverage']

    pred_cat = result['Predicted_DA_Category']
    actual_cat = result['Actual_DA_Category']
    prob_list = result['Probabilities']

    single_logloss = result['SingleDateLogLoss']
    single_brier = result['SingleDateBrier']

    lines = [
        f"Anchor Date (training cut-off): {result['AnchorDate'].date()}",
        f"Next Date (forecast target): {result['NextDate'].date()}",
        "",
        f"--- Regression ---",
        f"Q05: {q05:.2f}, Q50: {q50:.2f}, Q95: {q95:.2f}",
    ]
    if actual_levels is not None:
        lines.append(f"Actual DA Level: {actual_levels:.2f}")
    if single_cov is not None:
        lines.append(f"Single-Date Coverage (1 if Actual in [Q05,Q95] else 0): {single_cov}")

    lines += [
        "",
        f"--- Classification ---",
        f"Predicted Category: {pred_cat}, Probabilities={prob_list}",
    ]
    if actual_cat is not None:
        lines.append(f"Actual Category: {actual_cat} "
                     f"({'MATCH' if pred_cat == actual_cat else 'MISMATCH'})")

    if single_logloss is not None:
        lines.append(f"Single-Date Log Loss: {single_logloss:.3f}")
    if single_brier is not None:
        lines.append(f"Single-Date Brier Score: {single_brier:.3f}")

    return "\n".join(lines)


# ================================
# 9) MAIN
# ================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8065)
