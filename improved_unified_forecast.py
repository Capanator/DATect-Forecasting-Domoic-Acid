"""
Improved Unified Domoic Acid Forecasting Application
Fixes data leakage while preserving performance and original visualizations.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import random
from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
from joblib import Parallel, delayed
from tqdm import tqdm

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Configuration matching original
CONFIG = {
    "ENABLE_LAG_FEATURES": True,  # Now permanent - always enabled
    "ENABLE_LINEAR_LOGISTIC": True,
    "DATA_FILE": "final_output.parquet",
    "PORT_RETRO": 8071,
    "PORT_REALTIME": 8065,
    "NUM_RANDOM_ANCHORS_PER_SITE_EVAL": 50,  # Change this number to control forecasts per site
    "N_SPLITS_TS_GRIDSEARCH": 10,
    "MIN_TEST_DATE": "2008-01-01",
    "N_JOBS_EVAL": -1,
    "RANDOM_SEED": 42,
}

# No hyperparameter grids needed - using default parameters

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])


class ImprovedDAForecast:
    """Improved DA forecasting that fixes leakage but preserves performance."""
    
    def __init__(self):
        pass
        
    def create_numeric_transformer(self, df, drop_cols):
        """Create preprocessing transformer (from original)."""
        X = df.drop(columns=drop_cols, errors="ignore")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ])
        transformer = ColumnTransformer(
            [("num", numeric_pipeline, numeric_cols)],
            remainder="passthrough", 
            verbose_feature_names_out=False
        )
        transformer.set_output(transform="pandas")
        return transformer, X
    
    def add_lag_features(self, df, group_col, value_col, lags):
        """Add lag features (from original)."""
        df = df.copy()
        for lag in lags:
            df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
        return df
    
    def load_and_prepare_base_data(self, file_path):
        """Load and prepare base data (temporal features only - no leakage)."""
        print(f"[INFO] Loading {file_path}")
        data = pd.read_parquet(file_path, engine="pyarrow")
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values(["site", "date"], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Temporal features (safe - no future info)
        day_of_year = data["date"].dt.dayofyear
        data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
        data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

        # Target categorization (like past-forecasts-final.py)
        data["da-category"] = pd.cut(
            data["da"],
            bins=[-float("inf"), 5, 20, 40, float("inf")],
            labels=[0, 1, 2, 3],
            right=True,
        ).astype(pd.Int64Dtype())

        # Lag features (will be created per forecast to avoid leakage)
        if CONFIG["ENABLE_LAG_FEATURES"]:
            print("[INFO] Will create lag features per forecast")

        return data
    
    
    def get_model(self, task, model_type):
        """Get model based on task and model type (no hyperparameter tuning)."""
        if task == "regression":
            if model_type == "rf":
                return RandomForestRegressor(
                    n_estimators=100, 
                    random_state=CONFIG["RANDOM_SEED"], 
                    n_jobs=1
                )
            elif model_type == "linear":
                # Add regularization to prevent numerical overflow
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0, random_state=CONFIG["RANDOM_SEED"])
        elif task == "classification":
            if model_type == "rf":
                return RandomForestClassifier(
                    n_estimators=100, 
                    random_state=CONFIG["RANDOM_SEED"], 
                    n_jobs=1
                )
            elif model_type == "linear":
                return LogisticRegression(
                    solver="lbfgs", 
                    max_iter=1000, 
                    C=1.0,  # Add regularization
                    random_state=CONFIG["RANDOM_SEED"], 
                    n_jobs=1
                )
        else:
            raise ValueError(f"Unknown task/model combination: {task}/{model_type}")
    
    def forecast_single_anchor(self, anchor_info, full_data, min_target_date):
        """Process single anchor forecast (adapted from original but leak-free)."""
        site, anchor_date = anchor_info
        
        try:
            # Get site data
            site_data = full_data[full_data["site"] == site].copy()
            site_data.sort_values("date", inplace=True)
            
            # Add lag features if enabled
            if CONFIG["ENABLE_LAG_FEATURES"]:
                site_data = self.add_lag_features(site_data, "site", "da", [1, 2, 3])
            
            # Split data by anchor date (NO LEAKAGE - training only before anchor)
            train_df = site_data[site_data["date"] <= anchor_date].copy()
            test_df_candidates = site_data[site_data["date"] > anchor_date]
            
            if train_df.empty or test_df_candidates.empty:
                return None
                
            test_df_single_row = test_df_candidates.iloc[:1].copy()
            
            if test_df_single_row["date"].min() < min_target_date:
                return None
            
            # Always make both predictions (like past-forecasts-final.py)
            common_cols = ["date", "site"]
            drop_cols = common_cols + ["da", "da-category"]
            
            # Regression processing
            transformer_reg, X_train_reg = self.create_numeric_transformer(train_df, drop_cols)
            X_test_reg = test_df_single_row.drop(columns=drop_cols, errors="ignore")
            X_test_reg = X_test_reg.reindex(columns=X_train_reg.columns, fill_value=0)
            
            reg_model = self.get_model("regression", CONFIG["SELECTED_MODEL"])
            X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
            X_test_reg_processed = transformer_reg.transform(X_test_reg)
            reg_model.fit(X_train_reg_processed, train_df["da"])
            y_pred_reg = reg_model.predict(X_test_reg_processed)[0]
            
            # Classification processing
            transformer_cls, X_train_cls = self.create_numeric_transformer(train_df, drop_cols)
            X_test_cls = test_df_single_row.drop(columns=drop_cols, errors="ignore")
            X_test_cls = X_test_cls.reindex(columns=X_train_cls.columns, fill_value=0)
            
            # Check if we have multiple classes
            unique_classes = train_df["da-category"].dropna().nunique()
            if unique_classes < 2:
                y_pred_cls = np.nan
            else:
                cls_model = self.get_model("classification", CONFIG["SELECTED_MODEL"])
                X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
                X_test_cls_processed = transformer_cls.transform(X_test_cls)
                cls_model.fit(X_train_cls_processed, train_df["da-category"])
                y_pred_cls = cls_model.predict(X_test_cls_processed)[0]
            
            # Return result with both predictions
            result = test_df_single_row.copy()
            result["Predicted_da"] = y_pred_reg
            result["Predicted_da-category"] = y_pred_cls
            
            return result
            
        except Exception as e:
            print(f"Error processing anchor {site} at {anchor_date}: {e}")
            return None
    
    def evaluate_retrospective(self, data, n_anchors_per_site=200, min_test_date="2008-01-01"):
        """Run retrospective evaluation with leak-free anchors."""
        print(f"\n[INFO] Evaluating with {n_anchors_per_site} anchors per site")
        min_target_date = pd.Timestamp(min_test_date)
        
        # Generate anchor points
        anchor_infos = []
        for site in data["site"].unique():
            site_dates = data[data["site"] == site]["date"].sort_values().unique()
            if len(site_dates) > 1:
                # Only use dates that have future data for testing
                valid_anchors = [d for d in site_dates[:-1] if d >= min_target_date]
                if valid_anchors:
                    n_sample = min(len(valid_anchors), n_anchors_per_site)
                    selected_anchors = random.sample(list(valid_anchors), n_sample)
                    anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
        
        if not anchor_infos:
            print("[ERROR] No anchor points generated")
            return pd.DataFrame()
        
        print(f"[INFO] Generated {len(anchor_infos)} anchor points")
        
        # Process anchors in parallel
        results = Parallel(n_jobs=CONFIG["N_JOBS_EVAL"], verbose=1)(
            delayed(self.forecast_single_anchor)(ai, data, min_target_date) 
            for ai in tqdm(anchor_infos, desc="Processing Anchors")
        )
        
        # Filter successful results
        forecast_dfs = [df for df in results if df is not None]
        if not forecast_dfs:
            print("[ERROR] No successful forecasts")
            return pd.DataFrame()
            
        final_df = pd.concat(forecast_dfs).sort_values(["date", "site"]).drop_duplicates(["date", "site"])
        print(f"[INFO] Successfully processed {len(forecast_dfs)} forecasts")
        
        return final_df
    
    def forecast_realtime_single(self, data, forecast_date, site):
        """Single real-time forecast (from future-forecasts.py style but leak-free)."""
        try:
            # Get site data and sort
            df_site = data[data['site'] == site].copy()
            df_site.sort_values('date', inplace=True)
            
            # Add lag features
            if CONFIG["ENABLE_LAG_FEATURES"]:
                df_site = self.add_lag_features(df_site, "site", "da", [1, 2, 3])
            
            # Split data (NO LEAKAGE)
            df_before = df_site[df_site['date'] < forecast_date]
            if df_before.empty:
                return None
                
            anchor_date = df_before['date'].max()
            df_after = df_site[df_site['date'] >= forecast_date]
            test_date = df_after['date'].min() if not df_after.empty else None
            
            # Get forecast row (real or synthetic)
            if forecast_date in df_site['date'].values:
                df_forecast = df_site[df_site['date'] == forecast_date].copy()
            elif test_date is not None:
                df_forecast = df_site[df_site['date'] == test_date].copy()
            else:
                # Create synthetic forecast row
                last_row = df_site[df_site['date'] == anchor_date].iloc[0]
                new_row = last_row.copy()
                new_row['date'] = forecast_date
                new_row['da'] = np.nan
                df_forecast = pd.DataFrame([new_row])
            
            # Training data (categories already created globally)
            df_train = df_site[df_site['date'] <= anchor_date].copy()
            
            # Remove rows with NaN targets to prevent sklearn errors
            df_train = df_train.dropna(subset=['da'])
            if df_train.empty or len(df_train) < 5:
                return None
            
            # Feature processing
            base_drop_cols = ['date', 'site', 'da']
            train_drop_cols = base_drop_cols + ['da-category']  # Training data has categories
            forecast_drop_cols = base_drop_cols  # Forecast data doesn't have categories yet
            
            numeric_processor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])
            
            X_train_reg = df_train.drop(columns=[col for col in train_drop_cols if col in df_train.columns])
            y_train_reg = df_train['da']
            X_forecast_reg = df_forecast.drop(columns=[col for col in forecast_drop_cols if col in df_forecast.columns])
            
            # Preprocess features
            num_cols = X_train_reg.select_dtypes(include=[np.number]).columns
            preprocessor = ColumnTransformer([('num', numeric_processor, num_cols)], remainder='drop')
            
            X_train_processed = preprocessor.fit_transform(X_train_reg)
            X_forecast_processed = preprocessor.transform(X_forecast_reg)
            
            # Get actual values if available
            y_test_reg = None if df_forecast['da'].isnull().all() else df_forecast['da']
            actual_levels = float(y_test_reg.iloc[0]) if y_test_reg is not None else None
            
            # Real-time mode ALWAYS runs BOTH regression and classification (like original future-forecasts.py)
            # The --task flag only affects retrospective evaluation
            
            # REGRESSION: Train quantile regressors (Gradient Boosting)
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
            
            # Train standard regressor (Random Forest)
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=1
            )
            rf_model.fit(X_train_processed, y_train_reg)
            rf_pred = float(rf_model.predict(X_forecast_processed)[0])
            
            # Check coverage
            single_coverage = None
            if actual_levels is not None:
                single_coverage = 1.0 if gb_preds['q05'] <= actual_levels <= gb_preds['q95'] else 0.0
            
            # CLASSIFICATION: Train Random Forest classifier
            y_train_cls = df_train['da-category']
            unique_classes = y_train_cls.dropna().nunique()
            
            if unique_classes > 1:
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=1
                )
                clf.fit(X_train_processed, y_train_cls)
                pred_cat = int(clf.predict(X_forecast_processed)[0])
                prob_list = list(clf.predict_proba(X_forecast_processed)[0])
            else:
                pred_cat = None
                prob_list = None
            
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
                'site': site
            }
            
        except Exception as e:
            import traceback
            print(f"Error in real-time forecast for {site} at {forecast_date}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return None
    
    def create_level_range_graph(self, q05, q50, q95, actual_levels=None, rf_prediction=None):
        """Create gradient visualization for DA level forecast (from original future-forecasts.py)."""
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


class ImprovedForecastApp:
    """Improved forecast app with original-style dashboards."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.forecast_model = ImprovedDAForecast()
        self.data = None
        self.results_df = None
    
    def load_data(self):
        """Load data."""
        self.data = self.forecast_model.load_and_prepare_base_data(self.data_path)
        
    def run_retrospective_evaluation(self, n_anchors, min_test_date):
        """Run retrospective evaluation."""
        print(f"[INFO] Running {CONFIG['SELECTED_TASK']} evaluation with {CONFIG['SELECTED_MODEL']} model")
        
        # Run evaluation (no hyperparameter tuning)
        self.results_df = self.forecast_model.evaluate_retrospective(
            self.data, n_anchors, min_test_date
        )
        
        if self.results_df.empty:
            print("No results for evaluation")
            return
            
        # Print results (matching original format)
        df_reg = self.results_df.dropna(subset=["da", "Predicted_da"])
        
        # Handle both possible column name formats
        cat_cols = []
        if "da-category" in self.results_df.columns:
            cat_cols = ["da-category", "Predicted_da-category"]
        elif "da_category" in self.results_df.columns:
            cat_cols = ["da_category", "Predicted_da_category"]
        
        if cat_cols:
            df_cls = self.results_df.dropna(subset=cat_cols)
        else:
            df_cls = pd.DataFrame()  # Empty if no category columns found
        
        if not df_reg.empty:
            overall_r2 = r2_score(df_reg["da"], df_reg["Predicted_da"])
            overall_mae = mean_absolute_error(df_reg["da"], df_reg["Predicted_da"])
            print(f"[INFO] Overall Regression R2: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
            
        if not df_cls.empty and cat_cols:
            actual_col, pred_col = cat_cols
            overall_accuracy = accuracy_score(df_cls[actual_col], df_cls[pred_col])
            print(f"[INFO] Overall Classification Accuracy: {overall_accuracy:.4f}")
    
    def create_retrospective_dashboard(self, port):
        """Create retrospective dashboard with time series plots."""
        if self.results_df is None or self.results_df.empty:
            print("No results for dashboard")
            return
            
        app = dash.Dash(__name__)
        sites_list = sorted(self.results_df["site"].unique().tolist())

        app.layout = html.Div([
            html.H1("Domoic Acid Forecast Dashboard (Random Anchor Evaluation)"),
            html.Div([
                html.H3("Overall Analysis (Aggregated Random Anchor Forecasts)"),
                dcc.Dropdown(
                    id="site-dropdown",
                    options=[{"label": "All sites", "value": "All sites"}] + [
                        {"label": site, "value": site} for site in sites_list
                    ],
                    placeholder="Select site (or All sites)",
                    style={"width": "50%", "marginBottom": "15px"},
                ),
                dcc.Graph(id="analysis-graph"),
            ])
        ])

        @app.callback(
            Output("analysis-graph", "figure"),
            [Input("site-dropdown", "value")],
        )
        def update_graph(selected_site):
            df_plot = self.results_df.copy()
            
            if selected_site and selected_site != "All sites":
                df_plot = df_plot[df_plot["site"] == selected_site]
            
            # Show results based on the selected task
            if CONFIG["SELECTED_TASK"] == "regression":
                # Show regression results
                df_clean = df_plot.dropna(subset=['da', 'Predicted_da']).copy()
                if df_clean.empty:
                    return px.line(title="No regression data available")
                    
                df_clean = df_clean.sort_values('date')
                fig = go.Figure()
                
                if selected_site == "All sites" or not selected_site:
                    for site in df_clean['site'].unique():
                        site_data = df_clean[df_clean['site'] == site]
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['da'],
                            mode='lines+markers', name=f'{site} - Actual',
                            line=dict(dash='solid')
                        ))
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['Predicted_da'],
                            mode='lines+markers', name=f'{site} - Predicted',
                            line=dict(dash='dash')
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df_clean['date'], y=df_clean['da'],
                        mode='lines+markers', name='Actual DA',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_clean['date'], y=df_clean['Predicted_da'],
                        mode='lines+markers', name='Predicted DA',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                r2 = r2_score(df_clean['da'], df_clean['Predicted_da'])
                mae = mean_absolute_error(df_clean['da'], df_clean['Predicted_da'])
                
                fig.update_layout(
                    title=f"DA Level Forecast - {selected_site or 'All Sites'}<br>R² = {r2:.3f}, MAE = {mae:.3f}",
                    xaxis_title="Date",
                    yaxis_title="DA Levels",
                    hovermode='x unified'
                )
                
            elif CONFIG["SELECTED_TASK"] == "classification":
                # Show classification results
                cat_cols = []
                if 'da-category' in df_plot.columns and 'Predicted_da-category' in df_plot.columns:
                    cat_cols = ['da-category', 'Predicted_da-category']
                elif 'da_category' in df_plot.columns and 'Predicted_da_category' in df_plot.columns:
                    cat_cols = ['da_category', 'Predicted_da_category']
                
                df_clean = df_plot.dropna(subset=cat_cols).copy()
                if df_clean.empty:
                    return px.line(title="No classification data available")
                    
                df_clean = df_clean.sort_values('date')
                actual_col, pred_col = cat_cols
                fig = go.Figure()
                
                if selected_site == "All sites" or not selected_site:
                    for site in df_clean['site'].unique():
                        site_data = df_clean[df_clean['site'] == site]
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data[actual_col],
                            mode='lines+markers', name=f'{site} - Actual Category',
                            line=dict(dash='solid')
                        ))
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data[pred_col],
                            mode='lines+markers', name=f'{site} - Predicted Category',
                            line=dict(dash='dash')
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df_clean['date'], y=df_clean[actual_col],
                        mode='lines+markers', name='Actual Category',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_clean['date'], y=df_clean[pred_col],
                        mode='lines+markers', name='Predicted Category',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(df_clean[actual_col], df_clean[pred_col])
                
                fig.update_layout(
                    title=f"DA Category Forecast - {selected_site or 'All Sites'}<br>Accuracy = {accuracy:.3f}",
                    xaxis_title="Date",
                    yaxis_title="DA Category",
                    yaxis=dict(tickmode='array', tickvals=[0,1,2,3], 
                             ticktext=['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']),
                    hovermode='x unified'
                )
                
            else:
                return px.line(title=f"Unknown task: {CONFIG['SELECTED_TASK']}")
            
            return fig

        print(f"Starting retrospective dashboard on port {port}")
        app.run_server(debug=False, port=port)
    
    def create_realtime_dashboard(self, port):
        """Create real-time dashboard (matching future-forecasts.py interface)."""
        # Load data for real-time use
        raw_data = self.forecast_model.load_and_prepare_base_data(self.data_path)
        
        print(f"[INFO] Real-time forecasting (both regression and classification) with Random Forest + Gradient Boosting")
        
        available_sites = raw_data['site'].unique()
        initial_site = available_sites[0] if len(available_sites) > 0 else None
        min_forecast_date = pd.to_datetime("2010-01-01")

        app = dash.Dash(__name__)

        # Original UI Layout from future-forecasts.py
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
                return ("Please select both date and site.", go.Figure(), go.Figure())

            try:
                forecast_date = pd.to_datetime(forecast_date_str)
                result = self.forecast_model.forecast_realtime_single(raw_data, forecast_date, site)
                
                if not result:
                    return ("No forecast possible (insufficient data)", go.Figure(), go.Figure())

                # Format output (from future-forecasts.py style)
                CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
                
                lines = [
                    f"Forecast date (target): {result['ForecastPoint'].date()}",
                    f"Anchor date (training cutoff): {result['Anchordate'].date()}",
                ]

                if result['Testdate'] is not None:
                    lines.append(f"Test date (for accuracy): {result['Testdate'].date()}")
                else:
                    lines.append("Test date (for accuracy): N/A")

                # Real-time mode always shows both regression and classification results
                lines += [
                    "",
                    "--- Regression (da) ---",
                    f"Predicted Range (GB): {result['Predicted_da_Q05']:.2f} (Q05) – {result['Predicted_da_Q50']:.2f} (Q50) – {result['Predicted_da_Q95']:.2f} (Q95)",
                    f"Predicted Value (RF): {result['Predicted_da_RF']:.2f}",
                ]

                if result['Actual_da'] is not None:
                    within_range = result['SingledateCoverage']
                    status = 'Within GB Range ✅' if within_range else 'Outside GB Range ❌'
                    lines.append(f"Actual Value: {result['Actual_da']:.2f} ({status})")
                else:
                    lines.append("Actual Value: N/A (forecast beyond available data)")

                lines += ["", "--- Classification (da-category, Random Forest) ---"]
                
                if result['Predicted_da-category'] is not None:
                    lines.append(f"Predicted: {CATEGORY_LABELS[result['Predicted_da-category']]}")
                    if result['Probabilities'] is not None:
                        lines.append("Probabilities: " + ", ".join([
                            f"{label}: {prob * 100:.1f}%"
                            for label, prob in zip(CATEGORY_LABELS, result['Probabilities'])
                        ]))
                        
                    # Show actual category if available
                    if result['Actual_da'] is not None:
                        # Calculate actual category based on actual DA value
                        actual_da = result['Actual_da']
                        if actual_da <= 5:
                            actual_cat_idx = 0
                        elif actual_da <= 20:
                            actual_cat_idx = 1
                        elif actual_da <= 40:
                            actual_cat_idx = 2
                        else:
                            actual_cat_idx = 3
                        
                        match_status = "✅ MATCH" if result['Predicted_da-category'] == actual_cat_idx else "❌ MISMATCH"
                        lines.append(f"Actual: {CATEGORY_LABELS[actual_cat_idx]} {match_status}")
                    else:
                        lines.append("Actual Category: N/A")
                else:
                    lines.append("Predicted: N/A (insufficient training classes)")

                text_output = "\n".join(lines)

                # Create level range graph using original gradient visualization
                # Real-time mode always has quantiles (RF + GB), so always use gradient visualization
                level_fig = self.forecast_model.create_level_range_graph(
                    result['Predicted_da_Q05'],
                    result['Predicted_da_Q50'], 
                    result['Predicted_da_Q95'],
                    result['Actual_da'],
                    result['Predicted_da_RF']
                )

                # Create category graph (from future-forecasts.py)
                category_fig = go.Figure()
                
                if result['Probabilities'] is not None:
                    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
                    pred_cat = result['Predicted_da-category']
                    if pred_cat is not None and 0 <= pred_cat < len(colors):
                        colors[pred_cat] = '#2ca02c'  # Highlight predicted category

                    category_fig.add_trace(go.Bar(
                        x=CATEGORY_LABELS,
                        y=result['Probabilities'],
                        marker_color=colors,
                        text=[f"{p * 100:.1f}%" for p in result['Probabilities']],
                        textposition='auto'
                    ))

                category_fig.update_layout(
                    title="Category Probability Distribution (Random Forest)",
                    yaxis=dict(title="Probability", range=[0, 1.1]),
                    xaxis=dict(title="Category"),
                    showlegend=False,
                    height=400
                )

                return (text_output, level_fig, category_fig)

            except Exception as e:
                error_msg = f"Error generating forecast: {str(e)}"
                return (error_msg, go.Figure(), go.Figure())

        print(f"Starting real-time forecasting dashboard on port {port}")
        app.run_server(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description="Improved DA Forecasting Application")
    parser.add_argument('--mode', choices=['retrospective', 'realtime'], required=True)
    parser.add_argument('--task', choices=['regression', 'classification'], required=False, 
                       help='Task for retrospective mode (regression or classification). Ignored in realtime mode which runs both.')
    parser.add_argument('--model', choices=['rf', 'linear'], default='rf',
                       help='Model type for retrospective mode: rf (Random Forest) or linear (Linear/Logistic baseline). Ignored in realtime mode which always uses RF.')
    parser.add_argument('--data', default='final_output.parquet')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--anchors', type=int, default=None, 
                       help='Number of anchor forecasts per site (default: uses CONFIG value)')
    parser.add_argument('--min-test-date', default='2008-01-01')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'retrospective' and args.task is None:
        parser.error("--task is required for retrospective mode")
    
    # Update CONFIG based on command line arguments
    CONFIG["SELECTED_TASK"] = args.task if args.task else "both"  # "both" for realtime mode
    CONFIG["SELECTED_MODEL"] = args.model
    
    # Override anchors per site if specified
    if args.anchors is not None:
        CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"] = args.anchors
    
    # Initialize app
    app = ImprovedForecastApp(args.data)
    app.load_data()
    
    if args.mode == 'retrospective':
        port = args.port or CONFIG["PORT_RETRO"]
        n_anchors = CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"]
        print(f"[INFO] Using {n_anchors} anchor forecasts per site")
        app.run_retrospective_evaluation(n_anchors, args.min_test_date)
        app.create_retrospective_dashboard(port)
    else:  # realtime
        port = args.port or CONFIG["PORT_REALTIME"]
        app.create_realtime_dashboard(port)


if __name__ == "__main__":
    main()