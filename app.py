"""
Unified DATect Forecasting Pipeline - Dash Application
Combines both past evaluation and future forecasting in one app using original formats.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config
from data_processing import DataProcessor
from modeling import TimeSeriesForecaster, QuantileForecaster


def _run_single_evaluation_standalone(site: str, anchor_date: pd.Timestamp, 
                                     forecast_date: pd.Timestamp, data: pd.DataFrame,
                                     data_processor) -> Dict:
    """Standalone function for single evaluation forecast (for parallel processing)."""
    try:
        site_data = data[data['site'] == site].copy()
        site_data = site_data.sort_values('date')
        
        # Split data
        train_data = site_data[site_data['date'] <= anchor_date]
        forecast_row = site_data[site_data['date'] == forecast_date]
        
        if len(train_data) < config.MIN_TRAINING_POINTS or forecast_row.empty:
            return None
        
        # Get actual values
        actual_da = forecast_row['da'].iloc[0]
        actual_cat = _da_to_category_standalone(actual_da)
        
        # Features - use raw data
        X_train = train_data
        y_train = train_data['da']
        X_forecast = forecast_row
        
        # Regression model
        rf_model = TimeSeriesForecaster(
            model_type='random_forest',
            task='regression',
            include_lags=config.INCLUDE_LAG_FEATURES,
            random_state=config.RANDOM_STATE
        )
        rf_model.fit(X_train, y_train)
        pred_da = float(rf_model.predict(X_forecast)[0])
        
        # Classification model for proper category prediction
        rf_classifier = TimeSeriesForecaster(
            model_type='random_forest',
            task='classification',
            include_lags=config.INCLUDE_LAG_FEATURES,
            random_state=config.RANDOM_STATE
        )
        y_train_cls = train_data['da-category'] if 'da-category' in train_data.columns else train_data['da'].apply(_da_to_category_standalone)
        rf_classifier.fit(X_train, y_train_cls)
        pred_cat = int(rf_classifier.predict(X_forecast)[0])
        
        return {
            'site': site,
            'anchor_date': anchor_date,
            'forecast_date': forecast_date,
            'actual_da': actual_da,
            'predicted_da': pred_da,
            'actual_cat': actual_cat,
            'predicted_cat': pred_cat
        }
        
    except Exception as e:
        return None


def _da_to_category_standalone(da_value: float) -> int:
    """Convert DA value to category (standalone version)."""
    if da_value <= 5:
        return 0  # Low
    elif da_value <= 20:
        return 1  # Moderate
    elif da_value <= 40:
        return 2  # High
    else:
        return 3  # Extreme


class UnifiedForecastingApp:
    """Unified application combining original past and future forecasting formats."""
    
    def __init__(self, data_file: str = None):
        self.data_file = data_file or config.DATA_FILE
        self.data_processor = DataProcessor()
        self.data = None
        self.app = dash.Dash(__name__)
        
        # Store evaluation results for the past analysis
        self.evaluation_results = {}
        self.sites = []
        
        random.seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)
        
    def load_data(self):
        """Load and prepare data."""
        print("Loading data...")
        self.data = self.data_processor.load_data(self.data_file)
        self.sites = sorted(self.data['site'].unique())
        print(f"Loaded {len(self.data)} rows for {len(self.sites)} sites")
        
    def setup_layout(self):
        """Setup the unified dashboard layout with tabs for past and future."""
        min_forecast_date = pd.to_datetime(config.MIN_TEST_DATE)
        
        self.app.layout = html.Div([
            html.H1("DATect Unified Forecasting Pipeline", style={'textAlign': 'center'}),
            
            dcc.Tabs(id="main-tabs", value='future-tab', children=[
                # Future Forecasting Tab (Original future-forecasts.py format)
                dcc.Tab(label='Future Forecasting', value='future-tab', children=[
                    html.Div([
                        html.H3("Forecast by Specific Date & Site"),
                        
                        html.Label("Choose a site:"),
                        dcc.Dropdown(
                            id='site-dropdown-forecast',
                            options=[{'label': s, 'value': s} for s in self.sites],
                            value=self.sites[0] if self.sites else None,
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
                                html.Div(id='forecast-output-partial', 
                                        style={'whiteSpace': 'pre-wrap', 'marginTop': 20, 
                                              'border': '1px solid lightgrey', 'padding': '10px'}),
                                
                                # Graphs for forecast ranges
                                html.Div([
                                    dcc.Graph(id='level-range-graph', 
                                             style={'display': 'inline-block', 'width': '49%'}),
                                    dcc.Graph(id='category-range-graph', 
                                             style={'display': 'inline-block', 'width': '49%'})
                                ])
                            ],
                            style={'marginTop': 30}
                        )
                    ], style={'padding': '20px'})
                ]),
                
                # Past Evaluation Tab (with pre-selection format)
                dcc.Tab(label='Past Evaluation', value='past-tab', children=[
                    html.Div([
                        html.H1("Domoic Acid Past Evaluation (Random Anchor Forecasts)"),
                        
                        # Pre-evaluation controls
                        html.Div([
                            html.Label("Select Evaluation Type:"),
                            dcc.Dropdown(
                                id="evaluation-type-dropdown",
                                options=[
                                    {"label": "DA Levels (Regression)", "value": "DA_Level"},
                                    {"label": "DA Category (Classification)", "value": "da-category"},
                                ],
                                value="DA_Level",
                                clearable=False,
                                style={"width": "30%", "marginLeft": "10px"},
                            ),
                            
                            html.Label("Select Model Method:", style={"marginLeft": "20px"}),
                            dcc.Dropdown(
                                id="forecast-method-dropdown",
                                options=[
                                    {"label": "Random Forest", "value": "ml"}
                                ],
                                value="ml",
                                clearable=False,
                                style={"width": "20%", "marginLeft": "10px"},
                            ),
                            
                            html.Button('Run Evaluation', id='run-evaluation-btn', 
                                       style={'marginLeft': '20px', 'padding': '10px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
                            
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "30px", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),
                        
                        # Status and site selection
                        html.Div([
                            html.Div(id='evaluation-status', style={'marginBottom': '15px', 'fontSize': '14px'}),
                            
                            html.Label("Select Site for Analysis:"),
                            dcc.Dropdown(
                                id="site-dropdown",
                                options=[{"label": "All sites", "value": "All sites"}] + [
                                    {"label": site, "value": site} for site in self.sites
                                ],
                                value="All sites",
                                style={"width": "50%", "marginBottom": "20px"},
                            ),
                        ]),
                        
                        # Results visualization
                        dcc.Graph(id="analysis-graph", style={'height': '600px'}),
                    ], style={'padding': '20px'})
                ])
            ]),
            
            # Store for evaluation data
            dcc.Store(id="evaluation-data-store", data={}),
        ])
    
    def register_callbacks(self):
        """Register all Dash callbacks."""
        
        # Future forecasting callback (original format)
        @self.app.callback(
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
        def update_future_forecast(forecast_date_str, site):
            if not forecast_date_str or not site:
                return ("Please select both date and site.", go.Figure(), go.Figure())
            
            if self.data is None or self.data.empty:
                return ("Error: Data not loaded.", go.Figure(), go.Figure())
            
            try:
                forecast_date = pd.to_datetime(forecast_date_str)
                result = self.forecast_for_date(forecast_date, site)
                
                if not result:
                    return ("No forecast possible - insufficient historical data.", 
                           go.Figure(), go.Figure())
                
                # Format text output
                text_output = self.format_forecast_output(result)
                
                # Create visualizations
                level_fig = self.create_level_range_graph(
                    result['q05'], result['q50'], result['q95'],
                    result.get('actual'), result.get('rf_pred')
                )
                
                category_fig = self.create_category_graph(
                    result['probabilities'], result['pred_cat'], result.get('actual_cat')
                )
                
                return text_output, level_fig, category_fig
                
            except Exception as e:
                return f"Error: {str(e)}", go.Figure(), go.Figure()
        
        # Past evaluation callback (with evaluation type selection)
        @self.app.callback(
            [Output('evaluation-data-store', 'data'),
             Output('evaluation-status', 'children')],
            [Input('run-evaluation-btn', 'n_clicks'),
             Input('evaluation-type-dropdown', 'value'),
             Input('forecast-method-dropdown', 'value')],
            prevent_initial_call=True
        )
        def run_past_evaluation(n_clicks, evaluation_type, forecast_method):
            if not n_clicks:
                return {}, ""
                
            try:
                # Show running status
                total_forecasts = config.N_RANDOM_ANCHORS_PER_SITE * len(self.sites)
                running_msg = html.Div([
                    html.Span("🔄 Running evaluation... ", style={'color': 'blue', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span(f"Evaluation Type: {evaluation_type}", style={'fontSize': '12px'}),
                    html.Br(),
                    html.Span(f"Generating ~{total_forecasts} forecasts across {len(self.sites)} sites", 
                             style={'fontSize': '12px', 'color': 'gray'})
                ])
                
                # Run the evaluation with the selected type
                results = self.run_comprehensive_evaluation(evaluation_type)
                
                success_msg = html.Div([
                    html.Span("✅ Evaluation completed!", style={'color': 'green', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span(f"Ready to analyze {evaluation_type} results", style={'fontSize': '12px'})
                ])
                
                return results, success_msg
                
            except Exception as e:
                error_msg = html.Div([
                    html.Span("❌ Evaluation failed!", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span(f"Error: {str(e)}", style={'fontSize': '11px', 'color': 'red'})
                ])
                return {}, error_msg
        
        # Analysis graph callback (using evaluation type dropdown)
        @self.app.callback(
            Output("analysis-graph", "figure"),
            [
                Input("evaluation-type-dropdown", "value"),
                Input("site-dropdown", "value"), 
                Input("forecast-method-dropdown", "value"),
                Input("evaluation-data-store", "data")
            ]
        )
        def update_analysis_graph(evaluation_type, selected_site, forecast_method, evaluation_data):
            if not evaluation_data:
                return px.line(
                    title="Select evaluation type and click 'Run Evaluation' to see results",
                    template="plotly_white"
                ).update_layout(
                    annotations=[{
                        "text": "📊 No data available<br><br>1. Choose DA Levels or Category above<br>2. Click 'Run Evaluation'<br>3. Wait for processing to complete",
                        "xref": "paper", "yref": "paper",
                        "x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle",
                        "showarrow": False, "font": {"size": 16, "color": "gray"}
                    }]
                )
            
            return self.create_simple_analysis_graph(
                evaluation_data, evaluation_type, selected_site
            )
    
    def forecast_for_date(self, forecast_date: pd.Timestamp, site: str) -> Dict:
        """Generate forecast for specific date and site (future forecasting)."""
        try:
            # Get training and forecast data
            site_data = self.data[self.data['site'] == site].copy()
            site_data = site_data.sort_values('date')
            
            # Training data (before forecast date)
            train_data = site_data[site_data['date'] < forecast_date]
            if len(train_data) < config.MIN_TRAINING_POINTS:
                return None
            
            # Find actual value if available
            actual_row = site_data[site_data['date'] >= forecast_date]
            actual_da = actual_row['da'].iloc[0] if not actual_row.empty else None
            actual_cat = actual_row['da-category'].iloc[0] if not actual_row.empty and 'da-category' in actual_row.columns else None
            
            # Use raw data - let TimeSeriesForecaster handle preprocessing
            # Create synthetic forecast row
            last_train = train_data.iloc[-1:].copy()
            last_train['date'] = forecast_date
            last_train['da'] = np.nan
            
            # Train models with raw data
            X_train = train_data
            y_train = train_data['da']
            X_forecast = last_train
            
            # Random Forest prediction
            rf_model = TimeSeriesForecaster(
                model_type='random_forest', 
                task='regression', 
                include_lags=config.INCLUDE_LAG_FEATURES,
                random_state=config.RANDOM_STATE
            )
            rf_model.fit(X_train, y_train)
            rf_pred = float(rf_model.predict(X_forecast)[0])
            
            # Quantile predictions
            quantile_forecaster = QuantileForecaster(
                quantiles=config.QUANTILE_LEVELS, 
                random_state=config.RANDOM_STATE
            )
            quantile_forecaster.fit(X_train, y_train)
            quantile_preds = quantile_forecaster.predict(X_forecast)
            
            # Classification - train Random Forest classifier for proper probabilities
            rf_classifier = TimeSeriesForecaster(
                model_type='random_forest',
                task='classification',
                include_lags=config.INCLUDE_LAG_FEATURES,
                random_state=config.RANDOM_STATE
            )
            
            # Create classification training data
            y_train_cls = train_data['da-category'] if 'da-category' in train_data.columns else train_data['da'].apply(self._da_to_category)
            rf_classifier.fit(X_train, y_train_cls)
            
            # Get proper probabilities and prediction
            probabilities = list(rf_classifier.predict_proba(X_forecast)[0])
            pred_cat = int(rf_classifier.predict(X_forecast)[0])
            
            # Create actual category if needed
            actual_cat = self._da_to_category(actual_da) if actual_da is not None else None
            
            return {
                'site': site,
                'forecast_date': forecast_date,
                'anchor_date': train_data['date'].max(),
                'q05': quantile_preds['q05'][0],
                'q50': quantile_preds['q50'][0], 
                'q95': quantile_preds['q95'][0],
                'rf_pred': rf_pred,
                'pred_cat': pred_cat,
                'probabilities': probabilities,
                'actual': actual_da,
                'actual_cat': actual_cat
            }
            
        except Exception as e:
            print(f"Error in forecast_for_date: {e}")
            return None
    
    def run_comprehensive_evaluation(self, evaluation_type: str) -> Dict:
        """Run simple comprehensive evaluation without past-forecasts-final complexity."""
        print(f"Starting {evaluation_type} evaluation with {config.N_RANDOM_ANCHORS_PER_SITE} anchors per site...")
        
        # Simple anchor point generation
        anchor_points = []
        for site in self.sites:
            site_data = self.data[self.data['site'] == site].copy()
            site_data = site_data.sort_values('date')
            site_dates = site_data['date'].sort_values().unique()
            
            if len(site_dates) > config.MIN_TRAINING_POINTS + 1:
                eligible_anchors = site_dates[:-1]
                n_anchors = min(len(eligible_anchors), config.N_RANDOM_ANCHORS_PER_SITE)
                
                if n_anchors > 0:
                    selected_anchors = np.random.choice(eligible_anchors, size=n_anchors, replace=False)
                    for anchor_date in selected_anchors:
                        train_data = site_data[site_data['date'] <= anchor_date]
                        future_data = site_data[site_data['date'] > anchor_date]
                        
                        if len(train_data) >= config.MIN_TRAINING_POINTS and len(future_data) >= 1:
                            forecast_date = future_data['date'].iloc[0]
                            anchor_points.append((site, pd.Timestamp(anchor_date), pd.Timestamp(forecast_date)))
        
        print(f"Generated {len(anchor_points)} anchor points")
        
        # Run simple evaluation without complex parallel processing
        results = []
        for i, (site, anchor, forecast) in enumerate(anchor_points):
            if i % 50 == 0:
                print(f"Processing {i}/{len(anchor_points)} forecasts...")
                
            result = _run_single_evaluation_standalone(site, anchor, forecast, self.data, self.data_processor)
            if result is not None:
                results.append(result)
        
        print(f"Completed {len(results)} successful forecasts")
        return self._format_simple_evaluation_results(results, evaluation_type)
    
    
    def _format_simple_evaluation_results(self, results: List[Dict], evaluation_type: str) -> Dict:
        """Format results in simple format for display."""
        if not results:
            return {}
            
        # Create dataframe
        test_df = pd.DataFrame(results)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
        
        if evaluation_type == "DA_Level":
            overall_r2 = r2_score(test_df['actual_da'], test_df['predicted_da'])
            overall_mae = mean_absolute_error(test_df['actual_da'], test_df['predicted_da'])
            
            return {
                'results': {
                    'test_df': test_df.rename(columns={
                        'forecast_date': 'date',
                        'predicted_da': 'Predicted_da',
                        'actual_da': 'da'
                    }),
                    'overall_r2': overall_r2,
                    'overall_mae': overall_mae,
                    'evaluation_type': evaluation_type
                }
            }
        else:  # da-category
            overall_accuracy = accuracy_score(test_df['actual_cat'], test_df['predicted_cat'])
            
            return {
                'results': {
                    'test_df': test_df.rename(columns={
                        'forecast_date': 'date',
                        'predicted_cat': 'Predicted_da-category',
                        'actual_cat': 'da-category'
                    }),
                    'overall_accuracy': overall_accuracy,
                    'evaluation_type': evaluation_type
                }
            }
    
    def _da_to_category(self, da_value: float) -> int:
        """Convert DA value to category."""
        if da_value <= 5:
            return 0  # Low
        elif da_value <= 20:
            return 1  # Moderate
        elif da_value <= 40:
            return 2  # High
        else:
            return 3  # Extreme
    
    # Original visualization methods from future-forecasts.py
    def create_level_range_graph(self, q05, q50, q95, actual_levels=None, rf_prediction=None):
        """Create gradient visualization for DA level forecast (original format)."""
        fig = go.Figure()
        n_segments = 30
        range_width = q95 - q05
        max_distance = max(q50 - q05, q95 - q50) if range_width > 1e-6 else 1
        if max_distance <= 1e-6: 
            max_distance = 1

        base_color = (70, 130, 180)

        # Gradient confidence area
        for i in range(n_segments):
            x0 = q05 + (i / n_segments) * (range_width)
            x1 = q05 + ((i + 1) / n_segments) * (range_width)
            midpoint = (x0 + x1) / 2
            opacity = 1 - (abs(midpoint - q50) / max_distance) ** 0.5 if max_distance > 1e-6 else (0.8 if abs(midpoint - q50) < 1e-6 else 0.2)
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=0.4, y1=0.6,
                line=dict(width=0),
                fillcolor=f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {max(0, min(1, float(opacity)))})',
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
                x=[rf_prediction], y=[0.5],
                mode='markers',
                marker=dict(
                    size=14,
                    color='darkorange',
                    symbol='diamond-tall',
                    line=dict(width=1, color='black')
                ),
                name='Random Forest Pred.'
            ))

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

    def create_category_graph(self, probs, pred_cat, actual_cat=None):
        """Create bar chart for category probabilities (original format)."""
        colors = ['#1f77b4'] * len(config.CATEGORY_LABELS)
        if 0 <= pred_cat < len(colors):
            colors[pred_cat] = '#2ca02c'

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=config.CATEGORY_LABELS,
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

    def format_forecast_output(self, result):
        """Format forecast results as text for display (original format)."""
        lines = [
            f"Forecast date (target): {result['forecast_date'].date()}",
            f"Anchor date (training cutoff): {result['anchor_date'].date()}",
            "",
            "--- Regression (da) ---",
            f"Predicted Range (GB): {result['q05']:.2f} (Q05) – {result['q50']:.2f} (Q50) – {result['q95']:.2f} (Q95)",
            f"Predicted Value (RF): {result['rf_pred']:.2f}",
        ]

        if result['actual'] is not None:
            within_range = result['q05'] <= result['actual'] <= result['q95']
            status = 'Within GB Range ✅' if within_range else 'Outside GB Range ❌'
            lines.append(f"Actual Value: {result['actual']:.2f} ({status})")
        else:
            lines.append("Actual Value: N/A (forecast beyond available data)")

        lines += [
            "",
            "--- Classification (da-category, Random Forest) ---",
            f"Predicted: {config.CATEGORY_LABELS[result['pred_cat']]}",
            "Probabilities: " + ", ".join([
                f"{label}: {prob * 100:.1f}%"
                for label, prob in zip(config.CATEGORY_LABELS, result['probabilities'])
            ])
        ]

        if result['actual_cat'] is not None:
            match_status = "✅ MATCH" if result['pred_cat'] == result['actual_cat'] else "❌ MISMATCH"
            lines.append(f"Actual: {config.CATEGORY_LABELS[result['actual_cat']]} {match_status}")
        else:
            lines.append("Actual Category: N/A")

        return "\n".join(lines)
    
    def create_simple_analysis_graph(self, evaluation_data, evaluation_type, selected_site):
        """Create simple analysis graph for past evaluation."""
        if 'results' not in evaluation_data:
            return px.line(title="No evaluation results available")
        
        pred_data = evaluation_data['results']
        df_plot = pred_data.get("test_df", pd.DataFrame())
        
        if df_plot.empty:
            return px.line(title="No data available for analysis")

        # Setup based on evaluation type
        if evaluation_type == "DA_Level":
            y_axis_title = "Domoic Acid Levels"
            actual_col = "da" 
            pred_col = "Predicted_da"
            
            overall_r2 = pred_data.get("overall_r2", float("nan"))
            overall_mae = pred_data.get("overall_mae", float("nan"))
            performance_text = f"R² = {overall_r2:.3f}, MAE = {overall_mae:.2f}"
            
        else:  # da-category
            y_axis_title = "Domoic Acid Category"
            actual_col = "da-category"
            pred_col = "Predicted_da-category"
            
            # Convert to string for categorical plotting
            df_plot[actual_col] = df_plot[actual_col].astype(str)
            df_plot[pred_col] = df_plot[pred_col].astype(str)
            
            overall_accuracy = pred_data.get("overall_accuracy", float("nan"))
            performance_text = f"Accuracy = {overall_accuracy:.3f}"

        # Create time series plot
        metric_order = [actual_col, pred_col]
        df_plot_melted = pd.melt(
            df_plot,
            id_vars=["date", "site"], 
            value_vars=[actual_col, pred_col],
            var_name="Metric",
            value_name="Value",
        ).dropna(subset=["Value"])

        # Filter by site
        if selected_site != "All sites":
            df_plot_melted = df_plot_melted[df_plot_melted["site"] == selected_site]

        if df_plot_melted.empty:
            return px.line(title=f"No data to plot for {selected_site}")

        df_plot_melted = df_plot_melted.sort_values("date")
        
        # Create line chart
        plot_title = f"{y_axis_title} Evaluation - {selected_site or 'All sites'}"
        
        if selected_site == "All sites":
            fig = px.line(df_plot_melted, x="date", y="Value", 
                         color="site", line_dash="Metric",
                         title=plot_title,
                         category_orders={"Metric": metric_order},
                         template="plotly_white")
        else:
            fig = px.line(df_plot_melted, x="date", y="Value", 
                         color="Metric",
                         title=plot_title,
                         category_orders={"Metric": metric_order},
                         template="plotly_white")

        fig.update_layout(
            yaxis_title=y_axis_title, 
            xaxis_title="Date", 
            legend_title_text="Legend", 
            height=600,
            margin=dict(l=50, r=50, t=80, b=100),
            title_x=0.5,
            annotations=[{
                "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.12,
                "xanchor": "center", "yanchor": "top", "text": performance_text,
                "showarrow": False, "font": {"size": 14, "color": "darkblue"},
                "bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "lightgray", "borderwidth": 1
            }]
        )
        
        if evaluation_type == "da-category":
            cat_values = sorted(df_plot_melted["Value"].unique(), key=lambda x: str(x))
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=cat_values)
            
        return fig
    
    def run(self):
        """Run the unified application."""
        self.load_data()
        self.setup_layout()
        self.register_callbacks()
        
        print(f"Starting unified forecasting app on port {config.APP_PORT}")
        self.app.run_server(debug=config.DEBUG_MODE, host=config.APP_HOST, port=config.APP_PORT)


if __name__ == '__main__':
    app = UnifiedForecastingApp()
    app.run()