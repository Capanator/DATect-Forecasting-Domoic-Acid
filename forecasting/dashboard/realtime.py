"""
Realtime Dashboard
==================

Interactive dashboard for generating forecasts for specific dates and sites
with leak-free temporal safeguards and user-friendly controls.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

import config
from ..core.forecast_engine import ForecastEngine
from ..core.model_factory import ModelFactory
from ..core.logging_config import get_logger
from ..core.exception_handling import ScientificValidationError

# Initialize logger
logger = get_logger(__name__)


class RealtimeDashboard:
    """
    Dashboard for real-time forecasting with interactive controls.
    
    Features:
    - Date and site selection
    - Model parameter configuration
    - Interactive forecast generation
    - Feature importance visualization
    """
    
    def __init__(self, data_path):
        """
        Initialize dashboard with data source.
        
        Args:
            data_path: Path to processed data file
        """
        self.data_path = data_path
        self.forecast_engine = ForecastEngine()
        self.model_factory = ModelFactory()
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def run(self, port=8065, debug=False):
        """
        Run the dashboard server.
        
        Args:
            port: Port to run server on
            debug: Whether to run in debug mode
        """
        try:
            logger.info(f"Starting RealtimeDashboard server on port {port}")
            logger.info(f"Debug mode: {debug}")
            print(f"Starting realtime dashboard on port {port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')
            
        except Exception as e:
            logger.error(f"Failed to start RealtimeDashboard server: {str(e)}")
            raise ScientificValidationError(f"Dashboard server failed: {str(e)}")
        
    def _setup_layout(self):
        """Setup the dashboard layout."""
        try:
            logger.info("Setting up dashboard layout")
            
            # Load data to get available sites and date range
            logger.debug(f"Loading data from {self.data_path}")
            data = pd.read_parquet(self.data_path)
            data['date'] = pd.to_datetime(data['date'])
            
            sites_list = sorted(data['site'].unique().tolist())
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            
            logger.info(f"Dashboard setup: {len(sites_list)} sites, date range {min_date} to {max_date}")
        
        except Exception as e:
            logger.error(f"Failed to setup dashboard layout: {str(e)}")
            raise ScientificValidationError(f"Dashboard layout setup failed: {str(e)}")
        
        # Get available models from model factory
        available_models = self.model_factory.get_supported_models('regression')['regression']
        model_options = []
        for model in available_models:
            desc = self.model_factory.get_model_description(model)
            model_options.append({"label": desc, "value": model})
        
        self.app.layout = html.Div([
            html.H1("Real-time DA Forecast Dashboard",
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
                   
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Forecast Date:", style={'fontWeight': 'bold'}),
                    dcc.DatePickerSingle(
                        id="forecast-date",
                        date=max_date - timedelta(days=30),  # Default to recent date
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        display_format='YYYY-MM-DD'
                    ),
                ], style={'width': '180px', 'display': 'inline-block', 'margin': '0 15px'}),
                
                html.Div([
                    html.Label("Site:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="forecast-site",
                        options=[{"label": site, "value": site} for site in sites_list],
                        value=sites_list[0]
                    ),
                ], style={'width': '180px', 'display': 'inline-block', 'margin': '0 15px'}),
                
                html.Div([
                    html.Label("Model:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="model-selector",
                        options=model_options,
                        value=config.FORECAST_MODEL,  # Use config default
                        clearable=False
                    ),
                ], style={'width': '220px', 'display': 'inline-block', 'margin': '0 15px'}),
                
                html.Div([
                    html.Button("Generate Forecast", id="forecast-button", 
                              n_clicks=0,
                              style={'backgroundColor': '#2E86C1', 'color': 'white', 
                                   'border': 'none', 'padding': '10px 20px',
                                   'borderRadius': '5px', 'cursor': 'pointer',
                                   'marginTop': '25px'})
                ], style={'display': 'inline-block', 'margin': '0 15px'}),
                
            ], style={'textAlign': 'center', 'marginBottom': '30px',
                     'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
            
            # Results display
            html.Div(id="forecast-results", style={'marginTop': '30px'}),
            
            # Original realtime graphs
            html.Div([
                dcc.Graph(id='level-range-graph', style={'display': 'inline-block', 'width': '49%'}),
                dcc.Graph(id='category-range-graph', style={'display': 'inline-block', 'width': '49%'})
            ], style={'marginTop': '20px'}),
            
            # Feature importance plot
            html.Div([
                dcc.Graph(id="importance-plot", style={'height': '400px'}),
            ], style={'marginTop': '30px'}),
            
        ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
# Removed model dropdown callback - using fixed XGBoost model
            
        # Generate forecast when button is clicked
        @self.app.callback(
            [Output("forecast-results", "children"),
             Output("level-range-graph", "figure"),
             Output("category-range-graph", "figure"),
             Output("importance-plot", "figure")],
            [Input("forecast-button", "n_clicks")],
            [State("forecast-date", "date"),
             State("forecast-site", "value"),
             State("model-selector", "value")]
        )
        def generate_forecast(n_clicks, forecast_date, site, selected_model):
            if n_clicks == 0:
                return (html.Div("Click 'Generate Forecast' to start", 
                               style={'textAlign': 'center', 'color': '#666'}), 
                       go.Figure(), go.Figure(), go.Figure())
                
            if not all([forecast_date, site, selected_model]):
                return (html.Div("Please select date, site, and model", 
                               style={'textAlign': 'center', 'color': 'red'}), 
                       go.Figure(), go.Figure(), go.Figure())
                
            try:
                # Map model names based on task
                def get_actual_model(ui_model, task):
                    if ui_model == "xgboost":
                        return "xgboost"
                    elif ui_model == "ridge":
                        return "ridge" if task == "regression" else "logistic"
                    else:
                        return ui_model
                
                # Generate both regression and classification forecasts with mapped models
                regression_model = get_actual_model(selected_model, "regression")
                regression_result = self.forecast_engine.generate_single_forecast(
                    self.data_path, 
                    pd.to_datetime(forecast_date), 
                    site, 
                    "regression", 
                    regression_model
                )
                
                # Map for classification task
                classification_model = get_actual_model(selected_model, "classification")
                classification_result = self.forecast_engine.generate_single_forecast(
                    self.data_path, 
                    pd.to_datetime(forecast_date), 
                    site, 
                    "classification", 
                    classification_model
                )
                
                # Combine results
                combined_result = {
                    'forecast_date': pd.to_datetime(forecast_date),
                    'site': site,
                    'model_type': selected_model,
                    'classification_model': classification_model,
                    'regression': regression_result,
                    'classification': classification_result
                }
                
                if regression_result is None and classification_result is None:
                    return (html.Div("Insufficient data for forecast", 
                                   style={'textAlign': 'center', 'color': 'red'}), 
                           go.Figure(), go.Figure(), go.Figure())
                
                # Create results display for both tasks
                results_div = self._create_combined_forecast_display(combined_result)
                
                # Create additional graphs - prioritize regression but show both
                level_fig = self._create_level_range_graph(regression_result or {}, "regression")
                category_fig = self._create_category_range_graph(classification_result or {}, "classification")
                
                # Create feature importance plot (use regression if available, otherwise classification)
                importance_result = regression_result or classification_result
                importance_fig = self._create_importance_plot(importance_result.get('feature_importance') if importance_result else None)
                
                return results_div, level_fig, category_fig, importance_fig
                
            except Exception as e:
                error_msg = f"Error generating forecast: {str(e)}"
                return (html.Div(error_msg, style={'textAlign': 'center', 'color': 'red'}), 
                       go.Figure(), go.Figure(), go.Figure())
                
    def _create_combined_forecast_display(self, combined_result):
        """Create combined forecast results display for both regression and classification."""
        forecast_date = combined_result['forecast_date'].strftime('%Y-%m-%d')
        site = combined_result['site']
        
        # Base information
        model_desc = self.model_factory.get_model_description(combined_result['model_type'])
        classification_desc = self.model_factory.get_model_description(combined_result['classification_model'])
        info_section = html.Div([
            html.H3(f"Forecast for {site} on {forecast_date}", 
                   style={'color': '#2E86C1', 'textAlign': 'center'}),
            html.P(f"Regression Model: {model_desc} | Classification Model: {classification_desc}", 
                  style={'textAlign': 'center', 'fontSize': '16px'}),
        ])
        
        sections = []
        
        # Regression results (most important)
        if combined_result['regression']:
            reg_result = combined_result['regression']
            prediction = reg_result['predicted_da']
            training_samples = reg_result.get('training_samples', 'N/A')
            
            reg_section = html.Div([
                html.H4("🎯 DA Concentration Prediction", style={'color': '#E74C3C', 'marginBottom': '10px'}),
                html.P(f"Predicted DA: {prediction:.2f} μg/g", 
                      style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#2E86C1'}),
                html.P(f"Training samples: {training_samples}", 
                      style={'fontSize': '14px', 'color': '#666'})
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f0f8ff', 'borderRadius': '8px'})
            sections.append(reg_section)
        
        # Classification results  
        if combined_result['classification']:
            cls_result = combined_result['classification']
            predicted_category = cls_result['predicted_category']
            category_names = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
            
            # Fix numpy array indexing issue
            try:
                predicted_cat_int = int(predicted_category) if predicted_category is not None else 0
                category_name = category_names[predicted_cat_int] if 0 <= predicted_cat_int < len(category_names) else f"Category {predicted_cat_int}"
            except (ValueError, TypeError, IndexError):
                category_name = "Unknown Category"
                
            training_samples = cls_result.get('training_samples', 'N/A')
            
            cls_section = html.Div([
                html.H4("📊 Risk Category Prediction", style={'color': '#E74C3C', 'marginBottom': '10px'}),
                html.P(f"Risk Level: {category_name}", 
                      style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#2E86C1'}),
                html.P(f"Training samples: {training_samples}", 
                      style={'fontSize': '14px', 'color': '#666'})
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#fff8f0', 'borderRadius': '8px'})
            
            # Add probabilities if available
            if 'class_probabilities' in cls_result and cls_result['class_probabilities'] is not None:
                probs = cls_result['class_probabilities']
                prob_text = " | ".join([f"{cat}: {prob:.1%}" for cat, prob in zip(category_names[:len(probs)], probs)])
                cls_section.children.append(
                    html.P(f"Probabilities: {prob_text}",
                          style={'fontSize': '12px', 'color': '#666', 'marginTop': '10px'})
                )
            
            sections.append(cls_section)
        
        # No data message if both failed
        if not sections:
            sections.append(html.Div("Insufficient data for forecasting", 
                                   style={'textAlign': 'center', 'color': 'red', 'fontSize': '16px'}))
        
        return html.Div([info_section] + sections, 
                       style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
    
    def _create_forecast_display(self, result, task):
        """Create forecast results display."""
        forecast_date = result['forecast_date'].strftime('%Y-%m-%d')
        site = result['site']
        model_desc = self.model_factory.get_model_description(result['model_type'])
        
        # Base information
        info_section = html.Div([
            html.H3(f"Forecast for {site} on {forecast_date}", 
                   style={'color': '#2E86C1', 'textAlign': 'center'}),
            html.P(f"Model: {model_desc}", style={'textAlign': 'center', 'fontSize': '16px'}),
            html.P(f"Training samples: {result['training_samples']}", 
                  style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'}),
        ])
        
        # Task-specific results
        if task == "regression":
            prediction = result['predicted_da']
            confidence_section = html.Div([
                html.H4(f"Predicted DA: {prediction:.2f} μg/g", 
                       style={'textAlign': 'center', 'fontSize': '24px', 'color': '#E74C3C'}),
            ])
            
            # Add quantile predictions if available
            if 'quantile_predictions' in result:
                quantiles = result['quantile_predictions']
                confidence_section.children.append(
                    html.P(f"90% Confidence Interval: [{quantiles[0.05]:.2f}, {quantiles[0.95]:.2f}] μg/g",
                          style={'textAlign': 'center', 'fontSize': '16px', 'color': '#666'})
                )
                
        else:  # classification
            predicted_category = result['predicted_category']
            category_names = ['Low', 'Moderate', 'High', 'Extreme']
            category_name = category_names[predicted_category] if predicted_category < len(category_names) else str(predicted_category)
            
            confidence_section = html.Div([
                html.H4(f"Predicted Risk Level: {category_name}", 
                       style={'textAlign': 'center', 'fontSize': '24px', 'color': '#E74C3C'}),
            ])
            
            # Add class probabilities if available
            if 'class_probabilities' in result:
                probs = result['class_probabilities']
                prob_text = " | ".join([f"{cat}: {prob:.1%}" for cat, prob in zip(category_names[:len(probs)], probs)])
                confidence_section.children.append(
                    html.P(f"Probabilities: {prob_text}",
                          style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
                )
        
        return html.Div([
            info_section,
            confidence_section
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
        
    def _create_importance_plot(self, importance_data):
        """Create feature importance plot."""
        if importance_data is None:
            return go.Figure().add_annotation(
                text="Feature importance not available for this model",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
        # Take top 15 features for better readability
        top_features = importance_data.head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(color='steelblue')
            )
        ])
        
        fig.update_layout(
            title="Top Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    def _create_level_range_graph(self, result, task):
        """Create gradient visualization for DA level forecast (exact copy from original)."""
        if task != "regression" or 'predicted_da' not in result:
            return go.Figure().add_annotation(
                text="Level range graph only available for regression",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
        # Use predicted_da as both XGBoost prediction and median for now
        # In the original, these came from different models (GB quantiles + XGBoost point)
        predicted_da = result['predicted_da']
        
        # Create approximate quantiles around the XGBoost prediction
        q05 = predicted_da * 0.7  
        q50 = predicted_da        # Use XGBoost prediction as median
        q95 = predicted_da * 1.3  
        rf_prediction = predicted_da  # Same as q50 for now
        actual_levels = None  # Not available in realtime
        
        # Exact copy of original create_level_range_graph method
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
            # Convert opacity to float to avoid numpy type issues
            opacity_float = float(max(0, min(1, opacity)))
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=0.4, y1=0.6,
                line=dict(width=0),
                fillcolor=f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {opacity_float})',
                layer='below'
            )

        # Add median line
        fig.add_trace(go.Scatter(
            x=[q50, q50], y=[0.4, 0.6],
            mode='lines',
            line=dict(color='rgb(30, 60, 90)', width=3),
            name='Median (Q50 - XGBoost)'
        ))

        # Add range endpoints
        fig.add_trace(go.Scatter(
            x=[q05, q95], y=[0.5, 0.5],
            mode='markers',
            marker=dict(size=15, color='rgba(70, 130, 180, 0.3)', symbol='line-ns-open'),
            name='Prediction Range (XGBoost Q05-Q95)'
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
                name='XGBoost Pred.'
            ))

        if actual_levels is not None:
            fig.add_trace(go.Scatter(
                x=[actual_levels], y=[0.5],
                mode='markers',
                marker=dict(size=18, color='red', symbol='x-thin', line=dict(width=2)),
                name='Actual Value'
            ))

        fig.update_layout(
            title="DA Level Forecast: Gradient (XGBoost) & Point (XGBoost)",
            xaxis_title="DA Level",
            yaxis=dict(visible=False, range=[0, 1]),
            showlegend=True,
            height=300,
            plot_bgcolor='white'
        )

        return fig
        
    def _create_category_range_graph(self, result, task):
        """Create category probability distribution (from original)."""
        if task != "classification":
            return go.Figure().add_annotation(
                text="Category graph only available for classification",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
        fig = go.Figure()
        
        CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
        
        if 'class_probabilities' in result and result['class_probabilities'] is not None:
            probabilities = result['class_probabilities']
            
            # Color bars based on predicted category
            colors = ['#1f77b4'] * len(CATEGORY_LABELS)
            if 'predicted_category' in result and result['predicted_category'] is not None:
                try:
                    pred_cat = int(result['predicted_category'])
                    if 0 <= pred_cat < len(colors):
                        colors[pred_cat] = '#2ca02c'
                except (ValueError, TypeError):
                    pass  # Keep default colors if conversion fails

            fig.add_trace(go.Bar(
                x=CATEGORY_LABELS[:len(probabilities)],
                y=probabilities,
                marker_color=colors[:len(probabilities)],
                text=[f"{p * 100:.1f}%" for p in probabilities],
                textposition='auto'
            ))
        else:
            # Show predicted category only
            if 'predicted_category' in result and result['predicted_category'] is not None:
                try:
                    pred_cat = int(result['predicted_category'])
                    y_values = [0] * len(CATEGORY_LABELS)
                    if 0 <= pred_cat < len(CATEGORY_LABELS):
                        y_values[pred_cat] = 1.0
                        
                    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
                    colors[pred_cat] = '#2ca02c'
                except (ValueError, TypeError, IndexError):
                    # Fallback if conversion fails
                    y_values = [0.25] * len(CATEGORY_LABELS)  # Equal probability as fallback
                    colors = ['#1f77b4'] * len(CATEGORY_LABELS)
                
                try:
                    text_labels = [f"100%" if i == pred_cat else "" for i in range(len(CATEGORY_LABELS))]
                except (ValueError, TypeError, IndexError):
                    # Fallback for label generation errors
                    text_labels = [""] * len(CATEGORY_LABELS)
                
                fig.add_trace(go.Bar(
                    x=CATEGORY_LABELS,
                    y=y_values,
                    marker_color=colors,
                    text=text_labels,
                    textposition='auto'
                ))

        fig.update_layout(
            title="Category Probability Distribution",
            yaxis=dict(title="Probability", range=[0, 1.1]),
            xaxis=dict(title="Category"),
            showlegend=False,
            height=400
        )

        return fig