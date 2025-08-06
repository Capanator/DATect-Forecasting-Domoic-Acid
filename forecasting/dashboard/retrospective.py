from forecasting.core.logging_config import setup_logging, get_logger
from forecasting.core.exception_handling import safe_execute
"""
Retrospective Dashboard
======================

Interactive dashboard for visualizing historical forecasting performance
and model validation results with leak-free temporal safeguards.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

import config


class RetrospectiveDashboard:
    """
    Dashboard for retrospective forecasting evaluation.
    
    Features:
    - Interactive site selection
    - Time series visualization
    - Performance metrics
    - Model comparison plots
    """
    
    def __init__(self, results_df):
        """
        Initialize dashboard with results data.
        
        Args:
            results_df: DataFrame with forecasting results
        """
        self.results_df = results_df
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def run(self, port=8071, debug=False):
        """
        Run the dashboard server.
        
        Args:
            port: Port to run server on
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting retrospective dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)
        
    def _setup_layout(self):
        """Setup the dashboard layout."""
        sites_list = sorted(self.results_df["site"].unique().tolist())
        
        self.app.layout = html.Div([
            html.H1("Domoic Acid Forecast Dashboard",
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
                   
            html.Div([
                html.Label("Select Site:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id="site-dropdown",
                    options=[{"label": "All sites", "value": "All sites"}] + [
                        {"label": site, "value": site} for site in sites_list
                    ],
                    value="All sites",
                    style={'marginBottom': '20px'}
                ),
            ], style={'width': '300px', 'margin': '0 auto'}),
            
            # Performance metrics
            html.Div(id="metrics-display", style={'marginBottom': '30px'}),
            
            # Main plots
            html.Div([
                dcc.Graph(id="forecast-plot", style={'height': '500px'}),
            ]),
            
            html.Div([
                dcc.Graph(id="scatter-plot", style={'height': '400px'}),
            ]),
            
        ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("metrics-display", "children"),
             Output("forecast-plot", "figure"),
             Output("scatter-plot", "figure")],
            [Input("site-dropdown", "value")]
        )
        def update_dashboard(selected_site):
            # Filter data based on selection
            if selected_site == "All sites":
                filtered_df = self.results_df.copy()
                title_suffix = "All Sites"
            else:
                filtered_df = self.results_df[self.results_df["site"] == selected_site].copy()
                title_suffix = selected_site
                
            # Calculate metrics
            metrics_div = self._create_metrics_display(filtered_df)
            
            # Create plots
            forecast_fig = self._create_forecast_plot(filtered_df, title_suffix, selected_site)
            scatter_fig = self._create_scatter_plot(filtered_df, title_suffix)
            
            return metrics_div, forecast_fig, scatter_fig
            
    def _create_metrics_display(self, df):
        """Create metrics display component."""
        if df.empty:
            return html.Div("No data available", style={'textAlign': 'center'})
            
        metrics = []
        
        # Regression metrics (using original column names)
        if 'Predicted_da' in df.columns:
            valid_reg = df.dropna(subset=['da', 'Predicted_da'])
            if not valid_reg.empty:
                r2 = r2_score(valid_reg['da'], valid_reg['Predicted_da'])
                mae = mean_absolute_error(valid_reg['da'], valid_reg['Predicted_da'])
                
                metrics.extend([
                    html.Div([
                        html.H3("Regression Performance", style={'color': '#2E86C1'}),
                        html.P(f"R² Score: {r2:.4f}", style={'fontSize': '18px', 'margin': '5px 0'}),
                        html.P(f"MAE: {mae:.2f} μg/g", style={'fontSize': '18px', 'margin': '5px 0'}),
                        html.P(f"Forecasts: {len(valid_reg)}", style={'fontSize': '16px', 'color': '#666'})
                    ], style={'display': 'inline-block', 'margin': '0 30px', 'textAlign': 'center'})
                ])
                
        # Classification metrics (using original column names)
        if 'Predicted_da-category' in df.columns:
            valid_cls = df.dropna(subset=['da-category', 'Predicted_da-category'])
            if not valid_cls.empty:
                accuracy = accuracy_score(valid_cls['da-category'], valid_cls['Predicted_da-category'])
                
                metrics.extend([
                    html.Div([
                        html.H3("Classification Performance", style={'color': '#E74C3C'}),
                        html.P(f"Accuracy: {accuracy:.4f}", style={'fontSize': '18px', 'margin': '5px 0'}),
                        html.P(f"Forecasts: {len(valid_cls)}", style={'fontSize': '16px', 'color': '#666'})
                    ], style={'display': 'inline-block', 'margin': '0 30px', 'textAlign': 'center'})
                ])
                
        if not metrics:
            return html.Div("No valid predictions available", style={'textAlign': 'center'})
            
        return html.Div(metrics, style={
            'backgroundColor': '#f8f9fa', 
            'padding': '20px', 
            'borderRadius': '10px',
            'textAlign': 'center',
            'marginBottom': '20px'
        })
        
    def _create_forecast_plot(self, df, title_suffix, selected_site="All sites"):
        """Create time series forecast plot."""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
        fig = go.Figure()
        
        # Check if this is regression or classification data
        is_regression = 'Predicted_da' in df.columns
        is_classification = 'Predicted_da-category' in df.columns
        
        if is_regression:
            # Plot regression data (original style)
            valid_df = df.dropna(subset=['da', 'Predicted_da']).copy()
            
            if not valid_df.empty:
                # Sort by date for better line plotting
                valid_df = valid_df.sort_values('date')
                
                if selected_site == "All sites":
                    # Plot by site when showing all sites
                    for site in valid_df['site'].unique():
                        site_data = valid_df[valid_df['site'] == site]
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['da'],
                            mode='lines+markers', name=f'{site} - Actual',
                            line=dict(dash='solid'),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Actual DA: %{y:.2f} μg/g<extra></extra>',
                            text=[site] * len(site_data)
                        ))
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['Predicted_da'],
                            mode='lines+markers', name=f'{site} - Predicted',
                            line=dict(dash='dash'),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Predicted DA: %{y:.2f} μg/g<extra></extra>',
                            text=[site] * len(site_data)
                        ))
                else:
                    # Single site view
                    fig.add_trace(go.Scatter(
                        x=valid_df['date'], y=valid_df['da'],
                        mode='lines+markers', name='Actual DA',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Actual DA: %{y:.2f} μg/g<extra></extra>',
                        text=valid_df['site']
                    ))
                    fig.add_trace(go.Scatter(
                        x=valid_df['date'], y=valid_df['Predicted_da'],
                        mode='lines+markers', name='Predicted DA',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Predicted DA: %{y:.2f} μg/g<extra></extra>',
                        text=valid_df['site']
                    ))
                    
            fig.update_layout(
                title=f"DA Forecasting Results - {title_suffix}",
                xaxis_title="Date",
                yaxis_title="DA Concentration (μg/g)",
                hovermode='closest',
                height=500
            )
                
        elif is_classification:
            # Plot classification data (original style)
            valid_df = df.dropna(subset=['da-category', 'Predicted_da-category']).copy()
            
            if not valid_df.empty:
                # Sort by date for better line plotting
                valid_df = valid_df.sort_values('date')
                
                if selected_site == "All sites":
                    # Plot by site when showing all sites
                    for site in valid_df['site'].unique():
                        site_data = valid_df[valid_df['site'] == site]
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['da-category'],
                            mode='lines+markers', name=f'{site} - Actual Category',
                            line=dict(dash='solid'),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Actual Category: %{y}<extra></extra>',
                            text=[site] * len(site_data)
                        ))
                        fig.add_trace(go.Scatter(
                            x=site_data['date'], y=site_data['Predicted_da-category'],
                            mode='lines+markers', name=f'{site} - Predicted Category',
                            line=dict(dash='dash'),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Predicted Category: %{y}<extra></extra>',
                            text=[site] * len(site_data)
                        ))
                else:
                    # Single site view
                    fig.add_trace(go.Scatter(
                        x=valid_df['date'], y=valid_df['da-category'],
                        mode='lines+markers', name='Actual Category',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Actual Category: %{y}<extra></extra>',
                        text=valid_df['site']
                    ))
                    fig.add_trace(go.Scatter(
                        x=valid_df['date'], y=valid_df['Predicted_da-category'],
                        mode='lines+markers', name='Predicted Category',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Predicted Category: %{y}<extra></extra>',
                        text=valid_df['site']
                    ))
                    
            fig.update_layout(
                title=f"DA Category Forecasting Results - {title_suffix}",
                xaxis_title="Date",
                yaxis_title="DA Risk Category",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Low', 'Moderate', 'High', 'Extreme']
                ),
                hovermode='closest',
                height=500
            )
        else:
            fig.add_annotation(
                text="No forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
        
    def _create_scatter_plot(self, df, title_suffix):
        """Create actual vs predicted scatter plot."""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
        fig = go.Figure()
        
        # Check if this is regression or classification data
        is_regression = 'Predicted_da' in df.columns
        is_classification = 'Predicted_da-category' in df.columns
        
        if is_regression:
            valid_df = df.dropna(subset=['da', 'Predicted_da'])
            
            if valid_df.empty:
                return go.Figure().add_annotation(
                    text="No valid regression predictions available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            # Calculate R² for display
            r2 = r2_score(valid_df['da'], valid_df['Predicted_da'])
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=valid_df['da'],
                y=valid_df['Predicted_da'],
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Predictions',
                hovertemplate='<b>%{text}</b><br>' +
                            'Actual: %{x:.2f} μg/g<br>' +
                            'Predicted: %{y:.2f} μg/g<extra></extra>',
                text=valid_df['site']
            ))
            
            # Perfect prediction line
            max_val = max(valid_df['da'].max(), valid_df['Predicted_da'].max())
            min_val = min(valid_df['da'].min(), valid_df['Predicted_da'].min())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=f"Actual vs Predicted DA - {title_suffix} (R² = {r2:.3f})",
                xaxis_title="Actual DA (μg/g)",
                yaxis_title="Predicted DA (μg/g)",
                height=400
            )
            
        elif is_classification:
            valid_df = df.dropna(subset=['da-category', 'Predicted_da-category'])
            
            if valid_df.empty:
                return go.Figure().add_annotation(
                    text="No valid classification predictions available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                
            # Calculate accuracy for display
            accuracy = accuracy_score(valid_df['da-category'], valid_df['Predicted_da-category'])
            
            # Create confusion matrix-style scatter plot
            # Add some jitter to see overlapping points
            import numpy as np
            np.random.seed(42)  # For reproducible jitter
            jitter_strength = 0.1
            
            # Convert to numpy arrays and handle any numpy dtypes
            actual_cats = np.array(valid_df['da-category'].values, dtype=float)
            predicted_cats = np.array(valid_df['Predicted_da-category'].values, dtype=float)
            
            x_jitter = actual_cats + np.random.normal(0, jitter_strength, size=len(valid_df))
            y_jitter = predicted_cats + np.random.normal(0, jitter_strength, size=len(valid_df))
            
            fig.add_trace(go.Scatter(
                x=x_jitter,
                y=y_jitter,
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Predictions',
                hovertemplate='<b>%{text}</b><br>' +
                            'Actual Category: %{customdata[0]}<br>' +
                            'Predicted Category: %{customdata[1]}<extra></extra>',
                text=valid_df['site'],
                customdata=list(zip(valid_df['da-category'], valid_df['Predicted_da-category']))
            ))
            
            # Perfect prediction line
            fig.add_trace(go.Scatter(
                x=[0, 3], y=[0, 3],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=f"Actual vs Predicted Category - {title_suffix} (Accuracy = {accuracy:.3f})",
                xaxis=dict(
                    title="Actual DA Category",
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Low', 'Moderate', 'High', 'Extreme'],
                    range=[-0.5, 3.5]
                ),
                yaxis=dict(
                    title="Predicted DA Category",
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Low', 'Moderate', 'High', 'Extreme'],
                    range=[-0.5, 3.5]
                ),
                height=400
            )
        else:
            fig.add_annotation(
                text="No prediction data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig

# Setup logging
setup_logging(log_level='INFO', log_dir='./logs/', enable_file_logging=True)
logger = get_logger(__name__)
