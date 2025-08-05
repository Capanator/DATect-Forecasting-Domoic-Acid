"""
Unified Domoic Acid Forecasting Application

This replaces both past-forecasts-final.py and future-forecasts.py with a proper,
leak-free pipeline implementation.

Usage:
    python unified_forecast_app.py --mode retrospective  # For evaluation mode
    python unified_forecast_app.py --mode realtime      # For real-time forecasting
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from pipeline import DAForecastPipeline, DAForecastConfig


class ForecastApp:
    """Unified forecasting application."""
    
    def __init__(self, config_path: str = "final_output.parquet"):
        self.config_path = config_path
        self.pipeline = None
        self.data = None
        self.results_df = None
        
    def load_data(self):
        """Load and prepare data."""
        print("Loading data...")
        self.data = pd.read_parquet(self.config_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"Loaded {len(self.data)} records from {len(self.data['site'].unique())} sites")
        
    def initialize_pipeline(self, enable_lag_features: bool = True):
        """Initialize the forecasting pipeline."""
        config = DAForecastConfig.create_config(
            enable_lag_features=enable_lag_features,
            random_state=42,
            n_jobs=-1
        )
        
        self.pipeline = DAForecastPipeline(config)
        self.pipeline.fit(self.data)
        
    def run_retrospective_evaluation(self, n_anchors: int = 500, min_test_date: str = "2008-01-01"):
        """Run retrospective evaluation mode."""
        print(f"Running retrospective evaluation with {n_anchors} anchors per site...")
        
        self.results_df = self.pipeline.evaluate_retrospective(
            self.data, 
            n_anchors_per_site=n_anchors,
            min_test_date=min_test_date
        )
        
        # Evaluate performance
        overall_performance = self.pipeline.evaluate_performance(self.results_df)
        site_performance = self.pipeline.get_site_performance(self.results_df)
        
        print("\n" + "="*60)
        print("RETROSPECTIVE EVALUATION RESULTS")
        print("="*60)
        
        if 'regression_r2' in overall_performance:
            print(f"Overall Regression R²: {overall_performance['regression_r2']:.4f}")
            print(f"Overall Regression MAE: {overall_performance['regression_mae']:.4f}")
            print(f"Regression Samples: {overall_performance['regression_n_samples']}")
            
        if 'classification_accuracy' in overall_performance:
            print(f"Overall Classification Accuracy: {overall_performance['classification_accuracy']:.4f}")
            print(f"Classification Samples: {overall_performance['classification_n_samples']}")
            
        if 'quantile_coverage' in overall_performance:
            print(f"Quantile Coverage (90% interval): {overall_performance['quantile_coverage']:.4f}")
            print(f"Coverage Samples: {overall_performance['quantile_n_samples']}")
        
        print("\nSite-by-site performance:")
        print(site_performance.to_string(index=False))
        print("="*60)
        
        return self.results_df
    
    def create_retrospective_dashboard(self, port: int = 8071):
        """Create dashboard for retrospective evaluation results."""
        if self.results_df is None:
            raise ValueError("No retrospective results available. Run evaluation first.")
            
        app = dash.Dash(__name__)
        
        sites = sorted(self.results_df['site'].unique())
        
        app.layout = html.Div([
            html.H1("Domoic Acid Retrospective Forecast Evaluation"),
            
            html.Div([
                html.Label("Select Evaluation Type:"),
                dcc.Dropdown(
                    id='eval-type-dropdown',
                    options=[
                        {'label': 'DA Levels (Regression)', 'value': 'regression'},
                        {'label': 'DA Category (Classification)', 'value': 'classification'},
                        {'label': 'Quantile Coverage', 'value': 'quantile'}
                    ],
                    value='regression',
                    style={'width': '50%', 'marginBottom': '15px'}
                ),
                
                html.Label("Select Site:"),
                dcc.Dropdown(
                    id='site-dropdown',
                    options=[{'label': 'All Sites', 'value': 'all'}] + 
                           [{'label': site, 'value': site} for site in sites],
                    value='all',
                    style={'width': '50%', 'marginBottom': '15px'}
                )
            ]),
            
            dcc.Graph(id='evaluation-graph'),
            
            html.Div(id='performance-summary', 
                    style={'marginTop': '20px', 'padding': '10px', 
                          'border': '1px solid #ccc', 'backgroundColor': '#f9f9f9'})
        ])
        
        @app.callback(
            [Output('evaluation-graph', 'figure'),
             Output('performance-summary', 'children')],
            [Input('eval-type-dropdown', 'value'),
             Input('site-dropdown', 'value')]
        )
        def update_evaluation_display(eval_type, selected_site):
            df = self.results_df.copy()
            
            if selected_site != 'all':
                df = df[df['site'] == selected_site]
            
            if eval_type == 'regression':
                # Create regression evaluation plot
                df_clean = df.dropna(subset=['Actual_da', 'Predicted_da'])
                
                if df_clean.empty:
                    fig = px.scatter(title="No data available for regression evaluation")
                    summary = "No regression data available"
                else:
                    fig = px.scatter(
                        df_clean, x='Actual_da', y='Predicted_da', 
                        color='site' if selected_site == 'all' else None,
                        title=f"Regression Performance - {selected_site if selected_site != 'all' else 'All Sites'}"
                    )
                    
                    # Add perfect prediction line
                    min_val = min(df_clean['Actual_da'].min(), df_clean['Predicted_da'].min())
                    max_val = max(df_clean['Actual_da'].max(), df_clean['Predicted_da'].max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    # Calculate metrics
                    from sklearn.metrics import r2_score, mean_absolute_error
                    r2 = r2_score(df_clean['Actual_da'], df_clean['Predicted_da'])
                    mae = mean_absolute_error(df_clean['Actual_da'], df_clean['Predicted_da'])
                    
                    summary = f"""
                    **Regression Performance Summary**
                    - R² Score: {r2:.4f}
                    - Mean Absolute Error: {mae:.4f}
                    - Number of Predictions: {len(df_clean)}
                    """
                    
            elif eval_type == 'classification':
                # Create classification evaluation plot
                df_clean = df.dropna(subset=['Actual_da_category', 'Predicted_da_category'])
                
                if df_clean.empty:
                    fig = px.bar(title="No data available for classification evaluation")
                    summary = "No classification data available"
                else:
                    # Confusion matrix heatmap
                    from sklearn.metrics import confusion_matrix, accuracy_score
                    cm = confusion_matrix(df_clean['Actual_da_category'], 
                                        df_clean['Predicted_da_category'])
                    
                    fig = px.imshow(
                        cm, text_auto=True, aspect="auto",
                        title=f"Confusion Matrix - {selected_site if selected_site != 'all' else 'All Sites'}",
                        labels=dict(x="Predicted Category", y="Actual Category")
                    )
                    
                    accuracy = accuracy_score(df_clean['Actual_da_category'], 
                                            df_clean['Predicted_da_category'])
                    
                    summary = f"""
                    **Classification Performance Summary**
                    - Accuracy: {accuracy:.4f}
                    - Number of Predictions: {len(df_clean)}
                    """
                    
            else:  # quantile
                # Create quantile coverage plot
                df_clean = df.dropna(subset=['Actual_da', 'Predicted_da_q05', 'Predicted_da_q95'])
                
                if df_clean.empty:
                    fig = px.scatter(title="No data available for quantile evaluation")
                    summary = "No quantile data available"
                else:
                    # Coverage plot
                    df_clean = df_clean.sort_values('forecast_date')
                    df_clean['within_interval'] = (
                        (df_clean['Actual_da'] >= df_clean['Predicted_da_q05']) & 
                        (df_clean['Actual_da'] <= df_clean['Predicted_da_q95'])
                    )
                    
                    fig = go.Figure()
                    
                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=df_clean['forecast_date'],
                        y=df_clean['Predicted_da_q95'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_clean['forecast_date'],
                        y=df_clean['Predicted_da_q05'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        name='90% Prediction Interval',
                        fillcolor='rgba(0,100,80,0.2)'
                    ))
                    
                    # Add actual values
                    colors = ['green' if x else 'red' for x in df_clean['within_interval']]
                    fig.add_trace(go.Scatter(
                        x=df_clean['forecast_date'],
                        y=df_clean['Actual_da'],
                        mode='markers',
                        marker=dict(color=colors),
                        name='Actual Values'
                    ))
                    
                    fig.update_layout(
                        title=f"Quantile Coverage - {selected_site if selected_site != 'all' else 'All Sites'}",
                        xaxis_title="Forecast Date",
                        yaxis_title="DA Level"
                    )
                    
                    coverage = df_clean['within_interval'].mean()
                    summary = f"""
                    **Quantile Performance Summary**
                    - Coverage (90% interval): {coverage:.4f}
                    - Expected Coverage: 0.90
                    - Number of Predictions: {len(df_clean)}
                    """
            
            return fig, dcc.Markdown(summary)
        
        print(f"Starting retrospective evaluation dashboard on port {port}")
        app.run_server(debug=False, port=port)
    
    def create_realtime_dashboard(self, port: int = 8065):
        """Create dashboard for real-time forecasting."""
        app = dash.Dash(__name__)
        
        sites = sorted(self.data['site'].unique())
        min_date = self.data['date'].min().date()
        max_date = datetime.now().date()
        
        app.layout = html.Div([
            html.H1("Domoic Acid Real-Time Forecasting"),
            
            html.Div([
                html.Label("Select Site:"),
                dcc.Dropdown(
                    id='site-dropdown',
                    options=[{'label': site, 'value': site} for site in sites],
                    value=sites[0],
                    style={'width': '50%', 'marginBottom': '15px'}
                ),
                
                html.Label("Select Forecast Date:"),
                dcc.DatePickerSingle(
                    id='forecast-date-picker',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date + timedelta(days=365),
                    date=min_date + timedelta(days=30),
                    style={'marginBottom': '15px'}
                )
            ]),
            
            html.Div(id='forecast-results', style={'marginTop': '20px'}),
            
            html.Div([
                dcc.Graph(id='regression-plot', style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(id='classification-plot', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ])
        ])
        
        @app.callback(
            [Output('forecast-results', 'children'),
             Output('regression-plot', 'figure'),
             Output('classification-plot', 'figure')],
            [Input('site-dropdown', 'value'),
             Input('forecast-date-picker', 'date')]
        )
        def update_forecast(site, forecast_date_str):
            try:
                forecast_date = pd.to_datetime(forecast_date_str)
                result = self.pipeline.forecast_single(self.data, forecast_date, site)
                
                # Format results text
                results_text = f"""
                **Forecast for {site} on {forecast_date.date()}**
                
                **Regression Results:**
                - Point Prediction: {result.get('Predicted_da', 'N/A'):.2f}
                - 90% Prediction Interval: [{result.get('Predicted_da_q05', 'N/A'):.2f}, {result.get('Predicted_da_q95', 'N/A'):.2f}]
                - Median Prediction: {result.get('Predicted_da_q50', 'N/A'):.2f}
                
                **Classification Results:**
                - Predicted Category: {result.get('Predicted_da_category', 'N/A')}
                
                **Training Info:**
                - Anchor Date: {result.get('anchor_date', 'N/A').date() if result.get('anchor_date') else 'N/A'}
                - Test Date: {result.get('test_date', 'N/A').date() if result.get('test_date') else 'N/A'}
                """
                
                if 'Actual_da' in result:
                    results_text += f"\n**Actual Value:** {result['Actual_da']:.2f}"
                
                # Create regression plot
                reg_fig = go.Figure()
                
                if 'Predicted_da_q05' in result and 'Predicted_da_q95' in result:
                    # Add prediction interval
                    reg_fig.add_trace(go.Scatter(
                        x=[0, 1], y=[result['Predicted_da_q05'], result['Predicted_da_q05']],
                        fill=None, mode='lines', line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ))
                    reg_fig.add_trace(go.Scatter(
                        x=[0, 1], y=[result['Predicted_da_q95'], result['Predicted_da_q95']],
                        fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                        name='90% Prediction Interval', fillcolor='rgba(0,100,80,0.3)'
                    ))
                
                # Add point prediction
                if 'Predicted_da' in result:
                    reg_fig.add_trace(go.Scatter(
                        x=[0.5], y=[result['Predicted_da']],
                        mode='markers', marker=dict(size=15, color='blue'),
                        name='Point Prediction'
                    ))
                
                # Add actual value if available
                if 'Actual_da' in result:
                    reg_fig.add_trace(go.Scatter(
                        x=[0.5], y=[result['Actual_da']],
                        mode='markers', marker=dict(size=15, color='red', symbol='x'),
                        name='Actual Value'
                    ))
                
                reg_fig.update_layout(
                    title="DA Level Forecast",
                    yaxis_title="DA Level",
                    xaxis=dict(visible=False),
                    height=400
                )
                
                # Create classification plot
                cls_fig = go.Figure()
                
                if 'Prediction_probabilities' in result:
                    categories = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']
                    probs = result['Prediction_probabilities']
                    
                    cls_fig.add_trace(go.Bar(
                        x=categories, y=probs,
                        text=[f"{p:.1%}" for p in probs],
                        textposition='auto'
                    ))
                
                cls_fig.update_layout(
                    title="Category Probabilities",
                    yaxis_title="Probability",
                    height=400
                )
                
                return dcc.Markdown(results_text), reg_fig, cls_fig
                
            except Exception as e:
                error_msg = f"Error generating forecast: {str(e)}"
                empty_fig = px.scatter(title="Error")
                return dcc.Markdown(f"**{error_msg}**"), empty_fig, empty_fig
        
        print(f"Starting real-time forecasting dashboard on port {port}")
        app.run_server(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description="Unified DA Forecasting Application")
    parser.add_argument('--mode', choices=['retrospective', 'realtime'], required=True,
                       help="Run mode: retrospective evaluation or real-time forecasting")
    parser.add_argument('--data', default='final_output.parquet',
                       help="Path to data file")
    parser.add_argument('--port', type=int, default=None,
                       help="Port for dashboard (default: 8071 for retrospective, 8065 for realtime)")
    parser.add_argument('--anchors', type=int, default=500,
                       help="Number of anchor points per site for retrospective evaluation")
    parser.add_argument('--min-test-date', default='2008-01-01',
                       help="Minimum test date for retrospective evaluation")
    parser.add_argument('--enable-lag-features', action='store_true', default=True,
                       help="Enable lag features in the model")
    
    args = parser.parse_args()
    
    # Initialize app
    app = ForecastApp(args.data)
    app.load_data()
    app.initialize_pipeline(args.enable_lag_features)
    
    if args.mode == 'retrospective':
        port = args.port or 8071
        app.run_retrospective_evaluation(args.anchors, args.min_test_date)
        app.create_retrospective_dashboard(port)
    else:  # realtime
        port = args.port or 8065
        app.create_realtime_dashboard(port)


if __name__ == "__main__":
    main()