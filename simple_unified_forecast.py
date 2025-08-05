"""
Simplified Unified Domoic Acid Forecasting Application
Addresses data leakage issues without complex sklearn pipeline complications.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from joblib import Parallel, delayed
import random

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')


class SimpleDAForecast:
    """Simplified DA forecasting system without data leakage."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
    def create_features(self, data):
        """Create features without leakage - per forecast."""
        df = data.copy()
        
        # Temporal features
        df['date'] = pd.to_datetime(df['date'])
        day_of_year = df['date'].dt.dayofyear
        df['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Lag features (only for sites with enough history)
        for lag in [1, 2, 3]:
            df[f'da_lag_{lag}'] = df.groupby('site')['da'].shift(lag)
        
        return df
    
    def create_da_categories(self, da_values):
        """Create DA categories based on training data only."""
        return pd.cut(
            da_values,
            bins=[-float('inf'), 5, 20, 40, float('inf')],
            labels=[0, 1, 2, 3],
            right=True
        ).astype('Int64')
    
    def prepare_features(self, df):
        """Prepare features for modeling."""
        # Select numeric features only
        feature_cols = [col for col in df.columns if col not in ['date', 'site', 'da']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        return X_scaled
    
    def forecast_single(self, data, forecast_date, site):
        """Make a single forecast for a specific date and site."""
        try:
            # Filter to site and sort by date
            site_data = data[data['site'] == site].copy()
            site_data = site_data.sort_values('date').reset_index(drop=True)
            
            # Create features
            site_data = self.create_features(site_data)
            
            # Split data by forecast date
            train_data = site_data[site_data['date'] < forecast_date].copy()
            future_data = site_data[site_data['date'] >= forecast_date]
            
            if train_data.empty:
                return None
                
            # Get or create forecast row
            if future_data.empty:
                # Create synthetic forecast row
                forecast_row = train_data.iloc[-1:].copy()
                forecast_row['date'] = forecast_date
                forecast_row['da'] = np.nan
                test_date = None
                actual_da = None
            else:
                forecast_row = future_data.iloc[:1].copy()
                test_date = forecast_row['date'].iloc[0]
                actual_da = forecast_row['da'].iloc[0] if not pd.isna(forecast_row['da'].iloc[0]) else None
            
            # Prepare training data
            X_train = self.prepare_features(train_data)
            y_reg = train_data['da'].values
            
            # Create categories from training data only
            y_cls = self.create_da_categories(y_reg)
            
            # Remove NaN values
            valid_mask = ~(pd.isna(y_reg) | pd.isna(y_cls))
            if valid_mask.sum() < 5:  # Need minimum samples
                return None
                
            X_train_clean = X_train[valid_mask]
            y_reg_clean = y_reg[valid_mask]
            y_cls_clean = y_cls[valid_mask]
            
            # Prepare forecast features
            X_forecast = self.prepare_features(forecast_row)
            
            # Train regression model
            reg_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=1
            )
            reg_model.fit(X_train_clean, y_reg_clean)
            pred_da = reg_model.predict(X_forecast)[0]
            
            # Train classification model (if we have multiple classes)
            pred_category = None
            pred_proba = None
            if len(np.unique(y_cls_clean[~pd.isna(y_cls_clean)])) > 1:
                cls_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=1
                )
                cls_model.fit(X_train_clean, y_cls_clean)
                pred_category = cls_model.predict(X_forecast)[0]
                pred_proba = cls_model.predict_proba(X_forecast)[0]
            
            return {
                'site': site,
                'forecast_date': forecast_date,
                'anchor_date': train_data['date'].max(),
                'test_date': test_date,
                'Predicted_da': pred_da,
                'Actual_da': actual_da,
                'Predicted_da_category': pred_category,
                'Prediction_probabilities': pred_proba
            }
            
        except Exception as e:
            print(f"Error forecasting {site} at {forecast_date}: {e}")
            return None
    
    def generate_random_anchors(self, data, n_anchors_per_site=100, min_test_date="2008-01-01"):
        """Generate random anchor points for evaluation."""
        anchors = []
        min_date = pd.to_datetime(min_test_date)
        
        for site in data['site'].unique():
            site_data = data[data['site'] == site]
            dates = sorted(site_data['date'].unique())
            
            # Only use dates that have future data and meet minimum date requirement
            valid_dates = []
            for i, date in enumerate(dates[:-1]):  # Exclude last date
                if date >= min_date and i < len(dates) - 1:
                    future_dates = [d for d in dates[i+1:] if d >= min_date]
                    if future_dates:
                        valid_dates.append(date)
            
            if valid_dates:
                n_samples = min(len(valid_dates), n_anchors_per_site)
                selected_dates = random.sample(valid_dates, n_samples)
                anchors.extend([(site, date) for date in selected_dates])
        
        return anchors
    
    def evaluate_retrospective(self, data, n_anchors_per_site=100, min_test_date="2008-01-01"):
        """Run retrospective evaluation."""
        print("Generating anchor points...")
        anchors = self.generate_random_anchors(data, n_anchors_per_site, min_test_date)
        print(f"Generated {len(anchors)} anchor points")
        
        print("Running forecasts...")
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self.forecast_single)(data, anchor_date, site) 
            for site, anchor_date in anchors
        )
        
        # Filter successful results
        valid_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(valid_results)} forecasts")
        
        if not valid_results:
            return pd.DataFrame()
            
        return pd.DataFrame(valid_results)


class SimpleForecastApp:
    """Simple forecasting application with dashboards."""
    
    def __init__(self, data_path="final_output.parquet"):
        self.data_path = data_path
        self.forecast_model = SimpleDAForecast()
        self.data = None
        self.results_df = None
        
    def load_data(self):
        """Load and prepare data."""
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"Loaded {len(self.data)} records from {len(self.data['site'].unique())} sites")
        
    def run_retrospective_evaluation(self, n_anchors=100, min_test_date="2008-01-01"):
        """Run retrospective evaluation."""
        self.results_df = self.forecast_model.evaluate_retrospective(
            self.data, n_anchors, min_test_date
        )
        
        if self.results_df.empty:
            print("No successful forecasts generated")
            return
            
        # Calculate performance metrics
        reg_results = self.results_df.dropna(subset=['Actual_da', 'Predicted_da'])
        cls_results = self.results_df.dropna(subset=['Predicted_da_category'])
        cls_results = cls_results[cls_results['Predicted_da_category'].notna()]
        
        print("\n" + "="*60)
        print("RETROSPECTIVE EVALUATION RESULTS")
        print("="*60)
        
        if not reg_results.empty:
            overall_r2 = r2_score(reg_results['Actual_da'], reg_results['Predicted_da'])
            overall_mae = mean_absolute_error(reg_results['Actual_da'], reg_results['Predicted_da'])
            print(f"Regression R²: {overall_r2:.4f}")
            print(f"Regression MAE: {overall_mae:.4f}")
            print(f"Regression samples: {len(reg_results)}")
        
        if not cls_results.empty and 'Actual_da' in cls_results.columns:
            # Create actual categories for comparison
            actual_categories = self.forecast_model.create_da_categories(cls_results['Actual_da'])
            valid_cls = ~pd.isna(actual_categories)
            if valid_cls.sum() > 0:
                accuracy = accuracy_score(
                    actual_categories[valid_cls], 
                    cls_results.loc[valid_cls, 'Predicted_da_category']
                )
                print(f"Classification accuracy: {accuracy:.4f}")
                print(f"Classification samples: {valid_cls.sum()}")
        
        # Site-by-site performance
        if not reg_results.empty:
            site_performance = reg_results.groupby('site').agg({
                'Actual_da': 'count',
                'Predicted_da': lambda x: r2_score(reg_results.loc[x.index, 'Actual_da'], x) if len(x) > 1 else np.nan
            }).rename(columns={'Actual_da': 'n_samples', 'Predicted_da': 'r2'})
            
            print("\nSite Performance:")
            print(site_performance.to_string())
        
        print("="*60)
        
    def create_retrospective_dashboard(self, port=8071):
        """Create retrospective evaluation dashboard."""
        if self.results_df is None or self.results_df.empty:
            print("No results available for dashboard")
            return
            
        app = dash.Dash(__name__)
        sites = ['All Sites'] + sorted(self.results_df['site'].unique().tolist())
        
        app.layout = html.Div([
            html.H1("DA Retrospective Forecast Evaluation (Simplified Pipeline)"),
            
            html.Div([
                html.Label("Select Site:"),
                dcc.Dropdown(
                    id='site-dropdown',
                    options=[{'label': site, 'value': site} for site in sites],
                    value='All Sites',
                    style={'width': '50%', 'marginBottom': '15px'}
                )
            ]),
            
            dcc.Graph(id='scatter-plot'),
            html.Div(id='performance-stats')
        ])
        
        @app.callback(
            [Output('scatter-plot', 'figure'),
             Output('performance-stats', 'children')],
            [Input('site-dropdown', 'value')]
        )
        def update_plot(selected_site):
            df = self.results_df.dropna(subset=['Actual_da', 'Predicted_da'])
            
            if df.empty:
                return px.scatter(title="No data available"), "No data"
                
            if selected_site != 'All Sites':
                df = df[df['site'] == selected_site]
            
            if df.empty:
                return px.scatter(title=f"No data for {selected_site}"), "No data"
            
            # Create scatter plot
            fig = px.scatter(
                df, x='Actual_da', y='Predicted_da',
                color='site' if selected_site == 'All Sites' else None,
                title=f"Actual vs Predicted DA Levels - {selected_site}"
            )
            
            # Add perfect prediction line
            min_val = min(df['Actual_da'].min(), df['Predicted_da'].min())
            max_val = max(df['Actual_da'].max(), df['Predicted_da'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            # Calculate metrics
            r2 = r2_score(df['Actual_da'], df['Predicted_da'])
            mae = mean_absolute_error(df['Actual_da'], df['Predicted_da'])
            
            stats = html.Div([
                html.H3("Performance Statistics"),
                html.P(f"R² Score: {r2:.4f}"),
                html.P(f"Mean Absolute Error: {mae:.4f}"),
                html.P(f"Number of Predictions: {len(df)}")
            ])
            
            return fig, stats
        
        print(f"Starting retrospective dashboard on port {port}")
        app.run_server(debug=False, port=port)
    
    def create_realtime_dashboard(self, port=8065):
        """Create real-time forecasting dashboard."""
        app = dash.Dash(__name__)
        
        sites = sorted(self.data['site'].unique())
        min_date = self.data['date'].min().date()
        max_date = datetime.now().date()
        
        app.layout = html.Div([
            html.H1("DA Real-Time Forecasting (Simplified Pipeline)"),
            
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
                    id='date-picker',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date + timedelta(days=365),
                    date=min_date + timedelta(days=100),
                    style={'marginBottom': '15px'}
                )
            ]),
            
            html.Div(id='forecast-output'),
            dcc.Graph(id='forecast-viz')
        ])
        
        @app.callback(
            [Output('forecast-output', 'children'),
             Output('forecast-viz', 'figure')],
            [Input('site-dropdown', 'value'),
             Input('date-picker', 'date')]
        )
        def update_forecast(site, forecast_date_str):
            try:
                forecast_date = pd.to_datetime(forecast_date_str)
                result = self.forecast_model.forecast_single(self.data, forecast_date, site)
                
                if result is None:
                    return "No forecast possible (insufficient data)", px.bar()
                
                # Format results
                output = html.Div([
                    html.H3(f"Forecast for {site} on {forecast_date.date()}"),
                    html.P(f"Predicted DA Level: {result['Predicted_da']:.2f}"),
                    html.P(f"Anchor Date: {result['anchor_date'].date()}"),
                    html.P(f"Test Date: {result['test_date'].date() if result['test_date'] else 'N/A'}"),
                ])
                
                if result['Actual_da'] is not None:
                    output.children.append(html.P(f"Actual DA Level: {result['Actual_da']:.2f}"))
                
                # Create simple visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Predicted'],
                    y=[result['Predicted_da']],
                    name='Predicted DA'
                ))
                
                if result['Actual_da'] is not None:
                    fig.add_trace(go.Bar(
                        x=['Actual'],
                        y=[result['Actual_da']],
                        name='Actual DA'
                    ))
                
                fig.update_layout(title="DA Level Forecast", yaxis_title="DA Level")
                
                return output, fig
                
            except Exception as e:
                return f"Error: {e}", px.bar()
        
        print(f"Starting real-time dashboard on port {port}")
        app.run_server(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description="Simplified DA Forecasting Application")
    parser.add_argument('--mode', choices=['retrospective', 'realtime'], required=True)
    parser.add_argument('--data', default='final_output.parquet')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--anchors', type=int, default=100)
    parser.add_argument('--min-test-date', default='2008-01-01')
    
    args = parser.parse_args()
    
    # Initialize app
    app = SimpleForecastApp(args.data)
    app.load_data()
    
    if args.mode == 'retrospective':
        port = args.port or 8071
        app.run_retrospective_evaluation(args.anchors, args.min_test_date)
        app.create_retrospective_dashboard(port)
    else:  # realtime
        port = args.port or 8065
        app.create_realtime_dashboard(port)


if __name__ == "__main__":
    main()