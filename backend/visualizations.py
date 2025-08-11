"""
Backend visualization module for DATect web application.
Implements all visualization logic from the original Python analysis scripts.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
# Removed: from sklearn.model_selection import train_test_split  # Use temporal splits instead
from sklearn.inspection import permutation_importance
from scipy import signal
from scipy.stats import pearsonr
import warnings
import os
import os
import sys
import plotly.graph_objs as go
import plotly.io as pio
import pywt  # Wavelet analysis
from statsmodels.tsa.seasonal import STL  # Seasonal decomposition
from scipy.signal import windows  # For multitaper analysis

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_gradient_uncertainty_plot(gradient_quantiles, xgboost_prediction, actual_da=None):
    """
    Create advanced gradient visualization for DA level forecast with quantile uncertainty.
    
    Args:
        gradient_quantiles: dict with 'q05', 'q50', 'q95' keys
        xgboost_prediction: float, XGBoost point prediction 
        actual_da: float, optional actual DA value
    
    Returns:
        dict: Plotly figure JSON data
    """
    q05 = gradient_quantiles['q05']
    q50 = gradient_quantiles['q50'] 
    q95 = gradient_quantiles['q95']
    
    fig = go.Figure()
    
    # Create gradient confidence area with 30 segments
    n_segments = 30
    range_width = q95 - q05
    max_distance = max(q50 - q05, q95 - q50) if range_width > 1e-6 else 1
    if max_distance <= 1e-6:
        max_distance = 1  # Avoid division by zero
    
    base_color = (70, 130, 180)  # Steel blue
    
    # Generate gradient confidence bands
    for i in range(n_segments):
        x0 = q05 + (i / n_segments) * range_width
        x1 = q05 + ((i + 1) / n_segments) * range_width
        midpoint = (x0 + x1) / 2
        
        # Calculate opacity based on distance from median
        opacity = 1 - (abs(midpoint - q50) / max_distance) ** 0.5 if max_distance > 1e-6 else 0.8
        opacity = max(0.1, min(0.9, opacity))  # Ensure valid opacity range
        
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.35, y1=0.65,
            line=dict(width=0),
            fillcolor=f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {opacity})',
            layer='below'
        )
    
    # Add gradient boosting median line (q50)
    fig.add_trace(go.Scatter(
        x=[q50, q50], y=[0.35, 0.65],
        mode='lines',
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='GB Median (Q50)',
        hovertemplate='GB Median: %{x:.2f}<extra></extra>'
    ))
    
    # Add quantile range endpoints
    fig.add_trace(go.Scatter(
        x=[q05, q95], y=[0.5, 0.5],
        mode='markers',
        marker=dict(size=12, color='rgba(70, 130, 180, 0.4)', symbol='line-ns-open'),
        name='GB Range (Q05-Q95)',
        hovertemplate='GB Range: %{x:.2f}<extra></extra>'
    ))
    
    # Add XGBoost point prediction
    fig.add_trace(go.Scatter(
        x=[xgboost_prediction], y=[0.5],
        mode='markers',
        marker=dict(
            size=14,
            color='darkorange',
            symbol='diamond-tall',
            line=dict(width=2, color='black')
        ),
        name='XGBoost Prediction',
        hovertemplate='XGBoost: %{x:.2f}<extra></extra>'
    ))
    
    # Add actual value if available
    if actual_da is not None:
        fig.add_trace(go.Scatter(
            x=[actual_da], y=[0.5],
            mode='markers',
            marker=dict(size=16, color='red', symbol='x-thin', line=dict(width=3)),
            name='Actual Value',
            hovertemplate='Actual: %{x:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Advanced DA Level Forecast: Gradient Boosting Quantiles + XGBoost Point",
        xaxis_title="DA Level (μg/L)",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=True,
        height=350,
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return pio.to_json(fig)


def sophisticated_nan_handling_for_correlation(df, preserve_temporal=True):
    """
    Implements sophisticated NaN handling strategy from modular-forecast.
    """
    if df.empty:
        return df
        
    df_processed = df.copy()
    
    # Sort by date if available to maintain temporal integrity
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
    
    # Separate target variable (DA) from features
    if 'da' in df_processed.columns:
        # Identify numeric columns for imputation (excluding target)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'da']
        
        if feature_cols and len(df_processed) > 0:
            # Use median imputation for feature variables only
            imputer = SimpleImputer(strategy="median")
            df_processed[feature_cols] = imputer.fit_transform(df_processed[feature_cols])
    
    return df_processed


def generate_correlation_heatmap(data, site=None):
    """Generate correlation heatmap matching the original Python implementation."""
    
    if site:
        # Filter by site
        df = data[data['site'] == site].copy()
        title = f'Correlation Heatmap - {site}'
    else:
        # Use all data
        df = data.copy()
        title = 'Overall Correlation Heatmap'
    
    # Check if we have data
    if df.empty or len(df) < 2:
        # Return empty plot
        return {
            "data": [{
                "type": "heatmap",
                "z": [[0]],
                "x": ["No Data"],
                "y": ["No Data"],
                "colorscale": [
                    [0.0, "rgb(178, 24, 43)"],
                    [0.5, "rgb(255, 255, 255)"],
                    [1.0, "rgb(33, 102, 172)"]
                ],
                "showscale": False
            }],
            "layout": {
                "title": title + " - Insufficient Data",
                "height": 600,
                "width": 800
            }
        }
    
    # Apply sophisticated NaN handling
    df = sophisticated_nan_handling_for_correlation(df)
    
    # Drop non-numeric columns
    if 'site' in df.columns:
        df = df.drop(columns=['site'])
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Drop spatial columns
    cols_to_drop = [col for col in ['lon', 'lat'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    # Compute correlation matrix using pandas default (pairwise deletion)
    corr_matrix = numeric_df.corr(method='pearson')
    
    # Create annotations for each cell
    annotations = []
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            # Use white text for dark cells (strong correlations), black otherwise
            color = "white" if abs(value) > 0.7 else "black"
            annotations.append({
                "x": corr_matrix.columns[j],
                "y": corr_matrix.index[i],
                "text": f"{value:.2f}",
                "font": {"color": color, "size": 10},
                "showarrow": False
            })
    
    # Create plotly heatmap matching original style
    plot_data = {
        "data": [{
            "type": "heatmap",
            "z": corr_matrix.values.tolist(),
            "x": corr_matrix.columns.tolist(),
            "y": corr_matrix.index.tolist(),
            "colorscale": [
                [0.0, "rgb(178, 24, 43)"],  # -1: dark red
                [0.25, "rgb(239, 138, 98)"], # -0.5: light red
                [0.5, "rgb(255, 255, 255)"], # 0: white
                [0.75, "rgb(103, 169, 207)"], # 0.5: light blue
                [1.0, "rgb(33, 102, 172)"]   # +1: dark blue
            ],
            "zmid": 0,
            "zmin": -1,
            "zmax": 1,
            "colorbar": {
                "title": "Correlation (r)",
                "titleside": "right",
                "tickmode": "linear",
                "tick0": -1,
                "dtick": 0.5
            }
        }],
        "layout": {
            "title": {
                "text": title,
                "font": {"size": 20}
            },
            "height": 600,
            "width": 800,
            "xaxis": {
                "side": "bottom",
                "tickangle": -45,
                "tickfont": {"size": 12}
            },
            "yaxis": {
                "side": "left",
                "tickfont": {"size": 12}
            },
            "annotations": annotations,
            "margin": {"l": 150, "r": 150, "t": 100, "b": 150}
        }
    }
    
    return plot_data


def generate_sensitivity_analysis(data):
    """Generate sensitivity analysis plots including Sobol indices if possible."""
    
    # Apply sophisticated NaN handling
    df_processed = data.copy()
    
    # Maintain temporal ordering if date column exists
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
    
    # Remove samples with NaN target values
    df_clean = df_processed.dropna(subset=['da']).copy()
    
    # Identify feature columns
    exclude_cols = ['da', 'date', 'site', 'lon', 'lat']
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if feature_cols:
        # Use median imputation for features
        imputer = SimpleImputer(strategy="median")
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    
    # Compute absolute Pearson correlation with DA
    correlations = df_clean[feature_cols + ['da']].corr()['da'].drop('da').abs().sort_values(ascending=False)
    
    # Plot 1: Correlation Sensitivity Analysis
    plot1 = {
        "data": [{
            "type": "bar",
            "x": correlations.index.tolist(),
            "y": correlations.values.tolist(),
            "marker": {"color": "steelblue"}
        }],
        "layout": {
            "title": {
                "text": "Correlation Sensitivity Analysis: Impact on DA Levels",
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Input Variables",
                "tickangle": -45,
                "tickfont": {"size": 12}
            },
            "yaxis": {
                "title": "Absolute Pearson Correlation",
                "titlefont": {"size": 14}
            },
            "height": 500
        }
    }
    
    plots = [plot1]
    
    # Prepare data for model-based methods
    X = df_clean[feature_cols]
    y = df_clean['da']
    
    # Split data for training using temporal ordering (prevent data leakage)
    if 'date' in df_clean.columns:
        # Use temporal split: 75% earliest data for training, 25% latest for testing
        split_idx = int(len(df_clean) * 0.75)
        train_indices = df_clean.index[:split_idx]
        test_indices = df_clean.index[split_idx:]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
    else:
        # Fallback to chronological split by index order if no date column
        split_idx = int(len(X) * 0.75)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Try Sobol analysis if we have SALib and enough data
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        
        # Lower threshold to 50 rows (minimum for meaningful Sobol analysis)
        if len(X) >= 50:  # Need sufficient data for Sobol
            # Define the problem for SALib
            problem = {
                'num_vars': len(feature_cols),
                'names': feature_cols,
                'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_cols]
            }
            
            # Generate samples using Saltelli's sampling scheme
            # Adjust N based on available data
            N = min(64, max(8, len(X) // 20))  # Adaptive base sample size
            param_values = saltelli.sample(problem, N, calc_second_order=False)
            
            # Evaluate the model for all generated samples
            Y = model.predict(param_values)
            
            # Ensure Y is the correct shape
            if Y.ndim > 1:
                Y = Y.flatten()
            
            # Compute Sobol sensitivity indices
            sobol_indices = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            first_order = sobol_indices['S1']
            
            # Sort by importance
            sorted_idx = np.argsort(first_order)[::-1]
            sorted_features_sobol = [feature_cols[i] for i in sorted_idx]
            sorted_sobol = first_order[sorted_idx]
            
            # Plot: Sobol First Order Sensitivity
            plot_sobol = {
                "data": [{
                    "type": "bar",
                    "x": sorted_features_sobol[:10],  # Top 10 for clarity
                    "y": sorted_sobol[:10].tolist(),
                    "marker": {"color": "green"}
                }],
                "layout": {
                    "title": {
                        "text": "Sobol First Order Sensitivity Indices (Top 10)",
                        "font": {"size": 16}
                    },
                    "xaxis": {
                        "title": "Input Variables",
                        "tickangle": -45,
                        "tickfont": {"size": 12}
                    },
                    "yaxis": {
                        "title": "First Order Sobol Index",
                        "titlefont": {"size": 14}
                    },
                    "height": 500
                }
            }
            plots.append(plot_sobol)
        else:
            pass  # Insufficient data for Sobol analysis
    except ImportError as e:
        pass  # SALib not installed
    except Exception as e:
        # Sobol analysis failed, skip it silently
        pass
    
    # Compute permutation feature importance
    # IMPORTANT: Calculate on training data to get meaningful positive importances
    perm_result = permutation_importance(model, X_train, y_train, n_repeats=30, random_state=42)
    perm_importances = perm_result.importances_mean
    
    # Sort by importance
    sorted_idx = np.argsort(perm_importances)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_importances = perm_importances[sorted_idx]
    
    # Plot: Permutation Feature Importance
    plot2 = {
        "data": [{
            "type": "bar",
            "x": sorted_features,
            "y": sorted_importances.tolist(),
            "marker": {"color": "orange"}
        }],
        "layout": {
            "title": {
                "text": "Permutation Feature Importance",
                "font": {"size": 16}
            },
            "xaxis": {
                "title": "Input Variables",
                "tickangle": -45,
                "tickfont": {"size": 12}
            },
            "yaxis": {
                "title": "Decrease in Model Score",
                "titlefont": {"size": 14}
            },
            "height": 500
        }
    }
    plots.append(plot2)
    
    return plots


def generate_time_series_comparison(data, site=None):
    """Generate DA vs PN time series comparison."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    if site:
        # Filter by site
        site_data = data[data['site'] == site].copy()
        title = f'DA vs Pseudo-nitzschia Time Series - {site}'
    else:
        # Aggregate all sites
        site_data = data.groupby('date').agg({
            'da': 'mean',
            'pn': 'mean' if 'pn' in data.columns else lambda x: np.nan
        }).reset_index()
        title = 'DA vs Pseudo-nitzschia Time Series - All Sites Average'
    
    # Sort by date
    site_data = site_data.sort_values('date')
    
    # Check if we have PN data
    has_pn = 'pn' in site_data.columns and not site_data['pn'].isna().all()
    
    # Normalize data for comparison
    scaler = MinMaxScaler()
    traces = []
    
    # Add DA trace
    if 'da' in site_data.columns:
        da_values = site_data['da'].values.reshape(-1, 1)
        da_normalized = scaler.fit_transform(np.nan_to_num(da_values)).flatten()
        
        # Cap DA at 80 for visualization (as in original)
        da_capped = np.minimum(site_data['da'].fillna(0), 80)
        da_capped_normalized = scaler.fit_transform(da_capped.values.reshape(-1, 1)).flatten()
        
        traces.append({
            "x": site_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            "y": da_capped_normalized.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "DA (normalized, capped at 80)",
            "line": {"color": "red", "width": 2}
        })
    
    # Add PN trace if available
    if has_pn:
        pn_values = site_data['pn'].values.reshape(-1, 1)
        pn_normalized = scaler.fit_transform(np.nan_to_num(pn_values)).flatten()
        traces.append({
            "x": site_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            "y": pn_normalized.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Pseudo-nitzschia (normalized)",
            "line": {"color": "blue", "width": 2}
        })
    else:
        # Use MODIS chlorophyll as proxy if no PN data
        if 'modis-chla' in site_data.columns:
            chla_values = site_data['modis-chla'].values.reshape(-1, 1)
            chla_normalized = scaler.fit_transform(np.nan_to_num(chla_values)).flatten()
            traces.append({
                "x": site_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                "y": chla_normalized.tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "MODIS-Chlorophyll (normalized)",
                "line": {"color": "green", "width": 2}
            })
    
    plot_data = {
        "data": traces,
        "layout": {
            "title": title,
            "xaxis": {"title": "Date"},
            "yaxis": {
                "title": "Normalized Value (0-1)",
                "range": [0, 1]
            },
            "height": 500,
            "hovermode": "x unified",
            "showlegend": True
        }
    }
    
    return plot_data  # Already in correct format for direct use


def generate_waterfall_plot(data):
    """Generate waterfall plot matching the original implementation."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by latitude and date
    data = data.sort_values(['lat', 'date'])
    
    # Get latitude to site mapping
    lat_to_site = data.groupby('lat')['site'].first().to_dict()
    unique_lats = sorted(data['lat'].unique(), reverse=True)
    
    traces = []
    
    # Configuration from original
    LATITUDE_BASELINE_MULTIPLIER = 3
    DA_SCALING_FACTOR = 0.01
    
    for i, lat in enumerate(unique_lats):
        site_name = lat_to_site.get(lat, f"Lat {lat:.2f}")
        site_data = data[data['lat'] == lat].sort_values('date')
        
        # Calculate y-baseline for this latitude (represents DA = 0)
        baseline_y = lat * LATITUDE_BASELINE_MULTIPLIER
        
        # Scale DA values and add to baseline
        y_values = baseline_y + DA_SCALING_FACTOR * site_data['da'].fillna(0)
        
        traces.append({
            "x": site_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            "y": [float(v) if pd.notna(v) else None for v in y_values],
            "type": "scatter",
            "mode": "lines",
            "name": site_name,
            "line": {"width": 1},
            "hovertemplate": f"{site_name}<br>Date: %{{x}}<br>DA: %{{customdata:.2f}} μg/g<extra></extra>",
            "customdata": [float(v) if pd.notna(v) else None for v in site_data['da']]
        })
        
        # Add reference bars for this site
        # These show reference DA levels of 20, 50, 100 μg/g
        bar_da_levels = [20, 50, 100]
        bar_spacing_days = 120  # Spacing between bars
        bar_target_date = pd.to_datetime('2012-01-01')
        
        for idx, da_level in enumerate(bar_da_levels):
            # Calculate x position for each bar (spread them out)
            # Shift every other row to the right to avoid overlap
            row_offset_days = 500 if i % 2 == 1 else 0  # Odd rows shifted 500 days right
            bar_offset_days = (idx - 1) * bar_spacing_days + row_offset_days  # -120, 0, +120 days plus row offset
            bar_date = bar_target_date + pd.Timedelta(days=bar_offset_days)
            
            # Calculate y positions
            y_bar_base = baseline_y  # Bottom of bar (DA=0)
            y_bar_top = baseline_y + DA_SCALING_FACTOR * da_level  # Top of bar
            
            # Draw vertical line for reference bar
            traces.append({
                "x": [bar_date.strftime('%Y-%m-%d'), bar_date.strftime('%Y-%m-%d')],
                "y": [y_bar_base, y_bar_top],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "rgba(128, 128, 128, 0.7)", "width": 2},
                "showlegend": False,
                "hovertemplate": f"Reference: {da_level} μg/g<extra></extra>"
            })
            
            # Add label at top of bar
            traces.append({
                "x": [bar_date.strftime('%Y-%m-%d')],
                "y": [y_bar_top],
                "type": "scatter",
                "mode": "text",
                "text": [f"{da_level}"],
                "textposition": "top right",
                "textfont": {"size": 10, "color": "gray"},
                "showlegend": False,
                "hoverinfo": "skip"
            })
    
    # Create y-axis labels for sites - handle NaN values
    y_tick_positions = [float(lat * LATITUDE_BASELINE_MULTIPLIER) if pd.notna(lat) else 0 for lat in unique_lats]
    y_tick_labels = [f"{lat:.2f}°N" for lat in unique_lats]  # Just show latitude, site names are in legend
    
    plot_data = {
        "data": traces,
        "layout": {
            "title": {
                "text": "Absolute DA Levels Variation by Site/Latitude Over Time<br><sub>Reference bars show DA=20, 50, 100 μg/g</sub>",
                "font": {"size": 18}
            },
            "xaxis": {"title": "Date"},
            "yaxis": {
                "title": {
                    "text": "Latitude (°N) - Baseline represents DA=0",
                    "standoff": 30  # Add more space between title and labels
                },
                "tickmode": "array",
                "tickvals": y_tick_positions,
                "ticktext": y_tick_labels,
                "ticksuffix": "    ",  # Add more spacing after tick labels
                "ticklen": 10,  # Longer tick marks
                "tickwidth": 1
            },
            "height": 700,
            "hovermode": "x unified",
            "showlegend": True,
            "margin": {"l": 120}  # Increase left margin more for y-axis labels
        }
    }
    
    return plot_data


def generate_spectral_analysis(data, site=None):
    """Generate comprehensive spectral analysis with XGBoost comparison."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    if site:
        # Filter by site
        site_data = data[data['site'] == site].copy()
        site_name = site
    else:
        # Aggregate by date for all sites
        site_data = data.groupby('date')['da'].mean().reset_index()
        site_name = "All Sites"
    
    # Sort by date and remove NaN values
    site_data = site_data.sort_values('date')
    
    if 'da' in site_data.columns:
        da_values = site_data['da'].dropna().values
    else:
        da_values = site_data.dropna().values
    
    if len(da_values) < 20:
        return []
    
    # Optionally disable XGBoost comparison for performance (set SPECTRAL_ENABLE_XGB=0)
    if os.getenv('SPECTRAL_ENABLE_XGB', '1') not in ('1', 'true', 'TRUE'):
        xgb_predictions = None
        actual_for_comparison = da_values
    else:
        # Run actual XGBoost retrospective evaluation for comparison
        try:
            from forecasting.core.forecast_engine import ForecastEngine
            import config

            engine = ForecastEngine()
            n_anchors = int(os.getenv('SPECTRAL_N_ANCHORS', getattr(config, 'N_RANDOM_ANCHORS', 200)))
            results_df = engine.run_retrospective_evaluation(
                task="regression",
                model_type="xgboost",
                n_anchors=n_anchors
            )

            if site and results_df is not None and not results_df.empty:
                results_df = results_df[results_df['site'] == site]

            if results_df is not None and not results_df.empty:
                results_df = results_df.sort_values('date')
                xgb_predictions = results_df['Predicted_da'].dropna().values
                actual_for_comparison = results_df['da'].dropna().values
            else:
                xgb_predictions = None
                actual_for_comparison = da_values
        except Exception as e:
            print(f"XGBoost retrospective evaluation failed: {e}")
            xgb_predictions = None
            actual_for_comparison = da_values
    
    plots = []
    
    # 1. Power Spectral Density - Actual vs XGBoost
    freqs, psd = signal.welch(da_values, fs=1.0, nperseg=min(256, len(da_values)//4))
    
    # Find dominant frequencies
    dominant_idx = np.argsort(psd[1:])[-3:][::-1]  # Top 3 frequencies
    periods = 1 / freqs[1:]  # Convert to periods
    dominant_periods = periods[dominant_idx]
    
    traces = [{
        "x": freqs[1:].tolist(),
        "y": psd[1:].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": "Actual DA",
        "line": {"color": "blue", "width": 2}
    }]
    
    # Add XGBoost PSD if available
    if xgb_predictions is not None and len(xgb_predictions) >= 20:
        freqs_xgb, psd_xgb = signal.welch(xgb_predictions, fs=1.0, nperseg=min(256, len(xgb_predictions)//4))
        traces.append({
            "x": freqs_xgb[1:].tolist(),
            "y": psd_xgb[1:].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "XGBoost Predictions",
            "line": {"color": "red", "width": 2, "dash": "dash"}
        })
    
    plot1 = {
        "data": traces,
        "layout": {
            "title": f"Power Spectral Density - {site_name}",
            "xaxis": {
                "title": "Frequency (1/weeks)",
                "type": "log"
            },
            "yaxis": {
                "title": "Power",
                "type": "log"
            },
            "height": 400,
            "showlegend": True,
            "annotations": [{
                "x": 0.5,
                "y": 1.15,
                "xref": "paper",
                "yref": "paper",
                "text": f"Dominant periods: {', '.join([f'{p:.1f} weeks' for p in dominant_periods[:3]])}",
                "showarrow": False,
                "font": {"size": 12}
            }]
        }
    }
    plots.append(plot1)
    
    # 2. Periodogram - Actual vs XGBoost
    freqs_p, pgram = signal.periodogram(da_values, fs=1.0)
    
    traces2 = [{
        "x": (1/freqs_p[1:]).tolist(),
        "y": pgram[1:].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": "Actual DA",
        "line": {"color": "blue", "width": 2}
    }]
    
    # Add XGBoost periodogram if available
    if xgb_predictions is not None and len(xgb_predictions) >= 20:
        freqs_p_xgb, pgram_xgb = signal.periodogram(xgb_predictions, fs=1.0)
        traces2.append({
            "x": (1/freqs_p_xgb[1:]).tolist(),
            "y": pgram_xgb[1:].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "XGBoost Predictions",
            "line": {"color": "red", "width": 2, "dash": "dash"}
        })
    
    plot2 = {
        "data": traces2,
        "layout": {
            "title": f"Periodogram - {site_name}",
            "xaxis": {
                "title": "Period (weeks)",
                "type": "log"
            },
            "yaxis": {
                "title": "Power",
                "type": "log"
            },
            "height": 400,
            "showlegend": True
        }
    }
    plots.append(plot2)
    
    # 3. Spectrogram (time-varying spectral properties)
    if len(da_values) >= 32:
        window_size = min(32, len(da_values)//4)
        f, t, Sxx = signal.spectrogram(da_values, fs=1.0, nperseg=window_size, noverlap=window_size//2)
        
        plot3 = {
            "data": [{
                "type": "heatmap",
                "x": t.tolist(),
                "y": f.tolist(),
                "z": np.log10(Sxx + 1e-10).tolist(),  # Log scale for better visualization
                "colorscale": "Viridis",
                "colorbar": {"title": "log(Power)"}
            }],
            "layout": {
                "title": f"Spectrogram - {site_name}",
                "xaxis": {"title": "Time (weeks)"},
                "yaxis": {"title": "Frequency (1/weeks)"},
                "height": 400
            }
        }
        plots.append(plot3)
    
    # 4. Coherence plot if XGBoost predictions available
    if xgb_predictions is not None and len(xgb_predictions) >= 20 and len(actual_for_comparison) >= 20:
        # Ensure equal length
        min_len = min(len(actual_for_comparison), len(xgb_predictions))
        actual_trimmed = actual_for_comparison[:min_len]
        xgb_trimmed = xgb_predictions[:min_len]
        
        # Compute coherence
        freqs_coh, coherence = signal.coherence(actual_trimmed, xgb_trimmed, fs=1.0, nperseg=min(256, min_len//4))
        
        plot_coherence = {
            "data": [{
                "x": freqs_coh[1:].tolist(),
                "y": coherence[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Coherence",
                "line": {"color": "green", "width": 2}
            }],
            "layout": {
                "title": f"Coherence: Actual vs XGBoost - {site_name}",
                "xaxis": {
                    "title": "Frequency (1/weeks)",
                    "type": "log"
                },
                "yaxis": {
                    "title": "Coherence (0-1)",
                    "range": [0, 1]
                },
                "height": 400,
                "annotations": [{
                    "x": 0.5,
                    "y": 1.1,
                    "xref": "paper",
                    "yref": "paper",
                    "text": f"Mean coherence: {np.mean(coherence):.3f}",
                    "showarrow": False,
                    "font": {"size": 12}
                }]
            }
        }
        plots.append(plot_coherence)
    
    # 5. Summary statistics
    total_power = np.sum(psd)
    mean_da = np.mean(da_values)
    std_da = np.std(da_values)
    
    summary_lines = [
        f"Summary Statistics for {site_name}:",
        f"- Total spectral power: {total_power:.2f}",
        f"- Mean DA: {mean_da:.2f} μg/g",
        f"- Std DA: {std_da:.2f} μg/g",
        f"- Data points: {len(da_values)}"
    ]
    
    if xgb_predictions is not None:
        summary_lines.extend([
            f"- XGBoost predictions: {len(xgb_predictions)} points",
            f"- XGBoost mean: {np.mean(xgb_predictions):.2f} μg/g",
            f"- XGBoost std: {np.std(xgb_predictions):.2f} μg/g"
        ])
    
    summary_text = "\n".join(summary_lines)
    
    plot4 = {
        "data": [],
        "layout": {
            "title": f"Spectral Analysis Summary - {site_name}",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "height": 250,
            "annotations": [{
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "text": summary_text,
                "showarrow": False,
                "font": {"size": 14, "family": "monospace"},
                "align": "left"
            }]
        }
    }
    plots.append(plot4)
    
    return plots


def generate_advanced_spectral_analysis(data, site=None):
    """Generate comprehensive spectral analysis with advanced methods."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    if site:
        site_data = data[data['site'] == site].copy()
        site_name = site
    else:
        site_data = data.groupby('date')['da'].mean().reset_index()
        site_name = "All Sites"
    
    # Sort by date and remove NaN values
    site_data = site_data.sort_values('date')
    
    if 'da' in site_data.columns:
        da_values = site_data['da'].dropna().values
        dates = site_data[site_data['da'].notna()]['date'].values
    else:
        da_values = site_data.dropna().values
        dates = site_data.index
    
    if len(da_values) < 50:
        return []
    
    plots = []
    
    # 1. Multitaper Spectral Estimation
    # Using DPSS (Discrete Prolate Spheroidal Sequences) windows
    NW = 4  # Time-bandwidth product
    Kmax = 7  # Number of tapers
    
    # Create DPSS windows
    n_samples = len(da_values)
    dpss_tapers, dpss_eigen = windows.dpss(n_samples, NW, Kmax, return_ratios=True)
    
    # Compute multitaper spectrum
    mt_spectra = []
    for taper in dpss_tapers:
        windowed_signal = da_values * taper
        spectrum = np.abs(np.fft.rfft(windowed_signal))**2
        mt_spectra.append(spectrum)
    
    # Average the spectra
    mt_spectrum = np.mean(mt_spectra, axis=0)
    freqs_mt = np.fft.rfftfreq(n_samples, d=1.0)  # Weekly sampling
    
    # Jackknife confidence intervals
    jackknife_estimates = []
    for i in range(len(mt_spectra)):
        leave_one_out = [s for j, s in enumerate(mt_spectra) if j != i]
        jackknife_estimates.append(np.mean(leave_one_out, axis=0))
    
    mt_std = np.std(jackknife_estimates, axis=0)
    mt_lower = mt_spectrum - 2 * mt_std
    mt_upper = mt_spectrum + 2 * mt_std
    
    plot_multitaper = {
        "data": [
            {
                "x": freqs_mt[1:].tolist(),
                "y": mt_spectrum[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Multitaper Estimate",
                "line": {"color": "blue", "width": 2}
            },
            {
                "x": freqs_mt[1:].tolist(),
                "y": mt_lower[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "95% CI Lower",
                "line": {"color": "lightblue", "width": 1, "dash": "dash"},
                "showlegend": False
            },
            {
                "x": freqs_mt[1:].tolist(),
                "y": mt_upper[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "95% CI Upper",
                "line": {"color": "lightblue", "width": 1, "dash": "dash"},
                "fill": "tonexty",
                "fillcolor": "rgba(173, 216, 230, 0.2)",
                "showlegend": False
            }
        ],
        "layout": {
            "title": f"Multitaper Spectral Estimate - {site_name}",
            "xaxis": {"title": "Frequency (1/weeks)", "type": "log"},
            "yaxis": {"title": "Power", "type": "log"},
            "height": 400,
            "showlegend": True
        }
    }
    plots.append(plot_multitaper)
    
    # 2. Wavelet Analysis (Continuous Wavelet Transform)
    scales = np.arange(1, min(128, len(da_values)//2))
    wavelet = 'morl'  # Morlet wavelet
    
    coefficients, frequencies = pywt.cwt(da_values, scales, wavelet, sampling_period=1.0)
    power = np.abs(coefficients)**2
    
    # Convert frequencies to periods for better interpretation
    periods = 1 / frequencies
    
    plot_wavelet = {
        "data": [{
            "type": "heatmap",
            "x": list(range(len(da_values))),
            "y": periods.tolist(),
            "z": np.log10(power + 1e-10).tolist(),
            "colorscale": "Viridis",
            "colorbar": {"title": "log(Power)"}
        }],
        "layout": {
            "title": f"Wavelet Transform (Morlet) - {site_name}",
            "xaxis": {"title": "Time Index (weeks)"},
            "yaxis": {"title": "Period (weeks)", "type": "log"},
            "height": 400
        }
    }
    plots.append(plot_wavelet)
    
    # 3. Statistical Significance Testing with Surrogate Data
    n_surrogates = 100
    surrogate_spectra = []
    
    for _ in range(n_surrogates):
        # Create phase-randomized surrogate
        fft_original = np.fft.rfft(da_values)
        random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, len(fft_original)))
        # Keep DC component real
        random_phases[0] = 1
        surrogate_fft = fft_original * random_phases
        surrogate_signal = np.fft.irfft(surrogate_fft, n=len(da_values))
        
        # Compute spectrum of surrogate
        freqs_surr, psd_surr = signal.welch(surrogate_signal, fs=1.0, 
                                           nperseg=min(256, len(da_values)//4))
        surrogate_spectra.append(psd_surr)
    
    # Original spectrum
    freqs_orig, psd_orig = signal.welch(da_values, fs=1.0, 
                                       nperseg=min(256, len(da_values)//4))
    
    # Calculate significance thresholds
    surrogate_array = np.array(surrogate_spectra)
    significance_95 = np.percentile(surrogate_array, 95, axis=0)
    significance_99 = np.percentile(surrogate_array, 99, axis=0)
    
    # Find significant peaks
    significant_peaks_95 = psd_orig > significance_95
    significant_peaks_99 = psd_orig > significance_99
    
    plot_significance = {
        "data": [
            {
                "x": freqs_orig[1:].tolist(),
                "y": psd_orig[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Observed Spectrum",
                "line": {"color": "blue", "width": 2}
            },
            {
                "x": freqs_orig[1:].tolist(),
                "y": significance_95[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "95% Significance",
                "line": {"color": "orange", "width": 1, "dash": "dash"}
            },
            {
                "x": freqs_orig[1:].tolist(),
                "y": significance_99[1:].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "99% Significance",
                "line": {"color": "red", "width": 1, "dash": "dot"}
            }
        ],
        "layout": {
            "title": f"Spectral Significance Testing - {site_name}",
            "xaxis": {"title": "Frequency (1/weeks)", "type": "log"},
            "yaxis": {"title": "Power", "type": "log"},
            "height": 400,
            "showlegend": True,
            "annotations": [{
                "x": 0.5,
                "y": 1.1,
                "xref": "paper",
                "yref": "paper",
                "text": f"Peaks above lines are statistically significant (phase randomization test)",
                "showarrow": False,
                "font": {"size": 11}
            }]
        }
    }
    plots.append(plot_significance)
    
    # 4. Seasonal Decomposition (STL)
    try:
        # Ensure we have regular weekly data
        weekly_data = pd.Series(da_values, index=pd.date_range(start='2002-01-01', 
                                                               periods=len(da_values), 
                                                               freq='W'))
        
        # STL decomposition with annual seasonality (52 weeks)
        stl = STL(weekly_data, seasonal=53, trend=105)  # Flexible trend
        result = stl.fit()
        
        # Create subplots for decomposition
        decomp_traces = []
        
        # Original series
        decomp_traces.append({
            "x": list(range(len(da_values))),
            "y": da_values.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Original",
            "yaxis": "y",
            "line": {"color": "black", "width": 1}
        })
        
        # Trend
        decomp_traces.append({
            "x": list(range(len(result.trend))),
            "y": result.trend.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Trend",
            "yaxis": "y2",
            "line": {"color": "blue", "width": 2}
        })
        
        # Seasonal
        decomp_traces.append({
            "x": list(range(len(result.seasonal))),
            "y": result.seasonal.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Seasonal",
            "yaxis": "y3",
            "line": {"color": "green", "width": 1}
        })
        
        # Residual
        decomp_traces.append({
            "x": list(range(len(result.resid))),
            "y": result.resid.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Residual",
            "yaxis": "y4",
            "line": {"color": "red", "width": 1}
        })
        
        plot_stl = {
            "data": decomp_traces,
            "layout": {
                "title": f"STL Seasonal Decomposition - {site_name}",
                "xaxis": {"title": "Time (weeks)", "domain": [0, 1]},
                "yaxis": {"title": "Original", "domain": [0.75, 1]},
                "yaxis2": {"title": "Trend", "domain": [0.5, 0.73]},
                "yaxis3": {"title": "Seasonal", "domain": [0.25, 0.48]},
                "yaxis4": {"title": "Residual", "domain": [0, 0.23]},
                "height": 600,
                "showlegend": False
            }
        }
        plots.append(plot_stl)
        
    except Exception as e:
        print(f"STL decomposition failed: {e}")
    
    # 5. Autocorrelation and Partial Autocorrelation
    from statsmodels.tsa.stattools import acf, pacf
    
    max_lag = min(100, len(da_values)//3)
    acf_values = acf(da_values, nlags=max_lag)
    pacf_values = pacf(da_values, nlags=max_lag)
    
    # Confidence intervals
    ci = 1.96 / np.sqrt(len(da_values))
    
    plot_acf = {
        "data": [
            {
                "x": list(range(len(acf_values))),
                "y": acf_values.tolist(),
                "type": "bar",
                "name": "ACF",
                "marker": {"color": "blue"}
            },
            {
                "x": [0, max_lag],
                "y": [ci, ci],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "red", "dash": "dash"},
                "showlegend": False
            },
            {
                "x": [0, max_lag],
                "y": [-ci, -ci],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "red", "dash": "dash"},
                "showlegend": False
            }
        ],
        "layout": {
            "title": f"Autocorrelation Function - {site_name}",
            "xaxis": {"title": "Lag (weeks)"},
            "yaxis": {"title": "Correlation"},
            "height": 350,
            "showlegend": False
        }
    }
    plots.append(plot_acf)
    
    return plots


def generate_multisite_spectral_comparison(data):
    """Generate spectral comparison across all sites."""
    
    # Get unique sites
    unique_sites = data['site'].unique()
    
    plots = []
    
    # 1. Power Spectral Density Comparison
    psd_traces = []
    dominant_frequencies = {}
    
    for site in unique_sites:
        site_data = data[data['site'] == site].copy()
        site_data = site_data.sort_values('date')
        da_values = site_data['da'].dropna().values
        
        if len(da_values) < 50:
            continue
        
        # Compute PSD
        freqs, psd = signal.welch(da_values, fs=1.0, nperseg=min(256, len(da_values)//4))
        
        # Find dominant frequency
        dominant_idx = np.argmax(psd[1:10]) + 1  # Exclude DC, look at low frequencies
        dominant_freq = freqs[dominant_idx]
        dominant_period = 1 / dominant_freq
        dominant_frequencies[site] = dominant_period
        
        psd_traces.append({
            "x": freqs[1:].tolist(),
            "y": psd[1:].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": site,
            "line": {"width": 2}
        })
    
    plot_psd_comparison = {
        "data": psd_traces,
        "layout": {
            "title": "Power Spectral Density - All Sites Comparison",
            "xaxis": {"title": "Frequency (1/weeks)", "type": "log"},
            "yaxis": {"title": "Power", "type": "log"},
            "height": 500,
            "showlegend": True,
            "hovermode": "x unified"
        }
    }
    plots.append(plot_psd_comparison)
    
    # 2. Dominant Periods Bar Chart
    if dominant_frequencies:
        sites = list(dominant_frequencies.keys())
        periods = list(dominant_frequencies.values())
        
        plot_dominant = {
            "data": [{
                "x": sites,
                "y": periods,
                "type": "bar",
                "marker": {"color": periods, "colorscale": "Viridis"},
                "text": [f"{p:.1f} weeks" for p in periods],
                "textposition": "outside"
            }],
            "layout": {
                "title": "Dominant Periods by Site",
                "xaxis": {"title": "Site"},
                "yaxis": {"title": "Dominant Period (weeks)"},
                "height": 400
            }
        }
        plots.append(plot_dominant)
    
    # 3. Coherence Matrix Between Sites
    coherence_matrix = np.zeros((len(unique_sites), len(unique_sites)))
    
    for i, site1 in enumerate(unique_sites):
        for j, site2 in enumerate(unique_sites):
            if i <= j:
                site1_data = data[data['site'] == site1].sort_values('date')
                site2_data = data[data['site'] == site2].sort_values('date')
                
                # Merge on date to align time series
                merged = pd.merge(site1_data[['date', 'da']], 
                                site2_data[['date', 'da']], 
                                on='date', 
                                suffixes=('_1', '_2'))
                
                if len(merged) >= 50:
                    da1 = merged['da_1'].dropna().values
                    da2 = merged['da_2'].dropna().values
                    
                    if len(da1) >= 50 and len(da2) >= 50:
                        min_len = min(len(da1), len(da2))
                        da1 = da1[:min_len]
                        da2 = da2[:min_len]
                        
                        # Compute mean coherence
                        freqs_coh, coh = signal.coherence(da1, da2, fs=1.0, 
                                                         nperseg=min(64, min_len//4))
                        mean_coherence = np.mean(coh[1:])  # Exclude DC
                        coherence_matrix[i, j] = mean_coherence
                        coherence_matrix[j, i] = mean_coherence
    
    plot_coherence_matrix = {
        "data": [{
            "type": "heatmap",
            "x": unique_sites.tolist(),
            "y": unique_sites.tolist(),
            "z": coherence_matrix.tolist(),
            "colorscale": "RdBu",
            "colorbar": {"title": "Mean Coherence"},
            "zmin": 0,
            "zmax": 1
        }],
        "layout": {
            "title": "Spectral Coherence Between Sites",
            "xaxis": {"title": "Site", "tickangle": 45},
            "yaxis": {"title": "Site"},
            "height": 600
        }
    }
    plots.append(plot_coherence_matrix)
    
    # 4. Seasonal Strength Comparison
    seasonal_strengths = {}
    
    for site in unique_sites:
        site_data = data[data['site'] == site].sort_values('date')
        da_values = site_data['da'].dropna().values
        
        if len(da_values) >= 104:  # Need at least 2 years
            try:
                weekly_data = pd.Series(da_values[:104*2], 
                                      index=pd.date_range(start='2002-01-01', 
                                                         periods=min(104*2, len(da_values)), 
                                                         freq='W'))
                stl = STL(weekly_data, seasonal=53)
                result = stl.fit()
                
                # Calculate seasonal strength
                seasonal_var = np.var(result.seasonal)
                total_var = np.var(da_values[:len(result.seasonal)])
                seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
                seasonal_strengths[site] = seasonal_strength
            except:
                pass
    
    if seasonal_strengths:
        sites_seasonal = list(seasonal_strengths.keys())
        strengths = list(seasonal_strengths.values())
        
        plot_seasonal = {
            "data": [{
                "x": sites_seasonal,
                "y": strengths,
                "type": "bar",
                "marker": {"color": strengths, "colorscale": "RdYlGn"},
                "text": [f"{s:.2%}" for s in strengths],
                "textposition": "outside"
            }],
            "layout": {
                "title": "Seasonal Component Strength by Site",
                "xaxis": {"title": "Site"},
                "yaxis": {"title": "Seasonal Strength (% of total variance)"},
                "height": 400
            }
        }
        plots.append(plot_seasonal)
    
    return plots