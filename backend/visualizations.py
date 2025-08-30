"""
Backend visualization module for DATect web application.
Implements visualization logic from original analysis scripts.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from scipy import signal
from scipy.stats import pearsonr
import warnings
import os
import sys
import logging
import plotly.graph_objs as go
import plotly.io as pio

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_gradient_uncertainty_plot(gradient_quantiles, xgboost_prediction, actual_da=None):
    """Create gradient visualization with quantile uncertainty bands"""
    q05 = gradient_quantiles['q05']
    q50 = gradient_quantiles['q50'] 
    q95 = gradient_quantiles['q95']
    
    fig = go.Figure()
    
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


def generate_sensitivity_analysis(data, site=None):
    """Generate sensitivity analysis plots - complex for all sites, simple for single sites."""
    
    if site:
        # SINGLE SITE: Use simplified, fast analysis
        return _generate_single_site_sensitivity(data, site)
    else:
        # ALL SITES: Use complex analysis with Sobol, temporal splits, etc.
        return _generate_all_sites_sensitivity(data)

def _generate_single_site_sensitivity(data, site):
    """Fast, simplified sensitivity analysis for single site."""
    # Filter by site
    df = data[data['site'] == site].copy()
    title_suffix = f" - {site}"
    
    # Check if we have data
    if df.empty or len(df) < 10:
        return [{
            "data": [{
                "type": "bar",
                "x": ["Insufficient Data"],
                "y": [0],
                "marker": {"color": "gray"}
            }],
            "layout": {
                "title": f"Sensitivity Analysis: Insufficient Data{title_suffix}",
                "height": 500
            }
        }]
    
    # Remove samples with NaN target values
    df_clean = df.dropna(subset=['da']).copy()
    
    # Identify feature columns
    exclude_cols = ['da', 'date', 'site', 'lon', 'lat']
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not feature_cols:
        return [{
            "data": [{
                "type": "bar",
                "x": ["No Features"],
                "y": [0],
                "marker": {"color": "gray"}
            }],
            "layout": {
                "title": f"Sensitivity Analysis: No Features Available{title_suffix}",
                "height": 500
            }
        }]
    
    # Use median imputation for features
    imputer = SimpleImputer(strategy="median")
    df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    
    # Compute absolute Pearson correlation with DA
    correlations = df_clean[feature_cols + ['da']].corr()['da'].drop('da').abs().sort_values(ascending=False)
    correlations = correlations.dropna()
    
    if correlations.empty:
        return [{
            "data": [{
                "type": "bar",
                "x": ["No Valid Correlations"],
                "y": [0],
                "marker": {"color": "gray"}
            }],
            "layout": {
                "title": f"Sensitivity Analysis: No Valid Correlations{title_suffix}",
                "height": 500
            }
        }]
    
    plots = []
    
    # Plot 1: Correlation Sensitivity Analysis
    plot1 = {
        "data": [{
            "type": "bar",
            "x": correlations.index.tolist(),
            "y": correlations.values.tolist(),
            "marker": {"color": "steelblue"}
        }],
        "layout": {
            "title": f"Correlation Sensitivity Analysis{title_suffix}",
            "xaxis": {"title": "Input Variables", "tickangle": -45},
            "yaxis": {"title": "Absolute Pearson Correlation"},
            "height": 500
        }
    }
    plots.append(plot1)
    
    # Try to create model-based plots if we have enough data
    if len(df_clean) >= 20:
        try:
            X = df_clean[feature_cols]
            y = df_clean['da']
            
            # Simple train/test split
            split_idx = int(len(df_clean) * 0.75)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            
            if len(X_train) >= 10:
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Plot 2: Permutation Feature Importance (fast - only 5 repeats)
                perm_result = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=42)
                perm_importances = perm_result.importances_mean
                
                sorted_idx = np.argsort(perm_importances)[::-1]
                sorted_features = [feature_cols[i] for i in sorted_idx]
                sorted_importances = perm_importances[sorted_idx]
                
                plot2 = {
                    "data": [{
                        "type": "bar",
                        "x": sorted_features,
                        "y": sorted_importances.tolist(),
                        "marker": {"color": "orange"}
                    }],
                    "layout": {
                        "title": f"Permutation Feature Importance{title_suffix}",
                        "xaxis": {"title": "Input Variables", "tickangle": -45},
                        "yaxis": {"title": "Decrease in Model Score"},
                        "height": 500
                    }
                }
                plots.append(plot2)
                
                # Plot 3: Feature coefficients from linear model
                coefficients = np.abs(model.coef_)
                sorted_coeff_idx = np.argsort(coefficients)[::-1]
                sorted_coeff_features = [feature_cols[i] for i in sorted_coeff_idx]
                sorted_coefficients = coefficients[sorted_coeff_idx]
                
                plot3 = {
                    "data": [{
                        "type": "bar",
                        "x": sorted_coeff_features,
                        "y": sorted_coefficients.tolist(),
                        "marker": {"color": "green"}
                    }],
                    "layout": {
                        "title": f"Linear Model Coefficients (Absolute){title_suffix}",
                        "xaxis": {"title": "Input Variables", "tickangle": -45},
                        "yaxis": {"title": "Absolute Coefficient Value"},
                        "height": 500
                    }
                }
                plots.append(plot3)
        except Exception as e:
            logger.warning(f"Error in single site analysis{title_suffix}: {str(e)}")
    
    return plots

def _generate_all_sites_sensitivity(data):
    """Complex sensitivity analysis for all sites with Sobol, temporal splits, etc."""
    df_processed = data.copy()
    title_suffix = " - All Sites"
    
    # Maintain temporal ordering if date column exists
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
    
    # Remove samples with NaN target values
    df_clean = df_processed.dropna(subset=['da']).copy()
    
    # Check if we have sufficient data
    if len(df_clean) < 50:
        return [{
            "data": [{
                "type": "bar",
                "x": ["Insufficient Data"],
                "y": [0],
                "marker": {"color": "gray"}
            }],
            "layout": {
                "title": f"Sensitivity Analysis: Insufficient Data{title_suffix} ({len(df_clean)} samples)",
                "height": 500
            }
        }]
    
    # Identify feature columns
    exclude_cols = ['da', 'date', 'site', 'lon', 'lat']
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not feature_cols:
        return [{
            "data": [{
                "type": "bar",
                "x": ["No Features"],
                "y": [0],
                "marker": {"color": "gray"}
            }],
            "layout": {
                "title": f"Sensitivity Analysis: No Features Available{title_suffix}",
                "height": 500
            }
        }]
    
    # Use median imputation for features
    imputer = SimpleImputer(strategy="median")
    df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    
    # Compute absolute Pearson correlation with DA
    correlations = df_clean[feature_cols + ['da']].corr()['da'].drop('da').abs().sort_values(ascending=False)
    correlations = correlations.dropna()
    
    # Plot 1: Correlation Sensitivity Analysis
    plot1 = {
        "data": [{
            "type": "bar",
            "x": correlations.index.tolist(),
            "y": correlations.values.tolist(),
            "marker": {"color": "steelblue"}
        }],
        "layout": {
            "title": f"Correlation Sensitivity Analysis: Impact on DA Levels{title_suffix}",
            "xaxis": {"title": "Input Variables", "tickangle": -45, "tickfont": {"size": 12}},
            "yaxis": {"title": "Absolute Pearson Correlation", "titlefont": {"size": 14}},
            "height": 500
        }
    }
    
    plots = [plot1]
    
    # Prepare data for model-based methods using complex temporal splits
    X = df_clean[feature_cols]
    y = df_clean['da']
    
    # Complex temporal split: 75% earliest data for training, 25% latest for testing
    if 'date' in df_clean.columns:
        split_idx = int(len(df_clean) * 0.75)
        train_indices = df_clean.index[:split_idx]
        test_indices = df_clean.index[split_idx:]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
    else:
        # Fallback to chronological split by index order
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
        
        if len(X) >= 100:  # Higher threshold for Sobol analysis
            # Define the problem for SALib
            problem = {
                'num_vars': len(feature_cols),
                'names': feature_cols,
                'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_cols]
            }
            
            # Generate samples using Saltelli's sampling scheme
            N = min(128, max(16, len(X) // 15))  # More samples for better analysis
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
                    "x": sorted_features_sobol[:15],  # Top 15 for clarity
                    "y": sorted_sobol[:15].tolist(),
                    "marker": {"color": "green"}
                }],
                "layout": {
                    "title": f"Sobol First Order Sensitivity Indices (Top 15){title_suffix}",
                    "xaxis": {"title": "Input Variables", "tickangle": -45, "tickfont": {"size": 12}},
                    "yaxis": {"title": "First Order Sobol Index", "titlefont": {"size": 14}},
                    "height": 500
                }
            }
            plots.append(plot_sobol)
    except ImportError:
        pass  # SALib not installed
    except Exception as e:
        pass  # Sobol analysis failed, skip it
    
    # Compute permutation feature importance with lots of iterations
    try:
        perm_result = permutation_importance(model, X_train, y_train, n_repeats=50, random_state=42)
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
                "title": f"Permutation Feature Importance{title_suffix}",
                "xaxis": {"title": "Input Variables", "tickangle": -45, "tickfont": {"size": 12}},
                "yaxis": {"title": "Decrease in Model Score", "titlefont": {"size": 14}},
                "height": 500
            }
        }
        plots.append(plot2)
    except Exception as e:
        logger.warning(f"Error computing permutation importance{title_suffix}: {str(e)}")
    
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
    
    # Always enable XGBoost comparison in spectral analysis
    if True:  # Always enabled
        # Use cached XGBoost retrospective results instead of recomputing
        try:
            from pathlib import Path
            
            # Try to load cached retrospective results
            cache_dir = Path("cache/retrospective")
            cache_file = cache_dir / "regression_xgboost.parquet"
            
            if cache_file.exists():
                # Load cached results
                results_df = pd.read_parquet(cache_file)
                
                if site and not results_df.empty:
                    results_df = results_df[results_df['site'] == site]
                
                if not results_df.empty:
                    results_df = results_df.sort_values('date')
                    xgb_predictions = results_df['Predicted_da'].dropna().values
                    actual_for_comparison = results_df['da'].dropna().values
                else:
                    xgb_predictions = None
                    actual_for_comparison = da_values
            else:
                # Fallback: compute if cache doesn't exist
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
            logger.error(f"Loading XGBoost results failed: {e}")
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