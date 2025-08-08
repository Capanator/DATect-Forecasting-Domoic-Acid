"""
Backend visualization module for DATect web application.
Implements all visualization logic from the original Python analysis scripts.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy import signal
from scipy.stats import pearsonr
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


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
                "colorscale": "RdBu",
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
            "colorscale": "RdBu",  # Standard RdBu: red for negative, blue for positive
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
    """Generate sensitivity analysis plots matching the original implementation."""
    
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
    
    # Prepare data for model-based methods
    X = df_clean[feature_cols]
    y = df_clean['da']
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Compute permutation feature importance
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    perm_importances = perm_result.importances_mean
    
    # Sort by importance
    sorted_idx = np.argsort(perm_importances)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_importances = perm_importances[sorted_idx]
    
    # Plot 2: Permutation Feature Importance
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
    
    return [plot1, plot2]


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
        
        # Add reference bars (as in original)
        bar_da_levels = [20, 50, 100]
        bar_date = pd.to_datetime('2012-01-01')
        
        if bar_date in site_data['date'].values:
            for da_level in bar_da_levels:
                y_bar_top = baseline_y + DA_SCALING_FACTOR * da_level
                traces.append({
                    "x": [bar_date.strftime('%Y-%m-%d')],
                    "y": [y_bar_top],
                    "type": "scatter",
                    "mode": "markers",
                    "marker": {"size": 8, "color": "red"},
                    "showlegend": False,
                    "hovertemplate": f"Reference: {da_level} μg/g<extra></extra>"
                })
    
    # Create y-axis labels for sites - handle NaN values
    y_tick_positions = [float(lat * LATITUDE_BASELINE_MULTIPLIER) if pd.notna(lat) else 0 for lat in unique_lats]
    y_tick_labels = [f"{lat_to_site.get(lat, '')} ({lat:.2f}°N)" for lat in unique_lats]
    
    plot_data = {
        "data": traces,
        "layout": {
            "title": "Absolute DA Levels Variation by Site/Latitude Over Time",
            "xaxis": {"title": "Date"},
            "yaxis": {
                "title": "Latitude (°N) - Baseline represents DA=0",
                "tickmode": "array",
                "tickvals": y_tick_positions,
                "ticktext": y_tick_labels
            },
            "height": 700,
            "hovermode": "x unified",
            "showlegend": True
        }
    }
    
    return plot_data


def generate_spectral_analysis(data, site=None):
    """Generate comprehensive spectral analysis matching the original implementation."""
    
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
    
    plots = []
    
    # 1. Power Spectral Density
    freqs, psd = signal.welch(da_values, fs=1.0, nperseg=min(256, len(da_values)//4))
    
    # Find dominant frequencies
    dominant_idx = np.argsort(psd[1:])[-3:][::-1]  # Top 3 frequencies
    periods = 1 / freqs[1:]  # Convert to periods
    dominant_periods = periods[dominant_idx]
    
    plot1 = {
        "data": [{
            "x": freqs[1:].tolist(),
            "y": psd[1:].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Power Spectral Density",
            "line": {"color": "blue", "width": 2}
        }],
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
    
    # 2. Periodogram
    freqs_p, pgram = signal.periodogram(da_values, fs=1.0)
    
    plot2 = {
        "data": [{
            "x": (1/freqs_p[1:]).tolist(),
            "y": pgram[1:].tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Periodogram",
            "line": {"color": "red", "width": 2}
        }],
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
            "height": 400
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
    
    # 4. Summary statistics
    total_power = np.sum(psd)
    mean_da = np.mean(da_values)
    std_da = np.std(da_values)
    
    summary_text = f"""
    Summary Statistics for {site_name}:
    - Total spectral power: {total_power:.2f}
    - Mean DA: {mean_da:.2f} μg/g
    - Std DA: {std_da:.2f} μg/g
    - Data points: {len(da_values)}
    """
    
    plot4 = {
        "data": [],
        "layout": {
            "title": f"Spectral Analysis Summary - {site_name}",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "height": 200,
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