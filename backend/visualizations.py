"""
Backend visualization module for DATect web application.
Provides visualization functions for DATect web application.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from scipy import signal
import warnings
import os
import logging
import plotly.graph_objs as go
import plotly.io as pio

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


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
        max_distance = 1
    
    base_color = (70, 130, 180)
    
    for i in range(n_segments):
        x0 = q05 + (i / n_segments) * range_width
        x1 = q05 + ((i + 1) / n_segments) * range_width
        midpoint = (x0 + x1) / 2
        
        opacity = 1 - (abs(midpoint - q50) / max_distance) ** 0.5 if max_distance > 1e-6 else 0.8
        opacity = max(0.1, min(0.9, opacity))
        
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0.35, y1=0.65,
            line=dict(width=0),
            fillcolor=f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {opacity})',
            layer='below'
        )
    
    fig.add_trace(go.Scatter(
        x=[q50, q50], y=[0.35, 0.65],
        mode='lines',
        line=dict(color='rgb(30, 60, 90)', width=3),
        name='GB Median (Q50)',
        hovertemplate='GB Median: %{x:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[q05, q95], y=[0.5, 0.5],
        mode='markers',
        marker=dict(size=12, color='rgba(70, 130, 180, 0.4)', symbol='line-ns-open'),
        name='GB Range (Q05-Q95)',
        hovertemplate='GB Range: %{x:.2f}<extra></extra>'
    ))
    
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
    
    if actual_da is not None:
        fig.add_trace(go.Scatter(
            x=[actual_da], y=[0.5],
            mode='markers',
            marker=dict(size=16, color='red', symbol='x-thin', line=dict(width=3)),
            name='Actual Value',
            hovertemplate='Actual: %{x:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="DA Level Forecasts with Gradient Boosting Quantiles ",
        xaxis_title="DA Level (μg/L)",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=True,
        height=350,
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return pio.to_json(fig)


def sophisticated_nan_handling_for_correlation(df):
    """
    Implements sophisticated NaN handling strategy.
    """
    if df.empty:
        return df
        
    df_processed = df.copy()
    
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
    
    if 'da' in df_processed.columns:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'da']
        
        if feature_cols and len(df_processed) > 0:
            imputer = SimpleImputer(strategy="median")
            df_processed[feature_cols] = imputer.fit_transform(df_processed[feature_cols])
    
    return df_processed


def generate_correlation_heatmap(data, site=None):
    """Generate correlation heatmap."""
    
    if site:
        df = data[data['site'] == site].copy()
        title = f'Correlation Heatmap - {site}'
    else:
        df = data.copy()
        title = 'Overall Correlation Heatmap'
    
    df = sophisticated_nan_handling_for_correlation(df)
    
    exclude_cols = ['site', 'date', 'lon', 'lat']
    numeric_df = df.select_dtypes(include=['number']).drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
    
    corr_matrix = numeric_df.corr(method='pearson')
    
    annotations = []
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            color = "white" if abs(value) > 0.7 else "black"
            annotations.append({
                "x": corr_matrix.columns[j],
                "y": corr_matrix.index[i],
                "text": f"{value:.2f}",
                "font": {"color": color, "size": 10},
                "showarrow": False
            })
    
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
    """Generate sensitivity analysis plots including Sobol indices if possible."""
    
    if site:
        df_processed = data[data['site'] == site].copy()
        title_suffix = f" - {site}"
    else:
        df_processed = data.copy()
        title_suffix = " - All Sites"
    
    df_processed = sophisticated_nan_handling_for_correlation(df_processed)
    
    # Select numeric columns only, dropping non-relevant columns (same as correlation heatmap)
    exclude_cols = ['site', 'date', 'lon', 'lat']
    numeric_df = df_processed.select_dtypes(include=['number']).drop(columns=[col for col in exclude_cols if col in df_processed.columns], errors='ignore')
    
    df_clean = numeric_df.dropna(subset=['da']).copy()
    
    feature_cols = [col for col in df_clean.columns if col != 'da']
    
    X = df_clean[feature_cols]
    y = df_clean['da']
    
    correlations = df_clean[feature_cols + ['da']].corr()['da'].drop('da').abs().sort_values(ascending=False)
    
    plot1 = {
        "data": [{
            "type": "bar",
            "x": correlations.index.tolist(),
            "y": correlations.values.tolist(),
            "marker": {"color": "steelblue"}
        }],
        "layout": {
            "title": {
                "text": f"Correlation Sensitivity Analysis: Impact on DA Levels{title_suffix}",
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
    
    model = LinearRegression()
    model.fit(X, y)
    
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        
        if len(X) >= 50:
            problem = {
                'num_vars': len(feature_cols),
                'names': feature_cols,
                'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_cols]
            }
            
            N = min(64, max(8, len(X) // 20))
            param_values = saltelli.sample(problem, N, calc_second_order=False)
            
            Y = model.predict(param_values)
            
            if Y.ndim > 1:
                Y = Y.flatten()
            
            sobol_indices = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            first_order = sobol_indices['S1']
            
            sorted_idx = np.argsort(first_order)[::-1]
            sorted_features_sobol = [feature_cols[i] for i in sorted_idx]
            sorted_sobol = first_order[sorted_idx]
            
            plot_sobol = {
                "data": [{
                    "type": "bar",
                    "x": sorted_features_sobol[:10],
                    "y": sorted_sobol[:10].tolist(),
                    "marker": {"color": "green"}
                }],
                "layout": {
                    "title": {
                        "text": f"Sobol First Order Sensitivity Indices (Top 10){title_suffix}",
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
    except ImportError:
        pass
    except Exception:
        pass
    
    perm_result = permutation_importance(model, X, y, n_repeats=30, random_state=42)
    perm_importances = perm_result.importances_mean
    
    # Sort by importance
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
            "title": {
                "text": f"Permutation Feature Importance{title_suffix}",
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
        site_data = data[data['site'] == site].copy()
        title = f'DA vs Pseudo-nitzschia Time Series - {site}'
    else:
        site_data = data.groupby('date').agg({
            'da': 'mean',
            'pn': 'mean' if 'pn' in data.columns else lambda _: np.nan
        }).reset_index()
        title = 'DA vs Pseudo-nitzschia Time Series - All Sites Average'
    
    site_data = site_data.sort_values('date')
    
    has_pn = 'pn' in site_data.columns and not site_data['pn'].isna().all()
    
    scaler = MinMaxScaler()
    traces = []
    
    if 'da' in site_data.columns:
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
    
    return plot_data


def generate_waterfall_plot(data):
    """Generate waterfall plot visualization."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    data = data.sort_values(['lat', 'date'])
    
    lat_to_site = data.groupby('lat')['site'].first().to_dict()
    unique_lats = sorted(data['lat'].unique(), reverse=True)
    
    traces = []
    
    LATITUDE_BASELINE_MULTIPLIER = 3
    DA_SCALING_FACTOR = 0.01
    
    for i, lat in enumerate(unique_lats):
        site_name = lat_to_site.get(lat, f"Lat {lat:.2f}")
        site_data = data[data['lat'] == lat].sort_values('date')
        
        baseline_y = lat * LATITUDE_BASELINE_MULTIPLIER
        
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
        
        bar_da_levels = [20, 50, 100]
        bar_spacing_days = 120
        bar_target_date = pd.to_datetime('2012-01-01')
        
        for idx, da_level in enumerate(bar_da_levels):
            row_offset_days = 500 if i % 2 == 1 else 0
            bar_offset_days = (idx - 1) * bar_spacing_days + row_offset_days  # -120, 0, +120 days plus row offset
            bar_date = bar_target_date + pd.Timedelta(days=bar_offset_days)
            
            y_bar_base = baseline_y
            y_bar_top = baseline_y + DA_SCALING_FACTOR * da_level
            
            traces.append({
                "x": [bar_date.strftime('%Y-%m-%d'), bar_date.strftime('%Y-%m-%d')],
                "y": [y_bar_base, y_bar_top],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "rgba(128, 128, 128, 0.7)", "width": 2},
                "showlegend": False,
                "hovertemplate": f"Reference: {da_level} μg/g<extra></extra>"
            })
            
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
    
    y_tick_positions = [float(lat * LATITUDE_BASELINE_MULTIPLIER) if pd.notna(lat) else 0 for lat in unique_lats]
    y_tick_labels = [f"{lat:.2f}°N" for lat in unique_lats]
    
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
                    "standoff": 30
                },
                "tickmode": "array",
                "tickvals": y_tick_positions,
                "ticktext": y_tick_labels,
                "ticksuffix": "    ",
                "ticklen": 10,
                "tickwidth": 1
            },
            "height": 700,
            "hovermode": "x unified",
            "showlegend": True,
            "margin": {"l": 120}
        }
    }
    
    return plot_data


def generate_spectral_analysis(data, site=None):
    """Generate spectral analysis with optional XGBoost comparison."""
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    if site:
        site_data = data[data['site'] == site].copy()
        site_name = site
    else:
        site_data = data.groupby('date')['da'].mean().reset_index()
        site_name = "All Sites"
    
    site_data = site_data.sort_values('date')
    
    if 'da' in site_data.columns:
        da_values = site_data['da'].dropna().values
    else:
        da_values = site_data.dropna().values
    
    if len(da_values) < 20:
        return []
    
    if True:
        try:
            from pathlib import Path
            
            cache_dir = Path("cache/retrospective")
            cache_file = cache_dir / "regression_xgboost.parquet"
            
            if cache_file.exists():
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
                from forecasting.forecast_engine import ForecastEngine
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
    
    dominant_idx = np.argsort(psd[1:])[-3:][::-1]
    periods = 1 / freqs[1:]
    dominant_periods = periods[dominant_idx]
    
    traces = [{
        "x": freqs[1:].tolist(),
        "y": psd[1:].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": "Actual DA",
        "line": {"color": "blue", "width": 2}
    }]
    
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
                "z": np.log10(Sxx + 1e-10).tolist(),
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
        min_len = min(len(actual_for_comparison), len(xgb_predictions))
        actual_trimmed = actual_for_comparison[:min_len]
        xgb_trimmed = xgb_predictions[:min_len]
        
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