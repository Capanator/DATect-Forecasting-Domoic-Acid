#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for Domoic Acid Forecasting
=========================================================

This script provides comprehensive analysis of the DA forecasting dataset, combining
all existing visualizations with new comprehensive analyses following the site-by-site
+ comparison pattern proven successful in spectral analysis.

Includes:
- All existing visualizations from data-visualizations/ folder  
- New comprehensive analyses (quality assessment, patterns, distributions)
- Site-by-site analysis with unified comparison dashboard
- PNG exports for all analyses
- Combined all-sites analysis when applicable

Usage:
    python comprehensive-dataset-analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# Import SALib with error handling
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("Warning: SALib not available, Sobol analysis will be skipped")

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "final_output.parquet"
OUTPUT_DIR = "comprehensive-analysis-results"
DASH_PORT = 8085

# Waterfall plot configuration (from original waterfall plot.py)
WATERFALL_CONFIG = {
    'FIG_SIZE': (12, 8),
    'TITLE': "Absolute DA Levels Variation by Site/Latitude Over Time",
    'LATITUDE_BASELINE_MULTIPLIER': 3,
    'DA_SCALING_FACTOR': 0.01,
    'BAR_TARGET_DATE_STR': '2012-01-01',
    'BAR_DA_LEVELS': [20, 50, 100],
    'BAR_SPACING_DAYS': 120,
    'BAR_LINEWIDTH': 2.0,
    'BAR_LABEL_OFFSET_DAYS': 7,
    'BAR_LABEL_FONTSIZE': 4,
    'BAR_LABEL_BACKGROUND_ALPHA': 0.6,
    'LEGEND_FONTSIZE': 8,
    'LEGEND_POS': 'upper left',
    'LEGEND_ANCHOR': (1.03, 1),
    'X_AXIS_LABEL': "Date",
    'Y_AXIS_LABEL': "Latitude (°N) - Baseline represents DA=0",
    'DATE_FORMAT': '%Y-%m-%d',
    'X_AXIS_PADDING_DAYS': 120
}

class ComprehensiveDatasetAnalyzer:
    """
    Comprehensive analyzer for the DA forecasting dataset.
    
    Follows the proven site-by-site + comparison pattern from spectral analysis.
    """
    
    def __init__(self, data_file: str):
        """
        Initialize the analyzer.
        
        Args:
            data_file: Path to the parquet data file
        """
        self.data_file = data_file
        self.df = None
        self.sites = None
        self.site_data = {}
        self.analysis_results = {}
        self.all_sites_results = {}
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "site-analyses"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "png-exports"), exist_ok=True)
        
        # Load and prepare data
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load and prepare the dataset for analysis."""
        print(f"Loading data from {self.data_file}...")
        
        self.df = pd.read_parquet(self.data_file, engine="pyarrow")
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['site', 'date']).reset_index(drop=True)
        
        # Remove NaN rows for critical columns before analysis
        print(f"Original data shape: {self.df.shape}")
        
        # Remove rows where DA is NaN (our target variable)
        self.df = self.df.dropna(subset=['da'])
        print(f"After removing NaN DA values: {self.df.shape}")
        
        self.sites = sorted(self.df['site'].unique())
        
        # Prepare site-specific data
        for site in self.sites:
            site_df = self.df[self.df['site'] == site].copy()
            # Remove NaN rows for each site
            site_df = site_df.dropna(how='all')  # Remove completely empty rows
            self.site_data[site] = site_df
        
        print(f"Loaded {len(self.df)} records across {len(self.sites)} sites")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def create_correlation_heatmaps(self):
        """Create correlation heatmaps for each site and overall (from correlation heatmap.py)."""
        print("Creating correlation heatmaps...")
        
        # Overall correlation heatmap first
        print("Creating overall correlation heatmap for all data")
        overall_df = self.df.copy()
        
        # Drop non-numeric columns
        cols_to_drop = [col for col in ['lon', 'lat', 'site'] if col in overall_df.columns]
        if cols_to_drop:
            overall_df = overall_df.drop(columns=cols_to_drop)
        
        # Select numeric columns only
        numeric_df = overall_df.select_dtypes(include=['number'])
        
        # Remove rows with NaN values
        numeric_df = numeric_df.dropna()
        
        if len(numeric_df) > 0:
            # Compute correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create the heatmap using imshow with the RdBu colormap
            cax = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
            
            # Set tick marks and labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, fontsize=10, rotation=45)
            ax.set_yticklabels(corr_matrix.index, fontsize=10)
            
            # Title and axis labels
            ax.set_title('Overall Correlation Heatmap - All Sites', fontsize=16, pad=20)
            ax.set_xlabel('Variable', fontsize=14)
            ax.set_ylabel('Variable', fontsize=14)
            
            # Annotate each cell with the correlation value (rounded to 2 decimals)
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    value = corr_matrix.iloc[i, j]
                    # Use white text for dark cells (strong correlations), black otherwise
                    color = "white" if abs(value) > 0.7 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                           color=color, fontsize=8)
            
            # Add colorbar with a label
            cbar = fig.colorbar(cax, ax=ax)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Correlation (r)', rotation=270, labelpad=20, fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure as PNG
            png_path = os.path.join(OUTPUT_DIR, "png-exports", "correlation_heatmap_overall.png")
            plt.savefig(png_path, format="png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Overall heatmap saved at: {png_path}")
            
            # Store for all-sites analysis
            self.all_sites_results['correlation_heatmap'] = corr_matrix
        
        # Now process each site
        for site in self.sites:
            print(f"Processing correlation heatmap for site: {site}")
            
            site_df = self.site_data[site].copy()
            
            # Select numeric columns only (excluding the site column)
            if 'site' in site_df.columns:
                site_df = site_df.drop(columns=['site'])
            numeric_df = site_df.select_dtypes(include=['number'])
            
            # Remove rows with NaN values
            numeric_df = numeric_df.dropna()
            
            if len(numeric_df) > 10:  # Need sufficient data
                # Compute correlation matrix
                corr_matrix = numeric_df.corr()
                
                # Create the figure and axis
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Create the heatmap using imshow with the RdBu colormap
                cax = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
                
                # Set tick marks and labels
                ax.set_xticks(np.arange(len(corr_matrix.columns)))
                ax.set_yticks(np.arange(len(corr_matrix.index)))
                ax.set_xticklabels(corr_matrix.columns, fontsize=10, rotation=45)
                ax.set_yticklabels(corr_matrix.index, fontsize=10)
                
                # Title and axis labels
                ax.set_title(f'Correlation Heatmap - {site}', fontsize=16, pad=20)
                ax.set_xlabel('Variable', fontsize=14)
                ax.set_ylabel('Variable', fontsize=14)
                
                # Annotate each cell with the correlation value (rounded to 2 decimals)
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        value = corr_matrix.iloc[i, j]
                        # Use white text for dark cells (strong correlations), black otherwise
                        color = "white" if abs(value) > 0.7 else "black"
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                               color=color, fontsize=8)
                
                # Add colorbar with a label
                cbar = fig.colorbar(cax, ax=ax)
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label('Correlation (r)', rotation=270, labelpad=20, fontsize=14)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure as PNG
                png_path = os.path.join(OUTPUT_DIR, "png-exports", f"correlation_heatmap_{site.replace(' ', '_')}.png")
                plt.savefig(png_path, format="png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Heatmap for site '{site}' saved at: {png_path}")
                
                # Store in site results
                if site not in self.analysis_results:
                    self.analysis_results[site] = {}
                self.analysis_results[site]['correlation_heatmap'] = corr_matrix
    
    def create_waterfall_plot(self):
        """Create waterfall plot exactly as in waterfall plot.py."""
        print("Creating waterfall plot...")
        
        # Load data and remove NaN values
        df = self.df.copy()
        df = df.dropna(subset=['da', 'lat', 'site'])  # Remove NaN for essential columns
        
        print(f"Successfully loaded data for waterfall plot: {len(df)} records")
        
        # Prepare data (exactly from waterfall plot.py)
        df['date'] = pd.to_datetime(df['date'])
        lat_to_site = df.groupby('lat')['site'].first().to_dict()
        unique_lats = sorted(df['lat'].unique(), reverse=True)
        print(f"Data prepared for {len(unique_lats)} unique latitudes.")
        
        # Setup Plot
        fig, ax = plt.subplots(figsize=WATERFALL_CONFIG['FIG_SIZE'])
        y_tick_positions = []
        y_tick_labels = []
        
        # Prepare bar plotting parameters
        bar_target_dt = pd.to_datetime(WATERFALL_CONFIG['BAR_TARGET_DATE_STR'])
        bar_target_num = mdates.date2num(bar_target_dt)
        num_bars = len(WATERFALL_CONFIG['BAR_DA_LEVELS'])
        max_bar_extent_days = (num_bars - 1) / 2 * WATERFALL_CONFIG['BAR_SPACING_DAYS']
        
        # Determine overall date range for axis limits
        plot_min_date = df['date'].min()
        plot_max_date = df['date'].max()
        
        if pd.isna(plot_min_date) or pd.isna(plot_max_date):
            print("Warning: Could not determine data date range. Using bar target date for limits.")
            view_start_dt = bar_target_dt - pd.Timedelta(days=max_bar_extent_days + WATERFALL_CONFIG['X_AXIS_PADDING_DAYS'])
            view_end_dt = bar_target_dt + pd.Timedelta(days=max_bar_extent_days + WATERFALL_CONFIG['X_AXIS_PADDING_DAYS'])
        else:
            view_start_dt = min(plot_min_date, bar_target_dt - pd.Timedelta(days=max_bar_extent_days)) - pd.Timedelta(days=WATERFALL_CONFIG['X_AXIS_PADDING_DAYS'])
            view_end_dt = max(plot_max_date, bar_target_dt + pd.Timedelta(days=max_bar_extent_days)) + pd.Timedelta(days=WATERFALL_CONFIG['X_AXIS_PADDING_DAYS'])
        
        view_start_num = mdates.date2num(view_start_dt)
        view_end_num = mdates.date2num(view_end_dt)
        
        # Plot Data for Each Latitude
        print("Plotting data for each latitude...")
        for lat in unique_lats:
            group = df[df['lat'] == lat].sort_values(by='date').copy()
            
            time_nums = mdates.date2num(group['date'])
            da_values = group['da']
            
            # Check if there's any valid data to plot for this latitude
            has_valid_data = not da_values.empty and not da_values.isnull().all()
            
            # Calculate y-baseline for this latitude (represents DA = 0)
            baseline_y = lat * WATERFALL_CONFIG['LATITUDE_BASELINE_MULTIPLIER']
            
            # Handle potential NaNs by calculating directly; plot_date should handle plotting NaNs as gaps
            y_values = baseline_y + WATERFALL_CONFIG['DA_SCALING_FACTOR'] * da_values
            
            # Get site name for legend label
            site_name = lat_to_site.get(lat, f"Lat {lat:.2f}")
            
            # Plot the time series line
            line, = ax.plot_date(time_nums, y_values, '-', label=site_name, markersize=2, linewidth=1)
            line_color = line.get_color()
            
            # Store tick positions/labels for Y-axis (baseline still represents the latitude)
            y_tick_positions.append(baseline_y)
            y_tick_labels.append(f"{lat:.2f}")
            
            # Plot the reference bars for this latitude (only if valid data exists)
            if has_valid_data:
                start_offset_factor = - (num_bars - 1) / 2
                
                for index, da_level in enumerate(WATERFALL_CONFIG['BAR_DA_LEVELS']):
                    current_bar_offset_days = (start_offset_factor + index) * WATERFALL_CONFIG['BAR_SPACING_DAYS']
                    current_bar_x_num = bar_target_num + current_bar_offset_days
                    
                    # Base of the bar is at the baseline (DA=0)
                    y_bar_base = baseline_y
                    # Top of the bar corresponds to the specific da_level scaled from baseline
                    y_bar_top = baseline_y + WATERFALL_CONFIG['DA_SCALING_FACTOR'] * da_level
                    
                    # Draw the vertical bar
                    ax.vlines(x=current_bar_x_num,
                             ymin=y_bar_base,
                             ymax=y_bar_top,
                             color=line_color,
                             linestyle='-',
                             linewidth=WATERFALL_CONFIG['BAR_LINEWIDTH'],
                             alpha=0.85)
                    
                    # Add the label next to the bar top
                    ax.text(current_bar_x_num + WATERFALL_CONFIG['BAR_LABEL_OFFSET_DAYS'],
                           y_bar_top,
                           f"{da_level}",  # The DA value as text
                           fontsize=WATERFALL_CONFIG['BAR_LABEL_FONTSIZE'],
                           color=line_color,
                           verticalalignment='center',
                           horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', 
                                   alpha=WATERFALL_CONFIG['BAR_LABEL_BACKGROUND_ALPHA']))
        
        # Format Axes and Plot
        # X-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(WATERFALL_CONFIG['DATE_FORMAT']))
        ax.set_xlabel(WATERFALL_CONFIG['X_AXIS_LABEL'])
        ax.set_xlim(left=view_start_num, right=view_end_num)
        fig.autofmt_xdate(rotation=30, ha='right')
        
        # Y-axis
        ax.set_ylabel(WATERFALL_CONFIG['Y_AXIS_LABEL'])
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
        
        # Grid
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Title
        full_title = f"{WATERFALL_CONFIG['TITLE']}\n(Reference Bars show DA={WATERFALL_CONFIG['BAR_DA_LEVELS']})"
        ax.set_title(full_title, pad=15)
        
        # Legend
        ax.legend(loc=WATERFALL_CONFIG['LEGEND_POS'], 
                 bbox_to_anchor=WATERFALL_CONFIG['LEGEND_ANCHOR'], 
                 borderaxespad=0., 
                 fontsize=WATERFALL_CONFIG['LEGEND_FONTSIZE'])
        
        # Adjust layout
        plt.tight_layout(rect=[0.03, 0.03, 0.85, 0.94])
        
        # Save and Show Plot
        output_path = os.path.join(OUTPUT_DIR, "png-exports", "waterfall_plot_absolute_da.png")
        plt.savefig(output_path, format="png", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Waterfall plot saved to: '{output_path}'")
        
        # Store in all-sites results
        self.all_sites_results['waterfall_plot'] = {
            'created': True,
            'path': output_path,
            'n_sites': len(unique_lats),
            'date_range': (plot_min_date, plot_max_date)
        }
    
    def create_normalized_time_series(self):
        """Create normalized time series plots (from dsfjalskd.py)."""
        print("Creating normalized time series plots...")
        
        scaler = MinMaxScaler()
        
        # Create overall normalized time series for all sites
        overall_df = self.df.copy()
        overall_df = overall_df.dropna(subset=['modis-chla', 'da'])  # Remove NaN values
        
        if len(overall_df) > 0:
            # Normalize overall data
            overall_df['modis-chla_normalized'] = scaler.fit_transform(overall_df[['modis-chla']])
            overall_df['da_capped'] = np.minimum(overall_df['da'], 80)
            overall_df['da_normalized'] = scaler.fit_transform(overall_df[['da_capped']])
            
            # Create overall plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot by site with different colors
            for site in self.sites:
                site_data = overall_df[overall_df['site'] == site]
                if len(site_data) > 0:
                    ax.plot(site_data['date'], site_data['modis-chla_normalized'], 
                           alpha=0.7, linewidth=1, label=f'{site} - MODIS Chla')
                    ax.plot(site_data['date'], site_data['da_normalized'], 
                           alpha=0.7, linewidth=1, linestyle='--', label=f'{site} - DA')
            
            ax.set_title('Normalized MODIS-Chla and DA over Time - All Sites', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Normalized Value (0-1)', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save overall plot
            overall_path = os.path.join(OUTPUT_DIR, "png-exports", "normalized_timeseries_all_sites.png")
            plt.savefig(overall_path, format="png", bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Overall normalized time series saved: {overall_path}")
            
            # Store in all-sites results
            self.all_sites_results['normalized_timeseries'] = {
                'created': True,
                'path': overall_path,
                'n_records': len(overall_df)
            }
        
        # Create individual site plots
        for site in self.sites:
            site_df = self.site_data[site].copy()
            site_df = site_df.dropna(subset=['modis-chla', 'da'])  # Remove NaN values
            
            if len(site_df) > 10:  # Need sufficient data
                # Normalize data for this site
                site_df['modis-chla_normalized'] = scaler.fit_transform(site_df[['modis-chla']])
                site_df['da_capped'] = np.minimum(site_df['da'], 80)
                site_df['da_normalized'] = scaler.fit_transform(site_df[['da_capped']])
                
                # Create the plot
                plt.figure(figsize=(12, 6))
                plt.plot(site_df['date'], site_df['modis-chla_normalized'], 
                        label='modis-chla (Normalized)', linewidth=2)
                plt.plot(site_df['date'], site_df['da_normalized'], 
                        label='da (Normalized, Capped at 80)', linewidth=2)
                
                # Add title and labels
                plt.title(f'Normalized modis-chla and Capped da over Time - {site}', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Normalized Value', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Improve date formatting on x-axis
                plt.gcf().autofmt_xdate()
                
                # Set y-axis limits to 0 and 1 for normalized data
                plt.ylim(0, 1)
                
                plt.tight_layout()
                
                # Save plot
                site_path = os.path.join(OUTPUT_DIR, "png-exports", f"normalized_timeseries_{site.replace(' ', '_')}.png")
                plt.savefig(site_path, format="png", bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"Normalized time series for {site} saved: {site_path}")
                
                # Store in site results
                if site not in self.analysis_results:
                    self.analysis_results[site] = {}
                self.analysis_results[site]['normalized_timeseries'] = {
                    'created': True,
                    'path': site_path,
                    'n_records': len(site_df)
                }
    
    def create_sensitivity_analysis(self):
        """Create sensitivity analysis plots (from sensitivity test.py)."""
        print("Creating sensitivity analysis...")
        
        # Overall sensitivity analysis for all sites
        overall_df = self.df.copy()
        
        # Drop non-essential columns and remove NaN values
        cols_to_drop = [col for col in ['lon', 'lat', 'site'] if col in overall_df.columns]
        if cols_to_drop:
            overall_df = overall_df.drop(columns=cols_to_drop)
        
        # Keep only numeric columns and remove NaN values
        numeric_df = overall_df.select_dtypes(include=['number'])
        numeric_df = numeric_df.dropna()  # Remove all NaN rows
        
        target_var = 'da'
        if target_var in numeric_df.columns and len(numeric_df) > 50:
            
            # 1. Correlation Analysis
            correlations = numeric_df.corr()[target_var].drop(target_var).abs().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.bar(correlations.index[:15], correlations.values[:15])  # Top 15
            plt.title("Correlation Sensitivity Analysis: Impact on DA Levels - All Sites", fontsize=16)
            plt.xlabel("Input Variables", fontsize=14)
            plt.ylabel("Absolute Pearson Correlation", fontsize=14)
            plt.xticks(rotation=45, fontsize=10)
            plt.tight_layout()
            corr_path = os.path.join(OUTPUT_DIR, "png-exports", "sensitivity_correlation_all_sites.png")
            plt.savefig(corr_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Overall correlation sensitivity saved: {corr_path}")
            
            # 2. Permutation Feature Importance
            X = numeric_df.drop(columns=[target_var])
            y = numeric_df[target_var]
            feature_names = X.columns.tolist()
            
            if len(X) > 100:  # Need sufficient data
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                perm_importances = perm_result.importances_mean
                
                # Sort and plot top features
                perm_indices = np.argsort(perm_importances)[::-1][:15]  # Top 15
                
                plt.figure(figsize=(12, 8))
                plt.bar(range(len(perm_indices)), perm_importances[perm_indices])
                plt.title("Permutation Feature Importance - All Sites", fontsize=16)
                plt.xlabel("Input Variables", fontsize=14)
                plt.ylabel("Decrease in Model Score", fontsize=14)
                plt.xticks(range(len(perm_indices)), [feature_names[i] for i in perm_indices], rotation=45, fontsize=10)
                plt.tight_layout()
                perm_path = os.path.join(OUTPUT_DIR, "png-exports", "sensitivity_permutation_all_sites.png")
                plt.savefig(perm_path, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Overall permutation importance saved: {perm_path}")
                
                # 3. Sobol Analysis (if available)
                if SALIB_AVAILABLE and len(X) > 200:
                    try:
                        problem = {
                            'num_vars': len(feature_names),
                            'names': feature_names,
                            'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_names]
                        }
                        
                        param_values = saltelli.sample(problem, 64, calc_second_order=False)
                        Y = model.predict(param_values)
                        sobol_indices = sobol.analyze(problem, Y, print_to_console=False)
                        first_order = sobol_indices['S1']
                        
                        # Sort and plot
                        sobol_indices_sorted = np.argsort(first_order)[::-1][:15]  # Top 15
                        
                        plt.figure(figsize=(12, 8))
                        plt.bar(range(len(sobol_indices_sorted)), first_order[sobol_indices_sorted])
                        plt.title("Sobol First Order Sensitivity Indices - All Sites", fontsize=16)
                        plt.xlabel("Input Variables", fontsize=14)
                        plt.ylabel("First Order Sobol Index", fontsize=14)
                        plt.xticks(range(len(sobol_indices_sorted)), 
                                  [feature_names[i] for i in sobol_indices_sorted], rotation=45, fontsize=10)
                        plt.tight_layout()
                        sobol_path = os.path.join(OUTPUT_DIR, "png-exports", "sensitivity_sobol_all_sites.png")
                        plt.savefig(sobol_path, format='png', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Overall Sobol sensitivity saved: {sobol_path}")
                        
                    except Exception as e:
                        print(f"Sobol analysis failed: {e}")
            
            # Store in all-sites results
            self.all_sites_results['sensitivity_analysis'] = {
                'correlation_path': corr_path,
                'n_features': len(correlations),
                'top_correlations': correlations.head(10).to_dict()
            }
        
        # Individual site sensitivity analysis
        for site in self.sites:
            site_df = self.site_data[site].copy()
            
            # Process site data
            if 'site' in site_df.columns:
                site_df = site_df.drop(columns=['site'])
            
            numeric_df = site_df.select_dtypes(include=['number'])
            numeric_df = numeric_df.dropna()  # Remove NaN values
            
            if target_var in numeric_df.columns and len(numeric_df) > 20:
                correlations = numeric_df.corr()[target_var].drop(target_var).abs().sort_values(ascending=False)
                
                plt.figure(figsize=(10, 6))
                plt.bar(correlations.index[:10], correlations.values[:10])  # Top 10
                plt.title(f"Correlation Sensitivity Analysis - {site}", fontsize=14)
                plt.xlabel("Input Variables", fontsize=12)
                plt.ylabel("Absolute Pearson Correlation", fontsize=12)
                plt.xticks(rotation=45, fontsize=10)
                plt.tight_layout()
                site_corr_path = os.path.join(OUTPUT_DIR, "png-exports", f"sensitivity_correlation_{site.replace(' ', '_')}.png")
                plt.savefig(site_corr_path, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Sensitivity analysis for {site} saved: {site_corr_path}")
                
                # Store in site results
                if site not in self.analysis_results:
                    self.analysis_results[site] = {}
                self.analysis_results[site]['sensitivity_analysis'] = {
                    'correlation_path': site_corr_path,
                    'top_correlations': correlations.head(5).to_dict()
                }
    
    def create_site_comparison_plots(self):
        """Create comparison plots across all sites."""
        print("Creating site comparison plots...")
        
        # 1. Data Quality Comparison
        sites = []
        data_counts = []
        missing_percentages = []
        
        for site in self.sites:
            site_df = self.site_data[site]
            sites.append(site)
            data_counts.append(len(site_df))
            
            # Calculate average missing percentage
            missing_pct = site_df.isnull().mean().mean() * 100
            missing_percentages.append(missing_pct)
        
        # Data counts comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sites, data_counts)
        plt.title('Data Record Counts by Site', fontsize=16)
        plt.xlabel('Site', fontsize=14)
        plt.ylabel('Number of Records', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, data_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        count_path = os.path.join(OUTPUT_DIR, "png-exports", "comparison_data_counts.png")
        plt.savefig(count_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Data counts comparison saved: {count_path}")
        
        # Missing data comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sites, missing_percentages, color='orange', alpha=0.7)
        plt.title('Average Missing Data Percentage by Site', fontsize=16)
        plt.xlabel('Site', fontsize=14)
        plt.ylabel('Missing Data (%)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, pct in zip(bars, missing_percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_percentages)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        missing_path = os.path.join(OUTPUT_DIR, "png-exports", "comparison_missing_data.png")
        plt.savefig(missing_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Missing data comparison saved: {missing_path}")
        
        # 2. DA Statistics Comparison
        da_means = []
        da_maxes = []
        
        for site in self.sites:
            site_df = self.site_data[site]
            da_data = site_df['da'].dropna()
            if len(da_data) > 0:
                da_means.append(da_data.mean())
                da_maxes.append(da_data.max())
            else:
                da_means.append(0)
                da_maxes.append(0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Mean DA levels
        bars1 = ax1.bar(sites, da_means, color='skyblue', alpha=0.8)
        ax1.set_title('Mean DA Levels by Site', fontsize=14)
        ax1.set_ylabel('Mean DA (μg/g)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, mean_val in zip(bars1, da_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(da_means)*0.01,
                    f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Max DA levels
        bars2 = ax2.bar(sites, da_maxes, color='lightcoral', alpha=0.8)
        ax2.set_title('Maximum DA Levels by Site', fontsize=14)
        ax2.set_xlabel('Site', fontsize=12)
        ax2.set_ylabel('Max DA (μg/g)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, max_val in zip(bars2, da_maxes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(da_maxes)*0.01,
                    f'{max_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        da_stats_path = os.path.join(OUTPUT_DIR, "png-exports", "comparison_da_statistics.png")
        plt.savefig(da_stats_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"DA statistics comparison saved: {da_stats_path}")
        
        # Store comparison results
        self.all_sites_results['site_comparison'] = {
            'data_counts': dict(zip(sites, data_counts)),
            'missing_percentages': dict(zip(sites, missing_percentages)),
            'da_means': dict(zip(sites, da_means)),
            'da_maxes': dict(zip(sites, da_maxes)),
            'comparison_plots': [count_path, missing_path, da_stats_path]
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis for all sites."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATASET ANALYSIS")
        print("="*80)
        
        # Create all analyses
        print("\n1. Creating correlation heatmaps...")
        self.create_correlation_heatmaps()
        
        print("\n2. Creating waterfall plot...")
        self.create_waterfall_plot()
        
        print("\n3. Creating normalized time series...")
        self.create_normalized_time_series()
        
        print("\n4. Creating sensitivity analysis...")
        self.create_sensitivity_analysis()
        
        print("\n5. Creating site comparison plots...")
        self.create_site_comparison_plots()
        
        print(f"\nAnalysis complete! All PNG files saved to: {os.path.join(OUTPUT_DIR, 'png-exports')}")
        print(f"Site-specific analyses: {len(self.analysis_results)} sites")
        print(f"Overall analyses: {len(self.all_sites_results)} analysis types")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create a summary report of all analyses."""
        report_path = os.path.join(OUTPUT_DIR, "analysis_summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE DATASET ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.data_file}\n")
            f.write(f"Total Records: {len(self.df)}\n")
            f.write(f"Sites Analyzed: {len(self.sites)}\n")
            f.write(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}\n\n")
            
            f.write("SITES:\n")
            for i, site in enumerate(self.sites, 1):
                site_count = len(self.site_data[site])
                f.write(f"  {i:2d}. {site:<20} ({site_count:,} records)\n")
            
            f.write(f"\nANALYSES CREATED:\n")
            f.write(f"  • Correlation Heatmaps: {len(self.sites) + 1} files (1 overall + {len(self.sites)} per site)\n")
            f.write(f"  • Waterfall Plot: 1 file (all sites combined)\n")
            f.write(f"  • Normalized Time Series: {len(self.sites) + 1} files (1 overall + {len(self.sites)} per site)\n")
            f.write(f"  • Sensitivity Analysis: {len(self.sites) + 1} files (1 overall + {len(self.sites)} per site)\n")
            f.write(f"  • Site Comparison: 3 files (data counts, missing data, DA statistics)\n")
            
            total_png_files = (len(self.sites) + 1) * 3 + 1 + 3  # heatmaps + timeseries + sensitivity + waterfall + comparisons
            f.write(f"\nTOTAL PNG FILES GENERATED: ~{total_png_files}\n")
            f.write(f"OUTPUT DIRECTORY: {OUTPUT_DIR}/png-exports/\n")
            
            f.write(f"\nKEY FINDINGS:\n")
            if 'site_comparison' in self.all_sites_results:
                comp_data = self.all_sites_results['site_comparison']
                
                # Site with most data
                max_data_site = max(comp_data['data_counts'], key=comp_data['data_counts'].get)
                f.write(f"  • Site with most data: {max_data_site} ({comp_data['data_counts'][max_data_site]:,} records)\n")
                
                # Site with highest mean DA
                max_da_site = max(comp_data['da_means'], key=comp_data['da_means'].get)
                f.write(f"  • Site with highest mean DA: {max_da_site} ({comp_data['da_means'][max_da_site]:.3f} μg/g)\n")
                
                # Site with lowest missing data
                min_missing_site = min(comp_data['missing_percentages'], key=comp_data['missing_percentages'].get)
                f.write(f"  • Site with best data quality: {min_missing_site} ({comp_data['missing_percentages'][min_missing_site]:.1f}% missing)\n")
        
        print(f"\nSummary report saved: {report_path}")

    def create_interactive_dashboard(self):
        """Create comprehensive interactive dashboard."""
        print(f"\nCreating interactive dashboard on port {DASH_PORT}...")
        
        app = dash.Dash(__name__)
        app.title = "Comprehensive Dataset Analysis Dashboard"
        
        # Dashboard layout
        app.layout = html.Div([
            html.H1("Comprehensive Dataset Analysis Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Label("Select Analysis Type:"),
                dcc.Dropdown(
                    id='analysis-type',
                    options=[
                        {'label': 'Site Overview', 'value': 'overview'},
                        {'label': 'Correlation Heatmaps', 'value': 'correlation'},
                        {'label': 'Waterfall Plot', 'value': 'waterfall'}, 
                        {'label': 'Normalized Time Series', 'value': 'timeseries'},
                        {'label': 'Sensitivity Analysis', 'value': 'sensitivity'},
                        {'label': 'Site Comparison', 'value': 'comparison'}
                    ],
                    value='overview'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Select Site:"),
                dcc.Dropdown(
                    id='site-selector',
                    options=[{'label': 'All Sites', 'value': 'all'}] + 
                            [{'label': site, 'value': site} for site in self.sites],
                    value='all'
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'marginBottom': '20px'}),
            
            html.Div(id='analysis-content', style={'marginTop': '30px'})
        ])
        
        @app.callback(
            Output('analysis-content', 'children'),
            [Input('analysis-type', 'value'),
             Input('site-selector', 'value')]
        )
        def update_analysis_content(analysis_type, selected_site):
            if analysis_type == 'overview':
                return self._create_overview_dashboard(selected_site)
            elif analysis_type == 'correlation':
                return self._create_correlation_dashboard(selected_site)
            elif analysis_type == 'waterfall':
                return self._create_waterfall_dashboard()
            elif analysis_type == 'timeseries':
                return self._create_timeseries_dashboard(selected_site)
            elif analysis_type == 'sensitivity':
                return self._create_sensitivity_dashboard(selected_site)
            elif analysis_type == 'comparison':
                return self._create_comparison_dashboard()
            else:
                return html.Div("Select an analysis type")
        
        app.run_server(debug=False, port=DASH_PORT, host='0.0.0.0')
    
    def _create_overview_dashboard(self, selected_site):
        """Create overview dashboard content."""
        if selected_site == 'all':
            return html.Div([
                html.H3("Overall Dataset Overview"),
                html.P(f"Total Records: {len(self.df):,}"),
                html.P(f"Sites: {len(self.sites)}"),
                html.P(f"Date Range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"),
                html.P(f"DA Range: {self.df['da'].min():.3f} to {self.df['da'].max():.3f} μg/g"),
                
                html.H4("Site List:"),
                html.Ul([html.Li(f"{site} ({len(self.site_data[site]):,} records)") for site in self.sites])
            ])
        else:
            if selected_site in self.site_data:
                site_df = self.site_data[selected_site]
                return html.Div([
                    html.H3(f"Site Overview: {selected_site}"),
                    html.P(f"Records: {len(site_df):,}"),
                    html.P(f"Date Range: {site_df['date'].min().strftime('%Y-%m-%d')} to {site_df['date'].max().strftime('%Y-%m-%d')}"),
                    html.P(f"DA Mean: {site_df['da'].mean():.3f} μg/g"),
                    html.P(f"DA Range: {site_df['da'].min():.3f} to {site_df['da'].max():.3f} μg/g"),
                    html.P(f"Missing Data: {site_df.isnull().mean().mean()*100:.1f}%")
                ])
            else:
                return html.Div("Site not found")
    
    def _create_correlation_dashboard(self, selected_site):
        """Create correlation dashboard content."""
        return html.Div([
            html.H3(f"Correlation Analysis: {selected_site if selected_site != 'all' else 'All Sites'}"),
            html.P("Correlation heatmaps have been generated and saved as PNG files."),
            html.P(f"Check the output directory: {os.path.join(OUTPUT_DIR, 'png-exports')}")
        ])
    
    def _create_waterfall_dashboard(self):
        """Create waterfall dashboard content.""" 
        if 'waterfall_plot' in self.all_sites_results:
            waterfall_info = self.all_sites_results['waterfall_plot']
            return html.Div([
                html.H3("Waterfall Plot Analysis"),
                html.P(f"Waterfall plot created for {waterfall_info['n_sites']} sites"),
                html.P(f"Date range: {waterfall_info['date_range'][0].strftime('%Y-%m-%d')} to {waterfall_info['date_range'][1].strftime('%Y-%m-%d')}"),
                html.P(f"Saved to: {waterfall_info['path']}")
            ])
        else:
            return html.Div("Waterfall plot not available")
    
    def _create_timeseries_dashboard(self, selected_site):
        """Create timeseries dashboard content."""
        return html.Div([
            html.H3(f"Normalized Time Series: {selected_site if selected_site != 'all' else 'All Sites'}"),
            html.P("Normalized time series plots have been generated and saved as PNG files."),
            html.P("These show MODIS chlorophyll-a vs DA concentrations over time."),
            html.P(f"Check the output directory: {os.path.join(OUTPUT_DIR, 'png-exports')}")
        ])
    
    def _create_sensitivity_dashboard(self, selected_site):
        """Create sensitivity dashboard content."""
        return html.Div([
            html.H3(f"Sensitivity Analysis: {selected_site if selected_site != 'all' else 'All Sites'}"),
            html.P("Sensitivity analysis plots have been generated showing variable importance."),
            html.P("Includes correlation sensitivity and permutation importance analysis."),
            html.P(f"Check the output directory: {os.path.join(OUTPUT_DIR, 'png-exports')}")
        ])
    
    def _create_comparison_dashboard(self):
        """Create comparison dashboard content."""
        if 'site_comparison' in self.all_sites_results:
            comp_data = self.all_sites_results['site_comparison']
            
            # Find key statistics
            max_data_site = max(comp_data['data_counts'], key=comp_data['data_counts'].get)
            max_da_site = max(comp_data['da_means'], key=comp_data['da_means'].get) 
            min_missing_site = min(comp_data['missing_percentages'], key=comp_data['missing_percentages'].get)
            
            return html.Div([
                html.H3("Site Comparison Analysis"),
                html.H4("Key Findings:"),
                html.Ul([
                    html.Li(f"Site with most data: {max_data_site} ({comp_data['data_counts'][max_data_site]:,} records)"),
                    html.Li(f"Site with highest mean DA: {max_da_site} ({comp_data['da_means'][max_da_site]:.3f} μg/g)"),
                    html.Li(f"Site with best data quality: {min_missing_site} ({comp_data['missing_percentages'][min_missing_site]:.1f}% missing)")
                ]),
                html.P(f"Comparison plots saved to: {os.path.join(OUTPUT_DIR, 'png-exports')}")
            ])
        else:
            return html.Div("Comparison data not available")


def main():
    """Main function to run comprehensive dataset analysis."""
    print("Domoic Acid Dataset Comprehensive Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ComprehensiveDatasetAnalyzer(DATA_FILE)
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
    
    # Launch interactive dashboard
    print(f"\nLaunching interactive dashboard on http://localhost:{DASH_PORT}")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        analyzer.create_interactive_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")
        print("Analysis results are still available in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()