# -*- coding: utf-8 -*-
"""
Plots DA levels variation over time for different latitudes (sites),
plotting values relative to DA=0 for each latitude's baseline.
Includes vertical reference bars indicating specific DA levels
at a target date, spaced horizontally.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import os

# --- Configuration Parameters ---

# Input file
PARQUET_FILE = "final_output.parquet"
OUTPUT_DIR = "data-visualizations"
# MODIFIED: Updated filename to reflect change
OUTPUT_FILENAME = "waterfall_plot_absolute_da_v1.pdf"

# Plot Appearance
FIG_SIZE = (12, 8) # Adjusted size potentially needed for vertical range
TITLE = "Absolute DA Levels Variation by Site/Latitude Over Time" 

# Y-axis Scaling --- MIGHT NEED ADJUSTMENT ---
LATITUDE_BASELINE_MULTIPLIER = 3
DA_SCALING_FACTOR = 0.01 

# Reference Bar Settings
BAR_TARGET_DATE_STR = '2012-01-01'     # Central date for placing reference bars
BAR_DA_LEVELS = [20, 50, 100]    # DA values for the reference bars
BAR_SPACING_DAYS = 120                # Horizontal spacing between reference bars (days)
BAR_LINEWIDTH = 2.0                  # Linewidth of the vertical bars
BAR_LABEL_OFFSET_DAYS = 7            # Horizontal offset for bar value labels
BAR_LABEL_FONTSIZE = 4               # Fontsize for the bar value labels
BAR_LABEL_BACKGROUND_ALPHA = 0.6     # Transparency for the label background box

# Legend Settings
LEGEND_FONTSIZE = 8
LEGEND_POS = 'upper left'
LEGEND_ANCHOR = (1.03, 1) # Position relative to axes (x, y), >1 means outside

# Axis Settings
X_AXIS_LABEL = "Date"
Y_AXIS_LABEL = "Latitude (°N) - Baseline represents DA=0" 
DATE_FORMAT = '%Y-%m-%d'
X_AXIS_PADDING_DAYS = 120 # Extra space added to calculated x-limits

# --- Main Script ---

def create_da_waterfall_plot():
    """Loads data, creates, and saves the waterfall plot with absolute DA scaling."""

    # 1. Load Data
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Successfully loaded data from '{PARQUET_FILE}'.")

    # 2. Prepare Data
    df['date'] = pd.to_datetime(df['date'])
    lat_to_site = df.groupby('lat')['site'].first().to_dict()
    unique_lats = sorted(df['lat'].unique(), reverse=True)
    print(f"Data prepared for {len(unique_lats)} unique latitudes.")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    y_tick_positions = []
    y_tick_labels = []

    # Prepare bar plotting parameters
    bar_target_dt = pd.to_datetime(BAR_TARGET_DATE_STR)
    bar_target_num = mdates.date2num(bar_target_dt)
    num_bars = len(BAR_DA_LEVELS)
    max_bar_extent_days = (num_bars - 1) / 2 * BAR_SPACING_DAYS

    # Determine overall date range for axis limits
    plot_min_date = df['date'].min()
    plot_max_date = df['date'].max()

    if pd.isna(plot_min_date) or pd.isna(plot_max_date):
         print("Warning: Could not determine data date range. Using bar target date for limits.")
         view_start_dt = bar_target_dt - datetime.timedelta(days=max_bar_extent_days + X_AXIS_PADDING_DAYS)
         view_end_dt = bar_target_dt + datetime.timedelta(days=max_bar_extent_days + X_AXIS_PADDING_DAYS)
    else:
        view_start_dt = min(plot_min_date, bar_target_dt - datetime.timedelta(days=max_bar_extent_days)) - datetime.timedelta(days=X_AXIS_PADDING_DAYS)
        view_end_dt = max(plot_max_date, bar_target_dt + datetime.timedelta(days=max_bar_extent_days)) + datetime.timedelta(days=X_AXIS_PADDING_DAYS)

    view_start_num = mdates.date2num(view_start_dt)
    view_end_num = mdates.date2num(view_end_dt)


    # 4. Plot Data for Each Latitude
    print("Plotting data for each latitude...")
    for lat in unique_lats:
        group = df[df['lat'] == lat].sort_values(by='date').copy()

        time_nums = mdates.date2num(group['date'])
        da_values = group['da'] 

        # Check if there's any valid data to plot for this latitude
        has_valid_data = not da_values.empty and not da_values.isnull().all()

        # Calculate y-baseline for this latitude (represents DA = 0)
        baseline_y = lat * LATITUDE_BASELINE_MULTIPLIER

        # Handle potential NaNs by calculating directly; plot_date should handle plotting NaNs as gaps
        y_values = baseline_y + DA_SCALING_FACTOR * da_values

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

            for index, da_level in enumerate(BAR_DA_LEVELS):
                current_bar_offset_days = (start_offset_factor + index) * BAR_SPACING_DAYS
                current_bar_x_num = bar_target_num + current_bar_offset_days

                # Base of the bar is at the baseline (DA=0)
                y_bar_base = baseline_y
                # Top of the bar corresponds to the specific da_level scaled from baseline
                y_bar_top = baseline_y + DA_SCALING_FACTOR * da_level

                # Draw the vertical bar
                ax.vlines(x=current_bar_x_num,
                          ymin=y_bar_base,
                          ymax=y_bar_top,
                          color=line_color,
                          linestyle='-',
                          linewidth=BAR_LINEWIDTH,
                          alpha=0.85)

                # Add the label next to the bar top
                ax.text(current_bar_x_num + BAR_LABEL_OFFSET_DAYS,
                        y_bar_top,
                        f"{da_level}", # The DA value as text
                        fontsize=BAR_LABEL_FONTSIZE,
                        color=line_color,
                        verticalalignment='center',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=BAR_LABEL_BACKGROUND_ALPHA))


    # 5. Format Axes and Plot
    # X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_xlim(left=view_start_num, right=view_end_num)
    fig.autofmt_xdate(rotation=30, ha='right')

    # Y-axis
    ax.set_ylabel(Y_AXIS_LABEL) # Updated label
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # Grid
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

    # Title
    full_title = f"{TITLE}\n(Reference Bars show DA={BAR_DA_LEVELS})"
    ax.set_title(full_title, pad=15)

    # Legend
    ax.legend(loc=LEGEND_POS, bbox_to_anchor=LEGEND_ANCHOR, borderaxespad=0., fontsize=LEGEND_FONTSIZE)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 0.85, 0.94]) # Adjusted rect slightly


    # 6. Save and Show Plot
    effective_output_dir = OUTPUT_DIR
    if not os.path.exists(effective_output_dir):
        try:
            os.makedirs(effective_output_dir)
            print(f"Created output directory: '{effective_output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{effective_output_dir}': {e}. Attempting to save to current directory instead.")
            effective_output_dir = "."

    output_path = os.path.join(effective_output_dir, OUTPUT_FILENAME)

    plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=300)
    print(f"Plot successfully saved to: '{output_path}'")

    plt.show()


# --- Execution Guard ---
if __name__ == "__main__":
    create_da_waterfall_plot()