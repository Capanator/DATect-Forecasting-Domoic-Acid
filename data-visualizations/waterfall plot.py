import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Read Parquet data
try:
    df = pd.read_parquet("final_output.parquet")
except FileNotFoundError:
    print("Error: 'final_output.parquet' not found. Please ensure the file exists in the correct directory.")
    exit()

# --- Data Preparation ---
df['Date'] = pd.to_datetime(df['Date'])

lat_to_site = df.groupby('latitude')['Site'].first().to_dict()

# Get unique latitudes and sort them in DESCENDING order (North to South visually)
unique_lats = sorted(df['latitude'].unique(), reverse=True)

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(12, 9))

# --- Scaling Factors ---
scaling_factor_da = 0.01
baseline_multiplier = 5

annotation_x_offset_days = 10

# --- Plotting Loop ---
y_tick_positions = []
y_tick_labels = []

for lat in unique_lats:
    group = df[df['latitude'] == lat].sort_values(by='Date').copy()

    if group.empty:
        print(f"Warning: No data found for latitude {lat}. Skipping.")
        continue

    time_nums = mdates.date2num(group['Date'])
    da_values = group['DA_Levels']

    if da_values.isnull().all() or da_values.empty:
        min_da, max_da, mean_da = np.nan, np.nan, np.nan
    else:
        min_da = da_values.min()
        max_da = da_values.max()
        mean_da = da_values.mean()

    baseline = lat * baseline_multiplier

    if pd.notna(mean_da):
        y_values = baseline + scaling_factor_da * (da_values - mean_da)
    else:
        y_values = pd.Series([baseline] * len(time_nums), index=group.index)

    site_name = lat_to_site.get(lat, f"Lat {lat:.2f}")
    line, = ax.plot_date(time_nums, y_values, '-', label=site_name, markersize=2)

    y_tick_positions.append(baseline)
    y_tick_labels.append(f"{lat:.2f}")

    if len(time_nums) > 0 and pd.notna(min_da) and pd.notna(max_da):
        last_time_num = time_nums[-1]
        annotation_x_pos = last_time_num + annotation_x_offset_days
        annotation_y_pos = baseline

        annotation_text = f"DA: {min_da:.1f}-{max_da:.1f}"
        ax.text(annotation_x_pos, annotation_y_pos, annotation_text,
                fontsize=7, color=line.get_color(),
                verticalalignment='center', horizontalalignment='left')

# --- Axis Formatting ---
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate(rotation=30, ha='right')

ax.set_xlabel("Date")
ax.set_ylabel("Latitude(°N)")
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(y_tick_labels)

if not df['Date'].empty and not df['Date'].isnull().all():
    max_date = df['Date'].max()
    min_date = df['Date'].min()
    if pd.notna(max_date) and pd.notna(min_date):
        date_range_days = (max_date - min_date).days
        extra_space_factor = 0.06
        min_extra_days = annotation_x_offset_days * 1.5
        extra_space_days = max(min_extra_days, date_range_days * extra_space_factor)
        # Ensure enough space for annotations plus some padding
        required_right_limit = mdates.date2num(max_date) + annotation_x_offset_days * 2 # Give space for annotation text width
        ax.set_xlim(right=max(required_right_limit, mdates.date2num(max_date) + extra_space_days))


ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

ax.set_title("DA Levels Variation by Site/Latitude Over Time")

# --- MODIFICATION HERE: Increase the first value in bbox_to_anchor ---
# Original: bbox_to_anchor=(1.02, 1)
# New value (e.g., 1.05) moves it further right. Adjust as needed.
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=9)

# --- MODIFICATION HERE: Adjust the 'right' value in rect if needed ---
# Original: rect=[0, 0, 0.88, 1]
# Decrease the 'right' value (e.g., 0.85) to make more room for the legend on the right.
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjusted from 0.88

plt.savefig("data-visualizations/waterfall_plot.pdf", format="pdf", bbox_inches='tight')
plt.show()