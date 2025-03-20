import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read CSV data
df = pd.read_csv("data-visualizations/final_output.csv")

# Convert the 'Date' column to datetime (adjust format if necessary)
df['Date'] = pd.to_datetime(df['Date'])

# Create a mapping from latitude to its corresponding site (assumes one site per unique lat)
lat_to_site = df.groupby('lat')['Site'].first().to_dict()

# Sort unique latitudes in descending order (largest to smallest)
unique_lats = sorted(df['lat'].unique(), reverse=True)

plt.figure(figsize=(10, 7))

# Adjust these factors:
scaling_factor_da = 0.1   # Reduce the scale of DA Levels
baseline_multiplier = 15  # Increase the scale of latitude for greater vertical spacing

# Plot each latitude group separately with its own vertical baseline
for lat in unique_lats:
    group = df[df['lat'] == lat].sort_values(by='Date')
    
    # Convert Date to matplotlib's numeric format
    time_nums = mdates.date2num(group['Date'])
    
    # Extract DA Levels values
    da_values = group['DA Levels']
    
    # The baseline is the latitude multiplied by the baseline_multiplier.
    baseline = lat * baseline_multiplier
    y_values = baseline + scaling_factor_da * da_values
    
    # Use the site name from the mapping instead of the latitude value
    plt.plot_date(time_nums, y_values, '-', label=lat_to_site[lat])

# Format the x-axis to display dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# Label the axes
plt.xlabel("Date")
plt.ylabel("Scaled Latitude + Scaled DA Levels (Latitude X 10 + DA Levels X 0.1)")

# Optionally, add horizontal dashed lines at each latitude baseline for reference
for lat in unique_lats:
    plt.axhline(y=lat * baseline_multiplier, color='gray', linestyle='--', linewidth=0.5)

# Adjust the legend so it spans the full width and appears on two rows
# Change 'ncol=4' as needed to fit your number of sites
plt.legend(loc='lower center', bbox_to_anchor=(0, -0.3, 1, 0.3), mode="expand", ncol=4)

# Save the figure as a vector PDF
plt.savefig("data-visualizations/waterfall_plot.pdf", format="pdf")

plt.show()