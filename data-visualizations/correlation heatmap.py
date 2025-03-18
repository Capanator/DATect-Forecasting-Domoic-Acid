import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import matplotlib.dates as mdates

# Read CSV data
df = pd.read_csv("final_output.csv")

# Convert the time column to datetime (adjust format if necessary)
df['time'] = pd.to_datetime(df['time'])

# Group data by latitude
groups = df.groupby('latitude')

# Create 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a line for each latitude group to simulate a waterfall plot
for lat, group in groups:
    # Sort by time for each group
    group = group.sort_values(by='time')
    
    # Convert datetime to matplotlib’s numeric date format for proper scaling
    time_nums = mdates.date2num(group['time'])
    
    # Extract y values (da)
    da_values = group['da']
    
    # For the z-axis, use the constant latitude for each group
    lat_values = [lat] * len(group)
    
    ax.plot(time_nums, da_values, lat_values, label=f'Lat {lat}')

# Set axis labels
ax.set_xlabel("Time")
ax.set_ylabel("da")
ax.set_zlabel("Latitude")

# Format the x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

# Optional: add a legend (can be omitted if there are many latitudes)
ax.legend()

# Save the plot as a vector PDF
plt.savefig("waterfall_plot.pdf", format="pdf")

plt.show()