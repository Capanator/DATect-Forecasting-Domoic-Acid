import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler
import numpy as np # Import numpy for capping values

# Load the data from the CSV file
try:
    df = pd.read_csv('final_output.csv')
except FileNotFoundError:
    print("Error: final_output.csv not found. Please make sure the file is in the correct directory.")
    exit()

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Get unique site names
sites = df['site'].unique()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Iterate through each site and create a plot
for site in sites:
    # Filter data for the current site
    site_df = df[df['site'] == site].copy() # Create a copy to avoid SettingWithCopyWarning

    # Normalize 'modis-chla' for the current site
    site_df['modis-chla_normalized'] = scaler.fit_transform(site_df[['modis-chla']])

    # Normalize 'da' for the current site with a cap at 50
    # Apply the cap: values > 50 become 50
    site_df['da_capped'] = np.minimum(site_df['da'], 80)
    # Normalize the capped 'da' values
    site_df['da_normalized'] = scaler.fit_transform(site_df[['da_capped']])

    # Create the plot
    plt.figure(figsize=(12, 6)) # Set the figure size
    plt.plot(site_df['date'], site_df['modis-chla_normalized'], label='modis-chla (Normalized)')
    plt.plot(site_df['date'], site_df['da_normalized'], label='da (Normalized, Capped at 50)')

    # Add title and labels
    plt.title(f'Normalized modis-chla and Capped da over Time for Site: {site}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend() # Show legend
    plt.grid(True) # Add grid

    # Improve date formatting on x-axis
    plt.gcf().autofmt_xdate()

    # Set y-axis limits to 0 and 1 for normalized data
    plt.ylim(0, 1)

    # Show the plot for the current site
    plt.show()

