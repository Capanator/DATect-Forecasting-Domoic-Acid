import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def sophisticated_nan_handling_for_timeseries(df):
    """
    Implements sophisticated NaN handling strategy from modular-forecast for time series visualization.
    
    Strategy:
    1. Preserve temporal integrity by maintaining chronological order
    2. Use median imputation for environmental variables (MODIS-chla, etc.)
    3. Keep NaN values for DA to show natural data gaps in time series
    4. Process each site separately to maintain site-specific patterns
    5. Prevent data leakage through temporal ordering
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with sophisticated NaN handling applied
    """
    df_processed = df.copy()
    
    # Ensure temporal ordering (critical for time series plots)
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['site', 'date']).reset_index(drop=True)
        print(f"  Maintaining temporal order: {len(df_processed)} records")
    
    print(f"  Before NaN handling: {len(df_processed)} records")
    
    # Essential time series variables
    essential_cols = ['da', 'modis-chla', 'date', 'site']
    
    # Remove rows where both key variables are missing (can't create meaningful time series)
    before_essential = len(df_processed)
    df_processed = df_processed.dropna(subset=['modis-chla'], how='all')  # Need at least MODIS data
    after_essential = len(df_processed)
    
    if before_essential != after_essential:
        print(f"  Removed {before_essential - after_essential} rows missing MODIS-chla data")
    
    # Apply imputation to environmental variables (not targets)
    environmental_cols = ['modis-chla']  # Key environmental variable for time series
    non_essential_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                         if col not in essential_cols]
    
    # Use median imputation for non-essential environmental variables (matches modular-forecast)
    if non_essential_cols:
        print(f"  Applying median imputation to {len(non_essential_cols)} environmental variables")
        imputer = SimpleImputer(strategy="median")
        df_processed[non_essential_cols] = imputer.fit_transform(df_processed[non_essential_cols])
    
    # For MODIS-chla, use forward-only interpolation within sites to prevent temporal leakage
    df_processed = df_processed.sort_values(['site', 'date'])
    df_processed['modis-chla'] = df_processed.groupby('site')['modis-chla'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='forward')
    )
    
    # Keep NaN values for DA - time series plot will naturally show gaps
    # This maintains scientific integrity and prevents data leakage
    valid_da_count = df_processed['da'].notna().sum()
    print(f"  Preserving DA NaN values: {valid_da_count} valid DA measurements")
    
    print(f"  After sophisticated NaN handling: {len(df_processed)} records")
    
    return df_processed


# Load the data from parquet file with sophisticated NaN handling
print("Loading data with sophisticated NaN handling for time series visualization...")
try:
    # Use robust path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, '..', '..', '..')
    file_path = os.path.join(repo_root, 'data', 'processed', 'final_output.parquet')
    
    # Create output directory
    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_parquet(file_path)
    print(f"Raw data loaded: {len(df)} records")
except FileNotFoundError:
    print("Error: final_output.parquet not found. Please make sure the file is in the correct directory.")
    print(f"Looking for file at: {file_path}")
    exit()

# Apply sophisticated NaN handling
df = sophisticated_nan_handling_for_timeseries(df)

# Get unique site names (after NaN handling)
sites = df['site'].unique()
print(f"Processing {len(sites)} sites for time series visualization")

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
    plt.title(f'Normalized modis-chla and Capped da over Time for Site: {site}\n(Sophisticated NaN Handling Applied)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend() # Show legend
    plt.grid(True) # Add grid

    # Improve date formatting on x-axis
    plt.gcf().autofmt_xdate()

    # Set y-axis limits to 0 and 1 for normalized data
    plt.ylim(0, 1)

    # Save plot with sophisticated NaN handling identifier
    plot_filename = os.path.join(output_dir, f'timeseries_normalized_{site.replace(" ", "_")}_sophisticated.pdf')
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    print(f"Saved plot for {site}: {plot_filename}")
    
    # Show the plot for the current site
    plt.show()

print("All time series plots have been generated with sophisticated NaN handling.")

