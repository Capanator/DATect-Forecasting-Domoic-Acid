import pandas as pd
import numpy as np
import json
import os
import requests
import tempfile
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import traceback # Import traceback for detailed error printing

# Suppress specific warnings if needed
# warnings.filterwarnings("ignore", category=FutureWarning)

def process_dataset(url, data_type, site):
    """
    Downloads satellite data, averages values for each unique original timestamp
    within the queried spatial box, and returns the result. Assumes the default
    determined variable name is correct.

    Args:
        url (str): The URL to the NetCDF dataset.
        data_type (str): The type of data (e.g., 'temperature', 'chlorophyll').
        site (str): The site identifier.

    Returns:
        pd.DataFrame or None: A DataFrame with averaged data ('timestamp', 'site',
                              'data_type', 'value') for each original time point,
                              or None if processing fails or no valid data is found.
    """
    print(f"\nProcessing {data_type} for {site}...")
    print(f"URL: {url}")

    # --- Determine data variable name based on data_type/url ---
    # Simplified: No possible_vars or fallback logic.
    data_var = None

    if 'chlorophyll-anom' in data_type or 'chla_anomaly' in url:
        data_var = 'chla_anomaly'
    elif 'temperature-anom' in data_type or 'sst_anomaly' in url:
        data_var = 'sst_anomaly'
    elif 'chlorophyll' in data_type or 'chla' in url:
        data_var = 'chlorophyll' # Assumes this is the correct name in the NetCDF
    elif 'temperature' in data_type or 'sst' in url:
        data_var = 'sst' # Assumes this is the correct name in the NetCDF
    elif 'par' in data_type or 'par' in url:
        data_var = 'par'
    elif 'fluorescence' in data_type or 'fluorescence' in url:
        data_var = 'fluorescence' # Assumes this is the correct name in the NetCDF
    else:
        print(f"Warning: Could not determine data variable from data_type='{data_type}' or url='{url}'. Skipping.")
        return None

    if data_var is None:
         # This case should theoretically not be reached if the else above returns None
         print(f"Error: data_var is None after checks for data_type='{data_type}'. Logic error.")
         return None

    print(f"Using determined target variable: '{data_var}'")

    tmp_path = None
    try:
        # --- Download the NetCDF file ---
        try:
            print(f"Attempting download from {url}")
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            print("Download successful.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download data: {e}")
            return None

        # --- Save to temporary file ---
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        print(f"Data saved to temporary file: {tmp_path}")

        # --- Open and process the dataset ---
        with xr.open_dataset(tmp_path) as ds:
            print(f"Dataset opened. Variables: {list(ds.data_vars)}")

            # --- Check if the determined data variable exists ---
            # Simplified check: only look for data_var directly
            if data_var not in ds.data_vars:
                print(f"Error: Determined data variable '{data_var}' not found in dataset variables: {list(ds.data_vars)}")
                if data_var in ds.coords:
                     print(f"Note: '{data_var}' exists as a coordinate variable.")
                return None # Exit if the exact variable name isn't found

            print(f"Confirmed data variable exists: '{data_var}'")

            # --- Convert to DataFrame ---
            # Use data_var directly
            df = ds[data_var].to_dataframe().reset_index()

            # --- Clean data ---
            initial_rows = len(df)
            # Identify the actual column name for values, might be data_var or renamed by to_dataframe
            value_col_name = data_var
            if value_col_name not in df.columns:
                 value_col_candidates = [col for col in df.columns if col not in ['time', 'lat', 'lon', 'latitude', 'longitude', 'depth', 'altitude']]
                 if len(value_col_candidates) == 1:
                      value_col_name = value_col_candidates[0]
                      print(f"Value column identified as '{value_col_name}' after to_dataframe().")
                 else:
                      print(f"Error: Cannot uniquely identify the data value column after to_dataframe(). Candidates: {value_col_candidates}. All columns: {df.columns}")
                      return None

            # Drop rows where the identified value column is NaN
            df = df.dropna(subset=[value_col_name])
            if df.empty:
                print(f"No valid data found for {data_type} at {site} after removing NaN values (started with {initial_rows} rows).")
                return None
            print(f"Removed NaN values. Kept {len(df)} out of {initial_rows} rows.")

            # --- Time processing ---
            if 'time' not in df.columns:
                 print(f"Error: 'time' column not found in DataFrame. Columns: {df.columns}")
                 return None
            df['time'] = pd.to_datetime(df['time'])

            # --- Group by original timestamp and average ---
            averaged_df = df.groupby('time')[value_col_name].mean().reset_index()
            print(f"Averaged data over spatial dimensions for each unique timestamp. Result has {len(averaged_df)} time points.")

            # --- Add metadata and finalize ---
            averaged_df['site'] = site
            averaged_df['data_type'] = data_type

            # Rename columns for consistency
            averaged_df = averaged_df.rename(columns={
                'time': 'timestamp',
                value_col_name: 'value' # Use the identified value column name here
            })

            print(f"Successfully processed {data_type} for {site}.")
            return averaged_df[['timestamp', 'site', 'data_type', 'value']]

    except xr.backends.plugins.BackendError as e:
         print(f"Xarray Error opening {tmp_path} from {url}: {e}. File might be corrupted or invalid.")
         return None
    except FileNotFoundError:
        print(f"Error: Temporary file {tmp_path} not found.")
        return None
    except KeyError as e:
         print(f"Error: A required key/column was not found processing data from {url}: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred processing {url}: {e}")
        traceback.print_exc()
        return None
    finally:
        # --- Clean up the temporary file ---
        if tmp_path:
             try:
                 if os.path.exists(tmp_path):
                      os.unlink(tmp_path)
             except OSError as e:
                  print(f"Warning: Could not remove temporary file {tmp_path}: {e}")
             except Exception as e:
                  print(f"Warning: An unexpected error occurred during temporary file cleanup: {e}")


def main():
    # Load the satellite metadata
    metadata_file = 'satellite-metadata.json'
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Successfully loaded metadata from {metadata_file}")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {metadata_file}: {e}")
        return

    if 'end_date' not in metadata:
        print("Error: 'end_date' key missing from metadata file.")
        return

    valid_data_list = []
    tasks = []
    for data_type, sites in metadata.items():
        if data_type == 'end_date': continue
        if not isinstance(sites, dict):
             print(f"Warning: Skipping '{data_type}': expected a dictionary of sites, got {type(sites)}.")
             continue
        for site, url in sites.items():
             if not isinstance(url, str):
                  print(f"Warning: Skipping site '{site}' under '{data_type}': URL is not a string ({type(url)}).")
                  continue
             try:
                  processed_url = url.replace('{end_date}', metadata['end_date'])
                  tasks.append((data_type, site, processed_url))
             except Exception as e:
                  print(f"Error processing URL for {data_type}/{site}: {e}. Original URL: {url}")

    if not tasks:
         print("No valid tasks found in the metadata.")
         return

    # Process datasets
    for data_type, site, url in tqdm(tasks, desc="Processing datasets"):
        result = process_dataset(url, data_type, site)
        if result is not None and not result.empty:
            valid_data_list.append(result)

    # Combine all valid results
    if valid_data_list:
        try:
            valid_combined_df = pd.concat(valid_data_list, ignore_index=True)
            print(f"\nCombined data from {len(valid_data_list)} successful tasks into DataFrame with {len(valid_combined_df)} rows.")

            # --- Format Timestamp Column ---
            # Ensure timestamp column is datetime
            valid_combined_df['timestamp'] = pd.to_datetime(valid_combined_df['timestamp'])

            # Make timezone-aware (assume UTC if naive) and convert to UTC
            if valid_combined_df['timestamp'].dt.tz is None:
                valid_combined_df['timestamp'] = valid_combined_df['timestamp'].dt.tz_localize('UTC')
            else:
                valid_combined_df['timestamp'] = valid_combined_df['timestamp'].dt.tz_convert('UTC')

            # Format to ISO 8601 string with milliseconds using .apply()
            valid_combined_df['timestamp'] = valid_combined_df['timestamp'].apply(lambda x: x.isoformat(timespec='milliseconds')) # <-- FIXED LINE

            # Replace the timezone offset with 'Z' for the desired format
            valid_combined_df['timestamp'] = valid_combined_df['timestamp'].str.replace('+00:00', 'Z', regex=False)
            print("Formatted 'timestamp' column to ISO 8601 string (e.g., YYYY-MM-DDTHH:MM:SS.fffZ)")

            # --- Save to Parquet ---
            valid_output_file = 'satellite_data_averaged_by_timestamp.parquet'
            valid_combined_df.to_parquet(valid_output_file, index=False)

            print("-" * 30)
            print(f"Processing Complete.")
            print(f"Averaged data saved to {valid_output_file}")
            print(f"Processed {len(tasks)} total URL tasks.")
            print(f"Final DataFrame has {len(valid_combined_df)} rows, representing unique site-timestamp combinations.")
            print(f"Columns: {valid_combined_df.columns.tolist()}")
            print("-" * 30)
            print("\nSample of the final data:")
            print(valid_combined_df[['timestamp', 'site', 'data_type', 'value']].head())
            print("\nData Info:")
            valid_combined_df.info()

        except Exception as e:
             print(f"\nError during final data combination, formatting, or saving: {e}")
             traceback.print_exc()
    else:
        print("-" * 30)
        print("No valid data was processed successfully from any dataset.")
        print("-" * 30)

# Make sure the rest of your script (imports, process_dataset function) is included
# when you run this corrected main function.

if __name__ == "__main__":
    # Assuming process_dataset is defined above this point
    main()
