# =============================================================================
# DATECT FORECASTING CONFIGURATION FILE
# =============================================================================
# Configuration for Domoic Acid (DA) forecasting system
# This file contains all settings for data processing, modeling, and dashboards

# =============================================================================
# DATA SOURCES AND PATHS
# =============================================================================

# Original DA measurement files - CSV files containing historical DA toxin measurements
ORIGINAL_DA_FILES = {
    "twin-harbors": "./da-input/twin-harbors-da.csv",
    "long-beach": "./da-input/long-beach-da.csv",
    "quinault": "./da-input/quinault-da.csv",
    "kalaloch": "./da-input/kalaloch-da.csv",
    "copalis": "./da-input/copalis-da.csv",
    "newport": "./da-input/newport-da.csv",
    "gold-beach": "./da-input/gold-beach-da.csv",
    "coos-bay": "./da-input/coos-bay-da.csv",
    "clatsop-beach": "./da-input/clatsop-beach-da.csv",
    "cannon-beach": "./da-input/cannon-beach-da.csv"
}

# Original PN measurement files - CSV files containing Pseudo-nitzschia cell count data
ORIGINAL_PN_FILES = {
    "gold-beach-pn": "./pn-input/gold-beach-pn.csv",
    "coos-bay-pn": "./pn-input/coos-bay-pn.csv",
    "newport-pn": "./pn-input/newport-pn.csv",
    "clatsop-beach-pn": "./pn-input/clatsop-beach-pn.csv",
    "cannon-beach-pn": "./pn-input/cannon-beach-pn.csv",
    "kalaloch-pn": "./pn-input/kalaloch-pn.csv",
    "copalis-pn": "./pn-input/copalis-pn.csv",
    "long-beach-pn": "./pn-input/long-beach-pn.csv",
    "twin-harbors-pn": "./pn-input/twin-harbors-pn.csv",
    "quinault-pn": "./pn-input/quinault-pn.csv"
}

# =============================================================================
# ENVIRONMENTAL DATA URLS
# =============================================================================

# Climate indices from NOAA ERDDAP servers
PDO_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.nc?time%2CPDO&time%3E=2002-05-01&time%3C=2025-01-01T00%3A00%3A00Z"
ONI_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_ONI.nc?time%2CONI&time%3E=2002-05-01&time%3C=2024-12-01T00%3A00%3A00Z"

# Biologically Effective Upwelling Transport Index
BEUTI_URL = "https://oceanview.pfeg.noaa.gov/erddap/griddap/erdBEUTIdaily.nc?BEUTI%5B(2002-05-01):1:(2024-11-28T00:00:00Z)%5D%5B(42):1:(47.0)%5D"

# USGS streamflow data for Columbia River
STREAMFLOW_URL = "https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site=14246900&agencyCd=USGS&statCd=00003&parameterCd=00060&startDT=2002-06-01&endDT=2025-02-22"

# =============================================================================
# MONITORING SITES
# =============================================================================

# Site coordinates [latitude, longitude] for Pacific Coast monitoring locations
SITES = {
    "Kalaloch": [47.58597, -124.37914],
    "Quinault": [47.28439, -124.23612],
    "Copalis": [47.10565, -124.1805],
    "Twin Harbors": [46.79202, -124.09969],
    "Long Beach": [46.55835, -124.06088],
    "Clatsop Beach": [46.028889, -123.917222],
    "Cannon Beach": [45.881944, -123.959444],
    "Newport": [44.6, -124.05],
    "Coos Bay": [43.376389, -124.237222],
    "Gold Beach": [42.377222, -124.414167]
}

# =============================================================================
# DATE RANGES
# =============================================================================

# Primary data processing date range
START_DATE = "2003-01-01"
END_DATE = "2023-12-31"

# Output file path for processed dataset
FINAL_OUTPUT_PATH = "final_output.parquet"

# =============================================================================
# SATELLITE DATA CONFIGURATION
# =============================================================================

# MODIS satellite data URLs for oceanographic parameters
# Format: {start_date} and {end_date} will be replaced with actual dates
# {anom_start_date} is used for anomaly datasets

SATELLITE_DATA = {
    # MODIS Chlorophyll-a concentration (8-day composite)
    "modis-chla": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # MODIS Sea Surface Temperature (8-day composite)
    "modis-sst": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # MODIS Photosynthetically Available Radiation (8-day composite)
    "modis-par": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # MODIS Fluorescence Line Height (8-day composite)
    "modis-flur": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # MODIS Diffuse Attenuation Coefficient K490 (8-day composite)
    "modis-k490": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Chlorophyll-a Anomaly (monthly data, requires different coordinate system)
    "chla-anom": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(235.425):1:(235.625)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(235.5625):1:(235.7625)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(235.625):1:(235.825)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(235.7):1:(235.9)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(235.7375):1:(235.9375)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(235.8875):1:(236.0875)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(235.8375):1:(236.0375)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(235.55):1:(235.95)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(235.3625):1:(235.7625)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(235.1875):1:(235.5875)%5D"
    },
    
    # Sea Surface Temperature Anomaly (monthly data)
    "sst-anom": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(235.425):1:(235.625)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(235.5625):1:(235.7625)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(235.625):1:(235.825)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(235.7):1:(235.9)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(235.7375):1:(235.9375)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(235.8875):1:(236.0875)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(235.8375):1:(236.0375)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(235.55):1:(235.95)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(235.3625):1:(235.7625)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(235.1875):1:(235.5875)%5D"
    },
    
    # Satellite data date ranges
    "satellite_start_date": "2002-07-16T12:00:00Z",
    "satellite_anom_start_date": "2003-01-16T12:00:00Z",
    "satellite_end_date": "2025-01-16T12:00:00Z"
}

# =============================================================================
# FORECAST CONFIGURATION
# =============================================================================

# Operation Mode Configuration
# Controls which type of forecasting system to run
FORECAST_MODE = "retrospective"  # Options: "retrospective", "realtime"
# - "retrospective": Run historical validation with random anchor points
# - "realtime": Launch interactive dashboard for specific date/site predictions

# Task Configuration  
# Defines the prediction task type
FORECAST_TASK = "regression"  # Options: "regression", "classification"
# - "regression": Predict continuous DA levels (μg/g)
# - "classification": Predict categorical risk levels (Low/Moderate/High/Extreme)

# Model Configuration
# Specifies which machine learning algorithm to use
FORECAST_MODEL = "xgboost"  # Options: "rf", "xgboost", "stacking", "ridge", "logistic"
# - "rf": Random Forest (baseline model with good performance)
# - "xgboost": XGBoost (7.4% better than RF for regression)
# - "stacking": Stacking Ensemble (8.1% better than RF - BEST PERFORMER)
# - "ridge": Ridge Regression (linear method for regression only)
# - "logistic": Logistic Regression (linear method for classification only)


# Dashboard Configuration  
# Network port for web dashboard when running in realtime mode
DASHBOARD_PORT = 8065  # Options: Any available port (typically 8000-9000)
# - Must be available and not blocked by firewall
# - Used only in realtime mode

# Temporal Validation Settings
# These settings control data leakage prevention - CRITICAL for research validity
TEMPORAL_BUFFER_DAYS = 1  # Minimum days between training data and prediction target
SATELLITE_BUFFER_DAYS = 7  # Minimum days for satellite data temporal cutoff
CLIMATE_BUFFER_MONTHS = 2  # Minimum months for climate index reporting delays

# Model Performance Settings
MIN_TRAINING_SAMPLES = 3  # Minimum samples required to train a model
RANDOM_SEED = 42  # For reproducible results across runs

# Retrospective Evaluation Configuration
N_RANDOM_ANCHORS = 100  # Number of random anchor points for retrospective evaluation
# - Higher values: More thorough evaluation, longer runtime (recommended: 20-100)
# - Lower values: Faster evaluation, less comprehensive testing

# DA Category Thresholds (μg/g)
# Used for classification tasks and risk level assignment
DA_CATEGORY_BINS = [-float("inf"), 5, 20, 40, float("inf")]  # Bin edges matching original system
DA_CATEGORY_LABELS = [0, 1, 2, 3]  # Numeric labels for compatibility with ML models
# Category meanings:
# - 0: Low (0-5 μg/g) - safe for consumption
# - 1: Moderate (5-20 μg/g) - caution advised  
# - 2: High (20-40 μg/g) - avoid consumption
# - 3: Extreme (>40 μg/g) - health hazard