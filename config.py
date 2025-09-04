# DATect Forecasting Configuration
# Settings for data processing, modeling, and web interface

# Data Sources and Paths

# Historical DA toxin measurement files
ORIGINAL_DA_FILES = {
    "twin-harbors": "./data/raw/da-input/twin-harbors-da.csv",
    "long-beach": "./data/raw/da-input/long-beach-da.csv",
    "quinault": "./data/raw/da-input/quinault-da.csv",
    "kalaloch": "./data/raw/da-input/kalaloch-da.csv",
    "copalis": "./data/raw/da-input/copalis-da.csv",
    "newport": "./data/raw/da-input/newport-da.csv",
    "gold-beach": "./data/raw/da-input/gold-beach-da.csv",
    "coos-bay": "./data/raw/da-input/coos-bay-da.csv",
    "clatsop-beach": "./data/raw/da-input/clatsop-beach-da.csv",
    "cannon-beach": "./data/raw/da-input/cannon-beach-da.csv"
}

# Pseudo-nitzschia cell count data files
ORIGINAL_PN_FILES = {
    "gold-beach-pn": "./data/raw/pn-input/gold-beach-pn.csv",
    "coos-bay-pn": "./data/raw/pn-input/coos-bay-pn.csv",
    "newport-pn": "./data/raw/pn-input/newport-pn.csv",
    "clatsop-beach-pn": "./data/raw/pn-input/clatsop-beach-pn.csv",
    "cannon-beach-pn": "./data/raw/pn-input/cannon-beach-pn.csv",
    "kalaloch-pn": "./data/raw/pn-input/kalaloch-pn.csv",
    "copalis-pn": "./data/raw/pn-input/copalis-pn.csv",
    "long-beach-pn": "./data/raw/pn-input/long-beach-pn.csv",
    "twin-harbors-pn": "./data/raw/pn-input/twin-harbors-pn.csv",
    "quinault-pn": "./data/raw/pn-input/quinault-pn.csv"
}

# Environmental Data URLs

# Climate indices from NOAA ERDDAP
PDO_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.nc?time%2CPDO&time%3E=2002-05-01&time%3C=2025-01-01T00%3A00%3A00Z"
ONI_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_ONI.nc?time%2CONI&time%3E=2002-05-01&time%3C=2024-12-01T00%3A00%3A00Z"

# BEUTI (Biologically Effective Upwelling Transport Index)
BEUTI_URL = "https://oceanview.pfeg.noaa.gov/erddap/griddap/erdBEUTIdaily.nc?BEUTI%5B(2002-05-01):1:(2024-11-28T00:00:00Z)%5D%5B(42):1:(47.0)%5D"

# Columbia River streamflow data
STREAMFLOW_URL = "https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site=14246900&agencyCd=USGS&statCd=00003&parameterCd=00060&startDT=2002-06-01&endDT=2025-02-22"

# Monitoring Sites

# Pacific Coast monitoring site coordinates [lat, lon]
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

# Date Ranges
START_DATE = "2003-01-01"
END_DATE = "2023-12-31"

FINAL_OUTPUT_PATH = "./data/processed/final_output.parquet"

SATELLITE_CACHE_PATH = "./data/intermediate/satellite_data_intermediate.parquet"

# Satellite Data Configuration

# MODIS oceanographic data URLs with date placeholders

SATELLITE_DATA = {
    # Chlorophyll-a (8-day composite)
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
    
    # Sea Surface Temperature (8-day composite)
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
    
    # Photosynthetically Available Radiation (8-day composite)
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
    
    # Fluorescence Line Height (8-day composite)
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
    
    # Diffuse Attenuation Coefficient K490 (8-day composite)
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
    
    # Chlorophyll-a Anomaly (monthly, 0-360° longitude)
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
    
    # Sea Surface Temperature Anomaly (monthly)
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
    
    # Date ranges for satellite data
    "satellite_start_date": "2002-07-16T12:00:00Z",
    "satellite_anom_start_date": "2003-01-16T12:00:00Z",
    "satellite_end_date": "2025-01-16T12:00:00Z"
}

# Forecast Configuration

# Operation mode: "retrospective" (historical validation) or "realtime" (dashboard)
FORECAST_MODE = "retrospective"

# Task type: "regression" (continuous DA levels) or "classification" (risk categories)
FORECAST_TASK = "regression"

# ML algorithm: "xgboost" (primary) or "linear" (interpretable)
FORECAST_MODEL = "xgboost"

# Forecast Horizon Configuration
# How many weeks ahead to forecast from the data cutoff point
FORECAST_HORIZON_WEEKS = 1 
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7  # Derived days value for internal calculations


# Temporal Validation - prevents data leakage (handled by forecast horizon)

# Model Performance
MIN_TRAINING_SAMPLES = 3
RANDOM_SEED = 42

# Retrospective evaluation anchor points (higher = more thorough)
N_RANDOM_ANCHORS = 500

# Lag Feature Configuration

# Enable/disable lag features for time series modeling
USE_LAG_FEATURES = True  # Include DA lags as features (leak-safe)

# IMPORTANT: Lags are observation steps (≈ weeks), not days.
# For weekly DA, lag 1 = 1 week ago, 4 = ~1 month ago.
LAG_FEATURES = [1, 2, 3, 4] if USE_LAG_FEATURES else []

# DA Category Configuration

# Risk thresholds for classification: Low (0-5), Moderate (5-20), High (20-40), Extreme (>40 μg/g)
DA_CATEGORY_BINS = [-float("inf"), 5, 20, 40, float("inf")]
DA_CATEGORY_LABELS = [0, 1, 2, 3]

# Spike/Onset Emphasis Configuration
# Primary goal: Predict initial spike timing (crossing 20 ppm)
SPIKE_THRESHOLD_PPM = 20.0

# Weighting for regression training to emphasize high DA values
# Applied when y_train >= SPIKE_THRESHOLD_PPM
SPIKE_WEIGHT_MULT = 8.0

# Additional pre-spike weighting for samples that are below threshold but
# have a spike within the next PRE_SPIKE_WINDOW_DAYS (using only historical data
# up to the anchor date to avoid leakage). This helps the model learn the
# onset conditions that precede spikes.
PRE_SPIKE_WINDOW_DAYS = 14
PRE_SPIKE_WEIGHT_MULT = 6.0

# Extra emphasis on truly massive spikes
MASSIVE_SPIKE_THRESHOLD_PPM = 60.0
MASSIVE_SPIKE_WEIGHT_MULT = 4.0

# Naive baseline configuration used in analyses
NAIVE_BASELINE_LAG_DAYS = 7

# Onset timing evaluation tolerance
ONSET_TOLERANCE_DAYS = 3

# Onset classifier configuration
ONSET_WINDOW_DAYS = 7  # days to define "spike imminence"
ONSET_POS_WEIGHT = 6.0  # class weight multiplier for positive samples
ONSET_PROB_THRESHOLD = 0.20  # threshold to declare predicted onset

# Derived features from lags (leak-safe)
USE_DERIVED_LAG_FEATURES = True
# Derived windows use available lags; features are computed only from lag columns
DERIVED_LAG_SETTINGS = {
    "use_last3_mean": True,           # mean of da_lag_1..3 if available
    "use_weekly_change": True,        # da_lag_1 - da_lag_2 (week-over-week)
    "use_monthly_change": True,       # da_lag_1 - da_lag_4 (~month-over-month)
    "use_rising_flag": True,          # 1 if da_lag_1>da_lag_2>da_lag_3
    "use_below_streak3": True,        # count of last 3 lags below threshold
    # Safe naive anchor-snapshot features (computed without leakage)
    "use_naive_lag4": True,           # approximate monthly persistence
    "use_delta_last_vs_lag4": True,
    # Backward-compat toggles (kept true/false but ignored if related lags absent)
    "use_biweekly_change": False,
    "use_naive_lag7": False,
    "use_delta_last_vs_lag7": False
}

# XGBoost Parameter Overrides (optional)
# If provided, ModelFactory will use these instead of hard-coded defaults.
XGB_REGRESSION_PARAMS = {
    # More capacity but smoother learning
    "n_estimators": 1200,
    "max_depth": 6,
    "learning_rate": 0.05,
    # Regularization
    "reg_alpha": 0.0,
    "reg_lambda": 1.2,
    "min_child_weight": 2,
    "gamma": 0.0,
    # Subsampling
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    # Loss
    "objective": "reg:squarederror",
    # Speed/robustness
    "tree_method": "hist",
    "max_bin": 256,
}

XGB_CLASSIFICATION_PARAMS = {
    # Mirror regression stability with slightly lower depth
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.06,
    "reg_alpha": 0.0,
    "reg_lambda": 1.2,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_bin": 256,
}

# Class weights for DA category classification (0=Low, 1=Moderate, 2=High, 3=Extreme)
CLASS_WEIGHTS = {0: 1.0, 1: 1.4, 2: 2.5, 3: 3.5}

# Linear regression variant (for 'linear' model type)
LINEAR_REGRESSION_TYPE = 'ridge'  # 'ridge' or 'ols'
RIDGE_ALPHA = 1.0

# Experimental: Tweedie objective for heavy-tailed nonnegative targets
USE_TWEEDIE_REGRESSION = False
TWEEDIE_VARIANCE_POWER = 1.3  # in (1,2); 1.3 handles zeros + heavy tail
