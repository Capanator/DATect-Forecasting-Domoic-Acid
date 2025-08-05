"""
Configuration file for the unified DATect forecasting pipeline.
All configurable parameters are centralized here for easy management.
"""

# ================================
# MAIN PIPELINE CONFIGURATION
# ================================

# Data Configuration
DATA_FILE = 'final_output.parquet'
RANDOM_STATE = 42

# ================================
# FEATURE ENGINEERING
# ================================

# Lag Features Configuration
# Set to True to include lag features (da_lag_1, da_lag_2, da_lag_3)
# Set to False to disable lag features for faster processing or testing
INCLUDE_LAG_FEATURES = True

# Lag periods to create (only used if INCLUDE_LAG_FEATURES = True)
LAG_PERIODS = [1, 2, 3]

# Seasonal Features
INCLUDE_SEASONAL_FEATURES = True

# ================================
# PAST EVALUATION CONFIGURATION
# ================================

# Number of random anchor points per site for evaluation
# Original past-forecasts-final.py uses 500 per site
N_RANDOM_ANCHORS_PER_SITE = 50

# Minimum training points required for a valid forecast
MIN_TRAINING_POINTS = 10

# Parallel Processing (always on, like original)
N_JOBS_EVAL = -1  # Matches original CONFIG["N_JOBS_EVAL"]

# GridSearchCV removed - using fixed parameters for simplicity and speed

# Date range for random anchor generation (matching original)
MIN_TEST_DATE = '2008-01-01'  # Matches original CONFIG["MIN_TEST_DATE"]

# ================================
# FUTURE FORECASTING CONFIGURATION
# ================================

# Default model parameters (when not using GridSearch)
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': 1  # Match original n_jobs=1 for individual models
}

DEFAULT_GB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

# Quantile regression settings
QUANTILE_LEVELS = [0.05, 0.5, 0.95]

# ================================
# DASH APP CONFIGURATION
# ================================

# App settings (matching original)
APP_HOST = '127.0.0.1'
APP_PORT = 8071  # Matches original CONFIG["PORT"]
DEBUG_MODE = False  # Matches original debug=False

# UI Configuration
AVAILABLE_MODELS = ['random_forest', 'gradient_boosting']
AVAILABLE_METRICS = ['regression', 'classification']

# Category labels for classification
CATEGORY_LABELS = ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)']

# ================================
# PERFORMANCE SETTINGS
# ================================

# Memory management
MAX_WORKERS = 4  # For parallel processing when N_JOBS_EVAL is not -1

# Caching
ENABLE_RESULT_CACHING = False
CACHE_DIRECTORY = './cache'

# ================================
# LOGGING CONFIGURATION
# ================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'datect_pipeline.log'

# ================================
# VALIDATION SETTINGS
# ================================

# Data validation
MIN_DATA_POINTS_PER_SITE = 50
MAX_MISSING_VALUES_RATIO = 0.3

# Model validation
MIN_R2_THRESHOLD = 0.0  # Minimum R² to consider a model valid
MIN_ACCURACY_THRESHOLD = 0.0  # Minimum accuracy for classification

# ================================
# EXPORT SETTINGS
# ================================

# Results export
EXPORT_INDIVIDUAL_RESULTS = True
EXPORT_AGGREGATED_RESULTS = True
RESULTS_EXPORT_FORMAT = 'parquet'  # Options: 'parquet', 'csv', 'json'

# Visualization export
EXPORT_PLOTS = False
PLOT_EXPORT_FORMAT = 'png'  # Options: 'png', 'svg', 'pdf'
PLOT_EXPORT_DPI = 300