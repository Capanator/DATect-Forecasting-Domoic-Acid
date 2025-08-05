# Unified Domoic Acid Forecasting Pipeline

This unified pipeline replaces the fragile, monolithic `future-forecasts.py` and `past-forecasts-final.py` scripts with a properly engineered, modular system that prevents data leakage and follows software engineering best practices.

## Key Improvements

### 1. **Eliminated Data Leakage**
- **Fixed DA Category Creation**: Categories are now created within individual training sets, not globally across the entire dataset
- **Proper Lag Features**: Lag features are created only from historical data available at prediction time
- **Clean Train/Test Splits**: No information from future observations leaks into training data

### 2. **Modular Architecture**
- **`data_processing.py`**: Clean data handling and feature engineering
- **`modeling.py`**: Complete sklearn pipelines with proper preprocessing
- **`app.py`**: Unified dashboard for both evaluation and forecasting

### 3. **Proper sklearn Pipelines**
- All preprocessing (imputation, scaling, feature creation) happens within pipelines
- Prevents leakage between training and testing phases
- Ensures reproducible and consistent transformations

## Quick Start

### Installation Requirements
```bash
pip install pandas numpy scikit-learn plotly dash tqdm joblib pyarrow
```

### Running the Application
```bash
python3 app.py
```

Navigate to http://localhost:8080 to access the unified dashboard.

### Testing the Pipeline
```bash
python3 test_pipeline.py
```

## File Structure

```
├── data_processing.py      # Clean data processing without leakage
├── modeling.py            # sklearn pipelines and forecasting models  
├── app.py                # Unified dashboard application
├── test_pipeline.py      # Comprehensive testing suite
└── README_UNIFIED_PIPELINE.md
```

## Architecture Overview

### DataProcessor (`data_processing.py`)
- **`load_data()`**: Basic data loading and preparation
- **`get_train_test_split()`**: Clean temporal splits by site and anchor date
- **`prepare_training_data()`**: Creates features and targets for training (prevents leakage)
- **`prepare_forecast_data()`**: Prepares forecast data using only historical information
- **`create_da_categories()`**: Creates categorical targets from continuous values

### TimeSeriesForecaster (`modeling.py`)
- Complete sklearn pipelines for both regression and classification
- Automatic feature preprocessing (imputation, scaling)
- Supports multiple model types: Random Forest, Gradient Boosting, Linear models
- Prevents data leakage through proper pipeline structure

### UnifiedForecastingApp (`app.py`)
- **Future Forecasting Mode**: Generate predictions for specific dates/sites
- **Past Evaluation Mode**: Comprehensive retrospective evaluation with random anchors
- Interactive dashboard with visualizations
- Supports both point predictions and uncertainty quantification

## Usage Examples

### 1. Future Forecasting
```python
from data_processing import DataProcessor
from modeling import TimeSeriesForecaster

# Load and prepare data
processor = DataProcessor()
data = processor.load_data('final_output.parquet')

# Get training data up to anchor date
train_data, _ = processor.get_train_test_split(data, 'SiteName', anchor_date)
train_prepared = processor.prepare_training_data(train_data)
train_filtered = processor.filter_features(train_prepared)

# Create and train forecaster
forecaster = TimeSeriesForecaster(model_type='random_forest', task='regression')
forecaster.fit(train_filtered, train_prepared['da'])

# Make prediction
forecast_data = processor.prepare_forecast_data(forecast_row, train_prepared)
forecast_filtered = processor.filter_features(forecast_data)
prediction = forecaster.predict(forecast_filtered)
```

### 2. Model Optimization
```python
from modeling import ModelOptimizer

optimizer = ModelOptimizer(n_splits=5)
best_params = optimizer.optimize(forecaster, X_train, y_train, scoring='r2')
```

### 3. Uncertainty Quantification
```python
from modeling import QuantileForecaster

quantile_forecaster = QuantileForecaster(quantiles=[0.05, 0.5, 0.95])
quantile_forecaster.fit(X_train, y_train)
uncertainty_preds = quantile_forecaster.predict(X_test)
```

## Key Features

### Data Leakage Prevention
- ✅ DA categories created from training data only
- ✅ Lag features use only historical information
- ✅ No global statistics computed across train/test splits
- ✅ Proper time series cross-validation

### Robust Pipeline Design
- ✅ All preprocessing encapsulated in sklearn pipelines
- ✅ Automatic handling of missing values
- ✅ Consistent feature scaling and transformation
- ✅ Reproducible results with random seeds

### Comprehensive Evaluation
- ✅ Random anchor point evaluation
- ✅ Site-specific and overall performance metrics
- ✅ Both regression (R², MAE, RMSE) and classification (accuracy) metrics
- ✅ Uncertainty quantification with prediction intervals

### Interactive Dashboard
- ✅ Toggle between past evaluation and future forecasting
- ✅ Site and date selection
- ✅ Model comparison (Random Forest vs Linear)
- ✅ Real-time visualization of results

## Performance

The unified pipeline is designed for efficiency:
- Parallel processing for evaluation runs
- Optimized data structures and operations
- Configurable evaluation sample sizes
- Proper memory management

## Migration from Old Scripts

### Replacing `future-forecasts.py`
The new system provides the same functionality through the **Future Forecasting Mode** in the unified dashboard, but with proper data leakage prevention.

### Replacing `past-forecasts-final.py` 
The **Past Evaluation Mode** provides comprehensive retrospective evaluation with the same random anchor approach, but using clean pipelines.

## Configuration

Key configuration options in `app.py`:
```python
config = {
    'random_state': 42,           # Reproducibility
    'n_evaluation_samples': 200,  # Number of evaluation points
    'min_training_points': 10,    # Minimum training data required
    'include_lags': True,         # Enable lag features
    'port': 8080                  # Dashboard port
}
```

## Validation

The pipeline includes comprehensive validation:
- Unit tests for all major components
- Integration tests with real data
- Data leakage detection
- Performance benchmarking

Run `python3 test_pipeline.py` to verify everything works correctly.

## Next Steps

1. **Run the unified pipeline**: `python3 app.py`
2. **Compare results**: Verify predictions match expectations
3. **Customize models**: Add new model types or features as needed
4. **Scale up**: Increase evaluation samples for more robust statistics
5. **Deploy**: Set up production deployment if desired

The unified pipeline addresses all the critical issues mentioned in the feedback while providing a much more maintainable and robust forecasting system.