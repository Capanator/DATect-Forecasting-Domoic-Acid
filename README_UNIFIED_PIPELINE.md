# Unified Domoic Acid Forecasting Pipeline

This unified pipeline replaces both `past-forecasts-final.py` and `future-forecasts.py` with a proper, modular, leak-free forecasting system.

## Key Improvements

### 1. **Eliminates Data Leakage**
- **Problem Fixed**: The original code created `da-category` using `pd.cut()` on the entire dataset before train/test splits, causing look-ahead bias
- **Solution**: Category encoding now happens within sklearn pipelines on training data only
- **Problem Fixed**: Lag features were created on the entire dataframe at once
- **Solution**: Lag features are now created per forecast using proper time series splits

### 2. **Proper sklearn Pipeline Implementation**
- **Problem Fixed**: Original code only used Pipeline for imputation/scaling after manual feature engineering
- **Solution**: All preprocessing steps (feature creation, imputation, scaling) are now encapsulated in sklearn Pipelines
- **Benefit**: Prevents leakage and makes the process reproducible

### 3. **Modular Architecture**
- **Problem Fixed**: Monolithic files with mixed concerns
- **Solution**: Separated into logical modules:
  - `pipeline/feature_engineering.py` - Feature creation transformers
  - `pipeline/models.py` - Model definitions and training
  - `pipeline/data_splitter.py` - Time series splitting without leakage
  - `pipeline/evaluation.py` - Performance evaluation
  - `pipeline/forecast_pipeline.py` - Main orchestration

### 4. **Unified Interface**
- **Problem Fixed**: Two separate applications with different interfaces
- **Solution**: Single application with command-line arguments for different modes

## Usage

### Retrospective Evaluation Mode
```bash
python unified_forecast_app.py --mode retrospective --anchors 500
```
- Generates random anchor points for evaluation
- Runs time series cross-validation
- Provides comprehensive performance metrics
- Launches interactive dashboard on port 8071

### Real-time Forecasting Mode
```bash
python unified_forecast_app.py --mode realtime
```
- Interactive forecasting for specific dates/sites
- Provides point predictions and uncertainty intervals
- Shows quantile predictions and classification probabilities
- Launches dashboard on port 8065

### Command Line Options
```bash
python unified_forecast_app.py \
    --mode {retrospective|realtime} \
    --data final_output.parquet \
    --port 8071 \
    --anchors 500 \
    --min-test-date 2008-01-01 \
    --enable-lag-features
```

## Architecture

### Data Flow (Leak-Free)
1. **Data Loading**: Raw data loaded from parquet file
2. **Pipeline Fitting**: Feature transformers fitted on full dataset (learns scalers, encoders only)
3. **Per-Forecast Processing**:
   - Time series split (training < forecast date)
   - Feature pipeline applied to training data only
   - Models trained on processed training features
   - Predictions made on forecast data

### Feature Engineering Pipeline
```python
Pipeline([
    ('cleaner', DataCleaner()),
    ('temporal', TemporalFeatures()),
    ('category_encoder', CategoryEncoder()),  # Fits on training data only
    ('lag_features', LagFeatures()),           # Creates lags per forecast
    ('feature_selector', FeatureSelector())
])
```

### Model Training
- **Regression**: Random Forest + Quantile Gradient Boosting
- **Classification**: Random Forest with class probabilities
- **Validation**: TimeSeriesSplit for hyperparameter tuning
- **Training**: Per-forecast model fitting (no global models)

## Key Classes

### `DAForecastPipeline`
Main orchestration class that handles:
- Feature pipeline creation and fitting
- Single forecast generation
- Retrospective evaluation with random anchors
- Performance evaluation

### `TimeSeriesSplitter`
Handles proper time series data splitting:
- Ensures no future data leaks into training
- Creates synthetic forecast rows when needed
- Validates splits for data integrity

### `ModelTrainer` & `ModelPredictor`
Encapsulates model training and prediction:
- Proper sklearn Pipeline usage
- Error handling for edge cases
- Multiple model types (regression, classification, quantile)

## Performance Metrics

### Regression
- R² Score
- Mean Absolute Error (MAE)
- Quantile Coverage (90% prediction intervals)

### Classification
- Accuracy
- Confusion Matrix
- Per-class performance

### Evaluation
- Overall metrics across all forecasts
- Site-specific performance breakdown
- Coverage analysis for uncertainty quantification

## Data Requirements

The pipeline expects a parquet file with columns:
- `date`: Time series dates
- `site`: Location identifiers  
- `da`: Target variable (Domoic Acid levels)
- Feature columns: Oceanographic and climate data

## Configuration

Easily configurable via `DAForecastConfig`:
```python
config = DAForecastConfig.create_config(
    enable_lag_features=True,
    random_state=42,
    n_jobs=-1,
    quantiles=[0.05, 0.5, 0.95]
)
```

## Benefits Over Original Code

1. **No Data Leakage**: Proper time series validation
2. **Reproducible**: All preprocessing in sklearn Pipelines  
3. **Modular**: Clean separation of concerns
4. **Maintainable**: Well-documented, testable components
5. **Robust**: Proper error handling and validation
6. **Flexible**: Easy to modify models and features
7. **Unified**: Single interface for all forecasting needs

This implementation follows best practices for time series forecasting and machine learning pipeline design, ensuring reliable and unbiased model evaluation.