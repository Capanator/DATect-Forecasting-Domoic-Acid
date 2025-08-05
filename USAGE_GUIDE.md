# Unified DA Forecasting Pipeline - Usage Guide

## Command Line Interface

The unified pipeline now supports fine-grained control over tasks and models:

### Required Arguments
- `--mode {retrospective,realtime}`: Choose evaluation mode
- `--task {regression,classification}`: Required for retrospective mode only (ignored in realtime mode)

### Optional Arguments
- `--model {rf,linear}`: Model type for retrospective mode only (default: rf, ignored in realtime mode)
  - `rf`: Random Forest (primary model)
  - `linear`: Linear/Logistic Regression (baseline for comparison)
- `--anchors`: Number of anchor forecasts per site (default: 50, can override via command line)
- `--data`: Data file path (default: final_output.parquet)
- `--port`: Dashboard port
- `--min-test-date`: Minimum test date for evaluation (default: 2008-01-01)

### Configuration Changes
- **Lag features are now PERMANENT**: Always enabled (da_lag_1, da_lag_2, da_lag_3)
- **Default forecasts per site**: 50 (was 200) - change `CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"]` in code or use `--anchors` flag
- **Real-time mode runs BOTH tasks**: Always generates both regression and classification results

## Usage Examples

### Real-time Forecasting (Both Tasks)

**Basic real-time dashboard (both regression and classification):**
```bash
python3 improved_unified_forecast.py --mode realtime
```

**Real-time with custom port:**
```bash
python3 improved_unified_forecast.py --mode realtime --port 8066
```

### Retrospective Evaluation (Single Task)

**Random Forest Regression evaluation:**
```bash
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf
```

**Linear Regression baseline:**
```bash
python3 improved_unified_forecast.py --mode retrospective --task regression --model linear
```

**Random Forest Classification with more forecasts per site:**
```bash
python3 improved_unified_forecast.py --mode retrospective --task classification --model rf --anchors 100
```

**Logistic Regression Classification baseline:**
```bash
python3 improved_unified_forecast.py --mode retrospective --task classification --model linear
```

## Key Changes from Original

### ✅ Removed GridSearchCV
- No more time-consuming hyperparameter tuning
- Uses sensible default parameters
- Much faster startup times

### ✅ Split Regression/Classification
- Run only the task you need
- Separate evaluation for each task type
- Cleaner performance metrics

### ✅ Model Selection
- Choose between Random Forest (primary) and Linear models (baseline)
- Compare performance between model types
- Linear models serve as baseline comparison (as in original past-forecasts-final.py)

### ✅ Lag Features (Permanent)
- Lag features always enabled: da_lag_1, da_lag_2, da_lag_3 (1, 2, 3 weeks prior)
- No command line flag needed - permanently part of feature set
- Created per-forecast to avoid data leakage

## Performance Comparison Workflow

**Compare Random Forest vs Linear baseline for regression:**
```bash
# Run Random Forest
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf

# Run Linear baseline  
python3 improved_unified_forecast.py --mode retrospective --task regression --model linear

# Compare R² and MAE metrics
```

**Compare different numbers of forecasts per site:**
```bash
# Default (50 forecasts per site)
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf

# More forecasts per site for more robust evaluation
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf --anchors 200
```

## Dashboard Behavior

### Real-time Mode (`--mode realtime`)
**Always runs BOTH regression and classification** with Random Forest + Gradient Boosting
- **No `--task` flag needed** - automatically generates both types of forecasts
- **Regression**: Gradient visualization with quantile predictions (Q05, Q50, Q95) + RF point prediction
- **Classification**: Category probability distributions using Random Forest
- **Purpose**: Production forecasting with comprehensive results (like original `future-forecasts.py`)

### Retrospective Mode (`--mode retrospective`)
**Respects `--model` flag for comparison**
- **`--model rf`**: Uses Random Forest models
- **`--model linear`**: Uses Linear/Logistic models as baseline
- **Purpose**: Model evaluation and comparison

## Data Leakage Prevention

- ✅ Categories created per-forecast from training data only
- ✅ Lag features created within proper time series splits
- ✅ No global preprocessing that leaks future information
- ✅ Maintains temporal order in all operations

The pipeline is now more modular, faster, and gives you precise control over what you want to evaluate!