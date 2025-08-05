# Configuration Reference

## Quick Configuration Changes

### Change Number of Forecasts Per Site

**Method 1: Edit CONFIG in code (permanent)**
```python
# In improved_unified_forecast.py, line ~40
CONFIG = {
    "NUM_RANDOM_ANCHORS_PER_SITE_EVAL": 100,  # Change this number
    # ... other config
}
```

**Method 2: Use command line (temporary override)**
```bash
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf --anchors 100
```

### Common Values for Forecasts Per Site
- **50** (default): Fast evaluation, good for testing
- **100**: Balanced evaluation time vs accuracy
- **200**: More robust evaluation (original default)
- **500**: Very thorough evaluation (slow but comprehensive)

### Lag Features Configuration
Lag features are now **permanently enabled**. They include:
- `da_lag_1`: DA value from 1 week prior
- `da_lag_2`: DA value from 2 weeks prior  
- `da_lag_3`: DA value from 3 weeks prior

**To modify lag features, edit the code:**
```python
# In add_lag_features method, line ~92
for lag in [1, 2, 3]:  # Change these numbers to modify lags
    df[f'da_lag_{lag}'] = df.groupby('site')['da'].shift(lag)
```

### Other Configuration Options

**Change default ports:**
```python
CONFIG = {
    "PORT_RETRO": 8071,     # Retrospective dashboard port
    "PORT_REALTIME": 8065,  # Real-time dashboard port
}
```

**Change evaluation parameters:**
```python
CONFIG = {
    "MIN_TEST_DATE": "2008-01-01",  # Earliest date for test data
    "N_JOBS_EVAL": -1,              # Parallel jobs (-1 = use all cores)
    "RANDOM_SEED": 42,              # Random seed for reproducibility
}
```

## Performance Impact

| Anchors per Site | Total Forecasts* | Evaluation Time | Accuracy |
|------------------|------------------|-----------------|----------|
| 50               | ~500             | 1-2 minutes     | Good     |
| 100              | ~1,000           | 3-5 minutes     | Better   |
| 200              | ~2,000           | 6-10 minutes    | Best     |
| 500              | ~5,000           | 15-25 minutes   | Excellent|

*Approximate total across all 10 sites

## Quick Commands

**Fast evaluation (50 anchors per site):**
```bash
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf
```

**Robust evaluation (200 anchors per site):**
```bash
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf --anchors 200
```

**Compare models with consistent evaluation:**
```bash
# Random Forest
python3 improved_unified_forecast.py --mode retrospective --task regression --model rf --anchors 100

# Linear baseline  
python3 improved_unified_forecast.py --mode retrospective --task regression --model linear --anchors 100
```

**Real-time forecasting (always RF+GB, both tasks):**
```bash
# Real-time mode runs both regression and classification
python3 improved_unified_forecast.py --mode realtime

# No --task flag needed, --model flag ignored
python3 improved_unified_forecast.py --mode realtime --port 8066
```

## Model Behavior by Mode

### Real-time Mode (`--mode realtime`)
**Always runs BOTH regression and classification** with Random Forest + Gradient Boosting (like original `future-forecasts.py`)
- ✅ No `--task` flag needed - automatically runs both
- ✅ Random Forest for regression and classification
- ✅ Gradient Boosting for quantile predictions (Q05, Q50, Q95)
- ✅ Both gradient visualization and category probability charts
- ✅ Coverage analysis and category matching
- ⚠️ **Both `--task` and `--model` flags are ignored**

### Retrospective Mode (`--mode retrospective`) 
**Respects the `--model` flag for baseline comparison**
- `--model rf`: Uses Random Forest
- `--model linear`: Uses Linear/Logistic Regression (baseline)
- Used for comparing model performance