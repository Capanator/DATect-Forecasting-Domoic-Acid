# XGBoost Hyperparameter Optimization for DATect Forecasting

## 🎯 Objective
Find XGBoost parameter combinations that beat the current baseline performance:
- **R² > 0.3203**
- **MAE < 6.84** 
- **Spike F1 > 0.5796**

## 🚀 Speed Optimizations Applied

### 1. **Reduced Parameter Grid** (95% reduction)
- **Before:** ~130,000 parameter combinations
- **After:** 6,696 parameter combinations  
- **Speedup:** ~20x fewer evaluations

### 2. **Optimized XGBoost Settings**
- `tree_method='hist'` - Much faster tree construction
- `max_bin=256` - Reduced histogram bins
- `grow_policy='depthwise'` - More efficient growth
- `n_jobs=1` per model (parallel at higher level)

### 3. **Multi-Level Parallelization**
- **Configuration level:** All CPU cores evaluate different parameter sets
- **Site level:** Parallel evaluation across monitoring sites  
- **Anchor level:** Parallel processing of anchor points
- **Estimated speedup:** 5-10x depending on CPU cores

### 4. **Smart Filtering**
- **Early rejection:** Discard configs that don't beat baseline
- **Metric-based:** Must improve 2+ metrics or show significant R² gain
- **Memory efficient:** Process in adaptive batches

### 5. **Anchor Points** 
- **Default:** 100 anchor points per site (for maximum accuracy)
- **Speed option:** Can be reduced to 20-50 if needed (see usage section)

## 📊 Parameter Ranges Tested

### Core XGBoost Parameters
- **n_estimators:** 50 → 1,500 (capped at 1,200 for very high values)
- **max_depth:** 3 → 10 (focused on effective range)
- **learning_rate:** 0.03 → 0.2
- **subsample:** 0.8 → 0.9
- **colsample_bytree:** 0.8 → 0.9

### Regularization Parameters
- **reg_alpha:** 0.0 → 1.5
- **reg_lambda:** 0.0 → 1.5  
- **min_child_weight:** 1 → 3
- **gamma:** 0.0 → 0.3

### Spike Detection Parameters
- **spike_weight:** 2.0 → 20.0 (how much to weight DA spikes > threshold)
- **spike_threshold:** 15.0, 20.0, 25.0 μg/g

## 🏃‍♂️ Usage

### Standard Run (100 anchor points - Maximum accuracy)
```bash
python3 hyperparameter_tuning.py
```

### Quick Run (For testing/faster results)
```bash
python3 -c "
from hyperparameter_tuning import HyperparameterTuner
tuner = HyperparameterTuner(n_anchors_per_site=20)  # 20 anchors for speed
results = tuner.run_comprehensive_tuning()
"
```

## 📈 Expected Output

### Success Case
```
🎉 FOUND 15 CONFIGURATIONS THAT BEAT BASELINE!

🏆 BEST IMPROVEMENTS:
Best R² Improvement: +12.3% (from 0.3203 to 0.3596)
Parameters: {'n_estimators': 1200, 'max_depth': 8, 'learning_rate': 0.05, ...}

Best MAE Improvement: 8.7% (from 6.84 to 6.25)
Best Spike F1 Improvement: +15.2% (from 0.5796 to 0.6676)

TOP 10 CONFIGURATIONS THAT BEAT BASELINE:
 1. R²=0.3596(+12.3%) | MAE=6.25(+8.7%) | F1=0.6676(+15.2%)
    n_est=1200, depth=8, lr=0.05, spike_w=16.0, spike_th=20.0
 ...
```

### No Improvement Case
```
❌ NO CONFIGURATIONS BEAT THE CURRENT BASELINE!

Top 5 configurations (even though they don't beat baseline):
1. R²=0.3180, MAE=6.91, F1=0.5621
   n_est=1500, depth=10, lr=0.03
...
```

## 💾 Output Files

Results are saved to `xgboost_optimization_results_YYYYMMDD_HHMMSS.json` containing:
- Current baseline performance
- All configurations that beat baseline
- Best performing configurations by metric
- Runtime statistics
- Complete parameter details

## ⚡ Performance Estimates

**With all optimizations (100 anchor points):**
- **6,696 configurations** × **100 anchors/site** × **10 sites** = 6.7M evaluations
- **Estimated runtime:** 4-12 hours (depending on CPU cores)
- **Memory usage:** Moderate (batch processing prevents memory issues)

**Speed vs Accuracy Trade-offs (if needed):**
- 🎯 **Default:** 100 anchors/site (maximum accuracy)
- ⚖️ **Balanced:** 50 anchors/site (2x faster)  
- ⚡ **Quick:** 20 anchors/site (5x faster, still robust)

## 🔧 Further Speed Options

If still too slow, you can:
1. **Reduce sites:** Focus on 2-3 key monitoring locations
2. **Reduce anchors:** Use 10-15 anchors per site
3. **Reduce parameters:** Comment out high-end configurations
4. **Use subset:** Test only medium-range configurations first

The script is designed to find meaningful improvements while being practical to run!