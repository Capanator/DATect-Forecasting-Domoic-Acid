# XGBoost vs Naive Baseline Performance Analysis

## Executive Summary

This analysis compares the performance of XGBoost regression predictions against a naive baseline using the previous week's Domoic Acid (DA) concentration. **The naive baseline significantly outperforms XGBoost across all metrics**, confirming that DA concentrations exhibit strong week-to-week persistence that the XGBoost model fails to fully capture.

## Methodology

### Data Source
- **XGBoost Predictions**: `cache/retrospective/regression_xgboost.parquet` (5000 predictions)
- **Naive Baseline**: Previous week's DA value from `final_output.parquet`
- **Date Range**: Updated analysis with latest cached data
- **Sites**: 10 monitoring locations along the Pacific Coast

### Naive Baseline Strategy
For each prediction, use the actual DA value from exactly one week prior (±3 days window if exact match unavailable).
Maintains temporal integrity by only using data available before the anchor date.

### Metrics Evaluated
- **Regression**: R² Score, Mean Absolute Error (MAE)
- **Spike Detection**: F1 Score, Precision, Recall (threshold: >20 μg/g)

## Overall Performance Results

| Metric | XGBoost | Naive Baseline | Improvement |
|--------|---------|----------------|-------------|
| **R² Score** | 0.4934 | **0.7851** | **+59.1%** |
| **MAE (μg/g)** | 4.86 | **2.02** | **+140.3%** |
| **F1 Score** | 0.6142 | **0.8491** | **+38.2%** |
| **Precision** | 0.6352 | **0.8589** | **+35.2%** |
| **Recall** | 0.5947 | **0.8395** | **+41.2%** |

### Key Findings:
- **Naive baseline achieves 78.5% R²** vs XGBoost's 49.3%
- **MAE reduced by 140%**: 2.02 vs 4.86 μg/g
- **F1 score 38% higher**: Superior spike detection capability
- **Recall dramatically better**: 84.0% vs 59.5% for spike detection

## Site-Specific Performance

| Site | N | XGB R² | Naive R² | XGB MAE | Naive MAE | XGB F1 | Naive F1 | Spike Rate |
|------|---|--------|----------|---------|-----------|--------|----------|------------|
| Cannon Beach | 500 | 0.388 | **0.757** | 1.68 | **0.76** | 0.667 | **0.762** | 2.0% |
| Clatsop Beach | 500 | 0.507 | **0.780** | 5.37 | **2.57** | 0.622 | **0.868** | 13.0% |
| Coos Bay | 500 | 0.514 | **0.771** | 9.35 | **4.24** | 0.677 | **0.880** | 25.6% |
| Copalis | 500 | 0.474 | **0.935** | 3.64 | **1.10** | 0.597 | **0.895** | 7.6% |
| Gold Beach | 500 | 0.415 | **0.626** | 5.53 | **3.09** | 0.579 | **0.718** | 8.0% |
| Kalaloch | 500 | 0.256 | **0.863** | 4.84 | **1.18** | 0.459 | **0.815** | 5.2% |
| Long Beach | 500 | 0.613 | **0.901** | 3.79 | **1.31** | 0.682 | **0.891** | 10.2% |
| Newport | 500 | 0.169 | **0.540** | 7.08 | **3.33** | 0.488 | **0.714** | 7.8% |
| Quinault | 500 | 0.524 | **0.905** | 3.34 | **1.03** | 0.471 | **0.820** | 5.8% |
| Twin Harbors | 500 | 0.608 | **0.856** | 3.98 | **1.60** | 0.685 | **0.909** | 12.0% |

### Site-Specific Insights:
- **Naive baseline wins at every site** across all metrics
- **Copalis shows strongest persistence** (R² = 0.935 for naive)
- **Newport shows weakest performance** for both methods
- **Higher spike rate sites** show larger absolute MAE differences

## Spike Detection Performance

### Overall Spike Statistics:
- **Actual spikes (>20 μg/g)**: 486/5000 (9.7%)
- **XGBoost predicted spikes**: 455/5000 (9.1%) - **Under-predicts**
- **Naive predicted spikes**: 475/5000 (9.5%) - **Accurately captures rate**

### Detection Performance:
- **XGBoost**: Moderate precision (63.5%) but poor recall (59.5%)
- **Naive Baseline**: Balanced performance (85.9% precision, 84.0% recall)

## Implications and Conclusions

### 1. Strong Temporal Autocorrelation
The superior performance of the naive baseline reveals that **DA concentrations exhibit strong week-to-week persistence**. This suggests:
- Biological processes driving DA production change slowly
- Environmental conditions persist across weekly timescales
- Current XGBoost model may be overfitting to noise rather than capturing signal

### 2. Model Performance Issues
XGBoost's poor performance indicates:
- **Potential overfitting** to training data
- **Inadequate feature engineering** for temporal patterns
- **Missing key temporal dependencies** in current feature set
- **Need for explicit autoregressive features** (lagged DA values)

### 3. Recommendations

#### Immediate Actions:
1. **Add lagged DA features** to XGBoost model (1-week, 2-week, 4-week lags)
2. **Investigate hyperparameter tuning** to reduce overfitting
3. **Consider ensemble approach** combining XGBoost with temporal persistence

#### Strategic Considerations:
1. **Temporal models**: LSTM/GRU for sequence modeling
2. **Hybrid approach**: Use naive baseline as a feature in XGBoost
3. **Re-evaluate feature importance** focusing on persistence indicators

### 4. Scientific Insights
This analysis reveals that **Domoic Acid concentrations are highly predictable** from recent history alone, achieving 78.5% R² with a simple one-week lag. This finding has important implications for:
- **Monitoring strategies**: Focus resources on sites with lower persistence
- **Early warning systems**: Simple persistence models may be more reliable
- **Research priorities**: Understanding mechanisms behind temporal persistence

## Data Quality and Limitations

- **Temporal matching**: High success rate for naive predictions
- **No future data leakage**: Strict adherence to anchor date constraints
- **Comprehensive evaluation**: 5000 predictions across all sites
- **Updated analysis**: Based on latest cached results from September 2025

---

**Analysis Date**: September 2025  
**Dataset**: 5000 retrospective predictions across 10 Pacific Coast monitoring sites  
**Key Finding**: Simple temporal persistence continues to outperform complex XGBoost model by substantial margins across all performance metrics.
