# XGBoost vs Naive Baseline Performance Analysis

## Executive Summary

This analysis compares the performance of XGBoost regression predictions against a naive baseline using the previous week's Domoic Acid (DA) concentration. **Surprisingly, the naive baseline significantly outperforms XGBoost across all metrics**, suggesting that DA concentrations exhibit strong week-to-week persistence that the XGBoost model fails to fully capture.

## Methodology

### Data Source
- **XGBoost Predictions**: `cache/retrospective/regression_xgboost.parquet` (5,000 predictions)
- **Naive Baseline**: Previous week's DA value from `final_output.parquet`
- **Date Range**: January 2008 - December 2023
- **Sites**: 10 monitoring locations along the Pacific Coast

### Naive Baseline Strategy
For each prediction (e.g., predicting DA for Kalaloch on 2016-06-14):
- Use the actual DA value from exactly one week prior (2016-06-07)
- Maintain temporal integrity: only use data available before the anchor date
- 100% successful matches (5,000/5,000 predictions)

### Metrics Evaluated
- **Regression**: R² Score, Mean Absolute Error (MAE)
- **Spike Detection**: F1 Score, Precision, Recall (threshold: >20 μg/g)

## Overall Performance Results

| Metric | XGBoost | Naive Baseline | Improvement |
|--------|---------|----------------|-------------|
| **R² Score** | 0.4958 | **0.7851** | **+58.4%** |
| **MAE (μg/g)** | 4.23 | **2.02** | **+109.4%** |
| **F1 Score** | 0.5794 | **0.8491** | **+46.6%** |
| **Precision** | 0.8251 | **0.8589** | **+4.1%** |
| **Recall** | 0.4465 | **0.8395** | **+88.0%** |

### Key Findings:
- **Naive baseline achieves 78.5% R²** vs XGBoost's 49.6%
- **MAE reduced by over 50%**: 2.02 vs 4.23 μg/g
- **F1 score 46% higher**: Superior spike detection capability
- **Recall dramatically better**: 84% vs 45% for spike detection

## Site-Specific Performance

| Site | N | XGB R² | Naive R² | XGB MAE | Naive MAE | XGB F1 | Naive F1 | Spike Rate |
|------|---|--------|----------|---------|-----------|--------|----------|------------|
| Cannon Beach | 500 | 0.329 | **0.757** | 1.37 | **0.76** | 0.533 | **0.762** | 2.0% |
| Clatsop Beach | 500 | 0.458 | **0.780** | 4.86 | **2.57** | 0.527 | **0.868** | 13.0% |
| Coos Bay | 500 | 0.515 | **0.771** | 8.48 | **4.24** | 0.636 | **0.880** | 25.6% |
| Copalis | 500 | 0.558 | **0.935** | 3.14 | **1.10** | 0.645 | **0.895** | 7.6% |
| Gold Beach | 500 | 0.374 | **0.626** | 5.01 | **3.09** | 0.517 | **0.718** | 8.0% |
| Kalaloch | 500 | 0.570 | **0.863** | 3.52 | **1.18** | 0.542 | **0.815** | 5.2% |
| Long Beach | 500 | 0.551 | **0.901** | 3.62 | **1.31** | 0.486 | **0.891** | 10.2% |
| Newport | 500 | 0.138 | **0.540** | 5.86 | **3.33** | 0.418 | **0.714** | 7.8% |
| Quinault | 500 | 0.564 | **0.905** | 2.79 | **1.03** | 0.622 | **0.820** | 5.8% |
| Twin Harbors | 500 | 0.584 | **0.856** | 3.68 | **1.60** | 0.695 | **0.909** | 12.0% |

### Site-Specific Insights:
- **Naive baseline wins at every site** across all metrics
- **Copalis shows strongest persistence** (R² = 0.935 for naive)
- **Newport shows weakest performance** for both methods
- **Higher spike rate sites** (Coos Bay: 25.6%) show larger absolute MAE differences

## Spike Detection Performance

### Overall Spike Statistics:
- **Actual spikes (>20 μg/g)**: 486/5,000 (9.7%)
- **XGBoost predicted spikes**: 263/5,000 (5.3%) - **Under-predicts**
- **Naive predicted spikes**: 475/5,000 (9.5%) - **Accurately captures rate**

### Detection Performance:
- **XGBoost**: High precision (82.5%) but poor recall (44.7%)
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
This analysis reveals that **Domoic Acid concentrations are highly predictable** from recent history alone, achieving nearly 80% R² with a simple one-week lag. This finding has important implications for:
- **Monitoring strategies**: Focus resources on sites with lower persistence
- **Early warning systems**: Simple persistence models may be more reliable
- **Research priorities**: Understanding mechanisms behind temporal persistence

## Data Quality and Limitations

- **Perfect temporal matching**: 100% success rate for naive predictions
- **No future data leakage**: Strict adherence to anchor date constraints
- **Balanced evaluation**: Equal sample sizes (500) across all sites
- **Consistent time period**: 15-year span covers multiple oceanographic cycles

---

**Analysis Date**: September 2025  
**Dataset**: 5,000 retrospective predictions across 10 Pacific Coast monitoring sites  
**Key Finding**: Simple temporal persistence outperforms complex XGBoost model by substantial margins across all performance metrics.