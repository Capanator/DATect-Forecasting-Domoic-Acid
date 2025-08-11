# DATect Visualizations Guide

## Overview

DATect provides a comprehensive suite of scientific visualizations to help users understand domoic acid forecasting results, validate model performance, and explore oceanographic relationships. This guide explains how to interpret each visualization type and what insights they provide.

## Dashboard Visualizations (Real-time Forecasting)

### 1. Forecast Results Display

#### 1.1 Prediction Summary Cards
```
┌─────────────────────┐  ┌─────────────────────┐
│   Regression        │  │  Classification     │
│  15.2 μg/g         │  │   MODERATE RISK     │
│  ±3.1 uncertainty  │  │   68% confidence    │
└─────────────────────┘  └─────────────────────┘
```

**How to Interpret**:
- **Regression Value**: Continuous DA concentration prediction in μg/g
- **Uncertainty**: ±95% confidence interval from model ensemble
- **Risk Category**: Discrete risk level based on established thresholds
- **Confidence**: Model certainty in categorical prediction

**Risk Category Thresholds**:
- **Low (Green)**: 0-5 μg/g - Safe for consumption
- **Moderate (Yellow)**: 5-20 μg/g - Caution advised
- **High (Orange)**: 20-40 μg/g - Avoid consumption  
- **Extreme (Red)**: >40 μg/g - Health hazard

#### 1.2 Advanced Gradient Uncertainty Plot

```
     Gradient Boosting Uncertainty Visualization
     
Low                    Prediction                    High
├────────▓▓▓▓▓▓▓▓▓▓████▓▓▓▓▓▓▓▓────────┤
         Q05    Q50 (Median)    Q95
                  ◊ XGBoost Point Prediction
                     × Actual Value (if available)
```

**How to Interpret**:
- **Gradient Band**: Color intensity shows prediction confidence
  - **Dark Blue**: High confidence region around median
  - **Light Blue**: Lower confidence at quantile extremes
- **Q50 Line (Thick)**: Median prediction from gradient boosting
- **Q05/Q95 Markers**: 5th and 95th percentile bounds (90% confidence interval)
- **Diamond (XGBoost)**: Primary point prediction used for decisions
- **X Mark**: Actual measured value (when available for validation)

**Scientific Interpretation**:
- Wide gradient band = High prediction uncertainty
- Narrow gradient band = High prediction confidence  
- XGBoost point close to Q50 = Consistent model predictions
- Actual value within gradient band = Good model calibration

### 2. Feature Importance Visualizations

#### 2.1 XGBoost Feature Importance (Tree-based)
```
Feature Importance (XGBoost Gain)
│
├── SST_lag1              ████████████████████ 0.35
├── Chlorophyll_lag3      ████████████████ 0.28
├── PDO_current           ███████████ 0.18
├── Fluorescence_lag1     ██████ 0.12
├── BEUTI_lag3           ████ 0.07
└── ...
```

**How to Interpret**:
- **Gain Values**: Contribution to model performance (sum to 1.0)
- **Bar Length**: Relative importance compared to other features
- **Feature Names**: Include lag information and data source

**Top Features Typically Include**:
- **SST (Sea Surface Temperature)**: Ocean warming/cooling patterns
- **Chlorophyll**: Phytoplankton biomass indicator
- **PDO (Pacific Decadal Oscillation)**: Large-scale climate pattern
- **Fluorescence**: Phytoplankton health/stress indicator
- **BEUTI**: Coastal upwelling intensity

#### 2.2 Permutation Importance (Model-Agnostic)
```
Permutation Importance (Δ Performance)
│
├── PDO_current           ████████████████████ 0.42
├── SST_lag1              ███████████████ 0.31
├── Chlorophyll_anomaly   ██████████ 0.19
├── PN_counts_lag3        █████ 0.08
└── ...
```

**How to Interpret**:
- **Δ Performance**: How much model performance drops when feature is scrambled
- **Higher Values**: More critical features for accurate predictions
- **Different Rankings**: May differ from tree importance due to feature interactions

### 3. Time Series Comparison Plots

#### 3.1 DA vs. Pseudo-nitzschia Temporal Patterns
```
DA Concentration & Pseudo-nitzschia Cell Counts Over Time

40 ┤                                                          
   │    ●                                                     
30 ┤      ●                                              ●    
   │        ●          ●                               ●      
20 ┤          ●      ●   ●                           ●        
   │            ●  ●       ●                       ●          
10 ┤              ●         ●                   ●            
   │                         ●               ●               
 0 ┴─────────────────────────────────────────────────────────
   2020-01    2020-07    2021-01    2021-07    2022-01
   
   ● DA Concentration (μg/g)    ▲ Pseudo-nitzschia (cells/L)
```

**How to Interpret**:
- **Temporal Alignment**: Look for DA peaks following PN blooms (typical lag: 1-3 weeks)
- **Magnitude Correlation**: Larger PN blooms often predict higher DA levels
- **Seasonal Patterns**: Both typically peak in spring/summer upwelling season
- **Missing Correlations**: May indicate other environmental factors at play

#### 3.2 Environmental Drivers Time Series
```
Multi-Parameter Environmental Conditions

SST (°C)  ┤  ～～～～＼        ／～～～～
          │           ＼    ／
          │             ～～
          
CHL (mg/m³)┤     ▲▲▲            ▲▲
           │   ▲    ▲        ▲▲    ▲
           │ ▲        ▲    ▲        ▲
           
PDO Index ┤ ────────────────────────
          │     ＼      ／
          │       ＼  ／
          │         ～
```

**How to Interpret**:
- **SST Patterns**: Warm periods often correlate with bloom conditions
- **Chlorophyll Spikes**: Indicate phytoplankton blooms (potential DA producers)
- **PDO Oscillations**: Large-scale climate influence on regional oceanography
- **Multi-parameter Alignment**: Look for convergent conditions that trigger blooms

## Historical Analysis Visualizations

### 4. Correlation Heatmaps

#### 4.1 Feature Correlation Matrix
```
                    DA   SST  CHL  PDO  BEUTI  FLH
DA              │  1.00  0.24 0.31 -0.18  0.42  0.26 │
SST             │  0.24  1.00 0.19 -0.31  -0.15 0.22 │  
Chlorophyll     │  0.31  0.19 1.00  0.08   0.33 0.78 │
PDO             │ -0.18 -0.31 0.08  1.00   0.29 0.11 │
BEUTI           │  0.42 -0.15 0.33  0.29   1.00 0.31 │
Fluorescence    │  0.26  0.22 0.78  0.11   0.31 1.00 │
```

**Color Scale Interpretation**:
- **Red (Positive)**: Variables increase/decrease together
- **Blue (Negative)**: Variables move in opposite directions  
- **White (Near Zero)**: Little to no linear relationship

**Key Relationships to Look For**:
- **DA vs. BEUTI (Upwelling)**: Strong positive correlation expected
- **DA vs. Chlorophyll**: Moderate positive (bloom conditions)
- **SST vs. PDO**: Climate pattern influence on temperature
- **Chlorophyll vs. Fluorescence**: Phytoplankton biomass/health

#### 4.2 Lag-Correlation Analysis
```
Cross-Correlation: DA vs Environmental Variables

Correlation Coefficient
 1.0 ┤
     │        ●
 0.5 ┤      ●   ●
     │    ●       ●
 0.0 ┼──●───────────●──────────●──
     │                            
-0.5 ┤                          ●
     │
-1.0 ┴─────────────────────────────────
    -10   -5    0    5   10   15   20
         Lag (weeks)
         
    ● Optimal lag = 3 weeks (r = 0.45)
```

**How to Interpret**:
- **Peak Correlation**: Optimal time lag for prediction features
- **Negative Lags**: Environmental conditions leading DA events
- **Positive Lags**: DA events leading environmental changes (less common)
- **Multiple Peaks**: Complex relationships with seasonal components

### 5. Sensitivity Analysis (Sobol Indices)

#### 5.1 First-Order Sensitivity
```
Sobol First-Order Sensitivity Indices

Total Effect Decomposition:
├── SST_lag1           ██████████████████ 0.31 (Main Effect)
├── Chlorophyll_lag3   ████████████████ 0.27 (Main Effect)  
├── PDO_current        ██████████ 0.18 (Main Effect)
├── Interactions       ████████ 0.15 (Combined Effects)
└── Higher Order       ███ 0.09 (Complex Interactions)
```

**How to Interpret**:
- **First-Order**: Direct influence of each variable alone
- **Interactions**: Combined effects of multiple variables
- **Higher Order**: Complex multi-variable relationships
- **Total < 1.0**: Some variance unexplained (measurement noise, missing factors)

#### 5.2 Second-Order Interactions
```
Sobol Second-Order Interaction Matrix

            SST  CHL  PDO  BEUTI
SST      │   -   0.05 0.12  0.08 │
CHL      │  0.05  -   0.09  0.14 │
PDO      │  0.12 0.09  -    0.07 │
BEUTI    │  0.08 0.14 0.07   -   │
```

**How to Interpret**:
- **Diagonal**: Not applicable (self-interactions)
- **Off-diagonal**: Interaction strength between variable pairs
- **High Values (>0.10)**: Strong synergistic effects
- **Low Values (<0.05)**: Variables act mostly independently

### 6. Spectral Analysis

#### 6.1 Power Spectral Density
```
Frequency Domain Analysis: DA Time Series

Power Spectral Density
│
│  ●
│    ●
│      ●        ●
│        ●    ●   ●
│          ●●       ●
│                    ●●●●●●●
├─────────────────────────────────────
0   0.1  0.2  0.3  0.4  0.5  1.0  2.0
    Frequency (cycles/week)
    
Dominant Periods:
● 12-week seasonal cycle
● 4-week upwelling cycle
● 52-week annual cycle
```

**How to Interpret**:
- **Power Peaks**: Dominant cyclical patterns in DA data
- **Seasonal Cycles**: Expected annual and semi-annual patterns
- **Short-term Cycles**: Weather-driven upwelling/relaxation cycles
- **Broad Spectrum**: Complex system with multiple timescales

#### 6.2 Coherence Analysis
```
Coherence: DA vs Environmental Variables

Coherence²
 1.0 ┤     ●●●
     │   ●     ●
 0.8 ┤ ●         ●
     │●           ●
 0.6 ┤             ●●
     │               ●●●
 0.4 ┤                   ●●●●
     │
 0.0 ┴─────────────────────────────
    0.0   0.1   0.2   0.3   0.4
           Frequency (cycles/week)
           
High coherence at seasonal frequencies (0.02 cycles/week)
```

**How to Interpret**:
- **High Coherence (>0.8)**: Strong linear relationship at that frequency
- **Low Coherence (<0.4)**: Weak relationship or phase differences  
- **Frequency Bands**: Different timescales show different coupling strengths
- **Phase Information**: Can indicate lead/lag relationships (not shown)

## Model Performance Visualizations

### 7. Retrospective Validation Results

#### 7.1 Scatter Plot: Predicted vs. Actual
```
Model Performance: Predicted vs Actual DA

Actual DA (μg/g)
 50 ┤
    │     ●
 40 ┤   ●   ●
    │ ●       ●
 30 ┤●         ●
    │           ●
 20 ┤●●●●●●●●●●●●●
    │●●●●●●●●●●●●●●
 10 ┤●●●●●●●●●●●●●●●●
    │●●●●●●●●●●●●●●●●●
  0 ┴─────────────────────────────────
    0   10   20   30   40   50
         Predicted DA (μg/g)
         
    ─ Perfect prediction (y=x)
    R² = 0.37, MAE = 6.2 μg/g
```

**How to Interpret**:
- **Diagonal Line**: Perfect prediction reference
- **Point Scatter**: Actual model performance
- **R² Value**: Explained variance (0.37 = 37% explained)
- **MAE**: Average absolute error in same units as data
- **Clustering**: Most points in low-DA region (realistic distribution)

#### 7.2 Residual Analysis
```
Residual Analysis: Prediction Errors

Residuals (μg/g)
 20 ┤     ●
    │   ●   ●
 10 ┤ ●●●●●●●●●
    │●●●●●●●●●●●●●
  0 ┼●●●●●●●●●●●●●●●●●●●●●●●●●
    │●●●●●●●●●●●●●●●
-10 ┤   ●●●●●●●
    │     ●
-20 ┴─────────────────────────────
    0   10   20   30   40   50
         Predicted DA (μg/g)
         
    No systematic bias detected
```

**How to Interpret**:
- **Horizontal Band**: Good - no systematic over/under-prediction
- **Funnel Shape**: Increasing errors with higher predictions (common)
- **Outliers**: Check for unusual oceanographic conditions
- **Centered at Zero**: Unbiased predictions on average

### 8. Classification Performance

#### 8.1 Confusion Matrix
```
Classification Results: 4-Category Risk Levels

           Predicted
Actual  │ Low  Mod High Ext │
────────┼─────────────────┤
Low     │ 245   12    2   0 │  
Moderate│  18   87    8   1 │
High    │   3   15   22   4 │
Extreme │   0    1    5   8 │
        
Overall Accuracy: 79.4%
```

**How to Interpret**:
- **Diagonal**: Correct predictions (darker = better)
- **Off-diagonal**: Misclassifications
- **Near-diagonal**: Predictions close to correct category (acceptable)
- **Far off-diagonal**: Serious misclassifications (investigate causes)

#### 8.2 ROC Curves (Multi-class)
```
ROC Curves: Per-Category Performance

True Positive Rate
 1.0 ┤     ●●●●●●●
     │   ●●
 0.8 ┤ ●●          ●●●●●●●●
     │●           ●
 0.6 ┤          ●●
     │        ●●
 0.4 ┤      ●●
     │    ●●
 0.2 ┤  ●●
     │●●
 0.0 ┴─────────────────────────────
    0.0  0.2  0.4  0.6  0.8  1.0
         False Positive Rate
         
    Low Risk (AUC = 0.94)
    Moderate Risk (AUC = 0.82)
    High Risk (AUC = 0.76)  
    Extreme Risk (AUC = 0.88)
```

**How to Interpret**:
- **Curve Shape**: Closer to top-left corner = better performance
- **AUC Values**: Area under curve (1.0 = perfect, 0.5 = random)
- **Class Performance**: Low/Extreme risk easier to predict than Moderate/High
- **Operational Trade-offs**: Choose threshold based on false positive tolerance

## Site-Specific Visualizations

### 9. Spatial Performance Maps

#### 9.1 Site Performance Summary
```
Model Performance by Monitoring Site

Site              │  R²   MAE  Accuracy  Sample Size
──────────────────┼─────────────────────────────────
Cannon Beach      │ 0.42  5.8    81.2%        1,247
Newport           │ 0.39  6.1    78.9%        1,556  
Coos Bay          │ 0.35  6.8    76.4%        1,103
Gold Beach        │ 0.31  7.2    74.1%          892
Twin Harbors      │ 0.28  7.8    72.3%          743
...
```

**How to Interpret**:
- **R² Variation**: Site-specific predictability differences
- **MAE Patterns**: Error magnitude by location
- **Sample Size**: More data generally improves performance
- **Geographic Trends**: Northern sites often have better performance

### 10. Uncertainty Quantification

#### 10.1 Prediction Intervals
```
Uncertainty Quantification: 95% Prediction Intervals

DA Prediction with Uncertainty
 30 ┤
    │     ┌─●─┐     
 25 ┤   ┌─┴───┴─┐   
    │ ┌─┴───────┴─┐ 
 20 ┤─┴───────────┴─
    │               
 15 ┤     Point     
    │   Prediction   
 10 ┤               
    │ ┌─┬───────┬─┐ 
  5 ┤───┴───────┴───
    │     
  0 ┴─────────────────
    Site A  Site B
    
    ● Point prediction
    ├─┤ 95% confidence interval
    ┌─┐ Prediction interval
```

**How to Interpret**:
- **Point Prediction**: Most likely DA value
- **Confidence Interval**: Uncertainty in the mean prediction
- **Prediction Interval**: Range for individual measurements
- **Interval Width**: Wider intervals = higher uncertainty

## Troubleshooting Visualizations

### 11. Data Quality Indicators

#### 11.1 Missing Data Patterns
```
Data Completeness Heatmap

        2020  2021  2022  2023
        J F M A M J J A S O N D
Site A  │█████████████████████│ 98.2%
Site B  │████████░░███████████│ 89.4%  
Site C  │██████████████░░░████│ 85.7%
Site D  │████░░░██████████████│ 82.1%

█ Data available  ░ Missing data
```

**How to Interpret**:
- **Dark Areas**: Complete data coverage
- **Light Areas**: Missing data periods
- **Seasonal Gaps**: May indicate sampling challenges
- **Site Differences**: Operational constraints vary by location

### 12. Model Diagnostics

#### 12.1 Learning Curves
```
Model Training Progress

Performance Metric
 0.8 ┤
     │ ●●●●●●●●●●●●●●●●  ← Training Score
 0.6 ┤ 
     │   ●●●●●●●●●●●●●●  ← Validation Score
 0.4 ┤     
     │
 0.2 ┤
     │
 0.0 ┴─────────────────────────────────
    0    50   100  150  200  250  300
         Training Samples
         
    Convergence achieved at ~200 samples
```

**How to Interpret**:
- **Training Score**: Performance on training data
- **Validation Score**: Performance on held-out data
- **Gap Size**: Large gap indicates overfitting
- **Convergence**: Stable performance with more data
- **Optimal Size**: Minimum samples needed for reliable training

## Best Practices for Interpretation

### Guidelines for Users:

1. **Always Check Uncertainty**: Don't rely only on point predictions
2. **Consider Temporal Context**: Seasonal and annual patterns matter
3. **Validate Feature Importance**: Ensure scientifically reasonable results  
4. **Monitor Data Quality**: Missing data affects prediction reliability
5. **Compare Multiple Sites**: Spatial consistency indicates robust patterns
6. **Check Residuals**: Look for systematic biases in predictions
7. **Use Multiple Metrics**: No single metric tells the complete story
8. **Consider Operational Constraints**: Real-world deployment limitations

### Common Interpretation Pitfalls:

1. **Over-interpreting Correlations**: Correlation ≠ causation
2. **Ignoring Uncertainty**: Point predictions without confidence intervals
3. **Temporal Leakage**: Using inappropriate time periods for validation
4. **Sample Size Bias**: Small samples may show misleading patterns
5. **Seasonal Confounding**: Not accounting for natural cycles
6. **Scale Mismatches**: Comparing metrics across different scales

This comprehensive guide enables users to extract maximum scientific insight from DATect's visualization suite while avoiding common interpretation errors.