# Dataset Creation Scientific Decisions

**Document Purpose**: This document explains the scientific rationale behind key decisions made in the DATect dataset creation pipeline (`dataset-creation.py`). Each decision is documented with biological justification, temporal integrity considerations, and impact on model performance.

## Overview

The DATect dataset creation pipeline processes 21+ years of multi-source data into a unified weekly time series suitable for harmful algal bloom forecasting. Key priorities include:

1. **Temporal Integrity**: Preventing data leakage in retrospective analyses
2. **Biological Realism**: Reflecting natural toxin and bloom dynamics
3. **Scientific Rigor**: Using defensible interpolation and aggregation methods
4. **Forecasting Relevance**: Creating features available in real-world prediction scenarios

---

## 1. Temporal Interpolation of Toxin Data

### Problem Identified
**Original Implementation**: Linear interpolation for up to 6 weeks with unlimited forward-fill to 0
- Created artificial smoothing that unrealistically improved naive baseline performance
- No consideration of biological decay processes
- Temporal leakage risk in retrospective testing

### Scientific Decision: Biological Decay Interpolation

**Implementation** (Lines 883-955):
```python
# DA Parameters
da_max_gap_weeks = 2  # Conservative gap limit
da_decay_rate = 0.2   # Per week (3.5-week half-life)

# PN Parameters  
pn_max_gap_weeks = 4  # More aggressive due to sparser data
pn_decay_rate = 0.3   # Per week (2.3-week half-life)
```

**Biological Justification**:
- **DA Decay Rate**: 0.2/week falls within published razor clam depuration rates (0.02 ± 0.08 day⁻¹)
- **Exponential Decay**: Reflects natural toxin clearance through depuration, dilution, and degradation
- **Conservative Gap Limits**: DA limited to 2 weeks based on observed frequent zero measurements (48-52% of samples)

**Data Analysis Support**:
- Newport DA data: 48.2% zero measurements, gaps averaging 59 days
- Long Beach DA data: 0% zeros but 17.8% >20 μg/g, gaps averaging 55 days
- PN data: Much sparser (131-2000 records vs 700+ for DA), gaps 96-400+ days

**Temporal Integrity**: 
- Uses only past values for interpolation
- No forward-looking information that could create leakage in retrospective tests
- More realistic representation of real-world forecasting conditions

---

## 2. Satellite Data Temporal Alignment

### Problem Addressed
Satellite data availability varies by measurement type and reporting schedules, requiring careful temporal alignment to prevent future data leakage.

### Scientific Decision: Differential Temporal Buffers

**Implementation** (Lines 444-480):
```python
if is_anomaly_var:
    # Use data from 2 months before target (accounts for processing delays)
    safe_month_period = current_month_period - 2
else:
    # Regular satellite data: 1-week minimum buffer
    cutoff_date = target_ts - pd.Timedelta(days=7)
```

**Justification**:
- **Anomaly Data**: Requires longer time series for baseline calculation, uses 2-month buffer
- **Regular Satellite Data**: Near-real-time availability, uses 1-week safety buffer
- **No Fallback for Regular Data**: If data unavailable with safety buffer, leaves as NaN rather than risk leakage

---

## 3. Climate Index Temporal Alignment

### Scientific Decision: 2-Month Reporting Lag

**Implementation** (Lines 787-791):
```python
compiled_df['TargetPrevMonth'] = compiled_df['Date'].dt.to_period("M") - 2
```

**Justification**:
- **Reporting Reality**: Climate indices (PDO, ONI) have 1-2 month reporting delays
- **Forecasting Relevance**: Uses only climate data that would be available at prediction time
- **Conservative Approach**: 2-month lag ensures data availability in operational forecasting

---

## 4. BEUTI (Upwelling Index) Gap Filling

### Problem Identified
**Original Implementation**: `fillna(0)` - artificially created "neutral upwelling" periods

### Scientific Decision: Forward-Fill with Median Backup

**Implementation** (Lines 1082-1087):
```python
# Forward-fill BEUTI (upwelling patterns persist)
base_final_data["beuti"] = base_final_data.groupby('Site')["beuti"].fillna(method='ffill')
# Fill remaining with median (preserves natural distribution)
beuti_median = base_final_data["beuti"].median()
base_final_data["beuti"] = base_final_data["beuti"].fillna(beuti_median)
```

**Scientific Justification**:
- **BEUTI Range**: Can be legitimately negative (downwelling), zero (neutral), or positive (upwelling)
- **Persistence**: Upwelling patterns tend to persist over multiple weeks
- **Natural Distribution**: Median fill preserves realistic BEUTI value distribution

---

## 5. Weekly Aggregation Strategy

### Scientific Decision: ISO Week with Monday Anchor

**Implementation** (Lines 688, 747):
```python
df['Year-Week'] = df['Parsed_Date'].dt.strftime('%G-%V')
da_df_copy['Date'] = pd.to_datetime(da_df_copy['Year-Week'] + '-1', format='%G-%V-%w')
```

**Justification**:
- **Consistency**: ISO weeks provide consistent 52-53 week years
- **Monday Anchor**: Aligns with typical monitoring schedules
- **Temporal Coherence**: Ensures all data sources align to same weekly boundaries

---

## 6. Site Name Normalization

### Scientific Decision: Consistent Normalization Pipeline

**Implementation**: Multiple locations use `.replace('_', ' ').str.title()`

**Rationale**:
- **Data Integration**: Raw data files use various naming conventions (underscores, hyphens, case variations)
- **Consistent Matching**: Standardized format enables reliable joins across data sources
- **Human Readable**: Title case improves readability in outputs

---

## 7. Missing Data Philosophy

### DA/PN Extended Gaps
**Decision**: Fill gaps >2-4 weeks with 0
**Justification**: Extended sampling gaps likely represent periods of non-detectable toxin levels

### Environmental Data
**Decision**: Conservative gap filling (forward-fill, median backup)
**Justification**: Environmental conditions persist and should not default to artificial neutral states

### Satellite Data
**Decision**: No interpolation beyond safety buffers
**Justification**: Satellite data quality issues better handled by leaving as missing rather than interpolating

---

## 8. Data Quality Considerations

### Validation Steps
1. **Temporal Integrity Checks**: Ensure no future data contamination
2. **Biological Range Validation**: Check for negative DA values (biologically impossible)
3. **Site Coverage Validation**: Ensure all configured sites have data
4. **Date Range Validation**: Verify coverage spans full analysis period

### Quality Flags
- **Gap Reporting**: Document interpolation counts for transparency
- **Missing Data Reporting**: Track remaining NaN values after processing
- **Temporal Coverage**: Report date ranges and gap statistics

---

## Impact on Model Performance

### Positive Impacts
1. **Realistic Baselines**: Biological decay prevents artificial improvement of naive models
2. **Temporal Safety**: Buffers prevent data leakage in retrospective analysis
3. **Feature Relevance**: Uses only data available in real forecasting scenarios

### Tradeoffs
1. **Reduced Data Density**: Conservative gap limits reduce available training data
2. **Increased Complexity**: More sophisticated interpolation requires careful parameter tuning
3. **Validation Overhead**: Additional temporal integrity checks slow processing

---

## References

1. Trainer, V.L., et al. (2007). "Pseudo-nitzschia physiological ecology, phylogeny, toxicity, monitoring and impacts on ecosystem health." *Harmful Algae*, 14, 271-300.

2. Wekell, J.C., et al. (1994). "Occurrence of domoic acid in Washington State razor clam (*Siliqua patula*) populations." *Journal of Shellfish Research*, 13(1), 197-205.

3. Lefebvre, K.A., et al. (2002). "Detection of domoic acid in northern anchovies and California sea lions associated with an unusual mortality event." *Natural Toxins*, 6(6), 207-211.

---

## Revision History

- **Initial Version**: Linear interpolation, temporal leakage risks
- **Current Version**: Biological decay, temporal buffers, scientific gap filling
- **Future Considerations**: Site-specific decay rates, seasonal parameter adjustment