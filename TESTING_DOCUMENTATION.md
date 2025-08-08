# Testing Documentation

## 🧪 Comprehensive Testing Framework

DATect employs a multi-layered testing approach to ensure scientific validity and system reliability.

## Test Categories

### 1. Temporal Integrity Tests (CRITICAL)

**Location**: `analysis/scientific-validation/test_temporal_integrity.py`

These tests are the foundation of scientific validity - preventing data leakage in time series forecasting.

#### Test Suite Overview
```python
def test_no_future_leak_in_training_data():
    """Ensures no future data contaminates training sets"""
    
def test_training_data_temporal_cutoff():
    """Validates training data cutoff dates are respected"""
    
def test_satellite_buffer_period():
    """Confirms 7-day satellite processing buffer"""
    
def test_climate_reporting_delay():
    """Validates 2-month climate data reporting delays"""
    
def test_lag_features_temporal_safety():
    """Ensures lag features respect temporal boundaries"""
    
def test_chronological_train_test_split():
    """Validates chronological ordering in splits"""
    
def test_prediction_date_integrity():
    """Confirms predictions don't use future information"""
```

#### Running Temporal Tests
```bash
# Automatic execution (recommended)
python run_datect.py  # Runs all 7 tests before system startup

# Manual execution
python analysis/scientific-validation/test_temporal_integrity.py

# Pytest with verbose output
python -m pytest analysis/scientific-validation/test_temporal_integrity.py -v
```

#### Expected Output
```
✅ test_no_future_leak_in_training_data - PASSED
✅ test_training_data_temporal_cutoff - PASSED  
✅ test_satellite_buffer_period - PASSED
✅ test_climate_reporting_delay - PASSED
✅ test_lag_features_temporal_safety - PASSED
✅ test_chronological_train_test_split - PASSED
✅ test_prediction_date_integrity - PASSED

🎉 All temporal integrity tests passed (7/7)
```

### 2. Complete Pipeline Integration Tests

**Location**: `tools/testing/test_complete_pipeline.py`

Validates end-to-end system functionality from data processing through prediction.

#### Test Components
- Data loading and preprocessing validation
- Feature engineering pipeline testing
- Model training and validation
- Prediction generation and formatting
- API endpoint functionality
- Frontend-backend integration

#### Execution
```bash
# Full pipeline test
python tools/testing/test_complete_pipeline.py

# Expected: 21/21 components pass
```

### 3. Scientific Validation Suite

**Location**: `analysis/scientific-validation/`

#### Core Validation Scripts

**`run_scientific_validation.py`**
- Peer-review standard validation
- Model performance benchmarking
- Statistical significance testing
- Cross-validation results

**`advanced_acf_pacf.py`**
- Autocorrelation function analysis
- Partial autocorrelation analysis
- Optimal lag selection validation
- Statistical significance of lag features

**`performance_profiler.py`**
- System performance benchmarking
- Memory usage profiling
- Processing speed measurements
- Resource utilization analysis

#### Execution
```bash
# Complete scientific validation
python analysis/scientific-validation/run_scientific_validation.py

# Statistical lag analysis
python analysis/scientific-validation/advanced_acf_pacf.py

# Performance benchmarking
python analysis/scientific-validation/performance_profiler.py
```

### 4. Web Application Tests

#### Backend API Tests
```bash
# FastAPI test suite (if available)
cd backend && python -m pytest tests/ -v

# Manual API testing
curl http://localhost:8000/api/health
curl http://localhost:8000/api/visualizations/correlation/all
```

#### Frontend Tests
```bash
# React component tests (if available)
cd frontend && npm test

# Build validation
cd frontend && npm run build
```

## Test Data Requirements

### Processed Data File
- **Location**: `data/processed/final_output.parquet`
- **Generation**: `python dataset-creation.py` (30-60 minutes)
- **Validation**: Automatic in `run_datect.py`

### Raw Data Dependencies
- DA measurements: `data/raw/da-input/`
- Pseudo-nitzschia data: `data/raw/pn-input/`
- Satellite data: Downloaded automatically during processing

## Continuous Integration Testing

### Pre-Deployment Checklist
```bash
# 1. Run complete system validation
python run_datect.py

# 2. Validate all components manually
python analysis/scientific-validation/test_temporal_integrity.py
python tools/testing/test_complete_pipeline.py
python analysis/scientific-validation/run_scientific_validation.py

# 3. Performance benchmarking
python analysis/scientific-validation/performance_profiler.py
```

### Success Criteria
- ✅ Temporal Integrity: 7/7 tests pass
- ✅ Pipeline Integration: 21/21 components pass  
- ✅ Scientific Validation: All checks pass
- ✅ Performance: Meets benchmark standards

## Test Environment Setup

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Additional Test Dependencies
```bash
pip install pytest pytest-cov pytest-mock
```

### Data Prerequisites
```bash
# Generate test data if missing
python dataset-creation.py
```

## Test Results Interpretation

### Critical Failures
**Temporal Integrity Failure**: Any failure in the 7 temporal tests indicates potential data leakage. **DO NOT USE SYSTEM** until resolved.

**Pipeline Integration Failure**: Indicates system component malfunction. Debug specific failing component.

### Performance Benchmarks
- **Processing Speed**: Should achieve >80,000 samples/second
- **Memory Usage**: Should remain <300MB for full dataset
- **Model Performance**: R² should approach 0.529 for XGBoost
- **API Response Time**: <2 seconds for standard requests

### Warning Signs
- Tests passing intermittently (indicates race conditions)
- Performance degradation over time (memory leaks)
- Model performance below benchmarks (data quality issues)

## Testing Best Practices

### Before Code Changes
1. Run full test suite to establish baseline
2. Document current performance metrics
3. Identify potentially affected test categories

### After Code Changes
1. Run affected test categories first
2. Execute complete test suite
3. Compare performance metrics
4. Validate no regressions introduced

### Before Deployment
1. Clean environment test (fresh Python virtual environment)
2. Complete system startup test via `run_datect.py`
3. Manual smoke testing of key functionality
4. Performance validation under load

## Test Data Management

### Test Data Lifecycle
- **Generation**: Via `dataset-creation.py` from raw sources
- **Validation**: Automatic integrity checks during loading
- **Refresh**: Regenerate when raw data sources update
- **Backup**: Archive validated datasets for consistency

### Data Quality Checks
- Temporal ordering validation
- Missing value assessment
- Outlier detection and handling
- Feature completeness verification
- Measurement range validation

---

## 🚨 Critical Testing Requirements

### Must-Pass Tests
These tests are **NON-NEGOTIABLE** for scientific validity:

1. **All 7 temporal integrity tests** must pass (100%)
2. **Complete pipeline integration** must pass (21/21 components)
3. **Scientific validation suite** must complete without errors
4. **System startup** must succeed via `run_datect.py`

### Test Automation
The `run_datect.py` launcher automatically executes critical tests before system startup, ensuring no scientifically invalid system can be deployed.

**Failure at any test stage prevents system startup - this is intentional and critical for scientific integrity.**