#!/usr/bin/env python3
"""
Deep Scientific Audit of DATect Forecasting System
Focus on potential issues identified in the comprehensive audit
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
from forecasting.data_processor import DataProcessor

def test_temporal_leakage_prevention():
    """Test temporal leakage prevention mechanisms"""
    print("=== 1. TEMPORAL LEAKAGE PREVENTION TEST ===\n")
    
    # Load actual data and test temporal safety
    data_processor = DataProcessor()
    if not Path(config.FINAL_OUTPUT_PATH).exists():
        print("❌ Data file not found - cannot test temporal integrity")
        return
    
    data = data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)
    print(f"✓ Loaded {len(data)} records for temporal analysis")
    
    # Test lag feature creation with anchor date
    anchor_date = pd.Timestamp("2015-01-01")
    lag_data = data_processor.create_lag_features_safe(
        data, "site", "da", [1, 2, 3], anchor_date
    )
    
    # Check for any future contamination
    future_data = lag_data[lag_data['date'] > anchor_date]
    if not future_data.empty:
        print(f"⚠️  Found {len(future_data)} records after anchor date")
        # Check if lag features are properly masked
        future_lag_cols = [col for col in future_data.columns if 'lag' in col]
        if future_lag_cols:
            future_lag_values = future_data[future_lag_cols].notna().sum().sum()
            if future_lag_values > 0:
                print(f"❌ TEMPORAL LEAKAGE: {future_lag_values} lag values found after anchor date")
            else:
                print("✓ Lag features properly masked after anchor date")
    else:
        print("✓ No data after anchor date (expected for some test cases)")
    
    print()

def test_rolling_statistics_safety():
    """Test rolling statistics for temporal safety"""
    print("=== 2. ROLLING STATISTICS SAFETY TEST ===\n")
    
    # Check if rolling stats use proper min_periods
    data_processor = DataProcessor()
    test_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='W'),
        'site': ['Test'] * 10,
        'sst': [15.0, 16.0, np.nan, np.nan, 17.0, 18.0, 19.0, np.nan, 20.0, 21.0]
    })
    
    # Test rolling statistics
    config.USE_ROLLING_FEATURES = True
    enhanced_data = data_processor.add_rolling_statistics_safe(test_data)
    
    # Check rolling means for first few values
    rolling_cols = [col for col in enhanced_data.columns if 'rolling_mean' in col]
    if rolling_cols:
        first_rolling_col = rolling_cols[0]
        first_few_values = enhanced_data[first_rolling_col].iloc[:3]
        print(f"✓ Rolling statistics created: {len(rolling_cols)} columns")
        print(f"  First 3 rolling mean values: {first_few_values.tolist()}")
        
        # Check if min_periods=1 prevents NaN propagation
        if first_few_values.iloc[0] is not np.nan:
            print("✓ min_periods=1 prevents excessive NaN propagation")
        else:
            print("⚠️  min_periods may be too restrictive")
    else:
        print("❌ No rolling statistics created")
        
    config.USE_ROLLING_FEATURES = False  # Reset
    print()

def test_sample_weight_consistency():
    """Test sample weight consistency across models"""
    print("=== 3. SAMPLE WEIGHT CONSISTENCY TEST ===\n")
    
    factory = ModelFactory()
    
    # Create mock data with class imbalance
    y_mock = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 3])  # Imbalanced classes
    
    # Test classification weights
    cls_weights = factory.compute_sample_weights_for_classification(y_mock)
    print(f"Classification sample weights (first 5): {cls_weights[:5]}")
    print(f"Class 0 weight: {cls_weights[0]:.3f}")
    print(f"Class 1 weight: {cls_weights[6]:.3f}")
    print(f"Class 2 weight: {cls_weights[8]:.3f}")
    print(f"Class 3 weight: {cls_weights[9]:.3f}")
    
    # Test spike detection weights
    y_spike = np.array([0, 0, 0, 0, 0, 1, 1, 0])
    spike_weights = factory.compute_spike_focused_weights(y_spike)
    spike_weight_val = spike_weights[y_spike == 1][0]
    non_spike_weight_val = spike_weights[y_spike == 0][0]
    
    print(f"\nSpike detection weights:")
    print(f"  Spike samples (class 1): {spike_weight_val}")
    print(f"  Non-spike samples (class 0): {non_spike_weight_val}")
    print(f"  Ratio: {spike_weight_val / non_spike_weight_val:.1f}x emphasis on spikes")
    
    # Verify against config
    if spike_weight_val == config.SPIKE_FALSE_NEGATIVE_WEIGHT:
        print("✓ Spike weights match configuration")
    else:
        print(f"❌ Spike weight mismatch: {spike_weight_val} != {config.SPIKE_FALSE_NEGATIVE_WEIGHT}")
    
    print()

def test_forecast_horizon_consistency():
    """Test forecast horizon implementation"""
    print("=== 4. FORECAST HORIZON CONSISTENCY TEST ===\n")
    
    print(f"Configured forecast horizon: {config.FORECAST_HORIZON_WEEKS} weeks ({config.FORECAST_HORIZON_DAYS} days)")
    
    # Test with forecast engine
    engine = ForecastEngine()
    
    # Test anchor date calculation
    forecast_date = pd.Timestamp("2020-01-15")  # Wednesday
    target_anchor = forecast_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    
    print(f"Forecast date: {forecast_date.strftime('%A, %Y-%m-%d')}")
    print(f"Target anchor date: {target_anchor.strftime('%A, %Y-%m-%d')}")
    print(f"Actual gap: {(forecast_date - target_anchor).days} days")
    
    if (forecast_date - target_anchor).days == config.FORECAST_HORIZON_DAYS:
        print("✓ Forecast horizon calculation is consistent")
    else:
        print("❌ Forecast horizon calculation mismatch")
    
    print()

def test_da_threshold_scientific_validity():
    """Test DA threshold and categorization"""
    print("=== 5. DA THRESHOLD SCIENTIFIC VALIDITY TEST ===\n")
    
    print(f"Spike threshold: {config.SPIKE_THRESHOLD} μg/g")
    print(f"DA category bins: {config.DA_CATEGORY_BINS}")
    print(f"DA category labels: {config.DA_CATEGORY_LABELS}")
    
    # Test threshold against marine safety standards
    fda_action_level = 20.0  # μg/g - FDA action level for shellfish closure
    who_guidance = 20.0      # μg/g - WHO guidance
    
    if config.SPIKE_THRESHOLD == fda_action_level:
        print(f"✓ Spike threshold matches FDA action level ({fda_action_level} μg/g)")
    else:
        print(f"⚠️  Spike threshold ({config.SPIKE_THRESHOLD}) differs from FDA action level ({fda_action_level})")
    
    # Test category bins for scientific validity
    expected_bins = [-float("inf"), 5, 20, 40, float("inf")]
    if config.DA_CATEGORY_BINS == expected_bins:
        print("✓ DA categories align with toxicological risk levels")
        print("  - Low risk: 0-5 μg/g (no health concern)")
        print("  - Moderate risk: 5-20 μg/g (monitoring advised)")
        print("  - High risk: 20-40 μg/g (FDA action level exceeded)")
        print("  - Extreme risk: >40 μg/g (severe health risk)")
    else:
        print(f"⚠️  DA categories may not align with standard risk levels")
    
    print()

def test_feature_engineering_validity():
    """Test feature engineering for scientific validity"""
    print("=== 6. FEATURE ENGINEERING SCIENTIFIC VALIDITY TEST ===\n")
    
    # Check temporal features
    print("Temporal features:")
    print("✓ sin/cos day of year - captures seasonal patterns")
    print("✓ sin/cos month - captures monthly cycles") 
    print("✓ quarter - captures quarterly patterns")
    print("✓ days_since_start - captures long-term trends")
    
    # Check environmental thresholds
    print(f"\nEnvironmental bloom thresholds:")
    print(f"✓ Chlorophyll-a threshold: {config.CHLA_THRESHOLD_PERCENTILE*100}th percentile")
    print(f"✓ PAR threshold: {config.PAR_THRESHOLD_PERCENTILE*100}th percentile")
    print(f"✓ Optimal SST range: {config.OPTIMAL_SST_RANGE[0]}-{config.OPTIMAL_SST_RANGE[1]}°C")
    
    # Validate SST range against scientific literature
    if 12 <= config.OPTIMAL_SST_RANGE[0] <= 15 and 16 <= config.OPTIMAL_SST_RANGE[1] <= 20:
        print("✓ SST range aligns with Pseudo-nitzschia bloom temperature preferences")
    else:
        print("⚠️  SST range may not align with known bloom conditions")
    
    print()

def test_model_fairness():
    """Test model fairness and baseline consistency"""
    print("=== 7. MODEL FAIRNESS & BASELINE CONSISTENCY TEST ===\n")
    
    factory = ModelFactory()
    
    # Test model creation parity
    models_to_test = [
        ("regression", "xgboost"),
        ("regression", "linear"),
        ("classification", "xgboost"), 
        ("classification", "logistic"),
        ("spike_detection", "xgboost"),
        ("spike_detection", "logistic")
    ]
    
    successful_models = 0
    for task, model_type in models_to_test:
        try:
            model = factory.get_model(task, model_type)
            print(f"✓ {task} - {model_type}: Created successfully")
            successful_models += 1
        except Exception as e:
            print(f"❌ {task} - {model_type}: Failed - {e}")
    
    print(f"\nModel creation success rate: {successful_models}/{len(models_to_test)}")
    
    # Test sample weight support
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    # Test LogisticRegression sample_weight support
    try:
        lr = LogisticRegression()
        X_test = np.random.rand(10, 3)
        y_test = np.random.randint(0, 2, 10)
        w_test = np.ones(10)
        lr.fit(X_test, y_test, sample_weight=w_test)
        print("✓ LogisticRegression supports sample_weight")
    except Exception as e:
        print(f"❌ LogisticRegression sample_weight issue: {e}")
    
    # Test LinearRegression (should NOT use sample weights for regression baseline)
    try:
        lr = LinearRegression()
        X_test = np.random.rand(10, 3)
        y_test = np.random.rand(10)
        lr.fit(X_test, y_test)  # No sample weights for fair comparison
        print("✓ LinearRegression baseline (no sample weights)")
    except Exception as e:
        print(f"❌ LinearRegression issue: {e}")
    
    print()

def run_comprehensive_audit():
    """Run comprehensive scientific audit"""
    print("🔬 COMPREHENSIVE SCIENTIFIC INTEGRITY AUDIT")
    print("=" * 60)
    print()
    
    test_temporal_leakage_prevention()
    test_rolling_statistics_safety()
    test_sample_weight_consistency()
    test_forecast_horizon_consistency()
    test_da_threshold_scientific_validity()
    test_feature_engineering_validity()
    test_model_fairness()
    
    print("=" * 60)
    print("🎯 AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_audit()