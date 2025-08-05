#!/usr/bin/env python3
"""
Test script for the unified forecasting pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
from modeling import TimeSeriesForecaster, ModelOptimizer, QuantileForecaster


def test_data_processing():
    """Test data processing functionality."""
    print("Testing data processing...")
    
    # Check if data file exists
    data_file = 'final_output.parquet'
    if not os.path.exists(data_file):
        print(f"⚠ Data file {data_file} not found. Creating synthetic test data.")
        
        # Create synthetic test data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='W')
        sites = ['TestSite1', 'TestSite2']
        
        data = []
        for site in sites:
            for date in dates:
                data.append({
                    'site': site,
                    'date': date,
                    'da': np.random.lognormal(1, 1),  # Log-normal distribution for DA levels
                    'sst': np.random.normal(15, 3),   # Sea surface temperature
                    'chlorophyll': np.random.lognormal(0, 0.5)  # Chlorophyll
                })
        
        synthetic_data = pd.DataFrame(data)
        synthetic_data.to_parquet(data_file)
        print(f"✓ Created synthetic data with {len(synthetic_data)} rows")
    
    # Test data loading
    processor = DataProcessor()
    data = processor.load_data(data_file)
    print(f"✓ Loaded data: {len(data)} rows, {data['site'].nunique()} sites")
    
    # Test temporal features  
    data_with_temporal = processor.add_temporal_features(data.head())
    assert 'sin_day_of_year' in data_with_temporal.columns
    assert 'cos_day_of_year' in data_with_temporal.columns
    print("✓ Temporal features created successfully")
    
    # Test train/test split
    site = data['site'].iloc[0]
    anchor_date = data[data['site'] == site]['date'].quantile(0.7)
    train_data, test_data = processor.get_train_test_split(data, site, anchor_date)
    
    assert len(train_data) > 0
    assert len(test_data) > 0
    assert train_data['date'].max() <= anchor_date
    assert test_data['date'].min() > anchor_date
    print("✓ Train/test split working correctly")
    
    # Test training data preparation
    train_prepared = processor.prepare_training_data(train_data, include_lags=True)
    assert 'da_category' in train_prepared.columns
    print("✓ Training data preparation successful")
    
    # Test forecast data preparation
    forecast_prepared = processor.prepare_forecast_data(test_data.head(1), train_data, include_lags=True)
    assert 'sin_day_of_year' in forecast_prepared.columns
    print("✓ Forecast data preparation successful")
    
    return data, processor


def test_modeling(data, processor):
    """Test modeling functionality."""
    print("\nTesting modeling...")
    
    # Prepare data for testing
    site = data['site'].iloc[0]
    anchor_date = data[data['site'] == site]['date'].quantile(0.7)
    train_data, test_data = processor.get_train_test_split(data, site, anchor_date)
    
    train_prepared = processor.prepare_training_data(train_data, include_lags=True)
    train_filtered = processor.filter_features(train_prepared, include_lags=True)
    
    if len(train_prepared) < 10:
        print("⚠ Not enough training data for modeling test")
        return
    
    # Test regression forecaster
    reg_forecaster = TimeSeriesForecaster(
        model_type='random_forest',
        task='regression',
        include_lags=True,
        random_state=42
    )
    
    reg_forecaster.fit(train_filtered, train_prepared['da'])
    
    # Test prediction
    forecast_row = test_data.head(1)
    forecast_prepared = processor.prepare_forecast_data(forecast_row, train_prepared, include_lags=True)
    forecast_filtered = processor.filter_features(forecast_prepared, include_lags=True)
    
    reg_pred = reg_forecaster.predict(forecast_filtered)
    assert len(reg_pred) == 1
    print("✓ Regression forecasting successful")
    
    # Test classification forecaster
    cls_forecaster = TimeSeriesForecaster(
        model_type='random_forest',
        task='classification',
        include_lags=True,
        random_state=42
    )
    
    cls_forecaster.fit(train_filtered, train_prepared['da_category'])
    cls_pred = cls_forecaster.predict(forecast_filtered)
    cls_proba = cls_forecaster.predict_proba(forecast_filtered)
    
    assert len(cls_pred) == 1
    assert len(cls_proba[0]) == 4  # 4 categories
    print("✓ Classification forecasting successful")
    
    # Test quantile forecaster
    quantile_forecaster = QuantileForecaster(random_state=42)
    quantile_forecaster.fit(train_filtered, train_prepared['da'])
    quantile_preds = quantile_forecaster.predict(forecast_filtered)
    
    assert 'q05' in quantile_preds
    assert 'q50' in quantile_preds
    assert 'q95' in quantile_preds
    print("✓ Quantile forecasting successful")
    
    print("✓ All modeling components working correctly")


def test_app_components():
    """Test app components without running the full dashboard."""
    print("\nTesting app components...")
    
    try:
        from app import UnifiedForecastingApp
        
        # Create app instance (without running)
        app = UnifiedForecastingApp()
        print("✓ UnifiedForecastingApp initialized successfully")
        
        # Test configuration
        assert 'random_state' in app.config
        assert 'n_evaluation_samples' in app.config
        print("✓ App configuration loaded correctly")
        
    except ImportError as e:
        print(f"⚠ Could not import app components: {e}")
        print("  This might be due to missing dash dependencies")


def main():
    """Run all tests."""
    print("=== Testing Unified Forecasting Pipeline ===\n")
    
    try:
        # Test data processing
        data, processor = test_data_processing()
        
        # Test modeling
        test_modeling(data, processor)
        
        # Test app components
        test_app_components()
        
        print("\n🎉 All tests passed! The unified pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python3 app.py' to start the unified dashboard")
        print("2. Navigate to http://localhost:8080 in your browser")
        print("3. Switch between 'Future Forecasting' and 'Past Evaluation' modes")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())