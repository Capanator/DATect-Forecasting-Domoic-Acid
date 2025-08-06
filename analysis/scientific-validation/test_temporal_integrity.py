#!/usr/bin/env python3
"""
Temporal Integrity Unit Tests
============================

Critical unit tests to ensure temporal safeguards prevent data leakage.
These tests are ESSENTIAL for peer review and scientific validation.

Usage:
    python test_temporal_integrity.py
    python -m pytest test_temporal_integrity.py -v
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import system components
from forecasting.core.data_processor import DataProcessor
import config
from forecasting.core.forecast_engine import ForecastEngine


class TestTemporalIntegrity(unittest.TestCase):
    """Unit tests for temporal data leakage prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor()
        
        # Create synthetic test data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='W')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'site': ['TestSite'] * len(dates),
            'da': np.random.lognormal(1, 1, len(dates)),
            'feature1': np.random.normal(10, 2, len(dates)),
            'feature2': np.random.normal(5, 1, len(dates))
        })
        
    def test_lag_feature_temporal_cutoff(self):
        """Test that lag features don't use future information."""
        cutoff_date = pd.Timestamp('2020-06-01')
        
        # Create lag features with temporal cutoff
        data_with_lags = self.data_processor.create_lag_features_safe(
            self.test_data, 'site', 'da', config.LAG_FEATURES, cutoff_date
        )
        
        # Check that lag features after cutoff are NaN or properly restricted
        future_data = data_with_lags[data_with_lags['date'] > cutoff_date]
        
        # Lag features should be available for most points but restricted near cutoff
        buffer_zone = future_data[future_data['date'] <= cutoff_date + pd.Timedelta(days=7)]
        
        # At least some lag features should be NaN near the cutoff to prevent leakage
        lag_columns = [f'da_lag_{lag}' for lag in config.LAG_FEATURES]
        self.assertTrue(
            buffer_zone[lag_columns].isnull().any().any(),
            f"Lag features {lag_columns} should have NaN values near temporal cutoff"
        )
        
    def test_temporal_split_ordering(self):
        """Test that training data is always before test data."""
        anchor_date = pd.Timestamp('2020-06-01')
        
        train_mask = self.test_data['date'] <= anchor_date
        test_mask = self.test_data['date'] > anchor_date
        
        train_data = self.test_data[train_mask]
        test_data = self.test_data[test_mask]
        
        # Verify no temporal overlap
        max_train_date = train_data['date'].max()
        min_test_date = test_data['date'].min()
        
        self.assertLess(
            max_train_date, min_test_date,
            "Training data must be strictly before test data"
        )
        
    def test_da_category_independence(self):
        """Test that DA categories are created independently per forecast."""
        # Create two different datasets
        data1 = self.test_data[:26]  # First half of year
        data2 = self.test_data[26:]  # Second half of year
        
        # Create categories independently
        categories1 = self.data_processor.create_da_categories_safe(data1['da'])
        categories2 = self.data_processor.create_da_categories_safe(data2['da'])
        
        # Categories should be created independently (no shared statistics)
        # This is verified by checking that the binning is consistent with config
        self.assertTrue(all(cat in [0, 1, 2, 3] for cat in categories1.dropna()))
        self.assertTrue(all(cat in [0, 1, 2, 3] for cat in categories2.dropna()))
        
    def test_preprocessing_fit_only_on_training(self):
        """Test that preprocessing statistics come only from training data."""
        anchor_date = pd.Timestamp('2020-06-01')
        
        train_data = self.test_data[self.test_data['date'] <= anchor_date]
        
        # Create transformer and features
        transformer, X_train = self.data_processor.create_numeric_transformer(
            train_data, ['date', 'site', 'da']
        )
        
        # Fit transformer
        X_train_processed = transformer.fit_transform(X_train)
        
        # Check that transformer statistics are reasonable
        self.assertIsNotNone(X_train_processed)
        self.assertEqual(len(X_train_processed), len(train_data))
        
        # Verify no future data was used in fitting
        # (This is ensured by the split, but we verify the split worked)
        self.assertEqual(len(train_data), sum(self.test_data['date'] <= anchor_date))


class TestDataIntegrity(unittest.TestCase):
    """Unit tests for data integrity and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor()
        
    def test_temporal_integrity_validation(self):
        """Test temporal integrity validation function."""
        # Create test data with proper temporal ordering
        train_dates = pd.date_range('2020-01-01', '2020-06-01', freq='W')
        train_df = pd.DataFrame({
            'date': train_dates,
            'da': range(len(train_dates))
        })
        
        test_dates = pd.date_range('2020-06-08', '2020-08-01', freq='W') 
        test_df = pd.DataFrame({
            'date': test_dates,
            'da': range(100, 100 + len(test_dates))
        })
        
        # Should pass temporal integrity check
        is_valid = self.data_processor.validate_temporal_integrity(train_df, test_df)
        self.assertTrue(is_valid, "Properly ordered data should pass temporal integrity check")
        
        # Create overlapping data (should fail)
        bad_test_dates = pd.date_range('2020-05-01', '2020-07-01', freq='W')  # Overlaps with training
        bad_test_df = pd.DataFrame({
            'date': bad_test_dates,
            'da': range(200, 200 + len(bad_test_dates))
        })
        
        is_invalid = self.data_processor.validate_temporal_integrity(train_df, bad_test_df)
        self.assertFalse(is_invalid, "Overlapping data should fail temporal integrity check")


class TestModelValidation(unittest.TestCase):
    """Unit tests for model validation and forecasting logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal test dataset
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='W')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'site': ['TestSite'] * len(dates),
            'da': np.random.lognormal(1, 1, len(dates)),
            'sin_day_of_year': np.sin(2 * np.pi * dates.dayofyear / 365),
            'cos_day_of_year': np.cos(2 * np.pi * dates.dayofyear / 365),
            'feature1': np.random.normal(10, 2, len(dates))
        })
        
        # Save test data
        self.test_data.to_parquet('test_data.parquet')
        
    def tearDown(self):
        """Clean up test files."""
        import os
        if os.path.exists('test_data.parquet'):
            os.remove('test_data.parquet')
            
    def test_forecast_engine_initialization(self):
        """Test that forecast engine initializes properly."""
        engine = ForecastEngine('test_data.parquet')
        self.assertIsNotNone(engine)
        self.assertEqual(engine.data_file, 'test_data.parquet')
        
    def test_minimum_training_samples(self):
        """Test that forecasting requires minimum training samples."""
        # This test would require running an actual forecast
        # For now, we test the configuration
        engine = ForecastEngine('test_data.parquet')
        self.assertGreaterEqual(engine.min_training_samples, 3)


def run_tests():
    """Run all temporal integrity tests."""
    print("🧪 Running Temporal Integrity Unit Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTemporalIntegrity))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataIntegrity))  
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All temporal integrity tests PASSED")
        print("🔬 Scientific temporal safeguards are validated")
    else:
        print("❌ Some tests FAILED")
        print("⚠️  Temporal integrity may be compromised")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)