#!/usr/bin/env python3
"""
Security Improvements Test Suite
===============================

Tests for the new security and validation features added to the DATect system.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from forecasting.core.validation import (
    validate_url, sanitize_filename, validate_coordinate, 
    validate_date_range, validate_numeric_range, validate_config_structure
)
from forecasting.core.secure_download import SecureDownloader, secure_download_file
import config


class TestSecurityValidation(unittest.TestCase):
    """Test security validation functions."""
    
    def test_url_validation_allowed_domains(self):
        """Test URL validation with allowed domains."""
        # Valid URL
        valid_url = "https://oceanview.pfeg.noaa.gov/test/data.nc"
        is_valid, msg = validate_url(valid_url)
        self.assertTrue(is_valid, f"Valid URL rejected: {msg}")
        
        # Invalid domain
        invalid_url = "https://malicious-site.com/data.nc"
        is_valid, msg = validate_url(invalid_url)
        self.assertFalse(is_valid, "Malicious URL accepted")
        
        # Invalid scheme
        bad_scheme = "ftp://oceanview.pfeg.noaa.gov/data.nc"
        is_valid, msg = validate_url(bad_scheme)
        self.assertFalse(is_valid, "FTP scheme accepted")
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        # Dangerous filename
        dangerous = "../../../etc/passwd"
        safe = sanitize_filename(dangerous)
        self.assertNotIn("..", safe, "Directory traversal not sanitized")
        self.assertNotIn("/", safe, "Path separator not sanitized")
        
        # Normal filename
        normal = "data_file.nc"
        safe = sanitize_filename(normal)
        self.assertEqual(normal, safe, "Normal filename modified")
        
        # Special characters
        special = "file<>with\"bad'chars.nc"
        safe = sanitize_filename(special)
        self.assertNotIn("<", safe)
        self.assertNotIn(">", safe)
        self.assertNotIn('"', safe)
        self.assertNotIn("'", safe)
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinates
        is_valid, msg = validate_coordinate(45.0, -124.0)
        self.assertTrue(is_valid, f"Valid coordinates rejected: {msg}")
        
        # Invalid latitude
        is_valid, msg = validate_coordinate(95.0, -124.0)
        self.assertFalse(is_valid, "Invalid latitude accepted")
        
        # Invalid longitude  
        is_valid, msg = validate_coordinate(45.0, 200.0)
        self.assertFalse(is_valid, "Invalid longitude accepted")
    
    def test_date_range_validation(self):
        """Test date range validation."""
        # Valid date range
        is_valid, msg = validate_date_range("2020-01-01", "2020-12-31")
        self.assertTrue(is_valid, f"Valid date range rejected: {msg}")
        
        # Invalid range (start > end)
        is_valid, msg = validate_date_range("2020-12-31", "2020-01-01")
        self.assertFalse(is_valid, "Invalid date range accepted")
        
        # Future dates
        is_valid, msg = validate_date_range("2030-01-01", "2030-12-31")
        self.assertFalse(is_valid, "Future dates accepted")


class TestSecureDownloader(unittest.TestCase):
    """Test secure download functionality."""
    
    @patch('requests.Session.get')
    def test_secure_downloader_validation(self, mock_get):
        """Test secure downloader URL validation."""
        downloader = SecureDownloader()
        
        # Should reject invalid URL
        success, msg, filepath = downloader.validate_and_download(
            "https://malicious-site.com/data.nc"
        )
        self.assertFalse(success, "Invalid URL accepted by downloader")
        
        # Should not make HTTP request for invalid URL
        mock_get.assert_not_called()
    
    @patch('forecasting.core.secure_download.validate_url')
    def test_secure_download_file_validation(self, mock_validate):
        """Test secure_download_file validation."""
        # Mock validation failure
        mock_validate.return_value = (False, "Invalid URL")
        
        result = secure_download_file("https://invalid-url.com/data.nc")
        self.assertIsNone(result, "Download succeeded for invalid URL")


class TestConfigurationSecurity(unittest.TestCase):
    """Test configuration validation and security."""
    
    def test_config_structure_validation(self):
        """Test configuration structure validation."""
        # Valid config
        valid_config = {
            'ORIGINAL_DA_FILES': {'site1': './data/file1.csv'},
            'ORIGINAL_PN_FILES': {'site1': './data/file2.csv'},
            'SITES': {'Site1': [45.0, -124.0]},
            'PDO_URL': 'https://oceanview.pfeg.noaa.gov/test.nc',
            'ONI_URL': 'https://oceanview.pfeg.noaa.gov/test2.nc', 
            'BEUTI_URL': 'https://oceanview.pfeg.noaa.gov/test3.nc',
            'STREAMFLOW_URL': 'https://waterservices.usgs.gov/test.json'
        }
        
        is_valid, msg = validate_config_structure(valid_config)
        self.assertTrue(is_valid, f"Valid config rejected: {msg}")
        
        # Missing required section
        invalid_config = valid_config.copy()
        del invalid_config['SITES']
        
        is_valid, msg = validate_config_structure(invalid_config)
        self.assertFalse(is_valid, "Config with missing section accepted")
    
    def test_environment_detection(self):
        """Test environment detection and configuration."""
        # Test default environment
        self.assertIn(config.ENVIRONMENT, ['development', 'production', 'testing'])
        
        # Test security settings exist
        self.assertIsInstance(config.ALLOWED_DOMAINS, set)
        self.assertTrue(len(config.ALLOWED_DOMAINS) > 0)
        
        # Test timeout settings
        self.assertGreater(config.REQUEST_TIMEOUT_SECONDS, 0)
        self.assertGreater(config.MAX_RETRY_ATTEMPTS, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling improvements."""
    
    def test_handle_data_errors_decorator(self):
        """Test the handle_data_errors decorator."""
        from forecasting.core.exception_handling import handle_data_errors
        
        @handle_data_errors
        def failing_function():
            raise ValueError("Test error")
        
        # Should return None instead of raising
        result = failing_function()
        self.assertIsNone(result, "Error not handled by decorator")
    
    def test_data_integrity_validation(self):
        """Test data integrity validation."""
        from forecasting.core.exception_handling import validate_data_integrity
        import pandas as pd
        
        # Valid data
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        is_valid, msg = validate_data_integrity(df, ['col1', 'col2'])
        self.assertTrue(is_valid, f"Valid data rejected: {msg}")
        
        # Missing columns
        is_valid, msg = validate_data_integrity(df, ['col1', 'missing_col'])
        self.assertFalse(is_valid, "Missing columns not detected")
        
        # Insufficient rows
        is_valid, msg = validate_data_integrity(df, ['col1'], min_rows=10)
        self.assertFalse(is_valid, "Insufficient rows not detected")


def run_security_tests():
    """Run all security tests and report results."""
    print("Running DATect Security Improvement Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSecureDownloader))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All security tests passed!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ Some security tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)