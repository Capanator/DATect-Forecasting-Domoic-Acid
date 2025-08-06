"""
Comprehensive Data Validation Module
===================================

Production-ready data validation for DATect forecasting system.
Provides robust validation, sanitization, and quality checks for all data inputs.

Features:
- Schema validation for input data
- Range and type checking  
- Missing data analysis
- Outlier detection and handling
- Temporal consistency validation
- Feature compatibility verification

Usage:
    from forecasting.core.data_validation import DataValidator
    
    validator = DataValidator()
    
    # Validate input data
    is_valid, issues = validator.validate_prediction_data(df)
    
    # Sanitize data
    clean_data = validator.sanitize_data(df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings

from .logging_config import get_logger
from .exception_handling import ValidationError, safe_execute


class DataValidator:
    """
    Comprehensive data validation with schema checking, quality assessment, and sanitization.
    
    Features:
    - Input schema validation
    - Data type and range validation
    - Missing data assessment
    - Outlier detection and flagging
    - Temporal consistency checks
    - Feature compatibility validation
    """
    
    def __init__(self, strict_mode: bool = False):
        self.logger = get_logger(__name__)
        self.strict_mode = strict_mode
        
        # Define expected data schemas and ranges
        self._define_validation_schemas()
        
        self.logger.info(f"DataValidator initialized (strict_mode={strict_mode})")
    
    def _define_validation_schemas(self):
        """Define validation schemas and expected data ranges."""
        
        # Expected feature ranges for oceanographic data
        self.feature_ranges = {
            # Satellite oceanographic features
            'sst': (5.0, 25.0),           # Sea surface temperature (°C)
            'chlorophyll': (0.1, 100.0),  # Chlorophyll-a concentration (mg/m³)
            'par': (10.0, 70.0),          # Photosynthetically available radiation
            'fluorescence': (0.0, 5.0),   # Fluorescence line height
            'k490': (0.01, 2.0),          # Diffuse attenuation coefficient
            
            # Climate indices
            'PDO': (-3.0, 3.0),           # Pacific Decadal Oscillation
            'ONI': (-3.0, 3.0),           # Oceanic Niño Index
            'BEUTI': (-500.0, 500.0),     # Biologically Effective Upwelling Transport Index
            
            # Streamflow data
            'streamflow': (1000.0, 50000.0),  # USGS streamflow (cubic feet per second)
            
            # Temporal features
            'sin_day_of_year': (-1.0, 1.0),   # Sine day of year
            'cos_day_of_year': (-1.0, 1.0),   # Cosine day of year
            
            # Target and lag features
            'da': (0.0, 500.0),           # Domoic acid concentration (μg/g)
            'da_lag_1': (0.0, 500.0),     # Lag 1 DA concentration
            'da_lag_2': (0.0, 500.0),     # Lag 2 DA concentration
            'da_lag_3': (0.0, 500.0),     # Lag 3 DA concentration
            'pn': (0.0, 10000000.0),      # Pseudo-nitzschia cell count
            
            # DA risk categories
            'da-category': (0, 3)          # Risk categories: 0=Low, 1=Moderate, 2=High, 3=Extreme
        }
        
        # Required columns for different prediction tasks
        self.required_columns = {
            'regression': ['date', 'site'],  # Minimal requirements
            'classification': ['date', 'site'],
            'prediction_input': []  # No strict requirements for prediction input
        }
        
        # Expected data types
        self.expected_dtypes = {
            'date': ['datetime64[ns]', 'object'],
            'site': ['object', 'string'],
            'da': ['float64', 'int64', 'float32'],
            'da-category': ['int64', 'int32', 'category']
        }
        
        # Site validation
        self.valid_sites = {
            'Kalaloch', 'Quinault', 'Copalis', 'Twin Harbors', 'Long Beach',
            'Clatsop Beach', 'Cannon Beach', 'Newport', 'Coos Bay', 'Gold Beach'
        }
    
    def validate_prediction_data(self, data: pd.DataFrame, 
                                task_type: str = "prediction") -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive validation of prediction input data.
        
        Args:
            data: DataFrame to validate
            task_type: Type of task ("regression", "classification", "prediction")
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        try:
            self.logger.info(f"Validating data: {data.shape[0]:,} rows × {data.shape[1]} columns")
            
            # Basic data existence check
            if data is None or data.empty:
                validation_results['errors'].append("Data is empty or None")
                validation_results['is_valid'] = False
                return False, validation_results
            
            # Schema validation
            self._validate_schema(data, task_type, validation_results)
            
            # Data type validation
            self._validate_data_types(data, validation_results)
            
            # Range validation
            self._validate_ranges(data, validation_results)
            
            # Missing data assessment
            self._assess_missing_data(data, validation_results)
            
            # Temporal validation (if date column exists)
            if 'date' in data.columns:
                self._validate_temporal_consistency(data, validation_results)
            
            # Site validation (if site column exists)
            if 'site' in data.columns:
                self._validate_sites(data, validation_results)
            
            # Outlier detection
            self._detect_outliers(data, validation_results)
            
            # Feature correlation analysis
            self._analyze_feature_correlations(data, validation_results)
            
            # Overall quality assessment
            validation_results['data_quality'] = self._assess_overall_quality(data, validation_results)
            
            # Determine final validation status
            has_critical_errors = any('critical' in str(error).lower() for error in validation_results['errors'])
            validation_results['is_valid'] = len(validation_results['errors']) == 0 or not has_critical_errors
            
            # Log validation summary
            self._log_validation_summary(validation_results)
            
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            self.logger.error(f"Validation process failed: {str(e)}")
            validation_results['errors'].append(f"Validation process error: {str(e)}")
            validation_results['is_valid'] = False
            return False, validation_results
    
    def _validate_schema(self, data: pd.DataFrame, task_type: str, results: Dict[str, Any]):
        """Validate data schema against expected structure."""
        required_cols = self.required_columns.get(task_type, [])
        
        # Check for required columns
        missing_required = set(required_cols) - set(data.columns)
        if missing_required:
            results['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check for unexpected columns (informational only)
        expected_features = set(self.feature_ranges.keys()) | {'date', 'site'}
        unexpected_cols = set(data.columns) - expected_features
        if unexpected_cols:
            results['warnings'].append(f"Unexpected columns found: {unexpected_cols}")
    
    def _validate_data_types(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Validate data types against expected types."""
        for col in data.columns:
            if col in self.expected_dtypes:
                expected_types = self.expected_dtypes[col]
                actual_type = str(data[col].dtype)
                
                if actual_type not in expected_types:
                    results['warnings'].append(
                        f"Column '{col}' has type '{actual_type}', expected one of {expected_types}"
                    )
    
    def _validate_ranges(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Validate numeric values are within expected ranges."""
        for col in data.columns:
            if col in self.feature_ranges and pd.api.types.is_numeric_dtype(data[col]):
                min_val, max_val = self.feature_ranges[col]
                
                # Check for values outside valid range
                out_of_range = data[(data[col] < min_val) | (data[col] > max_val)]
                if not out_of_range.empty:
                    pct_invalid = len(out_of_range) / len(data) * 100
                    
                    message = (f"Column '{col}': {len(out_of_range)} values ({pct_invalid:.1f}%) "
                             f"outside valid range [{min_val}, {max_val}]")
                    
                    if pct_invalid > 10:  # More than 10% invalid
                        results['errors'].append(f"CRITICAL: {message}")
                    else:
                        results['warnings'].append(message)
                    
                    # Store details for analysis
                    results[f'{col}_range_violations'] = {
                        'count': len(out_of_range),
                        'percentage': pct_invalid,
                        'min_violation': data[col].min(),
                        'max_violation': data[col].max()
                    }
    
    def _assess_missing_data(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Assess missing data patterns and impact."""
        missing_summary = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = missing_count / len(data) * 100
            
            if missing_count > 0:
                missing_summary[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                # Categorize missing data severity
                if missing_pct > 50:
                    results['errors'].append(
                        f"Column '{col}' has excessive missing data: {missing_pct:.1f}%"
                    )
                elif missing_pct > 20:
                    results['warnings'].append(
                        f"Column '{col}' has significant missing data: {missing_pct:.1f}%"
                    )
        
        results['missing_data_summary'] = missing_summary
        
        # Overall missing data assessment
        total_missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        results['total_missing_percentage'] = round(total_missing_pct, 2)
        
        if total_missing_pct > 30:
            results['errors'].append(f"Overall missing data rate too high: {total_missing_pct:.1f}%")
    
    def _validate_temporal_consistency(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Validate temporal consistency and ordering."""
        try:
            date_col = pd.to_datetime(data['date'])
            
            # Check for duplicate dates within same site
            if 'site' in data.columns:
                duplicates = data.groupby(['date', 'site']).size()
                duplicate_pairs = duplicates[duplicates > 1]
                
                if not duplicate_pairs.empty:
                    results['warnings'].append(
                        f"Found {len(duplicate_pairs)} duplicate date-site combinations"
                    )
            
            # Check date range reasonableness
            min_date = date_col.min()
            max_date = date_col.max()
            
            if min_date.year < 2000:
                results['warnings'].append(f"Very early minimum date: {min_date}")
            
            if max_date > pd.Timestamp.now() + pd.Timedelta(days=365):
                results['warnings'].append(f"Future maximum date: {max_date}")
            
            # Check for large temporal gaps
            date_diff = date_col.sort_values().diff()
            large_gaps = date_diff[date_diff > pd.Timedelta(days=60)]  # 60+ day gaps
            
            if not large_gaps.empty:
                results['warnings'].append(
                    f"Found {len(large_gaps)} large temporal gaps (>60 days)"
                )
            
        except Exception as e:
            results['errors'].append(f"Temporal validation failed: {str(e)}")
    
    def _validate_sites(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Validate monitoring site names."""
        unique_sites = set(data['site'].unique())
        invalid_sites = unique_sites - self.valid_sites
        
        if invalid_sites:
            results['warnings'].append(f"Unrecognized sites: {invalid_sites}")
        
        results['site_summary'] = {
            'unique_sites': len(unique_sites),
            'valid_sites': list(unique_sites & self.valid_sites),
            'invalid_sites': list(invalid_sites)
        }
    
    def _detect_outliers(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Detect statistical outliers in numeric columns."""
        outlier_summary = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.feature_ranges:  # Only check known features
                # Use IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                
                if not outliers.empty:
                    outlier_pct = len(outliers) / len(data) * 100
                    
                    outlier_summary[col] = {
                        'count': len(outliers),
                        'percentage': round(outlier_pct, 2),
                        'lower_bound': round(lower_bound, 3),
                        'upper_bound': round(upper_bound, 3)
                    }
                    
                    if outlier_pct > 5:  # More than 5% outliers
                        results['warnings'].append(
                            f"Column '{col}' has many outliers: {outlier_pct:.1f}%"
                        )
        
        results['outlier_summary'] = outlier_summary
    
    def _analyze_feature_correlations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Analyze feature correlations for data quality assessment."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                
                # Find highly correlated features (potential redundancy)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.95:  # Very high correlation
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': round(corr_val, 3)
                            })
                
                if high_corr_pairs:
                    results['warnings'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
                    results['high_correlations'] = high_corr_pairs
                
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {str(e)}")
    
    def _assess_overall_quality(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality score and recommendations."""
        quality_score = 100.0
        
        # Deduct points for issues
        quality_score -= len(results['errors']) * 20  # 20 points per error
        quality_score -= len(results['warnings']) * 5   # 5 points per warning
        
        # Deduct for missing data
        missing_pct = results.get('total_missing_percentage', 0)
        quality_score -= missing_pct * 0.5  # 0.5 points per percent missing
        
        # Deduct for outliers
        outlier_summary = results.get('outlier_summary', {})
        total_outlier_pct = sum(info['percentage'] for info in outlier_summary.values())
        quality_score -= total_outlier_pct * 0.2  # 0.2 points per percent outliers
        
        quality_score = max(0.0, quality_score)  # Don't go below 0
        
        # Generate quality assessment
        if quality_score >= 90:
            quality_grade = "Excellent"
        elif quality_score >= 80:
            quality_grade = "Good"
        elif quality_score >= 70:
            quality_grade = "Fair"
        elif quality_score >= 50:
            quality_grade = "Poor"
        else:
            quality_grade = "Very Poor"
        
        return {
            'score': round(quality_score, 1),
            'grade': quality_grade,
            'total_issues': len(results['errors']) + len(results['warnings'])
        }
    
    def _log_validation_summary(self, results: Dict[str, Any]):
        """Log validation summary."""
        quality = results['data_quality']
        
        self.logger.info(f"Data validation complete:")
        self.logger.info(f"  Quality Score: {quality['score']}/100 ({quality['grade']})")
        self.logger.info(f"  Errors: {len(results['errors'])}")
        self.logger.info(f"  Warnings: {len(results['warnings'])}")
        self.logger.info(f"  Missing Data: {results.get('total_missing_percentage', 0):.1f}%")
        
        # Log critical issues
        for error in results['errors']:
            self.logger.error(f"  ✗ {error}")
        
        for warning in results['warnings'][:5]:  # Log first 5 warnings
            self.logger.warning(f"  ⚠ {warning}")
    
    def sanitize_data(self, data: pd.DataFrame, 
                     fix_types: bool = True,
                     handle_outliers: bool = True,
                     fill_missing: bool = False) -> pd.DataFrame:
        """
        Sanitize data by fixing common issues.
        
        Args:
            data: DataFrame to sanitize
            fix_types: Fix data type issues
            handle_outliers: Handle outliers (cap to valid ranges)
            fill_missing: Fill missing values with appropriate defaults
            
        Returns:
            Sanitized DataFrame
        """
        data_clean = data.copy()
        
        try:
            self.logger.info("Sanitizing data...")
            
            # Fix data types
            if fix_types:
                data_clean = self._fix_data_types(data_clean)
            
            # Handle outliers
            if handle_outliers:
                data_clean = self._handle_outliers(data_clean)
            
            # Fill missing values
            if fill_missing:
                data_clean = self._fill_missing_values(data_clean)
            
            self.logger.info("Data sanitization complete")
            
            return data_clean
            
        except Exception as e:
            self.logger.error(f"Data sanitization failed: {str(e)}")
            return data
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix common data type issues."""
        data_fixed = data.copy()
        
        # Fix date columns
        if 'date' in data_fixed.columns:
            try:
                data_fixed['date'] = pd.to_datetime(data_fixed['date'])
            except Exception as e:
                self.logger.warning(f"Could not fix date column: {e}")
        
        # Fix site columns
        if 'site' in data_fixed.columns:
            data_fixed['site'] = data_fixed['site'].astype(str)
        
        # Fix numeric columns
        for col in data_fixed.columns:
            if col in self.feature_ranges:
                try:
                    data_fixed[col] = pd.to_numeric(data_fixed[col], errors='coerce')
                except Exception:
                    pass
        
        return data_fixed
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers by capping to valid ranges."""
        data_capped = data.copy()
        
        for col in data_capped.columns:
            if col in self.feature_ranges and pd.api.types.is_numeric_dtype(data_capped[col]):
                min_val, max_val = self.feature_ranges[col]
                
                # Cap outliers to valid range
                original_outliers = ((data_capped[col] < min_val) | (data_capped[col] > max_val)).sum()
                
                if original_outliers > 0:
                    data_capped[col] = data_capped[col].clip(lower=min_val, upper=max_val)
                    self.logger.info(f"Capped {original_outliers} outliers in column '{col}'")
        
        return data_capped
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        data_filled = data.copy()
        
        # Fill numeric columns with median
        numeric_cols = data_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data_filled[col].isnull().any():
                fill_value = data_filled[col].median()
                data_filled[col].fillna(fill_value, inplace=True)
                self.logger.info(f"Filled missing values in '{col}' with median: {fill_value:.3f}")
        
        # Fill categorical columns with mode
        categorical_cols = data_filled.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if data_filled[col].isnull().any():
                fill_value = data_filled[col].mode().iloc[0] if not data_filled[col].mode().empty else 'unknown'
                data_filled[col].fillna(fill_value, inplace=True)
                self.logger.info(f"Filled missing values in '{col}' with mode: {fill_value}")
        
        return data_filled