"""
Data Quality Management
=======================

Comprehensive data quality validation, monitoring, and reporting for the
DATect forecasting system. Provides multi-layered validation, quality metrics,
and automated quality assurance.

This module provides:
- Multi-tier data validation (syntax, semantic, business rules)
- Data quality metrics and scoring
- Automated quality reports and alerts
- Data freshness and completeness monitoring
- Quality trend analysis and anomaly detection
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .logging_config import get_logger
from .exception_handling import ScientificValidationError
import config

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Data quality levels for classification."""
    EXCELLENT = "excellent"    # 95-100% quality score
    GOOD = "good"             # 85-94% quality score  
    ACCEPTABLE = "acceptable" # 70-84% quality score
    POOR = "poor"            # 50-69% quality score
    CRITICAL = "critical"    # <50% quality score


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    check_name: str
    passed: bool
    score: float  # 0-100
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    timestamp: str
    overall_score: float
    quality_level: QualityLevel
    total_records: int
    validation_results: List[ValidationResult]
    metrics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'quality_level': self.quality_level.value,
            'validation_results': [asdict(vr) for vr in self.validation_results]
        }


class DataValidator:
    """
    Comprehensive data validator with multi-tier validation approach.
    
    Implements three tiers of validation:
    1. Syntactic: Data types, formats, basic structure
    2. Semantic: Value ranges, logical consistency
    3. Business Rules: Domain-specific validation rules
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = []
        logger.info("Initialized DataValidator")
    
    def validate_dataset(self, df: pd.DataFrame, 
                        dataset_name: str = "unknown",
                        validation_config: Optional[Dict[str, Any]] = None) -> QualityReport:
        """
        Perform comprehensive validation of a dataset.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for reporting
            validation_config: Optional validation configuration
            
        Returns:
            Comprehensive quality report
        """
        logger.info(f"Starting comprehensive validation of {dataset_name}")
        
        self.validation_results = []
        
        try:
            # Tier 1: Syntactic validation
            self._validate_syntax(df, dataset_name)
            
            # Tier 2: Semantic validation
            self._validate_semantics(df, dataset_name)
            
            # Tier 3: Business rules validation
            self._validate_business_rules(df, dataset_name, validation_config)
            
            # Generate quality report
            report = self._generate_quality_report(df, dataset_name)
            
            logger.info(f"Validation completed for {dataset_name}: "
                       f"{report.quality_level.value} quality ({report.overall_score:.1f}%)")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during validation of {dataset_name}: {e}")
            # Return critical quality report on validation failure
            return QualityReport(
                dataset_name=dataset_name,
                timestamp=datetime.now().isoformat(),
                overall_score=0.0,
                quality_level=QualityLevel.CRITICAL,
                total_records=len(df) if df is not None else 0,
                validation_results=[ValidationResult(
                    check_name="validation_error",
                    passed=False,
                    score=0.0,
                    message=f"Validation failed: {str(e)}"
                )],
                metrics={},
                recommendations=["Fix validation errors before proceeding"]
            )
    
    def _validate_syntax(self, df: pd.DataFrame, dataset_name: str):
        """Perform syntactic validation (Tier 1)."""
        logger.debug(f"Performing syntactic validation for {dataset_name}")
        
        # Check 1: DataFrame structure
        if df is None:
            self.validation_results.append(ValidationResult(
                check_name="dataframe_exists",
                passed=False,
                score=0.0,
                message="DataFrame is None"
            ))
            return
        
        self.validation_results.append(ValidationResult(
            check_name="dataframe_exists",
            passed=True,
            score=100.0,
            message="DataFrame exists and is valid"
        ))
        
        # Check 2: Non-empty dataset
        is_empty = len(df) == 0
        self.validation_results.append(ValidationResult(
            check_name="non_empty_dataset",
            passed=not is_empty,
            score=0.0 if is_empty else 100.0,
            message=f"Dataset has {len(df)} records" if not is_empty else "Dataset is empty"
        ))
        
        # Check 3: Column structure
        has_columns = len(df.columns) > 0
        self.validation_results.append(ValidationResult(
            check_name="has_columns",
            passed=has_columns,
            score=0.0 if not has_columns else 100.0,
            message=f"Dataset has {len(df.columns)} columns" if has_columns else "No columns found"
        ))
        
        # Check 4: Data types consistency
        if not df.empty:
            mixed_types_count = 0
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if object column has mixed types
                    try:
                        unique_types = set(type(x).__name__ for x in df[col].dropna().iloc[:100])
                        if len(unique_types) > 1:
                            mixed_types_count += 1
                    except:
                        pass
            
            mixed_types_score = max(0, 100 - (mixed_types_count * 20))
            self.validation_results.append(ValidationResult(
                check_name="data_types_consistency",
                passed=mixed_types_count == 0,
                score=mixed_types_score,
                message=f"Found {mixed_types_count} columns with mixed data types",
                details={"mixed_type_columns": mixed_types_count}
            ))
    
    def _validate_semantics(self, df: pd.DataFrame, dataset_name: str):
        """Perform semantic validation (Tier 2)."""
        logger.debug(f"Performing semantic validation for {dataset_name}")
        
        if df.empty:
            return
        
        # Check 1: Date column validation
        if 'Date' in df.columns:
            date_quality = self._validate_date_column(df['Date'])
            self.validation_results.append(date_quality)
        
        # Check 2: Coordinate validation
        if 'lat' in df.columns and 'lon' in df.columns:
            coord_quality = self._validate_coordinates(df[['lat', 'lon']])
            self.validation_results.append(coord_quality)
        
        # Check 3: Numeric range validation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['lat', 'lon']:  # Skip coordinates (handled separately)
                range_quality = self._validate_numeric_ranges(df[col], col)
                self.validation_results.append(range_quality)
        
        # Check 4: Missing data patterns
        missing_quality = self._validate_missing_data_patterns(df)
        self.validation_results.append(missing_quality)
        
        # Check 5: Duplicate records
        duplicate_quality = self._validate_duplicates(df)
        self.validation_results.append(duplicate_quality)
    
    def _validate_business_rules(self, df: pd.DataFrame, dataset_name: str,
                                validation_config: Optional[Dict[str, Any]]):
        """Perform business rules validation (Tier 3)."""
        logger.debug(f"Performing business rules validation for {dataset_name}")
        
        if df.empty:
            return
        
        # Check 1: Temporal consistency
        if 'Date' in df.columns:
            temporal_quality = self._validate_temporal_consistency(df)
            self.validation_results.append(temporal_quality)
        
        # Check 2: Site-specific validation
        if 'Site' in df.columns:
            site_quality = self._validate_site_consistency(df)
            self.validation_results.append(site_quality)
        
        # Check 3: Domain-specific value ranges
        domain_quality = self._validate_domain_specific_ranges(df, dataset_name)
        self.validation_results.extend(domain_quality)
        
        # Check 4: Data completeness requirements
        completeness_quality = self._validate_completeness_requirements(df, dataset_name)
        self.validation_results.append(completeness_quality)
    
    def _validate_date_column(self, date_series: pd.Series) -> ValidationResult:
        """Validate date column quality."""
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(date_series):
                date_series = pd.to_datetime(date_series, errors='coerce')
            
            # Calculate quality metrics
            null_count = date_series.isna().sum()
            total_count = len(date_series)
            null_pct = (null_count / total_count * 100) if total_count > 0 else 100
            
            # Check date range reasonableness
            if total_count > 0 and null_count < total_count:
                min_date = date_series.min()
                max_date = date_series.max()
                
                # Reasonable range: 2000 to 2030
                reasonable_min = datetime(2000, 1, 1)
                reasonable_max = datetime(2030, 12, 31)
                
                unreasonable_count = sum([
                    (min_date < reasonable_min) if pd.notna(min_date) else 0,
                    (max_date > reasonable_max) if pd.notna(max_date) else 0
                ])
            else:
                unreasonable_count = 0
            
            # Calculate score
            score = max(0, 100 - null_pct - (unreasonable_count * 10))
            passed = score >= 70
            
            return ValidationResult(
                check_name="date_column_quality",
                passed=passed,
                score=score,
                message=f"Date quality: {null_pct:.1f}% null values, "
                       f"{unreasonable_count} unreasonable dates",
                details={
                    "null_percentage": null_pct,
                    "unreasonable_dates": unreasonable_count
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="date_column_quality",
                passed=False,
                score=0.0,
                message=f"Date validation error: {str(e)}"
            )
    
    def _validate_coordinates(self, coord_df: pd.DataFrame) -> ValidationResult:
        """Validate coordinate data quality."""
        try:
            lat_col = coord_df.iloc[:, 0]  # Assume first column is latitude
            lon_col = coord_df.iloc[:, 1]  # Assume second column is longitude
            
            # Check for null values
            lat_nulls = lat_col.isna().sum()
            lon_nulls = lon_col.isna().sum()
            total_nulls = lat_nulls + lon_nulls
            total_coords = len(coord_df) * 2
            null_pct = (total_nulls / total_coords * 100) if total_coords > 0 else 100
            
            # Check coordinate ranges
            valid_lat = ((lat_col >= -90) & (lat_col <= 90)).sum()
            valid_lon = ((lon_col >= -180) & (lon_col <= 180)).sum()
            total_valid = len(coord_df) - lat_col.isna().sum()
            
            range_score = ((valid_lat + valid_lon) / (total_valid * 2) * 100) if total_valid > 0 else 0
            
            # Overall score
            score = max(0, min(100 - null_pct, range_score))
            passed = score >= 80
            
            return ValidationResult(
                check_name="coordinate_quality",
                passed=passed,
                score=score,
                message=f"Coordinate quality: {null_pct:.1f}% missing, "
                       f"{range_score:.1f}% valid ranges",
                details={
                    "null_percentage": null_pct,
                    "valid_range_percentage": range_score
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="coordinate_quality",
                passed=False,
                score=0.0,
                message=f"Coordinate validation error: {str(e)}"
            )
    
    def _validate_numeric_ranges(self, series: pd.Series, column_name: str) -> ValidationResult:
        """Validate numeric column ranges."""
        try:
            # Skip non-numeric series
            if not pd.api.types.is_numeric_dtype(series):
                return ValidationResult(
                    check_name=f"numeric_range_{column_name}",
                    passed=True,
                    score=100.0,
                    message=f"{column_name} is not numeric - skipping range validation"
                )
            
            # Basic statistics
            null_count = series.isna().sum()
            total_count = len(series)
            null_pct = (null_count / total_count * 100) if total_count > 0 else 100
            
            if null_count == total_count:
                return ValidationResult(
                    check_name=f"numeric_range_{column_name}",
                    passed=False,
                    score=0.0,
                    message=f"{column_name}: All values are null"
                )
            
            # Check for infinite values
            finite_series = series.replace([np.inf, -np.inf], np.nan)
            inf_count = series.isna().sum() - finite_series.isna().sum()
            inf_pct = (inf_count / total_count * 100) if total_count > 0 else 0
            
            # Check for reasonable ranges (basic outlier detection)
            valid_series = finite_series.dropna()
            if len(valid_series) > 0:
                q1 = valid_series.quantile(0.25)
                q3 = valid_series.quantile(0.75)
                iqr = q3 - q1
                
                # Outliers beyond 3 IQR
                outlier_threshold = 3
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                
                outlier_count = ((valid_series < lower_bound) | (valid_series > upper_bound)).sum()
                outlier_pct = (outlier_count / len(valid_series) * 100) if len(valid_series) > 0 else 0
            else:
                outlier_pct = 0
            
            # Calculate score
            score = max(0, 100 - null_pct - inf_pct - min(outlier_pct, 20))
            passed = score >= 70
            
            return ValidationResult(
                check_name=f"numeric_range_{column_name}",
                passed=passed,
                score=score,
                message=f"{column_name}: {null_pct:.1f}% null, {inf_pct:.1f}% infinite, "
                       f"{outlier_pct:.1f}% outliers",
                details={
                    "null_percentage": null_pct,
                    "infinite_percentage": inf_pct,
                    "outlier_percentage": outlier_pct
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"numeric_range_{column_name}",
                passed=False,
                score=0.0,
                message=f"{column_name} range validation error: {str(e)}"
            )
    
    def _validate_missing_data_patterns(self, df: pd.DataFrame) -> ValidationResult:
        """Validate missing data patterns."""
        try:
            total_cells = df.size
            missing_cells = df.isna().sum().sum()
            missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 100
            
            # Check for columns with excessive missing data
            high_missing_cols = []
            for col in df.columns:
                col_missing_pct = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 100
                if col_missing_pct > 75:  # More than 75% missing
                    high_missing_cols.append(col)
            
            # Calculate score based on overall missing percentage
            if missing_pct <= 5:
                score = 100
            elif missing_pct <= 15:
                score = 85
            elif missing_pct <= 30:
                score = 70
            elif missing_pct <= 50:
                score = 50
            else:
                score = 25
            
            # Penalty for columns with excessive missing data
            score = max(0, score - len(high_missing_cols) * 10)
            
            passed = score >= 60
            
            return ValidationResult(
                check_name="missing_data_patterns",
                passed=passed,
                score=score,
                message=f"Missing data: {missing_pct:.1f}% overall, "
                       f"{len(high_missing_cols)} columns >75% missing",
                details={
                    "overall_missing_percentage": missing_pct,
                    "high_missing_columns": high_missing_cols
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="missing_data_patterns",
                passed=False,
                score=0.0,
                message=f"Missing data pattern validation error: {str(e)}"
            )
    
    def _validate_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Validate duplicate records."""
        try:
            total_records = len(df)
            if total_records == 0:
                return ValidationResult(
                    check_name="duplicate_records",
                    passed=True,
                    score=100.0,
                    message="No records to check for duplicates"
                )
            
            # Check for exact duplicates
            exact_duplicates = df.duplicated().sum()
            exact_dup_pct = (exact_duplicates / total_records * 100)
            
            # Check for potential key duplicates (Date + Site if available)
            key_duplicates = 0
            if 'Date' in df.columns and 'Site' in df.columns:
                key_duplicates = df.duplicated(subset=['Date', 'Site']).sum()
                key_dup_pct = (key_duplicates / total_records * 100)
            else:
                key_dup_pct = 0
            
            # Calculate score
            total_dup_pct = max(exact_dup_pct, key_dup_pct)
            score = max(0, 100 - total_dup_pct * 2)  # 2x penalty for duplicates
            passed = total_dup_pct < 5  # Less than 5% duplicates acceptable
            
            return ValidationResult(
                check_name="duplicate_records",
                passed=passed,
                score=score,
                message=f"Duplicates: {exact_dup_pct:.1f}% exact, {key_dup_pct:.1f}% key duplicates",
                details={
                    "exact_duplicate_percentage": exact_dup_pct,
                    "key_duplicate_percentage": key_dup_pct
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="duplicate_records",
                passed=False,
                score=0.0,
                message=f"Duplicate validation error: {str(e)}"
            )
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Validate temporal consistency."""
        try:
            if 'Date' not in df.columns:
                return ValidationResult(
                    check_name="temporal_consistency",
                    passed=True,
                    score=100.0,
                    message="No Date column - skipping temporal validation"
                )
            
            date_series = pd.to_datetime(df['Date'], errors='coerce')
            valid_dates = date_series.dropna()
            
            if len(valid_dates) < 2:
                return ValidationResult(
                    check_name="temporal_consistency",
                    passed=False,
                    score=50.0,
                    message="Insufficient valid dates for temporal consistency check"
                )
            
            # Check for chronological order (within sites if available)
            order_violations = 0
            total_sequences = 0
            
            if 'Site' in df.columns:
                # Check order within each site
                for site in df['Site'].unique():
                    site_data = df[df['Site'] == site].copy()
                    site_dates = pd.to_datetime(site_data['Date'], errors='coerce').dropna()
                    
                    if len(site_dates) > 1:
                        total_sequences += 1
                        # Check if dates are sorted
                        if not site_dates.is_monotonic_increasing:
                            order_violations += 1
            else:
                # Check overall order
                total_sequences = 1
                if not valid_dates.is_monotonic_increasing:
                    order_violations = 1
            
            # Calculate score
            if total_sequences == 0:
                score = 100.0
            else:
                order_score = (1 - order_violations / total_sequences) * 100
                score = order_score
            
            passed = score >= 80
            
            return ValidationResult(
                check_name="temporal_consistency",
                passed=passed,
                score=score,
                message=f"Temporal consistency: {order_violations}/{total_sequences} "
                       f"sequences out of order",
                details={
                    "order_violations": order_violations,
                    "total_sequences": total_sequences
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="temporal_consistency",
                passed=False,
                score=0.0,
                message=f"Temporal consistency validation error: {str(e)}"
            )
    
    def _validate_site_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Validate site data consistency."""
        try:
            if 'Site' not in df.columns:
                return ValidationResult(
                    check_name="site_consistency",
                    passed=True,
                    score=100.0,
                    message="No Site column - skipping site validation"
                )
            
            sites = df['Site'].dropna()
            total_records = len(sites)
            
            if total_records == 0:
                return ValidationResult(
                    check_name="site_consistency",
                    passed=False,
                    score=0.0,
                    message="No valid site data found"
                )
            
            # Check site name consistency (variations of same site)
            unique_sites = sites.unique()
            
            # Basic site name normalization check
            normalized_sites = sites.str.lower().str.strip().str.replace(' ', '')
            unique_normalized = normalized_sites.unique()
            
            # If normalized count is significantly different, might have inconsistent naming
            naming_inconsistency = len(unique_sites) - len(unique_normalized)
            inconsistency_pct = (naming_inconsistency / len(unique_sites) * 100) if len(unique_sites) > 0 else 0
            
            # Check for expected sites from configuration
            expected_sites = set()
            if hasattr(config, 'SITES'):
                expected_sites = set(config.SITES.keys())
                
            found_sites = set(unique_sites)
            missing_expected = expected_sites - found_sites
            unexpected_sites = found_sites - expected_sites if expected_sites else set()
            
            # Calculate score
            consistency_score = max(0, 100 - inconsistency_pct * 2)
            coverage_penalty = len(missing_expected) * 5
            unexpected_penalty = len(unexpected_sites) * 2
            
            score = max(0, consistency_score - coverage_penalty - unexpected_penalty)
            passed = score >= 70
            
            return ValidationResult(
                check_name="site_consistency",
                passed=passed,
                score=score,
                message=f"Site consistency: {len(unique_sites)} unique sites, "
                       f"{inconsistency_pct:.1f}% naming inconsistency",
                details={
                    "unique_sites": len(unique_sites),
                    "naming_inconsistency_percentage": inconsistency_pct,
                    "missing_expected_sites": list(missing_expected),
                    "unexpected_sites": list(unexpected_sites)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="site_consistency",
                passed=False,
                score=0.0,
                message=f"Site consistency validation error: {str(e)}"
            )
    
    def _validate_domain_specific_ranges(self, df: pd.DataFrame, 
                                       dataset_name: str) -> List[ValidationResult]:
        """Validate domain-specific value ranges."""
        results = []
        
        try:
            # Domain-specific validation rules
            domain_rules = {
                'DA_Levels': {'min': 0, 'max': 1000, 'unit': 'μg/g'},
                'PN_Levels': {'min': 0, 'max': 1000000, 'unit': 'cells/L'},
                'oni': {'min': -3, 'max': 3, 'unit': 'index'},
                'pdo': {'min': -3, 'max': 3, 'unit': 'index'},
                'discharge': {'min': 0, 'max': 1000000, 'unit': 'cfs'},
                'beuti': {'min': -100, 'max': 100, 'unit': 'index'}
            }
            
            # Check each domain-specific column
            for col_name, rules in domain_rules.items():
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    
                    if len(series) == 0:
                        results.append(ValidationResult(
                            check_name=f"domain_range_{col_name}",
                            passed=True,
                            score=100.0,
                            message=f"{col_name}: No data to validate"
                        ))
                        continue
                    
                    # Check range violations
                    below_min = (series < rules['min']).sum()
                    above_max = (series > rules['max']).sum()
                    total_violations = below_min + above_max
                    violation_pct = (total_violations / len(series) * 100) if len(series) > 0 else 0
                    
                    # Calculate score
                    score = max(0, 100 - violation_pct * 2)
                    passed = violation_pct < 5  # Less than 5% violations acceptable
                    
                    results.append(ValidationResult(
                        check_name=f"domain_range_{col_name}",
                        passed=passed,
                        score=score,
                        message=f"{col_name}: {violation_pct:.1f}% outside expected range "
                               f"({rules['min']}-{rules['max']} {rules['unit']})",
                        details={
                            "below_min": int(below_min),
                            "above_max": int(above_max),
                            "violation_percentage": violation_pct,
                            "expected_range": f"{rules['min']}-{rules['max']} {rules['unit']}"
                        }
                    ))
            
            return results
            
        except Exception as e:
            return [ValidationResult(
                check_name="domain_range_validation",
                passed=False,
                score=0.0,
                message=f"Domain range validation error: {str(e)}"
            )]
    
    def _validate_completeness_requirements(self, df: pd.DataFrame, 
                                          dataset_name: str) -> ValidationResult:
        """Validate data completeness requirements."""
        try:
            # Define completeness requirements by dataset type
            completeness_rules = {
                'climate': {'required_columns': ['Date'], 'min_records': 50},
                'satellite': {'required_columns': ['Date'], 'min_records': 100},
                'toxin': {'required_columns': ['Date', 'Site'], 'min_records': 10},
                'final': {'required_columns': ['Date', 'Site'], 'min_records': 100}
            }
            
            # Determine dataset type from name
            dataset_type = 'final'  # Default
            for key in completeness_rules.keys():
                if key in dataset_name.lower():
                    dataset_type = key
                    break
            
            rules = completeness_rules[dataset_type]
            
            # Check required columns
            missing_columns = [col for col in rules['required_columns'] if col not in df.columns]
            columns_score = 0 if missing_columns else 100
            
            # Check minimum records
            record_count = len(df)
            min_records = rules['min_records']
            
            if record_count >= min_records:
                records_score = 100
            elif record_count >= min_records * 0.5:  # At least 50% of minimum
                records_score = 70
            else:
                records_score = 30
            
            # Overall score
            overall_score = (columns_score + records_score) / 2
            passed = overall_score >= 70
            
            return ValidationResult(
                check_name="completeness_requirements",
                passed=passed,
                score=overall_score,
                message=f"Completeness: {record_count}/{min_records} records, "
                       f"{len(missing_columns)} missing required columns",
                details={
                    "record_count": record_count,
                    "minimum_required": min_records,
                    "missing_columns": missing_columns,
                    "dataset_type": dataset_type
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="completeness_requirements",
                passed=False,
                score=0.0,
                message=f"Completeness validation error: {str(e)}"
            )
    
    def _generate_quality_report(self, df: pd.DataFrame, dataset_name: str) -> QualityReport:
        """Generate comprehensive quality report."""
        # Calculate overall score (weighted average)
        if not self.validation_results:
            overall_score = 0.0
        else:
            total_score = sum(result.score for result in self.validation_results)
            overall_score = total_score / len(self.validation_results)
        
        # Determine quality level
        if overall_score >= 95:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 85:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 70:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_score >= 50:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.CRITICAL
        
        # Generate metrics
        metrics = {
            'total_validations': len(self.validation_results),
            'passed_validations': sum(1 for r in self.validation_results if r.passed),
            'failed_validations': sum(1 for r in self.validation_results if not r.passed),
            'average_score': overall_score
        }
        
        if not df.empty:
            metrics.update({
                'dataset_shape': list(df.shape),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'null_percentage': (df.isna().sum().sum() / df.size * 100) if df.size > 0 else 100
            })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_level, self.validation_results)
        
        return QualityReport(
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            quality_level=quality_level,
            total_records=len(df) if not df.empty else 0,
            validation_results=self.validation_results,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, quality_level: QualityLevel, 
                                 validation_results: List[ValidationResult]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # General recommendations based on quality level
        if quality_level == QualityLevel.CRITICAL:
            recommendations.append("CRITICAL: Dataset requires immediate attention before use")
            recommendations.append("Review all failed validations and data sources")
            
        elif quality_level == QualityLevel.POOR:
            recommendations.append("Dataset quality is poor - implement data cleaning")
            recommendations.append("Consider additional data validation steps")
            
        elif quality_level == QualityLevel.ACCEPTABLE:
            recommendations.append("Dataset is acceptable but has room for improvement")
            
        # Specific recommendations based on validation results
        failed_checks = [r for r in validation_results if not r.passed]
        
        for check in failed_checks:
            if 'missing_data' in check.check_name:
                recommendations.append("Implement data imputation strategies for missing values")
            elif 'duplicate' in check.check_name:
                recommendations.append("Remove or consolidate duplicate records")
            elif 'range' in check.check_name:
                recommendations.append(f"Review {check.check_name} - values outside expected ranges")
            elif 'temporal' in check.check_name:
                recommendations.append("Fix temporal ordering issues in dataset")
            elif 'site' in check.check_name:
                recommendations.append("Standardize site naming and check site consistency")
        
        # Add monitoring recommendations
        if quality_level.value in ['good', 'excellent']:
            recommendations.append("Continue monitoring data quality trends")
            recommendations.append("Set up automated quality alerts for production")
        
        return recommendations


class QualityMonitor:
    """
    Monitors data quality over time and generates alerts.
    """
    
    def __init__(self, report_dir: str = "./outputs/quality_reports/"):
        """
        Initialize quality monitor.
        
        Args:
            report_dir: Directory to store quality reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized QualityMonitor with report directory: {report_dir}")
    
    def save_report(self, report: QualityReport) -> str:
        """
        Save quality report to file.
        
        Args:
            report: Quality report to save
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_report_{report.dataset_name}_{timestamp}.json"
        filepath = self.report_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.info(f"Quality report saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise
    
    def load_report(self, filepath: str) -> QualityReport:
        """
        Load quality report from file.
        
        Args:
            filepath: Path to report file
            
        Returns:
            Loaded quality report
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct validation results
            validation_results = [
                ValidationResult(**vr) for vr in data['validation_results']
            ]
            
            # Reconstruct quality report
            report = QualityReport(
                dataset_name=data['dataset_name'],
                timestamp=data['timestamp'],
                overall_score=data['overall_score'],
                quality_level=QualityLevel(data['quality_level']),
                total_records=data['total_records'],
                validation_results=validation_results,
                metrics=data['metrics'],
                recommendations=data['recommendations']
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error loading quality report: {e}")
            raise
    
    def generate_quality_alert(self, report: QualityReport) -> Optional[Dict[str, Any]]:
        """
        Generate quality alert if thresholds are exceeded.
        
        Args:
            report: Quality report to check
            
        Returns:
            Alert dictionary if alert conditions are met, None otherwise
        """
        alert_conditions = [
            (report.quality_level == QualityLevel.CRITICAL, "CRITICAL"),
            (report.quality_level == QualityLevel.POOR, "WARNING"),
            (report.overall_score < 50, "LOW_QUALITY")
        ]
        
        alerts = []
        for condition, alert_type in alert_conditions:
            if condition:
                alerts.append({
                    'type': alert_type,
                    'message': f"Data quality alert for {report.dataset_name}",
                    'quality_level': report.quality_level.value,
                    'score': report.overall_score,
                    'timestamp': report.timestamp,
                    'recommendations': report.recommendations[:3]  # Top 3 recommendations
                })
        
        if alerts:
            logger.warning(f"Quality alerts generated for {report.dataset_name}: {len(alerts)} alerts")
            return alerts
        
        return None