"""
Data Freshness Validation
=========================

Validates data freshness and availability timing for the DATect forecasting
system. Ensures that data is current enough for reliable predictions while
respecting temporal safeguards.

This module provides:
- Data freshness validation with configurable thresholds
- Age-based data quality scoring
- Data availability timing checks
- Freshness trend monitoring
- Automated freshness alerts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger
from .exception_handling import ScientificValidationError
import config

logger = get_logger(__name__)


class FreshnessLevel(Enum):
    """Data freshness levels for classification."""
    FRESH = "fresh"          # Data is current and recent
    ACCEPTABLE = "acceptable"  # Data is somewhat old but usable  
    STALE = "stale"          # Data is old and may affect quality
    EXPIRED = "expired"      # Data is too old for reliable predictions


@dataclass
class FreshnessThreshold:
    """Configuration for data freshness thresholds."""
    data_source: str
    fresh_hours: int          # Hours for "fresh" classification
    acceptable_hours: int     # Hours for "acceptable" classification  
    stale_hours: int         # Hours for "stale" classification
    expired_hours: int       # Hours beyond which data is "expired"
    critical: bool = False   # Whether this data source is critical


@dataclass
class FreshnessResult:
    """Result of data freshness validation."""
    data_source: str
    freshness_level: FreshnessLevel
    age_hours: float
    threshold_config: FreshnessThreshold
    score: float  # 0-100 based on freshness
    message: str
    timestamp: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class DataFreshnessValidator:
    """
    Validates data freshness against configurable thresholds.
    
    Provides freshness scoring and classification for different data sources
    with domain-specific thresholds and requirements.
    """
    
    def __init__(self):
        """Initialize freshness validator with default thresholds."""
        self.thresholds = self._load_default_thresholds()
        logger.info("Initialized DataFreshnessValidator")
    
    def _load_default_thresholds(self) -> Dict[str, FreshnessThreshold]:
        """Load default freshness thresholds for different data sources."""
        return {
            'satellite': FreshnessThreshold(
                data_source='satellite',
                fresh_hours=24,      # Daily MODIS data
                acceptable_hours=72, # Up to 3 days acceptable
                stale_hours=168,     # 1 week stale
                expired_hours=720,   # 30 days expired
                critical=True        # Satellite data is critical
            ),
            'climate_pdo': FreshnessThreshold(
                data_source='climate_pdo',
                fresh_hours=720,     # Monthly data - 30 days fresh
                acceptable_hours=1440, # Up to 60 days acceptable  
                stale_hours=2160,    # 90 days stale
                expired_hours=4320,  # 180 days expired
                critical=False
            ),
            'climate_oni': FreshnessThreshold(
                data_source='climate_oni',
                fresh_hours=720,     # Monthly data - 30 days fresh
                acceptable_hours=1440, # Up to 60 days acceptable
                stale_hours=2160,    # 90 days stale  
                expired_hours=4320,  # 180 days expired
                critical=False
            ),
            'streamflow': FreshnessThreshold(
                data_source='streamflow',
                fresh_hours=24,      # Daily data
                acceptable_hours=72, # Up to 3 days acceptable
                stale_hours=168,     # 1 week stale
                expired_hours=720,   # 30 days expired
                critical=True        # Streamflow data is important
            ),
            'beuti': FreshnessThreshold(
                data_source='beuti',
                fresh_hours=168,     # Weekly data - 1 week fresh
                acceptable_hours=336, # Up to 2 weeks acceptable
                stale_hours=672,     # 4 weeks stale
                expired_hours=1344,  # 8 weeks expired
                critical=False
            ),
            'toxin': FreshnessThreshold(
                data_source='toxin',
                fresh_hours=168,     # Weekly sampling
                acceptable_hours=336, # Up to 2 weeks acceptable
                stale_hours=672,     # 4 weeks stale
                expired_hours=1344,  # 8 weeks expired
                critical=True        # Target data is critical
            )
        }
    
    def validate_data_freshness(self, df: pd.DataFrame, data_source: str,
                               date_column: str = 'Date') -> FreshnessResult:
        """
        Validate freshness of a dataset.
        
        Args:
            df: DataFrame to validate
            data_source: Source identifier for threshold lookup
            date_column: Name of the date column
            
        Returns:
            Freshness validation result
        """
        logger.info(f"Validating data freshness for {data_source}")
        
        try:
            # Get threshold configuration
            threshold = self.thresholds.get(data_source)
            if threshold is None:
                # Use generic threshold for unknown sources
                threshold = FreshnessThreshold(
                    data_source=data_source,
                    fresh_hours=24,
                    acceptable_hours=168,
                    stale_hours=720,
                    expired_hours=2160
                )
                logger.warning(f"Using default thresholds for unknown source: {data_source}")
            
            # Calculate data age
            age_hours = self._calculate_data_age(df, date_column)
            
            # Determine freshness level
            freshness_level = self._classify_freshness(age_hours, threshold)
            
            # Calculate freshness score
            score = self._calculate_freshness_score(age_hours, threshold)
            
            # Generate message
            message = self._generate_freshness_message(freshness_level, age_hours, threshold)
            
            result = FreshnessResult(
                data_source=data_source,
                freshness_level=freshness_level,
                age_hours=age_hours,
                threshold_config=threshold,
                score=score,
                message=message,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Freshness validation for {data_source}: "
                       f"{freshness_level.value} ({score:.1f}%, {age_hours:.1f}h old)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating freshness for {data_source}: {e}")
            
            # Return expired result on error
            return FreshnessResult(
                data_source=data_source,
                freshness_level=FreshnessLevel.EXPIRED,
                age_hours=float('inf'),
                threshold_config=threshold,
                score=0.0,
                message=f"Freshness validation error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _calculate_data_age(self, df: pd.DataFrame, date_column: str) -> float:
        """
        Calculate the age of the most recent data in hours.
        
        Args:
            df: DataFrame with date data
            date_column: Name of the date column
            
        Returns:
            Age in hours of the most recent data point
        """
        if df.empty or date_column not in df.columns:
            logger.warning("Empty dataset or missing date column")
            return float('inf')
        
        try:
            # Convert to datetime if needed
            date_series = pd.to_datetime(df[date_column], errors='coerce')
            
            # Remove null dates
            valid_dates = date_series.dropna()
            
            if len(valid_dates) == 0:
                logger.warning("No valid dates found in dataset")
                return float('inf')
            
            # Find most recent date
            most_recent = valid_dates.max()
            current_time = datetime.now()
            
            # Handle timezone-naive datetimes
            if most_recent.tzinfo is not None:
                # Convert to naive datetime for comparison
                most_recent = most_recent.replace(tzinfo=None)
            
            # Calculate age in hours
            age_delta = current_time - most_recent
            age_hours = age_delta.total_seconds() / 3600
            
            logger.debug(f"Most recent data: {most_recent}, Age: {age_hours:.1f} hours")
            
            return max(0, age_hours)  # Ensure non-negative age
            
        except Exception as e:
            logger.error(f"Error calculating data age: {e}")
            return float('inf')
    
    def _classify_freshness(self, age_hours: float, 
                          threshold: FreshnessThreshold) -> FreshnessLevel:
        """
        Classify data freshness based on age and thresholds.
        
        Args:
            age_hours: Age of data in hours
            threshold: Threshold configuration
            
        Returns:
            Freshness level classification
        """
        if age_hours <= threshold.fresh_hours:
            return FreshnessLevel.FRESH
        elif age_hours <= threshold.acceptable_hours:
            return FreshnessLevel.ACCEPTABLE
        elif age_hours <= threshold.stale_hours:
            return FreshnessLevel.STALE
        else:
            return FreshnessLevel.EXPIRED
    
    def _calculate_freshness_score(self, age_hours: float,
                                  threshold: FreshnessThreshold) -> float:
        """
        Calculate numeric freshness score (0-100).
        
        Args:
            age_hours: Age of data in hours
            threshold: Threshold configuration
            
        Returns:
            Freshness score from 0-100
        """
        if age_hours <= threshold.fresh_hours:
            # Fresh data: 85-100 score based on how recent
            relative_age = age_hours / threshold.fresh_hours
            return 100 - (relative_age * 15)  # Linear decrease from 100 to 85
            
        elif age_hours <= threshold.acceptable_hours:
            # Acceptable data: 70-85 score
            relative_age = (age_hours - threshold.fresh_hours) / (threshold.acceptable_hours - threshold.fresh_hours)
            return 85 - (relative_age * 15)  # Linear decrease from 85 to 70
            
        elif age_hours <= threshold.stale_hours:
            # Stale data: 40-70 score
            relative_age = (age_hours - threshold.acceptable_hours) / (threshold.stale_hours - threshold.acceptable_hours)
            return 70 - (relative_age * 30)  # Linear decrease from 70 to 40
            
        else:
            # Expired data: 0-40 score
            if age_hours <= threshold.expired_hours:
                relative_age = (age_hours - threshold.stale_hours) / (threshold.expired_hours - threshold.stale_hours)
                return 40 - (relative_age * 40)  # Linear decrease from 40 to 0
            else:
                return 0.0  # Completely expired
    
    def _generate_freshness_message(self, freshness_level: FreshnessLevel,
                                   age_hours: float,
                                   threshold: FreshnessThreshold) -> str:
        """
        Generate human-readable freshness message.
        
        Args:
            freshness_level: Freshness classification
            age_hours: Age in hours
            threshold: Threshold configuration
            
        Returns:
            Descriptive message about data freshness
        """
        # Convert hours to human-readable format
        if age_hours < 24:
            age_str = f"{age_hours:.1f} hours"
        elif age_hours < 168:  # Less than a week
            age_str = f"{age_hours/24:.1f} days"
        elif age_hours < 720:  # Less than a month
            age_str = f"{age_hours/168:.1f} weeks"
        else:
            age_str = f"{age_hours/720:.1f} months"
        
        messages = {
            FreshnessLevel.FRESH: f"Data is fresh ({age_str} old)",
            FreshnessLevel.ACCEPTABLE: f"Data is acceptable ({age_str} old)",
            FreshnessLevel.STALE: f"Data is stale ({age_str} old) - consider refreshing",
            FreshnessLevel.EXPIRED: f"Data is expired ({age_str} old) - refresh required"
        }
        
        base_message = messages.get(freshness_level, f"Data age: {age_str}")
        
        # Add criticality warning for critical sources
        if threshold.critical and freshness_level in [FreshnessLevel.STALE, FreshnessLevel.EXPIRED]:
            base_message += " - CRITICAL data source"
        
        return base_message
    
    def validate_multiple_sources(self, datasets: Dict[str, pd.DataFrame],
                                 date_column: str = 'Date') -> Dict[str, FreshnessResult]:
        """
        Validate freshness for multiple data sources.
        
        Args:
            datasets: Dictionary mapping source names to DataFrames
            date_column: Name of the date column in datasets
            
        Returns:
            Dictionary mapping source names to freshness results
        """
        logger.info(f"Validating freshness for {len(datasets)} data sources")
        
        results = {}
        
        for source_name, df in datasets.items():
            try:
                result = self.validate_data_freshness(df, source_name, date_column)
                results[source_name] = result
            except Exception as e:
                logger.error(f"Error validating {source_name}: {e}")
                results[source_name] = FreshnessResult(
                    data_source=source_name,
                    freshness_level=FreshnessLevel.EXPIRED,
                    age_hours=float('inf'),
                    threshold_config=self.thresholds.get(source_name),
                    score=0.0,
                    message=f"Validation error: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
        
        # Log summary
        fresh_count = sum(1 for r in results.values() if r.freshness_level == FreshnessLevel.FRESH)
        stale_count = sum(1 for r in results.values() if r.freshness_level in [FreshnessLevel.STALE, FreshnessLevel.EXPIRED])
        
        logger.info(f"Freshness summary: {fresh_count} fresh, {stale_count} stale/expired")
        
        return results
    
    def generate_freshness_report(self, results: Dict[str, FreshnessResult]) -> Dict[str, Any]:
        """
        Generate comprehensive freshness report.
        
        Args:
            results: Dictionary of freshness results
            
        Returns:
            Comprehensive freshness report
        """
        if not results:
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': 'No data sources to evaluate',
                'overall_score': 0.0,
                'recommendations': ['Add data sources for freshness monitoring']
            }
        
        # Calculate overall metrics
        overall_score = np.mean([r.score for r in results.values()])
        
        # Count by freshness level
        level_counts = {}
        for level in FreshnessLevel:
            level_counts[level.value] = sum(1 for r in results.values() if r.freshness_level == level)
        
        # Identify critical issues
        critical_issues = []
        critical_sources = []
        
        for source, result in results.items():
            if result.threshold_config and result.threshold_config.critical:
                critical_sources.append(source)
                if result.freshness_level in [FreshnessLevel.STALE, FreshnessLevel.EXPIRED]:
                    critical_issues.append(f"{source}: {result.freshness_level.value}")
        
        # Generate recommendations
        recommendations = []
        
        if critical_issues:
            recommendations.append(f"URGENT: Refresh critical data sources: {', '.join(critical_issues)}")
        
        stale_sources = [s for s, r in results.items() if r.freshness_level == FreshnessLevel.STALE]
        if stale_sources:
            recommendations.append(f"Consider refreshing stale sources: {', '.join(stale_sources)}")
        
        expired_sources = [s for s, r in results.items() if r.freshness_level == FreshnessLevel.EXPIRED]
        if expired_sources:
            recommendations.append(f"Expired sources require immediate refresh: {', '.join(expired_sources)}")
        
        if overall_score >= 80:
            recommendations.append("Data freshness is good - continue monitoring")
        elif overall_score >= 60:
            recommendations.append("Data freshness is acceptable - monitor trends")
        else:
            recommendations.append("Poor data freshness - implement more frequent updates")
        
        # Generate detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'total_sources': len(results),
            'critical_sources': len(critical_sources),
            'critical_issues': len(critical_issues),
            'freshness_distribution': level_counts,
            'source_details': {
                source: {
                    'freshness_level': result.freshness_level.value,
                    'score': result.score,
                    'age_hours': result.age_hours,
                    'message': result.message,
                    'critical': result.threshold_config.critical if result.threshold_config else False
                }
                for source, result in results.items()
            },
            'recommendations': recommendations
        }
        
        logger.info(f"Freshness report generated: {overall_score:.1f}% overall, "
                   f"{len(critical_issues)} critical issues")
        
        return report
    
    def update_threshold(self, data_source: str, threshold: FreshnessThreshold):
        """
        Update freshness threshold for a data source.
        
        Args:
            data_source: Name of the data source
            threshold: New threshold configuration
        """
        self.thresholds[data_source] = threshold
        logger.info(f"Updated freshness threshold for {data_source}")
    
    def get_threshold(self, data_source: str) -> Optional[FreshnessThreshold]:
        """
        Get freshness threshold for a data source.
        
        Args:
            data_source: Name of the data source
            
        Returns:
            Threshold configuration or None if not found
        """
        return self.thresholds.get(data_source)


def validate_pipeline_freshness(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Convenience function to validate freshness for entire pipeline.
    
    Args:
        datasets: Dictionary mapping source names to DataFrames
        
    Returns:
        Comprehensive freshness report
    """
    validator = DataFreshnessValidator()
    results = validator.validate_multiple_sources(datasets)
    report = validator.generate_freshness_report(results)
    
    return report