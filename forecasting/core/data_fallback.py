"""
Data Fallback Strategies
========================

Provides comprehensive fallback mechanisms for data retrieval and processing
when primary data sources fail or are temporarily unavailable.

This module provides:
- Cached data fallback with configurable expiration
- Alternative data source routing
- Data interpolation and gap-filling strategies  
- Graceful degradation for missing data sources
- Historical data substitution mechanisms
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import pickle
import hashlib

from .logging_config import get_logger
from .exception_handling import handle_data_errors, ScientificValidationError
from .data_freshness import DataFreshnessValidator, FreshnessLevel
import config

logger = get_logger(__name__)


class FallbackStrategy(Enum):
    """Data fallback strategy types."""
    CACHE = "cache"                    # Use cached data
    INTERPOLATION = "interpolation"    # Fill gaps with interpolation
    HISTORICAL = "historical"          # Use historical data from same period
    ALTERNATIVE_SOURCE = "alt_source"  # Use alternative data source
    DEFAULT_VALUES = "defaults"        # Use scientifically reasonable defaults
    SKIP = "skip"                     # Skip processing for this data source


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies."""
    primary_strategy: FallbackStrategy = FallbackStrategy.CACHE
    secondary_strategy: FallbackStrategy = FallbackStrategy.INTERPOLATION
    tertiary_strategy: FallbackStrategy = FallbackStrategy.HISTORICAL
    
    # Cache configuration
    cache_max_age_days: int = 7        # Max age for cached data
    cache_directory: str = "./data/cache"
    
    # Interpolation configuration
    interpolation_max_gap_days: int = 14    # Max gap to interpolate
    interpolation_method: str = "linear"    # pandas interpolation method
    
    # Historical substitution configuration
    historical_year_range: int = 3     # Years to look back for historical data
    historical_tolerance_days: int = 7  # Days tolerance for matching dates
    
    # Default values for scientific parameters
    default_values: Dict[str, float] = field(default_factory=lambda: {
        'sst': 12.0,        # Sea surface temperature (°C)
        'chlorophyll': 2.0,  # Chlorophyll-a concentration (mg/m³)
        'pdo': 0.0,         # Pacific Decadal Oscillation index
        'oni': 0.0,         # Oceanic Niño Index
        'beuti': 50.0,      # Biologically Effective Upwelling Transport Index
        'streamflow': 100.0  # Streamflow (cubic feet per second)
    })


@dataclass
class FallbackResult:
    """Result of fallback strategy application."""
    success: bool
    data: Optional[pd.DataFrame] = None
    strategy_used: Optional[FallbackStrategy] = None
    fallback_quality: float = 0.0  # Quality score 0-100
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.message and self.strategy_used:
            self.message = f"Applied {self.strategy_used.value} fallback strategy"


class DataFallbackManager:
    """
    Manages comprehensive data fallback strategies for resilient data processing.
    
    Provides multiple fallback mechanisms when primary data sources fail,
    ensuring the pipeline can continue operating with degraded but scientifically
    reasonable data.
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        """
        Initialize fallback manager.
        
        Args:
            config: Fallback configuration
        """
        self.config = config or FallbackConfig()
        self.cache_dir = Path(self.config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.freshness_validator = DataFreshnessValidator()
        
        # Track fallback usage for monitoring
        self.fallback_stats = {
            'total_requests': 0,
            'fallback_used': 0,
            'strategy_usage': {strategy.value: 0 for strategy in FallbackStrategy}
        }
        
        logger.info(f"Initialized DataFallbackManager with cache directory: {self.cache_dir}")
    
    @handle_data_errors
    def get_data_with_fallback(self, 
                             data_source: str,
                             fetch_function: Callable,
                             *args, **kwargs) -> FallbackResult:
        """
        Attempt to get data with comprehensive fallback strategies.
        
        Args:
            data_source: Name/identifier of the data source
            fetch_function: Primary function to fetch data
            *args, **kwargs: Arguments for fetch function
            
        Returns:
            FallbackResult with data and metadata
        """
        logger.info(f"Attempting to fetch data for {data_source} with fallback support")
        self.fallback_stats['total_requests'] += 1
        
        # Try primary data source first
        try:
            data = fetch_function(*args, **kwargs)
            if self._is_data_valid(data):
                logger.info(f"Primary data source succeeded for {data_source}")
                return FallbackResult(
                    success=True,
                    data=data,
                    strategy_used=None,
                    fallback_quality=100.0,
                    message="Primary data source successful"
                )
        except Exception as e:
            logger.warning(f"Primary data source failed for {data_source}: {e}")
        
        # Primary failed, try fallback strategies
        self.fallback_stats['fallback_used'] += 1
        
        strategies = [
            self.config.primary_strategy,
            self.config.secondary_strategy,
            self.config.tertiary_strategy
        ]
        
        for strategy in strategies:
            if strategy == FallbackStrategy.SKIP:
                continue
                
            try:
                result = self._apply_fallback_strategy(
                    data_source, strategy, fetch_function, *args, **kwargs
                )
                
                if result.success and self._is_data_valid(result.data):
                    self.fallback_stats['strategy_usage'][strategy.value] += 1
                    logger.info(f"Fallback strategy {strategy.value} succeeded for {data_source}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.value} failed for {data_source}: {e}")
        
        # All strategies failed
        logger.error(f"All fallback strategies failed for {data_source}")
        return FallbackResult(
            success=False,
            message=f"All fallback strategies failed for {data_source}"
        )
    
    def _apply_fallback_strategy(self, 
                               data_source: str, 
                               strategy: FallbackStrategy,
                               fetch_function: Callable,
                               *args, **kwargs) -> FallbackResult:
        """Apply specific fallback strategy."""
        logger.debug(f"Applying {strategy.value} fallback for {data_source}")
        
        if strategy == FallbackStrategy.CACHE:
            return self._fallback_to_cache(data_source)
            
        elif strategy == FallbackStrategy.INTERPOLATION:
            return self._fallback_to_interpolation(data_source)
            
        elif strategy == FallbackStrategy.HISTORICAL:
            return self._fallback_to_historical(data_source)
            
        elif strategy == FallbackStrategy.ALTERNATIVE_SOURCE:
            return self._fallback_to_alternative_source(data_source, fetch_function, *args, **kwargs)
            
        elif strategy == FallbackStrategy.DEFAULT_VALUES:
            return self._fallback_to_defaults(data_source)
            
        else:
            return FallbackResult(
                success=False,
                message=f"Unknown fallback strategy: {strategy}"
            )
    
    def _fallback_to_cache(self, data_source: str) -> FallbackResult:
        """Attempt to use cached data."""
        cache_file = self.cache_dir / f"{data_source}_cache.pkl"
        
        if not cache_file.exists():
            return FallbackResult(
                success=False,
                message=f"No cache file found for {data_source}"
            )
        
        try:
            # Load cached data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            cached_at = cache_data.get('timestamp')
            data = cache_data.get('data')
            
            if not cached_at or not isinstance(data, pd.DataFrame):
                return FallbackResult(
                    success=False,
                    message=f"Invalid cache format for {data_source}"
                )
            
            # Check cache age
            cache_age = datetime.now() - cached_at
            if cache_age.days > self.config.cache_max_age_days:
                return FallbackResult(
                    success=False,
                    message=f"Cache too old for {data_source}: {cache_age.days} days"
                )
            
            # Calculate quality based on cache age
            quality = max(50.0, 100.0 * (1 - cache_age.days / self.config.cache_max_age_days))
            
            return FallbackResult(
                success=True,
                data=data,
                strategy_used=FallbackStrategy.CACHE,
                fallback_quality=quality,
                message=f"Used cached data from {cached_at.strftime('%Y-%m-%d %H:%M')}",
                metadata={'cache_age_days': cache_age.days}
            )
            
        except Exception as e:
            return FallbackResult(
                success=False,
                message=f"Error reading cache for {data_source}: {e}"
            )
    
    def _fallback_to_interpolation(self, data_source: str) -> FallbackResult:
        """Attempt to use interpolated data from recent successful fetches."""
        # Look for recent partial data that can be interpolated
        recent_data_file = self.cache_dir / f"{data_source}_recent.pkl"
        
        if not recent_data_file.exists():
            return FallbackResult(
                success=False,
                message=f"No recent data found for interpolation: {data_source}"
            )
        
        try:
            with open(recent_data_file, 'rb') as f:
                recent_data = pickle.load(f)
            
            data = recent_data.get('data')
            if not isinstance(data, pd.DataFrame) or len(data) < 2:
                return FallbackResult(
                    success=False,
                    message=f"Insufficient data for interpolation: {data_source}"
                )
            
            # Apply interpolation
            interpolated_data = self._perform_interpolation(data, data_source)
            
            if interpolated_data is not None and len(interpolated_data) > 0:
                quality = 70.0  # Interpolated data has moderate quality
                return FallbackResult(
                    success=True,
                    data=interpolated_data,
                    strategy_used=FallbackStrategy.INTERPOLATION,
                    fallback_quality=quality,
                    message=f"Applied {self.config.interpolation_method} interpolation",
                    metadata={'interpolation_method': self.config.interpolation_method}
                )
            else:
                return FallbackResult(
                    success=False,
                    message=f"Interpolation failed for {data_source}"
                )
                
        except Exception as e:
            return FallbackResult(
                success=False,
                message=f"Error in interpolation fallback for {data_source}: {e}"
            )
    
    def _fallback_to_historical(self, data_source: str) -> FallbackResult:
        """Use historical data from the same time period in previous years."""
        historical_file = self.cache_dir / f"{data_source}_historical.pkl"
        
        if not historical_file.exists():
            return FallbackResult(
                success=False,
                message=f"No historical data available for {data_source}"
            )
        
        try:
            with open(historical_file, 'rb') as f:
                historical_data = pickle.load(f)
            
            # Find data from same period in previous years
            current_date = datetime.now()
            target_month_day = (current_date.month, current_date.day)
            
            matched_data = []
            for year_offset in range(1, self.config.historical_year_range + 1):
                target_year = current_date.year - year_offset
                
                for data_entry in historical_data:
                    entry_date = data_entry.get('date')
                    if isinstance(entry_date, datetime):
                        entry_month_day = (entry_date.month, entry_date.day)
                        
                        # Check if dates match within tolerance
                        if self._dates_match_within_tolerance(target_month_day, entry_month_day):
                            matched_data.append(data_entry)
                            break
            
            if matched_data:
                # Use most recent matching historical data
                best_match = max(matched_data, key=lambda x: x.get('date', datetime.min))
                data = best_match.get('data')
                
                if isinstance(data, pd.DataFrame) and len(data) > 0:
                    quality = 60.0  # Historical data has lower quality
                    return FallbackResult(
                        success=True,
                        data=data,
                        strategy_used=FallbackStrategy.HISTORICAL,
                        fallback_quality=quality,
                        message=f"Used historical data from {best_match['date'].year}",
                        metadata={'historical_year': best_match['date'].year}
                    )
            
            return FallbackResult(
                success=False,
                message=f"No suitable historical data found for {data_source}"
            )
            
        except Exception as e:
            return FallbackResult(
                success=False,
                message=f"Error in historical fallback for {data_source}: {e}"
            )
    
    def _fallback_to_alternative_source(self, 
                                      data_source: str,
                                      fetch_function: Callable,
                                      *args, **kwargs) -> FallbackResult:
        """Try alternative data sources or modified parameters."""
        # Define alternative sources or parameter modifications
        alternatives = self._get_alternative_sources(data_source)
        
        for alt_config in alternatives:
            try:
                # Modify fetch parameters based on alternative configuration
                alt_args = alt_config.get('args', args)
                alt_kwargs = {**kwargs, **alt_config.get('kwargs', {})}
                
                # Try alternative source
                data = fetch_function(*alt_args, **alt_kwargs)
                
                if self._is_data_valid(data):
                    quality = alt_config.get('quality', 50.0)
                    return FallbackResult(
                        success=True,
                        data=data,
                        strategy_used=FallbackStrategy.ALTERNATIVE_SOURCE,
                        fallback_quality=quality,
                        message=f"Used alternative source: {alt_config['name']}",
                        metadata={'alternative_source': alt_config['name']}
                    )
                    
            except Exception as e:
                logger.debug(f"Alternative source {alt_config['name']} failed: {e}")
                continue
        
        return FallbackResult(
            success=False,
            message=f"All alternative sources failed for {data_source}"
        )
    
    def _fallback_to_defaults(self, data_source: str) -> FallbackResult:
        """Generate data using scientifically reasonable default values."""
        try:
            # Generate minimal dataset with default values
            default_data = self._generate_default_dataset(data_source)
            
            if default_data is not None and len(default_data) > 0:
                return FallbackResult(
                    success=True,
                    data=default_data,
                    strategy_used=FallbackStrategy.DEFAULT_VALUES,
                    fallback_quality=30.0,  # Low quality but scientifically reasonable
                    message=f"Used default values for {data_source}",
                    metadata={'default_values_used': True}
                )
            else:
                return FallbackResult(
                    success=False,
                    message=f"Could not generate default dataset for {data_source}"
                )
                
        except Exception as e:
            return FallbackResult(
                success=False,
                message=f"Error generating defaults for {data_source}: {e}"
            )
    
    def cache_data(self, data_source: str, data: pd.DataFrame):
        """Cache data for future fallback use."""
        try:
            cache_file = self.cache_dir / f"{data_source}_cache.pkl"
            
            cache_data = {
                'timestamp': datetime.now(),
                'data': data.copy(),
                'source': data_source
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Cached data for {data_source}")
            
            # Also update recent data for interpolation
            self._update_recent_data(data_source, data)
            
        except Exception as e:
            logger.warning(f"Failed to cache data for {data_source}: {e}")
    
    def _update_recent_data(self, data_source: str, data: pd.DataFrame):
        """Update recent data store for interpolation fallback."""
        try:
            recent_file = self.cache_dir / f"{data_source}_recent.pkl"
            
            recent_data = {
                'timestamp': datetime.now(),
                'data': data.copy()
            }
            
            with open(recent_file, 'wb') as f:
                pickle.dump(recent_data, f)
                
        except Exception as e:
            logger.debug(f"Failed to update recent data for {data_source}: {e}")
    
    def _is_data_valid(self, data: Any) -> bool:
        """Check if data is valid for use."""
        if data is None:
            return False
        
        if isinstance(data, pd.DataFrame):
            return len(data) > 0 and not data.empty
        
        if isinstance(data, (list, tuple)):
            return len(data) > 0
        
        return True
    
    def _perform_interpolation(self, data: pd.DataFrame, data_source: str) -> Optional[pd.DataFrame]:
        """Perform interpolation on data gaps."""
        try:
            # Assume first column is date/time and subsequent are values
            if len(data.columns) < 2:
                return None
            
            data_copy = data.copy()
            
            # Sort by time column (assume first column is time)
            time_col = data_copy.columns[0]
            data_copy = data_copy.sort_values(time_col)
            
            # Apply interpolation to numeric columns
            numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Only interpolate gaps smaller than max_gap_days
                if self.config.interpolation_method == 'linear':
                    data_copy[col] = data_copy[col].interpolate(
                        method='linear',
                        limit=self.config.interpolation_max_gap_days
                    )
                else:
                    data_copy[col] = data_copy[col].interpolate(
                        method=self.config.interpolation_method
                    )
            
            return data_copy
            
        except Exception as e:
            logger.warning(f"Interpolation failed for {data_source}: {e}")
            return None
    
    def _dates_match_within_tolerance(self, date1: Tuple[int, int], date2: Tuple[int, int]) -> bool:
        """Check if two (month, day) tuples match within tolerance."""
        month1, day1 = date1
        month2, day2 = date2
        
        # Convert to day of year for easier comparison
        try:
            from datetime import date
            doy1 = date(2020, month1, day1).timetuple().tm_yday  # Use leap year for calculation
            doy2 = date(2020, month2, day2).timetuple().tm_yday
            
            return abs(doy1 - doy2) <= self.config.historical_tolerance_days
        except ValueError:
            return False
    
    def _get_alternative_sources(self, data_source: str) -> List[Dict[str, Any]]:
        """Get alternative source configurations for a data source."""
        alternatives = {
            'satellite': [
                {
                    'name': 'lower_resolution',
                    'quality': 60.0,
                    'kwargs': {'resolution': 'low'}
                },
                {
                    'name': 'extended_time_range',
                    'quality': 50.0,
                    'kwargs': {'time_buffer_days': 14}
                }
            ],
            'climate': [
                {
                    'name': 'monthly_average',
                    'quality': 70.0,
                    'kwargs': {'use_monthly_average': True}
                }
            ],
            'streamflow': [
                {
                    'name': 'nearby_gauge',
                    'quality': 60.0,
                    'kwargs': {'use_backup_gauge': True}
                }
            ]
        }
        
        # Return alternatives for the data source type
        for source_type, alts in alternatives.items():
            if source_type in data_source.lower():
                return alts
        
        return []
    
    def _generate_default_dataset(self, data_source: str) -> Optional[pd.DataFrame]:
        """Generate a minimal dataset with scientifically reasonable defaults."""
        try:
            # Determine appropriate defaults based on data source
            default_value = self._get_default_value_for_source(data_source)
            
            if default_value is None:
                return None
            
            # Generate minimal time series with default values
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days of default data
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            data = pd.DataFrame({
                'Date': date_range,
                'Value': [default_value] * len(date_range)
            })
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to generate default dataset for {data_source}: {e}")
            return None
    
    def _get_default_value_for_source(self, data_source: str) -> Optional[float]:
        """Get appropriate default value for a data source."""
        source_lower = data_source.lower()
        
        for key, value in self.config.default_values.items():
            if key in source_lower:
                return value
        
        # Generic defaults based on common oceanographic parameters
        if any(term in source_lower for term in ['temperature', 'sst']):
            return 12.0
        elif any(term in source_lower for term in ['chlorophyll', 'chla']):
            return 2.0
        elif any(term in source_lower for term in ['flow', 'discharge']):
            return 100.0
        elif any(term in source_lower for term in ['index', 'pdo', 'oni']):
            return 0.0
        
        return None
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fallback usage statistics."""
        if self.fallback_stats['total_requests'] > 0:
            fallback_rate = (self.fallback_stats['fallback_used'] / 
                           self.fallback_stats['total_requests'] * 100)
        else:
            fallback_rate = 0.0
        
        return {
            'total_requests': self.fallback_stats['total_requests'],
            'fallback_used': self.fallback_stats['fallback_used'],
            'fallback_rate_percent': fallback_rate,
            'strategy_usage': self.fallback_stats['strategy_usage'].copy(),
            'cache_directory': str(self.cache_dir),
            'cache_files': len(list(self.cache_dir.glob("*_cache.pkl"))),
            'report_timestamp': datetime.now().isoformat()
        }
    
    def cleanup_old_cache(self, max_age_days: int = None):
        """Clean up old cache files."""
        max_age = max_age_days or self.config.cache_max_age_days * 2  # Double the cache age for cleanup
        cutoff_date = datetime.now() - timedelta(days=max_age)
        
        try:
            cleaned_count = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        cache_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.debug(f"Error cleaning cache file {cache_file}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old cache files")
            return cleaned_count
            
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
            return 0


# Global fallback manager instance
fallback_manager = DataFallbackManager()


def get_data_with_fallback(data_source: str, fetch_function: Callable, *args, **kwargs) -> FallbackResult:
    """Convenience function to get data with fallback support."""
    return fallback_manager.get_data_with_fallback(data_source, fetch_function, *args, **kwargs)


def cache_successful_fetch(data_source: str, data: pd.DataFrame):
    """Convenience function to cache successful data fetches."""
    fallback_manager.cache_data(data_source, data)


def get_fallback_statistics() -> Dict[str, Any]:
    """Convenience function to get fallback statistics."""
    return fallback_manager.get_fallback_statistics()


def cleanup_old_fallback_cache(max_age_days: int = None):
    """Convenience function to clean up old cache files."""
    return fallback_manager.cleanup_old_cache(max_age_days)