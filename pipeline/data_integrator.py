"""
Data Integrator
===============

Coordinates the integration of all data sources into a unified dataset for
the DATect forecasting system. Handles temporal alignment, data merging,
and application of temporal safeguards.

This module provides:
- Unified data integration pipeline
- Temporal alignment of different data sources
- Site-week grid generation
- Climate indices integration with temporal buffers
- Toxin data merging and interpolation
- Comprehensive logging and error handling
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from forecasting.core.logging_config import get_logger
from forecasting.core.exception_handling import handle_data_errors, validate_data_integrity
from .temporal_safeguards import TemporalSafeguards
import config

logger = get_logger(__name__)


class DataIntegrator:
    """
    Integrates all data sources for the DATect forecasting system.
    
    Coordinates satellite, climate, streamflow, and toxin data integration
    with comprehensive temporal safeguards and data validation.
    """
    
    def __init__(self):
        """Initialize data integrator."""
        self.temporal_safeguards = TemporalSafeguards()
        logger.info("Initialized DataIntegrator")
    
    @handle_data_errors
    def generate_compiled_data(self, sites_dict: Dict[str, List[float]], 
                              start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """
        Generate base DataFrame with all Site-Week combinations.
        
        Args:
            sites_dict: Dictionary of site names to [lat, lon] coordinates
            start_dt: Start date for time range
            end_dt: End date for time range
            
        Returns:
            DataFrame with Site, Date, lat, lon columns or None on failure
        """
        logger.info(f"Generating weekly site-date grid from {start_dt.date()} to {end_dt.date()}")
        print(f"  Generating weekly entries from {start_dt.date()} to {end_dt.date()}")
        
        try:
            # Generate weekly date range (Mondays)
            weeks = pd.date_range(start_dt, end_dt, freq='W-MON', name='Date')
            logger.debug(f"Generated {len(weeks)} weekly dates")
            
            df_list = []
            invalid_sites = 0
            
            # Create Site-Week combinations
            for site, coords in sites_dict.items():
                # Validate coordinates
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    lat, lon = coords[0], coords[1]
                    
                    # Validate coordinate values
                    from forecasting.core.validation import validate_coordinate
                    is_valid, _ = validate_coordinate(lat, lon)
                    if not is_valid:
                        logger.warning(f"Invalid coordinates for site {site}: {coords}")
                        lat, lon = np.nan, np.nan
                        invalid_sites += 1
                else:
                    logger.warning(f"Invalid coordinate format for site {site}: {coords}")
                    lat, lon = np.nan, np.nan
                    invalid_sites += 1
                
                # Normalize site name
                normalized_site = self._normalize_site_name(site)
                
                # Create DataFrame for this site
                site_df = pd.DataFrame({
                    'Date': weeks,
                    'Site': normalized_site,
                    'lat': lat,
                    'lon': lon
                })
                
                df_list.append(site_df)
                logger.debug(f"Created {len(weeks)} weekly entries for site {site}")
            
            if invalid_sites > 0:
                logger.warning(f"{invalid_sites} sites had invalid coordinates")
            
            # Combine all site DataFrames
            compiled_df = pd.concat(df_list, ignore_index=True)
            compiled_df = compiled_df.sort_values(['Site', 'Date'])
            
            logger.info(f"Generated base DataFrame with {len(compiled_df)} site-week rows")
            print(f"  Generated base DataFrame with {len(compiled_df)} site-week rows.")
            
            return compiled_df
            
        except Exception as e:
            logger.error(f"Error generating compiled data: {e}")
            return None
    
    def _normalize_site_name(self, site_name: str) -> str:
        """
        Normalize site name for consistency.
        
        Args:
            site_name: Original site name
            
        Returns:
            Normalized site name in Title Case
        """
        return site_name.replace('_', ' ').replace('-', ' ').title()
    
    @handle_data_errors
    def compile_data(self, compiled_df: pd.DataFrame, oni_df: Optional[pd.DataFrame],
                    pdo_df: Optional[pd.DataFrame], streamflow_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge climate indices and streamflow data into base DataFrame.
        
        Applies temporal safeguards:
        - ONI and PDO use 2-month buffer to account for reporting delays
        - Streamflow uses backward fill with 7-day tolerance
        
        Args:
            compiled_df: Base DataFrame with Site-Date grid
            oni_df: ONI climate index data
            pdo_df: PDO climate index data
            streamflow_df: Daily streamflow data
            
        Returns:
            Enhanced DataFrame with environmental data
        """
        logger.info("Merging environmental data with temporal safeguards")
        print("\\n--- Merging Environmental Data ---")
        
        try:
            # Ensure Date is datetime type
            compiled_df['Date'] = pd.to_datetime(compiled_df['Date'])
            logger.debug("Converted Date column to datetime")
            
            # Apply temporal safeguards for climate indices
            compiled_df = self._merge_climate_indices(compiled_df, oni_df, pdo_df)
            
            # Merge streamflow data
            compiled_df = self._merge_streamflow_data(compiled_df, streamflow_df)
            
            # Final validation and sorting
            final_df = compiled_df.sort_values(['Site', 'Date'])
            
            logger.info(f"Environmental data integration complete: {len(final_df)} records")
            return final_df
            
        except Exception as e:
            logger.error(f"Error compiling environmental data: {e}")
            return compiled_df  # Return original if merge fails
    
    def _merge_climate_indices(self, compiled_df: pd.DataFrame, 
                              oni_df: Optional[pd.DataFrame], 
                              pdo_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge climate indices with temporal buffer to prevent data leakage.
        
        Args:
            compiled_df: Base DataFrame
            oni_df: ONI data
            pdo_df: PDO data
            
        Returns:
            DataFrame with climate indices merged
        """
        # Apply 2-month temporal buffer for climate indices
        # This accounts for reporting delays and prevents data leakage
        compiled_df['TargetPrevMonth'] = compiled_df['Date'].dt.to_period('M') - 2
        
        logger.debug("Applied 2-month temporal buffer for climate indices")
        
        # Sort for consistent merging
        compiled_df = compiled_df.sort_values(['Site', 'Date'])
        
        # Merge ONI data
        if oni_df is not None and not oni_df.empty:
            compiled_df = self._merge_single_climate_index(
                compiled_df, oni_df, 'oni', 'ONI'
            )
        else:
            logger.warning("ONI data not available - adding null column")
            compiled_df['oni'] = np.nan
        
        # Merge PDO data
        if pdo_df is not None and not pdo_df.empty:
            compiled_df = self._merge_single_climate_index(
                compiled_df, pdo_df, 'pdo', 'PDO'
            )
        else:
            logger.warning("PDO data not available - adding null column")
            compiled_df['pdo'] = np.nan
        
        # Clean up temporary column
        compiled_df.drop(columns=['TargetPrevMonth'], inplace=True, errors='ignore')
        
        return compiled_df
    
    def _merge_single_climate_index(self, compiled_df: pd.DataFrame, 
                                   climate_df: pd.DataFrame, 
                                   column_name: str, display_name: str) -> pd.DataFrame:
        """
        Merge a single climate index dataset.
        
        Args:
            compiled_df: Base DataFrame
            climate_df: Climate index DataFrame
            column_name: Name for the output column
            display_name: Display name for logging
            
        Returns:
            DataFrame with climate index merged
        """
        try:
            # Prepare climate data for merge
            climate_to_merge = climate_df[['Month', 'index']].rename(
                columns={'index': column_name, 'Month': 'ClimateIndexMonth'}
            )
            
            # Ensure uniqueness (safeguard)
            climate_to_merge = climate_to_merge.drop_duplicates(subset=['ClimateIndexMonth'])
            
            # Merge with temporal buffer
            compiled_df = pd.merge(
                compiled_df,
                climate_to_merge,
                left_on='TargetPrevMonth',
                right_on='ClimateIndexMonth',
                how='left'
            )
            
            # Clean up merge column
            compiled_df.drop(columns=['ClimateIndexMonth'], inplace=True, errors='ignore')
            
            # Log merge statistics
            non_null_count = compiled_df[column_name].notna().sum()
            total_count = len(compiled_df)
            logger.debug(f"Merged {display_name}: {non_null_count}/{total_count} records matched")
            
            return compiled_df
            
        except Exception as e:
            logger.error(f"Error merging {display_name} data: {e}")
            compiled_df[column_name] = np.nan  # Add null column on failure
            return compiled_df
    
    def _merge_streamflow_data(self, compiled_df: pd.DataFrame, 
                              streamflow_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge streamflow data with backward fill and tolerance.
        
        Args:
            compiled_df: Base DataFrame
            streamflow_df: Daily streamflow data
            
        Returns:
            DataFrame with streamflow data merged
        """
        if streamflow_df is None or streamflow_df.empty:
            logger.warning("Streamflow data not available - adding null column")
            compiled_df['discharge'] = np.nan
            return compiled_df
        
        try:
            # Prepare streamflow data
            streamflow_clean = streamflow_df.copy()
            streamflow_clean['Date'] = pd.to_datetime(streamflow_clean['Date'])
            streamflow_clean = streamflow_clean.sort_values('Date')
            
            # Ensure compiled_df is sorted by Date for merge_asof
            compiled_df = compiled_df.sort_values('Date')
            
            # Backward-looking merge with 7-day tolerance
            compiled_df = pd.merge_asof(
                compiled_df,
                streamflow_clean[['Date', 'Flow']],
                on='Date',
                direction='backward',
                tolerance=pd.Timedelta('7days')
            )
            
            # Rename column
            compiled_df.rename(columns={'Flow': 'discharge'}, inplace=True)
            
            # Log merge statistics
            non_null_count = compiled_df['discharge'].notna().sum()
            total_count = len(compiled_df)
            logger.debug(f"Merged streamflow: {non_null_count}/{total_count} records matched")
            
            return compiled_df
            
        except Exception as e:
            logger.error(f"Error merging streamflow data: {e}")
            compiled_df['discharge'] = np.nan
            return compiled_df
    
    @handle_data_errors
    def compile_da_pn(self, lt_df: pd.DataFrame, da_df: Optional[pd.DataFrame], 
                     pn_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge DA and PN data with temporal validation and interpolation.
        
        Args:
            lt_df: Long-term DataFrame with environmental data
            da_df: Weekly aggregated DA data
            pn_df: Weekly aggregated PN data
            
        Returns:
            Enhanced DataFrame with toxin data
        """
        logger.info("Merging DA and PN toxin data")
        print("\\n--- Merging DA and PN Data ---")
        
        try:
            lt_df_merged = lt_df.copy()
            
            # Merge DA data
            lt_df_merged = self._merge_toxin_data(
                lt_df_merged, da_df, 'DA', 'DA_Levels', 'DA_Levels_orig'
            )
            
            # Merge PN data
            lt_df_merged = self._merge_toxin_data(
                lt_df_merged, pn_df, 'PN', 'PN_Levels', 'PN_Levels_orig'
            )
            
            logger.info(f"Toxin data integration complete: {len(lt_df_merged)} records")
            return lt_df_merged
            
        except Exception as e:
            logger.error(f"Error compiling toxin data: {e}")
            return lt_df
    
    def _merge_toxin_data(self, lt_df: pd.DataFrame, toxin_df: Optional[pd.DataFrame],
                         data_type: str, level_col: str, output_col: str) -> pd.DataFrame:
        """
        Merge a single toxin dataset (DA or PN).
        
        Args:
            lt_df: Base DataFrame
            toxin_df: Toxin data (DA or PN)
            data_type: Data type name for logging
            level_col: Source level column name
            output_col: Output column name
            
        Returns:
            DataFrame with toxin data merged
        """
        if toxin_df is None or toxin_df.empty:
            logger.warning(f"{data_type} data not available - adding null column")
            lt_df[output_col] = np.nan
            return lt_df
        
        try:
            print(f"  Merging {data_type} data ({len(toxin_df)} records)...")
            
            # Prepare toxin data
            toxin_copy = toxin_df.copy()
            
            # Convert Year-Week to Date (Monday of that week)
            toxin_copy['Date'] = pd.to_datetime(
                toxin_copy['Year-Week'] + '-1', 
                format='%G-%V-%w'
            )
            
            # Clean and normalize data
            toxin_copy = toxin_copy.dropna(subset=['Date', 'Site', level_col])
            
            # Normalize site names for consistent merging
            lt_df['Site'] = lt_df['Site'].astype(str).str.replace('_', ' ').str.title()
            toxin_copy['Site'] = toxin_copy['Site'].astype(str).str.replace('_', ' ').str.title()
            
            # Merge data
            lt_df = pd.merge(
                lt_df, 
                toxin_copy[['Date', 'Site', level_col]], 
                on=['Date', 'Site'], 
                how='left'
            )
            
            # Rename column
            lt_df.rename(columns={level_col: output_col}, inplace=True)
            
            # Log merge statistics
            non_null_count = lt_df[output_col].notna().sum()
            total_count = len(lt_df)
            match_pct = (non_null_count / total_count * 100) if total_count > 0 else 0
            
            logger.info(f"Merged {data_type}: {non_null_count}/{total_count} records matched ({match_pct:.1f}%)")
            print(f"    {data_type} merge: {non_null_count}/{total_count} matches ({match_pct:.1f}%)")
            
            return lt_df
            
        except Exception as e:
            logger.error(f"Error merging {data_type} data: {e}")
            lt_df[output_col] = np.nan
            return lt_df
    
    @handle_data_errors
    def convert_and_fill(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final data processing with interpolation and validation.
        
        Args:
            data_df: Combined DataFrame with all data sources
            
        Returns:
            Final processed DataFrame ready for modeling
        """
        logger.info("Applying final data processing and interpolation")
        
        try:
            # Apply temporal safeguards validation
            validated_df = self.temporal_safeguards.validate_temporal_integrity(data_df)
            
            # Apply data interpolation and filling
            processed_df = self._apply_data_interpolation(validated_df)
            
            # Final validation
            final_df = self._final_data_validation(processed_df)
            
            logger.info(f"Final data processing complete: {len(final_df)} records ready for modeling")
            return final_df
            
        except Exception as e:
            logger.error(f"Error in final data processing: {e}")
            return data_df
    
    def _apply_data_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply interpolation strategies for missing data.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with interpolated values
        """
        df_processed = df.copy()
        
        # Define interpolation strategies for different columns
        interpolation_config = {
            'discharge': 'forward_fill',  # Forward fill streamflow (last known value)
            'oni': 'forward_fill',        # Forward fill climate indices
            'pdo': 'forward_fill',
            # Toxin data (DA/PN) should NOT be interpolated - keep as NaN for training
        }
        
        for column, strategy in interpolation_config.items():
            if column in df_processed.columns:
                missing_before = df_processed[column].isna().sum()
                
                if strategy == 'forward_fill':
                    # Group by site and forward fill
                    df_processed[column] = (df_processed.groupby('Site')[column]
                                          .fillna(method='ffill'))
                
                missing_after = df_processed[column].isna().sum()
                filled_count = missing_before - missing_after
                
                if filled_count > 0:
                    logger.debug(f"Interpolated {filled_count} missing values in {column}")
        
        return df_processed
    
    def _final_data_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform final data validation and quality checks.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Required columns for modeling
        required_columns = ['Date', 'Site', 'lat', 'lon']
        
        # Validate data integrity
        is_valid, error_msg = validate_data_integrity(df, required_columns, min_rows=1)
        if not is_valid:
            logger.error(f"Final data validation failed: {error_msg}")
        else:
            logger.info("Final data validation passed")
        
        # Log data quality statistics
        self._log_data_quality_stats(df)
        
        return df
    
    def _log_data_quality_stats(self, df: pd.DataFrame):
        """
        Log data quality statistics for monitoring.
        
        Args:
            df: Final DataFrame
        """
        stats = {
            'total_records': len(df),
            'unique_sites': df['Site'].nunique() if 'Site' in df.columns else 0,
            'date_range_days': (df['Date'].max() - df['Date'].min()).days if 'Date' in df.columns and not df.empty else 0,
        }
        
        # Calculate missing value percentages
        missing_stats = {}
        for column in df.columns:
            if column not in ['Date', 'Site']:
                missing_count = df[column].isna().sum()
                missing_pct = (missing_count / len(df) * 100) if len(df) > 0 else 0
                missing_stats[column] = missing_pct
        
        logger.info(f"Data quality summary: {stats['total_records']} records, "
                   f"{stats['unique_sites']} sites, {stats['date_range_days']} days")
        
        for column, missing_pct in missing_stats.items():
            if missing_pct > 50:
                logger.warning(f"High missing data in {column}: {missing_pct:.1f}%")
            elif missing_pct > 0:
                logger.debug(f"Missing data in {column}: {missing_pct:.1f}%")
    
    def integrate_all_data(self, satellite_data: Optional[pd.DataFrame],
                          climate_data: Dict[str, Optional[pd.DataFrame]],
                          streamflow_data: Optional[pd.DataFrame],
                          toxin_data: Dict[str, Optional[pd.DataFrame]],
                          sites_dict: Dict[str, List[float]],
                          start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Complete data integration pipeline.
        
        Args:
            satellite_data: Processed satellite data
            climate_data: Dictionary with ONI, PDO, BEUTI data
            streamflow_data: Processed streamflow data
            toxin_data: Dictionary with DA, PN data
            sites_dict: Site coordinates dictionary
            start_date: Start date for integration
            end_date: End date for integration
            
        Returns:
            Fully integrated DataFrame ready for modeling
        """
        logger.info("Starting complete data integration pipeline")
        
        try:
            # Generate base site-week grid
            compiled_df = self.generate_compiled_data(sites_dict, start_date, end_date)
            if compiled_df is None:
                logger.error("Failed to generate base compiled data")
                return None
            
            # Merge environmental data (climate + streamflow)
            environmental_df = self.compile_data(
                compiled_df,
                climate_data.get('ONI'),
                climate_data.get('PDO'), 
                streamflow_data
            )
            
            # Add satellite data if available
            if satellite_data is not None and not satellite_data.empty:
                # This would integrate with the satellite processor's add_satellite_data method
                logger.info("Satellite data integration would be implemented here")
                # environmental_df = satellite_processor.add_satellite_data(environmental_df, satellite_data)
            
            # Add BEUTI data if available
            if climate_data.get('BEUTI') is not None:
                # BEUTI has different structure (site-specific) - would need special handling
                logger.info("BEUTI data integration would be implemented here")
            
            # Merge toxin data
            final_df = self.compile_da_pn(
                environmental_df,
                toxin_data.get('DA'),
                toxin_data.get('PN')
            )
            
            # Apply final processing
            processed_df = self.convert_and_fill(final_df)
            
            logger.info("Complete data integration pipeline finished successfully")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in complete data integration: {e}")
            return None