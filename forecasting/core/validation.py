"""
Input Validation and Security Utilities
=======================================

Provides comprehensive input validation, URL security, and path sanitization
for the DATect forecasting system. Prevents security vulnerabilities while
maintaining scientific data processing capabilities.
"""

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urljoin
from typing import Optional, List, Dict, Tuple, Union
import logging

from .exception_handling import ScientificValidationError

logger = logging.getLogger(__name__)

# =============================================================================
# URL VALIDATION AND SECURITY
# =============================================================================

# Whitelist of allowed domains for external data sources
ALLOWED_DOMAINS = {
    'oceanview.pfeg.noaa.gov',  # NOAA ERDDAP servers
    'coastwatch.pfeg.noaa.gov', # NOAA CoastWatch
    'waterservices.usgs.gov',   # USGS water data
    'oceandata.sci.gsfc.nasa.gov', # NASA ocean color data
}

# Allowed URL schemes
ALLOWED_SCHEMES = {'https', 'http'}

# Allowed file extensions for downloads
ALLOWED_EXTENSIONS = {'.nc', '.csv', '.json', '.txt', '.dat'}


def validate_url(url: str, allowed_domains: Optional[set] = None) -> Tuple[bool, str]:
    """
    Validate URL against security criteria and domain whitelist.
    
    Args:
        url: URL to validate
        allowed_domains: Set of allowed domains (uses default if None)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_domains is None:
        allowed_domains = ALLOWED_DOMAINS
    
    try:
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"
        
        # Parse URL
        parsed = urlparse(url.strip())
        
        # Check scheme
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False, f"Invalid scheme '{parsed.scheme}'. Allowed: {ALLOWED_SCHEMES}"
        
        # Check domain
        domain = parsed.netloc.lower()
        if domain not in allowed_domains:
            return False, f"Domain '{domain}' not in whitelist: {allowed_domains}"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.\./',  # Directory traversal
            r'[<>"\']',  # HTML/script injection characters
            r'javascript:',  # JavaScript protocol
            r'data:',  # Data protocol
            r'file:',  # File protocol
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False, f"URL contains suspicious pattern: {pattern}"
        
        return True, "URL validation passed"
        
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem operations.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    if not filename or not isinstance(filename, str):
        raise ScientificValidationError("Filename must be a non-empty string")
    
    # First, remove directory traversal attempts completely
    sanitized = filename.replace('..', '').replace('/', '').replace('\\', '')
    
    # Remove other dangerous path components
    sanitized = sanitized.replace('~', '').replace('*', '').replace('?', '')
    
    # Remove or replace other dangerous characters
    # Keep alphanumeric, dots, hyphens, underscores, spaces
    sanitized = re.sub(r'[^\w\-_\.\s]', '_', sanitized)
    
    # Remove leading/trailing dots and spaces which can be problematic
    sanitized = sanitized.strip(' .')
    
    # Prevent empty filenames
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Enforce length limit
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        available_length = max_length - len(ext)
        sanitized = name[:available_length] + ext
    
    # Prevent reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = os.path.splitext(sanitized)[0].upper()
    if name_without_ext in reserved_names:
        sanitized = f"safe_{sanitized}"
    
    return sanitized


def validate_file_path(file_path: Union[str, Path], 
                      allowed_dirs: Optional[List[str]] = None,
                      must_exist: bool = False) -> Tuple[bool, str]:
    """
    Validate file path for security and accessibility.
    
    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directory prefixes
        must_exist: Whether file must exist
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not file_path:
            return False, "File path cannot be empty"
        
        # Convert to Path object
        path = Path(file_path).resolve()
        
        # Check for directory traversal attempts
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/'):
            if allowed_dirs is None:
                return False, "Absolute paths and directory traversal not allowed"
        
        # Validate against allowed directories
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    path.relative_to(Path(allowed_dir).resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                return False, f"Path not in allowed directories: {allowed_dirs}"
        
        # Check if file must exist
        if must_exist and not path.exists():
            return False, f"Required file does not exist: {path}"
        
        # Check file extension if it exists
        if path.suffix and path.suffix.lower() not in ALLOWED_EXTENSIONS:
            logger.warning(f"File extension '{path.suffix}' not in allowed list")
        
        return True, "File path validation passed"
        
    except Exception as e:
        return False, f"File path validation error: {str(e)}"


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_coordinate(lat: float, lon: float) -> Tuple[bool, str]:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        lat = float(lat)
        lon = float(lon)
        
        if not (-90 <= lat <= 90):
            return False, f"Invalid latitude: {lat}. Must be between -90 and 90"
        
        if not (-180 <= lon <= 180):
            return False, f"Invalid longitude: {lon}. Must be between -180 and 180"
        
        return True, "Coordinates validation passed"
        
    except (ValueError, TypeError) as e:
        return False, f"Coordinate validation error: {str(e)}"


def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
    """
    Validate date range for data requests.
    
    Args:
        start_date: Start date string (ISO format expected)
        end_date: End date string (ISO format expected)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from datetime import datetime
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Validate range
        if start_dt >= end_dt:
            return False, "Start date must be before end date"
        
        # Check for reasonable range (not too far in past/future)
        current_year = datetime.now().year
        min_year = 2000  # Satellite data availability
        max_year = current_year + 2  # Allow some future dates
        
        if start_dt.year < min_year or end_dt.year > max_year:
            return False, f"Date range must be between {min_year} and {max_year}"
        
        return True, "Date range validation passed"
        
    except Exception as e:
        return False, f"Date validation error: {str(e)}"


def validate_numeric_range(value: Union[int, float], 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          name: str = "value") -> Tuple[bool, str]:
    """
    Validate numeric value within specified range.
    
    Args:
        value: Numeric value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        numeric_value = float(value)
        
        # Check for NaN or infinite values
        if not isinstance(numeric_value, (int, float)) or \
           numeric_value != numeric_value or \
           abs(numeric_value) == float('inf'):
            return False, f"{name} must be a finite number"
        
        # Check range
        if min_val is not None and numeric_value < min_val:
            return False, f"{name} ({numeric_value}) must be >= {min_val}"
        
        if max_val is not None and numeric_value > max_val:
            return False, f"{name} ({numeric_value}) must be <= {max_val}"
        
        return True, f"{name} validation passed"
        
    except (ValueError, TypeError) as e:
        return False, f"{name} validation error: {str(e)}"


# =============================================================================
# SECURE TEMPORARY FILE HANDLING
# =============================================================================

def create_secure_temp_file(suffix: str = '', prefix: str = 'datect_', 
                           dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Create a secure temporary file with proper permissions.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        dir: Directory to create temp file in
        
    Returns:
        Tuple of (file_descriptor, file_path)
    """
    try:
        # Sanitize inputs
        suffix = sanitize_filename(suffix) if suffix else ''
        prefix = sanitize_filename(prefix) if prefix else 'datect_'
        
        # Create secure temporary file
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir
        )
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_path, 0o600)
        
        logger.debug(f"Created secure temporary file: {temp_path}")
        return fd, temp_path
        
    except Exception as e:
        logger.error(f"Failed to create secure temporary file: {e}")
        raise ScientificValidationError(f"Temporary file creation failed: {e}")


def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Securely clean up temporary files.
    
    Args:
        file_paths: List of temporary file paths to remove
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                # Overwrite with zeros before deletion (basic security)
                try:
                    with open(file_path, 'r+b') as f:
                        length = f.seek(0, 2)  # Get file size
                        f.seek(0)
                        f.write(b'\x00' * length)  # Overwrite with zeros
                except:
                    pass  # Skip overwrite if file is not writable
                
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config_structure(config_dict: Dict) -> Tuple[bool, str]:
    """
    Validate configuration dictionary structure.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_sections = [
        'ORIGINAL_DA_FILES',
        'ORIGINAL_PN_FILES',
        'SITES',
    ]
    
    required_urls = [
        'PDO_URL',
        'ONI_URL',
        'BEUTI_URL',
        'STREAMFLOW_URL',
    ]
    
    try:
        # Check required sections
        for section in required_sections:
            if section not in config_dict:
                return False, f"Missing required configuration section: {section}"
        
        # Check required URLs
        for url_key in required_urls:
            if url_key not in config_dict:
                return False, f"Missing required URL configuration: {url_key}"
            
            # Validate each URL
            is_valid, error_msg = validate_url(config_dict[url_key])
            if not is_valid:
                return False, f"Invalid {url_key}: {error_msg}"
        
        # Validate site coordinates
        if 'SITES' in config_dict:
            for site_name, coords in config_dict['SITES'].items():
                if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                    return False, f"Invalid coordinates for site {site_name}: must be [lat, lon]"
                
                is_valid, error_msg = validate_coordinate(coords[0], coords[1])
                if not is_valid:
                    return False, f"Invalid coordinates for site {site_name}: {error_msg}"
        
        return True, "Configuration validation passed"
        
    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"