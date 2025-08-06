"""
Secure Download Utilities
========================

Provides secure download functionality with validation, retry mechanisms,
and comprehensive error handling for external data sources.
"""

import os
import time
import requests
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from urllib.parse import urlparse

from .validation import validate_url, sanitize_filename, create_secure_temp_file, cleanup_temp_files
from .exception_handling import safe_execute, handle_data_errors
import config

logger = logging.getLogger(__name__)

# Track downloaded files for cleanup
_downloaded_files: List[str] = []


class SecureDownloader:
    """
    Secure downloader with validation, retry logic, and size limits.
    """
    
    def __init__(self, 
                 max_size_mb: int = None,
                 timeout_seconds: int = None,
                 max_retries: int = None,
                 retry_delay: int = None):
        """
        Initialize secure downloader with configuration.
        
        Args:
            max_size_mb: Maximum download size in MB
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial retry delay in seconds
        """
        self.max_size_bytes = (max_size_mb or config.MAX_DOWNLOAD_SIZE_MB) * 1024 * 1024
        self.timeout = timeout_seconds or config.REQUEST_TIMEOUT_SECONDS
        self.max_retries = max_retries or config.MAX_RETRY_ATTEMPTS
        self.retry_delay = retry_delay or config.RETRY_DELAY_SECONDS
        
        # Create session with reasonable defaults
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DATect-Forecasting/1.0 (Scientific Research)',
        })
    
    def validate_and_download(self, 
                            url: str, 
                            filename: Optional[str] = None,
                            temp_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Validate URL and download file securely.
        
        Args:
            url: URL to download from
            filename: Optional filename (generated if None)
            temp_dir: Optional temporary directory
            
        Returns:
            Tuple of (success, message, filepath)
        """
        try:
            # Validate URL
            is_valid, error_msg = validate_url(url, config.ALLOWED_DOMAINS)
            if not is_valid:
                return False, f"URL validation failed: {error_msg}", None
            
            # Generate secure filename if not provided
            if filename is None:
                parsed_url = urlparse(url)
                suggested_name = os.path.basename(parsed_url.path) or 'download'
                
                # Extract extension from URL or default to .dat
                if '.' in suggested_name:
                    name_part, ext_part = os.path.splitext(suggested_name)
                else:
                    name_part, ext_part = suggested_name, '.dat'
                
                filename = sanitize_filename(f"{name_part}{ext_part}")
            else:
                filename = sanitize_filename(filename)
            
            # Create secure temporary file if needed
            if temp_dir:
                os.makedirs(temp_dir, exist_ok=True)
                filepath = os.path.join(temp_dir, filename)
            else:
                # Use secure temporary file
                fd, filepath = create_secure_temp_file(
                    suffix=os.path.splitext(filename)[1],
                    prefix='download_',
                    dir=config.TEMP_DIR
                )
                os.close(fd)  # Close file descriptor, but keep the path
            
            # Attempt download with retries
            success, message = self._download_with_retries(url, filepath)
            
            if success:
                _downloaded_files.append(filepath)
                logger.info(f"Successfully downloaded: {url} -> {filepath}")
                return True, "Download successful", filepath
            else:
                # Clean up failed download
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False, message, None
                
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return False, f"Download failed: {str(e)}", None
    
    def _download_with_retries(self, url: str, filepath: str) -> Tuple[bool, str]:
        """
        Download file with retry logic and size validation.
        
        Args:
            url: URL to download
            filepath: Local file path to save to
            
        Returns:
            Tuple of (success, message)
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Retrying download attempt {attempt + 1}/{self.max_retries + 1} after {delay}s delay")
                    time.sleep(delay)
                
                # Make request with stream=True for size checking
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check content length if provided
                content_length = response.headers.get('content-length')
                if content_length:
                    size_bytes = int(content_length)
                    if size_bytes > self.max_size_bytes:
                        return False, f"File too large: {size_bytes / 1024 / 1024:.1f}MB > {self.max_size_bytes / 1024 / 1024}MB limit"
                
                # Download with size monitoring
                downloaded_size = 0
                chunk_size = 8192
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # Filter out keep-alive chunks
                            downloaded_size += len(chunk)
                            
                            # Check size limit during download
                            if downloaded_size > self.max_size_bytes:
                                return False, f"Download exceeded size limit: {downloaded_size / 1024 / 1024:.1f}MB"
                            
                            f.write(chunk)
                
                # Verify file was created and has content
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    return False, "Downloaded file is empty or was not created"
                
                logger.debug(f"Downloaded {downloaded_size / 1024:.1f}KB from {url}")
                return True, "Download successful"
                
            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout after {self.timeout}s"
                logger.warning(f"Download timeout (attempt {attempt + 1}): {url}")
                
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Connection error (attempt {attempt + 1}): {url}")
                
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code} - {e.response.reason}"
                logger.error(f"HTTP error (attempt {attempt + 1}): {url} - {last_error}")
                
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    break
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                logger.error(f"Request error (attempt {attempt + 1}): {url} - {last_error}")
                
            except IOError as e:
                last_error = f"File I/O error: {str(e)}"
                logger.error(f"File I/O error (attempt {attempt + 1}): {filepath} - {last_error}")
                break  # Don't retry on I/O errors
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error (attempt {attempt + 1}): {url} - {last_error}")
        
        return False, f"Download failed after {self.max_retries + 1} attempts. Last error: {last_error}"
    
    def close(self):
        """Close the session and release resources."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@handle_data_errors
def secure_download_file(url: str, 
                        filename: Optional[str] = None,
                        temp_dir: Optional[str] = None,
                        **kwargs) -> Optional[str]:
    """
    Convenience function for secure file download.
    
    Args:
        url: URL to download from
        filename: Optional filename (generated if None)
        temp_dir: Optional temporary directory
        **kwargs: Additional arguments for SecureDownloader
        
    Returns:
        Downloaded file path or None on failure
    """
    with SecureDownloader(**kwargs) as downloader:
        success, message, filepath = downloader.validate_and_download(url, filename, temp_dir)
        
        if success:
            return filepath
        else:
            logger.error(f"Download failed: {message}")
            return None


def generate_secure_filename(url: str, extension: str = '', temp_dir: Optional[str] = None) -> str:
    """
    Generate secure filename for URL with proper extension.
    
    Args:
        url: Source URL
        extension: File extension (including dot)
        temp_dir: Optional temporary directory
        
    Returns:
        Secure filename path
    """
    try:
        parsed = urlparse(url)
        base_name = os.path.basename(parsed.path) or 'download'
        
        # Remove existing extension if new one provided
        if extension and '.' in base_name:
            base_name = os.path.splitext(base_name)[0]
        
        # Add extension
        filename = f"{base_name}{extension}"
        
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        
        # Create full path
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            return os.path.join(temp_dir, safe_filename)
        else:
            return safe_filename
            
    except Exception as e:
        logger.warning(f"Error generating filename for {url}: {e}")
        timestamp = int(time.time())
        return f"download_{timestamp}{extension}"


def cleanup_downloaded_files():
    """Clean up all downloaded files tracked by this module."""
    global _downloaded_files
    
    if config.TEMP_FILE_CLEANUP:
        cleanup_temp_files(_downloaded_files)
        _downloaded_files.clear()
        logger.debug("Cleaned up all downloaded files")
    else:
        logger.debug(f"Skipping cleanup of {len(_downloaded_files)} downloaded files (cleanup disabled)")


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

def download_file(url: str, filename: str) -> str:
    """
    Legacy download function for backwards compatibility.
    
    Args:
        url: URL to download from  
        filename: Local filename to save to
        
    Returns:
        Path to downloaded file
        
    Raises:
        Exception: If download fails
    """
    logger.warning("Using legacy download_file function - consider upgrading to secure_download_file")
    
    # Use secure downloader but maintain old interface
    result = secure_download_file(url, filename)
    
    if result is None:
        raise Exception(f"Download failed for URL: {url}")
    
    return result