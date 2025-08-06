"""
Temporary File Management
=========================

Advanced temporary file management with automatic cleanup, resource tracking,
and failure recovery mechanisms for the DATect forecasting system.

This module provides:
- Comprehensive temporary directory management
- Automatic file cleanup with configurable policies
- Resource usage monitoring and limits
- Safe cleanup on interruption or failure
- File retention policies for debugging
- Cross-platform compatibility
"""

import os
import tempfile
import shutil
import atexit
import signal
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import psutil
import weakref
from contextlib import contextmanager

from .logging_config import get_logger
from .exception_handling import handle_data_errors
import config

logger = get_logger(__name__)


class CleanupPolicy(Enum):
    """Cleanup policy options."""
    IMMEDIATE = "immediate"        # Clean up immediately after use
    ON_EXIT = "on_exit"           # Clean up when program exits
    SCHEDULED = "scheduled"        # Clean up on schedule (time-based)
    SIZE_BASED = "size_based"     # Clean up when size limits exceeded
    MANUAL = "manual"             # Manual cleanup only


@dataclass
class TempFileConfig:
    """Configuration for temporary file management."""
    # Directory settings
    base_temp_dir: Optional[str] = None        # Base directory for temp files
    prefix: str = "datect_temp_"               # Prefix for temp directories
    max_temp_dirs: int = 10                    # Maximum concurrent temp directories
    
    # Cleanup settings
    cleanup_policy: CleanupPolicy = CleanupPolicy.ON_EXIT
    immediate_cleanup: bool = True             # Clean up files immediately when possible
    cleanup_interval_hours: int = 24          # Hours between scheduled cleanups
    
    # Size limits
    max_total_size_gb: float = 5.0            # Maximum total size in GB
    max_single_file_gb: float = 1.0           # Maximum single file size in GB
    warn_size_gb: float = 3.0                 # Size to trigger warnings
    
    # Retention settings
    retain_on_error: bool = True               # Keep files when errors occur
    max_retention_hours: int = 168             # Hours to retain files (7 days)
    debug_retention_hours: int = 24            # Hours to retain in debug mode
    
    # Monitoring settings
    monitor_interval_seconds: int = 300        # Resource monitoring interval
    enable_resource_monitoring: bool = True    # Enable resource usage tracking


@dataclass
class TempFileInfo:
    """Information about a temporary file or directory."""
    path: Path
    created_at: datetime
    size_bytes: int = 0
    purpose: str = ""
    cleanup_policy: CleanupPolicy = CleanupPolicy.ON_EXIT
    retain_on_error: bool = True
    accessed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.accessed_at is None:
            self.accessed_at = self.created_at
    
    @property
    def age_hours(self) -> float:
        """Get age of the file in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)


class TempFileManager:
    """
    Advanced temporary file manager with automatic cleanup and monitoring.
    
    Provides comprehensive management of temporary files and directories with
    configurable cleanup policies, resource monitoring, and error handling.
    """
    
    def __init__(self, config: Optional[TempFileConfig] = None):
        """
        Initialize temporary file manager.
        
        Args:
            config: Configuration for temp file management
        """
        self.config = config or TempFileConfig()
        self.tracked_files: Dict[str, TempFileInfo] = {}
        self.active_contexts: Set[str] = set()
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = threading.Event()
        
        # Set up base directory
        if self.config.base_temp_dir:
            self.base_dir = Path(self.config.base_temp_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.gettempdir()) / "datect_temp"
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_created': 0,
            'files_cleaned': 0,
            'total_size_cleaned_gb': 0.0,
            'cleanup_operations': 0,
            'errors_encountered': 0,
            'peak_usage_gb': 0.0
        }
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
        
        # Start monitoring if enabled
        if self.config.enable_resource_monitoring:
            self._start_resource_monitoring()
        
        logger.info(f"Initialized TempFileManager with base directory: {self.base_dir}")
    
    @contextmanager
    def managed_temp_dir(self, 
                        purpose: str = "processing",
                        cleanup_policy: Optional[CleanupPolicy] = None,
                        prefix: Optional[str] = None) -> Path:
        """
        Context manager for temporary directories with automatic cleanup.
        
        Args:
            purpose: Description of what the directory is used for
            cleanup_policy: Override default cleanup policy
            prefix: Custom prefix for directory name
            
        Yields:
            Path: Path to temporary directory
        """
        cleanup_policy = cleanup_policy or self.config.cleanup_policy
        prefix = prefix or self.config.prefix
        
        # Create temporary directory
        temp_dir = self.create_temp_dir(purpose, cleanup_policy, prefix)
        context_id = str(temp_dir)
        
        try:
            with self._lock:
                self.active_contexts.add(context_id)
            
            logger.debug(f"Created managed temp directory: {temp_dir}")
            yield temp_dir
            
        except Exception as e:
            logger.error(f"Error in managed temp directory context: {e}")
            if not self.config.retain_on_error:
                self._cleanup_single_path(temp_dir)
            raise
            
        finally:
            with self._lock:
                self.active_contexts.discard(context_id)
            
            # Cleanup based on policy
            if cleanup_policy == CleanupPolicy.IMMEDIATE:
                self._cleanup_single_path(temp_dir)
            
            logger.debug(f"Exited managed temp directory context: {temp_dir}")
    
    @handle_data_errors
    def create_temp_dir(self, 
                       purpose: str = "processing",
                       cleanup_policy: Optional[CleanupPolicy] = None,
                       prefix: Optional[str] = None) -> Path:
        """
        Create a tracked temporary directory.
        
        Args:
            purpose: Description of directory purpose
            cleanup_policy: Cleanup policy for this directory
            prefix: Custom prefix for directory name
            
        Returns:
            Path to created temporary directory
        """
        cleanup_policy = cleanup_policy or self.config.cleanup_policy
        prefix = prefix or self.config.prefix
        
        # Check limits
        self._check_resource_limits()
        
        # Create directory
        temp_dir = Path(tempfile.mkdtemp(
            prefix=prefix,
            dir=self.base_dir
        ))
        
        # Track the directory
        file_info = TempFileInfo(
            path=temp_dir,
            created_at=datetime.now(),
            purpose=purpose,
            cleanup_policy=cleanup_policy,
            retain_on_error=self.config.retain_on_error
        )
        
        with self._lock:
            self.tracked_files[str(temp_dir)] = file_info
            self.stats['files_created'] += 1
        
        logger.debug(f"Created temp directory: {temp_dir} (purpose: {purpose})")
        return temp_dir
    
    @handle_data_errors
    def create_temp_file(self,
                        suffix: str = "",
                        purpose: str = "processing",
                        cleanup_policy: Optional[CleanupPolicy] = None,
                        directory: Optional[Path] = None) -> Path:
        """
        Create a tracked temporary file.
        
        Args:
            suffix: File suffix/extension
            purpose: Description of file purpose  
            cleanup_policy: Cleanup policy for this file
            directory: Directory to create file in (uses default if None)
            
        Returns:
            Path to created temporary file
        """
        cleanup_policy = cleanup_policy or self.config.cleanup_policy
        directory = directory or self.base_dir
        
        # Check limits
        self._check_resource_limits()
        
        # Create file
        fd, temp_file_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=self.config.prefix,
            dir=directory
        )
        os.close(fd)  # Close file descriptor, keep file
        
        temp_file = Path(temp_file_path)
        
        # Track the file
        file_info = TempFileInfo(
            path=temp_file,
            created_at=datetime.now(),
            purpose=purpose,
            cleanup_policy=cleanup_policy,
            retain_on_error=self.config.retain_on_error
        )
        
        with self._lock:
            self.tracked_files[str(temp_file)] = file_info
            self.stats['files_created'] += 1
        
        logger.debug(f"Created temp file: {temp_file} (purpose: {purpose})")
        return temp_file
    
    def update_file_access(self, path: Union[str, Path]):
        """Update last access time for a tracked file."""
        path_str = str(path)
        with self._lock:
            if path_str in self.tracked_files:
                self.tracked_files[path_str].accessed_at = datetime.now()
    
    def get_tracked_files(self) -> Dict[str, TempFileInfo]:
        """Get copy of currently tracked files."""
        with self._lock:
            return self.tracked_files.copy()
    
    def get_total_size(self) -> float:
        """Get total size of tracked files in GB."""
        total_bytes = 0
        
        with self._lock:
            for file_info in self.tracked_files.values():
                if file_info.path.exists():
                    try:
                        if file_info.path.is_file():
                            total_bytes += file_info.path.stat().st_size
                        elif file_info.path.is_dir():
                            total_bytes += sum(
                                f.stat().st_size 
                                for f in file_info.path.rglob('*') 
                                if f.is_file()
                            )
                    except (OSError, PermissionError):
                        pass
        
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def cleanup_by_policy(self, policy: CleanupPolicy):
        """Clean up files matching specific policy."""
        cleaned_count = 0
        
        with self._lock:
            paths_to_clean = [
                info.path for info in self.tracked_files.values()
                if info.cleanup_policy == policy
            ]
        
        for path in paths_to_clean:
            if self._cleanup_single_path(path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} files with policy: {policy.value}")
        return cleaned_count
    
    def cleanup_old_files(self, max_age_hours: Optional[int] = None):
        """Clean up files older than specified age."""
        max_age = max_age_hours or self.config.max_retention_hours
        cutoff_time = datetime.now() - timedelta(hours=max_age)
        
        cleaned_count = 0
        
        with self._lock:
            old_files = [
                info.path for info in self.tracked_files.values()
                if info.created_at < cutoff_time
            ]
        
        for path in old_files:
            if self._cleanup_single_path(path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} files older than {max_age} hours")
        return cleaned_count
    
    def cleanup_by_size(self, target_size_gb: Optional[float] = None):
        """Clean up files to reach target size, removing oldest first."""
        target_size = target_size_gb or self.config.warn_size_gb
        current_size = self.get_total_size()
        
        if current_size <= target_size:
            return 0
        
        cleaned_count = 0
        
        # Sort files by age (oldest first)
        with self._lock:
            files_by_age = sorted(
                self.tracked_files.values(),
                key=lambda f: f.created_at
            )
        
        for file_info in files_by_age:
            if self.get_total_size() <= target_size:
                break
                
            if self._cleanup_single_path(file_info.path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} files to reach target size {target_size:.2f}GB")
        return cleaned_count
    
    def cleanup_all(self, force: bool = False):
        """Clean up all tracked files."""
        if not force:
            # Don't clean files in active contexts
            with self._lock:
                paths_to_clean = [
                    info.path for info in self.tracked_files.values()
                    if str(info.path) not in self.active_contexts
                ]
        else:
            with self._lock:
                paths_to_clean = list(self.tracked_files.keys())
        
        cleaned_count = 0
        for path in paths_to_clean:
            if self._cleanup_single_path(path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} tracked files")
        return cleaned_count
    
    def _cleanup_single_path(self, path: Union[str, Path]) -> bool:
        """Clean up a single path and remove from tracking."""
        path = Path(path)
        path_str = str(path)
        
        try:
            if path.exists():
                size_before = 0
                
                # Calculate size before deletion
                try:
                    if path.is_file():
                        size_before = path.stat().st_size
                    elif path.is_dir():
                        size_before = sum(
                            f.stat().st_size 
                            for f in path.rglob('*') 
                            if f.is_file()
                        )
                except (OSError, PermissionError):
                    pass
                
                # Remove the path
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                
                # Update statistics
                with self._lock:
                    self.stats['files_cleaned'] += 1
                    self.stats['total_size_cleaned_gb'] += size_before / (1024 ** 3)
                
                logger.debug(f"Cleaned up: {path}")
            
            # Remove from tracking
            with self._lock:
                self.tracked_files.pop(path_str, None)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")
            with self._lock:
                self.stats['errors_encountered'] += 1
            return False
    
    def _check_resource_limits(self):
        """Check if resource limits would be exceeded."""
        total_size = self.get_total_size()
        
        # Update peak usage
        with self._lock:
            if total_size > self.stats['peak_usage_gb']:
                self.stats['peak_usage_gb'] = total_size
        
        # Check size limits
        if total_size >= self.config.max_total_size_gb:
            logger.warning(f"Total temp file size limit reached: {total_size:.2f}GB")
            # Try to clean up old files
            self.cleanup_old_files()
            self.cleanup_by_size(self.config.warn_size_gb)
            
            # Check again
            if self.get_total_size() >= self.config.max_total_size_gb:
                raise RuntimeError(f"Temporary file size limit exceeded: {self.get_total_size():.2f}GB")
        
        elif total_size >= self.config.warn_size_gb:
            logger.warning(f"Temp file usage high: {total_size:.2f}GB (limit: {self.config.max_total_size_gb:.2f}GB)")
        
        # Check count limits
        with self._lock:
            if len(self.tracked_files) >= self.config.max_temp_dirs * 10:  # Rough limit
                logger.warning(f"High number of temp files: {len(self.tracked_files)}")
                self.cleanup_old_files()
    
    def _register_cleanup_handlers(self):
        """Register cleanup handlers for program exit and signals."""
        # Register atexit handler
        atexit.register(self._cleanup_on_exit)
        
        # Register signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (AttributeError, ValueError):
            # Some signals may not be available on all platforms
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, cleaning up temporary files...")
        self.cleanup_all(force=True)
    
    def _cleanup_on_exit(self):
        """Cleanup handler called on program exit."""
        try:
            logger.info("Program exiting, cleaning up temporary files...")
            self._shutdown.set()
            
            # Stop monitoring thread
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)
            
            # Clean up files based on policy
            if self.config.cleanup_policy == CleanupPolicy.ON_EXIT:
                self.cleanup_all(force=True)
            
        except Exception as e:
            logger.error(f"Error during exit cleanup: {e}")
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread."""
        def monitor():
            while not self._shutdown.wait(self.config.monitor_interval_seconds):
                try:
                    # Scheduled cleanup
                    if self.config.cleanup_policy == CleanupPolicy.SCHEDULED:
                        self.cleanup_old_files()
                    
                    # Size-based cleanup
                    if self.config.cleanup_policy == CleanupPolicy.SIZE_BASED:
                        if self.get_total_size() > self.config.warn_size_gb:
                            self.cleanup_by_size()
                    
                    # Update file sizes
                    self._update_file_sizes()
                    
                except Exception as e:
                    logger.debug(f"Error in resource monitoring: {e}")
        
        self._cleanup_thread = threading.Thread(target=monitor, daemon=True)
        self._cleanup_thread.start()
    
    def _update_file_sizes(self):
        """Update size information for tracked files."""
        with self._lock:
            for file_info in self.tracked_files.values():
                try:
                    if file_info.path.exists():
                        if file_info.path.is_file():
                            file_info.size_bytes = file_info.path.stat().st_size
                        elif file_info.path.is_dir():
                            file_info.size_bytes = sum(
                                f.stat().st_size 
                                for f in file_info.path.rglob('*') 
                                if f.is_file()
                            )
                except (OSError, PermissionError):
                    pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about temp file usage."""
        current_size = self.get_total_size()
        
        with self._lock:
            file_count = len(self.tracked_files)
            active_contexts = len(self.active_contexts)
            stats_copy = self.stats.copy()
        
        # Calculate additional metrics
        files_by_policy = {}
        files_by_age = {'< 1 hour': 0, '1-24 hours': 0, '> 24 hours': 0}
        
        with self._lock:
            for info in self.tracked_files.values():
                # Count by policy
                policy = info.cleanup_policy.value
                files_by_policy[policy] = files_by_policy.get(policy, 0) + 1
                
                # Count by age
                age_hours = info.age_hours
                if age_hours < 1:
                    files_by_age['< 1 hour'] += 1
                elif age_hours < 24:
                    files_by_age['1-24 hours'] += 1
                else:
                    files_by_age['> 24 hours'] += 1
        
        return {
            'current_file_count': file_count,
            'active_contexts': active_contexts,
            'current_size_gb': round(current_size, 3),
            'base_directory': str(self.base_dir),
            'files_by_cleanup_policy': files_by_policy,
            'files_by_age': files_by_age,
            'resource_monitoring_enabled': self.config.enable_resource_monitoring,
            **stats_copy,
            'report_timestamp': datetime.now().isoformat()
        }


# Global temp file manager instance
temp_manager = TempFileManager()


@contextmanager
def managed_temp_dir(purpose: str = "processing", **kwargs) -> Path:
    """Convenience context manager for temporary directories."""
    with temp_manager.managed_temp_dir(purpose, **kwargs) as temp_dir:
        yield temp_dir


def create_temp_dir(purpose: str = "processing", **kwargs) -> Path:
    """Convenience function to create temporary directory."""
    return temp_manager.create_temp_dir(purpose, **kwargs)


def create_temp_file(suffix: str = "", purpose: str = "processing", **kwargs) -> Path:
    """Convenience function to create temporary file."""
    return temp_manager.create_temp_file(suffix, purpose, **kwargs)


def cleanup_temp_files():
    """Convenience function to cleanup all temporary files."""
    return temp_manager.cleanup_all()


def get_temp_file_statistics() -> Dict[str, Any]:
    """Convenience function to get temp file statistics."""
    return temp_manager.get_statistics()