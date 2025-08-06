"""
Retry Mechanisms and Circuit Breaker
====================================

Advanced retry mechanisms with exponential backoff, jitter, and circuit breaker
pattern for resilient external API calls and data processing operations.

This module provides:
- Exponential backoff with configurable jitter
- Circuit breaker pattern for failing services
- Retry decorators for common use cases
- Fallback strategies and graceful degradation
- Comprehensive retry monitoring and logging
"""

import time
import random
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

from .logging_config import get_logger
from .exception_handling import ScientificValidationError
import config

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0      # Base delay in seconds
    max_delay: float = 60.0      # Maximum delay in seconds  
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True          # Add random jitter to delays
    jitter_factor: float = 0.1   # Maximum jitter as fraction of delay
    
    # Exceptions that trigger retries
    retryable_exceptions: Tuple = (
        ConnectionError,
        TimeoutError,
        IOError,
    )
    
    # Exceptions that should NOT be retried
    non_retryable_exceptions: Tuple = (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5    # Failures before opening circuit
    recovery_timeout: float = 60.0 # Seconds before attempting recovery
    success_threshold: int = 3     # Successes needed to close circuit
    timeout: float = 30.0         # Operation timeout in seconds


@dataclass 
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    exception: Optional[Exception] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RetryResult:
    """Result of retry operation."""
    success: bool
    result: Any = None
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_time: float = 0.0
    final_exception: Optional[Exception] = None


class CircuitBreaker:
    """
    Circuit breaker implementation for failing services.
    
    Prevents cascading failures by temporarily stopping requests
    to failing services and allowing them time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
        
        logger.info(f"Initialized CircuitBreaker '{name}' with {self.config.failure_threshold} failure threshold")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN state")
                else:
                    logger.warning(f"Circuit breaker '{self.name}' is OPEN - rejecting request")
                    raise ConnectionError(f"Circuit breaker '{self.name}' is open")
        
        try:
            # Execute function with timeout
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Handle success
            self._on_success()
            
            logger.debug(f"Circuit breaker '{self.name}' call succeeded in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle failure
            self._on_failure(e)
            raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' recovered - state: CLOSED")
            else:
                self.state = CircuitState.CLOSED
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed during recovery - state: OPEN")
                
            elif self.failure_count >= self.config.failure_threshold:
                # Too many failures - open circuit
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
            }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class RetryManager:
    """
    Advanced retry manager with exponential backoff and monitoring.
    """
    
    def __init__(self):
        """Initialize retry manager."""
        self.retry_stats = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
        
        logger.info("Initialized RetryManager")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """
        Get or create circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            config: Configuration for new circuit breakers
            
        Returns:
            Circuit breaker instance
        """
        if name not in self.circuit_breakers:
            with self.lock:
                if name not in self.circuit_breakers:
                    self.circuit_breakers[name] = CircuitBreaker(name, config)
        
        return self.circuit_breakers[name]
    
    def retry_with_backoff(self, func: Callable, config: RetryConfig = None,
                          circuit_breaker_name: Optional[str] = None) -> RetryResult:
        """
        Execute function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            config: Retry configuration
            circuit_breaker_name: Optional circuit breaker name
            
        Returns:
            Retry result with attempt information
        """
        config = config or RetryConfig()
        start_time = time.time()
        attempts = []
        
        # Get circuit breaker if specified
        circuit_breaker = None
        if circuit_breaker_name:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
        
        for attempt in range(config.max_attempts):
            attempt_start = time.time()
            
            try:
                # Execute function (with circuit breaker if specified)
                if circuit_breaker:
                    result = circuit_breaker.call(func)
                else:
                    result = func()
                
                # Success - record attempt and return
                attempt_time = time.time() - attempt_start
                attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    delay=0.0,
                    timestamp=datetime.now().isoformat()
                ))
                
                total_time = time.time() - start_time
                
                # Log success
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1} after {total_time:.2f}s")
                
                # Record stats
                self._record_retry_stats(func.__name__, attempts, True)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=total_time
                )
                
            except Exception as e:
                attempt_time = time.time() - attempt_start
                
                # Check if exception should be retried
                if isinstance(e, config.non_retryable_exceptions):
                    logger.error(f"Non-retryable exception in {func.__name__}: {e}")
                    
                    attempts.append(RetryAttempt(
                        attempt_number=attempt + 1,
                        delay=0.0,
                        exception=e,
                        timestamp=datetime.now().isoformat()
                    ))
                    
                    self._record_retry_stats(func.__name__, attempts, False)
                    
                    return RetryResult(
                        success=False,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        final_exception=e
                    )
                
                # Record attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    delay=0.0,
                    exception=e,
                    timestamp=datetime.now().isoformat()
                ))
                
                # Check if we should retry
                if attempt < config.max_attempts - 1 and isinstance(e, config.retryable_exceptions):
                    # Calculate delay with exponential backoff and jitter
                    delay = self._calculate_delay(attempt, config)
                    attempts[-1].delay = delay
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                 f"Retrying in {delay:.2f}s")
                    
                    time.sleep(delay)
                else:
                    # Final attempt or non-retryable exception
                    logger.error(f"All retry attempts failed for {func.__name__}: {e}")
                    
                    self._record_retry_stats(func.__name__, attempts, False)
                    
                    return RetryResult(
                        success=False,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        final_exception=e
                    )
        
        # Should not reach here
        return RetryResult(
            success=False,
            attempts=attempts,
            total_time=time.time() - start_time,
            final_exception=Exception("Max attempts reached")
        )
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter."""
        # Exponential backoff
        delay = config.base_delay * (config.backoff_factor ** attempt)
        
        # Cap at maximum delay
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _record_retry_stats(self, func_name: str, attempts: List[RetryAttempt], success: bool):
        """Record retry statistics for monitoring."""
        stats_entry = {
            'function': func_name,
            'timestamp': datetime.now().isoformat(),
            'attempts_count': len(attempts),
            'success': success,
            'total_delay': sum(a.delay for a in attempts)
        }
        
        with self.lock:
            self.retry_stats[func_name].append(stats_entry)
            
            # Keep only recent stats (last 100 entries per function)
            if len(self.retry_stats[func_name]) > 100:
                self.retry_stats[func_name] = self.retry_stats[func_name][-100:]
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics."""
        with self.lock:
            stats = {}
            
            for func_name, entries in self.retry_stats.items():
                if not entries:
                    continue
                
                total_calls = len(entries)
                successful_calls = sum(1 for e in entries if e['success'])
                failed_calls = total_calls - successful_calls
                
                total_attempts = sum(e['attempts_count'] for e in entries)
                avg_attempts = total_attempts / total_calls if total_calls > 0 else 0
                
                total_delay = sum(e['total_delay'] for e in entries)
                avg_delay = total_delay / total_calls if total_calls > 0 else 0
                
                success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
                
                stats[func_name] = {
                    'total_calls': total_calls,
                    'successful_calls': successful_calls,
                    'failed_calls': failed_calls,
                    'success_rate_pct': success_rate,
                    'average_attempts': avg_attempts,
                    'average_delay_seconds': avg_delay,
                    'total_retry_time_seconds': total_delay
                }
            
            # Circuit breaker stats
            circuit_stats = {}
            for name, breaker in self.circuit_breakers.items():
                circuit_stats[name] = breaker.get_state()
            
            return {
                'retry_statistics': stats,
                'circuit_breakers': circuit_stats,
                'report_timestamp': datetime.now().isoformat()
            }


# Global retry manager instance
retry_manager = RetryManager()


# Decorator functions for common retry patterns

def retry_on_failure(max_attempts: int = 3, base_delay: float = 1.0, 
                    backoff_factor: float = 2.0, circuit_breaker: Optional[str] = None):
    """
    Decorator to add retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts in seconds
        backoff_factor: Exponential backoff multiplier
        circuit_breaker: Optional circuit breaker name
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                backoff_factor=backoff_factor
            )
            
            result = retry_manager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
                circuit_breaker_name=circuit_breaker
            )
            
            if result.success:
                return result.result
            else:
                raise result.final_exception
        
        return wrapper
    return decorator


def retry_network_call(max_attempts: int = 5, circuit_breaker: str = None):
    """
    Decorator specifically for network calls with appropriate retry settings.
    
    Args:
        max_attempts: Maximum number of retry attempts
        circuit_breaker: Optional circuit breaker name
        
    Returns:
        Decorated function with network retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            IOError,
            OSError,
        )
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_manager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
                circuit_breaker_name=circuit_breaker
            )
            
            if result.success:
                return result.result
            else:
                raise result.final_exception
        
        return wrapper
    return decorator


def retry_data_processing(max_attempts: int = 3):
    """
    Decorator for data processing operations with appropriate retry settings.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with data processing retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=0.5,
        max_delay=5.0,
        backoff_factor=1.5,
        jitter=False,
        retryable_exceptions=(
            IOError,
            OSError,
        ),
        non_retryable_exceptions=(
            ValueError,
            TypeError,
            KeyError,
            ScientificValidationError,
        )
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_manager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config
            )
            
            if result.success:
                return result.result
            else:
                raise result.final_exception
        
        return wrapper
    return decorator


# Utility functions

def get_retry_statistics() -> Dict[str, Any]:
    """Get current retry statistics."""
    return retry_manager.get_retry_statistics()


def reset_circuit_breaker(name: str):
    """Reset circuit breaker to closed state."""
    if name in retry_manager.circuit_breakers:
        retry_manager.circuit_breakers[name].reset()
        logger.info(f"Circuit breaker '{name}' reset")
    else:
        logger.warning(f"Circuit breaker '{name}' not found")


def create_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Create or get circuit breaker."""
    return retry_manager.get_circuit_breaker(name, config)