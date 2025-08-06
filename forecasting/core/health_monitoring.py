"""
Health monitoring system for DATect forecasting service.
Provides system health checks, performance metrics, and monitoring capabilities.
"""

import time
import psutil
import platform
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    uptime_seconds: float


@dataclass
class ServiceMetrics:
    """Service-level metrics."""
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    average_response_time_ms: float
    last_prediction_time: Optional[datetime]
    models_loaded: int
    health_status: str


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, service_name: str = "datect-api"):
        self.service_name = service_name
        self.start_time = time.time()
        
        # Service metrics
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.response_times: List[float] = []
        self.last_prediction_time: Optional[datetime] = None
        self.models_loaded = 0
        
        # Health status
        self.health_status = "starting"
        
        logger.info(f"Health monitor initialized for {service_name}")
    
    def record_request(self, success: bool = True, response_time_ms: float = 0):
        """Record a service request."""
        self.total_requests += 1
        
        if success:
            self.successful_predictions += 1
            self.last_prediction_time = datetime.now(timezone.utc)
        else:
            self.failed_predictions += 1
        
        if response_time_ms > 0:
            self.response_times.append(response_time_ms)
            # Keep only last 1000 response times for memory efficiency
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
    
    def record_model_loaded(self):
        """Record that a model has been loaded."""
        self.models_loaded += 1
        self.health_status = "healthy"
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 ** 3)
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=round(memory_available_gb, 2),
                disk_usage_percent=round(disk_usage_percent, 1),
                disk_free_gb=round(disk_free_gb, 2),
                uptime_seconds=round(uptime_seconds, 1)
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                uptime_seconds=time.time() - self.start_time
            )
    
    def get_service_metrics(self) -> ServiceMetrics:
        """Get current service metrics."""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0.0
        )
        
        return ServiceMetrics(
            total_requests=self.total_requests,
            successful_predictions=self.successful_predictions,
            failed_predictions=self.failed_predictions,
            average_response_time_ms=round(avg_response_time, 2),
            last_prediction_time=self.last_prediction_time,
            models_loaded=self.models_loaded,
            health_status=self.health_status
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        system_metrics = self.get_system_metrics()
        service_metrics = self.get_service_metrics()
        
        # Determine overall health
        is_healthy = True
        issues = []
        
        # Check system resources
        if system_metrics.cpu_percent > 90:
            is_healthy = False
            issues.append(f"High CPU usage: {system_metrics.cpu_percent}%")
        
        if system_metrics.memory_percent > 90:
            is_healthy = False
            issues.append(f"High memory usage: {system_metrics.memory_percent}%")
        
        if system_metrics.disk_usage_percent > 90:
            is_healthy = False
            issues.append(f"High disk usage: {system_metrics.disk_usage_percent}%")
        
        # Check service health
        error_rate = (
            service_metrics.failed_predictions / max(service_metrics.total_requests, 1) * 100
            if service_metrics.total_requests > 0 else 0
        )
        
        if error_rate > 10:  # More than 10% error rate
            is_healthy = False
            issues.append(f"High error rate: {error_rate:.1f}%")
        
        status = "healthy" if is_healthy else "unhealthy"
        
        return {
            "service_name": self.service_name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "platform": {
                "system": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.machine()
            },
            "issues": issues,
            "system_metrics": asdict(system_metrics),
            "service_metrics": asdict(service_metrics)
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status with additional diagnostics."""
        basic_status = self.get_health_status()
        
        # Add detailed information
        try:
            # Process information
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_info_mb": round(process.memory_info().rss / (1024 ** 2), 1),
                "cpu_percent": round(process.cpu_percent(), 1),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
            # File system information
            model_artifacts_path = Path("./model_artifacts")
            data_files = {
                "model_artifacts_exists": model_artifacts_path.exists(),
                "model_artifacts_count": len(list(model_artifacts_path.glob("*"))) if model_artifacts_path.exists() else 0,
                "training_data_exists": Path("final_output.parquet").exists(),
                "training_data_size_mb": round(Path("final_output.parquet").stat().st_size / (1024 ** 2), 1) if Path("final_output.parquet").exists() else 0
            }
            
            basic_status.update({
                "process_info": process_info,
                "data_files": data_files,
                "recent_response_times": self.response_times[-10:] if len(self.response_times) >= 10 else self.response_times
            })
            
        except Exception as e:
            logger.warning(f"Could not get detailed status: {e}")
            basic_status["detailed_info_error"] = str(e)
        
        return basic_status


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def initialize_health_monitor(service_name: str = "datect-api") -> HealthMonitor:
    """Initialize the global health monitor."""
    global _health_monitor
    _health_monitor = HealthMonitor(service_name)
    return _health_monitor