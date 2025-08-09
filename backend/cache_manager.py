"""
Cache Manager for DATect API
============================

Manages cached pre-computed results for Google Cloud deployment.
Serves cached data instead of running expensive computations on the server.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages access to pre-computed cache files."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.enabled = self._should_enable_cache()
        
        if self.enabled:
            self.manifest = self._load_manifest()
            if not self.cache_dir.exists():
                logger.warning(f"Cache directory {cache_dir} not found. Run precompute_cache.py first.")
        else:
            self.manifest = None
            logger.info("Cache disabled for local development")
    
    def _should_enable_cache(self) -> bool:
        """Determine if cache should be enabled based on environment."""
        # Enable cache only in production or when explicitly requested
        if os.getenv("CACHE_DIR") == "/app/cache":  # Docker production
            return True
        if os.getenv("ENABLE_PRECOMPUTED_CACHE", "").lower() == "true":  # Explicit enable
            return True
        if os.getenv("NODE_ENV") == "production":  # Production environment
            return True
        
        # Disable for local development
        return False
            
    def _load_manifest(self) -> Optional[Dict]:
        """Load cache manifest file."""
        manifest_path = self.cache_dir / "manifest.json"
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache manifest: {e}")
                return None
        return None
        
    def is_cache_available(self) -> bool:
        """Check if cache is available and valid."""
        return self.enabled and self.manifest is not None and self.cache_dir.exists()
        
    def get_retrospective_forecast(self, task: str, model_type: str) -> Optional[List[Dict]]:
        """
        Get cached retrospective forecast results.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost" or "linear"
            
        Returns:
            List of forecast results or None if not cached
        """
        if not self.enabled:
            return None
        cache_file = self.cache_dir / "retrospective" / f"{task}_{model_type}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Clean any inf/nan values from cached data
                import math
                def clean_cached_data(item):
                    if isinstance(item, dict):
                        cleaned = {}
                        for k, v in item.items():
                            if isinstance(v, float):
                                if math.isinf(v) or math.isnan(v):
                                    cleaned[k] = None
                                else:
                                    cleaned[k] = v
                            else:
                                cleaned[k] = v
                        return cleaned
                    return item
                
                cleaned_data = [clean_cached_data(item) for item in data]
                logger.info(f"Served cached retrospective forecast: {task}+{model_type} ({len(cleaned_data)} records)")
                return cleaned_data
            except Exception as e:
                logger.error(f"Failed to load cached retrospective forecast: {e}")
                
        logger.warning(f"No cached data for retrospective forecast: {task}+{model_type}")
        return None
        
    def get_spectral_analysis(self, site: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get cached spectral analysis results.
        
        Args:
            site: Site name or None for aggregate
            
        Returns:
            Spectral analysis plots or None if not cached
        """
        if not self.enabled:
            return None
        site_name = site or "all_sites"
        cache_file = self.cache_dir / "spectral" / f"{site_name}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Served cached spectral analysis: {site_name}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached spectral analysis: {e}")
                
        logger.warning(f"No cached spectral analysis for site: {site_name}")
        return None
        
    def get_correlation_matrix(self, site: str) -> Optional[Dict]:
        """
        Get cached correlation matrix for a site.
        
        Args:
            site: Site name
            
        Returns:
            Correlation matrix data or None if not cached
        """
        if not self.enabled:
            return None
        cache_file = self.cache_dir / "visualizations" / f"{site}_correlation.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Served cached correlation matrix: {site}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached correlation matrix: {e}")
                
        logger.warning(f"No cached correlation matrix for site: {site}")
        return None
        
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics."""
        if not self.enabled:
            return {"available": False, "message": "Cache disabled for local development"}
        if not self.manifest:
            return {"available": False, "message": "No cache manifest found"}
            
        status = {
            "available": True,
            "generated_at": self.manifest.get("generated_at"),
            "cache_version": self.manifest.get("cache_version"),
            "total_files": len(self.manifest.get("files", {})),
            "categories": {}
        }
        
        # Count files by category
        for file_path in self.manifest.get("files", {}).keys():
            category = file_path.split('/')[0] if '/' in file_path else "root"
            status["categories"][category] = status["categories"].get(category, 0) + 1
            
        # Calculate total cache size
        total_size = sum(
            file_info.get("size_bytes", 0) 
            for file_info in self.manifest.get("files", {}).values()
        )
        status["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return status
        
    def list_available_forecasts(self) -> List[Dict[str, str]]:
        """List all available cached forecast combinations."""
        if not self.enabled:
            return []
        forecasts = []
        retrospective_dir = self.cache_dir / "retrospective"
        
        if retrospective_dir.exists():
            for json_file in retrospective_dir.glob("*.json"):
                if not json_file.name.endswith("_summary.json"):
                    # Parse filename like "classification_xgboost.json"
                    name_parts = json_file.stem.split("_", 1)
                    if len(name_parts) == 2:
                        task, model = name_parts
                        
                        # Load summary if available
                        summary_file = retrospective_dir / f"{json_file.stem}_summary.json"
                        summary = {}
                        if summary_file.exists():
                            try:
                                with open(summary_file, 'r') as f:
                                    summary = json.load(f)
                            except:
                                pass
                                
                        forecasts.append({
                            "task": task,
                            "model_type": model,
                            "predictions": summary.get("total_predictions", "unknown"),
                            "date_range": summary.get("date_range", {}),
                            "file_size_mb": round(json_file.stat().st_size / (1024*1024), 2)
                        })
                        
        return forecasts
        
    def list_available_spectral(self) -> List[str]:
        """List all available cached spectral analysis sites."""
        if not self.enabled:
            return []
        sites = []
        spectral_dir = self.cache_dir / "spectral"
        
        if spectral_dir.exists():
            for json_file in spectral_dir.glob("*.json"):
                site_name = json_file.stem
                # Convert "all_sites" back to None representation
                if site_name == "all_sites":
                    sites.append("(aggregate)")
                else:
                    sites.append(site_name)
                    
        return sorted(sites)


# Global cache manager instance
cache_manager = CacheManager()