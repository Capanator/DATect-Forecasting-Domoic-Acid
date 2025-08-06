"""
Model Persistence and Loading System
====================================

Production-ready model artifact management for DATect forecasting system.
Handles model training, persistence, loading, and metadata management.

Features:
- Model artifact saving with metadata
- Versioned model storage
- Model loading with validation
- Training pipeline persistence
- Model performance tracking
- Automatic model backup and recovery

Usage:
    from forecasting.core.model_persistence import ModelArtifactManager
    
    manager = ModelArtifactManager()
    
    # Save trained model
    manager.save_model(model, preprocessor, metadata, model_name="xgboost_v1.0")
    
    # Load model for prediction
    model, preprocessor = manager.load_model("xgboost_v1.0")
"""

import os
import pickle
import joblib
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from .logging_config import get_logger
from .exception_handling import safe_execute, handle_data_error, ModelError


class ModelArtifactManager:
    """
    Manages model artifacts with versioning, metadata, and validation.
    
    Features:
    - Atomic model saving/loading operations
    - Model metadata and performance tracking
    - Automatic backup and versioning
    - Data preprocessing pipeline persistence
    - Model validation and integrity checks
    """
    
    def __init__(self, artifacts_dir="./model_artifacts/", backup_dir="./model_backups/"):
        self.artifacts_dir = Path(artifacts_dir)
        self.backup_dir = Path(backup_dir)
        self.logger = get_logger(__name__)
        
        # Create directories if they don't exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ModelArtifactManager initialized: {self.artifacts_dir}")
    
    def save_model(self, model, preprocessor=None, metadata: Dict[str, Any] = None, 
                   model_name: str = "default_model", version: str = None) -> str:
        """
        Save trained model with preprocessing pipeline and metadata.
        
        Args:
            model: Trained machine learning model
            preprocessor: Data preprocessing pipeline (transformer)
            metadata: Dictionary containing model metadata
            model_name: Name identifier for the model
            version: Version string (auto-generated if None)
            
        Returns:
            String path to saved model artifact
            
        Raises:
            ModelError: If model saving fails
        """
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Create versioned model directory
            model_dir = self.artifacts_dir / f"{model_name}_v{version}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            full_metadata = self._prepare_metadata(model, preprocessor, metadata, model_name, version)
            
            # Save model components atomically
            self._save_model_components(model_dir, model, preprocessor, full_metadata)
            
            # Create backup
            self._create_model_backup(model_dir, model_name, version)
            
            # Update model registry
            self._update_model_registry(model_name, version, full_metadata)
            
            artifact_path = str(model_dir)
            self.logger.info(f"Model saved successfully: {artifact_path}")
            
            return artifact_path
            
        except Exception as e:
            error_msg = f"Failed to save model {model_name} v{version}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelError(error_msg, error_code="MODEL_SAVE_ERROR", 
                           context={'model_name': model_name, 'version': version})
    
    def load_model(self, model_name: str, version: str = "latest") -> Tuple[Any, Any, Dict]:
        """
        Load trained model with preprocessing pipeline and metadata.
        
        Args:
            model_name: Name identifier for the model
            version: Version string or "latest" for most recent
            
        Returns:
            Tuple of (model, preprocessor, metadata)
            
        Raises:
            ModelError: If model loading fails
        """
        try:
            # Resolve version
            if version == "latest":
                version = self._get_latest_version(model_name)
                if not version:
                    raise ModelError(f"No versions found for model {model_name}",
                                   error_code="MODEL_NOT_FOUND")
            
            model_dir = self.artifacts_dir / f"{model_name}_v{version}"
            
            if not model_dir.exists():
                raise ModelError(f"Model artifact not found: {model_dir}",
                               error_code="MODEL_NOT_FOUND")
            
            # Load model components
            model, preprocessor, metadata = self._load_model_components(model_dir)
            
            # Validate loaded model
            self._validate_loaded_model(model, preprocessor, metadata)
            
            self.logger.info(f"Model loaded successfully: {model_name} v{version}")
            
            return model, preprocessor, metadata
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name} v{version}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelError(error_msg, error_code="MODEL_LOAD_ERROR",
                           context={'model_name': model_name, 'version': version})
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to list of versions
        """
        models = {}
        
        try:
            for item in self.artifacts_dir.iterdir():
                if item.is_dir() and '_v' in item.name:
                    name_parts = item.name.split('_v')
                    if len(name_parts) == 2:
                        model_name = name_parts[0]
                        version = name_parts[1]
                        
                        if model_name not in models:
                            models[model_name] = []
                        models[model_name].append(version)
            
            # Sort versions for each model
            for model_name in models:
                models[model_name].sort(reverse=True)  # Most recent first
                
            self.logger.info(f"Found {len(models)} model types with {sum(len(v) for v in models.values())} total versions")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return {}
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name identifier for the model
            version: Version string to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            model_dir = self.artifacts_dir / f"{model_name}_v{version}"
            
            if not model_dir.exists():
                self.logger.warning(f"Model not found for deletion: {model_dir}")
                return False
            
            # Create backup before deletion
            backup_path = self._create_deletion_backup(model_dir, model_name, version)
            
            # Remove model directory
            import shutil
            shutil.rmtree(model_dir)
            
            # Update registry
            self._remove_from_registry(model_name, version)
            
            self.logger.info(f"Model deleted: {model_name} v{version} (backup: {backup_path})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name} v{version}: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name identifier for the model
            version: Version string or "latest"
            
        Returns:
            Dictionary containing model information
        """
        try:
            if version == "latest":
                version = self._get_latest_version(model_name)
            
            model_dir = self.artifacts_dir / f"{model_name}_v{version}"
            metadata_path = model_dir / "metadata.json"
            
            if not metadata_path.exists():
                return {}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add file system information
            metadata['artifact_path'] = str(model_dir)
            metadata['size_mb'] = self._get_directory_size(model_dir) / (1024 * 1024)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name} v{version}: {str(e)}")
            return {}
    
    def _prepare_metadata(self, model, preprocessor, user_metadata, model_name, version):
        """Prepare comprehensive metadata for model artifact."""
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'model_type': type(model).__name__,
            'preprocessor_type': type(preprocessor).__name__ if preprocessor else None,
            'has_preprocessor': preprocessor is not None,
            'python_version': '.'.join(map(str, __import__('sys').version_info[:3])),
            'dependencies': self._get_package_versions()
        }
        
        # Add model-specific information
        if hasattr(model, 'get_params'):
            metadata['model_params'] = model.get_params()
        
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importances'] = True
            metadata['n_features'] = len(model.feature_importances_)
        
        # Add user metadata
        if user_metadata:
            metadata.update(user_metadata)
        
        # Add checksums for integrity validation
        metadata['checksums'] = {}  # Will be populated during saving
        
        return metadata
    
    def _save_model_components(self, model_dir, model, preprocessor, metadata):
        """Save all model components atomically."""
        files_saved = []
        
        try:
            # Save model
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
            files_saved.append(model_path)
            metadata['checksums']['model'] = self._calculate_file_checksum(model_path)
            
            # Save preprocessor if provided
            if preprocessor is not None:
                preprocessor_path = model_dir / "preprocessor.joblib"
                joblib.dump(preprocessor, preprocessor_path)
                files_saved.append(preprocessor_path)
                metadata['checksums']['preprocessor'] = self._calculate_file_checksum(preprocessor_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            files_saved.append(metadata_path)
            
            # Create model info summary
            info_path = model_dir / "model_info.txt"
            self._create_model_summary(info_path, metadata)
            files_saved.append(info_path)
            
            self.logger.info(f"Saved {len(files_saved)} model components")
            
        except Exception as e:
            # Cleanup partial save on failure
            for file_path in files_saved:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass
            raise
    
    def _load_model_components(self, model_dir):
        """Load all model components with validation."""
        model_path = model_dir / "model.joblib"
        preprocessor_path = model_dir / "preprocessor.joblib" 
        metadata_path = model_dir / "metadata.json"
        
        # Load metadata first
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate checksums if available
        if 'checksums' in metadata:
            if 'model' in metadata['checksums']:
                expected = metadata['checksums']['model']
                actual = self._calculate_file_checksum(model_path)
                if expected != actual:
                    raise ModelError("Model file checksum validation failed",
                                   error_code="CHECKSUM_ERROR")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load preprocessor if exists
        preprocessor = None
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            
            # Validate preprocessor checksum
            if 'checksums' in metadata and 'preprocessor' in metadata['checksums']:
                expected = metadata['checksums']['preprocessor']
                actual = self._calculate_file_checksum(preprocessor_path)
                if expected != actual:
                    raise ModelError("Preprocessor file checksum validation failed",
                                   error_code="CHECKSUM_ERROR")
        
        return model, preprocessor, metadata
    
    def _validate_loaded_model(self, model, preprocessor, metadata):
        """Validate loaded model components."""
        # Check model type matches metadata
        if type(model).__name__ != metadata.get('model_type'):
            raise ModelError("Model type mismatch between artifact and metadata",
                           error_code="MODEL_TYPE_MISMATCH")
        
        # Check preprocessor consistency
        has_preprocessor = preprocessor is not None
        expected_preprocessor = metadata.get('has_preprocessor', False)
        
        if has_preprocessor != expected_preprocessor:
            raise ModelError("Preprocessor presence mismatch between artifact and metadata",
                           error_code="PREPROCESSOR_MISMATCH")
        
        # Check if model has required methods
        required_methods = ['predict']
        for method in required_methods:
            if not hasattr(model, method):
                raise ModelError(f"Loaded model missing required method: {method}",
                               error_code="INVALID_MODEL_INTERFACE")
    
    def _get_latest_version(self, model_name):
        """Get the latest version for a given model name."""
        models = self.list_models()
        if model_name in models and models[model_name]:
            return models[model_name][0]  # First item is latest (sorted descending)
        return None
    
    def _create_model_backup(self, model_dir, model_name, version):
        """Create backup of saved model."""
        try:
            import shutil
            backup_path = self.backup_dir / f"{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(model_dir, backup_path)
            self.logger.info(f"Model backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create model backup: {str(e)}")
            return None
    
    def _update_model_registry(self, model_name, version, metadata):
        """Update model registry with new model information."""
        registry_path = self.artifacts_dir / "model_registry.json"
        
        try:
            # Load existing registry
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {}
            
            # Add/update model entry
            if model_name not in registry:
                registry[model_name] = {}
            
            registry[model_name][version] = {
                'created_at': metadata['created_at'],
                'model_type': metadata['model_type'],
                'performance_metrics': metadata.get('performance_metrics', {}),
                'artifact_path': f"{model_name}_v{version}"
            }
            
            # Save updated registry
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to update model registry: {str(e)}")
    
    def _calculate_file_checksum(self, file_path):
        """Calculate SHA256 checksum for file integrity validation."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_package_versions(self):
        """Get versions of key packages for reproducibility."""
        packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'joblib']
        versions = {}
        
        for package in packages:
            try:
                module = __import__(package)
                versions[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not_installed'
        
        return versions
    
    def _create_model_summary(self, info_path, metadata):
        """Create human-readable model summary."""
        summary_lines = [
            f"DATect Model Artifact Summary",
            f"=" * 40,
            f"Model Name: {metadata['model_name']}",
            f"Version: {metadata['version']}",
            f"Created: {metadata['created_at']}",
            f"Model Type: {metadata['model_type']}",
            f"Has Preprocessor: {metadata['has_preprocessor']}",
            f"Python Version: {metadata['python_version']}",
            "",
            f"Dependencies:",
        ]
        
        for package, version in metadata.get('dependencies', {}).items():
            summary_lines.append(f"  {package}: {version}")
        
        if 'performance_metrics' in metadata:
            summary_lines.extend([
                "",
                "Performance Metrics:"
            ])
            for metric, value in metadata['performance_metrics'].items():
                if isinstance(value, float):
                    summary_lines.append(f"  {metric}: {value:.4f}")
                else:
                    summary_lines.append(f"  {metric}: {value}")
        
        with open(info_path, 'w') as f:
            f.write('\n'.join(summary_lines))
    
    def _get_directory_size(self, directory):
        """Calculate total size of directory in bytes."""
        total_size = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
    
    def _create_deletion_backup(self, model_dir, model_name, version):
        """Create backup before deletion."""
        try:
            import shutil
            backup_path = self.backup_dir / f"deleted_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(model_dir, backup_path)
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create deletion backup: {str(e)}")
            return None
    
    def _remove_from_registry(self, model_name, version):
        """Remove model from registry."""
        registry_path = self.artifacts_dir / "model_registry.json"
        
        try:
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                
                if model_name in registry and version in registry[model_name]:
                    del registry[model_name][version]
                    
                    # Remove model name if no versions left
                    if not registry[model_name]:
                        del registry[model_name]
                    
                    with open(registry_path, 'w') as f:
                        json.dump(registry, f, indent=2, default=str)
                        
        except Exception as e:
            self.logger.warning(f"Failed to remove from registry: {str(e)}")