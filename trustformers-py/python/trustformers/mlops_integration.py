#!/usr/bin/env python3
"""
MLOps Integration for Trustformers

This module provides comprehensive MLOps integrations including:
- MLflow tracking and model registry
- Weights & Biases (wandb) experiment tracking
- TensorBoard logging and visualization
- Generic experiment tracking interface
- Model lifecycle management
"""

import json
import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from collections import defaultdict
import pickle
import threading
from datetime import datetime
import shutil

try:
    import mlflow
    import mlflow.tracking
    from mlflow.models import ModelSignature
    from mlflow.models.signature import infer_signature
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

try:
    import tensorboardX
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        import torch
        HAS_TENSORBOARD = True
    except ImportError:
        SummaryWriter = None
        HAS_TENSORBOARD = False

# Import trustformers types
try:
    from . import Tensor, PreTrainedModel, AutoTokenizer
except ImportError:
    # Fallback for standalone usage
    Tensor = Any
    PreTrainedModel = Any
    AutoTokenizer = Any

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    name: str
    project: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelMetadata:
    """Metadata for model artifacts."""
    name: str
    version: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    framework: str = "trustformers"
    architecture: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    created_at: Optional[str] = None

class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""
    
    @abstractmethod
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment and return experiment ID."""
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        pass
    
    @abstractmethod
    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter."""
        pass
    
    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        pass
    
    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log an artifact file."""
        pass
    
    @abstractmethod
    def log_model(self, model: Any, model_name: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Log a model and return model URI."""
        pass
    
    @abstractmethod
    def end_experiment(self) -> None:
        """End the current experiment."""
        pass

class MLflowTracker(ExperimentTracker):
    """MLflow-based experiment tracker."""
    
    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.current_run = None
        self.experiment_id = None
        
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(config.name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    config.name,
                    tags={"description": config.description} if config.description else None
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            self.current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                tags=dict(zip(config.tags or [], config.tags or [])) if config.tags else None
            )
            
            if config.parameters:
                self.log_parameters(config.parameters)
            
            if config.metadata:
                for key, value in config.metadata.items():
                    mlflow.set_tag(key, value)
            
            return self.current_run.info.run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow experiment: {e}")
            raise
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to MLflow."""
        try:
            mlflow.log_metric(key, value, step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics, step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log parameter to MLflow."""
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to MLflow."""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        try:
            if artifact_name:
                # Copy to temp file with new name
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, artifact_name)
                shutil.copy2(artifact_path, temp_path)
                mlflow.log_artifact(temp_path)
                shutil.rmtree(temp_dir)
            else:
                mlflow.log_artifact(artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_path}: {e}")
    
    def log_model(self, model: Any, model_name: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Log model to MLflow."""
        try:
            # Save model to temporary directory
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "model")
            
            # Save model using pickle for now (can be enhanced for specific model types)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create signature if possible
            signature = None
            try:
                if hasattr(model, 'forward') or hasattr(model, '__call__'):
                    # Try to infer signature from a dummy input
                    pass  # Would need specific model implementation
            except:
                pass
            
            # Log the model
            model_info = mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=None,  # Custom model wrapper could be added
                artifacts={"model": model_path},
                signature=signature
            )
            
            # Register model if metadata provided
            if metadata and metadata.name:
                try:
                    model_version = mlflow.register_model(
                        model_info.model_uri,
                        metadata.name,
                        tags=dict(zip(metadata.tags or [], metadata.tags or [])) if metadata.tags else None
                    )
                    
                    # Update model version with metadata
                    client = mlflow.tracking.MlflowClient()
                    if metadata.description:
                        client.update_model_version(
                            metadata.name,
                            model_version.version,
                            description=metadata.description
                        )
                except Exception as e:
                    logger.warning(f"Failed to register model: {e}")
            
            shutil.rmtree(temp_dir)
            return model_info.model_uri
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
            raise
    
    def end_experiment(self) -> None:
        """End MLflow run."""
        try:
            if self.current_run:
                mlflow.end_run()
                self.current_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, entity: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize W&B tracker.
        
        Args:
            entity: W&B entity (team/user)
            api_key: W&B API key
        """
        if not HAS_WANDB:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        self.entity = entity
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        
        self.current_run = None
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start W&B run."""
        try:
            self.current_run = wandb.init(
                project=config.project or config.name,
                name=config.name,
                entity=self.entity,
                tags=config.tags,
                notes=config.description,
                config=config.parameters or {}
            )
            
            if config.metadata:
                wandb.config.update(config.metadata)
            
            return self.current_run.id
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            raise
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to W&B."""
        try:
            log_data = {key: value}
            if step is not None:
                log_data["step"] = step
            wandb.log(log_data)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to W&B."""
        try:
            log_data = metrics.copy()
            if step is not None:
                log_data["step"] = step
            wandb.log(log_data)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log parameter to W&B config."""
        try:
            wandb.config.update({key: value})
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to W&B."""
        try:
            wandb.config.update(params)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifact to W&B."""
        try:
            artifact = wandb.Artifact(
                artifact_name or os.path.basename(artifact_path),
                type="dataset"
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_path}: {e}")
    
    def log_model(self, model: Any, model_name: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Log model to W&B."""
        try:
            # Save model to temporary directory
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "model.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create model artifact
            model_artifact = wandb.Artifact(
                model_name,
                type="model",
                description=metadata.description if metadata else None,
                metadata=asdict(metadata) if metadata else None
            )
            model_artifact.add_file(model_path)
            
            wandb.log_artifact(model_artifact)
            shutil.rmtree(temp_dir)
            
            return f"wandb:///{wandb.run.entity}/{wandb.run.project}/{model_name}:latest"
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
            raise
    
    def end_experiment(self) -> None:
        """End W&B run."""
        try:
            if self.current_run:
                wandb.finish()
                self.current_run = None
        except Exception as e:
            logger.error(f"Failed to end W&B run: {e}")

class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize TensorBoard tracker.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        if not HAS_TENSORBOARD:
            raise ImportError("TensorBoard is not installed. Install with: pip install tensorboard or tensorboardX")
        
        self.log_dir = log_dir
        self.writer = None
        self.experiment_id = None
        self.step_counter = defaultdict(int)
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start TensorBoard logging."""
        try:
            self.experiment_id = str(uuid.uuid4())
            experiment_dir = os.path.join(self.log_dir, config.name, self.experiment_id)
            os.makedirs(experiment_dir, exist_ok=True)
            
            self.writer = SummaryWriter(experiment_dir)
            
            # Log configuration as text
            if config.parameters:
                config_text = "\n".join([f"{k}: {v}" for k, v in config.parameters.items()])
                self.writer.add_text("config/parameters", config_text)
            
            if config.metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in config.metadata.items()])
                self.writer.add_text("config/metadata", metadata_text)
            
            if config.description:
                self.writer.add_text("config/description", config.description)
            
            return self.experiment_id
        except Exception as e:
            logger.error(f"Failed to start TensorBoard logging: {e}")
            raise
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to TensorBoard."""
        try:
            if self.writer is None:
                return
            
            if step is None:
                step = self.step_counter[key]
                self.step_counter[key] += 1
            
            self.writer.add_scalar(key, value, step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to TensorBoard."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log parameter to TensorBoard as text."""
        try:
            if self.writer is None:
                return
            self.writer.add_text(f"parameters/{key}", str(value))
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to TensorBoard."""
        for key, value in params.items():
            self.log_parameter(key, value)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifact to TensorBoard (as file reference)."""
        try:
            if self.writer is None:
                return
            
            # Log artifact path as text
            name = artifact_name or os.path.basename(artifact_path)
            self.writer.add_text(f"artifacts/{name}", artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_path}: {e}")
    
    def log_model(self, model: Any, model_name: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Log model to TensorBoard."""
        try:
            if self.writer is None:
                return ""
            
            # Save model to experiment directory
            model_dir = os.path.join(os.path.dirname(self.writer.log_dir), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Log model metadata
            if metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in asdict(metadata).items() if v is not None])
                self.writer.add_text(f"models/{model_name}/metadata", metadata_text)
            
            self.writer.add_text(f"models/{model_name}/path", model_path)
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
            raise
    
    def end_experiment(self) -> None:
        """End TensorBoard logging."""
        try:
            if self.writer:
                self.writer.close()
                self.writer = None
        except Exception as e:
            logger.error(f"Failed to end TensorBoard logging: {e}")

class CompositeTracker(ExperimentTracker):
    """Composite tracker that logs to multiple backends."""
    
    def __init__(self, trackers: List[ExperimentTracker]):
        """Initialize composite tracker.
        
        Args:
            trackers: List of individual trackers
        """
        self.trackers = trackers
        self.experiment_ids = {}
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start experiment on all trackers."""
        for tracker in self.trackers:
            try:
                experiment_id = tracker.start_experiment(config)
                self.experiment_ids[type(tracker).__name__] = experiment_id
            except Exception as e:
                logger.error(f"Failed to start experiment on {type(tracker).__name__}: {e}")
        
        return str(uuid.uuid4())  # Return composite ID
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metric(key, value, step)
            except Exception as e:
                logger.error(f"Failed to log metric on {type(tracker).__name__}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Failed to log metrics on {type(tracker).__name__}: {e}")
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log parameter to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_parameter(key, value)
            except Exception as e:
                logger.error(f"Failed to log parameter on {type(tracker).__name__}: {e}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_parameters(params)
            except Exception as e:
                logger.error(f"Failed to log parameters on {type(tracker).__name__}: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                logger.error(f"Failed to log artifact on {type(tracker).__name__}: {e}")
    
    def log_model(self, model: Any, model_name: str, metadata: Optional[ModelMetadata] = None) -> str:
        """Log model to all trackers."""
        model_uris = {}
        for tracker in self.trackers:
            try:
                uri = tracker.log_model(model, model_name, metadata)
                model_uris[type(tracker).__name__] = uri
            except Exception as e:
                logger.error(f"Failed to log model on {type(tracker).__name__}: {e}")
        
        return json.dumps(model_uris)
    
    def end_experiment(self) -> None:
        """End experiment on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.end_experiment()
            except Exception as e:
                logger.error(f"Failed to end experiment on {type(tracker).__name__}: {e}")

class ModelRegistry:
    """Model registry for managing model lifecycle."""
    
    def __init__(self, storage_path: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            storage_path: Path to store model registry data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.storage_path / "models.json"
        self.models = self._load_models()
        
        self._lock = threading.Lock()
    
    def _load_models(self) -> Dict[str, List[Dict]]:
        """Load models from registry file."""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
        return {}
    
    def _save_models(self) -> None:
        """Save models to registry file."""
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, metadata: ModelMetadata, model_uri: str) -> str:
        """Register a model version.
        
        Args:
            metadata: Model metadata
            model_uri: URI to model artifact
            
        Returns:
            Model version ID
        """
        with self._lock:
            if metadata.name not in self.models:
                self.models[metadata.name] = []
            
            version_id = f"v{len(self.models[metadata.name]) + 1}"
            
            model_record = {
                "version": version_id,
                "uri": model_uri,
                "metadata": asdict(metadata),
                "registered_at": datetime.utcnow().isoformat(),
                "status": "staging"
            }
            
            self.models[metadata.name].append(model_record)
            self._save_models()
            
            return version_id
    
    def get_model(self, name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Optional[Dict]:
        """Get model by name and version or stage.
        
        Args:
            name: Model name
            version: Specific version (e.g., "v1")
            stage: Model stage ("staging", "production", "archived")
            
        Returns:
            Model record or None
        """
        if name not in self.models:
            return None
        
        models = self.models[name]
        
        if version:
            for model in models:
                if model["version"] == version:
                    return model
        elif stage:
            for model in models:
                if model.get("status") == stage:
                    return model
        else:
            # Return latest version
            return models[-1] if models else None
        
        return None
    
    def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a specific stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage ("staging", "production", "archived")
            
        Returns:
            Success status
        """
        with self._lock:
            if name not in self.models:
                return False
            
            for model in self.models[name]:
                if model["version"] == version:
                    # Demote other models from production if promoting to production
                    if stage == "production":
                        for other_model in self.models[name]:
                            if other_model.get("status") == "production":
                                other_model["status"] = "staging"
                    
                    model["status"] = stage
                    model["promoted_at"] = datetime.utcnow().isoformat()
                    self._save_models()
                    return True
            
            return False
    
    def list_models(self, name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List models in registry.
        
        Args:
            name: Specific model name to filter by
            
        Returns:
            Dictionary of models
        """
        if name:
            return {name: self.models.get(name, [])}
        return self.models.copy()
    
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete model or specific version.
        
        Args:
            name: Model name
            version: Specific version to delete (if None, deletes all versions)
            
        Returns:
            Success status
        """
        with self._lock:
            if name not in self.models:
                return False
            
            if version:
                self.models[name] = [
                    model for model in self.models[name]
                    if model["version"] != version
                ]
                if not self.models[name]:
                    del self.models[name]
            else:
                del self.models[name]
            
            self._save_models()
            return True

class ExperimentManager:
    """High-level experiment management interface."""
    
    def __init__(self, 
                 tracker: Optional[ExperimentTracker] = None,
                 registry: Optional[ModelRegistry] = None):
        """Initialize experiment manager.
        
        Args:
            tracker: Experiment tracker instance
            registry: Model registry instance
        """
        self.tracker = tracker
        self.registry = registry or ModelRegistry()
        self.current_experiment = None
        
    def create_tracker(self, 
                      backend: str = "tensorboard",
                      **kwargs) -> ExperimentTracker:
        """Create a tracker instance.
        
        Args:
            backend: Tracker backend ("mlflow", "wandb", "tensorboard", "composite")
            **kwargs: Backend-specific arguments
            
        Returns:
            Tracker instance
        """
        if backend == "mlflow":
            return MLflowTracker(**kwargs)
        elif backend == "wandb":
            return WandBTracker(**kwargs)
        elif backend == "tensorboard":
            return TensorBoardTracker(**kwargs)
        elif backend == "composite":
            trackers = kwargs.get("trackers", [])
            return CompositeTracker(trackers)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def start_experiment(self, 
                        name: str,
                        project: Optional[str] = None,
                        parameters: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None,
                        description: Optional[str] = None,
                        tracker_backend: str = "tensorboard",
                        **tracker_kwargs) -> str:
        """Start a new experiment.
        
        Args:
            name: Experiment name
            project: Project name
            parameters: Experiment parameters
            tags: Experiment tags
            description: Experiment description
            tracker_backend: Tracking backend
            **tracker_kwargs: Tracker-specific arguments
            
        Returns:
            Experiment ID
        """
        if self.tracker is None:
            self.tracker = self.create_tracker(tracker_backend, **tracker_kwargs)
        
        config = ExperimentConfig(
            name=name,
            project=project,
            parameters=parameters,
            tags=tags,
            description=description
        )
        
        experiment_id = self.tracker.start_experiment(config)
        self.current_experiment = experiment_id
        
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current experiment."""
        if self.tracker:
            self.tracker.log_metrics(metrics, step)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to current experiment."""
        if self.tracker:
            self.tracker.log_parameters(params)
    
    def save_model(self, 
                   model: Any,
                   name: str,
                   version: Optional[str] = None,
                   description: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   metrics: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
        """Save model to registry and tracker.
        
        Args:
            model: Model to save
            name: Model name
            version: Model version
            description: Model description
            tags: Model tags
            metrics: Model performance metrics
            
        Returns:
            Tuple of (model_uri, version_id)
        """
        metadata = ModelMetadata(
            name=name,
            version=version or "latest",
            description=description,
            tags=tags,
            metrics=metrics,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Log to tracker
        model_uri = ""
        if self.tracker:
            model_uri = self.tracker.log_model(model, name, metadata)
        
        # Register in registry
        version_id = self.registry.register_model(metadata, model_uri)
        
        return model_uri, version_id
    
    def load_model(self, name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Optional[Any]:
        """Load model from registry.
        
        Args:
            name: Model name
            version: Model version
            stage: Model stage
            
        Returns:
            Model object or None
        """
        model_record = self.registry.get_model(name, version, stage)
        if not model_record:
            return None
        
        try:
            # For simplicity, assume pickle format
            # In practice, this would handle different model formats
            model_uri = model_record["uri"]
            if model_uri.endswith(".pkl"):
                with open(model_uri, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
        
        return None
    
    def end_experiment(self) -> None:
        """End current experiment."""
        if self.tracker:
            self.tracker.end_experiment()
        self.current_experiment = None

# Global experiment manager instance
_global_manager = None

def get_experiment_manager() -> ExperimentManager:
    """Get global experiment manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ExperimentManager()
    return _global_manager

# Convenience functions
def start_experiment(name: str, **kwargs) -> str:
    """Start experiment using global manager."""
    return get_experiment_manager().start_experiment(name, **kwargs)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics using global manager."""
    get_experiment_manager().log_metrics(metrics, step)

def log_parameters(params: Dict[str, Any]) -> None:
    """Log parameters using global manager."""
    get_experiment_manager().log_parameters(params)

def save_model(model: Any, name: str, **kwargs) -> Tuple[str, str]:
    """Save model using global manager."""
    return get_experiment_manager().save_model(model, name, **kwargs)

def load_model(name: str, **kwargs) -> Optional[Any]:
    """Load model using global manager."""
    return get_experiment_manager().load_model(name, **kwargs)

def end_experiment() -> None:
    """End experiment using global manager."""
    get_experiment_manager().end_experiment()

__all__ = [
    'ExperimentConfig',
    'ModelMetadata', 
    'ExperimentTracker',
    'MLflowTracker',
    'WandBTracker',
    'TensorBoardTracker',
    'CompositeTracker',
    'ModelRegistry',
    'ExperimentManager',
    'get_experiment_manager',
    'start_experiment',
    'log_metrics',
    'log_parameters',
    'save_model',
    'load_model',
    'end_experiment'
]