"""
Distributed Training Support for TrustFormers

This module provides comprehensive distributed training capabilities including:
- PyTorch DistributedDataParallel (DDP) wrapper
- Horovod integration
- Multi-node coordination utilities
- Fault tolerance mechanisms
- Elastic training support

Features:
- Automatic distributed setup and teardown
- Multi-GPU and multi-node training
- Gradient synchronization and communication optimization
- Fault tolerance with checkpointing and recovery
- Elastic scaling for dynamic resource allocation
- Performance monitoring and optimization
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import socket
import hashlib

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.optim import ZeroRedundancyOptimizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

if TYPE_CHECKING:
    if TORCH_AVAILABLE:
        import torch
        import torch.nn as nn
        TorchModule = nn.Module
        TorchOptimizer = torch.optim.Optimizer
        TorchTensor = torch.Tensor
    else:
        TorchModule = Any
        TorchOptimizer = Any
        TorchTensor = Any
else:
    TorchModule = Any
    TorchOptimizer = Any
    TorchTensor = Any


class DistributedBackend:
    """Enumeration of supported distributed backends"""
    NCCL = "nccl"
    GLOO = "gloo" 
    MPI = "mpi"
    HOROVOD = "horovod"


class DistributedConfig:
    """Configuration for distributed training"""
    
    def __init__(
        self,
        backend: str = DistributedBackend.NCCL,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: str = "localhost",
        master_port: str = "29500",
        use_env: bool = True,
        timeout_seconds: int = 1800,
        find_unused_parameters: bool = False,
        bucket_cap_mb: int = 25,
        gradient_as_bucket_view: bool = True,
    ):
        self.backend = backend
        self.world_size = world_size or self._get_world_size()
        self.rank = rank or self._get_rank()
        self.local_rank = local_rank or self._get_local_rank()
        self.master_addr = master_addr
        self.master_port = master_port
        self.use_env = use_env
        self.timeout_seconds = timeout_seconds
        self.find_unused_parameters = find_unused_parameters
        self.bucket_cap_mb = bucket_cap_mb
        self.gradient_as_bucket_view = gradient_as_bucket_view
        
    def _get_world_size(self) -> int:
        """Get world size from environment or default to 1"""
        return int(os.environ.get("WORLD_SIZE", 1))
    
    def _get_rank(self) -> int:
        """Get rank from environment or default to 0"""
        return int(os.environ.get("RANK", 0))
    
    def _get_local_rank(self) -> int:
        """Get local rank from environment or default to 0"""
        return int(os.environ.get("LOCAL_RANK", 0))
    
    def setup_environment(self):
        """Setup environment variables for distributed training"""
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)


class DistributedManager:
    """Manages distributed training setup and coordination"""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._process_group = None
        
    def initialize(self):
        """Initialize distributed training"""
        if self._is_initialized:
            return
            
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for distributed training")
            
        self.config.setup_environment()
        
        if self.config.backend == DistributedBackend.HOROVOD:
            self._initialize_horovod()
        else:
            self._initialize_pytorch()
            
        self._is_initialized = True
        self.logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
    
    def _initialize_pytorch(self):
        """Initialize PyTorch distributed training"""
        if self.config.use_env:
            dist.init_process_group(
                backend=self.config.backend,
                timeout=torch.timedelta(seconds=self.config.timeout_seconds)
            )
        else:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.timedelta(seconds=self.config.timeout_seconds)
            )
            
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
    
    def _initialize_horovod(self):
        """Initialize Horovod distributed training"""
        if not HOROVOD_AVAILABLE:
            raise RuntimeError("Horovod is not available")
        hvd.init()
        
        # Set CUDA device for Horovod
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self._is_initialized and self.config.backend != DistributedBackend.HOROVOD:
            dist.destroy_process_group()
        self._is_initialized = False
        self.logger.info("Distributed training cleaned up")
    
    @property
    def rank(self) -> int:
        """Get current process rank"""
        if self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            return hvd.rank()
        elif self._is_initialized:
            return dist.get_rank()
        return self.config.rank
    
    @property
    def world_size(self) -> int:
        """Get total number of processes"""
        if self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            return hvd.size()
        elif self._is_initialized:
            return dist.get_world_size()
        return self.config.world_size
    
    @property
    def local_rank(self) -> int:
        """Get local rank within the node"""
        if self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            return hvd.local_rank()
        return self.config.local_rank
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0


class DDPModelWrapper:
    """Wrapper for PyTorch DistributedDataParallel"""
    
    def __init__(
        self,
        model: TorchModule,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
        bucket_cap_mb: int = 25,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DDP")
            
        self.original_model = model
        self.device_ids = device_ids
        self.output_device = output_device
        
        # Wrap model with DDP
        self.model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, *args, **kwargs):
        """Forward pass through DDP model"""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable"""
        return self.forward(*args, **kwargs)
    
    def state_dict(self):
        """Get state dict from original model"""
        return self.original_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict into original model"""
        return self.original_model.load_state_dict(state_dict)
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters"""
        return self.model.named_parameters()


class HorovodModelWrapper:
    """Wrapper for Horovod distributed training"""
    
    def __init__(self, model: TorchModule, optimizer: TorchOptimizer):
        if not HOROVOD_AVAILABLE:
            raise RuntimeError("Horovod is not available")
            
        self.model = model
        self.optimizer = optimizer
        
        # Broadcast parameters from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        
        # Broadcast optimizer state from rank 0 to all other processes
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        
        # Add Horovod distributed optimizer
        self.distributed_optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters()
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, *args, **kwargs):
        """Forward pass through model"""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable"""
        return self.forward(*args, **kwargs)
    
    def step(self):
        """Optimizer step with gradient averaging"""
        self.distributed_optimizer.step()
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()


class FaultToleranceManager:
    """Manages fault tolerance for distributed training"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        checkpoint_interval: int = 100,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.logger = logging.getLogger(__name__)
        self._failed_ranks = set()
        self._checkpoint_metadata = {}
    
    def save_checkpoint(
        self,
        model: TorchModule,
        optimizer: TorchOptimizer,
        epoch: int,
        step: int,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metadata': metadata or {},
            'timestamp': time.time(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        self._checkpoint_metadata[step] = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'timestamp': checkpoint['timestamp'],
        }
        
        # Save metadata
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._checkpoint_metadata, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_latest_checkpoint(
        self,
        model: TorchModule,
        optimizer: TorchOptimizer,
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint"""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        
        if not metadata_path.exists():
            self.logger.info("No checkpoint metadata found")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if not metadata:
            self.logger.info("No checkpoints available")
            return None
        
        # Find latest checkpoint
        latest_step = max(int(step) for step in metadata.keys())
        latest_checkpoint = metadata[str(latest_step)]
        
        checkpoint_path = Path(latest_checkpoint['path'])
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step"""
        return step % self.checkpoint_interval == 0


class ElasticTrainingManager:
    """Manages elastic training with dynamic scaling"""
    
    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 8,
        node_discovery_interval: float = 30.0,
        scaling_cooldown: float = 300.0,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.node_discovery_interval = node_discovery_interval
        self.scaling_cooldown = scaling_cooldown
        
        self.logger = logging.getLogger(__name__)
        self._active_nodes = set()
        self._last_scaling_time = time.monotonic()
        self._discovery_thread = None
        self._should_stop = threading.Event()
    
    def start_discovery(self):
        """Start node discovery thread"""
        if self._discovery_thread is not None:
            return
            
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True
        )
        self._discovery_thread.start()
        self.logger.info("Node discovery started")
    
    def stop_discovery(self):
        """Stop node discovery"""
        if self._discovery_thread is None:
            return
            
        self._should_stop.set()
        self._discovery_thread.join()
        self._discovery_thread = None
        self.logger.info("Node discovery stopped")
    
    def _discovery_loop(self):
        """Node discovery loop"""
        while not self._should_stop.wait(self.node_discovery_interval):
            try:
                self._discover_nodes()
                self._check_scaling_conditions()
            except Exception as e:
                self.logger.error(f"Error in node discovery: {e}")
    
    def _discover_nodes(self):
        """Discover available nodes"""
        # This is a simplified implementation
        # In practice, this would integrate with orchestration systems
        # like Kubernetes, SLURM, etc.
        pass
    
    def _check_scaling_conditions(self):
        """Check if scaling is needed"""
        current_time = time.monotonic()
        if current_time - self._last_scaling_time < self.scaling_cooldown:
            return
        
        # Implement scaling logic based on workload, resources, etc.
        # This is a placeholder for actual scaling implementation
        pass
    
    def scale_up(self, target_nodes: int):
        """Scale up to target number of nodes"""
        if target_nodes > self.max_nodes:
            target_nodes = self.max_nodes
            
        self.logger.info(f"Scaling up to {target_nodes} nodes")
        # Implement actual scaling logic
        # Use monotonic time for more reliable timing
        self._last_scaling_time = time.monotonic()
    
    def scale_down(self, target_nodes: int):
        """Scale down to target number of nodes"""
        if target_nodes < self.min_nodes:
            target_nodes = self.min_nodes
            
        self.logger.info(f"Scaling down to {target_nodes} nodes")
        # Implement actual scaling logic  
        # Use monotonic time for more reliable timing
        self._last_scaling_time = time.monotonic()


class DistributedTrainer:
    """High-level distributed training orchestrator"""
    
    def __init__(
        self,
        model: TorchModule,
        optimizer: TorchOptimizer,
        config: Optional[DistributedConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        fault_tolerance: bool = True,
        elastic_training: bool = False,
    ):
        self.config = config or DistributedConfig()
        self.dist_manager = DistributedManager(self.config)
        
        # Fault tolerance
        self.fault_tolerance_manager = None
        if fault_tolerance and checkpoint_dir:
            self.fault_tolerance_manager = FaultToleranceManager(checkpoint_dir)
        
        # Elastic training
        self.elastic_manager = None
        if elastic_training:
            self.elastic_manager = ElasticTrainingManager()
        
        # Model wrapper
        self.model_wrapper = None
        self.original_model = model
        self.optimizer = optimizer
        
        self.logger = logging.getLogger(__name__)
        
    def setup(self):
        """Setup distributed training"""
        self.dist_manager.initialize()
        
        # Setup model wrapper based on backend
        if self.config.backend == DistributedBackend.HOROVOD:
            self.model_wrapper = HorovodModelWrapper(self.original_model, self.optimizer)
        else:
            device_ids = [self.dist_manager.local_rank] if torch.cuda.is_available() else None
            self.model_wrapper = DDPModelWrapper(
                self.original_model,
                device_ids=device_ids,
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            )
        
        # Start elastic training if enabled
        if self.elastic_manager:
            self.elastic_manager.start_discovery()
        
        self.logger.info("Distributed trainer setup complete")
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.elastic_manager:
            self.elastic_manager.stop_discovery()
        
        self.dist_manager.cleanup()
        self.logger.info("Distributed trainer cleanup complete")
    
    def train_step(
        self,
        batch: Any,
        loss_fn: Callable,
        epoch: int,
        step: int,
    ) -> float:
        """Perform a single training step"""
        # Forward pass
        if hasattr(self.model_wrapper, 'forward'):
            outputs = self.model_wrapper.forward(batch)
        else:
            outputs = self.model_wrapper(batch)
        
        # Compute loss
        loss = loss_fn(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        if hasattr(self.model_wrapper, 'step'):
            self.model_wrapper.step()
        else:
            self.optimizer.step()
        
        # Zero gradients
        if hasattr(self.model_wrapper, 'zero_grad'):
            self.model_wrapper.zero_grad()
        else:
            self.optimizer.zero_grad()
        
        # Checkpoint if needed
        if (self.fault_tolerance_manager and 
            self.fault_tolerance_manager.should_checkpoint(step) and
            self.dist_manager.is_main_process):
            
            self.fault_tolerance_manager.save_checkpoint(
                self.original_model,
                self.optimizer,
                epoch,
                step,
                loss.item(),
            )
        
        return loss.item()
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint"""
        if not self.fault_tolerance_manager:
            return None
        
        return self.fault_tolerance_manager.load_latest_checkpoint(
            self.original_model,
            self.optimizer,
        )
    
    @contextmanager
    def distributed_context(self):
        """Context manager for distributed training"""
        try:
            self.setup()
            yield self
        finally:
            self.cleanup()


# Utility functions for distributed training
def get_free_port() -> int:
    """Get a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_logging_for_distributed(rank: int, log_level: int = logging.INFO):
    """Setup logging for distributed training"""
    logging.basicConfig(
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=log_level
    )


def all_gather_object(obj: Any, world_size: int) -> List[Any]:
    """Gather objects from all processes"""
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return [obj]
    
    gathered_objects = [None] * world_size
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from src rank to all processes"""
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier():
    """Synchronize all processes"""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: TorchTensor, op: str = "mean") -> TorchTensor:
    """Reduce tensor across all processes"""
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if op == "mean":
        rt /= dist.get_world_size()
    
    return rt


# Global instances for convenience
_default_config = None
_default_manager = None


def get_default_distributed_manager() -> DistributedManager:
    """Get default distributed manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = DistributedManager()
    return _default_manager


def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return (TORCH_AVAILABLE and 
            (dist.is_available() and dist.is_initialized()) or
            (HOROVOD_AVAILABLE and hvd.is_initialized()))


def get_rank() -> int:
    """Get current process rank"""
    if HOROVOD_AVAILABLE and hvd.is_initialized():
        return hvd.rank()
    elif TORCH_AVAILABLE and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if HOROVOD_AVAILABLE and hvd.is_initialized():
        return hvd.size()
    elif TORCH_AVAILABLE and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process"""
    return get_rank() == 0


# Export public API
__all__ = [
    # Core classes
    'DistributedConfig',
    'DistributedManager', 
    'DistributedTrainer',
    'DDPModelWrapper',
    'HorovodModelWrapper',
    
    # Advanced features
    'FaultToleranceManager',
    'ElasticTrainingManager',
    
    # Utilities
    'get_free_port',
    'setup_logging_for_distributed',
    'all_gather_object',
    'broadcast_object',
    'barrier',
    'reduce_tensor',
    
    # Global functions
    'get_default_distributed_manager',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_main_process',
    
    # Constants
    'DistributedBackend',
]