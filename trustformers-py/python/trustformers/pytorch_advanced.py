#!/usr/bin/env python3
"""
Advanced PyTorch Integration for Trustformers

This module provides advanced PyTorch features including:
- Gradient flow integration and analysis
- Mixed precision training support  
- Advanced optimization utilities
- Gradient hooks and monitoring
- Memory optimization for large models
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator, TYPE_CHECKING
from pathlib import Path
import json
import threading

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    GradScaler = None
    autocast = None
    SummaryWriter = None
    HAS_TORCH = False

if TYPE_CHECKING:
    if HAS_TORCH:
        import torch
        import torch.nn as nn
        TorchModule = nn.Module
        TorchTensor = torch.Tensor
        TorchOptimizer = torch.optim.Optimizer
    else:
        TorchModule = Any
        TorchTensor = Any
        TorchOptimizer = Any
else:
    TorchModule = Any
    TorchTensor = Any
    TorchOptimizer = Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Import existing PyTorch utilities
try:
    from .torch_utils import tensor_to_torch, torch_to_tensor
    from .torch_module_compat import TorchModuleWrapper
except ImportError:
    # Fallback imports if running standalone
    tensor_to_torch = None
    torch_to_tensor = None
    TorchModuleWrapper = None

# Import trustformers types
try:
    from . import Tensor, PreTrainedModel
except ImportError:
    Tensor = Any
    PreTrainedModel = Any

logger = logging.getLogger(__name__)

@dataclass
class GradientStats:
    """Statistics for gradient analysis."""
    layer_name: str
    grad_norm: float
    grad_mean: float
    grad_std: float
    grad_min: float
    grad_max: float
    param_norm: float
    update_ratio: float
    timestamp: float

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    enabled: bool = True
    init_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    dtype: str = "float16"  # "float16" or "bfloat16"
    opt_level: str = "O1"  # O0, O1, O2, O3
    loss_scale: Optional[float] = None
    dynamic_loss_scale: bool = True

class GradientFlowAnalyzer:
    """Analyzer for gradient flow through model layers."""
    
    def __init__(self, model: TorchModule, log_dir: Optional[str] = None):
        """Initialize gradient flow analyzer.
        
        Args:
            model: PyTorch model to analyze
            log_dir: Directory for logging results
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        self.model = model
        self.log_dir = Path(log_dir) if log_dir else None
        self.gradient_stats: List[GradientStats] = []
        self.hooks = []
        self.step_count = 0
        self.layer_stats = defaultdict(list)
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir / "gradient_flow") if SummaryWriter else None
        else:
            self.writer = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register backward hooks for gradient monitoring."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_grad_hook(name))
                self.hooks.append(hook)
    
    def _make_grad_hook(self, name: str) -> Callable:
        """Create gradient hook for a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Hook function
        """
        def grad_hook(grad):
            if grad is not None:
                self._analyze_gradient(name, grad)
        return grad_hook
    
    def _analyze_gradient(self, layer_name: str, grad: TorchTensor):
        """Analyze gradient statistics.
        
        Args:
            layer_name: Name of the layer
            grad: Gradient tensor
        """
        # Compute gradient statistics
        grad_norm = grad.norm().item()
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_min = grad.min().item()
        grad_max = grad.max().item()
        
        # Get parameter for comparison
        param = dict(self.model.named_parameters())[layer_name]
        param_norm = param.norm().item()
        
        # Compute update ratio (gradient norm / parameter norm)
        update_ratio = grad_norm / (param_norm + 1e-8)
        
        stats = GradientStats(
            layer_name=layer_name,
            grad_norm=grad_norm,
            grad_mean=grad_mean,
            grad_std=grad_std,
            grad_min=grad_min,
            grad_max=grad_max,
            param_norm=param_norm,
            update_ratio=update_ratio,
            timestamp=time.time()
        )
        
        self.gradient_stats.append(stats)
        self.layer_stats[layer_name].append(stats)
        
        # Log to tensorboard if available
        if self.writer:
            self.writer.add_scalar(f"gradient_norm/{layer_name}", grad_norm, self.step_count)
            self.writer.add_scalar(f"gradient_mean/{layer_name}", grad_mean, self.step_count)
            self.writer.add_scalar(f"gradient_std/{layer_name}", grad_std, self.step_count)
            self.writer.add_scalar(f"update_ratio/{layer_name}", update_ratio, self.step_count)
    
    def step(self):
        """Call after each optimization step."""
        self.step_count += 1
    
    def get_gradient_flow_report(self) -> Dict[str, Any]:
        """Generate comprehensive gradient flow report.
        
        Returns:
            Report dictionary
        """
        if not self.gradient_stats:
            return {"error": "No gradient statistics collected"}
        
        report = {
            "total_steps": self.step_count,
            "total_gradients_analyzed": len(self.gradient_stats),
            "layer_analysis": {},
            "overall_stats": {},
            "potential_issues": []
        }
        
        # Analyze each layer
        for layer_name, stats_list in self.layer_stats.items():
            if not stats_list:
                continue
                
            recent_stats = stats_list[-10:]  # Last 10 gradients
            
            avg_grad_norm = np.mean([s.grad_norm for s in recent_stats])
            avg_update_ratio = np.mean([s.update_ratio for s in recent_stats])
            grad_variance = np.var([s.grad_norm for s in recent_stats])
            
            layer_analysis = {
                "avg_gradient_norm": avg_grad_norm,
                "avg_update_ratio": avg_update_ratio,
                "gradient_variance": grad_variance,
                "num_updates": len(stats_list)
            }
            
            # Detect potential issues
            if avg_grad_norm < 1e-7:
                report["potential_issues"].append(f"Vanishing gradients in {layer_name}")
            elif avg_grad_norm > 10.0:
                report["potential_issues"].append(f"Exploding gradients in {layer_name}")
            
            if avg_update_ratio < 1e-6:
                report["potential_issues"].append(f"Very small updates in {layer_name}")
            elif avg_update_ratio > 1.0:
                report["potential_issues"].append(f"Very large updates in {layer_name}")
            
            report["layer_analysis"][layer_name] = layer_analysis
        
        # Overall statistics
        all_grad_norms = [s.grad_norm for s in self.gradient_stats[-100:]]  # Last 100
        if all_grad_norms:
            report["overall_stats"] = {
                "avg_gradient_norm": np.mean(all_grad_norms),
                "max_gradient_norm": np.max(all_grad_norms),
                "min_gradient_norm": np.min(all_grad_norms),
                "gradient_norm_std": np.std(all_grad_norms)
            }
        
        return report
    
    def plot_gradient_flow(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Plot gradient flow visualization.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure if available
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.layer_stats:
                logger.warning("No gradient statistics to plot")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Gradient norms over time
            ax1 = axes[0, 0]
            for layer_name, stats_list in self.layer_stats.items():
                grad_norms = [s.grad_norm for s in stats_list]
                ax1.plot(grad_norms, label=layer_name, alpha=0.7)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Gradient Norm')
            ax1.set_title('Gradient Norms Over Time')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_yscale('log')
            
            # Plot 2: Update ratios
            ax2 = axes[0, 1]
            for layer_name, stats_list in self.layer_stats.items():
                update_ratios = [s.update_ratio for s in stats_list]
                ax2.plot(update_ratios, label=layer_name, alpha=0.7)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Update Ratio')
            ax2.set_title('Update Ratios Over Time')
            ax2.set_yscale('log')
            
            # Plot 3: Gradient distribution (most recent)
            ax3 = axes[1, 0]
            recent_grad_norms = []
            layer_names = []
            for layer_name, stats_list in self.layer_stats.items():
                if stats_list:
                    recent_grad_norms.append(stats_list[-1].grad_norm)
                    layer_names.append(layer_name)
            
            if recent_grad_norms:
                ax3.bar(range(len(recent_grad_norms)), recent_grad_norms)
                ax3.set_xticks(range(len(layer_names)))
                ax3.set_xticklabels(layer_names, rotation=45, ha='right')
                ax3.set_ylabel('Gradient Norm')
                ax3.set_title('Most Recent Gradient Norms by Layer')
                ax3.set_yscale('log')
            
            # Plot 4: Gradient variance
            ax4 = axes[1, 1]
            variances = []
            for layer_name, stats_list in self.layer_stats.items():
                if len(stats_list) > 1:
                    grad_norms = [s.grad_norm for s in stats_list[-20:]]  # Last 20
                    variances.append(np.var(grad_norms))
                else:
                    variances.append(0)
            
            if variances and layer_names:
                ax4.bar(range(len(variances)), variances)
                ax4.set_xticks(range(len(layer_names)))
                ax4.set_xticklabels(layer_names, rotation=45, ha='right')
                ax4.set_ylabel('Gradient Variance')
                ax4.set_title('Gradient Variance by Layer')
                ax4.set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gradient flow plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def cleanup(self):
        """Clean up hooks and resources."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        if self.writer:
            self.writer.close()

class MixedPrecisionTrainer:
    """Mixed precision training manager."""
    
    def __init__(self, 
                 model: TorchModule,
                 optimizer: TorchOptimizer,
                 config: Optional[MixedPrecisionConfig] = None):
        """Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            config: Mixed precision configuration
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        self.model = model
        self.optimizer = optimizer
        self.config = config or MixedPrecisionConfig()
        
        # Initialize gradient scaler for mixed precision
        if self.config.enabled and torch.cuda.is_available():
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=True
            )
        else:
            self.scaler = None
            if self.config.enabled:
                logger.warning("CUDA not available, disabling mixed precision training")
                self.config.enabled = False
        
        # Statistics tracking
        self.step_count = 0
        self.scale_history = []
        self.overflow_count = 0
        self.effective_updates = 0
        
        # Determine precision dtype
        if self.config.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16 if self.config.enabled else torch.float32
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.config.enabled and autocast is not None:
            with autocast(dtype=self.dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: TorchTensor) -> TorchTensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Original loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def backward_and_step(self, 
                         loss: TorchTensor,
                         retain_graph: bool = False,
                         create_graph: bool = False) -> bool:
        """Perform backward pass and optimizer step with mixed precision.
        
        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
            create_graph: Whether to create graph for higher-order derivatives
            
        Returns:
            True if step was taken (no overflow), False otherwise
        """
        self.step_count += 1
        
        if self.scaler is not None:
            # Scale the loss to prevent underflow
            scaled_loss = self.scaler.scale(loss)
            
            # Backward pass
            scaled_loss.backward(retain_graph=retain_graph, create_graph=create_graph)
            
            # Check for overflow before stepping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping can be applied here
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track statistics
            current_scale = self.scaler.get_scale()
            self.scale_history.append(current_scale)
            
            # Check if step was skipped due to overflow
            if len(self.scale_history) > 1 and current_scale < self.scale_history[-2]:
                self.overflow_count += 1
                return False
            else:
                self.effective_updates += 1
                return True
        else:
            # Standard precision training
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)
            self.optimizer.step()
            self.effective_updates += 1
            return True
    
    def zero_grad(self):
        """Zero gradients with mixed precision awareness."""
        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_steps": self.step_count,
            "effective_updates": self.effective_updates,
            "overflow_count": self.overflow_count,
            "overflow_rate": self.overflow_count / max(self.step_count, 1),
            "mixed_precision_enabled": self.config.enabled,
            "dtype": str(self.dtype),
            "current_scale": self.scaler.get_scale() if self.scaler else 1.0,
            "scale_history_length": len(self.scale_history)
        }
        
        if self.scale_history:
            stats.update({
                "avg_scale": np.mean(self.scale_history),
                "max_scale": np.max(self.scale_history),
                "min_scale": np.min(self.scale_history),
                "scale_std": np.std(self.scale_history)
            })
        
        return stats
    
    def adjust_learning_rate(self, lr_factor: float):
        """Adjust learning rate based on mixed precision performance.
        
        Args:
            lr_factor: Factor to multiply learning rate by
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= lr_factor
    
    def save_state(self, filepath: str):
        """Save mixed precision training state.
        
        Args:
            filepath: Path to save state
        """
        state = {
            "config": self.config.__dict__,
            "step_count": self.step_count,
            "overflow_count": self.overflow_count,
            "effective_updates": self.effective_updates,
            "scale_history": self.scale_history[-100:],  # Keep last 100
            "scaler_state": self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(state, filepath)
        logger.info(f"Mixed precision state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load mixed precision training state.
        
        Args:
            filepath: Path to load state from
        """
        state = torch.load(filepath, map_location='cpu')
        
        self.step_count = state.get("step_count", 0)
        self.overflow_count = state.get("overflow_count", 0)
        self.effective_updates = state.get("effective_updates", 0)
        self.scale_history = state.get("scale_history", [])
        
        if self.scaler and state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])
        
        logger.info(f"Mixed precision state loaded from {filepath}")

class AdvancedOptimizer:
    """Advanced optimization utilities."""
    
    def __init__(self, 
                 model: TorchModule,
                 optimizer: TorchOptimizer,
                 scheduler: Optional[Any] = None):
        """Initialize advanced optimizer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Gradient monitoring
        self.gradient_analyzer = GradientFlowAnalyzer(model)
        
        # Statistics
        self.step_count = 0
        self.lr_history = []
        self.loss_history = deque(maxlen=1000)
        
    def step(self, 
             loss: TorchTensor,
             retain_graph: bool = False) -> Dict[str, float]:
        """Perform optimization step with monitoring.
        
        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
            
        Returns:
            Step statistics
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward(retain_graph=retain_graph)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        # Update analyzers
        self.gradient_analyzer.step()
        
        # Track statistics
        self.step_count += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        self.loss_history.append(loss.item())
        
        return {
            "step": self.step_count,
            "loss": loss.item(),
            "learning_rate": current_lr,
            "gradient_norm": grad_norm.item()
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report.
        
        Returns:
            Optimization report
        """
        report = {
            "total_steps": self.step_count,
            "gradient_flow": self.gradient_analyzer.get_gradient_flow_report(),
            "learning_rate_stats": {},
            "loss_stats": {}
        }
        
        # Learning rate statistics
        if self.lr_history:
            report["learning_rate_stats"] = {
                "current_lr": self.lr_history[-1],
                "initial_lr": self.lr_history[0],
                "avg_lr": np.mean(self.lr_history),
                "max_lr": np.max(self.lr_history),
                "min_lr": np.min(self.lr_history)
            }
        
        # Loss statistics
        if self.loss_history:
            recent_losses = list(self.loss_history)[-100:]  # Last 100
            report["loss_stats"] = {
                "current_loss": recent_losses[-1],
                "avg_loss": np.mean(recent_losses),
                "loss_trend": np.polyfit(range(len(recent_losses)), recent_losses, 1)[0],
                "loss_variance": np.var(recent_losses)
            }
        
        return report

class MemoryOptimizedTrainer:
    """Memory optimization utilities for large model training."""
    
    def __init__(self, model: TorchModule):
        """Initialize memory optimizer.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.activation_checkpointing = False
        self.cpu_offload = False
        
    def enable_activation_checkpointing(self):
        """Enable activation checkpointing to save memory."""
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            self.activation_checkpointing = True
            logger.info("Activation checkpointing enabled")
        else:
            logger.warning("Activation checkpointing not available")
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for model parameters."""
        self.cpu_offload = True
        logger.info("CPU offloading enabled")
    
    @contextmanager
    def memory_efficient_forward(self):
        """Context manager for memory-efficient forward pass."""
        if self.activation_checkpointing:
            # Implement activation checkpointing wrapper
            pass
        
        if self.cpu_offload:
            # Move model to GPU temporarily
            self.model.cuda()
            try:
                yield
            finally:
                # Move back to CPU
                self.model.cpu()
        else:
            yield
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Memory statistics
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                "max_reserved": torch.cuda.max_memory_reserved() / 1024**3
            }
        
        # Model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats["model_parameters"] = {
            "total": total_params,
            "trainable": trainable_params,
            "size_mb": total_params * 4 / 1024**2  # Assume float32
        }
        
        return stats

@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    enabled: bool = True
    accumulation_steps: int = 4
    max_batch_size: int = 32
    dynamic_batching: bool = True
    memory_threshold_gb: float = 8.0

@dataclass
class QuantizationConfig:
    """Configuration for quantization-aware training."""
    enabled: bool = False
    qconfig_spec: Optional[Dict] = None
    backend: str = "fbgemm"  # "fbgemm", "qnnpack"
    prepare_custom_config: Optional[Dict] = None
    convert_custom_config: Optional[Dict] = None

class GradientAccumulator:
    """Advanced gradient accumulation with dynamic batching."""
    
    def __init__(self, config: Optional[GradientAccumulationConfig] = None):
        """Initialize gradient accumulator.
        
        Args:
            config: Gradient accumulation configuration
        """
        self.config = config or GradientAccumulationConfig()
        self.accumulated_steps = 0
        self.batch_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=50)
        
    def should_accumulate(self, current_memory_gb: float) -> bool:
        """Determine if gradients should be accumulated.
        
        Args:
            current_memory_gb: Current memory usage in GB
            
        Returns:
            True if should accumulate, False if should step
        """
        if not self.config.enabled:
            return False
            
        # Always accumulate if we haven't reached minimum steps
        if self.accumulated_steps < self.config.accumulation_steps:
            return True
            
        # Dynamic batching based on memory
        if self.config.dynamic_batching:
            if current_memory_gb > self.config.memory_threshold_gb:
                return False  # Force step to free memory
                
        return self.accumulated_steps < self.config.accumulation_steps
    
    def step(self) -> bool:
        """Record accumulation step.
        
        Returns:
            True if should perform optimizer step
        """
        self.accumulated_steps += 1
        
        if self.accumulated_steps >= self.config.accumulation_steps:
            self.accumulated_steps = 0
            return True
        return False
    
    def reset(self):
        """Reset accumulation counter."""
        self.accumulated_steps = 0

class DynamicLossScaler:
    """Enhanced dynamic loss scaling with adaptive strategies."""
    
    def __init__(self, 
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 min_loss_scale: float = 1.0,
                 max_loss_scale: float = 2.**24):
        """Initialize dynamic loss scaler.
        
        Args:
            init_scale: Initial loss scale
            growth_factor: Factor to grow scale by
            backoff_factor: Factor to reduce scale by
            growth_interval: Steps between scale growth
            min_loss_scale: Minimum allowed scale
            max_loss_scale: Maximum allowed scale
        """
        self.current_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        
        self.growth_tracker = 0
        self.overflow_tracker = deque(maxlen=10)
        self.scale_history = deque(maxlen=1000)
        
    def scale(self, loss: TorchTensor) -> TorchTensor:
        """Scale loss tensor.
        
        Args:
            loss: Original loss
            
        Returns:
            Scaled loss
        """
        return loss * self.current_scale
    
    def unscale_gradients(self, optimizer: TorchOptimizer) -> bool:
        """Unscale gradients and check for overflow.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            True if no overflow detected
        """
        inv_scale = 1.0 / self.current_scale
        overflow_detected = False
        
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
                    
                    # Check for overflow/underflow
                    if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                        overflow_detected = True
                        break
            if overflow_detected:
                break
        
        self.overflow_tracker.append(overflow_detected)
        return not overflow_detected
    
    def update(self, optimizer_stepped: bool):
        """Update loss scale based on training progress.
        
        Args:
            optimizer_stepped: Whether optimizer step was taken
        """
        if optimizer_stepped:
            self.growth_tracker += 1
            
            # Grow scale if we've gone long enough without overflow
            if self.growth_tracker >= self.growth_interval:
                new_scale = min(self.current_scale * self.growth_factor, self.max_loss_scale)
                if new_scale != self.current_scale:
                    self.current_scale = new_scale
                    self.growth_tracker = 0
                    logger.debug(f"Increased loss scale to {self.current_scale}")
        else:
            # Reduce scale on overflow
            new_scale = max(self.current_scale * self.backoff_factor, self.min_loss_scale)
            if new_scale != self.current_scale:
                self.current_scale = new_scale
                self.growth_tracker = 0
                logger.debug(f"Reduced loss scale to {self.current_scale}")
        
        self.scale_history.append(self.current_scale)
    
    def get_adaptive_recommendation(self) -> float:
        """Get recommended scale adjustment based on overflow patterns.
        
        Returns:
            Recommended scale factor
        """
        if len(self.overflow_tracker) < 5:
            return 1.0
            
        recent_overflows = sum(self.overflow_tracker)
        overflow_rate = recent_overflows / len(self.overflow_tracker)
        
        if overflow_rate > 0.3:  # Too many overflows
            return 0.8
        elif overflow_rate < 0.05:  # Very stable
            return 1.2
        else:
            return 1.0

class AdvancedGradientClipper:
    """Advanced gradient clipping with adaptive strategies."""
    
    def __init__(self, 
                 max_norm: float = 1.0,
                 adaptive: bool = True,
                 percentile_clipping: bool = False,
                 percentile: float = 95.0):
        """Initialize advanced gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm
            adaptive: Whether to use adaptive clipping
            percentile_clipping: Whether to use percentile-based clipping
            percentile: Percentile for clipping threshold
        """
        self.max_norm = max_norm
        self.adaptive = adaptive
        self.percentile_clipping = percentile_clipping
        self.percentile = percentile
        
        self.grad_norm_history = deque(maxlen=1000)
        self.clip_history = deque(maxlen=1000)
        
    def clip_gradients(self, parameters) -> float:
        """Clip gradients with advanced strategies.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Total gradient norm before clipping
        """
        # Compute total gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'))
        
        if total_norm.isnan() or total_norm.isinf():
            logger.warning("NaN or Inf detected in gradients")
            # Zero out gradients
            for param in parameters:
                if param.grad is not None:
                    param.grad.zero_()
            return 0.0
        
        self.grad_norm_history.append(total_norm.item())
        
        # Determine clipping threshold
        if self.percentile_clipping and len(self.grad_norm_history) > 50:
            threshold = np.percentile(list(self.grad_norm_history), self.percentile)
        elif self.adaptive and len(self.grad_norm_history) > 10:
            # Adaptive clipping based on running statistics
            recent_norms = list(self.grad_norm_history)[-50:]
            mean_norm = np.mean(recent_norms)
            std_norm = np.std(recent_norms)
            threshold = mean_norm + 2 * std_norm
        else:
            threshold = self.max_norm
        
        # Apply clipping
        if total_norm > threshold:
            clip_factor = threshold / total_norm
            for param in parameters:
                if param.grad is not None:
                    param.grad.mul_(clip_factor)
            self.clip_history.append(True)
            logger.debug(f"Clipped gradients: norm={total_norm:.4f}, threshold={threshold:.4f}")
        else:
            self.clip_history.append(False)
        
        return total_norm.item()
    
    def get_clipping_stats(self) -> Dict[str, Any]:
        """Get gradient clipping statistics.
        
        Returns:
            Clipping statistics
        """
        if not self.grad_norm_history:
            return {}
            
        recent_norms = list(self.grad_norm_history)[-100:]
        recent_clips = list(self.clip_history)[-100:]
        
        return {
            "avg_grad_norm": np.mean(recent_norms),
            "max_grad_norm": np.max(recent_norms),
            "min_grad_norm": np.min(recent_norms),
            "grad_norm_std": np.std(recent_norms),
            "clip_rate": np.mean(recent_clips) if recent_clips else 0.0,
            "total_clips": sum(self.clip_history),
            "total_steps": len(self.clip_history)
        }

class QuantizationAwareTrainer:
    """Quantization-aware training support."""
    
    def __init__(self, 
                 model: TorchModule,
                 config: Optional[QuantizationConfig] = None):
        """Initialize quantization-aware trainer.
        
        Args:
            model: PyTorch model
            config: Quantization configuration
        """
        self.model = model
        self.config = config or QuantizationConfig()
        self.prepared_model = None
        self.is_prepared = False
        
    def prepare_model(self):
        """Prepare model for quantization-aware training."""
        if not self.config.enabled:
            logger.warning("Quantization not enabled")
            return
            
        try:
            from torch.quantization import prepare_qat, get_default_qat_qconfig
            
            # Set quantization config
            if self.config.qconfig_spec:
                self.model.qconfig = self.config.qconfig_spec
            else:
                self.model.qconfig = get_default_qat_qconfig(self.config.backend)
            
            # Prepare model
            self.prepared_model = prepare_qat(
                self.model, 
                inplace=False,
                **self.config.prepare_custom_config or {}
            )
            self.is_prepared = True
            logger.info("Model prepared for quantization-aware training")
            
        except ImportError:
            logger.error("PyTorch quantization not available")
            self.config.enabled = False
    
    def convert_model(self):
        """Convert trained model to quantized version."""
        if not self.is_prepared:
            logger.error("Model not prepared for quantization")
            return None
            
        try:
            from torch.quantization import convert
            
            # Set to eval mode before conversion
            self.prepared_model.eval()
            
            quantized_model = convert(
                self.prepared_model,
                inplace=False,
                **self.config.convert_custom_config or {}
            )
            
            logger.info("Model converted to quantized version")
            return quantized_model
            
        except ImportError:
            logger.error("PyTorch quantization not available")
            return None
    
    def get_model_size_reduction(self, quantized_model) -> Dict[str, float]:
        """Calculate model size reduction from quantization.
        
        Args:
            quantized_model: Quantized model
            
        Returns:
            Size reduction statistics
        """
        if not quantized_model:
            return {}
            
        # Calculate parameter counts
        original_params = sum(p.numel() for p in self.model.parameters())
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        
        # Estimate size reduction (quantized typically uses int8 vs float32)
        original_size_mb = original_params * 4 / 1024**2  # float32 = 4 bytes
        quantized_size_mb = quantized_params * 1 / 1024**2  # int8 = 1 byte (approximate)
        
        return {
            "original_params": original_params,
            "quantized_params": quantized_params,
            "original_size_mb": original_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "size_reduction_ratio": original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0,
            "size_reduction_percent": (1 - quantized_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0.0
        }

class AdvancedTrainingManager:
    """High-level training manager with all advanced features."""
    
    def __init__(self,
                 model: Union[TorchModule, TorchModuleWrapper],
                 optimizer: TorchOptimizer,
                 mixed_precision_config: Optional[MixedPrecisionConfig] = None,
                 gradient_accumulation_config: Optional[GradientAccumulationConfig] = None,
                 quantization_config: Optional[QuantizationConfig] = None,
                 log_dir: Optional[str] = None):
        """Initialize advanced training manager.
        
        Args:
            model: PyTorch model or wrapped Trustformers model
            optimizer: PyTorch optimizer
            mixed_precision_config: Mixed precision configuration
            gradient_accumulation_config: Gradient accumulation configuration
            quantization_config: Quantization configuration
            log_dir: Directory for logging
        """
        self.model = model
        self.optimizer = optimizer
        self.log_dir = log_dir
        
        # Initialize components
        self.mixed_precision = MixedPrecisionTrainer(
            model, optimizer, mixed_precision_config
        )
        self.advanced_optimizer = AdvancedOptimizer(model, optimizer)
        self.memory_optimizer = MemoryOptimizedTrainer(model)
        
        # NEW: Enhanced components
        self.gradient_accumulator = GradientAccumulator(gradient_accumulation_config)
        self.dynamic_loss_scaler = DynamicLossScaler()
        self.advanced_clipper = AdvancedGradientClipper()
        self.quantization_trainer = QuantizationAwareTrainer(model, quantization_config)
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.optimizer_steps = 0
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
    def train_step(self, 
                   batch: Dict[str, TorchTensor],
                   loss_fn: Callable,
                   batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Perform single training step with all optimizations.
        
        Args:
            batch: Input batch
            loss_fn: Loss function
            batch_size: Batch size for throughput calculation
            
        Returns:
            Step statistics
        """
        start_time = time.time()
        self.model.train()
        step_stats = {}
        
        # Get current memory usage
        current_memory_gb = 0.0
        if torch.cuda.is_available():
            current_memory_gb = torch.cuda.memory_allocated() / 1024**3
        
        # Check if we should accumulate gradients
        should_accumulate = self.gradient_accumulator.should_accumulate(current_memory_gb)
        
        # Memory optimization context
        with self.memory_optimizer.memory_efficient_forward():
            # Mixed precision forward pass
            with self.mixed_precision.autocast_context():
                outputs = self.model(**batch)
                
                # Compute loss
                if 'labels' in batch:
                    loss = loss_fn(outputs, batch['labels'])
                else:
                    loss = outputs.get('loss', outputs)
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulator.config.enabled:
                    loss = loss / self.gradient_accumulator.config.accumulation_steps
        
        # Enhanced backward pass with advanced scaling
        scaled_loss = self.dynamic_loss_scaler.scale(loss)
        scaled_loss.backward()
        
        # Check if we should perform optimizer step
        should_step = self.gradient_accumulator.step()
        step_taken = False
        
        if should_step or not self.gradient_accumulator.config.enabled:
            # Unscale gradients and check for overflow
            no_overflow = self.dynamic_loss_scaler.unscale_gradients(self.optimizer)
            
            if no_overflow:
                # Advanced gradient clipping
                grad_norm = self.advanced_clipper.clip_gradients(self.model.parameters())
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer_steps += 1
                step_taken = True
                
                # Update loss scaler
                self.dynamic_loss_scaler.update(True)
            else:
                # Skip step due to overflow
                self.dynamic_loss_scaler.update(False)
                logger.warning("Skipped optimizer step due to gradient overflow")
            
            # Zero gradients after step
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update gradient analyzer
            self.advanced_optimizer.gradient_analyzer.step()
        
        # Update counters
        self.step_count += 1
        
        # Performance tracking
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        if batch_size:
            throughput = batch_size / step_time
            self.throughput_history.append(throughput)
        
        # Collect comprehensive statistics
        step_stats.update({
            "step": self.step_count,
            "optimizer_step": self.optimizer_steps,
            "step_taken": step_taken,
            "loss": loss.item(),
            "scaled_loss": scaled_loss.item(),
            "step_time": step_time,
            "memory_gb": current_memory_gb,
            "should_accumulate": should_accumulate,
            "gradient_accumulation": {
                "accumulated_steps": self.gradient_accumulator.accumulated_steps,
                "total_accumulation_steps": self.gradient_accumulator.config.accumulation_steps
            },
            "loss_scaling": {
                "current_scale": self.dynamic_loss_scaler.current_scale,
                "overflow_detected": not no_overflow if should_step else False
            },
            "gradient_clipping": self.advanced_clipper.get_clipping_stats(),
            "mixed_precision_stats": self.mixed_precision.get_training_stats(),
            "memory_stats": self.memory_optimizer.get_memory_stats()
        })
        
        # Add throughput if available
        if self.throughput_history:
            step_stats["throughput_samples_per_sec"] = self.throughput_history[-1]
            step_stats["avg_throughput"] = np.mean(list(self.throughput_history)[-10:])
        
        return step_stats
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive training report with all optimization statistics.
        
        Returns:
            Complete training report
        """
        report = {
            "training_progress": {
                "total_steps": self.step_count,
                "optimizer_steps": self.optimizer_steps,
                "current_epoch": self.epoch_count,
                "effective_step_ratio": self.optimizer_steps / max(self.step_count, 1)
            },
            "mixed_precision": self.mixed_precision.get_training_stats(),
            "optimization": self.advanced_optimizer.get_optimization_report(),
            "memory": self.memory_optimizer.get_memory_stats(),
            "gradient_flow": self.advanced_optimizer.gradient_analyzer.get_gradient_flow_report()
        }
        
        # Add new optimization reports
        report["gradient_accumulation"] = {
            "enabled": self.gradient_accumulator.config.enabled,
            "accumulation_steps": self.gradient_accumulator.config.accumulation_steps,
            "current_accumulated": self.gradient_accumulator.accumulated_steps,
            "dynamic_batching": self.gradient_accumulator.config.dynamic_batching,
            "memory_threshold_gb": self.gradient_accumulator.config.memory_threshold_gb
        }
        
        report["loss_scaling"] = {
            "current_scale": self.dynamic_loss_scaler.current_scale,
            "scale_history_length": len(self.dynamic_loss_scaler.scale_history),
            "recent_overflows": sum(list(self.dynamic_loss_scaler.overflow_tracker)[-10:]),
            "adaptive_recommendation": self.dynamic_loss_scaler.get_adaptive_recommendation()
        }
        
        report["gradient_clipping"] = self.advanced_clipper.get_clipping_stats()
        
        report["quantization"] = {
            "enabled": self.quantization_trainer.config.enabled,
            "is_prepared": self.quantization_trainer.is_prepared,
            "backend": self.quantization_trainer.config.backend
        }
        
        # Performance statistics
        if self.step_times:
            recent_times = list(self.step_times)[-50:]
            report["performance"] = {
                "avg_step_time": np.mean(recent_times),
                "min_step_time": np.min(recent_times),
                "max_step_time": np.max(recent_times),
                "step_time_std": np.std(recent_times)
            }
        
        if self.throughput_history:
            recent_throughput = list(self.throughput_history)[-50:]
            report["performance"]["avg_throughput"] = np.mean(recent_throughput)
            report["performance"]["max_throughput"] = np.max(recent_throughput)
            report["performance"]["min_throughput"] = np.min(recent_throughput)
        
        return report
    
    def optimize_for_inference(self) -> TorchModule:
        """Optimize model for inference.
        
        Returns:
            Optimized model
        """
        logger.info("Optimizing model for inference...")
        
        # Switch to eval mode
        self.model.eval()
        
        # Quantization if enabled
        if self.quantization_trainer.config.enabled and self.quantization_trainer.is_prepared:
            quantized_model = self.quantization_trainer.convert_model()
            if quantized_model:
                logger.info("Applied quantization for inference")
                return quantized_model
        
        # Compile model if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(self.model)
                logger.info("Applied torch.compile for inference")
                return compiled_model
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
        
        return self.model
    
    def enable_quantization_training(self):
        """Enable and prepare quantization-aware training."""
        self.quantization_trainer.config.enabled = True
        self.quantization_trainer.prepare_model()
        
        if self.quantization_trainer.is_prepared:
            self.model = self.quantization_trainer.prepared_model
            logger.info("Quantization-aware training enabled")
        else:
            logger.error("Failed to enable quantization-aware training")
    
    def adjust_training_for_memory_pressure(self, memory_pressure_level: float):
        """Adjust training parameters based on memory pressure.
        
        Args:
            memory_pressure_level: 0.0 (no pressure) to 1.0 (high pressure)
        """
        if memory_pressure_level > 0.7:
            # High memory pressure - aggressive optimization
            self.gradient_accumulator.config.accumulation_steps = min(
                self.gradient_accumulator.config.accumulation_steps * 2, 16
            )
            self.gradient_accumulator.config.memory_threshold_gb *= 0.8
            logger.info("Applied aggressive memory optimization")
            
        elif memory_pressure_level > 0.5:
            # Medium memory pressure - moderate optimization
            self.gradient_accumulator.config.accumulation_steps = min(
                self.gradient_accumulator.config.accumulation_steps + 2, 8
            )
            logger.info("Applied moderate memory optimization")
            
        elif memory_pressure_level < 0.3:
            # Low memory pressure - can reduce accumulation
            self.gradient_accumulator.config.accumulation_steps = max(
                self.gradient_accumulator.config.accumulation_steps - 1, 1
            )
            logger.info("Reduced gradient accumulation due to low memory pressure")
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epoch_count": self.epoch_count
        }
        
        torch.save(checkpoint, filepath)
        
        # Save mixed precision state separately
        mp_path = filepath.replace('.pt', '_mixed_precision.pt')
        self.mixed_precision.save_state(mp_path)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def cleanup(self):
        """Clean up resources."""
        self.advanced_optimizer.gradient_analyzer.cleanup()

# Enhanced convenience functions
def create_mixed_precision_trainer(
    model: TorchModule,
    optimizer: TorchOptimizer,
    **kwargs
) -> MixedPrecisionTrainer:
    """Create mixed precision trainer with default config."""
    config = MixedPrecisionConfig(**kwargs)
    return MixedPrecisionTrainer(model, optimizer, config)

def analyze_gradient_flow(model: TorchModule, 
                         log_dir: Optional[str] = None) -> GradientFlowAnalyzer:
    """Create gradient flow analyzer."""
    return GradientFlowAnalyzer(model, log_dir)

def create_advanced_training_manager(
    model: Union[TorchModule, TorchModuleWrapper],
    optimizer: TorchOptimizer,
    mixed_precision_config: Optional[MixedPrecisionConfig] = None,
    gradient_accumulation_config: Optional[GradientAccumulationConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    **kwargs
) -> AdvancedTrainingManager:
    """Create advanced training manager with all optimization features.
    
    Args:
        model: PyTorch model or wrapped Trustformers model
        optimizer: PyTorch optimizer
        mixed_precision_config: Mixed precision configuration
        gradient_accumulation_config: Gradient accumulation configuration
        quantization_config: Quantization configuration
        **kwargs: Additional arguments for AdvancedTrainingManager
        
    Returns:
        Configured AdvancedTrainingManager instance
    """
    return AdvancedTrainingManager(
        model, 
        optimizer, 
        mixed_precision_config, 
        gradient_accumulation_config,
        quantization_config,
        **kwargs
    )

def create_gradient_accumulator(
    accumulation_steps: int = 4,
    dynamic_batching: bool = True,
    memory_threshold_gb: float = 8.0,
    **kwargs
) -> GradientAccumulator:
    """Create gradient accumulator with configuration.
    
    Args:
        accumulation_steps: Number of steps to accumulate
        dynamic_batching: Whether to use dynamic batching
        memory_threshold_gb: Memory threshold for dynamic batching
        **kwargs: Additional configuration options
        
    Returns:
        Configured GradientAccumulator instance
    """
    config = GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        dynamic_batching=dynamic_batching,
        memory_threshold_gb=memory_threshold_gb,
        **kwargs
    )
    return GradientAccumulator(config)

def create_quantization_trainer(
    model: TorchModule,
    backend: str = "fbgemm",
    **kwargs
) -> QuantizationAwareTrainer:
    """Create quantization-aware trainer.
    
    Args:
        model: PyTorch model
        backend: Quantization backend ("fbgemm", "qnnpack")
        **kwargs: Additional configuration options
        
    Returns:
        Configured QuantizationAwareTrainer instance
    """
    config = QuantizationConfig(enabled=True, backend=backend, **kwargs)
    return QuantizationAwareTrainer(model, config)

def optimize_model_for_inference(
    model: TorchModule,
    quantization: bool = False,
    compile_model: bool = True
) -> TorchModule:
    """Optimize model for inference with various strategies.
    
    Args:
        model: PyTorch model to optimize
        quantization: Whether to apply quantization
        compile_model: Whether to compile model (PyTorch 2.0+)
        
    Returns:
        Optimized model
    """
    optimized_model = model
    
    # Switch to eval mode
    optimized_model.eval()
    
    # Apply quantization if requested
    if quantization:
        try:
            from torch.quantization import quantize_dynamic
            optimized_model = quantize_dynamic(optimized_model, {torch.nn.Linear}, dtype=torch.qint8)
            logger.info("Applied dynamic quantization")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    # Compile model if available and requested
    if compile_model:
        try:
            if hasattr(torch, 'compile'):
                optimized_model = torch.compile(optimized_model)
                logger.info("Applied torch.compile")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    return optimized_model

def create_advanced_gradient_clipper(
    max_norm: float = 1.0,
    adaptive: bool = True,
    percentile_clipping: bool = False
) -> AdvancedGradientClipper:
    """Create advanced gradient clipper.
    
    Args:
        max_norm: Maximum gradient norm
        adaptive: Whether to use adaptive clipping
        percentile_clipping: Whether to use percentile-based clipping
        
    Returns:
        Configured AdvancedGradientClipper instance
    """
    return AdvancedGradientClipper(
        max_norm=max_norm,
        adaptive=adaptive,
        percentile_clipping=percentile_clipping
    )

__all__ = [
    # Dataclasses and configs
    'GradientStats',
    'MixedPrecisionConfig',
    'GradientAccumulationConfig',
    'QuantizationConfig',
    
    # Core optimization classes
    'GradientFlowAnalyzer',
    'MixedPrecisionTrainer', 
    'AdvancedOptimizer',
    'MemoryOptimizedTrainer',
    
    # Enhanced optimization classes
    'GradientAccumulator',
    'DynamicLossScaler',
    'AdvancedGradientClipper',
    'QuantizationAwareTrainer',
    
    # High-level managers
    'AdvancedTrainingManager',
    
    # Convenience functions
    'create_mixed_precision_trainer',
    'analyze_gradient_flow',
    'create_advanced_training_manager',
    'create_gradient_accumulator',
    'create_quantization_trainer',
    'optimize_model_for_inference',
    'create_advanced_gradient_clipper'
]