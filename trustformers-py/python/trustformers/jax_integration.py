"""
Advanced JAX Integration for TrustformeRS

Provides comprehensive JAX support including advanced gradient transformations,
JIT compilation patterns, device management, and ecosystem integration.
"""

import functools
import warnings
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, NamedTuple
import numpy as np
from .utils import logging

try:
    import jax
    import jax.numpy as jnp
    from jax import Array as JAXArray
    from jax import tree_util
    from jax.experimental import checkify
    from jax.experimental import host_callback
    import optax
    _jax_available = True
except ImportError:
    _jax_available = False
    JAXArray = None
    optax = None

logger = logging.get_logger(__name__)


class JAXDevice:
    """JAX device management utility"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
        self._devices = jax.devices()
        self._current_device = None
    
    @property
    def devices(self) -> List[Any]:
        """Get all available devices"""
        return self._devices
    
    @property
    def current_device(self) -> Any:
        """Get current device"""
        return self._current_device or self._devices[0]
    
    def set_device(self, device: Union[str, int, Any]):
        """Set current device"""
        if isinstance(device, str):
            device_type = device.upper()
            available_devices = [d for d in self._devices if d.platform.upper() == device_type]
            if not available_devices:
                raise ValueError(f"No {device_type} devices available")
            self._current_device = available_devices[0]
        elif isinstance(device, int):
            if device >= len(self._devices):
                raise ValueError(f"Device index {device} out of range")
            self._current_device = self._devices[device]
        else:
            self._current_device = device
    
    def put_on_device(self, array: JAXArray, device: Optional[Any] = None) -> JAXArray:
        """Put array on specific device"""
        target_device = device or self.current_device
        return jax.device_put(array, target_device)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "current_device": str(self.current_device),
            "available_devices": [str(d) for d in self._devices],
            "device_count": len(self._devices),
            "platform": self.current_device.platform,
        }


class JAXGradientTransforms:
    """Advanced gradient transformation utilities"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
    
    @staticmethod
    def value_and_grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0,
                      has_aux: bool = False) -> Callable:
        """Compute both value and gradient"""
        return jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)
    
    @staticmethod
    def hessian(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
        """Compute Hessian matrix"""
        return jax.hessian(fun, argnums=argnums)
    
    @staticmethod
    def jacobian(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
        """Compute Jacobian matrix"""
        return jax.jacobian(fun, argnums=argnums)
    
    @staticmethod
    def hvp(fun: Callable, primals: Any, tangents: Any) -> Tuple[Any, Any]:
        """Hessian-vector product"""
        return jax.jvp(jax.grad(fun), (primals,), (tangents,))
    
    @staticmethod
    def vjp(fun: Callable, primals: Any) -> Tuple[Any, Callable]:
        """Vector-Jacobian product"""
        return jax.vjp(fun, primals)
    
    @staticmethod
    def custom_gradient(fun: Callable, grad_fun: Callable) -> Callable:
        """Create function with custom gradient"""
        @jax.custom_vjp
        def custom_fun(x):
            return fun(x)
        
        def custom_fun_fwd(x):
            return custom_fun(x), x
        
        def custom_fun_bwd(residual, g):
            x = residual
            return (grad_fun(x, g),)
        
        custom_fun.defvjp(custom_fun_fwd, custom_fun_bwd)
        return custom_fun
    
    @staticmethod
    def stop_gradient(x: JAXArray) -> JAXArray:
        """Stop gradient computation"""
        return jax.lax.stop_gradient(x)
    
    @staticmethod
    def clip_gradients(grad_fn: Callable, max_norm: float = 1.0) -> Callable:
        """Clip gradients by global norm"""
        def clipped_grad_fn(*args, **kwargs):
            grads = grad_fn(*args, **kwargs)
            if isinstance(grads, dict):
                # Clip gradients in a dictionary
                total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads)))
                scale = max_norm / jnp.maximum(total_norm, max_norm)
                return jax.tree_map(lambda g: g * scale, grads)
            else:
                # Clip single gradient
                norm = jnp.linalg.norm(grads)
                scale = max_norm / jnp.maximum(norm, max_norm)
                return grads * scale
        
        return clipped_grad_fn
    
    @staticmethod
    def accumulate_gradients(grad_fns: List[Callable], weights: Optional[List[float]] = None) -> Callable:
        """Accumulate gradients from multiple functions"""
        if weights is None:
            weights = [1.0] * len(grad_fns)
        
        def accumulated_grad_fn(*args, **kwargs):
            grads = [w * grad_fn(*args, **kwargs) for w, grad_fn in zip(weights, grad_fns)]
            return jax.tree_map(lambda *g: sum(g), *grads)
        
        return accumulated_grad_fn


class JAXCompilation:
    """Advanced JIT compilation utilities"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
    
    @staticmethod
    def jit_with_cache(fun: Callable, static_argnums: Union[int, Tuple[int, ...]] = (),
                      device: Optional[Any] = None, donate_argnums: Union[int, Tuple[int, ...]] = ()) -> Callable:
        """JIT compile with caching and optimization"""
        return jax.jit(fun, static_argnums=static_argnums, device=device, donate_argnums=donate_argnums)
    
    @staticmethod
    def pmap(fun: Callable, axis_name: str = 'batch', devices: Optional[List[Any]] = None) -> Callable:
        """Parallel map across devices"""
        return jax.pmap(fun, axis_name=axis_name, devices=devices)
    
    @staticmethod
    def vmap_batch(fun: Callable, in_axes: Union[int, Tuple] = 0, out_axes: Union[int, Tuple] = 0) -> Callable:
        """Vectorized map with batch processing"""
        return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
    
    @staticmethod
    def scan(fun: Callable, init: Any, xs: Any, length: Optional[int] = None) -> Tuple[Any, Any]:
        """Scan operation for sequential processing"""
        return jax.lax.scan(fun, init, xs, length=length)
    
    @staticmethod
    def cond(pred: bool, true_fun: Callable, false_fun: Callable, *operands) -> Any:
        """Conditional execution"""
        return jax.lax.cond(pred, true_fun, false_fun, *operands)
    
    @staticmethod
    def while_loop(cond_fun: Callable, body_fun: Callable, init_val: Any) -> Any:
        """While loop execution"""
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    @staticmethod
    def checkpoint(fun: Callable, prevent_cse: bool = True) -> Callable:
        """Checkpoint computation for memory efficiency"""
        return jax.checkpoint(fun, prevent_cse=prevent_cse)
    
    @staticmethod
    def make_jaxpr(fun: Callable, *args, **kwargs) -> Any:
        """Create JAX expression for debugging"""
        return jax.make_jaxpr(fun)(*args, **kwargs)
    
    @staticmethod
    def profiler_trace(name: str):
        """Context manager for profiling"""
        return jax.profiler.trace(name)


class JAXOptimization:
    """JAX optimization utilities with Optax integration"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
        if optax is None:
            raise ImportError("Optax is not available. Install with: pip install optax")
    
    @staticmethod
    def create_optimizer(optimizer_name: str, learning_rate: float = 1e-3, **kwargs) -> Any:
        """Create Optax optimizer"""
        optimizers = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
            'rmsprop': optax.rmsprop,
            'adagrad': optax.adagrad,
            'adadelta': optax.adadelta,
            'adamax': optax.adamax,
            'lamb': optax.lamb,
        }
        
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        optimizer_fn = optimizers[optimizer_name]
        return optimizer_fn(learning_rate, **kwargs)
    
    @staticmethod
    def create_scheduler(scheduler_name: str, **kwargs) -> Any:
        """Create learning rate scheduler"""
        schedulers = {
            'constant': optax.constant_schedule,
            'linear': optax.linear_schedule,
            'cosine': optax.cosine_decay_schedule,
            'exponential': optax.exponential_decay,
            'polynomial': optax.polynomial_schedule,
            'warmup_cosine': optax.warmup_cosine_decay_schedule,
        }
        
        if scheduler_name not in schedulers:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        scheduler_fn = schedulers[scheduler_name]
        return scheduler_fn(**kwargs)
    
    @staticmethod
    def apply_gradients(optimizer: Any, grads: Any, params: Any, opt_state: Any) -> Tuple[Any, Any]:
        """Apply gradients using optimizer"""
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
    
    @staticmethod
    def gradient_clipping(max_norm: float = 1.0) -> Any:
        """Create gradient clipping transformation"""
        return optax.clip_by_global_norm(max_norm)
    
    @staticmethod
    def chain(*transformations) -> Any:
        """Chain multiple transformations"""
        return optax.chain(*transformations)


class JAXModelWrapper:
    """Wrapper for TrustformeRS models with JAX integration"""
    
    def __init__(self, model: Any, device_manager: Optional[JAXDevice] = None):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
        
        self.model = model
        self.device_manager = device_manager or JAXDevice()
        self.compiled_forward = None
        self.compiled_loss = None
        self.compiled_train_step = None
    
    def compile_forward(self, static_argnums: Tuple[int, ...] = ()) -> Callable:
        """Compile forward pass with JIT"""
        def forward_fn(params, inputs):
            # Convert JAX arrays to TrustformeRS tensors
            from .jax_utils import jax_to_tensor, tensor_to_jax
            
            # Convert inputs
            if isinstance(inputs, dict):
                tensor_inputs = {k: jax_to_tensor(v) for k, v in inputs.items()}
            else:
                tensor_inputs = jax_to_tensor(inputs)
            
            # Run model
            outputs = self.model(tensor_inputs)
            
            # Convert outputs back to JAX
            if hasattr(outputs, 'last_hidden_state'):
                # Model output object
                return tensor_to_jax(outputs.last_hidden_state)
            else:
                return tensor_to_jax(outputs)
        
        self.compiled_forward = jax.jit(forward_fn, static_argnums=static_argnums)
        return self.compiled_forward
    
    def compile_loss(self, loss_fn: Callable) -> Callable:
        """Compile loss function with JIT"""
        def jax_loss_fn(params, inputs, targets):
            outputs = self.compiled_forward(params, inputs)
            return loss_fn(outputs, targets)
        
        self.compiled_loss = jax.jit(jax_loss_fn)
        return self.compiled_loss
    
    def compile_train_step(self, optimizer: Any, loss_fn: Callable) -> Callable:
        """Compile full training step"""
        if self.compiled_loss is None:
            self.compile_loss(loss_fn)
        
        def train_step(params, opt_state, inputs, targets):
            # Compute gradients
            grads = jax.grad(self.compiled_loss)(params, inputs, targets)
            
            # Apply gradients
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Compute loss for logging
            loss = self.compiled_loss(params, inputs, targets)
            
            return new_params, new_opt_state, loss
        
        self.compiled_train_step = jax.jit(train_step)
        return self.compiled_train_step
    
    def create_params(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Create JAX parameters from model"""
        # This would extract parameters from the TrustformeRS model
        # For now, return a placeholder
        return {"model_params": jnp.ones(input_shape)}


class JAXTrainingLoop:
    """JAX-based training loop"""
    
    def __init__(self, model_wrapper: JAXModelWrapper, optimizer: Any, 
                 loss_fn: Callable, metrics: Optional[List[Callable]] = None):
        self.model_wrapper = model_wrapper
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        
        # Compile training components
        self.model_wrapper.compile_train_step(optimizer, loss_fn)
        
        # Initialize optimizer state
        self.opt_state = None
        self.params = None
    
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initialize parameters and optimizer state"""
        self.params = self.model_wrapper.create_params(input_shape)
        self.opt_state = self.optimizer.init(self.params)
    
    def train_step(self, inputs: JAXArray, targets: JAXArray) -> Dict[str, float]:
        """Single training step"""
        if self.params is None:
            raise ValueError("Must call initialize() first")
        
        # Perform training step
        self.params, self.opt_state, loss = self.model_wrapper.compiled_train_step(
            self.params, self.opt_state, inputs, targets
        )
        
        # Compute metrics
        metrics_values = {"loss": float(loss)}
        for metric in self.metrics:
            outputs = self.model_wrapper.compiled_forward(self.params, inputs)
            metric_value = metric(outputs, targets)
            metrics_values[metric.__name__] = float(metric_value)
        
        return metrics_values
    
    def evaluate(self, inputs: JAXArray, targets: JAXArray) -> Dict[str, float]:
        """Evaluate model"""
        if self.params is None:
            raise ValueError("Must call initialize() first")
        
        # Compute loss
        loss = self.model_wrapper.compiled_loss(self.params, inputs, targets)
        
        # Compute metrics
        metrics_values = {"loss": float(loss)}
        for metric in self.metrics:
            outputs = self.model_wrapper.compiled_forward(self.params, inputs)
            metric_value = metric(outputs, targets)
            metrics_values[metric.__name__] = float(metric_value)
        
        return metrics_values
    
    def predict(self, inputs: JAXArray) -> JAXArray:
        """Make predictions"""
        if self.params is None:
            raise ValueError("Must call initialize() first")
        
        return self.model_wrapper.compiled_forward(self.params, inputs)


class JAXCheckpointing:
    """JAX model checkpointing utilities"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
    
    @staticmethod
    def save_checkpoint(params: Any, opt_state: Any, filepath: str):
        """Save checkpoint"""
        import pickle
        
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'jax_version': jax.__version__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    @staticmethod
    def load_checkpoint(filepath: str) -> Dict[str, Any]:
        """Load checkpoint"""
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    @staticmethod
    def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 5) -> Any:
        """Create checkpoint manager"""
        # This would integrate with more sophisticated checkpoint management
        # For now, return a simple implementation
        return {
            'checkpoint_dir': checkpoint_dir,
            'max_to_keep': max_to_keep
        }


# Enhanced JAX ecosystem integration
class JAXEcosystem:
    """JAX ecosystem integration utilities"""
    
    def __init__(self):
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
    
    @staticmethod
    def setup_for_tpu():
        """Setup JAX for TPU"""
        try:
            import jax.tools.colab_tpu
            jax.tools.colab_tpu.setup_tpu()
            logger.info("TPU setup completed")
        except ImportError:
            logger.warning("TPU setup not available")
    
    @staticmethod
    def setup_memory_growth():
        """Setup memory growth for GPU"""
        import os
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information"""
        devices = jax.devices()
        memory_info = {}
        
        for i, device in enumerate(devices):
            try:
                # This is device-specific and may not work on all platforms
                memory_info[f'device_{i}'] = {
                    'device_type': device.device_kind,
                    'platform': device.platform,
                }
            except:
                memory_info[f'device_{i}'] = {
                    'device_type': 'unknown',
                    'platform': device.platform,
                }
        
        return memory_info
    
    @staticmethod
    def profile_computation(fun: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile JAX computation"""
        import time
        
        # Warm up
        fun(*args, **kwargs)
        
        # Time execution
        start_time = time.time()
        result = fun(*args, **kwargs)
        end_time = time.time()
        
        return {
            'execution_time': end_time - start_time,
            'result': result
        }


# Create global instances
jax_device_manager = JAXDevice() if _jax_available else None
jax_gradient_transforms = JAXGradientTransforms() if _jax_available else None
jax_compilation = JAXCompilation() if _jax_available else None
jax_optimization = JAXOptimization() if _jax_available and optax is not None else None
jax_checkpointing = JAXCheckpointing() if _jax_available else None
jax_ecosystem = JAXEcosystem() if _jax_available else None


# Export all classes and functions
__all__ = [
    'JAXDevice',
    'JAXGradientTransforms', 
    'JAXCompilation',
    'JAXOptimization',
    'JAXModelWrapper',
    'JAXTrainingLoop',
    'JAXCheckpointing',
    'JAXEcosystem',
    'jax_device_manager',
    'jax_gradient_transforms',
    'jax_compilation',
    'jax_optimization',
    'jax_checkpointing',
    'jax_ecosystem',
]