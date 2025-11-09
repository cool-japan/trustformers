"""
Custom Extensions System for TrustformeRS
=========================================

This module provides extensibility features including:
- Python callback system for custom hooks
- Custom layer support
- Hook registration for model lifecycle events
- Event system for component communication
- Plugin architecture for modular functionality

Classes:
--------
- CallbackManager: Centralized callback management
- HookRegistry: Model lifecycle hook registration
- EventBus: Component communication event system
- PluginManager: Plugin loading and management
- CustomLayer: Base class for Python-defined layers
- ExtensionConfig: Configuration for extension behavior
"""

import asyncio
import inspect
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Callable, Union, Type, 
    Protocol, TypeVar, Generic, Set, Tuple, AsyncGenerator
)
import threading
import time
import logging
from functools import wraps, partial
import importlib
import sys
from pathlib import Path

# Type definitions
T = TypeVar('T')
CallbackFunc = Callable[..., Any]
AsyncCallbackFunc = Callable[..., Any]  # Can be sync or async
HookFunc = Callable[..., Any]
EventHandler = Callable[..., Any]


@dataclass
class CallbackConfig:
    """Configuration for callback execution behavior"""
    timeout: Optional[float] = None
    max_retries: int = 0
    retry_delay: float = 0.1
    fail_fast: bool = True
    async_execution: bool = False
    priority: int = 0  # Higher priority executes first


@dataclass
class EventConfig:
    """Configuration for event handling"""
    async_delivery: bool = False
    max_handlers: Optional[int] = None
    timeout: Optional[float] = None
    buffer_size: int = 1000


class CallbackError(Exception):
    """Exception raised during callback execution"""
    pass


class HookError(Exception):
    """Exception raised during hook execution"""
    pass


class EventError(Exception):
    """Exception raised during event handling"""
    pass


class PluginError(Exception):
    """Exception raised during plugin operations"""
    pass


class CallbackResult:
    """Result of callback execution with metadata"""
    
    def __init__(self, success: bool, result: Any = None, error: Exception = None, 
                 execution_time: float = 0.0, callback_name: str = ""):
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.callback_name = callback_name
        self.timestamp = time.time()
    
    def __repr__(self):
        status = "Success" if self.success else "Failed"
        return f"CallbackResult({status}, {self.callback_name}, {self.execution_time:.3f}s)"


class CallbackManager:
    """Centralized callback management system"""
    
    def __init__(self, default_config: Optional[CallbackConfig] = None):
        self.default_config = default_config or CallbackConfig()
        self._callbacks: Dict[str, List[Tuple[CallbackFunc, CallbackConfig]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="callback")
        self._active_callbacks: Set[str] = set()
        self._callback_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_calls': 0, 'success_calls': 0, 'failed_calls': 0,
            'total_time': 0.0, 'avg_time': 0.0
        })
        self.logger = logging.getLogger(__name__ + '.CallbackManager')
    
    def register(self, event_type: str, callback: CallbackFunc, 
                 config: Optional[CallbackConfig] = None) -> str:
        """Register a callback for a specific event type"""
        config = config or self.default_config
        callback_id = f"{event_type}_{id(callback)}_{len(self._callbacks[event_type])}"
        
        with self._lock:
            self._callbacks[event_type].append((callback, config))
            # Sort by priority (higher first)
            self._callbacks[event_type].sort(key=lambda x: x[1].priority, reverse=True)
        
        self.logger.debug(f"Registered callback {callback_id} for event {event_type}")
        return callback_id
    
    def unregister(self, event_type: str, callback: CallbackFunc) -> bool:
        """Unregister a callback"""
        with self._lock:
            callbacks = self._callbacks[event_type]
            for i, (cb, _) in enumerate(callbacks):
                if cb is callback:
                    del callbacks[i]
                    self.logger.debug(f"Unregistered callback for event {event_type}")
                    return True
        return False
    
    def trigger(self, event_type: str, *args, **kwargs) -> List[CallbackResult]:
        """Trigger all callbacks for an event type"""
        if event_type in self._active_callbacks:
            self.logger.warning(f"Recursive callback detected for {event_type}")
            return []
        
        with self._lock:
            callbacks = self._callbacks[event_type].copy()
        
        if not callbacks:
            return []
        
        results = []
        self._active_callbacks.add(event_type)
        
        try:
            for callback, config in callbacks:
                result = self._execute_callback(callback, config, event_type, *args, **kwargs)
                results.append(result)
                
                # Update stats
                stats = self._callback_stats[event_type]
                stats['total_calls'] += 1
                if result.success:
                    stats['success_calls'] += 1
                else:
                    stats['failed_calls'] += 1
                stats['total_time'] += result.execution_time
                stats['avg_time'] = stats['total_time'] / stats['total_calls']
                
                if not result.success and config.fail_fast:
                    self.logger.error(f"Callback failed with fail_fast enabled: {result.error}")
                    break
        
        finally:
            self._active_callbacks.discard(event_type)
        
        return results
    
    def _execute_callback(self, callback: CallbackFunc, config: CallbackConfig, 
                         event_type: str, *args, **kwargs) -> CallbackResult:
        """Execute a single callback with error handling and retries"""
        callback_name = getattr(callback, '__name__', str(callback))
        
        for attempt in range(config.max_retries + 1):
            start_time = time.time()
            
            try:
                if config.timeout and not config.async_execution:
                    # Handle timeout for sync callbacks using thread pool
                    future = self._executor.submit(callback, *args, **kwargs)
                    result = future.result(timeout=config.timeout)
                elif config.async_execution and not asyncio.iscoroutinefunction(callback):
                    # Run sync callback in thread pool
                    future = self._executor.submit(callback, *args, **kwargs)
                    result = future.result(timeout=config.timeout)
                else:
                    # Direct execution
                    result = callback(*args, **kwargs)
                
                execution_time = time.time() - start_time
                return CallbackResult(True, result, None, execution_time, callback_name)
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                if attempt < config.max_retries:
                    self.logger.warning(f"Callback {callback_name} failed, retrying... ({attempt + 1}/{config.max_retries})")
                    time.sleep(config.retry_delay)
                    continue
                
                self.logger.error(f"Callback {callback_name} failed after {attempt + 1} attempts: {e}")
                return CallbackResult(False, None, e, execution_time, callback_name)
        
        # Should not reach here
        return CallbackResult(False, None, CallbackError("Unknown error"), 0.0, callback_name)
    
    async def trigger_async(self, event_type: str, *args, **kwargs) -> List[CallbackResult]:
        """Async version of trigger for async callbacks"""
        with self._lock:
            callbacks = self._callbacks[event_type].copy()
        
        if not callbacks:
            return []
        
        results = []
        
        for callback, config in callbacks:
            if asyncio.iscoroutinefunction(callback):
                start_time = time.time()
                try:
                    if config.timeout:
                        result = await asyncio.wait_for(callback(*args, **kwargs), config.timeout)
                    else:
                        result = await callback(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    callback_name = getattr(callback, '__name__', str(callback))
                    results.append(CallbackResult(True, result, None, execution_time, callback_name))
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    callback_name = getattr(callback, '__name__', str(callback))
                    results.append(CallbackResult(False, None, e, execution_time, callback_name))
            else:
                # Sync callback, execute normally
                result = self._execute_callback(callback, config, event_type, *args, **kwargs)
                results.append(result)
        
        return results
    
    def get_stats(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Get callback execution statistics"""
        if event_type:
            return self._callback_stats.get(event_type, {})
        return dict(self._callback_stats)
    
    def clear_stats(self):
        """Clear all statistics"""
        self._callback_stats.clear()
    
    def list_callbacks(self, event_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List registered callbacks"""
        if event_type:
            callbacks = self._callbacks.get(event_type, [])
            return {event_type: [getattr(cb[0], '__name__', str(cb[0])) for cb in callbacks]}
        
        result = {}
        for et, callbacks in self._callbacks.items():
            result[et] = [getattr(cb[0], '__name__', str(cb[0])) for cb in callbacks]
        return result
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


class HookRegistry:
    """Model lifecycle hook registration system"""
    
    # Standard hook points
    HOOK_POINTS = {
        'before_forward', 'after_forward',
        'before_backward', 'after_backward',
        'before_training_step', 'after_training_step',
        'before_validation_step', 'after_validation_step',
        'before_epoch', 'after_epoch',
        'before_save', 'after_save',
        'before_load', 'after_load',
        'on_error', 'on_warning'
    }
    
    def __init__(self):
        self.callback_manager = CallbackManager()
        self._model_hooks: Dict[str, Dict[str, List[HookFunc]]] = defaultdict(lambda: defaultdict(list))
        self._global_hooks: Dict[str, List[HookFunc]] = defaultdict(list)
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + '.HookRegistry')
    
    def register_hook(self, hook_point: str, hook_func: HookFunc, 
                     model_id: Optional[str] = None, priority: int = 0) -> str:
        """Register a hook for a specific hook point"""
        if hook_point not in self.HOOK_POINTS:
            raise HookError(f"Unknown hook point: {hook_point}. Valid points: {self.HOOK_POINTS}")
        
        config = CallbackConfig(priority=priority)
        
        if model_id:
            hook_id = f"{model_id}_{hook_point}_{id(hook_func)}"
            with self._lock:
                self._model_hooks[model_id][hook_point].append(hook_func)
        else:
            hook_id = f"global_{hook_point}_{id(hook_func)}"
            with self._lock:
                self._global_hooks[hook_point].append(hook_func)
        
        # Also register with callback manager
        event_type = f"hook_{hook_point}" if not model_id else f"hook_{model_id}_{hook_point}"
        self.callback_manager.register(event_type, hook_func, config)
        
        self.logger.debug(f"Registered hook {hook_id}")
        return hook_id
    
    def unregister_hook(self, hook_point: str, hook_func: HookFunc, 
                       model_id: Optional[str] = None) -> bool:
        """Unregister a hook"""
        with self._lock:
            if model_id:
                hooks = self._model_hooks[model_id][hook_point]
            else:
                hooks = self._global_hooks[hook_point]
            
            try:
                hooks.remove(hook_func)
                # Also unregister from callback manager
                event_type = f"hook_{hook_point}" if not model_id else f"hook_{model_id}_{hook_point}"
                self.callback_manager.unregister(event_type, hook_func)
                self.logger.debug(f"Unregistered hook for {hook_point}")
                return True
            except ValueError:
                return False
    
    def trigger_hooks(self, hook_point: str, model_id: Optional[str] = None, 
                     *args, **kwargs) -> List[CallbackResult]:
        """Trigger hooks for a specific hook point"""
        results = []
        
        # Trigger global hooks first
        if hook_point in self._global_hooks:
            global_results = self.callback_manager.trigger(f"hook_{hook_point}", *args, **kwargs)
            results.extend(global_results)
        
        # Trigger model-specific hooks
        if model_id and model_id in self._model_hooks:
            model_results = self.callback_manager.trigger(f"hook_{model_id}_{hook_point}", *args, **kwargs)
            results.extend(model_results)
        
        return results
    
    def list_hooks(self, model_id: Optional[str] = None) -> Dict[str, List[str]]:
        """List registered hooks"""
        if model_id:
            hooks = self._model_hooks.get(model_id, {})
            return {hp: [getattr(hf, '__name__', str(hf)) for hf in hooks.get(hp, [])] 
                   for hp in self.HOOK_POINTS}
        
        result = {}
        for hp in self.HOOK_POINTS:
            global_hooks = [getattr(hf, '__name__', str(hf)) for hf in self._global_hooks.get(hp, [])]
            result[hp] = global_hooks
        
        return result


class Event:
    """Event object containing event data and metadata"""
    
    def __init__(self, event_type: str, data: Any = None, source: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.id = f"{event_type}_{id(self)}_{int(self.timestamp * 1000)}"
        self._cancelled = False
    
    def cancel(self):
        """Cancel the event (prevents further processing)"""
        self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def __repr__(self):
        return f"Event({self.event_type}, {self.source}, {self.timestamp})"


class EventBus:
    """Component communication event system"""
    
    def __init__(self, config: Optional[EventConfig] = None):
        self.config = config or EventConfig()
        self._handlers: Dict[str, List[Tuple[EventHandler, int]]] = defaultdict(list)  # (handler, priority)
        self._wildcards: List[Tuple[str, EventHandler, int]] = []  # (pattern, handler, priority)
        self._event_buffer: List[Event] = []
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="event")
        self._stats = defaultdict(int)
        self.logger = logging.getLogger(__name__ + '.EventBus')
    
    def subscribe(self, event_type: str, handler: EventHandler, priority: int = 0) -> str:
        """Subscribe to events of a specific type"""
        if '*' in event_type:
            # Wildcard subscription
            handler_id = f"wildcard_{len(self._wildcards)}_{id(handler)}"
            with self._lock:
                self._wildcards.append((event_type, handler, priority))
                self._wildcards.sort(key=lambda x: x[2], reverse=True)  # Sort by priority
        else:
            # Exact match subscription
            handler_id = f"{event_type}_{len(self._handlers[event_type])}_{id(handler)}"
            with self._lock:
                self._handlers[event_type].append((handler, priority))
                self._handlers[event_type].sort(key=lambda x: x[1], reverse=True)  # Sort by priority
        
        self.logger.debug(f"Subscribed handler {handler_id} to {event_type}")
        return handler_id
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Unsubscribe from events"""
        with self._lock:
            if '*' in event_type:
                for i, (pattern, h, _) in enumerate(self._wildcards):
                    if pattern == event_type and h is handler:
                        del self._wildcards[i]
                        return True
            else:
                handlers = self._handlers[event_type]
                for i, (h, _) in enumerate(handlers):
                    if h is handler:
                        del handlers[i]
                        return True
        return False
    
    def publish(self, event: Event) -> List[Any]:
        """Publish an event to all subscribers"""
        if event.is_cancelled:
            return []
        
        # Add to buffer
        with self._lock:
            if len(self._event_buffer) >= self.config.buffer_size:
                self._event_buffer.pop(0)  # Remove oldest
            self._event_buffer.append(event)
        
        results = []
        handlers_to_call = []
        
        # Collect handlers
        with self._lock:
            # Exact match handlers
            exact_handlers = self._handlers.get(event.event_type, [])
            handlers_to_call.extend([(h, p) for h, p in exact_handlers])
            
            # Wildcard handlers
            import fnmatch
            for pattern, handler, priority in self._wildcards:
                if fnmatch.fnmatch(event.event_type, pattern):
                    handlers_to_call.append((handler, priority))
            
            # Sort by priority
            handlers_to_call.sort(key=lambda x: x[1], reverse=True)
            
            # Apply max_handlers limit
            if self.config.max_handlers:
                handlers_to_call = handlers_to_call[:self.config.max_handlers]
        
        # Execute handlers
        for handler, _ in handlers_to_call:
            if event.is_cancelled:
                break
            
            try:
                if self.config.async_delivery and not asyncio.iscoroutinefunction(handler):
                    # Execute in thread pool
                    future = self._executor.submit(handler, event)
                    if self.config.timeout:
                        result = future.result(timeout=self.config.timeout)
                    else:
                        result = future.result()
                else:
                    # Direct execution
                    result = handler(event)
                
                results.append(result)
                self._stats['successful_deliveries'] += 1
                
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")
                self._stats['failed_deliveries'] += 1
        
        self._stats['events_published'] += 1
        return results
    
    async def publish_async(self, event: Event) -> List[Any]:
        """Async version of publish"""
        if event.is_cancelled:
            return []
        
        handlers_to_call = []
        
        # Collect handlers (same logic as sync version)
        with self._lock:
            exact_handlers = self._handlers.get(event.event_type, [])
            handlers_to_call.extend([(h, p) for h, p in exact_handlers])
            
            import fnmatch
            for pattern, handler, priority in self._wildcards:
                if fnmatch.fnmatch(event.event_type, pattern):
                    handlers_to_call.append((handler, priority))
            
            handlers_to_call.sort(key=lambda x: x[1], reverse=True)
            
            if self.config.max_handlers:
                handlers_to_call = handlers_to_call[:self.config.max_handlers]
        
        results = []
        
        for handler, _ in handlers_to_call:
            if event.is_cancelled:
                break
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    if self.config.timeout:
                        result = await asyncio.wait_for(handler(event), self.config.timeout)
                    else:
                        result = await handler(event)
                else:
                    result = handler(event)
                
                results.append(result)
                self._stats['successful_deliveries'] += 1
                
            except Exception as e:
                self.logger.error(f"Async event handler failed: {e}")
                self._stats['failed_deliveries'] += 1
        
        self._stats['events_published'] += 1
        return results
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """Get recent events from buffer"""
        with self._lock:
            return self._event_buffer[-count:]
    
    def get_stats(self) -> Dict[str, int]:
        """Get event bus statistics"""
        return dict(self._stats)
    
    def clear_stats(self):
        """Clear statistics"""
        self._stats.clear()
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


# Global instances for convenience
callback_manager = CallbackManager()
hook_registry = HookRegistry()
event_bus = EventBus()


# Decorator utilities
def callback(event_type: str, config: Optional[CallbackConfig] = None):
    """Decorator to register a function as a callback"""
    def decorator(func):
        callback_manager.register(event_type, func, config)
        return func
    return decorator


def hook(hook_point: str, model_id: Optional[str] = None, priority: int = 0):
    """Decorator to register a function as a hook"""
    def decorator(func):
        hook_registry.register_hook(hook_point, func, model_id, priority)
        return func
    return decorator


def event_handler(event_type: str, priority: int = 0):
    """Decorator to register a function as an event handler"""
    def decorator(func):
        event_bus.subscribe(event_type, func, priority)
        return func
    return decorator


# Context managers
@contextmanager
def callback_context(event_type: str, callback: CallbackFunc, config: Optional[CallbackConfig] = None):
    """Context manager for temporary callback registration"""
    callback_id = callback_manager.register(event_type, callback, config)
    try:
        yield callback_id
    finally:
        callback_manager.unregister(event_type, callback)


@contextmanager
def hook_context(hook_point: str, hook_func: HookFunc, model_id: Optional[str] = None, priority: int = 0):
    """Context manager for temporary hook registration"""
    hook_id = hook_registry.register_hook(hook_point, hook_func, model_id, priority)
    try:
        yield hook_id
    finally:
        hook_registry.unregister_hook(hook_point, hook_func, model_id)


# Custom Layer Support
# ====================

class LayerProtocol(Protocol):
    """Protocol defining the interface for custom layers"""
    
    def forward(self, inputs: Any, *args, **kwargs) -> Any:
        """Forward pass computation"""
        ...
    
    def backward(self, grad_output: Any) -> Any:
        """Backward pass computation (optional)"""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get layer parameters"""
        ...
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set layer parameters"""
        ...


class CustomLayer(ABC):
    """Base class for custom Python-defined layers"""
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.config = kwargs
        self._parameters: Dict[str, Any] = {}
        self._training = True
        self._device = 'cpu'
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__ + f'.CustomLayer.{self.name}')
    
    @abstractmethod
    def forward(self, inputs: Any, *args, **kwargs) -> Any:
        """Forward pass computation - must be implemented by subclasses"""
        pass
    
    def backward(self, grad_output: Any) -> Any:
        """Backward pass computation - optional override"""
        return grad_output
    
    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        """Make layer callable"""
        # Trigger before_forward hooks
        self._trigger_hooks('before_forward', inputs, *args, **kwargs)
        
        try:
            # Perform forward pass
            output = self.forward(inputs, *args, **kwargs)
            
            # Trigger after_forward hooks
            self._trigger_hooks('after_forward', inputs, output, *args, **kwargs)
            
            return output
        
        except Exception as e:
            self._trigger_hooks('on_error', e, inputs, *args, **kwargs)
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get layer parameters"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set layer parameters"""
        self._parameters.update(params)
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self._training = mode
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    @property
    def training(self) -> bool:
        return self._training
    
    def to(self, device: str):
        """Move layer to device"""
        self._device = device
        return self
    
    @property
    def device(self) -> str:
        return self._device
    
    def register_hook(self, hook_type: str, hook_func: Callable):
        """Register a hook for this layer"""
        self._hooks[hook_type].append(hook_func)
    
    def _trigger_hooks(self, hook_type: str, *args, **kwargs):
        """Trigger hooks of a specific type"""
        for hook_func in self._hooks[hook_type]:
            try:
                hook_func(self, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook {hook_func} failed: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration"""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'config': self.config.copy(),
            'parameters': self.get_parameters(),
            'training': self.training,
            'device': self.device
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, training={self.training}, device={self.device})"


class LinearLayer(CustomLayer):
    """Example custom linear layer implementation"""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias
        
        # Initialize parameters (simplified - in real implementation would use proper initialization)
        import numpy as np
        self._parameters['weight'] = np.random.randn(input_size, output_size) * 0.1
        if bias:
            self._parameters['bias'] = np.zeros(output_size)
    
    def forward(self, inputs: Any, *args, **kwargs) -> Any:
        """Forward pass: inputs @ weight + bias"""
        import numpy as np
        
        # Convert inputs to numpy if needed
        if hasattr(inputs, 'numpy'):
            x = inputs.numpy()
        elif hasattr(inputs, '__array__'):
            x = np.array(inputs)
        else:
            x = np.array(inputs)
        
        # Linear transformation
        output = x @ self._parameters['weight']
        if self.use_bias:
            output = output + self._parameters['bias']
        
        # Convert back to original type if possible
        if hasattr(inputs, 'from_numpy'):
            return inputs.from_numpy(output)
        return output


class ActivationLayer(CustomLayer):
    """Example custom activation layer"""
    
    def __init__(self, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
    
    def forward(self, inputs: Any, *args, **kwargs) -> Any:
        """Apply activation function"""
        import numpy as np
        
        # Convert inputs to numpy if needed
        if hasattr(inputs, 'numpy'):
            x = inputs.numpy()
        elif hasattr(inputs, '__array__'):
            x = np.array(inputs)
        else:
            x = np.array(inputs)
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, x)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            output = np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Convert back to original type if possible
        if hasattr(inputs, 'from_numpy'):
            return inputs.from_numpy(output)
        return output


class CustomLayerRegistry:
    """Registry for custom layer types"""
    
    def __init__(self):
        self._layers: Dict[str, Type[CustomLayer]] = {}
        self._instances: Dict[str, CustomLayer] = {}
        self._lock = threading.RLock()
        
        # Register built-in layers
        self.register('linear', LinearLayer)
        self.register('activation', ActivationLayer)
    
    def register(self, layer_type: str, layer_class: Type[CustomLayer]):
        """Register a custom layer type"""
        with self._lock:
            self._layers[layer_type] = layer_class
    
    def create(self, layer_type: str, layer_id: Optional[str] = None, **kwargs) -> CustomLayer:
        """Create a custom layer instance"""
        if layer_type not in self._layers:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        layer_class = self._layers[layer_type]
        layer = layer_class(**kwargs)
        
        if layer_id:
            with self._lock:
                self._instances[layer_id] = layer
        
        return layer
    
    def get_instance(self, layer_id: str) -> Optional[CustomLayer]:
        """Get a layer instance by ID"""
        return self._instances.get(layer_id)
    
    def list_types(self) -> List[str]:
        """List available layer types"""
        return list(self._layers.keys())
    
    def list_instances(self) -> List[str]:
        """List active layer instances"""
        return list(self._instances.keys())


# Plugin Architecture
# ===================

class PluginInterface(ABC):
    """Base interface for plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown the plugin"""
        pass
    
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies"""
        return []
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        return {}


@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    dependencies: List[str] = field(default_factory=list)
    min_version: str = ""
    max_version: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)


class PluginManager:
    """Plugin loading and management system"""
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path.cwd() / "plugins"
        self._plugins: Dict[str, PluginInterface] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._load_order: List[str] = []
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + '.PluginManager')
        
        # Ensure plugin directory exists
        self.plugin_dir.mkdir(exist_ok=True)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directory"""
        plugins = []
        
        for plugin_path in self.plugin_dir.glob("*.py"):
            if plugin_path.name.startswith("__"):
                continue
            
            try:
                spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
                if spec and spec.loader:
                    plugins.append(plugin_path.stem)
            except Exception as e:
                self.logger.warning(f"Failed to discover plugin {plugin_path}: {e}")
        
        return plugins
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a single plugin"""
        if plugin_name in self._plugins:
            self.logger.warning(f"Plugin {plugin_name} already loaded")
            return True
        
        try:
            # Import the plugin module
            plugin_path = self.plugin_dir / f"{plugin_name}.py"
            if not plugin_path.exists():
                raise PluginError(f"Plugin file not found: {plugin_path}")
            
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if not spec or not spec.loader:
                raise PluginError(f"Failed to create spec for plugin: {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (inspect.isclass(attr) and 
                    issubclass(attr, PluginInterface) and 
                    attr != PluginInterface):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                raise PluginError(f"No plugin class found in {plugin_name}")
            
            # Create plugin instance
            plugin = plugin_class()
            
            # Get metadata
            metadata = PluginMetadata(
                name=plugin.name,
                version=plugin.version,
                dependencies=plugin.get_dependencies(),
                config_schema=plugin.get_config_schema()
            )
            
            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self._plugins:
                    self.logger.warning(f"Plugin {plugin_name} dependency {dep} not loaded")
            
            # Initialize plugin
            plugin.initialize(config or {})
            
            # Store plugin
            with self._lock:
                self._plugins[plugin_name] = plugin
                self._metadata[plugin_name] = metadata
                self._dependencies[plugin_name] = set(metadata.dependencies)
                self._load_order.append(plugin_name)
            
            self.logger.info(f"Loaded plugin: {plugin_name} v{plugin.version}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self._plugins:
            return False
        
        try:
            # Check for dependents
            dependents = self._get_dependents(plugin_name)
            if dependents:
                self.logger.warning(f"Plugin {plugin_name} has dependents: {dependents}")
                return False
            
            # Shutdown plugin
            plugin = self._plugins[plugin_name]
            plugin.shutdown()
            
            # Remove from registry
            with self._lock:
                del self._plugins[plugin_name]
                del self._metadata[plugin_name]
                del self._dependencies[plugin_name]
                self._load_order.remove(plugin_name)
            
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin instance"""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List loaded plugins"""
        return list(self._plugins.keys())
    
    def get_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata"""
        return self._metadata.get(plugin_name)
    
    def _get_dependents(self, plugin_name: str) -> List[str]:
        """Get plugins that depend on the given plugin"""
        dependents = []
        for name, deps in self._dependencies.items():
            if plugin_name in deps:
                dependents.append(name)
        return dependents
    
    def load_all_plugins(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, bool]:
        """Load all discovered plugins"""
        plugins = self.discover_plugins()
        results = {}
        
        for plugin_name in plugins:
            plugin_config = config.get(plugin_name, {}) if config else {}
            results[plugin_name] = self.load_plugin(plugin_name, plugin_config)
        
        return results
    
    def shutdown_all(self):
        """Shutdown all plugins"""
        # Shutdown in reverse load order
        for plugin_name in reversed(self._load_order.copy()):
            self.unload_plugin(plugin_name)


# Global instances
custom_layer_registry = CustomLayerRegistry()
plugin_manager = PluginManager()


# Utility functions
def create_custom_layer(layer_type: str, **kwargs) -> CustomLayer:
    """Convenience function to create a custom layer"""
    return custom_layer_registry.create(layer_type, **kwargs)


def register_layer_type(layer_type: str, layer_class: Type[CustomLayer]):
    """Convenience function to register a layer type"""
    custom_layer_registry.register(layer_type, layer_class)


# Configuration class for extensions
@dataclass
class ExtensionConfig:
    """Configuration for extension system behavior"""
    enable_callbacks: bool = True
    enable_hooks: bool = True
    enable_events: bool = True
    enable_plugins: bool = True
    enable_custom_layers: bool = True
    
    callback_config: CallbackConfig = field(default_factory=CallbackConfig)
    event_config: EventConfig = field(default_factory=EventConfig)
    
    plugin_dir: Optional[str] = None
    auto_load_plugins: bool = False
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Setup logging after initialization"""
        logging.basicConfig(level=getattr(logging, self.log_level.upper()))


# Configuration utility
def configure_extensions(config: ExtensionConfig):
    """Configure the extension system"""
    # Update global instances with new configurations
    global callback_manager, hook_registry, event_bus, plugin_manager
    
    if config.enable_callbacks:
        callback_manager = CallbackManager(config.callback_config)
    
    if config.enable_hooks:
        hook_registry = HookRegistry()
    
    if config.enable_events:
        event_bus = EventBus(config.event_config)
    
    if config.enable_plugins and config.plugin_dir:
        # Update the plugin_dir of the existing instance instead of creating new one
        plugin_manager.plugin_dir = Path(config.plugin_dir)
        # Ensure the directory exists
        plugin_manager.plugin_dir.mkdir(exist_ok=True)
        
        if config.auto_load_plugins:
            plugin_manager.load_all_plugins(config.plugin_configs)