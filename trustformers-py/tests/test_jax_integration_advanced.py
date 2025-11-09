"""
Tests for Advanced JAX Integration System
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import trustformers
from trustformers import Tensor

# Test if JAX is available
try:
    import jax
    import jax.numpy as jnp
    import optax
    _jax_available = True
except ImportError:
    _jax_available = False

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not _jax_available, reason="JAX not available")

if _jax_available:
    from trustformers.jax_integration import (
        JAXDevice, JAXGradientTransforms, JAXCompilation, JAXOptimization,
        JAXModelWrapper, JAXTrainingLoop, JAXCheckpointing, JAXEcosystem,
        jax_device_manager, jax_gradient_transforms, jax_compilation,
        jax_optimization, jax_checkpointing, jax_ecosystem
    )


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXDevice:
    """Test JAX device management"""
    
    def test_device_initialization(self):
        """Test device manager initialization"""
        device_manager = JAXDevice()
        assert device_manager.devices is not None
        assert len(device_manager.devices) > 0
        assert device_manager.current_device is not None
    
    def test_device_setting(self):
        """Test setting device"""
        device_manager = JAXDevice()
        original_device = device_manager.current_device
        
        # Test setting by index
        device_manager.set_device(0)
        assert device_manager.current_device == device_manager.devices[0]
        
        # Test setting by device object
        device_manager.set_device(original_device)
        assert device_manager.current_device == original_device
    
    def test_put_on_device(self):
        """Test putting array on device"""
        device_manager = JAXDevice()
        array = jnp.array([1, 2, 3])
        
        result = device_manager.put_on_device(array)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == array.shape
    
    def test_get_device_info(self):
        """Test getting device information"""
        device_manager = JAXDevice()
        info = device_manager.get_device_info()
        
        assert "current_device" in info
        assert "available_devices" in info
        assert "device_count" in info
        assert "platform" in info
        assert isinstance(info["device_count"], int)
        assert info["device_count"] > 0


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXGradientTransforms:
    """Test JAX gradient transformations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.grad_transforms = JAXGradientTransforms()
    
    def test_value_and_grad(self):
        """Test value and gradient computation"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        value_and_grad_fn = self.grad_transforms.value_and_grad(test_fn)
        x = jnp.array([1.0, 2.0, 3.0])
        
        value, grad = value_and_grad_fn(x)
        expected_value = jnp.sum(x ** 2)
        expected_grad = 2 * x
        
        assert jnp.allclose(value, expected_value)
        assert jnp.allclose(grad, expected_grad)
    
    def test_hessian(self):
        """Test Hessian computation"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        hessian_fn = self.grad_transforms.hessian(test_fn)
        x = jnp.array([1.0, 2.0])
        
        hessian = hessian_fn(x)
        expected_hessian = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        
        assert jnp.allclose(hessian, expected_hessian)
    
    def test_stop_gradient(self):
        """Test stop gradient"""
        x = jnp.array([1.0, 2.0, 3.0])
        stopped = self.grad_transforms.stop_gradient(x)
        
        assert jnp.allclose(stopped, x)
        # Test that gradient is blocked
        def test_fn(x):
            return jnp.sum(self.grad_transforms.stop_gradient(x) ** 2)
        
        grad_fn = jax.grad(test_fn)
        grad = grad_fn(x)
        
        assert jnp.allclose(grad, jnp.zeros_like(x))
    
    def test_clip_gradients(self):
        """Test gradient clipping"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        grad_fn = jax.grad(test_fn)
        clipped_grad_fn = self.grad_transforms.clip_gradients(grad_fn, max_norm=1.0)
        
        x = jnp.array([5.0, 5.0])  # Will produce large gradients
        clipped_grad = clipped_grad_fn(x)
        
        # Check that gradient norm is <= 1.0
        grad_norm = jnp.linalg.norm(clipped_grad)
        assert grad_norm <= 1.0 + 1e-6  # Allow for small numerical errors


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXCompilation:
    """Test JAX compilation utilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.compilation = JAXCompilation()
    
    def test_jit_with_cache(self):
        """Test JIT compilation with cache"""
        def test_fn(x):
            return x ** 2
        
        jit_fn = self.compilation.jit_with_cache(test_fn)
        x = jnp.array([1.0, 2.0, 3.0])
        
        result = jit_fn(x)
        expected = x ** 2
        
        assert jnp.allclose(result, expected)
    
    def test_vmap_batch(self):
        """Test vectorized map"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        vmap_fn = self.compilation.vmap_batch(test_fn)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = vmap_fn(x)
        expected = jnp.array([5.0, 25.0])  # 1^2 + 2^2 = 5, 3^2 + 4^2 = 25
        
        assert jnp.allclose(result, expected)
    
    def test_scan(self):
        """Test scan operation"""
        def scan_fn(carry, x):
            return carry + x, carry + x
        
        init = 0
        xs = jnp.array([1, 2, 3, 4])
        
        final_carry, ys = self.compilation.scan(scan_fn, init, xs)
        
        assert final_carry == 10  # 0 + 1 + 2 + 3 + 4
        assert jnp.allclose(ys, jnp.array([1, 3, 6, 10]))  # cumulative sums
    
    def test_cond(self):
        """Test conditional execution"""
        def true_fn(x):
            return x * 2
        
        def false_fn(x):
            return x * 3
        
        x = jnp.array([1.0, 2.0])
        
        result_true = self.compilation.cond(True, true_fn, false_fn, x)
        result_false = self.compilation.cond(False, true_fn, false_fn, x)
        
        assert jnp.allclose(result_true, x * 2)
        assert jnp.allclose(result_false, x * 3)
    
    def test_checkpoint(self):
        """Test computation checkpointing"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        checkpointed_fn = self.compilation.checkpoint(test_fn)
        x = jnp.array([1.0, 2.0, 3.0])
        
        result = checkpointed_fn(x)
        expected = jnp.sum(x ** 2)
        
        assert jnp.allclose(result, expected)


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXOptimization:
    """Test JAX optimization utilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.optimization = JAXOptimization()
    
    def test_create_optimizer(self):
        """Test optimizer creation"""
        optimizer = self.optimization.create_optimizer('adam', learning_rate=0.001)
        assert optimizer is not None
        
        # Test with different optimizers
        optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        for opt_name in optimizers:
            optimizer = self.optimization.create_optimizer(opt_name, learning_rate=0.01)
            assert optimizer is not None
    
    def test_create_optimizer_unknown(self):
        """Test unknown optimizer raises error"""
        with pytest.raises(ValueError):
            self.optimization.create_optimizer('unknown_optimizer')
    
    def test_create_scheduler(self):
        """Test scheduler creation"""
        scheduler = self.optimization.create_scheduler('constant', value=0.001)
        assert scheduler is not None
        
        # Test learning rate at different steps
        lr_0 = scheduler(0)
        lr_100 = scheduler(100)
        assert lr_0 == lr_100 == 0.001
    
    def test_create_scheduler_unknown(self):
        """Test unknown scheduler raises error"""
        with pytest.raises(ValueError):
            self.optimization.create_scheduler('unknown_scheduler')
    
    def test_apply_gradients(self):
        """Test gradient application"""
        optimizer = self.optimization.create_optimizer('sgd', learning_rate=0.1)
        
        # Initialize parameters and optimizer state
        params = {'w': jnp.array([1.0, 2.0])}
        opt_state = optimizer.init(params)
        
        # Apply gradients
        grads = {'w': jnp.array([0.1, 0.2])}
        new_params, new_opt_state = self.optimization.apply_gradients(
            optimizer, grads, params, opt_state
        )
        
        # Check that parameters were updated
        expected_params = {'w': jnp.array([0.99, 1.98])}  # params - lr * grads
        assert jnp.allclose(new_params['w'], expected_params['w'])
    
    def test_gradient_clipping(self):
        """Test gradient clipping transformation"""
        clipping = self.optimization.gradient_clipping(max_norm=1.0)
        assert clipping is not None
    
    def test_chain_transformations(self):
        """Test chaining transformations"""
        clipping = self.optimization.gradient_clipping(max_norm=1.0)
        optimizer = self.optimization.create_optimizer('adam', learning_rate=0.001)
        
        chained = self.optimization.chain(clipping, optimizer)
        assert chained is not None


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXModelWrapper:
    """Test JAX model wrapper"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock a simple model
        self.mock_model = MagicMock()
        self.mock_model.return_value = MagicMock()
        
        self.wrapper = JAXModelWrapper(self.mock_model)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization"""
        assert self.wrapper.model == self.mock_model
        assert self.wrapper.device_manager is not None
        assert self.wrapper.compiled_forward is None
        assert self.wrapper.compiled_loss is None
    
    def test_create_params(self):
        """Test parameter creation"""
        params = self.wrapper.create_params((10, 20))
        assert isinstance(params, dict)
        assert "model_params" in params
        assert params["model_params"].shape == (10, 20)


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXCheckpointing:
    """Test JAX checkpointing utilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.checkpointing = JAXCheckpointing()
    
    def test_checkpoint_roundtrip(self, tmp_path):
        """Test saving and loading checkpoint"""
        # Create test data
        params = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([0.5])}
        opt_state = {'step': 100}
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        self.checkpointing.save_checkpoint(params, opt_state, str(checkpoint_path))
        
        # Load checkpoint
        loaded = self.checkpointing.load_checkpoint(str(checkpoint_path))
        
        assert 'params' in loaded
        assert 'opt_state' in loaded
        assert 'jax_version' in loaded
        
        # Check parameter values
        assert jnp.allclose(loaded['params']['w'], params['w'])
        assert jnp.allclose(loaded['params']['b'], params['b'])
        assert loaded['opt_state']['step'] == opt_state['step']
    
    def test_create_checkpoint_manager(self):
        """Test checkpoint manager creation"""
        manager = self.checkpointing.create_checkpoint_manager("/tmp/checkpoints", max_to_keep=3)
        
        assert isinstance(manager, dict)
        assert manager['checkpoint_dir'] == "/tmp/checkpoints"
        assert manager['max_to_keep'] == 3


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestJAXEcosystem:
    """Test JAX ecosystem integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.ecosystem = JAXEcosystem()
    
    def test_setup_memory_growth(self):
        """Test memory growth setup"""
        import os
        
        # Clear environment variables
        if 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
            del os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']
        if 'XLA_PYTHON_CLIENT_PREALLOCATE' in os.environ:
            del os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']
        
        # Setup memory growth
        self.ecosystem.setup_memory_growth()
        
        # Check environment variables were set
        assert os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION') == '0.8'
        assert os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') == 'false'
    
    def test_get_memory_info(self):
        """Test getting memory information"""
        memory_info = self.ecosystem.get_memory_info()
        
        assert isinstance(memory_info, dict)
        assert len(memory_info) > 0
        
        # Check that device info is included
        for key, value in memory_info.items():
            assert key.startswith('device_')
            assert 'device_type' in value
            assert 'platform' in value
    
    def test_profile_computation(self):
        """Test computation profiling"""
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        x = jnp.array([1.0, 2.0, 3.0])
        profile_result = self.ecosystem.profile_computation(test_fn, x)
        
        assert 'execution_time' in profile_result
        assert 'result' in profile_result
        assert isinstance(profile_result['execution_time'], float)
        assert profile_result['execution_time'] >= 0
        assert jnp.allclose(profile_result['result'], jnp.sum(x ** 2))


@pytest.mark.skipif(not _jax_available, reason="JAX not available")
class TestGlobalInstances:
    """Test global instance availability"""
    
    def test_global_instances_exist(self):
        """Test that global instances exist when JAX is available"""
        assert jax_device_manager is not None
        assert jax_gradient_transforms is not None
        assert jax_compilation is not None
        assert jax_checkpointing is not None
        assert jax_ecosystem is not None
        
        # jax_optimization might be None if optax is not available
        if optax is not None:
            assert jax_optimization is not None
    
    def test_global_instances_functionality(self):
        """Test basic functionality of global instances"""
        # Test device manager
        info = jax_device_manager.get_device_info()
        assert isinstance(info, dict)
        assert 'device_count' in info
        
        # Test gradient transforms
        def test_fn(x):
            return jnp.sum(x ** 2)
        
        grad_fn = jax_gradient_transforms.value_and_grad(test_fn)
        x = jnp.array([1.0, 2.0])
        value, grad = grad_fn(x)
        
        assert jnp.allclose(value, 5.0)  # 1^2 + 2^2 = 5
        assert jnp.allclose(grad, jnp.array([2.0, 4.0]))  # 2*x


if __name__ == "__main__":
    pytest.main([__file__])