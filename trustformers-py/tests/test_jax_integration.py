"""
Tests for Advanced JAX Integration System
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import trustformers
from trustformers import Tensor


def jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_to_tensor():
    """Test JAX array to TrustformeRS tensor conversion."""
    import jax.numpy as jnp
    from trustformers import jax_to_tensor
    
    # Create JAX array
    jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to tensor
    tensor = jax_to_tensor(jax_array)
    
    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 2]
    np.testing.assert_array_almost_equal(tensor.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_tensor_to_jax():
    """Test TrustformeRS tensor to JAX array conversion."""
    import jax.numpy as jnp
    from trustformers import tensor_to_jax
    
    # Create tensor
    tensor = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    
    # Convert to JAX array
    jax_array = tensor_to_jax(tensor)
    
    assert jax_array.shape == (2, 2)
    np.testing.assert_array_almost_equal(np.array(jax_array), np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_batch_jax_conversion():
    """Test batch conversion between JAX arrays and tensors."""
    import jax.numpy as jnp
    from trustformers import batch_jax_to_tensor, batch_tensor_to_jax
    
    # Test list conversion
    jax_arrays = [
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
        jnp.array([5.0, 6.0])
    ]
    
    tensors = batch_jax_to_tensor(jax_arrays)
    assert len(tensors) == 3
    assert all(isinstance(t, Tensor) for t in tensors)
    
    # Convert back
    jax_arrays_back = batch_tensor_to_jax(tensors)
    assert len(jax_arrays_back) == 3
    
    # Test dict conversion
    jax_dict = {
        "input_ids": jnp.array([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": jnp.array([[1, 1, 1], [1, 1, 0]])
    }
    
    tensor_dict = batch_jax_to_tensor(jax_dict)
    assert set(tensor_dict.keys()) == set(jax_dict.keys())
    assert all(isinstance(t, Tensor) for t in tensor_dict.values())
    
    # Convert back
    jax_dict_back = batch_tensor_to_jax(tensor_dict)
    assert set(jax_dict_back.keys()) == set(jax_dict.keys())


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_ensure_jax_array():
    """Test ensuring inputs are JAX arrays."""
    import jax.numpy as jnp
    from trustformers import ensure_jax_array
    
    # Test with different input types
    
    # NumPy array
    np_array = np.array([1, 2, 3], dtype=np.float32)
    jax_result = ensure_jax_array(np_array)
    assert jax_result.shape == (3,)
    
    # List
    list_input = [1, 2, 3]
    jax_result = ensure_jax_array(list_input)
    assert jax_result.shape == (3,)
    
    # JAX array (should return as-is)
    jax_array = jnp.array([1, 2, 3])
    jax_result = ensure_jax_array(jax_array)
    assert jax_result is jax_array  # Should be the same object
    
    # TrustformeRS tensor
    tensor = Tensor([1, 2, 3])
    jax_result = ensure_jax_array(tensor)
    assert jax_result.shape == (3,)


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_tensor_wrapper():
    """Test JAXTensorWrapper functionality."""
    import jax.numpy as jnp
    from trustformers import JAXTensorWrapper
    
    wrapper = JAXTensorWrapper(prefer_jax=True)
    
    # Test JAX array to tensor
    jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    tensor = wrapper.jax_to_tensor(jax_array)
    assert isinstance(tensor, Tensor)
    
    # Test tensor to JAX array
    jax_back = wrapper.tensor_to_jax(tensor)
    assert jax_back.shape == (2, 2)
    
    # Test auto conversion with JAX preference
    converted = wrapper.auto_convert([1, 2, 3])
    assert converted.shape == (3,)


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_dataloader():
    """Test JAX dataloader creation."""
    import jax.numpy as jnp
    from trustformers import create_jax_dataloader
    
    # Test with dict of arrays
    data = {
        "input_ids": jnp.arange(100).reshape(10, 10),
        "attention_mask": jnp.ones((10, 10))
    }
    
    dataloader = create_jax_dataloader(data, batch_size=3, shuffle=False)
    batches = list(dataloader())
    
    assert len(batches) == 4  # 10 samples, batch_size=3, so 4 batches (last one smaller)
    assert all("input_ids" in batch and "attention_mask" in batch for batch in batches)
    assert batches[0]["input_ids"].shape[0] == 3  # First batch has 3 samples
    assert batches[-1]["input_ids"].shape[0] == 1  # Last batch has 1 sample


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_utility_functions():
    """Test JAX utility functions."""
    import jax
    import jax.numpy as jnp
    from trustformers import jax_grad_fn, jax_vmap_fn, jax_jit_fn
    
    # Test gradient function
    def simple_loss(x):
        return jnp.sum(x ** 2)
    
    grad_fn = jax_grad_fn(simple_loss)
    x = jnp.array([1.0, 2.0, 3.0])
    gradients = grad_fn(x)
    expected = 2 * x  # Gradient of x^2 is 2x
    np.testing.assert_array_almost_equal(gradients, expected)
    
    # Test vmap function
    def square(x):
        return x ** 2
    
    vmap_square = jax_vmap_fn(square)
    inputs = jnp.array([1.0, 2.0, 3.0])
    outputs = vmap_square(inputs)
    expected = jnp.array([1.0, 4.0, 9.0])
    np.testing.assert_array_almost_equal(outputs, expected)
    
    # Test JIT function
    jit_square = jax_jit_fn(square)
    result = jit_square(5.0)
    assert abs(result - 25.0) < 1e-6


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_pipeline_creation():
    """Test JAX pipeline creation."""
    from trustformers import create_jax_pipeline
    
    # Mock model (in real usage this would be a TrustformeRS model)
    mock_model = "test_model"
    
    pipeline_config = create_jax_pipeline(mock_model, task="text-classification")
    
    assert pipeline_config["model"] == mock_model
    assert pipeline_config["task"] == "text-classification"
    assert pipeline_config["framework"] == "jax"
    assert "convert_inputs" in pipeline_config
    assert "convert_outputs" in pipeline_config


@pytest.mark.skipif(not jax_available(), reason="JAX not installed")
def test_jax_config_setup():
    """Test JAX configuration setup."""
    from trustformers import setup_jax_config
    
    # This should run without errors
    setup_jax_config()


def test_jax_availability_check():
    """Test JAX availability checking."""
    from trustformers.jax_utils import jax_available
    
    # This should return True if JAX is installed, False otherwise
    available = jax_available()
    assert isinstance(available, bool)


def test_jax_import_errors():
    """Test proper error handling when JAX is not available."""
    # This test runs regardless of JAX availability
    # If JAX is not available, functions should raise ImportError
    
    if not jax_available():
        from trustformers.jax_utils import jax_to_tensor
        
        with pytest.raises(ImportError, match="JAX is not available"):
            # This should raise ImportError since JAX is not available
            import numpy as np
            fake_array = np.array([1, 2, 3])
            jax_to_tensor(fake_array)


if __name__ == "__main__":
    pytest.main([__file__])