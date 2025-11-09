"""
Basic tests for TrustformeRS Python package
"""

import pytest
import numpy as np


def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def test_import():
    """Test that the package can be imported."""
    import trustformers
    assert hasattr(trustformers, '__version__')


def test_tensor_creation():
    """Test basic tensor creation."""
    from trustformers import Tensor
    
    # Create from numpy arrays
    np_array1 = np.array([1, 2, 3], dtype=np.float32)
    tensor1 = Tensor(np_array1)
    assert tensor1.shape == [3]
    
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor2 = Tensor(np_array)
    assert tensor2.shape == [2, 2]


def test_tensor_creation_from_lists():
    """Test tensor creation from Python lists."""
    from trustformers import Tensor
    
    # Create from 1D Python list
    tensor1 = Tensor([1, 2, 3])
    assert tensor1.shape == [3]
    np.testing.assert_array_almost_equal(tensor1.numpy(), np.array([1, 2, 3], dtype=np.float32))
    
    # Create from 2D Python list
    tensor2 = Tensor([[1, 2], [3, 4]])
    assert tensor2.shape == [2, 2]
    np.testing.assert_array_almost_equal(tensor2.numpy(), np.array([[1, 2], [3, 4]], dtype=np.float32))
    
    # Create from 3D Python list
    tensor3 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor3.shape == [2, 2, 2]
    expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    np.testing.assert_array_almost_equal(tensor3.numpy(), expected)
    
    # Create from scalar
    tensor4 = Tensor(5.0)
    assert tensor4.shape == []
    
    # Create from mixed types (int/float)
    tensor5 = Tensor([1, 2.5, 3])
    assert tensor5.shape == [3]
    np.testing.assert_array_almost_equal(tensor5.numpy(), np.array([1.0, 2.5, 3.0], dtype=np.float32))


def test_tensor_operations():
    """Test basic tensor operations."""
    from trustformers import Tensor
    
    a = Tensor(np.array([1, 2, 3], dtype=np.float32))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32))
    
    # Addition
    c = a + b
    expected = np.array([5, 7, 9], dtype=np.float32)
    np.testing.assert_array_almost_equal(c.numpy(), expected)


def test_numpy_conversion():
    """Test NumPy conversion utilities."""
    from trustformers import numpy_to_tensor, tensor_to_numpy
    
    np_array = np.random.randn(3, 4).astype(np.float32)
    tensor = numpy_to_tensor(np_array)
    np_array_back = tensor_to_numpy(tensor)
    
    np.testing.assert_array_almost_equal(np_array, np_array_back)


def test_auto_tokenizer():
    """Test AutoTokenizer."""
    from trustformers import AutoTokenizer
    
    # AutoTokenizer can be instantiated
    assert AutoTokenizer is not None
    
    # Test tokenizer creation with different model types
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    assert bert_tokenizer is not None
    
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert gpt2_tokenizer is not None
    
    # Test that different model names create appropriate tokenizer types
    # BERT models should create WordPiece tokenizers
    bert_tokenizer2 = AutoTokenizer.from_pretrained("bert-large-uncased")
    assert bert_tokenizer2 is not None
    
    # GPT models should create BPE tokenizers  
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt-3.5-turbo")
    assert gpt_tokenizer is not None
    
    # T5 models should create SentencePiece tokenizers (fallback to BPE)
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    assert t5_tokenizer is not None


def test_pipeline():
    """Test pipeline creation."""
    from trustformers import pipeline
    
    # Pipeline function exists
    assert pipeline is not None
    
    # Test supported pipeline tasks
    # Text generation pipeline (uses GPT2 by default)
    text_gen_pipeline = pipeline("text-generation")
    assert text_gen_pipeline is not None
    
    # Text classification pipeline (uses BERT by default)
    text_class_pipeline = pipeline("text-classification")
    assert text_class_pipeline is not None
    
    # Sentiment analysis pipeline (alias for text-classification)
    sentiment_pipeline = pipeline("sentiment-analysis")
    assert sentiment_pipeline is not None
    
    # Test with explicit model name (using supported task)
    explicit_pipeline = pipeline("text-generation", model="gpt2")
    assert explicit_pipeline is not None
    
    # Test error handling for unsupported tasks
    try:
        pipeline("unsupported-task")
        assert False, "Should have raised ValueError for unsupported task"
    except ValueError as e:
        assert "Unknown task" in str(e)
        assert "Supported tasks" in str(e)


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_torch_conversion():
    """Test PyTorch conversion if available."""
    import torch
    from trustformers import torch_to_tensor, tensor_to_torch, Tensor
    
    torch_tensor = torch.randn(2, 3)
    trust_tensor = torch_to_tensor(torch_tensor)
    torch_back = tensor_to_torch(trust_tensor)
    
    assert torch.allclose(torch_tensor, torch_back)


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_torch_advanced_conversion():
    """Test advanced PyTorch conversion features."""
    import torch
    from trustformers import (
        torch_to_tensor, tensor_to_torch, 
        batch_torch_to_tensor, batch_tensor_to_torch,
        ensure_torch_tensor, Tensor
    )
    
    # Test different dtypes
    torch_int = torch.tensor([1, 2, 3], dtype=torch.int32)
    trust_tensor = torch_to_tensor(torch_int)
    torch_back = tensor_to_torch(trust_tensor)
    assert torch_back.dtype == torch.float32  # Should convert to float32
    
    # Test batch conversion
    torch_tensors = [torch.randn(2, 3) for _ in range(3)]
    trust_tensors = batch_torch_to_tensor(torch_tensors)
    torch_tensors_back = batch_tensor_to_torch(trust_tensors)
    
    assert len(torch_tensors_back) == 3
    for orig, back in zip(torch_tensors, torch_tensors_back):
        assert torch.allclose(orig, back)
    
    # Test dict batch conversion
    torch_dict = {
        "input_ids": torch.randint(0, 100, (2, 10)),
        "attention_mask": torch.ones(2, 10)
    }
    trust_dict = batch_torch_to_tensor(torch_dict)
    torch_dict_back = batch_tensor_to_torch(trust_dict)
    
    assert set(torch_dict.keys()) == set(torch_dict_back.keys())
    for key in torch_dict:
        assert torch.allclose(torch_dict[key].float(), torch_dict_back[key])
    
    # Test ensure_torch_tensor utility
    # From numpy
    import numpy as np
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    torch_result = ensure_torch_tensor(np_array)
    assert isinstance(torch_result, torch.Tensor)
    assert torch_result.shape == (2, 2)
    
    # From TrustformeRS Tensor
    trust_tensor = Tensor(np_array)
    torch_result = ensure_torch_tensor(trust_tensor)
    assert isinstance(torch_result, torch.Tensor)
    
    # From list
    torch_result = ensure_torch_tensor([1, 2, 3])
    assert isinstance(torch_result, torch.Tensor)
    assert torch_result.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__])