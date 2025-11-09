#!/usr/bin/env python3
"""
Quick start script for TrustformeRS Python

This script demonstrates basic usage and verifies the installation.
"""

import sys
import numpy as np


def check_import():
    """Check if TrustformeRS can be imported."""
    print("1. Checking TrustformeRS import...")
    try:
        import trustformers
        print(f"   ✓ TrustformeRS version: {trustformers.__version__}")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("\n   Please install TrustformeRS:")
        print("   pip install trustformers")
        return False


def test_tensor_operations():
    """Test basic tensor operations."""
    print("\n2. Testing tensor operations...")
    try:
        from trustformers import Tensor
        
        # Create tensors
        a = Tensor(np.array([1, 2, 3], dtype=np.float32))
        b = Tensor(np.array([4, 5, 6], dtype=np.float32))
        
        # Operations
        c = a + b
        print(f"   ✓ Tensor addition: [1,2,3] + [4,5,6] = {c.numpy()}")
        
        # Matrix multiplication
        mat_a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        mat_b = Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
        result = mat_a.matmul(mat_b)
        print(f"   ✓ Matrix multiplication successful, shape: {result.shape}")
        
        return True
    except Exception as e:
        print(f"   ✗ Tensor operations failed: {e}")
        return False


def test_numpy_integration():
    """Test NumPy integration."""
    print("\n3. Testing NumPy integration...")
    try:
        from trustformers import Tensor, numpy_to_tensor, tensor_to_numpy
        
        # NumPy to Tensor
        np_array = np.random.randn(2, 3).astype(np.float32)
        tensor = numpy_to_tensor(np_array)
        
        # Tensor to NumPy
        np_back = tensor_to_numpy(tensor)
        
        if np.allclose(np_array, np_back):
            print("   ✓ NumPy conversion working correctly")
            return True
        else:
            print("   ✗ NumPy conversion mismatch")
            return False
    except Exception as e:
        print(f"   ✗ NumPy integration failed: {e}")
        return False


def test_model_loading():
    """Test model loading."""
    print("\n4. Testing model loading...")
    try:
        from trustformers import AutoModel, BertConfig
        
        # Create a small config for testing
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        
        # This would normally load from hub
        # model = AutoModel.from_pretrained("bert-base-uncased")
        print("   ✓ Model imports working (actual loading requires model files)")
        return True
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        return False


def test_pipeline():
    """Test pipeline creation."""
    print("\n5. Testing pipeline API...")
    try:
        from trustformers import pipeline
        
        # Create pipeline
        classifier = pipeline("text-classification")
        print("   ✓ Pipeline creation successful")
        return True
    except Exception as e:
        print(f"   ✗ Pipeline creation failed: {e}")
        return False


def test_torch_integration():
    """Test PyTorch integration if available."""
    print("\n6. Testing PyTorch integration...")
    try:
        import torch
        from trustformers import torch_to_tensor, tensor_to_torch, Tensor
        
        # PyTorch to TrustformeRS
        torch_tensor = torch.randn(2, 3)
        trust_tensor = torch_to_tensor(torch_tensor)
        
        # TrustformeRS to PyTorch
        torch_back = tensor_to_torch(trust_tensor)
        
        if torch.allclose(torch_tensor, torch_back):
            print("   ✓ PyTorch integration working correctly")
            return True
        else:
            print("   ✗ PyTorch conversion mismatch")
            return False
    except ImportError:
        print("   - PyTorch not installed (optional)")
        return True
    except Exception as e:
        print(f"   ✗ PyTorch integration failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print("\n7. System Information:")
    print(f"   - Python version: {sys.version}")
    print(f"   - NumPy version: {np.__version__}")
    
    try:
        import torch
        print(f"   - PyTorch version: {torch.__version__}")
    except ImportError:
        print("   - PyTorch: not installed")
    
    try:
        import platform
        print(f"   - Platform: {platform.platform()}")
        print(f"   - Architecture: {platform.machine()}")
    except:
        pass


def main():
    """Run all tests."""
    print("TrustformeRS Python - Quick Start")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    if not check_import():
        sys.exit(1)
    
    all_passed &= test_tensor_operations()
    all_passed &= test_numpy_integration()
    all_passed &= test_model_loading()
    all_passed &= test_pipeline()
    all_passed &= test_torch_integration()
    
    print_system_info()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! TrustformeRS is ready to use.")
        print("\nNext steps:")
        print("1. Check out the examples/ directory for more demos")
        print("2. Read the documentation at https://trustformers.readthedocs.io")
        print("3. Try loading a real model with AutoModel.from_pretrained()")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()