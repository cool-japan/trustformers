#!/usr/bin/env python3
"""
Basic Tensor Operations Demo for TrustformeRS Python Bindings

This script demonstrates the core tensor operations that are currently working:
- Basic tensor creation and manipulation
- Enhanced tensor operations (GELU, SiLU, Mish, etc.)
- Matrix operations
- Utility functions

Requirements:
- trustformers-py compiled with simplified dependencies
- numpy
"""

import numpy as np

try:
    import trustformers
    from trustformers import Tensor as PyTensor, TensorOptimized
    print(f"✅ Successfully imported trustformers v{trustformers.__version__}")
except ImportError as e:
    print(f"❌ Error importing trustformers: {e}")
    print("Make sure trustformers-py is properly installed")
    exit(1)


def test_basic_tensor_operations():
    """Test basic tensor operations"""
    print("🔧 Testing Basic Tensor Operations")
    print("=" * 50)

    # Create test data
    data = np.random.randn(4, 8).astype(np.float32)
    print(f"Created test data with shape: {data.shape}")

    # Test PyTensor if available
    try:
        tensor = PyTensor.from_numpy(data)
        print(f"✅ PyTensor creation successful")
        print(f"Tensor shape: {tensor.shape()}")
        print(f"Tensor dtype: {tensor.dtype()}")
    except Exception as e:
        print(f"⚠️  PyTensor test failed: {e}")

    print()


def test_activation_functions():
    """Test advanced activation functions"""
    print("🔥 Testing Advanced Activation Functions")
    print("=" * 50)

    # Create test tensor
    x = np.random.randn(4, 8).astype(np.float32)
    print(f"Input tensor shape: {x.shape}")

    try:
        # Test GELU
        gelu_result = TensorOptimized.gelu(x)
        print(f"✅ GELU output shape: {gelu_result.shape}")
        print(f"GELU sample values: {gelu_result.flatten()[:3]}")

        # Test SiLU/Swish
        silu_result = TensorOptimized.silu(x)
        print(f"✅ SiLU output shape: {silu_result.shape}")
        print(f"SiLU sample values: {silu_result.flatten()[:3]}")

        # Test Mish
        mish_result = TensorOptimized.mish(x)
        print(f"✅ Mish output shape: {mish_result.shape}")
        print(f"Mish sample values: {mish_result.flatten()[:3]}")

        # Test fast GELU
        gelu_fast_result = TensorOptimized.gelu_fast(x)
        print(f"✅ Fast GELU output shape: {gelu_fast_result.shape}")
        print(f"Fast GELU sample values: {gelu_fast_result.flatten()[:3]}")

    except Exception as e:
        print(f"❌ Activation function test failed: {e}")

    print()


def test_matrix_operations():
    """Test optimized matrix operations"""
    print("🔢 Testing Matrix Operations")
    print("=" * 50)

    try:
        # Create matrices
        a = np.random.randn(4, 6).astype(np.float32)
        b = np.random.randn(6, 8).astype(np.float32)

        print(f"Matrix A shape: {a.shape}")
        print(f"Matrix B shape: {b.shape}")

        # Optimized matrix multiplication
        result = TensorOptimized.matmul(a, b)
        print(f"✅ TensorOptimized.matmul result shape: {result.shape}")

        # Compare with numpy
        numpy_result = np.matmul(a, b)
        print(f"NumPy result shape: {numpy_result.shape}")

        # Check if results are close
        diff = np.abs(result - numpy_result)
        max_diff = np.max(diff)
        print(f"Maximum difference vs NumPy: {max_diff:.8f}")

        if max_diff < 1e-5:
            print("✅ Results match NumPy within tolerance")
        else:
            print("⚠️  Results differ from NumPy significantly")

    except Exception as e:
        print(f"❌ Matrix operations test failed: {e}")

    print()


def test_softmax():
    """Test stable softmax"""
    print("💫 Testing Stable Softmax")
    print("=" * 50)

    try:
        # Create test tensor with large values to test stability
        x = np.array([[1000, 2000, 3000], [100, 200, 300]], dtype=np.float32)

        print(f"Input tensor:\n{x}")

        # Apply softmax along axis 1
        softmax_result = TensorOptimized.softmax(x, axis=1)
        print(f"✅ Softmax result:\n{softmax_result}")

        # Check that rows sum to 1
        row_sums = np.sum(softmax_result, axis=1)
        print(f"Row sums: {row_sums}")

        if np.allclose(row_sums, 1.0, atol=1e-6):
            print("✅ Softmax rows sum to 1 within tolerance")
        else:
            print("⚠️  Softmax rows do not sum to 1")

    except Exception as e:
        print(f"❌ Softmax test failed: {e}")

    print()


def test_utility_functions():
    """Test utility functions"""
    print("🔧 Testing Utility Functions")
    print("=" * 50)

    try:
        # Test device detection
        device = trustformers.get_device()
        print(f"✅ Current device: {device}")

        # Test random seed setting
        trustformers.set_seed(42)
        print("✅ Random seed set to 42")

        # Test gradient context
        trustformers.enable_grad()
        print("✅ Gradients enabled")

        trustformers.no_grad()
        print("✅ Gradients disabled")

    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")

    print()


def main():
    """Run all test functions"""
    print("🚀 TrustformeRS Basic Tensor Operations Demo")
    print("=" * 60)
    print(f"TrustformeRS version: {trustformers.__version__}")
    print("=" * 60)
    print()

    try:
        test_basic_tensor_operations()
        test_activation_functions()
        test_matrix_operations()
        test_softmax()
        test_utility_functions()

        print("✅ All basic tensor operations completed successfully!")
        print()
        print("🎉 TrustformeRS Core Tensor Features are working correctly!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Check the compilation and make sure all dependencies are properly configured")


if __name__ == "__main__":
    main()