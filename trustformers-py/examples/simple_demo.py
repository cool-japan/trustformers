#!/usr/bin/env python3
"""
Simple TrustformeRS Python Demo

This demo showcases the core tensor operations available in TrustformeRS Python bindings.
"""

import numpy as np

try:
    import trustformers
    from trustformers import TensorOptimized
    print("✅ Successfully imported TrustformeRS Python bindings")
except ImportError as e:
    print(f"❌ Failed to import TrustformeRS: {e}")
    exit(1)

def demo_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def main():
    """Run the comprehensive demo"""
    print("🚀 TrustformeRS Python Bindings - Simple Demo")
    print('='*60)

    # Basic tensor operations
    demo_section("Basic Tensor Operations")

    # Create input tensors
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    print("Input tensors:")
    print(f"A = {a}")
    print(f"B = {b}")

    # Matrix multiplication
    result = TensorOptimized.matmul(a, b)
    print(f"\nMatrix multiplication A @ B:")
    print(result)

    # Softmax
    result = TensorOptimized.softmax(a, axis=1)
    print(f"\nSoftmax(A, axis=1):")
    print(result)

    # Layer normalization
    print("\nLayer normalization:")
    result = TensorOptimized.layer_norm(a, None, None, eps=1e-5, normalized_shape=[2])
    print(result)

    # Activation functions
    demo_section("Activation Functions")

    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    print(f"Input: {x}")

    activations = [
        ("GELU", TensorOptimized.gelu),
        ("GELU Fast", TensorOptimized.gelu_fast),
        ("SiLU", TensorOptimized.silu),
        ("Mish", TensorOptimized.mish),
    ]

    for name, func in activations:
        result = func(x)
        print(f"{name:10}: {result}")

    # Attention operations
    demo_section("Attention Operations")

    # Create test tensors for attention
    batch_size, seq_len, hidden_size = 1, 3, 4
    query = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    key = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    value = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    try:
        attention_output = TensorOptimized.scaled_dot_product_attention(query, key, value)
        print(f"\nAttention output shape: {attention_output.shape}")
        print(f"Sample attention output:")
        print(attention_output)
    except Exception as e:
        print(f"Attention operation error: {e}")

    # Summary
    demo_section("Available Functions Summary")

    available_functions = [
        'matmul', 'softmax', 'layer_norm', 'gelu', 'gelu_fast',
        'silu', 'mish', 'scaled_dot_product_attention', 'apply_rotary_pos_emb'
    ]

    print("✅ Available TensorOptimized functions:")
    for func in available_functions:
        print(f"  - {func}")

    print("\n🎉 Demo completed successfully!")
    print("\nNote: This demo shows core functions available in TrustformeRS.")
    print("Advanced functions (Swish, ELU, Complex tensors, etc.) will be available in future releases.")

if __name__ == "__main__":
    main()