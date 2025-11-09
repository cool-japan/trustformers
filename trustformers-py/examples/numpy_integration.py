"""
NumPy integration examples for TrustformeRS Python bindings

This example demonstrates seamless integration between NumPy arrays
and TrustformeRS tensors.
"""

import numpy as np
from trustformers import (
    Tensor,
    BertModel,
    numpy_to_tensor,
    tensor_to_numpy,
    pad_sequence,
    batch_convert_to_tensors,
    concatenate,
    split,
    where,
    tensor_stats,
    create_attention_mask_from_lengths,
)


def basic_numpy_conversion():
    """Basic conversion between NumPy and TrustformeRS tensors."""
    print("=== Basic NumPy Conversion ===")
    
    # Create NumPy array
    np_array = np.random.randn(2, 3, 4).astype(np.float32)
    print(f"NumPy array shape: {np_array.shape}")
    
    # Convert to TrustformeRS tensor
    tensor = Tensor(np_array)
    print(f"Tensor shape: {tensor.shape}")
    
    # Convert back to NumPy
    np_array_back = tensor.numpy()
    print(f"Converted back shape: {np_array_back.shape}")
    print(f"Arrays equal: {np.allclose(np_array, np_array_back)}")
    print()


def tensor_operations_with_numpy():
    """Demonstrate tensor operations with NumPy compatibility."""
    print("=== Tensor Operations with NumPy ===")
    
    # Create tensors from NumPy arrays
    a = numpy_to_tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = numpy_to_tensor(np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32))
    
    # Perform operations
    c = a + b
    d = a.matmul(b.transpose())
    
    # Convert results to NumPy for verification
    c_np = tensor_to_numpy(c)
    d_np = tensor_to_numpy(d)
    
    print(f"a + b = \n{c_np}")
    print(f"a @ b.T = \n{d_np}")
    
    # Verify with NumPy
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)
    print(f"NumPy verification (a + b): {np.allclose(c_np, a_np + b_np)}")
    print(f"NumPy verification (a @ b.T): {np.allclose(d_np, a_np @ b_np.T)}")
    print()


def sequence_padding_example():
    """Demonstrate sequence padding for batch processing."""
    print("=== Sequence Padding ===")
    
    # Variable length sequences
    sequences = [
        np.array([1, 2, 3]),
        np.array([4, 5]),
        np.array([6, 7, 8, 9, 10]),
        np.array([11]),
    ]
    
    # Pad sequences
    padded_seqs, attention_mask = pad_sequence(sequences, padding_value=0)
    
    print(f"Padded sequences shape: {padded_seqs.shape}")
    print(f"Padded sequences:\n{padded_seqs}")
    print(f"Attention mask:\n{attention_mask}")
    print()


def batch_processing_example():
    """Demonstrate batch processing with NumPy arrays."""
    print("=== Batch Processing ===")
    
    # Create batch of embeddings
    batch_size = 4
    seq_length = 10
    hidden_size = 768
    
    # Simulate embeddings
    embeddings = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    
    # Create attention mask from lengths
    lengths = [8, 10, 6, 9]
    attention_mask = create_attention_mask_from_lengths(lengths, max_length=seq_length)
    
    # Convert to tensors
    batch_data = batch_convert_to_tensors({
        "embeddings": embeddings,
        "attention_mask": attention_mask,
    })
    
    print(f"Batch embeddings shape: {batch_data['embeddings'].shape}")
    print(f"Attention mask shape: {batch_data['attention_mask'].shape}")
    print(f"Attention mask:\n{attention_mask}")
    print()


def advanced_numpy_operations():
    """Demonstrate advanced NumPy operations on tensors."""
    print("=== Advanced NumPy Operations ===")
    
    # Create tensor
    tensor = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    
    # Get statistics
    stats = tensor_stats(tensor)
    print("Tensor statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Concatenate tensors
    tensor1 = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    tensor2 = Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
    concat_tensor = concatenate([tensor1, tensor2], axis=0)
    print(f"\nConcatenated tensor:\n{concat_tensor.numpy()}")
    
    # Split tensor
    split_tensors = split(concat_tensor, 2, axis=0)
    print(f"\nSplit tensors:")
    for i, t in enumerate(split_tensors):
        print(f"  Tensor {i}: {t.numpy()}")
    
    # Where operation
    condition = Tensor(np.array([[True, False], [False, True]]))
    x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    y = Tensor(np.array([[10, 20], [30, 40]], dtype=np.float32))
    result = where(condition, x, y)
    print(f"\nWhere operation result:\n{result.numpy()}")
    print()


def model_inference_with_numpy():
    """Demonstrate model inference with NumPy inputs."""
    print("=== Model Inference with NumPy ===")
    
    # Create mock model
    config = {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12}
    model = BertModel(config)
    
    # Create inputs as NumPy arrays
    batch_size = 2
    seq_length = 16
    
    # Random token IDs
    input_ids_np = np.random.randint(0, 30000, size=(batch_size, seq_length))
    attention_mask_np = np.ones((batch_size, seq_length))
    
    # Convert to tensors
    input_ids = Tensor(input_ids_np.astype(np.float32))
    attention_mask = Tensor(attention_mask_np.astype(np.float32))
    
    # Run inference
    outputs = model(input_ids, attention_mask)
    
    # Extract outputs as NumPy
    if hasattr(outputs, "last_hidden_state"):
        hidden_states_np = outputs.last_hidden_state.numpy()
    else:
        hidden_states_np = outputs["last_hidden_state"].numpy()
    
    print(f"Input shape: {input_ids_np.shape}")
    print(f"Output shape: {hidden_states_np.shape}")
    print(f"Output dtype: {hidden_states_np.dtype}")
    print()


def numpy_broadcasting_example():
    """Demonstrate NumPy broadcasting with tensors."""
    print("=== NumPy Broadcasting ===")
    
    # Create tensors with different shapes
    tensor1 = Tensor(np.array([[1, 2, 3]], dtype=np.float32))  # Shape: (1, 3)
    tensor2 = Tensor(np.array([[4], [5], [6]], dtype=np.float32))  # Shape: (3, 1)
    
    # Broadcasting will work
    result = tensor1 + tensor2
    
    print(f"Tensor 1 shape: {tensor1.shape}")
    print(f"Tensor 2 shape: {tensor2.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result.numpy()}")
    print()


def numpy_indexing_example():
    """Demonstrate NumPy-style indexing."""
    print("=== NumPy-style Indexing ===")
    
    # Create tensor
    tensor = Tensor(np.arange(24).reshape(2, 3, 4).astype(np.float32))
    print(f"Original tensor shape: {tensor.shape}")
    
    # Convert to numpy for indexing (current limitation)
    np_array = tensor.numpy()
    
    # Various indexing operations
    print(f"tensor[0, :, :] shape: {np_array[0, :, :].shape}")
    print(f"tensor[:, 1, :] shape: {np_array[:, 1, :].shape}")
    print(f"tensor[..., -1] shape: {np_array[..., -1].shape}")
    print(f"tensor[0, 1:, ::2] = {np_array[0, 1:, ::2]}")
    print()


def main():
    """Run all examples."""
    print("TrustformeRS NumPy Integration Examples")
    print("=" * 50)
    print()
    
    basic_numpy_conversion()
    tensor_operations_with_numpy()
    sequence_padding_example()
    batch_processing_example()
    advanced_numpy_operations()
    model_inference_with_numpy()
    numpy_broadcasting_example()
    numpy_indexing_example()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()