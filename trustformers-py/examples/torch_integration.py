"""
PyTorch integration examples for TrustformeRS Python bindings

This example demonstrates seamless integration between PyTorch tensors
and TrustformeRS tensors.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch is not installed. Install with: pip install torch")
    print("Skipping PyTorch integration examples.")
    TORCH_AVAILABLE = False
    exit(0)

from trustformers import (
    Tensor,
    BertModel,
    is_torch_available,
)

if is_torch_available():
    from trustformers import (
        torch_to_tensor,
        tensor_to_torch,
        batch_torch_to_tensor,
        batch_tensor_to_torch,
        ensure_torch_tensor,
        TorchTensorWrapper,
        create_torch_dataloader,
    )


def basic_torch_conversion():
    """Basic conversion between PyTorch and TrustformeRS tensors."""
    print("=== Basic PyTorch Conversion ===")
    
    # Create PyTorch tensor
    torch_tensor = torch.randn(2, 3, 4)
    print(f"PyTorch tensor shape: {torch_tensor.shape}")
    print(f"PyTorch tensor device: {torch_tensor.device}")
    print(f"PyTorch tensor requires_grad: {torch_tensor.requires_grad}")
    
    # Convert to TrustformeRS tensor
    trust_tensor = torch_to_tensor(torch_tensor)
    print(f"TrustformeRS tensor shape: {trust_tensor.shape}")
    
    # Convert back to PyTorch
    torch_tensor_back = tensor_to_torch(trust_tensor)
    print(f"Converted back shape: {torch_tensor_back.shape}")
    print(f"Tensors equal: {torch.allclose(torch_tensor, torch_tensor_back)}")
    print()


def gradient_tracking_example():
    """Demonstrate gradient tracking compatibility."""
    print("=== Gradient Tracking ===")
    
    # Create PyTorch tensor with gradients
    torch_tensor = torch.randn(3, 3, requires_grad=True)
    
    # Convert to TrustformeRS
    trust_tensor = torch_to_tensor(torch_tensor, requires_grad=True)
    
    # Convert back with gradients
    torch_tensor_back = tensor_to_torch(trust_tensor, requires_grad=True)
    
    # Perform operation
    result = torch_tensor_back.sum()
    result.backward()
    
    print(f"Original requires_grad: {torch_tensor.requires_grad}")
    print(f"Converted requires_grad: {torch_tensor_back.requires_grad}")
    print(f"Gradient shape: {torch_tensor_back.grad.shape}")
    print()


def device_compatibility():
    """Demonstrate device compatibility."""
    print("=== Device Compatibility ===")
    
    # Create TrustformeRS tensor
    trust_tensor = Tensor(np.random.randn(2, 3).astype(np.float32))
    
    # Convert to PyTorch on different devices
    if torch.cuda.is_available():
        torch_cuda = tensor_to_torch(trust_tensor, device="cuda")
        print(f"CUDA tensor device: {torch_cuda.device}")
    else:
        print("CUDA not available, using CPU")
    
    torch_cpu = tensor_to_torch(trust_tensor, device="cpu")
    print(f"CPU tensor device: {torch_cpu.device}")
    print()


def batch_conversion_example():
    """Demonstrate batch conversion."""
    print("=== Batch Conversion ===")
    
    # Create batch of PyTorch tensors
    torch_batch = {
        "input_ids": torch.randint(0, 1000, (4, 16)),
        "attention_mask": torch.ones(4, 16),
        "labels": torch.randint(0, 2, (4,)),
    }
    
    # Convert to TrustformeRS
    trust_batch = batch_torch_to_tensor(torch_batch)
    
    print("Batch converted to TrustformeRS:")
    for key, tensor in trust_batch.items():
        print(f"  {key}: shape {tensor.shape}")
    
    # Convert back
    torch_batch_back = batch_tensor_to_torch(trust_batch)
    
    print("\nBatch converted back to PyTorch:")
    for key, tensor in torch_batch_back.items():
        print(f"  {key}: shape {tensor.shape}, device {tensor.device}")
    print()


def model_integration_example():
    """Demonstrate integration with PyTorch models."""
    print("=== Model Integration ===")
    
    # Create TrustformeRS model
    config = {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12}
    trust_model = BertModel(config)
    
    # Create inputs as PyTorch tensors
    torch_input_ids = torch.randint(0, 30000, (2, 16))
    torch_attention_mask = torch.ones(2, 16)
    
    # Convert to TrustformeRS
    trust_input_ids = torch_to_tensor(torch_input_ids)
    trust_attention_mask = torch_to_tensor(torch_attention_mask)
    
    # Run inference
    outputs = trust_model(trust_input_ids, trust_attention_mask)
    
    # Get output as PyTorch tensor
    if hasattr(outputs, "last_hidden_state"):
        trust_hidden_states = outputs.last_hidden_state
    else:
        trust_hidden_states = outputs["last_hidden_state"]
    
    torch_hidden_states = tensor_to_torch(trust_hidden_states)
    
    print(f"Input shape: {torch_input_ids.shape}")
    print(f"Output shape: {torch_hidden_states.shape}")
    print(f"Output is PyTorch tensor: {isinstance(torch_hidden_states, torch.Tensor)}")
    print()


def pytorch_model_wrapper():
    """Demonstrate wrapping PyTorch models with TrustformeRS tensors."""
    print("=== PyTorch Model Wrapper ===")
    
    # Define a simple PyTorch model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create model
    model = SimpleModel(10, 20, 5)
    
    # Create TrustformeRS input
    trust_input = Tensor(np.random.randn(3, 10).astype(np.float32))
    
    # Convert to PyTorch for model
    torch_input = tensor_to_torch(trust_input)
    
    # Run model
    torch_output = model(torch_input)
    
    # Convert output back to TrustformeRS
    trust_output = torch_to_tensor(torch_output)
    
    print(f"Model input shape: {trust_input.shape}")
    print(f"Model output shape: {trust_output.shape}")
    print()


def dataloader_example():
    """Demonstrate DataLoader integration."""
    print("=== DataLoader Integration ===")
    
    # Create dataset with TrustformeRS tensors
    dataset = []
    for i in range(100):
        sample = {
            "input": Tensor(np.random.randn(10).astype(np.float32)),
            "label": np.random.randint(0, 2),
        }
        dataset.append(sample)
    
    # Create PyTorch DataLoader
    dataloader = create_torch_dataloader(dataset, batch_size=16, shuffle=True)
    
    # Iterate through one batch
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Input is PyTorch tensor: {isinstance(batch['input'], torch.Tensor)}")
        break
    print()


def mixed_operations_example():
    """Demonstrate mixed operations between PyTorch and TrustformeRS."""
    print("=== Mixed Operations ===")
    
    # Create tensors
    trust_a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    torch_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    # Convert for operations
    torch_a = tensor_to_torch(trust_a)
    trust_b = torch_to_tensor(torch_b)
    
    # PyTorch operations
    torch_result = torch_a @ torch_b
    print(f"PyTorch matmul result:\n{torch_result}")
    
    # TrustformeRS operations
    trust_result = trust_a.matmul(trust_b)
    print(f"TrustformeRS matmul result:\n{trust_result.numpy()}")
    
    # Verify results match
    trust_result_as_torch = tensor_to_torch(trust_result)
    print(f"Results match: {torch.allclose(torch_result, trust_result_as_torch)}")
    print()


def training_loop_example():
    """Demonstrate a simple training loop with mixed tensors."""
    print("=== Training Loop Example ===")
    
    # Create simple dataset
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32)
    
    # Create PyTorch model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(5):
        # Create TrustformeRS tensors
        trust_X = Tensor(X)
        trust_y = Tensor(y.reshape(-1, 1))
        
        # Convert to PyTorch
        torch_X = tensor_to_torch(trust_X)
        torch_y = tensor_to_torch(trust_y)
        
        # Forward pass
        outputs = model(torch_X)
        loss = criterion(outputs, torch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print()


def tensor_wrapper_example():
    """Demonstrate TorchTensorWrapper functionality."""
    print("=== Tensor Wrapper Example ===")
    
    # Create TrustformeRS tensor
    trust_tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    
    # Get PyTorch wrapper
    wrapper = TorchTensorWrapper(trust_tensor)
    
    # Use PyTorch methods
    print(f"Shape: {wrapper.shape}")
    print(f"Mean: {wrapper.mean().item()}")
    print(f"Sum: {wrapper.sum().item()}")
    
    # PyTorch operations
    result = wrapper.torch_tensor.t()  # Transpose
    print(f"Transposed shape: {result.shape}")
    print()


def main():
    """Run all examples."""
    if not TORCH_AVAILABLE:
        return
    
    print("TrustformeRS PyTorch Integration Examples")
    print("=" * 50)
    print()
    
    basic_torch_conversion()
    gradient_tracking_example()
    device_compatibility()
    batch_conversion_example()
    model_integration_example()
    pytorch_model_wrapper()
    dataloader_example()
    mixed_operations_example()
    training_loop_example()
    tensor_wrapper_example()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()