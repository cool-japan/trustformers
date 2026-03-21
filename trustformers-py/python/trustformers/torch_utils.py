"""
PyTorch interoperability utilities for TrustformeRS

Provides seamless conversion between PyTorch tensors and TrustformeRS tensors.
"""

from typing import Union, List, Dict, Tuple, Optional, Any
import numpy as np
import warnings

from . import Tensor
from .utils import is_torch_available

if is_torch_available():
    import torch
else:
    torch = None
    warnings.warn(
        "PyTorch is not installed. Install with: pip install torch\n"
        "PyTorch interoperability features will not be available."
    )


def torch_to_tensor(
    torch_tensor: "torch.Tensor",
    device: str = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """
    Convert PyTorch tensor to TrustformeRS Tensor.
    
    Args:
        torch_tensor: PyTorch tensor to convert
        device: Device to place tensor on (not used, for API compatibility)
        requires_grad: Whether tensor requires gradients
        
    Returns:
        TrustformeRS Tensor
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    # Convert to numpy (detach from computation graph and move to CPU)
    if torch_tensor.requires_grad:
        numpy_array = torch_tensor.detach().cpu().numpy()
    else:
        numpy_array = torch_tensor.cpu().numpy()
    
    # Ensure float32 for now (TrustformeRS default)
    if numpy_array.dtype != np.float32:
        numpy_array = numpy_array.astype(np.float32)
    
    # Create TrustformeRS tensor
    return Tensor(numpy_array, device=device, requires_grad=requires_grad)


def tensor_to_torch(
    tensor: Tensor,
    device: Optional[Union[str, "torch.device"]] = None,
    requires_grad: bool = False,
) -> "torch.Tensor":
    """
    Convert TrustformeRS Tensor to PyTorch tensor.
    
    Args:
        tensor: TrustformeRS Tensor
        device: PyTorch device to place tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        PyTorch tensor
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    # Get numpy array
    numpy_array = tensor.numpy()
    
    # Create PyTorch tensor
    torch_tensor = torch.from_numpy(numpy_array)
    
    # Set requires_grad
    if requires_grad:
        torch_tensor = torch_tensor.requires_grad_(True)
    
    # Move to device if specified
    if device is not None:
        torch_tensor = torch_tensor.to(device)
    
    return torch_tensor


def batch_torch_to_tensor(
    torch_tensors: Union[List["torch.Tensor"], Dict[str, "torch.Tensor"]],
    device: str = "cpu",
) -> Union[List[Tensor], Dict[str, Tensor]]:
    """
    Convert a batch of PyTorch tensors to TrustformeRS Tensors.
    
    Args:
        torch_tensors: List or dict of PyTorch tensors
        device: Device to place tensors on
        
    Returns:
        List or dict of TrustformeRS Tensors
    """
    if isinstance(torch_tensors, list):
        return [torch_to_tensor(t, device=device) for t in torch_tensors]
    elif isinstance(torch_tensors, dict):
        return {key: torch_to_tensor(t, device=device) for key, t in torch_tensors.items()}
    else:
        raise TypeError(f"Expected list or dict, got {type(torch_tensors)}")


def batch_tensor_to_torch(
    tensors: Union[List[Tensor], Dict[str, Tensor]],
    device: Optional[Union[str, "torch.device"]] = None,
) -> Union[List["torch.Tensor"], Dict[str, "torch.Tensor"]]:
    """
    Convert a batch of TrustformeRS Tensors to PyTorch tensors.
    
    Args:
        tensors: List or dict of TrustformeRS Tensors
        device: PyTorch device to place tensors on
        
    Returns:
        List or dict of PyTorch tensors
    """
    if isinstance(tensors, list):
        return [tensor_to_torch(t, device=device) for t in tensors]
    elif isinstance(tensors, dict):
        return {key: tensor_to_torch(t, device=device) for key, t in tensors.items()}
    else:
        raise TypeError(f"Expected list or dict, got {type(tensors)}")


def ensure_torch_tensor(
    data: Union["torch.Tensor", np.ndarray, List, Tensor],
    dtype: Optional["torch.dtype"] = None,
    device: Optional[Union[str, "torch.device"]] = None,
) -> "torch.Tensor":
    """
    Ensure data is a PyTorch tensor.
    
    Args:
        data: Input data (tensor, array, list, or TrustformeRS Tensor)
        dtype: Target PyTorch dtype
        device: Target device
        
    Returns:
        PyTorch tensor
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, Tensor):
        tensor = tensor_to_torch(data)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    if dtype is not None:
        tensor = tensor.to(dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def pad_sequence_torch(
    sequences: List["torch.Tensor"],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad PyTorch sequences to the same length.
    
    Args:
        sequences: List of PyTorch tensors
        batch_first: If True, output is (batch, seq, ...), else (seq, batch, ...)
        padding_value: Value to use for padding
        
    Returns:
        Tuple of (padded_sequences, lengths)
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    # Get lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Use PyTorch's pad_sequence
    from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
    
    padded = torch_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
    
    return padded, lengths


class TorchTensorWrapper:
    """
    Wrapper that makes TrustformeRS Tensor compatible with PyTorch operations.
    """
    
    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        self._torch_tensor = None
    
    @property
    def torch_tensor(self) -> "torch.Tensor":
        """Get the PyTorch tensor (cached)."""
        if self._torch_tensor is None:
            self._torch_tensor = tensor_to_torch(self.tensor)
        return self._torch_tensor
    
    def __getattr__(self, name):
        """Forward PyTorch method calls to the torch tensor."""
        if hasattr(torch.Tensor, name):
            attr = getattr(self.torch_tensor, name)
            if callable(attr):
                def method(*args, **kwargs):
                    # Convert TrustformeRS tensors in args
                    args = [arg.torch_tensor if isinstance(arg, TorchTensorWrapper) else arg 
                           for arg in args]
                    result = attr(*args, **kwargs)
                    if isinstance(result, torch.Tensor):
                        return torch_to_tensor(result)
                    return result
                return method
            return attr
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def backward(self, gradient=None):
        """Backward pass (for autograd compatibility)."""
        if self.torch_tensor.requires_grad:
            self.torch_tensor.backward(gradient)
    
    @property
    def grad(self):
        """Get gradient as TrustformeRS tensor."""
        if self.torch_tensor.grad is not None:
            return torch_to_tensor(self.torch_tensor.grad)
        return None
    
    def __repr__(self):
        return f"TorchTensorWrapper({self.tensor})"


# Monkey-patch Tensor class to add PyTorch compatibility
def _tensor_to_torch_method(self, device=None, requires_grad=False):
    """Convert to PyTorch tensor."""
    return tensor_to_torch(self, device=device, requires_grad=requires_grad)


def _tensor_from_torch_method(cls, torch_tensor, device="cpu", requires_grad=False):
    """Create from PyTorch tensor."""
    return torch_to_tensor(torch_tensor, device=device, requires_grad=requires_grad)


def _tensor_as_torch(self):
    """Get PyTorch-compatible wrapper."""
    return TorchTensorWrapper(self)


# Try to add methods to Tensor if it's already imported
try:
    Tensor.to_torch = _tensor_to_torch_method
    Tensor.from_torch = classmethod(_tensor_from_torch_method)
    Tensor.as_torch = _tensor_as_torch
except:
    pass


# PyTorch-specific utility functions
def create_torch_dataloader(
    dataset: List[Dict[str, Any]],
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn: Optional[callable] = None,
) -> "torch.utils.data.DataLoader":
    """
    Create PyTorch DataLoader from TrustformeRS data.
    
    Args:
        dataset: List of data samples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        collate_fn: Custom collate function
        
    Returns:
        PyTorch DataLoader
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    from torch.utils.data import DataLoader, Dataset
    
    class TrustformersDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            # Convert TrustformeRS tensors to PyTorch
            torch_sample = {}
            for key, value in sample.items():
                if isinstance(value, Tensor):
                    torch_sample[key] = tensor_to_torch(value)
                else:
                    torch_sample[key] = value
            return torch_sample
    
    dataset = TrustformersDataset(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def torch_optimizer_step(
    optimizer: "torch.optim.Optimizer",
    loss: Tensor,
    parameters: List[TorchTensorWrapper],
):
    """
    Perform optimizer step with TrustformeRS tensors.
    
    Args:
        optimizer: PyTorch optimizer
        loss: Loss tensor
        parameters: List of parameter wrappers
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    # Convert loss to PyTorch
    loss_torch = tensor_to_torch(loss, requires_grad=True)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss_torch.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Update TrustformeRS tensors with new values
    for param_wrapper in parameters:
        new_values = param_wrapper.torch_tensor.detach().cpu().numpy()
        # Update the underlying tensor data
        # This is a simplified version - in practice would need proper update method
        param_wrapper.tensor = Tensor(new_values)