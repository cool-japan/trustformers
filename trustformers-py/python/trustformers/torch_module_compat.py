"""
PyTorch nn.Module compatibility for TrustformeRS models.

This module provides wrappers that make TrustformeRS models compatible with
PyTorch's nn.Module interface, enabling seamless integration with PyTorch
training loops, optimizers, and other PyTorch ecosystem tools.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable, TYPE_CHECKING
import warnings
import numpy as np

from .utils import is_torch_available
from .torch_utils import tensor_to_torch, torch_to_tensor, batch_tensor_to_torch, batch_torch_to_tensor

if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
else:
    torch = None
    nn = None
    warnings.warn(
        "PyTorch is not installed. Install with: pip install torch\n"
        "torch.nn.Module compatibility features will not be available."
    )

if TYPE_CHECKING:
    if is_torch_available():
        import torch
        TorchTensor = TorchTensor
    else:
        TorchTensor = Any
else:
    TorchTensor = Any


class TorchModuleWrapper(nn.Module if nn else object):
    """
    Wrapper that makes TrustformeRS models compatible with torch.nn.Module.
    
    This wrapper automatically handles conversion between PyTorch tensors and
    TrustformeRS tensors, making it easy to use TrustformeRS models in PyTorch
    training loops and pipelines.
    """
    
    def __init__(self, trustformers_model: Any, device: str = "cpu"):
        """
        Initialize the wrapper.
        
        Args:
            trustformers_model: TrustformeRS model to wrap
            device: PyTorch device to use
        """
        if torch is None:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
            
        super().__init__()
        self.trustformers_model = trustformers_model
        self.device = torch.device(device) if torch else device
        
        # Register model parameters if available
        self._register_parameters()
    
    def _register_parameters(self):
        """Register TrustformeRS model parameters as PyTorch parameters."""
        # This is a simplified version - in practice would need to extract
        # actual parameters from the Rust backend
        try:
            if hasattr(self.trustformers_model, 'get_parameters'):
                params = self.trustformers_model.get_parameters()
                for name, param in params.items():
                    if hasattr(param, 'numpy'):
                        torch_param = tensor_to_torch(param, device=self.device, requires_grad=True)
                        self.register_parameter(name, nn.Parameter(torch_param))
        except:
            # Fallback: create dummy parameters for demonstration
            pass
    
    def forward(self, *args, **kwargs) -> Union[TorchTensor, Dict[str, TorchTensor], Tuple[TorchTensor, ...]]:
        """
        Forward pass with automatic tensor conversion.
        
        Args:
            *args: Positional arguments (will be converted to TrustformeRS tensors)
            **kwargs: Keyword arguments (tensors will be converted to TrustformeRS tensors)
            
        Returns:
            Model outputs converted to PyTorch tensors
        """
        # Convert PyTorch tensors to TrustformeRS tensors
        converted_args = []
        for arg in args:
            if torch is not None and isinstance(arg, torch.Tensor):
                converted_args.append(torch_to_tensor(arg))
            else:
                converted_args.append(arg)
        
        converted_kwargs = {}
        for key, value in kwargs.items():
            if torch is not None and isinstance(value, torch.Tensor):
                converted_kwargs[key] = torch_to_tensor(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (e.g., batch inputs)
                converted_dict = {}
                for k, v in value.items():
                    if torch is not None and isinstance(v, torch.Tensor):
                        converted_dict[k] = torch_to_tensor(v)
                    else:
                        converted_dict[k] = v
                converted_kwargs[key] = converted_dict
            else:
                converted_kwargs[key] = value
        
        # Call TrustformeRS model
        outputs = self.trustformers_model(*converted_args, **converted_kwargs)
        
        # Convert outputs back to PyTorch tensors
        return self._convert_outputs(outputs)
    
    def _convert_outputs(self, outputs: Any) -> Any:
        """Convert TrustformeRS outputs to PyTorch tensors."""
        if hasattr(outputs, 'numpy'):
            # Single tensor output
            return tensor_to_torch(outputs, device=self.device)
        elif isinstance(outputs, dict):
            # Dictionary output (common for model outputs)
            converted = {}
            for key, value in outputs.items():
                if hasattr(value, 'numpy'):
                    converted[key] = tensor_to_torch(value, device=self.device)
                elif isinstance(value, np.ndarray):
                    converted[key] = torch.from_numpy(value).to(self.device) if torch else value
                else:
                    converted[key] = value
            return converted
        elif isinstance(outputs, (tuple, list)):
            # Tuple/list output
            converted = []
            for item in outputs:
                if hasattr(item, 'numpy'):
                    converted.append(tensor_to_torch(item, device=self.device))
                elif isinstance(item, np.ndarray):
                    converted.append(torch.from_numpy(item).to(self.device) if torch else item)
                else:
                    converted.append(item)
            return tuple(converted) if isinstance(outputs, tuple) else converted
        else:
            return outputs
    
    def generate(self, *args, **kwargs) -> TorchTensor:
        """
        Text generation with PyTorch tensor outputs.
        
        Args:
            *args: Arguments for generation
            **kwargs: Keyword arguments for generation
            
        Returns:
            Generated tokens as PyTorch tensor
        """
        if not hasattr(self.trustformers_model, 'generate'):
            raise AttributeError(f"{type(self.trustformers_model).__name__} does not support generation")
        
        # Convert inputs
        converted_args = []
        for arg in args:
            if torch is not None and isinstance(arg, torch.Tensor):
                converted_args.append(torch_to_tensor(arg))
            else:
                converted_args.append(arg)
        
        converted_kwargs = {}
        for key, value in kwargs.items():
            if torch is not None and isinstance(value, torch.Tensor):
                converted_kwargs[key] = torch_to_tensor(value)
            else:
                converted_kwargs[key] = value
        
        # Generate
        outputs = self.trustformers_model.generate(*converted_args, **converted_kwargs)
        
        # Convert back to PyTorch
        return self._convert_outputs(outputs)
    
    def to(self, device):
        """Move model to device (PyTorch compatibility)."""
        super().to(device)
        self.device = torch.device(device) if torch else device
        return self
    
    def train(self, mode: bool = True):
        """Set training mode (PyTorch compatibility)."""
        super().train(mode)
        if hasattr(self.trustformers_model, 'train'):
            self.trustformers_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode (PyTorch compatibility)."""
        super().eval()
        if hasattr(self.trustformers_model, 'eval'):
            self.trustformers_model.eval()
        return self
    
    def save_pretrained(self, save_directory: str):
        """Save the wrapped model."""
        if hasattr(self.trustformers_model, 'save_pretrained'):
            self.trustformers_model.save_pretrained(save_directory)
        else:
            raise NotImplementedError("Wrapped model does not support save_pretrained")
    
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cpu", **kwargs):
        """Load a pretrained model and wrap it."""
        # This would need to be implemented based on the specific model types
        raise NotImplementedError("from_pretrained not yet implemented for wrapped models")


class TorchModuleBert(TorchModuleWrapper):
    """PyTorch nn.Module wrapper specifically for BERT models."""
    
    def __init__(self, bert_model: Any, device: str = "cpu"):
        super().__init__(bert_model, device)
        
        # Add BERT-specific functionality
        if hasattr(bert_model, 'config'):
            self.config = bert_model.config
    
    def forward(
        self,
        input_ids: TorchTensor,
        attention_mask: Optional[TorchTensor] = None,
        token_type_ids: Optional[TorchTensor] = None,
        position_ids: Optional[TorchTensor] = None,
        **kwargs
    ) -> Dict[str, TorchTensor]:
        """
        BERT forward pass with PyTorch tensors.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            
        Returns:
            Dictionary with last_hidden_state, pooler_output, etc.
        """
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        
        return super().forward(**inputs, **kwargs)


class TorchModuleGPT2(TorchModuleWrapper):
    """PyTorch nn.Module wrapper specifically for GPT-2 models."""
    
    def __init__(self, gpt2_model: Any, device: str = "cpu"):
        super().__init__(gpt2_model, device)
        
        if hasattr(gpt2_model, 'config'):
            self.config = gpt2_model.config
    
    def forward(
        self,
        input_ids: TorchTensor,
        attention_mask: Optional[TorchTensor] = None,
        past_key_values: Optional[Tuple[TorchTensor]] = None,
        **kwargs
    ) -> Dict[str, TorchTensor]:
        """
        GPT-2 forward pass with PyTorch tensors.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Past key values for faster generation
            
        Returns:
            Dictionary with logits, hidden_states, etc.
        """
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if past_key_values is not None:
            inputs["past_key_values"] = past_key_values
        
        return super().forward(**inputs, **kwargs)


class TorchLoss(nn.Module if nn else object):
    """
    PyTorch-compatible loss functions for TrustformeRS models.
    """
    
    def __init__(self, loss_type: str = "cross_entropy", **kwargs):
        """
        Initialize loss function.
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'mse', 'binary_cross_entropy')
            **kwargs: Additional arguments for the loss function
        """
        if torch is None:
            raise ImportError("PyTorch is not installed")
            
        super().__init__()
        self.loss_type = loss_type
        self.kwargs = kwargs
        
        # Initialize PyTorch loss function
        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(**kwargs)
        elif loss_type == "binary_cross_entropy":
            self.loss_fn = nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: TorchTensor, targets: TorchTensor) -> TorchTensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        return self.loss_fn(predictions, targets)


def create_torch_training_loop(
    model: TorchModuleWrapper,
    dataloader: Any,
    optimizer: Any,
    loss_fn: TorchLoss,
    num_epochs: int = 1,
    device: str = "cpu"
) -> Callable:
    """
    Create a standard PyTorch training loop for TrustformeRS models.
    
    Args:
        model: Wrapped TrustformeRS model
        dataloader: PyTorch DataLoader
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Training function
    """
    if torch is None:
        raise ImportError("PyTorch is not installed")
    
    def train():
        """Training function."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch is not None and isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(batch, dict):
                    if 'labels' in batch:
                        labels = batch.pop('labels')
                        outputs = model(**batch)
                    else:
                        outputs = model(**batch)
                        labels = None
                else:
                    outputs = model(batch)
                    labels = None
                
                # Compute loss
                if labels is not None:
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        loss = loss_fn(outputs['logits'], labels)
                    else:
                        loss = loss_fn(outputs, labels)
                else:
                    # Assume outputs contain loss
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        raise ValueError("No labels provided and no loss in outputs")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            print(f"Epoch {epoch} completed, Average Loss: {avg_epoch_loss:.4f}")
        
        return total_loss / num_epochs
    
    return train


# Convenience functions
def wrap_bert_model(bert_model: Any, device: str = "cpu") -> TorchModuleBert:
    """Wrap a TrustformeRS BERT model for PyTorch compatibility."""
    return TorchModuleBert(bert_model, device)


def wrap_gpt2_model(gpt2_model: Any, device: str = "cpu") -> TorchModuleGPT2:
    """Wrap a TrustformeRS GPT-2 model for PyTorch compatibility."""
    return TorchModuleGPT2(gpt2_model, device)


def wrap_model(model: Any, device: str = "cpu") -> TorchModuleWrapper:
    """Wrap any TrustformeRS model for PyTorch compatibility."""
    model_type = type(model).__name__.lower()
    
    if 'bert' in model_type:
        return TorchModuleBert(model, device)
    elif 'gpt' in model_type:
        return TorchModuleGPT2(model, device)
    else:
        return TorchModuleWrapper(model, device)


# Export main classes and functions
__all__ = [
    'TorchModuleWrapper',
    'TorchModuleBert', 
    'TorchModuleGPT2',
    'TorchLoss',
    'create_torch_training_loop',
    'wrap_bert_model',
    'wrap_gpt2_model', 
    'wrap_model'
]