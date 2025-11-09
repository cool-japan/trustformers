"""
Batch encoding utilities for efficient tokenization and processing.

This module provides utilities for batch encoding text inputs, handling
padding, truncation, and efficient tensor operations for large-scale
text processing workflows.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import warnings
from dataclasses import dataclass
import numpy as np

from .utils import logging
from .numpy_utils import pad_sequence, stack_numpy_arrays

logger = logging.get_logger(__name__)


@dataclass
class BatchEncoding:
    """
    Holds the output of the tokenizer's batch encoding methods.
    
    This class provides an interface similar to HuggingFace's BatchEncoding,
    with additional utilities for efficient batch processing.
    """
    
    data: Dict[str, np.ndarray]
    encodings: Optional[List[Dict[str, Any]]] = None
    
    def __init__(
        self, 
        data: Optional[Dict[str, Any]] = None,
        encodings: Optional[List[Dict[str, Any]]] = None,
        tensor_type: str = "np"
    ):
        """
        Initialize BatchEncoding.
        
        Args:
            data: Dictionary containing the tokenized data
            encodings: List of individual encoding dictionaries  
            tensor_type: Type of tensors to use ('np' for numpy)
        """
        self.data = data or {}
        self.encodings = encodings or []
        self.tensor_type = tensor_type
        
        # Ensure all data is numpy arrays
        self._ensure_numpy_arrays()
    
    def _ensure_numpy_arrays(self):
        """Ensure all data values are numpy arrays."""
        for key, value in self.data.items():
            if not isinstance(value, np.ndarray):
                if isinstance(value, (list, tuple)):
                    self.data[key] = np.array(value)
                else:
                    self.data[key] = np.array([value])
    
    def __getitem__(self, item: Union[str, int]) -> Union[np.ndarray, Dict[str, Any]]:
        """Get item from batch encoding."""
        if isinstance(item, str):
            return self.data[item]
        elif isinstance(item, int):
            # Get individual encoding
            if item < len(self.encodings):
                return self.encodings[item]
            else:
                # Extract from batch data
                result = {}
                for key, value in self.data.items():
                    if item < len(value):
                        result[key] = value[item]
                return result
        else:
            raise TypeError(f"Invalid key type: {type(item)}")
    
    def __setitem__(self, key: str, value: Any):
        """Set item in batch encoding."""
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.data[key] = value
    
    def __contains__(self, item: str) -> bool:
        """Check if key exists in data."""
        return item in self.data
    
    def keys(self):
        """Get data keys."""
        return self.data.keys()
    
    def values(self):
        """Get data values."""
        return self.data.values()
    
    def items(self):
        """Get data items."""
        return self.data.items()
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return dict(self.data)
    
    def convert_to_tensors(self, tensor_type: str = "np") -> "BatchEncoding":
        """
        Convert tensors to specified type.
        
        Args:
            tensor_type: Target tensor type ('np', 'torch', 'jax')
            
        Returns:
            New BatchEncoding with converted tensors
        """
        if tensor_type == "np":
            return self  # Already numpy
        elif tensor_type == "torch":
            try:
                import torch
                converted_data = {}
                for key, value in self.data.items():
                    converted_data[key] = torch.from_numpy(value)
                return BatchEncoding(converted_data, self.encodings, tensor_type)
            except ImportError:
                warnings.warn("PyTorch not available, keeping numpy tensors")
                return self
        elif tensor_type == "jax":
            try:
                import jax.numpy as jnp
                converted_data = {}
                for key, value in self.data.items():
                    converted_data[key] = jnp.array(value)
                return BatchEncoding(converted_data, self.encodings, tensor_type)
            except ImportError:
                warnings.warn("JAX not available, keeping numpy tensors")
                return self
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
    
    def to(self, device: str) -> "BatchEncoding":
        """
        Move tensors to device (PyTorch compatibility).
        
        Args:
            device: Target device
            
        Returns:
            New BatchEncoding with tensors on target device
        """
        if self.tensor_type == "torch":
            try:
                import torch
                converted_data = {}
                for key, value in self.data.items():
                    if hasattr(value, 'to'):
                        converted_data[key] = value.to(device)
                    else:
                        converted_data[key] = value
                return BatchEncoding(converted_data, self.encodings, self.tensor_type)
            except ImportError:
                warnings.warn("PyTorch not available for device placement")
                return self
        else:
            # For numpy/jax, device placement not directly supported
            return self
    
    @property
    def input_ids(self) -> np.ndarray:
        """Get input IDs."""
        return self.data.get('input_ids')
    
    @property
    def attention_mask(self) -> Optional[np.ndarray]:
        """Get attention mask."""
        return self.data.get('attention_mask')
    
    @property
    def token_type_ids(self) -> Optional[np.ndarray]:
        """Get token type IDs."""
        return self.data.get('token_type_ids')
    
    def pad(
        self,
        max_length: Optional[int] = None,
        padding_value: int = 0,
        attention_padding_value: int = 0,
        pad_to_multiple_of: Optional[int] = None
    ) -> "BatchEncoding":
        """
        Apply padding to the batch encoding.
        
        Args:
            max_length: Maximum sequence length
            padding_value: Value to use for padding input_ids
            attention_padding_value: Value to use for padding attention_mask
            pad_to_multiple_of: Pad to multiple of this value
            
        Returns:
            New padded BatchEncoding
        """
        padded_data = {}
        
        for key, sequences in self.data.items():
            if key in ['input_ids', 'token_type_ids']:
                pad_value = padding_value
            elif key == 'attention_mask':
                pad_value = attention_padding_value
            else:
                pad_value = 0
            
            if len(sequences.shape) > 1:
                # Already batched - pad sequences
                padded_sequences = pad_sequence(
                    [seq for seq in sequences],
                    padding_value=pad_value,
                    max_length=max_length,
                    pad_to_multiple_of=pad_to_multiple_of
                )
                padded_data[key] = padded_sequences
            else:
                # Single sequence
                padded_data[key] = sequences
        
        return BatchEncoding(padded_data, self.encodings, self.tensor_type)
    
    def truncate(self, max_length: int) -> "BatchEncoding":
        """
        Truncate sequences to maximum length.
        
        Args:
            max_length: Maximum sequence length
            
        Returns:
            New truncated BatchEncoding
        """
        truncated_data = {}
        
        for key, sequences in self.data.items():
            if len(sequences.shape) > 1:
                # Batch of sequences
                truncated_sequences = sequences[:, :max_length]
                truncated_data[key] = truncated_sequences
            else:
                # Single sequence
                truncated_data[key] = sequences[:max_length]
        
        return BatchEncoding(truncated_data, self.encodings, self.tensor_type)
    
    def batch_size(self) -> int:
        """Get batch size."""
        if not self.data:
            return 0
        first_key = next(iter(self.data.keys()))
        return len(self.data[first_key])
    
    def sequence_length(self) -> Optional[int]:
        """Get sequence length (if all sequences have same length)."""
        if not self.data:
            return None
        
        for key, value in self.data.items():
            if len(value.shape) > 1:
                return value.shape[1]
        return None


class BatchTokenizer:
    """
    Efficient batch tokenization utilities.
    
    Provides methods for efficient tokenization of large batches of text,
    with support for various padding and truncation strategies.
    """
    
    def __init__(self, tokenizer: Any, batch_size: int = 32):
        """
        Initialize batch tokenizer.
        
        Args:
            tokenizer: Underlying tokenizer
            batch_size: Default batch size for processing
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[str], List[Tuple[str, str]]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = "np",
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and encode a batch of sequences.
        
        Args:
            batch_text_or_text_pairs: Batch of text sequences or text pairs
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Truncation strategy  
            max_length: Maximum sequence length
            stride: Stride for overflowing tokens
            pad_to_multiple_of: Pad to multiple of this value
            return_tensors: Type of tensors to return
            return_token_type_ids: Whether to return token type IDs
            return_attention_mask: Whether to return attention mask
            return_overflowing_tokens: Whether to return overflowing tokens
            return_special_tokens_mask: Whether to return special tokens mask
            return_offsets_mapping: Whether to return offset mapping
            return_length: Whether to return sequence lengths
            verbose: Whether to show progress
            **kwargs: Additional arguments
            
        Returns:
            BatchEncoding with tokenized sequences
        """
        if not batch_text_or_text_pairs:
            return BatchEncoding()
        
        # Process in chunks if batch is large
        all_encodings = []
        
        for i in range(0, len(batch_text_or_text_pairs), self.batch_size):
            chunk = batch_text_or_text_pairs[i:i + self.batch_size]
            
            # Encode chunk
            chunk_encodings = []
            for text_or_pair in chunk:
                if isinstance(text_or_pair, (list, tuple)) and len(text_or_pair) == 2:
                    # Text pair
                    encoding = self.tokenizer.encode_plus(
                        text_or_pair[0],
                        text_or_pair[1],
                        add_special_tokens=add_special_tokens,
                        padding=False,  # We'll pad at the end
                        truncation=truncation,
                        max_length=max_length,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        **kwargs
                    )
                else:
                    # Single text
                    encoding = self.tokenizer.encode_plus(
                        text_or_pair,
                        add_special_tokens=add_special_tokens,
                        padding=False,  # We'll pad at the end
                        truncation=truncation,
                        max_length=max_length,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        **kwargs
                    )
                chunk_encodings.append(encoding)
            
            all_encodings.extend(chunk_encodings)
        
        # Combine all encodings into batch
        batch_data = {}
        
        # Get all keys from first encoding
        if all_encodings:
            first_encoding = all_encodings[0]
            for key in first_encoding.keys():
                sequences = [enc.get(key, []) for enc in all_encodings]
                
                # Handle different sequence types
                if key in ['input_ids', 'token_type_ids', 'attention_mask']:
                    if padding:
                        # Pad sequences
                        if key == 'attention_mask':
                            pad_value = 0
                        elif key == 'token_type_ids':
                            pad_value = 0
                        else:
                            pad_value = getattr(self.tokenizer, 'pad_token_id', 0)
                        
                        padded = pad_sequence(
                            sequences,
                            padding_value=pad_value,
                            max_length=max_length,
                            pad_to_multiple_of=pad_to_multiple_of
                        )
                        batch_data[key] = padded
                    else:
                        # Keep as list of sequences
                        batch_data[key] = sequences
                else:
                    # Other keys (e.g., special tokens mask)
                    try:
                        batch_data[key] = stack_numpy_arrays(sequences)
                    except:
                        batch_data[key] = sequences
        
        return BatchEncoding(batch_data, all_encodings, return_tensors or "np")
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "np"
    ) -> BatchEncoding:
        """
        Simple batch encoding method.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            padding: Whether to apply padding
            truncation: Whether to apply truncation
            max_length: Maximum sequence length
            return_tensors: Type of tensors to return
            
        Returns:
            BatchEncoding with encoded sequences
        """
        return self.batch_encode_plus(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
    
    def create_chunks(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None
    ) -> Iterator[List[str]]:
        """
        Create chunks of texts for batch processing.
        
        Args:
            texts: List of texts
            chunk_size: Size of each chunk (defaults to self.batch_size)
            
        Yields:
            Chunks of texts
        """
        chunk_size = chunk_size or self.batch_size
        
        for i in range(0, len(texts), chunk_size):
            yield texts[i:i + chunk_size]


def create_batch_encoding(
    input_ids: List[List[int]],
    attention_mask: Optional[List[List[int]]] = None,
    token_type_ids: Optional[List[List[int]]] = None,
    padding: bool = True,
    max_length: Optional[int] = None,
    pad_token_id: int = 0
) -> BatchEncoding:
    """
    Create a BatchEncoding from raw token sequences.
    
    Args:
        input_ids: List of input ID sequences
        attention_mask: List of attention mask sequences
        token_type_ids: List of token type ID sequences
        padding: Whether to apply padding
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        
    Returns:
        BatchEncoding object
    """
    data = {}
    
    # Process input_ids
    if padding:
        data['input_ids'] = pad_sequence(
            input_ids,
            padding_value=pad_token_id,
            max_length=max_length
        )
    else:
        data['input_ids'] = stack_numpy_arrays(input_ids)
    
    # Process attention_mask
    if attention_mask is not None:
        if padding:
            data['attention_mask'] = pad_sequence(
                attention_mask,
                padding_value=0,
                max_length=max_length
            )
        else:
            data['attention_mask'] = stack_numpy_arrays(attention_mask)
    elif padding:
        # Generate attention mask from input_ids
        attention_mask = []
        for seq in input_ids:
            mask = [1] * len(seq)
            attention_mask.append(mask)
        data['attention_mask'] = pad_sequence(
            attention_mask,
            padding_value=0,
            max_length=max_length
        )
    
    # Process token_type_ids
    if token_type_ids is not None:
        if padding:
            data['token_type_ids'] = pad_sequence(
                token_type_ids,
                padding_value=0,
                max_length=max_length
            )
        else:
            data['token_type_ids'] = stack_numpy_arrays(token_type_ids)
    
    return BatchEncoding(data)


# Export main classes and functions
__all__ = [
    'BatchEncoding',
    'BatchTokenizer', 
    'create_batch_encoding'
]