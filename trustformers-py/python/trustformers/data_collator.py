"""
Data collators for batching inputs during training and evaluation.

These classes handle the batching of inputs, including padding, attention masks,
and special token handling, compatible with HuggingFace's interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import warnings

import numpy as np
from .numpy_utils import pad_sequence, stack_numpy_arrays, ensure_numpy_array
from .utils import logging

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

logger = logging.get_logger(__name__)


@dataclass
class DataCollatorMixin:
    """
    Base mixin for all data collators.
    """
    tokenizer: Any = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "np"


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching inputs.
    """
    
    @abstractmethod
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Take a list of features and collate them into a batch.
        """
        pass


@dataclass
class DefaultDataCollator(DataCollator):
    """
    Very simple data collator that simply collates batches of dict-like objects
    and performs minimal processing.
    """
    return_tensors: str = "np"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of features into a batch.
        """
        if not features:
            return {}
            
        # Get all keys from the first feature
        first = features[0]
        batch = {}
        
        for key in first.keys():
            values = [feature[key] for feature in features if key in feature]
            
            if len(values) == 0:
                continue
                
            # Handle different types of values
            if isinstance(values[0], (int, float)):
                # Scalar values - convert to array
                batch[key] = np.array(values)
            elif isinstance(values[0], (list, tuple)):
                # List/tuple values - try to stack
                try:
                    batch[key] = stack_numpy_arrays(values)
                except:
                    # If stacking fails, keep as list
                    batch[key] = values
            elif hasattr(values[0], 'shape'):
                # Array-like values - try to stack
                try:
                    batch[key] = stack_numpy_arrays(values)
                except:
                    # If stacking fails, keep as list
                    batch[key] = values
            else:
                # Other types - keep as list
                batch[key] = values
                
        return batch


@dataclass
class DataCollatorWithPadding(DataCollator, DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received.
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate and pad a list of features into a batch.
        """
        if not features:
            return {}
            
        # Get all keys from features
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
            
        batch = {}
        
        for key in all_keys:
            values = [feature.get(key) for feature in features]
            
            # Skip None values
            values = [v for v in values if v is not None]
            if len(values) == 0:
                continue
                
            # Handle different key types
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                # These need padding
                if self.tokenizer is not None:
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                    if key == 'attention_mask':
                        pad_value = 0
                    elif key == 'token_type_ids':
                        pad_value = 0
                    else:
                        pad_value = pad_token_id
                else:
                    pad_value = 0
                    
                # Pad sequences
                padded = pad_sequence(
                    values, 
                    padding_value=pad_value,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
                batch[key] = padded
                
            elif key == 'labels':
                # Labels might need special handling
                if isinstance(values[0], (list, tuple, np.ndarray)):
                    # Sequence labels - pad with -100 (ignore index)
                    padded = pad_sequence(
                        values,
                        padding_value=-100,
                        max_length=self.max_length,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                    batch[key] = padded
                else:
                    # Classification labels - just stack
                    batch[key] = np.array(values)
                    
            else:
                # Other keys - try to handle generically
                if isinstance(values[0], (int, float)):
                    batch[key] = np.array(values)
                elif isinstance(values[0], (list, tuple, np.ndarray)):
                    try:
                        batch[key] = stack_numpy_arrays(values)
                    except:
                        # Try padding if stacking fails
                        try:
                            batch[key] = pad_sequence(values, padding_value=0)
                        except:
                            batch[key] = values
                else:
                    batch[key] = values
                    
        return batch


@dataclass
class DataCollatorForLanguageModeling(DataCollator, DataCollatorMixin):
    """
    Data collator for language modeling tasks (e.g., GPT, BERT MLM).
    """
    mlm: bool = True
    mlm_probability: float = 0.15
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate features and apply masking for language modeling.
        """
        # First apply standard padding
        padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        batch = padding_collator(features)
        
        if self.mlm and 'input_ids' in batch:
            # Apply MLM masking
            batch = self._apply_mlm_masking(batch)
            
        return batch
    
    def _apply_mlm_masking(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply masked language modeling to the batch.
        """
        input_ids = batch['input_ids']
        
        # Create labels (copy of input_ids)
        labels = input_ids.copy()
        
        # Create random mask
        probability_matrix = np.random.random(input_ids.shape)
        
        # Don't mask special tokens
        if self.tokenizer is not None:
            special_tokens_mask = self._get_special_tokens_mask(input_ids)
            probability_matrix = np.where(special_tokens_mask, 0.0, probability_matrix)
            
        # Create mask
        masked_indices = probability_matrix < self.mlm_probability
        
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~masked_indices] = -100
        
        # 80% of the time: replace with [MASK] token
        indices_replaced = masked_indices & (np.random.random(input_ids.shape) < 0.8)
        if self.tokenizer is not None and hasattr(self.tokenizer, 'mask_token_id'):
            input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time: replace with random token
        indices_random = masked_indices & ~indices_replaced & (np.random.random(input_ids.shape) < 0.5)
        if self.tokenizer is not None:
            vocab_size = getattr(self.tokenizer, 'vocab_size', 50000)
            random_words = np.random.randint(0, vocab_size, size=input_ids.shape)
            input_ids[indices_random] = random_words[indices_random]
        
        # 10% of the time: keep original token (no change needed)
        
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        
        return batch
    
    def _get_special_tokens_mask(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Get mask for special tokens.
        """
        if self.tokenizer is None:
            return np.zeros_like(input_ids, dtype=bool)
            
        special_tokens = []
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'unk_token_id']:
            if hasattr(self.tokenizer, attr):
                token_id = getattr(self.tokenizer, attr)
                if token_id is not None:
                    special_tokens.append(token_id)
        
        mask = np.zeros_like(input_ids, dtype=bool)
        for token_id in special_tokens:
            mask |= (input_ids == token_id)
            
        return mask


@dataclass
class DataCollatorForSeq2Seq(DataCollator, DataCollatorMixin):
    """
    Data collator for sequence-to-sequence tasks.
    """
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate features for seq2seq tasks.
        """
        # Separate input and label features
        input_features = []
        label_features = []
        
        for feature in features:
            input_feature = {k: v for k, v in feature.items() if k != 'labels'}
            input_features.append(input_feature)
            
            if 'labels' in feature:
                label_features.append({'input_ids': feature['labels']})
        
        # Collate inputs
        padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        batch = padding_collator(input_features)
        
        # Collate labels if present
        if label_features:
            # Use special pad token for labels
            label_collator = DataCollatorWithPadding(
                tokenizer=None,  # Don't use tokenizer for labels
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            
            # Override padding value for labels
            labels = [feature['input_ids'] for feature in label_features]
            batch['labels'] = pad_sequence(
                labels,
                padding_value=self.label_pad_token_id,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            
        return batch


@dataclass
class DataCollatorForTokenClassification(DataCollator, DataCollatorMixin):
    """
    Data collator for token classification tasks (NER, POS tagging).
    """
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate features for token classification.
        """
        # Use padding collator but handle labels specially
        padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        # Separate features without labels first
        features_without_labels = []
        labels = []
        
        for feature in features:
            feature_copy = feature.copy()
            if 'labels' in feature_copy:
                labels.append(feature_copy.pop('labels'))
            else:
                labels.append(None)
            features_without_labels.append(feature_copy)
        
        batch = padding_collator(features_without_labels)
        
        # Handle labels separately with proper padding
        if any(label is not None for label in labels):
            # Filter out None labels
            valid_labels = [label for label in labels if label is not None]
            if valid_labels:
                batch['labels'] = pad_sequence(
                    valid_labels,
                    padding_value=self.label_pad_token_id,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
        
        return batch


# Convenience function to get the appropriate data collator
def get_data_collator(
    task_type: str,
    tokenizer: Any = None,
    **kwargs
) -> DataCollator:
    """
    Get the appropriate data collator for a task type.
    
    Args:
        task_type: Type of task ('default', 'mlm', 'seq2seq', 'token_classification')
        tokenizer: Tokenizer to use for padding
        **kwargs: Additional arguments for the data collator
        
    Returns:
        Appropriate DataCollator instance
    """
    if task_type == 'mlm' or task_type == 'language_modeling':
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, **kwargs)
    elif task_type == 'seq2seq':
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, **kwargs)
    elif task_type == 'token_classification':
        return DataCollatorForTokenClassification(tokenizer=tokenizer, **kwargs)
    elif task_type == 'with_padding':
        return DataCollatorWithPadding(tokenizer=tokenizer, **kwargs)
    else:
        return DefaultDataCollator(**kwargs)