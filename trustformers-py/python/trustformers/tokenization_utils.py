"""
Tokenization utilities for TrustformeRS

Provides compatibility classes for tokenizers.
"""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import warnings


class BatchEncoding(dict):
    """
    Holds the output of tokenizer's encoding methods.
    """
    
    def __init__(self, data: Dict[str, Any], encoding=None):
        super().__init__(data)
        self._encoding = encoding
    
    @property
    def input_ids(self) -> List[List[int]]:
        """The input IDs."""
        return self["input_ids"]
    
    @property
    def attention_mask(self) -> List[List[int]]:
        """The attention mask."""
        return self.get("attention_mask", None)
    
    @property
    def token_type_ids(self) -> Optional[List[List[int]]]:
        """The token type IDs."""
        return self.get("token_type_ids", None)
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    
    def to(self, device: str) -> "BatchEncoding":
        """
        Send all values to device (for PyTorch compatibility).
        """
        if device != "cpu":
            warnings.warn(f"Device {device} not supported, using CPU")
        return self
    
    def convert_to_tensors(self, tensor_type: str = "np") -> "BatchEncoding":
        """
        Convert to tensors (NumPy arrays by default).
        """
        if tensor_type == "np":
            for key, value in self.items():
                if isinstance(value, list):
                    self[key] = np.array(value)
        elif tensor_type == "pt":
            try:
                import torch
                for key, value in self.items():
                    if isinstance(value, list):
                        self[key] = torch.tensor(value)
            except ImportError:
                raise ImportError("PyTorch is not installed. Install with: pip install torch")
        else:
            raise ValueError(f"Unsupported tensor_type: {tensor_type}")
        
        return self


class PreTrainedTokenizer:
    """
    Base class for all tokenizers.
    """
    
    model_max_length: int = 512
    padding_side: str = "right"
    pad_token: str = "[PAD]"
    pad_token_id: int = 0
    unk_token: str = "[UNK]"
    unk_token_id: int = 100
    cls_token: str = "[CLS]"
    cls_token_id: int = 101
    sep_token: str = "[SEP]"
    sep_token_id: int = 102
    mask_token: str = "[MASK]"
    mask_token_id: int = 103
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
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
        Main method to tokenize and encode text(s).
        """
        # This is a placeholder implementation
        # The actual implementation would call the Rust tokenizer
        
        if isinstance(text, str):
            text = [text]
        
        # Simple mock encoding
        encodings = {
            "input_ids": [[self.cls_token_id, 1, 2, 3, self.sep_token_id] for _ in text],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in text],
        }
        
        if return_token_type_ids is not False:
            encodings["token_type_ids"] = [[0, 0, 0, 0, 0] for _ in text]
        
        batch_encoding = BatchEncoding(encodings)
        
        if return_tensors:
            batch_encoding = batch_encoding.convert_to_tensors(return_tensors)
        
        return batch_encoding
    
    def encode(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> List[int]:
        """
        Encode text(s) to token IDs.
        """
        encodings = self(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        return encodings["input_ids"][0] if isinstance(text, str) else encodings["input_ids"]
    
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        """
        # Placeholder implementation
        if isinstance(token_ids, (int, np.integer)):
            token_ids = [token_ids]
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_tokens = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
            token_ids = [t for t in token_ids if t not in special_tokens]
        
        # Simple mock decoding
        return " ".join([f"token_{t}" for t in token_ids])
    
    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs.
        """
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens, 
                       clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for seq in sequences
        ]
    
    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer to directory.
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer config
        tokenizer_config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            config = {
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
            }
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load tokenizer from pretrained.
        """
        # This would normally load from HF Hub or local directory
        # For now, return a default instance
        return cls(**kwargs)


def tokenizer_from_pretrained(pretrained_model_name_or_path: str, **kwargs) -> PreTrainedTokenizer:
    """
    Load a tokenizer from a pretrained model name or path.
    """
    from . import AutoTokenizer
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)