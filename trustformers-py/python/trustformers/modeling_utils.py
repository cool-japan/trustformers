"""
Modeling utilities for TrustformeRS

Provides compatibility classes and utilities for model outputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import warnings

import numpy as np


class ModelOutput:
    """
    Base class for all model outputs as dataclass. Provides methods to easily 
    convert to tuples or dictionaries.
    """
    
    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/fields that are not None.
        """
        return tuple(self[k] for k in self.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert self to a dictionary containing all the attributes/fields.
        """
        return {k: v for k, v in self.items()}
    
    def __getitem__(self, k):
        """Get item like a dict."""
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return self.to_tuple()[k]
    
    def __setitem__(self, k, v):
        """Set item like a dict."""
        setattr(self, k, v)
    
    def keys(self):
        """Get keys/attributes."""
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    
    def values(self):
        """Get values."""
        return [self.__dict__[k] for k in self.keys()]
    
    def items(self):
        """Get items."""
        return [(k, self.__dict__[k]) for k in self.keys()]


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    
    Args:
        last_hidden_state: Sequence of hidden-states at the output of the last layer
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    last_hidden_state: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs with pooling.
    
    Args:
        last_hidden_state: Sequence of hidden-states at the output of the last layer
        pooler_output: Last layer hidden-state after pooling
        hidden_states: Hidden-states of the model at the output of each layer  
        attentions: Attention weights for each layer
    """
    last_hidden_state: np.ndarray = None
    pooler_output: Optional[np.ndarray] = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model outputs.
    
    Args:
        loss: Language modeling loss (optional, returned when labels are provided)
        logits: Prediction scores of the language modeling head
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    loss: Optional[np.ndarray] = None
    logits: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass  
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language model outputs.
    
    Args:
        loss: Masked language modeling loss (optional, returned when labels are provided)
        logits: Prediction scores of the language modeling head
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    loss: Optional[np.ndarray] = None
    logits: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    
    Args:
        loss: Classification loss (optional, returned when labels are provided)
        logits: Classification scores (before SoftMax)
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    loss: Optional[np.ndarray] = None
    logits: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    
    Args:
        loss: Classification loss (optional, returned when labels are provided)
        logits: Classification scores (before SoftMax)
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    loss: Optional[np.ndarray] = None
    logits: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.
    
    Args:
        loss: Total span extraction loss (optional, returned when labels are provided)
        start_logits: Span-start scores (before SoftMax)
        end_logits: Span-end scores (before SoftMax)
        hidden_states: Hidden-states of the model at the output of each layer
        attentions: Attention weights for each layer
    """
    loss: Optional[np.ndarray] = None
    start_logits: np.ndarray = None
    end_logits: np.ndarray = None
    hidden_states: Optional[Tuple[np.ndarray]] = None
    attentions: Optional[Tuple[np.ndarray]] = None


class PretrainedConfig:
    """
    Base class for all configuration classes.
    """
    
    model_type: str = ""
    is_composition: bool = False
    
    def __init__(self, **kwargs):
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        """
        output = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            output[key] = value
        return output
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a config from a Python dictionary of parameters.
        """
        config = cls(**config_dict)
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config
    
    def save_pretrained(self, save_directory: str):
        """
        Save a configuration object to a directory.
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "PretrainedConfig":
        """
        Instantiate a config from a pretrained configuration with auto-detection and validation.
        """
        import os
        import json
        
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            # This would normally download from HF Hub
            # For now, we'll use default configs
            config_file = None
        
        if config_file and os.path.exists(config_file):
            # Use the advanced config manager for loading
            try:
                from .config_manager import config_manager
                return config_manager.load_config(config_file, **kwargs)
            except ImportError:
                # Fallback to basic loading if config_manager is not available
                with open(config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                return cls.from_dict(config_dict, **kwargs)
        else:
            # Return default config
            warnings.warn(f"Could not find config for {pretrained_model_name_or_path}, using defaults")
            return cls(**kwargs)
    
    def validate(self) -> bool:
        """
        Validate the configuration using the config manager.
        """
        try:
            from .config_manager import config_manager
            return config_manager.validate_config(self)
        except ImportError:
            warnings.warn("Config manager not available, skipping validation")
            return True
    
    def migrate_to_version(self, target_version: str) -> "PretrainedConfig":
        """
        Migrate configuration to a target version.
        """
        try:
            from .config_manager import config_manager
            current_version = getattr(self, '_version', '1.0.0')
            migrated_dict = config_manager.migrate_config(
                self.to_dict(), current_version, target_version, self.model_type
            )
            return self.__class__.from_dict(migrated_dict)
        except ImportError:
            warnings.warn("Config manager not available, skipping migration")
            return self
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about this configuration.
        """
        try:
            from .config_manager import config_manager
            return config_manager.get_model_info(self.model_type)
        except ImportError:
            return {"model_type": self.model_type, "config_class": self.__class__.__name__}