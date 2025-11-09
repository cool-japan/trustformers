"""
Configuration Management System for TrustformeRS

Provides advanced configuration management with auto detection,
migration utilities, custom config support, and validation.
"""

import json
import os
import warnings
from typing import Dict, Any, Optional, Union, Type, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re
from packaging import version

from .modeling_utils import PretrainedConfig
from .configuration_bert import BertConfig
from .configuration_gpt2 import GPT2Config
from .configuration_llama import LlamaConfig
from .configuration_t5 import T5Config


@dataclass
class ConfigMetadata:
    """Metadata for configuration files"""
    version: str
    model_type: str
    framework: str = "trustformers"
    created_at: Optional[str] = None
    migration_applied: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationRule:
    """Configuration migration rule"""
    from_version: str
    to_version: str
    field_mappings: Dict[str, str] = field(default_factory=dict)
    value_transforms: Dict[str, callable] = field(default_factory=dict)
    deprecated_fields: List[str] = field(default_factory=list)
    new_fields: Dict[str, Any] = field(default_factory=dict)


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


class ConfigMigrationError(Exception):
    """Configuration migration error"""
    pass


class ConfigManager:
    """Advanced configuration management system"""
    
    # Registry of configuration classes
    CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
        "bert": BertConfig,
        "gpt2": GPT2Config,
        "llama": LlamaConfig,
        "t5": T5Config,
    }
    
    # Migration rules registry
    MIGRATION_RULES: Dict[str, List[MigrationRule]] = {
        "bert": [
            MigrationRule(
                from_version="1.0.0",
                to_version="1.1.0",
                field_mappings={},  # No field mappings for bert
                deprecated_fields=["pooler_fc_size"],
                new_fields={"classifier_dropout": None}
            ),
        ],
        "gpt2": [
            MigrationRule(
                from_version="1.0.0",
                to_version="1.1.0",
                field_mappings={"n_embd": "hidden_size", "n_layer": "num_layers"},
                value_transforms={
                    "activation_function": lambda x: "gelu_new" if x == "gelu" else x
                }
            ),
        ],
        "llama": [
            MigrationRule(
                from_version="1.0.0",
                to_version="1.1.0",
                new_fields={"rope_theta": 10000.0, "rope_scaling": None},
                deprecated_fields=["rotary_dim"]
            ),
        ],
    }
    
    # Configuration validation schemas
    VALIDATION_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "bert": {
            "vocab_size": {"type": int, "min": 1},
            "hidden_size": {"type": int, "min": 1},
            "num_hidden_layers": {"type": int, "min": 1},
            "num_attention_heads": {"type": int, "min": 1},
            "intermediate_size": {"type": int, "min": 1},
            "max_position_embeddings": {"type": int, "min": 1},
            "layer_norm_eps": {"type": float, "min": 1e-12},
        },
        "gpt2": {
            "vocab_size": {"type": int, "min": 1},
            "n_embd": {"type": int, "min": 1},
            "n_layer": {"type": int, "min": 1},
            "n_head": {"type": int, "min": 1},
            "n_positions": {"type": int, "min": 1},
        },
        "llama": {
            "vocab_size": {"type": int, "min": 1},
            "hidden_size": {"type": int, "min": 1},
            "num_hidden_layers": {"type": int, "min": 1},
            "num_attention_heads": {"type": int, "min": 1},
            "intermediate_size": {"type": int, "min": 1},
        },
    }
    
    def __init__(self):
        self.custom_configs: Dict[str, Type[PretrainedConfig]] = {}
        self.version_compatibility = "1.1.0"
    
    def register_config(self, model_type: str, config_class: Type[PretrainedConfig]):
        """Register a custom configuration class"""
        self.custom_configs[model_type] = config_class
        
    def detect_config_type(self, config_path: Union[str, Path]) -> Optional[str]:
        """Automatically detect configuration type from file"""
        config_path = Path(config_path)
        
        # Try to load the config file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
            
        # Check for explicit model_type
        if "model_type" in config_data:
            return config_data["model_type"]
            
        # Heuristic detection based on field patterns
        field_patterns = {
            "bert": ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"],
            "gpt2": ["n_embd", "n_layer", "n_head", "n_positions"],
            "llama": ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "rms_norm_eps"],
            "t5": ["d_model", "d_ff", "num_layers", "num_heads"],
        }
        
        scores = {}
        for model_type, required_fields in field_patterns.items():
            score = sum(1 for field in required_fields if field in config_data)
            if score > 0:
                scores[model_type] = score / len(required_fields)
        
        if scores:
            return max(scores, key=scores.get)
            
        return None
    
    def validate_config(self, config: Union[PretrainedConfig, Dict[str, Any]], 
                       model_type: Optional[str] = None) -> bool:
        """Validate configuration against schema"""
        if isinstance(config, PretrainedConfig):
            config_dict = config.to_dict()
            model_type = model_type or getattr(config, 'model_type', None)
        else:
            config_dict = config
            model_type = model_type or config_dict.get('model_type')
            
        if not model_type or model_type not in self.VALIDATION_SCHEMAS:
            warnings.warn(f"No validation schema for model type: {model_type}")
            return True
            
        schema = self.VALIDATION_SCHEMAS[model_type]
        errors = []
        
        for field, rules in schema.items():
            if field in config_dict:
                value = config_dict[field]
                
                # Type check
                if "type" in rules and not isinstance(value, rules["type"]):
                    errors.append(f"Field '{field}' must be of type {rules['type'].__name__}")
                
                # Range checks (only for numeric types)
                if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
                    errors.append(f"Field '{field}' must be >= {rules['min']}")
                    
                if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
                    errors.append(f"Field '{field}' must be <= {rules['max']}")
                    
                # Choice validation
                if "choices" in rules and value not in rules["choices"]:
                    errors.append(f"Field '{field}' must be one of {rules['choices']}")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
            
        return True
    
    def migrate_config(self, config_dict: Dict[str, Any], 
                      from_version: str, to_version: str,
                      model_type: str) -> Dict[str, Any]:
        """Migrate configuration from one version to another"""
        if model_type not in self.MIGRATION_RULES:
            warnings.warn(f"No migration rules for model type: {model_type}")
            return config_dict
            
        rules = self.MIGRATION_RULES[model_type]
        migrated_config = config_dict.copy()
        
        # Apply migration rules in sequence
        for rule in rules:
            if (version.parse(from_version) >= version.parse(rule.from_version) and
                version.parse(to_version) >= version.parse(rule.to_version)):
                
                # Apply field mappings
                for old_field, new_field in rule.field_mappings.items():
                    if old_field in migrated_config:
                        migrated_config[new_field] = migrated_config.pop(old_field)
                
                # Apply value transforms
                for field, transform in rule.value_transforms.items():
                    if field in migrated_config:
                        try:
                            migrated_config[field] = transform(migrated_config[field])
                        except Exception as e:
                            raise ConfigMigrationError(f"Failed to transform field '{field}': {e}")
                
                # Remove deprecated fields
                for field in rule.deprecated_fields:
                    if field in migrated_config:
                        warnings.warn(f"Field '{field}' is deprecated and will be removed")
                        migrated_config.pop(field)
                
                # Add new fields
                for field, default_value in rule.new_fields.items():
                    if field not in migrated_config:
                        migrated_config[field] = default_value
        
        return migrated_config
    
    def load_config(self, config_path: Union[str, Path], 
                   model_type: Optional[str] = None,
                   auto_migrate: bool = True) -> PretrainedConfig:
        """Load configuration with auto detection and migration"""
        config_path = Path(config_path)
        
        # Load configuration file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
            
        # Auto-detect model type if not provided
        if model_type is None:
            model_type = self.detect_config_type(config_path)
            if model_type is None:
                raise ValueError("Could not detect configuration type")
        
        # Check for version compatibility
        config_version = config_data.get("_version", "1.0.0")
        if auto_migrate and version.parse(config_version) < version.parse(self.version_compatibility):
            config_data = self.migrate_config(
                config_data, config_version, self.version_compatibility, model_type
            )
            config_data["_version"] = self.version_compatibility
            
        # Get config class
        config_class = self.get_config_class(model_type)
        
        # Create configuration instance
        config = config_class.from_dict(config_data)
        
        # Validate configuration
        self.validate_config(config, model_type)
        
        return config
    
    def get_config_class(self, model_type: str) -> Type[PretrainedConfig]:
        """Get configuration class for model type"""
        if model_type in self.custom_configs:
            return self.custom_configs[model_type]
        elif model_type in self.CONFIG_REGISTRY:
            return self.CONFIG_REGISTRY[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def save_config_with_metadata(self, config: PretrainedConfig, 
                                 save_path: Union[str, Path],
                                 metadata: Optional[ConfigMetadata] = None):
        """Save configuration with metadata"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = ConfigMetadata(
                version=self.version_compatibility,
                model_type=getattr(config, 'model_type', 'unknown')
            )
        
        # Save config with metadata
        config_dict = config.to_dict()
        config_dict["_version"] = metadata.version
        config_dict["_metadata"] = {
            "model_type": metadata.model_type,
            "framework": metadata.framework,
            "created_at": metadata.created_at,
            "migration_applied": metadata.migration_applied,
            "custom_fields": metadata.custom_fields
        }
        
        config_file = save_path / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
    
    def convert_from_huggingface(self, hf_config_path: Union[str, Path]) -> PretrainedConfig:
        """Convert HuggingFace config to TrustformeRS format"""
        hf_config_path = Path(hf_config_path)
        
        with open(hf_config_path, 'r', encoding='utf-8') as f:
            hf_config = json.load(f)
        
        # Detect model type
        model_type = hf_config.get("model_type", "").lower()
        
        # Convert field names if needed
        conversion_maps = {
            "bert": {
                "hidden_size": "hidden_size",
                "num_hidden_layers": "num_hidden_layers",
                "num_attention_heads": "num_attention_heads",
                "intermediate_size": "intermediate_size",
            },
            "gpt2": {
                "n_embd": "n_embd",
                "n_layer": "n_layer", 
                "n_head": "n_head",
                "n_positions": "n_positions",
            },
            "llama": {
                "hidden_size": "hidden_size",
                "num_hidden_layers": "num_hidden_layers",
                "num_attention_heads": "num_attention_heads",
                "intermediate_size": "intermediate_size",
            },
        }
        
        if model_type in conversion_maps:
            converted_config = {}
            for hf_field, tf_field in conversion_maps[model_type].items():
                if hf_field in hf_config:
                    converted_config[tf_field] = hf_config[hf_field]
            
            # Add remaining fields
            for key, value in hf_config.items():
                if key not in conversion_maps[model_type]:
                    converted_config[key] = value
            
            config_class = self.get_config_class(model_type)
            return config_class.from_dict(converted_config)
        else:
            raise ValueError(f"Unsupported HuggingFace model type: {model_type}")
    
    def create_custom_config(self, model_type: str, base_config: str = "bert", 
                           custom_fields: Dict[str, Any] = None) -> Type[PretrainedConfig]:
        """Create a custom configuration class"""
        base_class = self.get_config_class(base_config)
        custom_fields = custom_fields or {}
        
        # Create a custom config class dynamically
        def __init__(self, **kwargs):
            # Add custom fields with defaults
            for field, default_value in custom_fields.items():
                kwargs.setdefault(field, default_value)
            super(CustomConfig, self).__init__(**kwargs)
            
        # Create the class dynamically
        CustomConfig = type(
            f"CustomConfig_{model_type}",
            (base_class,),
            {
                "model_type": model_type,
                "__init__": __init__
            }
        )
        
        # Register the custom config
        self.register_config(model_type, CustomConfig)
        
        return CustomConfig
    
    def get_version_compatibility(self) -> str:
        """Get current version compatibility"""
        return self.version_compatibility
    
    def set_version_compatibility(self, version_str: str):
        """Set version compatibility"""
        self.version_compatibility = version_str
    
    def list_supported_models(self) -> List[str]:
        """List all supported model types"""
        return list(self.CONFIG_REGISTRY.keys()) + list(self.custom_configs.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a model type"""
        config_class = self.get_config_class(model_type)
        
        # Get default config
        default_config = config_class()
        
        return {
            "model_type": model_type,
            "config_class": config_class.__name__,
            "default_config": default_config.to_dict(),
            "validation_schema": self.VALIDATION_SCHEMAS.get(model_type, {}),
            "migration_rules": len(self.MIGRATION_RULES.get(model_type, [])),
        }


# Global config manager instance
config_manager = ConfigManager()