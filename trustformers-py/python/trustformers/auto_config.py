"""
Auto Configuration Module for TrustformeRS

Provides convenient functions for automatic configuration detection,
loading, and management without requiring users to directly interact
with the ConfigManager class.
"""

import os
import warnings
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

from .config_manager import config_manager, ConfigMetadata
from .modeling_utils import PretrainedConfig


class AutoConfig:
    """
    Convenient interface for automatic configuration management.
    
    This class provides static methods that wrap the ConfigManager
    functionality for easier use.
    """
    
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: Union[str, Path],
        model_type: Optional[str] = None,
        auto_migrate: bool = True,
        validate: bool = True,
        **kwargs
    ) -> PretrainedConfig:
        """
        Load a configuration from a pretrained model with automatic detection.
        
        Args:
            pretrained_model_name_or_path: Path to the model directory or config file
            model_type: Optional model type if auto-detection should be skipped
            auto_migrate: Whether to automatically migrate old config versions
            validate: Whether to validate the configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            PretrainedConfig: The loaded configuration
            
        Example:
            >>> config = AutoConfig.from_pretrained("path/to/model")
            >>> config = AutoConfig.from_pretrained("path/to/config.json", model_type="bert")
        """
        path = Path(pretrained_model_name_or_path)
        
        # Handle different path types
        if path.is_file() and path.suffix == '.json':
            config_file = path
        elif path.is_dir():
            config_file = path / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"No config.json found in {path}")
        else:
            raise ValueError(f"Invalid path: {path}")
        
        # Load configuration
        config = config_manager.load_config(
            config_file, 
            model_type=model_type, 
            auto_migrate=auto_migrate
        )
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            setattr(config, key, value)
        
        # Validate if requested
        if validate:
            config.validate()
        
        return config
    
    @staticmethod
    def detect_model_type(config_path: Union[str, Path]) -> Optional[str]:
        """
        Automatically detect the model type from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Optional[str]: The detected model type or None if detection failed
            
        Example:
            >>> model_type = AutoConfig.detect_model_type("config.json")
            >>> print(f"Detected model type: {model_type}")
        """
        return config_manager.detect_config_type(config_path)
    
    @staticmethod
    def validate_config(config: Union[PretrainedConfig, Dict[str, Any]], 
                       model_type: Optional[str] = None) -> bool:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration to validate
            model_type: Optional model type for validation
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
            
        Example:
            >>> config = BertConfig()
            >>> is_valid = AutoConfig.validate_config(config)
        """
        return config_manager.validate_config(config, model_type)
    
    @staticmethod
    def migrate_config(config_path: Union[str, Path], 
                      target_version: str,
                      output_path: Optional[Union[str, Path]] = None) -> PretrainedConfig:
        """
        Migrate a configuration file to a target version.
        
        Args:
            config_path: Path to the configuration file
            target_version: Target version to migrate to
            output_path: Optional path to save the migrated config
            
        Returns:
            PretrainedConfig: The migrated configuration
            
        Example:
            >>> migrated = AutoConfig.migrate_config("old_config.json", "1.1.0")
            >>> AutoConfig.migrate_config("config.json", "1.1.0", "new_config.json")
        """
        # Load the current config
        config = config_manager.load_config(config_path, auto_migrate=False)
        
        # Migrate to target version
        migrated = config.migrate_to_version(target_version)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / "config.json"
            
            metadata = ConfigMetadata(
                version=target_version,
                model_type=migrated.model_type,
                migration_applied=f"migrated_to_{target_version}"
            )
            
            config_manager.save_config_with_metadata(
                migrated, output_path.parent, metadata
            )
        
        return migrated
    
    @staticmethod
    def convert_from_huggingface(hf_config_path: Union[str, Path],
                               output_path: Optional[Union[str, Path]] = None) -> PretrainedConfig:
        """
        Convert a HuggingFace configuration to TrustformeRS format.
        
        Args:
            hf_config_path: Path to the HuggingFace configuration
            output_path: Optional path to save the converted config
            
        Returns:
            PretrainedConfig: The converted configuration
            
        Example:
            >>> config = AutoConfig.convert_from_huggingface("hf_config.json")
            >>> AutoConfig.convert_from_huggingface("hf_config.json", "trustformers_config.json")
        """
        config = config_manager.convert_from_huggingface(hf_config_path)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / "config.json"
            
            metadata = ConfigMetadata(
                version=config_manager.get_version_compatibility(),
                model_type=config.model_type,
                framework="trustformers",
                custom_fields={"converted_from": "huggingface"}
            )
            
            config_manager.save_config_with_metadata(
                config, output_path.parent, metadata
            )
        
        return config
    
    @staticmethod
    def create_custom_config(model_type: str,
                           base_model: str = "bert",
                           custom_fields: Optional[Dict[str, Any]] = None,
                           **kwargs) -> PretrainedConfig:
        """
        Create a custom configuration class and instance.
        
        Args:
            model_type: The model type identifier
            base_model: Base model to inherit from
            custom_fields: Dictionary of custom field defaults
            **kwargs: Additional configuration parameters
            
        Returns:
            PretrainedConfig: Instance of the custom configuration
            
        Example:
            >>> config = AutoConfig.create_custom_config(
            ...     "my_model",
            ...     base_model="bert",
            ...     custom_fields={"my_param": 42},
            ...     vocab_size=30000
            ... )
        """
        custom_fields = custom_fields or {}
        
        # Create the custom config class
        CustomConfig = config_manager.create_custom_config(
            model_type, base_model, custom_fields
        )
        
        # Create an instance
        return CustomConfig(**kwargs)
    
    @staticmethod
    def list_supported_models() -> List[str]:
        """
        List all supported model types.
        
        Returns:
            List[str]: List of supported model types
            
        Example:
            >>> models = AutoConfig.list_supported_models()
            >>> print(f"Supported models: {models}")
        """
        return config_manager.list_supported_models()
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a model type.
        
        Args:
            model_type: The model type to get info for
            
        Returns:
            Dict[str, Any]: Model information including defaults and schema
            
        Example:
            >>> info = AutoConfig.get_model_info("bert")
            >>> print(f"Default config: {info['default_config']}")
        """
        return config_manager.get_model_info(model_type)
    
    @staticmethod
    def register_custom_config(model_type: str, config_class: type):
        """
        Register a custom configuration class.
        
        Args:
            model_type: The model type identifier
            config_class: The configuration class to register
            
        Example:
            >>> class MyConfig(PretrainedConfig):
            ...     model_type = "my_model"
            >>> AutoConfig.register_custom_config("my_model", MyConfig)
        """
        config_manager.register_config(model_type, config_class)
    
    @staticmethod
    def get_version_info() -> Dict[str, str]:
        """
        Get version information for the configuration system.
        
        Returns:
            Dict[str, str]: Version information
            
        Example:
            >>> info = AutoConfig.get_version_info()
            >>> print(f"Current version: {info['current_version']}")
        """
        return {
            "current_version": config_manager.get_version_compatibility(),
            "supported_versions": ["1.0.0", "1.1.0"],
            "latest_version": "1.1.0"
        }
    
    @staticmethod
    def save_config(config: PretrainedConfig,
                   save_path: Union[str, Path],
                   include_metadata: bool = True):
        """
        Save a configuration to a file or directory.
        
        Args:
            config: Configuration to save
            save_path: Path to save the configuration
            include_metadata: Whether to include metadata in the saved file
            
        Example:
            >>> config = BertConfig()
            >>> AutoConfig.save_config(config, "saved_config/")
        """
        save_path = Path(save_path)
        
        if include_metadata:
            metadata = ConfigMetadata(
                version=config_manager.get_version_compatibility(),
                model_type=getattr(config, 'model_type', 'unknown'),
                framework="trustformers"
            )
            
            config_manager.save_config_with_metadata(config, save_path, metadata)
        else:
            # Use the standard save method
            if save_path.is_dir() or not save_path.suffix:
                save_path = save_path / "config.json"
            
            config.save_pretrained(str(save_path.parent))
    
    @staticmethod
    def check_compatibility(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Check configuration compatibility with current version.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict[str, Any]: Compatibility information
            
        Example:
            >>> compat = AutoConfig.check_compatibility("config.json")
            >>> if not compat['compatible']:
            ...     print(f"Migration needed: {compat['migration_required']}")
        """
        import json
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {
                "compatible": False,
                "error": str(e),
                "migration_required": False
            }
        
        config_version = config_data.get("_version", "1.0.0")
        current_version = config_manager.get_version_compatibility()
        
        from packaging import version
        
        compatible = version.parse(config_version) >= version.parse(current_version)
        migration_required = version.parse(config_version) < version.parse(current_version)
        
        return {
            "compatible": compatible,
            "config_version": config_version,
            "current_version": current_version,
            "migration_required": migration_required,
            "model_type": config_data.get("model_type", "unknown")
        }


# Convenience functions for common operations
def load_config(path: Union[str, Path], **kwargs) -> PretrainedConfig:
    """Convenience function to load a configuration"""
    return AutoConfig.from_pretrained(path, **kwargs)


def detect_model_type(path: Union[str, Path]) -> Optional[str]:
    """Convenience function to detect model type"""
    return AutoConfig.detect_model_type(path)


def validate_config(config: Union[PretrainedConfig, Dict[str, Any]], 
                   model_type: Optional[str] = None) -> bool:
    """Convenience function to validate configuration"""
    return AutoConfig.validate_config(config, model_type)


def migrate_config(config_path: Union[str, Path], 
                  target_version: str,
                  output_path: Optional[Union[str, Path]] = None) -> PretrainedConfig:
    """Convenience function to migrate configuration"""
    return AutoConfig.migrate_config(config_path, target_version, output_path)


def convert_from_huggingface(hf_config_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None) -> PretrainedConfig:
    """Convenience function to convert from HuggingFace"""
    return AutoConfig.convert_from_huggingface(hf_config_path, output_path)


def create_custom_config(model_type: str,
                        base_model: str = "bert",
                        custom_fields: Optional[Dict[str, Any]] = None,
                        **kwargs) -> PretrainedConfig:
    """Convenience function to create custom configuration"""
    return AutoConfig.create_custom_config(model_type, base_model, custom_fields, **kwargs)


def list_supported_models() -> List[str]:
    """Convenience function to list supported models"""
    return AutoConfig.list_supported_models()


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Convenience function to get model info"""
    return AutoConfig.get_model_info(model_type)


# Make AutoConfig the default export
__all__ = [
    "AutoConfig",
    "load_config",
    "detect_model_type", 
    "validate_config",
    "migrate_config",
    "convert_from_huggingface",
    "create_custom_config",
    "list_supported_models",
    "get_model_info"
]