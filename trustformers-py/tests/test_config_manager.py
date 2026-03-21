"""
Tests for the Configuration Management System
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from trustformers.config_manager import (
    ConfigManager, ConfigMetadata, MigrationRule, 
    ConfigValidationError, ConfigMigrationError, config_manager
)
from trustformers.configuration_bert import BertConfig
from trustformers.configuration_gpt2 import GPT2Config
from trustformers.configuration_llama import LlamaConfig
from trustformers.modeling_utils import PretrainedConfig


class TestConfigManager:
    """Test cases for ConfigManager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config_manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_config(self):
        """Test registering custom configuration"""
        class CustomConfig(PretrainedConfig):
            model_type = "custom"
            
            def __init__(self, custom_param=42, **kwargs):
                super().__init__(**kwargs)
                self.custom_param = custom_param
        
        self.config_manager.register_config("custom", CustomConfig)
        assert "custom" in self.config_manager.custom_configs
        assert self.config_manager.custom_configs["custom"] == CustomConfig
    
    def test_detect_config_type_explicit(self):
        """Test config type detection with explicit model_type"""
        config_data = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12
        }
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        detected_type = self.config_manager.detect_config_type(config_path)
        assert detected_type == "bert"
    
    def test_detect_config_type_heuristic(self):
        """Test config type detection using heuristics"""
        # GPT-2 config without explicit model_type
        config_data = {
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 1024
        }
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        detected_type = self.config_manager.detect_config_type(config_path)
        assert detected_type == "gpt2"
    
    def test_detect_config_type_unknown(self):
        """Test config type detection with unknown configuration"""
        config_data = {"unknown_param": "value"}
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        detected_type = self.config_manager.detect_config_type(config_path)
        assert detected_type is None
    
    def test_validate_config_success(self):
        """Test successful configuration validation"""
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        
        result = self.config_manager.validate_config(config)
        assert result is True
    
    def test_validate_config_failure(self):
        """Test configuration validation failure"""
        config_dict = {
            "model_type": "bert",
            "vocab_size": -1,  # Invalid: negative value
            "hidden_size": "invalid",  # Invalid: wrong type
            "num_hidden_layers": 0,  # Invalid: zero layers
        }
        
        with pytest.raises(ConfigValidationError):
            self.config_manager.validate_config(config_dict, "bert")
    
    def test_migrate_config_field_mapping(self):
        """Test configuration migration with field mapping"""
        config_dict = {
            "model_type": "bert",
            "hidden_dropout_prob": 0.1,
            "pooler_fc_size": 768,
            "vocab_size": 30522
        }
        
        migrated = self.config_manager.migrate_config(
            config_dict, "1.0.0", "1.1.0", "bert"
        )
        
        # Check deprecated field removal
        assert "pooler_fc_size" not in migrated
        
        # Check new field addition
        assert "classifier_dropout" in migrated
        assert migrated["classifier_dropout"] is None
    
    def test_migrate_config_value_transform(self):
        """Test configuration migration with value transformation"""
        config_dict = {
            "model_type": "gpt2",
            "activation_function": "gelu",
            "n_embd": 768
        }
        
        migrated = self.config_manager.migrate_config(
            config_dict, "1.0.0", "1.1.0", "gpt2"
        )
        
        # Check value transformation
        assert migrated["activation_function"] == "gelu_new"
        
        # Check field mapping
        assert "hidden_size" in migrated
        assert migrated["hidden_size"] == 768
    
    def test_load_config_basic(self):
        """Test basic configuration loading"""
        config_data = {
            "model_type": "bert",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072
        }
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_manager.load_config(config_path)
        assert isinstance(config, BertConfig)
        assert config.vocab_size == 30522
        assert config.hidden_size == 768
    
    def test_load_config_with_migration(self):
        """Test configuration loading with automatic migration"""
        config_data = {
            "model_type": "bert",
            "_version": "1.0.0",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "pooler_fc_size": 768
        }
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_manager.load_config(config_path, auto_migrate=True)
        assert isinstance(config, BertConfig)
        
        # Check migration was applied
        config_dict = config.to_dict()
        assert "hidden_dropout_prob" in config_dict  # Should still be there (no field mapping for bert)
        assert "pooler_fc_size" not in config_dict  # Should be removed (deprecated)
        assert "classifier_dropout" in config_dict  # Should be added (new field)
    
    def test_load_config_invalid_file(self):
        """Test loading configuration from invalid file"""
        config_path = Path(self.temp_dir) / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config(config_path)
    
    def test_load_config_invalid_json(self):
        """Test loading configuration from invalid JSON"""
        config_path = Path(self.temp_dir) / "invalid.json"
        with open(config_path, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ValueError):
            self.config_manager.load_config(config_path)
    
    def test_get_config_class_builtin(self):
        """Test getting built-in configuration class"""
        config_class = self.config_manager.get_config_class("bert")
        assert config_class == BertConfig
    
    def test_get_config_class_custom(self):
        """Test getting custom configuration class"""
        class CustomConfig(PretrainedConfig):
            model_type = "custom"
        
        self.config_manager.register_config("custom", CustomConfig)
        config_class = self.config_manager.get_config_class("custom")
        assert config_class == CustomConfig
    
    def test_get_config_class_unknown(self):
        """Test getting unknown configuration class"""
        with pytest.raises(ValueError):
            self.config_manager.get_config_class("unknown")
    
    def test_save_config_with_metadata(self):
        """Test saving configuration with metadata"""
        config = BertConfig(vocab_size=30522, hidden_size=768)
        metadata = ConfigMetadata(
            version="1.1.0",
            model_type="bert",
            custom_fields={"test_field": "test_value"}
        )
        
        save_path = Path(self.temp_dir) / "saved_config"
        self.config_manager.save_config_with_metadata(config, save_path, metadata)
        
        # Check that config was saved
        config_file = save_path / "config.json"
        assert config_file.exists()
        
        # Check metadata was included
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["_version"] == "1.1.0"
        assert saved_data["_metadata"]["model_type"] == "bert"
        assert saved_data["_metadata"]["custom_fields"]["test_field"] == "test_value"
    
    def test_convert_from_huggingface_bert(self):
        """Test converting HuggingFace BERT config"""
        hf_config = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 30522
        }
        
        config_path = Path(self.temp_dir) / "hf_config.json"
        with open(config_path, 'w') as f:
            json.dump(hf_config, f)
        
        config = self.config_manager.convert_from_huggingface(config_path)
        assert isinstance(config, BertConfig)
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
    
    def test_convert_from_huggingface_gpt2(self):
        """Test converting HuggingFace GPT-2 config"""
        hf_config = {
            "model_type": "gpt2",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 1024,
            "vocab_size": 50257
        }
        
        config_path = Path(self.temp_dir) / "hf_config.json"
        with open(config_path, 'w') as f:
            json.dump(hf_config, f)
        
        config = self.config_manager.convert_from_huggingface(config_path)
        assert isinstance(config, GPT2Config)
        assert config.n_embd == 768
        assert config.n_layer == 12
    
    def test_convert_from_huggingface_unsupported(self):
        """Test converting unsupported HuggingFace config"""
        hf_config = {
            "model_type": "unsupported",
            "param1": "value1"
        }
        
        config_path = Path(self.temp_dir) / "hf_config.json"
        with open(config_path, 'w') as f:
            json.dump(hf_config, f)
        
        with pytest.raises(ValueError):
            self.config_manager.convert_from_huggingface(config_path)
    
    def test_create_custom_config(self):
        """Test creating custom configuration class"""
        custom_fields = {
            "custom_param1": 42,
            "custom_param2": "default_value"
        }
        
        CustomConfig = self.config_manager.create_custom_config(
            "custom_model", "bert", custom_fields
        )
        
        # Test instantiation
        config = CustomConfig(vocab_size=30522, custom_param1=100)
        assert config.vocab_size == 30522
        assert config.custom_param1 == 100
        assert config.custom_param2 == "default_value"
        assert config.model_type == "custom_model"
        
        # Test that it was registered
        assert "custom_model" in self.config_manager.custom_configs
    
    def test_version_compatibility(self):
        """Test version compatibility methods"""
        # Test getting version
        version = self.config_manager.get_version_compatibility()
        assert version == "1.1.0"
        
        # Test setting version
        self.config_manager.set_version_compatibility("2.0.0")
        assert self.config_manager.get_version_compatibility() == "2.0.0"
    
    def test_list_supported_models(self):
        """Test listing supported models"""
        models = self.config_manager.list_supported_models()
        assert "bert" in models
        assert "gpt2" in models
        assert "llama" in models
        assert "t5" in models
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = self.config_manager.get_model_info("bert")
        assert info["model_type"] == "bert"
        assert info["config_class"] == "BertConfig"
        assert "default_config" in info
        assert "validation_schema" in info
        assert "migration_rules" in info


class TestConfigMetadata:
    """Test cases for ConfigMetadata"""
    
    def test_config_metadata_creation(self):
        """Test creating ConfigMetadata"""
        metadata = ConfigMetadata(
            version="1.0.0",
            model_type="bert",
            framework="trustformers",
            custom_fields={"test": "value"}
        )
        
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "bert"
        assert metadata.framework == "trustformers"
        assert metadata.custom_fields["test"] == "value"


class TestMigrationRule:
    """Test cases for MigrationRule"""
    
    def test_migration_rule_creation(self):
        """Test creating MigrationRule"""
        rule = MigrationRule(
            from_version="1.0.0",
            to_version="1.1.0",
            field_mappings={"old_field": "new_field"},
            deprecated_fields=["deprecated_field"],
            new_fields={"new_field": "default_value"}
        )
        
        assert rule.from_version == "1.0.0"
        assert rule.to_version == "1.1.0"
        assert rule.field_mappings["old_field"] == "new_field"
        assert "deprecated_field" in rule.deprecated_fields
        assert rule.new_fields["new_field"] == "default_value"


class TestPretrainedConfigEnhancements:
    """Test cases for enhanced PretrainedConfig functionality"""
    
    def test_validate_method(self):
        """Test config validation method"""
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        
        result = config.validate()
        assert result is True
    
    def test_migrate_to_version_method(self):
        """Test config migration method"""
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        
        migrated = config.migrate_to_version("1.1.0")
        assert isinstance(migrated, BertConfig)
    
    def test_get_config_info_method(self):
        """Test config info method"""
        config = BertConfig()
        info = config.get_config_info()
        assert "model_type" in info
        assert "config_class" in info


class TestGlobalConfigManager:
    """Test cases for global config manager instance"""
    
    def test_global_instance_exists(self):
        """Test that global config manager instance exists"""
        assert config_manager is not None
        assert isinstance(config_manager, ConfigManager)
    
    def test_global_instance_functionality(self):
        """Test basic functionality of global instance"""
        models = config_manager.list_supported_models()
        assert len(models) > 0
        assert "bert" in models


if __name__ == "__main__":
    pytest.main([__file__])