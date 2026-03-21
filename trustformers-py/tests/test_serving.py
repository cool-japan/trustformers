"""
Tests for the Model Serving infrastructure

This module tests the comprehensive serving functionality including:
- Model loading and management
- Batch processing 
- Load balancing
- A/B testing
- Version management
- FastAPI integration
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List

# Test serving module with optional dependencies
try:
    from trustformers.serving import (
        ModelConfig,
        ModelInstance,
        ModelStatus,
        InferenceRequest,
        InferenceResponse,
        HealthResponse,
        ABTestConfig,
        ModelVersionManager,
        LoadBalancer,
        LoadBalancingStrategy,
        BatchProcessor,
        BatchRequest,
        ABTestManager,
        MetricsCollector,
        ServingManager,
        create_app,
        serve_model,
    )
    SERVING_AVAILABLE = True
except ImportError:
    SERVING_AVAILABLE = False

# Skip all tests if serving dependencies not available
pytestmark = pytest.mark.skipif(not SERVING_AVAILABLE, reason="Serving dependencies not available")

class TestModelConfig:
    """Test ModelConfig functionality"""
    
    def test_model_config_creation(self):
        """Test basic model configuration creation"""
        config = ModelConfig(
            model_id="test-model",
            model_path="/path/to/model",
            version="1.0.0",
            max_batch_size=16,
            device="cpu"
        )
        
        assert config.model_id == "test-model"
        assert config.model_path == "/path/to/model"
        assert config.version == "1.0.0"
        assert config.max_batch_size == 16
        assert config.device == "cpu"
        assert config.weight == 1.0
        assert config.tags == {}
        assert config.metadata == {}
    
    def test_model_config_with_optional_params(self):
        """Test model configuration with optional parameters"""
        tags = {"env": "production", "team": "nlp"}
        metadata = {"description": "Test model", "accuracy": 0.95}
        
        config = ModelConfig(
            model_id="test-model",
            model_path="/path/to/model", 
            version="2.0.0",
            trust_remote_code=True,
            cache_dir="/cache",
            weight=1.5,
            tags=tags,
            metadata=metadata
        )
        
        assert config.trust_remote_code is True
        assert config.cache_dir == "/cache"
        assert config.weight == 1.5
        assert config.tags == tags
        assert config.metadata == metadata

class TestModelVersionManager:
    """Test ModelVersionManager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.version_manager = ModelVersionManager()
        self.config1 = ModelConfig(
            model_id="test-model",
            model_path="/path/to/model",
            version="1.0.0"
        )
        self.config2 = ModelConfig(
            model_id="test-model",
            model_path="/path/to/model",
            version="2.0.0"
        )
    
    def test_register_version(self):
        """Test version registration"""
        assert self.version_manager.register_version(self.config1) is True
        assert self.version_manager.register_version(self.config2) is True
        
        # Try to register same version again
        assert self.version_manager.register_version(self.config1) is False
    
    def test_get_active_version(self):
        """Test getting active version"""
        self.version_manager.register_version(self.config1)
        
        # First version should be active by default
        assert self.version_manager.get_active_version("test-model") == "1.0.0"
        
        # Non-existent model
        assert self.version_manager.get_active_version("non-existent") is None
    
    def test_set_active_version(self):
        """Test setting active version"""
        self.version_manager.register_version(self.config1)
        self.version_manager.register_version(self.config2)
        
        # Change active version
        assert self.version_manager.set_active_version("test-model", "2.0.0") is True
        assert self.version_manager.get_active_version("test-model") == "2.0.0"
        
        # Try to set non-existent version
        assert self.version_manager.set_active_version("test-model", "3.0.0") is False
    
    def test_list_versions(self):
        """Test listing versions"""
        self.version_manager.register_version(self.config1)
        self.version_manager.register_version(self.config2)
        
        versions = self.version_manager.list_versions("test-model")
        assert set(versions) == {"1.0.0", "2.0.0"}
        
        # Non-existent model
        assert self.version_manager.list_versions("non-existent") == []
    
    def test_get_config(self):
        """Test getting configuration"""
        self.version_manager.register_version(self.config1)
        self.version_manager.register_version(self.config2)
        
        # Get specific version
        config = self.version_manager.get_config("test-model", "1.0.0")
        assert config is not None
        assert config.version == "1.0.0"
        
        # Get active version (default)
        config = self.version_manager.get_config("test-model")
        assert config is not None
        assert config.version == "1.0.0"  # First registered is active

class TestLoadBalancer:
    """Test LoadBalancer functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Create mock instances
        self.config1 = ModelConfig(
            model_id="test-model",
            model_path="/path/to/model",
            version="1.0.0",
            weight=1.0
        )
        
        self.config2 = ModelConfig(
            model_id="test-model", 
            model_path="/path/to/model",
            version="1.0.0",
            weight=2.0
        )
        
        self.instance1 = ModelInstance(
            config=self.config1,
            model=MagicMock(),
            tokenizer=MagicMock(),
            status=ModelStatus.READY
        )
        
        self.instance2 = ModelInstance(
            config=self.config2,
            model=MagicMock(),
            tokenizer=MagicMock(),
            status=ModelStatus.READY
        )
    
    def test_add_remove_instance(self):
        """Test adding and removing instances"""
        self.load_balancer.add_instance(self.instance1)
        
        # Should get the instance back
        instance = self.load_balancer.get_instance("test-model", "1.0.0")
        assert instance is self.instance1
        
        # Remove instance
        self.load_balancer.remove_instance("test-model", "1.0.0", self.instance1)
        
        # Should not get instance back
        instance = self.load_balancer.get_instance("test-model", "1.0.0")
        assert instance is None
    
    def test_round_robin_strategy(self):
        """Test round robin load balancing"""
        self.load_balancer.add_instance(self.instance1)
        self.load_balancer.add_instance(self.instance2)
        
        # Should alternate between instances
        instance1 = self.load_balancer.get_instance("test-model", "1.0.0")
        instance2 = self.load_balancer.get_instance("test-model", "1.0.0")
        instance3 = self.load_balancer.get_instance("test-model", "1.0.0")
        
        assert instance1 is not instance2
        assert instance1 is instance3  # Should wrap around
    
    def test_least_loaded_strategy(self):
        """Test least loaded strategy"""
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.LEAST_LOADED)
        self.load_balancer.add_instance(self.instance1)
        self.load_balancer.add_instance(self.instance2)
        
        # Set different request counts
        self.instance1.request_count = 5
        self.instance2.request_count = 2
        
        # Should get instance with fewer requests
        instance = self.load_balancer.get_instance("test-model", "1.0.0")
        assert instance is self.instance2
    
    def test_unhealthy_instances_filtered(self):
        """Test that unhealthy instances are filtered out"""
        self.instance1.status = ModelStatus.ERROR
        self.instance2.status = ModelStatus.READY
        
        self.load_balancer.add_instance(self.instance1)
        self.load_balancer.add_instance(self.instance2)
        
        # Should only get healthy instance
        instance = self.load_balancer.get_instance("test-model", "1.0.0")
        assert instance is self.instance2
    
    def test_update_response_time(self):
        """Test response time tracking"""
        self.load_balancer.add_instance(self.instance1)
        
        # Update response time
        self.load_balancer.update_response_time(self.instance1, 0.5)
        assert self.instance1.avg_response_time == 0.5
        
        # Update again, should calculate average
        self.load_balancer.update_response_time(self.instance1, 1.0)
        assert self.instance1.avg_response_time == 0.75

class TestInferenceRequest:
    """Test InferenceRequest functionality"""
    
    def test_request_creation(self):
        """Test basic request creation"""
        request = InferenceRequest(
            model_id="test-model",
            inputs="Hello, world!",
            parameters={"max_length": 50}
        )
        
        assert request.model_id == "test-model"
        assert request.inputs == "Hello, world!"
        assert request.parameters == {"max_length": 50}
        assert request.priority == 0
        assert request.stream is False
        assert request.timeout == 30.0
        assert request.request_id is not None  # Should be auto-generated
    
    def test_batch_inputs(self):
        """Test request with batch inputs"""
        inputs = ["Hello", "World", "Test"]
        request = InferenceRequest(
            model_id="test-model",
            inputs=inputs
        )
        
        assert request.inputs == inputs
    
    def test_custom_request_id(self):
        """Test request with custom ID"""
        request = InferenceRequest(
            model_id="test-model",
            inputs="Test",
            request_id="custom-id-123"
        )
        
        assert request.request_id == "custom-id-123"
    
    def test_priority_validation(self):
        """Test priority validation"""
        # Valid priority
        request = InferenceRequest(
            model_id="test-model",
            inputs="Test",
            priority=50
        )
        assert request.priority == 50
        
        # Test with boundary values
        request_high = InferenceRequest(
            model_id="test-model",
            inputs="Test",
            priority=100
        )
        assert request_high.priority == 100
        
        request_low = InferenceRequest(
            model_id="test-model",
            inputs="Test",
            priority=-100
        )
        assert request_low.priority == -100

class TestBatchProcessor:
    """Test BatchProcessor functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.batch_processor = BatchProcessor(max_batch_size=4, max_wait_time=0.1)
    
    def teardown_method(self):
        """Cleanup after each test"""
        asyncio.run(self.batch_processor.shutdown())
    
    @pytest.mark.asyncio
    async def test_batch_request_creation(self):
        """Test batch request creation"""
        request = InferenceRequest(
            model_id="test-model",
            inputs="Test input",
            priority=10
        )
        
        batch_request = BatchRequest(request)
        assert batch_request.request is request
        assert batch_request.priority == 10
        assert batch_request.timestamp > 0
        assert isinstance(batch_request.future, asyncio.Future)
    
    @pytest.mark.asyncio
    async def test_batch_request_ordering(self):
        """Test batch request priority ordering"""
        request1 = InferenceRequest(model_id="test", inputs="Test1", priority=1)
        request2 = InferenceRequest(model_id="test", inputs="Test2", priority=10)
        request3 = InferenceRequest(model_id="test", inputs="Test3", priority=5)
        
        batch1 = BatchRequest(request1)
        batch2 = BatchRequest(request2)
        batch3 = BatchRequest(request3)
        
        # Higher priority should come first
        assert batch2 < batch3 < batch1

class TestABTestManager:
    """Test ABTestManager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.ab_manager = ABTestManager()
        self.test_config = ABTestConfig(
            test_name="bert-vs-roberta",
            model_a="bert-base-uncased",
            model_b="roberta-base",
            traffic_split=0.6,
            enabled=True
        )
    
    def test_create_test(self):
        """Test A/B test creation"""
        assert self.ab_manager.create_test(self.test_config) is True
        
        # Try to create same test again
        assert self.ab_manager.create_test(self.test_config) is False
    
    def test_route_request(self):
        """Test request routing"""
        self.ab_manager.create_test(self.test_config)
        
        # Test routing with different request IDs
        results = {}
        for i in range(100):
            model = self.ab_manager.route_request("bert-vs-roberta", f"request-{i}")
            if model:
                results[model] = results.get(model, 0) + 1
        
        # Should have routed to both models
        assert "bert-base-uncased" in results
        assert "roberta-base" in results
        
        # Check approximate split (allowing for randomness)
        total = sum(results.values())
        bert_ratio = results.get("bert-base-uncased", 0) / total
        assert 0.4 <= bert_ratio <= 0.8  # Approximately 60% with some variance
    
    def test_disabled_test(self):
        """Test disabled A/B test"""
        self.test_config.enabled = False
        self.ab_manager.create_test(self.test_config)
        
        # Should not route when disabled
        model = self.ab_manager.route_request("bert-vs-roberta", "test-request")
        assert model is None
    
    def test_record_results(self):
        """Test recording A/B test results"""
        self.ab_manager.create_test(self.test_config)
        
        # Record some results
        self.ab_manager.record_result("bert-vs-roberta", "bert-base-uncased", 0.5)
        self.ab_manager.record_result("bert-vs-roberta", "roberta-base", 0.7)
        self.ab_manager.record_result("bert-vs-roberta", "bert-base-uncased", 0.3, error=True)
        
        results = self.ab_manager.get_test_results("bert-vs-roberta")
        
        assert results["model_a_requests"] == 2
        assert results["model_b_requests"] == 1
        assert results["model_a_errors"] == 1
        assert results["model_b_errors"] == 0
        assert results["model_a_avg_time"] == 0.5  # Only non-error requests
        assert results["model_b_avg_time"] == 0.7
    
    def test_nonexistent_test(self):
        """Test operations on non-existent test"""
        model = self.ab_manager.route_request("nonexistent", "test-request")
        assert model is None
        
        results = self.ab_manager.get_test_results("nonexistent")
        assert results is None

@pytest.mark.skipif(not SERVING_AVAILABLE, reason="Serving dependencies not available")
class TestServingManager:
    """Test ServingManager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Reset singleton
        ServingManager._instance = None
        self.serving_manager = ServingManager.get_instance()
        
        self.config = ModelConfig(
            model_id="test-model",
            model_path="test-path",
            version="1.0.0",
            max_batch_size=4
        )
    
    def teardown_method(self):
        """Cleanup after each test"""
        asyncio.run(self.serving_manager.shutdown())
        ServingManager._instance = None
    
    def test_singleton_pattern(self):
        """Test singleton pattern"""
        manager1 = ServingManager.get_instance()
        manager2 = ServingManager.get_instance()
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_load_model_mock(self):
        """Test model loading with mocking"""
        with patch.object(self.serving_manager, 'instances', {}):
            # Mock the model/tokenizer loading since we don't have real models
            with patch('trustformers.serving.AutoModel') as mock_auto_model, \
                 patch('trustformers.serving.AutoTokenizer') as mock_auto_tokenizer:
                
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_auto_model.from_pretrained.return_value = mock_model
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                
                success = await self.serving_manager.load_model(self.config)
                
                assert success is True
                model_key = f"{self.config.model_id}:{self.config.version}"
                assert model_key in self.serving_manager.instances
                
                instance = self.serving_manager.instances[model_key]
                assert instance.status == ModelStatus.READY
                assert instance.config is self.config
    
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading"""
        # First load a model
        with patch('trustformers.serving.AutoModel'), \
             patch('trustformers.serving.AutoTokenizer'):
            await self.serving_manager.load_model(self.config)
        
        # Then unload it
        success = await self.serving_manager.unload_model("test-model", "1.0.0")
        assert success is True
        
        model_key = f"{self.config.model_id}:{self.config.version}"
        assert model_key not in self.serving_manager.instances
        
        # Try to unload non-existent model
        success = await self.serving_manager.unload_model("nonexistent", "1.0.0")
        assert success is False
    
    def test_get_health(self):
        """Test health status"""
        health = self.serving_manager.get_health()
        
        assert isinstance(health, HealthResponse)
        assert health.status in ["healthy", "unhealthy"]
        assert isinstance(health.models, dict)
        assert isinstance(health.system, dict)
        assert health.timestamp > 0

class TestMetricsCollector:
    """Test MetricsCollector functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.metrics = MetricsCollector()
    
    def test_record_request_no_prometheus(self):
        """Test recording request metrics without Prometheus"""
        # Should not raise error even without Prometheus
        self.metrics.record_request("test-model", "1.0.0", 0.5, "success")
        self.metrics.record_batch_size("test-model", "1.0.0", 8)
        self.metrics.set_model_load_time("test-model", "1.0.0", 2.0)
    
    def test_get_metrics_no_prometheus(self):
        """Test getting metrics without Prometheus"""
        metrics_data = self.metrics.get_metrics()
        assert metrics_data == ""  # Should return empty string without Prometheus

@pytest.mark.skipif(not SERVING_AVAILABLE, reason="Serving dependencies not available")
class TestServingIntegration:
    """Integration tests for serving functionality"""
    
    @pytest.mark.asyncio
    async def test_serve_model_function(self):
        """Test the serve_model convenience function"""
        with patch('trustformers.serving.AutoModel') as mock_auto_model, \
             patch('trustformers.serving.AutoTokenizer') as mock_auto_tokenizer:
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            serving_manager = await serve_model(
                model_path="test-model",
                model_id="custom-id",
                version="2.0.0",
                max_batch_size=16
            )
            
            assert isinstance(serving_manager, ServingManager)
            
            # Check that model was loaded
            model_key = "custom-id:2.0.0"
            assert model_key in serving_manager.instances
            
            instance = serving_manager.instances[model_key]
            assert instance.status == ModelStatus.READY
            assert instance.config.max_batch_size == 16
            
            # Cleanup
            await serving_manager.shutdown()

def test_optional_dependencies_graceful():
    """Test that missing optional dependencies are handled gracefully"""
    # This test verifies that the module can be imported even when optional deps are missing
    # The actual imports are tested at the module level with try/except
    
    # Test that we can create requests even without FastAPI
    request = InferenceRequest(
        model_id="test",
        inputs="test input"
    )
    assert request.model_id == "test"
    
    # Test that we can create configs
    config = ModelConfig(
        model_id="test",
        model_path="test-path",
        version="1.0.0"
    )
    assert config.model_id == "test"

if __name__ == "__main__":
    pytest.main([__file__])