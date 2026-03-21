"""
Tests for TrustformeRS Custom Extensions System
==============================================

Comprehensive test suite for:
- CallbackManager and callback system
- HookRegistry and model lifecycle hooks
- EventBus and component communication
- CustomLayer system and layer registration
- PluginManager and plugin architecture
"""

import asyncio
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the extensions module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

from trustformers.extensions import (
    # Core classes
    CallbackManager, HookRegistry, EventBus, PluginManager,
    CustomLayer, CustomLayerRegistry,
    
    # Configuration classes
    CallbackConfig, EventConfig, ExtensionConfig,
    
    # Event and result classes
    Event, CallbackResult,
    
    # Exception classes
    CallbackError, HookError, EventError, PluginError,
    
    # Example implementations
    LinearLayer, ActivationLayer,
    
    # Decorators and utilities
    callback, hook, event_handler,
    callback_context, hook_context,
    create_custom_layer, register_layer_type,
    configure_extensions,
    
    # Global instances
    callback_manager, hook_registry, event_bus,
    custom_layer_registry, plugin_manager
)


class TestCallbackManager:
    """Test CallbackManager functionality"""
    
    def setup_method(self):
        """Setup fresh callback manager for each test"""
        self.manager = CallbackManager()
        self.test_results = []
    
    def test_callback_registration(self):
        """Test callback registration and unregistration"""
        def test_callback(data):
            self.test_results.append(data)
        
        # Register callback
        callback_id = self.manager.register('test_event', test_callback)
        assert isinstance(callback_id, str)
        assert 'test_event' in callback_id
        
        # Trigger callback
        results = self.manager.trigger('test_event', 'test_data')
        assert len(results) == 1
        assert results[0].success
        assert self.test_results == ['test_data']
        
        # Unregister callback
        success = self.manager.unregister('test_event', test_callback)
        assert success
        
        # Verify callback no longer triggered
        self.test_results.clear()
        results = self.manager.trigger('test_event', 'test_data2')
        assert len(results) == 0
        assert self.test_results == []
    
    def test_callback_priority(self):
        """Test callback execution priority"""
        execution_order = []
        
        def high_priority_callback():
            execution_order.append('high')
        
        def low_priority_callback():
            execution_order.append('low')
        
        # Register with different priorities
        high_config = CallbackConfig(priority=10)
        low_config = CallbackConfig(priority=1)
        
        self.manager.register('priority_test', low_priority_callback, low_config)
        self.manager.register('priority_test', high_priority_callback, high_config)
        
        # Trigger and check order
        self.manager.trigger('priority_test')
        assert execution_order == ['high', 'low']
    
    def test_callback_error_handling(self):
        """Test callback error handling and retries"""
        def failing_callback():
            raise ValueError("Test error")
        
        def successful_callback():
            self.test_results.append('success')
        
        # Test fail_fast behavior
        config = CallbackConfig(fail_fast=True)
        self.manager.register('error_test', failing_callback, config)
        self.manager.register('error_test', successful_callback)
        
        results = self.manager.trigger('error_test')
        assert len(results) == 1  # Should stop after first failure
        assert not results[0].success
        assert isinstance(results[0].error, ValueError)
        assert self.test_results == []  # Successful callback not executed
        
        # Test without fail_fast
        self.manager = CallbackManager()
        config = CallbackConfig(fail_fast=False)
        self.manager.register('error_test', failing_callback, config)
        self.manager.register('error_test', successful_callback)
        
        results = self.manager.trigger('error_test')
        assert len(results) == 2
        assert not results[0].success
        assert results[1].success
        assert self.test_results == ['success']
    
    def test_callback_retries(self):
        """Test callback retry mechanism"""
        attempt_count = 0
        
        def flaky_callback():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Flaky error")
            return "success"
        
        config = CallbackConfig(max_retries=3, retry_delay=0.01)
        self.manager.register('retry_test', flaky_callback, config)
        
        results = self.manager.trigger('retry_test')
        assert len(results) == 1
        assert results[0].success
        assert results[0].result == "success"
        assert attempt_count == 3
    
    def test_callback_timeout(self):
        """Test callback timeout handling"""
        def slow_callback():
            time.sleep(0.2)
            return "slow_result"
        
        config = CallbackConfig(timeout=0.1)
        self.manager.register('timeout_test', slow_callback, config)
        
        results = self.manager.trigger('timeout_test')
        assert len(results) == 1
        assert not results[0].success
        # Note: Timeout behavior depends on execution mode
    
    @pytest.mark.asyncio
    async def test_async_callbacks(self):
        """Test async callback execution"""
        async def async_callback(data):
            await asyncio.sleep(0.01)
            self.test_results.append(f"async_{data}")
        
        def sync_callback(data):
            self.test_results.append(f"sync_{data}")
        
        self.manager.register('async_test', async_callback)
        self.manager.register('async_test', sync_callback)
        
        results = await self.manager.trigger_async('async_test', 'test')
        assert len(results) == 2
        assert all(r.success for r in results)
        assert 'async_test' in self.test_results
        assert 'sync_test' in self.test_results
    
    def test_callback_stats(self):
        """Test callback statistics tracking"""
        def test_callback():
            return "test_result"
        
        self.manager.register('stats_test', test_callback)
        
        # Trigger multiple times
        for _ in range(5):
            self.manager.trigger('stats_test')
        
        stats = self.manager.get_stats('stats_test')
        assert stats['total_calls'] == 5
        assert stats['success_calls'] == 5
        assert stats['failed_calls'] == 0
        assert stats['avg_time'] > 0
        
        # Clear stats
        self.manager.clear_stats()
        stats = self.manager.get_stats('stats_test')
        assert stats == {}


class TestHookRegistry:
    """Test HookRegistry functionality"""
    
    def setup_method(self):
        """Setup fresh hook registry for each test"""
        self.registry = HookRegistry()
        self.hook_results = []
    
    def test_hook_registration(self):
        """Test hook registration and triggering"""
        def before_forward_hook(model_input):
            self.hook_results.append(f"before_{model_input}")
        
        def after_forward_hook(model_input, model_output):
            self.hook_results.append(f"after_{model_input}_{model_output}")
        
        # Register hooks
        hook_id1 = self.registry.register_hook('before_forward', before_forward_hook)
        hook_id2 = self.registry.register_hook('after_forward', after_forward_hook)
        
        assert isinstance(hook_id1, str)
        assert isinstance(hook_id2, str)
        
        # Trigger hooks
        self.registry.trigger_hooks('before_forward', None, 'input_data')
        self.registry.trigger_hooks('after_forward', None, 'input_data', 'output_data')
        
        assert self.hook_results == ['before_input_data', 'after_input_data_output_data']
    
    def test_model_specific_hooks(self):
        """Test model-specific hook registration"""
        def global_hook():
            self.hook_results.append('global')
        
        def model_specific_hook():
            self.hook_results.append('model_specific')
        
        # Register hooks
        self.registry.register_hook('before_forward', global_hook)
        self.registry.register_hook('before_forward', model_specific_hook, model_id='bert-base')
        
        # Trigger global hooks
        self.registry.trigger_hooks('before_forward')
        assert self.hook_results == ['global']
        
        # Trigger model-specific hooks
        self.hook_results.clear()
        self.registry.trigger_hooks('before_forward', model_id='bert-base')
        assert 'global' in self.hook_results
        assert 'model_specific' in self.hook_results
    
    def test_hook_points_validation(self):
        """Test validation of hook points"""
        def test_hook():
            pass
        
        # Valid hook point
        self.registry.register_hook('before_forward', test_hook)
        
        # Invalid hook point
        with pytest.raises(HookError):
            self.registry.register_hook('invalid_hook_point', test_hook)
    
    def test_hook_unregistration(self):
        """Test hook unregistration"""
        def test_hook():
            self.hook_results.append('test')
        
        self.registry.register_hook('before_forward', test_hook)
        self.registry.trigger_hooks('before_forward')
        assert self.hook_results == ['test']
        
        # Unregister hook
        success = self.registry.unregister_hook('before_forward', test_hook)
        assert success
        
        # Verify hook no longer triggered
        self.hook_results.clear()
        self.registry.trigger_hooks('before_forward')
        assert self.hook_results == []
    
    def test_hook_listing(self):
        """Test hook listing functionality"""
        def hook1():
            pass
        
        def hook2():
            pass
        
        self.registry.register_hook('before_forward', hook1)
        self.registry.register_hook('after_forward', hook2, model_id='bert')
        
        # List all hooks
        hooks = self.registry.list_hooks()
        assert 'before_forward' in hooks
        assert 'after_forward' in hooks
        assert any('hook1' in str(h) for h in hooks['before_forward'])
        
        # List model-specific hooks
        model_hooks = self.registry.list_hooks(model_id='bert')
        assert 'after_forward' in model_hooks


class TestEventBus:
    """Test EventBus functionality"""
    
    def setup_method(self):
        """Setup fresh event bus for each test"""
        self.bus = EventBus()
        self.handler_results = []
    
    def test_event_subscription_and_publishing(self):
        """Test basic event subscription and publishing"""
        def test_handler(event):
            self.handler_results.append(f"handled_{event.data}")
        
        # Subscribe to events
        handler_id = self.bus.subscribe('test_event', test_handler)
        assert isinstance(handler_id, str)
        
        # Publish event
        event = Event('test_event', data='test_data')
        results = self.bus.publish(event)
        
        assert len(results) == 1
        assert self.handler_results == ['handled_test_data']
    
    def test_wildcard_subscriptions(self):
        """Test wildcard event subscriptions"""
        def wildcard_handler(event):
            self.handler_results.append(f"wildcard_{event.event_type}")
        
        def specific_handler(event):
            self.handler_results.append(f"specific_{event.event_type}")
        
        # Subscribe with wildcard
        self.bus.subscribe('model.*', wildcard_handler)
        self.bus.subscribe('model.bert.forward', specific_handler)
        
        # Publish matching events
        event1 = Event('model.bert.forward')
        event2 = Event('model.gpt2.forward')
        event3 = Event('tokenizer.encode')
        
        self.bus.publish(event1)
        self.bus.publish(event2)
        self.bus.publish(event3)
        
        # Check results
        expected = [
            'wildcard_model.bert.forward',
            'specific_model.bert.forward',
            'wildcard_model.gpt2.forward'
        ]
        assert sorted(self.handler_results) == sorted(expected)
    
    def test_event_priority(self):
        """Test event handler priority"""
        execution_order = []
        
        def high_priority_handler(event):
            execution_order.append('high')
        
        def low_priority_handler(event):
            execution_order.append('low')
        
        # Subscribe with different priorities
        self.bus.subscribe('priority_event', low_priority_handler, priority=1)
        self.bus.subscribe('priority_event', high_priority_handler, priority=10)
        
        # Publish event
        event = Event('priority_event')
        self.bus.publish(event)
        
        assert execution_order == ['high', 'low']
    
    def test_event_cancellation(self):
        """Test event cancellation mechanism"""
        def canceling_handler(event):
            self.handler_results.append('canceling')
            event.cancel()
        
        def normal_handler(event):
            self.handler_results.append('normal')
        
        # Subscribe handlers (canceling has higher priority)
        self.bus.subscribe('cancel_test', canceling_handler, priority=10)
        self.bus.subscribe('cancel_test', normal_handler, priority=1)
        
        # Publish event
        event = Event('cancel_test')
        self.bus.publish(event)
        
        # Only canceling handler should execute
        assert self.handler_results == ['canceling']
    
    @pytest.mark.asyncio
    async def test_async_event_publishing(self):
        """Test async event publishing"""
        async def async_handler(event):
            await asyncio.sleep(0.01)
            self.handler_results.append(f"async_{event.data}")
        
        def sync_handler(event):
            self.handler_results.append(f"sync_{event.data}")
        
        self.bus.subscribe('async_event', async_handler)
        self.bus.subscribe('async_event', sync_handler)
        
        event = Event('async_event', data='test')
        results = await self.bus.publish_async(event)
        
        assert len(results) == 2
        assert 'async_test' in self.handler_results
        assert 'sync_test' in self.handler_results
    
    def test_event_buffer_and_stats(self):
        """Test event buffer and statistics"""
        def test_handler(event):
            pass
        
        self.bus.subscribe('buffer_test', test_handler)
        
        # Publish multiple events
        for i in range(5):
            event = Event('buffer_test', data=i)
            self.bus.publish(event)
        
        # Check buffer
        recent_events = self.bus.get_recent_events(3)
        assert len(recent_events) == 3
        assert all(isinstance(e, Event) for e in recent_events)
        
        # Check stats
        stats = self.bus.get_stats()
        assert stats['events_published'] == 5
        assert stats['successful_deliveries'] == 5


class TestCustomLayer:
    """Test custom layer system"""
    
    def test_linear_layer(self):
        """Test LinearLayer implementation"""
        layer = LinearLayer(input_size=3, output_size=2, name='test_linear')
        
        # Test configuration
        assert layer.name == 'test_linear'
        assert layer.input_size == 3
        assert layer.output_size == 2
        assert layer.training
        
        # Test forward pass
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        output = layer(input_data)
        
        assert output.shape == (2, 2)  # batch_size x output_size
        assert isinstance(output, np.ndarray)
        
        # Test parameter access
        params = layer.get_parameters()
        assert 'weight' in params
        assert 'bias' in params
        assert params['weight'].shape == (3, 2)
        assert params['bias'].shape == (2,)
    
    def test_activation_layer(self):
        """Test ActivationLayer implementation"""
        relu_layer = ActivationLayer(activation='relu', name='test_relu')
        sigmoid_layer = ActivationLayer(activation='sigmoid')
        
        # Test ReLU
        input_data = np.array([[-1, 0, 1, 2]])
        relu_output = relu_layer(input_data)
        np.testing.assert_array_equal(relu_output, [[0, 0, 1, 2]])
        
        # Test Sigmoid
        sigmoid_output = sigmoid_layer(input_data)
        assert sigmoid_output.shape == input_data.shape
        assert np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1)
        
        # Test invalid activation
        with pytest.raises(ValueError):
            invalid_layer = ActivationLayer(activation='invalid')
            invalid_layer(input_data)
    
    def test_custom_layer_hooks(self):
        """Test custom layer hook system"""
        layer = LinearLayer(input_size=2, output_size=1)
        hook_results = []
        
        def before_hook(layer_instance, inputs, *args, **kwargs):
            hook_results.append('before')
        
        def after_hook(layer_instance, inputs, output, *args, **kwargs):
            hook_results.append('after')
        
        def error_hook(layer_instance, error, inputs, *args, **kwargs):
            hook_results.append('error')
        
        # Register hooks
        layer.register_hook('before_forward', before_hook)
        layer.register_hook('after_forward', after_hook)
        layer.register_hook('on_error', error_hook)
        
        # Test successful forward pass
        input_data = np.array([[1, 2]])
        output = layer(input_data)
        
        assert hook_results == ['before', 'after']
        
        # Test error handling (mock a failing forward)
        hook_results.clear()
        with patch.object(layer, 'forward', side_effect=ValueError("Test error")):
            with pytest.raises(ValueError):
                layer(input_data)
        
        assert 'before' in hook_results
        assert 'error' in hook_results
    
    def test_layer_training_mode(self):
        """Test layer training mode management"""
        layer = LinearLayer(input_size=2, output_size=1)
        
        # Default training mode
        assert layer.training
        
        # Set to eval mode
        layer.eval()
        assert not layer.training
        
        # Set back to training
        layer.train()
        assert layer.training
        
        # Test explicit training mode
        layer.train(False)
        assert not layer.training


class TestCustomLayerRegistry:
    """Test custom layer registry"""
    
    def setup_method(self):
        """Setup fresh registry for each test"""
        self.registry = CustomLayerRegistry()
    
    def test_layer_registration_and_creation(self):
        """Test layer type registration and instance creation"""
        # Built-in layers should be registered
        assert 'linear' in self.registry.list_types()
        assert 'activation' in self.registry.list_types()
        
        # Create layer instances
        linear = self.registry.create('linear', input_size=3, output_size=2)
        assert isinstance(linear, LinearLayer)
        
        activation = self.registry.create('activation', activation='relu')
        assert isinstance(activation, ActivationLayer)
        
        # Test unknown layer type
        with pytest.raises(ValueError):
            self.registry.create('unknown_layer')
    
    def test_custom_layer_registration(self):
        """Test registration of custom layer types"""
        class CustomTestLayer(CustomLayer):
            def forward(self, inputs, *args, **kwargs):
                return inputs * 2
        
        # Register custom layer
        self.registry.register('custom_test', CustomTestLayer)
        assert 'custom_test' in self.registry.list_types()
        
        # Create instance
        custom_layer = self.registry.create('custom_test', name='test_custom')
        assert isinstance(custom_layer, CustomTestLayer)
        assert custom_layer.name == 'test_custom'
    
    def test_layer_instance_management(self):
        """Test layer instance storage and retrieval"""
        # Create layer with ID
        layer = self.registry.create('linear', layer_id='test_layer_1', 
                                   input_size=2, output_size=1)
        
        # Retrieve by ID
        retrieved = self.registry.get_instance('test_layer_1')
        assert retrieved is layer
        
        # Test non-existent ID
        assert self.registry.get_instance('non_existent') is None
        
        # List instances
        instances = self.registry.list_instances()
        assert 'test_layer_1' in instances


class TestPluginSystem:
    """Test plugin management system"""
    
    def setup_method(self):
        """Setup plugin manager with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_dir = Path(self.temp_dir) / "plugins"
        self.plugin_dir.mkdir(exist_ok=True)
        self.manager = PluginManager(self.plugin_dir)
    
    def teardown_method(self):
        """Cleanup temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_plugin(self, name: str, content: str):
        """Helper to create test plugin files"""
        plugin_file = self.plugin_dir / f"{name}.py"
        plugin_file.write_text(content)
        return plugin_file
    
    def test_plugin_discovery(self):
        """Test plugin discovery functionality"""
        # Create test plugin files
        self.create_test_plugin("test_plugin1", """
from trustformers.extensions import PluginInterface

class TestPlugin1(PluginInterface):
    @property
    def name(self):
        return "TestPlugin1"
    
    @property  
    def version(self):
        return "1.0.0"
    
    def initialize(self, config):
        pass
    
    def shutdown(self):
        pass
""")
        
        self.create_test_plugin("test_plugin2", """
from trustformers.extensions import PluginInterface

class TestPlugin2(PluginInterface):
    @property
    def name(self):
        return "TestPlugin2"
    
    @property
    def version(self):
        return "1.0.0"
        
    def initialize(self, config):
        pass
    
    def shutdown(self):
        pass
""")
        
        # Discover plugins
        plugins = self.manager.discover_plugins()
        assert len(plugins) >= 2
        assert "test_plugin1" in plugins
        assert "test_plugin2" in plugins
    
    def test_plugin_loading_and_unloading(self):
        """Test plugin loading and unloading"""
        # Create test plugin
        self.create_test_plugin("loadable_plugin", """
from trustformers.extensions import PluginInterface

class LoadablePlugin(PluginInterface):
    def __init__(self):
        self.initialized = False
        self.shutdown_called = False
    
    @property
    def name(self):
        return "LoadablePlugin"
    
    @property
    def version(self):
        return "1.0.0"
    
    def initialize(self, config):
        self.initialized = True
        self.config = config
    
    def shutdown(self):
        self.shutdown_called = True
""")
        
        # Load plugin
        success = self.manager.load_plugin("loadable_plugin", {"test_config": "value"})
        assert success
        
        # Verify plugin is loaded
        assert "loadable_plugin" in self.manager.list_plugins()
        plugin = self.manager.get_plugin("loadable_plugin")
        assert plugin is not None
        assert plugin.initialized
        assert plugin.config == {"test_config": "value"}
        
        # Get metadata
        metadata = self.manager.get_metadata("loadable_plugin")
        assert metadata is not None
        assert metadata.name == "LoadablePlugin"
        assert metadata.version == "1.0.0"
        
        # Unload plugin
        success = self.manager.unload_plugin("loadable_plugin")
        assert success
        assert plugin.shutdown_called
        assert "loadable_plugin" not in self.manager.list_plugins()
    
    def test_plugin_dependencies(self):
        """Test plugin dependency handling"""
        # Create plugin with dependencies
        self.create_test_plugin("dependent_plugin", """
from trustformers.extensions import PluginInterface

class DependentPlugin(PluginInterface):
    @property
    def name(self):
        return "DependentPlugin"
    
    @property
    def version(self):
        return "1.0.0"
    
    def get_dependencies(self):
        return ["base_plugin"]
    
    def initialize(self, config):
        pass
    
    def shutdown(self):
        pass
""")
        
        # Create base plugin
        self.create_test_plugin("base_plugin", """
from trustformers.extensions import PluginInterface

class BasePlugin(PluginInterface):
    @property
    def name(self):
        return "BasePlugin"
    
    @property
    def version(self):
        return "1.0.0"
    
    def initialize(self, config):
        pass
    
    def shutdown(self):
        pass
""")
        
        # Load plugins
        self.manager.load_plugin("base_plugin")
        self.manager.load_plugin("dependent_plugin")
        
        # Try to unload base plugin (should fail due to dependency)
        success = self.manager.unload_plugin("base_plugin")
        assert not success  # Should fail due to dependent
        
        # Unload dependent first, then base
        self.manager.unload_plugin("dependent_plugin")
        success = self.manager.unload_plugin("base_plugin")
        assert success


class TestDecoratorsAndUtilities:
    """Test decorators and utility functions"""
    
    def setup_method(self):
        """Setup for decorator tests"""
        self.results = []
    
    def test_callback_decorator(self):
        """Test callback decorator"""
        @callback('decorated_event')
        def decorated_callback(data):
            self.results.append(f"decorated_{data}")
        
        # Trigger the event
        callback_manager.trigger('decorated_event', 'test')
        assert 'decorated_test' in self.results
    
    def test_hook_decorator(self):
        """Test hook decorator"""
        @hook('before_forward', priority=5)
        def decorated_hook(inputs):
            self.results.append(f"hook_{inputs}")
        
        # Trigger the hook
        hook_registry.trigger_hooks('before_forward', None, 'test_input')
        assert 'hook_test_input' in self.results
    
    def test_event_handler_decorator(self):
        """Test event handler decorator"""
        @event_handler('decorated_event_type')
        def decorated_handler(event):
            self.results.append(f"handled_{event.data}")
        
        # Publish event
        event = Event('decorated_event_type', data='test_data')
        event_bus.publish(event)
        assert 'handled_test_data' in self.results
    
    def test_callback_context_manager(self):
        """Test callback context manager"""
        def temp_callback(data):
            self.results.append(f"temp_{data}")
        
        # Use context manager
        with callback_context('temp_event', temp_callback):
            callback_manager.trigger('temp_event', 'test')
            assert 'temp_test' in self.results
        
        # Callback should be unregistered after context
        self.results.clear()
        callback_manager.trigger('temp_event', 'test2')
        assert self.results == []
    
    def test_hook_context_manager(self):
        """Test hook context manager"""
        def temp_hook(inputs):
            self.results.append(f"temp_hook_{inputs}")
        
        # Use context manager
        with hook_context('before_forward', temp_hook):
            hook_registry.trigger_hooks('before_forward', None, 'test')
            assert 'temp_hook_test' in self.results
        
        # Hook should be unregistered after context
        self.results.clear()
        hook_registry.trigger_hooks('before_forward', None, 'test2')
        # Should not contain the temp hook result
        temp_results = [r for r in self.results if 'temp_hook' in r]
        assert len(temp_results) == 0
    
    def test_layer_utility_functions(self):
        """Test layer utility functions"""
        # Test create_custom_layer
        layer = create_custom_layer('linear', input_size=2, output_size=1)
        assert isinstance(layer, LinearLayer)
        
        # Test register_layer_type
        class TestUtilityLayer(CustomLayer):
            def forward(self, inputs, *args, **kwargs):
                return inputs
        
        register_layer_type('test_utility', TestUtilityLayer)
        
        # Create instance of registered type
        test_layer = create_custom_layer('test_utility')
        assert isinstance(test_layer, TestUtilityLayer)


class TestConfiguration:
    """Test configuration system"""
    
    def test_extension_config(self):
        """Test ExtensionConfig class"""
        config = ExtensionConfig(
            enable_callbacks=True,
            enable_hooks=False,
            plugin_dir="/tmp/plugins",
            auto_load_plugins=True
        )
        
        assert config.enable_callbacks
        assert not config.enable_hooks
        assert config.plugin_dir == "/tmp/plugins"
        assert config.auto_load_plugins
    
    def test_configure_extensions(self):
        """Test extension configuration function"""
        # Create temporary plugin directory
        temp_dir = tempfile.mkdtemp()
        plugin_dir = Path(temp_dir) / "plugins"
        plugin_dir.mkdir(exist_ok=True)
        
        try:
            config = ExtensionConfig(
                enable_plugins=True,
                plugin_dir=str(plugin_dir),
                auto_load_plugins=False
            )
            
            configure_extensions(config)
            
            # Verify configuration applied
            assert plugin_manager.plugin_dir == plugin_dir
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_workflow(self):
        """Test complete workflow with callbacks, hooks, events, and layers"""
        workflow_results = []
        
        # Create custom layer
        layer = LinearLayer(input_size=2, output_size=1, name='workflow_layer')
        
        # Register callbacks
        def training_callback(step, loss):
            workflow_results.append(f"training_step_{step}_loss_{loss}")
        
        callback_manager.register('training_step', training_callback)
        
        # Register hooks
        def before_forward_hook(inputs):
            workflow_results.append(f"before_forward_{inputs.shape}")
        
        hook_registry.register_hook('before_forward', before_forward_hook)
        
        # Register event handlers
        def model_event_handler(event):
            workflow_results.append(f"model_event_{event.event_type}")
        
        event_bus.subscribe('model.*', model_event_handler)
        
        # Execute workflow
        # 1. Trigger training callback
        callback_manager.trigger('training_step', 1, 0.5)
        
        # 2. Process data through layer (with hooks)
        input_data = np.array([[1, 2]])
        hook_registry.trigger_hooks('before_forward', None, input_data)
        output = layer(input_data)
        
        # 3. Publish model events
        event_bus.publish(Event('model.forward.complete', data={'output_shape': output.shape}))
        
        # Verify all components worked
        assert len(workflow_results) >= 3
        assert any('training_step' in r for r in workflow_results)
        assert any('before_forward' in r for r in workflow_results)
        assert any('model_event' in r for r in workflow_results)
    
    def test_error_propagation(self):
        """Test error handling across components"""
        error_log = []
        
        # Create failing callback
        def failing_callback():
            raise ValueError("Callback failed")
        
        # Create error handler
        def error_handler(event):
            if 'error' in event.event_type:
                error_log.append(f"handled_error_{event.data}")
        
        callback_manager.register('error_test', failing_callback, 
                                CallbackConfig(fail_fast=False))
        event_bus.subscribe('system.error', error_handler)
        
        # Trigger failing callback
        results = callback_manager.trigger('error_test')
        
        # Publish error event
        error_event = Event('system.error', data='callback_failure')
        event_bus.publish(error_event)
        
        # Verify error handling
        assert len(results) == 1
        assert not results[0].success
        assert 'handled_error_callback_failure' in error_log


if __name__ == '__main__':
    pytest.main([__file__, '-v'])