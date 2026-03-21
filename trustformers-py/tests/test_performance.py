"""
Tests for the performance optimization module
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

# Test performance module with optional dependencies
try:
    from trustformers.performance import (
        BatchProcessor,
        ObjectPool,
        LazyLoader,
        MemoryPool,
        CallBatcher,
        PrewarmingCache,
        PerformanceMonitor,
        batched,
        lazy_init,
        pooled,
        monitored,
        batch_process,
        get_from_pool,
        return_to_pool,
        prewarm_cache,
        get_performance_stats,
        optimize_memory,
        cleanup_performance_resources,
    )
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Skip all tests if performance module not available
pytestmark = pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance module not available")

class TestBatchProcessor:
    """Test BatchProcessor functionality"""
    
    def test_batch_processor_creation(self):
        """Test basic batch processor creation"""
        processor = BatchProcessor(batch_size=4, max_workers=2)
        assert processor.batch_size == 4
        assert processor.max_workers == 2
        assert processor.adaptive_sizing is True
        
    def test_batch_processor_with_custom_settings(self):
        """Test batch processor with custom settings"""
        processor = BatchProcessor(
            batch_size=8,
            max_workers=4,
            adaptive_sizing=False,
            timeout=30.0
        )
        assert processor.batch_size == 8
        assert processor.max_workers == 4
        assert processor.adaptive_sizing is False
        assert processor.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test actual batch processing"""
        processor = BatchProcessor(batch_size=3, max_workers=1)
        
        def dummy_function(items):
            """Simple function for testing"""
            return [f"processed_{item}" for item in items]
        
        # Process items in batches
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(items, dummy_function)
        
        expected = ["processed_1", "processed_2", "processed_3", "processed_4", "processed_5"]
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self):
        """Test processing empty batch"""
        processor = BatchProcessor(batch_size=3)
        
        def dummy_function(items):
            return [f"processed_{item}" for item in items]
        
        results = await processor.process_batch([], dummy_function)
        assert results == []
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing functionality"""
        processor = BatchProcessor(batch_size=4, adaptive_sizing=True)
        
        # Test performance tracking
        processor.update_performance(processing_time=0.1, batch_size=4)
        processor.update_performance(processing_time=0.05, batch_size=8)
        
        # Should adjust batch size based on performance
        optimal_size = processor.get_optimal_batch_size()
        assert isinstance(optimal_size, int)
        assert optimal_size > 0

class TestObjectPool:
    """Test ObjectPool functionality"""
    
    def test_object_pool_creation(self):
        """Test basic object pool creation"""
        def factory():
            return {"value": 0}
        
        pool = ObjectPool(factory, max_size=5)
        assert pool.max_size == 5
        assert pool.current_size == 0
        
    def test_object_pool_get_return(self):
        """Test getting and returning objects"""
        def factory():
            return MagicMock()
        
        pool = ObjectPool(factory, max_size=3)
        
        # Get objects
        obj1 = pool.get()
        obj2 = pool.get()
        
        assert obj1 is not None
        assert obj2 is not None
        assert obj1 is not obj2
        
        # Return objects
        pool.return_object(obj1)
        pool.return_object(obj2)
        
        # Get again - should reuse
        obj3 = pool.get()
        assert obj3 is obj1 or obj3 is obj2
    
    def test_object_pool_reset_function(self):
        """Test object reset functionality"""
        def factory():
            obj = MagicMock()
            obj.reset = MagicMock()
            return obj
        
        def reset_func(obj):
            obj.reset()
        
        pool = ObjectPool(factory, max_size=2, reset_func=reset_func)
        
        obj = pool.get()
        pool.return_object(obj)
        
        # Reset should have been called
        obj.reset.assert_called_once()

class TestLazyLoader:
    """Test LazyLoader functionality"""
    
    def test_lazy_loader_creation(self):
        """Test basic lazy loader creation"""
        def expensive_operation():
            time.sleep(0.01)  # Simulate expensive operation
            return "expensive_result"
        
        loader = LazyLoader(expensive_operation)
        assert not loader.is_loaded
        
    def test_lazy_loading(self):
        """Test actual lazy loading"""
        call_count = 0
        
        def expensive_operation():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"
        
        loader = LazyLoader(expensive_operation)
        
        # First access should trigger loading
        result1 = loader.get()
        assert result1 == "result_1"
        assert loader.is_loaded
        assert call_count == 1
        
        # Second access should return cached result
        result2 = loader.get()
        assert result2 == "result_1"
        assert call_count == 1  # Should not call again
    
    def test_lazy_loader_with_args(self):
        """Test lazy loader with arguments"""
        def operation_with_args(x, y, z=None):
            return f"result_{x}_{y}_{z}"
        
        loader = LazyLoader(operation_with_args, args=(1, 2), kwargs={"z": 3})
        result = loader.get()
        assert result == "result_1_2_3"

class TestMemoryPool:
    """Test MemoryPool functionality"""
    
    def test_memory_pool_creation(self):
        """Test basic memory pool creation"""
        pool = MemoryPool(initial_size=10, growth_factor=1.5)
        assert pool.initial_size == 10
        assert pool.growth_factor == 1.5
        
    def test_memory_allocation(self):
        """Test memory allocation from pool"""
        pool = MemoryPool(initial_size=100)
        
        # Allocate memory
        mem1 = pool.allocate(50)
        mem2 = pool.allocate(30)
        
        assert mem1 is not None
        assert mem2 is not None
        assert len(mem1) == 50
        assert len(mem2) == 30
    
    def test_memory_deallocation(self):
        """Test memory deallocation"""
        pool = MemoryPool(initial_size=100)
        
        mem = pool.allocate(50)
        pool.deallocate(mem)
        
        # Should be able to reuse memory
        mem2 = pool.allocate(40)
        assert mem2 is not None

class TestCallBatcher:
    """Test CallBatcher functionality"""
    
    def test_call_batcher_creation(self):
        """Test basic call batcher creation"""
        def batch_function(calls):
            return [f"result_{call}" for call in calls]
        
        batcher = CallBatcher(batch_function, max_batch_size=5, max_wait_time=0.1)
        assert batcher.max_batch_size == 5
        assert batcher.max_wait_time == 0.1
    
    @pytest.mark.asyncio
    async def test_call_batching(self):
        """Test actual call batching"""
        def batch_function(calls):
            return [f"processed_{call}" for call in calls]
        
        batcher = CallBatcher(batch_function, max_batch_size=3, max_wait_time=0.1)
        
        # Make multiple calls
        tasks = [
            batcher.call("item1"),
            batcher.call("item2"),
            batcher.call("item3")
        ]
        
        results = await asyncio.gather(*tasks)
        expected = ["processed_item1", "processed_item2", "processed_item3"]
        assert results == expected

class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""
    
    def test_performance_monitor_creation(self):
        """Test basic performance monitor creation"""
        monitor = PerformanceMonitor()
        assert monitor.enabled is True
        
    def test_operation_timing(self):
        """Test operation timing"""
        monitor = PerformanceMonitor()
        
        with monitor.time_operation("test_op"):
            time.sleep(0.01)
        
        stats = monitor.get_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        assert stats["test_op"]["total_time"] > 0
    
    def test_memory_tracking(self):
        """Test memory usage tracking"""
        monitor = PerformanceMonitor()
        
        initial_memory = monitor.get_memory_usage()
        
        # Create some objects to use memory
        large_list = [i for i in range(1000)]
        
        final_memory = monitor.get_memory_usage()
        
        # Memory usage should have increased
        assert final_memory >= initial_memory
        
        # Cleanup
        del large_list
    
    def test_custom_metrics(self):
        """Test custom metrics tracking"""
        monitor = PerformanceMonitor()
        
        monitor.record_metric("custom_metric", 42)
        monitor.record_metric("custom_metric", 58)
        
        stats = monitor.get_stats()
        assert "custom_metric" in stats
        assert stats["custom_metric"]["count"] == 2
        assert stats["custom_metric"]["total"] == 100

class TestPrewarmingCache:
    """Test PrewarmingCache functionality"""
    
    def test_prewarming_cache_creation(self):
        """Test basic prewarming cache creation"""
        cache = PrewarmingCache(max_size=100)
        assert cache.max_size == 100
        
    def test_cache_operations(self):
        """Test basic cache operations"""
        cache = PrewarmingCache(max_size=10)
        
        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_prewarming(self):
        """Test cache prewarming functionality"""
        cache = PrewarmingCache(max_size=10)
        
        def expensive_function(key):
            time.sleep(0.001)  # Simulate expensive operation
            return f"computed_{key}"
        
        # Prewarm cache
        keys_to_prewarm = ["key1", "key2", "key3"]
        await cache.prewarm(keys_to_prewarm, expensive_function)
        
        # Values should be available immediately
        assert cache.get("key1") == "computed_key1"
        assert cache.get("key2") == "computed_key2"
        assert cache.get("key3") == "computed_key3"

class TestDecorators:
    """Test performance decorators"""
    
    def test_batched_decorator(self):
        """Test batched decorator"""
        @batched(batch_size=3)
        def process_items(items):
            return [f"processed_{item}" for item in items]
        
        # This should work but we can't easily test async behavior
        # in synchronous tests without complex setup
        assert process_items is not None
    
    def test_lazy_init_decorator(self):
        """Test lazy_init decorator"""
        init_count = 0
        
        @lazy_init
        def expensive_initialization():
            nonlocal init_count
            init_count += 1
            return "initialized"
        
        # First call should initialize
        result1 = expensive_initialization()
        assert result1 == "initialized"
        assert init_count == 1
        
        # Second call should return cached result
        result2 = expensive_initialization()
        assert result2 == "initialized"
        assert init_count == 1
    
    def test_pooled_decorator(self):
        """Test pooled decorator"""
        @pooled(max_pool_size=5)
        def create_resource():
            return MagicMock()
        
        # Test that decorator works
        resource = create_resource()
        assert resource is not None
    
    def test_monitored_decorator(self):
        """Test monitored decorator"""
        @monitored
        def monitored_function(x, y):
            return x + y
        
        result = monitored_function(2, 3)
        assert result == 5
        
        # Check that monitoring data exists
        stats = get_performance_stats()
        assert isinstance(stats, dict)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_batch_process_function(self):
        """Test batch_process utility function"""
        def simple_processor(items):
            return [item * 2 for item in items]
        
        items = [1, 2, 3, 4, 5]
        results = batch_process(items, simple_processor, batch_size=3)
        
        # Should process all items
        assert len(results) == 5
        assert results[0] == 2
        assert results[4] == 10
    
    def test_pool_functions(self):
        """Test get_from_pool and return_to_pool functions"""
        # These functions should work with the global pool
        obj = get_from_pool("test_pool")
        assert obj is not None
        
        return_to_pool("test_pool", obj)
        
        # Should be able to get it back
        obj2 = get_from_pool("test_pool")
        assert obj2 is obj  # Should be the same object
    
    def test_prewarm_cache_function(self):
        """Test prewarm_cache utility function"""
        def loader(key):
            return f"value_for_{key}"
        
        keys = ["a", "b", "c"]
        prewarm_cache(keys, loader, cache_name="test_cache")
        
        # This should complete without error
        # Detailed testing would require access to the cache internals
    
    def test_get_performance_stats(self):
        """Test get_performance_stats function"""
        stats = get_performance_stats()
        assert isinstance(stats, dict)
        
        # Stats should contain timing and memory information
        assert "timing" in stats or "memory" in stats or len(stats) == 0
    
    def test_optimize_memory(self):
        """Test optimize_memory function"""
        # This should run without error
        freed_memory = optimize_memory()
        assert isinstance(freed_memory, (int, float))
        assert freed_memory >= 0
    
    def test_cleanup_performance_resources(self):
        """Test cleanup_performance_resources function"""
        # This should run without error
        cleanup_performance_resources()
        
        # After cleanup, stats should be reset
        stats = get_performance_stats()
        # Stats might be empty or contain minimal data after cleanup

class TestPerformanceIntegration:
    """Integration tests for performance module"""
    
    def test_combined_performance_features(self):
        """Test using multiple performance features together"""
        # Create a batch processor with monitoring
        processor = BatchProcessor(batch_size=4, max_workers=2)
        monitor = PerformanceMonitor()
        
        def process_with_monitoring(items):
            with monitor.time_operation("batch_processing"):
                return [f"processed_{item}" for item in items]
        
        # This should work together
        items = list(range(10))
        
        # Note: We can't easily test async without proper async setup
        # But we can test that the components work together
        assert processor is not None
        assert monitor is not None
    
    def test_performance_optimization_workflow(self):
        """Test a complete performance optimization workflow"""
        # Initialize performance monitoring
        monitor = PerformanceMonitor()
        
        # Create object pool for expensive objects
        def expensive_factory():
            return MagicMock()
        
        pool = ObjectPool(expensive_factory, max_size=5)
        
        # Use lazy loading for initialization
        @lazy_init
        def get_expensive_resource():
            return "expensive_resource"
        
        # Test the workflow
        with monitor.time_operation("workflow"):
            # Get object from pool
            obj = pool.get()
            
            # Use lazy-loaded resource
            resource = get_expensive_resource()
            
            # Return object to pool
            pool.return_object(obj)
        
        # Check that everything worked
        stats = monitor.get_stats()
        assert "workflow" in stats
        assert resource == "expensive_resource"

if __name__ == "__main__":
    pytest.main([__file__])