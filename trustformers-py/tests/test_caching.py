"""
Tests for the caching system module
"""

import pytest
import time
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

# Test caching module
try:
    from trustformers.caching import (
        ModelCache,
        TokenizerCache,
        ResultCache,
        DiskCache,
        LRUCache,
        get_model_cache,
        get_tokenizer_cache,
        get_result_cache,
        memoize,
        cache_model,
        get_cached_model,
        cache_tokenizer,
        get_cached_tokenizer,
        clear_all_caches,
        get_cache_stats,
    )
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Skip all tests if caching module not available
pytestmark = pytest.mark.skipif(not CACHING_AVAILABLE, reason="Caching module not available")

class TestLRUCache:
    """Test LRUCache functionality"""
    
    def test_lru_cache_creation(self):
        """Test basic LRU cache creation"""
        cache = LRUCache(max_size=100)
        assert cache.max_size == 100
        assert cache.current_size == 0
        assert len(cache) == 0
    
    def test_lru_cache_with_ttl(self):
        """Test LRU cache with TTL"""
        cache = LRUCache(max_size=10, ttl=1.0)  # 1 second TTL
        assert cache.ttl == 1.0
    
    def test_basic_get_put(self):
        """Test basic get and put operations"""
        cache = LRUCache(max_size=5)
        
        # Put some items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = LRUCache(max_size=3)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2") 
        cache.put("key3", "value3")
        
        # Add one more - should evict key1 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_lru_access_ordering(self):
        """Test that access updates LRU ordering"""
        cache = LRUCache(max_size=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key4 - should evict key2 (now least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = LRUCache(max_size=10, ttl=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        assert cache.get("key1") is None
    
    def test_cache_update(self):
        """Test updating existing keys"""
        cache = LRUCache(max_size=5)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Update value
        cache.put("key1", "new_value1")
        assert cache.get("key1") == "new_value1"
        assert len(cache) == 1  # Should still be one item
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=5)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        
        # Add item and access
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = LRUCache(max_size=5)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert len(cache) == 2
        
        cache.clear()
        assert len(cache) == 0
        assert cache.get("key1") is None

class TestDiskCache:
    """Test DiskCache functionality"""
    
    def test_disk_cache_creation(self):
        """Test basic disk cache creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir, max_size_mb=100)
            assert cache.cache_dir == Path(tmpdir)
            assert cache.max_size_mb == 100
    
    def test_disk_cache_put_get(self):
        """Test basic put and get operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir)
            
            # Put and get simple data
            cache.put("key1", {"data": "value1"})
            result = cache.get("key1")
            assert result == {"data": "value1"}
    
    def test_disk_cache_binary_data(self):
        """Test caching binary data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir)
            
            # Put binary data
            binary_data = b"binary content"
            cache.put("binary_key", binary_data)
            
            result = cache.get("binary_key")
            assert result == binary_data
    
    def test_disk_cache_metadata(self):
        """Test disk cache metadata tracking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir)
            
            # Put data with metadata
            cache.put("key1", "value1", metadata={"size": 10, "type": "string"})
            
            # Get metadata
            metadata = cache.get_metadata("key1")
            assert metadata["size"] == 10
            assert metadata["type"] == "string"
            assert "timestamp" in metadata
    
    def test_disk_cache_size_limit(self):
        """Test disk cache size limiting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir, max_size_mb=0.001)  # Very small limit
            
            # Try to put large data
            large_data = "x" * 10000  # 10KB
            cache.put("large_key", large_data)
            
            # Should handle size limiting gracefully
            assert isinstance(cache.get_stats(), dict)
    
    def test_disk_cache_cleanup(self):
        """Test disk cache cleanup functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=tmpdir, ttl=0.1)  # 100ms TTL
            
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            
            # Wait for expiration
            time.sleep(0.15)
            
            # Run cleanup
            cache.cleanup()
            
            # Item should be cleaned up
            assert cache.get("key1") is None

class TestModelCache:
    """Test ModelCache functionality"""
    
    def test_model_cache_creation(self):
        """Test basic model cache creation"""
        cache = ModelCache(max_memory_mb=100, max_disk_mb=500)
        assert cache.max_memory_mb == 100
        assert cache.max_disk_mb == 500
    
    def test_model_cache_put_get(self):
        """Test model caching"""
        cache = ModelCache(max_memory_mb=100)
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        
        # Cache model
        cache.put("test-model", mock_model)
        
        # Retrieve model
        retrieved = cache.get("test-model")
        assert retrieved is mock_model
    
    def test_model_cache_memory_fallback(self):
        """Test memory to disk fallback"""
        cache = ModelCache(max_memory_mb=0.001, max_disk_mb=100)  # Very small memory
        
        # Create mock models
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        
        cache.put("model1", mock_model1)
        cache.put("model2", mock_model2)
        
        # Should handle fallback gracefully
        stats = cache.get_stats()
        assert isinstance(stats, dict)
    
    def test_model_cache_weak_references(self):
        """Test weak reference handling"""
        cache = ModelCache()
        
        # Create object that can be garbage collected
        class TestModel:
            def __init__(self, name):
                self.name = name
        
        model = TestModel("test")
        cache.put("test-model", model)
        
        # Should retrieve the model
        assert cache.get("test-model") is model
        
        # Delete reference
        del model
        
        # Weak reference should be cleaned up (implementation dependent)
        # This test mainly ensures no errors occur

class TestTokenizerCache:
    """Test TokenizerCache functionality"""
    
    def test_tokenizer_cache_creation(self):
        """Test basic tokenizer cache creation"""
        cache = TokenizerCache(max_size=50, ttl=3600)
        assert cache.max_size == 50
        assert cache.ttl == 3600
    
    def test_tokenizer_cache_operations(self):
        """Test tokenizer caching operations"""
        cache = TokenizerCache(max_size=10)
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.name = "test-tokenizer"
        
        # Cache tokenizer
        cache.put("test-tokenizer", mock_tokenizer)
        
        # Retrieve tokenizer
        retrieved = cache.get("test-tokenizer")
        assert retrieved is mock_tokenizer
    
    def test_tokenizer_cache_with_config(self):
        """Test tokenizer caching with configuration"""
        cache = TokenizerCache()
        
        mock_tokenizer = MagicMock()
        config = {"vocab_size": 30000, "do_lower_case": True}
        
        cache.put("tokenizer", mock_tokenizer, config=config)
        
        # Should be able to retrieve with config
        retrieved_config = cache.get_config("tokenizer")
        assert retrieved_config == config

class TestResultCache:
    """Test ResultCache functionality"""
    
    def test_result_cache_creation(self):
        """Test basic result cache creation"""
        cache = ResultCache(max_size=100)
        assert cache.max_size == 100
    
    def test_result_cache_operations(self):
        """Test result caching operations"""
        cache = ResultCache(max_size=10)
        
        # Cache function result
        result = {"output": "test result", "score": 0.95}
        cache.put("function_key", result)
        
        # Retrieve result
        retrieved = cache.get("function_key")
        assert retrieved == result
    
    def test_result_cache_key_generation(self):
        """Test automatic key generation"""
        cache = ResultCache()
        
        # Generate key from function arguments
        key = cache.generate_key("function_name", args=(1, 2), kwargs={"param": "value"})
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Same arguments should generate same key
        key2 = cache.generate_key("function_name", args=(1, 2), kwargs={"param": "value"})
        assert key == key2
        
        # Different arguments should generate different key
        key3 = cache.generate_key("function_name", args=(1, 3), kwargs={"param": "value"})
        assert key != key3

class TestMemoizeDecorator:
    """Test memoize decorator"""
    
    def test_memoize_basic(self):
        """Test basic memoization"""
        call_count = 0
        
        @memoize(max_size=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Call with different args
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_memoize_with_kwargs(self):
        """Test memoization with keyword arguments"""
        call_count = 0
        
        @memoize()
        def function_with_kwargs(x, y=1, z=2):
            nonlocal call_count
            call_count += 1
            return x + y + z
        
        # Calls with same effective arguments
        result1 = function_with_kwargs(1, y=2, z=3)
        result2 = function_with_kwargs(1, 2, 3)
        
        assert result1 == result2 == 6
        assert call_count == 1  # Should be cached
    
    def test_memoize_cache_stats(self):
        """Test memoize cache statistics"""
        @memoize(max_size=5)
        def test_function(x):
            return x * 2
        
        # Make some calls
        test_function(1)
        test_function(1)  # Cache hit
        test_function(2)
        test_function(3)
        test_function(1)  # Cache hit
        
        # Check if function has cache stats (implementation dependent)
        if hasattr(test_function, 'cache_info'):
            info = test_function.cache_info()
            assert info['hits'] >= 2
            assert info['misses'] >= 3

class TestCacheUtilities:
    """Test cache utility functions"""
    
    def test_get_cache_functions(self):
        """Test get_*_cache functions"""
        model_cache = get_model_cache()
        tokenizer_cache = get_tokenizer_cache()
        result_cache = get_result_cache()
        
        assert isinstance(model_cache, ModelCache)
        assert isinstance(tokenizer_cache, TokenizerCache)
        assert isinstance(result_cache, ResultCache)
        
        # Should return same instances on repeated calls
        assert get_model_cache() is model_cache
        assert get_tokenizer_cache() is tokenizer_cache
        assert get_result_cache() is result_cache
    
    def test_cache_model_functions(self):
        """Test cache_model and get_cached_model functions"""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        
        # Cache model
        success = cache_model("test-model", mock_model)
        assert success is True
        
        # Retrieve cached model
        retrieved = get_cached_model("test-model")
        assert retrieved is mock_model
        
        # Try to get non-existent model
        assert get_cached_model("non-existent") is None
    
    def test_cache_tokenizer_functions(self):
        """Test cache_tokenizer and get_cached_tokenizer functions"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.name = "test-tokenizer"
        
        # Cache tokenizer
        success = cache_tokenizer("test-tokenizer", mock_tokenizer)
        assert success is True
        
        # Retrieve cached tokenizer
        retrieved = get_cached_tokenizer("test-tokenizer")
        assert retrieved is mock_tokenizer
        
        # Try to get non-existent tokenizer
        assert get_cached_tokenizer("non-existent") is None
    
    def test_clear_all_caches(self):
        """Test clear_all_caches function"""
        # Add some items to caches
        cache_model("model1", MagicMock())
        cache_tokenizer("tokenizer1", MagicMock())
        
        # Clear all caches
        clear_all_caches()
        
        # Should be empty now
        assert get_cached_model("model1") is None
        assert get_cached_tokenizer("tokenizer1") is None
    
    def test_get_cache_stats(self):
        """Test get_cache_stats function"""
        stats = get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "model_cache" in stats
        assert "tokenizer_cache" in stats
        assert "result_cache" in stats
        
        # Each cache should have stats
        for cache_name, cache_stats in stats.items():
            assert isinstance(cache_stats, dict)

class TestCacheIntegration:
    """Integration tests for caching system"""
    
    def test_multi_level_caching(self):
        """Test using multiple cache levels together"""
        # Use both memory and disk caching
        model_cache = ModelCache(max_memory_mb=1, max_disk_mb=10)
        
        # Create multiple mock models
        models = [MagicMock() for _ in range(5)]
        for i, model in enumerate(models):
            model.name = f"model_{i}"
            model_cache.put(f"model_{i}", model)
        
        # Should be able to retrieve all models
        for i in range(5):
            retrieved = model_cache.get(f"model_{i}")
            assert retrieved is not None
    
    def test_cache_performance(self):
        """Test cache performance characteristics"""
        cache = LRUCache(max_size=1000)
        
        # Time cache operations
        start_time = time.time()
        
        # Put many items
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}")
        
        put_time = time.time() - start_time
        
        # Get many items
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        
        get_time = time.time() - start_time
        
        # Operations should be reasonably fast
        assert put_time < 1.0  # Less than 1 second for 100 puts
        assert get_time < 1.0  # Less than 1 second for 100 gets
    
    def test_cache_thread_safety(self):
        """Test cache thread safety (basic test)"""
        import threading
        
        cache = LRUCache(max_size=100)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    assert retrieved == value
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
    
    def test_cache_memory_management(self):
        """Test cache memory management"""
        cache = LRUCache(max_size=5)
        
        # Fill cache beyond capacity
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Should only have 5 items (max_size)
        assert len(cache) == 5
        
        # Should have the most recent items
        for i in range(5, 10):
            assert cache.get(f"key_{i}") == f"value_{i}"
        
        # Earlier items should be evicted
        for i in range(5):
            assert cache.get(f"key_{i}") is None

if __name__ == "__main__":
    pytest.main([__file__])