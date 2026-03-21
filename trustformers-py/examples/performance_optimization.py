"""
Performance optimization examples for TrustformeRS Python bindings

This example demonstrates various performance optimization techniques
including parallel processing, batching, caching, and memory management.
"""

import time
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import TrustformeRS performance utilities
from trustformers import (
    # Core components
    AutoModel, AutoTokenizer, pipeline,
    
    # Performance optimization
    BatchProcessor, BatchConfig, ObjectPool, LazyLoader,
    MemoryPool, CallBatcher, PrewarmingCache, PerformanceMonitor,
    ParallelTokenizer, ParallelInferenceEngine, AsyncInferenceEngine,
    
    # Decorators
    batched, lazy_init, pooled, monitored,
    
    # Utility functions
    batch_process, get_from_pool, return_to_pool,
    prewarm_cache, get_performance_stats, optimize_memory,
    cleanup_performance_resources,
    
    # Cache system
    get_model_cache, cache_model, get_cached_model,
    clear_all_caches, get_cache_stats,
)


def example_batch_processing():
    """Demonstrate efficient batch processing."""
    print("=== Batch Processing Example ===")
    
    # Create batch configuration
    config = BatchConfig(
        batch_size=16,
        max_workers=4,
        auto_batch_size=True,
        memory_limit_mb=1024
    )
    
    # Create batch processor
    processor = BatchProcessor(config)
    
    # Simulate tokenization function
    def tokenize_batch(texts: List[str]) -> List[Dict[str, Any]]:
        """Simulate batch tokenization."""
        time.sleep(0.01)  # Simulate processing time
        return [{"input_ids": list(range(len(text))), "text": text} for text in texts]
    
    # Create test data
    texts = [f"This is test sentence number {i}" for i in range(100)]
    
    # Process with batch processor
    start_time = time.time()
    results = processor.process_batch(texts, tokenize_batch, "tokenization")
    batch_time = time.time() - start_time
    
    # Process sequentially for comparison
    start_time = time.time()
    sequential_results = []
    for text in texts:
        result = tokenize_batch([text])
        sequential_results.extend(result)
    sequential_time = time.time() - start_time
    
    print(f"  Batch processing time: {batch_time:.3f}s")
    print(f"  Sequential processing time: {sequential_time:.3f}s")
    print(f"  Speedup: {sequential_time / batch_time:.2f}x")
    print(f"  Processed {len(results)} items")
    
    # Cleanup
    processor.close()
    print()


def example_parallel_tokenization():
    """Demonstrate parallel tokenization for high throughput."""
    print("=== Parallel Tokenization Example ===")
    
    # Create tokenizer (mock for this example)
    class MockTokenizer:
        def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
            time.sleep(0.001)  # Simulate tokenization time
            return {
                "input_ids": list(range(len(text) % 20 + 1)),
                "attention_mask": [1] * (len(text) % 20 + 1)
            }
    
    tokenizer = MockTokenizer()
    
    # Create parallel tokenizer
    parallel_tokenizer = ParallelTokenizer(tokenizer, max_workers=4)
    
    # Create test texts
    texts = [f"Test text number {i} with varying lengths" * (i % 3 + 1) for i in range(200)]
    
    # Tokenize in parallel
    start_time = time.time()
    results = parallel_tokenizer.tokenize_batch(texts)
    parallel_time = time.time() - start_time
    
    # Get statistics
    stats = parallel_tokenizer.get_stats()
    
    print(f"  Parallel tokenization time: {parallel_time:.3f}s")
    print(f"  Processed {len(results)} texts")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Average throughput: {stats['avg_throughput']:.2f} tokens/s")
    print(f"  Total texts: {stats['total_texts']}")
    
    # Cleanup
    parallel_tokenizer.close()
    print()


def example_parallel_inference():
    """Demonstrate parallel inference for high throughput."""
    print("=== Parallel Inference Example ===")
    
    # Create mock model
    class MockModel:
        def __call__(self, **kwargs) -> Dict[str, Any]:
            time.sleep(0.005)  # Simulate inference time
            batch_size = len(kwargs.get("input_ids", []))
            return {
                "logits": np.random.randn(batch_size, 10).tolist(),
                "hidden_states": np.random.randn(batch_size, 768).tolist()
            }
    
    model = MockModel()
    
    # Create parallel inference engine
    inference_engine = ParallelInferenceEngine(model, max_workers=4, batch_size=8)
    
    # Create test inputs
    inputs = [
        {"input_ids": list(range(20)), "attention_mask": [1] * 20}
        for _ in range(50)
    ]
    
    # Run parallel inference
    start_time = time.time()
    results = inference_engine.inference_batch(inputs)
    inference_time = time.time() - start_time
    
    # Get statistics
    stats = inference_engine.get_stats()
    
    print(f"  Parallel inference time: {inference_time:.3f}s")
    print(f"  Processed {len(results)} inputs")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Average latency: {stats['avg_latency']:.3f}s")
    print(f"  Average throughput: {stats['avg_throughput']:.2f} inferences/s")
    
    # Cleanup
    inference_engine.close()
    print()


def example_object_pooling():
    """Demonstrate object pooling for resource efficiency."""
    print("=== Object Pooling Example ===")
    
    # Create expensive object factory
    class ExpensiveObject:
        def __init__(self):
            self.created_at = time.time()
            self.usage_count = 0
            time.sleep(0.01)  # Simulate expensive initialization
        
        def use(self):
            self.usage_count += 1
            return f"Used {self.usage_count} times"
        
        def reset(self):
            self.usage_count = 0
    
    # Create object pool
    pool = ObjectPool(ExpensiveObject, max_size=10)
    
    # Use objects from pool
    start_time = time.time()
    for _ in range(50):
        obj = pool.acquire()
        result = obj.use()
        pool.release(obj)
    pooled_time = time.time() - start_time
    
    # Create objects directly for comparison
    start_time = time.time()
    for _ in range(50):
        obj = ExpensiveObject()
        result = obj.use()
    direct_time = time.time() - start_time
    
    # Get pool statistics
    stats = pool.stats()
    
    print(f"  Pooled creation time: {pooled_time:.3f}s")
    print(f"  Direct creation time: {direct_time:.3f}s")
    print(f"  Speedup: {direct_time / pooled_time:.2f}x")
    print(f"  Pool stats: {stats}")
    
    # Cleanup
    pool.clear()
    print()


def example_lazy_loading():
    """Demonstrate lazy loading for memory efficiency."""
    print("=== Lazy Loading Example ===")
    
    # Create expensive resource loader
    @lazy_init
    def load_large_model():
        print("    Loading large model...")
        time.sleep(0.1)  # Simulate model loading
        return {"model_size": "7B", "parameters": 7000000000}
    
    # Create multiple lazy loaders
    loaders = [load_large_model for _ in range(5)]
    
    # Check if loaded
    print(f"  Models loaded initially: {sum(loader.is_loaded() for loader in loaders)}")
    
    # Use some models
    start_time = time.time()
    model1 = loaders[0]()
    model2 = loaders[1]()
    load_time = time.time() - start_time
    
    print(f"  Models loaded after usage: {sum(loader.is_loaded() for loader in loaders)}")
    print(f"  Load time: {load_time:.3f}s")
    print(f"  Model 1: {model1}")
    
    # Reset a loader
    loaders[0].reset()
    print(f"  Models loaded after reset: {sum(loader.is_loaded() for loader in loaders)}")
    print()


def example_prewarming_cache():
    """Demonstrate prewarming cache for performance."""
    print("=== Prewarming Cache Example ===")
    
    # Create expensive computation
    def expensive_computation(key: str) -> str:
        time.sleep(0.05)  # Simulate computation
        return f"Result for {key}"
    
    # Create cache
    cache = PrewarmingCache(max_size=100)
    
    # Measure cache performance
    keys = [f"key_{i}" for i in range(20)]
    
    # First access (cold cache)
    start_time = time.time()
    results1 = []
    for key in keys:
        result = cache.get(key, lambda: expensive_computation(key))
        results1.append(result)
    cold_time = time.time() - start_time
    
    # Second access (warm cache)
    start_time = time.time()
    results2 = []
    for key in keys:
        result = cache.get(key, lambda: expensive_computation(key))
        results2.append(result)
    warm_time = time.time() - start_time
    
    print(f"  Cold cache time: {cold_time:.3f}s")
    print(f"  Warm cache time: {warm_time:.3f}s")
    print(f"  Speedup: {cold_time / warm_time:.2f}x")
    print(f"  Cache hit rate: {(len(keys) - len(set(keys))) / len(keys) * 100:.1f}%")
    
    # Cleanup
    cache.close()
    print()


def example_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("=== Performance Monitoring Example ===")
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Monitor different operations
    @monitored("fast_operation")
    def fast_operation():
        time.sleep(0.001)
        return "fast"
    
    @monitored("slow_operation")
    def slow_operation():
        time.sleep(0.05)
        return "slow"
    
    @monitored("variable_operation")
    def variable_operation(duration: float):
        time.sleep(duration)
        return f"took {duration}s"
    
    # Run operations
    for _ in range(10):
        fast_operation()
        slow_operation()
        variable_operation(np.random.uniform(0.01, 0.1))
    
    # Get performance report
    report = monitor.get_report()
    
    print("  Performance Report:")
    for op_name, stats in report.items():
        print(f"    {op_name}:")
        print(f"      Count: {stats['count']}")
        print(f"      Avg time: {stats['avg_time']*1000:.2f}ms")
        print(f"      Min time: {stats['min_time']*1000:.2f}ms")
        print(f"      Max time: {stats['max_time']*1000:.2f}ms")
        print(f"      Total time: {stats['total_time']*1000:.2f}ms")
    
    # Reset monitor
    monitor.reset()
    print()


def example_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("=== Memory Optimization Example ===")
    
    # Create memory pool for tensors
    common_shapes = [(100, 768), (50, 768), (200, 768)]
    memory_pool = MemoryPool(common_shapes, max_tensors_per_size=5)
    
    # Simulate tensor usage
    tensors = []
    for i in range(20):
        shape = common_shapes[i % len(common_shapes)]
        
        # Try to get from pool
        tensor = memory_pool.get_tensor(shape)
        if tensor is None:
            # Create new tensor
            tensor = np.random.randn(*shape).astype(np.float32)
        
        tensors.append(tensor)
    
    # Return tensors to pool
    for tensor in tensors:
        memory_pool.return_tensor(tensor)
    
    # Get pool statistics
    stats = memory_pool.get_stats()
    
    print(f"  Memory pool stats: {stats}")
    print(f"  Reuse rate: {stats['reuse_rate']:.2f}")
    
    # Run memory optimization
    print("  Running memory optimization...")
    optimize_memory()
    
    # Clear pool
    memory_pool.clear()
    print()


def example_batched_decorator():
    """Demonstrate batched function decorator."""
    print("=== Batched Function Decorator Example ===")
    
    @batched
    def process_text(text: str) -> str:
        """Process a single text."""
        time.sleep(0.001)  # Simulate processing
        return f"Processed: {text}"
    
    # Test single calls
    single_result = process_text("Hello")
    print(f"  Single call result: {single_result}")
    
    # Test batch calls (this would be handled by CallBatcher in real scenario)
    texts = ["Hello", "World", "Test"]
    batch_args = [(text,) for text in texts]
    batch_kwargs = [{}] * len(texts)
    
    # Note: This is a simplified example - real batching would be handled automatically
    print(f"  Batch processing would handle {len(texts)} texts automatically")
    print()


def example_global_performance_stats():
    """Demonstrate global performance statistics."""
    print("=== Global Performance Stats Example ===")
    
    # Get global performance stats
    stats = get_performance_stats()
    
    print("  Global Performance Statistics:")
    if stats:
        for operation, metrics in stats.items():
            print(f"    {operation}: {metrics}")
    else:
        print("    No performance data available yet")
    
    # Get cache statistics
    cache_stats = get_cache_stats()
    print(f"  Cache Statistics: {cache_stats}")
    print()


def benchmark_performance_features():
    """Benchmark performance features."""
    print("=== Performance Features Benchmark ===")
    
    # Create test data
    test_data = [f"Test string {i}" for i in range(1000)]
    
    def simple_process(items):
        return [item.upper() for item in items]
    
    # Benchmark batch processing
    batch_config = BatchConfig(batch_size=50, max_workers=4)
    processor = BatchProcessor(batch_config)
    
    start_time = time.time()
    batch_results = processor.process_batch(test_data, simple_process, "uppercase")
    batch_time = time.time() - start_time
    
    # Benchmark sequential processing
    start_time = time.time()
    sequential_results = simple_process(test_data)
    sequential_time = time.time() - start_time
    
    print(f"  Batch processing time: {batch_time:.3f}s")
    print(f"  Sequential processing time: {sequential_time:.3f}s")
    print(f"  Speedup: {sequential_time / batch_time:.2f}x")
    print(f"  Results match: {batch_results == sequential_results}")
    
    # Cleanup
    processor.close()
    print()


def main():
    """Run all performance optimization examples."""
    print("TrustformeRS Performance Optimization Examples")
    print("=" * 55)
    print()
    
    try:
        example_batch_processing()
        example_parallel_tokenization()
        example_parallel_inference()
        example_object_pooling()
        example_lazy_loading()
        example_prewarming_cache()
        example_performance_monitoring()
        example_memory_optimization()
        example_batched_decorator()
        example_global_performance_stats()
        benchmark_performance_features()
        
        print("All performance optimization examples completed successfully!")
        
    except Exception as e:
        print(f"Error in performance examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup performance resources
        cleanup_performance_resources()
        print("\nCleaned up performance resources")


if __name__ == "__main__":
    main()