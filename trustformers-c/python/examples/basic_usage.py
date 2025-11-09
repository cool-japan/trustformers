"""
Basic usage example for TrustformeRS-C Python bindings
"""

import numpy as np
import asyncio
from trustformers_c import (
    TrustformersC, 
    PerformanceOptimizer,
    PerformanceConfig,
    NumpyTensor,
    AsyncInference,
    AsyncPipeline,
    TensorOperations,
    get_version,
    get_build_info,
    get_platform_info,
    get_memory_usage,
    enable_simd,
    enable_dynamic_batching,
    enable_kernel_fusion,
    time_function,
    MemoryMonitor,
    benchmark_operation,
    DebugContext,
)

def basic_example():
    """Basic TrustformeRS-C usage example"""
    print("=== TrustformeRS-C Basic Example ===")
    
    # Initialize TrustformeRS-C
    with TrustformersC() as trustformers:
        # Get version and build info
        version = trustformers.get_version()
        build_info = trustformers.get_build_info()
        platform_info = trustformers.get_platform_info()
        
        print(f"TrustformeRS-C Version: {version}")
        print(f"Build Info: {build_info}")
        print(f"Platform: {platform_info}")
        
        # Check memory usage
        memory_stats = trustformers.get_memory_usage()
        print(f"Memory Usage: {memory_stats}")
        
        # Check available features
        features = ["simd", "dynamic_batching", "kernel_fusion", "gpu", "onnx"]
        for feature in features:
            available = trustformers.has_feature(feature)
            print(f"Feature '{feature}': {'Available' if available else 'Not Available'}")

def performance_optimization_example():
    """Performance optimization example"""
    print("\n=== Performance Optimization Example ===")
    
    # Create performance configuration
    config = PerformanceConfig()
    config.enable_simd = 1
    config.enable_dynamic_batching = 1
    config.enable_kernel_fusion = 1
    config.max_batch_size = 64
    config.target_latency_ms = 50
    config.num_threads = 0  # Auto-detect
    
    # Create performance optimizer
    optimizer = PerformanceOptimizer(config)
    
    # Create test matrices
    m, n, k = 512, 512, 512
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)
    
    # Time optimized matrix multiplication
    @time_function(name="optimized_matmul")
    def optimized_matmul():
        optimizer.optimize_matrix_operations(a, b, c, m, n, k)
        return c
    
    # Time regular NumPy multiplication
    @time_function(name="numpy_matmul")
    def numpy_matmul():
        return np.dot(a, b)
    
    # Compare performance
    print("Running optimized matrix multiplication...")
    result_optimized = optimized_matmul()
    
    print("Running NumPy matrix multiplication...")
    result_numpy = numpy_matmul()
    
    # Check accuracy
    error = np.mean(np.abs(result_optimized - result_numpy))
    print(f"Mean absolute error: {error:.6f}")
    
    # Get performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance Stats: {stats}")

def numpy_integration_example():
    """NumPy integration example"""
    print("\n=== NumPy Integration Example ===")
    
    # Create NumPy arrays
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    
    # Convert to NumpyTensor
    tensor_a = NumpyTensor(a)
    tensor_b = NumpyTensor(b)
    
    print(f"Tensor A shape: {tensor_a.shape}, dtype: {tensor_a.dtype}")
    print(f"Tensor B shape: {tensor_b.shape}, dtype: {tensor_b.dtype}")
    
    # Perform operations
    result_add = TensorOperations.element_wise_add(tensor_a, tensor_b)
    result_matmul = TensorOperations.matrix_multiply(tensor_a, tensor_b)
    result_relu = TensorOperations.relu(tensor_a)
    result_softmax = TensorOperations.softmax(tensor_a, axis=1)
    
    print(f"Addition result shape: {result_add.shape}")
    print(f"Matrix multiplication result shape: {result_matmul.shape}")
    print(f"ReLU result shape: {result_relu.shape}")
    print(f"Softmax result shape: {result_softmax.shape}")
    
    # Test memory efficiency
    print(f"Original tensor memory: {tensor_a.array.nbytes} bytes")
    print(f"Result tensor memory: {result_add.nbytes} bytes")

async def async_inference_example():
    """Asynchronous inference example"""
    print("\n=== Async Inference Example ===")
    
    # Mock model function
    def mock_model(x):
        """Mock model that simulates inference"""
        # Simulate some computation
        result = np.sin(x) * np.cos(x) + np.random.randn(*x.shape) * 0.1
        return result
    
    # Create async inference engine
    async with AsyncInference(
        model_function=mock_model,
        max_batch_size=8,
        max_wait_time=0.1,
        num_workers=2
    ) as engine:
        
        # Single inference
        input_data = np.random.randn(10, 10).astype(np.float32)
        result = await engine.infer(input_data)
        print(f"Single inference result shape: {result.output_data.shape}")
        print(f"Processing time: {result.processing_time:.4f} seconds")
        
        # Batch inference
        tasks = []
        for i in range(10):
            input_data = np.random.randn(10, 10).astype(np.float32)
            task = engine.infer(input_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        print(f"Batch inference completed: {len(results)} results")
        
        # Get statistics
        stats = engine.get_stats()
        print(f"Engine stats: {stats}")

async def async_pipeline_example():
    """Asynchronous pipeline example"""
    print("\n=== Async Pipeline Example ===")
    
    # Define pipeline stages
    def preprocess(x):
        """Preprocessing stage"""
        return x / 255.0 - 0.5
    
    def model_inference(x):
        """Model inference stage"""
        return np.tanh(x)
    
    def postprocess(x):
        """Postprocessing stage"""
        return x * 2.0 + 1.0
    
    stages = [preprocess, model_inference, postprocess]
    
    # Create async pipeline
    async with AsyncPipeline(stages=stages) as pipeline:
        
        # Process data through pipeline
        input_data = np.random.randint(0, 256, (32, 32, 3)).astype(np.float32)
        result = await pipeline.process(input_data)
        
        print(f"Pipeline input shape: {input_data.shape}")
        print(f"Pipeline output shape: {result.output_data.shape}")
        print(f"Pipeline processing time: {result.processing_time:.4f} seconds")

def memory_monitoring_example():
    """Memory monitoring example"""
    print("\n=== Memory Monitoring Example ===")
    
    # Create memory monitor
    monitor = MemoryMonitor(check_interval=0.1)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate memory-intensive operations
    arrays = []
    for i in range(10):
        # Create large array
        array = np.random.randn(1000, 1000).astype(np.float32)
        arrays.append(array)
        
        # Check memory periodically
        if i % 3 == 0:
            memory_stats = monitor.check_memory()
            print(f"Step {i}: Memory usage: {memory_stats}")
    
    # Stop monitoring and get stats
    final_stats = monitor.stop_monitoring()
    print(f"Memory monitoring results: {final_stats}")

def benchmarking_example():
    """Benchmarking example"""
    print("\n=== Benchmarking Example ===")
    
    # Create test data
    a = np.random.randn(500, 500).astype(np.float32)
    b = np.random.randn(500, 500).astype(np.float32)
    
    # Benchmark NumPy matrix multiplication
    numpy_results = benchmark_operation(
        np.dot, a, b,
        num_iterations=10,
        warmup_iterations=2
    )
    
    print("NumPy matrix multiplication benchmark:")
    print(f"  Mean time: {numpy_results['mean_time']:.6f} seconds")
    print(f"  Std time: {numpy_results['std_time']:.6f} seconds")
    print(f"  Min time: {numpy_results['min_time']:.6f} seconds")
    print(f"  Max time: {numpy_results['max_time']:.6f} seconds")
    
    # Benchmark TensorOperations
    tensor_results = benchmark_operation(
        TensorOperations.matrix_multiply, a, b,
        num_iterations=10,
        warmup_iterations=2
    )
    
    print("TensorOperations matrix multiplication benchmark:")
    print(f"  Mean time: {tensor_results['mean_time']:.6f} seconds")
    print(f"  Std time: {tensor_results['std_time']:.6f} seconds")
    print(f"  Min time: {tensor_results['min_time']:.6f} seconds")
    print(f"  Max time: {tensor_results['max_time']:.6f} seconds")
    
    # Compare performance
    speedup = numpy_results['mean_time'] / tensor_results['mean_time']
    print(f"Speedup: {speedup:.2f}x")

def debug_context_example():
    """Debug context example"""
    print("\n=== Debug Context Example ===")
    
    # Use debug context for detailed error reporting
    with DebugContext("matrix_operations"):
        # Simulate some operations
        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)
        
        # This will be logged with performance and memory stats
        result = np.dot(a, b)
        
        # Simulate potential error (commented out)
        # raise ValueError("Simulated error for demonstration")
        
        print(f"Operation completed successfully. Result shape: {result.shape}")

def main():
    """Main function to run all examples"""
    print("TrustformeRS-C Python Bindings Examples")
    print("=" * 50)
    
    try:
        # Basic functionality
        basic_example()
        
        # Performance optimization
        performance_optimization_example()
        
        # NumPy integration
        numpy_integration_example()
        
        # Memory monitoring
        memory_monitoring_example()
        
        # Benchmarking
        benchmarking_example()
        
        # Debug context
        debug_context_example()
        
        # Async examples (run in event loop)
        print("\n=== Running Async Examples ===")
        asyncio.run(async_examples())
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

async def async_examples():
    """Run async examples"""
    await async_inference_example()
    await async_pipeline_example()

if __name__ == "__main__":
    main()