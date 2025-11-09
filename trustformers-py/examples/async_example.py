"""
Example demonstrating async/await support in TrustformeRS

This example shows how to use the async utilities for non-blocking operations
with TrustformeRS models and tokenizers.
"""

import asyncio
import time
from typing import List

# Import TrustformeRS async utilities
from trustformers import (
    AsyncAutoModel, AsyncAutoTokenizer, AsyncPipeline,
    AsyncModelWrapper, AsyncTokenizerWrapper, AsyncBatchProcessor,
    AsyncResourceManager, wrap_model_async, wrap_tokenizer_async,
    async_pipeline, safe_async_call, get_async_executor
)

# Also import regular TrustformeRS for comparison
from trustformers import AutoModel, AutoTokenizer, pipeline


async def example_async_model_loading():
    """Example: Asynchronous model loading."""
    print("🔄 Loading models asynchronously...")
    
    # Start loading multiple models concurrently
    start_time = time.time()
    
    # Create tasks for concurrent loading
    tasks = [
        AsyncAutoModel.from_pretrained("bert-base-uncased"),
        AsyncAutoTokenizer.from_pretrained("bert-base-uncased"),
        AsyncAutoModel.from_pretrained("gpt2"),
        AsyncAutoTokenizer.from_pretrained("gpt2"),
    ]
    
    # Wait for all models to load
    try:
        bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer = await asyncio.gather(*tasks)
        
        loading_time = time.time() - start_time
        print(f"✅ All models loaded in {loading_time:.2f} seconds")
        
        return bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None


async def example_async_pipeline():
    """Example: Asynchronous pipeline creation and usage."""
    print("\n🔄 Creating async pipeline...")
    
    # Create async pipeline
    async_text_gen = await async_pipeline(
        "text-generation",
        model="gpt2"
    )
    
    # Use pipeline asynchronously
    prompts = [
        "The future of AI is",
        "Machine learning will",
        "Deep learning has"
    ]
    
    start_time = time.time()
    
    # Process multiple prompts concurrently
    tasks = [
        async_text_gen(prompt, max_length=50, num_return_sequences=1)
        for prompt in prompts
    ]
    
    results = await asyncio.gather(*tasks)
    
    processing_time = time.time() - start_time
    print(f"✅ Processed {len(prompts)} prompts in {processing_time:.2f} seconds")
    
    for i, result in enumerate(results):
        print(f"  Prompt {i+1}: {result}")
    
    return results


async def example_async_model_wrapper():
    """Example: Using AsyncModelWrapper for inference."""
    print("\n🔄 Using async model wrapper...")
    
    try:
        # Load model and tokenizer
        model = await AsyncAutoModel.from_pretrained("bert-base-uncased")
        tokenizer = await AsyncAutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Wrap for async operations
        async_model = wrap_model_async(model)
        async_tokenizer = wrap_tokenizer_async(tokenizer)
        
        # Prepare inputs
        texts = [
            "Hello world",
            "TrustformeRS is awesome",
            "Async processing is fast"
        ]
        
        # Tokenize texts concurrently
        tokenize_tasks = [
            async_tokenizer.encode(text, return_tensors="pt")
            for text in texts
        ]
        
        encoded_inputs = await asyncio.gather(*tokenize_tasks)
        
        # Run inference concurrently
        inference_tasks = [
            async_model.forward(input_ids=inputs)
            for inputs in encoded_inputs
        ]
        
        outputs = await asyncio.gather(*inference_tasks)
        
        print(f"✅ Processed {len(texts)} texts with async inference")
        
        return outputs
        
    except Exception as e:
        print(f"❌ Error in async model wrapper: {e}")
        return None


async def example_async_batch_processor():
    """Example: Using AsyncBatchProcessor for efficient batch processing."""
    print("\n🔄 Using async batch processor...")
    
    def process_batch(batch: List[str]) -> List[str]:
        """Simple batch processing function."""
        return [f"Processed: {item}" for item in batch]
    
    # Create batch processor
    batch_processor = AsyncBatchProcessor(batch_size=3)
    
    # Large list of items to process
    items = [f"Item {i}" for i in range(20)]
    
    start_time = time.time()
    
    # Process all items in batches
    results = await batch_processor.process_batches(items, process_batch)
    
    processing_time = time.time() - start_time
    print(f"✅ Processed {len(items)} items in {processing_time:.2f} seconds")
    print(f"  Results: {results[:5]}...")  # Show first 5 results
    
    return results


async def example_async_resource_manager():
    """Example: Using AsyncResourceManager for resource cleanup."""
    print("\n🔄 Using async resource manager...")
    
    class MockResource:
        def __init__(self, name: str):
            self.name = name
            self.cleaned = False
            print(f"  Created resource: {name}")
        
        def cleanup(self):
            self.cleaned = True
            print(f"  Cleaned up resource: {self.name}")
    
    async with AsyncResourceManager() as manager:
        # Create some resources
        resource1 = MockResource("Resource 1")
        resource2 = MockResource("Resource 2")
        
        # Add to manager for automatic cleanup
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        
        # Do some work...
        await asyncio.sleep(0.1)
        
        print("  Resources in use...")
    
    # Resources should be cleaned up automatically
    print("✅ Resources cleaned up automatically")


async def example_safe_async_call():
    """Example: Using safe_async_call for error handling."""
    print("\n🔄 Using safe async call...")
    
    async def risky_operation():
        await asyncio.sleep(0.1)
        raise ValueError("This operation failed!")
    
    async def slow_operation():
        await asyncio.sleep(2)
        return "This is slow"
    
    # Test error handling
    result1 = await safe_async_call(
        risky_operation(),
        default="Default value",
        reraise=False
    )
    print(f"  Result with error: {result1}")
    
    # Test timeout handling
    result2 = await safe_async_call(
        slow_operation(),
        timeout=0.5,
        default="Timeout default",
        reraise=False
    )
    print(f"  Result with timeout: {result2}")
    
    print("✅ Safe async call handled errors gracefully")


async def benchmark_async_vs_sync():
    """Benchmark: Compare async vs sync performance."""
    print("\n📊 Benchmarking async vs sync performance...")
    
    def sync_task(n: int) -> int:
        """Simulate a CPU-bound task."""
        time.sleep(0.1)  # Simulate work
        return n * 2
    
    async def async_task(n: int) -> int:
        """Async version of the task."""
        await asyncio.sleep(0.1)  # Simulate async work
        return n * 2
    
    tasks_count = 5
    
    # Sync version
    print(f"  Running {tasks_count} sync tasks...")
    start_time = time.time()
    sync_results = []
    for i in range(tasks_count):
        result = sync_task(i)
        sync_results.append(result)
    sync_time = time.time() - start_time
    
    # Async version
    print(f"  Running {tasks_count} async tasks...")
    start_time = time.time()
    async_tasks = [async_task(i) for i in range(tasks_count)]
    async_results = await asyncio.gather(*async_tasks)
    async_time = time.time() - start_time
    
    print(f"  Sync time: {sync_time:.2f} seconds")
    print(f"  Async time: {async_time:.2f} seconds")
    print(f"  Speedup: {sync_time / async_time:.2f}x")
    
    assert sync_results == async_results
    print("✅ Async version is faster for concurrent operations")


async def main():
    """Main async function demonstrating all examples."""
    print("🚀 TrustformeRS Async Utilities Demo")
    print("=" * 50)
    
    try:
        # Run all examples
        await example_async_model_loading()
        await example_async_pipeline()
        await example_async_model_wrapper()
        await example_async_batch_processor()
        await example_async_resource_manager()
        await example_safe_async_call()
        await benchmark_async_vs_sync()
        
        print("\n🎉 All async examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in async demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up async resources
        from trustformers import cleanup_async_executor
        cleanup_async_executor()
        print("\n🧹 Cleaned up async resources")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())