"""
Async/await support for TrustformeRS

Provides asynchronous versions of the main API functions for non-blocking operations.
"""

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, TypeVar, AsyncIterator
import inspect

from . import (
    AutoModel, AutoTokenizer, pipeline, Tensor,
    BertModel, GPT2Model, T5Model, LlamaModel,
    PreTrainedModel, PreTrainedTokenizer
)
from .utils import download_file, cached_path, logger

T = TypeVar('T')

# Global thread pool for async operations
_async_executor: Optional[ThreadPoolExecutor] = None
_async_lock = threading.Lock()


def get_async_executor() -> ThreadPoolExecutor:
    """Get or create the global async executor."""
    global _async_executor
    if _async_executor is None:
        with _async_lock:
            if _async_executor is None:
                _async_executor = ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="trustformers-async"
                )
    return _async_executor


def cleanup_async_executor():
    """Clean up the global async executor."""
    global _async_executor
    if _async_executor is not None:
        _async_executor.shutdown(wait=True)
        _async_executor = None


def async_wrapper(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to convert a synchronous function to async.
    
    Args:
        func: The synchronous function to wrap
        
    Returns:
        Async version of the function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        executor = get_async_executor()
        
        # Create a partial function with bound arguments
        bound_func = functools.partial(func, *args, **kwargs)
        
        # Run in executor
        result = await loop.run_in_executor(executor, bound_func)
        return result
    
    return wrapper


class AsyncAutoModel:
    """Async wrapper for AutoModel with non-blocking operations."""
    
    @staticmethod
    async def from_pretrained(
        model_name_or_path: str,
        *args,
        **kwargs
    ) -> PreTrainedModel:
        """
        Asynchronously load a model from pretrained weights.
        
        Args:
            model_name_or_path: Model name or path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Loaded model
        """
        logger.info(f"Async loading model: {model_name_or_path}")
        
        loop = asyncio.get_event_loop()
        executor = get_async_executor()
        
        # Run the blocking operation in executor
        model = await loop.run_in_executor(
            executor,
            functools.partial(AutoModel.from_pretrained, model_name_or_path, *args, **kwargs)
        )
        
        logger.info(f"Model loaded asynchronously: {type(model).__name__}")
        return model


class AsyncAutoTokenizer:
    """Async wrapper for AutoTokenizer with non-blocking operations."""
    
    @staticmethod
    async def from_pretrained(
        tokenizer_name_or_path: str,
        *args,
        **kwargs
    ) -> PreTrainedTokenizer:
        """
        Asynchronously load a tokenizer from pretrained.
        
        Args:
            tokenizer_name_or_path: Tokenizer name or path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Async loading tokenizer: {tokenizer_name_or_path}")
        
        loop = asyncio.get_event_loop()
        executor = get_async_executor()
        
        # Run the blocking operation in executor
        tokenizer = await loop.run_in_executor(
            executor,
            functools.partial(AutoTokenizer.from_pretrained, tokenizer_name_or_path, *args, **kwargs)
        )
        
        logger.info(f"Tokenizer loaded asynchronously: {type(tokenizer).__name__}")
        return tokenizer


class AsyncPipeline:
    """Async wrapper for pipeline operations."""
    
    def __init__(self, sync_pipeline):
        """
        Initialize with a synchronous pipeline.
        
        Args:
            sync_pipeline: The synchronous pipeline to wrap
        """
        self._pipeline = sync_pipeline
        
    async def __call__(
        self,
        inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]],
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Asynchronously run the pipeline on inputs.
        
        Args:
            inputs: Input text(s) or structured inputs
            **kwargs: Additional pipeline arguments
            
        Returns:
            Pipeline outputs
        """
        loop = asyncio.get_event_loop()
        executor = get_async_executor()
        
        # Run inference in executor
        result = await loop.run_in_executor(
            executor,
            functools.partial(self._pipeline, inputs, **kwargs)
        )
        
        return result
    
    @staticmethod
    async def create(
        task: str,
        model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        **kwargs
    ) -> 'AsyncPipeline':
        """
        Asynchronously create a pipeline.
        
        Args:
            task: Pipeline task (e.g., "text-generation")
            model: Model name/path or model instance
            tokenizer: Tokenizer name/path or tokenizer instance
            **kwargs: Additional pipeline arguments
            
        Returns:
            Async pipeline wrapper
        """
        loop = asyncio.get_event_loop()
        executor = get_async_executor()
        
        # Create pipeline asynchronously
        sync_pipeline = await loop.run_in_executor(
            executor,
            functools.partial(pipeline, task, model=model, tokenizer=tokenizer, **kwargs)
        )
        
        return AsyncPipeline(sync_pipeline)


class AsyncModelWrapper:
    """Async wrapper for any model with advanced features."""
    
    def __init__(self, model: PreTrainedModel, max_concurrent: int = 4):
        """
        Initialize async model wrapper.
        
        Args:
            model: The model to wrap
            max_concurrent: Maximum concurrent operations
        """
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._stats = {
            'total_requests': 0,
            'concurrent_requests': 0,
            'avg_response_time': 0.0
        }
    
    async def forward(self, *args, **kwargs) -> Any:
        """Async forward pass."""
        async with self.semaphore:
            import time
            start_time = time.time()
            
            self._stats['total_requests'] += 1
            self._stats['concurrent_requests'] += 1
            
            try:
                loop = asyncio.get_event_loop()
                executor = get_async_executor()
                
                result = await loop.run_in_executor(
                    executor,
                    functools.partial(self.model.forward, *args, **kwargs)
                )
                
                # Update stats
                end_time = time.time()
                response_time = end_time - start_time
                alpha = 0.1  # Smoothing factor
                self._stats['avg_response_time'] = (
                    alpha * response_time + 
                    (1 - alpha) * self._stats['avg_response_time']
                )
                
                return result
                
            finally:
                self._stats['concurrent_requests'] -= 1
    
    async def generate(self, input_ids: Tensor = None, **kwargs) -> Tensor:
        """Async text generation."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            executor = get_async_executor()
            
            if hasattr(self.model, 'generate'):
                # Call with keyword arguments if input_ids is provided that way
                if input_ids is not None:
                    kwargs['input_ids'] = input_ids
                result = await loop.run_in_executor(
                    executor,
                    functools.partial(self.model.generate, **kwargs)
                )
            else:
                # Raise AttributeError if generate method doesn't exist
                raise AttributeError(f"'{type(self.model).__name__}' object has no attribute 'generate'")
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async operation statistics."""
        return self._stats.copy()


class AsyncTokenizerWrapper:
    """Async wrapper for tokenizer with batch processing."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_concurrent: int = 8):
        """
        Initialize async tokenizer wrapper.
        
        Args:
            tokenizer: The tokenizer to wrap
            max_concurrent: Maximum concurrent operations
        """
        self.tokenizer = tokenizer
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def encode(self, text: str, **kwargs) -> List[int]:
        """Async text encoding."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            executor = get_async_executor()
            
            result = await loop.run_in_executor(
                executor,
                functools.partial(self.tokenizer.encode, text, **kwargs)
            )
            
            return result
    
    async def decode(self, token_ids: List[int], **kwargs) -> str:
        """Async token decoding."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            executor = get_async_executor()
            
            result = await loop.run_in_executor(
                executor,
                functools.partial(self.tokenizer.decode, token_ids, **kwargs)
            )
            
            return result
    
    async def batch_encode(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Async batch encoding."""
        # Process in smaller chunks for better concurrency
        chunk_size = max(1, len(texts) // 4)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        async def encode_chunk(chunk):
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                executor = get_async_executor()
                
                results = []
                for text in chunk:
                    result = await loop.run_in_executor(
                        executor,
                        functools.partial(self.tokenizer.encode, text, **kwargs)
                    )
                    results.append(result)
                return results
        
        # Process chunks concurrently
        tasks = [encode_chunk(chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for chunk in chunk_results for item in chunk]
    
    async def batch_encode_plus(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Async batch encode plus with attention masks."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            executor = get_async_executor()
            
            if hasattr(self.tokenizer, 'batch_encode_plus'):
                result = await loop.run_in_executor(
                    executor,
                    functools.partial(self.tokenizer.batch_encode_plus, texts, **kwargs)
                )
            else:
                # Fallback to individual encoding
                input_ids = []
                attention_mask = []
                for text in texts:
                    encoded = await loop.run_in_executor(
                        executor,
                        functools.partial(self.tokenizer.encode, text, **kwargs)
                    )
                    input_ids.append(encoded)
                    attention_mask.append([1] * len(encoded))
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            
            return result


class AsyncBatchProcessor:
    """Advanced async batch processor with streaming and backpressure."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_concurrent_batches: int = 4,
                 queue_size: int = 1000):
        """
        Initialize async batch processor.
        
        Args:
            batch_size: Size of each batch
            max_concurrent_batches: Maximum concurrent batches
            queue_size: Maximum queue size for backpressure
        """
        self.batch_size = batch_size
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._processing = False
        self._stats = {
            'total_items': 0,
            'total_batches': 0,
            'queue_size': 0
        }
    
    async def process_stream(self, 
                           items: AsyncIterator[Any],
                           processor: Callable[[List[Any]], Awaitable[List[Any]]]) -> AsyncIterator[Any]:
        """
        Process items from async stream in batches.
        
        Args:
            items: Async iterator of items to process
            processor: Async function to process batches
            
        Yields:
            Processed items
        """
        batch = []
        
        async for item in items:
            batch.append(item)
            self._stats['total_items'] += 1
            
            if len(batch) >= self.batch_size:
                async with self.semaphore:
                    results = await processor(batch)
                    for result in results:
                        yield result
                
                self._stats['total_batches'] += 1
                batch = []
        
        # Process remaining items
        if batch:
            async with self.semaphore:
                results = await processor(batch)
                for result in results:
                    yield result
            self._stats['total_batches'] += 1
    
    async def process_batches(self,
                             items: List[Any],
                             processor: Callable,
                             **kwargs) -> List[Any]:
        """
        Process a list of items in batches.
        
        Args:
            items: List of items to process
            processor: Function to process each batch
            **kwargs: Additional arguments to processor
            
        Returns:
            List of processed results
        """
        results = []
        
        # Split items into batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            async with self.semaphore:
                if asyncio.iscoroutinefunction(processor):
                    batch_results = await processor(batch, **kwargs)
                else:
                    # For sync functions, run in executor
                    loop = asyncio.get_event_loop()
                    executor = get_async_executor()
                    batch_results = await loop.run_in_executor(
                        executor,
                        functools.partial(processor, batch, **kwargs)
                    )
                
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
                
                self._stats['total_batches'] += 1
        
        self._stats['total_items'] += len(items)
        return results
    
    async def process_list(self,
                          items: List[Any],
                          processor: Callable[[List[Any]], Awaitable[List[Any]]]) -> List[Any]:
        """
        Process list of items in parallel batches.
        
        Args:
            items: List of items to process
            processor: Async function to process batches
            
        Returns:
            List of processed items
        """
        # Split into batches
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        async def process_batch(batch):
            async with self.semaphore:
                return await processor(batch)
        
        # Process batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [item for batch in batch_results for item in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        self._stats['queue_size'] = self.queue.qsize()
        return self._stats.copy()


class AsyncResourceManager:
    """Manage async resources with automatic cleanup."""
    
    def __init__(self):
        """Initialize async resource manager."""
        self._resources = {}
        self._locks = {}
    
    async def get_or_create(self, 
                           key: str,
                           factory: Callable[[], Awaitable[Any]]) -> Any:
        """
        Get or create a resource asynchronously.
        
        Args:
            key: Resource key
            factory: Async factory function
            
        Returns:
            The resource
        """
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            if key not in self._resources:
                self._resources[key] = await factory()
            return self._resources[key]
    
    async def release(self, key: str):
        """Release a resource."""
        if key in self._resources:
            resource = self._resources.pop(key)
            if hasattr(resource, 'close') and asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            elif hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup') and asyncio.iscoroutinefunction(resource.cleanup):
                await resource.cleanup()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
    
    async def cleanup_all(self):
        """Cleanup all resources."""
        for key in list(self._resources.keys()):
            await self.release(key)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_all()
    
    def add_resource(self, resource: Any, key: str = None):
        """Add a resource to be managed."""
        if key is None:
            key = f"resource_{len(self._resources)}"
        self._resources[key] = resource
        return key


class AsyncStreamProcessor:
    """Process streams of data asynchronously with backpressure control."""
    
    def __init__(self, processor_func: Callable = None, buffer_size: int = 100):
        """
        Initialize stream processor.
        
        Args:
            processor_func: Function to process items
            buffer_size: Size of internal buffer
        """
        self.buffer_size = buffer_size
        self.processor_func = processor_func
        self.result_queue = asyncio.Queue()
        self.processing = False
    
    async def map(self, 
                  stream: AsyncIterator[Any],
                  mapper: Callable[[Any], Awaitable[Any]],
                  max_concurrent: int = 10) -> AsyncIterator[Any]:
        """
        Map function over async stream with concurrency control.
        
        Args:
            stream: Input async stream
            mapper: Async mapping function
            max_concurrent: Maximum concurrent operations
            
        Yields:
            Mapped items
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        pending = set()
        
        async def process_item(item):
            async with semaphore:
                return await mapper(item)
        
        async for item in stream:
            # Start processing item
            task = asyncio.create_task(process_item(item))
            pending.add(task)
            
            # If we have too many pending, wait for some to complete
            if len(pending) >= max_concurrent:
                done, pending = await asyncio.wait(
                    pending, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    yield await task
        
        # Wait for remaining tasks
        if pending:
            for task in asyncio.as_completed(pending):
                yield await task
    
    async def filter(self,
                    stream: AsyncIterator[Any],
                    predicate: Callable[[Any], Awaitable[bool]]) -> AsyncIterator[Any]:
        """
        Filter async stream with async predicate.
        
        Args:
            stream: Input async stream
            predicate: Async predicate function
            
        Yields:
            Filtered items
        """
        async for item in stream:
            if await predicate(item):
                yield item
    
    async def batch(self,
                   stream: AsyncIterator[Any],
                   batch_size: int) -> AsyncIterator[List[Any]]:
        """
        Batch items from async stream.
        
        Args:
            stream: Input async stream
            batch_size: Size of each batch
            
        Yields:
            Batches of items
        """
        batch = []
        async for item in stream:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    async def start_processing(self):
        """Start processing."""
        self.processing = True
    
    async def stop_processing(self):
        """Stop processing."""
        self.processing = False
    
    async def add_item(self, item: Any):
        """Add an item to be processed."""
        if self.processor_func and self.processing:
            if asyncio.iscoroutinefunction(self.processor_func):
                result = await self.processor_func(item)
            else:
                result = self.processor_func(item)
            await self.result_queue.put(result)
    
    async def get_result(self):
        """Get a processed result."""
        return await self.result_queue.get()


class AsyncOperationError(Exception):
    """Exception for async operation errors."""
    
    def __init__(self, message: str, operation: str, original_error: Exception = None):
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


# Convenience functions for common async operations

async def async_download_file(url: str, 
                             local_path: str,
                             chunk_size: int = 8192,
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Asynchronously download a file with progress tracking.
    
    Args:
        url: URL to download from
        local_path: Local path to save to
        chunk_size: Size of each chunk
        progress_callback: Optional progress callback
        
    Returns:
        Path to downloaded file
    """
    loop = asyncio.get_event_loop()
    executor = get_async_executor()
    
    def sync_download():
        return download_file(url, local_path)
    
    try:
        result = await loop.run_in_executor(executor, sync_download)
        return result
    except Exception as e:
        raise AsyncOperationError(f"Failed to download {url}", "download", e)


async def async_cached_path(url_or_path: str, **kwargs) -> str:
    """
    Asynchronously get cached path.
    
    Args:
        url_or_path: URL or local path
        **kwargs: Additional arguments
        
    Returns:
        Path to cached file
    """
    loop = asyncio.get_event_loop()
    executor = get_async_executor()
    
    try:
        result = await loop.run_in_executor(
            executor,
            functools.partial(cached_path, url_or_path, **kwargs)
        )
        return result
    except Exception as e:
        raise AsyncOperationError(f"Failed to get cached path for {url_or_path}", "cache", e)


async def async_pipeline(task: str, **kwargs) -> AsyncPipeline:
    """
    Convenience function to create async pipeline.
    
    Args:
        task: Pipeline task
        **kwargs: Pipeline arguments
        
    Returns:
        Async pipeline
    """
    return await AsyncPipeline.create(task, **kwargs)


def wrap_model_async(model: PreTrainedModel, max_concurrent: int = 4) -> AsyncModelWrapper:
    """
    Wrap a model for async operations.
    
    Args:
        model: Model to wrap
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Async model wrapper
    """
    return AsyncModelWrapper(model, max_concurrent)


def wrap_tokenizer_async(tokenizer: PreTrainedTokenizer, max_concurrent: int = 8) -> AsyncTokenizerWrapper:
    """
    Wrap a tokenizer for async operations.
    
    Args:
        tokenizer: Tokenizer to wrap
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Async tokenizer wrapper
    """
    return AsyncTokenizerWrapper(tokenizer, max_concurrent)


async def safe_async_call(coro: Awaitable[T], 
                         timeout: Optional[float] = None,
                         fallback: Optional[T] = None) -> T:
    """
    Safely call an async function with timeout and fallback.
    
    Args:
        coro: Coroutine to call
        timeout: Optional timeout
        fallback: Fallback value on error
        
    Returns:
        Result or fallback
    """
    try:
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        else:
            return await coro
    except asyncio.TimeoutError:
        if fallback is not None:
            return fallback
        raise AsyncOperationError("Operation timed out", "timeout")
    except Exception as e:
        if fallback is not None:
            return fallback
        raise AsyncOperationError("Operation failed", "execution", e)


# Context managers for async resource management

class AsyncModel:
    """Async context manager for models."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.model = None
    
    async def __aenter__(self) -> AsyncModelWrapper:
        self.model = await AsyncAutoModel.from_pretrained(
            self.model_name_or_path, 
            **self.kwargs
        )
        return wrap_model_async(self.model)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.model and hasattr(self.model, 'cleanup'):
            await self.model.cleanup()


class AsyncTokenizer:
    """Async context manager for tokenizers."""
    
    def __init__(self, tokenizer_name_or_path: str, **kwargs):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.kwargs = kwargs
        self.tokenizer = None
    
    async def __aenter__(self) -> AsyncTokenizerWrapper:
        self.tokenizer = await AsyncAutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            **self.kwargs
        )
        return wrap_tokenizer_async(self.tokenizer)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tokenizer and hasattr(self.tokenizer, 'cleanup'):
            await self.tokenizer.cleanup()


# Advanced async utilities

class AsyncThrottler:
    """Rate limiting for async operations."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize throttler.
        
        Args:
            rate: Operations per second
            burst: Burst capacity
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to proceed."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on time passed
            self.tokens = min(self.burst, self.tokens + time_passed * self.rate)
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Need to wait
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


class AsyncCircuitBreaker:
    """Circuit breaker for async operations."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            timeout: Timeout before attempting reset
            expected_exception: Type of exception to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.lock = asyncio.Lock()
    
    async def call(self, coro: Awaitable[T]) -> T:
        """
        Call function through circuit breaker.
        
        Args:
            coro: Coroutine to call
            
        Returns:
            Result of coroutine
        """
        async with self.lock:
            if self.state == 'open':
                if (asyncio.get_event_loop().time() - self.last_failure_time) > self.timeout:
                    self.state = 'half-open'
                else:
                    raise AsyncOperationError("Circuit breaker is open", "circuit_breaker")
        
        try:
            result = await coro
            
            async with self.lock:
                if self.state == 'half-open':
                    self.state = 'closed'
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = asyncio.get_event_loop().time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
            
            raise AsyncOperationError("Operation failed and circuit breaker opened", "circuit_breaker", e)


# Global instances for convenience
_resource_manager = AsyncResourceManager()
_stream_processor = AsyncStreamProcessor()


def get_resource_manager() -> AsyncResourceManager:
    """Get global async resource manager."""
    return _resource_manager


def get_stream_processor() -> AsyncStreamProcessor:
    """Get global async stream processor."""
    return _stream_processor
