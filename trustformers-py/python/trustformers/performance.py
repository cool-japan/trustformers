"""
Performance optimization utilities for TrustformeRS

Provides binding optimization, batch processing, and other performance enhancements
to reduce Python/Rust overhead.
"""

import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Iterator
from functools import wraps, lru_cache
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc
import logging
from contextlib import contextmanager
import numpy as np
import os
import sys

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_workers: int = 4
    timeout: Optional[float] = None
    preserve_order: bool = True
    auto_batch_size: bool = True
    memory_limit_mb: Optional[float] = None


class BatchProcessor:
    """Efficient batch processing to reduce Python/Rust overhead."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4, 
                 timeout: Optional[float] = None, preserve_order: bool = True,
                 adaptive_sizing: bool = True, memory_limit_mb: Optional[float] = None,
                 config: BatchConfig = None):
        if config is not None:
            self.config = config
        else:
            self.config = BatchConfig(
                batch_size=batch_size,
                max_workers=max_workers, 
                timeout=timeout,
                preserve_order=preserve_order,
                auto_batch_size=adaptive_sizing,
                memory_limit_mb=memory_limit_mb
            )
        
        # Expose properties for direct access
        self.batch_size = self.config.batch_size
        self.max_workers = self.config.max_workers
        self.timeout = self.config.timeout
        self.preserve_order = self.config.preserve_order
        self.adaptive_sizing = self.config.auto_batch_size
        self.memory_limit_mb = self.config.memory_limit_mb
        
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._optimal_batch_sizes = {}
        self._lock = threading.RLock()
    
    async def process_batch(self, 
                           items: List[Any], 
                           processor_func: Callable[[List[Any]], List[Any]],
                           operation_name: str = "batch_op") -> List[Any]:
        """Process items in batches asynchronously."""
        if not items:
            return []
        
        # Determine optimal batch size
        batch_size = self._get_optimal_batch_size(operation_name, len(items))
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches
        if len(batches) == 1:
            # Single batch - process directly
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, processor_func, batches[0])
        
        # Multiple batches - process in parallel
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self.executor, processor_func, batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        return results
    
    def _get_optimal_batch_size(self, operation_name: str, total_items: int) -> int:
        """Get optimal batch size for operation."""
        if not self.config.auto_batch_size:
            return min(self.config.batch_size, total_items)
        
        with self._lock:
            if operation_name in self._optimal_batch_sizes:
                return min(self._optimal_batch_sizes[operation_name], total_items)
            
            # Start with default batch size
            return min(self.config.batch_size, total_items)
    
    def update_optimal_batch_size(self, operation_name: str, batch_size: int, duration: float):
        """Update optimal batch size based on performance."""
        with self._lock:
            if operation_name not in self._optimal_batch_sizes:
                self._optimal_batch_sizes[operation_name] = batch_size
            else:
                # Simple adaptive algorithm
                current_size = self._optimal_batch_sizes[operation_name]
                if duration < 0.1:  # Very fast - increase batch size
                    self._optimal_batch_sizes[operation_name] = min(current_size * 2, 1024)
                elif duration > 1.0:  # Too slow - decrease batch size
                    self._optimal_batch_sizes[operation_name] = max(current_size // 2, 1)
    
    def update_performance(self, processing_time: float, batch_size: int, operation_name: str = "default"):
        """Update performance metrics for adaptive batch sizing."""
        self.update_optimal_batch_size(operation_name, batch_size, processing_time)
    
    def get_optimal_batch_size(self, operation_name: str = "default") -> int:
        """Get optimal batch size for operation."""
        return self._get_optimal_batch_size(operation_name, self.batch_size)
    
    def close(self):
        """Close the batch processor."""
        self.executor.shutdown(wait=True)


class ObjectPool:
    """Object pool to reduce allocation overhead."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 100, reset_func: Optional[Callable[[Any], None]] = None):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self.pool = []
        self.lock = threading.RLock()
        self.created_count = 0
        self.reused_count = 0
    
    @property
    def current_size(self) -> int:
        """Get current pool size."""
        with self.lock:
            return len(self.pool)
    
    def acquire(self) -> Any:
        """Acquire object from pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.reused_count += 1
                return obj
            else:
                obj = self.factory()
                self.created_count += 1
                return obj
    
    def get(self) -> Any:
        """Get object from pool (alias for acquire)."""
        return self.acquire()
    
    def release(self, obj: Any) -> None:
        """Release object back to pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Apply reset function if provided, otherwise try object's own reset method
                if self.reset_func:
                    self.reset_func(obj)
                elif hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def put(self, obj: Any) -> None:
        """Put object back to pool (alias for release)."""
        self.release(obj)
    
    def return_object(self, obj: Any) -> None:
        """Return object to pool (alias for release)."""
        self.release(obj)
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': self.reused_count / max(self.created_count, 1)
            }
    
    def clear(self) -> None:
        """Clear the pool."""
        with self.lock:
            self.pool.clear()
            self.created_count = 0
            self.reused_count = 0


class LazyLoader:
    """Lazy loading to reduce initialization overhead."""
    
    def __init__(self, loader_func: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs or {}
        self._value = None
        self._loaded = False
        self._lock = threading.RLock()
    
    @property
    def is_loaded(self) -> bool:
        """Check if value has been loaded."""
        return self._loaded
    
    def get(self) -> Any:
        """Get the loaded value."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._value = self.loader_func(*self.args, **self.kwargs)
                    self._loaded = True
        return self._value
    
    def reset(self) -> None:
        """Reset the lazy loader."""
        with self._lock:
            self._value = None
            self._loaded = False


class MemoryPool:
    """Memory pool for efficient tensor allocation."""
    
    def __init__(self, initial_size: int = 100, growth_factor: float = 1.5, sizes: Optional[List[Tuple[int, ...]]] = None, max_tensors_per_size: int = 10):
        self.initial_size = initial_size
        self.growth_factor = growth_factor
        self.pools = {}
        self.max_tensors_per_size = max_tensors_per_size
        self.lock = threading.RLock()
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_memory_mb': 0
        }
        
        # Initialize pools for common sizes
        if sizes:
            for size in sizes:
                self.pools[size] = []
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: str = 'float32') -> Optional[Any]:
        """Get tensor from pool."""
        with self.lock:
            key = (shape, dtype)
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                self.stats['reuses'] += 1
                return tensor
            
            # Not available in pool
            self.stats['allocations'] += 1
            return None
    
    def return_tensor(self, tensor: Any) -> None:
        """Return tensor to pool."""
        with self.lock:
            try:
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype)
                key = (shape, dtype)
                
                if key not in self.pools:
                    self.pools[key] = []
                
                if len(self.pools[key]) < self.max_tensors_per_size:
                    # Zero out tensor before returning to pool
                    if hasattr(tensor, 'zero_'):
                        tensor.zero_()
                    self.pools[key].append(tensor)
            except Exception as e:
                logger.warning(f"Failed to return tensor to pool: {e}")
    
    def clear(self) -> None:
        """Clear all pools."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.stats['allocations'] = 0
            self.stats['reuses'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            total_tensors = sum(len(pool) for pool in self.pools.values())
            return {
                **self.stats,
                'total_tensors': total_tensors,
                'pool_sizes': {str(k): len(v) for k, v in self.pools.items()},
                'reuse_rate': self.stats['reuses'] / max(self.stats['allocations'], 1)
            }
    
    def allocate(self, size: int) -> List[float]:
        """Allocate memory block of specified size."""
        with self.lock:
            self.stats['allocations'] += 1
            # Simple allocation - return a list of the requested size
            return [0.0] * size
    
    def deallocate(self, memory: List[float]) -> None:
        """Deallocate memory block."""
        with self.lock:
            # Simple deallocation - just clear the list to help GC
            if memory:
                memory.clear()


class CallBatcher:
    """Batches multiple function calls to reduce overhead."""
    
    def __init__(self, batch_function: Callable, max_batch_size: int = 32, max_wait_time: float = 0.01):
        self.batch_function = batch_function
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_calls = []
        self.lock = threading.RLock()
        self.timer = None
    
    def add_call(self, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Add a call to the batch."""
        import threading
        
        # Create a future-like object
        result_event = threading.Event()
        result_container = {'result': None, 'exception': None}
        
        call_info = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'result_event': result_event,
            'result_container': result_container
        }
        
        with self.lock:
            self.pending_calls.append(call_info)
            
            # Start timer if not already running
            if self.timer is None:
                self.timer = threading.Timer(self.max_wait_time, self._execute_batch)
                self.timer.start()
            
            # Execute immediately if batch is full
            if len(self.pending_calls) >= self.max_batch_size:
                if self.timer:
                    self.timer.cancel()
                self._execute_batch()
        
        # Wait for result
        result_event.wait()
        
        if result_container['exception']:
            raise result_container['exception']
        
        return result_container['result']
    
    def _execute_batch(self):
        """Execute the current batch."""
        with self.lock:
            if not self.pending_calls:
                return
            
            calls = self.pending_calls.copy()
            self.pending_calls.clear()
            self.timer = None
        
        # Group calls by function
        function_groups = {}
        for call in calls:
            func_key = (call['func'], id(call['func']))
            if func_key not in function_groups:
                function_groups[func_key] = []
            function_groups[func_key].append(call)
        
        # Execute each group
        for func_key, group_calls in function_groups.items():
            func = group_calls[0]['func']
            
            # Check if function supports batching
            if hasattr(func, '_supports_batching') and func._supports_batching:
                # Batch the calls
                batch_args = [call['args'] for call in group_calls]
                batch_kwargs = [call['kwargs'] for call in group_calls]
                
                try:
                    batch_results = func(batch_args, batch_kwargs)
                    
                    # Distribute results
                    for i, call in enumerate(group_calls):
                        call['result_container']['result'] = batch_results[i]
                        call['result_event'].set()
                        
                except Exception as e:
                    # Set exception for all calls in group
                    for call in group_calls:
                        call['result_container']['exception'] = e
                        call['result_event'].set()
            else:
                # Execute calls individually
                for call in group_calls:
                    try:
                        result = func(*call['args'], **call['kwargs'])
                        call['result_container']['result'] = result
                    except Exception as e:
                        call['result_container']['exception'] = e
                    finally:
                        call['result_event'].set()
    
    async def call(self, *args, **kwargs):
        """Make an async call that will be batched."""
        # Return a coroutine that can be awaited
        loop = asyncio.get_event_loop()
        
        # Create a future for this call
        future = loop.create_future()
        
        # Store the call data
        call_data = {
            'args': args,
            'kwargs': kwargs,
            'future': future
        }
        
        with self.lock:
            self.pending_calls.append(call_data)
            
            # Start timer if not already running
            if self.timer is None:
                self.timer = threading.Timer(self.max_wait_time, self._execute_batch_async)
                self.timer.start()
            
            # Execute immediately if batch is full
            if len(self.pending_calls) >= self.max_batch_size:
                if self.timer:
                    self.timer.cancel()
                await self._execute_batch_async()
        
        return await future
    
    async def _execute_batch_async(self):
        """Execute the current batch asynchronously."""
        with self.lock:
            if not self.pending_calls:
                return
            
            calls = self.pending_calls.copy()
            self.pending_calls.clear()
            self.timer = None
        
        # Extract call arguments for batch processing
        call_args = [call['args'][0] if call['args'] else None for call in calls]
        
        try:
            # Use the batch function to process all calls
            results = self.batch_function(call_args)
            
            # Set results for each future
            for call, result in zip(calls, results):
                call['future'].set_result(result)
                
        except Exception as e:
            # Set exception for all futures
            for call in calls:
                call['future'].set_exception(e)


class ParallelTokenizer:
    """Multi-threaded tokenizer for high-throughput text processing."""
    
    def __init__(self, tokenizer, max_workers: int = None, chunk_size: int = 1000):
        self.tokenizer = tokenizer
        self.max_workers = max_workers or min(32, (4 * (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4)))
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._stats = {
            'total_texts': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_throughput': 0.0
        }
    
    def tokenize_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """Tokenize a batch of texts in parallel."""
        if not texts:
            return []
        
        start_time = time.time()
        
        # Split texts into chunks for parallel processing
        chunks = [texts[i:i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]
        
        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._tokenize_chunk, chunk, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
        
        # Update statistics
        end_time = time.time()
        self._update_stats(len(texts), sum(len(r.get('input_ids', [])) for r in results), end_time - start_time)
        
        return results
    
    def _tokenize_chunk(self, chunk: List[str], **kwargs) -> List[Dict]:
        """Tokenize a chunk of texts."""
        results = []
        for text in chunk:
            try:
                result = self.tokenizer(text, **kwargs)
                if hasattr(result, 'data'):
                    results.append(result.data)
                else:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to tokenize text: {e}")
                results.append({'input_ids': [], 'attention_mask': []})
        return results
    
    def tokenize_parallel_with_callback(self, texts: List[str], callback: Callable[[int, Dict], None], **kwargs):
        """Tokenize texts in parallel with progress callback."""
        chunk_size = min(self.chunk_size, len(texts) // self.max_workers + 1)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        def process_chunk_with_callback(chunk_idx, chunk):
            results = self._tokenize_chunk(chunk, **kwargs)
            for i, result in enumerate(results):
                callback(chunk_idx * chunk_size + i, result)
            return results
        
        futures = []
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(process_chunk_with_callback, i, chunk)
            futures.append(future)
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()
    
    def tokenize_streaming(self, texts: List[str], **kwargs) -> Iterator[Dict]:
        """Tokenize texts with streaming results."""
        chunk_size = min(self.chunk_size, len(texts) // self.max_workers + 1)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Submit all chunks
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._tokenize_chunk, chunk, **kwargs)
            futures.append(future)
        
        # Yield results as they complete
        for future in as_completed(futures):
            chunk_results = future.result()
            for result in chunk_results:
                yield result
    
    def _update_stats(self, num_texts: int, num_tokens: int, duration: float):
        """Update tokenization statistics."""
        self._stats['total_texts'] += num_texts
        self._stats['total_tokens'] += num_tokens
        self._stats['total_time'] += duration
        if self._stats['total_time'] > 0:
            self._stats['avg_throughput'] = self._stats['total_tokens'] / self._stats['total_time']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        return self._stats.copy()
    
    def close(self):
        """Close the parallel tokenizer."""
        self.executor.shutdown(wait=True)


class ParallelInferenceEngine:
    """Multi-threaded inference engine for high-throughput model inference."""
    
    def __init__(self, model, max_workers: int = None, batch_size: int = 32, queue_size: int = 1000):
        self.model = model
        self.max_workers = max_workers or min(8, (2 * (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4)))
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._request_queue = []
        self._queue_lock = threading.RLock()
        self._stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_latency': 0.0,
            'avg_throughput': 0.0,
            'queue_wait_time': 0.0
        }
    
    def inference_batch(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        """Run inference on a batch of inputs in parallel."""
        if not inputs:
            return []
        
        start_time = time.time()
        
        # Split inputs into smaller batches for parallel processing
        batches = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        
        # Process batches in parallel
        futures = []
        for batch in batches:
            future = self.executor.submit(self._inference_batch, batch, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
        
        # Update statistics
        end_time = time.time()
        self._update_inference_stats(len(inputs), end_time - start_time)
        
        return results
    
    def _inference_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Run inference on a single batch."""
        try:
            # Convert batch to model input format
            batch_input = self._prepare_batch_input(batch)
            
            # Run inference
            with torch.no_grad() if 'torch' in sys.modules else contextmanager(lambda: None)():
                outputs = self.model(**batch_input, **kwargs)
            
            # Convert outputs to list of dictionaries
            return self._process_batch_output(outputs, len(batch))
            
        except Exception as e:
            logger.error(f"Inference batch failed: {e}")
            # Return empty results for failed batch
            return [{'error': str(e)} for _ in batch]
    
    def _prepare_batch_input(self, batch: List[Dict]) -> Dict:
        """Prepare batch input for model."""
        # This is a simplified implementation - would need to be customized for specific models
        batch_input = {}
        
        # Collect all keys from the batch
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())
        
        # Stack tensors for each key
        for key in all_keys:
            values = []
            for item in batch:
                if key in item:
                    values.append(item[key])
                else:
                    # Pad with appropriate default values
                    if key == 'input_ids':
                        values.append([])
                    elif key == 'attention_mask':
                        values.append([])
                    else:
                        values.append(None)
            
            # Convert to appropriate tensor format
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                # Pad sequences to same length
                max_length = max(len(v) if v else 0 for v in values)
                padded_values = []
                for v in values:
                    if v:
                        padded = v + [0] * (max_length - len(v))
                        padded_values.append(padded)
                    else:
                        padded_values.append([0] * max_length)
                
                # Convert to tensor (this would use the actual tensor library)
                batch_input[key] = padded_values
        
        return batch_input
    
    def _process_batch_output(self, outputs, batch_size: int) -> List[Dict]:
        """Process batch output into list of individual results."""
        results = []
        
        # This is a simplified implementation - would need to be customized for specific outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            for i in range(batch_size):
                result = {
                    'logits': logits[i].tolist() if hasattr(logits[i], 'tolist') else logits[i],
                }
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    result['hidden_states'] = outputs.hidden_states[-1][i].tolist()
                results.append(result)
        else:
            # Fallback for unknown output format
            for i in range(batch_size):
                results.append({'output': f'result_{i}'})
        
        return results
    
    def inference_streaming(self, inputs: List[Dict], **kwargs) -> Iterator[Dict]:
        """Run inference with streaming results."""
        batches = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        
        # Submit all batches
        futures = []
        for batch in batches:
            future = self.executor.submit(self._inference_batch, batch, **kwargs)
            futures.append(future)
        
        # Yield results as they complete
        for future in as_completed(futures):
            batch_results = future.result()
            for result in batch_results:
                yield result
    
    def inference_with_priority(self, inputs: List[Dict], priority: int = 0, **kwargs) -> List[Dict]:
        """Run inference with priority queuing."""
        request = {
            'inputs': inputs,
            'kwargs': kwargs,
            'priority': priority,
            'timestamp': time.time(),
            'future': self.executor.submit(self._inference_batch, inputs, **kwargs)
        }
        
        with self._queue_lock:
            self._request_queue.append(request)
            self._request_queue.sort(key=lambda x: (-x['priority'], x['timestamp']))
        
        return request['future'].result()
    
    def _update_inference_stats(self, num_inferences: int, duration: float):
        """Update inference statistics."""
        self._stats['total_inferences'] += num_inferences
        self._stats['total_time'] += duration
        if self._stats['total_inferences'] > 0:
            self._stats['avg_latency'] = self._stats['total_time'] / self._stats['total_inferences']
            self._stats['avg_throughput'] = self._stats['total_inferences'] / self._stats['total_time']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        with self._queue_lock:
            queue_size = len(self._request_queue)
        
        return {
            **self._stats,
            'current_queue_size': queue_size,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }
    
    def close(self):
        """Close the parallel inference engine."""
        self.executor.shutdown(wait=True)


class AsyncInferenceEngine:
    """Asynchronous inference engine for non-blocking operations."""
    
    def __init__(self, model, max_concurrent: int = 10):
        self.model = model
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def inference_async(self, inputs: Dict, **kwargs) -> Dict:
        """Run asynchronous inference."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._blocking_inference,
            inputs,
            kwargs
        )
    
    def _blocking_inference(self, inputs: Dict, kwargs: Dict) -> Dict:
        """Run blocking inference with semaphore."""
        with self.semaphore:
            try:
                # Run inference
                outputs = self.model(**inputs, **kwargs)
                return {'outputs': outputs, 'error': None}
            except Exception as e:
                return {'outputs': None, 'error': str(e)}
    
    async def inference_batch_async(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        """Run asynchronous batch inference."""
        import asyncio
        
        tasks = []
        for input_data in inputs:
            task = self.inference_async(input_data, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Close the async inference engine."""
        self.executor.shutdown(wait=True)


class PrewarmingCache:
    """Cache with automatic prewarming capabilities."""
    
    def __init__(self, max_size: int = 1000, prewarm_ratio: float = 0.1):
        self.cache = {}
        self.max_size = max_size
        self.prewarm_ratio = prewarm_ratio
        self.access_count = {}
        self.lock = threading.RLock()
        self.prewarmer = ThreadPoolExecutor(max_workers=2)
    
    def get(self, key: str, loader_func: Callable[[], Any] = None) -> Any:
        """Get value from cache or load it."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
        
        if loader_func is None:
            return None  # Return None for missing keys when no loader provided
        
        # Load value
        value = loader_func()
        
        with self.lock:
            # Add to cache
            if len(self.cache) >= self.max_size:
                # Evict least accessed items
                sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
                for k, _ in sorted_items[:int(self.max_size * 0.1)]:
                    if k in self.cache:
                        del self.cache[k]
                    if k in self.access_count:
                        del self.access_count[k]
            
            self.cache[key] = value
            self.access_count[key] = 1
        
        # Trigger prewarming for related keys
        self._schedule_prewarming(key, loader_func)
        
        return value
    
    def _schedule_prewarming(self, key: str, loader_func: Callable[[], Any]):
        """Schedule prewarming of related keys."""
        # This is a simple implementation - could be enhanced with ML-based prediction
        related_keys = [f"{key}_variant_{i}" for i in range(3)]
        
        for related_key in related_keys:
            if related_key not in self.cache:
                self.prewarmer.submit(self._prewarm_key, related_key, loader_func)
    
    def _prewarm_key(self, key: str, loader_func: Callable[[], Any]):
        """Prewarm a specific key."""
        try:
            value = loader_func()
            with self.lock:
                if len(self.cache) < self.max_size and key not in self.cache:
                    self.cache[key] = value
                    self.access_count[key] = 0
        except Exception as e:
            logger.debug(f"Prewarming failed for key {key}: {e}")
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
    
    def close(self):
        """Close the prewarming cache."""
        self.prewarmer.shutdown(wait=True)
    
    def put(self, key: str, value: Any):
        """Put a value into the cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evict least accessed items
                sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
                for k, _ in sorted_items[:int(self.max_size * 0.1)]:
                    if k in self.cache:
                        del self.cache[k]
                    if k in self.access_count:
                        del self.access_count[k]
            
            self.cache[key] = value
            self.access_count[key] = self.access_count.get(key, 0) + 1
    
    async def prewarm(self, keys: List[str], loader_func: Callable[[str], Any]):
        """Prewarm cache with given keys."""
        async def async_load(key):
            try:
                value = loader_func(key)
                self.put(key, value)
            except Exception as e:
                logger.debug(f"Prewarming failed for key {key}: {e}")
        
        # Create async tasks for all keys
        tasks = [async_load(key) for key in keys]
        await asyncio.gather(*tasks, return_exceptions=True)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, enabled: bool = True):
        self.metrics = {}
        self.lock = threading.RLock()
        self.enabled = enabled
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self.lock:
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = {
                        'count': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'avg_time': 0.0,
                        'total_memory': 0.0,
                        'avg_memory': 0.0
                    }
                
                metrics = self.metrics[operation_name]
                metrics['count'] += 1
                metrics['total_time'] += duration
                metrics['min_time'] = min(metrics['min_time'], duration)
                metrics['max_time'] = max(metrics['max_time'], duration)
                metrics['avg_time'] = metrics['total_time'] / metrics['count']
                metrics['total_memory'] += memory_delta
                metrics['avg_memory'] = metrics['total_memory'] / metrics['count']
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        with self.lock:
            return {k: v.copy() for k, v in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time an operation (alias for measure)."""
        with self.measure(operation_name):
            yield
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB (public method)."""
        return self._get_memory_usage()
    
    def record_metric(self, metric_name: str, value: float):
        """Record a custom metric."""
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = {
                    'count': 0,
                    'total': 0.0,
                    'min': float('inf'),
                    'max': 0.0,
                    'avg': 0.0,
                    'values': []
                }
            
            metrics = self.metrics[metric_name]
            metrics['count'] += 1
            metrics['total'] += value
            metrics['min'] = min(metrics['min'], value)
            metrics['max'] = max(metrics['max'], value)
            metrics['avg'] = metrics['total'] / metrics['count']
            metrics['values'].append(value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics (alias for get_report)."""
        return self.get_report()


# Global instances
_batch_processor = None
_parallel_tokenizer = None
_parallel_inference_engine = None
_performance_monitor = PerformanceMonitor()


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor


def get_parallel_tokenizer(tokenizer) -> ParallelTokenizer:
    """Get parallel tokenizer instance."""
    global _parallel_tokenizer
    if _parallel_tokenizer is None or _parallel_tokenizer.tokenizer != tokenizer:
        if _parallel_tokenizer:
            _parallel_tokenizer.close()
        _parallel_tokenizer = ParallelTokenizer(tokenizer)
    return _parallel_tokenizer


def get_parallel_inference_engine(model) -> ParallelInferenceEngine:
    """Get parallel inference engine instance."""
    global _parallel_inference_engine
    if _parallel_inference_engine is None or _parallel_inference_engine.model != model:
        if _parallel_inference_engine:
            _parallel_inference_engine.close()
        _parallel_inference_engine = ParallelInferenceEngine(model)
    return _parallel_inference_engine


# Import required modules
import os
import sys


# Decorators for easy use
def batched(batch_size: int = 32):
    """Decorator to make function support batching."""
    def decorator(func):
        func._supports_batching = True
        func._batch_size = batch_size
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if this is a batch call
            if len(args) > 0 and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], (list, tuple)):
                # This is a batch call
                batch_args, batch_kwargs = args[0], args[1] if len(args) > 1 else kwargs
                results = []
                for i, (call_args, call_kwargs) in enumerate(zip(batch_args, batch_kwargs)):
                    result = func(*call_args, **call_kwargs)
                    results.append(result)
                return results
            else:
                # Regular call
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def lazy_init(loader_func):
    """Decorator for lazy initialization."""
    loader = LazyLoader(loader_func)
    
    def wrapper():
        return loader.get()
    
    wrapper.is_loaded = loader.is_loaded
    wrapper.reset = loader.reset
    return wrapper


def pooled(max_pool_size: int = 100):
    """Decorator to create pooled resources."""
    def decorator(factory_func):
        pool = ObjectPool(factory_func, max_pool_size)
        
        @contextmanager
        def get_pooled_resource():
            obj = pool.acquire()
            try:
                yield obj
            finally:
                pool.release(obj)
        
        get_pooled_resource.stats = pool.stats
        get_pooled_resource.clear = pool.clear
        return get_pooled_resource
    
    return decorator


def monitored(func_or_name=None):
    """Decorator to monitor function performance."""
    def decorator(func):
        operation_name = func_or_name if isinstance(func_or_name, str) else func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _performance_monitor.measure(operation_name):
                return func(*args, **kwargs)
        return wrapper
    
    if callable(func_or_name):
        # Called as @monitored without parentheses
        return decorator(func_or_name)
    else:
        # Called as @monitored() or @monitored("name")
        return decorator


# Convenience functions
def batch_process(items: List[Any], processor_func: Callable, batch_size: int = 32) -> List[Any]:
    """Convenience function for batch processing."""
    processor = get_batch_processor()
    processor.config.batch_size = batch_size
    
    # Run async process_batch in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(processor.process_batch(items, processor_func))


def get_from_pool(pool_name: str, factory_func: Callable = None):
    """Get object from named pool."""
    if not hasattr(get_from_pool, '_pools'):
        get_from_pool._pools = {}
    
    if pool_name not in get_from_pool._pools:
        if factory_func is None:
            # Use a default factory that creates simple objects
            factory_func = lambda: object()
        get_from_pool._pools[pool_name] = ObjectPool(factory_func)
    
    return get_from_pool._pools[pool_name].acquire()


def return_to_pool(pool_name: str, obj: Any):
    """Return object to named pool."""
    if hasattr(get_from_pool, '_pools') and pool_name in get_from_pool._pools:
        get_from_pool._pools[pool_name].release(obj)


def prewarm_cache(keys: List[str], loader_func: Callable, cache_name: str = "default"):
    """Prewarm cache with a specific key."""
    if not hasattr(prewarm_cache, '_caches'):
        prewarm_cache._caches = {}
    
    if cache_name not in prewarm_cache._caches:
        prewarm_cache._caches[cache_name] = PrewarmingCache()
    
    cache = prewarm_cache._caches[cache_name]
    for key in keys:
        cache.get(key, lambda k=key: loader_func(k))


def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    raw_stats = _performance_monitor.get_report()
    
    # Format stats to match test expectations
    if not raw_stats:
        return {}
    
    # Group stats by type 
    timing_stats = {}
    memory_stats = {}
    
    for operation_name, metrics in raw_stats.items():
        if 'total_time' in metrics:
            timing_stats[operation_name] = {
                'count': metrics.get('count', 0),
                'total_time': metrics.get('total_time', 0.0),
                'avg_time': metrics.get('avg_time', 0.0),
                'min_time': metrics.get('min_time', 0.0),
                'max_time': metrics.get('max_time', 0.0)
            }
        
        if 'total_memory' in metrics:
            memory_stats[operation_name] = {
                'total_memory': metrics.get('total_memory', 0.0),
                'avg_memory': metrics.get('avg_memory', 0.0)
            }
    
    result = {}
    if timing_stats:
        result['timing'] = timing_stats
    if memory_stats:
        result['memory'] = memory_stats
    
    return result


def optimize_memory():
    """Run memory optimization."""
    initial_memory = _performance_monitor.get_memory_usage()
    
    gc.collect()
    
    # Clear caches if they exist
    if hasattr(prewarm_cache, '_caches'):
        for cache in prewarm_cache._caches.values():
            cache.clear()
    
    # Clear pools
    if hasattr(get_from_pool, '_pools'):
        for pool in get_from_pool._pools.values():
            pool.clear()
    
    final_memory = _performance_monitor.get_memory_usage()
    freed_memory = initial_memory - final_memory
    return max(0, freed_memory)  # Return positive value for freed memory


def cleanup_performance_resources():
    """Cleanup all performance resources."""
    global _batch_processor, _parallel_tokenizer, _parallel_inference_engine
    
    if _batch_processor:
        _batch_processor.close()
        _batch_processor = None
    
    if _parallel_tokenizer:
        _parallel_tokenizer.close()
        _parallel_tokenizer = None
    
    if _parallel_inference_engine:
        _parallel_inference_engine.close()
        _parallel_inference_engine = None
    
    if hasattr(prewarm_cache, '_cache'):
        prewarm_cache._cache.close()
        delattr(prewarm_cache, '_cache')
    
    _performance_monitor.reset()
