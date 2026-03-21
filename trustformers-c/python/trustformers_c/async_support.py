"""
Asynchronous support for TrustformeRS-C
Provides async/await interfaces for non-blocking inference and batch processing
"""

import asyncio
import concurrent.futures
import threading
import time
import queue
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .core import TrustformersError, PerformanceOptimizer, PerformanceConfig
from .numpy_integration import NumpyTensor, ensure_numpy_array, TensorOperations

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class InferenceRequest:
    """Inference request data structure"""
    request_id: str
    input_data: Union[np.ndarray, NumpyTensor]
    priority: RequestPriority = RequestPriority.NORMAL
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class InferenceResult:
    """Inference result data structure"""
    request_id: str
    output_data: Union[np.ndarray, NumpyTensor]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

class AsyncInference:
    """
    Asynchronous inference engine with batching and priority handling
    """
    
    def __init__(self, 
                 model_function: Callable,
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.1,
                 num_workers: int = 1,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize async inference engine
        
        Args:
            model_function: Function to run inference (can be sync or async)
            max_batch_size: Maximum batch size for batching
            max_wait_time: Maximum time to wait for batch accumulation (seconds)
            num_workers: Number of worker threads
            performance_config: Performance optimization configuration
        """
        self.model_function = model_function
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        
        # Initialize performance optimizer
        if performance_config is not None:
            self.performance_optimizer = PerformanceOptimizer(performance_config)
        else:
            self.performance_optimizer = None
        
        # Request management
        self.request_queue = asyncio.Queue()
        self.result_futures = {}
        self.pending_requests = []
        
        # Worker management
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        self.running = False
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'average_batch_size': 0.0,
            'throughput_per_second': 0.0,
        }
        
        # Synchronization
        self.lock = asyncio.Lock()
    
    async def start(self):
        """Start the async inference engine"""
        if self.running:
            return
        
        self.running = True
        
        # Start batch processor
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self):
        """Stop the async inference engine"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel batch processor
        if hasattr(self, 'batch_processor_task'):
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def infer(self, 
                   input_data: Union[np.ndarray, NumpyTensor],
                   priority: RequestPriority = RequestPriority.NORMAL,
                   metadata: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Submit inference request and wait for result
        
        Args:
            input_data: Input data for inference
            priority: Request priority
            metadata: Additional metadata
            
        Returns:
            Inference result
        """
        request_id = f"req-{int(time.time() * 1000000)}-{id(input_data)}"
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Create future for result
        future = asyncio.Future()
        self.result_futures[request_id] = future
        
        # Submit request
        await self.request_queue.put(request)
        
        # Update stats
        async with self.lock:
            self.stats['total_requests'] += 1
        
        # Wait for result
        try:
            result = await future
            return result
        finally:
            # Clean up future
            self.result_futures.pop(request_id, None)
    
    async def infer_async(self,
                         input_data: Union[np.ndarray, NumpyTensor],
                         callback: Optional[Callable] = None,
                         priority: RequestPriority = RequestPriority.NORMAL,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit inference request without waiting for result
        
        Args:
            input_data: Input data for inference
            callback: Callback function to call when result is ready
            priority: Request priority
            metadata: Additional metadata
            
        Returns:
            Request ID
        """
        request_id = f"req-{int(time.time() * 1000000)}-{id(input_data)}"
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Submit request
        await self.request_queue.put(request)
        
        # Update stats
        async with self.lock:
            self.stats['total_requests'] += 1
        
        return request_id
    
    async def _batch_processor(self):
        """Process requests in batches"""
        while self.running:
            try:
                # Collect requests for batching
                batch_requests = []
                batch_start_time = time.time()
                
                # Get first request (blocking)
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=self.max_wait_time
                    )
                    batch_requests.append(request)
                except asyncio.TimeoutError:
                    continue
                
                # Collect additional requests (non-blocking)
                while (len(batch_requests) < self.max_batch_size and 
                       time.time() - batch_start_time < self.max_wait_time):
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(), 
                            timeout=0.01
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch_requests:
                    # Sort by priority
                    batch_requests.sort(key=lambda x: x.priority.value, reverse=True)
                    
                    # Process batch
                    asyncio.create_task(self._process_batch(batch_requests))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch_requests: List[InferenceRequest]):
        """Process a batch of requests"""
        start_time = time.time()
        
        try:
            # Prepare batch input
            batch_input = []
            for request in batch_requests:
                input_array = ensure_numpy_array(request.input_data)
                batch_input.append(input_array)
            
            # Stack inputs into batch
            if batch_input:
                batch_tensor = np.stack(batch_input, axis=0)
            else:
                return
            
            # Run inference
            if asyncio.iscoroutinefunction(self.model_function):
                batch_output = await self.model_function(batch_tensor)
            else:
                loop = asyncio.get_event_loop()
                batch_output = await loop.run_in_executor(
                    self.executor, 
                    self.model_function, 
                    batch_tensor
                )
            
            processing_time = time.time() - start_time
            
            # Split batch output and create results
            for i, request in enumerate(batch_requests):
                try:
                    if isinstance(batch_output, np.ndarray):
                        output_data = batch_output[i]
                    else:
                        output_data = batch_output[i] if hasattr(batch_output, '__getitem__') else batch_output
                    
                    result = InferenceResult(
                        request_id=request.request_id,
                        output_data=output_data,
                        processing_time=processing_time,
                        metadata=request.metadata
                    )
                    
                    # Handle result
                    await self._handle_result(request, result)
                    
                except Exception as e:
                    error_result = InferenceResult(
                        request_id=request.request_id,
                        output_data=None,
                        processing_time=processing_time,
                        metadata=request.metadata,
                        error=e
                    )
                    await self._handle_result(request, error_result)
            
            # Update stats
            async with self.lock:
                self.stats['completed_requests'] += len(batch_requests)
                self.stats['average_batch_size'] = (
                    (self.stats['average_batch_size'] * (self.stats['completed_requests'] - len(batch_requests)) + 
                     len(batch_requests)) / self.stats['completed_requests']
                )
                self.stats['average_latency'] = (
                    (self.stats['average_latency'] * (self.stats['completed_requests'] - len(batch_requests)) + 
                     processing_time) / self.stats['completed_requests']
                )
        
        except Exception as e:
            # Handle batch processing error
            for request in batch_requests:
                error_result = InferenceResult(
                    request_id=request.request_id,
                    output_data=None,
                    processing_time=time.time() - start_time,
                    metadata=request.metadata,
                    error=e
                )
                await self._handle_result(request, error_result)
            
            async with self.lock:
                self.stats['failed_requests'] += len(batch_requests)
    
    async def _handle_result(self, request: InferenceRequest, result: InferenceResult):
        """Handle inference result"""
        # Call callback if provided
        if request.callback:
            try:
                if asyncio.iscoroutinefunction(request.callback):
                    await request.callback(result)
                else:
                    request.callback(result)
            except Exception as e:
                print(f"Error in callback for request {request.request_id}: {e}")
        
        # Set future result if waiting
        if request.request_id in self.result_futures:
            future = self.result_futures[request.request_id]
            if not future.done():
                future.set_result(result)
    
    async def _worker(self, worker_name: str):
        """Worker thread (placeholder for future extensions)"""
        while self.running:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return self.stats.copy()
    
    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all pending requests to complete"""
        start_time = time.time()
        
        while self.result_futures:
            if timeout and time.time() - start_time > timeout:
                raise asyncio.TimeoutError("Timeout waiting for completion")
            
            await asyncio.sleep(0.01)

class AsyncPipeline:
    """
    Asynchronous pipeline for chaining multiple inference operations
    """
    
    def __init__(self, 
                 stages: List[Callable],
                 stage_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize async pipeline
        
        Args:
            stages: List of processing stages (functions)
            stage_configs: Configuration for each stage
        """
        self.stages = stages
        self.stage_configs = stage_configs or [{} for _ in stages]
        self.stage_engines = []
        
        # Create async inference engines for each stage
        for i, (stage, config) in enumerate(zip(self.stages, self.stage_configs)):
            engine = AsyncInference(
                model_function=stage,
                max_batch_size=config.get('max_batch_size', 32),
                max_wait_time=config.get('max_wait_time', 0.1),
                num_workers=config.get('num_workers', 1),
                performance_config=config.get('performance_config')
            )
            self.stage_engines.append(engine)
    
    async def start(self):
        """Start all pipeline stages"""
        for engine in self.stage_engines:
            await engine.start()
    
    async def stop(self):
        """Stop all pipeline stages"""
        for engine in self.stage_engines:
            await engine.stop()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def process(self, 
                     input_data: Union[np.ndarray, NumpyTensor],
                     priority: RequestPriority = RequestPriority.NORMAL) -> InferenceResult:
        """
        Process input through the entire pipeline
        
        Args:
            input_data: Input data
            priority: Request priority
            
        Returns:
            Final result
        """
        current_data = input_data
        
        for i, engine in enumerate(self.stage_engines):
            result = await engine.infer(current_data, priority=priority)
            
            if result.error:
                raise result.error
            
            current_data = result.output_data
        
        return InferenceResult(
            request_id=f"pipeline-{int(time.time() * 1000000)}",
            output_data=current_data,
            processing_time=0.0,  # Would need to track cumulative time
            metadata={}
        )

class AsyncBatchProcessor:
    """
    Asynchronous batch processor for efficient bulk processing
    """
    
    def __init__(self, 
                 processing_function: Callable,
                 batch_size: int = 32,
                 max_wait_time: float = 1.0):
        """
        Initialize async batch processor
        
        Args:
            processing_function: Function to process batches
            batch_size: Size of batches
            max_wait_time: Maximum wait time for batch accumulation
        """
        self.processing_function = processing_function
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        self.input_queue = asyncio.Queue()
        self.running = False
        self.processor_task = None
    
    async def start(self):
        """Start the batch processor"""
        if self.running:
            return
        
        self.running = True
        self.processor_task = asyncio.create_task(self._process_batches())
    
    async def stop(self):
        """Stop the batch processor"""
        if not self.running:
            return
        
        self.running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def add_to_batch(self, data: Union[np.ndarray, NumpyTensor]) -> None:
        """
        Add data to processing batch
        
        Args:
            data: Data to add to batch
        """
        await self.input_queue.put(data)
    
    async def _process_batches(self):
        """Process batches continuously"""
        while self.running:
            try:
                batch = []
                batch_start_time = time.time()
                
                # Collect batch
                while (len(batch) < self.batch_size and 
                       time.time() - batch_start_time < self.max_wait_time):
                    try:
                        data = await asyncio.wait_for(
                            self.input_queue.get(), 
                            timeout=0.1
                        )
                        batch.append(ensure_numpy_array(data))
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if not empty
                if batch:
                    batch_tensor = np.stack(batch, axis=0)
                    
                    if asyncio.iscoroutinefunction(self.processing_function):
                        await self.processing_function(batch_tensor)
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, 
                            self.processing_function, 
                            batch_tensor
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)

# Utility functions
async def run_async_inference(model_function: Callable,
                            input_data: Union[np.ndarray, NumpyTensor],
                            batch_size: int = 32,
                            max_wait_time: float = 0.1) -> InferenceResult:
    """
    Convenience function for single async inference
    
    Args:
        model_function: Model function to run
        input_data: Input data
        batch_size: Batch size for processing
        max_wait_time: Maximum wait time
        
    Returns:
        Inference result
    """
    async with AsyncInference(
        model_function=model_function,
        max_batch_size=batch_size,
        max_wait_time=max_wait_time
    ) as engine:
        return await engine.infer(input_data)

async def run_async_pipeline(stages: List[Callable],
                           input_data: Union[np.ndarray, NumpyTensor],
                           stage_configs: Optional[List[Dict[str, Any]]] = None) -> InferenceResult:
    """
    Convenience function for single async pipeline execution
    
    Args:
        stages: List of processing stages
        input_data: Input data
        stage_configs: Configuration for each stage
        
    Returns:
        Final result
    """
    async with AsyncPipeline(stages=stages, stage_configs=stage_configs) as pipeline:
        return await pipeline.process(input_data)