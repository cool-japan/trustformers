"""
Model Serving Infrastructure for TrustformeRS

This module provides a comprehensive FastAPI-based serving infrastructure for deploying
TrustformeRS models in production environments, including batching, load balancing,
model versioning, and A/B testing capabilities.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from urllib.parse import urlparse
import weakref

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Internal imports
from . import AutoModel, AutoTokenizer, Tensor
from .utils import create_logger

# Configure logging
logger = create_logger(__name__)

class ModelStatus(Enum):
    """Model deployment status"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"
    DEPRECATED = "deprecated"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"

@dataclass
class ModelConfig:
    """Configuration for a deployed model"""
    model_id: str
    model_path: str
    version: str
    max_batch_size: int = 8
    max_sequence_length: int = 512
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    weight: float = 1.0  # For load balancing
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelInstance:
    """Running model instance"""
    config: ModelConfig
    model: Any
    tokenizer: Any
    status: ModelStatus = ModelStatus.LOADING
    load_time: float = 0.0
    request_count: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    warmup_completed: bool = False

class InferenceRequest(BaseModel):
    """Request model for inference"""
    model_id: str
    inputs: Union[str, List[str]]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    version: Optional[str] = None
    priority: int = Field(default=0, ge=-100, le=100)
    request_id: Optional[str] = None
    stream: bool = False
    timeout: float = Field(default=30.0, gt=0)

    @validator('request_id', pre=True, always=True)
    def set_request_id(cls, v):
        return v or str(uuid.uuid4())

class InferenceResponse(BaseModel):
    """Response model for inference"""
    request_id: str
    model_id: str
    version: str
    outputs: Union[str, List[str], Dict[str, Any]]
    usage: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: Dict[str, Dict[str, Any]]
    system: Dict[str, Any]
    timestamp: float

class ABTestConfig(BaseModel):
    """A/B test configuration"""
    test_name: str
    model_a: str
    model_b: str
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchRequest:
    """Batched inference request"""
    def __init__(self, request: InferenceRequest):
        self.request = request
        self.future: asyncio.Future = asyncio.Future()
        self.timestamp = time.time()
        self.priority = request.priority

    def __lt__(self, other):
        # Higher priority first, then FIFO
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp

class ModelVersionManager:
    """Manages model versions and rollouts"""
    
    def __init__(self):
        self.versions: Dict[str, Dict[str, ModelConfig]] = defaultdict(dict)
        self.active_versions: Dict[str, str] = {}
        self.rollout_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def register_version(self, config: ModelConfig) -> bool:
        """Register a new model version"""
        with self._lock:
            model_id = config.model_id
            version = config.version
            
            if model_id in self.versions and version in self.versions[model_id]:
                logger.warning(f"Version {version} for model {model_id} already exists")
                return False
            
            self.versions[model_id][version] = config
            
            # Set as active if it's the first version
            if model_id not in self.active_versions:
                self.active_versions[model_id] = version
                
            logger.info(f"Registered model {model_id} version {version}")
            return True
    
    def get_active_version(self, model_id: str) -> Optional[str]:
        """Get the active version for a model"""
        return self.active_versions.get(model_id)
    
    def set_active_version(self, model_id: str, version: str) -> bool:
        """Set active version for a model"""
        with self._lock:
            if model_id not in self.versions or version not in self.versions[model_id]:
                return False
            
            self.active_versions[model_id] = version
            logger.info(f"Set active version for {model_id} to {version}")
            return True
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions for a model"""
        return list(self.versions.get(model_id, {}).keys())
    
    def get_config(self, model_id: str, version: Optional[str] = None) -> Optional[ModelConfig]:
        """Get configuration for a specific model version"""
        if model_id not in self.versions:
            return None
        
        if version is None:
            version = self.active_versions.get(model_id)
        
        return self.versions[model_id].get(version)

class LoadBalancer:
    """Load balancer for model instances"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: Dict[str, List[ModelInstance]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = Lock()
    
    def add_instance(self, instance: ModelInstance):
        """Add a model instance to the load balancer"""
        with self._lock:
            model_key = f"{instance.config.model_id}:{instance.config.version}"
            self.instances[model_key].append(instance)
            logger.info(f"Added instance for {model_key}")
    
    def remove_instance(self, model_id: str, version: str, instance: ModelInstance):
        """Remove a model instance from the load balancer"""
        with self._lock:
            model_key = f"{model_id}:{version}"
            if model_key in self.instances and instance in self.instances[model_key]:
                self.instances[model_key].remove(instance)
                logger.info(f"Removed instance for {model_key}")
    
    def get_instance(self, model_id: str, version: str) -> Optional[ModelInstance]:
        """Get the best instance based on load balancing strategy"""
        model_key = f"{model_id}:{version}"
        instances = self.instances.get(model_key, [])
        
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [inst for inst in instances if inst.status == ModelStatus.READY]
        if not healthy_instances:
            return None
        
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                instance = self._round_robin(healthy_instances, model_key)
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                instance = self._least_loaded(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                instance = self._weighted_round_robin(healthy_instances, model_key)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                instance = self._best_response_time(healthy_instances, model_key)
            else:  # RANDOM
                import random
                instance = random.choice(healthy_instances)
            
            instance.last_used = time.time()
            return instance
    
    def _round_robin(self, instances: List[ModelInstance], model_key: str) -> ModelInstance:
        """Round robin selection"""
        counter = self.counters[model_key]
        self.counters[model_key] = (counter + 1) % len(instances)
        return instances[counter]
    
    def _least_loaded(self, instances: List[ModelInstance]) -> ModelInstance:
        """Select least loaded instance"""
        return min(instances, key=lambda x: x.request_count)
    
    def _weighted_round_robin(self, instances: List[ModelInstance], model_key: str) -> ModelInstance:
        """Weighted round robin based on instance weights"""
        total_weight = sum(inst.config.weight for inst in instances)
        if total_weight == 0:
            return self._round_robin(instances, model_key)
        
        # Simplified weighted selection
        weights = [inst.config.weight / total_weight for inst in instances]
        import random
        return random.choices(instances, weights=weights, k=1)[0]
    
    def _best_response_time(self, instances: List[ModelInstance], model_key: str) -> ModelInstance:
        """Select instance with best response time"""
        return min(instances, key=lambda x: x.avg_response_time)
    
    def update_response_time(self, instance: ModelInstance, response_time: float):
        """Update response time statistics"""
        model_key = f"{instance.config.model_id}:{instance.config.version}"
        self.response_times[model_key].append(response_time)
        
        # Update moving average
        times = self.response_times[model_key]
        instance.avg_response_time = sum(times) / len(times)

class BatchProcessor:
    """Processes inference requests in batches"""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queues: Dict[str, asyncio.PriorityQueue] = defaultdict(asyncio.PriorityQueue)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
    
    async def add_request(self, request: InferenceRequest) -> InferenceResponse:
        """Add a request to the batch queue"""
        model_key = f"{request.model_id}:{request.version or 'default'}"
        batch_request = BatchRequest(request)
        
        await self.queues[model_key].put(batch_request)
        
        # Start processing task if not already running
        if model_key not in self.processing_tasks:
            self.processing_tasks[model_key] = asyncio.create_task(
                self._process_batch_queue(model_key)
            )
        
        # Wait for response
        try:
            return await asyncio.wait_for(batch_request.future, timeout=request.timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
    
    async def _process_batch_queue(self, model_key: str):
        """Process batched requests for a specific model"""
        queue = self.queues[model_key]
        
        while not self._shutdown:
            try:
                batch = []
                batch_start = time.time()
                
                # Collect requests for batch
                while len(batch) < self.max_batch_size:
                    try:
                        # Wait for requests with timeout
                        timeout = self.max_wait_time - (time.time() - batch_start)
                        if timeout <= 0 and batch:
                            break
                        
                        batch_request = await asyncio.wait_for(
                            queue.get(), timeout=max(0.001, timeout)
                        )
                        batch.append(batch_request)
                        
                    except asyncio.TimeoutError:
                        if batch:
                            break
                        continue
                
                if batch:
                    await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error processing batch for {model_key}: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        if not batch:
            return
        
        # Group by model and version
        first_request = batch[0].request
        model_id = first_request.model_id
        version = first_request.version
        
        try:
            # Get model instance from serving manager
            serving_manager = ServingManager.get_instance()
            instance = serving_manager.load_balancer.get_instance(model_id, version or "default")
            
            if not instance:
                error_response = HTTPException(
                    status_code=404, 
                    detail=f"Model {model_id}:{version} not available"
                )
                for batch_request in batch:
                    batch_request.future.set_exception(error_response)
                return
            
            # Prepare batch inputs
            inputs = [req.request.inputs for req in batch]
            parameters = batch[0].request.parameters  # Use first request's parameters
            
            # Record start time
            start_time = time.time()
            
            # Process batch
            if hasattr(instance.model, 'generate_batch'):
                # Custom batch generation
                outputs = instance.model.generate_batch(
                    inputs=inputs,
                    tokenizer=instance.tokenizer,
                    **parameters
                )
            else:
                # Fallback to individual processing
                outputs = []
                for input_text in inputs:
                    if isinstance(input_text, str):
                        tokens = instance.tokenizer.encode(input_text)
                        output_tokens = instance.model.generate(
                            tokens, 
                            max_length=parameters.get('max_length', 50)
                        )
                        output_text = instance.tokenizer.decode(output_tokens)
                        outputs.append(output_text)
                    else:
                        outputs.append(str(input_text))  # Fallback
            
            # Calculate timing
            processing_time = time.time() - start_time
            
            # Update instance statistics
            instance.request_count += len(batch)
            total_tokens = sum(len(str(inp).split()) for inp in inputs)
            instance.total_tokens += total_tokens
            
            # Update load balancer
            serving_manager.load_balancer.update_response_time(instance, processing_time)
            
            # Send responses
            for i, batch_request in enumerate(batch):
                try:
                    response = InferenceResponse(
                        request_id=batch_request.request.request_id,
                        model_id=model_id,
                        version=version or instance.config.version,
                        outputs=outputs[i] if i < len(outputs) else "",
                        usage={
                            "prompt_tokens": len(str(batch_request.request.inputs).split()),
                            "completion_tokens": len(str(outputs[i]).split()) if i < len(outputs) else 0,
                            "total_tokens": len(str(batch_request.request.inputs).split()) + 
                                          (len(str(outputs[i]).split()) if i < len(outputs) else 0)
                        },
                        timing={
                            "processing_time": processing_time,
                            "queue_time": start_time - batch_request.timestamp,
                            "total_time": time.time() - batch_request.timestamp
                        }
                    )
                    batch_request.future.set_result(response)
                except Exception as e:
                    batch_request.future.set_exception(e)
                    
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for batch_request in batch:
                if not batch_request.future.done():
                    batch_request.future.set_exception(
                        HTTPException(status_code=500, detail=str(e))
                    )
    
    async def shutdown(self):
        """Shutdown the batch processor"""
        self._shutdown = True
        for task in self.processing_tasks.values():
            task.cancel()
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)

class ABTestManager:
    """Manages A/B testing for models"""
    
    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "model_a_requests": 0,
            "model_b_requests": 0,
            "model_a_avg_time": 0.0,
            "model_b_avg_time": 0.0,
            "model_a_errors": 0,
            "model_b_errors": 0
        })
        self._lock = Lock()
    
    def create_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test"""
        with self._lock:
            if config.test_name in self.tests:
                return False
            
            self.tests[config.test_name] = config
            logger.info(f"Created A/B test: {config.test_name}")
            return True
    
    def route_request(self, test_name: str, request_id: str) -> Optional[str]:
        """Route request to model A or B based on test configuration"""
        test = self.tests.get(test_name)
        if not test or not test.enabled:
            return None
        
        # Simple hash-based routing for consistent user experience
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        if (hash_value % 100) / 100.0 < test.traffic_split:
            return test.model_a
        else:
            return test.model_b
    
    def record_result(self, test_name: str, model_id: str, response_time: float, error: bool = False):
        """Record A/B test results"""
        test = self.tests.get(test_name)
        if not test:
            return
        
        with self._lock:
            results = self.results[test_name]
            
            if model_id == test.model_a:
                results["model_a_requests"] += 1
                if error:
                    results["model_a_errors"] += 1
                else:
                    # Update average response time
                    total_requests = results["model_a_requests"]
                    current_avg = results["model_a_avg_time"]
                    results["model_a_avg_time"] = (
                        (current_avg * (total_requests - 1) + response_time) / total_requests
                    )
            elif model_id == test.model_b:
                results["model_b_requests"] += 1
                if error:
                    results["model_b_errors"] += 1
                else:
                    total_requests = results["model_b_requests"]
                    current_avg = results["model_b_avg_time"]
                    results["model_b_avg_time"] = (
                        (current_avg * (total_requests - 1) + response_time) / total_requests
                    )
    
    def get_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get results for an A/B test"""
        if test_name not in self.tests:
            return None
        
        return dict(self.results[test_name])

class MetricsCollector:
    """Collects and exports metrics"""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return
        
        # Request metrics
        self.request_count = Counter(
            'trustformers_requests_total',
            'Total requests processed',
            ['model_id', 'version', 'status']
        )
        
        self.request_duration = Histogram(
            'trustformers_request_duration_seconds',
            'Request processing time',
            ['model_id', 'version']
        )
        
        self.batch_size = Histogram(
            'trustformers_batch_size',
            'Batch size distribution',
            ['model_id', 'version']
        )
        
        # Model metrics
        self.model_load_time = Gauge(
            'trustformers_model_load_time_seconds',
            'Model loading time',
            ['model_id', 'version']
        )
        
        self.active_requests = Gauge(
            'trustformers_active_requests',
            'Currently active requests',
            ['model_id', 'version']
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'trustformers_memory_usage_bytes',
            'Memory usage by component',
            ['component']
        )
    
    def record_request(self, model_id: str, version: str, duration: float, status: str):
        """Record request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.request_count.labels(
            model_id=model_id, 
            version=version, 
            status=status
        ).inc()
        
        self.request_duration.labels(
            model_id=model_id, 
            version=version
        ).observe(duration)
    
    def record_batch_size(self, model_id: str, version: str, size: int):
        """Record batch size metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.batch_size.labels(
            model_id=model_id, 
            version=version
        ).observe(size)
    
    def set_model_load_time(self, model_id: str, version: str, load_time: float):
        """Set model load time metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.model_load_time.labels(
            model_id=model_id, 
            version=version
        ).set(load_time)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        return generate_latest()

class ServingManager:
    """Main serving manager that coordinates all components"""
    
    _instance = None
    _instance_lock = Lock()
    
    def __init__(self):
        self.version_manager = ModelVersionManager()
        self.load_balancer = LoadBalancer()
        self.batch_processor = BatchProcessor()
        self.ab_test_manager = ABTestManager()
        self.metrics = MetricsCollector()
        self.instances: Dict[str, ModelInstance] = {}
        self._shutdown = False
    
    @classmethod
    def get_instance(cls) -> 'ServingManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def load_model(self, config: ModelConfig) -> bool:
        """Load a model instance"""
        try:
            start_time = time.time()
            
            # Create instance
            instance = ModelInstance(config=config)
            model_key = f"{config.model_id}:{config.version}"
            
            # Load model and tokenizer
            logger.info(f"Loading model {model_key}")
            
            if hasattr(AutoModel, 'from_pretrained'):
                instance.model = AutoModel.from_pretrained(
                    config.model_path,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=config.cache_dir,
                    revision=config.revision
                )
            else:
                # Fallback for testing
                instance.model = type('MockModel', (), {
                    'generate': lambda self, tokens, **kwargs: tokens + [1, 2, 3],
                    'generate_batch': lambda self, inputs, tokenizer, **kwargs: [
                        f"Generated response for: {inp}" for inp in inputs
                    ]
                })()
            
            if hasattr(AutoTokenizer, 'from_pretrained'):
                instance.tokenizer = AutoTokenizer.from_pretrained(
                    config.model_path,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=config.cache_dir,
                    revision=config.revision
                )
            else:
                # Fallback for testing
                instance.tokenizer = type('MockTokenizer', (), {
                    'encode': lambda self, text: [1, 2, 3, 4],
                    'decode': lambda self, tokens: f"Decoded: {tokens}"
                })()
            
            # Record load time
            load_time = time.time() - start_time
            instance.load_time = load_time
            instance.status = ModelStatus.READY
            
            # Register with components
            self.version_manager.register_version(config)
            self.load_balancer.add_instance(instance)
            self.instances[model_key] = instance
            
            # Record metrics
            self.metrics.set_model_load_time(config.model_id, config.version, load_time)
            
            logger.info(f"Successfully loaded {model_key} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {config.model_id}:{config.version}: {e}")
            if model_key in self.instances:
                self.instances[model_key].status = ModelStatus.ERROR
                self.instances[model_key].error_message = str(e)
            return False
    
    async def unload_model(self, model_id: str, version: str) -> bool:
        """Unload a model instance"""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.instances:
            return False
        
        instance = self.instances[model_key]
        
        # Remove from load balancer
        self.load_balancer.remove_instance(model_id, version, instance)
        
        # Clean up
        del self.instances[model_key]
        
        logger.info(f"Unloaded model {model_key}")
        return True
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request"""
        # Handle A/B testing
        ab_test = request.parameters.get('ab_test')
        if ab_test:
            routed_model = self.ab_test_manager.route_request(ab_test, request.request_id)
            if routed_model:
                request.model_id = routed_model
        
        # Process through batch processor
        start_time = time.time()
        try:
            response = await self.batch_processor.add_request(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_request(
                request.model_id, 
                request.version or "default", 
                duration, 
                "success"
            )
            
            # Record A/B test results
            if ab_test:
                self.ab_test_manager.record_result(ab_test, request.model_id, duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_request(
                request.model_id, 
                request.version or "default", 
                duration, 
                "error"
            )
            
            if ab_test:
                self.ab_test_manager.record_result(ab_test, request.model_id, duration, error=True)
            
            raise e
    
    def get_health(self) -> HealthResponse:
        """Get health status"""
        models = {}
        for key, instance in self.instances.items():
            models[key] = {
                "status": instance.status.value,
                "load_time": instance.load_time,
                "request_count": instance.request_count,
                "avg_response_time": instance.avg_response_time,
                "last_used": instance.last_used,
                "error_message": instance.error_message
            }
        
        return HealthResponse(
            status="healthy" if any(inst.status == ModelStatus.READY for inst in self.instances.values()) else "unhealthy",
            models=models,
            system={
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "active_instances": len([inst for inst in self.instances.values() if inst.status == ModelStatus.READY]),
                "total_requests": sum(inst.request_count for inst in self.instances.values())
            },
            timestamp=time.time()
        )
    
    async def shutdown(self):
        """Shutdown the serving manager"""
        logger.info("Shutting down serving manager")
        self._shutdown = True
        await self.batch_processor.shutdown()

# FastAPI application factory
def create_app() -> FastAPI:
    """Create FastAPI application with serving endpoints"""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for serving functionality. Install with: pip install fastapi uvicorn")
    
    # Lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        serving_manager = ServingManager.get_instance()
        serving_manager._start_time = time.time()
        logger.info("TrustformeRS serving started")
        yield
        # Shutdown
        await serving_manager.shutdown()
        logger.info("TrustformeRS serving stopped")
    
    app = FastAPI(
        title="TrustformeRS Model Serving API",
        description="High-performance serving API for TrustformeRS models",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    serving_manager = ServingManager.get_instance()
    
    @app.post("/v1/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        """Process inference request"""
        return await serving_manager.inference(request)
    
    @app.post("/v1/models/load")
    async def load_model(config: ModelConfig):
        """Load a model"""
        success = await serving_manager.load_model(config)
        if success:
            return {"status": "success", "message": f"Model {config.model_id}:{config.version} loaded"}
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")
    
    @app.delete("/v1/models/{model_id}/{version}")
    async def unload_model(model_id: str, version: str):
        """Unload a model"""
        success = await serving_manager.unload_model(model_id, version)
        if success:
            return {"status": "success", "message": f"Model {model_id}:{version} unloaded"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    
    @app.get("/v1/models")
    async def list_models():
        """List all loaded models"""
        return {
            "models": [
                {
                    "model_id": inst.config.model_id,
                    "version": inst.config.version,
                    "status": inst.status.value,
                    "load_time": inst.load_time,
                    "request_count": inst.request_count
                }
                for inst in serving_manager.instances.values()
            ]
        }
    
    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        return serving_manager.get_health()
    
    @app.post("/v1/ab-tests")
    async def create_ab_test(config: ABTestConfig):
        """Create an A/B test"""
        success = serving_manager.ab_test_manager.create_test(config)
        if success:
            return {"status": "success", "message": f"A/B test {config.test_name} created"}
        else:
            raise HTTPException(status_code=400, detail="A/B test already exists")
    
    @app.get("/v1/ab-tests/{test_name}/results")
    async def get_ab_test_results(test_name: str):
        """Get A/B test results"""
        results = serving_manager.ab_test_manager.get_test_results(test_name)
        if results:
            return {"test_name": test_name, "results": results}
        else:
            raise HTTPException(status_code=404, detail="A/B test not found")
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        metrics_data = serving_manager.metrics.get_metrics()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    
    return app

# Convenience functions
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """Start the serving server"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI and uvicorn are required for serving. Install with: pip install fastapi uvicorn")
    
    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level
    )

async def serve_model(
    model_path: str,
    model_id: Optional[str] = None,
    version: str = "1.0.0",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 8,
    device: str = "auto"
) -> ServingManager:
    """Quickly serve a single model"""
    
    # Create config
    config = ModelConfig(
        model_id=model_id or Path(model_path).name,
        model_path=model_path,
        version=version,
        max_batch_size=max_batch_size,
        device=device
    )
    
    # Get serving manager and load model
    serving_manager = ServingManager.get_instance()
    success = await serving_manager.load_model(config)
    
    if not success:
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    logger.info(f"Model {config.model_id}:{version} ready for serving")
    return serving_manager

# Export main classes and functions
__all__ = [
    'ModelConfig',
    'ModelInstance', 
    'InferenceRequest',
    'InferenceResponse',
    'HealthResponse',
    'ABTestConfig',
    'ServingManager',
    'ModelVersionManager',
    'LoadBalancer',
    'BatchProcessor',
    'ABTestManager',
    'MetricsCollector',
    'create_app',
    'serve',
    'serve_model',
    'ModelStatus',
    'LoadBalancingStrategy'
]