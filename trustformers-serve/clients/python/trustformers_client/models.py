"""
Data models for TrustformeRS client library.

Defines Pydantic models for requests, responses, and server data structures.
Provides type safety and automatic validation for API interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator


class TaskType(str, Enum):
    """Supported inference task types."""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    FILL_MASK = "fill-mask"
    FEATURE_EXTRACTION = "feature-extraction"
    SENTENCE_SIMILARITY = "sentence-similarity"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    CONVERSATIONAL = "conversational"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_TO_TEXT = "image-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    AUDIO_CLASSIFICATION = "audio-classification"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    TEXT_TO_SPEECH = "text-to-speech"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Supported model architectures."""
    BERT = "bert"
    GPT2 = "gpt2"
    GPT3 = "gpt3"
    T5 = "t5"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    ELECTRA = "electra"
    ALBERT = "albert"
    DEBERTA = "deberta"
    LONGFORMER = "longformer"
    BART = "bart"
    PEGASUS = "pegasus"
    MARIAN = "marian"
    WAV2VEC2 = "wav2vec2"
    VIT = "vit"
    DEIT = "deit"
    CLIP = "clip"
    BLIP = "blip"
    CUSTOM = "custom"


class DeviceType(str, Enum):
    """Device types for inference."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"


class Priority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    """General status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelStatusEnum(str, Enum):
    """Model loading/status states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"


class InferenceRequest(BaseModel):
    """Request for single inference."""
    
    input_text: str = Field(..., description="Input text for inference")
    model_id: str = Field(..., description="Model identifier")
    task_type: Optional[TaskType] = Field(None, description="Type of task to perform")
    
    # Generation parameters
    max_length: Optional[int] = Field(50, ge=1, le=4096, description="Maximum output length")
    min_length: Optional[int] = Field(None, ge=0, description="Minimum output length")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    num_beams: Optional[int] = Field(1, ge=1, le=20, description="Number of beams for beam search")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    repetition_penalty: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    
    # Processing parameters
    return_tensors: Optional[bool] = Field(False, description="Whether to return raw tensors")
    return_attention: Optional[bool] = Field(False, description="Whether to return attention weights")
    return_hidden_states: Optional[bool] = Field(False, description="Whether to return hidden states")
    
    # Request metadata
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    priority: Optional[Priority] = Field(Priority.NORMAL, description="Request priority")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Request timeout in seconds")
    
    # Advanced parameters
    device: Optional[DeviceType] = Field(DeviceType.AUTO, description="Device to use for inference")
    enable_caching: Optional[bool] = Field(True, description="Whether to enable result caching")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom model-specific parameters")
    
    @validator('min_length')
    def min_length_must_be_less_than_max(cls, v, values):
        if v is not None and 'max_length' in values and v >= values['max_length']:
            raise ValueError('min_length must be less than max_length')
        return v
    
    @validator('input_text')
    def input_text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('input_text cannot be empty')
        return v


class InferenceResponse(BaseModel):
    """Response from single inference."""
    
    request_id: Optional[str] = Field(None, description="Request identifier")
    output_text: str = Field(..., description="Generated output text")
    output_tokens: Optional[List[str]] = Field(None, description="Generated tokens")
    
    # Confidence and scores
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    token_scores: Optional[List[float]] = Field(None, description="Per-token confidence scores")
    
    # Model outputs
    logits: Optional[List[List[float]]] = Field(None, description="Raw model logits")
    attention_weights: Optional[List[List[List[float]]]] = Field(None, description="Attention weights")
    hidden_states: Optional[List[List[List[float]]]] = Field(None, description="Hidden states")
    
    # Metadata
    model_id: str = Field(..., description="Model used for inference")
    task_type: Optional[TaskType] = Field(None, description="Task type performed")
    device_used: Optional[DeviceType] = Field(None, description="Device used for inference")
    
    # Performance metrics
    inference_time: Optional[float] = Field(None, ge=0.0, description="Inference time in seconds")
    tokens_per_second: Optional[float] = Field(None, ge=0.0, description="Generation speed")
    memory_used: Optional[int] = Field(None, ge=0, description="Memory used in bytes")
    
    # Generation details
    finish_reason: Optional[str] = Field(None, description="Why generation finished")
    num_generated_tokens: Optional[int] = Field(None, ge=0, description="Number of tokens generated")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Response creation time")
    
    # Caching info
    cache_hit: Optional[bool] = Field(None, description="Whether response came from cache")


class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""
    
    inputs: List[str] = Field(..., min_items=1, max_items=100, description="List of input texts")
    model_id: str = Field(..., description="Model identifier")
    task_type: Optional[TaskType] = Field(None, description="Type of task to perform")
    
    # Shared generation parameters
    max_length: Optional[int] = Field(50, ge=1, le=4096, description="Maximum output length")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    
    # Batch parameters
    batch_size: Optional[int] = Field(None, ge=1, le=32, description="Internal batch size for processing")
    return_all_outputs: Optional[bool] = Field(True, description="Whether to return all outputs")
    
    # Request metadata
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    priority: Optional[Priority] = Field(Priority.NORMAL, description="Request priority")
    timeout: Optional[int] = Field(60, ge=1, le=600, description="Request timeout in seconds")
    
    @validator('inputs')
    def inputs_must_not_be_empty(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All input texts must be non-empty')
        return v


class BatchInferenceResponse(BaseModel):
    """Response from batch inference."""
    
    request_id: Optional[str] = Field(None, description="Request identifier")
    results: List[InferenceResponse] = Field(..., description="Individual inference results")
    
    # Batch metadata
    total_inputs: int = Field(..., ge=1, description="Total number of inputs processed")
    successful_outputs: int = Field(..., ge=0, description="Number of successful outputs")
    failed_outputs: int = Field(..., ge=0, description="Number of failed outputs")
    
    # Performance metrics
    total_inference_time: Optional[float] = Field(None, ge=0.0, description="Total batch processing time")
    average_inference_time: Optional[float] = Field(None, ge=0.0, description="Average time per item")
    total_tokens_generated: Optional[int] = Field(None, ge=0, description="Total tokens generated")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Response creation time")
    
    @validator('successful_outputs', 'failed_outputs')
    def outputs_must_sum_to_total(cls, v, values):
        if 'total_inputs' in values and 'successful_outputs' in values:
            if values['successful_outputs'] + v != values['total_inputs']:
                raise ValueError('successful_outputs + failed_outputs must equal total_inputs')
        return v


class StreamingToken(BaseModel):
    """Individual token in streaming response."""
    
    token: str = Field(..., description="Generated token")
    text: str = Field(..., description="Decoded text so far")
    is_finished: bool = Field(False, description="Whether generation is complete")
    
    # Token metadata
    token_id: Optional[int] = Field(None, description="Token ID in vocabulary")
    logprob: Optional[float] = Field(None, description="Log probability of token")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Token confidence")
    
    # Position information
    position: Optional[int] = Field(None, ge=0, description="Token position in sequence")
    
    # Timing
    generated_at: Optional[datetime] = Field(None, description="When token was generated")
    
    # Special tokens
    is_special: Optional[bool] = Field(False, description="Whether token is special (EOS, etc.)")


class ModelInfo(BaseModel):
    """Information about a model."""
    
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Model metadata
    model_type: ModelType = Field(..., description="Model architecture type")
    task_types: List[TaskType] = Field(..., description="Supported task types")
    version: str = Field(..., description="Model version")
    
    # Model specifications
    max_sequence_length: int = Field(..., ge=1, description="Maximum input sequence length")
    vocabulary_size: Optional[int] = Field(None, ge=1, description="Vocabulary size")
    num_parameters: Optional[int] = Field(None, ge=1, description="Number of parameters")
    model_size_mb: Optional[float] = Field(None, ge=0.0, description="Model size in MB")
    
    # Supported features
    supports_streaming: bool = Field(False, description="Whether model supports streaming")
    supports_batching: bool = Field(True, description="Whether model supports batching")
    supports_attention_weights: bool = Field(False, description="Whether model can return attention")
    
    # Performance characteristics
    average_latency_ms: Optional[float] = Field(None, ge=0.0, description="Average inference latency")
    max_batch_size: Optional[int] = Field(None, ge=1, description="Maximum batch size")
    
    # Requirements
    required_memory_mb: Optional[float] = Field(None, ge=0.0, description="Required memory in MB")
    recommended_device: Optional[DeviceType] = Field(None, description="Recommended device type")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Model creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")


class ModelStatus(BaseModel):
    """Current status of a model."""
    
    model_id: str = Field(..., description="Model identifier")
    status: ModelStatusEnum = Field(..., description="Current model status")
    
    # Load information
    loaded_at: Optional[datetime] = Field(None, description="When model was loaded")
    load_time_seconds: Optional[float] = Field(None, ge=0.0, description="Time taken to load")
    
    # Resource usage
    memory_used_mb: Optional[float] = Field(None, ge=0.0, description="Memory used by model")
    device_used: Optional[DeviceType] = Field(None, description="Device model is loaded on")
    
    # Performance metrics
    total_requests: Optional[int] = Field(None, ge=0, description="Total requests served")
    successful_requests: Optional[int] = Field(None, ge=0, description="Successful requests")
    failed_requests: Optional[int] = Field(None, ge=0, description="Failed requests")
    average_latency_ms: Optional[float] = Field(None, ge=0.0, description="Average response time")
    
    # Health
    last_used_at: Optional[datetime] = Field(None, description="Last time model was used")
    error_message: Optional[str] = Field(None, description="Error message if status is FAILED")


class HealthStatus(BaseModel):
    """Server health status."""
    
    status: Status = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    # Service information
    service_name: str = Field("trustformers-serve", description="Service name")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime")
    
    # Resource health
    memory_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Memory usage")
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="CPU usage")
    gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage")
    
    # Request metrics
    total_requests: Optional[int] = Field(None, ge=0, description="Total requests served")
    requests_per_second: Optional[float] = Field(None, ge=0.0, description="Current RPS")
    average_response_time_ms: Optional[float] = Field(None, ge=0.0, description="Average response time")
    
    # Loaded models
    loaded_models: Optional[List[str]] = Field(None, description="List of loaded model IDs")
    
    # Additional details
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the service."""
    
    # Request metrics
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    
    # Timing metrics
    average_response_time_ms: float = Field(..., ge=0.0, description="Average response time")
    p50_response_time_ms: float = Field(..., ge=0.0, description="50th percentile response time")
    p95_response_time_ms: float = Field(..., ge=0.0, description="95th percentile response time")
    p99_response_time_ms: float = Field(..., ge=0.0, description="99th percentile response time")
    
    # Throughput metrics
    requests_per_second: float = Field(..., ge=0.0, description="Current requests per second")
    tokens_per_second: Optional[float] = Field(None, ge=0.0, description="Tokens generated per second")
    
    # Resource metrics
    memory_usage_mb: float = Field(..., ge=0.0, description="Current memory usage")
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU utilization")
    gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization")
    
    # Model metrics
    total_models_loaded: int = Field(..., ge=0, description="Number of loaded models")
    model_memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory used by models")
    
    # Cache metrics
    cache_hit_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cache hit rate")
    cache_size_mb: Optional[float] = Field(None, ge=0.0, description="Cache size")
    
    # Timestamps
    collected_at: datetime = Field(..., description="When metrics were collected")
    window_start: Optional[datetime] = Field(None, description="Metrics window start")
    window_end: Optional[datetime] = Field(None, description="Metrics window end")


class ServiceMetrics(BaseModel):
    """Service-level metrics."""
    
    # Service information
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    instance_id: str = Field(..., description="Service instance identifier")
    
    # Performance metrics
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    
    # Health metrics
    health_status: Status = Field(..., description="Current health status")
    last_health_check: datetime = Field(..., description="Last health check time")
    
    # Model metrics
    models: List[ModelStatus] = Field(..., description="Status of all models")
    
    # System metrics
    system_memory_total_mb: float = Field(..., ge=0.0, description="Total system memory")
    system_memory_available_mb: float = Field(..., ge=0.0, description="Available system memory")
    system_cpu_count: int = Field(..., ge=1, description="Number of CPU cores")
    system_gpu_count: Optional[int] = Field(None, ge=0, description="Number of GPUs")
    
    # Network metrics
    active_connections: int = Field(..., ge=0, description="Active network connections")
    total_bytes_sent: int = Field(..., ge=0, description="Total bytes sent")
    total_bytes_received: int = Field(..., ge=0, description="Total bytes received")
    
    # Collection metadata
    collected_at: datetime = Field(..., description="When metrics were collected")


# Additional specialized models for specific use cases

class TextGenerationRequest(InferenceRequest):
    """Specialized request for text generation tasks."""
    
    task_type: TaskType = Field(TaskType.TEXT_GENERATION, description="Fixed to text generation")
    prompt: str = Field(..., description="Generation prompt", alias="input_text")
    
    # Generation-specific parameters
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    include_prompt: Optional[bool] = Field(True, description="Whether to include prompt in output")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")


class ClassificationRequest(InferenceRequest):
    """Specialized request for classification tasks."""
    
    task_type: TaskType = Field(TaskType.TEXT_CLASSIFICATION, description="Fixed to classification")
    text: str = Field(..., description="Text to classify", alias="input_text")
    
    # Classification-specific parameters
    return_all_scores: Optional[bool] = Field(False, description="Return scores for all classes")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification threshold")


class QuestionAnsweringRequest(InferenceRequest):
    """Specialized request for question answering tasks."""
    
    task_type: TaskType = Field(TaskType.QUESTION_ANSWERING, description="Fixed to QA")
    question: str = Field(..., description="Question to answer")
    context: str = Field(..., description="Context for answering", alias="input_text")
    
    # QA-specific parameters
    max_answer_length: Optional[int] = Field(50, ge=1, description="Maximum answer length")
    min_answer_length: Optional[int] = Field(1, ge=1, description="Minimum answer length")
    top_k_answers: Optional[int] = Field(1, ge=1, le=10, description="Number of answers to return")
    
    @root_validator
    def create_input_text(cls, values):
        question = values.get('question')
        context = values.get('context')
        if question and context:
            values['input_text'] = f"Question: {question}\nContext: {context}"
        return values


# Error response models
class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    error_type: Optional[str] = Field(None, description="Error type/category")
    
    # Request context
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    # Additional details
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestion: Optional[str] = Field(None, description="Suggested resolution")
    
    # Retry information
    retryable: Optional[bool] = Field(None, description="Whether the request can be retried")
    retry_after_seconds: Optional[int] = Field(None, description="Suggested retry delay")


class ValidationError(BaseModel):
    """Validation error details."""
    
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="The invalid value")
    expected_type: Optional[str] = Field(None, description="Expected value type")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details."""
    
    error_type: str = Field("validation_error", description="Fixed error type")
    validation_errors: List[ValidationError] = Field(..., description="Specific validation errors")