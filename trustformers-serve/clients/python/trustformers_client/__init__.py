"""
TrustformeRS Client Library for Python

A comprehensive Python client for interacting with TrustformeRS serving infrastructure.
Provides both synchronous and asynchronous APIs for inference, model management, and monitoring.
"""

from .client import TrustformersClient, AsyncTrustformersClient
from .models import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ModelInfo,
    ModelStatus,
    HealthStatus,
    PerformanceMetrics,
    ServiceMetrics,
)
from .exceptions import (
    TrustformersError,
    TrustformersAPIError,
    TrustformersTimeoutError,
    TrustformersConnectionError,
    TrustformersAuthenticationError,
)
from .streaming import StreamingClient, AsyncStreamingClient
from .auth import AuthConfig, APIKeyAuth, JWTAuth, OAuth2Auth
from .monitoring import MonitoringClient, PerformanceMonitor
from .batch import BatchManager, AsyncBatchManager

__version__ = "0.1.0"
__author__ = "TrustformeRS Team"
__email__ = "support@trustformers.ai"
__description__ = "Python client library for TrustformeRS serving infrastructure"

__all__ = [
    # Main client classes
    "TrustformersClient",
    "AsyncTrustformersClient",
    
    # Data models
    "InferenceRequest",
    "InferenceResponse", 
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "ModelInfo",
    "ModelStatus",
    "HealthStatus",
    "PerformanceMetrics",
    "ServiceMetrics",
    
    # Exceptions
    "TrustformersError",
    "TrustformersAPIError",
    "TrustformersTimeoutError",
    "TrustformersConnectionError",
    "TrustformersAuthenticationError",
    
    # Streaming
    "StreamingClient",
    "AsyncStreamingClient",
    
    # Authentication
    "AuthConfig",
    "APIKeyAuth",
    "JWTAuth",
    "OAuth2Auth",
    
    # Monitoring
    "MonitoringClient",
    "PerformanceMonitor",
    
    # Batch processing
    "BatchManager",
    "AsyncBatchManager",
]

# Default configuration
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_STREAM_BUFFER_SIZE = 1024

# Version info
VERSION_INFO = {
    "version": __version__,
    "python_requires": ">=3.8",
    "dependencies": [
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "websockets>=11.0.0",
        "asyncio-mqtt>=0.13.0",
        "ujson>=5.7.0",
        "aiofiles>=23.1.0",
    ],
    "optional_dependencies": {
        "jwt": ["PyJWT>=2.6.0"],
        "oauth": ["authlib>=1.2.0"],
        "monitoring": ["prometheus-client>=0.16.0"],
        "visualization": ["matplotlib>=3.7.0", "plotly>=5.14.0"],
    }
}

def get_version():
    """Get the current version of the client library."""
    return __version__

def get_client_info():
    """Get comprehensive client library information."""
    return {
        "name": "trustformers-client",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "supported_apis": [
            "REST API v1",
            "gRPC API",
            "WebSocket Streaming", 
            "GraphQL API",
            "Server-Sent Events",
        ],
        "features": [
            "Synchronous and asynchronous clients",
            "Batch processing",
            "Streaming inference",
            "Model management",
            "Performance monitoring",
            "Authentication (API Key, JWT, OAuth2)",
            "Automatic retries and error handling",
            "Type safety with Pydantic models",
            "Comprehensive logging",
            "Prometheus metrics integration",
        ]
    }

# Configure default logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def configure_logging(
    level: str = "INFO",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    enable_file_logging: bool = False,
    log_file: str = "trustformers_client.log"
):
    """
    Configure logging for the TrustformeRS client library.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format
        enable_file_logging: Whether to log to file in addition to console
        log_file: File path for logging (if file logging is enabled)
    """
    # Configure root logger for this package
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    package_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)
    
    logger.info(f"TrustformeRS client logging configured at {level} level")

# Export key constants
SUPPORTED_MODEL_TYPES = [
    "bert",
    "gpt2", 
    "gpt3",
    "t5",
    "roberta",
    "distilbert",
    "electra",
    "albert",
    "deberta",
    "longformer",
    "bart",
    "pegasus",
    "marian",
    "wav2vec2",
    "vit",
    "deit",
    "clip",
    "blip",
    "custom",
]

SUPPORTED_TASKS = [
    "text-generation",
    "text-classification", 
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "fill-mask",
    "feature-extraction",
    "sentence-similarity",
    "zero-shot-classification",
    "conversational",
    "image-classification",
    "object-detection",
    "image-to-text",
    "text-to-image",
    "audio-classification",
    "automatic-speech-recognition",
    "text-to-speech",
]

# Quick start example
QUICK_START_EXAMPLE = '''
# Quick Start Example

from trustformers_client import TrustformersClient, InferenceRequest

# Initialize client
client = TrustformersClient(base_url="http://localhost:8080")

# Check server health
health = client.get_health()
print(f"Server status: {health.status}")

# Run inference
request = InferenceRequest(
    input_text="Hello, how are you?",
    model_id="gpt2-small",
    max_length=50
)

response = client.infer(request)
print(f"Generated text: {response.output_text}")

# Batch inference
batch_request = BatchInferenceRequest(
    inputs=["Text 1", "Text 2", "Text 3"],
    model_id="bert-base-uncased"
)

batch_response = client.batch_infer(batch_request)
for result in batch_response.results:
    print(f"Result: {result.output}")

# Streaming inference
for token in client.stream_infer(request):
    print(token.text, end="", flush=True)
'''