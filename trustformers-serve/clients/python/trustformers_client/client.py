"""
TrustformeRS Client Implementation

Provides both synchronous and asynchronous HTTP clients for interacting with TrustformeRS servers.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Iterator, AsyncIterator, Any
from urllib.parse import urljoin

import httpx
import ujson as json
from pydantic import ValidationError

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
    StreamingToken,
)
from .exceptions import (
    TrustformersError,
    TrustformersAPIError,
    TrustformersTimeoutError,
    TrustformersConnectionError,
    TrustformersAuthenticationError,
)
from .auth import AuthConfig
from .retry import RetryConfig, exponential_backoff
from .monitoring import RequestTracker

logger = logging.getLogger(__name__)


class TrustformersClient:
    """
    Synchronous client for TrustformeRS serving infrastructure.
    
    Provides methods for inference, model management, health checks, and monitoring.
    Supports authentication, retries, and comprehensive error handling.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth: Optional[AuthConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
        enable_monitoring: bool = True,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the TrustformeRS client.
        
        Args:
            base_url: Base URL of the TrustformeRS server
            auth: Authentication configuration
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            verify_ssl: Whether to verify SSL certificates
            headers: Additional headers to include in requests
            enable_monitoring: Whether to enable request monitoring
            user_agent: Custom user agent string
        """
        self.base_url = base_url.rstrip('/')
        self.auth = auth
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl
        self.enable_monitoring = enable_monitoring
        
        # Default headers
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent or f"trustformers-client-python/0.1.0",
        }
        if headers:
            self._headers.update(headers)
        
        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=timeout,
            verify=verify_ssl,
            headers=self._headers,
        )
        
        # Initialize monitoring
        if enable_monitoring:
            self._tracker = RequestTracker()
        else:
            self._tracker = None
        
        # Retry configuration
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=retry_delay,
            max_delay=60.0,
            exponential_base=2.0,
        )
        
        logger.info(f"Initialized TrustformeRS client for {base_url}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the HTTP client and cleanup resources."""
        if hasattr(self, '_client'):
            self._client.close()
        logger.debug("TrustformeRS client closed")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.auth:
            return {}
        return self.auth.get_headers()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        stream: bool = False,
        **kwargs
    ) -> httpx.Response:
        """
        Make an HTTP request with retries and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response
            **kwargs: Additional arguments for httpx
            
        Returns:
            HTTP response object
            
        Raises:
            TrustformersAPIError: For API errors
            TrustformersTimeoutError: For timeout errors
            TrustformersConnectionError: For connection errors
            TrustformersAuthenticationError: For authentication errors
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        headers = {**self._headers, **self._get_auth_headers()}
        
        request_start = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self._tracker:
                    self._tracker.record_request_start(method, endpoint)
                
                response = self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    stream=stream,
                    **kwargs
                )
                
                request_duration = time.time() - request_start
                
                if self._tracker:
                    self._tracker.record_request_end(
                        method, endpoint, response.status_code, request_duration
                    )
                
                # Handle authentication errors
                if response.status_code == 401:
                    raise TrustformersAuthenticationError(
                        "Authentication failed. Check your credentials."
                    )
                
                # Handle client errors (4xx)
                if 400 <= response.status_code < 500:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('error', 'Client error')
                    except Exception:
                        error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                    
                    raise TrustformersAPIError(
                        error_message,
                        status_code=response.status_code,
                        response_data=error_data if 'error_data' in locals() else None
                    )
                
                # Handle server errors (5xx) - these should be retried
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        delay = exponential_backoff(attempt, self._retry_config)
                        logger.warning(
                            f"Server error (attempt {attempt + 1}/{self.max_retries + 1}): "
                            f"HTTP {response.status_code}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', 'Server error')
                        except Exception:
                            error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                        
                        raise TrustformersAPIError(
                            error_message,
                            status_code=response.status_code,
                            response_data=error_data if 'error_data' in locals() else None
                        )
                
                # Success case
                return response
                
            except httpx.TimeoutException as e:
                last_exception = TrustformersTimeoutError(f"Request timed out: {e}")
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, self._retry_config)
                    logger.warning(
                        f"Timeout error (attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                
            except httpx.ConnectError as e:
                last_exception = TrustformersConnectionError(f"Connection failed: {e}")
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, self._retry_config)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                
            except Exception as e:
                last_exception = TrustformersError(f"Unexpected error: {e}")
                break
        
        # If we reach here, all retries have been exhausted
        if last_exception:
            raise last_exception
        else:
            raise TrustformersError("All retry attempts exhausted")
    
    def get_health(self) -> HealthStatus:
        """
        Get server health status.
        
        Returns:
            Health status information
        """
        response = self._make_request("GET", "/health")
        return HealthStatus.model_validate(response.json())
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed server health information.
        
        Returns:
            Detailed health status including dependencies
        """
        response = self._make_request("GET", "/health/detailed")
        return response.json()
    
    def get_readiness(self) -> Dict[str, Any]:
        """
        Get server readiness status.
        
        Returns:
            Readiness status information
        """
        response = self._make_request("GET", "/health/readiness")
        return response.json()
    
    def get_liveness(self) -> Dict[str, Any]:
        """
        Get server liveness status.
        
        Returns:
            Liveness status information
        """
        response = self._make_request("GET", "/health/liveness")
        return response.json()
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run inference on a single input.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        response = self._make_request(
            "POST", 
            "/v1/infer",
            data=request.model_dump()
        )
        return InferenceResponse.model_validate(response.json())
    
    def batch_infer(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """
        Run inference on a batch of inputs.
        
        Args:
            request: Batch inference request
            
        Returns:
            Batch inference response
        """
        response = self._make_request(
            "POST",
            "/v1/batch_infer", 
            data=request.model_dump()
        )
        return BatchInferenceResponse.model_validate(response.json())
    
    def stream_infer(self, request: InferenceRequest) -> Iterator[StreamingToken]:
        """
        Run streaming inference with token-by-token generation.
        
        Args:
            request: Inference request
            
        Yields:
            Streaming tokens as they are generated
        """
        try:
            with self._client.stream(
                "POST",
                urljoin(self.base_url, "/v1/stream_infer"),
                headers={**self._headers, **self._get_auth_headers()},
                json=request.model_dump(),
                timeout=None  # Disable timeout for streaming
            ) as response:
                
                if response.status_code != 200:
                    response.read()  # Consume the response
                    raise TrustformersAPIError(
                        f"Streaming failed with status {response.status_code}",
                        status_code=response.status_code
                    )
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            token_data = json.loads(data_str)
                            yield StreamingToken.model_validate(token_data)
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.warning(f"Failed to parse streaming token: {e}")
                            continue
                            
        except httpx.TimeoutException as e:
            raise TrustformersTimeoutError(f"Streaming request timed out: {e}")
        except httpx.ConnectError as e:
            raise TrustformersConnectionError(f"Streaming connection failed: {e}")
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all available models.
        
        Returns:
            List of model information
        """
        response = self._make_request("GET", "/v1/models")
        models_data = response.json()
        return [ModelInfo.model_validate(model) for model in models_data]
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information
        """
        response = self._make_request("GET", f"/v1/models/{model_id}")
        return ModelInfo.model_validate(response.json())
    
    def get_model_status(self, model_id: str) -> ModelStatus:
        """
        Get the status of a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model status
        """
        response = self._make_request("GET", f"/v1/models/{model_id}/status")
        return ModelStatus.model_validate(response.json())
    
    def load_model(self, model_id: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a model on the server.
        
        Args:
            model_id: Model identifier
            model_config: Optional model configuration
            
        Returns:
            Load operation result
        """
        data = {"model_id": model_id}
        if model_config:
            data["config"] = model_config
            
        response = self._make_request("POST", f"/v1/models/{model_id}/load", data=data)
        return response.json()
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """
        Unload a model from the server.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Unload operation result
        """
        response = self._make_request("POST", f"/v1/models/{model_id}/unload")
        return response.json()
    
    def get_metrics(self) -> ServiceMetrics:
        """
        Get server metrics.
        
        Returns:
            Service metrics
        """
        response = self._make_request("GET", "/metrics")
        return ServiceMetrics.model_validate(response.json())
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        response = self._make_request("GET", "/v1/metrics/performance")
        return PerformanceMetrics.model_validate(response.json())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Server statistics
        """
        response = self._make_request("GET", "/stats")
        return response.json()


class AsyncTrustformersClient:
    """
    Asynchronous client for TrustformeRS serving infrastructure.
    
    Provides async/await methods for inference, model management, and monitoring.
    Ideal for high-performance applications and concurrent request handling.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth: Optional[AuthConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
        enable_monitoring: bool = True,
        user_agent: Optional[str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ):
        """
        Initialize the async TrustformeRS client.
        
        Args:
            base_url: Base URL of the TrustformeRS server
            auth: Authentication configuration
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            verify_ssl: Whether to verify SSL certificates
            headers: Additional headers to include in requests
            enable_monitoring: Whether to enable request monitoring
            user_agent: Custom user agent string
            max_connections: Maximum number of connections in the pool
            max_keepalive_connections: Maximum keepalive connections
        """
        self.base_url = base_url.rstrip('/')
        self.auth = auth
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl
        self.enable_monitoring = enable_monitoring
        
        # Default headers
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent or f"trustformers-client-python-async/0.1.0",
        }
        if headers:
            self._headers.update(headers)
        
        # HTTP client limits
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections
        )
        
        # Initialize async HTTP client
        self._client = httpx.AsyncClient(
            timeout=timeout,
            verify=verify_ssl,
            headers=self._headers,
            limits=limits,
        )
        
        # Initialize monitoring
        if enable_monitoring:
            self._tracker = RequestTracker()
        else:
            self._tracker = None
        
        # Retry configuration
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=retry_delay,
            max_delay=60.0,
            exponential_base=2.0,
        )
        
        logger.info(f"Initialized async TrustformeRS client for {base_url}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        if hasattr(self, '_client'):
            await self._client.aclose()
        logger.debug("Async TrustformeRS client closed")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.auth:
            return {}
        return self.auth.get_headers()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        stream: bool = False,
        **kwargs
    ) -> httpx.Response:
        """
        Make an async HTTP request with retries and error handling.
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        headers = {**self._headers, **self._get_auth_headers()}
        
        request_start = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self._tracker:
                    self._tracker.record_request_start(method, endpoint)
                
                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    **kwargs
                )
                
                request_duration = time.time() - request_start
                
                if self._tracker:
                    self._tracker.record_request_end(
                        method, endpoint, response.status_code, request_duration
                    )
                
                # Handle authentication errors
                if response.status_code == 401:
                    raise TrustformersAuthenticationError(
                        "Authentication failed. Check your credentials."
                    )
                
                # Handle client errors (4xx)
                if 400 <= response.status_code < 500:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('error', 'Client error')
                    except Exception:
                        error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                    
                    raise TrustformersAPIError(
                        error_message,
                        status_code=response.status_code,
                        response_data=error_data if 'error_data' in locals() else None
                    )
                
                # Handle server errors (5xx) - these should be retried
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        delay = exponential_backoff(attempt, self._retry_config)
                        logger.warning(
                            f"Server error (attempt {attempt + 1}/{self.max_retries + 1}): "
                            f"HTTP {response.status_code}. Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', 'Server error')
                        except Exception:
                            error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                        
                        raise TrustformersAPIError(
                            error_message,
                            status_code=response.status_code,
                            response_data=error_data if 'error_data' in locals() else None
                        )
                
                # Success case
                return response
                
            except httpx.TimeoutException as e:
                last_exception = TrustformersTimeoutError(f"Request timed out: {e}")
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, self._retry_config)
                    logger.warning(
                        f"Timeout error (attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                
            except httpx.ConnectError as e:
                last_exception = TrustformersConnectionError(f"Connection failed: {e}")
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, self._retry_config)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                
            except Exception as e:
                last_exception = TrustformersError(f"Unexpected error: {e}")
                break
        
        # If we reach here, all retries have been exhausted
        if last_exception:
            raise last_exception
        else:
            raise TrustformersError("All retry attempts exhausted")
    
    async def get_health(self) -> HealthStatus:
        """Get server health status."""
        response = await self._make_request("GET", "/health")
        return HealthStatus.model_validate(response.json())
    
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a single input."""
        response = await self._make_request(
            "POST", 
            "/v1/infer",
            data=request.model_dump()
        )
        return InferenceResponse.model_validate(response.json())
    
    async def batch_infer(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """Run inference on a batch of inputs."""
        response = await self._make_request(
            "POST",
            "/v1/batch_infer", 
            data=request.model_dump()
        )
        return BatchInferenceResponse.model_validate(response.json())
    
    async def stream_infer(self, request: InferenceRequest) -> AsyncIterator[StreamingToken]:
        """Run streaming inference with token-by-token generation."""
        try:
            async with self._client.stream(
                "POST",
                urljoin(self.base_url, "/v1/stream_infer"),
                headers={**self._headers, **self._get_auth_headers()},
                json=request.model_dump(),
                timeout=None  # Disable timeout for streaming
            ) as response:
                
                if response.status_code != 200:
                    await response.aread()  # Consume the response
                    raise TrustformersAPIError(
                        f"Streaming failed with status {response.status_code}",
                        status_code=response.status_code
                    )
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            token_data = json.loads(data_str)
                            yield StreamingToken.model_validate(token_data)
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.warning(f"Failed to parse streaming token: {e}")
                            continue
                            
        except httpx.TimeoutException as e:
            raise TrustformersTimeoutError(f"Streaming request timed out: {e}")
        except httpx.ConnectError as e:
            raise TrustformersConnectionError(f"Streaming connection failed: {e}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        response = await self._make_request("GET", "/v1/models")
        models_data = response.json()
        return [ModelInfo.model_validate(model) for model in models_data]
    
    async def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        response = await self._make_request("GET", f"/v1/models/{model_id}")
        return ModelInfo.model_validate(response.json())
    
    async def get_metrics(self) -> ServiceMetrics:
        """Get server metrics."""
        response = await self._make_request("GET", "/metrics")
        return ServiceMetrics.model_validate(response.json())

# Convenience functions for quick setup
def create_client(
    base_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    **kwargs
) -> TrustformersClient:
    """
    Create a TrustformeRS client with simple API key authentication.
    
    Args:
        base_url: Server URL
        api_key: API key for authentication
        **kwargs: Additional client arguments
        
    Returns:
        Configured TrustformeRS client
    """
    from .auth import APIKeyAuth
    
    auth = None
    if api_key:
        auth = APIKeyAuth(api_key)
    
    return TrustformersClient(base_url=base_url, auth=auth, **kwargs)

def create_async_client(
    base_url: str = "http://localhost:8080", 
    api_key: Optional[str] = None,
    **kwargs
) -> AsyncTrustformersClient:
    """
    Create an async TrustformeRS client with simple API key authentication.
    
    Args:
        base_url: Server URL
        api_key: API key for authentication
        **kwargs: Additional client arguments
        
    Returns:
        Configured async TrustformeRS client
    """
    from .auth import APIKeyAuth
    
    auth = None
    if api_key:
        auth = APIKeyAuth(api_key)
    
    return AsyncTrustformersClient(base_url=base_url, auth=auth, **kwargs)