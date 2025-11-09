"""
TrustformeRS Client Exceptions

Defines a hierarchy of exceptions for different error conditions when interacting
with TrustformeRS serving infrastructure.
"""

from typing import Any, Dict, Optional


class TrustformersError(Exception):
    """
    Base exception for all TrustformeRS client errors.
    
    All other exceptions in the client library inherit from this base class.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.message}', {self.details})"


class TrustformersAPIError(TrustformersError):
    """
    Exception for API-related errors.
    
    Raised when the server returns an error response (4xx or 5xx status codes).
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize the API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response_data: Raw response data from the server
            request_id: Request identifier if available
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.request_id = request_id
        
        # Extract additional details from response
        details = {
            "status_code": status_code,
            "request_id": request_id,
        }
        if response_data:
            details.update(response_data)
        
        self.details = details
    
    def __str__(self) -> str:
        parts = [self.message]
        
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        
        return " - ".join(parts)
    
    @property
    def is_client_error(self) -> bool:
        """Check if this is a client error (4xx)."""
        return self.status_code is not None and 400 <= self.status_code < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if this is a server error (5xx)."""
        return self.status_code is not None and self.status_code >= 500
    
    @property
    def is_retryable(self) -> bool:
        """Check if this error might be retryable."""
        # Generally, only server errors and some specific client errors are retryable
        if self.is_server_error:
            return True
        
        # Some 4xx errors might be retryable
        if self.status_code in [408, 429]:  # Timeout, Rate Limited
            return True
        
        return False


class TrustformersTimeoutError(TrustformersError):
    """
    Exception for request timeout errors.
    
    Raised when a request takes longer than the configured timeout.
    """
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        """
        Initialize the timeout error.
        
        Args:
            message: Error message
            timeout_seconds: The timeout value that was exceeded
        """
        super().__init__(message, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class TrustformersConnectionError(TrustformersError):
    """
    Exception for connection-related errors.
    
    Raised when there are network connectivity issues or the server is unreachable.
    """
    
    def __init__(self, message: str, base_url: Optional[str] = None):
        """
        Initialize the connection error.
        
        Args:
            message: Error message
            base_url: The server URL that failed to connect
        """
        super().__init__(message, {"base_url": base_url})
        self.base_url = base_url


class TrustformersAuthenticationError(TrustformersError):
    """
    Exception for authentication and authorization errors.
    
    Raised when authentication fails or the user lacks necessary permissions.
    """
    
    def __init__(
        self,
        message: str,
        auth_type: Optional[str] = None,
        required_scopes: Optional[list] = None,
    ):
        """
        Initialize the authentication error.
        
        Args:
            message: Error message
            auth_type: Type of authentication that failed
            required_scopes: Required scopes/permissions if applicable
        """
        details = {}
        if auth_type:
            details["auth_type"] = auth_type
        if required_scopes:
            details["required_scopes"] = required_scopes
        
        super().__init__(message, details)
        self.auth_type = auth_type
        self.required_scopes = required_scopes or []


class TrustformersValidationError(TrustformersError):
    """
    Exception for input validation errors.
    
    Raised when request parameters fail validation before being sent to the server.
    """
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, list]] = None,
        invalid_fields: Optional[list] = None,
    ):
        """
        Initialize the validation error.
        
        Args:
            message: Error message
            field_errors: Mapping of field names to error messages
            invalid_fields: List of invalid field names
        """
        details = {}
        if field_errors:
            details["field_errors"] = field_errors
        if invalid_fields:
            details["invalid_fields"] = invalid_fields
        
        super().__init__(message, details)
        self.field_errors = field_errors or {}
        self.invalid_fields = invalid_fields or []


class TrustformersModelError(TrustformersError):
    """
    Exception for model-related errors.
    
    Raised when there are issues with model loading, availability, or configuration.
    """
    
    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        model_status: Optional[str] = None,
    ):
        """
        Initialize the model error.
        
        Args:
            message: Error message
            model_id: ID of the model that caused the error
            model_status: Current status of the model
        """
        details = {}
        if model_id:
            details["model_id"] = model_id
        if model_status:
            details["model_status"] = model_status
        
        super().__init__(message, details)
        self.model_id = model_id
        self.model_status = model_status


class TrustformersResourceError(TrustformersError):
    """
    Exception for resource-related errors.
    
    Raised when there are insufficient resources (memory, GPU, etc.) for the request.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required_amount: Optional[float] = None,
        available_amount: Optional[float] = None,
    ):
        """
        Initialize the resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, gpu, etc.)
            required_amount: Amount of resource required
            available_amount: Amount of resource available
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if required_amount is not None:
            details["required_amount"] = required_amount
        if available_amount is not None:
            details["available_amount"] = available_amount
        
        super().__init__(message, details)
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount


class TrustformersRateLimitError(TrustformersAPIError):
    """
    Exception for rate limiting errors.
    
    Raised when the client has exceeded the rate limit for API requests.
    """
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        rate_limit_type: Optional[str] = None,
    ):
        """
        Initialize the rate limit error.
        
        Args:
            message: Error message
            retry_after_seconds: Seconds to wait before retrying
            rate_limit_type: Type of rate limit (requests_per_minute, etc.)
        """
        super().__init__(message, status_code=429)
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_type = rate_limit_type
        
        if retry_after_seconds:
            self.details["retry_after_seconds"] = retry_after_seconds
        if rate_limit_type:
            self.details["rate_limit_type"] = rate_limit_type


class TrustformersStreamingError(TrustformersError):
    """
    Exception for streaming-related errors.
    
    Raised when there are issues with streaming inference requests.
    """
    
    def __init__(
        self,
        message: str,
        stream_id: Optional[str] = None,
        tokens_received: Optional[int] = None,
    ):
        """
        Initialize the streaming error.
        
        Args:
            message: Error message
            stream_id: ID of the streaming request
            tokens_received: Number of tokens received before error
        """
        details = {}
        if stream_id:
            details["stream_id"] = stream_id
        if tokens_received is not None:
            details["tokens_received"] = tokens_received
        
        super().__init__(message, details)
        self.stream_id = stream_id
        self.tokens_received = tokens_received


class TrustformersBatchError(TrustformersError):
    """
    Exception for batch processing errors.
    
    Raised when there are issues with batch inference requests.
    """
    
    def __init__(
        self,
        message: str,
        batch_id: Optional[str] = None,
        successful_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        partial_results: Optional[list] = None,
    ):
        """
        Initialize the batch error.
        
        Args:
            message: Error message
            batch_id: ID of the batch request
            successful_items: Number of successfully processed items
            failed_items: Number of failed items
            partial_results: Partial results if available
        """
        details = {
            "batch_id": batch_id,
            "successful_items": successful_items,
            "failed_items": failed_items,
        }
        
        super().__init__(message, details)
        self.batch_id = batch_id
        self.successful_items = successful_items
        self.failed_items = failed_items
        self.partial_results = partial_results or []


# Utility functions for exception handling

def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error might be retryable
    """
    # TrustformersAPIError has built-in retry logic
    if isinstance(error, TrustformersAPIError):
        return error.is_retryable
    
    # Connection and timeout errors are generally retryable
    if isinstance(error, (TrustformersConnectionError, TrustformersTimeoutError)):
        return True
    
    # Rate limit errors are retryable after waiting
    if isinstance(error, TrustformersRateLimitError):
        return True
    
    # Resource errors might be retryable after some time
    if isinstance(error, TrustformersResourceError):
        return True
    
    # Other errors are generally not retryable
    return False


def get_retry_delay(error: Exception) -> Optional[float]:
    """
    Get the recommended retry delay for an error.
    
    Args:
        error: Exception to check
        
    Returns:
        Recommended delay in seconds, or None if not applicable
    """
    if isinstance(error, TrustformersRateLimitError) and error.retry_after_seconds:
        return float(error.retry_after_seconds)
    
    # Default exponential backoff delays
    if isinstance(error, TrustformersConnectionError):
        return 2.0
    
    if isinstance(error, TrustformersTimeoutError):
        return 5.0
    
    if isinstance(error, TrustformersResourceError):
        return 10.0
    
    if isinstance(error, TrustformersAPIError) and error.is_server_error:
        return 3.0
    
    return None


def extract_error_details(error: Exception) -> Dict[str, Any]:
    """
    Extract structured error details from an exception.
    
    Args:
        error: Exception to extract details from
        
    Returns:
        Dictionary of error details
    """
    details = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "retryable": is_retryable_error(error),
    }
    
    # Add retry delay if available
    retry_delay = get_retry_delay(error)
    if retry_delay:
        details["retry_delay_seconds"] = retry_delay
    
    # Add exception-specific details
    if hasattr(error, 'details'):
        details.update(error.details)
    
    return details


# Exception mapping for HTTP status codes
HTTP_STATUS_TO_EXCEPTION = {
    400: TrustformersValidationError,
    401: TrustformersAuthenticationError,
    403: TrustformersAuthenticationError,
    404: TrustformersAPIError,
    408: TrustformersTimeoutError,
    409: TrustformersAPIError,
    422: TrustformersValidationError,
    429: TrustformersRateLimitError,
    500: TrustformersAPIError,
    502: TrustformersConnectionError,
    503: TrustformersResourceError,
    504: TrustformersTimeoutError,
}


def exception_from_response(
    status_code: int,
    message: str,
    response_data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> TrustformersError:
    """
    Create an appropriate exception from an HTTP response.
    
    Args:
        status_code: HTTP status code
        message: Error message
        response_data: Response data from server
        request_id: Request identifier
        
    Returns:
        Appropriate exception instance
    """
    exception_class = HTTP_STATUS_TO_EXCEPTION.get(status_code, TrustformersAPIError)
    
    # Handle special cases
    if status_code == 429:
        retry_after = None
        if response_data and "retry_after_seconds" in response_data:
            retry_after = response_data["retry_after_seconds"]
        
        return TrustformersRateLimitError(
            message,
            retry_after_seconds=retry_after,
            rate_limit_type=response_data.get("rate_limit_type") if response_data else None,
        )
    
    if exception_class == TrustformersAPIError:
        return TrustformersAPIError(
            message,
            status_code=status_code,
            response_data=response_data,
            request_id=request_id,
        )
    
    # For other exception types, pass what they need
    if exception_class in [TrustformersValidationError]:
        field_errors = response_data.get("field_errors") if response_data else None
        invalid_fields = response_data.get("invalid_fields") if response_data else None
        return exception_class(message, field_errors=field_errors, invalid_fields=invalid_fields)
    
    if exception_class == TrustformersAuthenticationError:
        auth_type = response_data.get("auth_type") if response_data else None
        required_scopes = response_data.get("required_scopes") if response_data else None
        return exception_class(message, auth_type=auth_type, required_scopes=required_scopes)
    
    # Default case
    return exception_class(message)