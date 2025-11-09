/**
 * TrustformeRS TypeScript Client Errors
 * 
 * Defines error classes for different error conditions when interacting
 * with TrustformeRS serving infrastructure.
 */

/**
 * Base error for all TrustformeRS client errors
 */
export class TrustformersError extends Error {
  public readonly details: Record<string, any>;

  constructor(message: string, details: Record<string, any> = {}) {
    super(message);
    this.name = 'TrustformersError';
    this.details = details;
    
    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, TrustformersError);
    }
  }

  toString(): string {
    if (Object.keys(this.details).length > 0) {
      return `${this.message} (details: ${JSON.stringify(this.details)})`;
    }
    return this.message;
  }
}

/**
 * Error for API-related errors
 * 
 * Thrown when the server returns an error response (4xx or 5xx status codes)
 */
export class TrustformersAPIError extends TrustformersError {
  public readonly status?: number;
  public readonly responseData?: Record<string, any>;
  public readonly requestId?: string;

  constructor(
    message: string,
    status?: number,
    responseData?: Record<string, any>,
    requestId?: string
  ) {
    const details: Record<string, any> = {
      status_code: status,
      request_id: requestId,
      ...responseData,
    };
    
    super(message, details);
    this.name = 'TrustformersAPIError';
    this.status = status;
    this.responseData = responseData || {};
    this.requestId = requestId;
  }

  toString(): string {
    const parts = [this.message];
    
    if (this.status) {
      parts.push(`HTTP ${this.status}`);
    }
    
    if (this.requestId) {
      parts.push(`Request ID: ${this.requestId}`);
    }
    
    return parts.join(' - ');
  }

  /**
   * Check if this is a client error (4xx)
   */
  get isClientError(): boolean {
    return this.status !== undefined && this.status >= 400 && this.status < 500;
  }

  /**
   * Check if this is a server error (5xx)
   */
  get isServerError(): boolean {
    return this.status !== undefined && this.status >= 500;
  }

  /**
   * Check if this error might be retryable
   */
  get isRetryable(): boolean {
    // Generally, only server errors and some specific client errors are retryable
    if (this.isServerError) {
      return true;
    }
    
    // Some 4xx errors might be retryable
    if (this.status === 408 || this.status === 429) { // Timeout, Rate Limited
      return true;
    }
    
    return false;
  }
}

/**
 * Error for request timeout errors
 * 
 * Thrown when a request takes longer than the configured timeout
 */
export class TrustformersTimeoutError extends TrustformersError {
  public readonly timeoutSeconds?: number;

  constructor(message: string, timeoutSeconds?: number) {
    super(message, { timeout_seconds: timeoutSeconds });
    this.name = 'TrustformersTimeoutError';
    this.timeoutSeconds = timeoutSeconds;
  }
}

/**
 * Error for connection-related errors
 * 
 * Thrown when there are network connectivity issues or the server is unreachable
 */
export class TrustformersConnectionError extends TrustformersError {
  public readonly baseUrl?: string;

  constructor(message: string, baseUrl?: string) {
    super(message, { base_url: baseUrl });
    this.name = 'TrustformersConnectionError';
    this.baseUrl = baseUrl;
  }
}

/**
 * Error for authentication and authorization errors
 * 
 * Thrown when authentication fails or the user lacks necessary permissions
 */
export class TrustformersAuthenticationError extends TrustformersError {
  public readonly authType?: string;
  public readonly requiredScopes?: string[];

  constructor(
    message: string,
    authType?: string,
    requiredScopes?: string[]
  ) {
    const details: Record<string, any> = {};
    if (authType) {
      details.auth_type = authType;
    }
    if (requiredScopes) {
      details.required_scopes = requiredScopes;
    }
    
    super(message, details);
    this.name = 'TrustformersAuthenticationError';
    this.authType = authType;
    this.requiredScopes = requiredScopes || [];
  }
}

/**
 * Error for input validation errors
 * 
 * Thrown when request parameters fail validation before being sent to the server
 */
export class TrustformersValidationError extends TrustformersError {
  public readonly fieldErrors?: Record<string, string[]>;
  public readonly invalidFields?: string[];

  constructor(
    message: string,
    fieldErrors?: Record<string, string[]>,
    invalidFields?: string[]
  ) {
    const details: Record<string, any> = {};
    if (fieldErrors) {
      details.field_errors = fieldErrors;
    }
    if (invalidFields) {
      details.invalid_fields = invalidFields;
    }
    
    super(message, details);
    this.name = 'TrustformersValidationError';
    this.fieldErrors = fieldErrors || {};
    this.invalidFields = invalidFields || [];
  }
}

/**
 * Error for model-related errors
 * 
 * Thrown when there are issues with model loading, availability, or configuration
 */
export class TrustformersModelError extends TrustformersError {
  public readonly modelId?: string;
  public readonly modelStatus?: string;

  constructor(
    message: string,
    modelId?: string,
    modelStatus?: string
  ) {
    const details: Record<string, any> = {};
    if (modelId) {
      details.model_id = modelId;
    }
    if (modelStatus) {
      details.model_status = modelStatus;
    }
    
    super(message, details);
    this.name = 'TrustformersModelError';
    this.modelId = modelId;
    this.modelStatus = modelStatus;
  }
}

/**
 * Error for resource-related errors
 * 
 * Thrown when there are insufficient resources (memory, GPU, etc.) for the request
 */
export class TrustformersResourceError extends TrustformersError {
  public readonly resourceType?: string;
  public readonly requiredAmount?: number;
  public readonly availableAmount?: number;

  constructor(
    message: string,
    resourceType?: string,
    requiredAmount?: number,
    availableAmount?: number
  ) {
    const details: Record<string, any> = {};
    if (resourceType) {
      details.resource_type = resourceType;
    }
    if (requiredAmount !== undefined) {
      details.required_amount = requiredAmount;
    }
    if (availableAmount !== undefined) {
      details.available_amount = availableAmount;
    }
    
    super(message, details);
    this.name = 'TrustformersResourceError';
    this.resourceType = resourceType;
    this.requiredAmount = requiredAmount;
    this.availableAmount = availableAmount;
  }
}

/**
 * Error for rate limiting errors
 * 
 * Thrown when the client has exceeded the rate limit for API requests
 */
export class TrustformersRateLimitError extends TrustformersAPIError {
  public readonly retryAfterSeconds?: number;
  public readonly rateLimitType?: string;

  constructor(
    message: string,
    retryAfterSeconds?: number,
    rateLimitType?: string
  ) {
    super(message, 429);
    this.name = 'TrustformersRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
    this.rateLimitType = rateLimitType;
    
    if (retryAfterSeconds) {
      this.details.retry_after_seconds = retryAfterSeconds;
    }
    if (rateLimitType) {
      this.details.rate_limit_type = rateLimitType;
    }
  }
}

/**
 * Error for streaming-related errors
 * 
 * Thrown when there are issues with streaming inference requests
 */
export class TrustformersStreamingError extends TrustformersError {
  public readonly streamId?: string;
  public readonly tokensReceived?: number;

  constructor(
    message: string,
    streamId?: string,
    tokensReceived?: number
  ) {
    const details: Record<string, any> = {};
    if (streamId) {
      details.stream_id = streamId;
    }
    if (tokensReceived !== undefined) {
      details.tokens_received = tokensReceived;
    }
    
    super(message, details);
    this.name = 'TrustformersStreamingError';
    this.streamId = streamId;
    this.tokensReceived = tokensReceived;
  }
}

/**
 * Error for batch processing errors
 * 
 * Thrown when there are issues with batch inference requests
 */
export class TrustformersBatchError extends TrustformersError {
  public readonly batchId?: string;
  public readonly successfulItems?: number;
  public readonly failedItems?: number;
  public readonly partialResults?: any[];

  constructor(
    message: string,
    batchId?: string,
    successfulItems?: number,
    failedItems?: number,
    partialResults?: any[]
  ) {
    const details: Record<string, any> = {
      batch_id: batchId,
      successful_items: successfulItems,
      failed_items: failedItems,
    };
    
    super(message, details);
    this.name = 'TrustformersBatchError';
    this.batchId = batchId;
    this.successfulItems = successfulItems;
    this.failedItems = failedItems;
    this.partialResults = partialResults || [];
  }
}

// Utility functions for error handling

/**
 * Check if an error is retryable
 */
export function isRetryableError(error: Error): boolean {
  // TrustformersAPIError has built-in retry logic
  if (error instanceof TrustformersAPIError) {
    return error.isRetryable;
  }
  
  // Connection and timeout errors are generally retryable
  if (error instanceof TrustformersConnectionError || error instanceof TrustformersTimeoutError) {
    return true;
  }
  
  // Rate limit errors are retryable after waiting
  if (error instanceof TrustformersRateLimitError) {
    return true;
  }
  
  // Resource errors might be retryable after some time
  if (error instanceof TrustformersResourceError) {
    return true;
  }
  
  // Other errors are generally not retryable
  return false;
}

/**
 * Get the recommended retry delay for an error
 */
export function getRetryDelay(error: Error): number | null {
  if (error instanceof TrustformersRateLimitError && error.retryAfterSeconds) {
    return error.retryAfterSeconds;
  }
  
  // Default exponential backoff delays
  if (error instanceof TrustformersConnectionError) {
    return 2.0;
  }
  
  if (error instanceof TrustformersTimeoutError) {
    return 5.0;
  }
  
  if (error instanceof TrustformersResourceError) {
    return 10.0;
  }
  
  if (error instanceof TrustformersAPIError && error.isServerError) {
    return 3.0;
  }
  
  return null;
}

/**
 * Extract structured error details from an exception
 */
export function extractErrorDetails(error: Error): Record<string, any> {
  const details: Record<string, any> = {
    error_type: error.constructor.name,
    message: error.message,
    retryable: isRetryableError(error),
  };
  
  // Add retry delay if available
  const retryDelay = getRetryDelay(error);
  if (retryDelay !== null) {
    details.retry_delay_seconds = retryDelay;
  }
  
  // Add exception-specific details
  if (error instanceof TrustformersError) {
    Object.assign(details, error.details);
  }
  
  return details;
}

// HTTP status code to exception mapping
const HTTP_STATUS_TO_EXCEPTION: Record<number, typeof TrustformersError> = {
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
};

/**
 * Create an appropriate exception from an HTTP response
 */
export function exceptionFromResponse(
  statusCode: number,
  message: string,
  responseData?: Record<string, any>,
  requestId?: string
): TrustformersError {
  const ExceptionClass = HTTP_STATUS_TO_EXCEPTION[statusCode] || TrustformersAPIError;
  
  // Handle special cases
  if (statusCode === 429) {
    const retryAfter = responseData?.retry_after_seconds;
    const rateLimitType = responseData?.rate_limit_type;
    
    return new TrustformersRateLimitError(message, retryAfter, rateLimitType);
  }
  
  if (ExceptionClass === TrustformersAPIError) {
    return new TrustformersAPIError(message, statusCode, responseData, requestId);
  }
  
  // For other exception types, handle constructor variations
  if (ExceptionClass === TrustformersValidationError) {
    const fieldErrors = responseData?.field_errors;
    const invalidFields = responseData?.invalid_fields;
    return new TrustformersValidationError(message, fieldErrors, invalidFields);
  }
  
  if (ExceptionClass === TrustformersAuthenticationError) {
    const authType = responseData?.auth_type;
    const requiredScopes = responseData?.required_scopes;
    return new TrustformersAuthenticationError(message, authType, requiredScopes);
  }
  
  if (ExceptionClass === TrustformersTimeoutError) {
    const timeoutSeconds = responseData?.timeout_seconds;
    return new TrustformersTimeoutError(message, timeoutSeconds);
  }
  
  if (ExceptionClass === TrustformersConnectionError) {
    const baseUrl = responseData?.base_url;
    return new TrustformersConnectionError(message, baseUrl);
  }
  
  if (ExceptionClass === TrustformersResourceError) {
    const resourceType = responseData?.resource_type;
    const requiredAmount = responseData?.required_amount;
    const availableAmount = responseData?.available_amount;
    return new TrustformersResourceError(message, resourceType, requiredAmount, availableAmount);
  }
  
  // Default case
  return new ExceptionClass(message);
}