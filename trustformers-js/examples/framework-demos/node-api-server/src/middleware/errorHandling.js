/**
 * Error Handling Middleware
 * Centralized error handling for the API server
 */

import logger from '../utils/logger.js';

/**
 * 404 Not Found handler
 */
export function notFoundHandler(req, res, next) {
  const error = new Error(`Route not found: ${req.method} ${req.path}`);
  error.status = 404;
  
  logger.warn('Route not found', {
    method: req.method,
    path: req.path,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    requestId: req.requestId,
  });
  
  res.status(404).json({
    success: false,
    error: '404 Not Found',
    message: `Route ${req.method} ${req.path} not found`,
    requestId: req.requestId,
    timestamp: new Date().toISOString(),
    availableEndpoints: {
      documentation: '/docs',
      health: '/health',
      analytics: '/api/v1/analytics',
    },
  });
}

/**
 * Global error handler
 */
export function errorHandler(error, req, res, next) {
  // Default error status
  const status = error.status || error.statusCode || 500;
  const isOperational = error.isOperational || status < 500;
  
  // Log error with context
  logger.errorWithContext(error, {
    endpoint: req.path,
    method: req.method,
    requestId: req.requestId,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    body: req.body ? JSON.stringify(req.body).substring(0, 500) : undefined,
    status,
    isOperational,
  });
  
  // Don't leak error details in production for 5xx errors
  const shouldExposeError = process.env.NODE_ENV === 'development' || isOperational;
  
  const errorResponse = {
    success: false,
    error: getErrorType(status),
    message: shouldExposeError ? error.message : 'Internal server error',
    requestId: req.requestId,
    timestamp: new Date().toISOString(),
  };
  
  // Add error details in development
  if (process.env.NODE_ENV === 'development') {
    errorResponse.stack = error.stack;
    errorResponse.details = {
      name: error.name,
      code: error.code,
      status: error.status,
    };
  }
  
  // Add retry information for rate limiting
  if (status === 429) {
    errorResponse.retryAfter = error.retryAfter || '15 minutes';
  }
  
  // Add validation details for 400 errors
  if (status === 400 && error.details) {
    errorResponse.validationErrors = error.details;
  }
  
  res.status(status).json(errorResponse);
}

/**
 * Get user-friendly error type based on status code
 */
function getErrorType(status) {
  switch (status) {
    case 400:
      return 'Bad Request';
    case 401:
      return 'Unauthorized';
    case 403:
      return 'Forbidden';
    case 404:
      return 'Not Found';
    case 409:
      return 'Conflict';
    case 422:
      return 'Validation Error';
    case 429:
      return 'Rate Limit Exceeded';
    case 500:
      return 'Internal Server Error';
    case 502:
      return 'Bad Gateway';
    case 503:
      return 'Service Unavailable';
    case 504:
      return 'Gateway Timeout';
    default:
      return status >= 500 ? 'Server Error' : 'Client Error';
  }
}

/**
 * Async error wrapper for route handlers
 */
export function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

/**
 * Create operational error
 */
export function createError(message, status = 500, details = null) {
  const error = new Error(message);
  error.status = status;
  error.isOperational = true;
  if (details) {
    error.details = details;
  }
  return error;
}