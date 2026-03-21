/**
 * Logger utility using Winston
 * Provides structured logging for the TrustformeRS API server
 */

import winston from 'winston';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const NODE_ENV = process.env.NODE_ENV || 'development';
const LOG_LEVEL = process.env.LOG_LEVEL || (NODE_ENV === 'production' ? 'info' : 'debug');

// Custom log format
const logFormat = winston.format.combine(
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss.SSS'
  }),
  winston.format.errors({ stack: true }),
  winston.format.json(),
  winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
    const logObject = {
      timestamp,
      level,
      message,
      ...(stack && { stack }),
      ...(Object.keys(meta).length > 0 && { meta })
    };
    
    return JSON.stringify(logObject);
  })
);

// Console format for development
const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({
    format: 'HH:mm:ss.SSS'
  }),
  winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
    const metaStr = Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : '';
    const stackStr = stack ? `\n${stack}` : '';
    return `${timestamp} [${level}]: ${message}${metaStr}${stackStr}`;
  })
);

// Create transports
const transports = [];

// Console transport
if (NODE_ENV !== 'test') {
  transports.push(
    new winston.transports.Console({
      level: LOG_LEVEL,
      format: NODE_ENV === 'production' ? logFormat : consoleFormat,
      handleExceptions: true,
      handleRejections: true,
    })
  );
}

// File transports for production
if (NODE_ENV === 'production') {
  // Error log file
  transports.push(
    new winston.transports.File({
      filename: join(__dirname, '../../logs/error.log'),
      level: 'error',
      format: logFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
      handleExceptions: true,
      handleRejections: true,
    })
  );
  
  // Combined log file
  transports.push(
    new winston.transports.File({
      filename: join(__dirname, '../../logs/combined.log'),
      level: LOG_LEVEL,
      format: logFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 10,
    })
  );
}

// Create logger instance
const logger = winston.createLogger({
  level: LOG_LEVEL,
  format: logFormat,
  defaultMeta: {
    service: 'trustformers-api',
    version: process.env.npm_package_version || '1.0.0',
    environment: NODE_ENV,
    pid: process.pid,
  },
  transports,
  exitOnError: false,
});

// Add request correlation ID if available
logger.addRequestId = (req, res, next) => {
  const requestId = req.header('X-Request-ID') || 
                   req.header('X-Correlation-ID') || 
                   Math.random().toString(36).substring(2, 15);
  
  req.requestId = requestId;
  res.setHeader('X-Request-ID', requestId);
  
  // Add to logger context
  req.logger = logger.child({ requestId });
  
  next();
};

// Performance logging helper
logger.performance = (operation, duration, metadata = {}) => {
  logger.info('Performance metric', {
    operation,
    duration: `${duration}ms`,
    ...metadata,
  });
};

// API request logging helper
logger.apiRequest = (req, res, duration) => {
  const { method, url, ip, headers } = req;
  const { statusCode } = res;
  
  logger.info('API Request', {
    method,
    url,
    statusCode,
    duration: `${duration}ms`,
    ip,
    userAgent: headers['user-agent'],
    requestId: req.requestId,
    contentLength: res.get('content-length'),
  });
};

// ML operation logging helper
logger.mlOperation = (operation, model, input, output, duration, metadata = {}) => {
  logger.info('ML Operation', {
    operation,
    model,
    inputSize: typeof input === 'string' ? input.length : JSON.stringify(input).length,
    outputSize: typeof output === 'string' ? output.length : JSON.stringify(output).length,
    duration: `${duration}ms`,
    ...metadata,
  });
};

// Error logging helper with context
logger.errorWithContext = (error, context = {}) => {
  logger.error(error.message, {
    error: {
      name: error.name,
      message: error.message,
      stack: error.stack,
      code: error.code,
    },
    context,
  });
};

// Security event logging
logger.security = (event, details = {}) => {
  logger.warn('Security Event', {
    event,
    timestamp: new Date().toISOString(),
    ...details,
  });
};

// Health check logging
logger.health = (component, status, details = {}) => {
  const level = status === 'healthy' ? 'info' : 'warn';
  logger[level]('Health Check', {
    component,
    status,
    ...details,
  });
};

// Startup/shutdown logging
logger.lifecycle = (event, details = {}) => {
  logger.info('Lifecycle Event', {
    event,
    timestamp: new Date().toISOString(),
    ...details,
  });
};

// Rate limiting logging
logger.rateLimit = (ip, endpoint, remainingRequests = 0) => {
  logger.warn('Rate Limit', {
    event: 'rate_limit_exceeded',
    ip,
    endpoint,
    remainingRequests,
    timestamp: new Date().toISOString(),
  });
};

// Validation error logging
logger.validation = (errors, context = {}) => {
  logger.warn('Validation Error', {
    errors: Array.isArray(errors) ? errors : [errors],
    ...context,
  });
};

// Business logic logging
logger.business = (event, details = {}) => {
  logger.info('Business Event', {
    event,
    timestamp: new Date().toISOString(),
    ...details,
  });
};

// Development helpers
if (NODE_ENV === 'development') {
  // Enable debug logging for development
  logger.debug('Logger initialized in development mode');
  
  // Log uncaught exceptions in development
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', { error: error.message, stack: error.stack });
  });
  
  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection', { reason, promise });
  });
}

// Log startup information
logger.lifecycle('logger_initialized', {
  level: LOG_LEVEL,
  transports: transports.map(t => t.constructor.name),
  environment: NODE_ENV,
});

export default logger;