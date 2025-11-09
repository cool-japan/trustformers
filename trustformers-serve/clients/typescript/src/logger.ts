/**
 * TrustformeRS TypeScript Client Logger
 * 
 * Provides logging functionality for TrustformeRS client library.
 * Supports multiple log levels, formatters, and output destinations.
 */

/**
 * Log levels in order of severity
 */
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4,
}

/**
 * Log entry interface
 */
export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: Date;
  context?: string;
  metadata?: Record<string, any>;
  error?: Error;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  /** Enable or disable logging */
  enabled: boolean;
  /** Minimum log level to output */
  level: LogLevel;
  /** Logger name/prefix */
  prefix?: string;
  /** Custom formatter function */
  formatter?: LogFormatter;
  /** Output destinations */
  outputs?: LogOutput[];
  /** Include timestamps in logs */
  includeTimestamp: boolean;
  /** Include stack traces for errors */
  includeStackTrace: boolean;
  /** Maximum log message length */
  maxMessageLength?: number;
  /** Context to include in all log messages */
  context?: Record<string, any>;
}

/**
 * Log formatter interface
 */
export interface LogFormatter {
  format(entry: LogEntry): string;
}

/**
 * Log output interface
 */
export interface LogOutput {
  write(formattedMessage: string, entry: LogEntry): void;
}

/**
 * Default logger configuration
 */
const DEFAULT_LOGGER_CONFIG: LoggerConfig = {
  enabled: true,
  level: LogLevel.INFO,
  includeTimestamp: true,
  includeStackTrace: true,
  outputs: [],
};

/**
 * Console log output
 */
export class ConsoleOutput implements LogOutput {
  write(formattedMessage: string, entry: LogEntry): void {
    switch (entry.level) {
      case LogLevel.DEBUG:
        console.debug(formattedMessage);
        break;
      case LogLevel.INFO:
        console.info(formattedMessage);
        break;
      case LogLevel.WARN:
        console.warn(formattedMessage);
        break;
      case LogLevel.ERROR:
        console.error(formattedMessage);
        break;
    }
  }
}

/**
 * Buffer log output for collecting logs in memory
 */
export class BufferOutput implements LogOutput {
  private buffer: Array<{ message: string; entry: LogEntry }> = [];
  private readonly maxSize: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  write(formattedMessage: string, entry: LogEntry): void {
    this.buffer.push({ message: formattedMessage, entry });
    
    if (this.buffer.length > this.maxSize) {
      this.buffer = this.buffer.slice(-this.maxSize);
    }
  }

  getBuffer(): Array<{ message: string; entry: LogEntry }> {
    return [...this.buffer];
  }

  clear(): void {
    this.buffer = [];
  }

  export(): string {
    return this.buffer.map(item => item.message).join('\n');
  }
}

/**
 * File log output (Node.js only)
 */
export class FileOutput implements LogOutput {
  private readonly filePath: string;
  private writeQueue: string[] = [];
  private writing = false;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  write(formattedMessage: string, entry: LogEntry): void {
    if (typeof require === 'undefined') {
      console.warn('FileOutput is only supported in Node.js environments');
      return;
    }

    this.writeQueue.push(formattedMessage + '\n');
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.writing || this.writeQueue.length === 0) {
      return;
    }

    this.writing = true;

    try {
      const fs = require('fs').promises;
      const messages = this.writeQueue.splice(0);
      const content = messages.join('');
      await fs.appendFile(this.filePath, content);
    } catch (error) {
      console.error('Failed to write to log file:', error);
    } finally {
      this.writing = false;
      
      // Process any messages that were added while writing
      if (this.writeQueue.length > 0) {
        this.processQueue();
      }
    }
  }
}

/**
 * Simple text formatter
 */
export class SimpleFormatter implements LogFormatter {
  format(entry: LogEntry): string {
    const parts: string[] = [];

    if (entry.timestamp) {
      parts.push(`[${entry.timestamp.toISOString()}]`);
    }

    parts.push(`[${LogLevel[entry.level]}]`);

    if (entry.context) {
      parts.push(`[${entry.context}]`);
    }

    parts.push(entry.message);

    if (entry.metadata && Object.keys(entry.metadata).length > 0) {
      parts.push(`- ${JSON.stringify(entry.metadata)}`);
    }

    if (entry.error && entry.error.stack) {
      parts.push(`\n${entry.error.stack}`);
    }

    return parts.join(' ');
  }
}

/**
 * JSON formatter
 */
export class JSONFormatter implements LogFormatter {
  format(entry: LogEntry): string {
    const logObject = {
      timestamp: entry.timestamp.toISOString(),
      level: LogLevel[entry.level],
      message: entry.message,
      context: entry.context,
      metadata: entry.metadata,
      error: entry.error ? {
        name: entry.error.name,
        message: entry.error.message,
        stack: entry.error.stack,
      } : undefined,
    };

    return JSON.stringify(logObject);
  }
}

/**
 * Colorized console formatter
 */
export class ColorizedFormatter implements LogFormatter {
  private readonly colors = {
    [LogLevel.DEBUG]: '\x1b[36m', // Cyan
    [LogLevel.INFO]: '\x1b[32m',  // Green
    [LogLevel.WARN]: '\x1b[33m',  // Yellow
    [LogLevel.ERROR]: '\x1b[31m', // Red
  };
  private readonly reset = '\x1b[0m';

  format(entry: LogEntry): string {
    const color = this.colors[entry.level] || '';
    const levelName = LogLevel[entry.level];
    
    let message = `${color}[${levelName}]${this.reset} ${entry.message}`;
    
    if (entry.context) {
      message = `${color}[${entry.context}]${this.reset} ${message}`;
    }
    
    if (entry.metadata && Object.keys(entry.metadata).length > 0) {
      message += ` ${JSON.stringify(entry.metadata)}`;
    }
    
    if (entry.error && entry.error.stack) {
      message += `\n${entry.error.stack}`;
    }
    
    return message;
  }
}

/**
 * Logger class
 */
export class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...DEFAULT_LOGGER_CONFIG, ...config };
    
    // Set default outputs if none provided
    if (!this.config.outputs || this.config.outputs.length === 0) {
      this.config.outputs = [new ConsoleOutput()];
    }
    
    // Set default formatter if none provided
    if (!this.config.formatter) {
      this.config.formatter = typeof window !== 'undefined' 
        ? new ColorizedFormatter() 
        : new SimpleFormatter();
    }
  }

  /**
   * Log a debug message
   */
  debug(message: string, metadata?: Record<string, any>): void {
    this.log(LogLevel.DEBUG, message, metadata);
  }

  /**
   * Log an info message
   */
  info(message: string, metadata?: Record<string, any>): void {
    this.log(LogLevel.INFO, message, metadata);
  }

  /**
   * Log a warning message
   */
  warn(message: string, metadata?: Record<string, any>): void {
    this.log(LogLevel.WARN, message, metadata);
  }

  /**
   * Log an error message
   */
  error(message: string, error?: Error, metadata?: Record<string, any>): void {
    this.log(LogLevel.ERROR, message, metadata, error);
  }

  /**
   * Log a message at specified level
   */
  log(level: LogLevel, message: string, metadata?: Record<string, any>, error?: Error): void {
    if (!this.config.enabled || level < this.config.level) {
      return;
    }

    // Truncate message if needed
    if (this.config.maxMessageLength && message.length > this.config.maxMessageLength) {
      message = message.substring(0, this.config.maxMessageLength) + '...';
    }

    // Create log entry
    const entry: LogEntry = {
      level,
      message: this.config.prefix ? `[${this.config.prefix}] ${message}` : message,
      timestamp: new Date(),
      context: this.config.prefix,
      metadata: { ...this.config.context, ...metadata },
      error: error,
    };

    // Format and output
    const formattedMessage = this.config.formatter!.format(entry);
    
    for (const output of this.config.outputs!) {
      try {
        output.write(formattedMessage, entry);
      } catch (outputError) {
        console.error('Logger output error:', outputError);
      }
    }
  }

  /**
   * Create a child logger with additional context
   */
  child(context: string, additionalConfig?: Partial<LoggerConfig>): Logger {
    const childPrefix = this.config.prefix 
      ? `${this.config.prefix}:${context}`
      : context;

    return new Logger({
      ...this.config,
      ...additionalConfig,
      prefix: childPrefix,
      context: { ...this.config.context, ...additionalConfig?.context },
    });
  }

  /**
   * Update logger configuration
   */
  updateConfig(updates: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...updates };
  }

  /**
   * Get current configuration
   */
  getConfig(): LoggerConfig {
    return { ...this.config };
  }

  /**
   * Enable/disable logging
   */
  setEnabled(enabled: boolean): void {
    this.config.enabled = enabled;
  }

  /**
   * Set log level
   */
  setLevel(level: LogLevel): void {
    this.config.level = level;
  }

  /**
   * Add output destination
   */
  addOutput(output: LogOutput): void {
    if (!this.config.outputs) {
      this.config.outputs = [];
    }
    this.config.outputs.push(output);
  }

  /**
   * Remove all outputs
   */
  clearOutputs(): void {
    this.config.outputs = [];
  }

  /**
   * Time a function execution
   */
  async time<T>(label: string, fn: () => Promise<T>): Promise<T> {
    const start = Date.now();
    this.debug(`${label} started`);
    
    try {
      const result = await fn();
      const duration = Date.now() - start;
      this.debug(`${label} completed`, { duration: `${duration}ms` });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      this.error(`${label} failed`, error as Error, { duration: `${duration}ms` });
      throw error;
    }
  }

  /**
   * Log function entry and exit
   */
  trace<T>(label: string, fn: () => T): T {
    this.debug(`→ ${label}`);
    
    try {
      const result = fn();
      this.debug(`← ${label}`);
      return result;
    } catch (error) {
      this.debug(`✗ ${label}`, { error: (error as Error).message });
      throw error;
    }
  }
}

// Global logger instance
let globalLogger: Logger | null = null;

/**
 * Get or create global logger instance
 */
export function getGlobalLogger(): Logger {
  if (!globalLogger) {
    globalLogger = new Logger();
  }
  return globalLogger;
}

/**
 * Set global logger instance
 */
export function setGlobalLogger(logger: Logger): void {
  globalLogger = logger;
}

/**
 * Create logger with configuration
 */
export function createLogger(config: Partial<LoggerConfig> = {}): Logger {
  return new Logger(config);
}

/**
 * Create logger for specific environment
 */
export function createEnvironmentLogger(environment: 'development' | 'staging' | 'production'): Logger {
  const configs = {
    development: {
      enabled: true,
      level: LogLevel.DEBUG,
      formatter: new ColorizedFormatter(),
    },
    staging: {
      enabled: true,
      level: LogLevel.INFO,
      formatter: new JSONFormatter(),
    },
    production: {
      enabled: true,
      level: LogLevel.WARN,
      formatter: new JSONFormatter(),
      includeStackTrace: false,
    },
  };

  return new Logger(configs[environment]);
}

/**
 * Logger middleware for request/response logging
 */
export class LoggerMiddleware {
  constructor(private logger: Logger) {}

  logRequest(method: string, url: string, headers?: Record<string, string>): void {
    this.logger.debug('HTTP Request', {
      method,
      url,
      headers: this.sanitizeHeaders(headers),
    });
  }

  logResponse(
    method: string,
    url: string,
    statusCode: number,
    duration: number,
    size?: number
  ): void {
    const level = statusCode >= 400 ? LogLevel.WARN : LogLevel.DEBUG;
    this.logger.log(level, 'HTTP Response', {
      method,
      url,
      statusCode,
      duration: `${duration}ms`,
      size: size ? `${size} bytes` : undefined,
    });
  }

  logError(method: string, url: string, error: Error): void {
    this.logger.error('HTTP Error', error, {
      method,
      url,
    });
  }

  private sanitizeHeaders(headers?: Record<string, string>): Record<string, string> | undefined {
    if (!headers) return undefined;

    const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];
    const sanitized: Record<string, string> = {};

    for (const [key, value] of Object.entries(headers)) {
      if (sensitiveHeaders.includes(key.toLowerCase())) {
        sanitized[key] = '[REDACTED]';
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }
}

// Convenience logging functions that use global logger
export const log = {
  debug: (message: string, metadata?: Record<string, any>) => 
    getGlobalLogger().debug(message, metadata),
  info: (message: string, metadata?: Record<string, any>) => 
    getGlobalLogger().info(message, metadata),
  warn: (message: string, metadata?: Record<string, any>) => 
    getGlobalLogger().warn(message, metadata),
  error: (message: string, error?: Error, metadata?: Record<string, any>) => 
    getGlobalLogger().error(message, error, metadata),
};