/**
 * TrustformeRS TypeScript Client Monitoring
 * 
 * Provides request tracking, performance monitoring, and metrics collection
 * for TrustformeRS client interactions.
 */

/**
 * Request tracking data
 */
export interface RequestRecord {
  /** Request method */
  method: string;
  /** Request endpoint */
  endpoint: string;
  /** Request start timestamp */
  startTime: number;
  /** Request end timestamp */
  endTime?: number;
  /** Request duration in milliseconds */
  duration?: number;
  /** HTTP status code */
  statusCode?: number;
  /** Request size in bytes */
  requestSize?: number;
  /** Response size in bytes */
  responseSize?: number;
  /** Whether request was successful */
  success?: boolean;
  /** Error message if request failed */
  error?: string;
  /** Request ID for correlation */
  requestId?: string;
  /** Additional metadata */
  metadata?: Record<string, any>;
}

/**
 * Aggregated metrics for requests
 */
export interface RequestMetrics {
  /** Total number of requests */
  totalRequests: number;
  /** Number of successful requests */
  successfulRequests: number;
  /** Number of failed requests */
  failedRequests: number;
  /** Success rate (0.0-1.0) */
  successRate: number;
  /** Average response time in milliseconds */
  averageResponseTime: number;
  /** Minimum response time in milliseconds */
  minResponseTime: number;
  /** Maximum response time in milliseconds */
  maxResponseTime: number;
  /** 50th percentile response time */
  p50ResponseTime: number;
  /** 95th percentile response time */
  p95ResponseTime: number;
  /** 99th percentile response time */
  p99ResponseTime: number;
  /** Total data transferred (bytes) */
  totalDataTransferred: number;
  /** Requests per second (over the last minute) */
  requestsPerSecond: number;
  /** Error rate (0.0-1.0) */
  errorRate: number;
  /** Most common errors */
  topErrors: Array<{ error: string; count: number }>;
}

/**
 * Request tracker for monitoring API interactions
 */
export class RequestTracker {
  private records: RequestRecord[] = [];
  private activeRequests = new Map<string, RequestRecord>();
  private readonly maxRecords: number;
  private readonly enableDetailed: boolean;

  constructor(options: {
    maxRecords?: number;
    enableDetailed?: boolean;
  } = {}) {
    this.maxRecords = options.maxRecords ?? 1000;
    this.enableDetailed = options.enableDetailed ?? true;
  }

  /**
   * Record the start of a request
   */
  recordRequestStart(
    method: string,
    endpoint: string,
    requestId?: string,
    metadata?: Record<string, any>
  ): string {
    const id = requestId || this.generateRequestId();
    const record: RequestRecord = {
      method,
      endpoint,
      startTime: this.now(),
      requestId: id,
      metadata,
    };

    this.activeRequests.set(id, record);
    return id;
  }

  /**
   * Record the end of a request
   */
  recordRequestEnd(
    method: string,
    endpoint: string,
    statusCode: number,
    duration?: number,
    requestId?: string,
    error?: string,
    requestSize?: number,
    responseSize?: number
  ): void {
    const id = requestId || this.findActiveRequest(method, endpoint);
    const activeRecord = this.activeRequests.get(id);

    if (!activeRecord) {
      // Create a minimal record if we don't have the start
      const record: RequestRecord = {
        method,
        endpoint,
        startTime: this.now() - (duration || 0),
        endTime: this.now(),
        duration: duration || 0,
        statusCode,
        success: statusCode >= 200 && statusCode < 400,
        error,
        requestSize,
        responseSize,
        requestId: id,
      };
      this.addRecord(record);
      return;
    }

    // Complete the existing record
    const endTime = this.now();
    activeRecord.endTime = endTime;
    activeRecord.duration = duration ?? (endTime - activeRecord.startTime);
    activeRecord.statusCode = statusCode;
    activeRecord.success = statusCode >= 200 && statusCode < 400;
    activeRecord.error = error;
    activeRecord.requestSize = requestSize;
    activeRecord.responseSize = responseSize;

    this.activeRequests.delete(id);
    this.addRecord(activeRecord);
  }

  /**
   * Add a complete request record
   */
  private addRecord(record: RequestRecord): void {
    this.records.push(record);

    // Maintain max records limit
    if (this.records.length > this.maxRecords) {
      this.records = this.records.slice(-this.maxRecords);
    }
  }

  /**
   * Get request statistics
   */
  getStats(): RequestMetrics {
    const completedRecords = this.records.filter(r => r.endTime !== undefined);
    
    if (completedRecords.length === 0) {
      return this.createEmptyMetrics();
    }

    const successfulRequests = completedRecords.filter(r => r.success).length;
    const failedRequests = completedRecords.length - successfulRequests;
    const durations = completedRecords.map(r => r.duration!).filter(d => d !== undefined);
    const totalDataTransferred = completedRecords.reduce(
      (sum, r) => sum + (r.requestSize || 0) + (r.responseSize || 0),
      0
    );

    // Calculate percentiles
    const sortedDurations = [...durations].sort((a, b) => a - b);
    const p50 = this.percentile(sortedDurations, 0.5);
    const p95 = this.percentile(sortedDurations, 0.95);
    const p99 = this.percentile(sortedDurations, 0.99);

    // Calculate requests per second (last minute)
    const oneMinuteAgo = this.now() - 60000;
    const recentRequests = completedRecords.filter(r => r.startTime > oneMinuteAgo);
    const requestsPerSecond = recentRequests.length / 60;

    // Get top errors
    const errorCounts = new Map<string, number>();
    completedRecords
      .filter(r => r.error)
      .forEach(r => {
        const count = errorCounts.get(r.error!) || 0;
        errorCounts.set(r.error!, count + 1);
      });

    const topErrors = Array.from(errorCounts.entries())
      .map(([error, count]) => ({ error, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    return {
      totalRequests: completedRecords.length,
      successfulRequests,
      failedRequests,
      successRate: successfulRequests / completedRecords.length,
      averageResponseTime: durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : 0,
      minResponseTime: durations.length > 0 ? Math.min(...durations) : 0,
      maxResponseTime: durations.length > 0 ? Math.max(...durations) : 0,
      p50ResponseTime: p50,
      p95ResponseTime: p95,
      p99ResponseTime: p99,
      totalDataTransferred,
      requestsPerSecond,
      errorRate: failedRequests / completedRecords.length,
      topErrors,
    };
  }

  /**
   * Get detailed request history
   */
  getRequestHistory(options: {
    limit?: number;
    method?: string;
    endpoint?: string;
    successOnly?: boolean;
    failuresOnly?: boolean;
    since?: number;
  } = {}): RequestRecord[] {
    let filtered = [...this.records];

    if (options.method) {
      filtered = filtered.filter(r => r.method === options.method);
    }

    if (options.endpoint) {
      filtered = filtered.filter(r => r.endpoint === options.endpoint);
    }

    if (options.successOnly) {
      filtered = filtered.filter(r => r.success === true);
    }

    if (options.failuresOnly) {
      filtered = filtered.filter(r => r.success === false);
    }

    if (options.since) {
      filtered = filtered.filter(r => r.startTime >= options.since);
    }

    // Sort by start time (most recent first)
    filtered.sort((a, b) => b.startTime - a.startTime);

    if (options.limit) {
      filtered = filtered.slice(0, options.limit);
    }

    return filtered;
  }

  /**
   * Clear all tracking data
   */
  reset(): void {
    this.records = [];
    this.activeRequests.clear();
  }

  /**
   * Get endpoints by frequency
   */
  getEndpointStats(): Array<{ endpoint: string; count: number; averageResponseTime: number; errorRate: number }> {
    const endpointMap = new Map<string, RequestRecord[]>();

    this.records.forEach(record => {
      const key = `${record.method} ${record.endpoint}`;
      if (!endpointMap.has(key)) {
        endpointMap.set(key, []);
      }
      endpointMap.get(key)!.push(record);
    });

    return Array.from(endpointMap.entries()).map(([endpoint, records]) => {
      const completedRecords = records.filter(r => r.endTime !== undefined);
      const failedRecords = completedRecords.filter(r => r.success === false);
      const durations = completedRecords
        .map(r => r.duration!)
        .filter(d => d !== undefined);

      return {
        endpoint,
        count: completedRecords.length,
        averageResponseTime: durations.length > 0 
          ? durations.reduce((a, b) => a + b, 0) / durations.length 
          : 0,
        errorRate: completedRecords.length > 0 
          ? failedRecords.length / completedRecords.length 
          : 0,
      };
    }).sort((a, b) => b.count - a.count);
  }

  private createEmptyMetrics(): RequestMetrics {
    return {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      successRate: 0,
      averageResponseTime: 0,
      minResponseTime: 0,
      maxResponseTime: 0,
      p50ResponseTime: 0,
      p95ResponseTime: 0,
      p99ResponseTime: 0,
      totalDataTransferred: 0,
      requestsPerSecond: 0,
      errorRate: 0,
      topErrors: [],
    };
  }

  private percentile(sortedArray: number[], p: number): number {
    if (sortedArray.length === 0) return 0;
    
    const index = (sortedArray.length - 1) * p;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
      return sortedArray[lower];
    }
    
    const weight = index - lower;
    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private findActiveRequest(method: string, endpoint: string): string {
    for (const [id, record] of this.activeRequests.entries()) {
      if (record.method === method && record.endpoint === endpoint) {
        return id;
      }
    }
    return this.generateRequestId();
  }

  private now(): number {
    return typeof performance !== 'undefined' && performance.now 
      ? performance.now() + performance.timeOrigin
      : Date.now();
  }
}

/**
 * Performance tracker for monitoring client performance
 */
export class PerformanceTracker {
  private marks = new Map<string, number>();
  private measures = new Map<string, { duration: number; timestamp: number }>();

  /**
   * Mark a performance point
   */
  mark(name: string): void {
    this.marks.set(name, this.now());
    
    // Use Performance API if available
    if (typeof performance !== 'undefined' && performance.mark) {
      performance.mark(name);
    }
  }

  /**
   * Measure time between two marks
   */
  measure(name: string, startMark: string, endMark?: string): number {
    const startTime = this.marks.get(startMark);
    if (!startTime) {
      throw new Error(`Start mark '${startMark}' not found`);
    }

    const endTime = endMark ? this.marks.get(endMark) : this.now();
    if (endMark && !endTime) {
      throw new Error(`End mark '${endMark}' not found`);
    }

    const duration = endTime! - startTime;
    this.measures.set(name, { duration, timestamp: this.now() });

    // Use Performance API if available
    if (typeof performance !== 'undefined' && performance.measure) {
      try {
        performance.measure(name, startMark, endMark);
      } catch {
        // Ignore if marks don't exist in Performance API
      }
    }

    return duration;
  }

  /**
   * Get all measurements
   */
  getMeasures(): Record<string, { duration: number; timestamp: number }> {
    return Object.fromEntries(this.measures);
  }

  /**
   * Clear all marks and measures
   */
  clear(): void {
    this.marks.clear();
    this.measures.clear();
    
    if (typeof performance !== 'undefined' && performance.clearMarks) {
      performance.clearMarks();
      performance.clearMeasures();
    }
  }

  /**
   * Time a function execution
   */
  async time<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const startMark = `${name}_start`;
    const endMark = `${name}_end`;
    
    this.mark(startMark);
    
    try {
      const result = await fn();
      this.mark(endMark);
      this.measure(name, startMark, endMark);
      return result;
    } catch (error) {
      this.mark(endMark);
      this.measure(name, startMark, endMark);
      throw error;
    }
  }

  private now(): number {
    return typeof performance !== 'undefined' && performance.now 
      ? performance.now()
      : Date.now();
  }
}

/**
 * Monitoring client for comprehensive observability
 */
export class MonitoringClient {
  private readonly requestTracker: RequestTracker;
  private readonly performanceTracker: PerformanceTracker;
  private readonly enabledFeatures: Set<string>;

  constructor(options: {
    enableRequestTracking?: boolean;
    enablePerformanceTracking?: boolean;
    maxRequestRecords?: number;
    features?: string[];
  } = {}) {
    this.enabledFeatures = new Set(options.features || ['requests', 'performance']);
    
    if (options.enableRequestTracking !== false && this.enabledFeatures.has('requests')) {
      this.requestTracker = new RequestTracker({
        maxRecords: options.maxRequestRecords,
      });
    } else {
      this.requestTracker = new RequestTracker({ maxRecords: 0 });
    }

    if (options.enablePerformanceTracking !== false && this.enabledFeatures.has('performance')) {
      this.performanceTracker = new PerformanceTracker();
    } else {
      this.performanceTracker = new PerformanceTracker();
    }
  }

  /**
   * Get request tracker instance
   */
  getRequestTracker(): RequestTracker {
    return this.requestTracker;
  }

  /**
   * Get performance tracker instance
   */
  getPerformanceTracker(): PerformanceTracker {
    return this.performanceTracker;
  }

  /**
   * Generate a comprehensive monitoring report
   */
  generateReport(): {
    requestMetrics: RequestMetrics;
    endpointStats: Array<{ endpoint: string; count: number; averageResponseTime: number; errorRate: number }>;
    performanceMeasures: Record<string, { duration: number; timestamp: number }>;
    systemInfo: Record<string, any>;
  } {
    return {
      requestMetrics: this.requestTracker.getStats(),
      endpointStats: this.requestTracker.getEndpointStats(),
      performanceMeasures: this.performanceTracker.getMeasures(),
      systemInfo: this.getSystemInfo(),
    };
  }

  /**
   * Reset all monitoring data
   */
  reset(): void {
    this.requestTracker.reset();
    this.performanceTracker.clear();
  }

  /**
   * Export monitoring data as JSON
   */
  exportData(): string {
    return JSON.stringify(this.generateReport(), null, 2);
  }

  private getSystemInfo(): Record<string, any> {
    const info: Record<string, any> = {
      timestamp: new Date().toISOString(),
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
    };

    // Browser-specific info
    if (typeof window !== 'undefined') {
      info.environment = 'browser';
      info.url = window.location.href;
      info.referrer = document.referrer;
      
      if (navigator.connection) {
        info.connection = {
          effectiveType: (navigator.connection as any).effectiveType,
          downlink: (navigator.connection as any).downlink,
          rtt: (navigator.connection as any).rtt,
        };
      }
    }

    // Node.js-specific info
    if (typeof process !== 'undefined') {
      info.environment = 'node';
      info.nodeVersion = process.version;
      info.platform = process.platform;
      info.arch = process.arch;
    }

    return info;
  }
}

// Utility functions

/**
 * Create a monitoring client with preset configurations
 */
export function createMonitoringClient(
  type: 'minimal' | 'standard' | 'comprehensive'
): MonitoringClient {
  const configs = {
    minimal: {
      enableRequestTracking: true,
      enablePerformanceTracking: false,
      maxRequestRecords: 100,
      features: ['requests'],
    },
    standard: {
      enableRequestTracking: true,
      enablePerformanceTracking: true,
      maxRequestRecords: 500,
      features: ['requests', 'performance'],
    },
    comprehensive: {
      enableRequestTracking: true,
      enablePerformanceTracking: true,
      maxRequestRecords: 1000,
      features: ['requests', 'performance', 'system'],
    },
  };

  return new MonitoringClient(configs[type]);
}