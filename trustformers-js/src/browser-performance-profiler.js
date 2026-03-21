/* global screen */

/**
 * Browser-Specific Performance Profiler for TrustformeRS
 *
 * Advanced performance profiling tailored for browser environments
 * with support for various browser APIs and DevTools integration.
 *
 * Features:
 * - Performance API integration (marks, measures, navigation timing)
 * - Long task monitoring (>50ms blocking)
 * - Memory pressure detection
 * - FPS monitoring
 * - Network waterfall analysis
 * - WebGPU performance counters
 * - DevTools Performance panel integration
 * - Real-time performance dashboard
 * - Automatic bottleneck detection
 * - Performance budgets and alerts
 *
 * @module browser-performance-profiler
 */

/**
 * Performance metric types
 * @enum {string}
 */
export const MetricType = {
  TIMING: 'timing',
  MEMORY: 'memory',
  FPS: 'fps',
  LONG_TASK: 'long_task',
  NETWORK: 'network',
  GPU: 'gpu',
  CUSTOM: 'custom'
};

/**
 * Performance budget thresholds
 */
export const PerformanceBudgets = {
  // Timing budgets (ms)
  FIRST_PAINT: 1000,
  FIRST_CONTENTFUL_PAINT: 1500,
  TIME_TO_INTERACTIVE: 3500,
  LARGEST_CONTENTFUL_PAINT: 2500,

  // Inference budgets (ms)
  MODEL_LOAD: 5000,
  FIRST_INFERENCE: 500,
  SUBSEQUENT_INFERENCE: 100,

  // Memory budgets (MB)
  HEAP_SIZE: 100,
  TOTAL_MEMORY: 500,

  // FPS budget
  MIN_FPS: 30,

  // Long task threshold (ms)
  LONG_TASK_THRESHOLD: 50
};

/**
 * Browser Performance Profiler
 */
export class BrowserPerformanceProfiler {
  /**
   * Create a performance profiler
   * @param {Object} config - Configuration
   */
  constructor(config = {}) {
    this.config = {
      // Enable specific profiling features
      enableMemoryProfiling: true,
      enableFPSMonitoring: true,
      enableLongTaskMonitoring: true,
      enableNetworkMonitoring: true,
      enableGPUProfiling: true,

      // Sampling intervals
      memoryCheckInterval: 1000, // 1 second
      fpsCheckInterval: 100,      // 100ms

      // Performance budgets
      budgets: { ...PerformanceBudgets },

      // Callbacks
      onBudgetExceeded: null,
      onLongTask: null,
      onMemoryPressure: null,

      ...config
    };

    // Profiling state
    this.metrics = new Map();
    this.marks = new Map();
    this.longTasks = [];
    this.networkRequests = [];

    // Monitoring state
    this.monitoringInterval = null;
    this.fpsMonitor = null;
    this.longTaskObserver = null;
    this.memoryPressureObserver = null;

    // Statistics
    this.stats = {
      totalMarks: 0,
      totalMeasures: 0,
      totalLongTasks: 0,
      averageFPS: 0,
      peakMemoryUsage: 0
    };

    // Initialize browser APIs
    this.initializeBrowserAPIs();
  }

  /**
   * Initialize browser performance APIs
   */
  initializeBrowserAPIs() {
    // Check Performance API
    this.hasPerformanceAPI = typeof performance !== 'undefined';

    // Check Performance Observer API
    this.hasPerformanceObserver = typeof PerformanceObserver !== 'undefined';

    // Check Memory API
    this.hasMemoryAPI = performance && performance.memory;

    // Check Navigation Timing API
    this.hasNavigationTiming = performance && performance.timing;

    // Check Resource Timing API
    this.hasResourceTiming = performance && performance.getEntriesByType;

    // Check Long Tasks API
    this.hasLongTasksAPI = this.hasPerformanceObserver;

    // Check Layout Instability API (CLS)
    this.hasLayoutInstabilityAPI = this.hasPerformanceObserver;
  }

  /**
   * Start profiling
   */
  start() {
    // Performance profiling started

    // Start memory monitoring
    if (this.config.enableMemoryProfiling && this.hasMemoryAPI) {
      this.startMemoryMonitoring();
    }

    // Start FPS monitoring
    if (this.config.enableFPSMonitoring) {
      this.startFPSMonitoring();
    }

    // Start long task monitoring
    if (this.config.enableLongTaskMonitoring && this.hasLongTasksAPI) {
      this.startLongTaskMonitoring();
    }

    // Start network monitoring
    if (this.config.enableNetworkMonitoring && this.hasResourceTiming) {
      this.startNetworkMonitoring();
    }

    // Mark profiler start
    this.mark('profiler_start');
  }

  /**
   * Stop profiling
   */
  stop() {
    // Performance profiling stopped

    // Stop monitoring intervals
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    if (this.fpsMonitor) {
      cancelAnimationFrame(this.fpsMonitor);
    }

    // Disconnect observers
    if (this.longTaskObserver) {
      this.longTaskObserver.disconnect();
    }

    if (this.memoryPressureObserver) {
      this.memoryPressureObserver.disconnect();
    }

    // Mark profiler stop
    this.mark('profiler_stop');
    this.measure('total_profiling_duration', 'profiler_start', 'profiler_stop');
  }

  /**
   * Create a performance mark
   * @param {string} name - Mark name
   * @param {Object} [detail] - Additional details
   */
  mark(name, detail = {}) {
    if (!this.hasPerformanceAPI) return;

    const timestamp = performance.now();

    try {
      performance.mark(name, { detail });
    } catch {
      // Fallback for older browsers
      performance.mark(name);
    }

    this.marks.set(name, {
      timestamp,
      detail
    });

    this.stats.totalMarks++;
  }

  /**
   * Create a performance measure
   * @param {string} name - Measure name
   * @param {string} startMark - Start mark name
   * @param {string} [endMark] - End mark name (current time if not provided)
   * @returns {number} Duration in milliseconds
   */
  measure(name, startMark, endMark) {
    if (!this.hasPerformanceAPI) return 0;

    try {
      const measure = performance.measure(name, startMark, endMark);
      const { duration } = measure;

      this.metrics.set(name, {
        type: MetricType.TIMING,
        value: duration,
        timestamp: performance.now(),
        startMark,
        endMark
      });

      this.stats.totalMeasures++;

      // Check against budgets
      this.checkBudget(name, duration);

      return duration;
    } catch (error) {
      console.warn(`Failed to measure ${name}:`, error);
      return 0;
    }
  }

  /**
   * Profile an async operation
   * @param {string} name - Operation name
   * @param {Function} fn - Async function to profile
   * @returns {Promise<any>} Function result
   */
  async profileAsync(name, fn) {
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

  /**
   * Profile a synchronous operation
   * @param {string} name - Operation name
   * @param {Function} fn - Function to profile
   * @returns {any} Function result
   */
  profileSync(name, fn) {
    const startMark = `${name}_start`;
    const endMark = `${name}_end`;

    this.mark(startMark);

    try {
      const result = fn();
      this.mark(endMark);
      this.measure(name, startMark, endMark);
      return result;
    } catch (error) {
      this.mark(endMark);
      this.measure(name, startMark, endMark);
      throw error;
    }
  }

  /**
   * Start memory monitoring
   */
  startMemoryMonitoring() {
    this.monitoringInterval = setInterval(() => {
      if (!performance.memory) return;

      const memoryInfo = {
        usedJSHeapSize: performance.memory.usedJSHeapSize / (1024 * 1024), // MB
        totalJSHeapSize: performance.memory.totalJSHeapSize / (1024 * 1024),
        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / (1024 * 1024)
      };

      this.metrics.set('memory', {
        type: MetricType.MEMORY,
        value: memoryInfo,
        timestamp: performance.now()
      });

      // Update peak memory
      if (memoryInfo.usedJSHeapSize > this.stats.peakMemoryUsage) {
        this.stats.peakMemoryUsage = memoryInfo.usedJSHeapSize;
      }

      // Check memory pressure
      const usagePercent = (memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit) * 100;
      if (usagePercent > 80) {
        this.handleMemoryPressure(memoryInfo);
      }

      // Check budget
      if (memoryInfo.usedJSHeapSize > this.config.budgets.HEAP_SIZE) {
        this.handleBudgetExceeded('memory_heap', memoryInfo.usedJSHeapSize);
      }
    }, this.config.memoryCheckInterval);
  }

  /**
   * Start FPS monitoring
   */
  startFPSMonitoring() {
    let lastTime = performance.now();
    let frames = 0;
    let fps = 0;

    const measureFPS = () => {
      const currentTime = performance.now();
      frames++;

      if (currentTime >= lastTime + 1000) {
        fps = Math.round((frames * 1000) / (currentTime - lastTime));

        this.metrics.set('fps', {
          type: MetricType.FPS,
          value: fps,
          timestamp: currentTime
        });

        this.stats.averageFPS = fps;

        // Check budget
        if (fps < this.config.budgets.MIN_FPS) {
          this.handleBudgetExceeded('fps', fps);
        }

        frames = 0;
        lastTime = currentTime;
      }

      this.fpsMonitor = requestAnimationFrame(measureFPS);
    };

    measureFPS();
  }

  /**
   * Start long task monitoring
   */
  startLongTaskMonitoring() {
    try {
      this.longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > this.config.budgets.LONG_TASK_THRESHOLD) {
            const longTask = {
              duration: entry.duration,
              startTime: entry.startTime,
              name: entry.name,
              attribution: entry.attribution
            };

            this.longTasks.push(longTask);
            this.stats.totalLongTasks++;

            if (this.config.onLongTask) {
              this.config.onLongTask(longTask);
            }

            console.warn(`Long task detected: ${entry.duration.toFixed(2)}ms`);
          }
        }
      });

      this.longTaskObserver.observe({ entryTypes: ['longtask'] });
    } catch (error) {
      console.warn('Long task monitoring not supported:', error);
    }
  }

  /**
   * Start network monitoring
   */
  startNetworkMonitoring() {
    if (!this.hasPerformanceObserver) return;

    try {
      const networkObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'resource') {
            const resource = {
              name: entry.name,
              duration: entry.duration,
              transferSize: entry.transferSize,
              encodedBodySize: entry.encodedBodySize,
              decodedBodySize: entry.decodedBodySize,
              type: entry.initiatorType,
              startTime: entry.startTime
            };

            this.networkRequests.push(resource);
          }
        }
      });

      networkObserver.observe({ entryTypes: ['resource'] });
    } catch (error) {
      console.warn('Network monitoring not supported:', error);
    }
  }

  /**
   * Check performance budget
   * @param {string} metric - Metric name
   * @param {number} value - Metric value
   */
  checkBudget(metric, value) {
    const budgetKey = metric.toUpperCase().replace(/-/g, '_');
    const budget = this.config.budgets[budgetKey];

    if (budget && value > budget) {
      this.handleBudgetExceeded(metric, value, budget);
    }
  }

  /**
   * Handle budget exceeded
   * @param {string} metric - Metric name
   * @param {number} value - Actual value
   * @param {number} [budget] - Budget value
   */
  handleBudgetExceeded(metric, value, budget) {
    console.warn(`Performance budget exceeded: ${metric} = ${value} (budget: ${budget})`);

    if (this.config.onBudgetExceeded) {
      this.config.onBudgetExceeded({ metric, value, budget });
    }
  }

  /**
   * Handle memory pressure
   * @param {Object} memoryInfo - Memory information
   */
  handleMemoryPressure(memoryInfo) {
    console.warn('Memory pressure detected:', memoryInfo);

    if (this.config.onMemoryPressure) {
      this.config.onMemoryPressure(memoryInfo);
    }
  }

  /**
   * Get Web Vitals metrics
   * @returns {Object} Web Vitals
   */
  getWebVitals() {
    const vitals = {};

    if (!this.hasPerformanceAPI) return vitals;

    // First Contentful Paint (FCP)
    const fcpEntries = performance.getEntriesByName('first-contentful-paint');
    if (fcpEntries.length > 0) {
      vitals.FCP = fcpEntries[0].startTime;
    }

    // Largest Contentful Paint (LCP)
    const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
    if (lcpEntries.length > 0) {
      vitals.LCP = lcpEntries[lcpEntries.length - 1].startTime;
    }

    // First Input Delay (FID) - requires user interaction
    const fidEntries = performance.getEntriesByType('first-input');
    if (fidEntries.length > 0) {
      vitals.FID = fidEntries[0].processingStart - fidEntries[0].startTime;
    }

    // Cumulative Layout Shift (CLS)
    const clsEntries = performance.getEntriesByType('layout-shift');
    if (clsEntries.length > 0) {
      vitals.CLS = clsEntries.reduce((sum, entry) => sum + entry.value, 0);
    }

    // Time to First Byte (TTFB)
    if (this.hasNavigationTiming) {
      const nav = performance.timing;
      vitals.TTFB = nav.responseStart - nav.requestStart;
    }

    return vitals;
  }

  /**
   * Get navigation timing breakdown
   * @returns {Object} Navigation timing
   */
  getNavigationTiming() {
    if (!this.hasNavigationTiming) return {};

    const nav = performance.timing;
    const navigation = {};

    // Calculate timing phases
    navigation.redirect = nav.redirectEnd - nav.redirectStart;
    navigation.dns = nav.domainLookupEnd - nav.domainLookupStart;
    navigation.tcp = nav.connectEnd - nav.connectStart;
    navigation.request = nav.responseStart - nav.requestStart;
    navigation.response = nav.responseEnd - nav.responseStart;
    navigation.domProcessing = nav.domContentLoadedEventStart - nav.domLoading;
    navigation.domContentLoaded = nav.domContentLoadedEventEnd - nav.domContentLoadedEventStart;
    navigation.loadEvent = nav.loadEventEnd - nav.loadEventStart;

    // Total time
    navigation.total = nav.loadEventEnd - nav.navigationStart;

    return navigation;
  }

  /**
   * Generate comprehensive report
   * @returns {Object} Performance report
   */
  generateReport() {
    return {
      timestamp: new Date().toISOString(),
      duration: this.marks.has('profiler_start')
        ? performance.now() - this.marks.get('profiler_start').timestamp
        : 0,

      // Summary statistics
      stats: this.stats,

      // Web Vitals
      webVitals: this.getWebVitals(),

      // Navigation timing
      navigationTiming: this.getNavigationTiming(),

      // Current metrics
      metrics: Object.fromEntries(this.metrics),

      // Long tasks
      longTasks: this.longTasks.slice(-20), // Last 20

      // Network requests
      networkRequests: this.networkRequests.slice(-50), // Last 50

      // Memory info
      memory: this.metrics.get('memory')?.value || null,

      // FPS
      fps: this.metrics.get('fps')?.value || null,

      // Browser info
      browser: this.getBrowserInfo(),

      // Device info
      device: this.getDeviceInfo()
    };
  }

  /**
   * Get browser information
   * @returns {Object} Browser info
   */
  getBrowserInfo() {
    const ua = navigator.userAgent;

    return {
      userAgent: ua,
      vendor: navigator.vendor,
      platform: navigator.platform,
      language: navigator.language,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      hardwareConcurrency: navigator.hardwareConcurrency || 'unknown'
    };
  }

  /**
   * Get device information
   * @returns {Object} Device info
   */
  getDeviceInfo() {
    return {
      screen: {
        width: screen.width,
        height: screen.height,
        availWidth: screen.availWidth,
        availHeight: screen.availHeight,
        colorDepth: screen.colorDepth,
        pixelDepth: screen.pixelDepth
      },
      devicePixelRatio: window.devicePixelRatio || 1,
      touchSupport: 'ontouchstart' in window || navigator.maxTouchPoints > 0
    };
  }

  /**
   * Export report as JSON
   * @returns {string} JSON report
   */
  exportJSON() {
    return JSON.stringify(this.generateReport(), null, 2);
  }

  /**
   * Export report to console
   */
  printReport() {
    /* eslint-disable no-console */
    console.group('📊 Performance Report');
    console.table(this.stats);
    console.groupEnd();

    console.group('🌐 Web Vitals');
    console.table(this.getWebVitals());
    console.groupEnd();

    if (this.longTasks.length > 0) {
      console.group('⚠️ Long Tasks');
      console.table(this.longTasks.slice(-10));
      console.groupEnd();
    }

    console.log('Full report:', this.generateReport());
    /* eslint-enable no-console */
  }
}

/**
 * Create a browser performance profiler
 * @param {Object} config - Configuration
 * @returns {BrowserPerformanceProfiler}
 */
export function createBrowserProfiler(config) {
  return new BrowserPerformanceProfiler(config);
}

export default {
  BrowserPerformanceProfiler,
  MetricType,
  PerformanceBudgets,
  createBrowserProfiler
};
