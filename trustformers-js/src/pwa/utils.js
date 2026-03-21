/**
 * PWA utility functions for TrustformeRS
 * Provides helper functions for Progressive Web App features
 */

/**
 * Check if the app is running as a PWA
 * @returns {boolean}
 */
export function isPWA() {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true ||
    document.referrer.includes('android-app://')
  );
}

/**
 * Check if the app is running in fullscreen mode
 * @returns {boolean}
 */
export function isFullscreen() {
  return window.matchMedia('(display-mode: fullscreen)').matches;
}

/**
 * Check if the device is mobile
 * @returns {boolean}
 */
export function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

/**
 * Check if the device supports touch
 * @returns {boolean}
 */
export function isTouch() {
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
}

/**
 * Get device capabilities
 * @returns {Object}
 */
export function getDeviceCapabilities() {
  return {
    serviceWorker: 'serviceWorker' in navigator,
    pushNotifications: 'Notification' in window && 'serviceWorker' in navigator,
    indexedDB: 'indexedDB' in window,
    webAssembly: 'WebAssembly' in window,
    webGL: (() => {
      try {
        const canvas = document.createElement('canvas');
        return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
      } catch (e) {
        return false;
      }
    })(),
    webGPU: 'gpu' in navigator,
    backgroundSync:
      'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype,
    periodicBackgroundSync:
      'serviceWorker' in navigator && 'periodicSync' in window.ServiceWorkerRegistration.prototype,
    share: 'share' in navigator,
    contactsAPI: 'contacts' in navigator,
    fileSystemAccess: 'showOpenFilePicker' in window,
    clipboardAPI: 'clipboard' in navigator,
    wakelock: 'wakeLock' in navigator,
    deviceMemory: 'deviceMemory' in navigator ? navigator.deviceMemory : null,
    hardwareConcurrency: navigator.hardwareConcurrency || null,
    connection:
      navigator.connection || navigator.mozConnection || navigator.webkitConnection || null,
  };
}

/**
 * Get network information
 * @returns {Object}
 */
export function getNetworkInfo() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;

  if (!connection) {
    return {
      online: navigator.onLine,
      type: 'unknown',
      effectiveType: 'unknown',
      downlink: null,
      rtt: null,
      saveData: false,
    };
  }

  return {
    online: navigator.onLine,
    type: connection.type || 'unknown',
    effectiveType: connection.effectiveType || 'unknown',
    downlink: connection.downlink || null,
    rtt: connection.rtt || null,
    saveData: connection.saveData || false,
  };
}

/**
 * Estimate memory requirements for model
 * @param {Object} modelInfo - Model information
 * @returns {Object}
 */
export function estimateMemoryRequirements(modelInfo) {
  const { parameters, precision = 'fp32', architecture } = modelInfo;

  let bytesPerParameter;
  switch (precision) {
    case 'fp16':
      bytesPerParameter = 2;
      break;
    case 'fp32':
      bytesPerParameter = 4;
      break;
    case 'int8':
      bytesPerParameter = 1;
      break;
    case 'int4':
      bytesPerParameter = 0.5;
      break;
    default:
      bytesPerParameter = 4;
  }

  const modelSize = parameters * bytesPerParameter;
  const overhead = modelSize * 0.3; // 30% overhead for activations, gradients, etc.
  const totalRequired = modelSize + overhead;

  // Architecture-specific adjustments
  let multiplier = 1;
  if (architecture === 'transformer') {
    multiplier = 1.5; // Transformers need more memory for attention
  }

  return {
    modelSize: Math.round(modelSize),
    overhead: Math.round(overhead),
    totalRequired: Math.round(totalRequired * multiplier),
    recommended: Math.round(totalRequired * multiplier * 1.5), // 50% buffer
  };
}

/**
 * Check if device can run model
 * @param {Object} modelInfo - Model information
 * @returns {Object}
 */
export function checkModelCompatibility(modelInfo) {
  const capabilities = getDeviceCapabilities();
  const memoryReq = estimateMemoryRequirements(modelInfo);
  const deviceMemoryGB = capabilities.deviceMemory || 4; // Default to 4GB if unknown
  const deviceMemoryBytes = deviceMemoryGB * 1024 * 1024 * 1024;

  const result = {
    compatible: true,
    warnings: [],
    requirements: memoryReq,
    recommendations: [],
  };

  // Check WebAssembly support
  if (!capabilities.webAssembly) {
    result.compatible = false;
    result.warnings.push('WebAssembly not supported');
  }

  // Check memory requirements
  if (memoryReq.totalRequired > deviceMemoryBytes * 0.7) {
    // Use max 70% of device memory
    result.compatible = false;
    result.warnings.push(
      `Model requires ${Math.round(
        memoryReq.totalRequired / 1024 / 1024 / 1024
      )}GB but device only has ${deviceMemoryGB}GB`
    );
  } else if (memoryReq.totalRequired > deviceMemoryBytes * 0.5) {
    result.warnings.push('Model may cause memory pressure');
    result.recommendations.push('Close other applications before running');
  }

  // Check GPU capabilities for acceleration
  if (capabilities.webGL) {
    result.recommendations.push('WebGL acceleration available');
  } else {
    result.warnings.push('WebGL not available - CPU inference only');
  }

  if (capabilities.webGPU) {
    result.recommendations.push('WebGPU acceleration available');
  }

  // Network recommendations
  const networkInfo = getNetworkInfo();
  if (networkInfo.saveData) {
    result.recommendations.push('Data saver mode detected - consider downloading models on WiFi');
  }

  if (networkInfo.effectiveType === '2g' || networkInfo.effectiveType === 'slow-2g') {
    result.warnings.push('Slow network connection detected');
    result.recommendations.push('Consider pre-downloading models');
  }

  return result;
}

/**
 * Format bytes to human readable string
 * @param {number} bytes - Number of bytes
 * @param {number} decimals - Number of decimal places
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Get cache size information
 * @returns {Promise<Object>}
 */
export async function getCacheSize() {
  if (!('storage' in navigator) || !('estimate' in navigator.storage)) {
    return { quota: null, usage: null, available: null };
  }

  try {
    const estimate = await navigator.storage.estimate();
    return {
      quota: estimate.quota,
      usage: estimate.usage,
      available: estimate.quota - estimate.usage,
      quotaFormatted: formatBytes(estimate.quota),
      usageFormatted: formatBytes(estimate.usage),
      availableFormatted: formatBytes(estimate.quota - estimate.usage),
    };
  } catch (error) {
    console.error('Failed to get storage estimate:', error);
    return { quota: null, usage: null, available: null };
  }
}

/**
 * Request persistent storage
 * @returns {Promise<boolean>}
 */
export async function requestPersistentStorage() {
  if (!('storage' in navigator) || !('persist' in navigator.storage)) {
    return false;
  }

  try {
    const granted = await navigator.storage.persist();
    console.warn(`Persistent storage ${granted ? 'granted' : 'denied'}`);
    return granted;
  } catch (error) {
    console.error('Failed to request persistent storage:', error);
    return false;
  }
}

/**
 * Check if storage is persistent
 * @returns {Promise<boolean>}
 */
export async function isStoragePersistent() {
  if (!('storage' in navigator) || !('persisted' in navigator.storage)) {
    return false;
  }

  try {
    return await navigator.storage.persisted();
  } catch (error) {
    console.error('Failed to check storage persistence:', error);
    return false;
  }
}

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @param {boolean} immediate - Execute immediately
 * @returns {Function}
 */
export function debounce(func, wait, immediate = false) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func(...args);
  };
}

/**
 * Throttle function
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function}
 */
export function throttle(func, limit) {
  let inThrottle;
  return function executedFunction(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Create event emitter for PWA events
 * @returns {Object}
 */
export function createEventEmitter() {
  const events = {};

  return {
    on(event, callback) {
      if (!events[event]) {
        events[event] = [];
      }
      events[event].push(callback);
    },

    off(event, callback) {
      if (events[event]) {
        events[event] = events[event].filter(cb => cb !== callback);
      }
    },

    emit(event, data) {
      if (events[event]) {
        events[event].forEach(callback => {
          try {
            callback(data);
          } catch (error) {
            console.error('Event callback error:', error);
          }
        });
      }
    },

    once(event, callback) {
      const onceCallback = data => {
        callback(data);
        this.off(event, onceCallback);
      };
      this.on(event, onceCallback);
    },
  };
}

/**
 * Performance monitor for PWA
 * @returns {Object}
 */
export function createPerformanceMonitor() {
  const metrics = {
    memory: [],
    timing: [],
    navigation: [],
  };

  const monitor = {
    start() {
      this.startTime = performance.now();
      this.recordMemory();
    },

    end(label = 'operation') {
      const endTime = performance.now();
      const duration = endTime - this.startTime;

      metrics.timing.push({
        label,
        duration,
        timestamp: Date.now(),
      });

      this.recordMemory();
      return duration;
    },

    recordMemory() {
      if (performance.memory) {
        metrics.memory.push({
          used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize,
          limit: performance.memory.jsHeapSizeLimit,
          timestamp: Date.now(),
        });
      }
    },

    getMetrics() {
      return {
        ...metrics,
        navigation: performance.getEntriesByType('navigation'),
        resources: performance.getEntriesByType('resource'),
      };
    },

    clear() {
      metrics.memory.length = 0;
      metrics.timing.length = 0;
      metrics.navigation.length = 0;
    },
  };

  return monitor;
}

/**
 * Wake lock utility
 * @returns {Object}
 */
export function createWakeLock() {
  let wakeLock = null;

  return {
    async request() {
      if (!('wakeLock' in navigator)) {
        console.warn('Wake Lock API not supported');
        return false;
      }

      try {
        wakeLock = await navigator.wakeLock.request('screen');
        console.warn('Wake lock acquired');

        wakeLock.addEventListener('release', () => {
          console.warn('Wake lock released');
        });

        return true;
      } catch (error) {
        console.error('Wake lock request failed:', error);
        return false;
      }
    },

    async release() {
      if (wakeLock) {
        await wakeLock.release();
        wakeLock = null;
      }
    },

    get active() {
      return wakeLock && !wakeLock.released;
    },
  };
}

/**
 * Battery status utility
 * @returns {Promise<Object>}
 */
export async function getBatteryStatus() {
  if (!('getBattery' in navigator)) {
    return {
      supported: false,
      level: null,
      charging: null,
      chargingTime: null,
      dischargingTime: null,
    };
  }

  try {
    const battery = await navigator.getBattery();
    return {
      supported: true,
      level: battery.level,
      charging: battery.charging,
      chargingTime: battery.chargingTime,
      dischargingTime: battery.dischargingTime,
    };
  } catch (error) {
    console.error('Battery status check failed:', error);
    return {
      supported: false,
      level: null,
      charging: null,
      chargingTime: null,
      dischargingTime: null,
    };
  }
}

export default {
  isPWA,
  isFullscreen,
  isMobile,
  isTouch,
  getDeviceCapabilities,
  getNetworkInfo,
  estimateMemoryRequirements,
  checkModelCompatibility,
  formatBytes,
  getCacheSize,
  requestPersistentStorage,
  isStoragePersistent,
  debounce,
  throttle,
  createEventEmitter,
  createPerformanceMonitor,
  createWakeLock,
  getBatteryStatus,
};
