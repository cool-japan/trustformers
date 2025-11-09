/**
 * TrustformeRS WASM Svelte Integration
 * 
 * Complete Svelte package for integrating TrustformeRS WASM
 * into Svelte applications.
 */

// Core bindings
export {
  // Initialization
  initializeTrustformers,
  enableWebGPU,
  initWasm,
  getWasmInstance,
  
  // Tensor management
  createTensor,
  deleteTensor,
  getTensorData,
  cleanupTensors,
  
  // Operations
  addTensors,
  matrixMultiply,
  
  // Stores
  wasmState,
  tensorStore,
  memoryUsage,
  isWasmReady,
  tensorCount,
  webGpuAvailable,
  performanceMetrics,
  
  // Utilities
  tensorVisualization,
  createTensorSubscription,
  createReactiveTensorOp,
  measureOperation,
  createErrorHandler,
  updateMemoryUsage
} from '../svelte_bindings.js';

// Components
export { default as TrustformersProvider } from './TrustformersProvider.svelte';
export { default as TensorVisualization } from './TensorVisualization.svelte';
export { default as TensorOperations } from './TensorOperations.svelte';

// Re-export Svelte stores for convenience
import { writable, derived, readable } from 'svelte/store';
export { writable, derived, readable };

// Plugin for SvelteKit
export function svelteKitPlugin() {
  return {
    name: 'trustformers-wasm',
    configureServer(server) {
      // Configure server for WASM files
      server.middlewares.use('/trustformers-wasm', (req, res, next) => {
        if (req.url.endsWith('.wasm')) {
          res.setHeader('Content-Type', 'application/wasm');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        }
        next();
      });
    }
  };
}

// Utilities for common Svelte patterns
export const svelteUtils = {
  // Create a reactive tensor that updates when dependencies change
  createReactiveTensor(dependencies, computation) {
    return derived(dependencies, (values) => {
      try {
        return computation(values);
      } catch (error) {
        console.error('Reactive tensor computation error:', error);
        return null;
      }
    });
  },

  // Create a tensor store with built-in cleanup
  createManagedTensorStore(initialValue = null) {
    const { subscribe, set, update } = writable(initialValue);
    let currentTensorId = null;

    return {
      subscribe,
      set: (value) => {
        if (currentTensorId) {
          deleteTensor(currentTensorId);
        }
        currentTensorId = value;
        set(value);
      },
      update: (updater) => {
        update((current) => {
          if (current) {
            deleteTensor(current);
          }
          const newValue = updater(current);
          currentTensorId = newValue;
          return newValue;
        });
      },
      destroy: () => {
        if (currentTensorId) {
          deleteTensor(currentTensorId);
          currentTensorId = null;
        }
        set(null);
      }
    };
  },

  // Batch operations with automatic cleanup
  async withTensorBatch(operations) {
    const createdTensors = [];
    try {
      const results = await operations((shape, data) => {
        const id = createTensor(shape, data);
        createdTensors.push(id);
        return id;
      });
      return results;
    } finally {
      // Clean up intermediate tensors
      createdTensors.forEach(id => {
        try {
          deleteTensor(id);
        } catch (error) {
          console.warn('Failed to cleanup tensor:', id, error);
        }
      });
    }
  },

  // Performance monitoring wrapper
  withPerformanceTracking(name, fn) {
    return measureOperation(name, fn);
  }
};

// TypeScript definitions (if using TypeScript)
export const types = {
  // Tensor shape type
  Shape: /** @type {number[]} */ null,
  
  // Tensor data type
  TensorData: /** @type {Float32Array | number[]} */ null,
  
  // WASM state interface
  WasmState: /** @type {{
    initialized: boolean,
    loading: boolean,
    error: string | null,
    device: 'cpu' | 'gpu',
    webGpuSupported: boolean
  }} */ null,
  
  // Tensor info interface
  TensorInfo: /** @type {{
    id: number,
    shape: number[],
    createdAt: Date,
    device: string,
    operation?: string
  }} */ null
};

// Default configuration
export const defaultConfig = {
  autoInitialize: true,
  enableWebGPU: false,
  maxTensors: 100,
  memoryThreshold: 100 * 1024 * 1024, // 100MB
  performanceLogging: false,
  errorReporting: true
};

// Configuration store
export const config = writable(defaultConfig);

// Global error handler
export const globalErrorHandler = createErrorHandler('TrustformersGlobal');

// Initialize with configuration
export async function initializeWithConfig(userConfig = {}) {
  const finalConfig = { ...defaultConfig, ...userConfig };
  config.set(finalConfig);

  try {
    if (finalConfig.autoInitialize) {
      await initializeTrustformers();
      
      if (finalConfig.enableWebGPU) {
        try {
          await enableWebGPU();
        } catch (error) {
          if (finalConfig.errorReporting) {
            console.warn('WebGPU initialization failed:', error);
          }
        }
      }
    }
    
    return true;
  } catch (error) {
    globalErrorHandler.handle(error, 'initialization');
    throw error;
  }
}

// Export version information
export const version = '1.0.0';
export const buildInfo = {
  version,
  buildDate: new Date().toISOString(),
  features: [
    'wasm',
    'webgpu',
    'svelte',
    'typescript',
    'reactive-stores'
  ]
};

// Development helpers
export const dev = {
  // Log all store updates in development
  enableStoreLogging() {
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      wasmState.subscribe(state => console.log('WASM State:', state));
      tensorStore.subscribe(tensors => console.log('Tensors:', tensors.size));
      memoryUsage.subscribe(usage => console.log('Memory:', usage));
    }
  },

  // Performance debugging
  enablePerformanceLogging() {
    performanceMetrics.subscribe(metrics => {
      console.log('Performance Metrics:', metrics);
    });
  },

  // Tensor debugging
  logTensorInfo(tensorId) {
    try {
      const data = getTensorData(tensorId);
      console.log(`Tensor ${tensorId}:`, {
        shape: data.shape,
        size: data.data.length,
        memory: data.data.byteLength,
        preview: Array.from(data.data.slice(0, 10))
      });
    } catch (error) {
      console.error(`Failed to log tensor ${tensorId}:`, error);
    }
  }
};

// Lifecycle hooks for SvelteKit
export const lifecycle = {
  // Call this in app.html or layout.svelte
  onMount: async (options = {}) => {
    await initializeWithConfig(options);
  },

  // Call this when the app is being destroyed
  onDestroy: () => {
    cleanupTensors();
  },

  // Call this when navigating away (SPA)
  onNavigate: () => {
    // Optionally clean up tensors on navigation
    cleanupTensors();
  }
};

// Export everything as default for convenience
export default {
  // Core functions
  initializeTrustformers,
  enableWebGPU,
  createTensor,
  deleteTensor,
  addTensors,
  matrixMultiply,
  cleanupTensors,
  
  // Stores
  wasmState,
  tensorStore,
  memoryUsage,
  isWasmReady,
  tensorCount,
  webGpuAvailable,
  performanceMetrics,
  
  // Components (for dynamic imports)
  TrustformersProvider: () => import('./TrustformersProvider.svelte'),
  TensorVisualization: () => import('./TensorVisualization.svelte'),
  TensorOperations: () => import('./TensorOperations.svelte'),
  
  // Utilities
  svelteUtils,
  config,
  globalErrorHandler,
  initializeWithConfig,
  svelteKitPlugin,
  version,
  buildInfo,
  dev,
  lifecycle
};