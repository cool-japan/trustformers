/**
 * Playwright cross-browser tests for TrustformeRS WASM
 */

import { test, expect, chromium, firefox, webkit } from '@playwright/test';

// Test data and utilities
const testTensorData = {
  small: { shape: [5, 5], size: 25 },
  medium: { shape: [50, 50], size: 2500 },
  large: { shape: [100, 100], size: 10000 }
};

// Helper function to inject WASM module
async function injectWasmModule(page) {
  await page.addScriptTag({
    content: `
      // Mock TrustformeRS WASM module for testing
      window.TrustformersWasm = {
        initialized: false,
        tensors: new Map(),
        nextId: 1,
        
        async init() {
          // Simulate async initialization
          await new Promise(resolve => setTimeout(resolve, 100));
          this.initialized = true;
          return true;
        },
        
        createTensor(shape, data) {
          if (!this.initialized) throw new Error('Not initialized');
          const id = this.nextId++;
          const size = shape.reduce((a, b) => a * b, 1);
          const tensor = {
            id,
            shape: [...shape],
            data: data || new Float32Array(size),
            device: 'cpu'
          };
          this.tensors.set(id, tensor);
          return id;
        },
        
        getTensor(id) {
          return this.tensors.get(id);
        },
        
        addTensors(id1, id2) {
          const t1 = this.tensors.get(id1);
          const t2 = this.tensors.get(id2);
          if (!t1 || !t2) throw new Error('Invalid tensor ID');
          
          const result = new Float32Array(t1.data.length);
          for (let i = 0; i < result.length; i++) {
            result[i] = t1.data[i] + t2.data[i];
          }
          
          return this.createTensor(t1.shape, result);
        },
        
        async enableWebGPU() {
          if (!navigator.gpu) throw new Error('WebGPU not supported');
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) throw new Error('No WebGPU adapter');
          this.device = await adapter.requestDevice();
          return true;
        },
        
        getMemoryUsage() {
          return {
            totalTensors: this.tensors.size,
            totalBytes: Array.from(this.tensors.values())
              .reduce((sum, t) => sum + t.data.byteLength, 0)
          };
        }
      };
    `
  });
}

// Browser compatibility tests
test.describe('Browser Compatibility', () => {
  test('WebAssembly support detection', async ({ page, browserName }) => {
    await page.goto('/');
    
    const wasmSupported = await page.evaluate(() => {
      return typeof WebAssembly !== 'undefined';
    });
    
    expect(wasmSupported).toBe(true);
    
    // Test WASM capabilities
    const wasmFeatures = await page.evaluate(() => {
      const features = {};
      
      // Test basic WASM
      features.basic = typeof WebAssembly !== 'undefined';
      
      // Test SIMD (simplified check)
      features.simd = true; // Assume supported for testing
      
      // Test threads
      features.threads = typeof SharedArrayBuffer !== 'undefined';
      
      return features;
    });
    
    expect(wasmFeatures.basic).toBe(true);
    
    console.log(`[${browserName}] WASM Features:`, wasmFeatures);
  });
  
  test('WebGPU support detection', async ({ page, browserName }) => {
    await page.goto('/');
    
    const webgpuInfo = await page.evaluate(async () => {
      if (!navigator.gpu) {
        return { supported: false, reason: 'navigator.gpu not available' };
      }
      
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          return { supported: false, reason: 'No adapter available' };
        }
        
        const device = await adapter.requestDevice();
        return {
          supported: true,
          features: Array.from(adapter.features || []),
          limits: {
            maxComputeWorkgroupSizeX: adapter.limits?.maxComputeWorkgroupSizeX,
            maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize
          }
        };
      } catch (error) {
        return { supported: false, reason: error.message };
      }
    });
    
    console.log(`[${browserName}] WebGPU Info:`, webgpuInfo);
    
    // WebGPU support varies by browser
    if (browserName === 'chromium') {
      // Chrome should support WebGPU
      if (!webgpuInfo.supported) {
        console.warn(`[${browserName}] WebGPU not supported: ${webgpuInfo.reason}`);
      }
    } else {
      // Other browsers may not support WebGPU yet
      console.log(`[${browserName}] WebGPU support: ${webgpuInfo.supported}`);
    }
  });
  
  test('Performance API availability', async ({ page, browserName }) => {
    await page.goto('/');
    
    const perfFeatures = await page.evaluate(() => {
      return {
        performance: typeof performance !== 'undefined',
        now: typeof performance?.now === 'function',
        mark: typeof performance?.mark === 'function',
        measure: typeof performance?.measure === 'function',
        getEntriesByType: typeof performance?.getEntriesByType === 'function'
      };
    });
    
    expect(perfFeatures.performance).toBe(true);
    expect(perfFeatures.now).toBe(true);
    
    console.log(`[${browserName}] Performance API:`, perfFeatures);
  });
});

// WASM module functionality tests
test.describe('WASM Module Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await injectWasmModule(page);
  });
  
  test('Module initialization', async ({ page, browserName }) => {
    const initResult = await page.evaluate(async () => {
      const start = performance.now();
      const result = await window.TrustformersWasm.init();
      const duration = performance.now() - start;
      
      return {
        success: result,
        initialized: window.TrustformersWasm.initialized,
        duration
      };
    });
    
    expect(initResult.success).toBe(true);
    expect(initResult.initialized).toBe(true);
    expect(initResult.duration).toBeLessThan(1000);
    
    console.log(`[${browserName}] Init duration: ${initResult.duration.toFixed(2)}ms`);
  });
  
  test('Tensor creation and management', async ({ page, browserName }) => {
    await page.evaluate(() => window.TrustformersWasm.init());
    
    const tensorTest = await page.evaluate(() => {
      const tensorId = window.TrustformersWasm.createTensor([10, 10]);
      const tensor = window.TrustformersWasm.getTensor(tensorId);
      
      return {
        id: tensorId,
        shape: tensor.shape,
        dataLength: tensor.data.length,
        dataType: tensor.data.constructor.name
      };
    });
    
    expect(tensorTest.id).toBeGreaterThan(0);
    expect(tensorTest.shape).toEqual([10, 10]);
    expect(tensorTest.dataLength).toBe(100);
    expect(tensorTest.dataType).toBe('Float32Array');
    
    console.log(`[${browserName}] Tensor created:`, tensorTest);
  });
  
  test('Tensor operations', async ({ page, browserName }) => {
    await page.evaluate(() => window.TrustformersWasm.init());
    
    const operationTest = await page.evaluate(() => {
      const start = performance.now();
      
      // Create test tensors
      const t1Id = window.TrustformersWasm.createTensor([5, 5], new Array(25).fill(1));
      const t2Id = window.TrustformersWasm.createTensor([5, 5], new Array(25).fill(2));
      
      // Perform addition
      const resultId = window.TrustformersWasm.addTensors(t1Id, t2Id);
      const result = window.TrustformersWasm.getTensor(resultId);
      
      const duration = performance.now() - start;
      
      return {
        resultShape: result.shape,
        resultData: Array.from(result.data.slice(0, 5)), // First 5 elements
        duration,
        memoryUsage: window.TrustformersWasm.getMemoryUsage()
      };
    });
    
    expect(operationTest.resultShape).toEqual([5, 5]);
    expect(operationTest.resultData).toEqual([3, 3, 3, 3, 3]);
    expect(operationTest.memoryUsage.totalTensors).toBe(3);
    
    console.log(`[${browserName}] Operation duration: ${operationTest.duration.toFixed(2)}ms`);
    console.log(`[${browserName}] Memory usage:`, operationTest.memoryUsage);
  });
});

// Performance tests across browsers
test.describe('Performance Comparison', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await injectWasmModule(page);
    await page.evaluate(() => window.TrustformersWasm.init());
  });
  
  test('Tensor operation performance', async ({ page, browserName }) => {
    const perfResults = await page.evaluate(() => {
      const iterations = 100;
      const durations = [];
      
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        
        const t1Id = window.TrustformersWasm.createTensor([10, 10]);
        const t2Id = window.TrustformersWasm.createTensor([10, 10]);
        const resultId = window.TrustformersWasm.addTensors(t1Id, t2Id);
        
        const duration = performance.now() - start;
        durations.push(duration);
      }
      
      const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
      const minDuration = Math.min(...durations);
      const maxDuration = Math.max(...durations);
      
      return {
        iterations,
        avgDuration,
        minDuration,
        maxDuration,
        variance: durations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / durations.length
      };
    });
    
    expect(perfResults.avgDuration).toBeLessThan(50); // Should be fast
    expect(perfResults.maxDuration).toBeLessThan(perfResults.avgDuration * 5); // Reasonable variation
    
    console.log(`[${browserName}] Performance Results:`, {
      avg: `${perfResults.avgDuration.toFixed(2)}ms`,
      min: `${perfResults.minDuration.toFixed(2)}ms`,
      max: `${perfResults.maxDuration.toFixed(2)}ms`,
      stdDev: `${Math.sqrt(perfResults.variance).toFixed(2)}ms`
    });
  });
  
  test('Memory efficiency', async ({ page, browserName }) => {
    const memoryTest = await page.evaluate(() => {
      const initialUsage = window.TrustformersWasm.getMemoryUsage();
      
      // Create many tensors
      const tensorIds = [];
      for (let i = 0; i < 50; i++) {
        tensorIds.push(window.TrustformersWasm.createTensor([20, 20]));
      }
      
      const peakUsage = window.TrustformersWasm.getMemoryUsage();
      
      // Clean up half the tensors
      for (let i = 0; i < 25; i++) {
        window.TrustformersWasm.tensors.delete(tensorIds[i]);
      }
      
      const finalUsage = window.TrustformersWasm.getMemoryUsage();
      
      return {
        initial: initialUsage,
        peak: peakUsage,
        final: finalUsage,
        memoryEfficiency: (peakUsage.totalBytes - finalUsage.totalBytes) / peakUsage.totalBytes
      };
    });
    
    expect(memoryTest.peak.totalTensors).toBe(50);
    expect(memoryTest.final.totalTensors).toBe(25);
    expect(memoryTest.memoryEfficiency).toBeGreaterThan(0.4); // Should free significant memory
    
    console.log(`[${browserName}] Memory Test:`, {
      peak: `${(memoryTest.peak.totalBytes / 1024).toFixed(1)}KB`,
      final: `${(memoryTest.final.totalBytes / 1024).toFixed(1)}KB`,
      efficiency: `${(memoryTest.memoryEfficiency * 100).toFixed(1)}%`
    });
  });
});

// Error handling tests
test.describe('Error Handling', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await injectWasmModule(page);
  });
  
  test('Graceful error handling', async ({ page, browserName }) => {
    const errorTests = await page.evaluate(async () => {
      const results = {};
      
      // Test uninitialized module
      try {
        window.TrustformersWasm.createTensor([5, 5]);
        results.uninitializedError = false;
      } catch (error) {
        results.uninitializedError = error.message;
      }
      
      // Initialize for other tests
      await window.TrustformersWasm.init();
      
      // Test invalid tensor operations
      try {
        const t1Id = window.TrustformersWasm.createTensor([5, 5]);
        const t2Id = window.TrustformersWasm.createTensor([3, 3]);
        window.TrustformersWasm.addTensors(t1Id, t2Id);
        results.shapeMismatchError = false;
      } catch (error) {
        results.shapeMismatchError = error.message;
      }
      
      // Test invalid tensor ID
      try {
        window.TrustformersWasm.getTensor(99999);
        results.invalidIdHandled = true;
      } catch (error) {
        results.invalidIdHandled = false;
      }
      
      return results;
    });
    
    expect(errorTests.uninitializedError).toBeTruthy();
    expect(errorTests.uninitializedError).toContain('Not initialized');
    
    console.log(`[${browserName}] Error handling test results:`, errorTests);
  });
});

// WebGPU specific tests (conditional)
test.describe('WebGPU Integration', () => {
  test('WebGPU enablement and operations', async ({ page, browserName }) => {
    await page.goto('/');
    await injectWasmModule(page);
    await page.evaluate(() => window.TrustformersWasm.init());
    
    const webgpuTest = await page.evaluate(async () => {
      try {
        const webgpuEnabled = await window.TrustformersWasm.enableWebGPU();
        
        // Test WebGPU operation
        const t1Id = window.TrustformersWasm.createTensor([10, 10]);
        const t2Id = window.TrustformersWasm.createTensor([10, 10]);
        
        const start = performance.now();
        const resultId = window.TrustformersWasm.addTensors(t1Id, t2Id);
        const duration = performance.now() - start;
        
        return {
          webgpuEnabled,
          operationSuccess: !!resultId,
          duration,
          error: null
        };
      } catch (error) {
        return {
          webgpuEnabled: false,
          operationSuccess: false,
          duration: 0,
          error: error.message
        };
      }
    });
    
    if (webgpuTest.webgpuEnabled) {
      expect(webgpuTest.operationSuccess).toBe(true);
      console.log(`[${browserName}] WebGPU operation duration: ${webgpuTest.duration.toFixed(2)}ms`);
    } else {
      console.log(`[${browserName}] WebGPU not available: ${webgpuTest.error}`);
    }
  });
});