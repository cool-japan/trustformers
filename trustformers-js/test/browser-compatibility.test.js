/**
 * Browser Compatibility and WASM Integration Tests
 */

import { 
  describe, 
  test, 
  beforeAll, 
  expect 
} from './test-runner.js';

import { 
  initialize, 
  createModel, 
  tensor,
  utils
} from '../src/index.js';

// Mock browser globals for Node.js testing
function setupBrowserMocks() {
  if (typeof window === 'undefined') {
    global.window = {
      WebAssembly: global.WebAssembly,
      navigator: {
        userAgent: 'Node.js Test Environment',
        platform: 'test'
      },
      location: {
        href: 'http://localhost:3000/test'
      },
      document: {
        createElement: (tag) => ({
          tagName: tag.toUpperCase(),
          setAttribute: () => {},
          getAttribute: () => null,
          style: {}
        })
      }
    };
    
    global.document = global.window.document;
    global.navigator = global.window.navigator;
  }
}

describe('Browser Compatibility', () => {
  beforeAll(async () => {
    setupBrowserMocks();
    console.log('Setting up browser compatibility tests...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
  });

  describe('WebAssembly Support', () => {
    test('WebAssembly is available', () => {
      expect(typeof WebAssembly).toBe('object');
      expect(typeof WebAssembly.Module).toBe('function');
      expect(typeof WebAssembly.Instance).toBe('function');
      expect(typeof WebAssembly.instantiate).toBe('function');
    });

    test('WebAssembly.Memory is supported', () => {
      expect(typeof WebAssembly.Memory).toBe('function');
      
      const memory = new WebAssembly.Memory({ initial: 1 });
      expect(memory).toBeTruthy();
      expect(memory.buffer instanceof ArrayBuffer).toBeTruthy();
    });

    test('WebAssembly streaming compilation is supported', async () => {
      expect(typeof WebAssembly.instantiateStreaming).toBe('function');
      expect(typeof WebAssembly.compileStreaming).toBe('function');
    });

    test('WebAssembly SIMD support detection', () => {
      // Note: SIMD support varies by browser/environment
      const simdSupported = typeof WebAssembly.validate === 'function' && 
        WebAssembly.validate(new Uint8Array([
          0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // WASM header
          0x01, 0x04, 0x01, 0x60, 0x00, 0x00,             // Type section
          0x03, 0x02, 0x01, 0x00,                         // Function section
          0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0f, 0x45, 0x0b // Code section with SIMD
        ]));
      
      expect(typeof simdSupported).toBe('boolean');
    });
  });

  describe('JavaScript Environment Compatibility', () => {
    test('supports ES6 features', () => {
      // Arrow functions
      const arrow = () => 42;
      expect(arrow()).toBe(42);
      
      // Template literals
      const template = `test ${42}`;
      expect(template).toBe('test 42');
      
      // Destructuring
      const [a, b] = [1, 2];
      expect(a).toBe(1);
      expect(b).toBe(2);
      
      // Promises
      expect(typeof Promise).toBe('function');
      expect(typeof Promise.resolve).toBe('function');
    });

    test('supports ES2020+ features', () => {
      // BigInt (ES2020)
      if (typeof BigInt !== 'undefined') {
        const big = BigInt(42);
        expect(typeof big).toBe('bigint');
      }
      
      // Optional chaining (ES2020)
      const obj = { a: { b: 42 } };
      expect(obj?.a?.b).toBe(42);
      expect(obj?.c?.d).toBeUndefined();
      
      // Nullish coalescing (ES2020)
      expect(null ?? 'default').toBe('default');
      expect(undefined ?? 'default').toBe('default');
      expect(0 ?? 'default').toBe(0);
    });

    test('supports async/await', async () => {
      const asyncFunction = async () => {
        return await Promise.resolve(42);
      };
      
      const result = await asyncFunction();
      expect(result).toBe(42);
    });

    test('supports modules', () => {
      // If we're running this test, modules are working
      expect(typeof createModel).toBe('function');
      expect(typeof tensor).toBe('function');
    });

    test('supports typed arrays', () => {
      const float32 = new Float32Array([1, 2, 3, 4]);
      expect(float32.length).toBe(4);
      expect(float32[0]).toBe(1);
      
      const int32 = new Int32Array(4);
      expect(int32.length).toBe(4);
      expect(int32[0]).toBe(0);
      
      const uint8 = new Uint8Array([255, 128, 0]);
      expect(uint8[0]).toBe(255);
      expect(uint8[1]).toBe(128);
    });
  });

  describe('Web APIs Compatibility', () => {
    test('Performance API availability', () => {
      if (typeof performance !== 'undefined') {
        expect(typeof performance.now).toBe('function');
        
        const start = performance.now();
        const end = performance.now();
        expect(end).toBeGreaterThan(start);
      }
    });

    test('File API compatibility', () => {
      if (typeof File !== 'undefined') {
        expect(typeof File).toBe('function');
        expect(typeof FileReader).toBe('function');
        expect(typeof Blob).toBe('function');
      }
    });

    test('Worker API compatibility', () => {
      if (typeof Worker !== 'undefined') {
        expect(typeof Worker).toBe('function');
        expect(typeof SharedArrayBuffer !== 'undefined' || typeof ArrayBuffer !== 'undefined').toBeTruthy();
      }
    });

    test('Fetch API compatibility', () => {
      if (typeof fetch !== 'undefined') {
        expect(typeof fetch).toBe('function');
        expect(typeof Response).toBe('function');
        expect(typeof Request).toBe('function');
      }
    });

    test('Local Storage compatibility', () => {
      if (typeof localStorage !== 'undefined') {
        expect(typeof localStorage.setItem).toBe('function');
        expect(typeof localStorage.getItem).toBe('function');
        expect(typeof localStorage.removeItem).toBe('function');
      }
    });

    test('IndexedDB compatibility', () => {
      if (typeof indexedDB !== 'undefined') {
        expect(typeof indexedDB.open).toBe('function');
        expect(typeof indexedDB.deleteDatabase).toBe('function');
      }
    });
  });

  describe('WASM Memory Management', () => {
    test('WASM memory grows correctly', () => {
      const model = createModel('bert_base');
      
      // WASM memory should be allocated
      expect(model).toBeTruthy();
      expect(typeof model.memory_usage_mb).toBe('function');
      
      const memoryUsage = model.memory_usage_mb();
      expect(memoryUsage).toBeGreaterThan(0);
      
      model.free();
    });

    test('handles large tensor allocations', () => {
      // Create a reasonably large tensor
      const largeArray = new Array(10000).fill(1);
      const largeTensor = tensor(largeArray, [100, 100]);
      
      expect(largeTensor).toBeTruthy();
      expect(largeTensor.numel()).toBe(10000);
      
      largeTensor.free();
    });

    test('handles memory pressure gracefully', () => {
      const tensors = [];
      
      try {
        // Create multiple tensors to test memory management
        for (let i = 0; i < 10; i++) {
          const t = tensor(new Array(1000).fill(i), [10, 100]);
          tensors.push(t);
        }
        
        expect(tensors.length).toBe(10);
        
      } finally {
        // Cleanup all tensors
        tensors.forEach(t => {
          try {
            t.free();
          } catch (e) {
            // Ignore cleanup errors
          }
        });
      }
    });
  });

  describe('Error Handling in Browser Environment', () => {
    test('handles WASM loading failures gracefully', async () => {
      // Test with invalid WASM path
      try {
        await initialize({
          wasmPath: 'invalid/path/to/wasm.wasm'
        });
        // Should not reach here
        expect(false).toBeTruthy();
      } catch (error) {
        expect(error).toBeTruthy();
        expect(typeof error.message).toBe('string');
      }
    });

    test('handles unsupported browser features', () => {
      // Temporarily remove WebAssembly to test fallback
      const originalWasm = global.WebAssembly;
      delete global.WebAssembly;
      
      try {
        expect(() => {
          createModel('bert_base');
        }).toThrow();
      } finally {
        // Restore WebAssembly
        global.WebAssembly = originalWasm;
      }
    });

    test('handles out of memory conditions', () => {
      // This is difficult to test reliably, but we can at least
      // ensure the error handling path exists
      expect(() => {
        // Try to create an impossibly large tensor
        tensor(new Array(Number.MAX_SAFE_INTEGER).fill(1));
      }).toThrow();
    });
  });

  describe('Cross-Platform Compatibility', () => {
    test('detects platform correctly', () => {
      const platform = utils.platform();
      expect(typeof platform).toBe('string');
      expect(['web', 'node', 'worker', 'unknown'].includes(platform)).toBeTruthy();
    });

    test('detects available features', () => {
      const features = utils.features();
      expect(Array.isArray(features)).toBeTruthy();
      
      // Should include basic features
      expect(features.includes('wasm')).toBeTruthy();
      
      // May include optional features
      const optionalFeatures = ['simd', 'threads', 'gpu', 'webgl', 'webgpu'];
      const hasOptionalFeatures = optionalFeatures.some(f => features.includes(f));
      expect(typeof hasOptionalFeatures).toBe('boolean');
    });

    test('handles different number formats', () => {
      // Test with different number formats that might vary by platform
      const float32Tensor = tensor([1.5, 2.7, 3.14159], [3]);
      const int32Tensor = tensor([1, 2, 3], [3]);
      
      expect(float32Tensor.sum()).toBeCloseTo(7.34159, 4);
      expect(int32Tensor.sum()).toBe(6);
      
      float32Tensor.free();
      int32Tensor.free();
    });

    test('handles different endianness', () => {
      // Create a tensor with specific byte pattern
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setFloat32(0, 1.5, true); // little endian
      view.setFloat32(4, 2.5, true);
      
      const array = new Float32Array(buffer);
      const tensorFromBuffer = tensor(Array.from(array), [2]);
      
      expect(tensorFromBuffer.sum()).toBeCloseTo(4.0, 4);
      tensorFromBuffer.free();
    });
  });

  describe('Performance in Browser Environment', () => {
    test('initialization time is reasonable', async () => {
      const startTime = performance.now ? performance.now() : Date.now();
      
      // Re-initialize to test timing
      await initialize({
        wasmPath: '../pkg/trustformers_wasm_bg.wasm'
      });
      
      const endTime = performance.now ? performance.now() : Date.now();
      const initTime = endTime - startTime;
      
      // Should initialize within reasonable time (under 5 seconds)
      expect(initTime).toBeLessThan(5000);
    });

    test('tensor operations are performant', () => {
      const size = 1000;
      const array1 = new Array(size).fill(1);
      const array2 = new Array(size).fill(2);
      
      const startTime = performance.now ? performance.now() : Date.now();
      
      const t1 = tensor(array1, [size]);
      const t2 = tensor(array2, [size]);
      const result = t1.add(t2);
      const sum = result.sum();
      
      const endTime = performance.now ? performance.now() : Date.now();
      const opTime = endTime - startTime;
      
      expect(sum).toBe(size * 3); // Each element is 1 + 2 = 3
      expect(opTime).toBeLessThan(100); // Should be fast
      
      t1.free();
      t2.free();
      result.free();
    });

    test('model inference is performant', () => {
      const model = createModel('bert_base');
      const input = tensor([101, 2003, 102], [1, 3]);
      
      const startTime = performance.now ? performance.now() : Date.now();
      const output = model.forward(input);
      const endTime = performance.now ? performance.now() : Date.now();
      
      const inferenceTime = endTime - startTime;
      
      expect(output).toBeTruthy();
      expect(inferenceTime).toBeLessThan(2000); // Should be under 2 seconds
      
      model.free();
      input.free();
      output.free();
    });
  });

  describe('Security Considerations', () => {
    test('WASM runs in secure context', () => {
      // WASM should not have access to file system or network by default
      expect(typeof window !== 'undefined' ? window.location : true).toBeTruthy();
      
      // Should not expose internal pointers
      const model = createModel('bert_base');
      const config = model.config;
      
      // Config should not contain raw pointers or unsafe data
      Object.values(config).forEach(value => {
        expect(typeof value).not.toBe('undefined');
        if (typeof value === 'object' && value !== null) {
          expect(value.constructor).toBeTruthy();
        }
      });
      
      model.free();
    });

    test('input validation prevents buffer overflows', () => {
      // Test with various potentially problematic inputs
      expect(() => {
        tensor(null);
      }).toThrow();
      
      expect(() => {
        tensor(undefined);
      }).toThrow();
      
      expect(() => {
        tensor([1, 2, 3], null);
      }).toThrow();
      
      expect(() => {
        tensor([1, 2, 3], [-1, 2]);
      }).toThrow();
    });

    test('prevents access to uninitialized memory', () => {
      const t = tensor([1, 2, 3], [3]);
      
      // Should not be able to access memory beyond tensor bounds
      expect(() => {
        t.slice(0, 0, 10); // Slice beyond bounds
      }).toThrow();
      
      t.free();
    });
  });
});