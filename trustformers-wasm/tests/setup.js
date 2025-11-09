/**
 * Jest setup file for cross-browser testing
 */

import { jest } from '@jest/globals';

// Mock WebGPU for environments that don't support it
global.mockWebGPU = () => {
    if (typeof navigator === 'undefined') {
        global.navigator = {};
    }
    
    if (!navigator.gpu) {
        navigator.gpu = {
            requestAdapter: jest.fn().mockResolvedValue({
                requestDevice: jest.fn().mockResolvedValue({
                    createCommandEncoder: jest.fn().mockReturnValue({
                        beginRenderPass: jest.fn().mockReturnValue({
                            end: jest.fn()
                        }),
                        finish: jest.fn().mockReturnValue({})
                    }),
                    queue: {
                        submit: jest.fn()
                    }
                }),
                features: new Set(['shader-f16']),
                limits: {
                    maxTextureDimension1D: 8192,
                    maxTextureDimension2D: 8192,
                    maxTextureDimension3D: 2048,
                    maxTextureArrayLayers: 256,
                    maxBindGroups: 4,
                    maxDynamicUniformBuffersPerPipelineLayout: 8,
                    maxComputeWorkgroupSizeX: 256,
                    maxComputeWorkgroupSizeY: 256,
                    maxComputeWorkgroupSizeZ: 64
                }
            })
        };
    }
};

// Mock WebAssembly for testing
global.mockWebAssembly = () => {
    if (typeof WebAssembly === 'undefined') {
        global.WebAssembly = {
            Module: jest.fn().mockImplementation(() => ({})),
            Instance: jest.fn().mockImplementation(() => ({})),
            Memory: jest.fn().mockImplementation(() => ({})),
            Global: jest.fn().mockImplementation(() => ({})),
            instantiate: jest.fn().mockResolvedValue({
                instance: {},
                module: {}
            }),
            instantiateStreaming: jest.fn().mockResolvedValue({
                instance: {},
                module: {}
            }),
            compile: jest.fn().mockResolvedValue({}),
            compileStreaming: jest.fn().mockResolvedValue({})
        };
    }
};

// Mock SharedArrayBuffer for testing
global.mockSharedArrayBuffer = () => {
    if (typeof SharedArrayBuffer === 'undefined') {
        global.SharedArrayBuffer = class MockSharedArrayBuffer extends ArrayBuffer {
            constructor(length) {
                super(length);
                this.byteLength = length;
            }
        };
    }
};

// Mock performance API
global.mockPerformance = () => {
    if (typeof performance === 'undefined') {
        global.performance = {
            now: jest.fn(() => Date.now()),
            mark: jest.fn(),
            measure: jest.fn(),
            getEntriesByName: jest.fn(() => []),
            getEntriesByType: jest.fn(() => []),
            clearMarks: jest.fn(),
            clearMeasures: jest.fn()
        };
    }
};

// Mock navigator properties for testing
global.mockNavigator = () => {
    if (typeof navigator === 'undefined') {
        global.navigator = {};
    }
    
    const defaultNavigator = {
        userAgent: 'Mozilla/5.0 (Test Environment) TestBrowser/1.0',
        vendor: 'Test Inc.',
        platform: 'TestOS',
        language: 'en-US',
        hardwareConcurrency: 4,
        deviceMemory: 8,
        onLine: true,
        cookieEnabled: true
    };
    
    Object.assign(navigator, defaultNavigator);
};

// Mock localStorage for testing
global.mockLocalStorage = () => {
    if (typeof localStorage === 'undefined') {
        const store = {};
        global.localStorage = {
            getItem: jest.fn((key) => store[key] || null),
            setItem: jest.fn((key, value) => {
                store[key] = value;
            }),
            removeItem: jest.fn((key) => {
                delete store[key];
            }),
            clear: jest.fn(() => {
                Object.keys(store).forEach(key => delete store[key]);
            }),
            get length() {
                return Object.keys(store).length;
            },
            key: jest.fn((index) => Object.keys(store)[index] || null)
        };
    }
};

// Browser-specific setup
const setupBrowserEnvironment = () => {
    // Detect test environment
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
    const isJSDOM = global.window && global.window.navigator && 
                   global.window.navigator.userAgent.includes('jsdom');
    
    if (isNode || isJSDOM) {
        // Setup mocks for Node.js/JSDOM environment
        mockNavigator();
        mockWebGPU();
        mockWebAssembly();
        mockSharedArrayBuffer();
        mockPerformance();
        mockLocalStorage();
        
        // Mock Worker for Node.js environment
        if (typeof Worker === 'undefined') {
            global.Worker = class MockWorker {
                constructor(scriptURL) {
                    this.scriptURL = scriptURL;
                    this.onmessage = null;
                    this.onerror = null;
                }
                
                postMessage(message) {
                    // Simulate async message handling
                    setTimeout(() => {
                        if (this.onmessage) {
                            this.onmessage({ data: `echo: ${JSON.stringify(message)}` });
                        }
                    }, 10);
                }
                
                terminate() {
                    // Mock termination
                }
            };
        }
        
        // Mock OffscreenCanvas
        if (typeof OffscreenCanvas === 'undefined') {
            global.OffscreenCanvas = class MockOffscreenCanvas {
                constructor(width, height) {
                    this.width = width;
                    this.height = height;
                }
                
                getContext(contextType) {
                    return {
                        canvas: this,
                        drawImage: jest.fn(),
                        fillRect: jest.fn(),
                        clearRect: jest.fn(),
                        putImageData: jest.fn(),
                        getImageData: jest.fn(() => ({
                            data: new Uint8ClampedArray(this.width * this.height * 4),
                            width: this.width,
                            height: this.height
                        }))
                    };
                }
                
                transferToImageBitmap() {
                    return {
                        width: this.width,
                        height: this.height,
                        close: jest.fn()
                    };
                }
            };
        }
    }
};

// Error handling setup
const setupErrorHandling = () => {
    // Global error handler for unhandled promise rejections
    if (typeof process !== 'undefined') {
        process.on('unhandledRejection', (reason, promise) => {
            console.error('Unhandled Promise Rejection:', reason);
        });
    }
    
    // Global error handler for uncaught exceptions
    if (typeof window !== 'undefined') {
        window.addEventListener('error', (event) => {
            console.error('Uncaught Error:', event.error);
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled Promise Rejection:', event.reason);
        });
    }
};

// Test utilities
global.testUtils = {
    // Simulate browser-specific behavior
    simulateBrowser: (browserName) => {
        const userAgents = {
            chrome: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            firefox: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            safari: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            edge: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
        };
        
        if (navigator && userAgents[browserName]) {
            Object.defineProperty(navigator, 'userAgent', {
                value: userAgents[browserName],
                writable: true
            });
        }
    },
    
    // Wait for a condition to be true
    waitFor: async (condition, timeout = 5000, interval = 100) => {
        const start = Date.now();
        while (Date.now() - start < timeout) {
            if (await condition()) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, interval));
        }
        throw new Error(`Condition not met within ${timeout}ms`);
    },
    
    // Create a test tensor data array
    createTestTensorData: (shape, fillValue = 0) => {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Array(size).fill(fillValue);
    },
    
    // Generate random tensor data
    generateRandomTensorData: (shape, min = -1, max = 1) => {
        const size = shape.reduce((a, b) => a * b, 1);
        return Array.from({ length: size }, () => Math.random() * (max - min) + min);
    },
    
    // Compare arrays with tolerance
    arrayEquals: (a, b, tolerance = 1e-6) => {
        if (a.length !== b.length) return false;
        return a.every((val, i) => Math.abs(val - b[i]) <= tolerance);
    }
};

// Browser feature detection mocks
global.browserFeatures = {
    webassembly: true,
    webgpu: process.env.WEBGPU_SUPPORT === 'true',
    sharedarraybuffer: true,
    webworkers: true,
    offscreencanvas: true,
    simd: true,
    bigint64array: true
};

// Setup everything
setupBrowserEnvironment();
setupErrorHandling();

// Console output for test info
console.log('🧪 TrustformeRS WASM Test Environment Setup Complete');
console.log('📊 Available Features:', global.browserFeatures);

// Increase timeout for WASM operations
jest.setTimeout(30000);