/**
 * Svelte bindings for TrustformeRS WASM
 * 
 * This module provides Svelte-specific bindings, stores, and components
 * for seamless integration with Svelte applications.
 */

import { writable, derived, readable, get } from 'svelte/store';
import { createEventDispatcher, onMount, onDestroy } from 'svelte';

// Core WASM interface (mock for now)
class TrustformersWasmCore {
    constructor() {
        this.initialized = false;
        this.tensors = new Map();
        this.nextTensorId = 1;
        this.device = 'cpu';
        this.webGpuAdapter = null;
        this.webGpuDevice = null;
    }

    async init() {
        // Simulate WASM initialization
        await new Promise(resolve => setTimeout(resolve, 100));
        this.initialized = true;
        return true;
    }

    async enableWebGPU() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        
        this.webGpuAdapter = await navigator.gpu.requestAdapter();
        if (!this.webGpuAdapter) {
            throw new Error('No WebGPU adapter available');
        }
        
        this.webGpuDevice = await this.webGpuAdapter.requestDevice();
        this.device = 'gpu';
        return true;
    }

    createTensor(shape, data = null) {
        if (!this.initialized) {
            throw new Error('WASM not initialized');
        }

        const id = this.nextTensorId++;
        const size = shape.reduce((a, b) => a * b, 1);
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(size),
            device: this.device,
            dtype: 'f32'
        };
        
        this.tensors.set(id, tensor);
        return id;
    }

    getTensor(id) {
        return this.tensors.get(id);
    }

    deleteTensor(id) {
        return this.tensors.delete(id);
    }

    addTensors(id1, id2) {
        const t1 = this.tensors.get(id1);
        const t2 = this.tensors.get(id2);
        
        if (!t1 || !t2) throw new Error('Invalid tensor ID');
        if (t1.shape.length !== t2.shape.length) throw new Error('Shape mismatch');
        
        const result = new Float32Array(t1.data.length);
        for (let i = 0; i < result.length; i++) {
            result[i] = t1.data[i] + t2.data[i];
        }
        
        return this.createTensor(t1.shape, result);
    }

    matmul(id1, id2) {
        const t1 = this.tensors.get(id1);
        const t2 = this.tensors.get(id2);
        
        if (!t1 || !t2) throw new Error('Invalid tensor ID');
        if (t1.shape.length !== 2 || t2.shape.length !== 2) {
            throw new Error('Matrix multiplication requires 2D tensors');
        }
        
        const [m, k] = t1.shape;
        const [k2, n] = t2.shape;
        if (k !== k2) throw new Error('Invalid dimensions for matrix multiplication');
        
        const result = new Float32Array(m * n);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) {
                    sum += t1.data[i * k + p] * t2.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        
        return this.createTensor([m, n], result);
    }

    softmax(id, axis = -1) {
        const tensor = this.tensors.get(id);
        if (!tensor) throw new Error('Invalid tensor ID');
        
        const result = new Float32Array(tensor.data.length);
        
        // Simple softmax implementation for 1D case
        if (tensor.shape.length === 1) {
            const max = Math.max(...tensor.data);
            let sum = 0;
            
            for (let i = 0; i < tensor.data.length; i++) {
                result[i] = Math.exp(tensor.data[i] - max);
                sum += result[i];
            }
            
            for (let i = 0; i < result.length; i++) {
                result[i] /= sum;
            }
        } else {
            // For multi-dimensional tensors, copy data for now
            result.set(tensor.data);
        }
        
        return this.createTensor(tensor.shape, result);
    }

    relu(id) {
        const tensor = this.tensors.get(id);
        if (!tensor) throw new Error('Invalid tensor ID');
        
        const result = new Float32Array(tensor.data.length);
        for (let i = 0; i < tensor.data.length; i++) {
            result[i] = Math.max(0, tensor.data[i]);
        }
        
        return this.createTensor(tensor.shape, result);
    }

    getMemoryUsage() {
        const totalTensors = this.tensors.size;
        const totalBytes = Array.from(this.tensors.values())
            .reduce((sum, t) => sum + t.data.byteLength, 0);
        
        return {
            totalTensors,
            totalBytes,
            estimatedMemoryMB: totalBytes / (1024 * 1024),
            device: this.device
        };
    }
}

// Global WASM instance
let wasmInstance = null;

// Initialize WASM instance
export async function initWasm() {
    if (!wasmInstance) {
        wasmInstance = new TrustformersWasmCore();
        await wasmInstance.init();
    }
    return wasmInstance;
}

// Get WASM instance
export function getWasmInstance() {
    if (!wasmInstance) {
        throw new Error('WASM not initialized. Call initWasm() first.');
    }
    return wasmInstance;
}

// Svelte stores for WASM state management
export const wasmState = writable({
    initialized: false,
    loading: false,
    error: null,
    device: 'cpu',
    webGpuSupported: false
});

export const tensorStore = writable(new Map());

export const memoryUsage = writable({
    totalTensors: 0,
    totalBytes: 0,
    estimatedMemoryMB: 0,
    device: 'cpu'
});

// Derived stores
export const isWasmReady = derived(
    wasmState,
    $wasmState => $wasmState.initialized && !$wasmState.loading && !$wasmState.error
);

export const tensorCount = derived(
    tensorStore,
    $tensorStore => $tensorStore.size
);

export const webGpuAvailable = readable(
    typeof navigator !== 'undefined' && !!navigator.gpu,
    (set) => {
        // Check for WebGPU support changes
        if (typeof navigator !== 'undefined') {
            const checkWebGpu = () => set(!!navigator.gpu);
            checkWebGpu();
            
            // Listen for potential WebGPU availability changes
            const interval = setInterval(checkWebGpu, 5000);
            return () => clearInterval(interval);
        }
    }
);

// WASM management functions
export async function initializeTrustformers() {
    wasmState.update(state => ({ ...state, loading: true, error: null }));
    
    try {
        const instance = await initWasm();
        
        wasmState.update(state => ({
            ...state,
            initialized: true,
            loading: false,
            device: instance.device
        }));
        
        updateMemoryUsage();
        return instance;
    } catch (error) {
        wasmState.update(state => ({
            ...state,
            loading: false,
            error: error.message
        }));
        throw error;
    }
}

export async function enableWebGPU() {
    const instance = getWasmInstance();
    
    try {
        await instance.enableWebGPU();
        wasmState.update(state => ({
            ...state,
            device: 'gpu',
            webGpuSupported: true
        }));
        updateMemoryUsage();
        return true;
    } catch (error) {
        wasmState.update(state => ({
            ...state,
            error: error.message
        }));
        throw error;
    }
}

export function updateMemoryUsage() {
    if (wasmInstance) {
        const usage = wasmInstance.getMemoryUsage();
        memoryUsage.set(usage);
    }
}

// Tensor management functions
export function createTensor(shape, data = null) {
    const instance = getWasmInstance();
    const tensorId = instance.createTensor(shape, data);
    
    tensorStore.update(store => {
        const newStore = new Map(store);
        newStore.set(tensorId, {
            id: tensorId,
            shape,
            createdAt: new Date(),
            device: instance.device
        });
        return newStore;
    });
    
    updateMemoryUsage();
    return tensorId;
}

export function deleteTensor(tensorId) {
    const instance = getWasmInstance();
    const deleted = instance.deleteTensor(tensorId);
    
    if (deleted) {
        tensorStore.update(store => {
            const newStore = new Map(store);
            newStore.delete(tensorId);
            return newStore;
        });
        updateMemoryUsage();
    }
    
    return deleted;
}

export function getTensorData(tensorId) {
    const instance = getWasmInstance();
    return instance.getTensor(tensorId);
}

// High-level operations
export function addTensors(tensorId1, tensorId2) {
    const instance = getWasmInstance();
    const resultId = instance.addTensors(tensorId1, tensorId2);
    
    const tensor1 = instance.getTensor(tensorId1);
    tensorStore.update(store => {
        const newStore = new Map(store);
        newStore.set(resultId, {
            id: resultId,
            shape: tensor1.shape,
            createdAt: new Date(),
            device: instance.device,
            operation: 'add'
        });
        return newStore;
    });
    
    updateMemoryUsage();
    return resultId;
}

export function matrixMultiply(tensorId1, tensorId2) {
    const instance = getWasmInstance();
    const resultId = instance.matmul(tensorId1, tensorId2);
    
    const tensor1 = instance.getTensor(tensorId1);
    const tensor2 = instance.getTensor(tensorId2);
    const resultShape = [tensor1.shape[0], tensor2.shape[1]];
    
    tensorStore.update(store => {
        const newStore = new Map(store);
        newStore.set(resultId, {
            id: resultId,
            shape: resultShape,
            createdAt: new Date(),
            device: instance.device,
            operation: 'matmul'
        });
        return newStore;
    });
    
    updateMemoryUsage();
    return resultId;
}

// Svelte actions for DOM integration
export function tensorVisualization(node, { tensorId, options = {} }) {
    let canvas;
    let ctx;
    
    function setupCanvas() {
        canvas = document.createElement('canvas');
        canvas.width = options.width || 200;
        canvas.height = options.height || 200;
        ctx = canvas.getContext('2d');
        node.appendChild(canvas);
    }
    
    function renderTensor() {
        if (!tensorId || !ctx) return;
        
        try {
            const tensor = getTensorData(tensorId);
            if (!tensor) return;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Simple visualization for 2D tensors
            if (tensor.shape.length === 2) {
                const [rows, cols] = tensor.shape;
                const cellWidth = canvas.width / cols;
                const cellHeight = canvas.height / rows;
                
                for (let i = 0; i < rows; i++) {
                    for (let j = 0; j < cols; j++) {
                        const value = tensor.data[i * cols + j];
                        const intensity = Math.min(255, Math.max(0, Math.abs(value) * 255));
                        const color = value >= 0 ? `rgb(${intensity}, 0, 0)` : `rgb(0, 0, ${intensity})`;
                        
                        ctx.fillStyle = color;
                        ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                    }
                }
            } else {
                // For other shapes, draw a simple bar chart
                const values = Array.from(tensor.data.slice(0, 50)); // First 50 values
                const barWidth = canvas.width / values.length;
                const maxValue = Math.max(...values.map(Math.abs));
                
                values.forEach((value, index) => {
                    const barHeight = (Math.abs(value) / maxValue) * canvas.height;
                    const color = value >= 0 ? '#4CAF50' : '#F44336';
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(index * barWidth, canvas.height - barHeight, barWidth - 1, barHeight);
                });
            }
        } catch (error) {
            console.warn('Error rendering tensor:', error);
        }
    }
    
    setupCanvas();
    renderTensor();
    
    return {
        update(newOptions) {
            if (newOptions.tensorId !== tensorId) {
                tensorId = newOptions.tensorId;
                renderTensor();
            }
        },
        destroy() {
            if (canvas && canvas.parentNode) {
                canvas.parentNode.removeChild(canvas);
            }
        }
    };
}

// Svelte store utilities
export function createTensorSubscription(tensorId) {
    return derived(
        [tensorStore, memoryUsage],
        ([$tensorStore, $memoryUsage]) => {
            const tensorInfo = $tensorStore.get(tensorId);
            if (!tensorInfo) return null;
            
            try {
                const tensorData = getTensorData(tensorId);
                return {
                    ...tensorInfo,
                    data: tensorData?.data,
                    memoryUsage: tensorData?.data?.byteLength || 0
                };
            } catch (error) {
                return { ...tensorInfo, error: error.message };
            }
        }
    );
}

// Reactive tensor operations
export function createReactiveTensorOp(operation, inputs, options = {}) {
    return derived(
        [tensorStore, wasmState],
        ([$tensorStore, $wasmState]) => {
            if (!$wasmState.initialized || !inputs.every(id => $tensorStore.has(id))) {
                return { status: 'waiting', result: null };
            }
            
            try {
                let resultId;
                
                switch (operation) {
                    case 'add':
                        resultId = addTensors(inputs[0], inputs[1]);
                        break;
                    case 'matmul':
                        resultId = matrixMultiply(inputs[0], inputs[1]);
                        break;
                    case 'softmax':
                        {
                            const instance = getWasmInstance();
                            resultId = instance.softmax(inputs[0], options.axis);
                            break;
                        }
                    case 'relu':
                        {
                            const instance = getWasmInstance();
                            resultId = instance.relu(inputs[0]);
                            break;
                        }
                    default:
                        throw new Error(`Unknown operation: ${operation}`);
                }
                
                return { status: 'success', result: resultId };
            } catch (error) {
                return { status: 'error', error: error.message };
            }
        }
    );
}

// Performance monitoring
export const performanceMetrics = writable({
    operationCounts: {},
    averageTimes: {},
    lastOperationTime: 0
});

export function measureOperation(operationName, operationFn) {
    return async (...args) => {
        const startTime = performance.now();
        
        try {
            const result = await operationFn(...args);
            const endTime = performance.now();
            const duration = endTime - startTime;
            
            performanceMetrics.update(metrics => {
                const newMetrics = { ...metrics };
                newMetrics.operationCounts[operationName] = (newMetrics.operationCounts[operationName] || 0) + 1;
                
                const previousAvg = newMetrics.averageTimes[operationName] || 0;
                const count = newMetrics.operationCounts[operationName];
                newMetrics.averageTimes[operationName] = (previousAvg * (count - 1) + duration) / count;
                newMetrics.lastOperationTime = duration;
                
                return newMetrics;
            });
            
            return result;
        } catch (error) {
            const endTime = performance.now();
            const duration = endTime - startTime;
            
            performanceMetrics.update(metrics => ({
                ...metrics,
                lastOperationTime: duration
            }));
            
            throw error;
        }
    };
}

// Error handling utilities
export function createErrorHandler(storeName = 'default') {
    const errorStore = writable(null);
    
    return {
        store: errorStore,
        handle: (error, context = '') => {
            console.error(`[${storeName}] ${context}:`, error);
            errorStore.set({
                message: error.message || error,
                context,
                timestamp: new Date(),
                stack: error.stack
            });
        },
        clear: () => errorStore.set(null)
    };
}

// Cleanup utilities
export function cleanupTensors() {
    const currentTensors = get(tensorStore);
    for (const tensorId of currentTensors.keys()) {
        deleteTensor(tensorId);
    }
}

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', cleanupTensors);
}

// Export for use in Svelte components
export default {
    // Core functions
    initializeTrustformers,
    enableWebGPU,
    createTensor,
    deleteTensor,
    getTensorData,
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
    
    // Utilities
    tensorVisualization,
    createTensorSubscription,
    createReactiveTensorOp,
    measureOperation,
    createErrorHandler,
    updateMemoryUsage
};