# TrustformeRS WASM API Reference

This document provides a complete reference for the TrustformeRS WebAssembly API.

## Table of Contents

- [Core Classes](#core-classes)
- [Tensor Operations](#tensor-operations)
- [Configuration Objects](#configuration-objects)
- [Utility Functions](#utility-functions)
- [Error Handling](#error-handling)
- [Type Definitions](#type-definitions)

## Core Classes

### TrustformersWasm

Main entry point for TrustformeRS functionality.

```typescript
class TrustformersWasm {
    constructor();
    readonly version: string;
}
```

**Example:**
```javascript
const tf = new TrustformersWasm();
console.log(tf.version); // "0.1.0-alpha.1"
```

### InferenceSession

Manages model loading and inference execution.

```typescript
class InferenceSession {
    constructor(task_type: string);
    
    // Initialization
    initialize_with_auto_device(): Promise<void>;
    initialize_with_device(device_type: string): Promise<void>;
    
    // Model management
    load_model(model_data: Uint8Array): Promise<void>;
    load_model_with_quantization(model_data: Uint8Array): Promise<void>;
    unload_model(): void;
    
    // Inference
    predict(input: WasmTensor): any;
    predict_with_batching(input: WasmTensor, priority: Priority): Promise<any>;
    
    // Device management
    readonly current_device_type: string;
    get_device_capabilities(): DeviceCapabilities;
    force_device_type(device_type: string): void;
    
    // Configuration
    enable_quantization(config: QuantizationConfig): void;
    disable_quantization(): void;
    enable_batch_processing(config: BatchConfig): void;
    disable_batch_processing(): void;
    enable_debug_logging(config: DebugConfig): void;
    
    // Storage
    initialize_storage(max_size_mb: number): Promise<void>;
    get_cache_stats(): CacheStats;
    clear_cache(): void;
    
    // Streaming
    create_generation_stream(): GenerationStream;
    
    // Cleanup
    cleanup(): void;
}
```

**Task Types:**
- `"text-generation"` - Language model text generation
- `"text-classification"` - Text classification tasks
- `"image-captioning"` - Multimodal image captioning
- `"translation"` - Language translation
- `"code-completion"` - Code completion tasks

**Device Types:**
- `"CPU"` - CPU execution
- `"GPU"` - WebGPU acceleration
- `"Auto"` - Automatic selection

**Example:**
```javascript
const session = new InferenceSession('text-generation');
await session.initialize_with_auto_device();

const modelData = new Uint8Array(/* model bytes */);
await session.load_model(modelData);

const result = session.predict(inputTensor);
```

### WasmTensor

Represents multi-dimensional arrays for model input/output.

```typescript
class WasmTensor {
    constructor(data: Float32Array, shape: number[]);
    
    readonly data: Float32Array;
    readonly shape: number[];
    readonly size: number;
    readonly dtype: string;
    
    // Operations
    reshape(new_shape: number[]): WasmTensor;
    clone(): WasmTensor;
    to_cpu(): WasmTensor;
    to_gpu(): WasmTensor;
    
    // Data access
    get(indices: number[]): number;
    set(indices: number[], value: number): void;
    
    // Serialization
    to_array(): Float32Array;
    static from_array(data: Float32Array, shape: number[]): WasmTensor;
}
```

**Example:**
```javascript
// Create 2x3 matrix
const data = new Float32Array([1, 2, 3, 4, 5, 6]);
const tensor = new WasmTensor(data, [2, 3]);

console.log(tensor.shape); // [2, 3]
console.log(tensor.size);  // 6

// Reshape to 3x2
const reshaped = tensor.reshape([3, 2]);
console.log(reshaped.shape); // [3, 2]
```

## Configuration Objects

### QuantizationConfig

Configuration for model quantization.

```typescript
class QuantizationConfig {
    precision: QuantizationPrecision;
    calibration_samples: number;
    enable_dynamic: boolean;
    preserve_accuracy: boolean;
    
    constructor();
}

enum QuantizationPrecision {
    FP32 = "FP32",
    FP16 = "FP16", 
    Int8 = "Int8",
    Int4 = "Int4"
}
```

**Example:**
```javascript
const quantConfig = new QuantizationConfig();
quantConfig.precision = QuantizationPrecision.Int8;
quantConfig.calibration_samples = 100;
quantConfig.enable_dynamic = true;

session.enable_quantization(quantConfig);
```

### BatchConfig

Configuration for batch processing.

```typescript
class BatchConfig {
    max_batch_size: number;
    timeout_ms: number;
    strategy: BatchingStrategy;
    priority_levels: number;
    
    constructor();
    
    static real_time(): BatchConfig;
    static throughput_optimized(): BatchConfig;
    static memory_efficient(): BatchConfig;
}

enum BatchingStrategy {
    Immediate = "Immediate",
    FixedSize = "FixedSize", 
    Dynamic = "Dynamic",
    Adaptive = "Adaptive"
}

enum Priority {
    Low = "Low",
    Normal = "Normal",
    High = "High",
    Critical = "Critical"
}
```

**Example:**
```javascript
const batchConfig = BatchConfig.real_time();
// Or customize:
// batchConfig.max_batch_size = 8;
// batchConfig.timeout_ms = 100;

session.enable_batch_processing(batchConfig);
```

### DebugConfig

Configuration for debugging and logging.

```typescript
class DebugConfig {
    level: LogLevel;
    enable_performance_monitoring: boolean;
    enable_memory_tracking: boolean;
    log_tensor_shapes: boolean;
    max_log_entries: number;
    
    constructor();
}

enum LogLevel {
    Off = "Off",
    Error = "Error",
    Warn = "Warn", 
    Info = "Info",
    Debug = "Debug",
    Trace = "Trace"
}
```

**Example:**
```javascript
const debugConfig = new DebugConfig();
debugConfig.level = LogLevel.Info;
debugConfig.enable_performance_monitoring = true;
debugConfig.enable_memory_tracking = true;

session.enable_debug_logging(debugConfig);
```

## Type Definitions

### DeviceCapabilities

Information about compute device capabilities.

```typescript
interface DeviceCapabilities {
    device_type: string;
    memory_mb: number;
    compute_units: number;
    supports_fp16: boolean;
    supports_int8: boolean;
    max_texture_size: number;
    vendor: string;
}
```

### CacheStats

Statistics about model cache usage.

```typescript
interface CacheStats {
    total_size_mb: number;
    used_size_mb: number;
    model_count: number;
    hit_rate: number;
    eviction_count: number;
}
```

### MemoryStats

System memory usage statistics.

```typescript
interface MemoryStats {
    wasm_memory: number;      // WASM heap size in bytes
    gpu_memory: number;       // GPU memory usage in bytes  
    model_memory: number;     // Model weights memory in bytes
    cache_memory: number;     // Cache memory usage in bytes
    peak_memory: number;      // Peak memory usage in bytes
}
```

### GenerationStream

Iterator for streaming text generation.

```typescript
interface GenerationStream {
    [Symbol.asyncIterator](): AsyncIterator<GenerationToken>;
    cancel(): void;
}

interface GenerationToken {
    token: string;
    token_id: number;
    logprobs?: number[];
    finish_reason?: string;
    metadata?: any;
}
```

**Example:**
```javascript
const stream = session.create_generation_stream();

for await (const token of stream) {
    console.log('Token:', token.token);
    
    if (token.finish_reason) {
        console.log('Generation finished:', token.finish_reason);
        break;
    }
}
```

## Utility Functions

### Memory Management

```typescript
function get_memory_stats(): MemoryStats;
function force_gc(): void;
function set_memory_limit(limit_mb: number): void;
```

**Example:**
```javascript
import { get_memory_stats, force_gc } from 'trustformers-wasm';

const stats = get_memory_stats();
console.log('Memory usage:', stats.wasm_memory / 1024 / 1024, 'MB');

// Force garbage collection
force_gc();
```

### Device Detection

```typescript
function is_webgpu_supported(): boolean;
function is_mobile_device(): boolean;
function get_browser_info(): BrowserInfo;
```

**Example:**
```javascript
import { 
    is_webgpu_supported, 
    is_mobile_device,
    get_browser_info 
} from 'trustformers-wasm';

if (is_webgpu_supported()) {
    console.log('WebGPU is available');
}

if (is_mobile_device()) {
    console.log('Running on mobile device');
}

const browserInfo = get_browser_info();
console.log('Browser:', browserInfo.name, browserInfo.version);
```

### Performance Utilities

```typescript
function measure_performance<T>(fn: () => T): PerformanceResult<T>;
function benchmark_operations(): BenchmarkResults;
```

**Example:**
```javascript
import { measure_performance } from 'trustformers-wasm';

const result = measure_performance(() => {
    return session.predict(inputTensor);
});

console.log('Execution time:', result.duration_ms, 'ms');
console.log('Result:', result.value);
```

## Error Handling

### Error Types

TrustformeRS WASM defines several specific error types:

```typescript
class TrustformersError extends Error {
    readonly code: string;
    readonly context?: any;
}

class ModelLoadError extends TrustformersError {}
class InferenceError extends TrustformersError {}  
class DeviceError extends TrustformersError {}
class MemoryError extends TrustformersError {}
class ConfigurationError extends TrustformersError {}
```

### Error Codes

- `E1001` - Model file format invalid
- `E1002` - Model size exceeds memory limit
- `E1003` - Unsupported model architecture
- `E2001` - Inference input shape mismatch
- `E2002` - Inference computation failed
- `E3001` - Device initialization failed
- `E3002` - WebGPU not available
- `E4001` - Out of memory
- `E4002` - Memory allocation failed
- `E5001` - Invalid configuration parameter

**Example:**
```javascript
try {
    await session.load_model(modelData);
} catch (error) {
    if (error instanceof ModelLoadError) {
        console.error('Model loading failed:', error.message);
        console.error('Error code:', error.code);
    } else if (error instanceof MemoryError) {
        console.error('Not enough memory:', error.message);
        // Try enabling quantization
        session.enable_quantization(new QuantizationConfig());
    }
}
```

## Advanced Usage

### Custom Model Loading

```typescript
interface ModelLoader {
    load_from_url(url: string, progress_callback?: (progress: number) => void): Promise<Uint8Array>;
    load_from_file(file: File): Promise<Uint8Array>;
    load_with_auth(url: string, auth_token: string): Promise<Uint8Array>;
}
```

### Plugin System

```typescript
interface Plugin {
    name: string;
    version: string;
    initialize(session: InferenceSession): Promise<void>;
    process(input: any): Promise<any>;
    cleanup(): void;
}

function register_plugin(plugin: Plugin): void;
function unregister_plugin(plugin_name: string): void;
```

### Custom Operators

```typescript
interface CustomOperator {
    name: string;
    forward(inputs: WasmTensor[]): WasmTensor[];
    supports_gpu: boolean;
}

function register_operator(operator: CustomOperator): void;
```

## Environment Variables

TrustformeRS WASM can be configured via environment variables:

- `TRUSTFORMERS_LOG_LEVEL` - Set default log level
- `TRUSTFORMERS_MEMORY_LIMIT` - Set memory limit in MB
- `TRUSTFORMERS_CACHE_SIZE` - Set cache size in MB
- `TRUSTFORMERS_DEVICE_TYPE` - Force device type
- `TRUSTFORMERS_THREADS` - Number of worker threads

## Browser Compatibility

### Feature Support Matrix

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebAssembly | 57+ | 52+ | 11+ | 16+ |
| WebGPU | 94+ | 113+ | 18+ | 94+ |
| SharedArrayBuffer | 68+ | 79+ | 15.2+ | 79+ |
| OffscreenCanvas | 69+ | 105+ | 16.4+ | 79+ |

### Polyfills

For older browsers, consider these polyfills:

```html
<!-- WebAssembly polyfill for very old browsers -->
<script src="https://cdn.jsdelivr.net/npm/@webassemblyjs/wasm-loader@1.9.0/lib/index.js"></script>

<!-- SharedArrayBuffer polyfill -->
<script src="https://cdn.jsdelivr.net/npm/sharedarraybuffer-polyfill@1.0.0/dist/sharedarraybuffer.js"></script>
```

## Performance Tips

1. **Use WebGPU when available** for 10x+ speedup
2. **Enable quantization** to reduce memory usage by 50-75%
3. **Use batch processing** for multiple requests
4. **Cache models** to avoid repeated loading
5. **Monitor memory usage** to prevent OOM errors
6. **Use streaming** for long text generation

## Migration Guide

### From v0.0.x to v0.1.x

- Replace `WasmModel` with `InferenceSession`
- Update tensor creation: `new Tensor()` → `new WasmTensor()`
- Add device initialization before model loading
- Update error handling for new error types

**Before:**
```javascript
const model = new WasmModel();
await model.load(modelData);
const result = model.predict(input);
```

**After:**
```javascript
const session = new InferenceSession('text-generation');
await session.initialize_with_auto_device();
await session.load_model(modelData);
const result = session.predict(input);
```

## Examples

See the [examples directory](../examples/) for complete working examples:

- [Text Generation Playground](../examples/text-generation-playground.html)
- [Image Captioning Demo](../examples/image-captioning-demo.html)  
- [Code Completion Example](../examples/code-completion-demo.html)
- [Multilingual Translation](../examples/multilingual-translation-demo.html)
- [Chat Interface](../examples/chat-interface-demo.html)

## Contributing

To contribute to TrustformeRS WASM:

1. Read the [contribution guidelines](../CONTRIBUTING.md)
2. Set up the [development environment](../docs/development.md)
3. Run tests: `wasm-pack test --chrome --headless`
4. Submit a pull request

## License

TrustformeRS WASM is licensed under the [MIT License](../LICENSE).