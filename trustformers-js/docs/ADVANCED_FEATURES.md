# TrustformeRS Advanced Features Guide

This guide covers the cutting-edge features added to trustformers-js for enterprise-grade ML deployment.

## Table of Contents

1. [WebNN Backend](#webnn-backend)
2. [Advanced Quantization](#advanced-quantization)
3. [Benchmark Suite](#benchmark-suite)
4. [Advanced Caching](#advanced-caching)

---

## WebNN Backend

The WebNN (Web Neural Network API) backend provides hardware-accelerated machine learning operations with native support for CPU, GPU, and NPU devices.

### Features

- **Hardware Acceleration**: Native acceleration on CPU, GPU, and NPU
- **Graph-based Computation**: Optimized computation graphs for maximum performance
- **Operator Fusion**: Advanced optimization through graph-level operation fusion
- **Multi-device Support**: Automatic device selection and capability detection
- **Advanced Operations**: Attention mechanisms, layer normalization, quantization

### Quick Start

```javascript
import { initWebNN, WebNNBackend, isWebNNAvailable } from 'trustformers-js';

// Check if WebNN is available
if (isWebNNAvailable()) {
  console.log('WebNN is supported!');
}

// Initialize WebNN backend
const success = await initWebNN({
  deviceType: 'gpu',  // 'cpu', 'gpu', or 'npu'
  powerPreference: 'high-performance'  // 'default', 'high-performance', or 'low-power'
});

if (success) {
  console.log('WebNN backend initialized successfully');
}
```

### Capability Detection

```javascript
import { WebNNCapabilities } from 'trustformers-js';

const caps = await WebNNCapabilities.detect();

console.log('Available devices:', caps.devices);  // ['gpu', 'cpu']
console.log('Supported operations:', caps.supportedOps);  // ['matmul', 'conv2d', ...]
console.log('Features:', caps.features);
```

### Using WebNN Operations

```javascript
import { WebNNBackend } from 'trustformers-js';

const backend = new WebNNBackend({ deviceType: 'gpu' });
await backend.initialize();

// Create tensors
const a = { shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) };
const b = { shape: [3, 2], data: new Float32Array([1, 2, 3, 4, 5, 6]) };

// Matrix multiplication
const result = await backend.matmul(a, b);

// Element-wise addition
const c = { shape: [6], data: new Float32Array([1, 2, 3, 4, 5, 6]) };
const d = { shape: [6], data: new Float32Array([1, 1, 1, 1, 1, 1]) };
const sum = await backend.add(c, d);

// Activation functions
const input = { shape: [4], data: new Float32Array([-1, 0, 1, 2]) };
const activated = await backend.relu(input);
```

### Building Custom Graphs

```javascript
import { WebNNBackend, WebNNOperations } from 'trustformers-js';

const backend = new WebNNBackend({ deviceType: 'gpu' });
await backend.initialize();

// Build attention mechanism
const graph = await backend.buildAttentionGraph(768, 12, {
  // hiddenSize: 768, numHeads: 12
});

// Execute graph
const output = await backend.executeGraph('attention', {
  query: queryData,
  key: keyData,
  value: valueData
});
```

### Quantization with WebNN

```javascript
// Quantize tensor to INT8
const quantized = await backend.quantizeInt8(input, scale, zeroPoint);

// Dequantize back to FP32
const dequantized = await backend.dequantizeInt8(quantized, scale, zeroPoint);
```

### Performance Monitoring

```javascript
const stats = backend.getStatistics();

console.log('Graphs created:', stats.graphsCreated);
console.log('Executions completed:', stats.executionsCompleted);
console.log('Average execution time:', stats.averageExecutionTime, 'ms');
```

---

## Advanced Quantization

Comprehensive model quantization support for reducing model size and improving inference speed.

### Supported Formats

- **FP32**: 32-bit floating point (baseline)
- **FP16**: 16-bit floating point (~50% size reduction)
- **INT8**: 8-bit integer (~75% size reduction)
- **INT4**: 4-bit integer (~87.5% size reduction)
- **GGML**: llama.cpp compatible formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)

### Float16 Quantization

```javascript
import { Float16Utils } from 'trustformers-js';

// Single value conversion
const fp32Value = 3.14159;
const fp16Bits = Float16Utils.float32ToFloat16(fp32Value);
const backToFP32 = Float16Utils.float16ToFloat32(fp16Bits);

// Array conversion
const fp32Array = new Float32Array([1.0, 2.5, 3.14, -1.5]);
const fp16Array = Float16Utils.float32ArrayToFloat16Array(fp32Array);
const fp32Back = Float16Utils.float16ArrayToFloat32Array(fp16Array);

console.log('Compression ratio: 50%');
```

### INT8 Quantization

```javascript
import { Int8Quantizer } from 'trustformers-js';

const weights = new Float32Array(1000).map(() => Math.random() * 10 - 5);

// Symmetric quantization
const quantized = Int8Quantizer.quantize(weights);
console.log('Quantized data:', quantized.data);  // Int8Array
console.log('Scale:', quantized.scale);
console.log('Zero point:', quantized.zeroPoint);

// Dequantize
const restored = Int8Quantizer.dequantize(
  quantized.data,
  quantized.scale,
  quantized.zeroPoint
);

// Per-channel quantization
const shape = [32, 128];  // [out_channels, in_features]
const perChannel = Int8Quantizer.quantizePerChannel(weights, shape, 0);
console.log('Scales per channel:', perChannel.scales);
```

### INT4 Quantization

```javascript
import { Int4Quantizer } from 'trustformers-js';

const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);

// Quantize to 4-bit (packed in Uint8Array)
const quantized = Int4Quantizer.quantize(data);
console.log('Original size:', data.byteLength, 'bytes');
console.log('Quantized size:', quantized.data.byteLength, 'bytes');
console.log('Compression: 87.5%');

// Dequantize
const restored = Int4Quantizer.dequantize(
  quantized.data,
  quantized.scale,
  quantized.zeroPoint,
  quantized.originalLength
);
```

### GGML Quantization

```javascript
import { GGMLQuantizer, QuantizationType } from 'trustformers-js';

const weights = new Float32Array(1024).map(() => Math.random() * 10 - 5);

// GGML Q4_0: 4-bit with block-wise scaling
const q4_0 = GGMLQuantizer.quantizeQ4_0(weights);
console.log('Format:', q4_0.type);  // 'ggml_q4_0'
console.log('Block size:', q4_0.blockSize);  // 32
console.log('Num blocks:', q4_0.numBlocks);

// Dequantize
const restored = GGMLQuantizer.dequantizeQ4_0(q4_0.data, weights.length);

// GGML Q8_0: 8-bit with block-wise scaling
const q8_0 = GGMLQuantizer.quantizeQ8_0(weights);
const restored8 = GGMLQuantizer.dequantizeQ8_0(q8_0.data, weights.length);
```

### Quantization Calibration

```javascript
import { QuantizationCalibrator } from 'trustformers-js';

const calibrator = new QuantizationCalibrator();

// Collect statistics from calibration data
const calibrationData = new Float32Array(10000).map(() => Math.random() * 20 - 10);
calibrator.collectStatistics('layer1', calibrationData);

// Compute optimal scale using KL divergence minimization
const optimalScale = calibrator.computeOptimalScale('layer1', 8);  // 8-bit
console.log('Optimal scale:', optimalScale);
```

### Mixed Precision Quantization

```javascript
import { MixedPrecisionQuantizer, QuantizationType } from 'trustformers-js';

const quantizer = new MixedPrecisionQuantizer();

// Configure different layers with different precision
quantizer.setLayerQuantization('attention.q_proj', QuantizationType.FP16);
quantizer.setLayerQuantization('attention.k_proj', QuantizationType.FP16);
quantizer.setLayerQuantization('attention.v_proj', QuantizationType.FP16);
quantizer.setLayerQuantization('mlp.fc1', QuantizationType.INT8);
quantizer.setLayerQuantization('mlp.fc2', QuantizationType.INT8);
quantizer.setLayerQuantization('embeddings', QuantizationType.INT4);

// Quantize layers
const weights = new Float32Array(1000);
const quantizedAttention = quantizer.quantizeLayer('attention.q_proj', weights);
const quantizedMLP = quantizer.quantizeLayer('mlp.fc1', weights);

// Generate report
const report = quantizer.generateReport();
console.log('Layer configurations:', report.layers);
console.log('Average bits per weight:', report.avgBitsPerWeight);
```

### Quantization-Aware Training

```javascript
import { QuantizationAwareTraining } from 'trustformers-js';

// Simulate quantization during forward pass
const weights = new Float32Array([1.5, 2.7, -3.2, 4.1]);
const scale = 0.1;
const zeroPoint = 0;

const fakeQuantized = QuantizationAwareTraining.fakeQuantize(
  weights,
  scale,
  zeroPoint,
  8  // 8-bit
);

// Use fakeQuantized in training to learn quantization-robust weights
```

---

## Benchmark Suite

Comprehensive performance testing framework with statistical analysis and reporting.

### Quick Benchmark

```javascript
import { PerformanceStats } from 'trustformers-js';

// Collect measurements
const measurements = [];
for (let i = 0; i < 50; i++) {
  const start = performance.now();
  // Your operation here
  const result = someOperation();
  const end = performance.now();
  measurements.push(end - start);
}

// Calculate statistics
const stats = PerformanceStats.calculate(measurements);

console.log('Mean:', stats.mean, 'ms');
console.log('Median:', stats.median, 'ms');
console.log('Std Dev:', stats.stdDev, 'ms');
console.log('P95:', stats.p95, 'ms');
console.log('P99:', stats.p99, 'ms');
console.log('Coefficient of Variation:', stats.coefficientOfVariation, '%');
```

### Performance Comparison

```javascript
import { PerformanceStats } from 'trustformers-js';

// Baseline measurements
const baselineMeasurements = [100, 105, 102, 108, 103];
const baseline = PerformanceStats.calculate(baselineMeasurements);

// Current measurements
const currentMeasurements = [90, 88, 92, 89, 91];
const current = PerformanceStats.calculate(currentMeasurements);

// Compare
const comparison = PerformanceStats.compare(baseline, current);

console.log('Improvement:', comparison.improvementPercent, '%');
console.log('Faster:', comparison.faster);
console.log('Significant:', comparison.significant);
console.log('Status:', comparison.status);  // 'faster', 'slower', or 'similar'
```

### Tensor Operation Benchmarks

```javascript
import { TensorBenchmarks } from 'trustformers-js';

const backend = getYourBackend();  // WebGL, WebGPU, or WebNN
const benchmarker = new TensorBenchmarks(backend);

// Benchmark matrix multiplication
const matmulResults = await benchmarker.benchmarkMatMul([
  [64, 64],
  [128, 128],
  [256, 256],
  [512, 512]
]);

for (const result of matmulResults) {
  console.log(`${result.name}: ${result.timing.mean.toFixed(2)}ms`);
}

// Benchmark element-wise operations
const elementwiseResults = await benchmarker.benchmarkElementWise();

// Benchmark activations
const activationResults = await benchmarker.benchmarkActivations();

// Benchmark reductions
const reductionResults = await benchmarker.benchmarkReductions();
```

### Model Benchmarks

```javascript
import { ModelBenchmarks } from 'trustformers-js';

const benchmarker = new ModelBenchmarks(model, tokenizer);

// Benchmark text generation
const prompts = [
  "Once upon a time",
  "The quick brown fox",
  "In a galaxy far, far away"
];

const results = await benchmarker.benchmarkTextGeneration(prompts);

for (const result of results) {
  console.log(`Prompt: "${result.prompt}"`);
  console.log(`  Tokenization: ${result.tokenization.mean.toFixed(2)}ms`);
  console.log(`  Inference: ${result.inference.mean.toFixed(2)}ms`);
  console.log(`  Decoding: ${result.decoding.mean.toFixed(2)}ms`);
  console.log(`  End-to-end: ${result.endToEnd.mean.toFixed(2)}ms`);
}

// Benchmark batch processing
const batchResults = await benchmarker.benchmarkBatchProcessing([1, 4, 8, 16]);

for (const result of batchResults) {
  console.log(`Batch size ${result.batchSize}:`);
  console.log(`  Latency: ${result.timing.mean.toFixed(2)}ms`);
  console.log(`  Throughput: ${result.throughput.toFixed(0)} samples/sec`);
}

// Benchmark different precision settings
const precisionResults = await benchmarker.benchmarkPrecision(
  input,
  ['fp32', 'fp16', 'int8', 'int4']
);
```

### Backend Comparison

```javascript
import { BackendComparison } from 'trustformers-js';

const backends = [
  { name: 'WebGL', backend: webglBackend },
  { name: 'WebGPU', backend: webgpuBackend },
  { name: 'WebNN', backend: webnnBackend },
  { name: 'WASM', backend: wasmBackend }
];

const comparison = new BackendComparison(backends);

// Compare specific operation
const matmulComparison = await comparison.compareOperation(
  'matmul',
  () => createMatmulTest()
);

console.log('Fastest backend:', matmulComparison._comparison.fastest);

for (const result of matmulComparison._comparison.results) {
  console.log(`${result.backend}: ${result.mean.toFixed(2)}ms (${result.relativeSpeed}x)`);
}

// Full comparison
const fullResults = await comparison.runFullComparison();
```

### Memory Benchmarks

```javascript
import { MemoryBenchmark } from 'trustformers-js';

// Benchmark allocation patterns
const allocationResults = await MemoryBenchmark.benchmarkAllocation([
  1000,
  10000,
  100000,
  1000000
]);

for (const result of allocationResults) {
  console.log(`Size ${result.size}:`);
  console.log(`  Allocated: ${result.allocated} bytes`);
  console.log(`  Per tensor: ${result.perTensor} bytes`);
}

// Benchmark pooling efficiency
const poolingResults = await MemoryBenchmark.benchmarkPooling(memoryPool);

console.log('With pooling:', poolingResults.withPool.mean, 'ms');
console.log('Without pooling:', poolingResults.withoutPool.mean, 'ms');
```

### Full Benchmark Suite

```javascript
import { BenchmarkSuite } from 'trustformers-js';

const suite = new BenchmarkSuite({
  warmupRuns: 5,
  benchmarkRuns: 50,
  timeout: 60000,
  collectMemoryStats: true,
  backends: ['wasm', 'webgl', 'webgpu', 'webnn'],
  reportFormat: 'both'  // 'json', 'html', or 'both'
});

// Run all benchmarks
const results = await suite.runAll(backends);

// Generate HTML report
const htmlReport = suite.generateHTMLReport();

// Save to file (Node.js)
await suite.saveReport('benchmark-report.html', 'html');

// Generate JSON report
const jsonReport = suite.generateJSONReport();
await suite.saveReport('benchmark-report.json', 'json');
```

---

## Advanced Caching

Multi-level caching strategies for optimizing model inference and data access.

### LRU Cache

Least Recently Used cache with size and memory limits.

```javascript
import { LRUCache } from 'trustformers-js';

const cache = new LRUCache({
  maxSize: 100,  // Maximum number of items
  maxMemory: 100 * 1024 * 1024,  // 100MB maximum memory
  onEvict: (key, value) => {
    console.log(`Evicted: ${key}`);
  }
});

// Set values
cache.set('model_weights', weights);
cache.set('activations', activations, { size: 1024 * 1024 });  // Specify size

// Get values
const value = cache.get('model_weights');

// Check existence
if (cache.has('activations')) {
  console.log('Cache hit!');
}

// Statistics
const stats = cache.getStatistics();
console.log('Hit rate:', stats.hitRate);
console.log('Memory used:', stats.currentMemory, 'bytes');
console.log('Evictions:', stats.evictions);
```

### TTL Cache

Time-To-Live cache with automatic expiration.

```javascript
import { TTLCache } from 'trustformers-js';

const cache = new TTLCache({
  defaultTTL: 60000,  // 1 minute default
  checkInterval: 10000,  // Check for expired items every 10s
  maxSize: 1000
});

// Set with default TTL
cache.set('key1', 'value1');

// Set with custom TTL
cache.set('key2', 'value2', 30000);  // 30 seconds

// Get value (returns null if expired)
const value = cache.get('key1');

// Statistics
const stats = cache.getStatistics();
console.log('Expirations:', stats.expirations);

// Cleanup
cache.dispose();  // Clear intervals and cache
```

### LFU Cache

Least Frequently Used cache.

```javascript
import { LFUCache } from 'trustformers-js';

const cache = new LFUCache({
  maxSize: 100
});

// Set values
cache.set('frequently_used', data1);
cache.set('rarely_used', data2);

// Access patterns determine eviction
for (let i = 0; i < 100; i++) {
  cache.get('frequently_used');  // Increase frequency
}

cache.get('rarely_used');  // Lower frequency

// When cache is full, 'rarely_used' will be evicted first

const stats = cache.getStatistics();
console.log('Hit rate:', stats.hitRate);
```

### Multi-Level Cache

Combines L1 (LRU), L2 (TTL), and L3 (LFU) caches.

```javascript
import { MultiLevelCache } from 'trustformers-js';

const cache = new MultiLevelCache({
  l1Size: 50,  // Fast, small
  l1Memory: 50 * 1024 * 1024,  // 50MB
  l2Size: 200,  // Medium
  l2TTL: 300000,  // 5 minutes
  l3Size: 1000  // Large, slower
});

// Set value (stored in all levels)
await cache.set('key', value);

// Get value (searches L1 → L2 → L3)
const result = await cache.get('key');

// Statistics show hit distribution
const stats = cache.getStatistics();
console.log('L1 hits:', stats.l1Hits);
console.log('L2 hits:', stats.l2Hits);
console.log('L3 hits:', stats.l3Hits);
console.log('Overall hit rate:', stats.hitRate);

// Cleanup
cache.dispose();
```

### Persistent Cache

Uses IndexedDB for persistence.

```javascript
import { PersistentCache } from 'trustformers-js';

const cache = new PersistentCache({
  dbName: 'trustformers-cache',
  storeName: 'models',
  version: 1
});

// Initialize (async)
await cache.initialize();

// Set value (persisted to IndexedDB)
await cache.set('model_weights', weightsData);

// Get value (from memory cache or IndexedDB)
const weights = await cache.get('model_weights');

// Delete
await cache.delete('model_weights');

// Clear all
await cache.clear();
```

### Cache Manager

High-level cache management with prefetching and warming.

```javascript
import { CacheManager } from 'trustformers-js';

const manager = new CacheManager({
  useMultiLevel: true,
  l1Size: 50,
  l2Size: 200,
  l3Size: 1000
});

// Get with automatic loading
const loader = async (key) => {
  // Load data from network/disk
  const data = await fetch(`/models/${key}`);
  return data.arrayBuffer();
};

const value = await manager.get('model_weights', loader);

// Prefetch in background
await manager.prefetch(
  ['model1', 'model2', 'model3'],
  loader
);

// Warm cache on startup
await manager.warm({
  'common_model': preloadedData1,
  'tokenizer': preloadedData2
});

// Statistics
const stats = manager.getStatistics();
console.log('Cache statistics:', stats);

// Clear
manager.clear();
```

### Cache Patterns

#### Model Caching

```javascript
const modelCache = new LRUCache({ maxSize: 5, maxMemory: 500 * 1024 * 1024 });

async function loadModel(modelName) {
  // Check cache first
  let model = modelCache.get(modelName);

  if (!model) {
    // Load from network
    model = await fetch(`/models/${modelName}`).then(r => r.arrayBuffer());
    modelCache.set(modelName, model, { size: model.byteLength });
  }

  return model;
}
```

#### Inference Result Caching

```javascript
const resultCache = new TTLCache({ defaultTTL: 300000 });  // 5 minutes

async function cachedInference(input) {
  const cacheKey = hashInput(input);

  let result = resultCache.get(cacheKey);

  if (!result) {
    result = await model.forward(input);
    resultCache.set(cacheKey, result);
  }

  return result;
}
```

---

## Performance Tips

### WebNN Backend

1. **Device Selection**: Use GPU for large operations, CPU for small ones
2. **Graph Reuse**: Build graphs once, execute multiple times
3. **Batch Operations**: Combine multiple operations into a single graph
4. **Avoid Frequent Transfers**: Minimize CPU ↔ GPU data transfers

### Quantization

1. **Calibration**: Use representative data for calibration
2. **Mixed Precision**: Keep attention layers in higher precision
3. **GGML for LLMs**: Use GGML formats for large language models
4. **Profile**: Measure accuracy vs. speed tradeoffs

### Benchmarking

1. **Warmup**: Always run warmup iterations
2. **Multiple Runs**: Use 50+ runs for statistical significance
3. **Stable Environment**: Minimize background processes
4. **Memory Cleanup**: Clear caches between benchmarks

### Caching

1. **Multi-Level**: Use multi-level cache for diverse access patterns
2. **Size Limits**: Set appropriate memory limits
3. **TTL**: Use TTL for time-sensitive data
4. **Prefetch**: Prefetch commonly accessed models

---

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebNN | 116+ | - | - | 116+ |
| Quantization | ✅ | ✅ | ✅ | ✅ |
| Benchmarking | ✅ | ✅ | ✅ | ✅ |
| LRU/TTL/LFU Cache | ✅ | ✅ | ✅ | ✅ |
| IndexedDB Cache | ✅ | ✅ | ✅ | ✅ |

## Next Steps

- See [examples/advanced-features-demo.html](../examples/advanced-features-demo.html) for interactive demonstrations
- Run [test/advanced-features.test.js](../test/advanced-features.test.js) for integration tests
- Check [TODO.md](../TODO.md) for upcoming features

## Support

For issues or questions:
- GitHub Issues: https://github.com/trustformers/trustformers/issues
- Documentation: https://trustformers.ai/docs
