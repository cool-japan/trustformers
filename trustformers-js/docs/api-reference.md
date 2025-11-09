# TrustformeRS JavaScript API Reference

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Initialization](#initialization)
4. [Core API](#core-api)
5. [Enhanced Performance API](#enhanced-performance-api)
6. [Memory Management](#memory-management)
7. [Performance Profiling](#performance-profiling)
8. [Zero-Copy Operations](#zero-copy-operations)
9. [WebGL Backend](#webgl-backend)
10. [Model Operations](#model-operations)
11. [Tensor Operations](#tensor-operations)
12. [Pipeline API](#pipeline-api)
13. [Streaming](#streaming)
14. [Error Handling](#error-handling)
15. [Examples](#examples)

## Installation

### NPM
```bash
npm install trustformers
```

### CDN
```html
<script type="module">
  import * as tf from 'https://cdn.jsdelivr.net/npm/trustformers@latest/dist/trustformers.js';
</script>
```

## Quick Start

```javascript
import { initialize, tensor, createModel } from 'trustformers';

// Initialize TrustformeRS
await initialize();

// Create a tensor
const data = [1, 2, 3, 4];
const shape = [2, 2];
const t = tensor(data, shape);

// Create a model
const model = createModel('bert_base');

// Run inference
const outputs = await model.forward(t);
```

## Initialization

### Basic Initialization

```javascript
import { initialize } from 'trustformers';

await initialize({
  wasmPath: './trustformers_wasm_bg.wasm', // Path to WASM file
  initPanicHook: true                       // Enable better error messages
});
```

### Enhanced Initialization with Performance Optimizations

```javascript
import { initializeEnhanced } from 'trustformers';

const capabilities = await initializeEnhanced({
  // Basic options
  wasmPath: './trustformers_wasm_bg.wasm',
  initPanicHook: true,
  
  // Performance features
  enableWebGL: true,        // Enable WebGL backend
  enableMemoryPool: true,   // Enable memory pooling
  enableProfiling: true,    // Enable performance profiling
  enableZeroCopy: true,     // Enable zero-copy transfers
  
  // Feature-specific options
  webglOptions: {
    canvas: document.getElementById('canvas'), // Optional canvas
    memory: { maxPoolSize: 50 }
  },
  memoryOptions: {
    maxTotalMemory: 1024 * 1024 * 1024, // 1GB limit
    maxPoolSize: 100
  },
  profilingOptions: {
    detailed: true,
    autoReport: false
  }
});

console.log('Available capabilities:', capabilities);
// { wasm: true, webgl: true, webgpu: false, memoryPool: true, profiling: true, zeroCopy: true }
```

## Core API

### Tensor Creation

#### `tensor(data, shape)`
Create a tensor from JavaScript array.

```javascript
const t = tensor([1, 2, 3, 4], [2, 2]);
```

#### `zeros(shape)`
Create a tensor filled with zeros.

```javascript
const t = zeros([3, 3]);
```

#### `ones(shape)`
Create a tensor filled with ones.

```javascript
const t = ones([2, 4]);
```

#### `randn(shape)`
Create a tensor with random normal distribution.

```javascript
const t = randn([10, 10]);
```

### Model Creation

#### `createModel(configOrType)`
Create a model from configuration or type string.

```javascript
// From predefined type
const model = createModel('bert_base');

// From custom configuration
const config = createModelConfig('llama_7b');
config.num_layers = 24;
const model = createModel(config);
```

#### `createModelConfig(modelType)`
Create model configuration for specific architecture.

```javascript
const config = createModelConfig('gpt2_base');
// Modify configuration as needed
config.vocab_size = 50000;
```

### Tokenizer Creation

#### `createTokenizer(type, vocab)`
Create a tokenizer instance.

```javascript
const tokenizer = createTokenizer('wordpiece');
// Load vocabulary if available
tokenizer.load_vocab(vocab);
```

## Enhanced Performance API

### Enhanced Tensor Operations

The enhanced API provides automatic backend selection and performance optimization.

#### `enhanced_tensor_ops.matmul(a, b, options)`
Enhanced matrix multiplication with automatic backend selection.

```javascript
import { enhanced_tensor_ops } from 'trustformers';

const result = await enhanced_tensor_ops.matmul(a, b, {
  backend: 'auto',  // 'auto', 'webgl', 'webgpu', 'wasm'
  profile: true     // Enable profiling
});
```

#### `enhanced_tensor_ops.elementWise(a, b, operation, options)`
Enhanced element-wise operations.

```javascript
const result = await enhanced_tensor_ops.elementWise(a, b, 'add', {
  backend: 'webgl',
  profile: true
});
```

#### `enhanced_tensor_ops.activation(tensor, activation, options)`
Enhanced activation functions.

```javascript
const result = await enhanced_tensor_ops.activation(input, 'relu', {
  backend: 'auto',
  profile: true
});
```

### Enhanced Tensor Utilities

#### `enhanced_tensor_utils.createTensor(data, shape, options)`
Create tensor with automatic memory management.

```javascript
import { enhanced_tensor_utils } from 'trustformers';

const tensor = enhanced_tensor_utils.createTensor([1, 2, 3, 4], [2, 2], {
  dtype: 'f32',
  useMemoryPool: true,
  zeroCopy: true
});
```

#### `enhanced_tensor_utils.zeros(shape, options)`
Create zeros tensor with pooling.

```javascript
const tensor = enhanced_tensor_utils.zeros([100, 100], {
  dtype: 'f32',
  useMemoryPool: true
});
```

#### `enhanced_tensor_utils.releaseTensor(tensor)`
Release tensor back to memory pool.

```javascript
enhanced_tensor_utils.releaseTensor(tensor);
```

### Enhanced Inference

#### `enhanced_inference.runInference(model, inputs, options)`
Run optimized model inference.

```javascript
import { enhanced_inference } from 'trustformers';

const outputs = await enhanced_inference.runInference(model, inputs, {
  profile: true,
  useMemoryPool: true,
  autoCleanup: true,
  timeout: 30000,
  backend: 'auto'
});
```

#### `enhanced_inference.batchInference(model, batchInputs, options)`
Batch inference with optimizations.

```javascript
const results = await enhanced_inference.batchInference(model, batches, {
  batchSize: 32,
  profile: true
});
```

## Memory Management

### Memory Statistics

#### `memory.getStats()`
Get current memory usage statistics.

```javascript
const stats = memory.getStats();
console.log('Memory usage:', stats);
// { used: 12345678, limit: 2147483648, wasm: 1234567, js: 11111111 }
```

#### `memory.getUsage()`
Get WASM memory usage in bytes.

```javascript
const usage = memory.getUsage();
console.log('WASM memory:', usage, 'bytes');
```

### Memory Pool Management

#### `getMemoryManager(options)`
Get or create global memory manager.

```javascript
import { getMemoryManager } from 'trustformers';

const manager = getMemoryManager({
  tensor: {
    maxPoolSize: 100,
    maxTotalMemory: 1024 * 1024 * 1024
  }
});

const stats = manager.getStats();
```

#### `withMemoryManagement(operation, inputs, options)`
Execute operation with automatic memory management.

```javascript
import { withMemoryManagement } from 'trustformers';

const result = await withMemoryManagement(
  () => model.forward(inputs),
  [inputs],
  { autoRelease: true }
);
```

## Performance Profiling

### Profiler API

#### `getProfiler(options)`
Get or create global profiler.

```javascript
import { getProfiler } from 'trustformers';

const profiler = getProfiler({
  enabled: true,
  detailed: true,
  autoReport: false
});
```

#### `profile(name, fn, metadata)`
Profile any function.

```javascript
import { profile } from 'trustformers';

const result = await profile('my_operation', async () => {
  return await someExpensiveOperation();
}, { custom: 'metadata' });
```

### Performance Sessions

#### `performance.startSession(name, metadata)`
Start performance monitoring session.

```javascript
import { performance } from 'trustformers';

const sessionId = performance.startSession('inference_batch', {
  batchSize: 32,
  modelType: 'bert'
});
```

#### `performance.endSession()`
End current performance session.

```javascript
const report = performance.endSession();
console.log('Session report:', report);
```

#### `performance.getReport()`
Get comprehensive performance report.

```javascript
const report = performance.getReport();
console.log('Performance report:', report);
```

### Method Profiling Decorator

#### `@profileMethod(operationName)`
Decorator for profiling class methods.

```javascript
import { profileMethod } from 'trustformers';

class MyModel {
  @profileMethod('forward_pass')
  async forward(inputs) {
    // Method implementation
  }
}
```

## Zero-Copy Operations

### Zero-Copy Tensor Views

#### `createZeroCopyTensor(data, shape, dtype)`
Create zero-copy tensor from JavaScript data.

```javascript
import { createZeroCopyTensor } from 'trustformers';

const data = new Float32Array([1, 2, 3, 4]);
const tensor = createZeroCopyTensor(data, [2, 2], 'f32');

// Get direct view of tensor data (no copy)
const view = tensor.getView('f32');
view[0] = 10; // Modifies underlying tensor data
```

#### `wrapTensor(wasmTensor, metadata)`
Wrap existing WASM tensor in zero-copy view.

```javascript
import { wrapTensor } from 'trustformers';

const view = wrapTensor(wasmTensor, { source: 'model_output' });
```

#### `transferTensor(tensor, targetFormat)`
Efficient tensor data transfer.

```javascript
import { transferTensor } from 'trustformers';

// Convert to JavaScript format
const jsData = transferTensor(tensor, 'js');

// Convert to WebGL format
const webglData = transferTensor(tensor, 'webgl');
```

### Zero-Copy Tensor View Methods

```javascript
const view = createZeroCopyTensor(data, shape);

// Shape and metadata
console.log(view.shape);      // [2, 2]
console.log(view.dtype);      // 'f32'
console.log(view.elementCount); // 4
console.log(view.byteSize);   // 16

// Data access
const f32View = view.getView('f32');
const i32View = view.getView('i32');

// Operations (create new views)
const sliced = view.slice([0, 0], [1, 2]);
const reshaped = view.reshape([1, 4]);
const transposed = view.transpose();

// Data manipulation
view.setData(newData, offset);
view.fill(value, start, end);

// Statistics
const stats = view.getStats();
console.log(stats); // { min, max, sum, mean, length }

// Cleanup
view.dispose();
```

## WebGL Backend

### WebGL Backend Creation

#### `createWebGLBackend(canvas)`
Create and initialize WebGL backend.

```javascript
import { createWebGLBackend } from 'trustformers';

const backend = await createWebGLBackend();
console.log('WebGL info:', backend.getInfo());
```

### WebGL Operations

```javascript
// Matrix multiplication
const result = await backend.matmul(tensorA, tensorB);

// Element-wise operations
const sum = await backend.elementWise(a, b, 'add');
const product = await backend.elementWise(a, b, 'mul');

// Activation functions
const relu_out = await backend.activation(input, 'relu');
const sigmoid_out = await backend.activation(input, 'sigmoid');

// Check support
if (backend.isSupported()) {
  console.log('WebGL backend ready');
}

// Cleanup
backend.dispose();
```

## Model Operations

### Model Loading

#### `Pipeline.fromPretrained(task, modelName)`
Load pipeline from pretrained model.

```javascript
const pipeline = await Pipeline.fromPretrained('text-generation', 'gpt2');
```

### Model Inference

#### Basic Forward Pass

```javascript
const outputs = await model.forward(inputs);
```

#### Async Inference with Options

```javascript
const outputs = await async_utils.runInference(model, inputs, {
  autoCleanup: true,
  timeout: 30000
});
```

### Pipeline Operations

#### Text Generation Pipeline

```javascript
const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');
const pipeline = Pipeline.textGeneration(model, tokenizer, {
  max_length: 100,
  temperature: 0.8
});

const output = await pipeline.generate("Hello world");
```

#### Text Classification Pipeline

```javascript
const pipeline = Pipeline.textClassification(model, tokenizer, ['positive', 'negative']);
const result = await pipeline.classify("This is great!");
```

#### Question Answering Pipeline

```javascript
const pipeline = Pipeline.questionAnswering(model, tokenizer);
const answer = await pipeline.answer({
  question: "What is AI?",
  context: "Artificial Intelligence (AI) is..."
});
```

## Tensor Operations

### Basic Operations

#### Arithmetic Operations

```javascript
// Element-wise operations
const sum = tensor_ops.add(a, b);
const diff = tensor_ops.sub(a, b);
const product = tensor_ops.mul(a, b);
const quotient = tensor_ops.div(a, b);

// Matrix multiplication
const result = tensor_ops.matmul(a, b);

// Scalar operations
const scaled = tensor_ops.mulScalar(tensor, 2.0);
const shifted = tensor_ops.addScalar(tensor, 1.0);
```

#### Shape Operations

```javascript
// Reshape
const reshaped = tensor_ops.reshape(tensor, [4, 2]);

// Transpose
const transposed = tensor_ops.transpose(tensor);
const custom_transpose = tensor_ops.transpose(tensor, [1, 0, 2]);

// Squeeze/Unsqueeze
const squeezed = tensor_ops.squeeze(tensor);
const unsqueezed = tensor_ops.unsqueeze(tensor, 1);
```

#### Slicing and Indexing

```javascript
// Slice tensor
const sliced = tensor_ops.slice(tensor, [0, 0], [2, 2], 1);

// Index select
const selected = tensor_ops.indexSelect(tensor, 0, [0, 2, 4]);
```

#### Reduction Operations

```javascript
// Sum
const total = tensor_ops.sum(tensor);
const dim_sum = tensor_ops.sum(tensor, 1, true);

// Mean
const avg = tensor_ops.mean(tensor);
const dim_avg = tensor_ops.mean(tensor, 0, false);

// Min/Max
const minimum = tensor_ops.min(tensor);
const maximum = tensor_ops.max(tensor);
```

#### Comparison Operations

```javascript
const equal = tensor_ops.eq(a, b);
const greater = tensor_ops.gt(a, b);
const lesser = tensor_ops.lt(a, b);
```

#### Concatenation and Stacking

```javascript
// Concatenate tensors
const concatenated = tensor_ops.cat([tensor1, tensor2, tensor3], 0);

// Stack tensors
const stacked = tensor_ops.stack([tensor1, tensor2], 1);
```

#### Mathematical Functions

```javascript
const exp_tensor = tensor_ops.exp(tensor);
const log_tensor = tensor_ops.log(tensor);
const sqrt_tensor = tensor_ops.sqrt(tensor);
const pow_tensor = tensor_ops.pow(tensor, 2);
const abs_tensor = tensor_ops.abs(tensor);
```

#### Normalization

```javascript
// Layer normalization
const layer_norm = tensor_ops.layerNorm(tensor, [128], 1e-5);

// Batch normalization
const batch_norm = tensor_ops.batchNorm(tensor, running_mean, running_var, weight, bias, 1e-5);
```

### Activation Functions

```javascript
// Basic activations
const relu_out = activations.relu(tensor);
const leaky_relu_out = activations.leakyRelu(tensor, 0.01);
const gelu_out = activations.gelu(tensor);
const swish_out = activations.swish(tensor);

// Probability activations
const sigmoid_out = activations.sigmoid(tensor);
const tanh_out = activations.tanh(tensor);
const softmax_out = activations.softmax(tensor, -1);
const log_softmax_out = activations.logSoftmax(tensor, -1);
```

### Advanced Tensor Utilities

#### Typed Array Creation

```javascript
// From typed arrays
const tensor = tensor_utils.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2]);

// From nested arrays
const tensor = tensor_utils.fromNestedArray([[1, 2], [3, 4]]);

// Random tensors with distributions
const normal_tensor = tensor_utils.random([10, 10], 'normal', { mean: 0, std: 1 });
const uniform_tensor = tensor_utils.random([5, 5], 'uniform', { low: 0, high: 1 });
```

## Streaming

### Streaming Text Generation

```javascript
// Async generator for streaming
for await (const chunk of streaming.textGeneration(model, tokenizer, config)) {
  console.log('Generated:', chunk.text);
  if (chunk.done) break;
}
```

### Streaming Tokenization

```javascript
// Stream tokenization
for await (const token of streaming.tokenize(tokenizer, longText)) {
  console.log('Token:', token);
}
```

### Async Batch Processing

```javascript
// Process tensors in batches
const results = await async_utils.processBatch(
  tensors,
  async (tensor) => await model.forward(tensor),
  32 // batch size
);
```

## WebGPU Support

### WebGPU Status

```javascript
// Check WebGPU availability
if (webgpu.isAvailable()) {
  console.log('WebGPU is available');
  
  // Get device information
  const deviceInfo = await webgpu.getDeviceInfo();
  console.log('Device:', deviceInfo);
  
  // Create WebGPU operations
  const ops = webgpu.createOps();
  const result = await ops.matmul(a, b);
}

// Get status
const status = webgpu.getStatus();
console.log('WebGPU status:', status);
```

## Error Handling

### Common Error Patterns

```javascript
try {
  await initialize();
  const model = createModel('bert_base');
  const result = await model.forward(inputs);
} catch (error) {
  if (error.message.includes('not initialized')) {
    console.error('TrustformeRS not initialized');
  } else if (error.message.includes('shape mismatch')) {
    console.error('Tensor shape error:', error.message);
  } else {
    console.error('Unexpected error:', error);
  }
}
```

### Memory Errors

```javascript
try {
  const largeTensor = zeros([10000, 10000]);
} catch (error) {
  if (error.message.includes('Memory limit exceeded')) {
    console.error('Out of memory. Try smaller tensors or enable memory pooling.');
    performance.cleanup(); // Trigger cleanup
  }
}
```

### Performance Warnings

```javascript
import { getProfiler } from 'trustformers';

const profiler = getProfiler({
  warningThresholds: {
    operationTime: 1000, // 1 second
    memoryUsage: 512 * 1024 * 1024 // 512MB
  }
});

// Warnings will be logged automatically for slow operations
```

## Examples

### Complete Inference Example

```javascript
import { 
  initializeEnhanced, 
  createModel, 
  createTokenizer,
  enhanced_inference,
  performance 
} from 'trustformers';

async function runInference() {
  // Initialize with all optimizations
  const capabilities = await initializeEnhanced({
    enableWebGL: true,
    enableMemoryPool: true,
    enableProfiling: true,
    enableZeroCopy: true
  });
  
  // Start performance session
  const sessionId = performance.startSession('bert_inference');
  
  try {
    // Create model and tokenizer
    const model = createModel('bert_base');
    const tokenizer = createTokenizer('wordpiece');
    
    // Prepare input
    const text = "Hello, this is a test sentence.";
    const tokens = tokenizer.encode(text);
    const inputs = { input_ids: tokens };
    
    // Run optimized inference
    const outputs = await enhanced_inference.runInference(model, inputs, {
      profile: true,
      useMemoryPool: true,
      autoCleanup: true,
      backend: 'auto'
    });
    
    console.log('Inference results:', outputs);
    
  } finally {
    // End session and get report
    const report = performance.endSession();
    console.log('Performance report:', report);
  }
}

runInference().catch(console.error);
```

### Memory-Optimized Batch Processing

```javascript
import { 
  initializeEnhanced,
  enhanced_tensor_utils,
  enhanced_inference,
  getMemoryManager 
} from 'trustformers';

async function batchProcessing() {
  await initializeEnhanced({ enableMemoryPool: true });
  
  const model = createModel('gpt2_base');
  const memoryManager = getMemoryManager();
  
  // Create batch inputs using memory pool
  const batchInputs = [];
  for (let i = 0; i < 100; i++) {
    const input = enhanced_tensor_utils.createTensor(
      generateRandomData(), 
      [1, 512], 
      { useMemoryPool: true }
    );
    batchInputs.push(input);
  }
  
  try {
    // Process in batches with automatic memory management
    const results = await enhanced_inference.batchInference(model, batchInputs, {
      batchSize: 16,
      profile: true
    });
    
    console.log('Processed', results.length, 'inputs');
    console.log('Memory stats:', memoryManager.getStats());
    
  } finally {
    // Clean up all tensors
    batchInputs.forEach(tensor => enhanced_tensor_utils.releaseTensor(tensor));
    memoryManager.cleanup();
  }
}
```

### Zero-Copy Operations Example

```javascript
import { 
  initializeEnhanced,
  createZeroCopyTensor,
  transferTensor,
  enhanced_tensor_ops 
} from 'trustformers';

async function zeroCopyExample() {
  await initializeEnhanced({ enableZeroCopy: true });
  
  // Create zero-copy tensor from existing data
  const data = new Float32Array(1000 * 1000); // 1M elements
  data.fill(1.0);
  
  const tensor = createZeroCopyTensor(data, [1000, 1000], 'f32');
  
  // Direct view access (no copy)
  const view = tensor.getView('f32');
  console.log('First element:', view[0]); // 1.0
  
  // Modify data directly
  view[0] = 42.0;
  console.log('Modified:', view[0]); // 42.0
  
  // Efficient operations
  const result = await enhanced_tensor_ops.matmul(tensor, tensor, {
    backend: 'webgl',
    profile: true
  });
  
  // Transfer to different formats efficiently
  const webglData = transferTensor(result, 'webgl');
  const jsArray = transferTensor(result, 'js');
  
  console.log('Result shape:', result.shape);
  
  // Cleanup
  tensor.dispose();
  result.dispose();
}
```

### WebGL Backend Example

```javascript
import { 
  initializeEnhanced,
  createWebGLBackend,
  enhanced_tensor_utils 
} from 'trustformers';

async function webglExample() {
  const capabilities = await initializeEnhanced({ enableWebGL: true });
  
  if (!capabilities.webgl) {
    console.error('WebGL not available');
    return;
  }
  
  // Get WebGL backend
  const backend = await createWebGLBackend();
  console.log('WebGL info:', backend.getInfo());
  
  // Create test tensors
  const a = enhanced_tensor_utils.createTensor(
    Array.from({length: 100}, () => Math.random()), 
    [10, 10]
  );
  const b = enhanced_tensor_utils.createTensor(
    Array.from({length: 100}, () => Math.random()), 
    [10, 10]
  );
  
  // WebGL matrix multiplication
  const result = await backend.matmul(a, b);
  console.log('WebGL matmul result shape:', result.shape);
  
  // WebGL element-wise operations
  const sum = await backend.elementWise(a, b, 'add');
  const activated = await backend.activation(sum, 'relu');
  
  console.log('WebGL operations completed');
  
  // Cleanup
  backend.dispose();
}
```

## Browser Compatibility

- **WebAssembly**: Chrome 57+, Firefox 52+, Safari 11+, Edge 16+
- **WebGL**: Chrome 9+, Firefox 4+, Safari 5.1+, Edge 12+
- **WebGPU**: Chrome 113+, Firefox (experimental), Safari (experimental)
- **ES Modules**: Chrome 61+, Firefox 60+, Safari 10.1+, Edge 16+

## Performance Tips

1. **Use Enhanced APIs**: The enhanced APIs automatically select the best backend and provide memory optimizations.

2. **Enable Memory Pooling**: Reduces allocation overhead for frequently created tensors.

3. **Use Zero-Copy When Possible**: Eliminates memory copying for large tensors.

4. **Profile Your Code**: Use the built-in profiler to identify bottlenecks.

5. **Batch Operations**: Process multiple inputs together for better throughput.

6. **Choose Appropriate Backends**: WebGPU > WebGL > WASM for compute-intensive operations.

7. **Clean Up Resources**: Always dispose tensors and release memory when done.

8. **Monitor Memory Usage**: Use memory statistics to prevent out-of-memory errors.

## TypeScript Support

TrustformeRS includes comprehensive TypeScript definitions:

```typescript
import { initialize, tensor, createModel, Tensor, Model } from 'trustformers';

const t: Tensor = tensor([1, 2, 3, 4], [2, 2]);
const model: Model = createModel('bert_base');
const outputs: Tensor = await model.forward(t);
```

See `src/index.d.ts` for complete type definitions.