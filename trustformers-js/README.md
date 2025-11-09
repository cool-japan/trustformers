# TrustformeRS JavaScript API

A modern JavaScript/TypeScript API for the TrustformeRS WebAssembly library, providing comprehensive tensor operations, streaming capabilities, and async processing utilities.

## Features

- 🚀 **Comprehensive Tensor Operations**: 40+ tensor operations including arithmetic, shape manipulation, reductions, and mathematical functions
- 🌊 **Streaming Support**: Real-time text generation and tokenization with async generators
- ⚡ **Async/Await API**: Modern JavaScript patterns with Promise-based operations
- 🔧 **TypeScript Support**: Full TypeScript definitions with comprehensive type safety
- 🎯 **Memory Management**: Automatic cleanup and memory monitoring utilities
- 🖥️ **WebGPU Support**: Hardware acceleration when available
- 📊 **Batch Processing**: Efficient batch processing with configurable batch sizes
- 🛡️ **Error Handling**: Robust error handling and recovery mechanisms
- 🧠 **Multiple Models**: Support for BERT, GPT-2, T5, LLaMA, and Mistral architectures
- 💾 **Memory Efficient**: Optimized for browser environments

## Installation

```bash
npm install trustformers
```

## Quick Start

```javascript
import { initialize, tensor, tensor_ops, activations, createModel, Pipeline } from 'trustformers';

// Initialize the WASM module
await initialize();

// Create and manipulate tensors
const a = tensor([1, 2, 3, 4], [2, 2]);
const b = tensor([5, 6, 7, 8], [2, 2]);

// Perform operations
const sum = tensor_ops.add(a, b);
const product = tensor_ops.matmul(a, b);
const activated = activations.relu(sum);

// Create models and pipelines
const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');
const pipeline = Pipeline.textGeneration(model, tokenizer);

// Generate text
const result = await pipeline.generate("The future of AI is");
```

## Core API

### Initialization

```javascript
await initialize({
  wasmPath: './trustformers_wasm_bg.wasm',  // Optional custom path
  initPanicHook: true  // Enable better error messages
});
```

### Tensor Operations

#### Basic Tensor Creation

```javascript
import { tensor, zeros, ones, randn, tensor_utils } from 'trustformers';

// Create tensors
const t1 = tensor([1, 2, 3, 4], [2, 2]);
const t2 = zeros([3, 3]);
const t3 = ones([2, 4]);
const t4 = randn([5, 5]);

// Create from nested arrays
const nested = tensor_utils.fromNestedArray([[1, 2], [3, 4]]);

// Create from typed arrays
const float_data = new Float32Array([1, 2, 3, 4]);
const from_typed = tensor_utils.fromTypedArray(float_data, [2, 2]);

// Create random tensors with distributions
const normal = tensor_utils.random([3, 3], 'normal', { mean: 0, std: 1 });
const uniform = tensor_utils.random([3, 3], 'uniform', { low: -1, high: 1 });
```

#### Comprehensive Tensor Operations

```javascript
import { tensor_ops } from 'trustformers';

// Arithmetic operations
const sum = tensor_ops.add(a, b);
const diff = tensor_ops.sub(a, b);
const product = tensor_ops.mul(a, b);
const quotient = tensor_ops.div(a, b);
const matrix_mult = tensor_ops.matmul(a, b);

// Scalar operations
const scaled = tensor_ops.mulScalar(a, 2.5);
const shifted = tensor_ops.addScalar(a, 10);

// Shape operations
const reshaped = tensor_ops.reshape(a, [4, 2]);
const transposed = tensor_ops.transpose(a, [1, 0]);
const squeezed = tensor_ops.squeeze(a, 0);
const unsqueezed = tensor_ops.unsqueeze(a, 1);

// Slicing and indexing
const sliced = tensor_ops.slice(a, [0, 1], [2, 3]);
const selected = tensor_ops.indexSelect(a, 0, [0, 2]);

// Reduction operations
const sum_all = tensor_ops.sum(a);
const mean_dim = tensor_ops.mean(a, 1, true);
const max_vals = tensor_ops.max(a, 0);
const min_vals = tensor_ops.min(a, 1);

// Mathematical functions
const exp_result = tensor_ops.exp(a);
const log_result = tensor_ops.log(a);
const sqrt_result = tensor_ops.sqrt(a);
const pow_result = tensor_ops.pow(a, 2);
const abs_result = tensor_ops.abs(a);

// Concatenation and stacking
const concatenated = tensor_ops.cat([a, b, c], 0);
const stacked = tensor_ops.stack([a, b, c], 1);
```

### Activation Functions

```javascript
import { activations } from 'trustformers';

const input = tensor([-2, -1, 0, 1, 2], [5]);

// Various activation functions
const relu = activations.relu(input);
const leaky_relu = activations.leakyRelu(input, 0.1);
const gelu = activations.gelu(input);
const swish = activations.swish(input);
const sigmoid = activations.sigmoid(input);
const tanh = activations.tanh(input);
const softmax = activations.softmax(input, 0);
const log_softmax = activations.logSoftmax(input, 0);
```

### Streaming Operations

```javascript
import { streaming } from 'trustformers';

// Stream text generation
const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');

for await (const chunk of streaming.textGeneration(model, tokenizer, {
  max_length: 100,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.9
})) {
  process.stdout.write(chunk);
}

// Stream tokenization
const text = "Natural language processing is fascinating.";
const tokens = [];
for await (const token of streaming.tokenize(tokenizer, text)) {
  tokens.push(token);
}
```

### Async Processing

```javascript
import { async_utils } from 'trustformers';

// Batch processing
const tensors = Array.from({ length: 100 }, () => randn([10, 10]));

const results = await async_utils.processBatch(
  tensors,
  async (tensor) => {
    // Simulate async processing
    await new Promise(resolve => setTimeout(resolve, 10));
    return tensor_ops.mean(tensor);
  },
  10 // batch size
);

// Inference with timeout and cleanup
const result = await async_utils.runInference(model, input, {
  autoCleanup: true,
  timeout: 5000
});
```

## Pipeline API

### Text Generation

```javascript
const pipeline = Pipeline.textGeneration(model, tokenizer, {
  max_length: 100,
  temperature: 0.8,
  top_p: 0.9,
  do_sample: true
});

const generated = await pipeline.generate("Once upon a time");
```

### Text Classification

```javascript
const pipeline = Pipeline.textClassification(
  model, 
  tokenizer,
  ['positive', 'negative', 'neutral']
);

const result = await pipeline.classify("This product is amazing!");
// { label: 'positive', score: 0.98, all_scores: Float32Array }
```

### Question Answering

```javascript
const pipeline = Pipeline.questionAnswering(model, tokenizer);

const answer = await pipeline.answer(
  "What is the capital of France?",
  "Paris is the capital of France. It is known for the Eiffel Tower."
);
// { answer: 'Paris', start: 0, end: 5, score: 0.95 }
```

## Memory Management

```javascript
import { memory } from 'trustformers';

// Get memory statistics
const stats = memory.getStats();
console.log(`Used: ${stats.used_mb} MB / ${stats.limit_mb} MB`);

// Monitor memory during processing
console.log('Memory before:', memory.getStats());

// Create and process tensors
const tensors = Array.from({ length: 50 }, () => randn([100, 100]));

console.log('Memory at peak:', memory.getStats());

// Clean up
tensors.forEach(tensor => tensor.free());

console.log('Memory after cleanup:', memory.getStats());
```

## WebGPU Support

```javascript
import { webgpu } from 'trustformers';

// Check WebGPU availability
if (webgpu.isAvailable()) {
  console.log('WebGPU is available');
  console.log('Status:', webgpu.getStatus());
  
  // Get device information
  const deviceInfo = await webgpu.getDeviceInfo();
  console.log('Device:', deviceInfo);
  
  // Create WebGPU operations
  const ops = webgpu.createOps();
} else {
  console.log('WebGPU is not available');
}
```

## Examples

### Basic Tensor Operations

```javascript
import { initialize, tensor, tensor_ops, activations } from 'trustformers';

await initialize();

// Create tensors
const a = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
const b = tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);

// Perform operations
const sum = tensor_ops.add(a, b);
const product = tensor_ops.matmul(a, tensor_ops.transpose(b));
const activated = activations.relu(sum);

console.log('Sum:', sum.toString());
console.log('Product:', product.toString());
console.log('Activated:', activated.toString());
```

### Streaming Text Generation

```javascript
import { initialize, createModel, createTokenizer, streaming } from 'trustformers';

await initialize();

const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');

// Stream text generation
for await (const chunk of streaming.textGeneration(model, tokenizer, {
  max_length: 100,
  temperature: 0.8
})) {
  process.stdout.write(chunk);
}
```

### Async Batch Processing

```javascript
import { initialize, async_utils, tensor_utils } from 'trustformers';

await initialize();

const tensors = Array.from({ length: 10 }, () => 
  tensor_utils.random([100, 100], 'normal')
);

// Process in batches
const results = await async_utils.processBatch(
  tensors,
  async (tensor) => {
    // Simulate async processing
    await new Promise(resolve => setTimeout(resolve, 10));
    return tensor_ops.mean(tensor);
  },
  3 // batch size
);

console.log('Results:', results);
```

## Running Examples

```bash
# Run basic tensor operations
npm run example:tensor

# Run enhanced tensor operations demo
npm run example:enhanced

# Run streaming and async demo
npm run example:streaming

# Run all examples
npm run examples:all
```

## TypeScript Support

The library includes comprehensive TypeScript definitions:

```typescript
import { 
  Tensor, 
  tensor_ops, 
  activations, 
  async_utils,
  streaming,
  Model,
  Tokenizer,
  GenerationConfig
} from 'trustformers';

// Full type safety
const a: Tensor = tensor([1, 2, 3, 4], [2, 2]);
const b: Tensor = tensor([5, 6, 7, 8], [2, 2]);

// Typed operations
const sum: Tensor = tensor_ops.add(a, b);
const relu: Tensor = activations.relu(sum);

// Async operations with proper typing
async function processData(): Promise<Tensor> {
  const result = await async_utils.runInference(model, input, {
    autoCleanup: true,
    timeout: 5000
  });
  return result;
}

// Streaming with async generators
async function* generateText(): AsyncGenerator<string, void, unknown> {
  yield* streaming.textGeneration(model, tokenizer, {
    max_length: 100,
    temperature: 0.8
  });
}
```

## Advanced Features

### Custom Tensor Distributions

```javascript
import { tensor_utils } from 'trustformers';

// Normal distribution
const normal = tensor_utils.random([3, 3], 'normal', { 
  mean: 0, 
  std: 1 
});

// Uniform distribution
const uniform = tensor_utils.random([3, 3], 'uniform', { 
  low: -1, 
  high: 1 
});

// Binomial distribution
const binomial = tensor_utils.random([3, 3], 'binomial', { 
  n: 10, 
  p: 0.5 
});
```

### Performance Monitoring

```javascript
import { utils, memory } from 'trustformers';

// Performance timing
const timer = utils.timer('operation');

// Perform operations
const result = tensor_ops.matmul(a, b);

// Log timing
timer.log_elapsed();

// Memory monitoring
const memoryBefore = memory.getStats();
// ... perform memory-intensive operations ...
const memoryAfter = memory.getStats();

console.log('Memory used:', memoryAfter.used - memoryBefore.used);
```

## Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import { initialize, tensor, tensor_ops, activations } from './trustformers.js';
    
    async function run() {
      await initialize({ wasmPath: './trustformers_wasm_bg.wasm' });
      
      const a = tensor([1, 2, 3, 4], [2, 2]);
      const b = tensor([5, 6, 7, 8], [2, 2]);
      
      const sum = tensor_ops.add(a, b);
      const activated = activations.relu(sum);
      
      console.log('Result:', activated.toString());
    }
    
    run().catch(console.error);
  </script>
</head>
<body>
  <h1>TrustformeRS Browser Demo</h1>
  <p>Check the console for results!</p>
</body>
</html>
```

## Performance Tips

1. **Use batch processing** for multiple operations to avoid blocking the main thread
2. **Enable WebGPU** when available for hardware acceleration
3. **Monitor memory usage** and call `free()` on tensors when done
4. **Use appropriate batch sizes** (typically 8-32) for optimal performance
5. **Consider streaming** for real-time applications
6. **Reuse models** - create models once and reuse them for multiple inferences
7. **Use tensor_ops** for efficient operations instead of individual tensor methods

## Browser Support

- Chrome 80+
- Firefox 72+
- Safari 14+
- Edge 80+

Requires WebAssembly support and modern JavaScript features (ES2020+).

## API Reference

For complete API documentation, see the TypeScript definitions in `src/index.d.ts`.

## Contributing

Contributions are welcome! Please see the main TrustformeRS repository for contribution guidelines.

## License

MIT License - see the LICENSE file for details.