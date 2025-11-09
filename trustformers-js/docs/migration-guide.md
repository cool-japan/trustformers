# Migration Guide to TrustformeRS

This guide helps you migrate from other popular machine learning libraries to TrustformeRS JavaScript API.

## Table of Contents

1. [From TensorFlow.js](#from-tensorflowjs)
2. [From PyTorch (Python to JS)](#from-pytorch-python-to-js)
3. [From Hugging Face Transformers.js](#from-hugging-face-transformersjs)
4. [From ML5.js](#from-ml5js)
5. [From Brain.js](#from-brainjs)
6. [Common Migration Patterns](#common-migration-patterns)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## From TensorFlow.js

### Basic Setup Migration

**TensorFlow.js:**
```javascript
import * as tf from '@tensorflow/tfjs';
await tf.ready();
```

**TrustformeRS:**
```javascript
import { initializeEnhanced } from 'trustformers';
const capabilities = await initializeEnhanced({
  enableWebGL: true,
  enableMemoryPool: true,
  enableProfiling: true
});
```

### Tensor Operations Migration

**TensorFlow.js:**
```javascript
// Create tensors
const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor2d([[5, 6], [7, 8]]);

// Operations
const sum = tf.add(a, b);
const product = tf.matMul(a, b);
const relu = tf.relu(a);

// Cleanup
a.dispose();
b.dispose();
sum.dispose();
```

**TrustformeRS:**
```javascript
import { enhanced_tensor_utils, enhanced_tensor_ops } from 'trustformers';

// Create tensors (with automatic memory management)
const a = enhanced_tensor_utils.createTensor([[1, 2], [3, 4]], [2, 2]);
const b = enhanced_tensor_utils.createTensor([[5, 6], [7, 8]], [2, 2]);

// Operations (with automatic backend selection)
const sum = await enhanced_tensor_ops.elementWise(a, b, 'add');
const product = await enhanced_tensor_ops.matmul(a, b);
const relu = await enhanced_tensor_ops.activation(a, 'relu');

// Automatic cleanup with memory pooling
enhanced_tensor_utils.releaseTensor(a);
enhanced_tensor_utils.releaseTensor(b);
enhanced_tensor_utils.releaseTensor(sum);
```

### Model Loading Migration

**TensorFlow.js:**
```javascript
const model = await tf.loadLayersModel('/path/to/model.json');
const prediction = model.predict(inputTensor);
```

**TrustformeRS:**
```javascript
import { createModel, enhanced_inference } from 'trustformers';

const model = createModel('bert_base'); // or load from config
const prediction = await enhanced_inference.runInference(model, inputTensor, {
  profile: true,
  useMemoryPool: true,
  autoCleanup: true
});
```

### Data Pipeline Migration

**TensorFlow.js:**
```javascript
const dataset = tf.data.array(data)
  .batch(32)
  .map(x => tf.cast(x, 'float32'));

await dataset.forEachAsync(batch => {
  const prediction = model.predict(batch);
  // Process prediction
  prediction.dispose();
});
```

**TrustformeRS:**
```javascript
import { enhanced_inference, async_utils } from 'trustformers';

const batches = [];
for (let i = 0; i < data.length; i += 32) {
  batches.push(data.slice(i, i + 32));
}

const results = await enhanced_inference.batchInference(model, batches, {
  batchSize: 32,
  profile: true
});
```

## From PyTorch (Python to JS)

### Tensor Creation Migration

**PyTorch (Python):**
```python
import torch

# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.zeros(3, 3)
z = torch.randn(10, 10)

# Operations
result = torch.matmul(x, x.T)
activated = torch.relu(result)
```

**TrustformeRS:**
```javascript
import { enhanced_tensor_utils, enhanced_tensor_ops } from 'trustformers';

// Create tensors
const x = enhanced_tensor_utils.createTensor([[1, 2], [3, 4]], [2, 2], { dtype: 'f32' });
const y = enhanced_tensor_utils.zeros([3, 3]);
const z = tensor_utils.random([10, 10], 'normal');

// Operations
const xT = x.transpose();
const result = await enhanced_tensor_ops.matmul(x, xT);
const activated = await enhanced_tensor_ops.activation(result, 'relu');
```

### Model Definition Migration

**PyTorch (Python):**
```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)

model = SimpleModel()
output = model(input_tensor)
```

**TrustformeRS:**
```javascript
import { createModelConfig, createModel } from 'trustformers';

// Define model configuration
const config = createModelConfig('custom');
config.layers = [
  { type: 'linear', input_size: 784, output_size: 10 },
  { type: 'relu' }
];

const model = createModel(config);
const output = await model.forward(inputTensor);
```

### Training Migration

**PyTorch (Python):**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch.data)
    loss = criterion(output, batch.labels)
    loss.backward()
    optimizer.step()
```

**TrustformeRS:**
```javascript
// Note: TrustformeRS focuses on inference. For training, consider:
// 1. Training in Python/PyTorch and exporting to TrustformeRS
// 2. Using TrustformeRS for inference in production

import { createModel } from 'trustformers';

// Load pre-trained model
const model = createModel('bert_base');
// or load from exported PyTorch model
const model = await loadFromPyTorch('/path/to/model');

// Use for inference
const predictions = await model.forward(batch);
```

## From Hugging Face Transformers.js

### Model Loading Migration

**Transformers.js:**
```javascript
import { pipeline } from '@xenova/transformers';

const classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
const result = await classifier('I love TrustformeRS!');
```

**TrustformeRS:**
```javascript
import { Pipeline, createModel, createTokenizer } from 'trustformers';

// Method 1: Use pipeline factory
const classifier = await Pipeline.fromPretrained('text-classification', 'distilbert-base-uncased');
const result = await classifier.classify('I love TrustformeRS!');

// Method 2: Manual setup with more control
const model = createModel('distilbert_base');
const tokenizer = createTokenizer('wordpiece');
const pipeline = Pipeline.textClassification(model, tokenizer, ['negative', 'positive']);
const result = await pipeline.classify('I love TrustformeRS!');
```

### Text Generation Migration

**Transformers.js:**
```javascript
const generator = await pipeline('text-generation', 'Xenova/gpt2');
const output = await generator('Hello, my name is', {
  max_length: 50,
  temperature: 0.8
});
```

**TrustformeRS:**
```javascript
import { Pipeline, createModel, createTokenizer, streaming } from 'trustformers';

const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');
const generator = Pipeline.textGeneration(model, tokenizer, {
  max_length: 50,
  temperature: 0.8
});

// Standard generation
const output = await generator.generate('Hello, my name is');

// Streaming generation
for await (const chunk of streaming.textGeneration(model, tokenizer, {
  prompt: 'Hello, my name is',
  max_length: 50
})) {
  console.log(chunk.text);
}
```

### Feature Extraction Migration

**Transformers.js:**
```javascript
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
const embeddings = await extractor('Hello world');
```

**TrustformeRS:**
```javascript
import { createModel, createTokenizer } from 'trustformers';

const model = createModel('bert_base');
const tokenizer = createTokenizer('wordpiece');

// Extract embeddings
const tokens = tokenizer.encode('Hello world');
const outputs = await model.forward({ input_ids: tokens });
const embeddings = outputs.last_hidden_state;
```

## From ML5.js

### Image Classification Migration

**ML5.js:**
```javascript
const classifier = ml5.imageClassifier('MobileNet', modelReady);

function modelReady() {
  classifier.classify(img, gotResult);
}

function gotResult(error, results) {
  console.log(results);
}
```

**TrustformeRS:**
```javascript
import { createModel, enhanced_inference } from 'trustformers';

const model = createModel('mobilenet_v2');

async function classifyImage(imageData) {
  try {
    const results = await enhanced_inference.runInference(model, imageData, {
      profile: true,
      autoCleanup: true
    });
    console.log(results);
    return results;
  } catch (error) {
    console.error('Classification error:', error);
  }
}

// Usage
const results = await classifyImage(img);
```

### Pose Detection Migration

**ML5.js:**
```javascript
const poseNet = ml5.poseNet(video, modelReady);
poseNet.on('pose', gotPoses);

function gotPoses(poses) {
  console.log(poses);
}
```

**TrustformeRS:**
```javascript
import { createModel, enhanced_inference } from 'trustformers';

const poseModel = createModel('posenet');

async function detectPoses(videoFrame) {
  const poses = await enhanced_inference.runInference(poseModel, videoFrame, {
    backend: 'webgl', // Use GPU acceleration
    profile: true
  });
  return poses;
}

// Continuous detection
async function startPoseDetection(video) {
  const stream = video.captureStream();
  const reader = stream.getVideoTracks()[0].getReader();
  
  while (true) {
    const { value: frame } = await reader.read();
    const poses = await detectPoses(frame);
    console.log(poses);
  }
}
```

## From Brain.js

### Neural Network Migration

**Brain.js:**
```javascript
const net = new brain.NeuralNetwork();

net.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
]);

const output = net.run([1, 0]);
```

**TrustformeRS:**
```javascript
import { createModelConfig, createModel } from 'trustformers';

// Define a simple neural network
const config = createModelConfig('feedforward');
config.layers = [
  { type: 'linear', input_size: 2, output_size: 3 },
  { type: 'relu' },
  { type: 'linear', input_size: 3, output_size: 1 },
  { type: 'sigmoid' }
];

const model = createModel(config);

// Note: For training, you would typically:
// 1. Train in Python/PyTorch and export to TrustformeRS
// 2. Load pre-trained weights

// For inference
const input = tensor([1, 0], [1, 2]);
const output = await model.forward(input);
```

### LSTM Migration

**Brain.js:**
```javascript
const lstm = new brain.recurrent.LSTM();

lstm.train([
  { input: 'hello', output: 'world' },
  { input: 'foo', output: 'bar' }
]);

const output = lstm.run('hello');
```

**TrustformeRS:**
```javascript
import { createModel, createTokenizer } from 'trustformers';

// Use a pre-trained transformer model (more powerful than LSTM)
const model = createModel('gpt2_base');
const tokenizer = createTokenizer('bpe');

async function generateResponse(input) {
  const tokens = tokenizer.encode(input);
  const output = await model.generate(tokens, {
    max_length: 50,
    temperature: 0.8
  });
  return tokenizer.decode(output);
}

const response = await generateResponse('hello');
```

## Common Migration Patterns

### Memory Management

**Old Pattern (Manual Cleanup):**
```javascript
// TensorFlow.js style
const tensor1 = tf.tensor([1, 2, 3]);
const tensor2 = tf.tensor([4, 5, 6]);
const result = tf.add(tensor1, tensor2);

// Manual cleanup required
tensor1.dispose();
tensor2.dispose();
result.dispose();
```

**New Pattern (Automatic Management):**
```javascript
// TrustformeRS style
import { enhanced_tensor_utils, withMemoryManagement } from 'trustformers';

const result = await withMemoryManagement(async () => {
  const tensor1 = enhanced_tensor_utils.createTensor([1, 2, 3], [3]);
  const tensor2 = enhanced_tensor_utils.createTensor([4, 5, 6], [3]);
  return await enhanced_tensor_ops.elementWise(tensor1, tensor2, 'add');
}, [], { autoRelease: true });
```

### Asynchronous Operations

**Old Pattern (Callback-based):**
```javascript
// ML5.js style
classifier.classify(image, (error, results) => {
  if (error) {
    console.error(error);
  } else {
    console.log(results);
  }
});
```

**New Pattern (Promise/Async-Await):**
```javascript
// TrustformeRS style
try {
  const results = await enhanced_inference.runInference(model, image, {
    profile: true,
    timeout: 10000
  });
  console.log(results);
} catch (error) {
  console.error('Inference failed:', error);
}
```

### Batch Processing

**Old Pattern (Loop-based):**
```javascript
// Manual batching
const results = [];
for (const item of data) {
  const result = await model.predict(item);
  results.push(result);
}
```

**New Pattern (Optimized Batching):**
```javascript
// TrustformeRS optimized batching
const results = await enhanced_inference.batchInference(model, data, {
  batchSize: 32,
  profile: true
});
```

## Performance Optimization

### Enable All Performance Features

```javascript
import { initializeEnhanced } from 'trustformers';

// Initialize with all performance optimizations
const capabilities = await initializeEnhanced({
  enableWebGL: true,        // GPU acceleration
  enableMemoryPool: true,   // Memory optimization
  enableProfiling: true,    // Performance monitoring
  enableZeroCopy: true,     // Zero-copy transfers
  
  webglOptions: {
    canvas: document.getElementById('canvas')
  },
  memoryOptions: {
    maxTotalMemory: 2 * 1024 * 1024 * 1024, // 2GB
    maxPoolSize: 200
  },
  profilingOptions: {
    detailed: true,
    autoReport: false
  }
});

console.log('Available optimizations:', capabilities);
```

### Use Performance Monitoring

```javascript
import { performance } from 'trustformers';

// Start performance session
const sessionId = performance.startSession('model_inference');

try {
  // Your model operations
  const result = await enhanced_inference.runInference(model, inputs);
  
  // Get performance metrics
  const report = performance.getReport();
  console.log('Performance metrics:', report);
  
} finally {
  // End session
  const sessionReport = performance.endSession();
  console.log('Session report:', sessionReport);
}
```

### Optimize Memory Usage

```javascript
import { enhanced_tensor_utils, getMemoryManager } from 'trustformers';

// Use memory-managed tensors
const tensor = enhanced_tensor_utils.createTensor(data, shape, {
  useMemoryPool: true,
  zeroCopy: true
});

// Monitor memory usage
const memoryStats = getMemoryManager().getStats();
console.log('Memory efficiency:', memoryStats.memoryEfficiency);

// Force cleanup when needed
performance.cleanup();
```

## Troubleshooting

### Common Migration Issues

#### 1. WebAssembly Loading Errors

**Problem:**
```
Error: Failed to initialize TrustformeRS: WebAssembly module not found
```

**Solution:**
```javascript
await initializeEnhanced({
  wasmPath: './path/to/trustformers_wasm_bg.wasm', // Correct path
  initPanicHook: true
});
```

#### 2. Memory Limit Exceeded

**Problem:**
```
Error: Memory limit exceeded: allocation would exceed 1GB
```

**Solution:**
```javascript
await initializeEnhanced({
  enableMemoryPool: true,
  memoryOptions: {
    maxTotalMemory: 4 * 1024 * 1024 * 1024, // Increase to 4GB
    maxPoolSize: 500
  }
});
```

#### 3. WebGL Not Available

**Problem:**
```
Warning: WebGL backend initialization failed
```

**Solution:**
```javascript
// Check WebGL availability first
if (!window.WebGLRenderingContext) {
  console.warn('WebGL not supported, falling back to CPU');
}

await initializeEnhanced({
  enableWebGL: !!window.WebGLRenderingContext,
  // Fallback to CPU-only mode
});
```

#### 4. Model Loading Errors

**Problem:**
```
Error: Unknown model type: my_custom_model
```

**Solution:**
```javascript
// Use explicit configuration
const config = createModelConfig('bert_base');
config.vocab_size = 30522;
config.hidden_size = 768;
// ... customize config

const model = createModel(config);
```

### Performance Debugging

#### Enable Detailed Profiling

```javascript
import { getProfiler } from 'trustformers';

const profiler = getProfiler({
  enabled: true,
  detailed: true,
  warningThresholds: {
    operationTime: 500,  // Warn for ops > 500ms
    memoryUsage: 100 * 1024 * 1024  // Warn for > 100MB
  }
});

// Operations will be automatically profiled
const result = await enhanced_tensor_ops.matmul(a, b);

// Get detailed report
const report = profiler.generateReport();
console.log('Optimization recommendations:', report.recommendations);
```

#### Memory Leak Detection

```javascript
import { getMemoryManager } from 'trustformers';

const manager = getMemoryManager();

// Monitor memory usage over time
setInterval(() => {
  const stats = manager.getStats();
  console.log('Memory usage:', stats.tensor.currentMemoryUsage);
  
  if (stats.tensor.memoryEfficiency < 0.7) {
    console.warn('Low memory efficiency, consider cleanup');
    manager.cleanup();
  }
}, 5000);
```

### Migration Checklist

- [ ] Replace tensor creation with `enhanced_tensor_utils`
- [ ] Update operations to use `enhanced_tensor_ops`
- [ ] Enable performance optimizations in initialization
- [ ] Replace manual memory management with automatic pooling
- [ ] Update callback-based code to async/await
- [ ] Add error handling for async operations
- [ ] Enable performance profiling for optimization
- [ ] Test with different backend configurations
- [ ] Verify memory usage is within limits
- [ ] Add performance monitoring to production code

### Getting Help

1. **Check the API Documentation**: Complete reference at `docs/api-reference.md`
2. **Enable Debug Logging**: Set profiling options to get detailed logs
3. **Use Performance Profiler**: Identify bottlenecks and optimization opportunities
4. **Check System Compatibility**: Verify browser support for required features
5. **Community Support**: Check GitHub issues for similar migration problems

---

This migration guide should help you successfully transition to TrustformeRS while taking advantage of its performance optimizations and modern JavaScript features.