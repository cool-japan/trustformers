# TrustformeRS Node.js Bindings

High-performance transformer library with C API bindings for Node.js. Provides TypeScript-first interface with full type safety and memory management.

## Features

- **TypeScript-first**: Complete type definitions and type-safe APIs
- **High Performance**: Direct FFI bindings to optimized C library
- **Memory Management**: Automatic resource cleanup and leak detection
- **Cross-platform**: Supports Windows, macOS, and Linux (x64 and ARM64)
- **Multiple Tasks**: Text generation, classification, question answering, and more
- **Streaming Support**: Real-time inference with streaming callbacks
- **Batch Processing**: Efficient batch operations for high throughput
- **Performance Monitoring**: Built-in profiling and optimization hints
- **Conversation Management**: Stateful conversation handling
- **GPU Support**: CUDA and ROCm acceleration when available

## Installation

```bash
npm install @trustformers/c-bindings
```

### Prerequisites

- Node.js 16.0.0 or higher
- TrustformeRS C library (built and available in system path)

## Quick Start

### Basic Usage

```javascript
const { trustformers, createTextGenerationPipeline, quickSetup } = require('@trustformers/c-bindings');

async function main() {
  // Quick setup with optimization
  quickSetup({
    logLevel: 3,
    memoryLimitMb: 1024,
    enableProfiling: true,
    optimizationLevel: 2
  });

  // Create a text generation pipeline
  const pipeline = await createTextGenerationPipeline('gpt2', {
    device: 'auto',
    maxLength: 512
  });

  // Generate text
  const result = await pipeline.run("The future of AI is", {
    maxLength: 100,
    temperature: 0.8,
    topK: 50
  });

  console.log('Generated:', result.text);

  // Cleanup
  pipeline.dispose();
}

main().catch(console.error);
```

### TypeScript Usage

```typescript
import {
  trustformers,
  createTextGenerationPipeline,
  GenerationConfig,
  PipelineConfig
} from '@trustformers/c-bindings';

async function generateText(): Promise<void> {
  const config: PipelineConfig = {
    task: 'text-generation',
    model: {
      modelPath: 'gpt2',
      device: 'auto',
      quantization: 'fp16'
    }
  };

  const pipeline = await createTextGenerationPipeline(config.model.modelPath, {
    device: config.model.device,
    maxLength: 512
  });

  const generationConfig: GenerationConfig = {
    maxLength: 100,
    temperature: 0.8,
    topK: 50,
    topP: 0.95,
    doSample: true
  };

  const result = await pipeline.run("Hello, world!", generationConfig);
  console.log('Generated:', result.text);

  pipeline.dispose();
}
```

## API Reference

### Core Classes

#### TrustformeRS

Main API class providing access to library functions and utilities.

```typescript
const api = TrustformeRS.getInstance();

// Get version and build info
console.log(api.getVersion());
console.log(api.getBuildInfo());

// Memory management
const memoryUsage = api.getMemoryUsage();
api.memoryCleanup();

// Performance monitoring
const metrics = api.getPerformanceMetrics();
api.startProfiling();
const report = api.stopProfiling();
```

#### Model

Represents a loaded transformer model.

```typescript
import { Model } from '@trustformers/c-bindings';

const model = new Model({
  modelPath: 'path/to/model',
  device: 'auto',
  quantization: 'fp16'
});

// Get model information
const metadata = model.getMetadata();
const config = model.getConfig();

// Validate model
const validation = model.validate();

// Quantize model
model.quantize('int8', 'path/to/output');

// Cleanup
model.dispose();
```

#### Tokenizer

Handles text tokenization and encoding/decoding.

```typescript
import { Tokenizer } from '@trustformers/c-bindings';

const tokenizer = new Tokenizer({
  tokenizerPath: 'path/to/tokenizer.json',
  addSpecialTokens: true,
  maxLength: 512
});

// Encode text
const encoded = tokenizer.encode("Hello, world!", {
  padding: false,
  truncation: true,
  returnAttentionMask: true
});

// Decode tokens
const decoded = tokenizer.decode(encoded.inputIds, {
  skipSpecialTokens: true
});

// Batch operations
const batchEncoded = tokenizer.encodeBatch([
  "First text",
  "Second text"
]);

// Cleanup
tokenizer.dispose();
```

#### Pipeline

High-level interface for inference tasks.

```typescript
import { Pipeline } from '@trustformers/c-bindings';

const pipeline = new Pipeline({
  task: 'text-generation',
  model: 'gpt2',
  device: 'auto'
});

// Single inference
const result = await pipeline.run("Input text", {
  maxLength: 100,
  temperature: 0.8
});

// Batch inference
const results = await pipeline.runBatch([
  "First input",
  "Second input"
]);

// Streaming inference
await pipeline.stream("Input text", {}, (chunk, isComplete) => {
  console.log('Chunk:', chunk);
  if (isComplete) console.log('Complete!');
});

// Conversation management
const conversationId = pipeline.startConversation("System prompt");
pipeline.addConversationTurn(conversationId, 'user', 'Hello!');
const response = await pipeline.generateConversationResponse(conversationId);

// Cleanup
pipeline.dispose();
```

### Convenience Functions

```typescript
import {
  createTextGenerationPipeline,
  createClassificationPipeline,
  createQuestionAnsweringPipeline,
  loadModel,
  loadTokenizer,
  quickSetup,
  benchmark
} from '@trustformers/c-bindings';

// Quick pipeline creation
const textPipeline = await createTextGenerationPipeline('gpt2');
const classificationPipeline = await createClassificationPipeline('bert-base');
const qaPipeline = await createQuestionAnsweringPipeline('distilbert');

// Resource loading
const model = loadModel('path/to/model', { device: 'cuda' });
const tokenizer = loadTokenizer('path/to/tokenizer.json');

// Library setup
quickSetup({
  logLevel: 3,
  memoryLimitMb: 2048,
  enableProfiling: true
});

// Benchmarking
const benchmarkResult = await benchmark({
  modelPath: 'gpt2',
  iterations: 10
});
```

## Examples

### Text Generation

```typescript
import { createTextGenerationPipeline } from '@trustformers/c-bindings';

const pipeline = await createTextGenerationPipeline('gpt2');

const result = await pipeline.run("The future of artificial intelligence", {
  maxLength: 150,
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
  repetitionPenalty: 1.1
});

console.log('Generated:', result.text);
pipeline.dispose();
```

### Text Classification

```typescript
import { createClassificationPipeline } from '@trustformers/c-bindings';

const pipeline = await createClassificationPipeline('distilbert-base-uncased');

const results = await pipeline.runBatch([
  "I love this product!",
  "This is terrible.",
  "It's okay, could be better."
]);

results.forEach((result, index) => {
  console.log(`Text ${index + 1}: ${result.label} (${result.score})`);
});

pipeline.dispose();
```

### Question Answering

```typescript
import { createQuestionAnsweringPipeline } from '@trustformers/c-bindings';

const pipeline = await createQuestionAnsweringPipeline('distilbert-base-cased');

const context = "TrustformeRS is a high-performance transformer library written in Rust.";
const question = "What is TrustformeRS?";

const result = await pipeline.run(`${question} [SEP] ${context}`);

console.log('Answer:', result.answer);
console.log('Score:', result.score);

pipeline.dispose();
```

### Conversation

```typescript
import { createTextGenerationPipeline } from '@trustformers/c-bindings';

const pipeline = await createTextGenerationPipeline('gpt2');

const conversationId = pipeline.startConversation(
  "You are a helpful assistant. Be concise and friendly."
);

// User input
pipeline.addConversationTurn(conversationId, 'user', 'Hello! How are you?');
let response = await pipeline.generateConversationResponse(conversationId);
console.log('Assistant:', response.text);

// Continue conversation
pipeline.addConversationTurn(conversationId, 'assistant', response.text);
pipeline.addConversationTurn(conversationId, 'user', 'What can you help me with?');
response = await pipeline.generateConversationResponse(conversationId);
console.log('Assistant:', response.text);

pipeline.dispose();
```

### Streaming Generation

```typescript
import { createTextGenerationPipeline } from '@trustformers/c-bindings';

const pipeline = await createTextGenerationPipeline('gpt2');

await pipeline.stream("In the distant future", {
  maxLength: 200,
  temperature: 0.8
}, (chunk, isComplete) => {
  process.stdout.write(chunk);
  if (isComplete) {
    console.log('\n[Generation complete]');
  }
});

pipeline.dispose();
```

### Memory Monitoring

```typescript
import { trustformers } from '@trustformers/c-bindings';

// Basic memory usage
const basicUsage = trustformers.getMemoryUsage();
console.log('Memory usage:', {
  total: (basicUsage.totalMemoryBytes / 1024 / 1024).toFixed(2) + ' MB',
  peak: (basicUsage.peakMemoryBytes / 1024 / 1024).toFixed(2) + ' MB',
  models: basicUsage.allocatedModels
});

// Advanced memory monitoring
const advancedUsage = trustformers.getAdvancedMemoryUsage();
console.log('Advanced memory:', {
  fragmentation: (advancedUsage.fragmentationRatio * 100).toFixed(2) + '%',
  pressureLevel: advancedUsage.pressureLevel,
  allocationRate: advancedUsage.allocationRate.toFixed(2) + '/min'
});

// Memory leak detection
const leakReport = trustformers.checkMemoryLeaks();
console.log('Leak report:', leakReport);

// Cleanup
trustformers.memoryCleanup();
```

### Performance Optimization

```typescript
import { trustformers } from '@trustformers/c-bindings';

// Apply optimizations
trustformers.applyOptimizations({
  enableTracking: true,
  enableCaching: true,
  cacheSizeMb: 512,
  numThreads: 0, // Auto-detect
  enableSimd: true,
  optimizeBatchSize: true,
  memoryOptimizationLevel: 3
});

// Start profiling
trustformers.startProfiling();

// ... perform operations ...

// Get profiling report
const report = trustformers.stopProfiling();
console.log('Profiling report:', report);

// Get performance metrics
const metrics = trustformers.getPerformanceMetrics();
console.log('Performance:', {
  score: metrics.performanceScore.toFixed(2),
  avgTime: metrics.avgOperationTimeMs.toFixed(2) + ' ms',
  cacheHitRate: (metrics.cacheHitRate * 100).toFixed(2) + '%'
});
```

## Error Handling

```typescript
import { TrustformersNativeError, TrustformersError } from '@trustformers/c-bindings';

try {
  const pipeline = await createTextGenerationPipeline('invalid-model');
} catch (error) {
  if (error instanceof TrustformersNativeError) {
    console.log('Native error code:', error.code);
    console.log('Error message:', error.message);
    console.log('Context:', error.context);
    
    // Handle specific error types
    switch (error.code) {
      case TrustformersError.FileNotFound:
        console.log('Model file not found');
        break;
      case TrustformersError.OutOfMemory:
        console.log('Insufficient memory');
        break;
      default:
        console.log('Other error occurred');
    }
  } else {
    console.log('Standard error:', error.message);
  }
}
```

## Building from Source

1. Clone the repository:
```bash
git clone https://github.com/cool-japan/trustformers.git
cd trustformers/trustformers-c/nodejs
```

2. Install dependencies:
```bash
npm install
```

3. Build the TypeScript sources:
```bash
npm run build
```

4. Run examples:
```bash
npm run example
```

5. Run tests:
```bash
npm test
```

## Platform Support

- **Windows**: x64, ARM64
- **macOS**: x64, ARM64 (Apple Silicon)
- **Linux**: x64, ARM64

### GPU Support

- **CUDA**: NVIDIA GPUs with CUDA 11.0+
- **ROCm**: AMD GPUs with ROCm 4.0+
- **Metal**: Apple Silicon GPUs

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches improve throughput but require more memory
2. **Enable optimizations**: Use `quickSetup()` or `applyOptimizations()` for best performance
3. **Choose the right device**: Use GPU acceleration when available
4. **Monitor memory usage**: Use memory monitoring functions to prevent OOM errors
5. **Use quantization**: FP16 or INT8 quantization can significantly improve performance
6. **Reuse resources**: Keep models and tokenizers loaded for multiple operations

## Troubleshooting

### Common Issues

1. **Library not found**: Ensure TrustformeRS C library is built and in system path
2. **Model loading fails**: Check model path and format compatibility
3. **Out of memory**: Reduce batch size or enable memory optimizations
4. **Slow performance**: Enable optimizations and use appropriate device

### Debug Mode

Enable debug logging:
```typescript
import { trustformers } from '@trustformers/c-bindings';

trustformers.setLogLevel(5); // Trace level
```

### Memory Debugging

```typescript
// Check for memory leaks
const leakReport = trustformers.checkMemoryLeaks();
console.log('Potential leaks:', leakReport.potential_leaks);

// Monitor memory pressure
const usage = trustformers.getAdvancedMemoryUsage();
if (usage.pressureLevel >= 2) {
  console.warn('High memory pressure detected');
  trustformers.memoryCleanup();
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0

- Initial release
- TypeScript-first API
- Complete FFI bindings
- Memory management
- Performance monitoring
- Cross-platform support
- Examples and documentation