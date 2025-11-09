# Getting Started with TrustformeRS WASM

Welcome to TrustformeRS WASM! This guide will help you get up and running with transformer models in WebAssembly for browser and edge deployment.

## Quick Start

### Installation

#### Via NPM
```bash
npm install trustformers-wasm
```

#### Via CDN
```html
<script type="module">
  import init, { TrustformersWasm } from 'https://unpkg.com/trustformers-wasm/pkg/trustformers_wasm.js';
  
  async function run() {
    await init();
    const tf = new TrustformersWasm();
    console.log('TrustformeRS Version:', tf.version);
  }
  
  run();
</script>
```

### Basic Usage

#### 1. Initialize TrustformeRS

```javascript
import init, { 
  TrustformersWasm, 
  InferenceSession, 
  WasmTensor 
} from 'trustformers-wasm';

// Initialize WASM module
await init();

// Create TrustformeRS instance
const tf = new TrustformersWasm();
console.log('Version:', tf.version);
```

#### 2. Create an Inference Session

```javascript
// Create session for text generation
const session = new InferenceSession('text-generation');

// Initialize with automatic device selection
await session.initialize_with_auto_device();

// Or specify device type
await session.initialize_with_device('GPU'); // or 'CPU'
```

#### 3. Load a Model

```javascript
// Load model from URL or ArrayBuffer
const modelUrl = 'https://example.com/model.bin';
const response = await fetch(modelUrl);
const modelData = new Uint8Array(await response.arrayBuffer());

await session.load_model(modelData);
```

#### 4. Run Inference

```javascript
// Create input tensor
const inputText = "Hello, world!";
const inputData = new Float32Array(inputText.length);
for (let i = 0; i < inputText.length; i++) {
    inputData[i] = inputText.charCodeAt(i) / 255.0;
}
const inputTensor = new WasmTensor(inputData, [1, inputText.length]);

// Run inference
const result = session.predict(inputTensor);
console.log('Prediction result:', result);
```

## Core Concepts

### Inference Sessions

Inference sessions manage the lifecycle of model execution:

- **Text Generation**: For language models like GPT-2, T5
- **Text Classification**: For classification tasks
- **Image Captioning**: For multimodal models like BLIP-2
- **Translation**: For translation models like M2M-100

```javascript
const textGenSession = new InferenceSession('text-generation');
const classificationSession = new InferenceSession('text-classification');
const captioningSession = new InferenceSession('image-captioning');
const translationSession = new InferenceSession('translation');
```

### Tensors

TrustformeRS uses `WasmTensor` for data input/output:

```javascript
// Create tensor from data
const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
const tensor = new WasmTensor(data, [2, 2]); // 2x2 matrix

// Access tensor properties
console.log('Shape:', tensor.shape);
console.log('Data:', tensor.data);
```

### Device Management

Choose between CPU and GPU execution:

```javascript
// Check WebGPU support
import { is_webgpu_supported } from 'trustformers-wasm';
if (is_webgpu_supported()) {
    await session.initialize_with_device('GPU');
} else {
    await session.initialize_with_device('CPU');
}

// Get current device
const deviceType = session.current_device_type;
console.log('Using device:', deviceType);
```

## Advanced Features

### Quantization

Reduce model size and improve performance:

```javascript
import { QuantizationConfig, QuantizationPrecision } from 'trustformers-wasm';

const quantConfig = new QuantizationConfig();
quantConfig.precision = QuantizationPrecision.Int8;
quantConfig.calibration_samples = 100;

session.enable_quantization(quantConfig);
```

### Batch Processing

Process multiple inputs efficiently:

```javascript
import { BatchConfig, BatchingStrategy, Priority } from 'trustformers-wasm';

// Enable batch processing
const batchConfig = new BatchConfig();
batchConfig.max_batch_size = 8;
batchConfig.timeout_ms = 100;
batchConfig.strategy = BatchingStrategy.Dynamic;

session.enable_batch_processing(batchConfig);

// Submit batch request
const result = await session.predict_with_batching(inputTensor, Priority.High);
```

### Streaming Generation

Generate text incrementally:

```javascript
// Enable streaming
const stream = session.create_generation_stream();

// Process tokens as they're generated
for await (const token of stream) {
    console.log('Generated token:', token);
    if (token.finish_reason) {
        break;
    }
}
```

### Model Storage

Cache models in browser storage:

```javascript
// Initialize storage with size limit (MB)
await session.initialize_storage(500);

// Models are automatically cached after loading
// Check cache status
const cacheStats = session.get_cache_stats();
console.log('Cache size:', cacheStats.total_size_mb);
console.log('Cached models:', cacheStats.model_count);
```

## Performance Optimization

### WebGPU Acceleration

For best performance, enable WebGPU:

```javascript
// Check WebGPU support
if (is_webgpu_supported()) {
    await session.initialize_with_device('GPU');
    
    // Get GPU capabilities
    const capabilities = session.get_device_capabilities();
    console.log('GPU memory:', capabilities.memory_mb);
    console.log('Compute units:', capabilities.compute_units);
}
```

### Memory Management

Monitor and optimize memory usage:

```javascript
import { get_memory_stats } from 'trustformers-wasm';

// Get memory statistics
const memStats = get_memory_stats();
console.log('WASM memory:', memStats.wasm_memory / 1024 / 1024, 'MB');
console.log('GPU memory:', memStats.gpu_memory / 1024 / 1024, 'MB');

// Clean up when done
session.cleanup();
```

### Performance Monitoring

Track performance metrics:

```javascript
import { DebugConfig, LogLevel } from 'trustformers-wasm';

const debugConfig = new DebugConfig();
debugConfig.level = LogLevel.Info;
debugConfig.enable_performance_monitoring = true;
debugConfig.enable_memory_tracking = true;

session.enable_debug_logging(debugConfig);

// Performance metrics are automatically logged
```

## Error Handling

Handle errors gracefully:

```javascript
try {
    await session.load_model(modelData);
    const result = session.predict(inputTensor);
} catch (error) {
    if (error.name === 'ModelLoadError') {
        console.error('Failed to load model:', error.message);
    } else if (error.name === 'InferenceError') {
        console.error('Inference failed:', error.message);
    } else {
        console.error('Unexpected error:', error);
    }
}
```

## Examples

### Text Generation

```javascript
import init, { TrustformersWasm, InferenceSession, WasmTensor } from 'trustformers-wasm';

async function generateText() {
    await init();
    
    const session = new InferenceSession('text-generation');
    await session.initialize_with_auto_device();
    
    // Load GPT-2 model (example)
    const response = await fetch('/models/gpt2-small.bin');
    const modelData = new Uint8Array(await response.arrayBuffer());
    await session.load_model(modelData);
    
    // Create input
    const prompt = "The future of AI is";
    const inputData = new Float32Array(prompt.length);
    for (let i = 0; i < prompt.length; i++) {
        inputData[i] = prompt.charCodeAt(i) / 255.0;
    }
    const inputTensor = new WasmTensor(inputData, [1, prompt.length]);
    
    // Generate
    const result = session.predict(inputTensor);
    console.log('Generated text:', result);
}

generateText();
```

### Image Captioning

```javascript
async function captionImage() {
    await init();
    
    const session = new InferenceSession('image-captioning');
    await session.initialize_with_auto_device();
    
    // Load BLIP-2 model
    const modelData = await loadModel('/models/blip2.bin');
    await session.load_model(modelData);
    
    // Process image
    const imageData = await loadImageAsFloat32Array('image.jpg');
    const imageTensor = new WasmTensor(imageData, [1, 3, 224, 224]);
    
    // Generate caption
    const caption = session.predict(imageTensor);
    console.log('Caption:', caption);
}
```

### Translation

```javascript
async function translateText() {
    await init();
    
    const session = new InferenceSession('translation');
    await session.initialize_with_auto_device();
    
    // Load M2M-100 model
    const modelData = await loadModel('/models/m2m-100.bin');
    await session.load_model(modelData);
    
    // Translate
    const sourceText = "Hello, how are you?";
    const inputTensor = createTensorFromText(sourceText);
    
    const translation = session.predict(inputTensor);
    console.log('Translation:', translation);
}
```

## Browser Compatibility

TrustformeRS WASM supports modern browsers with:

- **WebAssembly**: All major browsers (Chrome 57+, Firefox 52+, Safari 11+)
- **WebGPU**: Chrome 94+, Firefox 113+, Safari 18+ (experimental)
- **SharedArrayBuffer**: Requires cross-origin isolation for multi-threading

### Cross-Origin Isolation

For optimal performance with SharedArrayBuffer:

```html
<!-- Add these headers to your HTML or server config -->
<meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
<meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
```

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check model format compatibility
   - Verify file size and network conditions
   - Ensure sufficient memory

2. **WebGPU not available**
   - Enable experimental WebGPU in browser flags
   - Fall back to CPU execution
   - Check browser compatibility

3. **Memory errors**
   - Reduce model size or enable quantization
   - Clear cache periodically
   - Use smaller batch sizes

### Debug Mode

Enable detailed logging:

```javascript
const debugConfig = new DebugConfig();
debugConfig.level = LogLevel.Debug;
debugConfig.enable_performance_monitoring = true;
debugConfig.enable_memory_tracking = true;
debugConfig.log_tensor_shapes = true;

session.enable_debug_logging(debugConfig);
```

## Next Steps

- Try the [interactive examples](./examples/)
- Read the [API reference](./api-reference.md)
- Check out the [performance guide](./performance-guide.md)
- Learn about [deployment options](./deployment-guide.md)

## Community

- [GitHub Repository](https://github.com/trustformers/trustformers)
- [Documentation](https://docs.trustformers.ai)
- [Discord Community](https://discord.gg/trustformers)
- [Report Issues](https://github.com/trustformers/trustformers/issues)