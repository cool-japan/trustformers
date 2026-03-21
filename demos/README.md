# TrustformeRS WebAssembly Demos

This directory contains interactive web demos showcasing the capabilities of TrustformeRS running in the browser via WebAssembly.

## Available Demos

### 1. Main Demo (`index.html`)
A comprehensive demo showcasing various TrustformeRS features:
- **Tensor Operations**: Create and manipulate tensors with various mathematical operations
- **Text Classification**: Classify text sentiment using transformer models
- **Text Generation**: Generate text from prompts (requires trained models)
- **Question Answering**: Extract answers from context
- **Memory & Performance Monitoring**: Track WASM memory usage and performance

### 2. Tensor Playground (`tensor-playground.html`)
An interactive playground for experimenting with tensor operations:
- Create custom tensors with editable values
- Perform unary operations: transpose, ReLU, softmax, normalize, etc.
- Perform binary operations: addition, multiplication, matrix multiplication
- Visualize tensor values with charts
- Real-time statistics (sum, mean, min, max)

### 3. Performance Benchmark (`benchmark.html`)
A benchmarking tool to measure TrustformeRS performance:
- Tensor operation benchmarks with configurable sizes
- Model inference benchmarks for different architectures
- Performance charts and metrics
- GFLOPS calculations for numerical operations
- Memory usage tracking

### 4. WebGPU Support Demo (`webgpu-demo.html`)
A demo showcasing WebGPU support and GPU acceleration capabilities:
- WebGPU availability detection
- GPU-accelerated tensor operations (with CPU fallback)
- Performance comparison between CPU and GPU operations
- Browser compatibility information
- Status monitoring for WebGPU features

## Running the Demos

### Prerequisites
1. Build the WASM module first:
   ```bash
   cd ../trustformers-wasm
   wasm-pack build --target web --out-dir pkg
   ```

2. Serve the files using a local web server (required for WASM):
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Node.js
   npx http-server -p 8000
   
   # Using Rust
   cargo install basic-http-server
   basic-http-server
   ```

3. Open your browser and navigate to:
   - Main demo: `http://localhost:8000/demos/index.html`
   - Tensor playground: `http://localhost:8000/demos/tensor-playground.html`
   - Benchmarks: `http://localhost:8000/demos/benchmark.html`

## Browser Requirements

- **Modern Browser**: Chrome 89+, Firefox 89+, Safari 15+, Edge 89+
- **WebAssembly Support**: Required (all modern browsers support this)
- **SIMD Support**: Optional but recommended for better performance
  - Enable in Chrome: `chrome://flags/#enable-webassembly-simd`
  - Enable in Firefox: `about:config` → `javascript.options.wasm_simd`

## Features Demonstrated

### Tensor Operations
- Creation: zeros, ones, random normal
- Arithmetic: add, subtract, multiply, divide
- Linear algebra: matrix multiplication, transpose
- Activations: ReLU, GELU, softmax, sigmoid, tanh
- Reductions: sum, mean
- Shape manipulation: reshape, slice

### Model Support
- **BERT**: For classification and question answering
- **GPT-2**: For text generation
- **T5**: For sequence-to-sequence tasks
- **LLaMA**: For large language model inference
- **Mistral**: For efficient inference

### Pipeline API
- Text Classification with customizable labels
- Text Generation with configurable parameters
- Question Answering with context extraction
- Batch processing support

## Performance Notes

1. **Initial Load**: First load may take a few seconds to download and initialize WASM
2. **Memory**: Large models may require significant memory (check browser limits)
3. **SIMD**: Enable SIMD in your browser for 2-4x performance improvement
4. **Model Weights**: Demo uses random weights; real models need trained weights

## Limitations

- **Model Weights**: Demos use randomly initialized weights for demonstration
- **Vocabulary**: Simple vocabulary for tokenization examples
- **Browser Memory**: Large tensors may hit browser memory limits
- **WebGPU**: Not yet implemented (CPU-only for now)

## Troubleshooting

### CORS Errors
If you see CORS errors, ensure you're serving files from a web server, not opening HTML files directly.

### WASM Not Loading
- Check browser console for errors
- Verify WASM file path is correct
- Ensure browser supports WebAssembly

### Performance Issues
- Enable SIMD support in browser flags
- Reduce tensor sizes or batch sizes
- Close other browser tabs to free memory

### Memory Errors
- Browser memory limits vary (typically 2-4GB)
- Reduce model size or batch size
- Monitor memory usage in the demos

## Future Enhancements

- [ ] WebGPU backend for GPU acceleration
- [ ] Model weight loading from URLs
- [ ] Real tokenizer vocabularies
- [ ] More model architectures
- [ ] Audio and vision demos
- [ ] Model quantization demos

## Contributing

To add a new demo:
1. Create a new HTML file in this directory
2. Import the TrustformeRS JavaScript API
3. Initialize the WASM module
4. Add your demo functionality
5. Update this README with the new demo