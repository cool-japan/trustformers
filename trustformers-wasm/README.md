# TrustformeRS WebAssembly

WebAssembly bindings for the TrustformeRS transformer library, enabling transformer models to run directly in web browsers and Node.js environments with WebGPU hardware acceleration.

**Version:** 0.1.4 | **Status:** Stable | **Tests:** ~130 | **SLoC:** 55,721 | **Last Updated:** 2026-07-02

## Features

- **WebGPU Backend**: GPU compute via direct `web-sys`/`js-sys` bindings to the browser WebGPU API (no `wgpu` crate dependency), with automatic CPU fallback — see "WebGPU Notes" below for current dispatch-path coverage
- **Web Workers Parallelism**: Multi-threaded inference via SharedArrayBuffer
- **IndexedDB Caching**: Persistent model and KV-cache storage in the browser
- **BERT WASM Model**: Complete BERT implementation running in-browser
- **React/Vue/Angular/Web Components**: First-class framework bindings
- **Streaming Inference**: Token-by-token generation with streaming API
- **SIMD Support**: Hardware-accelerated tensor ops where available
- **Mobile Optimization**: Battery-aware, network-adaptive loading
- **SciRS2 (optional)**: `scirs2-core` wired in as an optional dependency behind the `scirs2` feature for future tensor-op acceleration (not yet called from any in-crate tensor op)

## Building

### Prerequisites

- Rust (latest stable)
- wasm-pack (`curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`)

### Build Commands

```bash
# Build for all targets
./build.sh

# Or build individually:
wasm-pack build --target web --out-dir pkg-web
wasm-pack build --target bundler --out-dir pkg-bundler
wasm-pack build --target nodejs --out-dir pkg-node
```

## Usage

### Browser (Direct)

```html
<script type="module">
import init, { TrustformersWasm, WasmTensor } from './pkg-web/trustformers_wasm.js';

async function run() {
    await init();

    const tf = new TrustformersWasm();
    console.log('Version:', tf.version);  // "0.1.4"

    // Create and manipulate tensors
    const tensor = WasmTensor.new([1, 2, 3, 4], [2, 2]);
    const result = tensor.add(tensor);
    console.log('Result:', result.data);
}

run();
</script>
```

### Node.js

```javascript
const { TrustformersWasm, WasmTensor } = require('./pkg-node/trustformers_wasm.js');

const tf = new TrustformersWasm();
const tensor = WasmTensor.new([1, 2, 3, 4], [2, 2]);
console.log(tensor.toString());
```

### Webpack/Bundler

```javascript
import * as wasm from './pkg-bundler/trustformers_wasm';

async function run() {
    await wasm.default();

    const tf = new wasm.TrustformersWasm();
    // Use the library...
}
```

## API Overview

This crate exposes roughly **2,276 public items** (functions, structs, enums, and traits, including trait/impl-block methods) across 102 source files under `src/`; about 621 of those are top-level module-level declarations.

### Core Classes

#### `TrustformersWasm`
Main entry point for the library.

```javascript
const tf = new TrustformersWasm();
console.log(tf.version);     // "0.1.4"
console.log(tf.initialized); // true
```

#### `WasmTensor`
Core tensor operations.

```javascript
// Creation
const a = WasmTensor.new([1, 2, 3, 4], [2, 2]);
const b = WasmTensor.zeros([3, 3]);
const c = WasmTensor.ones([2, 4]);
const d = WasmTensor.randn([5, 5]);

// Operations
const sum = a.add(b);
const prod = a.matmul(b);
const transposed = a.transpose();

// Activations
const relu_out = a.relu();
const gelu_out = a.gelu();
const softmax_out = a.softmax(-1);
```

#### `Linear`
Fully connected layer.

```javascript
const linear = new Linear(input_size, output_size, use_bias);
const output = linear.forward(input_tensor);
```

#### `BertModelWasm`
BERT model running entirely in WASM.

```javascript
const config = BertConfig.tiny();
const model = new BertModelWasm(config);
const output = model.forward(input_ids, attention_mask);
```

### WebGPU Backend

```javascript
import { is_webgpu_available, GpuTensorFactory } from './pkg-web/trustformers_wasm.js';

// Check whether the browser exposes navigator.gpu at all
console.log('WebGPU available:', is_webgpu_available());

// create_tensor() tries WebGPU first and falls back to CPU automatically
// (see "WebGPU Notes" below for current dispatch-path coverage)
const a = await GpuTensorFactory.create_tensor([1, 2, 3, 4], [2, 2]);
const b = await GpuTensorFactory.create_tensor([1, 1, 1, 1], [2, 2]);
const sum = await a.add(b);
console.log('Result:', sum.data, 'backend:', sum.backend_info());
```

### Framework Bindings

#### React

```jsx
import { useTrustformers, TrustformersProvider } from 'trustformers-react';

function App() {
    const { model, generate, isLoading } = useTrustformers('bert-base');
    return (
        <TrustformersProvider>
            <InferenceComponent model={model} onGenerate={generate} />
        </TrustformersProvider>
    );
}
```

#### Vue

```javascript
import { useTrustformers } from 'trustformers-vue';

export default {
    setup() {
        const { model, tokenizer, generate } = useTrustformers('bert-base');
        return { model, generate };
    }
}
```

#### Angular

```typescript
import { TrustformersService } from 'trustformers-angular';

@Injectable({ providedIn: 'root' })
export class AppComponent {
    constructor(private tf: TrustformersService) {}

    async generate(prompt: string) {
        return this.tf.generate(prompt).pipe(toArray()).toPromise();
    }
}
```

#### Web Components

```html
<trustformers-inference-engine model="bert-base"></trustformers-inference-engine>
<trustformers-model-loader src="./models/bert.bin"></trustformers-model-loader>
<trustformers-performance-monitor></trustformers-performance-monitor>
```

### Utilities

```javascript
// Performance measurement
const timer = new Timer("My Operation");
// ... do work ...
console.log(`Elapsed: ${timer.elapsed()}ms`);

// Memory statistics
const stats = get_memory_stats();
console.log(`Memory used: ${stats.used_mb} MB`);

// Feature detection
console.log(`SIMD enabled: ${enable_simd()}`);
console.log(`Features: ${features()}`);
```

## Feature Flags

- `webgpu` — WebGPU compute backend via direct `web-sys`/`js-sys` bindings to the browser API (no `wgpu` crate dependency); gates `compute::webgpu`, `compute::gpu_tensor`, `compute::webgpu_simple`
- `web-workers` — Web Workers-based multi-threaded execution (`src/compute/web_workers.rs`)
- `shared-memory` — SharedArrayBuffer-backed cross-thread shared memory (`src/compute/threads.rs`)
- `kernel-fusion` — Fused transformer kernel patterns (MHA, FFN, LayerNorm+Residual, RMSNorm, SwiGLU) in `compute/webgpu/kernel_fusion.rs` + `advanced_fusion_patterns.rs`; note these modules currently compile whenever `webgpu` is enabled, so this flag is not yet an independent source-level gate
- `async-executor` — Async task executor for WebGPU dispatch (`compute/webgpu/async_executor.rs`); like `kernel-fusion`, currently compiles under `webgpu` regardless of this flag
- `indexeddb` — IndexedDB model/KV-cache persistence; also gates the top-level `storage` module itself, so `memory64`/`streaming-loader`/`model-splitting` below need this enabled too
- `memory64` — WASM memory64 addressing for models >4GB (`src/storage/memory64.rs`)
- `streaming-loader` — Progressive chunked model loading (`src/storage/streaming_loader.rs`, `progressive_loader.rs`)
- `model-splitting` — Splits large models into chunks for loading (`src/storage/model_splitting.rs`)
- `react-components` — React hooks and component library (`src/react_components.rs`)
- `vue-components` — Vue composables and plugin (`src/vue_components.rs`)
- `angular-components` — Angular services and directives (`src/angular_components.rs`)
- `web-components` — Framework-agnostic custom elements (`src/web_components/`)
- `playground` — Interactive browser playground (`src/playground.rs`)
- `streaming-generation` — Token-by-token streaming inference (`src/streaming_generation.rs`)
- `mobile-optimization` — Battery/network-adaptive loading, touch gestures, camera integration, device-capability detection (`src/mobile.rs`, `touch_gestures.rs`, `camera_integration.rs`, `device_capability*`)
- `scirs2` — Enables the optional `scirs2-core` dependency; not yet consumed by any tensor op in this crate (reserved for future acceleration)
- `console_panic` — Routes Rust panics to the browser console via `console_error_panic_hook` (part of `default`)
- `dlmalloc-alloc` — Swaps the global allocator to `dlmalloc` (`src/allocator.rs`) for wasm32 (part of `default`)
- `default` — `console_panic` + `dlmalloc-alloc`
- `size-optimized` — Same composition as `default` today (`dlmalloc-alloc` + `console_panic`)
- `performance-optimized` — `dlmalloc-alloc` + `kernel-fusion` + `async-executor` + `scirs2`; does **not** include `webgpu` itself, so the fusion/executor code (gated behind `webgpu` at the module level) won't actually compile in unless `webgpu` is enabled too
- `minimal` — Smallest viable build: `dlmalloc-alloc` only
- `full` — Enables every additive feature except `webgpu` and `console_panic` (web-workers, shared-memory, kernel-fusion, async-executor, indexeddb, memory64, streaming-loader, model-splitting, react-components, vue-components, angular-components, web-components, playground, streaming-generation, mobile-optimization, scirs2, dlmalloc-alloc); combine with `--features full,webgpu` for GPU support too

## WebGPU Notes

This crate has **no dependency on the native `wgpu` crate**. WebGPU support is implemented by calling the browser's WebGPU API directly through hand-written `web-sys`/`js-sys` bindings:

- **Types**: `GpuAdapter`, `GpuDevice`, `GpuQueue`, etc. are `js_sys::Object` aliases (`src/compute/webgpu/types.rs`), with extension traits (`GpuDeviceExt`, `GpuAdapterExt`, `GpuQueueExt`, `GpuBufferExt`) that use JS reflection for methods web-sys doesn't bind natively.
- **Device negotiation**: `navigator.gpu` → `requestAdapter()` → `requestDevice()`, each awaited via `wasm_bindgen_futures::JsFuture` with explicit null/undefined checks (`GpuTensor::init_webgpu` in `src/compute/gpu_tensor.rs`; `WebGPUOps::initialize` in `src/compute/webgpu_simple.rs`).
- **Shared backend handle**: the negotiated backend is wrapped in `Rc<RefCell<WebGPUBackend>>` (`src/compute/gpu_tensor.rs`) for cheap sharing across derived tensors plus interior mutability for pipeline caching; `RefCell` borrows are scoped so none is ever held across an `.await`.
- **CPU fallback is real** at multiple levels: `WebGPUBackend::is_available()` probes for `navigator.gpu` before attempting GPU init; `GpuTensorFactory::create_tensor` falls back silently on any initialization error; per-op methods on `GpuTensor` (`matmul`/`add`/`relu`) route to CPU tensor math whenever no GPU backend is active.
- **Two dispatch paths of different completeness** coexist — know which one you're using:
  - `WebGPUOps` (`src/compute/webgpu_simple.rs`) is fully wired end-to-end: it compiles 7 real WGSL compute shaders (matmul, add, relu, sigmoid, tanh, gelu, softmax), builds storage buffers/bind groups/command encoders, dispatches compute passes, and reads results back via a staging buffer + `map_async`/`getMappedRange`.
  - `WebGPUBackend`/`SimpleGpuOps` (`src/compute/webgpu/backend.rs`, `simple_ops.rs`) — the path behind the `Rc<RefCell>`-wrapped `GpuTensor` — allocate real GPU buffers and pipelines, but their dispatch methods (`dispatch_add`/`dispatch_relu`/`dispatch_matmul`, and `SimpleGpuOps::matmul`/`softmax`/`layer_norm`/`attention`) currently execute the CPU fallback path by explicit documented design; GPU dispatch for these ops isn't wired in yet.
  - **Recommendation**: use `WebGPUOps` directly if you need guaranteed end-to-end GPU execution today; `GpuTensor` is convenient but currently CPU-backed for most ops even when a GPU device was successfully acquired.

## Examples

See the `examples/` directory for complete examples:

- `index.html` / `playground.html` — Interactive browser demo
- `demo/` — Full-featured playground application
- Node.js example in `examples/`

## Performance Tips

1. **Enable WebGPU**: Use Chrome 113+ / Edge 113+ for 50-100x speedup
2. **Enable SIMD**: Compile with WASM SIMD128 target feature
3. **Batch operations**: Process multiple inputs together
4. **Use IndexedDB caching**: Avoid re-downloading models between sessions
5. **Enable kernel fusion**: `webgpu` + `kernel-fusion` features
6. **Reuse tensors**: Minimize allocations in hot loops

## Testing

This crate has two distinct test layers:

- **Rust unit tests** — in-source tests use plain `#[test]` attributes (163 occurrences across 39 files; 0 `#[wasm_bindgen_test]`, despite `wasm-bindgen-test` being a dev-dependency), so they compile and run as ordinary host-target Rust tests; no browser or `wasm-pack` runner is required for this layer.
- **Browser/E2E tests** — a separate JS-driven suite under `tests/*.js` (Playwright + Jest: cross-browser, e2e, performance, visual-regression, memory-leak checks), run via `npm test` / `npx playwright test` from `tests/package.json`, independent of the Rust test binary.

```bash
# Run the Rust unit tests (host target)
cargo test
cargo nextest run

# Run with specific features
cargo test --features webgpu

# Check that the actual wasm32 build compiles
cargo check --target wasm32-unknown-unknown

# Run the browser/E2E JS suite
cd tests && npm test               # Jest-based suite
cd tests && npx playwright test    # Playwright cross-browser/e2e suite
```

~130 unit tests with 100% pass rate for this crate, covering:
- Core tensor operations
- WebGPU backend (mock device)
- BERT forward pass
- Framework binding contracts
- Streaming generation
- IndexedDB model cache

Workspace-wide (`cargo nextest run --workspace --all-features`, 2026-07-01): 18,102 passed / 0 failed / 119 skipped; 0 clippy warnings; 0 rustdoc warnings.

## Limitations

- WebGPU requires Chrome 113+, Edge 113+, or Safari (experimental)
- SharedArrayBuffer requires cross-origin isolation headers
- SIMD requires WASM SIMD128 browser support
- Memory typically capped at 2-4GB (use `memory64` + quantization for large models)
- `WebGPUBackend`/`SimpleGpuOps` (the dispatch path behind `GpuTensor`) currently execute the CPU fallback for matmul/add/relu/softmax/layer_norm/attention by documented design — use `WebGPUOps` (`compute::webgpu_simple`) directly for guaranteed end-to-end GPU dispatch today (see "WebGPU Notes")
- 0 `todo!()`/`unimplemented!()` macros in source, but several documented simplifications remain (none block compilation or panic): a no-op cache-clear recovery action (`src/error.rs`), fixed-bytes-per-element quantization stats (`src/optimization/quantization/quantizer.rs`), hardcoded device-capability probes (`src/device_capability/detector.rs`), fixed-constant (non-bit-width-aware) basic quantization math (`src/optimization/quantization/algorithms/basic.rs`), synthesized `blob:`/`data:` URLs in place of `URL.createObjectURL()` (`src/storage/model_splitting.rs`, `src/compute/threads.rs`), and default (non-queried) device capabilities (`src/compute/webgpu/mod.rs`)

## License

Apache-2.0
