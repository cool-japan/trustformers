# trustformers-debug TODO List

**Version:** 0.1.0 | **Status:** Alpha | **Tests:** 216 | **SLoC:** 61,841 | **Updated:** 2026-03-21

## Overview

The `trustformers-debug` crate provides debugging and visualization tools for model development and troubleshooting. It includes profilers, memory analyzers, graph visualizers, flame graph generation, AI code analysis, and interactive debugging interfaces with VS Code integration.

**Key Responsibilities:**
- Tensor/gradient analysis with NaN/Inf detection
- Dead neuron detection
- Memory profiling with deadlock-safe mutex scoping
- Visualization via Plotters, Ratatui, and TensorBoard
- Performance profiling and flame graph generation
- AI code analysis with architecture smell detection
- VS Code integration via LSP/DAP diagnostics
- Interactive debugging interface
- Export to various visualization formats

**Feature Flags:** `visual`, `video`, `gif`, `wasm`, `atomics`, `headless`

---

## Current Status

### Implementation Status
- [x] **ALPHA** - Core debugging infrastructure implemented
- [x] **ZERO COMPILATION ERRORS** - Clean compilation
- [x] **COMPREHENSIVE TOOLS** - Full debugging suite
- [x] **VISUALIZATION SUPPORT** - Plotters, Ratatui, TensorBoard export
- [x] **INTERACTIVE MODE** - Real-time debugging
- [x] **FLAME GRAPHS** - Inferno-compatible flamegraph output
- [x] **AI CODE ANALYSIS** - Architecture smell detection, anti-pattern matching
- [x] **VS CODE INTEGRATION** - LSP diagnostic JSON and DAP event emission

### Feature Coverage
- **Profiling:** CPU, memory (deadlock-safe), latency analysis, flame graphs
- **Visualization:** Plotters (`visual`), Ratatui TUI (`headless`), TensorBoard, GIF (`gif`), video (`video`)
- **Analysis:** Gradient flow, weight distribution, NaN/Inf detection, dead neurons, numerical stability
- **AI Analysis:** Architecture pattern matching, anti-pattern detection, actionable suggestions
- **VS Code:** LSP diagnostic output, DAP event emission, tensor shape hover annotations
- **Export:** TensorBoard, Netron, GraphViz, JSON

---

## Completed Features

### Profiling Tools

#### Performance Profiler

**Comprehensive performance analysis**

- [x] **Metrics**
  - Layer-wise execution time
  - Memory usage per layer
  - GPU utilization
  - FLOPS calculation
  - Throughput (tokens/sec)

- [x] **Reports**
  - Summary statistics
  - Bottleneck identification
  - Optimization recommendations
  - Comparison across runs

**Example:**
```rust
use trustformers_debug::Profiler;

let profiler = Profiler::new()?;

// Profile model forward pass
profiler.start("forward")?;
let output = model.forward(input)?;
profiler.stop("forward")?;

// Get report
let report = profiler.report()?;
println!("{}", report);
// Layer "attention": 15.2ms (45% of total), 2.3GB memory
// Layer "ffn": 18.5ms (55% of total), 1.8GB memory
```

---

#### Memory Profiler

**Deadlock-safe memory usage tracking**

- [x] **Features**
  - Peak memory tracking with RAII-scoped mutex guards (no `unwrap`, no lock poisoning)
  - Memory allocation timeline with per-layer attribution
  - Leak detection
  - Fragmentation analysis
  - Thread-safe snapshot API with bounded async scope

**Example:**
```rust
use trustformers_debug::MemoryProfiler;

let mem_profiler = MemoryProfiler::new()?;

mem_profiler.start()?;
let model = load_model("gpt2")?;
mem_profiler.snapshot("model_loaded")?;

let output = model.forward(input)?;
mem_profiler.snapshot("forward_complete")?;

let report = mem_profiler.report()?;
println!("Peak memory: {} GB", report.peak_gb());
println!("Leaks detected: {}", report.leaks().len());
```

---

#### Flame Graph Generator

**Inferno-compatible call stack visualization**

- [x] **Features**
  - Collapsed stack format compatible with `inferno-flamegraph`
  - Per-layer timing attribution
  - Forward and backward pass separation
  - Export to `/tmp/` for safe temporary handling

**Example:**
```rust
use trustformers_debug::profiler::FlameGraphBuilder;

let mut builder = FlameGraphBuilder::new();
builder.record_frame("forward", "transformer_block_0", 15_200);
builder.record_frame("forward/attention", "multi_head_attn", 9_800);

// Write to temp file
let path = std::env::temp_dir().join("trustformers_flame.txt");
builder.write_to_file(&path)?;
// Run: inferno-flamegraph < /tmp/trustformers_flame.txt > flame.svg
```

---

### Visualization Tools

#### Computation Graph Visualizer

**Visual representation of model architecture**

- [x] **Export Formats**
  - GraphViz (DOT)
  - Netron (ONNX)
  - TensorBoard graph
  - Custom JSON format

- [x] **Features**
  - Node annotations (shape, dtype, device)
  - Edge labels (tensor dimensions)
  - Subgraph clustering
  - Interactive exploration

**Example:**
```rust
use trustformers_debug::GraphVisualizer;

let viz = GraphVisualizer::new()?;

// Visualize model
viz.visualize_model(&model, "model.dot")?;
viz.export_to_netron(&model, "model.onnx")?;

// Open in browser
viz.serve_interactive(&model, 8080)?;
// Navigate to http://localhost:8080
```

---

#### Activation Visualizer (`visual` feature)

**Inspect layer activations**

- [x] **Features**
  - Heatmaps for attention weights (via Plotters)
  - Distribution histograms
  - Activation statistics (mean, std, min, max)
  - Outlier detection

**Example:**
```rust
use trustformers_debug::ActivationVisualizer;

let viz = ActivationVisualizer::new()?;

// Register hooks
viz.register_forward_hook(&model, "layer.0.attention")?;

// Run forward pass
let output = model.forward(input)?;

// Get activations
let activations = viz.get_activations("layer.0.attention")?;

// Visualize
viz.plot_heatmap(&activations, "attention_heatmap.png")?;
viz.plot_distribution(&activations, "activation_dist.png")?;
```

---

#### Attention Visualizer (`visual` feature)

**Visualize attention patterns**

- [x] **Features**
  - Attention weight heatmaps (Plotters)
  - Head-by-head visualization
  - Token-to-token attention flow
  - BertViz-style HTML export

**Example:**
```rust
use trustformers_debug::AttentionVisualizer;

let viz = AttentionVisualizer::new()?;

// Get attention weights
let attention = model.get_attention_weights(input)?;

// Visualize
viz.plot_attention_heatmap(&attention, "attention.png")?;
viz.export_to_bertviz(&attention, tokens, "attention.html")?;
```

---

#### Terminal Visualization (`headless` feature)

**Ratatui TUI dashboards and ASCII plots for headless environments**

- [x] **Features**
  - Ratatui-based TUI training dashboard
  - ASCII histogram and line plots
  - Live updating metrics panel
  - No GUI dependency

**Example:**
```rust
use trustformers_debug::TerminalVisualizer;

let terminal_viz = TerminalVisualizer::new(80, 24);

// ASCII histogram
let histogram = terminal_viz.ascii_histogram(&values, 10);
println!("{}", histogram);

// ASCII line plot
let line_plot = terminal_viz.ascii_line_plot(&x_values, &y_values, "Training Loss");
println!("{}", line_plot);
```

---

#### TensorBoard Integration

**Export to TensorBoard**

- [x] **Features**
  - Scalar logging
  - Histogram logging
  - Graph visualization
  - Embedding projector

**Example:**
```rust
use trustformers_debug::TensorBoardWriter;

let writer = TensorBoardWriter::new("runs/experiment1")?;

// Log scalars
writer.add_scalar("loss", loss_value, step)?;

// Log histograms
writer.add_histogram("layer.0.weight", weights, step)?;

// Log graph
writer.add_graph(&model)?;

// Log embeddings
writer.add_embedding(embeddings, labels, step)?;
```

---

### Analysis Tools

#### Gradient Flow Analyzer

**Analyze gradient propagation**

- [x] **Features**
  - Gradient norm tracking
  - Vanishing/exploding gradient detection
  - Layer-wise gradient statistics
  - Gradient clipping recommendations

**Example:**
```rust
use trustformers_debug::GradientAnalyzer;

let analyzer = GradientAnalyzer::new()?;

// Register backward hooks
analyzer.register_hooks(&model)?;

// Backward pass
loss.backward()?;

// Analyze gradients
let report = analyzer.analyze()?;
println!("Vanishing gradients in: {:?}", report.vanishing_layers());
println!("Exploding gradients in: {:?}", report.exploding_layers());
```

---

#### Weight Distribution Analyzer

**Analyze weight distributions**

- [x] **Features**
  - Histogram plots
  - Statistical summaries
  - Dead neuron detection
  - Weight initialization validation

**Example:**
```rust
use trustformers_debug::WeightAnalyzer;

let analyzer = WeightAnalyzer::new()?;

// Analyze weights
let report = analyzer.analyze_model(&model)?;

println!("Dead neurons: {}", report.dead_neurons().len());
println!("Mean weight: {:.4}", report.mean());
println!("Std weight: {:.4}", report.std());

// Plot distributions
analyzer.plot_weight_histogram(&model, "weights.png")?;
```

---

#### Numerical Stability Checker

**Detect NaN, Inf, and numerical issues**

- [x] **Checks**
  - NaN detection
  - Inf detection
  - Underflow/overflow detection
  - Precision loss detection

**Example:**
```rust
use trustformers_debug::StabilityChecker;

let checker = StabilityChecker::new()?;

// Check model outputs
checker.check_tensor(&output)?;

// Get report
let issues = checker.get_issues()?;
for issue in issues {
    println!("Issue in {}: {:?}", issue.layer, issue.kind);
}
```

---

### AI Code Analysis

**Automated architecture smell and anti-pattern detection**

- [x] **Features**
  - Pattern matching for gradient checkpoint misuse
  - Redundant recomputation detection
  - Degenerate attention head identification
  - Excessive depth without skip connection warnings
  - Actionable suggestions with layer-level annotations

**Example:**
```rust
use trustformers_debug::ai_analysis::CodeAnalyzer;

let analyzer = CodeAnalyzer::new();
let report = analyzer.analyze_model_config(&model_config)?;

for suggestion in report.suggestions() {
    println!("[{}] {}: {}", suggestion.severity, suggestion.layer, suggestion.message);
}
```

---

### VS Code Integration

**LSP diagnostic and DAP event output**

- [x] **Features**
  - Structured JSON diagnostic output (LSP DiagnosticSeverity format)
  - DAP (Debug Adapter Protocol) event emission for breakpoints
  - Tensor shape annotations in hover-compatible JSON
  - Compatible with Rust Analyzer extension

**Example:**
```rust
use trustformers_debug::vscode::DiagnosticEmitter;

let emitter = DiagnosticEmitter::new();
emitter.emit_tensor_shape_diagnostic("layer.0.attention", &[1, 12, 512, 512])?;
// Outputs JSON-RPC notification compatible with VS Code LSP client
```

---

### Interactive Debugging

#### Debug Console

**Interactive debugging interface**

- [x] **Features**
  - REPL-style interface
  - Tensor inspection
  - Layer-wise execution
  - Breakpoints
  - Variable watching

**Example:**
```rust
use trustformers_debug::DebugConsole;

let console = DebugConsole::new()?;

// Set breakpoint
console.breakpoint("layer.0.attention")?;

// Run with debugging
console.run(&model, input)?;

// Interactive session:
// > inspect layer.0.attention.output
// Tensor(shape=[1, 12, 512, 512], dtype=f32, device=cuda:0)
// > stats layer.0.attention.output
// mean=0.0234, std=0.982, min=-2.31, max=3.45
```

---

### Export and Integration

#### Netron Export

**Export for Netron visualizer**

- [x] **Features**
  - ONNX export
  - Model metadata
  - Interactive exploration

---

## Known Limitations

- Some visualizations require the `visual` feature flag (GUI environment)
- Large models may take time to visualize
- GPU profiling requires CUDA/ROCm support
- Interactive debugging may slow down execution
- `wasm` feature disables filesystem I/O; use in-memory buffers only

---

## Future Enhancements

### High Priority
- Enhanced profiling for distributed training across multiple ranks
- Better visualization for very large models (>100B params)
- Real-time debugging dashboard with WebSocket streaming
- More export formats (Perfetto, Tracy)

### Performance
- Faster graph generation for large architectures
- Reduced overhead for profiling hooks (lock-free ring buffer)
- Better memory efficiency for long training runs

### Features
- More interactive visualizations (animated gradient flow)
- Integration with MLflow experiment tracking
- Custom visualization plugins
- Automated performance tuning recommendations

---

## Development Guidelines

### Code Standards
- **File Size:** <2000 lines per file
- **Testing:** Comprehensive test coverage
- **Documentation:** Examples for all tools
- **Performance:** Minimal overhead when disabled
- **Temp files:** Always use `std::env::temp_dir()` in tests

### Build & Test Commands

```bash
# Build
cargo build --release -p trustformers-debug

# Run tests
cargo test -p trustformers-debug

# Run with all features
cargo test -p trustformers-debug --all-features

# Run examples
cargo run --example profiler
cargo run --example visualizer
cargo run --example interactive_debug
```

---

**Last Updated:** 2026-03-21 - 0.1.0 Alpha Release
**Status:** Alpha - core features implemented, API may change
**Tests:** 216 (100% pass rate)
**Tools:** Profiling, flame graphs, visualization (Plotters/Ratatui/TensorBoard), analysis, AI code analysis, VS Code integration
