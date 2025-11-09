# trustformers-debug TODO List

## Overview

The `trustformers-debug` crate provides debugging and visualization tools for model development and troubleshooting. It includes profilers, memory analyzers, graph visualizers, and interactive debugging interfaces.

**Key Responsibilities:**
- Model profiling and performance analysis
- Memory usage tracking and leak detection
- Computation graph visualization
- Layer-wise activation inspection
- Gradient flow analysis
- Interactive debugging interface
- Export to various visualization formats

---

## Current Status

### Implementation Status
✅ **PRODUCTION-READY** - Complete debugging infrastructure
✅ **ZERO COMPILATION ERRORS** - Clean compilation
✅ **COMPREHENSIVE TOOLS** - Full debugging suite
✅ **VISUALIZATION SUPPORT** - Multiple export formats
✅ **INTERACTIVE MODE** - Real-time debugging

### Feature Coverage
- **Profiling:** CPU, GPU, memory, latency analysis
- **Visualization:** Graph viz, activation maps, attention patterns
- **Analysis:** Gradient flow, weight distribution, numerical stability
- **Export:** TensorBoard, Netron, GraphViz, JSON

---

## Completed Features

### Profiling Tools

#### Performance Profiler

**Comprehensive performance analysis**

- ✅ **Metrics**
  - Layer-wise execution time
  - Memory usage per layer
  - GPU utilization
  - FLOPS calculation
  - Throughput (tokens/sec)

- ✅ **Reports**
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

**Memory usage tracking**

- ✅ **Features**
  - Peak memory tracking
  - Memory allocation timeline
  - Leak detection
  - Fragmentation analysis

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

### Visualization Tools

#### Computation Graph Visualizer

**Visual representation of model architecture**

- ✅ **Export Formats**
  - GraphViz (DOT)
  - Netron (ONNX)
  - TensorBoard graph
  - Custom JSON format

- ✅ **Features**
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

#### Activation Visualizer

**Inspect layer activations**

- ✅ **Features**
  - Heatmaps for attention weights
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

#### Attention Visualizer

**Visualize attention patterns**

- ✅ **Features**
  - Attention weight heatmaps
  - Head-by-head visualization
  - Token-to-token attention flow
  - BertViz-style visualizations

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

### Analysis Tools

#### Gradient Flow Analyzer

**Analyze gradient propagation**

- ✅ **Features**
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

- ✅ **Features**
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

**Detect numerical issues**

- ✅ **Checks**
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

### Interactive Debugging

#### Debug Console

**Interactive debugging interface**

- ✅ **Features**
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

#### TensorBoard Integration

**Export to TensorBoard**

- ✅ **Features**
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

#### Netron Export

**Export for Netron visualizer**

- ✅ **Features**
  - ONNX export
  - Model metadata
  - Interactive exploration

---

## Known Limitations

- Some visualizations require GUI environment
- Large models may take time to visualize
- GPU profiling requires CUDA/ROCm support
- Interactive debugging may slow down execution

---

## Future Enhancements

### High Priority
- Enhanced profiling for distributed training
- Better visualization for large models
- Real-time debugging dashboard
- More export formats

### Performance
- Faster graph generation
- Reduced overhead for profiling
- Better memory efficiency

### Features
- More interactive visualizations
- Integration with MLflow
- Custom visualization plugins
- Automated performance tuning

---

## Development Guidelines

### Code Standards
- **File Size:** <2000 lines per file
- **Testing:** Comprehensive test coverage
- **Documentation:** Examples for all tools
- **Performance:** Minimal overhead when disabled

### Build & Test Commands

```bash
# Build
cargo build --release -p trustformers-debug

# Run tests
cargo test -p trustformers-debug

# Run examples
cargo run --example profiler
cargo run --example visualizer
cargo run --example interactive_debug
```

---

## Usage Examples

### Basic Profiling

```rust
use trustformers::AutoModel;
use trustformers_debug::Profiler;

// Load model
let model = AutoModel::from_pretrained("gpt2")?;

// Create profiler
let profiler = Profiler::new()?;

// Profile inference
profiler.start("inference")?;
let output = model.forward(input)?;
profiler.stop("inference")?;

// Print report
println!("{}", profiler.report()?);
```

### Visualization

```rust
use trustformers_debug::GraphVisualizer;

let viz = GraphVisualizer::new()?;

// Export to GraphViz
viz.visualize_model(&model, "model.dot")?;

// Convert to PNG
std::process::Command::new("dot")
    .args(&["-Tpng", "model.dot", "-o", "model.png"])
    .output()?;

// Or export to Netron
viz.export_to_netron(&model, "model.onnx")?;
```

### Interactive Debugging

```rust
use trustformers_debug::DebugConsole;

let console = DebugConsole::new()?;

// Set breakpoints
console.breakpoint("layer.0")?;
console.breakpoint("layer.11")?;

// Run with debugging
console.debug(&model, input)?;
```

---

**Last Updated:** Refactored for alpha.1 release
**Status:** Production-ready debugging tools
**Tools:** Profiling, visualization, analysis, interactive debugging
