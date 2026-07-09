# trustformers-debug TODO List

**Version:** 0.2.1 | **Status:** Alpha | **Tests:** ~899 | **SLoC:** ~101,000 | **Updated:** 2026-07-09

## Overview

The `trustformers-debug` crate provides debugging and visualization tools for model development and troubleshooting. It includes profilers, memory analyzers, graph visualizers, flame graph generation, AI code analysis, interpretability/simulation tooling, and interactive debugging interfaces with VS Code integration.

**Key Responsibilities:**
- Tensor/gradient analysis with NaN/Inf detection
- Dead neuron detection
- Memory profiling with deadlock-safe mutex scoping
- Visualization via Plotters, Ratatui, and TensorBoard
- Performance profiling and flame graph generation
- Model interpretability (SHAP, LIME, feature attribution, counterfactual, attention analysis)
- Simulation & robustness testing (what-if, perturbation, adversarial, edge-case discovery)
- AI code analysis with architecture smell detection
- VS Code integration via LSP/DAP diagnostics
- Interactive, guided, and tutorial-based debugging interfaces
- Export to various visualization and data formats (TensorBoard, Netron, Excel, JSON, ...)

**Feature Flags:** `visual`, `video`, `gif`, `wasm`, `atomics`, `headless`, `cuda`, `rocm`, `tpu` (the last three are reserved placeholders for future GPU/TPU backends; see README's Feature Flags section for what they do today)

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
- [x] **INTERPRETABILITY** - SHAP, LIME, feature attribution, counterfactual, and attention-pattern analysis
- [x] **SIMULATION & GUIDED TOOLS** - What-if/perturbation/adversarial/edge-case testing, guided debugger, tutorial mode

### Feature Coverage
- **Profiling:** CPU, memory (deadlock-safe), latency analysis, flame graphs
- **Visualization:** Plotters (`visual`), Ratatui TUI (`headless`), TensorBoard, GIF (`gif`), video (`video`)
- **Analysis:** Gradient flow, weight distribution, NaN/Inf detection, dead neurons, numerical stability
- **Interpretability:** SHAP, LIME, 13 feature-attribution methods (Integrated Gradients, Grad-CAM, etc.), counterfactual generation, attention-pattern analysis
- **Simulation & Robustness:** What-if analysis, perturbation/robustness testing, adversarial probing, edge-case discovery
- **AI Analysis:** Architecture pattern matching, anti-pattern detection, actionable suggestions
- **VS Code:** LSP diagnostic output, DAP event emission, tensor shape hover annotations
- **Guided Learning:** Step-by-step guided debugger, lesson-based tutorial mode
- **Export:** TensorBoard, Netron, Excel (real `.xlsx` via `oxiarc-archive`), GraphViz, JSON

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

#### Interpretability Analyzer

**SHAP, LIME, feature attribution, counterfactual, and attention-pattern analysis**

- [x] **SHAP Analysis** — Shapley-value feature contributions with background-dataset sampling (`analyze_shap`)
- [x] **LIME Analysis** — local surrogate-model explanations via perturbation sampling (`analyze_lime`)
- [x] **Feature Attribution** — 13 methods: Integrated Gradients, Gradient×Input, SmoothGrad, Gradient SHAP, DeepLIFT, LRP, Guided Backprop, Grad-CAM, Grad-CAM++, Score-CAM, Expected Gradients, Attention Rollout, Path Integrated Gradients (`analyze_feature_attribution`, `AttributionMethod`)
- [x] **Counterfactual Generation** — minimal-change counterfactuals, feature sensitivity, decision-boundary crossing analysis, actionable insights (`generate_counterfactuals`)
- [x] **Attention Pattern Analysis** — per-layer/per-head statistics, head-specialization typing, attention-flow tracing (`analyze_attention`)

**Example:**
```rust
use std::collections::HashMap;
use trustformers_debug::{InterpretabilityAnalyzer, InterpretabilityConfig};

let mut analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

let mut instance: HashMap<String, f64> = HashMap::new();
instance.insert("feature_a".to_string(), 0.7);
instance.insert("feature_b".to_string(), 1.2);

let model_predictions = vec![0.65, 0.70, 0.68];
let background_data = vec![instance.clone()];

let shap_result = analyzer
    .analyze_shap(&instance, &model_predictions, &background_data)
    .await?;
println!("Top SHAP feature: {:?}", shap_result.top_positive_features.first());

let report = analyzer.generate_report().await?;
println!("SHAP analyses recorded: {}", report.shap_analyses_count);
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

### Simulation & Robustness Testing

**Systematic model-behavior probing (`simulation_tools`)**

- [x] **What-If Analysis** — scenario generation, impact analysis, feature-sensitivity and decision-boundary exploration (`analyze_what_if`)
- [x] **Perturbation Testing** — robustness scoring across perturbation intensities, sensitivity-hotspot identification, failure-mode analysis (`test_perturbations`)
- [x] **Adversarial Probing** — adversarial-example generation (FGSM/PGD/CW/DeepFool), attack-success analysis, certified-robustness estimation, defense recommendations (`probe_adversarial`)
- [x] **Edge Case Discovery** — automated edge-case search, classification, coverage analysis, risk assessment (`discover_edge_cases`)

**Example:**
```rust
use std::collections::HashMap;
use trustformers_debug::{SimulationAnalyzer, SimulationConfig};

let mut analyzer = SimulationAnalyzer::new(SimulationConfig::default());

let mut base_input: HashMap<String, f64> = HashMap::new();
base_input.insert("age".to_string(), 35.0);
base_input.insert("income".to_string(), 55000.0);

let model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync> =
    Box::new(|input| input.values().sum::<f64>() / 1000.0);

let robustness = analyzer.test_perturbations(&base_input, model_fn).await?;
println!("Robustness score: {:.3}", robustness.robustness_assessment.robustness_score);

let report = analyzer.generate_report().await?;
println!("Perturbation tests run: {}", report.perturbation_tests_count);
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

#### Guided Debugger

**Step-by-step guided debugging wizard**

- [x] **Features**
  - Automatic 6-step plan: health check, gradient analysis, architecture analysis, memory profiling, performance profiling, anomaly detection
  - Progress tracking, step skipping, and reset

**Example:**
```rust
use trustformers_debug::GuidedDebugger;

let mut wizard = GuidedDebugger::new();

while !wizard.is_complete() {
    let step_name = wizard.current_step().map(|s| s.name.clone());
    let result = wizard.execute_current_step().await?;
    println!("[{:.0}%] {:?} -> {:?}", wizard.progress(), step_name, result);
}
```

---

#### Tutorial Mode

**Lesson-based interactive tutorial for onboarding**

- [x] **Features**
  - Built-in lessons (Getting Started, One-Line Debugging, Guided Debugging) with objectives, example code, tips, and common mistakes
  - Per-lesson navigation and completion tracking

**Example:**
```rust
use trustformers_debug::TutorialMode;

let mut tutorial = TutorialMode::new();

while !tutorial.is_complete() {
    if let Some(lesson) = tutorial.current_lesson() {
        println!("Lesson: {}", lesson.title);
    }
    tutorial.complete_current_lesson()?;
}

println!("Tutorial progress: {:.0}%", tutorial.progress());
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

#### Excel (.xlsx) Export

**Real Office Open XML (OOXML) workbook export**

- [x] **Features**
  - Genuine `.xlsx` package (`[Content_Types].xml`, `_rels/.rels`, `xl/workbook.xml`, `xl/_rels/workbook.xml.rels`, `xl/worksheets/sheet1.xml`) built with COOLJAPAN's pure-Rust `oxiarc-archive` crate (`oxiarc_archive::zip::ZipWriter`)
  - Replaces the previous CSV-with-`.xlsx`-extension placeholder
  - Selected via the `ExportFormat::Excel` variant on `DataExportManager`

**Example:**
```rust
use trustformers_debug::data_export::{
    DataExportManager, ExportConfig, ExportFormat, ExportableData, ExportOptions,
};

let mut manager = DataExportManager::new(ExportConfig::default());

// `export_data: Vec<ExportableData>` populated from your debug session
let job_id = manager.start_export(
    "debug_metrics".to_string(),
    export_data,
    ExportFormat::Excel,
    "report.xlsx".to_string(),
    ExportOptions::default(),
)?;
```

---

## Known Limitations

- Some visualizations require the `visual` feature flag (GUI environment)
- Large models may take time to visualize
- GPU profiling requires CUDA/ROCm support
- `cuda`/`rocm`/`tpu` feature flags are placeholders for future GPU/TPU backends; today they only change which hardware-specific recommendation text `performance_tuning::PerformanceTuner` surfaces, not actual GPU/TPU kernel execution
- Interactive debugging may slow down execution
- `wasm` feature disables filesystem I/O; use in-memory buffers only

---

## Future Enhancements

### High Priority
- [x] **DONE** Enhanced profiling for distributed training across multiple ranks: `distributed_profiling::DistributedProfiler` (re-exported at crate root) tracks per-rank node registration, communication/synchronization events, load-balance analysis, and bottleneck detection with recommendations. Present in the tree since 0.1.0 but never previously reflected here.
- [x] **DONE** Better visualization for very large models (>100B params): `large_model_viz::LargeModelVisualizer` (re-exported at crate root) resolves the previous open questions on sampling strategy (Uniform/Adaptive/Representative/Importance-based), hierarchical view (layer grouping with per-group summaries), and LOD approach (configurable memory budget with sampled vs. full-detail layers). Present in the tree since 0.1.0 but never previously reflected here.
- [x] **DONE** Real-time debugging dashboard with WebSocket/SSE streaming (`dashboard_ws` module)
- [x] **DONE** More export formats: Perfetto (`export::perfetto`) and Tracy (`export::tracy`)

### Performance
- [ ] Faster graph generation for large architectures
- [x] **DONE** Reduced overhead for profiling hooks: lock-free SPSC ring buffer (`ring_buffer::LockFreeRingBuffer`)
- [ ] Better memory efficiency for long training runs
  - **Refinement needed:** Target metric? (e.g., peak RSS reduction %? allocation count reduction?)

### Features
- [x] **DONE** More interactive visualizations (animated gradient flow): `visualization::gradient_animation::GradientFlowAnimator` — frame-by-frame gradient recording, JSON/CSV/ASCII-heatmap export, health classification, summary report
- [x] **DONE** Integration with MLflow experiment tracking: `tracking::mlflow::MlflowClient` / `MlflowExperiment` — local file-based MLflow backend writing the canonical `mlruns/` layout; supports experiments, runs, metrics, params, tags, artifacts
- [x] **DONE** Automated performance regression detection: `regression::detector::RegressionDetector` — baseline save/load (JSON), statistical significance (z-score), severity-graded alerts (Minor/Moderate/Severe/Critical), human-readable reports, actionable recommendations
- [x] Custom visualization plugins (implemented 2026-04-24 via `visualization_plugins` real rendering)
- [x] Automated performance tuning recommendations (implemented 2026-04-24 via `performance_tuning`)

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

**Last Updated:** 2026-07-09 - v0.2.1 Development
**Status:** Alpha - core features implemented, API may change
**Tests:** ~899 (100% pass rate)
**Tools:** Profiling, flame graphs, visualization (Plotters/Ratatui/TensorBoard), analysis, interpretability (SHAP/LIME/attribution/counterfactual/attention), simulation & robustness testing, guided debugger, tutorial mode, AI code analysis, VS Code integration, Excel/.xlsx (real OOXML), Perfetto/Tracy export, lock-free ring buffer, SSE streaming dashboard
