# trustformers-debug TODO List

**Version:** 0.2.1 (unreleased) | **Status:** Alpha | **Tests:** 1696 passing / 0 failing (`cargo nextest run -p trustformers-debug --no-fail-fast`, measured 2026-08-25) | **SLoC:** 88,454 (`tokei`, verified 2026-08-24 — this crate had no wave-4 work item, so the change from ~101,000 likely reflects measurement scope rather than code change; not investigated) | **Updated:** 2026-08-25

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

**Feature Flags:** `visual`, `image`, `gif`, `wasm`, `atomics`, `headless`, `cuda`, `rocm`, `tpu` (the last three are reserved placeholders for future GPU/TPU backends; see README's Feature Flags section for what they do today)

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
- **Visualization:** Plotters (`visual`), Ratatui TUI (`headless`), TensorBoard, GIF (`gif`), PNG heatmaps (`image`)
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
  - Netron (JSON graph — Netron opens it directly)
  - TensorBoard graph
  - Custom JSON format
  - **Not implemented:** real ONNX protobuf. `NetronExporter::export` with
    `ExportFormat::Onnx` returns a structured error naming the missing protobuf
    encoder. It previously wrote JSON bytes to the caller's `.onnx` path.

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
viz.export_to_netron(&model, "model.json")?; // Netron reads the JSON graph

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
- [x] **DONE** (2026-08-25) `kernel_optimizer::PerformanceRegressionDetector` — a separate, kernel-level regression detector (part of `KernelOptimizationReport`) now records real per-kernel execution-time history and compares it against an established baseline with a genuine Welch's t-test (`kernel_optimizer::analysis::compare_to_baseline`/`detect_regression`); `get_status` returns `Ok(None)` when there is not yet enough real history rather than a fabricated "stable, 95% confident" constant. Previously structurally incapable of ever reporting a regression (`has_regression` was hardcoded `false`).
- [x] Custom visualization plugins (implemented 2026-04-24 via `visualization_plugins` real rendering)
- [x] Automated performance tuning recommendations (implemented 2026-04-24 via `performance_tuning`)

---

## Wave 6c honesty sweep (2026-08-25) — behaviour changes to be aware of

A crate-wide triage of every remaining "placeholder / simplified / would be /
for now / simulated" marker replaced fabricated values with real computations or
honest absences. The user-visible consequences:

- **Real implementations added.** `DebugVisualizer` now renders real SVG
  (axes/polylines/binned bars/colour-mapped cells) and returns the path it wrote;
  `TensorInspector` spectral analysis uses a real `nalgebra` SVD (real numerical
  rank, real 2-norm condition number, Roy–Vetterli effective rank);
  `regression_detector` uses the exact Student-t distribution;
  `FlameGraphProfiler` captures real stacks via `std::backtrace` and renders a
  real inline SVG flame graph; `simulation_tools` FGSM/PGD are real
  finite-difference gradient attacks with L-infinity projection;
  `model_diagnostics::AdvancedAnalytics` does real PCA on the correlation matrix;
  `Profiler` reads real process RSS/CPU via `sysinfo`; `WeightAnalyzer` computes
  real 64-bin Shannon entropy.
- **Many published fields became `Option`.** Where a value could not be measured
  it is now `None` instead of a constant: LLM alignment/bias/dialog scores and
  `HealthSummary::{score,status}`, `GradientFlow::{gradient_max,gradient_min,
  dead_neurons_ratio,active_neurons_ratio}`, per-layer memory throughout
  `gradient_debugger::performance_tracking`, `CpuBottleneckAnalysis` PMU
  counters, `MemorySnapshot` memory figures, `IoProfile::queue_time`,
  environmental `EnergyEfficiencyMetrics`/`ComparativeEfficiency`,
  `KernelOptimizationSummaryReport::overall_optimization_score`,
  `TeamMetrics` peak day / growth rate, `LayerLRRecommendation::confidence`,
  `Scenario::confidence`, `CriticalGradientPath::optimization_potential`.
- **Some calls now refuse instead of pretending.** `ExportFormat::Onnx`
  (netron), `ExportFormat::MessagePack` (realtime dashboard), raster/video
  `ImageFormat`s in `DebugVisualizer`, `AdvancedMLDebugger::analyze_model_sensitivity`,
  the C&W/DeepFool/UAP/Boundary adversarial methods, and every
  `ErrorRecoverySystem` strategy except notification (which really emits) —
  each returns a structured error or `success: false` naming exactly what is
  missing.
- **Renames for accuracy.** `KernelOptimizationAnalyzer::new_stub` →
  `new_empty`; `MLPredictor`/`MLPrediction` → `WindowDispersionScorer`/
  `WindowDispersionScore` (nothing was ever trained); `WebSocketServer` →
  `DashboardEndpoint` with a real bounded update queue
  (`InteractiveDashboard::drain_pending_updates`) replacing a no-op
  `broadcast_update`; `SpectralAnalysis::eigenvalues` → `singular_values`;
  `InformationContent::{effective_rank,compression_ratio}` →
  `{value_distribution_perplexity,distinct_value_fraction}`;
  `HallucinationAnalysisResult::hallucination_probability` → `hedging_signal`.
- **`VisualizationConfig::default()` now uses `ImageFormat::SVG`** (was `PNG`,
  which the default Pure-Rust feature set cannot encode, so every default-config
  plot call would now error).

---

## Wave 6d honesty round (2026-08-25) — further behaviour changes

Follow-up to the Wave 6c sweep, driven by the Wave-6c verification findings.

- **`TrainingMonitor::clear_alert` was deleting every alert.** Its body was
  `retain(|a| !matches!(&a.alert_type, _alert_type))`; a bare identifier in
  pattern position is an irrefutable binding, not a comparison, so the arm
  always matched. It now compares the discriminant, and a regression test
  covers it. (A `matches!(x, ident)` binding fires `unused_variables` unless the
  identifier is underscore-prefixed, so the bug can only hide behind a `_`
  prefix; `grep -rnE 'matches!\(.*,\s*_[A-Za-z]\w*\s*[,)]'` over the
  workspace returns this one site and nothing else, and the remaining
  `matches!(x, _)` hits use the literal wildcard.)
- **`Profiler` CPU usage is a real two-sample `sysinfo` measurement.** The
  previous code refreshed a brand-new `System` once, which can only ever report
  `Some(0.0)`. The profiler now keeps a long-lived sampler primed in `new()` and
  reports `None` until `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` has elapsed — it
  never blocks the caller waiting for the interval.
- **`BatchMetrics` really accumulates.** `update_from_report`/`finalize` were
  empty bodies, so every `analyze_batch` report published `Default` (all
  averages `0.0`). Averages are now `Option`s over the responses that actually
  carried each sub-analysis, and `flagged_responses_count` /
  `critical_issues_count` count real safety signals.
- **`FactualityChecker` no longer scores factuality.** Nothing queries a
  knowledge base, so `factuality_score` is `None`. `verified_claims` /
  `unverified_claims` are renamed to `claim_like_sentences` /
  `uncertainty_indicator_hits` (what they count), and a real
  `uncertainty_density` is published alongside. The `contains("fact") ? 0.9 :
  0.7` ladder is gone, and the module header now describes the module as it is.
- **Alignment aggregates are `Option`.** `AlignmentMetrics::{overall_alignment_
  score, value_consistency_score, behavioral_drift, alignment_trend}` were
  seeded at construction (0.85 / 0.9 / 0.1 / Stable) and never updated;
  `LLMHealthReport::overall_health_score` averaged them anyway. It now averages
  only the terms that exist, and the "alignment drift" critical issue can no
  longer be raised by an unmeasured score.
- **One Student-t implementation.** `kernel_optimizer::analysis` and
  `differential_debugging` computed two-sided p-values as
  `2 * (1 - statrs_cdf(|t|))`, which underflows to exactly `0.0` in the upper
  tail (t=12, df=120: true p 2.8e-22). Both now delegate to
  `trustformers_core::statistics::student_t_two_sided_p_value`.
- **Kernel baseline statistics are unit-invariant.** `PerformanceDistribution`
  stored its mean/std-dev/percentiles as `Duration` (whole nanoseconds), so the
  published p-value depended on whether the same measurements were expressed in
  seconds, milliseconds or microseconds — and a microsecond-scale std-dev could
  round-trip to `0 ns`. The fields are now `f64` **seconds**
  (`mean_secs`/`std_dev_secs`/`outlier_threshold_secs`), with a test asserting
  identical p-values across the three scales.
- **`StatisticalTest::power` → `observed_power: Option<f64>`,** computed from
  the observed non-centrality and the critical value rather than being `1 - p`.
- **`RegressionDetection::{confidence, statistical_significance}` → one
  `p_value: Option<f64>`.** The two fields published the same `1 - p` in one
  branch and opposite quantities in another; the change-point branch invented
  `0.8` / `0.01` and now reports `None` (it runs no test).
- **`ErrorRecoverySystem` starts with no verdict.** `HealthMetrics` no longer
  seeds `recovery_success_rate: 1.0` / `memory_health_score: 1.0` /
  `stability_score: 1.0`; the first two are `Option`, the unmeasurable ones are
  permanently `None`, and `average_response_time_ms` became a really-computed
  `average_recovery_time_ms`.
- **Environmental figures stop inventing regional data.**
  `get_carbon_intensity`/`get_renewable_percentage` return `Option` instead of
  the "global average fallbacks" 500 gCO2/kWh and 30%; `record_emissions` and
  the cost path fail with `EnvironmentalMonitorError::UnknownRegion`.
  `EnergyMeasurement::{utilization, efficiency_ratio}` are `Option` —
  `record_session` used to stamp every measurement `utilization: 0.8`, which
  produced a published `efficiency_lost_percentage` of exactly 20% and a "GPU
  underutilization" bottleneck for every session. New
  `CarbonFootprintTracker::{set_carbon_intensity, set_renewable_percentage}`
  make that refusal actionable — the intensity map was private with no setter.
- **Computation-graph estimates come from shapes.** `create_graph` passed an
  empty shape slice, so every node took a constant (1M FLOPs for MatMul, 1024
  bytes, `Some(1_000_000)` parameters). `flop_count`/`memory_usage` are now
  `Option`, `estimate_parameters` derives real counts from the weight shapes,
  and the new `create_graph_with_shapes`/`OperationSpec` entry point lets a
  caller supply the shapes that make all three real.
- **LIME reports a real local fit.** `local_r_squared` (`0.75`),
  `local_fidelity` (`0.85`), per-feature `p_value` (`0.05`), `stability`
  (`0.8`), `confidence_interval` (`coeff ± 0.1`), `std_prediction` (`0.1`),
  `density` (`0.5`) and `neighborhood_coverage` (`0.8`) were all constants.
  `interpretability::lime::fit_local_surrogate` now computes R², per-coefficient
  standard errors, t-based p-values and confidence intervals from the
  perturbation sample; `local_fidelity` was removed (R² *is* the fidelity
  measure) and `stability`/`density` are honest `None`.
- **Rustdoc is warning-free.** `cargo doc -p trustformers-debug --no-deps`
  reported 54 warnings (public docs linking to private items, plus stale
  module-level intra-doc links from earlier waves); all are resolved and the
  command now emits none.
- **Miscellaneous.** `behavior_analysis` correlation pairs carry a real
  Pearson p-value (was `0.01`), `FeatureGroup::group_importance` (a copy of
  `average_correlation`) was removed and `AnalysisSummary::analysis_coverage`
  is `None` (was a flat `1.0`); `health_checker` trends are computed for all
  four series (three were the literal `Trend::Stable`) and the baseline
  comparison measures against the first real assessment rather than the
  literals 0.8/0.7/0.6; `PerformanceOptimizer` reads real process RSS and
  reports `None` CPU instead of `0`/`0.0` (which made every budget check pass);
  `TeamMetrics::avg_response_time` is `None` (was a flat 15.0 minutes);
  `sustainability` recommendations report the measured gap to target instead of
  multiplying it by an invented 0.2.

### Known remaining markers (triaged, not yet fixed)

Live-but-unfixed `// Simplified` sites, for a future round:
`kernel_optimizer.rs:892,941,1009` (`get_analysis`/siblings return fabricated
launch-config results), `advanced_ml_debugging.rs:920,999`,
`flame_graph_profiler.rs:899,936` (`call_count: 1`),
`memory_profiler.rs:540` (`largest_free_block` = total free memory),
`tensor_inspector.rs:856,861,866` (MSE/MAE/cosine from means only),
`ai_code_analyzer.rs:809,824,839`, `neural_network_debugging.rs:294`,
`llm_debugging.rs:1372`, `graph_visualizer.rs:367`,
`realtime_dashboard.rs:878` (`model_accuracy` naming),
`profiler/gpu.rs:70`, `profiler/mod.rs:332` (`total_memory: 0`),
`performance/optimization.rs:344,461,468,478,489` (background-task bodies that
sleep and format a string), `simulation_tools/analyzer.rs:487,667,681,1011,1077`,
`gradient_debugger/enhanced_analysis.rs:631,945,1006,1007,1052,1053,1054,1171,
1174,1202`, `environmental_monitor/efficiency_analysis.rs:226`,
`environmental_monitor/mod.rs:393` (carbon pricing constant, now documented as
stated rather than measured), `differential_debugging.rs:1447`,
`health_checker.rs:540`.

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

**Last Updated:** 2026-08-25 - v0.2.1 Development
**Status:** Alpha - core features implemented, API may change
**Tests:** 1696 (100% pass rate, measured 2026-08-25)
**Tools:** Profiling, flame graphs, visualization (Plotters/Ratatui/TensorBoard), analysis, interpretability (SHAP/LIME/attribution/counterfactual/attention), simulation & robustness testing, guided debugger, tutorial mode, AI code analysis, VS Code integration, Excel/.xlsx (real OOXML), Perfetto/Tracy export, lock-free ring buffer, SSE streaming dashboard
