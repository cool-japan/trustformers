# TrustformeRS Debug

**Version:** 0.2.1 | **Status:** Alpha | **Tests:** ~899 | **SLoC:** ~101,000 | **Updated:** 2026-07-09

Advanced debugging and analysis tools for TrustformeRS machine learning models.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `visual` | Enable graphical plot output via Plotters (opt-in — see note below) |
| `video` | Enable video frame export for training animations |
| `gif` | Enable animated GIF export for visualization sequences |
| `wasm` | Enable WebAssembly-compatible debugging (no filesystem I/O) |
| `atomics` | Enable atomic counters for lock-free profiling in multi-threaded contexts |
| `headless` | Enable terminal/ASCII visualization for server environments (Ratatui) |
| `cuda` | Reserved for a future NVIDIA CUDA GPU backend (see note below) |
| `rocm` | Reserved for a future AMD ROCm GPU backend (see note below) |
| `tpu` | Reserved for a future Google TPU backend (see note below) |

`default = []` — this crate is Pure-Rust by default (COOLJAPAN Pure-Rust policy). `visual` is deliberately **not** part of `default`: it pulls in `plotters` → `image` → `png` → `flate2`/`miniz_oxide`, and compression crates other than `oxiarc-archive` are policy-banned in default builds. Enable it explicitly with `--features visual` when graphical (PNG/SVG) plot output is needed.

`cuda`/`rocm`/`tpu` are placeholder flags for future GPU/TPU backend work — no GPU/TPU kernels are compiled or executed anywhere in the crate today. The only place they currently have any effect is the Performance Profiler's `performance_tuning::PerformanceTuner`, whose compile-time `detect_hardware()` check picks which hardware-specific tuning recommendations to surface (`HardwareType::NvidiaGpu`/`AmdGpu`/`Tpu` vs. the `Cpu`/`AppleSilicon` defaults).

## Overview

TrustformeRS Debug provides comprehensive debugging capabilities for deep learning models, including tensor inspection, gradient flow analysis, model diagnostics, performance profiling, automated debugging hooks, model interpretability (SHAP/LIME/attention analysis), and simulation-based robustness testing. These tools help identify training issues, performance bottlenecks, and model problems early in the development process.

## Features

### Tensor Inspector
- **Real-time Analysis**: Inspect tensor statistics, distributions, and patterns
- **Anomaly Detection**: Automatically detect NaN, infinite, and extreme values
- **Memory Tracking**: Monitor memory usage and identify memory leaks
- **Comparison Tools**: Compare tensors across different training steps or layers

### Gradient Debugger
- **Flow Analysis**: Track gradient flow through model layers
- **Problem Detection**: Identify vanishing and exploding gradients
- **Dead Neuron Detection**: Find layers with inactive neurons
- **Trend Analysis**: Monitor gradient patterns over training steps

### Model Diagnostics
- **Training Dynamics**: Analyze convergence, stability, and learning efficiency
- **Architecture Analysis**: Evaluate model structure and parameter efficiency
- **Performance Monitoring**: Track training metrics and detect anomalies
- **Health Assessment**: Overall model health scoring and recommendations

### Debugging Hooks
- **Automated Monitoring**: Set up conditional debugging triggers
- **Layer-specific Tracking**: Monitor specific layers or operations
- **Custom Callbacks**: Implement custom debugging logic
- **Alert System**: Real-time notifications for debugging events

### Visualization

Multiple backends depending on enabled features:

- **Plotters** (`visual` feature): Static PNG/SVG plots, histograms, scatter charts, line graphs
- **Ratatui** (`headless` feature): Terminal-based TUI dashboards and ASCII plots for headless server environments
- **TensorBoard**: Event file export compatible with TensorBoard scalar/histogram/graph viewers
- **GIF export** (`gif` feature): Animated training progress visualizations
- **Video export** (`video` feature): MP4-compatible frame sequences for training recordings

### Large-Model Visualization

Memory-bounded visualization for models too large to render layer-by-layer (`LargeModelVisualizer`, `large_model_viz` module, re-exported at the crate root):

- **Smart Layer Sampling**: Uniform, Adaptive, Representative, and Importance-based (parameter-count/compute-cost weighted) sampling strategies pick a representative subset of layers instead of rendering every layer
- **Hierarchical Rendering**: Layers are grouped into clusters with per-group summary statistics for very deep architectures
- **Memory-Bounded Caching**: Configurable memory budget (`max_memory_mb`) that warns when accumulated layer metadata exceeds it
- **Output Formats**: Text summary, JSON metadata, static PNG/SVG, and interactive SVG/HTML

### Memory Profiling

Deadlock-safe memory profiling using scoped mutex guards:

- Peak memory tracking with RAII-scoped lock acquisition (no `unwrap`, no lock poisoning)
- Memory allocation timeline with per-layer attribution
- Fragmentation analysis and leak detection
- Thread-safe snapshot API with bounded scope to prevent deadlock across async boundaries

### Performance Profiler
- **Execution Timing**: Profile layer and operation execution times
- **Memory Profiling**: Track memory usage patterns with deadlock-safe mutex scoping
- **Bottleneck Detection**: Identify performance bottlenecks automatically
- **Optimization Suggestions**: Get recommendations for performance improvements
- **Flame Graphs**: Generate Inferno-compatible flamegraph data for call stack visualization

### Distributed Profiling

Multi-node, multi-rank profiling for distributed training (`DistributedProfiler`, `distributed_profiling` module, re-exported at the crate root):

- **Per-Rank Node Registration**: Track each node by rank and role, recording communication and synchronization events as training progresses
- **Communication & Synchronization Analysis**: Aggregated summaries of cross-node communication patterns and synchronization overhead
- **Load Balance Analysis**: Detect work imbalance across ranks
- **Bottleneck Detection**: Automatic identification of distributed-training bottlenecks with actionable recommendations
- **Real-time Stats & Export**: Live statistics snapshots (`get_realtime_stats`) plus full JSON report export (`export_json`)

### AI Code Analysis

Automated model code analysis powered by built-in heuristics:

- Pattern matching for common anti-patterns (gradient checkpoint misuse, redundant recomputation)
- Architecture smell detection (excessive depth without skip connections, degenerate attention heads)
- Actionable suggestions with line-level annotations compatible with VS Code diagnostics

### VS Code Integration

Language Server Protocol (LSP)-compatible diagnostic output:

- Emit structured JSON diagnostics consumable by the VS Code Rust Analyzer extension
- Breakpoint-compatible debug session with DAP (Debug Adapter Protocol) event emission
- In-editor tensor shape annotations via hover provider protocol

### Interpretability

Feature-attribution and explainability toolkit (`InterpretabilityAnalyzer`) for model predictions:

- **SHAP Analysis**: Shapley-value feature contributions with background-dataset sampling
- **LIME Analysis**: local surrogate-model explanations via perturbation-based neighborhood sampling
- **Feature Attribution**: Integrated Gradients, Gradient×Input, SmoothGrad, Gradient SHAP, DeepLIFT, LRP, Guided Backprop, Grad-CAM/Grad-CAM++/Score-CAM, Expected Gradients, Attention Rollout, Path Integrated Gradients
- **Counterfactual Analysis**: minimal-change counterfactual generation, feature-sensitivity and decision-boundary analysis, actionable insights
- **Attention Pattern Analysis**: per-layer/per-head attention statistics, head-specialization classification, attention-flow tracing

### Simulation & Robustness Testing

Systematic model-behavior probing (`SimulationAnalyzer` in `simulation_tools`):

- **What-If Analysis**: scenario generation, impact analysis, feature-sensitivity and decision-boundary exploration
- **Perturbation Testing**: robustness scoring across perturbation intensities, sensitivity-hotspot identification, failure-mode analysis
- **Adversarial Probing**: adversarial-example generation (FGSM/PGD/CW/DeepFool), attack-success analysis, certified-robustness estimation, defense recommendations
- **Edge Case Discovery**: automated edge-case search, classification, coverage analysis, and risk assessment

### Guided Debugging

Step-by-step debugging wizard (`GuidedDebugger`) that walks through health check, gradient analysis, architecture analysis, memory profiling, performance profiling, and anomaly detection in sequence, with progress tracking and per-step skip/reset control.

### Interactive Tutorial

Lesson-based walkthrough mode (`TutorialMode`) for onboarding new users to the debugging toolkit, with per-lesson objectives, example code, tips, and progress tracking.

### Export and Integration

- **Netron Export**: ONNX-compatible export for the [Netron](https://netron.app) model visualizer
- **Excel (.xlsx) Export**: genuine Office Open XML (OOXML) workbook generation — `[Content_Types].xml`, relationships, workbook, and worksheet parts written via COOLJAPAN's pure-Rust `oxiarc-archive` crate (`oxiarc_archive::zip::ZipWriter`), replacing the previous CSV-with-`.xlsx`-extension placeholder

## Quick Start

### Basic Usage

```rust
use trustformers_debug::{debug_session, DebugConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a debug session
    let mut debug_session = debug_session();
    debug_session.start().await?;

    // Inspect a tensor (use plain Vec, not ndarray)
    let tensor: Vec<f64> = (0..1000).map(|i| i as f64 / 999.0).collect();
    let tensor_id = debug_session.tensor_inspector_mut()
        .inspect_tensor(&tensor, "my_tensor", Some("layer_1"), Some("forward"))?;

    // Record gradient flow
    let gradients = vec![0.01, 0.02, 0.015, 0.008]; // Sample gradients
    debug_session.gradient_debugger_mut()
        .record_gradient_flow("layer_1", &gradients)?;

    // Generate report
    let report = debug_session.stop().await?;
    println!("Debug Summary: {:?}", report.summary());

    Ok(())
}
```

### Advanced Configuration

```rust
use trustformers_debug::{DebugConfig, DebugSession};

let config = DebugConfig {
    enable_tensor_inspection: true,
    enable_gradient_debugging: true,
    enable_model_diagnostics: true,
    enable_visualization: true,
    max_tracked_tensors: 1000,
    max_gradient_history: 200,
    output_dir: Some("./debug_output".to_string()),
    sampling_rate: 0.1, // Sample 10% of operations for performance
};

let mut debug_session = DebugSession::new(config);
```

### Using Debugging Hooks

```rust
use trustformers_debug::{HookBuilder, HookTrigger, HookAction, AlertSeverity};

let mut debug_session = debug_session();

// Create a hook to monitor specific layers
let hook = HookBuilder::new("Layer Monitor")
    .trigger(HookTrigger::EveryForward)
    .action(HookAction::InspectTensor)
    .layer_patterns(vec!["attention.*".to_string()])
    .max_executions(100)
    .build();

let hook_id = debug_session.hooks_mut().register_hook(hook)?;

// Execute hooks during model execution
let tensor_data = vec![1.0, 2.0, 3.0, 4.0];
let results = debug_session.hooks_mut().execute_hooks(
    "attention_layer_1",
    &tensor_data,
    &[2, 2],
    true, // is_forward
    None
);
```

### Visualization

```rust
use trustformers_debug::{DebugVisualizer, VisualizationConfig};

let config = VisualizationConfig {
    output_directory: "./plots".to_string(),
    ..Default::default()
};

let mut visualizer = DebugVisualizer::new(config);

// Plot tensor distribution
let values = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2];
visualizer.plot_tensor_distribution("weights", &values, 10)?;

// Plot gradient flow over time
let steps = vec![0, 1, 2, 3, 4];
let gradients = vec![0.1, 0.08, 0.06, 0.04, 0.03];
visualizer.plot_gradient_flow("layer1", &steps, &gradients)?;

// Create dashboard
let plot_names = visualizer.get_plot_names();
let dashboard = visualizer.create_dashboard(&plot_names)?;
```

### Flame Graph Generation

```rust
use trustformers_debug::profiler::FlameGraphBuilder;

let mut builder = FlameGraphBuilder::new();
builder.record_frame("forward", "transformer_block_0", 15_200);
builder.record_frame("forward/attention", "multi_head_attn", 9_800);
builder.record_frame("forward/ffn", "feed_forward", 5_400);

// Write Inferno-compatible collapsed stack format
builder.write_to_file("/tmp/trustformers_flame.txt")?;
// Run: inferno-flamegraph < /tmp/trustformers_flame.txt > flame.svg
```

### Interpretability Analysis (SHAP)

```rust
use std::collections::HashMap;
use trustformers_debug::{InterpretabilityAnalyzer, InterpretabilityConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

    let mut instance: HashMap<String, f64> = HashMap::new();
    instance.insert("feature_a".to_string(), 0.7);
    instance.insert("feature_b".to_string(), 1.2);

    let model_predictions = vec![0.65, 0.70, 0.68];
    let background_data = vec![instance.clone()];

    // SHAP feature attribution for this instance
    let shap_result = analyzer
        .analyze_shap(&instance, &model_predictions, &background_data)
        .await?;
    println!("Top SHAP feature: {:?}", shap_result.top_positive_features.first());

    // Aggregate report across all analyses run so far
    let report = analyzer.generate_report().await?;
    println!("SHAP analyses recorded: {}", report.shap_analyses_count);

    Ok(())
}
```

## Key Components

### Tensor Inspector

The tensor inspector provides detailed analysis of tensor values:

- **Statistical Analysis**: Mean, standard deviation, min/max, norms
- **Distribution Analysis**: Histograms, percentiles, outlier detection
- **Quality Checks**: NaN/infinite value detection, sparsity analysis
- **Memory Tracking**: Memory usage per tensor, leak detection

```rust
let tensor_id = debug_session.tensor_inspector_mut()
    .inspect_tensor(&my_tensor, "layer_weights", Some("linear_1"), Some("parameter"))?;

// Get tensor information
let info = debug_session.tensor_inspector().get_tensor_info(tensor_id);
println!("Tensor stats: {:?}", info.unwrap().stats);
```

### Gradient Debugger

Monitor gradient flow to identify training problems:

- **Flow Tracking**: Record gradient norms, means, and patterns
- **Problem Detection**: Vanishing/exploding gradients, dead neurons
- **Trend Analysis**: Gradient evolution over training steps
- **Health Assessment**: Overall gradient flow health scoring

```rust
// Record gradients for a layer
debug_session.gradient_debugger_mut()
    .record_gradient_flow("transformer_block_0", &gradient_values)?;

// Analyze gradient health
let analysis = debug_session.gradient_debugger().analyze_gradient_flow();
println!("Gradient health: {:?}", analysis.overall_health);
```

### Model Diagnostics

Comprehensive model-level analysis:

- **Training Dynamics**: Convergence analysis, stability assessment
- **Architecture Evaluation**: Parameter efficiency, layer analysis
- **Performance Tracking**: Training metrics, memory usage
- **Anomaly Detection**: Performance degradation, instability detection

```rust
// Record model architecture
let arch_info = ModelArchitectureInfo {
    total_parameters: 125_000_000,
    trainable_parameters: 125_000_000,
    model_size_mb: 500.0,
    layer_count: 24,
    // ... other fields
};
debug_session.model_diagnostics_mut().record_architecture(arch_info);

// Record training metrics
let metrics = ModelPerformanceMetrics {
    training_step: 100,
    loss: 0.5,
    accuracy: Some(0.85),
    learning_rate: 1e-4,
    // ... other fields
};
debug_session.model_diagnostics_mut().record_performance(metrics);
```

### Performance Profiler

Identify performance bottlenecks:

- **Execution Timing**: Layer and operation profiling
- **Memory Analysis**: Memory usage patterns and leaks (deadlock-safe scoped locks)
- **Bottleneck Detection**: Automatic performance issue identification
- **Flame Graph Export**: Inferno-compatible collapsed stack format
- **Optimization Recommendations**: Actionable performance suggestions

```rust
let profiler = debug_session.profiler_mut();

// Profile layer execution
profiler.start_timer("layer_forward");
// ... layer computation ...
let duration = profiler.end_timer("layer_forward");

// Record detailed layer metrics
profiler.record_layer_execution(
    "transformer_block_0",
    "TransformerBlock",
    forward_time,
    Some(backward_time),
    memory_usage,
    parameter_count
);
```

## Terminal Visualization

For headless environments (`headless` feature), use Ratatui-based TUI dashboards or ASCII-based visualizations:

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

## Integration with Training

Integrate debugging seamlessly into your training loop:

```rust
use trustformers_debug::{debug_session, ModelPerformanceMetrics};

let mut debug_session = debug_session();
debug_session.start().await?;

for epoch in 0..num_epochs {
    for (step, batch) in train_dataloader.enumerate() {
        // Forward pass with tensor inspection
        let output = model.forward(&batch);
        debug_session.tensor_inspector_mut()
            .inspect_tensor(&output, "model_output", None, Some("forward"))?;

        // Backward pass with gradient tracking
        let loss = criterion(&output, &batch.labels);
        loss.backward();

        for (name, param) in model.named_parameters() {
            if let Some(grad) = param.grad() {
                let grad_values: Vec<f64> = grad.iter().cloned().collect();
                debug_session.gradient_debugger_mut()
                    .record_gradient_flow(name, &grad_values)?;
            }
        }

        // Record performance metrics
        if step % 100 == 0 {
            let metrics = ModelPerformanceMetrics {
                training_step: step,
                loss: loss.item(),
                accuracy: calculate_accuracy(&output, &batch.labels),
                learning_rate: optimizer.get_lr(),
                batch_size: batch.size(),
                throughput_samples_per_sec: calculate_throughput(),
                memory_usage_mb: get_memory_usage(),
                gpu_utilization: get_gpu_utilization(),
                timestamp: chrono::Utc::now(),
            };
            debug_session.model_diagnostics_mut().record_performance(metrics);
        }
    }
}

// Generate comprehensive debug report
let report = debug_session.stop().await?;
```

## Best Practices

### Performance Considerations

1. **Sampling**: Use sampling rate < 1.0 for expensive operations
2. **Selective Monitoring**: Monitor only critical layers during training
3. **Batch Processing**: Process debug operations in batches when possible
4. **Memory Management**: Clear debug data periodically for long training runs
5. **Feature Flags**: Enable only needed features to minimize binary size and overhead

### Debugging Workflow

1. **Start Simple**: Begin with basic tensor inspection and gradient monitoring
2. **Add Hooks**: Use automated hooks for continuous monitoring
3. **Analyze Patterns**: Look for trends in gradient flow and model metrics
4. **Visualize Results**: Create dashboards for comprehensive analysis
5. **Act on Insights**: Apply recommendations from debug reports

### Common Use Cases

- **Training Instability**: Use gradient debugger to identify vanishing/exploding gradients
- **Memory Issues**: Monitor tensor memory usage with deadlock-safe profiling
- **Performance Optimization**: Profile layer execution and generate flame graphs
- **Model Health**: Track training dynamics and convergence patterns
- **Debugging New Architectures**: Comprehensive monitoring during model development
- **Headless Servers**: Use `headless` feature with Ratatui for terminal dashboards

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `debug_session_demo.rs`: Complete debugging workflow examples
- `flame_graph_demo.rs`: Generating and viewing flame graphs
- `headless_dashboard_demo.rs`: Terminal TUI dashboard with Ratatui
- `vs_code_integration_demo.rs`: VS Code diagnostic JSON output

## Contributing

We welcome contributions to improve TrustformeRS Debug! Please see the main repository contributing guidelines.

## License

Licensed under Apache License, Version 2.0 ([LICENSE](../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).
