# TrustformeRS Debug

Advanced debugging and analysis tools for TrustformeRS machine learning models.

## Overview

TrustformeRS Debug provides comprehensive debugging capabilities for deep learning models, including tensor inspection, gradient flow analysis, model diagnostics, performance profiling, and automated debugging hooks. These tools help identify training issues, performance bottlenecks, and model problems early in the development process.

## Features

### 🔍 Tensor Inspector
- **Real-time Analysis**: Inspect tensor statistics, distributions, and patterns
- **Anomaly Detection**: Automatically detect NaN, infinite, and extreme values
- **Memory Tracking**: Monitor memory usage and identify memory leaks
- **Comparison Tools**: Compare tensors across different training steps or layers

### 📈 Gradient Debugger
- **Flow Analysis**: Track gradient flow through model layers
- **Problem Detection**: Identify vanishing and exploding gradients
- **Dead Neuron Detection**: Find layers with inactive neurons
- **Trend Analysis**: Monitor gradient patterns over training steps

### 🏥 Model Diagnostics
- **Training Dynamics**: Analyze convergence, stability, and learning efficiency
- **Architecture Analysis**: Evaluate model structure and parameter efficiency
- **Performance Monitoring**: Track training metrics and detect anomalies
- **Health Assessment**: Overall model health scoring and recommendations

### 🎣 Debugging Hooks
- **Automated Monitoring**: Set up conditional debugging triggers
- **Layer-specific Tracking**: Monitor specific layers or operations
- **Custom Callbacks**: Implement custom debugging logic
- **Alert System**: Real-time notifications for debugging events

### 📊 Visualization
- **Interactive Plots**: Generate comprehensive debugging visualizations
- **Dashboard Creation**: Build debugging dashboards with multiple plots
- **Terminal Visualization**: ASCII plots for headless environments
- **Export Capabilities**: Save plots and data in multiple formats

### ⚡ Performance Profiler
- **Execution Timing**: Profile layer and operation execution times
- **Memory Profiling**: Track memory usage patterns
- **Bottleneck Detection**: Identify performance bottlenecks automatically
- **Optimization Suggestions**: Get recommendations for performance improvements

## Quick Start

### Basic Usage

```rust
use trustformers_debug::{debug_session, DebugConfig};
use ndarray::Array;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a debug session
    let mut debug_session = debug_session();
    debug_session.start().await?;

    // Inspect a tensor
    let tensor = Array::linspace(0.0, 1.0, 1000);
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
- **Memory Analysis**: Memory usage patterns and leaks
- **Bottleneck Detection**: Automatic performance issue identification
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

For headless environments, use ASCII-based visualizations:

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

### Debugging Workflow

1. **Start Simple**: Begin with basic tensor inspection and gradient monitoring
2. **Add Hooks**: Use automated hooks for continuous monitoring
3. **Analyze Patterns**: Look for trends in gradient flow and model metrics
4. **Visualize Results**: Create dashboards for comprehensive analysis
5. **Act on Insights**: Apply recommendations from debug reports

### Common Use Cases

- **Training Instability**: Use gradient debugger to identify vanishing/exploding gradients
- **Memory Issues**: Monitor tensor memory usage and detect leaks
- **Performance Optimization**: Profile layer execution and identify bottlenecks
- **Model Health**: Track training dynamics and convergence patterns
- **Debugging New Architectures**: Comprehensive monitoring during model development

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `debug_session_demo.rs`: Complete debugging workflow examples
- More examples coming soon...

## Contributing

We welcome contributions to improve TrustformeRS Debug! Please see the main repository contributing guidelines.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.