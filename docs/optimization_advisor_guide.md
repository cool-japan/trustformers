# Performance Optimization Advisor Guide

The TrustformeRS Performance Optimization Advisor analyzes your model and runtime configuration to provide actionable optimization suggestions tailored to your hardware and use case.

## Overview

The Optimization Advisor combines:
- Model architecture analysis
- Performance profiling data
- Hardware capabilities detection
- Best practices knowledge base

To provide specific, prioritized optimization recommendations with expected performance improvements.

## Quick Start

```rust
use trustformers_core::performance::{OptimizationAdvisor, AnalysisContext, HardwareInfo};

// Create advisor
let advisor = OptimizationAdvisor::new();

// Create analysis context
let context = AnalysisContext {
    model_graph: Some(model_graph),
    profile_results: Some(profile),
    latency_metrics: Some(latency),
    memory_metrics: Some(memory),
    throughput_metrics: Some(throughput),
    hardware_info: HardwareInfo::default(),
    current_config: config,
};

// Get optimization report
let report = advisor.analyze(&context)?;

// Save as markdown
std::fs::write("optimizations.md", report.to_markdown())?;
```

## Analysis Context

The advisor analyzes multiple aspects of your system:

### 1. Model Architecture
```rust
// Provide model graph for architecture analysis
let mut visualizer = AutoVisualizer::new();
let graph = visualizer.visualize_bert_model(12)?;

context.model_graph = Some(graph);
```

### 2. Performance Metrics
```rust
// Add profiling results
let profile = profiler.get_results();
context.profile_results = Some(profile);

// Add latency metrics
context.latency_metrics = Some(LatencyMetrics {
    mean_ms: 50.0,
    p99_ms: 100.0,
    // ...
});
```

### 3. Hardware Information
```rust
let hardware = HardwareInfo {
    cpu_model: Some("Intel Xeon Gold 6330".to_string()),
    cpu_cores: 28,
    gpu_model: Some("NVIDIA A100".to_string()),
    gpu_memory_mb: Some(40960),
    system_memory_mb: 256000,
    simd_capabilities: vec!["AVX512".to_string()],
};
```

### 4. Current Configuration
```rust
let config = HashMap::from([
    ("mode".to_string(), "inference".to_string()),
    ("batch_size".to_string(), "1".to_string()),
    ("quantization".to_string(), "false".to_string()),
]);
```

## Optimization Categories

The advisor provides suggestions in several categories:

### Architecture Optimizations
- Sparse attention patterns for long sequences
- Model pruning and distillation
- Layer fusion opportunities

### Memory Optimizations
- Memory fragmentation reduction
- Gradient checkpointing
- KV-cache for inference

### Compute Optimizations
- Kernel fusion
- Mixed precision training
- Flash Attention

### Quantization
- INT8/INT4 quantization
- Dynamic quantization
- SmoothQuant for LLMs

### Parallelization
- Multi-threading configuration
- Data parallelism
- Pipeline parallelism

### Hardware-Specific
- GPU tensor cores utilization
- SIMD optimizations
- NUMA-aware allocation

### Data Pipeline
- Dynamic batching
- Prefetching strategies
- I/O optimization

## Understanding Suggestions

Each suggestion includes:

```rust
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub impact: ImpactLevel,        // Critical, High, Medium, Low
    pub difficulty: Difficulty,      // Easy, Medium, Hard
    pub title: String,
    pub description: String,
    pub expected_improvement: PerformanceImprovement,
    pub implementation_steps: Vec<String>,
    pub code_examples: Option<Vec<CodeExample>>,
    pub warnings: Vec<String>,
}
```

### Impact Levels
- **Critical**: Must-fix performance issues
- **High**: Significant performance gains (>30%)
- **Medium**: Moderate improvements (10-30%)
- **Low**: Minor optimizations (<10%)

### Difficulty Levels
- **Easy**: Configuration changes
- **Medium**: Code modifications
- **Hard**: Architectural changes

## Working with Reports

### Get High-Impact Suggestions
```rust
let critical = report.high_impact_suggestions();
for suggestion in critical {
    println!("{}: {}", suggestion.title, suggestion.description);
}
```

### Filter by Category
```rust
let memory_opts = report.suggestions_by_category(
    OptimizationCategory::Memory
);
```

### Get Easy Wins
```rust
let easy_wins = report.easy_suggestions();
```

### Export Report
```rust
// Markdown format
let markdown = report.to_markdown();

// JSON format
let json = serde_json::to_string_pretty(&report)?;
```

## Built-in Optimization Rules

### 1. Attention Optimization
Detects long sequences and suggests sparse attention:
```rust
// Triggered when sequence length > 512
// Suggests: Sparse attention, Linear attention variants
// Expected: 30% latency reduction, 40% memory reduction
```

### 2. Memory Fragmentation
Identifies memory fragmentation issues:
```rust
// Triggered when peak memory > 2x current usage
// Suggests: Memory pools, tensor recycling
// Expected: 30% memory reduction
```

### 3. Quantization
Recommends quantization for large models:
```rust
// Triggered for models > 100M parameters
// Suggests: INT8/INT4 quantization
// Expected: 75% memory reduction, 40% speedup
```

### 4. Parallelization
Detects underutilized CPU cores:
```rust
// Triggered when cores > 4 and parallelization disabled
// Suggests: Enable multi-threading
// Expected: 50% latency reduction
```

### 5. Kernel Fusion
Identifies fusable operations:
```rust
// Triggered by many small operations
// Suggests: Fused kernels
// Expected: 20% latency reduction
```

### 6. KV-Cache
Recommends caching for inference:
```rust
// Triggered in inference mode without cache
// Suggests: Enable KV-cache
// Expected: 3-5x generation speedup
```

### 7. Batch Size
Detects suboptimal batching:
```rust
// Triggered by batch_size=1 on GPU
// Suggests: Increase batch size
// Expected: 3x throughput increase
```

### 8. Mixed Precision
Checks GPU capabilities:
```rust
// Triggered on Ampere+ GPUs without FP16
// Suggests: Enable mixed precision
// Expected: 2x speedup, 50% memory reduction
```

### 9. Flash Attention
Recommends efficient attention:
```rust
// Triggered for attention on compatible GPUs
// Suggests: Flash Attention implementation
// Expected: 2-4x attention speedup
```

### 10. Gradient Checkpointing
Memory-compute tradeoff:
```rust
// Triggered by high memory usage
// Suggests: Enable gradient checkpointing
// Expected: 50% memory reduction, 20% slowdown
```

## Custom Optimization Rules

Create custom rules for your specific use cases:

```rust
use trustformers_core::performance::{OptimizationRule, OptimizationSuggestion};

struct MyCustomRule;

impl OptimizationRule for MyCustomRule {
    fn analyze(&self, context: &AnalysisContext) 
        -> Result<Option<OptimizationSuggestion>> {
        
        // Check for specific conditions
        if let Some(graph) = &context.model_graph {
            if /* your condition */ {
                return Ok(Some(OptimizationSuggestion {
                    id: "custom_optimization".to_string(),
                    category: OptimizationCategory::Architecture,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Medium,
                    title: "Your Optimization".to_string(),
                    // ... other fields
                }));
            }
        }
        
        Ok(None)
    }
}

// Add to advisor
let mut advisor = OptimizationAdvisor::new();
advisor.add_rule(Box::new(MyCustomRule));
```

## Example: Full Analysis Pipeline

```rust
use trustformers_core::performance::*;

fn optimize_model(model: &Model) -> Result<()> {
    // 1. Profile the model
    let mut profiler = PerformanceProfiler::new();
    profiler.start_profiling();
    
    // Run inference
    let output = model.forward(&input)?;
    
    let profile = profiler.stop_profiling();
    
    // 2. Collect metrics
    let latency = measure_latency(&model)?;
    let memory = measure_memory(&model)?;
    let throughput = measure_throughput(&model)?;
    
    // 3. Get model graph
    let graph = model.to_graph()?;
    
    // 4. Create context
    let context = AnalysisContext {
        model_graph: Some(graph),
        profile_results: Some(profile),
        latency_metrics: Some(latency),
        memory_metrics: Some(memory),
        throughput_metrics: Some(throughput),
        hardware_info: HardwareInfo::detect()?,
        current_config: model.get_config(),
    };
    
    // 5. Get recommendations
    let advisor = OptimizationAdvisor::new();
    let report = advisor.analyze(&context)?;
    
    // 6. Apply easy wins
    for suggestion in report.easy_suggestions() {
        println!("Applying: {}", suggestion.title);
        // Apply suggestion...
    }
    
    // 7. Save full report
    std::fs::write(
        "optimization_report.md", 
        report.to_markdown()
    )?;
    
    Ok(())
}
```

## Best Practices

### 1. Iterative Optimization
Apply optimizations incrementally:
```rust
// 1. Start with easy, high-impact changes
// 2. Measure impact after each change
// 3. Move to more complex optimizations
```

### 2. Validation
Always validate after optimization:
```rust
// Before optimization
let baseline_accuracy = evaluate_model(&model)?;

// Apply optimization
apply_optimization(&mut model, &suggestion)?;

// Validate
let new_accuracy = evaluate_model(&model)?;
assert!((baseline_accuracy - new_accuracy).abs() < 0.01);
```

### 3. Profile-Guided Optimization
Use real workload data:
```rust
// Profile on representative data
let profile = profile_on_real_data(&model)?;

// Analyze with actual profile
context.profile_results = Some(profile);
```

### 4. Hardware-Aware Optimization
Consider deployment hardware:
```rust
// Development machine
let dev_hardware = HardwareInfo::detect()?;

// Production hardware
let prod_hardware = HardwareInfo {
    gpu_model: Some("NVIDIA T4".to_string()),
    gpu_memory_mb: Some(16384),
    // ...
};

// Optimize for production
context.hardware_info = prod_hardware;
```

## Common Optimization Workflows

### Inference Optimization
```rust
let context = AnalysisContext {
    // ... other fields
    current_config: HashMap::from([
        ("mode".to_string(), "inference".to_string()),
    ]),
};

// Will suggest: KV-cache, quantization, batch optimization
```

### Training Optimization
```rust
let context = AnalysisContext {
    // ... other fields
    current_config: HashMap::from([
        ("mode".to_string(), "training".to_string()),
    ]),
};

// Will suggest: Mixed precision, gradient checkpointing
```

### Memory-Constrained Optimization
```rust
let context = AnalysisContext {
    memory_metrics: Some(high_memory_usage),
    hardware_info: HardwareInfo {
        system_memory_mb: 8192, // Limited memory
        // ...
    },
    // ...
};

// Will prioritize memory reduction suggestions
```

## Troubleshooting

### No Suggestions Generated
- Ensure context has sufficient information
- Check that hardware info is populated
- Verify current_config reflects actual setup

### Suggestions Don't Apply
- Some suggestions are hardware-specific
- Check warnings for prerequisites
- Validate hardware capabilities

### Performance Regression
- Apply optimizations one at a time
- Measure impact after each change
- Some optimizations trade off (e.g., memory vs compute)

## Integration Examples

### With Continuous Benchmarking
```rust
let benchmark = ContinuousBenchmark::new(config);
let results = benchmark.run()?;

// Use benchmark results for optimization
let context = AnalysisContext {
    latency_metrics: Some(results.latency),
    throughput_metrics: Some(results.throughput),
    // ...
};
```

### With Model Export
```rust
// Optimize before export
let report = advisor.analyze(&context)?;

// Apply quantization if suggested
if report.suggestions.iter().any(|s| s.id == "quantization") {
    model = quantize_model(model)?;
}

// Export optimized model
exporter.export(&model, "optimized_model.onnx")?;
```

### With A/B Testing
```rust
// Test optimization impact
let ab_test = ABTestManager::new();

// Original model
let variant_a = Variant::new("original", model.clone());

// Optimized model
let optimized = apply_optimizations(model, &report)?;
let variant_b = Variant::new("optimized", optimized);

let results = ab_test.run_test(vec![variant_a, variant_b])?;
```

## Performance Tips

1. **Run analysis during development** - Catch issues early
2. **Profile on target hardware** - Optimizations are hardware-specific
3. **Consider the full pipeline** - Include data loading and preprocessing
4. **Monitor in production** - Real-world patterns may differ
5. **Document optimizations** - Track what was applied and why

## Next Steps

- See [Performance Tuning](./performance_tuning.md) for detailed optimization techniques
- Check [Benchmarking Guide](./benchmarking.md) for performance measurement
- Read [Quantization Guide](./quantization.md) for compression techniques
- Explore [Parallelization Guide](./parallelization.md) for scaling strategies