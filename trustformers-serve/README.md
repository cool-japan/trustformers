# TrustformeRS Serve

High-performance inference server for TrustformeRS models with advanced batching and optimization capabilities.

## Features

### Dynamic Batching System

The dynamic batching system automatically groups inference requests to maximize throughput while maintaining low latency. Key features include:

- **Intelligent Request Aggregation**: Automatically collects requests into optimal batch sizes
- **Priority-based Scheduling**: Process critical requests first with configurable priority levels
- **Adaptive Batching**: Dynamically adjusts batch size and timeout based on load patterns
- **Memory-aware Batching**: Prevents OOM by tracking memory usage per batch
- **Continuous Batching**: Special mode for LLM text generation with KV cache management
- **Sequence Bucketing**: Groups similar-length sequences to minimize padding overhead

### Configuration Options

```rust
use trustformers_serve::{BatchingConfig, BatchingMode, OptimizationTarget};
use std::time::Duration;

let config = BatchingConfig {
    max_batch_size: 32,                           // Maximum requests per batch
    min_batch_size: 4,                            // Minimum batch size before timeout
    max_wait_time: Duration::from_millis(50),     // Maximum wait time for batch formation
    enable_adaptive_batching: true,               // Enable load-based adaptation
    mode: BatchingMode::Dynamic,                  // Batching mode
    optimization_target: OptimizationTarget::Balanced,  // Optimization goal
    memory_limit: Some(1024 * 1024 * 100),       // 100MB memory limit
    enable_priority_scheduling: true,             // Enable priority-based scheduling
    ..Default::default()
};
```

### Batching Modes

1. **Fixed**: Constant batch size
2. **Dynamic**: Variable batch size based on queue depth
3. **Adaptive**: Automatically adjusts based on load patterns
4. **Continuous**: Special mode for LLM generation with incremental decoding

### Optimization Targets

- **Throughput**: Maximize requests per second
- **Latency**: Minimize response time
- **Balanced**: Balance between throughput and latency
- **Cost**: Optimize for cloud deployment costs

### Performance Monitoring

Built-in metrics collection provides real-time insights:

- Request throughput (req/s)
- Latency percentiles (p50, p95, p99)
- Batch size distribution
- GPU/memory utilization
- Queue depth and wait times
- Automatic optimization suggestions

## Usage Example

```rust
use trustformers_serve::{
    DynamicBatchingService, BatchingConfig,
    Request, RequestInput, Priority,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Configure batching
    let config = BatchingConfig::default();
    
    // Create and start service
    let service = DynamicBatchingService::new(config);
    service.start().await?;
    
    // Submit request
    let request = Request {
        id: RequestId::new(),
        input: RequestInput::Text {
            text: "Hello, world!".to_string(),
            max_length: Some(100),
        },
        priority: Priority::Normal,
        submitted_at: Instant::now(),
        deadline: None,
        metadata: Default::default(),
    };
    
    let result = service.submit_request(request).await?;
    println!("Result: {:?}", result);
    
    // Get statistics
    let stats = service.get_stats().await;
    println!("Throughput: {:.1} req/s", stats.metrics_summary.throughput_rps);
    
    Ok(())
}
```

## Advanced Features

### Memory-aware Batching

Prevents out-of-memory errors by tracking memory usage:

```rust
let config = BatchingConfig {
    memory_limit: Some(1024 * 1024 * 512), // 512MB limit
    dynamic_config: DynamicBatchConfig {
        memory_aware: true,
        padding_strategy: PaddingStrategy::Minimal,
        enable_bucketing: true,
        bucket_boundaries: vec![128, 256, 512, 1024],
        ..Default::default()
    },
    ..Default::default()
};
```

### Priority Scheduling

Handle critical requests with higher priority:

```rust
let critical_request = Request {
    priority: Priority::Critical,
    deadline: Some(Instant::now() + Duration::from_millis(100)),
    ..default_request
};
```

### Continuous Batching for LLMs

Optimized for text generation with incremental decoding:

```rust
let config = BatchingConfig {
    mode: BatchingMode::Continuous,
    optimization_target: OptimizationTarget::Throughput,
    ..Default::default()
};
```

## Performance Tips

1. **Batch Size**: Start with max_batch_size = 32 and adjust based on GPU memory
2. **Timeout**: Lower timeouts (10-50ms) for latency-sensitive applications
3. **Bucketing**: Enable sequence bucketing to reduce padding overhead
4. **Memory Limits**: Set appropriate memory limits to prevent OOM
5. **Monitoring**: Use built-in metrics to identify optimization opportunities

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `dynamic_batching_demo.rs`: Complete demonstration of all batching features
- More examples coming soon...

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.