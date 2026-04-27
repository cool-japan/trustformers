# TrustformeRS Serve

**Version:** 0.1.1 | **Status:** Stable | **Tests:** 216 | **SLoC:** 206,636 | **Updated:** 2026-04-25

High-performance inference server for TrustformeRS models with advanced batching, multi-protocol APIs, cloud-native deployment, and comprehensive observability.

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

### Multi-Protocol APIs

- **REST (Axum)**: HTTP/1.1 and HTTP/2, streaming via SSE and WebSockets
- **gRPC (Tonic)**: High-throughput binary protocol with bidirectional streaming
- **GraphQL (async-graphql)**: Flexible query API with subscriptions

### SLO Monitoring and Observability

Built-in SLO (Service Level Objective) monitoring with Prometheus metrics exported via `once_cell` lazy statics for zero-cost initialization:

- Request throughput (req/s) and token throughput (tokens/s)
- Latency percentiles (p50, p90, p95, p99) with SLO breach alerting
- Batch size distribution and queue depth histograms
- GPU/memory utilization gauges
- Cache hit rate and eviction counters
- Automatic SLO violation detection and alerting

```rust
use trustformers_serve::monitoring::{SloConfig, SloThresholds};

let slo = SloConfig {
    p99_latency_ms: 200.0,
    p95_latency_ms: 100.0,
    availability_target: 0.999,
    error_rate_threshold: 0.001,
};
```

### Distributed Tracing

Full OpenTelemetry-compatible distributed tracing with Jaeger and Zipkin exporters:

- Per-request span creation with context propagation
- Trace sampling (head-based and tail-based)
- Baggage propagation across service boundaries
- Integration with service mesh (Istio, Linkerd)

### Performance Optimizer with NUMA/Topology Detection

Platform-aware performance optimizer that detects hardware topology to maximize throughput:

- **Linux**: Reads CPU topology via `/sys/devices/system/cpu/` sysfs, NUMA node distances from `/sys/devices/system/node/`
- **macOS**: Queries CPU topology via `sysctl hw.physicalcpu`, `hw.logicalcpu`, `hw.cachesize`
- Automatic thread affinity binding to NUMA nodes
- Memory allocation policy optimized for NUMA topology
- Cache-line aware data structure layout

### Speculative Decoding

Accelerates LLM text generation using draft models:

- Draft model generates candidate tokens in parallel
- Verifier model accepts or rejects in a single forward pass
- Configurable draft length (typically 4–8 tokens)
- Automatic fallback when speculative quality degrades
- Up to 3x throughput improvement for autoregressive models

### Kernel Fusion

GPU kernel optimization to reduce memory bandwidth pressure:

- **Vertical fusion**: Sequential elementwise operations fused into single kernel
- **Horizontal fusion**: Independent parallel operations batched together
- **Producer-consumer fusion**: Eliminates intermediate tensor materialization
- **Multi-pattern fusion**: Combined patterns for attention and FFN blocks
- Reduced kernel launch overhead and improved L2 cache utilization

### Message Queue Integration

Asynchronous request ingestion via:

- **Apache Kafka**: High-throughput topic-based routing, consumer groups, exactly-once semantics
- **RabbitMQ**: AMQP protocol, priority queues, dead-letter exchanges, TTL-based expiry

### Cloud Provider Support

Native integrations for major cloud platforms:

- **AWS**: EKS deployment, SageMaker endpoint compatibility, S3 model storage, CloudWatch metrics
- **GCP**: GKE autopilot, Vertex AI serving, GCS model storage, Cloud Monitoring
- **Azure**: AKS deployment, Azure ML serving, Blob Storage, Azure Monitor

### GDPR Compliance

Data protection and privacy controls:

- Request/response data anonymization with configurable PII redaction
- Right-to-erasure support with audit trail
- Consent management with per-user opt-in/opt-out
- Data processing records (ROPA) generation
- Comprehensive audit logs with tamper-evident storage

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

### Speculative Decoding Configuration

```rust
use trustformers_serve::serving::SpeculativeConfig;

let spec_config = SpeculativeConfig {
    draft_model_path: "/models/gpt2-small".to_string(),
    draft_steps: 5,
    acceptance_threshold: 0.8,
    fallback_on_low_acceptance: true,
};
```

## Performance Tips

1. **Batch Size**: Start with max_batch_size = 32 and adjust based on GPU memory
2. **Timeout**: Lower timeouts (10-50ms) for latency-sensitive applications
3. **Bucketing**: Enable sequence bucketing to reduce padding overhead
4. **Memory Limits**: Set appropriate memory limits to prevent OOM
5. **NUMA Binding**: Enable topology detection for multi-socket servers
6. **Speculative Decoding**: Use for autoregressive generation to improve throughput
7. **Monitoring**: Use built-in SLO metrics to identify SLA violations early

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `dynamic_batching_demo.rs`: Complete demonstration of all batching features
- `speculative_decoding_demo.rs`: Speculative decoding with draft models
- `kafka_integration_demo.rs`: Message queue ingestion patterns
- `cloud_deployment_demo.rs`: AWS/GCP/Azure deployment examples

## License

Licensed under Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).
