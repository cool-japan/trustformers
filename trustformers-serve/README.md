# TrustformeRS Serve

**Version:** 0.2.1 (unreleased) | **Status:** Stable | **Tests:** 5,572 passed, 1 skipped, 0 failed as of 2026-08-18, not independently re-run this pass — see root `TODO.md` for the current workspace-wide baseline (20,629 passed / 43 skipped / 0 failed, 2026-08-24) | **SLoC:** 273,756 (`tokei`, verified 2026-08-24 — down from 278,397 on 2026-08-18, mainly the `resource_manager/` placeholder tree, 5,972 lines, deleted outright this cycle) | **Updated:** 2026-08-24

High-performance inference server for TrustformeRS models with advanced batching, multi-protocol APIs, cloud-native deployment, and comprehensive observability.

## Features

### Cargo Features

Only three things in this crate are actually opt-in Cargo features:

```toml
[features]
default = []
kafka = ["dep:rdkafka"]       # requires cmake + system librdkafka
lambda = ["dep:lambda-web"]   # kept opt-in: lambda-web depends on banned brotli/brotli-decompressor
swagger-ui = ["dep:utoipa-swagger-ui"]  # kept opt-in: build-dep pulls banned `zip` crate; CDN-hosted /docs Swagger UI works without this feature
```

`default = []` does **not** mean a minimal/lightweight build — it only means these three extras are off by default. Everything else is an unconditional dependency compiled in regardless of feature flags: all AWS SDK crates (SageMaker, SQS, CloudWatch, Lambda client), `async-nats`, `lapin` (RabbitMQ), `redis`, the full OpenTelemetry stack, `tonic`/`tonic-prost` (gRPC), `async-graphql`, `axum` (REST), and `utoipa` (OpenAPI spec generation). There is no `aws` feature to enable — the AWS SDKs are always compiled in.
>
> **Updated 2026-08-18**: the Azure crates (`azure_core`, `azure_identity`, `azure_mgmt_machinelearningservices`, `azure_mgmt_web`) and GCP crates (`google-cloud-functions-v2`, `google-cloud-gax`) this paragraph used to list here were removed from `Cargo.toml` — none had a call site anywhere in `src/`, and the Azure/GCP serverless providers return a structured `NotImplemented` error rather than calling any SDK, so no capability was lost. `AzureMachineLearningProvider`'s inference path (`src/cloud_providers/rest.rs`) calls Azure's REST API directly over `reqwest` instead of the SDK, so Azure inference still works without that dependency.

> **Updated 2026-08-24**: `src/resource_manager/`, a placeholder resource-management tree that fabricated network ports (always `vec![8080]`), temp-directory paths (never actually created), database connection ids (synthesized), GPU device stats (echoed indices plus a constant 8192MB), and monitoring efficiency (a constant `0.75`), is deleted outright. The crate's public `ResourceManagementSystem` name — previously the unprefixed alias for that fake tree — now resolves directly to the real, tested `resource_management/` tree instead (`ModularResourceManagementSystem` kept as a compatibility alias for the same type).

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
- **gRPC (Tonic)**: High-throughput binary protocol with bidirectional streaming (proto compilation and serving restored in 0.1.4, migrated to the tonic 0.14 split `tonic-build`/`tonic-prost-build` API)
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

Asynchronous request ingestion via a shared `MessageQueueProducer`/`MessageQueueConsumer` trait abstraction:

- **Apache Kafka** (`kafka` feature, requires system librdkafka): production-ready wire-protocol implementation — high-throughput topic-based routing, consumer groups, exactly-once semantics.
- **RabbitMQ, Redis Streams, NATS, AWS SQS**: as of 2026-08-18 these call real client libraries (`src/message_queue/{rabbitmq,nats,redis_streams,sqs}.rs` use `lapin`/`async_nats`/`redis`/the AWS SQS SDK respectively) rather than the no-op scaffold this section described through 0.2.0. Not independently exercised against a live broker in this documentation pass — see [TODO.md](TODO.md) for the exact verification state.

### Cloud Provider Support

A unified multi-cloud provider abstraction (`CloudProvider` trait) with health-check orchestration and unified request/response types is implemented and tested across AWS, GCP, and Azure. Deployment-side integrations are real.

- **AWS**: EKS deployment, S3 model storage, CloudWatch metrics. SageMaker inference goes through the provider abstraction.
- **GCP**: GKE autopilot, GCS model storage, Cloud Monitoring. Vertex AI inference goes through the provider abstraction.
- **Azure**: AKS deployment, Blob Storage, Azure Monitor. Azure ML inference (`AzureMachineLearningProvider`) makes a real `reqwest` REST call to the configured endpoint (`POST {base}/chat/completions` with an `api-key` header) — confirmed by reading `src/cloud_providers/rest.rs` on 2026-08-18. AWS/GCP inference: `grep -c "Mock response\|example.com/endpoint" src/cloud_providers.rs` returns 0 (confirmed 2026-08-24), matching this crate's own `CHANGELOG.md` entry that all 6 provider integrations lost the shared canned-response fabrication — still not individually exercised against a live AWS/GCP endpoint in this documentation pass.

See [TODO.md](TODO.md) for the current verification state of message queues and cloud providers.

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
