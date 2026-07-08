# trustformers-serve TODO List

**Version:** 0.1.4 | **Status:** Stable | **Tests:** ~4,321 | **Public API Items:** 7,319 | **SLoC:** 283,692 | **Updated:** 2026-07-02

## Overview

The `trustformers-serve` crate provides high-performance inference serving infrastructure for production deployment of transformer models. It includes REST/gRPC/GraphQL APIs, dynamic batching, distributed serving, and comprehensive monitoring.

This is the largest crate in the `trustformers` workspace by public API surface: **7,319** public items (top-level `pub fn`/`struct`/`enum`/`trait` declarations plus indented `pub fn` methods inside impl blocks) — for comparison, the sibling `trustformers-training` crate has 846. Counting only top-level declarations (excluding impl-block methods) gives a more conservative 4,999.

**Key Responsibilities:**
- REST API with dynamic batching and caching
- gRPC API for high-throughput serving (proto compilation/serving restored in 0.1.4 — see "gRPC API (Tonic)" below)
- GraphQL API for flexible queries
- Distributed serving with load balancing
- Model management (hot-swapping, versioning, A/B testing)
- Hardware acceleration (CUDA, ROCm, Metal, XLA, Vulkan)
- Kubernetes deployment with autoscaling
- Monitoring and observability (Prometheus with once_cell lazy statics, Jaeger, OpenTelemetry)
- SLO monitoring and breach alerting
- NUMA/topology-aware performance optimizer (Linux sysfs, macOS sysctl)
- Speculative decoding with draft models
- Kernel fusion for GPU operations
- Message queue integration (Kafka — production; RabbitMQ/Redis Streams/NATS/SQS — interface complete, real backend wiring pending)
- Cloud provider support (AWS, GCP, Azure — orchestration layer real; per-provider inference calls simulated pending real SDK integration)
- GDPR compliance

---

## Current Status

### Implementation Status
- [x] **PRODUCTION-READY** - Complete serving infrastructure
- [x] **ZERO COMPILATION ERRORS** - Clean compilation
- [x] **COMPREHENSIVE TESTING** - ~4,321 tests passing, 0 failures (workspace-wide `cargo nextest run --workspace --all-features`, 2026-07-01: 18,102 passed / 0 failed / 119 skipped, 0 clippy warnings, 0 rustdoc warnings)
- [x] **REQUEST QUEUING** - Priority queue with deadline awareness and cancellation (`queue` module)
- [x] **PRIORITY SCHEDULING** - WRR, EDF, fair queuing, priority, and FIFO strategies (`scheduler` module)
- [x] **HARDWARE ACCELERATED** - CUDA, ROCm, Metal support
- [x] **KUBERNETES READY** - Helm charts, autoscaling, monitoring

### Feature Coverage
- **APIs:** REST (Axum), gRPC (Tonic), GraphQL (async-graphql)
- **Performance:** Dynamic batching, result caching, kernel fusion, speculative decoding
- **Distribution:** Load balancing, failover, health checks, disaster recovery
- **Monitoring:** Prometheus metrics (once_cell lazy statics), Jaeger tracing, OpenTelemetry, SLO monitoring
- **Security:** Authentication, TLS, GDPR compliance, encryption
- **Deployment:** Docker, Kubernetes, Helm, service mesh integration
- **Cloud:** AWS (EKS, S3, CloudWatch), GCP (GKE, GCS), Azure (AKS, Blob) — deployment orchestration real; per-provider inference (SageMaker/Vertex AI/Azure ML) simulated pending real SDK integration
- **Messaging:** Kafka (production, feature-gated); RabbitMQ/Redis Streams/NATS/SQS (interface/scaffold, no-op backend — real broker wiring pending)

---

## Completed Features

### API Implementations

#### REST API (Axum)

**High-performance REST API with Axum framework**

- [x] **Endpoints**
  - `/v1/generate` - Text generation
  - `/v1/embeddings` - Text embeddings
  - `/v1/classify` - Text classification
  - `/v1/models` - Model management (list, load, unload)
  - `/health` - Health checks
  - `/metrics` - Prometheus metrics

- [x] **Features**
  - Request validation with serde
  - Streaming responses (SSE, WebSockets)
  - CORS support
  - Compression (gzip, brotli)
  - Rate limiting
  - Authentication middleware

**Example:**
```bash
# Text generation
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100}'

# Stream generation
curl -N -X POST http://localhost:8080/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "stream": true}'
```

---

#### gRPC API (Tonic)

**High-throughput binary protocol**

- [x] **Restored in 0.1.4** — proto compilation and serving re-enabled; build migrated to the tonic 0.14 split `tonic-build`/`tonic-prost-build` API (`build.rs` runs `tonic_prost_build::configure().build_server(true).build_client(true).compile_protos(...)` against `proto/inference.proto`; see workspace `CHANGELOG.md` `[0.1.4] - 2026-07-01`)
- [x] **Services**
  - InferenceService - Model inference
  - ModelService - Model management
  - HealthService - Health checks

- [x] **Features**
  - Protocol Buffers (protobuf)
  - Bidirectional streaming
  - Interceptors for auth/logging
  - Connection pooling
  - Load balancing (client-side and server-side)

**Example:**
```rust
// Client usage
let mut client = InferenceServiceClient::connect("http://[::1]:9090").await?;

let request = tonic::Request::new(GenerateRequest {
    prompt: "Once upon a time".to_string(),
    max_tokens: 100,
    ..Default::default()
});

let response = client.generate(request).await?;
println!("Response: {}", response.into_inner().text);
```

---

#### GraphQL API

**Flexible query-based API**

- [x] **Schema**
  - Query (health, models, metrics)
  - Mutation (generate, load_model, unload_model)
  - Subscription (streaming generation, metrics updates)

- [x] **Features**
  - Introspection
  - Batching
  - DataLoader pattern
  - Field-level authorization

**Example:**
```graphql
# Query health
query {
  health {
    status
    uptime
    modelsLoaded
    activeRequests
  }
}

# Generate text
mutation {
  generate(prompt: "Hello, world!", maxTokens: 50) {
    text
    tokens
    latencyMs
  }
}

# Subscribe to generation stream
subscription {
  streamGenerate(prompt: "Once upon a time") {
    token
    isDone
  }
}
```

---

### Performance Optimization

#### Dynamic Batching

**Automatic request batching for throughput**

- [x] **Strategies**
  - Time-based batching (max wait time)
  - Size-based batching (max batch size)
  - Dynamic batching (adaptive based on load)
  - Priority-based batching

- [x] **Configuration**
  - Configurable batch size (1-256)
  - Timeout (1-1000ms)
  - Priority queues
  - Fairness policies

**Example:**
```rust
let batching_config = BatchingConfig {
    max_batch_size: 32,
    max_wait_time_ms: 10,
    strategy: BatchingStrategy::Dynamic,
    enable_priority: true,
};

let server = TrustformersServer::new(config)
    .with_batching(batching_config)?;
```

---

#### Speculative Decoding

**Accelerated autoregressive generation**

- [x] **Features**
  - Draft model generates N candidate tokens in parallel
  - Verifier model accepts/rejects in single forward pass
  - Configurable draft length (1-16 tokens)
  - Automatic fallback on low acceptance rate
  - Up to 3x throughput improvement

**Example:**
```rust
let spec_config = SpeculativeConfig {
    draft_model_path: "/models/gpt2-small".to_string(),
    draft_steps: 5,
    acceptance_threshold: 0.8,
    fallback_on_low_acceptance: true,
};
```

---

#### Kernel Fusion

**GPU kernel optimization**

- [x] **Fusion Patterns**
  - Vertical fusion (sequential ops)
  - Horizontal fusion (parallel ops)
  - Producer-consumer fusion
  - Multi-pattern fusion

- [x] **Benefits**
  - Reduced kernel launches
  - Improved memory bandwidth
  - Lower latency
  - Higher throughput

---

#### Result Caching

**Multi-tier caching for latency reduction**

- [x] **Cache Tiers**
  - L1: In-memory cache (LRU, LFU, ARC)
  - L2: Redis distributed cache
  - L3: Disk-based cache

- [x] **Features**
  - TTL-based expiration
  - Cache warming
  - Invalidation strategies
  - Compression
  - Sharding

**Example:**
```rust
let cache_config = CacheConfig {
    tiers: vec![
        TierConfig {
            tier_type: TierType::Memory,
            size_mb: 1024,
            eviction: EvictionPolicy::LRU,
        },
        TierConfig {
            tier_type: TierType::Redis,
            size_mb: 10240,
            eviction: EvictionPolicy::LRU,
        },
    ],
    ttl_seconds: 3600,
    enable_compression: true,
};
```

---

### Performance Optimizer with NUMA/Topology Detection

**Platform-aware hardware topology optimization**

- [x] **Linux**
  - CPU topology via `/sys/devices/system/cpu/` sysfs
  - NUMA node distances from `/sys/devices/system/node/`
  - Thread affinity binding per NUMA node

- [x] **macOS**
  - CPU topology via `sysctl hw.physicalcpu`, `hw.logicalcpu`, `hw.cachesize`
  - Unified memory topology detection

- [x] **Features**
  - Automatic thread affinity assignment
  - NUMA-aware memory allocation policies
  - Cache-line aligned data structures

---

### Monitoring and Observability

#### SLO Monitoring with Prometheus

**Comprehensive SLO tracking with once_cell lazy statics**

- [x] **Metrics** (exported via `once_cell::sync::Lazy` for zero-cost initialization)
  - Request count, latency (p50, p90, p95, p99)
  - Throughput (requests/sec, tokens/sec)
  - Error rate
  - Model-specific metrics
  - GPU utilization
  - Memory usage
  - Cache hit rate
  - Batch size histograms
  - Queue depth gauges

- [x] **SLO Breach Alerting**
  - Configurable p99 latency thresholds
  - Error rate threshold monitoring
  - Availability target tracking
  - Webhook and PagerDuty integration for breach notifications

**Example:**
```rust
// Metrics are automatically exported at /metrics via once_cell lazy statics
// Access with Prometheus scrape config:
// - job_name: 'trustformers'
//   static_configs:
//   - targets: ['localhost:8080']
```

---

#### Distributed Tracing

**Request tracing with Jaeger/OpenTelemetry**

- [x] **Features**
  - Span creation for each operation
  - Context propagation
  - Trace sampling
  - Baggage items
  - Integration with Jaeger/Zipkin

**Example:**
```rust
let tracing_config = TracingConfig {
    exporter: TracingExporter::Jaeger,
    jaeger_endpoint: "http://localhost:14268/api/traces".to_string(),
    sampling_rate: 0.1, // 10% sampling
    service_name: "trustformers-serve".to_string(),
};

server.enable_tracing(tracing_config)?;
```

---

### Message Queue Integration

#### Apache Kafka

**High-throughput asynchronous request ingestion — production-ready wire protocol**

- [x] **Features** (feature-gated behind the `kafka` Cargo feature, requires system librdkafka via the `rdkafka` crate)
  - Topic-based routing for request types
  - Consumer group support
  - Exactly-once semantics
  - Configurable partition assignment
  - Back-pressure with bounded queues

#### RabbitMQ / Redis Streams / NATS / AWS SQS

**Trait-based interface/scaffold — not yet wired to a real broker**

- [x] **Interface/orchestration: done** — `MessageQueueProducer`/`MessageQueueConsumer` traits fully implemented for all four backends (`RabbitMQProducer`/`Consumer`, `RedisProducer`/`Consumer`, `NatsProducer`/`Consumer`, `SqsProducer`/`Consumer` in `src/message_queue.rs`), sufficient to exercise the request routing/orchestration layer end-to-end today.
- [ ] **Real backend integration: pending / currently simulated** — all four share one `impl_placeholder_backend!` macro: `send_message`/`send_batch` fabricate a success result with no network I/O, `poll` always returns an empty vec, and transactions/flush/close are no-ops. The `redis` and `lapin` crates are already unconditional Cargo.toml dependencies, but their APIs are never called anywhere in the crate. Previously documented as delivered "AMQP 0-9-1 protocol support / dead-letter exchanges / publisher confirms" — corrected here; that wire-protocol work is still open. See it tracked under [Future Enhancements](#future-enhancements).

---

### Cloud Provider Support

- [x] **Unified provider abstraction: done** — `CloudProvider` trait, health-check orchestration, and unified request/response types implemented and tested across AWS/GCP/Azure/HuggingFace/OpenAI/Anthropic provider stand-ins (`src/cloud_providers.rs`)
- [x] **AWS**: EKS deployment, S3 model storage, CloudWatch metrics
- [x] **GCP**: GKE autopilot, GCS model storage, Cloud Monitoring
- [x] **Azure**: AKS, Blob Storage, Azure Monitor
- [ ] **Real per-provider inference/deployment calls: pending / currently simulated** — `AwsSagemakerProvider`, `GoogleVertexAiProvider`, `AzureMachineLearningProvider` (and the `HuggingFaceProvider`/`OpenAiProvider`/`AnthropicProvider` stand-ins) all share one `impl_provider!` macro whose `inference()` unconditionally returns a fabricated `OutputData::Text("Mock response")` with canned latency/cost/token metadata, and whose `deploy_model()` returns a canned `https://example.com/endpoint` URL, regardless of which provider is selected. The corresponding AWS/GCP/Azure SDK crates are unconditional dependencies but are never invoked by this code path. Previously documented as "SageMaker endpoint compatibility" / "Vertex AI serving compatibility" / "Azure ML serving compatibility" — corrected here; real SDK-backed inference/deployment per provider is still open work. See it tracked under [Future Enhancements](#future-enhancements).

---

### Model Management

#### Hot-Swapping

**Zero-downtime model updates**

- [x] **Features**
  - Atomic model replacement
  - Gradual rollout
  - Rollback support
  - Version tracking

**Example:**
```rust
// Load new model version
server.load_model("gpt2-v2", "/path/to/model")?;

// Swap models atomically
server.swap_model("gpt2", "gpt2-v2")?;

// Rollback if needed
server.rollback_model("gpt2")?;
```

---

#### A/B Testing

**Traffic splitting for model comparison**

- [x] **Features**
  - Percentage-based routing
  - User-based routing
  - Request-based routing
  - Metrics collection per variant

**Example:**
```rust
let ab_config = ABTestConfig {
    variants: vec![
        Variant { model: "gpt2-v1", weight: 0.9 },
        Variant { model: "gpt2-v2", weight: 0.1 },
    ],
    routing_key: RoutingKey::UserId,
};

server.enable_ab_test("gpt2", ab_config)?;
```

---

### Hardware Acceleration

#### CUDA Support

**NVIDIA GPU acceleration**

- [x] **Features**
  - cuDNN integration
  - cuBLAS for GEMM
  - Multi-GPU support
  - CUDA Graphs for optimization
  - Tensor Cores (FP16, INT8)

---

#### ROCm Support

**AMD GPU acceleration**

- [x] **Features**
  - MIOpen integration
  - rocBLAS for GEMM
  - HIP kernels
  - Multi-GPU support

---

#### Metal Support

**Apple Silicon acceleration**

- [x] **Features**
  - Metal Performance Shaders (MPS)
  - Metal compute kernels
  - Unified memory
  - Neural Engine integration

---

### Security

#### Authentication

**Multi-method authentication**

- [x] **Methods**
  - API keys
  - JWT tokens
  - OAuth2
  - mTLS

---

#### TLS/HTTPS

**Encrypted connections**

- [x] **Features**
  - TLS 1.2/1.3 support
  - Certificate management
  - mTLS for client authentication
  - ACME (Let's Encrypt) integration

---

#### GDPR Compliance

**Privacy and data protection**

- [x] **Features**
  - Data anonymization with configurable PII redaction
  - Right to be forgotten
  - Consent management
  - Data processing records (ROPA)
  - Audit logs with tamper-evident storage

---

### Kubernetes Deployment

#### Helm Charts

**Kubernetes deployment**

- [x] **Resources**
  - Deployment
  - Service (ClusterIP, LoadBalancer)
  - Ingress
  - HorizontalPodAutoscaler
  - PodDisruptionBudget
  - ServiceMonitor (Prometheus)

**Example:**
```bash
# Install with Helm
helm install trustformers ./helm/trustformers \
  --set image.tag=v0.1.0 \
  --set replicas=3 \
  --set resources.limits.nvidia.com/gpu=1
```

---

#### Autoscaling

**Automatic scaling based on metrics**

- [x] **Metrics-Based**
  - CPU utilization
  - Memory utilization
  - Request rate
  - Queue depth
  - Custom metrics (latency, error rate)

---

## Known Limitations

- Maximum batch size 256 (hardware dependent)
- GraphQL subscriptions require WebSocket support
- CUDA requires NVIDIA GPUs with compute capability 7.0+
- ROCm requires AMD GPUs (RX 5000 series+)
- Kubernetes autoscaling requires metrics-server
- RabbitMQ, Redis Streams, NATS, and AWS SQS message-queue backends are trait-complete but currently no-op placeholders (no real broker I/O); only Apache Kafka is a genuine wire-protocol implementation — see "Message Queue Integration" above
- Per-provider cloud inference/deployment (AWS SageMaker, GCP Vertex AI, Azure ML) is simulated/mocked pending real SDK integration; the multi-cloud orchestration layer itself (`CloudProvider` trait, health checks, unified types) is real — see "Cloud Provider Support" above
- The AWS Lambda serverless adapter (`src/serverless/awslambdaprovider_traits.rs`) is not yet wired to real AWS Lambda: `deploy()` fabricates an ARN using a hardcoded placeholder AWS account ID, `invoke()` echoes the input payload back instead of invoking the function, and `get_metrics()` returns hardcoded constants; the struct holds a real `aws_sdk_lambda::Client` field but it is unused at its one call site

---

## Security Notes

- `cargo audit` (workspace-wide run, 2026-07-01) found 7 `rustls-webpki` advisories pulled in transitively: via the AWS SDK stack (`aws-smithy-http-client` → `rustls` 0.21), and via `async-nats` 0.46 → `rustls-webpki` 0.102.8.
- `cargo update --dry-run` confirmed there is no safe patch-level fix available — resolving this requires a major version bump of the AWS SDK crates and/or `async-nats`.
- **Deferred by explicit maintainer decision.** This is a larger cross-cutting upgrade (AWS SDK crates are unconditional dependencies throughout this crate) rather than a quick patch, so it is tracked here rather than fixed immediately. Revisit when the AWS SDK for Rust or `async-nats` ship a `rustls`/`rustls-webpki` upgrade.

---

## Future Enhancements

### High Priority
- [x] Fix TestPerformanceMonitoringConfig field drift (completed 2026-07-05)
  - Goal: 6 sub-configuration types (`AnalyticsConfig`, `EventConfig`, `HistoricalDataConfig`, `AlertConfig`, `DashboardConfig`, `SubscriptionConfig`) already existed and were fully real, but the top-level `TestPerformanceMonitoringConfig` struct had never grown fields to hold them, so every sub-system constructor in `service.rs` fell back to `Default::default()` instead of the caller's real configuration.
  - Fix: added the 6 fields (+ `Default` impl) to `TestPerformanceMonitoringConfig`; added the 2 previously-missing leaf fields referenced by dead commented-out call sites (`audit_trail_enabled: bool` on `HistoricalDataConfig`, `compliance_logging: bool` on `EventConfig`, `rate_limiting_enabled: bool` on `AlertConfig`) plus `compliance_reporting: bool` on `ReportConfig`; rewired `TestPerformanceMonitoringService::new()` in `service.rs` to pass `config.analytics_config.clone()` / `.event_config` / `.historical_data_config` / `.alert_config` / `.dashboard_config` / `.subscription_config` into each sub-manager constructor instead of `Default::default()`; uncommented the now-valid field assignments in `create_compliance_focused_service`/`create_resource_efficient_service` in `mod.rs`.
  - Files: `test_performance_monitoring/types/config.rs`, `types/events.rs` (`EventConfig`'s `Default` impl lives here after an earlier SplitRS split), `service.rs`, `mod.rs`.
  - Tests: extended `test_test_performance_monitoring_config_default` / `test_report_config_default` / `test_historical_data_config_default` / `test_alert_config_default` and added `test_event_config_default` in `types/config.rs`; extended `test_specialized_service_creation` in `mod.rs` to assert the compliance-focused/resource-efficient services' *resulting* config actually carries the requested flags (via a new `TestPerformanceMonitoringService::config()` accessor) rather than only checking `.is_ok()`, which would have passed even if the fields were silently ignored.
  - Behavior confirmed real, not cosmetic, for most of the 6: `EventConfig.channel_capacity` now sizes the real `broadcast::channel` inside `EventManager`, `buffer_size` sizes its `CircularEventBuffer`, and `compression_enabled`/`indexing_config`/`retention_config`/`correlation_config`/`pattern_config`/`aggregation_config`/`enrichment_config` all reach their respective sub-components; `HistoricalDataConfig.compression_enabled`/`indexing_config`/`partitioning_strategy`/`storage_optimization` reach `HistoricalDataManager`'s `CompressionEngine`/`TimeSeriesStore`; `DashboardConfig.layout`/`refresh_interval` reach `DashboardManager`'s `WidgetManager`/`LayoutEngine`. By contrast, `AlertConfig` and `SubscriptionConfig` are now threaded through as real, stored objects but remain otherwise inert today — `AlertRuleEngine::new` takes `_config: &AlertConfig` (deliberately unused) and every other `AlertManager` sub-component takes no config at all, and `SubscriptionManager` only stores its config without reading any field from it — the same "real code, no live consumer yet" pattern already flagged elsewhere in this file (SemanticCache/GraphQL model_service), noted here rather than silently implied as fully wired.
- [~] Mount SemanticCache as an opt-in caching tier (planned 2026-07-05)
  - Goal: the already-complete, already-tested (15 tests, 510 lines) SemanticCache becomes part of the compiled crate.
  - Design: add `pub mod semantic_cache;` + re-exports to caching/mod.rs. Add an Option<Arc<SemanticCache>> tier to CachingService, gated by a new config flag, mirroring how distributed_cache is already gated. Define a small EmbeddingProvider trait as the lookup seam — do NOT fabricate embeddings: when none is supplied (always, today — no real embedding generation exists anywhere in this crate), semantic lookup is simply skipped and result_cache is used alone, exactly as today.
  - Files: trustformers-serve/src/caching/mod.rs, caching/semantic_cache.rs, caching/config.rs.
  - Tests: the file's existing 15 unit tests run once mounted; a new integration test using a deterministic test-double EmbeddingProvider to verify tier composition.
  - Documented caveat, not a blocker: the live inference endpoint uses a third, separate, ad-hoc REQUEST_CACHE static today — not CachingService at all. Mounting SemanticCache here does not make it reachable from real requests; that rewiring plus real embedding generation is a separate, larger follow-up.
  - Risk: same "real code, no live consumer yet" pattern as the GraphQL model_service item — document both that way rather than implying either is fully live.
- ~~Better request scheduling algorithms~~ ✅ Done — priority queue + WRR/EDF/fair/FIFO scheduler
- [ ] Improved GPU memory management
  - **Refinement needed:** target metric (peak GPU memory %, allocation fragmentation?), which strategy (buddy allocator? memory pool tunability?)?
- [ ] WebAssembly serving for edge deployment (WASM-compiled inference server, complements trustformers-wasm)

### Performance
- [ ] Further kernel fusion optimizations
  - **Refinement needed:** which ops? attention+layernorm? ffn fused? target inference speedup %.
- [ ] Dynamic precision selection (auto-select fp32/fp16/bf16/int8 based on hardware and accuracy tolerance)
- [ ] Better batching strategies for variable-length generation (continuous batching / PagedAttention-style batching)

### Features
- [ ] Auth: OIDC (OpenID Connect) provider integration
- [ ] Auth: SAML 2.0 SSO integration
- [ ] Enhanced monitoring dashboards
  - **Refinement needed:** Grafana dashboards? Prometheus alert rules? What metrics to surface?
- [~] Wire real Welch's t-test + implement Mann-Whitney U for A/B tests (planned 2026-07-05)
  - Goal: AbTestManager::compute_results() uses the crate's own rigorous, already-tested (27 tests) Welch's t-test instead of a cruder homegrown z-test, plus a new Mann-Whitney U test (zero existing implementation confirmed).
  - Design: decide up front — bounded reservoir sampling, NOT an unbounded Vec<f64> (unbounded growth is a real production memory risk). Add reservoir-sampled raw-latency retention to ExperimentVariantStats. Rewire compute_results() to call the existing statistics::welch_t_test on the reservoir samples. Implement mann_whitney_u_test(control, treatment, alpha) in statistics.rs following welch_t_test's exact structure. Report both TTestResult and MannWhitneyResult on ExperimentResult.
  - Files: trustformers-serve/src/ab_testing/statistics.rs, ab_testing/mod.rs.
  - Tests: pure numerical unit tests mirroring statistics.rs's existing 27-test style; an AbTestManager integration test with a real (not faked) latency distribution.
  - Risk: check for other callers of the homegrown StatisticalTest/two_sample_z_test before removing it (it's pub).
- [ ] Real-time model updates with zero-downtime hot-reload (blue-green model swap with atomic pointer update)
- [ ] Real broker wiring for message-queue backends: RabbitMQ (AMQP 0-9-1, dead-letter exchanges, publisher confirms), Redis Streams, NATS, and AWS SQS currently share a no-op `impl_placeholder_backend!` macro (`src/message_queue.rs`) — only Kafka is genuinely wired today
- [ ] Real SDK-backed inference/deployment for cloud providers: `AwsSagemakerProvider`, `GoogleVertexAiProvider`, and `AzureMachineLearningProvider` currently share a simulated `impl_provider!` macro (`src/cloud_providers.rs`) that returns a fixed mock response regardless of provider
- [~] Wire real AWS Lambda calls in serverless adapter (planned 2026-07-05)
  - Goal: deploy/update/invoke/get_metrics make real AWS calls instead of fabricating every response.
  - Design: with_aws_config() also builds/stores the already-present-but-unused CloudWatch client. deploy() -> real client.create_function(). update() MUST get its own real body (update_function_code/update_function_configuration) — it currently delegates to deploy(), which will start erroring once deploy is real (CreateFunction fails on an existing function name). invoke() -> real client.invoke(), surfacing function_error as Err. get_metrics() -> parallel cloudwatch_client.get_metric_statistics() calls. Document, don't fabricate: cost_usd and cold_starts can't be fully sourced from these two SDKs alone — mark as approximations in code comments.
  - Files: trustformers-serve/src/serverless/awslambdaprovider_traits.rs, serverless/types.rs.
  - Tests: NO live AWS calls — use the AWS SDK's own test-replay HTTP client with canned responses; also test that deploy/invoke/get_metrics return Err (not fabricated success) when no client is configured.
  - Risk: update()'s current delegation-to-deploy() breaking is the single most important cross-effect to get right.

---

## Development Guidelines

### Code Standards
- **File Size:** <2000 lines per file
- **Testing:** Comprehensive unit and integration tests
- **Documentation:** API documentation with examples
- **Error Handling:** Use `TrustformersResult<T>`

### Build & Test Commands

```bash
# Build
cargo build --release

# Run tests
cargo test --all-features

# Run server
cargo run --release --bin trustformers-serve

# Build Docker image
docker build -t trustformers-serve:latest .

# Run with Docker
docker run -p 8080:8080 trustformers-serve:latest

# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Configuration Example

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8080
  grpc_port: 9090

batching:
  max_batch_size: 32
  max_wait_time_ms: 10
  strategy: "dynamic"

speculative:
  enabled: true
  draft_model: "/models/gpt2-small"
  draft_steps: 5

cache:
  enabled: true
  size_mb: 1024
  ttl_seconds: 3600

models:
  - name: "gpt2"
    path: "/models/gpt2"
    device: "cuda:0"
    max_batch_size: 16
  - name: "bert"
    path: "/models/bert"
    device: "cuda:1"
    max_batch_size: 32

monitoring:
  prometheus:
    enabled: true
    port: 9090
  jaeger:
    enabled: true
    endpoint: "http://localhost:14268/api/traces"
  slo:
    p99_latency_ms: 200
    availability_target: 0.999
```

---

## API Examples

### REST API

```bash
# Generate text
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# Get embeddings
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "bert-base-uncased"
  }'

# List models
curl http://localhost:8080/v1/models

# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

### gRPC API

```rust
use trustformers_serve::proto::inference_service_client::InferenceServiceClient;
use trustformers_serve::proto::GenerateRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = InferenceServiceClient::connect("http://[::1]:9090").await?;

    let request = tonic::Request::new(GenerateRequest {
        prompt: "Once upon a time".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    });

    let response = client.generate(request).await?;
    println!("Generated: {}", response.into_inner().text);

    Ok(())
}
```

---

**Last Updated:** 2026-07-02 - v0.1.4
**Status:** Production-ready serving infrastructure (see Known Limitations / Security Notes for the mock/placeholder subsystems and the deferred audit finding)
**Tests:** ~4,321 passing, 0 failing (workspace-wide `cargo nextest run --workspace --all-features`)
**Public API:** 7,319 items (largest crate in the `trustformers` workspace by this measure)
**APIs:** REST, gRPC (proto compilation restored in 0.1.4), GraphQL
**Deployment:** Docker, Kubernetes, Helm
**Cloud:** AWS, GCP, Azure (orchestration real; per-provider inference simulated — see Cloud Provider Support)
**Messaging:** Kafka (production); RabbitMQ/Redis Streams/NATS/SQS (interface/scaffold, no-op backend)
