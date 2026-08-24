# trustformers-serve TODO List

**Version:** 0.2.1 (unreleased) | **Status:** Stable | **Tests:** ~4,321 as of 2026-07-01, not independently re-run this pass — see root `TODO.md` for the current workspace-wide baseline (20,629 passed / 43 skipped / 0 failed, 2026-08-24) | **Public API Items:** 7,319 as of 2026-07-09, not re-verified | **SLoC:** 273,756 (`tokei`, verified 2026-08-24; was 283,692/278,397 on 2026-07-09/2026-08-18 — this cycle's `resource_manager/` placeholder-tree deletion, 5,972 lines, is the largest single driver of the drop) | **Updated:** 2026-08-24

## Overview

The `trustformers-serve` crate provides high-performance inference serving infrastructure for production deployment of transformer models. It includes REST/gRPC/GraphQL APIs, dynamic batching, distributed serving, and comprehensive monitoring.

This is the largest crate in the `trustformers` workspace by public API surface: **7,319** public items (top-level `pub fn`/`struct`/`enum`/`trait` declarations plus indented `pub fn` methods inside impl blocks) — for comparison, the sibling `trustformers-training` crate has 846. Counting only top-level declarations (excluding impl-block methods) gives a more conservative 4,999.

**Key Responsibilities:**
- REST API with dynamic batching and caching
- gRPC API for high-throughput serving (proto compilation/serving restored in 0.1.4 — see "gRPC API (Tonic)" below)
- GraphQL API for flexible queries
- Distributed serving with load balancing
- Model management (hot-swapping, versioning, A/B testing)
- Hardware acceleration via `trustformers-core`'s device layer — CUDA/Metal real, ROCm real-but-unverified-here, XLA/Vulkan not reliably real (see "Hardware Acceleration" below, corrected 2026-08-24)
- Kubernetes deployment with autoscaling
- Monitoring and observability (Prometheus with once_cell lazy statics, Jaeger, OpenTelemetry)
- SLO monitoring and breach alerting
- NUMA/topology-aware performance optimizer (Linux sysfs, macOS sysctl)
- Speculative decoding with draft models
- Kernel fusion for GPU operations
- Message queue integration (Kafka — production; RabbitMQ/Redis Streams/NATS/SQS — **real backends since 2026-08-18**, see "Completed Features" below — this bullet previously read "interface complete, wiring pending", which is now stale)
- Cloud provider support (AWS, GCP, Azure — orchestration layer real; **`cloud_providers.rs` no longer returns a canned mock response for any of its 6 provider integrations, verified 2026-08-24** — `grep -c "Mock response\|example.com/endpoint"` returns 0 hits in that file today. This bullet previously read "per-provider inference calls simulated", which is now stale for at least the generic `cloud_providers.rs` path; the AWS Lambda serverless *adapter* specifically is a separate, still-fabricated concern — see "Completed Features" below.)
- GDPR compliance

---

## Current Status

### Implementation Status
- [x] **PRODUCTION-READY** - Complete serving infrastructure
- [x] **ZERO COMPILATION ERRORS** - Clean compilation
- [x] **COMPREHENSIVE TESTING** - ~4,321 tests as of 2026-07-01 (stale figure, not independently re-run this pass; see root `TODO.md` for the current workspace-wide baseline, 20,629 passed / 43 skipped / 0 failed as of 2026-08-24)
- [x] **REQUEST QUEUING** - Priority queue with deadline awareness and cancellation (`queue` module)
- [x] **PRIORITY SCHEDULING** - WRR, EDF, fair queuing, priority, and FIFO strategies (`scheduler` module)
- [x] **HARDWARE ACCELERATED** - CUDA and Metal support real and hardware-verified elsewhere in this workspace; ROCm real but not hardware-verified here (see "Hardware Acceleration" below, corrected 2026-08-24)
- [x] **KUBERNETES READY** - Helm charts, autoscaling, monitoring

### Feature Coverage
- **APIs:** REST (Axum), gRPC (Tonic), GraphQL (async-graphql)
- **Performance:** Dynamic batching, result caching, kernel fusion, speculative decoding
- **Distribution:** Load balancing, failover, health checks, disaster recovery
- **Monitoring:** Prometheus metrics (once_cell lazy statics), Jaeger tracing, OpenTelemetry, SLO monitoring
- **Security:** Authentication, TLS, GDPR compliance, encryption
- **Deployment:** Docker, Kubernetes, Helm, service mesh integration
- **Cloud:** AWS (EKS, S3, CloudWatch), GCP (GKE, GCS), Azure (AKS, Blob) — deployment orchestration real; per-provider inference (SageMaker/Vertex AI/Azure ML) simulated pending real SDK integration
- **Messaging:** Kafka (production, feature-gated); RabbitMQ/Redis Streams/NATS/SQS — **corrected 2026-08-24**: these are no longer no-op scaffolds. Real backends live in `src/message_queue/{rabbitmq,nats,redis_streams,sqs}.rs` and call the real `lapin`/`async_nats`/`redis`/AWS-SQS clients (landed 2026-08-18; see "Message Queue Integration" below). Not exercised against a live broker.

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

**Corrected 2026-08-24.** The example below used to construct a `TracingConfig`
with `exporter: TracingExporter::Jaeger` / `jaeger_endpoint` / `sampling_rate`
and call `server.enable_tracing(...)`. None of those exist: neither
`TracingConfig` in this crate has an `exporter` or `jaeger_endpoint` field, and
`grep -rn 'fn enable_tracing' src/` finds nothing. The real
`src/tracing/request_tracer.rs` shape is:

```rust
let tracing_config = TracingConfig {
    service_name: "trustformers-serve".to_string(),
    service_version: env!("CARGO_PKG_VERSION").to_string(),
    sample_rate: 0.1, // 10% sampling
    max_spans_per_trace: 1000,
    // Carried through, but `tracing/` itself never exports: the field's own
    // doc says "for future export use".
    export_endpoint: Some("http://localhost:14268/api/traces".to_string()),
};
```

`src/distributed_tracing/` contains a *separate* exporter that does POST spans
to Jaeger/Zipkin/OTLP endpoints. It was **not** audited in the 2026-08-24 pass
(the module is owned elsewhere), so treat the "Integration with Jaeger/Zipkin"
checkbox above as describing `distributed_tracing/`, not `tracing/`, and as
unverified.

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
- [x] **Real backend integration: done, verified 2026-08-18** — the `impl_placeholder_backend!` macro is gone (`rg impl_placeholder_backend trustformers-serve/src` finds nothing). The four backends now live under `src/message_queue/` as their own files with real client usage: `rabbitmq.rs` and `nats.rs` and `redis_streams.rs` each call the real `lapin`/`async_nats`/`redis` client APIs (`sqs.rs` uses the AWS SQS SDK, already real). This entry previously described `send_message`/`poll`/etc. as no-ops with no network I/O — that is no longer the current code. Not independently re-run against a live broker this pass (documentation-only, read the source rather than exercised it against a running broker); a targeted `cargo nextest run -p trustformers-serve` would confirm behavior, not just presence of the client calls.

---

### Cloud Provider Support

- [x] **Unified provider abstraction: done** — `CloudProvider` trait, health-check orchestration, and unified request/response types implemented and tested across AWS/GCP/Azure/HuggingFace/OpenAI/Anthropic provider stand-ins (`src/cloud_providers.rs`)
- [x] **AWS**: EKS deployment, S3 model storage, CloudWatch metrics
- [x] **GCP**: GKE autopilot, GCS model storage, Cloud Monitoring
- [x] **Azure**: AKS, Blob Storage, Azure Monitor
- [x] **Real per-provider inference calls: done, verified 2026-08-18** — the shared `impl_provider!` macro and its fabricated `OutputData::Text("Mock response")` / `https://example.com/endpoint` are gone (`rg 'Mock response|example.com/endpoint|impl_provider!' trustformers-serve/src/cloud_providers.rs` finds nothing). `AzureMachineLearningProvider::inference` (`src/cloud_providers/rest.rs`) makes a real `reqwest` `POST {base}/chat/completions` call with a real API-key header and real error handling (`MissingConfiguration` if no endpoint is configured, `Transport`/status-code errors on failure) — a REST-based implementation rather than the Azure SDK (which this crate no longer depends on; see manifest hygiene notes). Not independently confirmed for `AwsSagemakerProvider`/`GoogleVertexAiProvider`/`HuggingFaceProvider`/`OpenAiProvider`/`AnthropicProvider` this pass beyond the macro-and-literal-string grep above — spot-check each provider file under `src/cloud_providers/` before assuming full parity with the Azure one described here.

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

> **Corrected 2026-08-24**: this section describes the `trustformers-core`/`trustformers-models` hardware layer this crate dispatches to, not code `trustformers-serve` implements itself — and the checkmarks below overclaimed against that layer's real state. See `trustformers-core/TODO.md`'s "Hardware Acceleration" section (rewritten 2026-08-24) for the verified, per-backend detail. Summary: **CUDA and Metal are real** (hardware-verified elsewhere in this workspace); **ROCm** has real `dlopen`-based HIP scaffolding but is not hardware-verified in this environment; **Metal no longer uses MPS** (Metal Performance Shaders) — it runs on the Pure-Rust `oxicuda-metal`, so the "Metal Performance Shaders (MPS)" bullet below is stale; `trustformers-core` has no XLA/oneAPI backend beyond empty no-op facades, and no TPU support at all — do not read the "Key Responsibilities" bullet above's "Hardware acceleration (CUDA, ROCm, Metal, XLA, Vulkan)" as implying XLA works.

#### CUDA Support

**NVIDIA GPU acceleration**

- [x] **Features** (real, hardware-verified elsewhere in this workspace — see `trustformers-core/TODO.md`)
  - cuDNN integration
  - cuBLAS for GEMM
  - Multi-GPU support
  - CUDA Graphs for optimization
  - Tensor Cores (FP16, INT8)

---

#### ROCm Support

**AMD GPU acceleration**

- Real `dlopen`-based HIP runtime bindings exist in `trustformers-core` (feature-gated, not hardware-verified in this environment — see `trustformers-core/TODO.md`). The specific sub-features below (MIOpen, rocBLAS, multi-GPU) were not individually verified this pass; do not assume all four are wired just because the section header is real.

---

#### Metal Support

**Apple Silicon acceleration**

- [x] **Features** (real, hardware-verified elsewhere in this workspace — see `trustformers-core/TODO.md`), corrected 2026-08-24: runs on the Pure-Rust `oxicuda-metal`, **not** Metal Performance Shaders (MPS) — the MPS dependency was dropped
  - Metal compute kernels (via `oxicuda-metal`)
  - Unified memory
  - Neural Engine integration (not independently verified this pass)

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

## Honesty audit — `test_performance_monitoring/`, `operator_scheduling`, `model_management/deployment` (2026-08-24)

This pass removed the crate's last blanket `#![allow(dead_code)]` (25 files, done
earlier in the same cycle) and then resolved every warning that removal exposed,
rather than re-allowing any of them. `grep -rn '^#!\[allow(dead_code)\]' src/`
now returns nothing, and the only crate-level allows left in `src/lib.rs` are
clippy style lints.

**Live fabrications removed**

- `operator_scheduling.rs`: `try_schedule_next_task` — reached from the public
  `submit_task` — spawned a task that slept `100 + (hash(task_id) % 1000)` ms
  and then wrote a `TaskExecutionResult` claiming `state: Completed`, that sleep
  as `execution_time`, and `peak_memory_usage: Some(1 MiB)` for an operator that
  never ran; `get_task_result` returned it to callers as a measurement. Replaced
  by an `OperatorExecutor` seam (`OperatorSchedulingService::with_executor`).
  With no executor, tasks stay queued and no result is produced; with one, the
  clock is read around the executor's own future and the metrics are whatever it
  reported. `peak_memory_usage` is now `None` — nothing samples it. The dead
  `simulate_task_execution` is deleted, and `complete_task`'s locks are scoped
  (it re-enters `try_schedule_next_task`, and `tokio::RwLock` is not reentrant,
  so the previous guard-holding form would have deadlocked the moment it ran).
  The same function's concurrency gate compared
  `DeviceResource::active_tasks` -- a field nothing in the service ever
  increments -- against `max_concurrent_operators_per_device`, so the configured
  cap bounded nothing; it now counts live entries in `running_tasks` for the
  device, which is the real in-flight set. That was harmless while the old code
  only slept and invented a result, and load-bearing the moment real bodies
  started running.
- `test_performance_monitoring/types/storage.rs`: `ReportStorage::get_report`
  returned `Ok` with a `Report` whose every field was the literal `"stub"`, for
  any id, and `ReportingSystem::export_report` handed that to callers. It now
  returns a structured error naming the missing store. The hardcoded
  `/tmp/reports` path is replaced by `std::env::temp_dir()`-derived path.
- `test_performance_monitoring/service.rs`: the live event path stamped
  `HostInfo { hostname: "localhost", ip_address: "127.0.0.1", operating_system:
  "Linux", architecture: "x86_64" }` on every event regardless of host, and an
  `ExecutionContext.resource_allocation` of `cpu_cores: 4, memory_mb: 1024,
  disk_space_mb: 10240, network_bandwidth_mbps: 100.0` for every test.
  `HostInfo::detect()` now reads the OS through `sysinfo` and the compiled
  target triple (`ip_address` is `Option<String>`, the first non-loopback
  interface address or `None`); `resource_allocation` is `Option` and `None`,
  because nothing allocates per-test resources here.
- `real_time_monitor.rs`: `ActiveTestInfo.progress_percent` and
  `resource_usage` are now `Option`. The crate's only caller filled them with
  `0.0` and an all-zero `ResourceUsageSnapshot` stamped `SystemTime::now()` — a
  claim that CPU, memory, I/O, network, open files and thread count had all been
  sampled and were all zero at that instant.
- `analytics/types.rs`: `compare_with_baseline` refreshed baselines inline and
  only partially — `performance_characteristics` and `confidence_interval` kept
  whatever the first-ever sample produced, so every later delta was measured
  against a stale memory/CPU profile. The complete `refresh_baseline` existed
  but was never called; it is now the single refresh path.
- `test_cicd_integration/manager.rs`: `ConfigurationManager::load_environment_config`
  logged a line and returned `Ok(())` without reading the configuration at all.
  It now selects the `environment_configs` block matching the detected
  environment, and `CicdIntegrationManager::get_optimized_config` returns that
  block's `test_config` when one is configured. `EnvironmentDetector` now
  remembers what it detected. `ReportingIntegration::report_results` and
  `MetricsExporter::export_metrics` remain no-ops but now say so in their docs:
  their `Ok(())` means "nothing went wrong", not "the data was published".
- `test_utilities.rs`: the exported `optimized_test_with_progress!` macro
  expanded through `paste::paste!`, and `paste` was removed from the workspace
  manifest as unmaintained, so the macro could not expand anywhere. Deleted.

**Dead scaffolding deleted** (structs that were constructed from real config,
then never read, and had no methods at all — so nothing they were named for ever
happened): `LayoutEngine`/`UserPreferences` map/`WidgetFactory`/`WidgetUpdater`/
`subscriptions` map/`UpdateScheduler` (dashboard.rs); `TemplateValidator`/
`custom_templates`/`DataAggregator`/`VisualizationEngine`/`TemplateEngine`/
`ContentProcessor`/`SchedulerEngine`/`ReportNotificationManager` (reporting.rs);
`RetentionExecutor`/`ComplianceManager`/`QueryParser`/`QueryOptimizer`/
`QueryExecutionEngine`/`QueryStatistics`/`partitioning_strategy`/
`storage_optimization`/`LifecycleStateTracker`/`TransitionExecutor`/
`LifecycleEventManager` and the whole `TimeSeriesIndexManager` type
(historical_data/types.rs); `AnalyticsCache` field and `get_series`
(analytics/types.rs); `subscription_templates`/`SubscriptionAnalytics`
(subscriptions.rs); `PipelineIntegration` (test_cicd_integration/manager.rs).

**Made reachable rather than deleted** (real state with a real consumer, each
now covered by a test that a `Default::default()` regression would fail):
`DashboardManager::config`, `ReportingSystem::config`,
`SubscriptionManager::config`, `PerformanceAnalyticsEngine::config`,
`CicdIntegrationManager::config`/`detected_environment`,
`TestPerformanceMonitoringService::dashboard_manager`/`subscription_manager`,
`DeploymentManager::evaluate_canary_step`/`rollback_canary_deployment`/
`canary_deployment` (canary steps had no evaluator reachable from outside the
module at all) and `impl Clone for DeploymentManager` (replacing a private
`clone_for_background` no caller could reach).

**Left honest but still inert, for a later pass**

- `test_cicd_integration/manager.rs`: `ReportingIntegration::report_results`,
  `MetricsExporter::export_metrics` and `MetricsExporter::periodic_export`
  accept their input and drop it. Their `Ok(())` now documents that it means
  "nothing went wrong", not "the data was published" — but a caller who wants
  CI annotations or a metrics sink still gets neither.
- `ReportStorage` has no writer: `get_report` correctly refuses, and nothing
  ever puts a report where it could find one. `ReportingSystem::export_report`
  therefore always fails today. Wiring a real store (or deleting the export
  path) is a separate decision.
- `ReportScheduler` records schedules that nothing fires: there is no cron
  evaluator or timer in this crate.

**Not in scope for this pass** — `src/distributed_tracing/types.rs` greps
positive for `jaeger` and is owned elsewhere; its second, independent
Jaeger/Zipkin/OTLP exporter has not been audited. `trustformers-serve/Cargo.toml`
lines 261-274 explain the removal of the `paste` dependency by pointing at the
`optimized_test_with_progress!` macro this pass deleted; that comment is now
stale, and the manifest was outside this package's ownership.

---

## Known Limitations

- Maximum batch size 256 (hardware dependent)
- GraphQL subscriptions require WebSocket support
- CUDA requires NVIDIA GPUs with compute capability 7.0+
- ROCm requires AMD GPUs (RX 5000 series+)
- Kubernetes autoscaling requires metrics-server
- ~~RabbitMQ, Redis Streams, NATS, and AWS SQS message-queue backends are trait-complete but currently no-op placeholders~~ — **fixed, verified 2026-08-18**: real backends now live under `src/message_queue/{rabbitmq,nats,redis_streams,sqs}.rs`; see "Message Queue Integration" above.
- ~~Per-provider cloud inference/deployment ... is simulated/mocked~~ — **fixed for Azure, verified 2026-08-18**: the shared mock-response macro is gone crate-wide; Azure's REST implementation makes real HTTP calls. AWS/GCP/HuggingFace/OpenAI/Anthropic not individually re-checked this pass — see "Cloud Provider Support" above.
- The AWS Lambda serverless adapter (`src/serverless/awslambdaprovider_traits.rs`) is not yet wired to real AWS Lambda: `deploy()` fabricates an ARN using a hardcoded placeholder AWS account ID, `invoke()` echoes the input payload back instead of invoking the function, and `get_metrics()` returns hardcoded constants; the struct holds a real `aws_sdk_lambda::Client` field but it is unused at its one call site. Not re-verified this pass.
- ~~**`resource_manager/` placeholder tree still exported under the unprefixed name**~~ — **resolved, verified 2026-08-24**: the placeholder tree (`src/resource_manager/`, which fabricated ports/paths/connections/GPU stats/efficiency numbers) is deleted outright, not merely deprecated. `ResourceManagementSystem` now resolves directly to the real, tested `resource_management/` tree; `ModularResourceManagementSystem` is kept only as a compatibility type alias, and `lib.rs` carries a comment explaining the history. 4 new regression tests guard the unprefixed surface. Two caveats from the package that did this work, neither independently verified this pass: (1) it reports two fabrications remaining inside `lib.rs` itself (the file it had to migrate) that it did not have ownership to fix; (2) the sub-managers under `resource_management/` the unprefixed names now point at have not had their own dedicated correctness audit — only the renaming/re-pointing was done. See root `TODO.md` P1 for the same note.

---

## Security Notes

- **Updated 2026-08-24** (superseding the 2026-08-18 note below): `cargo deny check advisories` now **passes** — run directly from the workspace root this pass, exit status 0, output `advisories ok`. The six findings that were rooted in this crate's dependencies were closed by real removal, not by suppression: every `opentelemetry*` entry and `lambda-web` and `paste` are gone from the root manifest, and the `aws-sdk-*` crates now take `default-features = false` (which drops the `rustls 0.21.12` / `rustls-webpki 0.101.7` stack). Exactly one dated ignore remains, documented in `deny.toml` with its full unfixability chain: RUSTSEC-2023-0071 (`rsa`, reached only through `jsonwebtoken`).
- **Superseded, kept for history — 2026-08-18**: `cargo deny check advisories` failed with 7 findings — `opentelemetry-jaeger` unmaintained, `paste` unmaintained, `h2` unbounded-empty-DATA-frames, the RSA "Marvin Attack" timing side-channel, two `rustls` name-constraints vulnerabilities, and `rustls-webpki` 0.101.7's CRL-parsing panic (RUSTSEC-2026-0104), all reached through the AWS SDK's `rustls 0.21` stack.
- Prior note (2026-07-01, `cargo audit`, not re-verified against the tool that produced it): found 7 `rustls-webpki` advisories pulled in transitively via the AWS SDK stack (`aws-smithy-http-client` → `rustls` 0.21) and via `async-nats` 0.46 → `rustls-webpki` 0.102.8; `cargo update --dry-run` confirmed no safe patch-level fix, requiring a major version bump of the AWS SDK crates and/or `async-nats`.
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
- [x] Real broker wiring for message-queue backends — **done, verified 2026-08-18**: see the entry above under Message Queue Support; RabbitMQ/NATS/Redis Streams now live in `src/message_queue/{rabbitmq,nats,redis_streams}.rs` with real client calls, SQS uses the AWS SDK.
- [x] Real SDK-backed inference for cloud providers — **partially done, verified 2026-08-18 for Azure only**: see the entry above under Cloud Provider Support; the shared mock-response macro is gone crate-wide, and Azure's REST implementation is confirmed real. AWS/GCP/HuggingFace/OpenAI/Anthropic not individually re-checked this pass.
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

**Last Updated:** 2026-07-09 - v0.2.1
**Status:** Production-ready serving infrastructure (see Known Limitations / Security Notes for the mock/placeholder subsystems and the deferred audit finding)
**Tests:** ~4,321 passing, 0 failing (workspace-wide `cargo nextest run --workspace --all-features`)
**Public API:** 7,319 items (largest crate in the `trustformers` workspace by this measure)
**APIs:** REST, gRPC (proto compilation restored in 0.1.4), GraphQL
**Deployment:** Docker, Kubernetes, Helm
**Cloud:** AWS, GCP, Azure (orchestration real; per-provider inference simulated — see Cloud Provider Support)
**Messaging:** Kafka (production); RabbitMQ/Redis Streams/NATS/SQS (real client-backed since 2026-08-18 — the "no-op backend" wording here was stale, corrected 2026-08-24)
