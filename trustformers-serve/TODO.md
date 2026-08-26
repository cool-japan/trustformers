# trustformers-serve TODO List

**Version:** 0.2.1 | **Status:** Stable | **Tests:** ~4,321 as of 2026-07-01, not independently re-run this pass — see root `TODO.md` for the current workspace-wide baseline (21,370 passed / 41 skipped / 0 failed, 2026-08-26) | **Public API Items:** 7,319 as of 2026-07-09, not re-verified | **SLoC:** 273,756 (`tokei`, verified 2026-08-24; was 283,692/278,397 on 2026-07-09/2026-08-18 — this cycle's `resource_manager/` placeholder-tree deletion, 5,972 lines, is the largest single driver of the drop) | **Updated:** 2026-08-24

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
- [x] **COMPREHENSIVE TESTING** - ~4,321 tests as of 2026-07-01 (stale figure, not independently re-run this pass; see root `TODO.md` for the current workspace-wide baseline, 21,370 passed / 41 skipped / 0 failed as of 2026-08-26)
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
to Jaeger/Zipkin/OTLP endpoints — treat the "Integration with Jaeger/Zipkin"
checkbox above as describing `distributed_tracing/`, not `tracing/`. It was
**not** audited in the 2026-08-24 pass (the module was owned elsewhere at the
time); it was audited end-to-end in Wave 6 (2026-08-25) — see "`distributed_tracing/`
audit (Wave 6)" below for the fabrications found and fixed, and for why two
exporters genuinely exist rather than one.

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

**`resource_management/gpu_manager/**` audit (first pass over this tree)**

The data path is genuinely real: `discover_gpu_devices` and
`collect_device_metrics` shell out to `nvidia-smi` and return `Ok(None)` when it
is unavailable or silent, so a host with no NVIDIA GPU gets an empty catalogue
rather than an invented one, and `run_benchmark` already refused to synthesise a
score (now covered by a regression test). One real fabrication was found and
fixed:

- `alert_system/types.rs:557` and `monitoring/types.rs:459` both computed GPU
  memory usage as `memory_usage_mb / 24576.0` -- i.e. they assumed **every** GPU
  has exactly 24 GiB of VRAM. On a 12 GiB card that halved the true percentage
  and silenced genuine memory alerts; on an 80 GiB card it inflated it and would
  fire false ones. `GpuRealTimeMetrics` now carries
  `total_memory_mb: Option<u64>` (filled from what discovery read for that
  device, threaded through the monitoring hand-off), and both consumers use the
  new `memory_usage_percent()`, which returns `None` when the size is unknown so
  the threshold is skipped rather than evaluated against a guess.

A second, subtler one in the same tree: `GpuTelemetrySample`'s doc claimed
"sensors the driver marks as unavailable are represented as `None` or `NaN`
rather than as a plausible number", but only `fan_percent` was `Option`. The
`nvidia-smi` parse fell back to `0` for `utilization_percent`, `memory_used_mb`
and both clocks -- so an unreadable utilization sensor reported an **idle** GPU,
the direction that hides a problem. `gpu_scheduler::update_gpu_memory_monitoring`
wrote that `0` straight into its status table, `dynamic_gpu_allocation` passed it
on as a real utilization, and the health check compared it against
`utilization_threshold` and passed. (Temperature and power were already `NaN` on
failure, which no comparison passes, so those two were safe.) Every field is now
`Option`; the three consumers skip an absent reading instead of writing a zero,
the health check raises "Utilization sensor is unavailable" as an explicit issue,
and `collect_device_metrics` skips a sample it cannot represent honestly rather
than publishing an "idle, empty" GPU into the monitoring stream.

A third, with the widest blast radius: `GpuDeviceInfo::utilization_percent` is
written as `0.0` by `discover_gpu_devices` and **never updated by anything**, yet
five call sites used it as the *fallback* when a live load reading was missing.
The effect was uniform and always permissive:

- `load_balancer`'s least-loaded and weighted selectors scored an unmonitored
  device as `0.0` load, i.e. perfectly idle, so it beat every measured device and
  won the selection. Least-loaded now treats a missing reading as `INFINITY`
  (never the least loaded), the weighted selector skips an unscoreable device,
  and the memory-optimized scorer charges the full load penalty.
- `GpuConstraintType::MaxUtilization` compared against that same `0.0` in both
  `manager::verify_constraint` and `load_balancer`, so a `MaxUtilization`
  constraint was satisfied by every device unconditionally — the limit an
  operator set enforced nothing. `verify_constraint` now takes the live reading
  (prefetched before the `parking_lot` guards, since `device_telemetry` is
  async) and *fails* the constraint when the driver will not report utilization:
  an unverifiable limit is not a met one. The `load_balancer` copy has no live
  reading available at that point and now returns `false` rather than claiming
  the constraint is met.
- The health check's `performance_ok` fell back to it, so it was decided by a
  constant; it now raises "Utilization sensor is unavailable" instead.

`GpuConstraintType::MinPerformance`/`PowerLimit`/`TemperatureLimit` still return
`true` unchecked; they are now commented as "this selector does not evaluate
this constraint", not "the device satisfies it". `MinPerformance` genuinely
cannot be evaluated: it needs a benchmark score, and `run_benchmark` correctly
refuses to invent one.

Two `gpu_manager` tests were themselves asserting against the fabrication and
were rewritten: `test_health_check` asserted `performance_ok` on a synthetic
device (true only because the check read the record constant), and
`test_unhealthy_device_detection` set `device.utilization_percent = 99.0` and
asserted the check noticed — proving nothing about a real device once the value
stopped being read. Both now assert the real invariant: a device whose driver
answers nothing cannot be certified healthy.

The remaining `Ok(true)` returns in `manager.rs` (`assess_device_health`,
`check_performance_requirements`) were checked and are real: each is the tail of
a function that returns `Ok(false)` on a failed condition above it.

**`gpu_manager/load_balancer` audit, continued (Wave 6)**

The first pass above fixed `select_least_loaded`, `select_performance_based`
(via `calculate_performance_score`) and `select_memory_optimized` (via
`calculate_memory_score`) to stop trusting `GpuDeviceInfo::utilization_percent`
as a fallback for a device with no live load reading. It missed that
`select_hybrid` — reachable via `LoadBalancingStrategy::Hybrid` — carries its
own, separate copy of the `LeastLoaded` scoring logic rather than calling
`select_least_loaded`, and that copy still had the original bug verbatim: an
unmonitored device scored `1.0 - 0.0 = 1.0` and, being added into the hybrid's
per-strategy score sum, beat every genuinely measured device regardless of how
many other strategies were mixed in. It now scores a missing reading as
`f32::NEG_INFINITY` — added into any finite total from the hybrid's other
strategies, this still leaves the device unelectable, unlike simply omitting
it from that round's scoring would. The `PerformanceBased` and
`MemoryOptimized` branches inside `select_hybrid` call the same
`calculate_performance_score`/`calculate_memory_score` helpers the dedicated
selectors use, so they inherited the first pass's fix automatically and needed
no separate change; the catch-all branch for any other strategy assigns every
device the same neutral `1.0`, which does not discriminate for or against an
unmonitored device either. A new regression test
(`test_hybrid_strategy_unmonitored_device_never_wins`) pins a device with no
`update_device_load` call against one measured at 99% utilization and asserts
the hybrid selector still prefers the measured, heavily-loaded device.

**`resource_management/statistics.rs` audit (Wave 6)**

`StatisticsCollector`'s analytics engine (`AnalyticsEngine` /
`AnomalyDetector` / `PerformancePredictor` / `BottleneckAnalyzer` /
`MetricsAggregator`) held real, growing histories of
`SystemPerformanceSnapshot`/`ResourceUtilizationSnapshot` the whole time —
every analysis method on top of them was simply ignoring the data and
returning a constant instead:

- `PerformancePredictor::predict` ignored its snapshots and returned
  `predicted_value: 75.0`, `confidence_interval: (70.0, 80.0)`, `confidence:
  0.85` for every metric, every horizon. It now fits ordinary least squares
  over every finite recorded reading of the requested metric (elapsed seconds
  since the first reading vs. value) and extrapolates to the requested
  horizon; the confidence interval is the fit's residual-based prediction
  interval (a fixed 1.96-sigma multiplier under a normal approximation, since
  this module has no inverse-t implementation for a proper t-distribution
  critical value); `confidence` is the fit's R², `0.0` — not a fabricated
  "perfect fit" — when the recorded values have no variance to explain.
  Refuses when the metric name is unrecognized, no snapshots exist, or fewer
  than 2 finite readings of the metric exist.
- `AnomalyDetector::detect_anomalies` and `BottleneckAnalyzer::analyze_bottlenecks`
  both ignored their snapshots and returned `Ok(vec![])` unconditionally.
  Anomaly detection now applies a three-sigma rule per known metric (mean and
  sample standard deviation over every finite reading; a metric with fewer
  than two finite readings, or zero variance, contributes no anomalies rather
  than dividing by zero); bottleneck analysis now compares the latest
  snapshot's utilization of each known resource against its configured
  `BottleneckAnalysisConfig` threshold (a 0.85 default for a resource with no
  dedicated entry), reporting one bottleneck per resource at or above
  threshold, ranked by measured severity. Both refuse only when no snapshot
  has ever been recorded.
- `MetricsAggregator::aggregate_utilization_metrics` logged a message and did
  nothing, called on the live `record_utilization` path — so
  `get_aggregated_metrics` always returned an empty map. It now computes every
  configured `AggregationMethod` (mean/median/min/max/std-dev/percentile/sum/
  count) plus every configured percentile, over the most recent
  `rolling_window_size` entries, for every series the recorded
  `ResourceUtilizationSnapshot`s actually carry (each resource category's four
  sub-metrics, the same per custom resource, and the six flat `SystemMetrics`
  fields) — not an invented composite "overall utilization" figure.
- `get_performance_statistics` published `efficiency_score: 0.85` as a
  constant inside otherwise-real statistics. It is now the mean of each
  snapshot's own measured `overall_efficiency` (itself a real average of
  whichever subsystem occupancy signals had recorded activity, computed by
  `ResourceManagementSystem::get_performance_snapshot` — not fabricated here).

14 new tests lock these in, including refusal cases (empty history, single
snapshot, unknown metric name) and positive cases (a near-perfect linear
history yields R² > 0.99 and a prediction that continues the trend; a
20-reading constant series plus one 99.0 outlier is flagged, a flat series is
not; simultaneous CPU/memory threshold breaches are both reported and ranked
by measured severity; aggregation over two recorded snapshots produces the
correct mean/min/max).

**`gpu_manager/health_monitor` audit, continued (Wave 6)**

Two more fabrications in the same tree, both at device-health *birth* rather
than during a live check:

- `create_initial_health_status` used to construct a fully-healthy record
  outright (`is_healthy: true`, `health_score: 1.0`, every `*_ok` flag `true`,
  `current_temperature: 45.0`, `current_power: 150.0`,
  `consecutive_healthy_checks: 1`) for a device that had never once been
  probed — readable via `get_health_status` in the window between
  `initialize_device_health` running and the background health-check loop's
  first real tick. It now reports the same shape
  `perform_comprehensive_health_check` reports for a driver that answered
  nothing: `is_healthy: false`, `f32::NAN` for the three sensor readings (not
  a plausible number), `consecutive_healthy_checks: 0`, and an `issues` list
  naming each sensor as unread. Memory and hardware status are the exception —
  those come from the device record itself (capacity/availability, reported
  status), which really is known at discovery time, so they are computed for
  real here exactly as the live check computes them, not marked unknown.
  `create_initial_analytics` no longer seeds `health_history` with one
  fabricated `(now, 1.0)` sample either — an empty history correctly reports
  `HealthTrend::Unknown` with zero confidence until a real check contributes
  the first point. Downstream: `allocate_gpu_devices`/`check_availability`
  already refuse a device with `is_healthy: false`, so a device now waits for
  its first real probe before it can be allocated — the honest reading of "we
  do not yet know" rather than "assumed fine". No other consumer reads these
  fields (`GpuHealthStatus`/`GpuHealthAnalytics` are not re-exported past
  `gpu_manager`).
- `compute_trend_analysis`'s `confidence` used to be a declared ladder keyed
  only on how many samples the history held (20+ → 0.9, 10+ → 0.7, 5+ → 0.5,
  else 0.2) — the previous pass through this file only documented that as a
  known limitation rather than fixing it. It is now the least-squares fit's
  own coefficient of determination (R², computed in `update_analytics_metrics`
  alongside the slope it already fit): how much of the health score's variance
  the linear trend actually explains, `0.0` before there are two samples or
  when the history has no variance to explain (a frozen, always-identical
  reading no longer manufactures a high-confidence trend).

**`distributed_tracing/` audit (Wave 6) — the second Jaeger/Zipkin/OTLP exporter**

`src/distributed_tracing/` and `src/tracing/` are not accidental duplicates:
`tracing/` (via `tracing::legacy::export_traces`) only ever serializes spans
to a byte buffer for a "Load JSON File" UI flow or manual inspection — it has
no HTTP client anywhere in it, and its own `export_endpoint` field doc already
says "for future export use". `distributed_tracing/`'s `TracingManager` is the
only module in this crate that actually POSTs spans to a live Jaeger, Zipkin
or OTLP collector, on a background batching loop. Given that, the right move
is not a merge but auditing this exporter to the standard the advisories pass
applied to `tracing/`, and sharing types where they genuinely are the same
shape rather than re-deriving them — done below. Real fabrications and bugs
found and fixed:

- `SamplingStrategy::Adaptive`'s `should_sample` branch read `let current_load
  = 0.5;` — a hardcoded constant, so for any given config the branch sampled
  at exactly `min_rate` or exactly `max_rate` and never actually adapted to
  load the way the variant's name and doc promise. `sysinfo` is already a
  workspace dependency (see `resource_management::manager`'s identical
  CPU-usage pattern); a cached `System` instance (`CpuLoadMonitor`, refreshed
  at most once per 500ms — `should_sample` runs on every span start, and
  `sysinfo` itself documents that refreshing more often makes readings less
  accurate) now backs it with a real reading.
- `convert_to_jaeger_format`'s `tags` and `process.tags` serialized
  `span.attributes` (a `HashMap<String, String>`) directly, producing a flat
  JSON object. Jaeger's real `model.KeyValue` tag is `{key, type, value}` — an
  object keyed by attribute name is not a list of those, and a reader
  expecting the documented shape could not parse it as tags at all. This now
  builds `tracing::legacy::export::JaegerTag`, the already-audited type for
  the exact same shape, rather than a second, differently-wrong one.
- `convert_to_otlp_format` nested spans under
  `instrumentationLibrarySpans`/`instrumentationLibrary` — the pre-1.0 OTLP
  field names, renamed to `scopeSpans`/`scope` when OTLP went stable; a
  current collector does not recognize the old names (see
  `tracing::legacy::export`'s `OtlpScopeSpans`/`OtlpScope`, which already use
  the current ones). `startTimeUnixNano`/`endTimeUnixNano` were plain JSON
  numbers; OTLP/JSON represents protobuf `fixed64` fields as strings
  precisely so a 64-bit nanosecond timestamp survives a round-trip through a
  language whose numbers are `f64`-precision, which these otherwise would not.
  Separately, `resourceSpans[].resource.attributes` never carried
  `service.name` at all — the one resource attribute OTLP semantic
  conventions require to identify which service emitted a trace (compare
  `tracing::legacy::export::OpenTelemetryExport::from_spans`, which already
  sets it) — every export from this function reported spans attributable to
  no service. It is now always the first resource attribute.
- Both the Jaeger and OTLP converters defaulted a root span's
  `parentSpanID`/`parentSpanId` to `""` via `unwrap_or_default()`; a reader
  cannot distinguish an empty-string parent from no parent. Both now emit
  `null` for a root span, like the Zipkin converter already did.
- `export_to_jaeger` accepted `username`/`password` from
  `TracingBackend::Jaeger` all the way down to this function and then
  silently discarded them (`_username`, `_password`): a caller who configured
  credentials for an authenticated collector got unauthenticated requests,
  reported back only as a generic export failure with no indication the
  credentials were never sent. Now applied via HTTP Basic auth when a
  username is present.
- Both `export_loop` (on a failed batch) and `flush` (on any failure) used to
  clear the queue and *then* attempt the export, so a failed export lost the
  spans for good — fire-and-forget in the literal sense. Both now put the
  spans back at the front of the queue (bounded by `max_span_queue_size`,
  like any other queued span — not a separately backed-off indefinite retry)
  before propagating the error, so the next tick, or a caller that retries,
  can still recover them. `shutdown()` calls `flush()`, so a failure right at
  shutdown no longer silently discards whatever was still queued. A
  successful `flush()` now also counts toward `TracingStats::spans_exported`,
  which previously only `export_loop` updated.

Checked and already correct: `convert_to_zipkin_format`'s flat `"tags":
span.attributes` *is* the real Zipkin v2 tag shape (unlike Jaeger's), and its
`annotations`/`localEndpoint`/`kind` mapping matches the spec. The OTLP
per-attribute `{key, value: {stringValue}}` shape and the numeric
`SpanKind` → OTLP-kind mapping (`Internal=1 .. Consumer=5`) were both already
correct. `src/distributed_tracing/**` otherwise still logs via the `log`
facade crate rather than `tracing` (e.g. `use log::{debug, error, info,
warn};` at the top of `types.rs`) — noted here rather than changed, since a
facade migration across the whole subtree is a larger, separate change than
this audit's scope and is not itself a fabrication.

Named honestly rather than fixed, for the same reason (real but out of this
audit's scope): `export_loop`'s and `flush()`'s requeue fixes above each make
their *own* call to `export_spans` safe against loss, but the two are not
mutually exclusive — `flush()` can run concurrently with the background
loop's own tick (nothing serializes them), so in the narrow window where both
observe a non-empty queue at the same instant, the queue could in principle
be drained by one path while the other is mid-retry. Nothing in this crate
calls `flush()` from a second task today (only `shutdown()`, itself an
explicit one-shot call site), so the race is not reachable on any real path
found in this audit — worth a `Mutex`-guarded single "is a
drain-and-export in flight" flag if a genuinely concurrent caller of
`flush()` is ever added.

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

**Resolved in Wave 6** (was "Not in scope for this pass" here) — both items
below were flagged by this pass as out of its ownership and have since been
closed by the package that owns them:

- `src/distributed_tracing/types.rs`'s second, independent Jaeger/Zipkin/OTLP
  exporter has now been audited end-to-end; see "`distributed_tracing/` audit
  (Wave 6)" above.
- `trustformers-serve/Cargo.toml` lines 261-274's comment justifying the
  `paste` dev-dependency removal, which pointed at the
  `optimized_test_with_progress!` macro this pass deleted, has been updated to
  say the macro is gone rather than describe it as a still-present
  never-expanded trap.

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
