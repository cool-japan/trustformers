# Deployment Best Practices for TrustformeRS

This guide provides comprehensive best practices for deploying TrustformeRS models in production environments, covering containerization, serving strategies, monitoring, scaling, and optimization.

## Table of Contents

1. [Pre-deployment Checklist](#pre-deployment-checklist)
2. [Model Optimization for Production](#model-optimization-for-production)
3. [Containerization](#containerization)
4. [Serving Strategies](#serving-strategies)
5. [Infrastructure Setup](#infrastructure-setup)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Best Practices](#security-best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Scaling Strategies](#scaling-strategies)
10. [Deployment Scenarios](#deployment-scenarios)
11. [Troubleshooting](#troubleshooting)

## Pre-deployment Checklist

### Model Validation

```rust
use trustformers::validation::{ModelValidator, ValidationConfig};

// Validate model before deployment
let validator = ModelValidator::new(ValidationConfig {
    check_weights: true,
    check_architecture: true,
    validate_outputs: true,
    test_batch_sizes: vec![1, 8, 32],
    test_sequence_lengths: vec![128, 512, 1024],
});

validator.validate(&model)?;

// Performance benchmarking
let perf_report = validator.benchmark_performance(&model)?;
println!("Latency P99: {} ms", perf_report.latency_p99);
println!("Throughput: {} tokens/sec", perf_report.throughput);
```

### Resource Requirements

```rust
use trustformers::deployment::{estimate_resources, ResourceEstimator};

let estimator = ResourceEstimator::new();
let requirements = estimator.estimate(&model, expected_qps, max_batch_size)?;

println!("Minimum RAM: {} GB", requirements.min_ram_gb);
println!("Recommended GPU VRAM: {} GB", requirements.gpu_vram_gb);
println!("CPU cores needed: {}", requirements.cpu_cores);
```

## Model Optimization for Production

### 1. Model Quantization

```rust
use trustformers::quantization::{quantize_for_deployment, QuantizationConfig};

// INT8 quantization for inference
let quant_config = QuantizationConfig {
    bits: 8,
    symmetric: true,
    per_channel: true,
    calibration_samples: 1000,
    optimize_for_inference: true,
};

let quantized_model = quantize_for_deployment(&model, &quant_config)?;

// Dynamic quantization for variable workloads
let dynamic_model = model.enable_dynamic_quantization()?;
```

### 2. Model Pruning

```rust
use trustformers::pruning::{structured_prune, PruningConfig};

let pruning_config = PruningConfig {
    target_sparsity: 0.5,
    structured: true,
    preserve_accuracy_threshold: 0.99,
};

let pruned_model = structured_prune(&model, &pruning_config, &validation_data)?;
```

### 3. Compilation and Optimization

```rust
use trustformers::compile::{compile_for_inference, CompilerConfig};

let compiler_config = CompilerConfig {
    backend: "inductor",
    optimize_for_latency: true,
    enable_fusion: true,
    target_hardware: "nvidia-a100",
};

let compiled_model = compile_for_inference(&model, &compiler_config)?;
```

## Containerization

### Docker Configuration

```dockerfile
# Dockerfile for TrustformeRS model serving
FROM rust:1.75 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libssl-dev \
    pkg-config

# Copy source
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build with optimizations
RUN cargo build --release --features "cuda,serving"

# Runtime image
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary and models
COPY --from=builder /app/target/release/trustformers-serve /usr/local/bin/
COPY models /models

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Configuration
ENV RUST_LOG=info
ENV MODEL_PATH=/models
ENV BIND_ADDRESS=0.0.0.0:8080

EXPOSE 8080

CMD ["trustformers-serve"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  model-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
      - ./config:/config:ro
    environment:
      - RUST_LOG=info
      - MODEL_CONFIG=/config/serving.toml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Serving Strategies

### 1. HTTP REST API Server

```rust
use trustformers::serving::{ModelServer, ServerConfig};
use axum::{Router, Json};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct PredictRequest {
    text: String,
    max_length: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    output: String,
    tokens_generated: usize,
    latency_ms: f64,
}

// Create server
let server_config = ServerConfig {
    model_path: "/models/my-model",
    max_batch_size: 32,
    max_sequence_length: 2048,
    timeout_ms: 30000,
    enable_metrics: true,
};

let model_server = ModelServer::new(server_config)?;

// Define routes
let app = Router::new()
    .route("/predict", post(predict_handler))
    .route("/health", get(health_handler))
    .route("/metrics", get(metrics_handler))
    .with_state(model_server);

// Handler implementation
async fn predict_handler(
    State(server): State<ModelServer>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, StatusCode> {
    let start = Instant::now();
    
    let output = server
        .predict(&request.text, request.max_length, request.temperature)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(PredictResponse {
        output: output.text,
        tokens_generated: output.token_count,
        latency_ms: start.elapsed().as_millis() as f64,
    }))
}
```

### 2. gRPC Service

```proto
// trustformers.proto
syntax = "proto3";

package trustformers;

service ModelService {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc StreamPredict(PredictRequest) returns (stream StreamResponse);
    rpc BatchPredict(BatchPredictRequest) returns (BatchPredictResponse);
}

message PredictRequest {
    string text = 1;
    int32 max_length = 2;
    float temperature = 3;
    float top_p = 4;
}

message PredictResponse {
    string output = 1;
    int32 tokens_generated = 2;
    float latency_ms = 3;
}
```

```rust
use tonic::{transport::Server, Request, Response, Status};
use trustformers::serving::grpc::{ModelServiceServer, ModelServiceImpl};

// Implement gRPC service
impl ModelService for ModelServiceImpl {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        
        let output = self.model
            .generate(&req.text, req.max_length as usize)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        Ok(Response::new(PredictResponse {
            output: output.text,
            tokens_generated: output.token_count as i32,
            latency_ms: output.latency_ms,
        }))
    }
}

// Start gRPC server
let addr = "[::1]:50051".parse()?;
let model_service = ModelServiceImpl::new(model)?;

Server::builder()
    .add_service(ModelServiceServer::new(model_service))
    .serve(addr)
    .await?;
```

### 3. WebSocket for Streaming

```rust
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};

async fn handle_websocket(
    ws_stream: WebSocketStream<TcpStream>,
    model: Arc<Model>,
) {
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    
    while let Some(msg) = ws_receiver.next().await {
        if let Ok(Message::Text(text)) = msg {
            let request: GenerateRequest = serde_json::from_str(&text).unwrap();
            
            // Stream tokens as they're generated
            let mut token_stream = model.generate_stream(&request.prompt).await;
            
            while let Some(token) = token_stream.next().await {
                let response = json!({
                    "token": token.text,
                    "is_final": token.is_final,
                });
                
                ws_sender
                    .send(Message::Text(response.to_string()))
                    .await
                    .ok();
            }
        }
    }
}
```

## Infrastructure Setup

### 1. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trustformers-model-server
  labels:
    app: trustformers
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trustformers
  template:
    metadata:
      labels:
        app: trustformers
    spec:
      containers:
      - name: model-server
        image: trustformers/model-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/models/production"
        - name: MAX_BATCH_SIZE
          value: "32"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: trustformers-service
spec:
  selector:
    app: trustformers
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trustformers-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trustformers-model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "30"
```

### 2. AWS ECS Task Definition

```json
{
  "family": "trustformers-model-server",
  "taskRoleArn": "arn:aws:iam::account-id:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account-id:role/ecsExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "placementConstraints": [
    {
      "type": "memberOf",
      "expression": "attribute:ecs.instance-type =~ p3.*"
    }
  ],
  "cpu": "4096",
  "memory": "15360",
  "containerDefinitions": [
    {
      "name": "model-server",
      "image": "trustformers/model-server:latest",
      "memoryReservation": 14336,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_S3_PATH",
          "value": "s3://my-models/production/latest"
        },
        {
          "name": "AWS_REGION",
          "value": "us-west-2"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trustformers",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "model-server"
        }
      }
    }
  ]
}
```

## Monitoring and Observability

### 1. Metrics Collection

```rust
use prometheus::{Registry, Counter, Histogram, Gauge};
use trustformers::monitoring::{MetricsCollector, MetricConfig};

// Define metrics
lazy_static! {
    static ref REQUEST_COUNTER: Counter = Counter::new(
        "trustformers_requests_total", 
        "Total number of requests"
    ).unwrap();
    
    static ref REQUEST_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "trustformers_request_duration_seconds",
            "Request duration in seconds"
        ).buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();
    
    static ref MODEL_QUEUE_SIZE: Gauge = Gauge::new(
        "trustformers_queue_size",
        "Current queue size"
    ).unwrap();
    
    static ref GPU_UTILIZATION: Gauge = Gauge::new(
        "trustformers_gpu_utilization_percent",
        "GPU utilization percentage"
    ).unwrap();
}

// Instrument your code
pub async fn instrumented_predict(
    model: &Model,
    input: &str,
) -> Result<Output> {
    REQUEST_COUNTER.inc();
    let timer = REQUEST_DURATION.start_timer();
    
    let result = model.predict(input).await;
    
    timer.observe_duration();
    
    result
}

// Export metrics endpoint
async fn metrics_handler() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode_to_string(&metric_families).unwrap()
}
```

### 2. Distributed Tracing

```rust
use opentelemetry::{trace::{Tracer, SpanKind}, global};
use opentelemetry_jaeger::JaegerTracer;

// Initialize tracer
fn init_tracer() -> Result<JaegerTracer> {
    opentelemetry_jaeger::new_pipeline()
        .with_service_name("trustformers-model-server")
        .with_endpoint("http://jaeger:14268/api/traces")
        .install_batch(opentelemetry::runtime::Tokio)
}

// Trace inference requests
pub async fn traced_inference(
    model: &Model,
    request: InferenceRequest,
) -> Result<InferenceResponse> {
    let tracer = global::tracer("trustformers");
    
    let mut span = tracer
        .span_builder("model.inference")
        .with_kind(SpanKind::Server)
        .with_attributes(vec![
            KeyValue::new("model.name", model.name()),
            KeyValue::new("input.length", request.text.len() as i64),
        ])
        .start(&tracer);
    
    // Tokenization span
    let tokens = {
        let _tokenize_span = tracer
            .span_builder("model.tokenize")
            .start(&tracer);
        
        model.tokenize(&request.text)?
    };
    
    // Forward pass span
    let output = {
        let _forward_span = tracer
            .span_builder("model.forward")
            .with_attributes(vec![
                KeyValue::new("batch_size", 1),
                KeyValue::new("sequence_length", tokens.len() as i64),
            ])
            .start(&tracer);
        
        model.forward(&tokens)?
    };
    
    span.set_attribute(KeyValue::new("output.tokens", output.len() as i64));
    
    Ok(InferenceResponse::from(output))
}
```

### 3. Logging Best Practices

```rust
use tracing::{info, warn, error, debug, instrument};
use tracing_subscriber::prelude::*;

// Setup structured logging
pub fn setup_logging() {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true);
    
    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

// Instrument functions
#[instrument(skip(model), fields(model_name = %model.name()))]
pub async fn serve_request(
    model: &Model,
    request: Request,
) -> Result<Response> {
    info!(
        request_id = %request.id,
        input_length = request.text.len(),
        "Processing inference request"
    );
    
    let start = Instant::now();
    
    match model.generate(&request.text, request.params).await {
        Ok(output) => {
            info!(
                request_id = %request.id,
                duration_ms = start.elapsed().as_millis(),
                tokens_generated = output.token_count,
                "Request completed successfully"
            );
            Ok(Response::Success(output))
        }
        Err(e) => {
            error!(
                request_id = %request.id,
                error = %e,
                "Request failed"
            );
            Err(e)
        }
    }
}
```

## Security Best Practices

### 1. API Authentication

```rust
use jsonwebtoken::{encode, decode, Header, Validation};
use axum::middleware::from_fn;

// JWT middleware
async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = headers
        .get(AUTHORIZATION)
        .and_then(|auth| auth.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let claims = decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_SECRET.as_ref()),
        &Validation::default(),
    )
    .map_err(|_| StatusCode::UNAUTHORIZED)?
    .claims;
    
    // Add claims to request extensions
    request.extensions_mut().insert(claims);
    
    Ok(next.run(request).await)
}

// Apply to routes
let app = Router::new()
    .route("/predict", post(predict_handler))
    .layer(from_fn(auth_middleware));
```

### 2. Input Validation

```rust
use validator::{Validate, ValidationError};

#[derive(Deserialize, Validate)]
struct PredictRequest {
    #[validate(length(min = 1, max = 10000))]
    text: String,
    
    #[validate(range(min = 1, max = 2048))]
    max_length: Option<u32>,
    
    #[validate(range(min = 0.1, max = 2.0))]
    temperature: Option<f32>,
}

// Validate and sanitize inputs
fn validate_and_sanitize(request: &PredictRequest) -> Result<()> {
    request.validate()?;
    
    // Additional security checks
    if contains_injection_patterns(&request.text) {
        return Err(Error::InvalidInput("Potentially malicious input detected"));
    }
    
    Ok(())
}
```

### 3. Rate Limiting

```rust
use tower_governor::{Governor, GovernorConfigBuilder};

// Configure rate limiter
let governor_conf = GovernorConfigBuilder::default()
    .per_second(10) // 10 requests per second
    .burst_size(30)
    .finish()
    .unwrap();

let app = Router::new()
    .route("/predict", post(predict_handler))
    .layer(Governor::new(governor_conf));

// Per-user rate limiting
let rate_limiter = RateLimiter::new(
    RedisStore::new("redis://localhost:6379"),
    RateLimitConfig {
        window_secs: 60,
        max_requests: 100,
    },
);

async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let user_id = extract_user_id(&headers)?;
    
    if !limiter.check_rate_limit(&user_id).await? {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    
    Ok(next.run(request).await)
}
```

## Performance Optimization

### 1. Request Batching

```rust
use trustformers::serving::{BatchProcessor, BatchConfig};

// Configure dynamic batching
let batch_config = BatchConfig {
    max_batch_size: 32,
    max_wait_time_ms: 50,
    enable_padding_optimization: true,
};

let batch_processor = BatchProcessor::new(model, batch_config);

// Process requests in batches
let results = batch_processor
    .process_requests(incoming_requests)
    .await?;
```

### 2. Caching Strategy

```rust
use moka::future::Cache;

// Create LRU cache for frequent queries
let cache: Cache<String, PredictResponse> = Cache::builder()
    .max_capacity(10_000)
    .time_to_live(Duration::from_secs(3600))
    .build();

async fn cached_predict(
    cache: &Cache<String, PredictResponse>,
    model: &Model,
    request: &PredictRequest,
) -> Result<PredictResponse> {
    let cache_key = format!("{:?}", request);
    
    // Check cache
    if let Some(cached) = cache.get(&cache_key).await {
        return Ok(cached);
    }
    
    // Compute and cache
    let response = model.predict(request).await?;
    cache.insert(cache_key, response.clone()).await;
    
    Ok(response)
}
```

### 3. Connection Pooling

```rust
use deadpool::managed::{Pool, Manager};

// Database connection pool
let db_pool = Pool::builder(PostgresManager::new(db_config))
    .max_size(32)
    .wait_timeout(Some(Duration::from_secs(30)))
    .runtime(Runtime::Tokio1)
    .build()?;

// Redis connection pool for caching
let redis_pool = Pool::builder(RedisManager::new(redis_url)?)
    .max_size(16)
    .build()?;
```

## Scaling Strategies

### 1. Horizontal Scaling

```rust
use trustformers::cluster::{ClusterManager, NodeConfig};

// Setup cluster manager
let cluster_config = ClusterConfig {
    node_discovery: "kubernetes", // or "consul", "etcd"
    health_check_interval_secs: 30,
    rebalance_threshold: 0.2,
};

let cluster_manager = ClusterManager::new(cluster_config)?;

// Register node
cluster_manager.register_node(NodeConfig {
    id: node_id(),
    capacity: ModelCapacity {
        max_concurrent_requests: 100,
        max_batch_size: 32,
    },
})?;

// Load balancing
let load_balancer = LoadBalancer::new(
    cluster_manager,
    BalancingStrategy::LeastConnections,
);
```

### 2. Auto-scaling Configuration

```yaml
# Kubernetes VPA (Vertical Pod Autoscaler)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: trustformers-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trustformers-model-server
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: model-server
      minAllowed:
        cpu: "2"
        memory: "4Gi"
      maxAllowed:
        cpu: "16"
        memory: "32Gi"
        
# Custom metrics for scaling
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: kube-system
data:
  config.yaml: |
    rules:
    - seriesQuery: 'trustformers_queue_depth{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^trustformers_(.*)$"
        as: "${1}"
      metricsQuery: 'avg_over_time(<<.Series>>{<<.LabelMatchers>>}[2m])'
```

## Deployment Scenarios

### 1. Edge Deployment

```rust
use trustformers::edge::{EdgeOptimizer, EdgeConfig};

// Optimize for edge devices
let edge_config = EdgeConfig {
    max_model_size_mb: 500,
    quantization_bits: 4,
    enable_model_splitting: true,
    target_device: "raspberry-pi-4",
};

let edge_model = EdgeOptimizer::optimize(model, edge_config)?;

// Deploy with resource constraints
let edge_server = EdgeServer::new(
    edge_model,
    ServerConfig {
        max_memory_mb: 2048,
        cpu_threads: 4,
        enable_swap: false,
    },
)?;
```

### 2. Multi-Region Deployment

```rust
use trustformers::geo::{RegionManager, ReplicationConfig};

// Configure multi-region setup
let region_config = ReplicationConfig {
    primary_region: "us-east-1",
    replica_regions: vec!["eu-west-1", "ap-southeast-1"],
    sync_strategy: SyncStrategy::Eventual,
    failover_timeout_secs: 30,
};

let region_manager = RegionManager::new(region_config)?;

// Route requests based on latency
async fn route_request(
    region_manager: &RegionManager,
    request: Request,
) -> Result<Response> {
    let optimal_region = region_manager
        .find_optimal_region(&request.client_location)
        .await?;
    
    optimal_region.process_request(request).await
}
```

### 3. Hybrid Cloud Deployment

```rust
use trustformers::hybrid::{HybridOrchestrator, CloudProvider};

// Configure hybrid deployment
let orchestrator = HybridOrchestrator::new()
    .add_provider(CloudProvider::AWS, aws_config)
    .add_provider(CloudProvider::OnPremise, on_prem_config)
    .with_burst_policy(BurstPolicy {
        threshold_qps: 1000,
        burst_to: CloudProvider::AWS,
        scale_down_after_mins: 15,
    });

// Handle traffic with bursting
orchestrator.handle_request(request).await?;
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

```rust
// Monitor memory usage
use trustformers::monitoring::MemoryMonitor;

let memory_monitor = MemoryMonitor::new();
memory_monitor.set_threshold_percent(90);

memory_monitor.on_threshold_exceeded(|| {
    warn!("Memory usage critical, clearing caches");
    clear_model_cache();
    force_garbage_collection();
});
```

#### 2. High Latency Spikes

```rust
// Diagnose latency issues
use trustformers::profiling::{LatencyProfiler, trace_request};

#[trace_request]
async fn diagnose_slow_request(model: &Model, input: &str) -> Result<Output> {
    let profiler = LatencyProfiler::new();
    
    profiler.mark("tokenization_start");
    let tokens = model.tokenize(input)?;
    profiler.mark("tokenization_end");
    
    profiler.mark("inference_start");
    let output = model.forward(&tokens)?;
    profiler.mark("inference_end");
    
    if profiler.total_duration() > Duration::from_millis(100) {
        warn!("Slow request detected: {}", profiler.report());
    }
    
    Ok(output)
}
```

#### 3. Model Loading Failures

```rust
// Robust model loading
use trustformers::deployment::{ModelLoader, LoadStrategy};

let loader = ModelLoader::new()
    .with_retry_policy(RetryPolicy {
        max_attempts: 3,
        backoff_ms: 1000,
    })
    .with_fallback_path("/models/backup");

let model = match loader.load(primary_path).await {
    Ok(model) => model,
    Err(e) => {
        error!("Failed to load primary model: {}", e);
        loader.load_fallback().await?
    }
};
```

## Best Practices Summary

1. **Always validate models before deployment**
2. **Use health checks and readiness probes**
3. **Implement proper monitoring and alerting**
4. **Enable request tracing for debugging**
5. **Use connection pooling for external services**
6. **Implement circuit breakers for resilience**
7. **Cache frequently requested results**
8. **Use rate limiting to prevent abuse**
9. **Regularly update dependencies and base images**
10. **Test deployment configurations in staging**

## Next Steps

- Review the [Performance Tuning Guide](./performance_tuning.md) for optimization techniques
- Check [Security Guidelines](./security.md) for additional security measures
- See [Monitoring Setup](./monitoring.md) for detailed observability configuration
- Join our [Deployment Community](https://github.com/trustformers/trustformers/discussions/categories/deployment) for support

Remember: Always test your deployment configuration thoroughly in a staging environment before going to production!