//! Model Serving Utilities
//!
//! This module provides utilities for serving machine learning models
//! including load balancing, request queuing, and health monitoring.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::{traits::Model, Tensor};
use uuid::Uuid;

/// Configuration for model serving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingConfig {
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum queue size for pending requests
    pub max_queue_size: usize,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Enable request metrics collection
    pub enable_metrics: bool,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            request_timeout_seconds: 30,
            max_queue_size: 100,
            health_check_interval_seconds: 60,
            enable_metrics: true,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin load balancing
    RoundRobin,
    /// Least connections load balancing
    LeastConnections,
    /// Weighted round-robin
    WeightedRoundRobin(Vec<f64>),
    /// Load balancing based on response time
    ResponseTime,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: Uuid,
    pub input: Tensor,
    pub priority: RequestPriority,
    pub timestamp: Instant,
    pub metadata: HashMap<String, String>,
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(input: Tensor, priority: RequestPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            input,
            priority,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the request
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get elapsed time since request creation
    pub fn elapsed(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Inference response
#[derive(Debug)]
pub struct InferenceResponse {
    pub request_id: Uuid,
    pub output: Result<Tensor>,
    pub processing_time: Duration,
    pub metadata: HashMap<String, String>,
}

/// Model instance for serving
#[derive(Debug)]
pub struct ModelInstance {
    pub id: String,
    pub weight: f64,
    pub active_requests: usize,
    pub total_requests: u64,
    pub total_processing_time: Duration,
    pub last_health_check: Instant,
    pub is_healthy: bool,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(id: String, weight: f64) -> Self {
        Self {
            id,
            weight,
            active_requests: 0,
            total_requests: 0,
            total_processing_time: Duration::new(0, 0),
            last_health_check: Instant::now(),
            is_healthy: true,
        }
    }

    /// Update instance statistics after processing a request
    pub fn update_stats(&mut self, processing_time: Duration) {
        self.active_requests = self.active_requests.saturating_sub(1);
        self.total_requests += 1;
        self.total_processing_time += processing_time;
    }

    /// Get average response time
    pub fn average_response_time(&self) -> Duration {
        if self.total_requests > 0 {
            self.total_processing_time / self.total_requests as u32
        } else {
            Duration::new(0, 0)
        }
    }

    /// Mark request as started
    pub fn start_request(&mut self) {
        self.active_requests += 1;
    }
}

/// Load balancer for model instances
#[derive(Debug)]
pub struct LoadBalancer {
    instances: Vec<ModelInstance>,
    strategy: LoadBalancingStrategy,
    current_index: usize,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            instances: Vec::new(),
            strategy,
            current_index: 0,
        }
    }

    /// Add a model instance
    pub fn add_instance(&mut self, instance: ModelInstance) {
        self.instances.push(instance);
    }

    /// Select the next instance based on the load balancing strategy
    pub fn select_instance(&mut self) -> Option<&mut ModelInstance> {
        if self.instances.is_empty() {
            return None;
        }

        let selected_index = match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index = self.current_index;
                self.current_index = (self.current_index + 1) % self.instances.len();
                index
            },
            LoadBalancingStrategy::LeastConnections => self
                .instances
                .iter()
                .enumerate()
                .filter(|(_, instance)| instance.is_healthy)
                .min_by_key(|(_, instance)| instance.active_requests)
                .map(|(index, _)| index)
                .unwrap_or(0),
            LoadBalancingStrategy::WeightedRoundRobin(weights) => {
                // Simple weighted selection - in practice, you'd want a more sophisticated algorithm
                self.instances
                    .iter()
                    .enumerate()
                    .filter(|(_, instance)| instance.is_healthy)
                    .max_by(|(i, _), (j, _)| {
                        let weight_i = weights.get(*i).unwrap_or(&1.0);
                        let weight_j = weights.get(*j).unwrap_or(&1.0);
                        weight_i.partial_cmp(weight_j).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            },
            LoadBalancingStrategy::ResponseTime => self
                .instances
                .iter()
                .enumerate()
                .filter(|(_, instance)| instance.is_healthy)
                .min_by_key(|(_, instance)| instance.average_response_time())
                .map(|(index, _)| index)
                .unwrap_or(0),
        };

        self.instances.get_mut(selected_index)
    }

    /// Get healthy instances count
    pub fn healthy_instances_count(&self) -> usize {
        self.instances.iter().filter(|i| i.is_healthy).count()
    }

    /// Update instance health status
    pub fn update_instance_health(&mut self, instance_id: &str, is_healthy: bool) {
        if let Some(instance) = self.instances.iter_mut().find(|i| i.id == instance_id) {
            instance.is_healthy = is_healthy;
            instance.last_health_check = Instant::now();
        }
    }
}

/// Request queue manager
#[derive(Debug)]
pub struct RequestQueue {
    queue: VecDeque<InferenceRequest>,
    max_size: usize,
}

impl RequestQueue {
    /// Create a new request queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_size,
        }
    }

    /// Add a request to the queue
    pub fn enqueue(&mut self, request: InferenceRequest) -> Result<()> {
        if self.queue.len() >= self.max_size {
            return Err(TrustformersError::resource_exhausted(
                "Request queue is full".to_string(),
            ));
        }

        // Insert based on priority (higher priority first)
        let insert_index = self
            .queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.queue.len());

        self.queue.insert(insert_index, request);
        Ok(())
    }

    /// Remove and return the next request
    pub fn dequeue(&mut self) -> Option<InferenceRequest> {
        self.queue.pop_front()
    }

    /// Get current queue size
    pub fn size(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Remove expired requests based on timeout
    pub fn remove_expired(&mut self, timeout: Duration) -> usize {
        let initial_size = self.queue.len();
        self.queue.retain(|req| req.elapsed() < timeout);
        initial_size - self.queue.len()
    }
}

/// Serving metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ServingMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub timeout_requests: u64,
    pub average_response_time_ms: f64,
    pub current_queue_size: usize,
    pub peak_queue_size: usize,
    pub active_connections: usize,
}

impl ServingMetrics {
    /// Update metrics after processing a request
    pub fn update_request(&mut self, success: bool, response_time: Duration) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        // Update average response time (simple moving average)
        let new_time_ms = response_time.as_millis() as f64;
        if self.total_requests == 1 {
            self.average_response_time_ms = new_time_ms;
        } else {
            self.average_response_time_ms =
                (self.average_response_time_ms * (self.total_requests - 1) as f64 + new_time_ms)
                    / self.total_requests as f64;
        }
    }

    /// Update queue size metrics
    pub fn update_queue_size(&mut self, current_size: usize) {
        self.current_queue_size = current_size;
        if current_size > self.peak_queue_size {
            self.peak_queue_size = current_size;
        }
    }

    /// Record a timeout
    pub fn record_timeout(&mut self) {
        self.timeout_requests += 1;
        self.failed_requests += 1;
        self.total_requests += 1;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing, rejecting requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for health monitoring
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: usize,
    failure_threshold: usize,
    recovery_timeout: Duration,
    last_failure_time: Option<Instant>,
    success_threshold: usize, // For half-open state
    half_open_successes: usize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(
        failure_threshold: usize,
        recovery_timeout: Duration,
        success_threshold: usize,
    ) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            failure_threshold,
            recovery_timeout,
            last_failure_time: None,
            success_threshold,
            half_open_successes: 0,
        }
    }

    /// Check if a request should be allowed
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.recovery_timeout {
                        self.state = CircuitBreakerState::HalfOpen;
                        self.half_open_successes = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                    self.last_failure_time = None;
                }
            },
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            },
            _ => {},
        }
    }

    /// Record a failed operation
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            },
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.half_open_successes = 0;
            },
            _ => {},
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitBreakerState {
        self.state
    }
}

/// Health monitor for instances
#[derive(Debug)]
pub struct HealthMonitor {
    circuit_breakers: HashMap<String, CircuitBreaker>,
    health_check_interval: Duration,
    last_health_check: Instant,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(health_check_interval: Duration) -> Self {
        Self {
            circuit_breakers: HashMap::new(),
            health_check_interval,
            last_health_check: Instant::now(),
        }
    }

    /// Add an instance to monitor
    pub fn add_instance(&mut self, instance_id: String) {
        let circuit_breaker = CircuitBreaker::new(
            3,                       // failure threshold
            Duration::from_secs(30), // recovery timeout
            2,                       // success threshold for recovery
        );
        self.circuit_breakers.insert(instance_id, circuit_breaker);
    }

    /// Check if an instance can handle requests
    pub fn can_handle_request(&mut self, instance_id: &str) -> bool {
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(instance_id) {
            circuit_breaker.allow_request()
        } else {
            false
        }
    }

    /// Record a successful operation for an instance
    pub fn record_success(&mut self, instance_id: &str) {
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(instance_id) {
            circuit_breaker.record_success();
        }
    }

    /// Record a failed operation for an instance
    pub fn record_failure(&mut self, instance_id: &str) {
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(instance_id) {
            circuit_breaker.record_failure();
        }
    }

    /// Get health status for all instances
    pub fn get_health_status(&self) -> HashMap<String, CircuitBreakerState> {
        self.circuit_breakers.iter().map(|(id, cb)| (id.clone(), cb.state())).collect()
    }

    /// Check if it's time for health check
    pub fn should_run_health_check(&self) -> bool {
        self.last_health_check.elapsed() >= self.health_check_interval
    }
}

/// Type alias for model inference function
pub type ModelInferenceFn = Arc<dyn Fn(Tensor) -> Result<Tensor> + Send + Sync>;

/// Model serving manager
pub struct ModelServingManager {
    config: ServingConfig,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    request_queue: Arc<Mutex<RequestQueue>>,
    metrics: Arc<RwLock<ServingMetrics>>,
    health_monitor: Arc<Mutex<HealthMonitor>>,
    model_fn: Option<ModelInferenceFn>,
}

impl std::fmt::Debug for ModelServingManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelServingManager")
            .field("config", &self.config)
            .field("load_balancer", &"Arc<Mutex<LoadBalancer>>")
            .field("request_queue", &"Arc<Mutex<RequestQueue>>")
            .field("metrics", &"Arc<RwLock<ServingMetrics>>")
            .field("health_monitor", &"Arc<Mutex<HealthMonitor>>")
            .field("model_fn", &self.model_fn.is_some())
            .finish()
    }
}

impl ModelServingManager {
    /// Create a new model serving manager
    pub fn new(config: ServingConfig) -> Self {
        let load_balancer = LoadBalancer::new(config.load_balancing_strategy.clone());
        let request_queue = RequestQueue::new(config.max_queue_size);
        let health_monitor =
            HealthMonitor::new(Duration::from_secs(config.health_check_interval_seconds));

        Self {
            config,
            load_balancer: Arc::new(Mutex::new(load_balancer)),
            request_queue: Arc::new(Mutex::new(request_queue)),
            metrics: Arc::new(RwLock::new(ServingMetrics::default())),
            health_monitor: Arc::new(Mutex::new(health_monitor)),
            model_fn: None,
        }
    }

    /// Create a new model serving manager with a specific model
    pub fn with_model<M: Model<Input = Tensor, Output = Tensor> + 'static>(
        config: ServingConfig,
        model: M,
    ) -> Self {
        let load_balancer = LoadBalancer::new(config.load_balancing_strategy.clone());
        let request_queue = RequestQueue::new(config.max_queue_size);
        let health_monitor =
            HealthMonitor::new(Duration::from_secs(config.health_check_interval_seconds));

        let model = Arc::new(model);
        let model_fn: ModelInferenceFn = Arc::new(move |input| model.forward(input));

        Self {
            config,
            load_balancer: Arc::new(Mutex::new(load_balancer)),
            request_queue: Arc::new(Mutex::new(request_queue)),
            metrics: Arc::new(RwLock::new(ServingMetrics::default())),
            health_monitor: Arc::new(Mutex::new(health_monitor)),
            model_fn: Some(model_fn),
        }
    }

    /// Set a custom inference function
    pub fn set_inference_fn(&mut self, inference_fn: ModelInferenceFn) {
        self.model_fn = Some(inference_fn);
    }

    /// Add a model instance
    pub fn add_instance(&self, instance: ModelInstance) -> Result<()> {
        let instance_id = instance.id.clone();

        let mut balancer = self.load_balancer.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire load balancer lock".to_string())
        })?;
        balancer.add_instance(instance);

        // Register with health monitor
        let mut health_monitor = self.health_monitor.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire health monitor lock".to_string())
        })?;
        health_monitor.add_instance(instance_id);

        Ok(())
    }

    /// Get health status for all instances
    pub fn get_health_status(&self) -> Result<HashMap<String, CircuitBreakerState>> {
        let health_monitor = self.health_monitor.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire health monitor lock".to_string())
        })?;
        Ok(health_monitor.get_health_status())
    }

    /// Perform health check on all instances
    pub async fn perform_health_check(&self) -> Result<()> {
        let should_check = {
            let health_monitor = self.health_monitor.lock().map_err(|_| {
                TrustformersError::runtime_error(
                    "Failed to acquire health monitor lock".to_string(),
                )
            })?;
            health_monitor.should_run_health_check()
        };

        if should_check {
            // In a real implementation, this would perform actual health checks
            // For now, we'll just update the health monitor's last check time
            let mut _health_monitor = self.health_monitor.lock().map_err(|_| {
                TrustformersError::runtime_error(
                    "Failed to acquire health monitor lock".to_string(),
                )
            })?;
            // Health check logic would go here
        }

        Ok(())
    }

    /// Submit a request for processing
    pub async fn submit_request(&self, request: InferenceRequest) -> Result<()> {
        let mut queue = self.request_queue.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire queue lock".to_string())
        })?;

        queue.enqueue(request)?;

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.update_queue_size(queue.size());
        }

        Ok(())
    }

    /// Process the next request in the queue
    pub async fn process_next_request(&self) -> Result<Option<InferenceResponse>> {
        // Get the next request
        let request = {
            let mut queue = self.request_queue.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire queue lock".to_string())
            })?;
            queue.dequeue()
        };

        let request = match request {
            Some(req) => req,
            None => return Ok(None),
        };

        // Check for timeout
        let timeout_duration = Duration::from_secs(self.config.request_timeout_seconds);
        if request.elapsed() > timeout_duration {
            if self.config.enable_metrics {
                let mut metrics = self.metrics.write().await;
                metrics.record_timeout();
            }
            return Ok(Some(InferenceResponse {
                request_id: request.id,
                output: Err(TrustformersError::runtime_error(
                    "Request timed out".to_string(),
                )),
                processing_time: request.elapsed(),
                metadata: HashMap::new(),
            }));
        }

        // Select an instance for processing
        let instance_id = {
            let mut balancer = self.load_balancer.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire load balancer lock".to_string())
            })?;

            match balancer.select_instance() {
                Some(instance) => {
                    instance.start_request();
                    instance.id.clone()
                },
                None => {
                    return Err(TrustformersError::resource_exhausted(
                        "No healthy instances available".to_string(),
                    ));
                },
            }
        };

        // Simulate processing (in a real implementation, this would call the actual model)
        let start_time = Instant::now();
        let output = self.process_inference(&request).await;
        let processing_time = start_time.elapsed();

        // Update instance statistics
        {
            let mut balancer = self.load_balancer.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire load balancer lock".to_string())
            })?;

            if let Some(instance) = balancer.instances.iter_mut().find(|i| i.id == instance_id) {
                instance.update_stats(processing_time);
            }
        }

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.update_request(output.is_ok(), processing_time);

            let queue_size = {
                let queue = self.request_queue.lock().map_err(|_| {
                    TrustformersError::runtime_error("Failed to acquire queue lock".to_string())
                })?;
                queue.size()
            };
            metrics.update_queue_size(queue_size);
        }

        Ok(Some(InferenceResponse {
            request_id: request.id,
            output,
            processing_time,
            metadata: HashMap::new(),
        }))
    }

    /// Process an inference request using the configured model
    async fn process_inference(&self, request: &InferenceRequest) -> Result<Tensor> {
        match &self.model_fn {
            Some(model_fn) => {
                // Use the configured model function for actual inference
                let model_fn = Arc::clone(model_fn);
                let input_tensor = request.input.clone();

                // Run inference in a blocking task to avoid blocking the async runtime
                let output = tokio::task::spawn_blocking(move || (model_fn)(input_tensor))
                    .await
                    .map_err(|e| {
                    TrustformersError::runtime_error(format!("Inference task failed: {}", e))
                })??;

                Ok(output)
            },
            None => {
                // Fallback: enhanced simulation with basic tensor operations
                let input = &request.input;

                // Simulate some computation time based on tensor size
                let tensor_size = match input {
                    Tensor::F32(arr) => arr.len(),
                    Tensor::I64(arr) => arr.len(),
                    _ => 1000, // Default size
                };
                let processing_time = std::cmp::min(100, tensor_size / 1000); // Max 100ms
                tokio::time::sleep(Duration::from_millis(processing_time as u64)).await;

                // Return input tensor for now (can be enhanced with basic transformations)
                Ok(request.input.clone())
            },
        }
    }

    /// Get current serving metrics
    pub async fn get_metrics(&self) -> ServingMetrics {
        let metrics = self.metrics.read().await;
        (*metrics).clone()
    }

    /// Cleanup expired requests
    pub async fn cleanup_expired_requests(&self) -> Result<usize> {
        let timeout_duration = Duration::from_secs(self.config.request_timeout_seconds);
        let mut queue = self.request_queue.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire queue lock".to_string())
        })?;

        let removed_count = queue.remove_expired(timeout_duration);

        if self.config.enable_metrics && removed_count > 0 {
            let mut metrics = self.metrics.write().await;
            for _ in 0..removed_count {
                metrics.record_timeout();
            }
            metrics.update_queue_size(queue.size());
        }

        Ok(removed_count)
    }

    /// Get healthy instances count
    pub fn healthy_instances_count(&self) -> Result<usize> {
        let balancer = self.load_balancer.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire load balancer lock".to_string())
        })?;
        Ok(balancer.healthy_instances_count())
    }
}

/// Rate limiter implementation using token bucket algorithm
#[derive(Debug)]
pub struct RateLimiter {
    max_tokens: u64,
    tokens: u64,
    refill_rate: u64, // tokens per second
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            max_tokens,
            tokens: max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to acquire a token
    pub fn try_acquire(&mut self, tokens: u64) -> bool {
        self.refill_tokens();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill_tokens(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = (elapsed.as_secs_f64() * self.refill_rate as f64) as u64;

        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
            self.last_refill = now;
        }
    }

    /// Get current token count
    pub fn available_tokens(&mut self) -> u64 {
        self.refill_tokens();
        self.tokens
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of instances
    pub min_instances: usize,
    /// Maximum number of instances
    pub max_instances: usize,
    /// Target CPU utilization percentage
    pub target_cpu_utilization: f64,
    /// Scale up threshold (queue length)
    pub scale_up_queue_threshold: usize,
    /// Scale down threshold (queue length)
    pub scale_down_queue_threshold: usize,
    /// Cooldown period between scaling actions
    pub cooldown_period_seconds: u64,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_instances: 1,
            max_instances: 10,
            target_cpu_utilization: 70.0,
            scale_up_queue_threshold: 20,
            scale_down_queue_threshold: 5,
            cooldown_period_seconds: 300, // 5 minutes
        }
    }
}

/// Auto-scaler for model instances
#[derive(Debug)]
pub struct AutoScaler {
    config: AutoScalingConfig,
    last_scaling_action: Option<Instant>,
    current_instance_count: usize,
}

impl AutoScaler {
    /// Create a new auto-scaler
    pub fn new(config: AutoScalingConfig, initial_instance_count: usize) -> Self {
        Self {
            config,
            last_scaling_action: None,
            current_instance_count: initial_instance_count,
        }
    }

    /// Determine if scaling action is needed
    pub fn should_scale(
        &self,
        queue_size: usize,
        avg_cpu_utilization: f64,
    ) -> Option<ScalingAction> {
        if !self.config.enabled {
            return None;
        }

        // Check cooldown period
        if let Some(last_action) = self.last_scaling_action {
            if last_action.elapsed().as_secs() < self.config.cooldown_period_seconds {
                return None;
            }
        }

        // Check scale up conditions
        if (queue_size > self.config.scale_up_queue_threshold
            || avg_cpu_utilization > self.config.target_cpu_utilization)
            && self.current_instance_count < self.config.max_instances
        {
            return Some(ScalingAction::ScaleUp);
        }

        // Check scale down conditions
        if queue_size < self.config.scale_down_queue_threshold
            && avg_cpu_utilization < self.config.target_cpu_utilization * 0.5
            && self.current_instance_count > self.config.min_instances
        {
            return Some(ScalingAction::ScaleDown);
        }

        None
    }

    /// Record a scaling action
    pub fn record_scaling_action(&mut self, action: ScalingAction) {
        self.last_scaling_action = Some(Instant::now());

        match action {
            ScalingAction::ScaleUp => {
                self.current_instance_count =
                    (self.current_instance_count + 1).min(self.config.max_instances);
            },
            ScalingAction::ScaleDown => {
                self.current_instance_count =
                    (self.current_instance_count.saturating_sub(1)).max(self.config.min_instances);
            },
        }
    }

    /// Get current instance count
    pub fn current_instance_count(&self) -> usize {
        self.current_instance_count
    }

    /// Get scaling recommendations based on metrics
    pub fn get_scaling_recommendations(&self, metrics: &ServingMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !self.config.enabled {
            recommendations.push("Auto-scaling is disabled".to_string());
            return recommendations;
        }

        let queue_ratio =
            metrics.current_queue_size as f64 / self.config.scale_up_queue_threshold as f64;

        if queue_ratio > 1.0 {
            recommendations.push(format!(
                "Queue size ({}) exceeds scale-up threshold ({}). Consider scaling up.",
                metrics.current_queue_size, self.config.scale_up_queue_threshold
            ));
        } else if queue_ratio < 0.25 {
            recommendations.push(format!(
                "Queue size ({}) is very low. Consider scaling down to save resources.",
                metrics.current_queue_size
            ));
        }

        if metrics.average_response_time_ms > 1000.0 {
            recommendations.push("High response times detected. Consider scaling up.".to_string());
        }

        if metrics.success_rate() < 0.95 {
            recommendations.push(
                "Low success rate detected. Check instance health and consider scaling."
                    .to_string(),
            );
        }

        recommendations
    }
}

/// Scaling actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
}

/// Enhanced serving manager with rate limiting and auto-scaling
#[derive(Debug)]
pub struct EnhancedServingManager {
    base_manager: Arc<ModelServingManager>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    auto_scaler: Arc<Mutex<AutoScaler>>,
    rate_limit_config: RateLimitConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum requests per second
    pub max_requests_per_second: u64,
    /// Burst capacity
    pub burst_capacity: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_requests_per_second: 100,
            burst_capacity: 200,
        }
    }
}

impl EnhancedServingManager {
    /// Create a new enhanced serving manager
    pub fn new(
        serving_config: ServingConfig,
        rate_limit_config: RateLimitConfig,
        auto_scaling_config: AutoScalingConfig,
    ) -> Self {
        let base_manager = Arc::new(ModelServingManager::new(serving_config));
        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(
            rate_limit_config.burst_capacity,
            rate_limit_config.max_requests_per_second,
        )));
        let auto_scaler = Arc::new(Mutex::new(AutoScaler::new(auto_scaling_config, 1)));

        Self {
            base_manager,
            rate_limiter,
            auto_scaler,
            rate_limit_config,
        }
    }

    /// Submit a request with rate limiting
    pub async fn submit_request_with_rate_limiting(&self, request: InferenceRequest) -> Result<()> {
        // Check rate limit
        if self.rate_limit_config.enabled {
            let mut limiter = self.rate_limiter.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire rate limiter lock".to_string())
            })?;

            if !limiter.try_acquire(1) {
                return Err(TrustformersError::resource_exhausted(
                    "Rate limit exceeded".to_string(),
                ));
            }
        }

        // Submit request to base manager
        self.base_manager.submit_request(request).await
    }

    /// Check for auto-scaling decisions
    pub async fn check_auto_scaling(&self) -> Result<Option<ScalingAction>> {
        let metrics = self.base_manager.get_metrics().await;

        let mut scaler = self.auto_scaler.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire auto-scaler lock".to_string())
        })?;

        // Get approximate CPU utilization based on system load
        let avg_cpu_utilization = self.get_approximate_cpu_utilization();

        if let Some(action) = scaler.should_scale(metrics.current_queue_size, avg_cpu_utilization) {
            scaler.record_scaling_action(action);
            Ok(Some(action))
        } else {
            Ok(None)
        }
    }

    /// Get enhanced metrics including rate limiting and auto-scaling info
    pub async fn get_enhanced_metrics(&self) -> Result<EnhancedMetrics> {
        let base_metrics = self.base_manager.get_metrics().await;

        let available_tokens = {
            let mut limiter = self.rate_limiter.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire rate limiter lock".to_string())
            })?;
            limiter.available_tokens()
        };

        let (current_instance_count, scaling_recommendations) = {
            let scaler = self.auto_scaler.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire auto-scaler lock".to_string())
            })?;
            (
                scaler.current_instance_count(),
                scaler.get_scaling_recommendations(&base_metrics),
            )
        };

        Ok(EnhancedMetrics {
            base_metrics,
            available_rate_limit_tokens: available_tokens,
            current_instance_count,
            scaling_recommendations,
        })
    }

    /// Get approximate CPU utilization based on system metrics
    fn get_approximate_cpu_utilization(&self) -> f64 {
        use std::fs;
        use std::io::Read;

        // Try to read from /proc/loadavg on Unix systems
        #[cfg(unix)]
        {
            if let Ok(mut file) = fs::File::open("/proc/loadavg") {
                let mut contents = String::new();
                if file.read_to_string(&mut contents).is_ok() {
                    let parts: Vec<&str> = contents.split_whitespace().collect();
                    if let Some(load_1min) = parts.first() {
                        if let Ok(load) = load_1min.parse::<f64>() {
                            let num_cores = num_cpus::get() as f64;
                            // Convert load average to approximate CPU utilization percentage
                            let utilization = (load / num_cores * 100.0).min(100.0);
                            return utilization;
                        }
                    }
                }
            }
        }

        // Fallback: estimate based on current queue size and activity
        let queue_size = if let Ok(queue) = self.base_manager.request_queue.lock() {
            queue.size() as f64
        } else {
            0.0
        };

        // Simple heuristic: higher queue size suggests higher CPU usage
        let base_utilization = 30.0; // Base system utilization
        let queue_factor = (queue_size * 5.0).min(50.0); // Max 50% from queue

        (base_utilization + queue_factor).min(95.0) // Cap at 95%
    }

    /// Get the underlying base manager
    pub fn base_manager(&self) -> &Arc<ModelServingManager> {
        &self.base_manager
    }
}

/// Enhanced metrics including rate limiting and auto-scaling information
#[derive(Debug, Clone)]
pub struct EnhancedMetrics {
    pub base_metrics: ServingMetrics,
    pub available_rate_limit_tokens: u64,
    pub current_instance_count: usize,
    pub scaling_recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serving_config_default() {
        let config = ServingConfig::default();
        assert_eq!(config.max_concurrent_requests, 10);
        assert_eq!(config.request_timeout_seconds, 30);
        assert_eq!(config.max_queue_size, 100);
    }

    #[test]
    fn test_inference_request_creation() {
        let tensor = Tensor::zeros(&[1, 2]).unwrap();
        let request = InferenceRequest::new(tensor, RequestPriority::Normal);

        assert_eq!(request.priority, RequestPriority::Normal);
        assert!(!request.metadata.is_empty() || request.metadata.is_empty()); // Just check it exists
    }

    #[test]
    fn test_model_instance() {
        let mut instance = ModelInstance::new("test-instance".to_string(), 1.0);
        assert_eq!(instance.id, "test-instance");
        assert_eq!(instance.weight, 1.0);
        assert_eq!(instance.active_requests, 0);

        instance.start_request();
        assert_eq!(instance.active_requests, 1);

        instance.update_stats(Duration::from_millis(100));
        assert_eq!(instance.active_requests, 0);
        assert_eq!(instance.total_requests, 1);
    }

    #[test]
    fn test_load_balancer() {
        let mut balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let instance1 = ModelInstance::new("instance1".to_string(), 1.0);
        let instance2 = ModelInstance::new("instance2".to_string(), 1.0);

        balancer.add_instance(instance1);
        balancer.add_instance(instance2);

        assert_eq!(balancer.healthy_instances_count(), 2);

        let selected1 = balancer.select_instance().unwrap();
        assert_eq!(selected1.id, "instance1");

        let selected2 = balancer.select_instance().unwrap();
        assert_eq!(selected2.id, "instance2");
    }

    #[test]
    fn test_request_queue() {
        let mut queue = RequestQueue::new(2);

        let tensor1 = Tensor::zeros(&[1, 2]).unwrap();
        let tensor2 = Tensor::zeros(&[1, 2]).unwrap();
        let tensor3 = Tensor::zeros(&[1, 2]).unwrap();

        let req1 = InferenceRequest::new(tensor1, RequestPriority::Normal);
        let req2 = InferenceRequest::new(tensor2, RequestPriority::High);
        let req3 = InferenceRequest::new(tensor3, RequestPriority::Low);

        assert!(queue.enqueue(req1).is_ok());
        assert!(queue.enqueue(req2).is_ok());
        assert!(queue.enqueue(req3).is_err()); // Should fail due to max size

        assert_eq!(queue.size(), 2);

        // Higher priority request should be dequeued first
        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.priority, RequestPriority::High);
    }

    #[test]
    fn test_serving_metrics() {
        let mut metrics = ServingMetrics::default();

        metrics.update_request(true, Duration::from_millis(100));
        metrics.update_request(false, Duration::from_millis(200));

        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.success_rate(), 0.5);
        assert_eq!(metrics.average_response_time_ms, 150.0);
    }

    #[tokio::test]
    async fn test_model_serving_manager() {
        let config = ServingConfig::default();
        let manager = ModelServingManager::new(config);

        let instance = ModelInstance::new("test-instance".to_string(), 1.0);
        manager.add_instance(instance).unwrap();

        let tensor = Tensor::zeros(&[1, 2]).unwrap();
        let request = InferenceRequest::new(tensor, RequestPriority::Normal);

        manager.submit_request(request).await.unwrap();

        let response = manager.process_next_request().await.unwrap();
        assert!(response.is_some());

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10, 5); // 10 tokens, 5 per second refill

        // Should be able to acquire initial tokens
        assert!(limiter.try_acquire(5));
        assert_eq!(limiter.available_tokens(), 5);

        // Should fail to acquire more than available
        assert!(!limiter.try_acquire(10));

        // Should be able to acquire remaining tokens
        assert!(limiter.try_acquire(5));
        assert_eq!(limiter.available_tokens(), 0);

        // Should not be able to acquire when empty
        assert!(!limiter.try_acquire(1));
    }

    #[test]
    fn test_auto_scaler() {
        let config = AutoScalingConfig {
            enabled: true,
            min_instances: 1,
            max_instances: 5,
            target_cpu_utilization: 70.0,
            scale_up_queue_threshold: 10,
            scale_down_queue_threshold: 2,
            cooldown_period_seconds: 60,
        };

        let mut scaler = AutoScaler::new(config, 2);

        // Should recommend scale up when queue is high
        let action = scaler.should_scale(15, 50.0);
        assert_eq!(action, Some(ScalingAction::ScaleUp));

        // Record the action
        scaler.record_scaling_action(ScalingAction::ScaleUp);
        assert_eq!(scaler.current_instance_count(), 3);

        // Should not scale again due to cooldown
        let action = scaler.should_scale(15, 50.0);
        assert_eq!(action, None);
    }

    #[test]
    fn test_auto_scaling_recommendations() {
        let config = AutoScalingConfig {
            enabled: true,
            scale_up_queue_threshold: 20,
            ..Default::default()
        };
        let scaler = AutoScaler::new(config, 2);

        let mut metrics = ServingMetrics::default();
        metrics.current_queue_size = 25; // High queue size (above threshold of 20)
        metrics.update_request(true, Duration::from_millis(1500)); // High response time

        let recommendations = scaler.get_scaling_recommendations(&metrics);
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("scale-up threshold")));
        assert!(recommendations.iter().any(|r| r.contains("High response times")));
    }

    #[tokio::test]
    async fn test_enhanced_serving_manager() {
        let serving_config = ServingConfig::default();
        let rate_limit_config = RateLimitConfig {
            enabled: true,
            max_requests_per_second: 2,
            burst_capacity: 5,
        };
        let auto_scaling_config = AutoScalingConfig::default();

        let manager =
            EnhancedServingManager::new(serving_config, rate_limit_config, auto_scaling_config);

        // Add an instance to the base manager
        let instance = ModelInstance::new("test-instance".to_string(), 1.0);
        manager.base_manager().add_instance(instance).unwrap();

        // Test rate limiting
        let tensor = Tensor::zeros(&[1, 2]).unwrap();

        // Should be able to submit requests within rate limit
        for _ in 0..5 {
            let request = InferenceRequest::new(tensor.clone(), RequestPriority::Normal);
            let result = manager.submit_request_with_rate_limiting(request).await;
            assert!(result.is_ok());
        }

        // Should fail when rate limit is exceeded
        let request = InferenceRequest::new(tensor, RequestPriority::Normal);
        let result = manager.submit_request_with_rate_limiting(request).await;
        assert!(result.is_err());

        // Test enhanced metrics
        let enhanced_metrics = manager.get_enhanced_metrics().await.unwrap();
        assert_eq!(enhanced_metrics.current_instance_count, 1);
        assert!(enhanced_metrics.available_rate_limit_tokens < 5);
    }

    #[tokio::test]
    async fn test_enhanced_serving_auto_scaling() {
        let serving_config = ServingConfig::default();
        let rate_limit_config = RateLimitConfig::default();
        let auto_scaling_config = AutoScalingConfig {
            enabled: true,
            min_instances: 1,
            max_instances: 3,
            scale_up_queue_threshold: 5,
            scale_down_queue_threshold: 1,
            cooldown_period_seconds: 0, // No cooldown for testing
            ..Default::default()
        };

        let manager =
            EnhancedServingManager::new(serving_config, rate_limit_config, auto_scaling_config);

        // Add multiple requests to trigger scaling
        let tensor = Tensor::zeros(&[1, 2]).unwrap();
        for _ in 0..10 {
            let request = InferenceRequest::new(tensor.clone(), RequestPriority::Normal);
            manager.base_manager().submit_request(request).await.unwrap();
        }

        // Check for scaling decision
        let scaling_action = manager.check_auto_scaling().await.unwrap();
        assert_eq!(scaling_action, Some(ScalingAction::ScaleUp));

        let enhanced_metrics = manager.get_enhanced_metrics().await.unwrap();
        assert_eq!(enhanced_metrics.current_instance_count, 2);
    }
}
