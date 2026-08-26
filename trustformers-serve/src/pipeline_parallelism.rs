//! Pipeline Parallelism for TrustformeRS Inference Server
//!
//! Implements pipeline parallelism to enable efficient processing of large models
//! by splitting computation across multiple stages that can run in parallel.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock, Semaphore};
use uuid::Uuid;

/// Pipeline stage identifier
pub type StageId = usize;

/// Request identifier for tracking through pipeline
pub type PipelineRequestId = Uuid;

/// Pipeline parallelism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Buffer size for inter-stage communication
    pub stage_buffer_size: usize,
    /// Maximum number of concurrent requests per stage
    pub max_concurrent_per_stage: usize,
    /// Stage timeout in milliseconds
    pub stage_timeout_ms: u64,
    /// Enable adaptive stage assignment
    pub enable_adaptive_assignment: bool,
    /// Pipeline warmup time in seconds
    pub warmup_time_seconds: u64,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Enable stage profiling
    pub enable_profiling: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 4,
            stage_buffer_size: 32,
            max_concurrent_per_stage: 8,
            stage_timeout_ms: 5000,
            enable_adaptive_assignment: true,
            warmup_time_seconds: 30,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            enable_profiling: true,
        }
    }
}

/// Load balancing strategy for pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Assign to least loaded stage
    LeastLoaded,
    /// Assign based on stage capacity
    CapacityBased,
    /// Adaptive assignment based on performance
    Adaptive,
}

/// Pipeline request wrapper
#[derive(Debug, Clone)]
pub struct PipelineRequest {
    /// Unique request identifier
    pub id: PipelineRequestId,
    /// Original request data
    pub data: Vec<u8>,
    /// Request metadata
    pub metadata: RequestMetadata,
    /// Stage-specific intermediate results
    pub stage_results: HashMap<StageId, StageResult>,
    /// Request creation time
    pub created_at: Instant,
    /// Stage entry times
    pub stage_times: HashMap<StageId, Instant>,
}

/// Request metadata for pipeline processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Model ID for processing
    pub model_id: String,
    /// Request priority
    pub priority: RequestPriority,
    /// Required device type
    pub device_type: DeviceType,
    /// Estimated processing complexity
    pub complexity_score: f32,
    /// Client-provided correlation ID
    pub correlation_id: Option<String>,
}

/// Request priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialOrd, Ord, PartialEq, Eq)]
pub enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Device type for stage assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Gpu(usize),
    Tpu,
    Any,
}

/// Stage processing result
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Processed data
    pub data: Vec<u8>,
    /// Processing time
    pub processing_time: Duration,
    /// Stage-specific metrics
    pub metrics: StageMetrics,
    /// Whether this is the final result
    pub is_final: bool,
}

/// Per-stage metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageMetrics {
    /// Processing latency in microseconds
    pub latency_us: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f32>,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Number of operations performed
    pub operation_count: u64,
}

/// Pipeline stage definition
///
/// The two channel ends are the *same* queue: `input_tx` is how a producer —
/// [`PipelineParallelismManager::submit_request`] for the first stage, the
/// preceding stage thereafter — hands a request to this stage, and `input_rx`
/// is what this stage's processing loop reads from.
///
/// Before 0.2.1 the stage instead held its own detached `output_tx` whose
/// receiver was dropped at construction, so requests submitted to a stage went
/// nowhere and the "completed" response was fabricated by a timer.
#[derive(Debug)]
pub struct PipelineStage {
    /// Stage identifier
    pub id: StageId,
    /// Stage name/description
    pub name: String,
    /// Sending end of this stage's own input queue, cloned by whoever feeds it.
    pub input_tx: mpsc::Sender<PipelineRequest>,
    /// Receiving end of this stage's input queue, drained by its processing
    /// loop. Wrapped so the loop can hold it across awaits without keeping the
    /// stage's own lock.
    pub input_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<PipelineRequest>>>,
    /// Semaphore for controlling concurrency
    pub concurrency_limit: Arc<Semaphore>,
    /// Stage configuration
    pub config: StageConfig,
    /// Stage statistics
    pub stats: Arc<StageStats>,
}

/// Stage-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Device assignment for this stage
    pub device: DeviceType,
    /// Model layers assigned to this stage
    pub layer_range: (usize, usize),
    /// Stage-specific processing parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Memory limit for this stage
    pub memory_limit_mb: Option<usize>,
    /// Processing timeout
    pub timeout: Duration,
}

/// Stage performance statistics
#[derive(Debug, Default)]
pub struct StageStats {
    /// Total requests processed
    pub requests_processed: AtomicU64,
    /// Total processing time
    pub total_processing_time_us: AtomicU64,
    /// Current active requests
    pub active_requests: AtomicUsize,
    /// Total errors encountered
    pub error_count: AtomicU64,
    /// Peak memory usage
    pub peak_memory_usage: AtomicU64,
    /// Average queue depth
    pub avg_queue_depth: AtomicU64,
}

impl StageStats {
    pub fn record_request_start(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_complete(&self, processing_time: Duration) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_us
            .fetch_add(processing_time.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_avg_latency_us(&self) -> f64 {
        let total_requests = self.requests_processed.load(Ordering::Relaxed);
        if total_requests == 0 {
            return 0.0;
        }
        self.total_processing_time_us.load(Ordering::Relaxed) as f64 / total_requests as f64
    }
}

/// The work one pipeline stage performs on a request.
///
/// The pipeline machinery here owns scheduling, back-pressure, ordering and
/// accounting; it deliberately owns no opinion about what a "stage" computes.
/// That belongs to the caller, who installs an executor with
/// [`PipelineParallelismManager::with_stage_executor`].
///
/// Without an installed executor the manager refuses work outright — see
/// [`PipelineError::NoStageExecutor`]. It does **not** substitute a stand-in
/// that sleeps for a plausible interval and reports invented utilisation
/// figures, which is what this module did before 0.2.1: `submit_request`
/// returned a fixed 100 ms later with a `"dummy"` model id, and every stage
/// recorded a hardcoded 80% GPU utilisation and 100 operations into statistics
/// that were then exported as measurements.
#[async_trait::async_trait]
pub trait StageExecutor: Send + Sync {
    /// Run `stage_id`'s share of the work for `request`.
    ///
    /// Implementations report their own measured [`StageMetrics`]; the pipeline
    /// fills in the wall-clock latency it observed regardless, so an executor
    /// that leaves `latency_us` at zero is corrected rather than believed.
    async fn execute(
        &self,
        request: &PipelineRequest,
        stage_id: StageId,
    ) -> Result<StageResult, PipelineError>;
}

/// Main pipeline parallelism manager
#[derive(Clone)]
pub struct PipelineParallelismManager {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Pipeline stages
    stages: Arc<RwLock<Vec<Arc<RwLock<PipelineStage>>>>>,
    /// Stage assignment strategy
    assignment_strategy: Arc<RwLock<StageAssignmentStrategy>>,
    /// Pipeline statistics
    stats: Arc<PipelineStats>,
    /// Request tracker for monitoring
    request_tracker: Arc<RwLock<HashMap<PipelineRequestId, RequestTracker>>>,
    /// The work each stage performs. `None` until
    /// [`PipelineParallelismManager::with_stage_executor`] installs one, and
    /// while it is `None` the pipeline accepts no requests.
    stage_executor: Option<Arc<dyn StageExecutor>>,
    /// Completion slots for in-flight requests, keyed by request id.
    ///
    /// The final stage sends the finished request through its slot; the
    /// `submit_request` call that is waiting receives it. This is what makes the
    /// returned request the one that actually traversed the pipeline instead of
    /// a value constructed after a fixed sleep.
    completions: Arc<RwLock<HashMap<PipelineRequestId, oneshot::Sender<PipelineRequest>>>>,
}

/// Request tracking information
#[derive(Debug, Clone)]
pub struct RequestTracker {
    /// Request metadata
    pub metadata: RequestMetadata,
    /// Current stage
    pub current_stage: Option<StageId>,
    /// Stage completion times
    pub stage_completion_times: HashMap<StageId, Instant>,
    /// Total processing start time
    pub start_time: Instant,
}

/// Stage assignment strategy
#[derive(Debug)]
pub struct StageAssignmentStrategy {
    /// Current round-robin index
    rr_index: AtomicUsize,
    // 0.2.1: `stage_loads: HashMap<StageId, AtomicUsize>` and
    // `performance_history: VecDeque<StagePerformanceSnapshot>` lived here.
    // Both were constructed empty and never inserted into or read, so
    // "adaptive assignment" had no history to adapt from and load tracking
    // tracked nothing; only `rr_index` ever moved. Both are deleted rather than
    // left as containers that are permanently empty.
    // `StagePerformanceSnapshot` stays: it is the public shape a real
    // history would hold.
}

/// Performance snapshot for adaptive assignment
#[derive(Debug, Clone)]
pub struct StagePerformanceSnapshot {
    /// Stage ID
    pub stage_id: StageId,
    /// Timestamp
    pub timestamp: Instant,
    /// Average latency
    pub avg_latency_ms: f32,
    /// Queue depth
    pub queue_depth: usize,
    /// Success rate
    pub success_rate: f32,
}

/// Pipeline-wide statistics
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Total requests processed
    pub total_requests: AtomicU64,
    /// Successfully completed requests
    pub completed_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Total pipeline latency
    pub total_latency_us: AtomicU64,
    /// Current requests in pipeline
    pub active_requests: AtomicUsize,
    /// Pipeline throughput (requests per second)
    pub throughput_rps: AtomicU64,
}

impl PipelineParallelismManager {
    /// Create a new pipeline parallelism manager
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let stats = Arc::new(PipelineStats::default());
        let assignment_strategy = Arc::new(RwLock::new(StageAssignmentStrategy::new()));

        Ok(Self {
            config,
            stages: Arc::new(RwLock::new(Vec::new())),
            assignment_strategy,
            stats,
            request_tracker: Arc::new(RwLock::new(HashMap::new())),
            stage_executor: None,
            completions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Install the executor that performs each stage's work.
    ///
    /// Until one is installed, [`Self::submit_request`] fails with
    /// [`PipelineError::NoStageExecutor`] rather than returning a synthesized
    /// response.
    #[must_use]
    pub fn with_stage_executor(mut self, executor: Arc<dyn StageExecutor>) -> Self {
        self.stage_executor = Some(executor);
        self
    }

    /// Whether this pipeline can actually process a request.
    pub fn has_stage_executor(&self) -> bool {
        self.stage_executor.is_some()
    }

    /// Initialize pipeline stages
    pub async fn initialize_stages(&self, stage_configs: Vec<StageConfig>) -> Result<()> {
        if stage_configs.len() != self.config.num_stages {
            return Err(anyhow::anyhow!(
                "Stage config count ({}) doesn't match configured stages ({})",
                stage_configs.len(),
                self.config.num_stages
            ));
        }

        let mut stages = Vec::new();

        for (i, stage_config) in stage_configs.into_iter().enumerate() {
            // One queue per stage, both ends retained: the sender is what
            // producers clone, the receiver is what this stage's loop drains.
            // Retaining both is what makes the pipeline actually connected —
            // previously the receiving end of the stage's own output channel
            // was dropped immediately, so nothing could ever flow.
            let (input_tx, input_rx) = mpsc::channel(self.config.stage_buffer_size);

            let stage = PipelineStage {
                id: i,
                name: format!("Stage-{}", i),
                input_tx,
                input_rx: Arc::new(tokio::sync::Mutex::new(input_rx)),
                concurrency_limit: Arc::new(Semaphore::new(self.config.max_concurrent_per_stage)),
                config: stage_config,
                stats: Arc::new(StageStats::default()),
            };

            stages.push(Arc::new(RwLock::new(stage)));
        }

        *self.stages.write().await = stages;

        // Start stage processing tasks
        self.start_stage_processors().await?;

        Ok(())
    }

    /// Submit a request and wait for the pipeline to finish it.
    ///
    /// The returned request is the one that actually traversed the stages,
    /// carrying each stage's real [`StageResult`] in
    /// [`PipelineRequest::stage_results`]. It is delivered through a completion
    /// channel the final stage sends on, so nothing is returned until the work
    /// is genuinely done.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::NoStageExecutor`] when no executor has been installed
    ///   with [`Self::with_stage_executor`] — the pipeline cannot compute
    ///   anything and says so instead of synthesizing a response.
    /// * [`PipelineError::ConfigurationError`] when the stages have not been
    ///   initialised, or the assigned stage does not exist.
    /// * [`PipelineError::StageTimeout`] when the configured
    ///   `stage_timeout_ms` budget elapses for every stage without the request
    ///   emerging.
    /// * [`PipelineError::ProcessingError`] when a stage fails, or the pipeline
    ///   shuts down while the request is in flight.
    pub async fn submit_request(
        &self,
        request: PipelineRequest,
    ) -> Result<PipelineRequest, PipelineError> {
        if self.stage_executor.is_none() {
            return Err(PipelineError::NoStageExecutor);
        }

        let request_id = request.id;
        let (completion_tx, completion_rx) = oneshot::channel();

        let stage_id = self.assign_request_to_stage(&request).await.map_err(|e| {
            PipelineError::ConfigurationError {
                message: e.to_string(),
            }
        })?;

        // Register the completion slot *before* the request can reach the final
        // stage, so a fast pipeline cannot complete into a slot that does not
        // exist yet.
        self.completions.write().await.insert(request_id, completion_tx);

        let tracker = RequestTracker {
            metadata: request.metadata.clone(),
            current_stage: Some(stage_id),
            stage_completion_times: HashMap::new(),
            start_time: Instant::now(),
        };
        self.request_tracker.write().await.insert(request_id, tracker);

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats.active_requests.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();

        // Hand the request to its first stage. Every early return from here on
        // must undo the bookkeeping above, or the pipeline reports work that is
        // not in flight.
        let send_outcome = {
            let stages = self.stages.read().await;
            match stages.get(stage_id) {
                Some(stage) => {
                    let sender = stage.read().await.input_tx.clone();
                    sender.send(request).await.map_err(|_| PipelineError::ProcessingError {
                        error: format!("stage {stage_id} is no longer accepting requests"),
                    })
                },
                None => Err(PipelineError::ConfigurationError {
                    message: format!(
                        "stage {stage_id} does not exist; call initialize_stages() first"
                    ),
                }),
            }
        };
        if let Err(e) = send_outcome {
            self.abandon_request(request_id).await;
            return Err(e);
        }

        // A request may legitimately visit every stage, so the whole-pipeline
        // budget is the per-stage budget times the number of stages.
        let budget = Duration::from_millis(self.config.stage_timeout_ms)
            .saturating_mul(self.config.num_stages.max(1) as u32);

        let outcome = match tokio::time::timeout(budget, completion_rx).await {
            Ok(Ok(completed)) => Ok(completed),
            Ok(Err(_)) => Err(PipelineError::ProcessingError {
                error: "the pipeline dropped the request before it completed".to_string(),
            }),
            Err(_) => Err(PipelineError::StageTimeout { stage_id }),
        };

        match outcome {
            Ok(completed) => {
                let elapsed = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
                self.stats.completed_requests.fetch_add(1, Ordering::Relaxed);
                self.stats.total_latency_us.fetch_add(elapsed, Ordering::Relaxed);
                self.stats.active_requests.fetch_sub(1, Ordering::Relaxed);
                self.request_tracker.write().await.remove(&request_id);
                self.completions.write().await.remove(&request_id);
                Ok(completed)
            },
            Err(e) => {
                self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                self.abandon_request(request_id).await;
                Err(e)
            },
        }
    }

    /// Drop the bookkeeping for a request that never entered, or never left,
    /// the pipeline.
    ///
    /// `active_requests` is decremented here rather than only on the success
    /// path, so a rejected or timed-out request cannot leave the gauge showing
    /// work that is not happening.
    async fn abandon_request(&self, request_id: PipelineRequestId) {
        self.completions.write().await.remove(&request_id);
        self.request_tracker.write().await.remove(&request_id);
        self.stats.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Assign request to optimal stage
    async fn assign_request_to_stage(&self, request: &PipelineRequest) -> Result<StageId> {
        let strategy = self.assignment_strategy.read().await;

        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index = strategy.rr_index.fetch_add(1, Ordering::Relaxed);
                Ok(index % self.config.num_stages)
            },
            LoadBalancingStrategy::LeastLoaded => self.find_least_loaded_stage().await,
            LoadBalancingStrategy::CapacityBased => self.find_capacity_based_stage(request).await,
            LoadBalancingStrategy::Adaptive => self.find_adaptive_stage(request).await,
        }
    }

    /// Find least loaded stage
    async fn find_least_loaded_stage(&self) -> Result<StageId> {
        let stages = self.stages.read().await;
        let mut min_load = usize::MAX;
        let mut best_stage = 0;

        for (i, stage) in stages.iter().enumerate() {
            let stage_guard = stage.read().await;
            let load = stage_guard.stats.active_requests.load(Ordering::Relaxed);
            if load < min_load {
                min_load = load;
                best_stage = i;
            }
        }

        Ok(best_stage)
    }

    /// Find stage based on capacity
    async fn find_capacity_based_stage(&self, _request: &PipelineRequest) -> Result<StageId> {
        // Simplified implementation - in practice, consider device capabilities,
        // memory requirements, etc.
        self.find_least_loaded_stage().await
    }

    /// Find stage using adaptive strategy
    async fn find_adaptive_stage(&self, _request: &PipelineRequest) -> Result<StageId> {
        // Simplified implementation - in practice, use ML-based prediction
        // based on request characteristics and historical performance
        self.find_least_loaded_stage().await
    }

    /// Start processing tasks for all stages
    async fn start_stage_processors(&self) -> Result<()> {
        let stages = self.stages.read().await;

        for (i, stage) in stages.iter().enumerate() {
            let stage_clone = stage.clone();
            let config = self.config.clone();
            // The *next* stage's sender is what this stage forwards to.
            let next_sender = match stages.get(i + 1) {
                Some(next) => Some(next.read().await.input_tx.clone()),
                None => None,
            };
            let executor = self.stage_executor.clone();
            let completions = Arc::clone(&self.completions);

            tokio::spawn(async move {
                Self::process_stage(stage_clone, next_sender, config, executor, completions).await;
            });
        }

        Ok(())
    }

    /// Drain one stage's queue, running the installed executor on each request.
    ///
    /// A request either moves on to `next_sender`, or — when this is the last
    /// stage, or the executor marked the result final — is delivered to the
    /// caller waiting on its completion slot. A request that fails, or that
    /// finishes with nowhere to go and no waiter, is dropped after being counted
    /// as an error; it is never silently reported as a success.
    async fn process_stage(
        stage: Arc<RwLock<PipelineStage>>,
        next_sender: Option<mpsc::Sender<PipelineRequest>>,
        _config: PipelineConfig,
        executor: Option<Arc<dyn StageExecutor>>,
        completions: Arc<RwLock<HashMap<PipelineRequestId, oneshot::Sender<PipelineRequest>>>>,
    ) {
        let (stage_id, receiver, semaphore, stats) = {
            let guard = stage.read().await;
            (
                guard.id,
                Arc::clone(&guard.input_rx),
                Arc::clone(&guard.concurrency_limit),
                Arc::clone(&guard.stats),
            )
        };

        loop {
            let mut request = {
                let mut rx = receiver.lock().await;
                match rx.recv().await {
                    Some(req) => req,
                    // Every sender is gone: the pipeline is shutting down.
                    None => break,
                }
            };

            // Permit held for RAII; on a closed semaphore (shutdown) proceed unlimited.
            let _permit = semaphore.acquire().await.ok();

            stats.record_request_start();
            let start_time = Instant::now();
            request.stage_times.insert(stage_id, start_time);

            let outcome = match &executor {
                Some(executor) => executor.execute(&request, stage_id).await,
                // Unreachable through `submit_request`, which refuses before a
                // request can be queued; stated explicitly rather than assumed.
                None => Err(PipelineError::NoStageExecutor),
            };

            let processing_time = start_time.elapsed();

            match outcome {
                Ok(mut stage_result) => {
                    // The stage's own latency claim is replaced by what the
                    // pipeline actually measured, so the exported figure cannot
                    // be an executor's guess.
                    stage_result.processing_time = processing_time;
                    stage_result.metrics.latency_us =
                        u64::try_from(processing_time.as_micros()).unwrap_or(u64::MAX);

                    let is_final = stage_result.is_final;
                    request.stage_results.insert(stage_id, stage_result);
                    stats.record_request_complete(processing_time);

                    match (is_final, &next_sender) {
                        // More pipeline to traverse.
                        (false, Some(next)) => {
                            if next.send(request).await.is_err() {
                                // The next stage is gone; the request cannot
                                // complete, and its waiter must not hang.
                                stats.record_error();
                                break;
                            }
                        },
                        // Finished: hand it back to whoever is waiting.
                        _ => {
                            let waiter = completions.write().await.remove(&request.id);
                            match waiter {
                                Some(slot) => {
                                    // A closed slot means the submitter gave up
                                    // (timed out); nothing further to do.
                                    let _ = slot.send(request);
                                },
                                None => {
                                    // Completed work nobody is waiting for: an
                                    // abandoned or duplicate request. Counted,
                                    // not reported as a success.
                                    stats.record_error();
                                },
                            }
                        },
                    }
                },
                Err(e) => {
                    tracing::warn!("pipeline stage {stage_id} failed a request: {e}");
                    stats.record_error();
                    // Drop the completion slot so the submitter fails fast
                    // instead of waiting out the full timeout budget.
                    completions.write().await.remove(&request.id);
                },
            }
        }
    }

    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStatsSummary {
        let stages = self.stages.read().await;
        let mut stage_stats = Vec::new();

        for stage in stages.iter() {
            let stage_guard = stage.read().await;
            stage_stats.push(StageStatsSummary {
                stage_id: stage_guard.id,
                name: stage_guard.name.clone(),
                requests_processed: stage_guard.stats.requests_processed.load(Ordering::Relaxed),
                active_requests: stage_guard.stats.active_requests.load(Ordering::Relaxed),
                avg_latency_us: stage_guard.stats.get_avg_latency_us(),
                error_count: stage_guard.stats.error_count.load(Ordering::Relaxed),
            });
        }

        PipelineStatsSummary {
            total_requests: self.stats.total_requests.load(Ordering::Relaxed),
            completed_requests: self.stats.completed_requests.load(Ordering::Relaxed),
            failed_requests: self.stats.failed_requests.load(Ordering::Relaxed),
            active_requests: self.stats.active_requests.load(Ordering::Relaxed),
            avg_pipeline_latency_us: {
                let total = self.stats.total_latency_us.load(Ordering::Relaxed);
                let completed = self.stats.completed_requests.load(Ordering::Relaxed);
                if completed > 0 {
                    total as f64 / completed as f64
                } else {
                    0.0
                }
            },
            throughput_rps: self.stats.throughput_rps.load(Ordering::Relaxed),
            stage_stats,
        }
    }

    /// Update pipeline configuration
    pub async fn update_config(&mut self, new_config: PipelineConfig) -> Result<()> {
        // Validate configuration changes
        if new_config.num_stages != self.config.num_stages {
            return Err(anyhow::anyhow!(
                "Cannot change number of stages in running pipeline"
            ));
        }

        self.config = new_config;
        Ok(())
    }
}

impl StageAssignmentStrategy {
    fn new() -> Self {
        Self {
            rr_index: AtomicUsize::new(0),
        }
    }
}

/// Pipeline statistics summary
#[derive(Debug, Serialize)]
pub struct PipelineStatsSummary {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub active_requests: usize,
    pub avg_pipeline_latency_us: f64,
    pub throughput_rps: u64,
    pub stage_stats: Vec<StageStatsSummary>,
}

/// Stage statistics summary
#[derive(Debug, Serialize)]
pub struct StageStatsSummary {
    pub stage_id: StageId,
    pub name: String,
    pub requests_processed: u64,
    pub active_requests: usize,
    pub avg_latency_us: f64,
    pub error_count: u64,
}

/// Pipeline parallelism error types
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Stage not found: {stage_id}")]
    StageNotFound { stage_id: StageId },

    #[error("Pipeline configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Stage timeout: {stage_id}")]
    StageTimeout { stage_id: StageId },

    #[error("Communication error: {message}")]
    CommunicationError { message: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Processing error: {error}")]
    ProcessingError { error: String },

    #[error(
        "no stage executor is installed: this pipeline cannot process requests. \
         Install one with PipelineParallelismManager::with_stage_executor()"
    )]
    NoStageExecutor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let manager =
            PipelineParallelismManager::new(config).expect("test operation should succeed");
        assert_eq!(manager.config.num_stages, 4);
    }

    #[tokio::test]
    async fn test_stage_assignment() {
        let config = PipelineConfig::default();
        let manager =
            PipelineParallelismManager::new(config).expect("test operation should succeed");

        let request = PipelineRequest {
            id: Uuid::new_v4(),
            data: vec![1, 2, 3],
            metadata: RequestMetadata {
                model_id: "test".to_string(),
                priority: RequestPriority::Normal,
                device_type: DeviceType::Any,
                complexity_score: 1.0,
                correlation_id: None,
            },
            stage_results: HashMap::new(),
            created_at: Instant::now(),
            stage_times: HashMap::new(),
        };

        let stage_id = manager
            .assign_request_to_stage(&request)
            .await
            .expect("async operation should succeed in test");
        assert!(stage_id < manager.config.num_stages);
    }

    #[test]
    fn test_stage_stats() {
        let stats = StageStats::default();

        stats.record_request_start();
        assert_eq!(stats.active_requests.load(Ordering::Relaxed), 1);

        stats.record_request_complete(Duration::from_millis(100));
        assert_eq!(stats.active_requests.load(Ordering::Relaxed), 0);
        assert_eq!(stats.requests_processed.load(Ordering::Relaxed), 1);
    }

    // ── Regression tests: the pipeline used to fabricate its own results ──

    /// An executor that performs real, verifiable work: it appends the stage id
    /// to the request payload, so the returned data proves which stages ran and
    /// in what order.
    struct AppendStageIdExecutor {
        /// Stage after which the pipeline is done.
        final_stage: StageId,
    }

    #[async_trait::async_trait]
    impl StageExecutor for AppendStageIdExecutor {
        async fn execute(
            &self,
            request: &PipelineRequest,
            stage_id: StageId,
        ) -> Result<StageResult, PipelineError> {
            let mut data = request
                .stage_results
                .get(&stage_id.wrapping_sub(1))
                .map(|previous| previous.data.clone())
                .unwrap_or_else(|| request.data.clone());
            data.push(u8::try_from(stage_id).unwrap_or(u8::MAX));

            Ok(StageResult {
                data,
                // Deliberately wrong: the pipeline must overwrite both of these
                // with what it actually measured.
                processing_time: Duration::ZERO,
                metrics: StageMetrics::default(),
                is_final: stage_id == self.final_stage,
            })
        }
    }

    /// An executor that always fails, to check the error path.
    struct FailingExecutor;

    #[async_trait::async_trait]
    impl StageExecutor for FailingExecutor {
        async fn execute(
            &self,
            _request: &PipelineRequest,
            stage_id: StageId,
        ) -> Result<StageResult, PipelineError> {
            Err(PipelineError::ProcessingError {
                error: format!("stage {stage_id} refused"),
            })
        }
    }

    fn test_config(num_stages: usize) -> PipelineConfig {
        PipelineConfig {
            num_stages,
            stage_timeout_ms: 2_000,
            ..PipelineConfig::default()
        }
    }

    fn stage_configs(count: usize) -> Vec<StageConfig> {
        (0..count)
            .map(|i| StageConfig {
                device: DeviceType::Any,
                layer_range: (i, i + 1),
                parameters: HashMap::new(),
                memory_limit_mb: None,
                timeout: Duration::from_secs(1),
            })
            .collect()
    }

    fn test_request(data: Vec<u8>) -> PipelineRequest {
        PipelineRequest {
            id: Uuid::new_v4(),
            data,
            metadata: RequestMetadata {
                model_id: "test-model".to_string(),
                priority: RequestPriority::Normal,
                device_type: DeviceType::Any,
                complexity_score: 1.0,
                correlation_id: None,
            },
            stage_results: HashMap::new(),
            created_at: Instant::now(),
            stage_times: HashMap::new(),
        }
    }

    /// Regression: `submit_request` slept 100 ms and returned a fabricated
    /// request — a fixed 100-byte zero payload with the model id `"dummy"` and
    /// no stage results — regardless of what was submitted or whether any stage
    /// had run. The response must now be the request that really traversed the
    /// pipeline.
    #[tokio::test]
    async fn submit_request_returns_the_request_that_actually_traversed_the_pipeline() {
        let manager = PipelineParallelismManager::new(test_config(2))
            .expect("manager builds")
            .with_stage_executor(Arc::new(AppendStageIdExecutor { final_stage: 1 }));
        manager.initialize_stages(stage_configs(2)).await.expect("stages initialise");

        let request = test_request(vec![7, 7, 7]);
        let submitted_id = request.id;
        let completed = manager.submit_request(request).await.expect("pipeline completes");

        assert_eq!(
            completed.id, submitted_id,
            "the answer must be for this request"
        );
        assert_eq!(
            completed.metadata.model_id, "test-model",
            "the old code returned the literal model id \"dummy\""
        );
        assert_ne!(
            completed.data,
            vec![0u8; 100],
            "the old code returned a fixed 100-byte zero payload"
        );
        assert!(
            !completed.stage_results.is_empty(),
            "the old code returned no stage results at all"
        );

        // The executor appends its stage id, so the final payload proves both
        // stages ran, in order, on the submitted bytes.
        let last_stage = completed.stage_results.keys().max().copied().expect("a stage ran");
        assert_eq!(
            completed.stage_results[&last_stage].data,
            vec![7, 7, 7, 0, 1]
        );
    }

    /// Regression: every stage recorded a hardcoded 10 ms latency, 1 MB of
    /// memory, 80% GPU utilisation and 100 operations into statistics that
    /// `get_stats()` exports. Latency must now be measured, and an executor that
    /// reports nothing must not have numbers invented for it.
    #[tokio::test]
    async fn stage_metrics_are_measured_rather_than_invented() {
        let manager = PipelineParallelismManager::new(test_config(1))
            .expect("manager builds")
            .with_stage_executor(Arc::new(AppendStageIdExecutor { final_stage: 0 }));
        manager.initialize_stages(stage_configs(1)).await.expect("stages initialise");

        let completed =
            manager.submit_request(test_request(vec![1])).await.expect("pipeline completes");
        let result = completed.stage_results.get(&0).expect("stage 0 ran");

        // The executor returned Duration::ZERO and a default StageMetrics; the
        // pipeline replaced the latency with its own measurement.
        assert!(
            result.processing_time > Duration::ZERO,
            "processing time must be measured, not taken from the executor's claim"
        );
        assert_ne!(
            result.metrics.latency_us, 10_000,
            "10_000 µs was the old hardcoded value"
        );
        assert_eq!(
            result.metrics.latency_us,
            u64::try_from(result.processing_time.as_micros()).unwrap_or(u64::MAX)
        );
        // Fields the executor genuinely did not measure stay at zero rather
        // than being filled with plausible-looking numbers.
        assert_eq!(result.metrics.gpu_utilization, None, "0.8 was invented");
        assert_eq!(result.metrics.operation_count, 0, "100 was invented");
        assert_eq!(result.metrics.memory_usage, 0, "1 MB was invented");

        let stats = manager.get_stats().await;
        assert_eq!(stats.completed_requests, 1);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(
            stats.active_requests, 0,
            "the in-flight gauge must return to zero"
        );
        assert!(stats.avg_pipeline_latency_us > 0.0);
    }

    /// A pipeline with no executor must refuse work rather than answer with a
    /// synthesized response.
    #[tokio::test]
    async fn a_pipeline_without_an_executor_refuses_requests() {
        let manager = PipelineParallelismManager::new(test_config(2)).expect("manager builds");
        assert!(!manager.has_stage_executor());
        manager.initialize_stages(stage_configs(2)).await.expect("stages initialise");

        let error = manager
            .submit_request(test_request(vec![1, 2, 3]))
            .await
            .expect_err("must not fabricate a response");
        assert!(matches!(error, PipelineError::NoStageExecutor));

        // A refused request must not be counted as in flight.
        assert_eq!(manager.get_stats().await.active_requests, 0);
    }

    /// A failing stage surfaces as an error promptly, and is counted as a
    /// failure rather than reported as a completion.
    #[tokio::test]
    async fn a_failing_stage_is_reported_as_a_failure() {
        let manager = PipelineParallelismManager::new(test_config(1))
            .expect("manager builds")
            .with_stage_executor(Arc::new(FailingExecutor));
        manager.initialize_stages(stage_configs(1)).await.expect("stages initialise");

        let error = manager
            .submit_request(test_request(vec![1]))
            .await
            .expect_err("a failing stage must not produce a completion");
        assert!(matches!(error, PipelineError::ProcessingError { .. }));

        let stats = manager.get_stats().await;
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.active_requests, 0);
        assert_eq!(stats.stage_stats[0].error_count, 1);
    }

    /// Submitting before `initialize_stages` must be an error, not a wait.
    #[tokio::test]
    async fn submitting_to_an_uninitialised_pipeline_is_a_configuration_error() {
        let manager = PipelineParallelismManager::new(test_config(2))
            .expect("manager builds")
            .with_stage_executor(Arc::new(AppendStageIdExecutor { final_stage: 1 }));

        let error = manager
            .submit_request(test_request(vec![1]))
            .await
            .expect_err("there is no stage to run on");
        assert!(matches!(error, PipelineError::ConfigurationError { .. }));
        assert_eq!(manager.get_stats().await.active_requests, 0);
    }
}
