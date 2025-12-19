//! Request Aggregator for Dynamic Batching

use crate::batching::{config::*, metrics::MetricsCollector};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

/// Unique request identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RequestId(pub Uuid);

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Inference request
#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub input: RequestInput,
    pub priority: Priority,
    pub submitted_at: Instant,
    pub deadline: Option<Instant>,
    pub metadata: HashMap<String, String>,
}

/// Request input data
#[derive(Debug, Clone)]
pub enum RequestInput {
    Text {
        text: String,
        max_length: Option<usize>,
    },
    TokenIds {
        ids: Vec<u32>,
        attention_mask: Option<Vec<u8>>,
    },
    Image {
        data: Vec<u8>,
        height: u32,
        width: u32,
    },
    Multimodal {
        text: Option<String>,
        image: Option<Vec<u8>>,
        audio: Option<Vec<f32>>,
    },
}

impl RequestInput {
    /// Get the sequence length for padding calculations
    pub fn sequence_length(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len() / 4, // Approximate tokens
            Self::TokenIds { ids, .. } => ids.len(),
            Self::Image { .. } => 1, // Treat as single sequence
            Self::Multimodal { text, .. } => text.as_ref().map(|t| t.len() / 4).unwrap_or(1),
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_estimate(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len() * 2, // Unicode consideration
            Self::TokenIds {
                ids,
                attention_mask,
            } => ids.len() * 4 + attention_mask.as_ref().map(|m| m.len()).unwrap_or(0),
            Self::Image { data, .. } => data.len(),
            Self::Multimodal { text, image, audio } => {
                text.as_ref().map(|t| t.len() * 2).unwrap_or(0)
                    + image.as_ref().map(|i| i.len()).unwrap_or(0)
                    + audio.as_ref().map(|a| a.len() * 4).unwrap_or(0)
            },
        }
    }
}

/// Batch of requests
#[derive(Debug)]
pub struct RequestBatch {
    pub id: Uuid,
    pub requests: Vec<Request>,
    pub created_at: Instant,
    pub total_memory: usize,
    pub max_sequence_length: usize,
    pub priority: Priority,
}

impl RequestBatch {
    fn new(requests: Vec<Request>) -> Self {
        let total_memory = requests.iter().map(|r| r.input.memory_estimate()).sum();

        let max_sequence_length =
            requests.iter().map(|r| r.input.sequence_length()).max().unwrap_or(0);

        let priority = requests.iter().map(|r| r.priority).max().unwrap_or(Priority::Normal);

        Self {
            id: Uuid::new_v4(),
            requests,
            created_at: Instant::now(),
            total_memory,
            max_sequence_length,
            priority,
        }
    }
}

/// Batching strategy implementation
pub trait BatchingStrategy: Send + Sync {
    /// Determine if a batch should be formed
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool;

    /// Select requests for the next batch
    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request>;
}

/// First-come, first-served strategy
pub struct FCFSStrategy;

impl BatchingStrategy for FCFSStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        if queue.is_empty() {
            return false;
        }

        // Check if we have enough requests
        if queue.len() >= config.max_batch_size {
            return true;
        }

        // Check timeout for the oldest request
        if let Some(oldest) = queue.front() {
            let wait_time = current_time.duration_since(oldest.submitted_at);
            if wait_time >= config.max_wait_time {
                return true;
            }
        }

        // Check if we meet minimum batch size
        queue.len() >= config.min_batch_size
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        let batch_size = queue.len().min(config.max_batch_size);
        queue.drain(..batch_size).collect()
    }
}

/// Priority-based strategy
pub struct PriorityStrategy;

impl BatchingStrategy for PriorityStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        FCFSStrategy.should_form_batch(queue, config, current_time)
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        // Sort by priority and select highest priority requests
        let mut all_requests: Vec<_> = queue.drain(..).collect();
        all_requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        let batch_size = all_requests.len().min(config.max_batch_size);
        let selected = all_requests.drain(..batch_size).collect();

        // Put remaining requests back
        queue.extend(all_requests);

        selected
    }
}

/// Continuous batching strategy for LLM serving
pub struct ContinuousBatchingStrategy {
    active_sequences: Arc<Mutex<HashMap<RequestId, SequenceState>>>,
}

impl Default for ContinuousBatchingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousBatchingStrategy {
    pub fn new() -> Self {
        Self {
            active_sequences: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add new sequence to active tracking
    pub async fn add_sequence(&self, request_id: RequestId, initial_tokens: Vec<u32>) {
        let state = SequenceState {
            request_id: request_id.clone(),
            generated_tokens: initial_tokens,
            is_finished: false,
            last_updated: Instant::now(),
        };

        self.active_sequences.lock().await.insert(request_id, state);
    }

    /// Update sequence with new token
    pub async fn update_sequence(&self, request_id: &RequestId, new_token: u32) -> bool {
        if let Some(state) = self.active_sequences.lock().await.get_mut(request_id) {
            state.generated_tokens.push(new_token);
            state.last_updated = Instant::now();
            true
        } else {
            false
        }
    }

    /// Mark sequence as finished
    pub async fn finish_sequence(&self, request_id: &RequestId) {
        if let Some(state) = self.active_sequences.lock().await.get_mut(request_id) {
            state.is_finished = true;
        }
    }

    /// Get active sequence count
    pub async fn active_count(&self) -> usize {
        self.active_sequences.lock().await.len()
    }
}

impl BatchingStrategy for ContinuousBatchingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        _config: &BatchingConfig,
        _current_time: Instant,
    ) -> bool {
        // For continuous batching, always process if there are requests
        !queue.is_empty()
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        // Select all available requests up to max batch size
        let batch_size = queue.len().min(config.max_batch_size);
        queue.drain(..batch_size).collect()
    }
}

/// Sequence packing strategy for better memory efficiency
pub struct SequencePackingStrategy;

impl BatchingStrategy for SequencePackingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        if queue.is_empty() {
            return false;
        }

        // Pack based on sequence length buckets
        let total_length: usize = queue.iter().map(|r| r.input.sequence_length()).sum();

        let avg_length = total_length / queue.len();
        let target_packed_length = config.max_batch_size * avg_length;

        total_length >= target_packed_length
            || queue.len() >= config.max_batch_size
            || queue
                .front()
                .map(|r| current_time.duration_since(r.submitted_at) >= config.max_wait_time)
                .unwrap_or(false)
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        if !config.dynamic_config.enable_bucketing {
            return queue.drain(..queue.len().min(config.max_batch_size)).collect();
        }

        // Group requests by sequence length buckets
        let mut buckets: HashMap<usize, Vec<Request>> = HashMap::new();
        let mut remaining_requests = Vec::new();

        while let Some(request) = queue.pop_front() {
            let length = request.input.sequence_length();
            let bucket = find_bucket(length, &config.dynamic_config.bucket_boundaries);

            if let Some(bucket_size) = bucket {
                buckets.entry(bucket_size).or_default().push(request);
            } else {
                remaining_requests.push(request);
            }
        }

        // Select from the bucket with most requests
        let mut selected = Vec::new();
        if let Some((_, mut bucket_requests)) =
            buckets.into_iter().max_by_key(|(_, requests)| requests.len())
        {
            let take_count = bucket_requests.len().min(config.max_batch_size);
            selected.extend(bucket_requests.drain(..take_count));

            // Put back remaining requests from this bucket
            remaining_requests.extend(bucket_requests);
        }

        // Put back all remaining requests
        for request in remaining_requests {
            queue.push_back(request);
        }

        selected
    }
}

/// Request coalescing strategy for similar requests
pub struct CoalescingStrategy {
    similarity_threshold: f32,
}

impl CoalescingStrategy {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
        }
    }

    /// Calculate similarity between two requests
    fn calculate_similarity(&self, req1: &Request, req2: &Request) -> f32 {
        match (&req1.input, &req2.input) {
            (RequestInput::Text { text: t1, .. }, RequestInput::Text { text: t2, .. }) => {
                self.text_similarity(t1, t2)
            },
            (RequestInput::TokenIds { ids: i1, .. }, RequestInput::TokenIds { ids: i2, .. }) => {
                self.token_similarity(i1, i2)
            },
            _ => 0.0, // Different input types
        }
    }

    fn text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity on words
        let words1: std::collections::HashSet<_> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn token_similarity(&self, tokens1: &[u32], tokens2: &[u32]) -> f32 {
        // Simple overlap percentage
        let min_len = tokens1.len().min(tokens2.len());
        if min_len == 0 {
            return 0.0;
        }

        let common =
            tokens1.iter().zip(tokens2.iter()).take(min_len).filter(|(a, b)| a == b).count();

        common as f32 / min_len as f32
    }
}

impl BatchingStrategy for CoalescingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        FCFSStrategy.should_form_batch(queue, config, current_time)
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        if queue.len() <= 1 {
            return queue.drain(..).collect();
        }

        let mut selected = Vec::new();
        let mut remaining = Vec::new();

        // Take first request as anchor
        if let Some(anchor) = queue.pop_front() {
            selected.push(anchor);

            // Find similar requests
            while let Some(request) = queue.pop_front() {
                let similarity = self.calculate_similarity(&selected[0], &request);

                if similarity >= self.similarity_threshold && selected.len() < config.max_batch_size
                {
                    selected.push(request);
                } else {
                    remaining.push(request);
                }
            }
        }

        // Put back non-selected requests
        for request in remaining {
            queue.push_back(request);
        }

        selected
    }
}

/// Sequence state for continuous batching
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub request_id: RequestId,
    pub generated_tokens: Vec<u32>,
    pub is_finished: bool,
    pub last_updated: Instant,
}

/// Find the appropriate bucket for a sequence length
fn find_bucket(length: usize, boundaries: &[usize]) -> Option<usize> {
    boundaries.iter().find(|&&boundary| length <= boundary).copied()
}

/// Batch aggregator
pub struct BatchAggregator {
    config: BatchingConfig,
    queue: Arc<Mutex<VecDeque<Request>>>,
    strategy: Box<dyn BatchingStrategy>,
    response_channels: Arc<Mutex<HashMap<RequestId, oneshot::Sender<ProcessingResult>>>>,
    metrics: Arc<MetricsCollector>,
    batch_tx: mpsc::Sender<RequestBatch>,
    batch_rx: Arc<Mutex<mpsc::Receiver<RequestBatch>>>,
}

impl BatchAggregator {
    pub fn new(config: BatchingConfig, metrics: Arc<MetricsCollector>) -> Self {
        let strategy: Box<dyn BatchingStrategy> = match config.mode {
            BatchingMode::Fixed => Box::new(FCFSStrategy),
            BatchingMode::Dynamic => {
                if config.dynamic_config.enable_bucketing {
                    Box::new(SequencePackingStrategy)
                } else if config.enable_priority_scheduling {
                    Box::new(PriorityStrategy)
                } else {
                    Box::new(FCFSStrategy)
                }
            },
            BatchingMode::Adaptive => {
                // Use coalescing strategy for adaptive mode
                Box::new(CoalescingStrategy::new(0.3)) // 30% similarity threshold
            },
            BatchingMode::Continuous => Box::new(ContinuousBatchingStrategy::new()),
        };

        let (batch_tx, batch_rx) = mpsc::channel(100);

        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            strategy,
            response_channels: Arc::new(Mutex::new(HashMap::new())),
            metrics,
            batch_tx,
            batch_rx: Arc::new(Mutex::new(batch_rx)),
        }
    }

    /// Add a request to the aggregator
    pub async fn add_request(
        &mut self,
        request: Request,
    ) -> Result<oneshot::Receiver<ProcessingResult>> {
        let (tx, rx) = oneshot::channel();

        // Store response channel
        self.response_channels.lock().await.insert(request.id.clone(), tx);

        // Add to queue
        self.queue.lock().await.push_back(request);

        // Try to form batch
        self.try_form_batch().await?;

        Ok(rx)
    }

    /// Try to form a batch if conditions are met
    async fn try_form_batch(&self) -> Result<()> {
        let mut queue = self.queue.lock().await;

        if self.strategy.should_form_batch(&queue, &self.config, Instant::now()) {
            let requests = self.strategy.select_requests(&mut queue, &self.config);

            if !requests.is_empty() {
                let batch = RequestBatch::new(requests);
                self.metrics.record_batch_formed(&batch);
                self.batch_tx.send(batch).await?;
            }
        }

        Ok(())
    }

    /// Get the next batch (called by scheduler)
    pub async fn get_next_batch(&self) -> Option<RequestBatch> {
        self.batch_rx.lock().await.recv().await
    }

    /// Process batch results
    pub async fn process_results(
        &self,
        _batch_id: Uuid,
        results: HashMap<RequestId, ProcessingResult>,
    ) -> Result<()> {
        let mut channels = self.response_channels.lock().await;

        for (request_id, result) in results {
            if let Some(tx) = channels.remove(&request_id) {
                let _ = tx.send(result);
            }
        }

        Ok(())
    }

    /// Update configuration
    pub async fn update_config(&mut self, config: BatchingConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Apply optimization suggestions
    pub async fn apply_optimizations(&mut self, suggestions: Vec<OptimizationSuggestion>) {
        for suggestion in suggestions {
            match suggestion {
                OptimizationSuggestion::IncreaseBatchSize(size) => {
                    self.config.max_batch_size = size;
                },
                OptimizationSuggestion::DecreaseBatchSize(size) => {
                    self.config.max_batch_size = size;
                },
                OptimizationSuggestion::AdjustTimeout(timeout) => {
                    self.config.max_wait_time = timeout;
                },
                OptimizationSuggestion::EnableBucketing => {
                    self.config.dynamic_config.enable_bucketing = true;
                },
            }
        }
    }

    /// Get aggregator statistics
    pub fn get_stats(&self) -> AggregatorStats {
        AggregatorStats {
            queue_depth: 0, // Would need async access to queue
            pending_requests: 0,
            total_batches_formed: 0,
            avg_batch_size: 0.0,
        }
    }
}

/// Processing result
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub request_id: RequestId,
    pub output: ProcessingOutput,
    pub latency_ms: u64,
    pub batch_id: Uuid,
}

/// Processing output
#[derive(Debug, Clone)]
pub enum ProcessingOutput {
    Text(String),
    Tokens(Vec<u32>),
    Embeddings(Vec<f32>),
    Classification(Vec<(String, f32)>),
    Error(String),
}

/// Optimization suggestions
#[derive(Debug, Clone)]
pub enum OptimizationSuggestion {
    IncreaseBatchSize(usize),
    DecreaseBatchSize(usize),
    AdjustTimeout(Duration),
    EnableBucketing,
}

/// Adaptive batching strategy that adjusts based on load and latency
pub struct AdaptiveBatchingStrategy {
    config: AdaptiveConfig,
    load_tracker: Arc<Mutex<LoadTracker>>,
    performance_history: Arc<Mutex<VecDeque<PerformanceMetric>>>,
    last_adjustment: Arc<Mutex<Instant>>,
}

impl AdaptiveBatchingStrategy {
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            config,
            load_tracker: Arc::new(Mutex::new(LoadTracker::new())),
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Record performance metric for adaptive decisions
    pub async fn record_performance(&self, latency: Duration, throughput: f32, batch_size: usize) {
        let metric = PerformanceMetric {
            timestamp: Instant::now(),
            latency,
            throughput,
            batch_size,
        };

        let mut history = self.performance_history.lock().await;
        history.push_back(metric);

        // Keep only recent history
        let cutoff = Instant::now() - self.config.prediction_window;
        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate optimal batch size based on current conditions
    async fn calculate_optimal_batch_size(&self, config: &BatchingConfig) -> usize {
        let load = self.load_tracker.lock().await.current_load();
        let mut optimal_size = config.max_batch_size;

        // Adjust based on load
        if load < self.config.low_load_threshold {
            // Low load: prefer latency, use smaller batches
            optimal_size = (config.max_batch_size as f32 * 0.5) as usize;
        } else if load > self.config.high_load_threshold {
            // High load: prefer throughput, use larger batches
            optimal_size = self.config.high_load_batch_size.min(config.max_batch_size);
        }

        // Adjust based on recent performance if prediction is enabled
        if self.config.enable_prediction {
            if let Some(predicted_size) = self.predict_optimal_size().await {
                // Weighted average of load-based and prediction-based size
                optimal_size =
                    ((optimal_size as f32 * 0.7) + (predicted_size as f32 * 0.3)) as usize;
            }
        }

        optimal_size.clamp(config.min_batch_size, config.max_batch_size)
    }

    /// Predict optimal batch size based on historical performance
    async fn predict_optimal_size(&self) -> Option<usize> {
        let history = self.performance_history.lock().await;
        if history.len() < 3 {
            return None;
        }

        // Find batch size with best throughput/latency ratio
        let mut best_ratio = 0.0;
        let mut best_size = 0;

        for metric in history.iter() {
            let ratio = metric.throughput / metric.latency.as_secs_f32();
            if ratio > best_ratio {
                best_ratio = ratio;
                best_size = metric.batch_size;
            }
        }

        Some(best_size)
    }
}

impl BatchingStrategy for AdaptiveBatchingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        if queue.is_empty() {
            return false;
        }

        // Check if we should wait for more requests based on load
        let load = self
            .load_tracker
            .try_lock()
            .map(|tracker| tracker.current_load())
            .unwrap_or(0.0);

        if load < self.config.low_load_threshold && queue.len() < config.min_batch_size {
            // Low load: wait a bit longer for better batching
            let oldest_request_age =
                current_time.duration_since(queue.front().unwrap().submitted_at);
            oldest_request_age > self.config.low_load_timeout
        } else {
            // Normal/high load: form batch more aggressively
            queue.len() >= config.min_batch_size
                || current_time.duration_since(queue.front().unwrap().submitted_at)
                    > config.max_wait_time
        }
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        // Calculate adaptive batch size
        let optimal_size = {
            // Use a simple heuristic if we can't get the optimal size
            let load = self
                .load_tracker
                .try_lock()
                .map(|tracker| tracker.current_load())
                .unwrap_or(0.0);

            if load < self.config.low_load_threshold {
                (config.max_batch_size as f32 * 0.5) as usize
            } else if load > self.config.high_load_threshold {
                self.config.high_load_batch_size.min(config.max_batch_size)
            } else {
                config.max_batch_size
            }
        };

        let batch_size = queue.len().clamp(config.min_batch_size, optimal_size);

        // Select requests with priority consideration
        let mut all_requests: Vec<_> = queue.drain(..).collect();
        all_requests.sort_by(|a, b| {
            // Sort by priority first, then by submission time
            match b.priority.cmp(&a.priority) {
                std::cmp::Ordering::Equal => a.submitted_at.cmp(&b.submitted_at),
                other => other,
            }
        });

        let selected = all_requests.drain(..batch_size.min(all_requests.len())).collect();

        // Put remaining requests back
        queue.extend(all_requests);

        selected
    }
}

/// Load-aware batching strategy
pub struct LoadAwareBatchingStrategy {
    load_tracker: Arc<Mutex<LoadTracker>>,
    base_config: BatchingConfig,
}

impl LoadAwareBatchingStrategy {
    pub fn new(base_config: BatchingConfig) -> Self {
        Self {
            load_tracker: Arc::new(Mutex::new(LoadTracker::new())),
            base_config,
        }
    }

    pub async fn update_load(&self, requests_per_second: f32) {
        self.load_tracker.lock().await.update(requests_per_second);
    }
}

impl BatchingStrategy for LoadAwareBatchingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        if queue.is_empty() {
            return false;
        }

        let load = self
            .load_tracker
            .try_lock()
            .map(|tracker| tracker.current_load())
            .unwrap_or(0.0);

        // Adjust timing based on load
        let timeout = if load > 50.0 {
            config.max_wait_time / 2 // High load: reduce wait time
        } else if load < 10.0 {
            config.max_wait_time * 2 // Low load: wait longer for better batching
        } else {
            config.max_wait_time
        };

        queue.len() >= config.min_batch_size
            || current_time.duration_since(queue.front().unwrap().submitted_at) > timeout
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        let load = self
            .load_tracker
            .try_lock()
            .map(|tracker| tracker.current_load())
            .unwrap_or(0.0);

        // Adjust batch size based on load
        let batch_size = if load > 50.0 {
            config.max_batch_size // High load: maximize throughput
        } else if load < 10.0 {
            (config.max_batch_size / 2).max(config.min_batch_size) // Low load: optimize latency
        } else {
            (config.max_batch_size * 3 / 4).max(config.min_batch_size) // Medium load: balanced
        };

        let actual_size = queue.len().min(batch_size);
        queue.drain(..actual_size).collect()
    }
}

/// Predictive batching strategy using machine learning concepts
pub struct PredictiveBatchingStrategy {
    predictor: Arc<Mutex<SimplePredictor>>,
    performance_history: Arc<Mutex<VecDeque<PerformanceMetric>>>,
}

impl Default for PredictiveBatchingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictiveBatchingStrategy {
    pub fn new() -> Self {
        Self {
            predictor: Arc::new(Mutex::new(SimplePredictor::new())),
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub async fn train_predictor(&self, metrics: Vec<PerformanceMetric>) {
        self.predictor.lock().await.train(metrics);
    }
}

impl BatchingStrategy for PredictiveBatchingStrategy {
    fn should_form_batch(
        &self,
        queue: &VecDeque<Request>,
        config: &BatchingConfig,
        current_time: Instant,
    ) -> bool {
        if queue.is_empty() {
            return false;
        }

        // Use predictor to determine if now is a good time to batch
        let predicted_latency = self
            .predictor
            .try_lock()
            .ok()
            .and_then(|p| p.predict_latency(queue.len()))
            .unwrap_or(50.0); // Default fallback

        // Form batch if predicted latency is acceptable or timeout exceeded
        let should_wait = predicted_latency < 100.0 && queue.len() < config.max_batch_size;
        let timeout_exceeded =
            current_time.duration_since(queue.front().unwrap().submitted_at) > config.max_wait_time;

        !should_wait || timeout_exceeded
    }

    fn select_requests(
        &self,
        queue: &mut VecDeque<Request>,
        config: &BatchingConfig,
    ) -> Vec<Request> {
        // Use predictor to find optimal batch size
        let optimal_size = self
            .predictor
            .try_lock()
            .ok()
            .and_then(|p| p.predict_optimal_batch_size())
            .unwrap_or(config.max_batch_size);

        let batch_size = queue.len().clamp(config.min_batch_size, optimal_size);
        queue.drain(..batch_size).collect()
    }
}

/// Load tracker for adaptive strategies
#[derive(Debug)]
struct LoadTracker {
    recent_loads: VecDeque<(Instant, f32)>,
    current_load: f32,
}

impl LoadTracker {
    fn new() -> Self {
        Self {
            recent_loads: VecDeque::new(),
            current_load: 0.0,
        }
    }

    fn update(&mut self, load: f32) {
        let now = Instant::now();
        self.recent_loads.push_back((now, load));

        // Remove old entries (keep last 60 seconds)
        let cutoff = now - Duration::from_secs(60);
        while let Some((time, _)) = self.recent_loads.front() {
            if *time < cutoff {
                self.recent_loads.pop_front();
            } else {
                break;
            }
        }

        // Calculate average load
        if !self.recent_loads.is_empty() {
            self.current_load = self.recent_loads.iter().map(|(_, load)| load).sum::<f32>()
                / self.recent_loads.len() as f32;
        }
    }

    fn current_load(&self) -> f32 {
        self.current_load
    }
}

/// Performance metric for adaptive learning
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub timestamp: Instant,
    pub latency: Duration,
    pub throughput: f32,
    pub batch_size: usize,
}

/// Simple predictor for batch optimization
#[derive(Debug)]
struct SimplePredictor {
    latency_samples: Vec<(usize, f32)>, // (batch_size, latency_ms)
    optimal_size: usize,
}

impl SimplePredictor {
    fn new() -> Self {
        Self {
            latency_samples: Vec::new(),
            optimal_size: 16, // Default
        }
    }

    fn train(&mut self, metrics: Vec<PerformanceMetric>) {
        self.latency_samples.clear();

        for metric in &metrics {
            self.latency_samples
                .push((metric.batch_size, metric.latency.as_millis() as f32));
        }

        // Find batch size with best throughput/latency ratio
        if let Some(best) = metrics.iter().max_by(|a, b| {
            let ratio_a = a.throughput / a.latency.as_secs_f32();
            let ratio_b = b.throughput / b.latency.as_secs_f32();
            ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.optimal_size = best.batch_size;
        }
    }

    fn predict_latency(&self, batch_size: usize) -> Option<f32> {
        if self.latency_samples.is_empty() {
            return None;
        }

        // Simple linear interpolation
        let closest = self
            .latency_samples
            .iter()
            .min_by_key(|(size, _)| (*size as i32 - batch_size as i32).abs())?;

        Some(closest.1)
    }

    fn predict_optimal_batch_size(&self) -> Option<usize> {
        if self.optimal_size > 0 {
            Some(self.optimal_size)
        } else {
            None
        }
    }
}

/// Aggregator statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct AggregatorStats {
    pub queue_depth: usize,
    pub pending_requests: usize,
    pub total_batches_formed: usize,
    pub avg_batch_size: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_memory_estimate() {
        let text_input = RequestInput::Text {
            text: "Hello world".to_string(),
            max_length: None,
        };
        assert!(text_input.memory_estimate() > 0);

        let token_input = RequestInput::TokenIds {
            ids: vec![1, 2, 3, 4, 5],
            attention_mask: Some(vec![1, 1, 1, 1, 1]),
        };
        assert_eq!(token_input.memory_estimate(), 25); // 5*4 + 5*1
    }

    #[test]
    fn test_batch_creation() {
        let requests = vec![Request {
            id: RequestId::new(),
            input: RequestInput::Text {
                text: "Test".to_string(),
                max_length: None,
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
        }];

        let batch = RequestBatch::new(requests);
        assert_eq!(batch.requests.len(), 1);
        assert!(batch.total_memory > 0);
    }
}
