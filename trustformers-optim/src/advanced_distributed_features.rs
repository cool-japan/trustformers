//! # Advanced Distributed Training Features
//!
//! This module provides cutting-edge features for distributed training that extend
//! the enhanced distributed training framework with:
//!
//! - **Auto-Scaling**: Dynamic GPU allocation based on workload and performance
//! - **Advanced Fault Recovery**: Sophisticated checkpoint management and node recovery
//! - **Performance Optimization**: ML-based performance tuning and resource optimization
//! - **Elastic Training**: Dynamic worker scaling during training
//! - **Communication Optimization**: Advanced topology-aware communication patterns
//! - **Memory Management**: Advanced memory pressure detection and optimization
//!
//! ## Key Features
//!
//! 1. **Elastic Scaling**: Automatically add/remove nodes based on workload
//! 2. **Smart Checkpointing**: Differential checkpoints with automatic validation
//! 3. **Performance ML**: Machine learning models for performance prediction and optimization
//! 4. **Network Topology Optimization**: Automatic topology discovery and optimization
//! 5. **Memory Pressure Management**: Predictive memory management with preemptive optimization
//! 6. **Load Balancing**: Sophisticated load balancing with performance modeling
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{
//!     EnhancedDistributedTrainer,
//!     AutoScaler, AutoScalerConfig, ScalingStrategy,
//!     PerformanceMLOptimizer, MLOptimizerConfig,
//! };
//! # use trustformers_optim::{AveragedAdam, DistributedConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create auto-scaling configuration
//! let auto_scaler = AutoScaler::new(AutoScalerConfig::default())
//!     .with_min_nodes(2)
//!     .with_max_nodes(64)
//!     .with_scaling_strategy(ScalingStrategy::Performance)
//!     .with_scale_up_threshold(0.85)
//!     .with_scale_down_threshold(0.6);
//!
//! // Enable ML-based performance optimization
//! let ml_optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default())
//!     .with_prediction_horizon(100)
//!     .with_optimization_frequency(50);
//!
//! // Advanced distributed trainer; auto-scaling and ML optimization are applied
//! // to it independently via `update_and_scale` / `optimize_performance`
//! # let config = DistributedConfig::new();
//! # let optimizer = AveragedAdam::for_distributed_training();
//! let trainer = EnhancedDistributedTrainer::new(config, optimizer)?;
//! # let _ = (auto_scaler, ml_optimizer, trainer);
//! # Ok(())
//! # }
//! ```

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

pub mod checkpoint_format;

use crate::enhanced_distributed_training::{DistributedConfig, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Auto-scaling configuration for dynamic node management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalerConfig {
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Scaling strategy
    pub strategy: ScalingStrategy,
    /// Threshold for scaling up (GPU utilization %)
    pub scale_up_threshold: f32,
    /// Threshold for scaling down (GPU utilization %)
    pub scale_down_threshold: f32,
    /// Cooldown period between scaling operations
    pub scaling_cooldown: Duration,
    /// Enable predictive scaling
    pub predictive_scaling: bool,
    /// Cost optimization priority (0.0 = performance, 1.0 = cost)
    pub cost_priority: f32,
}

impl Default for AutoScalerConfig {
    fn default() -> Self {
        Self {
            min_nodes: 1,
            max_nodes: 16,
            strategy: ScalingStrategy::Performance,
            scale_up_threshold: 0.85,
            scale_down_threshold: 0.6,
            scaling_cooldown: Duration::from_secs(300), // 5 minutes
            predictive_scaling: true,
            cost_priority: 0.3, // Slightly favor performance
        }
    }
}

/// Scaling strategies for auto-scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// Scale based on performance metrics
    Performance,
    /// Scale based on queue length
    QueueBased,
    /// Scale based on predicted workload
    Predictive,
    /// Scale based on cost-performance optimization
    CostOptimized,
    /// Custom scaling strategy
    Custom(String),
}

/// Auto-scaler for dynamic node management
pub struct AutoScaler {
    config: AutoScalerConfig,
    current_nodes: usize,
    last_scaling_action: Instant,
    performance_history: VecDeque<PerformanceMetrics>,
    scaling_history: Vec<ScalingEvent>,
    workload_predictor: WorkloadPredictor,
    cost_optimizer: CostOptimizer,
}

impl AutoScaler {
    pub fn new(config: AutoScalerConfig) -> Self {
        Self {
            current_nodes: config.min_nodes,
            config,
            last_scaling_action: Instant::now(),
            performance_history: VecDeque::with_capacity(1000),
            scaling_history: Vec::new(),
            workload_predictor: WorkloadPredictor::new(),
            cost_optimizer: CostOptimizer::new(),
        }
    }

    /// Builder pattern for configuration
    pub fn with_min_nodes(mut self, min_nodes: usize) -> Self {
        self.config.min_nodes = min_nodes;
        // Also update current_nodes if it's below the new minimum
        if self.current_nodes < min_nodes {
            self.current_nodes = min_nodes;
        }
        self
    }

    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.config.max_nodes = max_nodes;
        self
    }

    pub fn with_scaling_strategy(mut self, strategy: ScalingStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn with_scale_up_threshold(mut self, threshold: f32) -> Self {
        self.config.scale_up_threshold = threshold;
        self
    }

    pub fn with_scale_down_threshold(mut self, threshold: f32) -> Self {
        self.config.scale_down_threshold = threshold;
        self
    }

    /// Update performance metrics and decide on scaling
    pub fn update_and_scale(&mut self, metrics: &PerformanceMetrics) -> Result<ScalingDecision> {
        // Add metrics to history
        self.performance_history.push_back(metrics.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Check cooldown period
        if self.last_scaling_action.elapsed() < self.config.scaling_cooldown {
            return Ok(ScalingDecision::NoAction);
        }

        // Analyze current performance
        let avg_utilization =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;
        let _avg_memory =
            metrics.memory_usage.iter().sum::<f32>() / metrics.memory_usage.len() as f32;

        // Make scaling decision based on strategy
        let decision = match &self.config.strategy {
            ScalingStrategy::Performance => self.performance_based_scaling(avg_utilization)?,
            ScalingStrategy::QueueBased => self.queue_based_scaling(metrics)?,
            ScalingStrategy::Predictive => self.predictive_scaling(metrics)?,
            ScalingStrategy::CostOptimized => {
                self.cost_optimized_scaling(avg_utilization, metrics)?
            },
            ScalingStrategy::Custom(name) => self.custom_scaling(name, metrics)?,
        };

        // Execute scaling decision
        match &decision {
            ScalingDecision::ScaleUp(nodes) => {
                self.execute_scale_up(*nodes)?;
            },
            ScalingDecision::ScaleDown(nodes) => {
                self.execute_scale_down(*nodes)?;
            },
            ScalingDecision::NoAction => {},
        }

        Ok(decision)
    }

    fn performance_based_scaling(&self, avg_utilization: f32) -> Result<ScalingDecision> {
        if avg_utilization > self.config.scale_up_threshold
            && self.current_nodes < self.config.max_nodes
        {
            // Calculate number of nodes to add based on utilization
            let target_utilization = 0.75; // Target 75% utilization
            let utilization_ratio = avg_utilization / target_utilization;
            let nodes_to_add =
                ((utilization_ratio - 1.0) * self.current_nodes as f32).ceil() as usize;
            let nodes_to_add = nodes_to_add.min(self.config.max_nodes - self.current_nodes);

            Ok(ScalingDecision::ScaleUp(nodes_to_add))
        } else if avg_utilization < self.config.scale_down_threshold
            && self.current_nodes > self.config.min_nodes
        {
            // Calculate number of nodes to remove
            let target_utilization = 0.8; // Target 80% utilization when scaling down
            let required_nodes =
                (avg_utilization * self.current_nodes as f32 / target_utilization).ceil() as usize;
            let nodes_to_remove = self.current_nodes.saturating_sub(required_nodes);
            let nodes_to_remove = nodes_to_remove.min(self.current_nodes - self.config.min_nodes);

            if nodes_to_remove > 0 {
                Ok(ScalingDecision::ScaleDown(nodes_to_remove))
            } else {
                Ok(ScalingDecision::NoAction)
            }
        } else {
            Ok(ScalingDecision::NoAction)
        }
    }

    fn queue_based_scaling(&self, metrics: &PerformanceMetrics) -> Result<ScalingDecision> {
        // Simplified queue-based scaling (would integrate with actual queue metrics)
        let throughput_ratio = metrics.throughput / 1000.0; // Assume baseline 1000 samples/sec

        if throughput_ratio < 0.5 && self.current_nodes < self.config.max_nodes {
            Ok(ScalingDecision::ScaleUp(1))
        } else if throughput_ratio > 2.0 && self.current_nodes > self.config.min_nodes {
            Ok(ScalingDecision::ScaleDown(1))
        } else {
            Ok(ScalingDecision::NoAction)
        }
    }

    fn predictive_scaling(&mut self, metrics: &PerformanceMetrics) -> Result<ScalingDecision> {
        if !self.config.predictive_scaling {
            return self.performance_based_scaling(
                metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32,
            );
        }

        // Update workload predictor
        self.workload_predictor.update_metrics(metrics);

        // Without enough history there is no prediction to make. Falling back
        // to the *measured* utilization keeps the decision grounded in real
        // data; inventing a "conservative 0.75" would fabricate the input the
        // whole branch is about to act on.
        if !self.workload_predictor.can_predict() {
            return self.performance_based_scaling(
                metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32,
            );
        }

        // Get prediction for next 10 minutes
        let predicted_load = self.workload_predictor.predict_workload(Duration::from_secs(600))?;

        // Utilization the cluster is sized for. Both branches solve the same
        // equation — `nodes * target = load * current_nodes` — so the sizing is
        // derived from the configuration rather than from magic constants.
        let target = self.config.scale_up_threshold.clamp(0.05, 1.0);

        // Make scaling decision based on prediction
        if predicted_load > self.config.scale_up_threshold * 1.1 && // Add 10% buffer
           self.current_nodes < self.config.max_nodes
        {
            let required = (predicted_load / target * self.current_nodes as f32).ceil() as usize;
            let nodes_to_add = required.saturating_sub(self.current_nodes).max(1);
            Ok(ScalingDecision::ScaleUp(
                nodes_to_add.min(self.config.max_nodes - self.current_nodes),
            ))
        } else if predicted_load < self.config.scale_down_threshold * 0.9 && // Add 10% buffer
                  self.current_nodes > self.config.min_nodes
        {
            let target_nodes =
                ((predicted_load / target * self.current_nodes as f32).ceil() as usize).max(1);
            let nodes_to_remove = self.current_nodes.saturating_sub(target_nodes);
            if nodes_to_remove > 0 {
                Ok(ScalingDecision::ScaleDown(
                    nodes_to_remove.min(self.current_nodes - self.config.min_nodes),
                ))
            } else {
                Ok(ScalingDecision::NoAction)
            }
        } else {
            Ok(ScalingDecision::NoAction)
        }
    }

    fn cost_optimized_scaling(
        &mut self,
        avg_utilization: f32,
        metrics: &PerformanceMetrics,
    ) -> Result<ScalingDecision> {
        // Calculate cost-performance ratio
        let current_cost = self.cost_optimizer.calculate_current_cost(self.current_nodes, metrics);

        // Evaluate scale up cost-benefit
        if avg_utilization > self.config.scale_up_threshold
            && self.current_nodes < self.config.max_nodes
        {
            let scale_up_cost =
                self.cost_optimizer.calculate_scale_up_cost(self.current_nodes + 1, metrics);
            let cost_benefit_ratio = current_cost / scale_up_cost;

            if cost_benefit_ratio > (1.0 - self.config.cost_priority) {
                Ok(ScalingDecision::ScaleUp(1))
            } else {
                Ok(ScalingDecision::NoAction)
            }
        } else if avg_utilization < self.config.scale_down_threshold
            && self.current_nodes > self.config.min_nodes
        {
            let scale_down_cost =
                self.cost_optimizer.calculate_scale_down_cost(self.current_nodes - 1, metrics);
            let cost_savings = current_cost - scale_down_cost;

            if cost_savings > current_cost * 0.1 {
                // At least 10% savings
                Ok(ScalingDecision::ScaleDown(1))
            } else {
                Ok(ScalingDecision::NoAction)
            }
        } else {
            Ok(ScalingDecision::NoAction)
        }
    }

    /// Dispatch a caller-named scaling strategy.
    ///
    /// # Errors
    ///
    /// Always. [`ScalingStrategy::Custom`] names a policy this crate does not
    /// implement and has no callback for; answering
    /// [`ScalingDecision::NoAction`] would be indistinguishable from a policy
    /// that ran and decided to do nothing.
    fn custom_scaling(&self, name: &str, _metrics: &PerformanceMetrics) -> Result<ScalingDecision> {
        Err(TrustformersError::not_implemented(format!(
            "custom scaling strategy `{name}` has no implementation registered; select one of \
             ScalingStrategy::{{Performance, QueueBased, Predictive, CostOptimized}} or drive the \
             scaling decision yourself"
        )))
    }

    fn execute_scale_up(&mut self, nodes: usize) -> Result<()> {
        log::info!(
            "scaling up: adding {} nodes (current: {})",
            nodes,
            self.current_nodes
        );

        self.current_nodes += nodes;
        self.last_scaling_action = Instant::now();

        self.scaling_history.push(ScalingEvent {
            timestamp: SystemTime::now(),
            action: ScalingAction::ScaleUp,
            nodes_changed: nodes,
            reason: "Performance threshold exceeded".to_string(),
        });

        // In a real implementation, this would:
        // 1. Request new nodes from cloud provider
        // 2. Initialize nodes with training environment
        // 3. Add nodes to communication topology
        // 4. Redistribute workload

        Ok(())
    }

    fn execute_scale_down(&mut self, nodes: usize) -> Result<()> {
        log::info!(
            "scaling down: removing {} nodes (current: {})",
            nodes,
            self.current_nodes
        );

        self.current_nodes -= nodes;
        self.last_scaling_action = Instant::now();

        self.scaling_history.push(ScalingEvent {
            timestamp: SystemTime::now(),
            action: ScalingAction::ScaleDown,
            nodes_changed: nodes,
            reason: "Low utilization detected".to_string(),
        });

        // In a real implementation, this would:
        // 1. Gracefully remove nodes from training
        // 2. Migrate workload to remaining nodes
        // 3. Update communication topology
        // 4. Terminate removed nodes

        Ok(())
    }

    pub fn get_current_nodes(&self) -> usize {
        self.current_nodes
    }

    pub fn get_scaling_history(&self) -> &[ScalingEvent] {
        &self.scaling_history
    }
}

/// Scaling decision types
#[derive(Debug, Clone)]
pub enum ScalingDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    NoAction,
}

/// Scaling event for tracking scaling history
#[derive(Debug, Clone)]
pub struct ScalingEvent {
    pub timestamp: SystemTime,
    pub action: ScalingAction,
    pub nodes_changed: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
}

/// Workload predictor using simple ML models
pub struct WorkloadPredictor {
    historical_data: VecDeque<(Instant, f32)>, // (timestamp, utilization)
    trend_analyzer: TrendAnalyzer,
    seasonal_analyzer: SeasonalAnalyzer,
}

impl Default for WorkloadPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadPredictor {
    pub fn new() -> Self {
        Self {
            historical_data: VecDeque::with_capacity(10000),
            trend_analyzer: TrendAnalyzer::new(),
            seasonal_analyzer: SeasonalAnalyzer::new(),
        }
    }

    /// Samples required before a prediction is meaningful.
    pub const MIN_SAMPLES: usize = 10;

    pub fn update_metrics(&mut self, metrics: &PerformanceMetrics) {
        if metrics.gpu_utilization.is_empty() {
            // No telemetry was recorded; there is nothing to learn from.
            return;
        }
        let avg_utilization =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;

        self.historical_data.push_back((Instant::now(), avg_utilization));
        if self.historical_data.len() > 10000 {
            self.historical_data.pop_front();
        }

        self.trend_analyzer.update(avg_utilization);
        self.seasonal_analyzer.update(SystemTime::now(), avg_utilization);
    }

    /// Number of utilization samples observed so far.
    pub fn sample_count(&self) -> usize {
        self.historical_data.len()
    }

    /// Whether enough history has accumulated for
    /// [`WorkloadPredictor::predict_workload`] to answer.
    pub fn can_predict(&self) -> bool {
        self.historical_data.len() >= Self::MIN_SAMPLES
    }

    /// Predict mean GPU utilization `horizon` from now.
    ///
    /// # Errors
    ///
    /// When fewer than [`WorkloadPredictor::MIN_SAMPLES`] samples have been
    /// recorded. Earlier revisions returned a hard-coded `0.75` here, which the
    /// auto-scaler then acted on as though it were a measurement.
    pub fn predict_workload(&self, horizon: Duration) -> Result<f32> {
        if !self.can_predict() {
            return Err(TrustformersError::invalid_state(format!(
                "workload prediction needs at least {} utilization samples, have {}",
                Self::MIN_SAMPLES,
                self.historical_data.len()
            )));
        }

        // Simple prediction combining trend and seasonal components
        let trend_prediction = self.trend_analyzer.predict(horizon)?;
        let seasonal_prediction = self.seasonal_analyzer.predict(horizon)?;

        // Weighted combination
        let prediction = trend_prediction * 0.7 + seasonal_prediction * 0.3;

        // Clamp to reasonable bounds
        Ok(prediction.clamp(0.0, 1.0))
    }
}

/// Simple trend analyzer
pub struct TrendAnalyzer {
    values: VecDeque<f32>,
    window_size: usize,
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalyzer {
    pub fn new() -> Self {
        Self {
            values: VecDeque::with_capacity(100),
            window_size: 50,
        }
    }

    pub fn update(&mut self, value: f32) {
        self.values.push_back(value);
        if self.values.len() > self.window_size {
            self.values.pop_front();
        }
    }

    pub fn predict(&self, _horizon: Duration) -> Result<f32> {
        if self.values.len() < 10 {
            return Ok(0.75); // Default
        }

        // Simple linear trend calculation
        let values: Vec<f32> = self.values.iter().cloned().collect();
        let n = values.len() as f32;

        let x_sum = (0..values.len()).sum::<usize>() as f32;
        let y_sum = values.iter().sum::<f32>();
        let xy_sum = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum::<f32>();
        let x2_sum = (0..values.len()).map(|i| (i * i) as f32).sum::<f32>();

        // Linear regression slope
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
        let intercept = (y_sum - slope * x_sum) / n;

        // Predict for next point
        let next_x = values.len() as f32;
        let prediction = slope * next_x + intercept;

        Ok(prediction)
    }
}

/// Utilization aggregated by UTC hour-of-day.
///
/// Samples are bucketed by the wall-clock hour they were observed at, so a
/// prediction for a future time reads the bucket that time falls in. An earlier
/// revision derived the bucket from `Instant::elapsed()`, which measures the
/// *age* of the sample and therefore put every observation in bucket 0.
pub struct SeasonalAnalyzer {
    hourly_patterns: HashMap<u32, Vec<f32>>, // UTC hour-of-day -> values
    last_update: Option<SystemTime>,
}

impl Default for SeasonalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SeasonalAnalyzer {
    pub fn new() -> Self {
        Self {
            hourly_patterns: HashMap::new(),
            last_update: None,
        }
    }

    /// UTC hour-of-day (`0..24`) that `at` falls in.
    ///
    /// Times before the Unix epoch are clamped to hour 0; this crate carries no
    /// calendar dependency, and hour-of-day needs none.
    pub fn hour_of_day(at: SystemTime) -> u32 {
        let seconds = at
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        ((seconds / 3600) % 24) as u32
    }

    /// Record `value` in the bucket for the UTC hour `at` falls in.
    pub fn update(&mut self, at: SystemTime, value: f32) {
        let hour = Self::hour_of_day(at);
        let bucket = self.hourly_patterns.entry(hour).or_default();
        bucket.push(value);
        // Bounded history: keep the most recent 100 samples for this hour.
        if bucket.len() > 100 {
            let excess = bucket.len() - 100;
            bucket.drain(0..excess);
        }

        self.last_update = Some(at);
    }

    /// Predicted utilization `horizon` from now.
    ///
    /// Reads the bucket for the UTC hour that `now + horizon` falls in. When
    /// that hour has no samples yet the mean over every recorded hour is
    /// returned instead — still a measurement, just a coarser one.
    ///
    /// # Errors
    ///
    /// When nothing has been recorded at all. There is no honest number to
    /// return in that case.
    pub fn predict(&self, horizon: Duration) -> Result<f32> {
        self.predict_at(SystemTime::now() + horizon)
    }

    /// Predicted utilization for the UTC hour that `at` falls in.
    ///
    /// The absolute-time form of [`SeasonalAnalyzer::predict`]; taking the
    /// instant explicitly makes the bucket selection reproducible.
    pub fn predict_at(&self, at: SystemTime) -> Result<f32> {
        if self.hourly_patterns.is_empty() {
            return Err(TrustformersError::invalid_state(
                "seasonal prediction requires at least one recorded sample".to_string(),
            ));
        }

        let target_hour = Self::hour_of_day(at);
        if let Some(values) = self.hourly_patterns.get(&target_hour) {
            if !values.is_empty() {
                return Ok(values.iter().sum::<f32>() / values.len() as f32);
            }
        }

        let mut total = 0.0f32;
        let mut count = 0usize;
        for values in self.hourly_patterns.values() {
            total += values.iter().sum::<f32>();
            count += values.len();
        }
        if count == 0 {
            return Err(TrustformersError::invalid_state(
                "seasonal prediction requires at least one recorded sample".to_string(),
            ));
        }
        Ok(total / count as f32)
    }
}

/// Cost optimizer for cost-performance trade-offs
pub struct CostOptimizer {
    cost_model: CostModel,
    performance_model: PerformanceModel,
}

impl Default for CostOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CostOptimizer {
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::new(),
            performance_model: PerformanceModel::new(),
        }
    }

    pub fn calculate_current_cost(&self, nodes: usize, metrics: &PerformanceMetrics) -> f32 {
        self.cost_model.calculate_cost(nodes, metrics)
    }

    pub fn calculate_scale_up_cost(&self, new_nodes: usize, metrics: &PerformanceMetrics) -> f32 {
        self.cost_model.calculate_cost(new_nodes, metrics)
    }

    pub fn calculate_scale_down_cost(&self, new_nodes: usize, metrics: &PerformanceMetrics) -> f32 {
        self.cost_model.calculate_cost(new_nodes, metrics)
    }
}

/// Simple cost model
pub struct CostModel {
    cost_per_node_hour: f32,
    bandwidth_cost_factor: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    pub fn new() -> Self {
        Self {
            cost_per_node_hour: 3.0,    // $3 per GPU hour
            bandwidth_cost_factor: 0.1, // $0.1 per GB
        }
    }

    pub fn calculate_cost(&self, nodes: usize, metrics: &PerformanceMetrics) -> f32 {
        let compute_cost = nodes as f32 * self.cost_per_node_hour;
        let bandwidth_cost = metrics.bandwidth_utilization * self.bandwidth_cost_factor;
        compute_cost + bandwidth_cost
    }
}

/// Simple performance model
pub struct PerformanceModel {
    scaling_efficiency: f32,
}

impl Default for PerformanceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceModel {
    pub fn new() -> Self {
        Self {
            scaling_efficiency: 0.85, // 85% scaling efficiency
        }
    }

    pub fn predict_performance(&self, nodes: usize, base_throughput: f32) -> f32 {
        base_throughput * nodes as f32 * self.scaling_efficiency
    }
}

/// Smart checkpoint manager with differential checkpointing
pub struct SmartCheckpointManager {
    config: CheckpointConfig,
    checkpoint_history: Vec<CheckpointInfo>,
    compression_enabled: bool,
    validation_enabled: bool,
    differential_enabled: bool,
    checkpoint_dir: PathBuf,
    /// Model state as of the most recent checkpoint; the baseline that
    /// differential checkpoints are diffed against.
    baseline_state: HashMap<String, Tensor>,
    /// Absolute change below which an element is considered unchanged.
    differential_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Base checkpoint frequency (steps)
    pub base_frequency: usize,
    /// Enable adaptive frequency based on performance
    pub adaptive_frequency: bool,
    /// Maximum checkpoint file size (MB)
    pub max_file_size_mb: usize,
    /// Number of checkpoints to retain
    pub retention_count: usize,
    /// Enable checkpoint compression
    pub compression: bool,
    /// Enable checkpoint validation
    pub validation: bool,
    /// Enable differential checkpointing
    pub differential: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            base_frequency: 1000,
            adaptive_frequency: true,
            max_file_size_mb: 1024, // 1GB
            retention_count: 5,
            compression: true,
            validation: true,
            differential: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub step: usize,
    pub timestamp: SystemTime,
    pub file_path: PathBuf,
    pub file_size: usize,
    pub validation_passed: bool,
    pub is_differential: bool,
    pub base_checkpoint: Option<usize>, // For differential checkpoints
}

impl SmartCheckpointManager {
    pub fn new(config: CheckpointConfig, checkpoint_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&checkpoint_dir)?;

        let compression_enabled = config.compression;
        let validation_enabled = config.validation;
        let differential_enabled = config.differential;

        Ok(Self {
            config,
            checkpoint_history: Vec::new(),
            compression_enabled,
            validation_enabled,
            differential_enabled,
            checkpoint_dir,
            baseline_state: HashMap::new(),
            differential_threshold: 0.0,
        })
    }

    /// Set the absolute change below which an element is treated as unchanged
    /// by differential checkpointing. The default, `0.0`, records every element
    /// that differs at all (lossless).
    pub fn with_differential_threshold(mut self, threshold: f32) -> Self {
        self.differential_threshold = threshold.max(0.0);
        self
    }

    /// Model state the next differential checkpoint will be diffed against.
    pub fn baseline_state(&self) -> &HashMap<String, Tensor> {
        &self.baseline_state
    }

    pub fn should_checkpoint(&self, step: usize, performance_metrics: &PerformanceMetrics) -> bool {
        if step.is_multiple_of(self.config.base_frequency) {
            return true;
        }

        if self.config.adaptive_frequency {
            // Adaptive checkpointing based on performance trends
            self.adaptive_checkpoint_decision(step, performance_metrics)
        } else {
            false
        }
    }

    fn adaptive_checkpoint_decision(&self, _step: usize, metrics: &PerformanceMetrics) -> bool {
        // Checkpoint more frequently during unstable training
        let avg_gpu_util =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;
        let performance_variance = self.calculate_performance_variance(metrics);

        // High variance or low utilization suggests potential instability
        performance_variance > 0.1 || avg_gpu_util < 0.5
    }

    fn calculate_performance_variance(&self, metrics: &PerformanceMetrics) -> f32 {
        if metrics.gpu_utilization.is_empty() {
            return 0.0;
        }

        let mean =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;
        let variance = metrics.gpu_utilization.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / metrics.gpu_utilization.len() as f32;

        variance.sqrt()
    }

    pub fn create_checkpoint(
        &mut self,
        step: usize,
        model_state: &HashMap<String, Tensor>,
    ) -> Result<CheckpointInfo> {
        let timestamp = SystemTime::now();

        // Determine checkpoint type
        let is_differential = self.differential_enabled && !self.checkpoint_history.is_empty();
        let base_checkpoint = if is_differential {
            self.checkpoint_history.last().map(|c| c.step)
        } else {
            None
        };

        // Create checkpoint file path
        let filename = if is_differential {
            let base = base_checkpoint.ok_or_else(|| {
                TrustformersError::invalid_state(
                    "Base checkpoint must exist when differential checkpointing is enabled"
                        .to_string(),
                )
            })?;
            format!("checkpoint_step_{}_diff_{}.ckpt", step, base)
        } else {
            format!("checkpoint_step_{}_full.ckpt", step)
        };
        let file_path = self.checkpoint_dir.join(filename);

        // Create checkpoint data
        let checkpoint_data = if is_differential {
            self.create_differential_checkpoint(model_state)?
        } else {
            self.create_full_checkpoint(model_state)?
        };

        // Compress if enabled
        let final_data = if self.compression_enabled {
            self.compress_checkpoint(&checkpoint_data)?
        } else {
            checkpoint_data
        };

        // Write checkpoint file
        std::fs::write(&file_path, &final_data)?;
        let file_size = final_data.len();

        // Validate checkpoint if enabled
        let validation_passed = if self.validation_enabled {
            self.validate_checkpoint(&file_path)?
        } else {
            true
        };

        let checkpoint_info = CheckpointInfo {
            step,
            timestamp,
            file_path,
            file_size,
            validation_passed,
            is_differential,
            base_checkpoint,
        };

        self.checkpoint_history.push(checkpoint_info.clone());

        // The state just written becomes the baseline for the next differential
        // checkpoint.
        self.baseline_state = model_state.clone();

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        log::info!(
            "checkpoint created: step {}, {:.2} MiB, {}",
            step,
            file_size as f32 / (1024.0 * 1024.0),
            if is_differential { "differential" } else { "full" }
        );

        Ok(checkpoint_info)
    }

    /// Serialize the complete model state.
    ///
    /// Tensor payloads are IEEE-754 `f32` little-endian bytes, so a
    /// save→load round trip is bit-identical. See
    /// [`checkpoint_format`](self::checkpoint_format) for the layout.
    fn create_full_checkpoint(&self, model_state: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
        checkpoint_format::encode_full(model_state)
    }

    /// Serialize only the elements that changed since the previous checkpoint.
    ///
    /// The baseline is the state captured at the last successful
    /// [`SmartCheckpointManager::create_checkpoint`]. Elements whose absolute
    /// change does not exceed [`SmartCheckpointManager::differential_threshold`]
    /// are omitted entirely.
    fn create_differential_checkpoint(
        &self,
        model_state: &HashMap<String, Tensor>,
    ) -> Result<Vec<u8>> {
        let base_step = self.checkpoint_history.last().map(|c| c.step).ok_or_else(|| {
            TrustformersError::invalid_state(
                "differential checkpointing requires a previous checkpoint".to_string(),
            )
        })?;

        checkpoint_format::encode_differential(
            model_state,
            &self.baseline_state,
            base_step,
            self.differential_threshold,
        )
    }

    /// Losslessly compress a serialized checkpoint (zero-run-length encoding).
    fn compress_checkpoint(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(checkpoint_format::compress(data))
    }

    /// Validate a checkpoint by fully parsing it back, not by looking at its
    /// size.
    ///
    /// Differential checkpoints are validated against the manager's baseline
    /// state, which is exactly what a restore would use.
    fn validate_checkpoint(&self, file_path: &PathBuf) -> Result<bool> {
        let raw = std::fs::read(file_path)?;
        let payload = match checkpoint_format::decompress(&raw) {
            Ok(payload) => payload,
            Err(_) => return Ok(false),
        };

        if checkpoint_format::is_differential(&payload) {
            Ok(checkpoint_format::decode_differential(&payload, &self.baseline_state).is_ok())
        } else {
            Ok(checkpoint_format::decode_full(&payload).is_ok())
        }
    }

    /// Restore the model state recorded at `step`.
    ///
    /// Differential checkpoints are replayed on top of the nearest preceding
    /// full checkpoint, so any step in the retained history can be restored.
    pub fn load_checkpoint(&self, step: usize) -> Result<HashMap<String, Tensor>> {
        let target =
            self.checkpoint_history
                .iter()
                .position(|info| info.step == step)
                .ok_or_else(|| {
                    TrustformersError::invalid_input(format!(
                        "no checkpoint recorded for step {step}"
                    ))
                })?;

        // Walk back to the most recent full checkpoint.
        let mut anchor = target;
        while self.checkpoint_history[anchor].is_differential {
            if anchor == 0 {
                return Err(TrustformersError::invalid_state(
                    "checkpoint history starts with a differential checkpoint; the base is gone"
                        .to_string(),
                ));
            }
            anchor -= 1;
        }

        let mut state = checkpoint_format::decode_full(&self.read_payload(anchor)?)?;
        for index in (anchor + 1)..=target {
            let payload = self.read_payload(index)?;
            let (_, next) = checkpoint_format::decode_differential(&payload, &state)?;
            state = next;
        }

        Ok(state)
    }

    fn read_payload(&self, index: usize) -> Result<Vec<u8>> {
        let info = self.checkpoint_history.get(index).ok_or_else(|| {
            TrustformersError::invalid_input(format!("checkpoint index {index} is out of range"))
        })?;
        let raw = std::fs::read(&info.file_path)?;
        checkpoint_format::decompress(&raw)
    }

    /// Drop the oldest checkpoints beyond the retention count.
    ///
    /// A full checkpoint is never dropped while a differential checkpoint still
    /// depends on it, because doing so would make every dependent checkpoint
    /// unrestorable.
    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.checkpoint_history.len() <= self.config.retention_count {
            return Ok(());
        }

        let mut to_remove = self.checkpoint_history.len() - self.config.retention_count;
        while to_remove > 0 {
            let next_is_dependent =
                self.checkpoint_history.get(1).is_some_and(|info| info.is_differential);
            if next_is_dependent {
                log::debug!(
                    "retaining checkpoint at step {} because later differential checkpoints \
                     depend on it",
                    self.checkpoint_history[0].step
                );
                break;
            }

            let removed = self.checkpoint_history.remove(0);
            if let Err(err) = std::fs::remove_file(&removed.file_path) {
                log::warn!(
                    "failed to remove old checkpoint {}: {err}",
                    removed.file_path.display()
                );
            }
            to_remove -= 1;
        }

        Ok(())
    }

    pub fn get_latest_checkpoint(&self) -> Option<&CheckpointInfo> {
        self.checkpoint_history.last()
    }

    pub fn get_checkpoint_history(&self) -> &[CheckpointInfo] {
        &self.checkpoint_history
    }
}

/// Performance ML optimizer using machine learning for performance optimization
pub struct PerformanceMLOptimizer {
    config: MLOptimizerConfig,
    performance_model: Arc<Mutex<MLPerformanceModel>>,
    optimization_history: Vec<OptimizationResult>,
    last_optimization: Instant,
}

#[derive(Debug, Clone)]
pub struct MLOptimizerConfig {
    /// Prediction horizon (steps)
    pub prediction_horizon: usize,
    /// Optimization frequency (steps)
    pub optimization_frequency: usize,
    /// Enable automatic parameter tuning
    pub auto_tuning: bool,
    /// Learning rate for ML model updates
    pub model_learning_rate: f32,
    /// Enable advanced feature engineering
    pub feature_engineering: bool,
}

impl Default for MLOptimizerConfig {
    fn default() -> Self {
        Self {
            prediction_horizon: 100,
            optimization_frequency: 50,
            auto_tuning: true,
            model_learning_rate: 0.001,
            feature_engineering: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub timestamp: SystemTime,
    pub optimization_type: OptimizationType,
    /// Fraction of step time this change is *predicted* to save, derived from
    /// the metrics that were measured and the parameter change that was
    /// actually applied.
    ///
    /// This is a prediction, never an observation: the change has not run yet
    /// when the result is produced. It is always a function of
    /// [`PerformanceMetrics`] and the applied delta — never a constant.
    /// Compare consecutive [`PerformanceMetrics::step_time`] samples for the
    /// realised effect.
    pub performance_improvement: f32,
    pub parameters_changed: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    BatchSizeOptimization,
    LearningRateScheduling,
    CommunicationPatternOptimization,
    MemoryOptimization,
    CompressionOptimization,
}

impl PerformanceMLOptimizer {
    pub fn new(config: MLOptimizerConfig) -> Self {
        Self {
            config,
            performance_model: Arc::new(Mutex::new(MLPerformanceModel::new())),
            optimization_history: Vec::new(),
            // Initialize to a time in the past so first optimization can run immediately
            last_optimization: Instant::now() - Duration::from_secs(120),
        }
    }

    pub fn with_prediction_horizon(mut self, horizon: usize) -> Self {
        self.config.prediction_horizon = horizon;
        self
    }

    pub fn with_optimization_frequency(mut self, frequency: usize) -> Self {
        self.config.optimization_frequency = frequency;
        self
    }

    pub fn should_optimize(&self, step: usize) -> bool {
        step.is_multiple_of(self.config.optimization_frequency)
            && self.last_optimization.elapsed() > Duration::from_secs(60) // At least 1 minute between optimizations
    }

    pub fn optimize_performance(
        &mut self,
        current_metrics: &PerformanceMetrics,
        training_config: &mut DistributedConfig,
    ) -> Result<Vec<OptimizationResult>> {
        let mut optimizations = Vec::new();

        // Update ML model with current metrics
        {
            let mut model = self.performance_model.lock().map_err(|_| {
                TrustformersError::lock_error("performance model mutex poisoned".to_string())
            })?;
            model.update_training_data(current_metrics)?;
        }

        // Perform different types of optimizations
        if self.config.auto_tuning {
            // Batch size optimization
            if let Some(result) = self.optimize_batch_sizes(current_metrics, training_config)? {
                optimizations.push(result);
            }

            // Compression optimization
            if let Some(result) = self.optimize_compression(current_metrics, training_config)? {
                optimizations.push(result);
            }

            // Communication pattern optimization
            if let Some(result) = self.optimize_communication(current_metrics, training_config)? {
                optimizations.push(result);
            }
        }

        self.optimization_history.extend(optimizations.clone());
        self.last_optimization = Instant::now();

        Ok(optimizations)
    }

    /// Interconnect bandwidth, in MB/s, below which gradient compression is
    /// worth enabling: roughly a single saturated 1 GbE link.
    pub const SLOW_INTERCONNECT_MBPS: f32 = 125.0;

    fn optimize_batch_sizes(
        &self,
        metrics: &PerformanceMetrics,
        config: &mut DistributedConfig,
    ) -> Result<Option<OptimizationResult>> {
        let avg_utilization =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;
        let avg_memory =
            metrics.memory_usage.iter().sum::<f32>() / metrics.memory_usage.len() as f32;

        // Predict optimal batch size based on utilization and memory
        let model = self.performance_model.lock().map_err(|_| {
            TrustformersError::lock_error("performance model mutex poisoned".to_string())
        })?;
        let predicted_optimal_batch =
            model.predict_optimal_batch_size(avg_utilization, avg_memory)?;

        let current_batch = config.dynamic_batching.initial_batch_size as f32;
        if !(current_batch > 0.0 && predicted_optimal_batch > 0.0) {
            return Ok(None);
        }
        let size_change = (predicted_optimal_batch - current_batch) / current_batch;

        if size_change.abs() <= 0.1 {
            // Less than a 10% change is not worth disturbing the schedule for.
            return Ok(None);
        }

        // The batch size is an integer; derive the prediction from the value
        // that is actually written, not from the un-truncated estimate.
        let applied_batch = predicted_optimal_batch as usize;
        if applied_batch == 0 {
            return Ok(None);
        }
        config.dynamic_batching.initial_batch_size = applied_batch;
        let applied = applied_batch as f32;

        // A larger batch amortizes the fixed per-step collective over more
        // samples, so it removes `1 - current/new` of the communication phase.
        // A *smaller* batch is chosen to relieve memory pressure and predicts no
        // step-time saving at all — reporting the raw size delta as a
        // "performance improvement" would invert the sign of a slowdown.
        let predicted = if applied > current_batch {
            (metrics.communication_overhead * (1.0 - current_batch / applied)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut params_changed = HashMap::new();
        params_changed.insert("batch_size".to_string(), applied);

        Ok(Some(OptimizationResult {
            timestamp: SystemTime::now(),
            optimization_type: OptimizationType::BatchSizeOptimization,
            performance_improvement: predicted,
            parameters_changed: params_changed,
        }))
    }

    /// Tighten the gradient-compression ratio when communication dominates the
    /// step.
    ///
    /// The predicted saving follows directly from the change that is applied:
    /// transferred bytes scale with the target ratio, so shrinking it from
    /// `old` to `new` removes `1 - new/old` of the communication time, and
    /// communication is [`PerformanceMetrics::communication_overhead`] of the
    /// step. No constant is invented.
    fn optimize_compression(
        &self,
        metrics: &PerformanceMetrics,
        config: &mut DistributedConfig,
    ) -> Result<Option<OptimizationResult>> {
        const FLOOR: f32 = 0.05;
        const TIGHTEN: f32 = 0.8;

        if metrics.communication_overhead <= 0.3 {
            return Ok(None);
        }

        let old_ratio = config.compression.target_ratio;
        let new_ratio = (old_ratio * TIGHTEN).max(FLOOR);
        if !(old_ratio.is_finite() && old_ratio > 0.0) || new_ratio >= old_ratio {
            // Already at the floor: there is nothing left to tighten, so there
            // is no optimization to report.
            return Ok(None);
        }
        config.compression.target_ratio = new_ratio;

        let payload_reduction = 1.0 - new_ratio / old_ratio;
        let predicted = (metrics.communication_overhead * payload_reduction).clamp(0.0, 1.0);

        let mut params_changed = HashMap::new();
        params_changed.insert("compression_ratio".to_string(), new_ratio);

        Ok(Some(OptimizationResult {
            timestamp: SystemTime::now(),
            optimization_type: OptimizationType::CompressionOptimization,
            performance_improvement: predicted,
            parameters_changed: params_changed,
        }))
    }

    /// Turn gradient compression on when the interconnect is the bottleneck.
    ///
    /// This is the only communication knob [`DistributedConfig`] exposes — it
    /// carries no topology, bucket-size or overlap setting — so when
    /// compression is already enabled there is nothing to change and the
    /// function reports no optimization rather than an imagined one.
    ///
    /// `bandwidth_utilization` is a measured MB/s figure; below
    /// [`SLOW_INTERCONNECT_MBPS`](Self::SLOW_INTERCONNECT_MBPS) the link is
    /// slower than a single 1 GbE hop and compression pays for itself.
    fn optimize_communication(
        &self,
        metrics: &PerformanceMetrics,
        config: &mut DistributedConfig,
    ) -> Result<Option<OptimizationResult>> {
        if metrics.bandwidth_utilization >= Self::SLOW_INTERCONNECT_MBPS
            || config.compression.enabled
        {
            return Ok(None);
        }

        config.compression.enabled = true;

        // Enabling compression removes `1 - target_ratio` of the transferred
        // bytes from a phase that takes `communication_overhead` of the step.
        let ratio = config.compression.target_ratio.clamp(0.0, 1.0);
        let predicted = (metrics.communication_overhead * (1.0 - ratio)).clamp(0.0, 1.0);

        let mut params_changed = HashMap::new();
        params_changed.insert("compression_enabled".to_string(), 1.0);
        params_changed.insert("compression_ratio".to_string(), ratio);

        Ok(Some(OptimizationResult {
            timestamp: SystemTime::now(),
            optimization_type: OptimizationType::CommunicationPatternOptimization,
            performance_improvement: predicted,
            parameters_changed: params_changed,
        }))
    }

    pub fn get_optimization_history(&self) -> &[OptimizationResult] {
        &self.optimization_history
    }
}

/// Simple ML performance model
pub struct MLPerformanceModel {
    training_data: Vec<(Vec<f32>, f32)>, // (features, target)
    model_weights: Vec<f32>,
    learning_rate: f32,
}

impl Default for MLPerformanceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MLPerformanceModel {
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            model_weights: vec![0.5, 0.3, 0.2, 0.1], // Simple linear model weights
            learning_rate: 0.001,
        }
    }

    pub fn update_training_data(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        // Extract features from metrics
        let features = vec![
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32,
            metrics.memory_usage.iter().sum::<f32>() / metrics.memory_usage.len() as f32,
            metrics.communication_overhead,
            metrics.bandwidth_utilization,
        ];

        let target = metrics.throughput;

        self.training_data.push((features, target));

        // Keep only recent training data
        if self.training_data.len() > 1000 {
            self.training_data.drain(0..500);
        }

        // Simple online learning update
        if self.training_data.len() > 10 {
            self.update_model_weights()?;
        }

        Ok(())
    }

    fn update_model_weights(&mut self) -> Result<()> {
        if self.training_data.is_empty() {
            return Ok(());
        }

        // Simple gradient descent update
        for (features, target) in &self.training_data {
            let prediction = self.predict_with_features(features)?;
            let error = target - prediction;

            // Update weights
            for i in 0..self.model_weights.len().min(features.len()) {
                self.model_weights[i] += self.learning_rate * error * features[i];
            }
        }

        Ok(())
    }

    pub fn predict_optimal_batch_size(
        &self,
        gpu_utilization: f32,
        memory_usage: f32,
    ) -> Result<f32> {
        // Simple heuristic for batch size prediction
        let utilization_factor = if gpu_utilization < 0.7 {
            1.2
        } else if gpu_utilization > 0.9 {
            0.8
        } else {
            1.0
        };
        let memory_factor = if memory_usage > 0.9 {
            0.7
        } else if memory_usage < 0.5 {
            1.3
        } else {
            1.0
        };

        let base_batch_size = 32.0_f32;
        let optimal_batch: f32 = base_batch_size * utilization_factor * memory_factor;

        Ok(optimal_batch.clamp(8.0_f32, 256.0_f32)) // Clamp to reasonable range
    }

    fn predict_with_features(&self, features: &[f32]) -> Result<f32> {
        let prediction = features
            .iter()
            .zip(self.model_weights.iter())
            .map(|(&f, &w)| f * w)
            .sum::<f32>();

        Ok(prediction.max(0.0)) // Ensure non-negative prediction
    }
}

#[cfg(test)]
mod tests;
