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
            ScalingStrategy::Custom(_) => self.custom_scaling(metrics)?,
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

        // Get prediction for next 10 minutes
        let predicted_load = self.workload_predictor.predict_workload(Duration::from_secs(600))?;

        // Make scaling decision based on prediction
        if predicted_load > self.config.scale_up_threshold * 1.1 && // Add 10% buffer
           self.current_nodes < self.config.max_nodes
        {
            let nodes_to_add =
                ((predicted_load - 0.75) * self.current_nodes as f32).ceil() as usize;
            Ok(ScalingDecision::ScaleUp(
                nodes_to_add.min(self.config.max_nodes - self.current_nodes),
            ))
        } else if predicted_load < self.config.scale_down_threshold * 0.9 && // Add 10% buffer
                  self.current_nodes > self.config.min_nodes
        {
            let target_nodes = (predicted_load / 0.8 * self.current_nodes as f32).ceil() as usize;
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

    fn custom_scaling(&self, _metrics: &PerformanceMetrics) -> Result<ScalingDecision> {
        // Placeholder for custom scaling logic
        Ok(ScalingDecision::NoAction)
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

    pub fn update_metrics(&mut self, metrics: &PerformanceMetrics) {
        let avg_utilization =
            metrics.gpu_utilization.iter().sum::<f32>() / metrics.gpu_utilization.len() as f32;
        let now = Instant::now();

        self.historical_data.push_back((now, avg_utilization));
        if self.historical_data.len() > 10000 {
            self.historical_data.pop_front();
        }

        self.trend_analyzer.update(avg_utilization);
        self.seasonal_analyzer.update(now, avg_utilization);
    }

    pub fn predict_workload(&self, horizon: Duration) -> Result<f32> {
        if self.historical_data.len() < 10 {
            // Not enough data for prediction
            return Ok(0.75); // Default conservative estimate
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

/// Simple seasonal analyzer
pub struct SeasonalAnalyzer {
    hourly_patterns: HashMap<u32, Vec<f32>>, // hour -> values
    last_update: Option<Instant>,
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

    pub fn update(&mut self, timestamp: Instant, value: f32) {
        // Simplified: use milliseconds modulo as hour approximation
        let pseudo_hour = (timestamp.elapsed().as_secs() / 3600) % 24;

        self.hourly_patterns.entry(pseudo_hour as u32).or_default().push(value);

        // Keep only recent values (last 100 per hour)
        for values in self.hourly_patterns.values_mut() {
            if values.len() > 100 {
                values.drain(0..50); // Remove oldest 50
            }
        }

        self.last_update = Some(timestamp);
    }

    pub fn predict(&self, _horizon: Duration) -> Result<f32> {
        if self.hourly_patterns.is_empty() {
            return Ok(0.75); // Default
        }

        // Simple average of all patterns
        let all_values: Vec<f32> =
            self.hourly_patterns.values().flat_map(|v| v.iter()).cloned().collect();

        if all_values.is_empty() {
            Ok(0.75)
        } else {
            Ok(all_values.iter().sum::<f32>() / all_values.len() as f32)
        }
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
        config.dynamic_batching.initial_batch_size = predicted_optimal_batch as usize;

        // A larger batch amortizes the fixed per-step collective over more
        // samples, so it removes `1 - current/new` of the communication phase.
        // A *smaller* batch is chosen to relieve memory pressure and predicts no
        // step-time saving at all — reporting the raw size delta as a
        // "performance improvement" would invert the sign of a slowdown.
        let predicted = if predicted_optimal_batch > current_batch {
            (metrics.communication_overhead * (1.0 - current_batch / predicted_optimal_batch))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut params_changed = HashMap::new();
        params_changed.insert("batch_size".to_string(), predicted_optimal_batch);

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
mod tests {
    use super::*;

    fn scratch_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "trustformers-ckpt-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn model_state(scale: f32) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        state.insert(
            "encoder.weight".to_string(),
            Tensor::from_slice(
                &[0.11 * scale, -0.25 * scale, 0.5 * scale, -0.75 * scale],
                &[2, 2],
            )
            .expect("tensor must build in test"),
        );
        state.insert(
            "encoder.bias".to_string(),
            Tensor::from_slice(&[-0.03 * scale, 0.07 * scale], &[2])
                .expect("tensor must build in test"),
        );
        state
    }

    fn assert_states_equal(actual: &HashMap<String, Tensor>, expected: &HashMap<String, Tensor>) {
        assert_eq!(actual.len(), expected.len(), "parameter count");
        for (name, tensor) in expected {
            let restored =
                actual.get(name).unwrap_or_else(|| panic!("parameter `{name}` was lost"));
            assert_eq!(restored.shape(), tensor.shape(), "`{name}` shape");
            assert_eq!(
                restored.to_vec_f32().expect("read"),
                tensor.to_vec_f32().expect("read"),
                "`{name}` values must be bit-identical"
            );
        }
    }

    #[test]
    fn checkpoint_save_load_round_trip_is_bit_identical() {
        let dir = scratch_dir("full");
        let config = CheckpointConfig {
            differential: false,
            compression: false,
            ..CheckpointConfig::default()
        };
        let mut manager =
            SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

        let state = model_state(1.0);
        let info = manager.create_checkpoint(10, &state).expect("checkpoint in test");
        assert!(info.validation_passed, "a real checkpoint must validate");
        assert!(!info.is_differential);

        let restored = manager.load_checkpoint(10).expect("load in test");
        assert_states_equal(&restored, &state);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_does_not_zero_sub_unit_weights() {
        // Regression: the previous serializer wrote `tensor.to_vec_u8()`,
        // which casts f32 values to u8 and mapped every weight in (-1, 1) to 0.
        let dir = scratch_dir("subunit");
        let config = CheckpointConfig {
            differential: false,
            compression: true,
            ..CheckpointConfig::default()
        };
        let mut manager =
            SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

        let mut state = HashMap::new();
        state.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.004, -0.9, 0.5, 0.125], &[4])
                .expect("tensor must build in test"),
        );

        manager.create_checkpoint(1, &state).expect("checkpoint in test");
        let restored = manager.load_checkpoint(1).expect("load in test");
        let values = restored.get("w").expect("parameter must survive").to_vec_f32().expect("read");

        assert_eq!(values, vec![0.004f32, -0.9, 0.5, 0.125]);
        assert!(values.iter().all(|value| *value != 0.0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn differential_checkpoints_replay_onto_the_base() {
        let dir = scratch_dir("diff");
        let config = CheckpointConfig {
            differential: true,
            compression: true,
            retention_count: 10,
            ..CheckpointConfig::default()
        };
        let mut manager =
            SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

        let first = model_state(1.0);
        let second = model_state(2.0);
        let third = model_state(3.0);

        let full = manager.create_checkpoint(1, &first).expect("checkpoint in test");
        assert!(!full.is_differential);
        let diff_one = manager.create_checkpoint(2, &second).expect("checkpoint in test");
        assert!(diff_one.is_differential);
        let diff_two = manager.create_checkpoint(3, &third).expect("checkpoint in test");
        assert!(diff_two.is_differential);
        assert!(diff_two.validation_passed);

        assert_states_equal(&manager.load_checkpoint(1).expect("load in test"), &first);
        assert_states_equal(&manager.load_checkpoint(2).expect("load in test"), &second);
        assert_states_equal(&manager.load_checkpoint(3).expect("load in test"), &third);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn differential_checkpoint_is_smaller_than_a_full_one() {
        let dir = scratch_dir("size");
        let config = CheckpointConfig {
            differential: true,
            compression: false,
            retention_count: 10,
            ..CheckpointConfig::default()
        };
        let mut manager =
            SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

        let mut state = HashMap::new();
        state.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.5f32; 256], &[256]).expect("tensor must build in test"),
        );
        let full = manager.create_checkpoint(1, &state).expect("checkpoint in test");

        // Change a single element.
        let mut changed: Vec<f32> = vec![0.5f32; 256];
        changed[7] = -1.25;
        state.insert(
            "w".to_string(),
            Tensor::from_slice(&changed, &[256]).expect("tensor must build in test"),
        );
        let diff = manager.create_checkpoint(2, &state).expect("checkpoint in test");

        assert!(
            diff.file_size < full.file_size,
            "a one-element delta must be smaller than the full state ({} vs {})",
            diff.file_size,
            full.file_size
        );
        assert_states_equal(&manager.load_checkpoint(2).expect("load in test"), &state);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loading_an_unknown_step_errors() {
        let dir = scratch_dir("missing");
        let manager = SmartCheckpointManager::new(CheckpointConfig::default(), dir.clone())
            .expect("manager must build in test");
        assert!(manager.load_checkpoint(42).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auto_scaler_config() {
        let config = AutoScalerConfig {
            min_nodes: 2,
            max_nodes: 32,
            ..AutoScalerConfig::default()
        };

        // Validate the modified configuration
        assert_eq!(config.min_nodes, 2);
        assert_eq!(config.max_nodes, 32);
    }

    #[test]
    fn test_auto_scaler_creation() {
        let config = AutoScalerConfig::default();
        let auto_scaler = AutoScaler::new(config)
            .with_min_nodes(2)
            .with_max_nodes(16)
            .with_scaling_strategy(ScalingStrategy::Performance);

        assert_eq!(auto_scaler.get_current_nodes(), 2);
        assert!(matches!(
            auto_scaler.config.strategy,
            ScalingStrategy::Performance
        ));
    }

    #[test]
    fn test_workload_predictor() {
        let mut predictor = WorkloadPredictor::new();

        // Add some test data
        let metrics = PerformanceMetrics {
            throughput: 1000.0,
            gpu_utilization: vec![0.8, 0.7, 0.9],
            memory_usage: vec![0.6, 0.7, 0.5],
            communication_overhead: 0.2,
            compression_ratio: 0.1,
            bandwidth_utilization: 0.8,
            step_time: Duration::from_millis(100),
        };

        predictor.update_metrics(&metrics);

        let prediction = predictor
            .predict_workload(Duration::from_secs(600))
            .expect("Operation failed in test");
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn test_checkpoint_manager() {
        let config = CheckpointConfig::default();
        let temp_dir = std::env::temp_dir().join("test_checkpoints");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let manager = SmartCheckpointManager::new(config, temp_dir).expect("Construction failed");

        let metrics = PerformanceMetrics {
            throughput: 1000.0,
            gpu_utilization: vec![0.8],
            memory_usage: vec![0.6],
            communication_overhead: 0.2,
            compression_ratio: 0.1,
            bandwidth_utilization: 0.8,
            step_time: Duration::from_millis(100),
        };

        assert!(manager.should_checkpoint(1000, &metrics));
        assert!(!manager.should_checkpoint(999, &metrics));
    }

    #[test]
    fn test_ml_optimizer() {
        let config = MLOptimizerConfig::default();
        let optimizer = PerformanceMLOptimizer::new(config)
            .with_prediction_horizon(50)
            .with_optimization_frequency(25);

        assert_eq!(optimizer.config.prediction_horizon, 50);
        assert_eq!(optimizer.config.optimization_frequency, 25);

        assert!(optimizer.should_optimize(25));
        assert!(!optimizer.should_optimize(24));
    }

    #[test]
    fn test_trend_analyzer() {
        let mut analyzer = TrendAnalyzer::new();

        // Add increasing trend
        for i in 0..20 {
            analyzer.update(i as f32 * 0.1);
        }

        let prediction =
            analyzer.predict(Duration::from_secs(60)).expect("Operation failed in test");
        assert!(prediction > 1.0); // Should predict increasing trend
    }

    // ── ML optimizer: predictions must be derived, never constants ────────
    //
    // The previous implementation returned `performance_improvement: 0.15` for
    // every compression change and `0.08` for every "communication" change —
    // the latter without touching the configuration at all. These tests fail
    // against that code because they vary the measured metrics and require the
    // reported prediction to move with them.

    fn metrics(communication_overhead: f32, bandwidth_mbps: f32) -> PerformanceMetrics {
        PerformanceMetrics {
            throughput: 100.0,
            gpu_utilization: vec![0.75],
            memory_usage: vec![0.6],
            communication_overhead,
            compression_ratio: 1.0,
            bandwidth_utilization: bandwidth_mbps,
            step_time: Duration::from_millis(100),
        }
    }

    #[test]
    fn compression_prediction_tracks_the_measured_communication_overhead() {
        let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

        let mut light = DistributedConfig::new();
        light.compression.target_ratio = 0.5;
        let light_result = optimizer
            .optimize_compression(&metrics(0.4, 1000.0), &mut light)
            .expect("optimization must succeed in test")
            .expect("high communication overhead must trigger a change in test");

        let mut heavy = DistributedConfig::new();
        heavy.compression.target_ratio = 0.5;
        let heavy_result = optimizer
            .optimize_compression(&metrics(0.8, 1000.0), &mut heavy)
            .expect("optimization must succeed in test")
            .expect("high communication overhead must trigger a change in test");

        assert!(
            heavy_result.performance_improvement > light_result.performance_improvement,
            "a heavier communication phase must predict a larger saving: {} vs {}",
            heavy_result.performance_improvement,
            light_result.performance_improvement
        );

        // 20% fewer bytes out of a phase that is 80% of the step.
        assert!((heavy_result.performance_improvement - 0.8 * 0.2).abs() < 1e-5);

        // And the configuration really changed.
        assert!((heavy.compression.target_ratio - 0.4).abs() < 1e-6);
    }

    #[test]
    fn compression_at_the_floor_reports_no_optimization() {
        let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());
        let mut config = DistributedConfig::new();
        config.compression.target_ratio = 0.05;

        assert!(optimizer
            .optimize_compression(&metrics(0.9, 1000.0), &mut config)
            .expect("optimization must succeed in test")
            .is_none());
        assert!((config.compression.target_ratio - 0.05).abs() < 1e-6);
    }

    #[test]
    fn communication_optimization_applies_a_real_change_or_reports_none() {
        let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

        // Fast link: nothing to do.
        let mut fast = DistributedConfig::new();
        assert!(optimizer
            .optimize_communication(&metrics(0.5, 10_000.0), &mut fast)
            .expect("optimization must succeed in test")
            .is_none());
        assert!(!fast.compression.enabled);

        // Slow link with compression off: the knob is really turned.
        let mut slow = DistributedConfig::new();
        slow.compression.enabled = false;
        slow.compression.target_ratio = 0.25;
        let result = optimizer
            .optimize_communication(&metrics(0.5, 10.0), &mut slow)
            .expect("optimization must succeed in test")
            .expect("a slow interconnect must enable compression in test");
        assert!(slow.compression.enabled, "the config must actually change");
        assert!((result.performance_improvement - 0.5 * 0.75).abs() < 1e-5);

        // Already enabled: no further knob exists, so nothing is claimed.
        assert!(optimizer
            .optimize_communication(&metrics(0.5, 10.0), &mut slow)
            .expect("optimization must succeed in test")
            .is_none());
    }

    #[test]
    fn shrinking_the_batch_never_reports_a_speedup() {
        let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

        // High utilization and high memory pressure => the model shrinks the
        // batch. The old code reported the negative size delta as an
        // "improvement".
        let mut config = DistributedConfig::new();
        config.dynamic_batching.initial_batch_size = 128;
        let pressured = PerformanceMetrics {
            gpu_utilization: vec![0.95],
            memory_usage: vec![0.95],
            ..metrics(0.5, 1000.0)
        };

        let result = optimizer
            .optimize_batch_sizes(&pressured, &mut config)
            .expect("optimization must succeed in test")
            .expect("a >10% size change must be reported in test");

        assert!(config.dynamic_batching.initial_batch_size < 128);
        assert_eq!(
            result.performance_improvement, 0.0,
            "a batch reduction taken for memory headroom predicts no speedup"
        );
    }

    #[test]
    fn growing_the_batch_predicts_amortised_communication() {
        let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

        let mut config = DistributedConfig::new();
        config.dynamic_batching.initial_batch_size = 8;
        let idle = PerformanceMetrics {
            gpu_utilization: vec![0.5],
            memory_usage: vec![0.3],
            ..metrics(0.5, 1000.0)
        };

        let result = optimizer
            .optimize_batch_sizes(&idle, &mut config)
            .expect("optimization must succeed in test")
            .expect("an idle GPU must grow the batch in test");

        let new_batch = config.dynamic_batching.initial_batch_size as f32;
        assert!(new_batch > 8.0);
        let expected = 0.5 * (1.0 - 8.0 / new_batch);
        assert!(
            (result.performance_improvement - expected).abs() < 1e-4,
            "predicted {} but the derivation gives {expected}",
            result.performance_improvement
        );
    }
}
