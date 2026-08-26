//! # Enhanced Multi-GPU Distributed Training Framework
//!
//! This module provides advanced distributed training capabilities building upon
//! the existing multi-node infrastructure with focus on:
//! - Modern GPU communication patterns (NCCL integration)
//! - Advanced gradient compression and quantization
//! - Dynamic load balancing and fault tolerance
//! - Integration with cutting-edge optimizers (Averaged Adam, etc.)
//! - Real-time performance monitoring and auto-tuning
//!
//! ## Key Features
//!
//! 1. **GPU-Optimized Communication**: NCCL-based all-reduce with topology awareness
//! 2. **Advanced Gradient Compression**: Multiple compression algorithms with adaptive selection
//! 3. **Dynamic Load Balancing**: Automatic workload redistribution based on GPU performance
//! 4. **Fault Tolerance**: Automatic recovery from node failures with checkpoint restoration
//! 5. **Performance Auto-Tuning**: Real-time optimization of batch sizes and communication patterns
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{AveragedAdam, CompressionType, DistributedConfig, EnhancedDistributedTrainer};
//! # use std::collections::HashMap;
//! # use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create distributed configuration
//! let config = DistributedConfig::new()
//!     .with_gpus(8)
//!     .with_gradient_compression(CompressionType::PowerSGD { rank: 4 })
//!     .with_dynamic_batching(true)
//!     .with_fault_tolerance(true);
//!
//! // Initialize Averaged Adam for distributed training
//! let optimizer = AveragedAdam::for_distributed_training();
//!
//! // Create enhanced distributed trainer
//! let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)?;
//!
//! // Register model parameters
//! # let model_parameters: HashMap<String, Tensor> = HashMap::new();
//! trainer.register_model(model_parameters)?;
//!
//! // Training loop with automatic optimization
//! # let data_loader: Vec<HashMap<String, Tensor>> = Vec::new();
//! for batch in data_loader {
//!     trainer.train_step(batch)?;
//! }
//! # Ok(())
//! # }
//! ```

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use crate::averaged_adam::{AveragedAdam, AveragedAdamConfig};
use crate::multinode::{MultiNodeConfig, MultiNodeTrainer};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::parallel::CommunicationBackend;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Enhanced distributed training configuration with modern GPU optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of GPUs to use
    pub num_gpus: usize,
    /// GPU device IDs to use
    pub gpu_ids: Vec<usize>,
    /// Communication backend (NCCL preferred for GPUs)
    pub backend: CommunicationBackend,
    /// Gradient compression configuration
    pub compression: CompressionConfig,
    /// Dynamic batching configuration
    pub dynamic_batching: DynamicBatchingConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Performance monitoring settings
    pub monitoring: MonitoringConfig,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimizationConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            gpu_ids: vec![0],
            backend: CommunicationBackend::Nccl,
            compression: CompressionConfig::default(),
            dynamic_batching: DynamicBatchingConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            monitoring: MonitoringConfig::default(),
            memory_optimization: MemoryOptimizationConfig::default(),
        }
    }
}

impl DistributedConfig {
    /// Create new distributed configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of GPUs
    pub fn with_gpus(mut self, num_gpus: usize) -> Self {
        self.num_gpus = num_gpus;
        self.gpu_ids = (0..num_gpus).collect();
        self
    }

    /// Set specific GPU IDs
    pub fn with_gpu_ids(mut self, gpu_ids: Vec<usize>) -> Self {
        self.num_gpus = gpu_ids.len();
        self.gpu_ids = gpu_ids;
        self
    }

    /// Enable gradient compression
    pub fn with_gradient_compression(mut self, compression_type: CompressionType) -> Self {
        self.compression.enabled = true;
        self.compression.algorithm = compression_type;
        self
    }

    /// Enable dynamic batching
    pub fn with_dynamic_batching(mut self, enabled: bool) -> Self {
        self.dynamic_batching.enabled = enabled;
        self
    }

    /// Enable fault tolerance
    pub fn with_fault_tolerance(mut self, enabled: bool) -> Self {
        self.fault_tolerance.enabled = enabled;
        self
    }

    /// Set communication backend
    pub fn with_backend(mut self, backend: CommunicationBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Gradient compression algorithms for efficient communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression (baseline)
    None,
    /// Top-K sparsification
    TopK { k: usize },
    /// Random sparsification
    RandomSparsification { ratio: f32 },
    /// Quantization to lower precision
    Quantization { bits: u8 },
    /// PowerSGD low-rank compression
    PowerSGD { rank: usize },
    /// 1-Bit SGD compression
    OneBitSGD,
    /// Adaptive compression based on gradient statistics
    Adaptive,
}

/// Gradient compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionType,
    /// Compression ratio target (0.1 = 90% reduction)
    pub target_ratio: f32,
    /// Enable error feedback for compression
    pub error_feedback: bool,
    /// Adaptive compression threshold
    pub adaptive_threshold: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionType::TopK { k: 1000 },
            target_ratio: 0.1,
            error_feedback: true,
            adaptive_threshold: 0.01,
        }
    }
}

/// Dynamic batching configuration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchingConfig {
    pub enabled: bool,
    /// Initial batch size per GPU
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Target GPU utilization percentage
    pub target_utilization: f32,
    /// Batch size adjustment frequency (steps)
    pub adjustment_frequency: usize,
}

impl Default for DynamicBatchingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_batch_size: 32,
            min_batch_size: 8,
            max_batch_size: 128,
            target_utilization: 0.85,
            adjustment_frequency: 100,
        }
    }
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub enabled: bool,
    /// Checkpoint frequency (steps)
    pub checkpoint_frequency: usize,
    /// Maximum number of retries for failed operations
    pub max_retries: usize,
    /// Heartbeat interval for node health monitoring
    pub heartbeat_interval: Duration,
    /// Enable automatic node replacement
    pub auto_replacement: bool,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_frequency: 1000,
            max_retries: 3,
            heartbeat_interval: Duration::from_secs(10),
            auto_replacement: false,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    /// Enable real-time performance metrics
    pub real_time_metrics: bool,
    /// Enable automatic performance tuning
    pub auto_tuning: bool,
    /// Metrics collection frequency
    pub collection_frequency: Duration,
    /// Enable bandwidth monitoring
    pub bandwidth_monitoring: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            real_time_metrics: true,
            auto_tuning: false,
            collection_frequency: Duration::from_secs(1),
            bandwidth_monitoring: true,
        }
    }
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Enable offloading to CPU memory
    pub cpu_offloading: bool,
    /// Memory pool size for efficient allocation
    pub memory_pool_size_gb: f32,
    /// Enable automatic garbage collection
    pub auto_gc: bool,
    /// Memory usage threshold for triggering optimizations
    pub memory_threshold: f32,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            gradient_checkpointing: false,
            cpu_offloading: false,
            memory_pool_size_gb: 4.0,
            auto_gc: true,
            memory_threshold: 0.9,
        }
    }
}

/// Enhanced distributed trainer with modern GPU optimizations
pub struct EnhancedDistributedTrainer<T: Optimizer + StatefulOptimizer> {
    config: DistributedConfig,
    optimizer: T,
    multi_node_trainer: Option<MultiNodeTrainer<T>>,
    performance_monitor: PerformanceMonitor,
    gradient_compressor: GradientCompressor,
    dynamic_batcher: DynamicBatcher,
    fault_handler: FaultHandler,
    step_count: usize,
    start_time: Instant,
    gpu_contexts: Vec<Arc<GpuContext>>,
    parameter_registry: HashMap<String, ParameterInfo>,
    /// Gradients produced by the last [`EnhancedDistributedTrainer::train_step`]
    /// after compression, reduction and decompression, kept so the owner of the
    /// parameter tensors can apply them (see
    /// [`EnhancedDistributedTrainer::apply_reduced_gradients`]).
    reduced_gradients: HashMap<String, Tensor>,
}

/// GPU context for managing device-specific operations.
///
/// Every metric is `None` until a **real** sample is supplied through
/// [`EnhancedDistributedTrainer::record_gpu_telemetry`]. The pure-Rust default
/// build links no GPU runtime (NVML/ROCm SMI are C libraries), so this crate
/// cannot query a device itself. An earlier revision filled these fields with
/// `0.8 + random()` and friends, which made every downstream decision — dynamic
/// batch sizing, bottleneck detection, auto-scaling — act on invented data.
#[derive(Debug)]
pub struct GpuContext {
    /// Device index this context tracks.
    pub device_id: usize,
    /// Fraction of device memory in use, in `[0, 1]`; `None` when unknown.
    pub memory_usage: Arc<Mutex<Option<f32>>>,
    /// Device utilization in `[0, 1]`; `None` when unknown.
    pub utilization: Arc<Mutex<Option<f32>>>,
    /// Device temperature in degrees Celsius; `None` when unknown.
    pub temperature: Arc<Mutex<Option<f32>>>,
    /// Achieved interconnect bandwidth in MB/s; `None` when unknown.
    pub communication_bandwidth: Arc<Mutex<Option<f32>>>,
}

/// One real telemetry reading for a single device.
///
/// The embedder is responsible for obtaining these numbers (NVML, ROCm SMI,
/// `nvidia-smi`, a cluster metrics endpoint, …) and feeding them in with
/// [`EnhancedDistributedTrainer::record_gpu_telemetry`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuTelemetrySample {
    /// Device utilization as a fraction in `[0, 1]`.
    pub utilization: f32,
    /// Fraction of device memory in use, in `[0, 1]`.
    pub memory_usage: f32,
    /// Device temperature in degrees Celsius.
    pub temperature_celsius: f32,
    /// Achieved interconnect bandwidth in MB/s.
    pub communication_bandwidth_mb_s: f32,
}

impl GpuTelemetrySample {
    /// Validate the ranges a caller can get wrong silently.
    fn validate(&self, device_id: usize) -> Result<()> {
        for (label, value, upper) in [
            ("utilization", self.utilization, 1.0_f32),
            ("memory_usage", self.memory_usage, 1.0),
        ] {
            if !value.is_finite() || !(0.0..=upper).contains(&value) {
                return Err(TrustformersError::invalid_input(format!(
                    "GPU {device_id} telemetry `{label}` must be a finite fraction in [0, {upper}], got {value}"
                )));
            }
        }
        if !self.temperature_celsius.is_finite() {
            return Err(TrustformersError::invalid_input(format!(
                "GPU {device_id} telemetry `temperature_celsius` must be finite, got {}",
                self.temperature_celsius
            )));
        }
        if !self.communication_bandwidth_mb_s.is_finite() || self.communication_bandwidth_mb_s < 0.0
        {
            return Err(TrustformersError::invalid_input(format!(
                "GPU {device_id} telemetry `communication_bandwidth_mb_s` must be finite and \
                 non-negative, got {}",
                self.communication_bandwidth_mb_s
            )));
        }
        Ok(())
    }
}

/// Parameter information for distributed training
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub size: usize,
    pub device_id: usize,
    pub is_sharded: bool,
}

/// Performance metrics for distributed training.
///
/// # Device telemetry is opt-in
///
/// `gpu_utilization`, `memory_usage` and `bandwidth_utilization` are derived
/// from samples the embedder recorded with
/// [`EnhancedDistributedTrainer::record_gpu_telemetry`]. When a device has never
/// been sampled there is nothing to report, and this crate does **not** invent a
/// number: the two vectors are then left **empty** and `bandwidth_utilization`
/// is `0.0`. Treat an empty vector as "unknown", never as "0% utilized".
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Samples per second, measured by the throughput tracker.
    pub throughput: f32,
    /// Per-GPU utilization in `[0, 1]`; empty when no telemetry was recorded.
    pub gpu_utilization: Vec<f32>,
    /// Per-GPU memory usage in `[0, 1]`; empty when no telemetry was recorded.
    pub memory_usage: Vec<f32>,
    /// Fraction of step time spent communicating.
    pub communication_overhead: f32,
    /// Compression ratio achieved by the gradient codec.
    pub compression_ratio: f32,
    /// Mean interconnect bandwidth in MB/s; `0.0` when no telemetry was
    /// recorded.
    pub bandwidth_utilization: f32,
    /// Wall-clock time of the training step.
    pub step_time: Duration,
}

/// Real-time performance monitoring
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics_history: Vec<PerformanceMetrics>,
    last_collection: Instant,
    throughput_tracker: ThroughputTracker,
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics_history: Vec::new(),
            last_collection: Instant::now(),
            throughput_tracker: ThroughputTracker::new(),
        }
    }

    /// Read one `Option<f32>` metric from every context.
    ///
    /// Returns `None` unless *every* device has a recorded sample: a partially
    /// populated vector would silently misalign device indices with values.
    fn read_metric(
        gpu_contexts: &[Arc<GpuContext>],
        select: impl Fn(&GpuContext) -> &Arc<Mutex<Option<f32>>>,
        label: &str,
    ) -> Result<Option<Vec<f32>>> {
        let mut values = Vec::with_capacity(gpu_contexts.len());
        for ctx in gpu_contexts {
            let guard = select(ctx).lock().map_err(|_| {
                TrustformersError::lock_error(format!("GPU context {label} mutex poisoned"))
            })?;
            match *guard {
                Some(value) => values.push(value),
                None => return Ok(None),
            }
        }
        if values.is_empty() {
            return Ok(None);
        }
        Ok(Some(values))
    }

    /// Collect one metrics sample.
    ///
    /// Device-derived fields are populated only from telemetry the embedder
    /// recorded; see [`PerformanceMetrics`] for the "unknown" encoding.
    pub fn collect_metrics(
        &mut self,
        gpu_contexts: &[Arc<GpuContext>],
    ) -> Result<PerformanceMetrics> {
        let now = Instant::now();
        let step_time = now - self.last_collection;
        self.last_collection = now;

        let gpu_utilization =
            Self::read_metric(gpu_contexts, |ctx| &ctx.utilization, "utilization")?
                .unwrap_or_default();

        let memory_usage =
            Self::read_metric(gpu_contexts, |ctx| &ctx.memory_usage, "memory_usage")?
                .unwrap_or_default();

        let bandwidth_utilization = match Self::read_metric(
            gpu_contexts,
            |ctx| &ctx.communication_bandwidth,
            "communication_bandwidth",
        )? {
            Some(values) => values.iter().sum::<f32>() / values.len() as f32,
            None => 0.0,
        };

        let throughput = self.throughput_tracker.calculate_throughput();

        let metrics = PerformanceMetrics {
            throughput,
            gpu_utilization,
            memory_usage,
            communication_overhead: 0.0, // Will be calculated based on timing
            compression_ratio: 0.0,      // Will be set by compression module
            bandwidth_utilization,
            step_time,
        };

        self.metrics_history.push(metrics.clone());

        // Keep only recent metrics
        if self.metrics_history.len() > 1000 {
            self.metrics_history.drain(0..500);
        }

        Ok(metrics)
    }

    pub fn get_recent_metrics(&self, count: usize) -> &[PerformanceMetrics] {
        let start = self.metrics_history.len().saturating_sub(count);
        &self.metrics_history[start..]
    }

    pub fn analyze_performance_trends(&self) -> PerformanceAnalysis {
        if self.metrics_history.len() < 10 {
            return PerformanceAnalysis::default();
        }

        let recent_metrics = self.get_recent_metrics(100);

        let avg_throughput =
            recent_metrics.iter().map(|m| m.throughput).sum::<f32>() / recent_metrics.len() as f32;

        // Only samples that actually carry device telemetry contribute; an
        // empty `gpu_utilization` means "unknown", and averaging it in as 0.0
        // (or dividing by zero) would manufacture a utilization figure.
        let mut util_samples = 0usize;
        let mut util_total = 0.0f32;
        for m in recent_metrics {
            if m.gpu_utilization.is_empty() {
                continue;
            }
            util_total += m.gpu_utilization.iter().sum::<f32>() / m.gpu_utilization.len() as f32;
            util_samples += 1;
        }
        let avg_gpu_util = if util_samples == 0 { 0.0 } else { util_total / util_samples as f32 };

        let avg_comm_overhead =
            recent_metrics.iter().map(|m| m.communication_overhead).sum::<f32>()
                / recent_metrics.len() as f32;

        PerformanceAnalysis {
            average_throughput: avg_throughput,
            average_gpu_utilization: avg_gpu_util,
            average_communication_overhead: avg_comm_overhead,
            performance_trend: self.calculate_trend(),
            bottleneck_analysis: self.identify_bottlenecks(recent_metrics),
        }
    }

    fn calculate_trend(&self) -> PerformanceTrend {
        if self.metrics_history.len() < 20 {
            return PerformanceTrend::Stable;
        }

        let recent = self.get_recent_metrics(10);
        let older =
            &self.metrics_history[self.metrics_history.len() - 20..self.metrics_history.len() - 10];

        let recent_avg = recent.iter().map(|m| m.throughput).sum::<f32>() / recent.len() as f32;
        let older_avg = older.iter().map(|m| m.throughput).sum::<f32>() / older.len() as f32;

        // A zero (or non-finite) baseline carries no trend information; saying
        // "stable" is honest, dividing by it would produce inf/NaN.
        if !older_avg.is_finite() || older_avg.abs() < f32::EPSILON {
            return PerformanceTrend::Stable;
        }
        let change_ratio = (recent_avg - older_avg) / older_avg;

        if change_ratio > 0.05 {
            PerformanceTrend::Improving
        } else if change_ratio < -0.05 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    fn identify_bottlenecks(&self, metrics: &[PerformanceMetrics]) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        if metrics.is_empty() {
            return bottlenecks;
        }

        // Check GPU utilization. Samples without recorded telemetry carry empty
        // vectors and are therefore skipped rather than reported as "0% used".
        for m in metrics.iter() {
            for (gpu_id, &util) in m.gpu_utilization.iter().enumerate() {
                if util < 0.7 {
                    bottlenecks.push(Bottleneck::LowGpuUtilization {
                        gpu_id,
                        utilization: util,
                    });
                }
            }
        }

        // Check communication overhead
        let avg_comm =
            metrics.iter().map(|m| m.communication_overhead).sum::<f32>() / metrics.len() as f32;
        if avg_comm > 0.3 {
            bottlenecks.push(Bottleneck::HighCommunicationOverhead { overhead: avg_comm });
        }

        // Check memory usage
        for m in metrics {
            for (gpu_id, &memory) in m.memory_usage.iter().enumerate() {
                if memory > 0.95 {
                    bottlenecks.push(Bottleneck::HighMemoryUsage {
                        gpu_id,
                        usage: memory,
                    });
                }
            }
        }

        bottlenecks
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub average_throughput: f32,
    pub average_gpu_utilization: f32,
    pub average_communication_overhead: f32,
    pub performance_trend: PerformanceTrend,
    pub bottleneck_analysis: Vec<Bottleneck>,
}

impl Default for PerformanceAnalysis {
    fn default() -> Self {
        Self {
            average_throughput: 0.0,
            average_gpu_utilization: 0.0,
            average_communication_overhead: 0.0,
            performance_trend: PerformanceTrend::Stable,
            bottleneck_analysis: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

#[derive(Debug, Clone)]
pub enum Bottleneck {
    LowGpuUtilization { gpu_id: usize, utilization: f32 },
    HighCommunicationOverhead { overhead: f32 },
    HighMemoryUsage { gpu_id: usize, usage: f32 },
    InsufficientBandwidth { bandwidth_mbps: f32 },
}

/// Throughput tracking utility
pub struct ThroughputTracker {
    sample_count: usize,
    start_time: Instant,
    last_reset: Instant,
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputTracker {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            sample_count: 0,
            start_time: now,
            last_reset: now,
        }
    }

    pub fn record_samples(&mut self, count: usize) {
        self.sample_count += count;
    }

    pub fn calculate_throughput(&self) -> f32 {
        let elapsed = self.last_reset.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.sample_count as f32 / elapsed
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.last_reset = Instant::now();
    }
}

pub mod compression;

pub use compression::{CompressedData, CompressedGradient, CompressionStats, GradientCompressor};

/// Dynamic batching for optimal GPU utilization
pub struct DynamicBatcher {
    config: DynamicBatchingConfig,
    current_batch_sizes: Vec<usize>,
    utilization_history: Vec<Vec<f32>>,
    adjustment_counter: usize,
}

impl DynamicBatcher {
    pub fn new(config: DynamicBatchingConfig, num_gpus: usize) -> Self {
        let current_batch_sizes = vec![config.initial_batch_size; num_gpus];
        Self {
            config,
            current_batch_sizes,
            utilization_history: Vec::new(),
            adjustment_counter: 0,
        }
    }

    pub fn get_batch_sizes(&self) -> &[usize] {
        &self.current_batch_sizes
    }

    pub fn update_batch_sizes(&mut self, gpu_utilizations: &[f32]) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        self.utilization_history.push(gpu_utilizations.to_vec());
        self.adjustment_counter += 1;

        if self.adjustment_counter < self.config.adjustment_frequency {
            return Ok(false);
        }

        // Reset counter
        self.adjustment_counter = 0;

        // Calculate average utilization for each GPU
        let avg_utilizations = self.calculate_average_utilizations();
        let mut adjusted = false;

        // The history may carry more entries than this batcher has devices (a
        // caller can pass a longer slice); indexing beyond `current_batch_sizes`
        // would panic, so the shorter of the two bounds the loop.
        let tracked = avg_utilizations.len().min(self.current_batch_sizes.len());
        for (gpu_id, &avg_util) in avg_utilizations.iter().enumerate().take(tracked) {
            let current_batch = self.current_batch_sizes[gpu_id];
            let new_batch = if avg_util < self.config.target_utilization - 0.05 {
                // Utilization too low - increase batch size
                (current_batch + 8).min(self.config.max_batch_size)
            } else if avg_util > self.config.target_utilization + 0.05 {
                // Utilization too high - decrease batch size
                (current_batch.saturating_sub(8)).max(self.config.min_batch_size)
            } else {
                current_batch
            };

            if new_batch != current_batch {
                self.current_batch_sizes[gpu_id] = new_batch;
                adjusted = true;

                log::debug!(
                    "GPU {}: adjusted batch size {} -> {} (utilization: {:.1}%)",
                    gpu_id,
                    current_batch,
                    new_batch,
                    avg_util * 100.0
                );
            }
        }

        // Clear old history
        if self.utilization_history.len() > 1000 {
            self.utilization_history.drain(0..500);
        }

        Ok(adjusted)
    }

    fn calculate_average_utilizations(&self) -> Vec<f32> {
        if self.utilization_history.is_empty() {
            return vec![0.0; self.current_batch_sizes.len()];
        }

        let num_gpus = self.current_batch_sizes.len();
        let mut sums = vec![0.0; num_gpus];
        let mut counts = vec![0; num_gpus];

        for utilizations in &self.utilization_history {
            for (i, &util) in utilizations.iter().enumerate() {
                if i < num_gpus {
                    sums[i] += util;
                    counts[i] += 1;
                }
            }
        }

        sums.into_iter()
            .zip(counts)
            .map(|(sum, count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect()
    }
}

/// Callback that attempts to bring the job back after a node loss.
type RecoveryPolicy = Box<dyn FnMut(usize) -> Result<bool> + Send>;

/// Fault tolerance handler for robust distributed training
pub struct FaultHandler {
    config: FaultToleranceConfig,
    failed_nodes: Vec<usize>,
    checkpoint_manager: CheckpointManager,
    heartbeat_tracker: HeartbeatTracker,
    recovery_policy: Option<RecoveryPolicy>,
}

impl std::fmt::Debug for FaultHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaultHandler")
            .field("config", &self.config)
            .field("failed_nodes", &self.failed_nodes)
            .field("has_recovery_policy", &self.recovery_policy.is_some())
            .finish()
    }
}

impl FaultHandler {
    pub fn new(config: FaultToleranceConfig) -> Self {
        let checkpoint_frequency = config.checkpoint_frequency;
        let heartbeat_interval = config.heartbeat_interval;

        Self {
            config,
            failed_nodes: Vec::new(),
            checkpoint_manager: CheckpointManager::new(checkpoint_frequency),
            heartbeat_tracker: HeartbeatTracker::new(heartbeat_interval),
            recovery_policy: None,
        }
    }

    pub fn should_checkpoint(&self, step: usize) -> bool {
        step.is_multiple_of(self.config.checkpoint_frequency)
    }

    pub fn handle_node_failure(&mut self, node_id: usize) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        self.failed_nodes.push(node_id);
        log::warn!("node {} failed, attempting recovery", node_id);

        if self.config.auto_replacement {
            // Attempt to restore from checkpoint and continue training
            self.recover_from_failure(node_id)
        } else {
            Ok(false)
        }
    }

    /// Attempt to recover from the loss of `node_id`.
    ///
    /// Recovery means re-forming the communicator over the surviving nodes,
    /// restoring the latest checkpoint and redistributing the workload — none
    /// of which this handler can do on its own: it owns neither the process
    /// group nor the model state. It therefore either delegates to the recovery
    /// policy the embedder installed with
    /// [`FaultHandler::set_recovery_policy`], or reports
    /// [`TrustformersError::not_implemented`]. Returning `Ok(true)` without
    /// having recovered anything would tell the training loop it is safe to
    /// continue on a broken communicator.
    fn recover_from_failure(&mut self, node_id: usize) -> Result<bool> {
        match self.recovery_policy.as_mut() {
            Some(policy) => {
                let recovered = policy(node_id)?;
                if recovered {
                    log::info!("recovery policy reported node {node_id} recovered");
                } else {
                    log::warn!("recovery policy could not recover node {node_id}");
                }
                Ok(recovered)
            },
            None => Err(TrustformersError::not_implemented(
                "automatic node recovery: FaultHandler owns neither the process group nor the \
                 model state, so it cannot re-form the communicator or reload a checkpoint. \
                 Install a policy with FaultHandler::set_recovery_policy, or disable \
                 FaultToleranceConfig::auto_replacement and handle the failure in the training \
                 loop"
                    .to_string(),
            )),
        }
    }

    /// Install the callback invoked when a node fails and
    /// [`FaultToleranceConfig::auto_replacement`] is enabled.
    ///
    /// The callback receives the failed node id and returns whether training can
    /// continue.
    pub fn set_recovery_policy<F>(&mut self, policy: F)
    where
        F: FnMut(usize) -> Result<bool> + Send + 'static,
    {
        self.recovery_policy = Some(Box::new(policy));
    }
}

/// Checkpoint management for fault tolerance
pub struct CheckpointManager {
    frequency: usize,
    last_checkpoint: usize,
}

impl CheckpointManager {
    pub fn new(frequency: usize) -> Self {
        Self {
            frequency,
            last_checkpoint: 0,
        }
    }

    pub fn should_save(&self, step: usize) -> bool {
        step - self.last_checkpoint >= self.frequency
    }
}

/// Heartbeat tracking for node health monitoring
pub struct HeartbeatTracker {
    interval: Duration,
    last_heartbeat: HashMap<usize, Instant>,
}

impl HeartbeatTracker {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_heartbeat: HashMap::new(),
        }
    }

    pub fn record_heartbeat(&mut self, node_id: usize) {
        self.last_heartbeat.insert(node_id, Instant::now());
    }

    pub fn check_failed_nodes(&self) -> Vec<usize> {
        let now = Instant::now();
        self.last_heartbeat
            .iter()
            .filter_map(|(&node_id, &last_time)| {
                if now - last_time > self.interval * 3 {
                    // Allow 3x interval before marking as failed
                    Some(node_id)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<T: Optimizer + StatefulOptimizer + Clone> EnhancedDistributedTrainer<T> {
    /// Create new enhanced distributed trainer
    pub fn new(config: DistributedConfig, optimizer: T) -> Result<Self> {
        // Initialize GPU contexts
        let gpu_contexts = config
            .gpu_ids
            .iter()
            .map(|&id| {
                Arc::new(GpuContext {
                    device_id: id,
                    // No telemetry until the embedder records some.
                    memory_usage: Arc::new(Mutex::new(None)),
                    utilization: Arc::new(Mutex::new(None)),
                    temperature: Arc::new(Mutex::new(None)),
                    communication_bandwidth: Arc::new(Mutex::new(None)),
                })
            })
            .collect();

        // Create multi-node trainer if needed
        let multi_node_trainer = if config.num_gpus > 1 {
            let multi_config = MultiNodeConfig {
                num_nodes: 1,
                devices_per_node: config.num_gpus,
                node_rank: 0,
                local_rank: 0,
                global_rank: 0,
                zero_config: Default::default(),
                gradient_compression: config.compression.enabled,
                comm_backend: config.backend,
                overlap_comm_compute: true,
                gradient_bucket_size_mb: 25,
            };
            Some(MultiNodeTrainer::new(multi_config, optimizer.clone())?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            optimizer,
            multi_node_trainer,
            performance_monitor: PerformanceMonitor::new(config.monitoring),
            gradient_compressor: GradientCompressor::new(config.compression),
            dynamic_batcher: DynamicBatcher::new(config.dynamic_batching, config.num_gpus),
            fault_handler: FaultHandler::new(config.fault_tolerance),
            step_count: 0,
            start_time: Instant::now(),
            gpu_contexts,
            parameter_registry: HashMap::new(),
            reduced_gradients: HashMap::new(),
        })
    }

    /// Register model parameters for distributed training
    pub fn register_model(&mut self, parameters: HashMap<String, Tensor>) -> Result<()> {
        // Register parameters with multi-node trainer if available
        if let Some(ref mut trainer) = self.multi_node_trainer {
            trainer.register_parameters(parameters.clone())?;
        }

        // Build parameter registry
        for (name, tensor) in parameters {
            let param_info = ParameterInfo {
                name: name.clone(),
                shape: tensor.shape().to_vec(),
                size: tensor.shape().iter().product(),
                device_id: 0, // Simplified device assignment
                is_sharded: false,
            };
            self.parameter_registry.insert(name, param_info);
        }

        log::info!(
            "registered {} parameters for distributed training",
            self.parameter_registry.len()
        );
        Ok(())
    }

    /// Perform one training step with enhanced distributed optimizations.
    ///
    /// The step compresses `gradients` with the configured codec, reduces them
    /// across the multi-node group when one is configured, and decompresses the
    /// result. The reduced gradients are retained; because this trainer holds
    /// parameter *metadata* only (see [`ParameterInfo`]), the caller — who owns
    /// the parameter tensors — applies them with
    /// [`EnhancedDistributedTrainer::apply_reduced_gradients`] or reads them via
    /// [`EnhancedDistributedTrainer::take_reduced_gradients`].
    ///
    /// Dynamic batch sizing runs only when device telemetry has been recorded
    /// for every device (see
    /// [`EnhancedDistributedTrainer::record_gpu_telemetry`]); without it there
    /// is nothing to base a resize on and the batch sizes are left alone.
    pub fn train_step(&mut self, gradients: HashMap<String, Tensor>) -> Result<TrainingStepResult> {
        let step_start = Instant::now();

        // Compress gradients
        let compressed_gradients = self.gradient_compressor.compress_gradients(&gradients)?;

        // Update dynamic batch sizes when real utilization samples exist.
        let batch_size_adjusted = match self.recorded_gpu_utilizations()? {
            Some(utilizations) => self.dynamic_batcher.update_batch_sizes(&utilizations)?,
            None => {
                log::debug!(
                    "skipping dynamic batch sizing: no GPU telemetry recorded (call \
                     EnhancedDistributedTrainer::record_gpu_telemetry)"
                );
                false
            },
        };

        // Reduce and decompress, then keep the result for the parameter owner.
        let mut decompressed: HashMap<String, Tensor> =
            HashMap::with_capacity(compressed_gradients.len());
        for (name, compressed) in &compressed_gradients {
            decompressed.insert(name.clone(), compressed.decompress()?);
        }

        if let Some(ref mut trainer) = self.multi_node_trainer {
            trainer.update_gradients(decompressed.clone())?;
            trainer.optimizer_step()?;
        }
        self.reduced_gradients = decompressed;

        self.step_count += 1;

        // Signal that a checkpoint is due. This trainer holds parameter
        // *metadata* only (see `parameter_registry`), so it cannot serialize
        // model state itself; the owner drives
        // `SmartCheckpointManager::create_checkpoint` with the real tensors.
        // Claiming "checkpoint saved" here would be a fabrication.
        if self.fault_handler.should_checkpoint(self.step_count) {
            log::info!(
                "checkpoint interval reached at step {}; call \
                 SmartCheckpointManager::create_checkpoint with the model state",
                self.step_count
            );
        }

        // Collect performance metrics
        let performance_metrics = self.performance_monitor.collect_metrics(&self.gpu_contexts)?;

        let step_time = step_start.elapsed();

        Ok(TrainingStepResult {
            step: self.step_count,
            step_time,
            compression_ratio: self
                .gradient_compressor
                .get_compression_stats()
                .average_compression_ratio,
            batch_size_adjusted,
            performance_metrics,
        })
    }

    /// Record a real telemetry reading for one device.
    ///
    /// This crate cannot read a GPU itself: NVML and ROCm SMI are C libraries
    /// and the default build is pure Rust. Rather than inventing plausible
    /// numbers, every device metric stays `None` until the embedder calls this
    /// with values it obtained from the platform. Metrics gathered here drive
    /// dynamic batch sizing, bottleneck detection and auto-scaling, so a
    /// fabricated sample would propagate into real training decisions.
    ///
    /// # Errors
    ///
    /// Fails when `device_id` is not one of the configured devices or when the
    /// sample is out of range (see [`GpuTelemetrySample`]).
    pub fn record_gpu_telemetry(
        &mut self,
        device_id: usize,
        sample: GpuTelemetrySample,
    ) -> Result<()> {
        sample.validate(device_id)?;

        let ctx =
            self.gpu_contexts.iter().find(|ctx| ctx.device_id == device_id).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "device {device_id} is not part of this trainer; configured devices: {:?}",
                    self.gpu_contexts.iter().map(|ctx| ctx.device_id).collect::<Vec<_>>()
                ))
            })?;

        let store = |slot: &Arc<Mutex<Option<f32>>>, value: f32, label: &str| -> Result<()> {
            let mut guard = slot.lock().map_err(|_| {
                TrustformersError::lock_error(format!("GPU context {label} mutex poisoned"))
            })?;
            *guard = Some(value);
            Ok(())
        };

        store(&ctx.utilization, sample.utilization, "utilization")?;
        store(&ctx.memory_usage, sample.memory_usage, "memory_usage")?;
        store(&ctx.temperature, sample.temperature_celsius, "temperature")?;
        store(
            &ctx.communication_bandwidth,
            sample.communication_bandwidth_mb_s,
            "communication_bandwidth",
        )?;
        Ok(())
    }

    /// Utilization of every configured device, or `None` when at least one
    /// device has never been sampled.
    fn recorded_gpu_utilizations(&self) -> Result<Option<Vec<f32>>> {
        let mut values = Vec::with_capacity(self.gpu_contexts.len());
        for ctx in &self.gpu_contexts {
            let guard = ctx.utilization.lock().map_err(|_| {
                TrustformersError::lock_error("GPU context utilization mutex poisoned".to_string())
            })?;
            match *guard {
                Some(value) => values.push(value),
                None => return Ok(None),
            }
        }
        Ok(if values.is_empty() { None } else { Some(values) })
    }

    /// Take the gradients produced by the last
    /// [`EnhancedDistributedTrainer::train_step`], leaving the trainer empty.
    pub fn take_reduced_gradients(&mut self) -> HashMap<String, Tensor> {
        std::mem::take(&mut self.reduced_gradients)
    }

    /// Borrow the gradients produced by the last
    /// [`EnhancedDistributedTrainer::train_step`].
    pub fn reduced_gradients(&self) -> &HashMap<String, Tensor> {
        &self.reduced_gradients
    }

    /// Apply the gradients from the last [`EnhancedDistributedTrainer::train_step`]
    /// to `parameters` with this trainer's optimizer.
    ///
    /// Parameters are visited in sorted-name order so every rank performs the
    /// same update sequence. Returns the number of parameters updated.
    ///
    /// # Errors
    ///
    /// Fails when a gradient has no matching parameter — a silent skip would
    /// leave part of the model un-trained without any signal.
    pub fn apply_reduced_gradients(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
    ) -> Result<usize> {
        let mut names: Vec<String> = self.reduced_gradients.keys().cloned().collect();
        names.sort();

        for name in &names {
            let gradient = self.reduced_gradients.get(name).ok_or_else(|| {
                TrustformersError::invalid_input(format!("gradient `{name}` vanished"))
            })?;
            let parameter = parameters.get_mut(name).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "no parameter named `{name}` to apply its gradient to"
                ))
            })?;
            self.optimizer.update(parameter, gradient)?;
        }
        self.optimizer.step();
        Ok(names.len())
    }

    /// Get comprehensive training statistics.
    ///
    /// `gpu_utilization` and `memory_usage` are empty when no device telemetry
    /// has been recorded; see [`EnhancedDistributedTrainer::record_gpu_telemetry`].
    pub fn get_training_stats(&self) -> DistributedTrainingStats {
        let performance_analysis = self.performance_monitor.analyze_performance_trends();
        let compression_stats = self.gradient_compressor.get_compression_stats();

        let collect_known = |select: fn(&GpuContext) -> &Arc<Mutex<Option<f32>>>| -> Vec<f32> {
            let mut values = Vec::with_capacity(self.gpu_contexts.len());
            for ctx in &self.gpu_contexts {
                match *select(ctx).lock().unwrap_or_else(|poisoned| poisoned.into_inner()) {
                    Some(value) => values.push(value),
                    None => return Vec::new(),
                }
            }
            values
        };

        let memory_usage: Vec<f32> = collect_known(|ctx| &ctx.memory_usage);
        let gpu_utilization: Vec<f32> = collect_known(|ctx| &ctx.utilization);

        DistributedTrainingStats {
            total_steps: self.step_count,
            training_time: self.start_time.elapsed(),
            average_throughput: performance_analysis.average_throughput,
            gpu_utilization,
            memory_usage,
            compression_ratio: compression_stats.average_compression_ratio,
            communication_overhead: performance_analysis.average_communication_overhead,
            batch_sizes: self.dynamic_batcher.get_batch_sizes().to_vec(),
            failed_nodes: self.fault_handler.failed_nodes.clone(),
            performance_trend: performance_analysis.performance_trend,
            bottlenecks: performance_analysis.bottleneck_analysis,
        }
    }

    /// Render the training statistics as a human-readable report.
    ///
    /// Prefer this over [`Self::print_training_stats`] inside libraries: it
    /// returns the text instead of writing to stdout.
    pub fn training_stats_report(&self) -> String {
        use std::fmt::Write as _;

        let stats = self.get_training_stats();
        let mut report = String::new();

        // Writing into a String is infallible, so the results are discarded
        // deliberately rather than unwrapped.
        let _ = writeln!(report, "Enhanced distributed training statistics");
        let _ = writeln!(report, "Training progress:");
        let _ = writeln!(report, "  total steps: {}", stats.total_steps);
        let _ = writeln!(
            report,
            "  training time: {:.2} minutes",
            stats.training_time.as_secs_f32() / 60.0
        );
        let _ = writeln!(
            report,
            "  average throughput: {:.1} samples/sec",
            stats.average_throughput
        );

        let _ = writeln!(report, "GPU performance:");
        for (index, (&utilization, &memory)) in
            stats.gpu_utilization.iter().zip(&stats.memory_usage).enumerate()
        {
            let _ = writeln!(
                report,
                "  GPU {}: utilization {:.1}%, memory {:.1}%",
                index,
                utilization * 100.0,
                memory * 100.0
            );
        }

        let _ = writeln!(report, "Optimization metrics:");
        let _ = writeln!(
            report,
            "  compression ratio: {:.1}%",
            stats.compression_ratio * 100.0
        );
        let _ = writeln!(
            report,
            "  communication overhead: {:.1}%",
            stats.communication_overhead * 100.0
        );
        let _ = writeln!(report, "  performance trend: {:?}", stats.performance_trend);

        if !stats.bottlenecks.is_empty() {
            let _ = writeln!(report, "Identified bottlenecks:");
            for bottleneck in &stats.bottlenecks {
                match bottleneck {
                    Bottleneck::LowGpuUtilization {
                        gpu_id,
                        utilization,
                    } => {
                        let _ = writeln!(
                            report,
                            "  - GPU {} low utilization: {:.1}%",
                            gpu_id,
                            utilization * 100.0
                        );
                    },
                    Bottleneck::HighCommunicationOverhead { overhead } => {
                        let _ = writeln!(
                            report,
                            "  - high communication overhead: {:.1}%",
                            overhead * 100.0
                        );
                    },
                    Bottleneck::HighMemoryUsage { gpu_id, usage } => {
                        let _ = writeln!(
                            report,
                            "  - GPU {} high memory usage: {:.1}%",
                            gpu_id,
                            usage * 100.0
                        );
                    },
                    Bottleneck::InsufficientBandwidth { bandwidth_mbps } => {
                        let _ = writeln!(
                            report,
                            "  - insufficient bandwidth: {:.0} Mbps",
                            bandwidth_mbps
                        );
                    },
                }
            }
        }

        report
    }

    /// Write [`Self::training_stats_report`] to stdout.
    ///
    /// This is an explicit, caller-initiated escape hatch for binaries and
    /// examples; nothing on the training path writes to stdout. Library callers
    /// should prefer [`Self::log_training_stats`], which routes the same report
    /// through the `log` facade so the host application controls the sink.
    pub fn print_training_stats(&self) {
        println!("{}", self.training_stats_report());
    }

    /// Emit [`Self::training_stats_report`] at `info` level through the `log`
    /// facade.
    pub fn log_training_stats(&self) {
        log::info!("{}", self.training_stats_report());
    }

    /// Whether the fault handler considers a checkpoint due at the current
    /// step. The caller owns the model state and drives
    /// [`crate::advanced_distributed_features::SmartCheckpointManager`].
    pub fn checkpoint_due(&self) -> bool {
        self.fault_handler.should_checkpoint(self.step_count)
    }

    /// Optimize hyperparameters for the current distributed setup.
    ///
    /// # Errors
    ///
    /// Distributed-aware hyperparameter optimization is **not implemented**.
    /// The crate ships [`crate::hyperparameter_tuning`], but wiring it here
    /// requires an evaluation callback (a way to run a trial and score it) that
    /// this trainer does not have. Rather than returning an unmodified clone of
    /// the optimizer while reporting success, this returns
    /// [`TrustformersError`] describing what is missing whenever auto-tuning is
    /// requested.
    ///
    /// With `config.monitoring.auto_tuning == false` the call is a no-op and
    /// returns the current optimizer unchanged, which is honest: no
    /// optimization was requested and none was performed.
    pub fn optimize_hyperparameters(&mut self) -> Result<T> {
        if self.config.monitoring.auto_tuning {
            return Err(TrustformersError::not_implemented(
                "distributed hyperparameter optimization: \
                 EnhancedDistributedTrainer has no trial-evaluation callback, so no search can \
                 be run. Drive crate::hyperparameter_tuning::HyperparameterTuner directly with \
                 your own objective function, or disable config.monitoring.auto_tuning"
                    .to_string(),
            ));
        }

        Ok(self.optimizer.clone())
    }
}

/// Result of a training step
#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    pub step: usize,
    pub step_time: Duration,
    pub compression_ratio: f32,
    pub batch_size_adjusted: bool,
    pub performance_metrics: PerformanceMetrics,
}

/// Comprehensive distributed training statistics
#[derive(Debug, Clone)]
pub struct DistributedTrainingStats {
    pub total_steps: usize,
    pub training_time: Duration,
    pub average_throughput: f32,
    pub gpu_utilization: Vec<f32>,
    pub memory_usage: Vec<f32>,
    pub compression_ratio: f32,
    pub communication_overhead: f32,
    pub batch_sizes: Vec<usize>,
    pub failed_nodes: Vec<usize>,
    pub performance_trend: PerformanceTrend,
    pub bottlenecks: Vec<Bottleneck>,
}

// Extension trait for Averaged Adam distributed training
impl AveragedAdam {
    /// Create Averaged Adam configuration optimized for distributed training
    pub fn for_distributed_training() -> Self {
        let config = AveragedAdamConfig {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            averaging_coeff: 0.9999, // Higher averaging for distributed stability
            use_averaged: true,
            averaging_warmup: 1000, // Longer warmup for distributed training
        };

        AveragedAdam::new(
            config.lr,
            config.betas,
            config.eps,
            config.weight_decay,
            config.averaging_coeff,
        )
    }

    /// Create configuration for large-scale distributed training
    pub fn for_large_scale_distributed(world_size: usize) -> Self {
        // Adjust hyperparameters based on world size
        let lr_scale = (world_size as f32).sqrt();
        let config = AveragedAdamConfig {
            lr: 1e-3 * lr_scale,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01 / lr_scale, // Reduce weight decay for larger batch sizes
            averaging_coeff: 1.0 - (1.0 - 0.999) / world_size as f32, // Adjust averaging
            use_averaged: true,
            averaging_warmup: 1000 + world_size * 10, // Scale warmup with world size
        };

        AveragedAdam::new(
            config.lr,
            config.betas,
            config.eps,
            config.weight_decay,
            config.averaging_coeff,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam::Adam;

    #[test]
    fn test_distributed_config_creation() {
        let config = DistributedConfig::new()
            .with_gpus(4)
            .with_gradient_compression(CompressionType::TopK { k: 1000 })
            .with_dynamic_batching(true)
            .with_fault_tolerance(true);

        assert_eq!(config.num_gpus, 4);
        assert_eq!(config.gpu_ids, vec![0, 1, 2, 3]);
        assert!(config.compression.enabled);
        assert!(config.dynamic_batching.enabled);
        assert!(config.fault_tolerance.enabled);
    }

    #[test]
    fn test_gradient_compression() {
        let config = CompressionConfig {
            enabled: true,
            algorithm: CompressionType::TopK { k: 5 },
            target_ratio: 0.1,
            error_feedback: false,
            adaptive_threshold: 0.01,
        };

        let mut compressor = GradientCompressor::new(config);
        let gradient = Tensor::ones(&[10]).expect("Failed to create tensor");
        let mut gradients = HashMap::new();
        gradients.insert("test".to_string(), gradient);

        let compressed =
            compressor.compress_gradients(&gradients).expect("Operation failed in test");
        assert!(compressed.contains_key("test"));

        let compressed_grad = &compressed["test"];
        assert!(compressed_grad.compression_ratio <= 1.0);
    }

    #[test]
    fn test_performance_monitor() {
        let config = MonitoringConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        let gpu_contexts = vec![Arc::new(GpuContext {
            device_id: 0,
            memory_usage: Arc::new(Mutex::new(Some(0.8))),
            utilization: Arc::new(Mutex::new(Some(0.9))),
            temperature: Arc::new(Mutex::new(Some(75.0))),
            communication_bandwidth: Arc::new(Mutex::new(Some(1000.0))),
        })];

        let metrics = monitor.collect_metrics(&gpu_contexts).expect("Operation failed in test");
        assert_eq!(metrics.gpu_utilization, vec![0.9]);
        assert_eq!(metrics.memory_usage, vec![0.8]);
        assert_eq!(metrics.bandwidth_utilization, 1000.0);
    }

    /// Regression: an earlier revision filled every device metric with
    /// `0.8 + random()` on each step, so callers received invented telemetry.
    /// Without a recorded sample the metrics must now be *absent*, never
    /// plausible-looking noise.
    #[test]
    fn unsampled_devices_report_no_telemetry() {
        let mut monitor = PerformanceMonitor::new(MonitoringConfig::default());
        let gpu_contexts = vec![Arc::new(GpuContext {
            device_id: 0,
            memory_usage: Arc::new(Mutex::new(None)),
            utilization: Arc::new(Mutex::new(None)),
            temperature: Arc::new(Mutex::new(None)),
            communication_bandwidth: Arc::new(Mutex::new(None)),
        })];

        let metrics = monitor.collect_metrics(&gpu_contexts).expect("collect must succeed in test");
        assert!(
            metrics.gpu_utilization.is_empty(),
            "unknown utilization must stay empty, got {:?}",
            metrics.gpu_utilization
        );
        assert!(metrics.memory_usage.is_empty());
        assert_eq!(metrics.bandwidth_utilization, 0.0);
    }

    #[test]
    fn train_step_does_not_invent_gpu_telemetry() {
        let config = DistributedConfig::new().with_gpus(1);
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut trainer =
            EnhancedDistributedTrainer::new(config, optimizer).expect("trainer must build in test");

        let mut gradients = HashMap::new();
        gradients.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.1f32, -0.2, 0.3], &[3]).expect("tensor must build in test"),
        );

        let result = trainer.train_step(gradients).expect("train step must succeed in test");
        assert!(
            result.performance_metrics.gpu_utilization.is_empty(),
            "no telemetry was recorded, so none may be reported: {:?}",
            result.performance_metrics.gpu_utilization
        );
        assert!(!result.batch_size_adjusted);
    }

    #[test]
    fn recorded_telemetry_is_reported_verbatim() {
        let config = DistributedConfig::new().with_gpu_ids(vec![3]);
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut trainer =
            EnhancedDistributedTrainer::new(config, optimizer).expect("trainer must build in test");

        trainer
            .record_gpu_telemetry(
                3,
                GpuTelemetrySample {
                    utilization: 0.42,
                    memory_usage: 0.17,
                    temperature_celsius: 61.5,
                    communication_bandwidth_mb_s: 512.0,
                },
            )
            .expect("recording telemetry must succeed in test");

        let stats = trainer.get_training_stats();
        assert_eq!(stats.gpu_utilization, vec![0.42]);
        assert_eq!(stats.memory_usage, vec![0.17]);

        // An unknown device is rejected instead of being silently created.
        assert!(trainer
            .record_gpu_telemetry(
                9,
                GpuTelemetrySample {
                    utilization: 0.5,
                    memory_usage: 0.5,
                    temperature_celsius: 50.0,
                    communication_bandwidth_mb_s: 1.0,
                },
            )
            .is_err());

        // Out-of-range samples are rejected.
        assert!(trainer
            .record_gpu_telemetry(
                3,
                GpuTelemetrySample {
                    utilization: 1.5,
                    memory_usage: 0.5,
                    temperature_celsius: 50.0,
                    communication_bandwidth_mb_s: 1.0,
                },
            )
            .is_err());
    }

    /// Regression: the single-device branch of `train_step` used to decompress
    /// each gradient into `_grad` and drop it, so training was a no-op.
    #[test]
    fn train_step_gradients_reach_the_optimizer() {
        let config = DistributedConfig::new().with_gpus(1);
        let optimizer = Adam::new(0.1, (0.9, 0.999), 1e-8, 0.0);
        let mut trainer =
            EnhancedDistributedTrainer::new(config, optimizer).expect("trainer must build in test");

        let mut parameters = HashMap::new();
        parameters.insert(
            "w".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).expect("tensor must build in test"),
        );
        let before = parameters["w"].to_vec_f32().expect("tensor read must succeed in test");

        let mut gradients = HashMap::new();
        gradients.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.5f32, 0.5, 0.5], &[3]).expect("tensor must build in test"),
        );

        trainer.train_step(gradients).expect("train step must succeed in test");
        assert_eq!(trainer.reduced_gradients().len(), 1);

        let updated = trainer
            .apply_reduced_gradients(&mut parameters)
            .expect("applying gradients must succeed in test");
        assert_eq!(updated, 1);

        let after = parameters["w"].to_vec_f32().expect("tensor read must succeed in test");
        assert_ne!(before, after, "a positive gradient must move the parameter");
        for (old, new) in before.iter().zip(&after) {
            assert!(
                new < old,
                "descent must decrease each weight: {old} -> {new}"
            );
        }
    }

    /// Regression: `recover_from_failure` used to log and return `Ok(true)`,
    /// telling the training loop the job had recovered when nothing happened.
    #[test]
    fn node_recovery_requires_a_real_policy() {
        let mut handler = FaultHandler::new(FaultToleranceConfig {
            enabled: true,
            checkpoint_frequency: 10,
            max_retries: 3,
            heartbeat_interval: Duration::from_secs(1),
            auto_replacement: true,
        });

        assert!(
            handler.handle_node_failure(2).is_err(),
            "no recovery policy is installed, so recovery cannot be claimed"
        );

        handler.set_recovery_policy(|node_id| Ok(node_id != 7));
        assert!(handler.handle_node_failure(2).expect("policy must run in test"));
        assert!(!handler.handle_node_failure(7).expect("policy must run in test"));
    }

    #[test]
    fn test_dynamic_batcher() {
        let config = DynamicBatchingConfig {
            enabled: true,
            initial_batch_size: 32,
            min_batch_size: 8,
            max_batch_size: 128,
            target_utilization: 0.8,
            adjustment_frequency: 1, // Adjust every step for testing
        };

        let mut batcher = DynamicBatcher::new(config, 2);
        assert_eq!(batcher.get_batch_sizes(), &[32, 32]);

        // Simulate low utilization
        let low_utilization = vec![0.5, 0.6];
        let _adjusted =
            batcher.update_batch_sizes(&low_utilization).expect("Operation failed in test");

        // Should increase batch sizes due to low utilization
        // Note: May not adjust on first call due to frequency requirements
        let final_sizes = batcher.get_batch_sizes();
        assert_eq!(final_sizes.len(), 2);
    }

    #[test]
    fn test_averaged_adam_distributed_config() {
        let _optimizer = AveragedAdam::for_distributed_training();
        // Test that it creates a valid configuration
        // In actual implementation, would verify specific parameters
    }

    #[test]
    fn test_enhanced_distributed_trainer_creation() {
        let config = DistributedConfig::new().with_gpus(1);
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        // A single-device trainer needs no external runtime, so construction
        // must succeed unconditionally; swallowing the error here would hide a
        // real regression.
        let trainer = EnhancedDistributedTrainer::new(config, optimizer)
            .expect("single-device trainer must build in test");
        assert_eq!(trainer.config.num_gpus, 1);
        assert_eq!(trainer.step_count, 0);
    }

    #[test]
    fn optimize_hyperparameters_reports_not_implemented_instead_of_a_success_banner() {
        // The previous implementation printed "✅ Hyperparameter optimization
        // completed (placeholder)" and returned a clone of the input optimizer.
        let mut config = DistributedConfig::new().with_gpus(1);
        config.monitoring.auto_tuning = true;
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)
            .expect("single-device trainer must build in test");

        let Err(error) = trainer.optimize_hyperparameters() else {
            panic!("auto-tuning must not report success without running a search in test");
        };
        let message = error.to_string();
        assert!(
            message.contains("hyperparameter") && message.contains("not implemented")
                || message.contains("HyperparameterTuner"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn optimize_hyperparameters_is_a_no_op_when_auto_tuning_is_off() {
        let mut config = DistributedConfig::new().with_gpus(1);
        config.monitoring.auto_tuning = false;
        let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)
            .expect("single-device trainer must build in test");

        // No optimization was requested, so returning the current optimizer
        // unchanged is the honest answer.
        assert!(trainer.optimize_hyperparameters().is_ok());
    }
}
