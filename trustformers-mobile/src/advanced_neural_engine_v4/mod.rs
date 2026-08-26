//! Advanced Neural Engine v4 Optimization
//!
//! This module provides cutting-edge optimization techniques for Apple's latest Neural Engine
//! hardware (A17 Pro, M3 series, and newer), including advanced graph optimization,
//! dynamic compilation, and hardware-specific acceleration patterns.
//!
//! # Features
//!
//! - **Multi-Core Neural Engine Utilization**: Leverage all 16 cores in A17 Pro Neural Engine
//! - **Dynamic Graph Recompilation**: Real-time graph optimization based on runtime patterns
//! - **Advanced Memory Hierarchy Management**: Optimal usage of Neural Engine memory tiers
//! - **Precision-Aware Quantization**: Hardware-native quantization schemes (INT4, INT8, FP16)
//! - **Thermal-Aware Performance Scaling**: Dynamic performance adjustment based on thermal state
//! - **Concurrent Execution Pipeline**: Overlapped compute and memory operations
//! - **Advanced Attention Mechanisms**: Hardware-optimized attention patterns for transformers
//! - **Custom Kernel Fusion**: Complex operation fusion for maximum throughput
//!
//! # Honesty
//!
//! This module previously did not compile at all (it imported `crate::ios`,
//! `crate::coreml` and `crate::neural_engine_v3`, none of which export the
//! types it named) and every analytic it reported was a hardcoded constant:
//! attention returned a 1x1 zero tensor, execution echoed its input, and
//! `total_compilations: 100` / `cache_hit_rate: 0.85` /
//! `overall_improvement: 0.25` were literals.
//!
//! It has been rewritten around two rules:
//!
//! 1. **Attention is real.** [`AdvancedNeuralEngineV4::execute_optimized_attention`]
//!    computes genuine multi-head scaled dot-product attention — on the Metal
//!    GPU with the `metal` feature, on real CPU kernels otherwise.
//! 2. **Analytics are measured or absent.** Every figure is either derived from
//!    executions that actually happened, or reported as
//!    [`Availability::NotAvailable`] with a reason. There is no
//!    "optimization effectiveness" or "bottleneck analysis" API, because this
//!    crate has no baseline to measure improvement against and any such number
//!    would be invented.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;

/// Configuration for Neural Engine v4 optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEngineV4Config {
    /// Enable multi-core Neural Engine utilization
    pub enable_multi_core: bool,
    /// Number of Neural Engine cores to use (auto-detect if None)
    pub num_cores: Option<usize>,
    /// Dynamic graph recompilation settings
    pub dynamic_recompilation: DynamicRecompilationConfig,
    /// Memory hierarchy optimization
    pub memory_optimization: MemoryHierarchyConfig,
    /// Precision and quantization settings
    pub precision_config: PrecisionConfig,
    /// Thermal management configuration
    pub thermal_config: ThermalManagementConfig,
    /// Concurrent execution settings
    pub concurrency_config: ConcurrencyConfig,
    /// Advanced attention optimization
    pub attention_config: AttentionOptimizationConfig,
}

impl Default for NeuralEngineV4Config {
    fn default() -> Self {
        Self {
            enable_multi_core: true,
            num_cores: None, // Auto-detect
            dynamic_recompilation: DynamicRecompilationConfig::default(),
            memory_optimization: MemoryHierarchyConfig::default(),
            precision_config: PrecisionConfig::default(),
            thermal_config: ThermalManagementConfig::default(),
            concurrency_config: ConcurrencyConfig::default(),
            attention_config: AttentionOptimizationConfig::default(),
        }
    }
}

/// Dynamic graph recompilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRecompilationConfig {
    /// Enable runtime graph optimization
    pub enabled: bool,
    /// Minimum number of executions before triggering recompilation
    pub min_executions: usize,
    /// Performance improvement threshold for recompilation
    pub performance_threshold: f32,
    /// Maximum compilation time budget (ms)
    pub compilation_time_budget_ms: u64,
    /// Enable speculative compilation
    pub enable_speculative_compilation: bool,
    /// Graph analysis depth
    pub analysis_depth: usize,
}

impl Default for DynamicRecompilationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_executions: 10,
            performance_threshold: 0.05, // 5% improvement threshold
            compilation_time_budget_ms: 500,
            enable_speculative_compilation: true,
            analysis_depth: 3,
        }
    }
}

/// Memory hierarchy optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHierarchyConfig {
    /// Enable advanced memory prefetching
    pub enable_prefetching: bool,
    /// Cache tier optimization strategy
    pub cache_strategy: CacheStrategy,
    /// Memory bandwidth optimization
    pub bandwidth_optimization: BandwidthOptimization,
    /// Buffer pooling configuration
    pub buffer_pooling: BufferPoolingConfig,
}

impl Default for MemoryHierarchyConfig {
    fn default() -> Self {
        Self {
            enable_prefetching: true,
            cache_strategy: CacheStrategy::Adaptive,
            bandwidth_optimization: BandwidthOptimization::Aggressive,
            buffer_pooling: BufferPoolingConfig::default(),
        }
    }
}

/// Cache optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// Conservative caching with minimal eviction
    Conservative,
    /// Balanced caching strategy
    Balanced,
    /// Adaptive caching based on usage patterns
    Adaptive,
    /// Aggressive caching for maximum performance
    Aggressive,
}

/// Memory bandwidth optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthOptimization {
    /// Minimal bandwidth optimization
    Minimal,
    /// Balanced bandwidth usage
    Balanced,
    /// Aggressive bandwidth optimization
    Aggressive,
}

/// Buffer pooling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPoolingConfig {
    /// Enable buffer pooling
    pub enabled: bool,
    /// Maximum pool size in bytes
    pub max_pool_size_bytes: usize,
    /// Buffer alignment requirements
    pub alignment_bytes: usize,
    /// Pool growth strategy
    pub growth_strategy: PoolGrowthStrategy,
}

impl Default for BufferPoolingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_pool_size_bytes: 256 * 1024 * 1024, // 256MB
            alignment_bytes: 64,                    // 64-byte alignment for Neural Engine
            growth_strategy: PoolGrowthStrategy::Exponential,
        }
    }
}

/// Buffer pool growth strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolGrowthStrategy {
    /// Linear growth
    Linear,
    /// Exponential growth
    Exponential,
    /// Fibonacci growth
    Fibonacci,
}

/// Precision and quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Default precision for operations
    pub default_precision: NeuralEnginePrecision,
    /// Mixed precision configuration
    pub mixed_precision: MixedPrecisionConfig,
    /// Quantization settings
    pub quantization: QuantizationConfig,
    /// Sparsity exploitation settings
    pub sparsity_config: SparsityConfig,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: NeuralEnginePrecision::FP16,
            mixed_precision: MixedPrecisionConfig::default(),
            quantization: QuantizationConfig::default(),
            sparsity_config: SparsityConfig::default(),
        }
    }
}

/// Neural Engine supported precision types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralEnginePrecision {
    /// 4-bit integer quantization
    INT4,
    /// 8-bit integer quantization
    INT8,
    /// 16-bit floating point
    FP16,
    /// Mixed precision (automatic selection)
    Mixed,
}

/// Mixed precision optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Loss scaling factor
    pub loss_scale: f32,
    /// Gradient clipping threshold
    pub gradient_clip_threshold: f32,
    /// Operations to force in FP16
    pub force_fp16_ops: Vec<String>,
    /// Operations to force in FP32
    pub force_fp32_ops: Vec<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            loss_scale: 65536.0,
            gradient_clip_threshold: 1.0,
            force_fp16_ops: vec![
                "conv2d".to_string(),
                "matmul".to_string(),
                "attention".to_string(),
            ],
            force_fp32_ops: vec![
                "softmax".to_string(),
                "layer_norm".to_string(),
                "loss".to_string(),
            ],
        }
    }
}

/// Quantization configuration for Neural Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Enable adaptive quantization
    pub adaptive_quantization: bool,
    /// Per-channel vs per-tensor quantization
    pub per_channel_quantization: bool,
    /// Calibration dataset size
    pub calibration_samples: usize,
    /// Quantization-aware training settings
    pub qat_config: Option<QATConfig>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            adaptive_quantization: true,
            per_channel_quantization: true,
            calibration_samples: 1000,
            qat_config: Some(QATConfig::default()),
        }
    }
}

/// Quantization-Aware Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATConfig {
    /// QAT learning rate
    pub learning_rate: f32,
    /// QAT warmup steps
    pub warmup_steps: usize,
    /// Fake quantization noise
    pub fake_quant_noise: f32,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            warmup_steps: 1000,
            fake_quant_noise: 0.1,
        }
    }
}

/// Sparsity exploitation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityConfig {
    /// Enable structured sparsity optimization
    pub enable_structured_sparsity: bool,
    /// Enable unstructured sparsity optimization
    pub enable_unstructured_sparsity: bool,
    /// Minimum sparsity ratio for optimization
    pub min_sparsity_ratio: f32,
    /// Sparsity pattern cache size
    pub pattern_cache_size: usize,
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            enable_structured_sparsity: true,
            enable_unstructured_sparsity: true,
            min_sparsity_ratio: 0.1, // 10% sparsity threshold
            pattern_cache_size: 1000,
        }
    }
}

/// Thermal management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalManagementConfig {
    /// Enable thermal-aware performance scaling
    pub enabled: bool,
    /// Target thermal state
    pub target_thermal_state: NeuralEngineThermalState,
    /// Performance scaling strategy
    pub scaling_strategy: ThermalScalingStrategy,
    /// Temperature monitoring interval
    pub monitoring_interval_ms: u64,
    /// Emergency throttling threshold
    pub emergency_throttle_threshold: f32,
}

impl Default for ThermalManagementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_thermal_state: NeuralEngineThermalState::Fair,
            scaling_strategy: ThermalScalingStrategy::Adaptive,
            monitoring_interval_ms: 100,
            emergency_throttle_threshold: 0.5, // 50% performance reduction
        }
    }
}

/// Thermal scaling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalScalingStrategy {
    /// Linear performance scaling
    Linear,
    /// Exponential performance scaling
    Exponential,
    /// Adaptive scaling based on workload
    Adaptive,
    /// Step-wise scaling
    Stepped,
}

/// Concurrency configuration for Neural Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Enable concurrent execution
    pub enabled: bool,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Pipeline depth
    pub pipeline_depth: usize,
    /// Enable memory/compute overlap
    pub enable_memory_compute_overlap: bool,
    /// Dependency tracking strategy
    pub dependency_strategy: DependencyStrategy,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_ops: 4,
            pipeline_depth: 3,
            enable_memory_compute_overlap: true,
            dependency_strategy: DependencyStrategy::Aggressive,
        }
    }
}

/// Dependency tracking strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyStrategy {
    /// Conservative dependency tracking
    Conservative,
    /// Balanced dependency analysis
    Balanced,
    /// Aggressive dependency optimization
    Aggressive,
}

/// Attention mechanism optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionOptimizationConfig {
    /// Enable Flash Attention optimization
    pub enable_flash_attention: bool,
    /// Enable attention caching
    pub enable_attention_caching: bool,
    /// Attention head fusion strategy
    pub head_fusion_strategy: AttentionFusionStrategy,
    /// Key-value cache compression
    pub kv_cache_compression: KVCacheConfig,
    /// Attention sparsity patterns
    pub sparsity_patterns: Vec<AttentionSparsityPattern>,
}

impl Default for AttentionOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_flash_attention: true,
            enable_attention_caching: true,
            head_fusion_strategy: AttentionFusionStrategy::Adaptive,
            kv_cache_compression: KVCacheConfig::default(),
            sparsity_patterns: vec![
                AttentionSparsityPattern::LocalWindow { window_size: 128 },
                AttentionSparsityPattern::Strided { stride: 4 },
            ],
        }
    }
}

/// Attention fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionFusionStrategy {
    /// No fusion
    None,
    /// Fuse adjacent heads
    Adjacent,
    /// Adaptive fusion based on similarity
    Adaptive,
    /// Full multi-head fusion
    Full,
}

/// Key-Value cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// Enable KV cache compression
    pub enable_compression: bool,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Maximum cache size
    pub max_cache_size_mb: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_ratio: 0.5, // 50% compression
            eviction_policy: CacheEvictionPolicy::LRU,
            max_cache_size_mb: 512,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Random eviction
    Random,
    /// First In, First Out
    FIFO,
}

/// Attention sparsity patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionSparsityPattern {
    /// Local attention window
    LocalWindow { window_size: usize },
    /// Strided attention pattern
    Strided { stride: usize },
    /// Random sparse attention
    Random { sparsity_ratio: f32 },
    /// Block sparse attention
    BlockSparse { block_size: usize },
}

// ─── Runtime state ───────────────────────────────────────────────────────────

/// Device description used to size the engine.
///
/// The previous version of this module referenced an `IOSDeviceInfo` type that
/// did not exist anywhere in the crate, so it could never compile; this is a
/// self-contained replacement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuralEngineDeviceInfo {
    /// Marketing name of the device, e.g. `"iPhone 15 Pro"`.
    pub device_name: String,
    /// SoC name, e.g. `"A17 Pro"`. Drives core-count detection.
    pub chip_name: String,
    /// Reported Neural Engine generation, e.g. `"v4"`.
    pub neural_engine_version: String,
    /// Installed memory in GiB.
    pub memory_gb: usize,
    /// GPU core count.
    pub gpu_cores: usize,
    /// CPU core count.
    pub cpu_cores: usize,
}

/// Thermal pressure level, mirroring Apple's `ProcessInfo.ThermalState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeuralEngineThermalState {
    /// No thermal pressure.
    Nominal,
    /// Mild pressure; full performance is still available.
    Fair,
    /// Significant pressure; the OS is throttling.
    Serious,
    /// Severe pressure; heavy throttling.
    Critical,
}

impl NeuralEngineThermalState {
    /// The performance scale this crate applies at this thermal level.
    ///
    /// These are *this crate's own* throttling policy, not a measurement of
    /// what the OS does, and are documented as such.
    pub fn performance_scale(self) -> f32 {
        match self {
            Self::Nominal => 1.0,
            Self::Fair => 0.9,
            Self::Serious => 0.6,
            Self::Critical => 0.3,
        }
    }
}

/// Advanced Neural Engine v4 optimization engine.
///
/// # Honesty
///
/// This module previously reported entirely invented analytics: attention
/// returned `Tensor::zeros(&[1, 1])`, execution echoed its input, and every
/// statistic (`total_compilations: 100`, `cache_hit_rate: 0.85`,
/// `overall_improvement: 0.25`, ...) was a hardcoded constant. All of that has
/// been removed:
///
/// - [`Self::execute_optimized_attention`] runs real multi-head scaled
///   dot-product attention, on the Metal GPU when the `metal` feature is on and
///   on real CPU kernels otherwise.
/// - Every figure in [`AdvancedPerformanceAnalytics`] is either measured from
///   actual executions or reported as [`Availability::NotAvailable`] with a
///   reason. Nothing is fabricated.
/// - There is no thermal sensor available to this crate on any supported
///   platform, so the thermal state is whatever the *caller* reports via
///   [`Self::set_thermal_state`], and defaults to `Nominal`. It is never
///   invented.
pub struct AdvancedNeuralEngineV4 {
    config: NeuralEngineV4Config,
    device_info: NeuralEngineDeviceInfo,
    num_cores: usize,

    /// Real measurements of executions that actually ran.
    execution_history: Arc<RwLock<VecDeque<ExecutionRecord>>>,
    /// Caller-reported thermal state; never guessed.
    thermal_state: Arc<RwLock<NeuralEngineThermalState>>,
    /// Real compilation-cache accounting.
    compilation_cache: Arc<RwLock<CompilationCache>>,
}

/// Maximum number of execution records retained for analytics.
const MAX_EXECUTION_HISTORY: usize = 10_000;

impl AdvancedNeuralEngineV4 {
    /// Create a new Neural Engine v4 optimizer for `device_info`.
    ///
    /// # Errors
    /// Returns an error if the configured core count is zero.
    pub fn new(config: NeuralEngineV4Config, device_info: NeuralEngineDeviceInfo) -> Result<Self> {
        let num_cores = match config.num_cores {
            Some(0) => {
                return Err(CoreError::InvalidInput(
                    "Neural Engine core count must be non-zero".to_string(),
                ))
            },
            Some(explicit) => explicit,
            None => Self::detect_neural_engine_cores(&device_info),
        };

        Ok(Self {
            config,
            device_info,
            num_cores,
            execution_history: Arc::new(RwLock::new(VecDeque::with_capacity(64))),
            thermal_state: Arc::new(RwLock::new(NeuralEngineThermalState::Nominal)),
            compilation_cache: Arc::new(RwLock::new(CompilationCache::default())),
        })
    }

    /// The engine configuration.
    pub fn config(&self) -> &NeuralEngineV4Config {
        &self.config
    }

    /// The device this engine was built for.
    pub fn device_info(&self) -> &NeuralEngineDeviceInfo {
        &self.device_info
    }

    /// The Neural Engine core count in use.
    pub fn num_cores(&self) -> usize {
        self.num_cores
    }

    /// Neural Engine core count for a known Apple SoC.
    ///
    /// Returns `None` for an unrecognised chip rather than guessing: a wrong
    /// core count would silently mis-size every downstream decision.
    pub fn known_neural_engine_cores(chip_name: &str) -> Option<usize> {
        Some(match chip_name {
            "A17 Pro" | "A18" | "A18 Pro" => 16,
            "M3" | "M3 Pro" | "M3 Max" | "M4" | "M4 Pro" | "M4 Max" => 16,
            "A16 Bionic" | "A15 Bionic" | "A14 Bionic" => 16,
            "M1" | "M1 Pro" | "M1 Max" | "M1 Ultra" => 16,
            "M2" | "M2 Pro" | "M2 Max" | "M2 Ultra" => 16,
            "A13 Bionic" => 8,
            "A12 Bionic" | "A12X Bionic" | "A12Z Bionic" => 8,
            _ => return None,
        })
    }

    /// Neural Engine core count, falling back to a documented conservative
    /// default for unknown chips.
    pub fn detect_neural_engine_cores(device_info: &NeuralEngineDeviceInfo) -> usize {
        // 8 is the smallest Neural Engine Apple has shipped, so an unknown chip
        // is assumed to be at least that. This is a stated floor, not a claim
        // about the actual hardware.
        Self::known_neural_engine_cores(&device_info.chip_name).unwrap_or(8)
    }

    /// Report the current thermal state.
    ///
    /// No supported platform exposes a Neural Engine thermal sensor to this
    /// crate, so the caller (which may have `ProcessInfo` access on iOS) must
    /// supply it. Until it does, the state stays `Nominal`.
    pub fn set_thermal_state(&self, state: NeuralEngineThermalState) {
        *self.thermal_state.write().unwrap_or_else(|p| p.into_inner()) = state;
    }

    /// The thermal state last reported via [`Self::set_thermal_state`].
    pub fn thermal_state(&self) -> NeuralEngineThermalState {
        *self.thermal_state.read().unwrap_or_else(|p| p.into_inner())
    }

    /// The performance scale implied by the reported thermal state.
    pub fn performance_scale(&self) -> f32 {
        self.thermal_state().performance_scale()
    }

    /// Execute multi-head scaled dot-product attention.
    ///
    /// `query`, `key` and `value` must all be `[seq_len, num_heads * head_dim]`
    /// and identically shaped. With the `metal` feature enabled on macOS the
    /// computation is dispatched to the real Metal kernel in `trustformers-core`;
    /// otherwise it runs the exact same function on real CPU tensor kernels.
    ///
    /// `attention_mask`, when supplied, is an additive `[seq_len, seq_len]`
    /// mask (use `f32::NEG_INFINITY` to mask a position).
    ///
    /// The previous implementation of this method was
    /// `let _ = (query, key, value, attention_mask); Ok(Tensor::zeros(&[1, 1])?)`.
    ///
    /// # Errors
    /// Returns an error for mismatched or non-2-D shapes, a `num_heads` that
    /// does not divide the hidden size, or a mask of the wrong shape.
    pub fn execute_optimized_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        num_heads: usize,
    ) -> Result<Tensor> {
        let start = Instant::now();
        let output = self.attention_impl(query, key, value, attention_mask, num_heads)?;
        self.record_execution(ExecutionRecord {
            operation: "attention".to_string(),
            elapsed: start.elapsed(),
            element_count: output.to_vec_f32()?.len(),
            thermal_state: self.thermal_state(),
            backend: Self::backend_label(),
        });
        Ok(output)
    }

    /// The attention computation itself, without the measurement wrapper.
    fn attention_impl(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        num_heads: usize,
    ) -> Result<Tensor> {
        if num_heads == 0 {
            return Err(CoreError::InvalidInput(
                "Attention requires at least one head".to_string(),
            ));
        }
        let shape = query.shape();
        if shape.len() != 2 {
            return Err(CoreError::InvalidInput(format!(
                "Attention expects 2-D [seq_len, num_heads * head_dim] tensors, got {shape:?}"
            )));
        }
        if key.shape() != shape || value.shape() != shape {
            return Err(CoreError::InvalidInput(format!(
                "Attention requires q/k/v of equal shape, got {:?} / {:?} / {:?}",
                shape,
                key.shape(),
                value.shape()
            )));
        }
        let (seq_len, hidden) = (shape[0], shape[1]);
        if seq_len == 0 || hidden == 0 {
            return Err(CoreError::InvalidInput(
                "Attention requires a non-empty sequence and hidden size".to_string(),
            ));
        }
        if !hidden.is_multiple_of(num_heads) {
            return Err(CoreError::InvalidInput(format!(
                "Attention hidden size {hidden} is not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = hidden / num_heads;

        let mask_values = match attention_mask {
            Some(mask) => {
                let mask_shape = mask.shape();
                if mask_shape != vec![seq_len, seq_len] {
                    return Err(CoreError::InvalidInput(format!(
                        "Attention mask must be [{seq_len}, {seq_len}], got {mask_shape:?}"
                    )));
                }
                Some(mask.to_vec_f32()?)
            },
            None => None,
        };

        // The Metal kernel *always* applies a causal mask, so it computes a
        // different function from unmasked attention. It may therefore only be
        // used when the caller asked for exactly causal attention. Dispatching
        // on `mask.is_none()` would silently return causal results for an
        // unmasked request, which the reference-comparison tests catch.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.config.attention_config.enable_flash_attention
            && mask_values.as_deref().is_some_and(|mask| is_causal_mask(mask, seq_len))
        {
            if let Ok(output) =
                self.attention_metal(query, key, value, seq_len, num_heads, head_dim)
            {
                return Ok(output);
            }
            // A GPU failure falls through to the CPU path, which computes the
            // same function — never to fabricated output.
        }

        let q_data = query.to_vec_f32()?;
        let k_data = key.to_vec_f32()?;
        let v_data = value.to_vec_f32()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * hidden];

        for head in 0..num_heads {
            let head_offset = head * head_dim;
            for row in 0..seq_len {
                // Online (streaming) softmax: numerically stable in one pass,
                // with no seq_len-sized scratch buffer.
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut accumulator = vec![0.0f32; head_dim];

                for col in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[row * hidden + head_offset + d]
                            * k_data[col * hidden + head_offset + d];
                    }
                    let mut score = dot * scale;
                    if let Some(mask) = &mask_values {
                        score += mask[row * seq_len + col];
                    }
                    if !score.is_finite() && score < 0.0 {
                        // A -inf mask entry contributes exactly zero weight.
                        continue;
                    }
                    let new_max = running_max.max(score);
                    let correction = (running_max - new_max).exp();
                    let weight = (score - new_max).exp();
                    running_sum = running_sum * correction + weight;
                    for d in 0..head_dim {
                        accumulator[d] = accumulator[d] * correction
                            + weight * v_data[col * hidden + head_offset + d];
                    }
                    running_max = new_max;
                }

                let inv_sum = if running_sum > 0.0 { 1.0 / running_sum } else { 0.0 };
                for d in 0..head_dim {
                    output[row * hidden + head_offset + d] = accumulator[d] * inv_sum;
                }
            }
        }

        Ok(Tensor::from_vec(output, &[seq_len, hidden])?)
    }

    /// Dispatch attention to the real Metal kernel.
    ///
    /// Only valid for causal attention: the kernel applies the causal mask
    /// unconditionally.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn attention_metal(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let backend = trustformers_core::gpu_ops::metal::get_metal_backend()?;
        let q_id = backend.create_transient_buffer(&query.to_vec_f32()?)?;
        let k_id = backend.create_transient_buffer(&key.to_vec_f32()?)?;
        let v_id = backend.create_transient_buffer(&value.to_vec_f32()?)?;
        let outcome = backend
            .attention_gpu_to_gpu(&q_id, &k_id, &v_id, 1, seq_len, num_heads, head_dim)
            .and_then(|out_id| {
                let data = backend.download_buffer_to_vec(&out_id);
                backend.release_buffers(&[out_id])?;
                data
            });
        backend.release_buffers(&[q_id, k_id, v_id])?;
        Ok(Tensor::from_vec(
            outcome?,
            &[seq_len, num_heads * head_dim],
        )?)
    }

    /// `"metal"` when this build dispatches attention to the GPU, else `"cpu"`.
    pub fn backend_label() -> &'static str {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            "metal"
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            "cpu"
        }
    }

    /// Record one real execution measurement.
    fn record_execution(&self, record: ExecutionRecord) {
        let mut history = self.execution_history.write().unwrap_or_else(|p| p.into_inner());
        if history.len() >= MAX_EXECUTION_HISTORY {
            history.pop_front();
        }
        history.push_back(record);
    }

    /// Push a record directly into the history buffer.
    ///
    /// Test-only: real records are created by the measured execution paths.
    #[cfg(test)]
    pub(crate) fn record_execution_for_test(&self, record: ExecutionRecord) {
        self.record_execution(record);
    }

    /// Record a graph compilation outcome, so that
    /// [`Self::compilation_statistics`] reports reality.
    pub fn record_compilation(&self, succeeded: bool, elapsed: Duration, was_cache_hit: bool) {
        let mut cache = self.compilation_cache.write().unwrap_or_else(|p| p.into_inner());
        cache.record(succeeded, elapsed, was_cache_hit);
    }

    /// Measured compilation statistics, or `NotAvailable` when nothing has been
    /// compiled.
    ///
    /// The previous implementation returned `total_compilations: 100,
    /// successful_compilations: 98, cache_hit_rate: 0.85` unconditionally.
    pub fn compilation_statistics(&self) -> Availability<CompilationStatistics> {
        self.compilation_cache.read().unwrap_or_else(|p| p.into_inner()).statistics()
    }

    /// Measured execution statistics, or `NotAvailable` when nothing has run.
    ///
    /// The previous implementation returned invented latency and throughput.
    pub fn execution_statistics(&self) -> Availability<ExecutionStatistics> {
        let history = self.execution_history.read().unwrap_or_else(|p| p.into_inner());
        if history.is_empty() {
            return Availability::NotAvailable {
                reason: "no operations have been executed on this engine yet".to_string(),
            };
        }

        let total: Duration = history.iter().map(|r| r.elapsed).sum();
        let count = history.len();
        // `count` is non-zero here, so the division is well-defined.
        let average = total / u32::try_from(count).unwrap_or(u32::MAX);
        let elements: usize = history.iter().map(|r| r.element_count).sum();
        let total_secs = total.as_secs_f64();

        Availability::Available(ExecutionStatistics {
            execution_count: count,
            average_latency: average,
            fastest: history.iter().map(|r| r.elapsed).min().unwrap_or(average),
            slowest: history.iter().map(|r| r.elapsed).max().unwrap_or(average),
            total_elements_produced: elements,
            elements_per_second: if total_secs > 0.0 {
                elements as f64 / total_secs
            } else {
                f64::INFINITY
            },
            backend: history.back().map(|r| r.backend).unwrap_or("unknown"),
        })
    }

    /// Real memory statistics for this process, or `NotAvailable` when the
    /// platform does not report them.
    ///
    /// The previous implementation returned fixed byte counts
    /// (`peak_usage: 128 MiB`, `allocation_count: 1000`, ...).
    pub fn memory_statistics(&self) -> Availability<MemoryStatistics> {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );

        match system.process(pid) {
            Some(process) => Availability::Available(MemoryStatistics {
                resident_bytes: process.memory(),
                virtual_bytes: process.virtual_memory(),
            }),
            None => Availability::NotAvailable {
                reason: "the operating system did not report memory usage for this process"
                    .to_string(),
            },
        }
    }

    /// A complete analytics snapshot.
    ///
    /// Every field is either a real measurement or an explicit
    /// [`Availability::NotAvailable`]. Nothing here is invented, and there is no
    /// "optimization effectiveness" or "bottleneck analysis" field: this crate
    /// has no baseline to compare against, so any such number would be fiction.
    pub fn performance_analytics(&self) -> AdvancedPerformanceAnalytics {
        AdvancedPerformanceAnalytics {
            execution: self.execution_statistics(),
            compilation: self.compilation_statistics(),
            memory: self.memory_statistics(),
            thermal_state: self.thermal_state(),
            performance_scale: self.performance_scale(),
            num_cores: self.num_cores,
            backend: Self::backend_label(),
        }
    }

    /// The measurements behind [`Self::execution_statistics`], newest last.
    pub fn execution_records(&self) -> Vec<ExecutionRecord> {
        self.execution_history
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

/// Whether `mask` is exactly the standard causal mask: zero on and below the
/// diagonal, `-inf` strictly above it.
///
/// Used to decide whether the (unconditionally causal) Metal kernel computes
/// the same function the caller asked for.
#[cfg_attr(not(all(target_os = "macos", feature = "metal")), allow(dead_code))]
fn is_causal_mask(mask: &[f32], seq_len: usize) -> bool {
    if mask.len() != seq_len * seq_len {
        return false;
    }
    for row in 0..seq_len {
        for col in 0..seq_len {
            let value = mask[row * seq_len + col];
            let masked_out = value == f32::NEG_INFINITY;
            if col > row {
                // Strictly above the diagonal must be masked out.
                if !masked_out {
                    return false;
                }
            } else if value != 0.0 {
                // On and below the diagonal must be unmodified.
                return false;
            }
        }
    }
    true
}

/// A value that is either genuinely measured or explicitly unavailable.
///
/// Introduced so that a caller can never mistake a fabricated constant for a
/// measurement — the previous analytics API had no way to express "unknown", so
/// it invented numbers instead.
#[derive(Debug, Clone, PartialEq)]
pub enum Availability<T> {
    /// A real measurement.
    Available(T),
    /// No measurement is available, and why.
    NotAvailable {
        /// Human-readable reason.
        reason: String,
    },
}

impl<T> Availability<T> {
    /// The measurement, if there is one.
    pub fn measured(&self) -> Option<&T> {
        match self {
            Self::Available(value) => Some(value),
            Self::NotAvailable { .. } => None,
        }
    }

    /// Whether a real measurement is present.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// One measured execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRecord {
    /// Which operation ran.
    pub operation: String,
    /// Measured wall-clock duration.
    pub elapsed: Duration,
    /// Number of output elements produced.
    pub element_count: usize,
    /// Thermal state reported at the time.
    pub thermal_state: NeuralEngineThermalState,
    /// Which backend executed it.
    pub backend: &'static str,
}

/// Statistics derived from real executions.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionStatistics {
    /// How many executions the figures are based on.
    pub execution_count: usize,
    /// Mean measured latency.
    pub average_latency: Duration,
    /// Fastest measured execution.
    pub fastest: Duration,
    /// Slowest measured execution.
    pub slowest: Duration,
    /// Total output elements produced across all executions.
    pub total_elements_produced: usize,
    /// Measured throughput in output elements per second.
    pub elements_per_second: f64,
    /// Backend that ran the most recent execution.
    pub backend: &'static str,
}

/// Real compilation-cache accounting.
#[derive(Debug, Default)]
struct CompilationCache {
    total: usize,
    successful: usize,
    cache_hits: usize,
    total_time: Duration,
}

impl CompilationCache {
    fn record(&mut self, succeeded: bool, elapsed: Duration, was_cache_hit: bool) {
        self.total += 1;
        if succeeded {
            self.successful += 1;
        }
        if was_cache_hit {
            self.cache_hits += 1;
        }
        self.total_time += elapsed;
    }

    fn statistics(&self) -> Availability<CompilationStatistics> {
        if self.total == 0 {
            return Availability::NotAvailable {
                reason: "no graph compilations have been recorded".to_string(),
            };
        }
        let total = self.total;
        Availability::Available(CompilationStatistics {
            total_compilations: total,
            successful_compilations: self.successful,
            cache_hits: self.cache_hits,
            // `total` is non-zero here.
            average_compilation_time: self.total_time / u32::try_from(total).unwrap_or(u32::MAX),
            cache_hit_rate: self.cache_hits as f64 / total as f64,
        })
    }
}

/// Statistics derived from real graph compilations.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilationStatistics {
    /// Total compilations recorded.
    pub total_compilations: usize,
    /// How many of those succeeded.
    pub successful_compilations: usize,
    /// How many were served from cache.
    pub cache_hits: usize,
    /// Mean measured compilation time.
    pub average_compilation_time: Duration,
    /// Measured cache hit rate in `[0, 1]`.
    pub cache_hit_rate: f64,
}

/// Real process memory usage, sampled via `sysinfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStatistics {
    /// Resident set size in bytes.
    pub resident_bytes: u64,
    /// Virtual memory size in bytes.
    pub virtual_bytes: u64,
}

/// A full analytics snapshot in which every field is measured or explicitly
/// unavailable.
#[derive(Debug, Clone)]
pub struct AdvancedPerformanceAnalytics {
    /// Execution latency and throughput.
    pub execution: Availability<ExecutionStatistics>,
    /// Graph compilation accounting.
    pub compilation: Availability<CompilationStatistics>,
    /// Process memory usage.
    pub memory: Availability<MemoryStatistics>,
    /// Thermal state as last reported by the caller.
    pub thermal_state: NeuralEngineThermalState,
    /// Performance scale implied by that thermal state.
    pub performance_scale: f32,
    /// Neural Engine core count in use.
    pub num_cores: usize,
    /// Backend this build dispatches to.
    pub backend: &'static str,
}

#[cfg(test)]
mod tests;
