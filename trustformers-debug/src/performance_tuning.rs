//! Automated Performance Tuning Recommendations
//!
//! This module analyzes profiling data and generates actionable performance
//! optimization recommendations for transformer models.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance model defining how improvement estimates are computed.
#[derive(Debug, Clone)]
pub enum PerformanceModel {
    /// Rule-based estimation using configurable per-category multipliers.
    ///
    /// Each multiplier scales the contribution of a recommendation's speedup
    /// to the total estimated improvement. Default multipliers are all `1.0`,
    /// which reproduces the original additive behaviour.
    RuleBased {
        /// Multiplier for each recommendation category (defaults to 1.0 if absent).
        multipliers: HashMap<RecommendationCategory, f64>,
    },
}

impl Default for PerformanceModel {
    fn default() -> Self {
        let mut multipliers = HashMap::new();
        multipliers.insert(RecommendationCategory::Memory, 1.0);
        multipliers.insert(RecommendationCategory::Compute, 1.0);
        multipliers.insert(RecommendationCategory::BatchSize, 1.0);
        multipliers.insert(RecommendationCategory::Layer, 1.0);
        multipliers.insert(RecommendationCategory::Hardware, 1.0);
        multipliers.insert(RecommendationCategory::DataLoading, 1.0);
        multipliers.insert(RecommendationCategory::Architecture, 1.0);
        PerformanceModel::RuleBased { multipliers }
    }
}

/// Performance tuning analyzer
#[derive(Debug)]
pub struct PerformanceTuner {
    /// Configuration for the tuner
    config: TunerConfig,
    /// Historical performance data
    history: Vec<PerformanceSnapshot>,
}

/// Configuration for performance tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Enable memory optimization suggestions
    pub enable_memory_tuning: bool,
    /// Enable compute optimization suggestions
    pub enable_compute_tuning: bool,
    /// Enable batch size optimization
    pub enable_batch_tuning: bool,
    /// Enable layer-specific tuning
    pub enable_layer_tuning: bool,
    /// Minimum confidence threshold (0.0-1.0)
    pub confidence_threshold: f64,
    /// Target hardware type
    pub target_hardware: HardwareType,
    /// Number of recent snapshots to consider for data-loading analysis
    pub data_loading_window: usize,
    /// Performance model used for improved-performance estimation
    #[serde(skip)]
    pub performance_model: PerformanceModel,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            enable_memory_tuning: true,
            enable_compute_tuning: true,
            enable_batch_tuning: true,
            enable_layer_tuning: true,
            confidence_threshold: 0.7,
            target_hardware: HardwareType::Auto,
            data_loading_window: 10,
            performance_model: PerformanceModel::default(),
        }
    }
}

/// Target hardware type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareType {
    /// Auto-detect hardware
    Auto,
    /// NVIDIA GPU (CUDA)
    NvidiaGpu,
    /// AMD GPU (ROCm)
    AmdGpu,
    /// Apple Silicon (Metal)
    AppleSilicon,
    /// CPU only
    Cpu,
    /// TPU
    Tpu,
}

/// Performance snapshot for analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Total execution time (ms)
    pub total_time_ms: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Peak memory (MB)
    pub peak_memory_mb: f64,
    /// GPU utilization (0-100)
    pub gpu_utilization: f64,
    /// Throughput (samples/sec)
    pub throughput: f64,
    /// Batch size used
    pub batch_size: usize,
    /// Layer timings (layer name -> time in ms)
    pub layer_timings: HashMap<String, f64>,
    /// Memory per layer (layer name -> memory in MB)
    pub layer_memory: HashMap<String, f64>,
    /// Override hardware type for this snapshot (overrides TunerConfig::target_hardware)
    pub hardware_type: Option<HardwareType>,
    /// Fraction of time (0.0-1.0) spent waiting on I/O during this snapshot
    pub io_wait_pct: Option<f32>,
    /// Samples processed per second during this snapshot
    pub batch_throughput_per_sec: Option<f32>,
    /// Theoretical peak GPU throughput (samples/sec) for the detected GPU
    pub gpu_peak_throughput: Option<f32>,
    /// Number of transformer layers in the model
    pub model_depth: Option<usize>,
    /// Number of attention heads
    pub num_heads: Option<usize>,
    /// Current KV-cache size in bytes
    pub kv_cache_bytes: Option<u64>,
    /// Sequence length for this batch
    pub seq_len: Option<usize>,
}

/// Tuning recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: Priority,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Short title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected impact
    pub expected_impact: ImpactEstimate,
    /// Implementation difficulty
    pub difficulty: Difficulty,
    /// Specific actions to take
    pub actions: Vec<String>,
    /// Code example (if applicable)
    pub code_example: Option<String>,
}

/// Recommendation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Memory optimization
    Memory,
    /// Compute optimization
    Compute,
    /// Batch processing
    BatchSize,
    /// Layer-specific optimization
    Layer,
    /// Hardware configuration
    Hardware,
    /// Data loading
    DataLoading,
    /// Model architecture
    Architecture,
}

/// Priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical (blocking performance)
    Critical,
}

/// Implementation difficulty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    /// Easy to implement
    Easy,
    /// Moderate effort required
    Moderate,
    /// Significant effort required
    Hard,
}

/// Expected performance impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEstimate {
    /// Expected speedup (e.g., 1.5 = 50% faster)
    pub speedup: f64,
    /// Expected memory reduction (MB)
    pub memory_reduction_mb: f64,
    /// Expected throughput improvement (%)
    pub throughput_improvement: f64,
}

/// Complete tuning report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningReport {
    /// All recommendations sorted by priority
    pub recommendations: Vec<Recommendation>,
    /// Current performance summary
    pub current_performance: PerformanceSummary,
    /// Estimated performance after applying recommendations
    pub estimated_performance: PerformanceSummary,
    /// Analysis timestamp
    pub timestamp: u64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Average execution time (ms)
    pub avg_time_ms: f64,
    /// Average memory usage (MB)
    pub avg_memory_mb: f64,
    /// Average throughput (samples/sec)
    pub avg_throughput: f64,
    /// GPU utilization (%)
    pub gpu_utilization: f64,
    /// Efficiency score (0-100)
    pub efficiency_score: f64,
}

impl PerformanceTuner {
    /// Create a new performance tuner
    pub fn new(config: TunerConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Record a performance snapshot
    pub fn record_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        self.history.push(snapshot);

        // Keep only last 100 snapshots
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Detect the effective hardware type at runtime using cfg flags.
    ///
    /// Each branch is guarded by a disjoint cfg predicate so that exactly one
    /// branch is compiled on any given target, eliminating unreachable-code
    /// and unexpected-cfg warnings.
    fn detect_hardware(&self) -> HardwareType {
        #[cfg(target_os = "macos")]
        return HardwareType::AppleSilicon;

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        return HardwareType::NvidiaGpu;

        #[cfg(all(not(target_os = "macos"), not(feature = "cuda"), feature = "rocm"))]
        return HardwareType::AmdGpu;

        #[cfg(all(
            not(target_os = "macos"),
            not(feature = "cuda"),
            not(feature = "rocm"),
            feature = "tpu"
        ))]
        return HardwareType::Tpu;

        #[cfg(all(
            not(target_os = "macos"),
            not(feature = "cuda"),
            not(feature = "rocm"),
            not(feature = "tpu")
        ))]
        HardwareType::Cpu
    }

    /// Return the detected hardware type (public API for `detect_hardware`).
    pub fn detected_hardware(&self) -> HardwareType {
        self.detect_hardware()
    }

    /// Analyze performance and generate recommendations
    pub fn analyze(&self) -> Result<TuningReport> {
        let mut recommendations = Vec::new();

        if self.history.is_empty() {
            anyhow::bail!("No performance data available");
        }

        // Generate different types of recommendations
        if self.config.enable_memory_tuning {
            recommendations.extend(self.analyze_memory());
        }

        if self.config.enable_compute_tuning {
            recommendations.extend(self.analyze_compute());
        }

        if self.config.enable_batch_tuning {
            recommendations.extend(self.analyze_batch_size());
        }

        if self.config.enable_layer_tuning {
            recommendations.extend(self.analyze_layers());
        }

        // Hardware analysis is always relevant regardless of other flags.
        recommendations.extend(self.analyze_hardware(&self.history));

        // Data-loading analysis only when any snapshot carries io_wait_pct data.
        if self.history.iter().any(|s| s.io_wait_pct.is_some()) {
            recommendations.extend(self.analyze_data_loading(&self.history));
        }

        // Architecture analysis only when any snapshot carries seq_len data.
        if self.history.iter().any(|s| s.seq_len.is_some()) {
            recommendations.extend(self.analyze_architecture(&self.history));
        }

        // Filter by confidence threshold
        recommendations.retain(|r| r.confidence >= self.config.confidence_threshold);

        // Sort by priority (highest first)
        recommendations.sort_by_key(|item| std::cmp::Reverse(item.priority));

        let current_perf = self.compute_current_performance();
        let estimated_perf = self.estimate_improved_performance(&recommendations);

        Ok(TuningReport {
            recommendations,
            current_performance: current_perf,
            estimated_performance: estimated_perf,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        })
    }

    /// Analyze memory usage patterns
    fn analyze_memory(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        let avg_memory =
            self.history.iter().map(|s| s.memory_usage_mb).sum::<f64>() / self.history.len() as f64;

        let peak_memory = self.history.iter().map(|s| s.peak_memory_mb).fold(0.0, f64::max);

        // Check for high memory fragmentation
        if peak_memory > avg_memory * 1.5 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Memory,
                priority: Priority::High,
                confidence: 0.85,
                title: "Reduce memory fragmentation".to_string(),
                description: format!(
                    "Peak memory ({:.1}MB) is significantly higher than average ({:.1}MB). \
                     This indicates memory fragmentation.",
                    peak_memory, avg_memory
                ),
                expected_impact: ImpactEstimate {
                    speedup: 1.1,
                    memory_reduction_mb: (peak_memory - avg_memory) * 0.5,
                    throughput_improvement: 5.0,
                },
                difficulty: Difficulty::Moderate,
                actions: vec![
                    "Enable gradient checkpointing to reduce activation memory".to_string(),
                    "Use torch.cuda.empty_cache() or equivalent after large operations".to_string(),
                    "Consider using mixed precision training (FP16/BF16)".to_string(),
                ],
                code_example: Some(
                    "# Enable gradient checkpointing\n\
                     model.gradient_checkpointing_enable()\n\
                     \n\
                     # Use automatic mixed precision\n\
                     with torch.cuda.amp.autocast():\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}output = model(input)"
                        .to_string(),
                ),
            });
        }

        // Check for excessive memory usage
        if avg_memory > 8000.0 && self.config.target_hardware == HardwareType::Cpu {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Memory,
                priority: Priority::High,
                confidence: 0.9,
                title: "Reduce memory footprint for CPU execution".to_string(),
                description: format!(
                    "Average memory usage ({:.1}GB) is high for CPU execution. \
                     Consider model compression techniques.",
                    avg_memory / 1024.0
                ),
                expected_impact: ImpactEstimate {
                    speedup: 1.3,
                    memory_reduction_mb: avg_memory * 0.4,
                    throughput_improvement: 15.0,
                },
                difficulty: Difficulty::Moderate,
                actions: vec![
                    "Apply 8-bit or 4-bit quantization".to_string(),
                    "Use dynamic quantization for linear layers".to_string(),
                    "Consider model distillation to a smaller model".to_string(),
                ],
                code_example: Some(
                    "# Apply 8-bit quantization\n\
                     quantized_model = torch.quantization.quantize_dynamic(\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}model, {torch.nn.Linear}, dtype=torch.qint8\n\
                     )"
                    .to_string(),
                ),
            });
        }

        recommendations
    }

    /// Analyze compute patterns
    fn analyze_compute(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        let avg_gpu_util =
            self.history.iter().map(|s| s.gpu_utilization).sum::<f64>() / self.history.len() as f64;

        // Check for low GPU utilization
        if avg_gpu_util < 50.0 && self.config.target_hardware != HardwareType::Cpu {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Compute,
                priority: Priority::High,
                confidence: 0.88,
                title: "Improve GPU utilization".to_string(),
                description: format!(
                    "Average GPU utilization ({:.1}%) is low. GPU is underutilized.",
                    avg_gpu_util
                ),
                expected_impact: ImpactEstimate {
                    speedup: 1.8,
                    memory_reduction_mb: 0.0,
                    throughput_improvement: 40.0,
                },
                difficulty: Difficulty::Easy,
                actions: vec![
                    "Increase batch size to maximize GPU occupancy".to_string(),
                    "Use DataLoader with num_workers > 0 to prevent CPU bottleneck".to_string(),
                    "Enable pin_memory for faster host-to-device transfers".to_string(),
                    "Use compiled models (torch.compile)".to_string(),
                ],
                code_example: Some(
                    "# Optimize data loading\n\
                     dataloader = DataLoader(\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}dataset,\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}batch_size=32,\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}num_workers=4,  # Parallel data loading\n\
                     \u{00a0}\u{00a0}\u{00a0}\u{00a0}pin_memory=True  # Faster transfers\n\
                     )"
                    .to_string(),
                ),
            });
        }

        recommendations
    }

    /// Analyze batch size efficiency
    fn analyze_batch_size(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        if let Some(last_snapshot) = self.history.last() {
            let batch_size = last_snapshot.batch_size;

            // Check if batch size is too small
            if batch_size < 16 && self.config.target_hardware != HardwareType::Cpu {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::BatchSize,
                    priority: Priority::Medium,
                    confidence: 0.75,
                    title: "Increase batch size".to_string(),
                    description: format!(
                        "Current batch size ({}) is small. Larger batches improve GPU utilization.",
                        batch_size
                    ),
                    expected_impact: ImpactEstimate {
                        speedup: 1.5,
                        memory_reduction_mb: 0.0,
                        throughput_improvement: 30.0,
                    },
                    difficulty: Difficulty::Easy,
                    actions: vec![
                        format!("Increase batch size to {} or higher", batch_size * 2),
                        "Monitor memory usage to find optimal batch size".to_string(),
                        "Use gradient accumulation if memory is limited".to_string(),
                    ],
                    code_example: Some(
                        "# Gradient accumulation for effective larger batch\n\
                         accumulation_steps = 4\n\
                         for i, batch in enumerate(dataloader):\n\
                         \u{00a0}\u{00a0}\u{00a0}\u{00a0}loss = model(batch) / accumulation_steps\n\
                         \u{00a0}\u{00a0}\u{00a0}\u{00a0}loss.backward()\n\
                         \u{00a0}\u{00a0}\u{00a0}\u{00a0}if (i + 1) % accumulation_steps == 0:\n\
                         \u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}optimizer.step()\n\
                         \u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}optimizer.zero_grad()"
                            .to_string()
                    ),
                });
            }
        }

        recommendations
    }

    /// Analyze layer-specific bottlenecks
    fn analyze_layers(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        if let Some(snapshot) = self.history.last() {
            let total_time: f64 = snapshot.layer_timings.values().sum();

            // Find layers that take >20% of total time
            for (layer_name, &time) in &snapshot.layer_timings {
                let percentage = (time / total_time) * 100.0;

                if percentage > 20.0 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::Layer,
                        priority: Priority::Medium,
                        confidence: 0.8,
                        title: format!("Optimize {} layer", layer_name),
                        description: format!(
                            "Layer '{}' takes {:.1}% of total execution time ({:.2}ms). \
                             Consider layer-specific optimizations.",
                            layer_name, percentage, time
                        ),
                        expected_impact: ImpactEstimate {
                            speedup: 1.2,
                            memory_reduction_mb: 0.0,
                            throughput_improvement: 15.0,
                        },
                        difficulty: Difficulty::Moderate,
                        actions: vec![
                            "Use fused operations for this layer type".to_string(),
                            "Check if layer can benefit from Flash Attention".to_string(),
                            "Consider layer pruning if accuracy allows".to_string(),
                        ],
                        code_example: None,
                    });
                }
            }
        }

        recommendations
    }

    /// Analyze hardware-specific optimisation opportunities.
    ///
    /// Determines the effective hardware type from the latest snapshot's
    /// `hardware_type` field (if set), then falls back to the config value
    /// (resolving `Auto` via `detect_hardware`).
    fn analyze_hardware(&self, snapshots: &[PerformanceSnapshot]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        let effective_hw = snapshots.last().and_then(|s| s.hardware_type).unwrap_or_else(|| {
            if self.config.target_hardware == HardwareType::Auto {
                self.detect_hardware()
            } else {
                self.config.target_hardware
            }
        });

        let hw_recs: &[(&str, &str)] = match effective_hw {
            HardwareType::NvidiaGpu => &[
                (
                    "Enable TF32 matmul for Ampere+ GPUs",
                    "Enable TF32 matmul for Ampere+ GPUs (torch.backends.cuda.matmul.allow_tf32)",
                ),
                (
                    "cuDNN deterministic algorithms",
                    "Consider cuDNN deterministic algorithms for reproducibility (may reduce throughput)",
                ),
            ],
            HardwareType::AmdGpu => &[
                (
                    "ROCm hipBLAS strided-batched GEMM",
                    "Enable ROCm hipBLAS strided-batched GEMM for batch inference",
                ),
                (
                    "bf16 precision with ROCm MI200+",
                    "Use bf16 precision with ROCm MI200+ for 2x throughput",
                ),
            ],
            HardwareType::AppleSilicon => &[
                (
                    "Metal Performance Shaders fused kernels",
                    "Enable Metal Performance Shaders fused kernels via trustformers metal backend",
                ),
                (
                    "f16 precision on Apple Silicon",
                    "Use f16 precision on Apple Silicon where model accuracy permits",
                ),
            ],
            HardwareType::Cpu => &[
                (
                    "Enable AVX-512 via scirs2-core SIMD",
                    "Enable AVX-512 via scirs2-core SIMD features if not already active",
                ),
                (
                    "AMX acceleration on Apple M-series",
                    "Use AMX acceleration on Apple M-series CPUs for matrix operations",
                ),
            ],
            HardwareType::Tpu => &[
                (
                    "bf16 precision for TPU v4+",
                    "Use bf16 precision with matmul_precision=highest for TPU v4+",
                ),
                (
                    "XLA sharding for tensor parallelism",
                    "Enable XLA sharding for tensor parallelism across TPU cores",
                ),
            ],
            HardwareType::Auto => {
                // Resolve Auto by detecting actual hardware and re-running.
                let resolved = self.detect_hardware();
                let resolved_snap: Vec<PerformanceSnapshot> = snapshots
                    .iter()
                    .cloned()
                    .map(|mut s| {
                        s.hardware_type = Some(resolved);
                        s
                    })
                    .collect();
                return self.analyze_hardware(&resolved_snap);
            }
        };

        for (title, description) in hw_recs {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Hardware,
                priority: Priority::Medium,
                confidence: 0.8,
                title: (*title).to_string(),
                description: (*description).to_string(),
                expected_impact: ImpactEstimate {
                    speedup: 1.15,
                    memory_reduction_mb: 0.0,
                    throughput_improvement: 10.0,
                },
                difficulty: Difficulty::Easy,
                actions: vec![(*description).to_string()],
                code_example: None,
            });
        }

        recommendations
    }

    /// Analyse data-loading bottlenecks from recent snapshots.
    fn analyze_data_loading(&self, snapshots: &[PerformanceSnapshot]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        let window = self.config.data_loading_window.min(snapshots.len());
        if window == 0 {
            return recommendations;
        }
        let recent = &snapshots[snapshots.len() - window..];

        // Average io_wait_pct over the window (ignoring None entries).
        let (io_sum, io_count) = recent.iter().fold((0.0_f64, 0_usize), |(acc, n), s| {
            if let Some(pct) = s.io_wait_pct {
                (acc + pct as f64, n + 1)
            } else {
                (acc, n)
            }
        });

        if io_count > 0 {
            let avg_io_wait = io_sum / io_count as f64;

            if avg_io_wait > 0.15 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::DataLoading,
                    priority: Priority::High,
                    confidence: 0.82,
                    title: "Increase data-loader worker count".to_string(),
                    description: format!(
                        "Increase data-loader worker count (current I/O wait {:.1}% suggests bottleneck)",
                        avg_io_wait * 100.0
                    ),
                    expected_impact: ImpactEstimate {
                        speedup: 1.2,
                        memory_reduction_mb: 0.0,
                        throughput_improvement: 20.0,
                    },
                    difficulty: Difficulty::Easy,
                    actions: vec![
                        format!(
                            "Increase data-loader worker count (current I/O wait {:.1}% suggests bottleneck)",
                            avg_io_wait * 100.0
                        ),
                        "Enable dataset prefetch to overlap data loading with model computation"
                            .to_string(),
                        "Memory-map large weight files to reduce I/O overhead".to_string(),
                    ],
                    code_example: None,
                });
            }
        }

        // Check compute/GPU utilisation ratio.
        let last = snapshots.last();
        if let Some(snap) = last {
            if let (Some(bt), Some(gpt)) = (snap.batch_throughput_per_sec, snap.gpu_peak_throughput)
            {
                if gpt > 0.0 && (bt / gpt) < 0.5 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::DataLoading,
                        priority: Priority::Medium,
                        confidence: 0.75,
                        title: "Parallelise tokenization across CPU workers".to_string(),
                        description:
                            "Parallelise tokenization across CPU workers to reduce preprocessing bottleneck"
                                .to_string(),
                        expected_impact: ImpactEstimate {
                            speedup: 1.25,
                            memory_reduction_mb: 0.0,
                            throughput_improvement: 25.0,
                        },
                        difficulty: Difficulty::Moderate,
                        actions: vec![
                            "Parallelise tokenization across CPU workers to reduce preprocessing bottleneck"
                                .to_string(),
                            "Move data preprocessing to CPU worker pool to better utilise GPU"
                                .to_string(),
                        ],
                        code_example: None,
                    });
                }
            }
        }

        recommendations
    }

    /// Analyse model architecture for structural optimisation opportunities.
    fn analyze_architecture(&self, snapshots: &[PerformanceSnapshot]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Use the latest snapshot that carries seq_len.
        let seq_snap = snapshots.iter().rev().find(|s| s.seq_len.is_some());

        if let Some(snap) = seq_snap {
            let seq_len = snap.seq_len.unwrap_or(0);

            if seq_len > 1024 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::Architecture,
                    priority: Priority::High,
                    confidence: 0.88,
                    title: "Enable Flash Attention for long sequences".to_string(),
                    description: format!(
                        "Enable Flash Attention for seq_len={} (reduces memory O(n^2) to O(n))",
                        seq_len
                    ),
                    expected_impact: ImpactEstimate {
                        speedup: 1.6,
                        memory_reduction_mb: 0.0,
                        throughput_improvement: 30.0,
                    },
                    difficulty: Difficulty::Moderate,
                    actions: vec![format!(
                        "Enable Flash Attention for seq_len={} (reduces memory O(n^2) to O(n))",
                        seq_len
                    )],
                    code_example: None,
                });
            }

            if let (Some(kv_bytes), Some(gpu_peak)) =
                (snap.kv_cache_bytes, snap.gpu_peak_throughput)
            {
                let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
                // Heuristic: KV cache > 50% of the GPU memory budget (proxy via peak throughput).
                if kv_bytes as f64 > gpu_peak as f64 * 0.5 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::Architecture,
                        priority: Priority::High,
                        confidence: 0.78,
                        title: "Reduce KV heads using Grouped-Query Attention".to_string(),
                        description: format!(
                            "Reduce KV heads using Grouped-Query Attention (GQA) — current KV cache {:.0} MB exceeds 50% of GPU memory budget",
                            kv_mb
                        ),
                        expected_impact: ImpactEstimate {
                            speedup: 1.3,
                            memory_reduction_mb: kv_mb * 0.5,
                            throughput_improvement: 20.0,
                        },
                        difficulty: Difficulty::Hard,
                        actions: vec![format!(
                            "Reduce KV heads using Grouped-Query Attention (GQA) — current KV cache {:.0} MB exceeds 50% of GPU memory budget",
                            kv_mb
                        )],
                        code_example: None,
                    });
                }
            }

            if seq_len > 4096 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::Architecture,
                    priority: Priority::Medium,
                    confidence: 0.82,
                    title: "Consider sliding-window attention".to_string(),
                    description:
                        "Consider sliding-window attention (Mistral-style) to reduce quadratic memory growth at very long contexts"
                            .to_string(),
                    expected_impact: ImpactEstimate {
                        speedup: 1.4,
                        memory_reduction_mb: 0.0,
                        throughput_improvement: 25.0,
                    },
                    difficulty: Difficulty::Hard,
                    actions: vec![
                        "Consider sliding-window attention (Mistral-style) to reduce quadratic memory growth at very long contexts"
                            .to_string(),
                    ],
                    code_example: None,
                });
            }

            if let Some(num_heads) = snap.num_heads {
                if num_heads > 32 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::Architecture,
                        priority: Priority::Medium,
                        confidence: 0.76,
                        title: "Multi-Query Attention to reduce KV memory".to_string(),
                        description: format!(
                            "Multi-Query Attention (single KV head) could reduce memory by {}x while retaining most accuracy",
                            num_heads
                        ),
                        expected_impact: ImpactEstimate {
                            speedup: 1.2,
                            memory_reduction_mb: 0.0,
                            throughput_improvement: 15.0,
                        },
                        difficulty: Difficulty::Hard,
                        actions: vec![format!(
                            "Multi-Query Attention (single KV head) could reduce memory by {}x while retaining most accuracy",
                            num_heads
                        )],
                        code_example: None,
                    });
                }
            }
        }

        // model_depth-based recommendations (use latest snapshot with depth set).
        let depth_snap = snapshots.iter().rev().find(|s| s.model_depth.is_some());
        if let Some(snap) = depth_snap {
            if let Some(depth) = snap.model_depth {
                if depth > 48 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::Architecture,
                        priority: Priority::Medium,
                        confidence: 0.84,
                        title: "Enable gradient checkpointing for deep models".to_string(),
                        description: format!(
                            "Enable gradient checkpointing for models with >48 layers — reduces activation memory at ~33% compute overhead (current depth: {})",
                            depth
                        ),
                        expected_impact: ImpactEstimate {
                            speedup: 0.85, // slight slowdown due to recomputation
                            memory_reduction_mb: 0.0,
                            throughput_improvement: 0.0,
                        },
                        difficulty: Difficulty::Easy,
                        actions: vec![
                            "Enable gradient checkpointing for models with >48 layers — reduces activation memory at ~33% compute overhead"
                                .to_string(),
                        ],
                        code_example: None,
                    });
                }
            }
        }

        recommendations
    }

    /// Compute current performance summary
    fn compute_current_performance(&self) -> PerformanceSummary {
        let count = self.history.len() as f64;

        let avg_time = self.history.iter().map(|s| s.total_time_ms).sum::<f64>() / count;

        let avg_memory = self.history.iter().map(|s| s.memory_usage_mb).sum::<f64>() / count;

        let avg_throughput = self.history.iter().map(|s| s.throughput).sum::<f64>() / count;

        let avg_gpu = self.history.iter().map(|s| s.gpu_utilization).sum::<f64>() / count;

        // Compute efficiency score (0-100)
        let efficiency = (avg_gpu.min(100.0) + (avg_throughput / 10.0).min(100.0)) / 2.0;

        PerformanceSummary {
            avg_time_ms: avg_time,
            avg_memory_mb: avg_memory,
            avg_throughput,
            gpu_utilization: avg_gpu,
            efficiency_score: efficiency,
        }
    }

    /// Estimate performance after applying recommendations.
    ///
    /// Uses `TunerConfig::performance_model` to scale each recommendation's
    /// speedup contribution, replacing hard-coded magic numbers with
    /// user-configurable multipliers.
    fn estimate_improved_performance(
        &self,
        recommendations: &[Recommendation],
    ) -> PerformanceSummary {
        let current = self.compute_current_performance();

        let total_speedup: f64 = match &self.config.performance_model {
            PerformanceModel::RuleBased { multipliers } => {
                recommendations
                    .iter()
                    .map(|r| {
                        let m = multipliers.get(&r.category).copied().unwrap_or(1.0);
                        m * (r.expected_impact.speedup - 1.0)
                    })
                    .sum::<f64>()
                    + 1.0
            },
        };

        let total_memory_reduction: f64 =
            recommendations.iter().map(|r| r.expected_impact.memory_reduction_mb).sum();

        let total_throughput_improvement: f64 =
            recommendations.iter().map(|r| r.expected_impact.throughput_improvement).sum();

        PerformanceSummary {
            avg_time_ms: current.avg_time_ms / total_speedup,
            avg_memory_mb: (current.avg_memory_mb - total_memory_reduction).max(0.0),
            avg_throughput: current.avg_throughput * (1.0 + total_throughput_improvement / 100.0),
            gpu_utilization: (current.gpu_utilization * 1.2).min(95.0),
            efficiency_score: (current.efficiency_score * 1.3).min(100.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_snapshot() -> PerformanceSnapshot {
        PerformanceSnapshot {
            timestamp: 0,
            total_time_ms: 100.0,
            memory_usage_mb: 1000.0,
            peak_memory_mb: 2000.0,
            gpu_utilization: 40.0,
            throughput: 20.0,
            batch_size: 8,
            layer_timings: {
                let mut t = HashMap::new();
                t.insert("attention".to_string(), 60.0);
                t.insert("ffn".to_string(), 30.0);
                t.insert("other".to_string(), 10.0);
                t
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_tuner_creation() {
        let config = TunerConfig::default();
        let _tuner = PerformanceTuner::new(config);
    }

    #[test]
    fn test_snapshot_recording() {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        let snapshot = PerformanceSnapshot {
            timestamp: 0,
            total_time_ms: 100.0,
            memory_usage_mb: 500.0,
            peak_memory_mb: 600.0,
            gpu_utilization: 75.0,
            throughput: 50.0,
            batch_size: 16,
            layer_timings: HashMap::new(),
            layer_memory: HashMap::new(),
            ..Default::default()
        };

        tuner.record_snapshot(snapshot);
        assert_eq!(tuner.history.len(), 1);
    }

    #[test]
    fn test_analysis_with_data() -> Result<()> {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        // Add some sample data
        for i in 0..10 {
            let snapshot = PerformanceSnapshot {
                timestamp: i,
                ..base_snapshot()
            };

            tuner.record_snapshot(snapshot);
        }

        let report = tuner.analyze()?;

        // Should have recommendations
        assert!(!report.recommendations.is_empty());

        // Should have current and estimated performance
        assert!(report.current_performance.avg_time_ms > 0.0);
        assert!(report.estimated_performance.avg_time_ms > 0.0);

        Ok(())
    }

    #[test]
    fn test_analyze_hardware_nvidia_produces_hardware_recommendation() -> Result<()> {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        let snapshot = PerformanceSnapshot {
            hardware_type: Some(HardwareType::NvidiaGpu),
            ..base_snapshot()
        };
        tuner.record_snapshot(snapshot);

        let report = tuner.analyze()?;
        let hw_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| r.category == RecommendationCategory::Hardware)
            .collect();

        assert!(
            !hw_recs.is_empty(),
            "Expected at least one Hardware recommendation for NvidiaGpu"
        );
        Ok(())
    }

    #[test]
    fn test_analyze_hardware_apple_silicon() -> Result<()> {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        let snapshot = PerformanceSnapshot {
            hardware_type: Some(HardwareType::AppleSilicon),
            ..base_snapshot()
        };
        tuner.record_snapshot(snapshot);

        let report = tuner.analyze()?;
        let hw_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| r.category == RecommendationCategory::Hardware)
            .collect();

        assert!(
            !hw_recs.is_empty(),
            "Expected at least one Hardware recommendation for AppleSilicon"
        );
        Ok(())
    }

    #[test]
    fn test_analyze_data_loading_high_io_wait() -> Result<()> {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        let snapshot = PerformanceSnapshot {
            io_wait_pct: Some(0.3),
            gpu_peak_throughput: Some(100.0),
            batch_throughput_per_sec: Some(20.0),
            ..base_snapshot()
        };
        tuner.record_snapshot(snapshot);

        let report = tuner.analyze()?;
        let dl_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| r.category == RecommendationCategory::DataLoading)
            .collect();

        assert!(
            !dl_recs.is_empty(),
            "Expected at least one DataLoading recommendation for high I/O wait"
        );
        Ok(())
    }

    #[test]
    fn test_analyze_architecture_long_context() -> Result<()> {
        let mut tuner = PerformanceTuner::new(TunerConfig::default());

        let snapshot = PerformanceSnapshot {
            seq_len: Some(4096),
            ..base_snapshot()
        };
        tuner.record_snapshot(snapshot);

        let report = tuner.analyze()?;
        let arch_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| {
                r.category == RecommendationCategory::Architecture
                    && r.description.contains("Flash Attention")
            })
            .collect();

        assert!(
            !arch_recs.is_empty(),
            "Expected at least one Architecture recommendation mentioning Flash Attention"
        );
        Ok(())
    }

    #[test]
    fn test_detected_hardware_returns_non_auto() {
        let tuner = PerformanceTuner::new(TunerConfig::default());
        let hw = tuner.detected_hardware();
        assert_ne!(
            hw,
            HardwareType::Auto,
            "detected_hardware should never return Auto"
        );
    }

    #[test]
    fn test_performance_model_rule_based_configurable() -> Result<()> {
        // Build a custom model where Memory multiplier is 2.0 (doubles each Memory rec's contribution).
        let mut multipliers = HashMap::new();
        for cat in &[
            RecommendationCategory::Memory,
            RecommendationCategory::Compute,
            RecommendationCategory::BatchSize,
            RecommendationCategory::Layer,
            RecommendationCategory::Hardware,
            RecommendationCategory::DataLoading,
            RecommendationCategory::Architecture,
        ] {
            multipliers.insert(*cat, 1.0);
        }
        multipliers.insert(RecommendationCategory::Memory, 2.0);

        let config = TunerConfig {
            performance_model: PerformanceModel::RuleBased { multipliers },
            ..TunerConfig::default()
        };

        let mut tuner = PerformanceTuner::new(config);

        // Add snapshots that trigger a Memory recommendation (fragmentation).
        for i in 0..5 {
            let snapshot = PerformanceSnapshot {
                timestamp: i,
                memory_usage_mb: 1000.0,
                peak_memory_mb: 3000.0, // >1.5x average triggers memory rec
                ..base_snapshot()
            };
            tuner.record_snapshot(snapshot);
        }

        let report = tuner.analyze()?;

        // With a 2x multiplier on Memory, the speedup contribution of Memory recommendations
        // is doubled. The estimated throughput should exceed that of the default model.
        let default_config = TunerConfig::default();
        let mut default_tuner = PerformanceTuner::new(default_config);
        for i in 0..5 {
            let snapshot = PerformanceSnapshot {
                timestamp: i,
                memory_usage_mb: 1000.0,
                peak_memory_mb: 3000.0,
                ..base_snapshot()
            };
            default_tuner.record_snapshot(snapshot);
        }
        let default_report = default_tuner.analyze()?;

        // Custom model with multiplier=2 should estimate at least as well as default (multiplier=1).
        // Both have the same recs, but the custom model scales Memory speedup by 2x.
        assert!(
            report.estimated_performance.avg_time_ms
                <= default_report.estimated_performance.avg_time_ms + 1e-6,
            "Custom multiplier=2.0 should produce at least as optimistic a time estimate as default"
        );

        Ok(())
    }
}
