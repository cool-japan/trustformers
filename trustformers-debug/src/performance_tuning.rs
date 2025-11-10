//! Automated Performance Tuning Recommendations
//!
//! This module analyzes profiling data and generates actionable performance
//! optimization recommendations for transformer models.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

        // Filter by confidence threshold
        recommendations.retain(|r| r.confidence >= self.config.confidence_threshold);

        // Sort by priority (highest first)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        let current_perf = self.compute_current_performance();
        let estimated_perf = self.estimate_improved_performance(&recommendations);

        Ok(TuningReport {
            recommendations,
            current_performance: current_perf,
            estimated_performance: estimated_perf,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
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

    /// Estimate performance after applying recommendations
    fn estimate_improved_performance(
        &self,
        recommendations: &[Recommendation],
    ) -> PerformanceSummary {
        let current = self.compute_current_performance();

        // Aggregate expected improvements
        let total_speedup: f64 =
            recommendations.iter().map(|r| r.expected_impact.speedup - 1.0).sum::<f64>() + 1.0;

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
                total_time_ms: 100.0,
                memory_usage_mb: 1000.0,
                peak_memory_mb: 2000.0, // High fragmentation
                gpu_utilization: 40.0,  // Low utilization
                throughput: 20.0,
                batch_size: 8, // Small batch
                layer_timings: {
                    let mut timings = HashMap::new();
                    timings.insert("attention".to_string(), 60.0);
                    timings.insert("ffn".to_string(), 30.0);
                    timings.insert("other".to_string(), 10.0);
                    timings
                },
                layer_memory: HashMap::new(),
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
}
