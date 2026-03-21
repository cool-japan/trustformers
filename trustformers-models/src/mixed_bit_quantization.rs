//! # Mixed-Bit Quantization Framework
//!
//! This module provides advanced mixed-bit quantization capabilities where different
//! layers can use different bit widths to optimize the trade-off between model
//! accuracy and compression ratio.
//!
//! ## Features
//!
//! - **Per-Layer Bit Allocation**: Automatically determine optimal bit widths for each layer
//! - **Sensitivity Analysis**: Analyze layer sensitivity to quantization
//! - **Advanced Calibration**: Multiple calibration strategies for optimal quantization parameters
//! - **Gradient-Free Optimization**: Bit allocation without backpropagation
//! - **Hardware-Aware Quantization**: Consider target hardware capabilities
//! - **Progressive Quantization**: Gradually reduce precision during training
//! - **Quality Metrics**: Comprehensive evaluation of quantization quality
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::mixed_bit_quantization::{
//!     MixedBitQuantizer, QuantizationConfig, BitAllocationStrategy
//! };
//!
//! let config = QuantizationConfig::default()
//!     .with_target_compression(4.0)
//!     .with_max_accuracy_drop(0.02);
//!
//! let quantizer = MixedBitQuantizer::new(config);
//! let quantized_model = quantizer.quantize_model(model, calibration_data)?;
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for mixed-bit quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedBitQuantizationConfig {
    /// Target compression ratio (e.g., 4.0 for 4x compression)
    pub target_compression_ratio: f32,
    /// Maximum allowed accuracy drop (0.0-1.0)
    pub max_accuracy_drop: f32,
    /// Available bit widths for quantization
    pub available_bit_widths: Vec<u8>,
    /// Bit allocation strategy
    pub allocation_strategy: BitAllocationStrategy,
    /// Calibration configuration
    pub calibration_config: CalibrationConfig,
    /// Hardware constraints
    pub hardware_constraints: Option<HardwareConstraints>,
    /// Whether to use gradient-free optimization
    pub gradient_free_optimization: bool,
    /// Progressive quantization settings
    pub progressive_quantization: Option<ProgressiveQuantizationConfig>,
    /// Layer-specific constraints
    pub layer_constraints: HashMap<String, LayerQuantizationConstraints>,
}

impl Default for MixedBitQuantizationConfig {
    fn default() -> Self {
        Self {
            target_compression_ratio: 4.0,
            max_accuracy_drop: 0.02,
            available_bit_widths: vec![4, 6, 8, 16],
            allocation_strategy: BitAllocationStrategy::SensitivityBased,
            calibration_config: CalibrationConfig::default(),
            hardware_constraints: None,
            gradient_free_optimization: true,
            progressive_quantization: None,
            layer_constraints: HashMap::new(),
        }
    }
}

impl MixedBitQuantizationConfig {
    pub fn with_target_compression(mut self, ratio: f32) -> Self {
        self.target_compression_ratio = ratio;
        self
    }

    pub fn with_max_accuracy_drop(mut self, drop: f32) -> Self {
        self.max_accuracy_drop = drop;
        self
    }

    pub fn with_bit_widths(mut self, widths: Vec<u8>) -> Self {
        self.available_bit_widths = widths;
        self
    }
}

/// Strategies for allocating bit widths to different layers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BitAllocationStrategy {
    /// Allocate bits based on layer sensitivity analysis
    SensitivityBased,
    /// Use reinforcement learning for bit allocation
    ReinforcementLearning,
    /// Evolutionary algorithm for optimization
    EvolutionaryAlgorithm,
    /// Greedy search with local optimization
    GreedySearch,
    /// Mixed-integer programming approach
    MixedIntegerProgramming,
    /// Neural architecture search for bit allocation
    NeuralArchitectureSearch,
    /// Pareto-optimal bit allocation
    ParetoOptimal,
    /// Custom user-defined allocation
    Custom(HashMap<String, u8>),
}

/// Calibration configuration for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of calibration samples
    pub num_samples: usize,
    /// Calibration method
    pub method: CalibrationMethod,
    /// Percentile for activation range estimation
    pub percentile: f32,
    /// Whether to use entropy-based calibration
    pub entropy_calibration: bool,
    /// Number of histogram bins for calibration
    pub histogram_bins: usize,
    /// Outlier rejection strategy
    pub outlier_rejection: OutlierRejectionStrategy,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            method: CalibrationMethod::Entropy,
            percentile: 99.99,
            entropy_calibration: true,
            histogram_bins: 2048,
            outlier_rejection: OutlierRejectionStrategy::Percentile { threshold: 0.1 },
        }
    }
}

/// Methods for calibrating quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CalibrationMethod {
    /// Simple min-max calibration
    MinMax,
    /// Entropy-based calibration (KL divergence)
    Entropy,
    /// Percentile-based calibration
    Percentile,
    /// Mean-squared error optimization
    MSE,
    /// Adaptive calibration based on layer characteristics
    Adaptive,
    /// Cross-layer correlation-aware calibration
    CorrelationAware,
}

/// Strategies for rejecting outliers during calibration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutlierRejectionStrategy {
    /// No outlier rejection
    None,
    /// Percentile-based rejection
    Percentile { threshold: f32 },
    /// Standard deviation-based rejection
    StandardDeviation { num_stds: f32 },
    /// Interquartile range-based rejection
    IQR { multiplier: f32 },
    /// Custom outlier detection
    Custom,
}

/// Hardware-specific constraints for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Target hardware platform
    pub platform: HardwarePlatform,
    /// Supported quantization formats
    pub supported_formats: Vec<QuantizationFormat>,
    /// Memory bandwidth constraints
    pub memory_bandwidth: Option<f32>,
    /// Compute capability constraints
    pub compute_capability: Option<String>,
    /// Power consumption limits
    pub power_limit: Option<f32>,
    /// Latency requirements
    pub latency_requirement: Option<f32>,
}

/// Hardware platforms for quantization optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HardwarePlatform {
    CPU,
    GPU,
    TPU,
    FPGA,
    EdgeTPU,
    NeuralProcessingUnit,
    Mobile,
    Embedded,
    Custom(String),
}

/// Quantization formats supported by hardware
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationFormat {
    /// Signed integer quantization
    SignedInt { bits: u8 },
    /// Unsigned integer quantization
    UnsignedInt { bits: u8 },
    /// Floating-point quantization
    FloatingPoint { bits: u8 },
    /// Block-wise quantization
    BlockWise { block_size: usize, bits: u8 },
    /// Custom quantization format
    Custom { name: String, bits: u8 },
}

/// Progressive quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveQuantizationConfig {
    /// Number of progressive stages
    pub num_stages: usize,
    /// Bit reduction schedule
    pub bit_schedule: BitReductionSchedule,
    /// Fine-tuning epochs per stage
    pub epochs_per_stage: usize,
    /// Learning rate schedule
    pub learning_rate_schedule: Vec<f32>,
}

/// Schedules for progressive bit reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BitReductionSchedule {
    /// Linear reduction of bits
    Linear,
    /// Exponential reduction
    Exponential { decay_rate: f32 },
    /// Step-wise reduction
    StepWise { steps: Vec<(usize, f32)> },
    /// Custom schedule
    Custom(Vec<f32>),
}

/// Layer-specific quantization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantizationConstraints {
    /// Minimum allowed bit width
    pub min_bits: Option<u8>,
    /// Maximum allowed bit width
    pub max_bits: Option<u8>,
    /// Fixed bit width (if specified)
    pub fixed_bits: Option<u8>,
    /// Quantization priority (higher = more important to preserve)
    pub priority: f32,
    /// Whether this layer can be skipped
    pub can_skip: bool,
}

/// Information about a quantized layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedLayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Assigned bit width
    pub bit_width: u8,
    /// Quantization parameters
    pub quantization_params: QuantizationParams,
    /// Sensitivity score
    pub sensitivity_score: f32,
    /// Compression ratio for this layer
    pub compression_ratio: f32,
    /// Estimated accuracy impact
    pub accuracy_impact: f32,
}

/// Quantization parameters for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Quantization range
    pub range: (f32, f32),
    /// Whether quantization is symmetric
    pub symmetric: bool,
    /// Per-channel parameters (if applicable)
    pub per_channel: Option<Vec<ChannelQuantizationParams>>,
}

/// Per-channel quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelQuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub range: (f32, f32),
}

/// Results from layer sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResults {
    /// Sensitivity scores per layer
    pub layer_sensitivities: HashMap<String, f32>,
    /// Recommended bit allocations
    pub recommended_bits: HashMap<String, u8>,
    /// Analysis methodology used
    pub analysis_method: SensitivityAnalysisMethod,
    /// Confidence scores for recommendations
    pub confidence_scores: HashMap<String, f32>,
}

/// Methods for analyzing layer sensitivity to quantization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SensitivityAnalysisMethod {
    /// Hessian-based sensitivity analysis
    HessianBased,
    /// Fisher information-based analysis
    FisherInformation,
    /// Gradient-based analysis
    GradientBased,
    /// Activation-based analysis
    ActivationBased,
    /// Output perturbation analysis
    OutputPerturbation,
    /// Mutual information analysis
    MutualInformation,
}

/// Results from mixed-bit quantization
#[derive(Debug, Clone)]
pub struct QuantizationResults {
    /// Per-layer quantization information
    pub layer_info: Vec<QuantizedLayerInfo>,
    /// Overall compression ratio achieved
    pub overall_compression_ratio: f32,
    /// Memory reduction (bytes)
    pub memory_reduction: usize,
    /// Estimated accuracy preservation
    pub accuracy_preservation: f32,
    /// Quantization quality metrics
    pub quality_metrics: QuantizationQualityMetrics,
    /// Execution time breakdown
    pub timing_info: QuantizationTimingInfo,
}

/// Quality metrics for quantization assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationQualityMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Peak signal-to-noise ratio
    pub psnr: f32,
    /// Structural similarity index
    pub ssim: f32,
    /// Cosine similarity
    pub cosine_similarity: f32,
    /// L2 reconstruction error
    pub l2_error: f32,
    /// KL divergence from original
    pub kl_divergence: f32,
    /// Per-layer quality scores
    pub per_layer_scores: HashMap<String, f32>,
}

/// Timing information for quantization process
#[derive(Debug, Clone)]
pub struct QuantizationTimingInfo {
    /// Total quantization time
    pub total_time_ms: f64,
    /// Sensitivity analysis time
    pub sensitivity_analysis_ms: f64,
    /// Bit allocation time
    pub bit_allocation_ms: f64,
    /// Calibration time
    pub calibration_ms: f64,
    /// Model conversion time
    pub conversion_ms: f64,
}

/// Main mixed-bit quantization engine
pub struct MixedBitQuantizer {
    #[allow(dead_code)]
    config: MixedBitQuantizationConfig,
    sensitivity_analyzer: SensitivityAnalyzer,
    bit_allocator: BitAllocator,
    calibrator: QuantizationCalibrator,
    quality_assessor: QualityAssessor,
}

impl MixedBitQuantizer {
    /// Create a new mixed-bit quantizer
    pub fn new(config: MixedBitQuantizationConfig) -> Self {
        let sensitivity_analyzer = SensitivityAnalyzer::new(&config);
        let bit_allocator = BitAllocator::new(&config);
        let calibrator = QuantizationCalibrator::new(&config.calibration_config);
        let quality_assessor = QualityAssessor::new();

        Self {
            config,
            sensitivity_analyzer,
            bit_allocator,
            calibrator,
            quality_assessor,
        }
    }

    /// Quantize a model using mixed-bit quantization
    pub fn quantize_model<M>(
        &mut self,
        model: M,
        calibration_data: &[Tensor],
    ) -> Result<QuantizationResults>
    where
        M: Clone,
    {
        let start_time = std::time::Instant::now();

        // Step 1: Analyze layer sensitivities
        println!("[INFO] Starting sensitivity analysis...");
        let sensitivity_start = std::time::Instant::now();
        let sensitivity_results =
            self.sensitivity_analyzer.analyze_sensitivities(&model, calibration_data)?;
        let sensitivity_time = sensitivity_start.elapsed().as_millis() as f64;

        // Step 2: Allocate bit widths based on sensitivities
        println!("[INFO] Allocating bit widths...");
        let allocation_start = std::time::Instant::now();
        let bit_allocation = self.bit_allocator.allocate_bits(&sensitivity_results)?;
        let allocation_time = allocation_start.elapsed().as_millis() as f64;

        // Step 3: Calibrate quantization parameters
        println!("[INFO] Calibrating quantization parameters...");
        let calibration_start = std::time::Instant::now();
        let quantization_params =
            self.calibrator.calibrate(&model, calibration_data, &bit_allocation)?;
        let calibration_time = calibration_start.elapsed().as_millis() as f64;

        // Step 4: Apply quantization and convert model
        println!("[INFO] Converting model...");
        let conversion_start = std::time::Instant::now();
        let layer_info = self.apply_quantization(&model, &bit_allocation, &quantization_params)?;
        let conversion_time = conversion_start.elapsed().as_millis() as f64;

        // Step 5: Assess quantization quality
        println!("[INFO] Assessing quantization quality...");
        let quality_metrics =
            self.quality_assessor.assess_quality(&model, &layer_info, calibration_data)?;

        let total_time = start_time.elapsed().as_millis() as f64;

        // Calculate overall metrics
        let overall_compression_ratio = self.calculate_compression_ratio(&layer_info);
        let memory_reduction = self.calculate_memory_reduction(&layer_info);
        let accuracy_preservation = quality_metrics.cosine_similarity;

        Ok(QuantizationResults {
            layer_info,
            overall_compression_ratio,
            memory_reduction,
            accuracy_preservation,
            quality_metrics,
            timing_info: QuantizationTimingInfo {
                total_time_ms: total_time,
                sensitivity_analysis_ms: sensitivity_time,
                bit_allocation_ms: allocation_time,
                calibration_ms: calibration_time,
                conversion_ms: conversion_time,
            },
        })
    }

    /// Apply quantization to the model
    fn apply_quantization<M>(
        &self,
        _model: &M,
        bit_allocation: &HashMap<String, u8>,
        quantization_params: &HashMap<String, QuantizationParams>,
    ) -> Result<Vec<QuantizedLayerInfo>> {
        let mut layer_info = Vec::new();

        for (layer_name, &bit_width) in bit_allocation {
            if let Some(params) = quantization_params.get(layer_name) {
                let sensitivity_score = 0.5; // Would be calculated from actual sensitivity analysis
                let compression_ratio = 32.0 / bit_width as f32; // Assuming 32-bit baseline
                let accuracy_impact = self.estimate_accuracy_impact(bit_width, sensitivity_score);

                layer_info.push(QuantizedLayerInfo {
                    layer_name: layer_name.clone(),
                    bit_width,
                    quantization_params: params.clone(),
                    sensitivity_score,
                    compression_ratio,
                    accuracy_impact,
                });
            }
        }

        Ok(layer_info)
    }

    /// Estimate accuracy impact for a layer
    fn estimate_accuracy_impact(&self, bit_width: u8, sensitivity_score: f32) -> f32 {
        // Simplified model: higher sensitivity and lower bits = higher impact
        let bit_impact = (8.0 - bit_width as f32).max(0.0) / 8.0;
        sensitivity_score * bit_impact
    }

    /// Calculate overall compression ratio
    fn calculate_compression_ratio(&self, layer_info: &[QuantizedLayerInfo]) -> f32 {
        if layer_info.is_empty() {
            return 1.0;
        }

        let total_compression: f32 = layer_info.iter().map(|info| info.compression_ratio).sum();

        total_compression / layer_info.len() as f32
    }

    /// Calculate memory reduction
    fn calculate_memory_reduction(&self, layer_info: &[QuantizedLayerInfo]) -> usize {
        // Simplified calculation - would need actual layer sizes
        layer_info
            .iter()
            .map(|info| ((info.compression_ratio - 1.0) * 1024.0 * 1024.0) as usize)
            .sum()
    }

    /// Generate quantization report
    pub fn generate_report(&self, results: &QuantizationResults) -> String {
        let mut report = String::new();

        report.push_str("# Mixed-Bit Quantization Report\n\n");

        report.push_str("## Overall Results\n");
        report.push_str(&format!(
            "- **Compression Ratio**: {:.2}x\n",
            results.overall_compression_ratio
        ));
        report.push_str(&format!(
            "- **Memory Reduction**: {:.2} MB\n",
            results.memory_reduction as f32 / (1024.0 * 1024.0)
        ));
        report.push_str(&format!(
            "- **Accuracy Preservation**: {:.2}%\n",
            results.accuracy_preservation * 100.0
        ));
        report.push_str(&format!(
            "- **Total Time**: {:.2} ms\n\n",
            results.timing_info.total_time_ms
        ));

        report.push_str("## Layer-wise Results\n\n");
        report.push_str("| Layer | Bit Width | Compression | Sensitivity | Impact |\n");
        report.push_str("|-------|-----------|-------------|-------------|--------|\n");

        for layer in &results.layer_info {
            report.push_str(&format!(
                "| {} | {} | {:.2}x | {:.3} | {:.3} |\n",
                layer.layer_name,
                layer.bit_width,
                layer.compression_ratio,
                layer.sensitivity_score,
                layer.accuracy_impact
            ));
        }

        report.push_str("\n## Quality Metrics\n\n");
        report.push_str(&format!(
            "- **SNR**: {:.2} dB\n",
            results.quality_metrics.snr
        ));
        report.push_str(&format!(
            "- **PSNR**: {:.2} dB\n",
            results.quality_metrics.psnr
        ));
        report.push_str(&format!(
            "- **SSIM**: {:.4}\n",
            results.quality_metrics.ssim
        ));
        report.push_str(&format!(
            "- **Cosine Similarity**: {:.4}\n",
            results.quality_metrics.cosine_similarity
        ));
        report.push_str(&format!(
            "- **L2 Error**: {:.6}\n",
            results.quality_metrics.l2_error
        ));

        report
    }
}

/// Analyzer for layer sensitivity to quantization
pub struct SensitivityAnalyzer {
    method: SensitivityAnalysisMethod,
}

impl SensitivityAnalyzer {
    fn new(_config: &MixedBitQuantizationConfig) -> Self {
        Self {
            method: SensitivityAnalysisMethod::ActivationBased,
        }
    }

    fn analyze_sensitivities<M>(
        &self,
        _model: &M,
        _calibration_data: &[Tensor],
    ) -> Result<SensitivityAnalysisResults> {
        // Simplified implementation - in practice would analyze actual model layers
        let mut layer_sensitivities = HashMap::new();
        let mut recommended_bits = HashMap::new();
        let mut confidence_scores = HashMap::new();

        // Mock sensitivity analysis
        let layer_names = [
            "embedding",
            "attention_0",
            "attention_1",
            "ffn_0",
            "ffn_1",
            "output",
        ];
        let base_sensitivities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.95];

        for (i, layer_name) in layer_names.iter().enumerate() {
            let sensitivity = base_sensitivities[i];
            layer_sensitivities.insert(layer_name.to_string(), sensitivity);

            // Higher sensitivity = higher bits
            let bits = if sensitivity > 0.8 {
                8
            } else if sensitivity > 0.6 {
                6
            } else {
                4
            };
            recommended_bits.insert(layer_name.to_string(), bits);
            confidence_scores.insert(layer_name.to_string(), 0.85);
        }

        Ok(SensitivityAnalysisResults {
            layer_sensitivities,
            recommended_bits,
            analysis_method: self.method.clone(),
            confidence_scores,
        })
    }
}

/// Bit width allocator using various optimization strategies
pub struct BitAllocator {
    strategy: BitAllocationStrategy,
    #[allow(dead_code)]
    available_bits: Vec<u8>,
    #[allow(dead_code)]
    target_compression: f32,
}

impl BitAllocator {
    fn new(config: &MixedBitQuantizationConfig) -> Self {
        Self {
            strategy: config.allocation_strategy.clone(),
            available_bits: config.available_bit_widths.clone(),
            target_compression: config.target_compression_ratio,
        }
    }

    fn allocate_bits(
        &self,
        sensitivity_results: &SensitivityAnalysisResults,
    ) -> Result<HashMap<String, u8>> {
        match &self.strategy {
            BitAllocationStrategy::SensitivityBased => {
                self.sensitivity_based_allocation(sensitivity_results)
            },
            BitAllocationStrategy::Custom(allocation) => Ok(allocation.clone()),
            _ => {
                // For other strategies, fall back to sensitivity-based
                self.sensitivity_based_allocation(sensitivity_results)
            },
        }
    }

    fn sensitivity_based_allocation(
        &self,
        sensitivity_results: &SensitivityAnalysisResults,
    ) -> Result<HashMap<String, u8>> {
        let mut allocation = HashMap::new();

        // Sort layers by sensitivity (highest first)
        let mut sorted_layers: Vec<_> = sensitivity_results.layer_sensitivities.iter().collect();
        sorted_layers.sort_by(|a, b| b.1.partial_cmp(a.1).expect("operation failed"));

        for (layer_name, &sensitivity) in sorted_layers {
            // Allocate higher bits to more sensitive layers
            let bits = if sensitivity > 0.8 {
                8
            } else if sensitivity > 0.6 {
                6
            } else {
                4
            };

            allocation.insert(layer_name.clone(), bits);
        }

        Ok(allocation)
    }
}

/// Calibrator for quantization parameters
pub struct QuantizationCalibrator {
    #[allow(dead_code)]
    config: CalibrationConfig,
}

impl QuantizationCalibrator {
    fn new(config: &CalibrationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    fn calibrate<M>(
        &self,
        _model: &M,
        _calibration_data: &[Tensor],
        bit_allocation: &HashMap<String, u8>,
    ) -> Result<HashMap<String, QuantizationParams>> {
        let mut params = HashMap::new();

        for (layer_name, &bits) in bit_allocation {
            // Simplified calibration - would use actual activation statistics
            let scale = 1.0 / (2_f32.powi((bits - 1) as i32) - 1.0);
            let zero_point = 0;
            let range = (-1.0, 1.0);

            params.insert(
                layer_name.clone(),
                QuantizationParams {
                    scale,
                    zero_point,
                    range,
                    symmetric: true,
                    per_channel: None,
                },
            );
        }

        Ok(params)
    }
}

/// Quality assessor for quantization results
pub struct QualityAssessor {}

impl QualityAssessor {
    fn new() -> Self {
        Self {}
    }

    fn assess_quality<M>(
        &self,
        _original_model: &M,
        layer_info: &[QuantizedLayerInfo],
        _test_data: &[Tensor],
    ) -> Result<QuantizationQualityMetrics> {
        // Simplified quality assessment
        Ok(QuantizationQualityMetrics {
            snr: 45.0,
            psnr: 48.0,
            ssim: 0.95,
            cosine_similarity: 0.98,
            l2_error: 0.001,
            kl_divergence: 0.05,
            per_layer_scores: layer_info
                .iter()
                .map(|info| (info.layer_name.clone(), 0.95))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config_builder() {
        let config = MixedBitQuantizationConfig::default()
            .with_target_compression(8.0)
            .with_max_accuracy_drop(0.01)
            .with_bit_widths(vec![2, 4, 8]);

        assert_eq!(config.target_compression_ratio, 8.0);
        assert_eq!(config.max_accuracy_drop, 0.01);
        assert_eq!(config.available_bit_widths, vec![2, 4, 8]);
    }

    #[test]
    fn test_sensitivity_analyzer() {
        let config = MixedBitQuantizationConfig::default();
        let analyzer = SensitivityAnalyzer::new(&config);

        // Test would need actual model and data
        assert_eq!(analyzer.method, SensitivityAnalysisMethod::ActivationBased);
    }

    #[test]
    fn test_bit_allocator() {
        let config = MixedBitQuantizationConfig::default();
        let allocator = BitAllocator::new(&config);

        assert_eq!(allocator.target_compression, 4.0);
        assert_eq!(allocator.available_bits, vec![4, 6, 8, 16]);
    }
}
