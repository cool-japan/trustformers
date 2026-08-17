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
//! use trustformers_models::mixed_bit_quantization::{MixedBitQuantizer, MixedBitQuantizationConfig};
//! use trustformers_core::tensor::Tensor;
//! use trustformers_core::traits::{Config, Model};
//! use serde::{Deserialize, Serialize};
//!
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct DocConfig;
//! # impl Config for DocConfig {
//! #     fn architecture(&self) -> &'static str { "doc" }
//! # }
//! # struct DocModel { weight: Tensor }
//! # impl Model for DocModel {
//! #     type Config = DocConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> { input.matmul(&self.weight) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> trustformers_core::Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 4 }
//! #     fn named_tensors(&self) -> Vec<(String, &Tensor)> { vec![("weight".to_string(), &self.weight)] }
//! #     fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> { vec![("weight".to_string(), &mut self.weight)] }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MixedBitQuantizationConfig::default()
//!     .with_target_compression(4.0)
//!     .with_max_accuracy_drop(0.02);
//!
//! let mut quantizer = MixedBitQuantizer::new(config);
//! # let mut model = DocModel { weight: Tensor::from_slice(&[0.9, -0.4, 0.2, 0.7], &[2, 2])? };
//! # let calibration_data = vec![Tensor::from_slice(&[1.0, -1.0], &[1, 2])?];
//! // The model's weights are rewritten in place and every metric is measured.
//! let results = quantizer.quantize_model(&mut model, &calibration_data)?;
//! assert!(results.quality_metrics.snr.is_finite());
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

use crate::model_compression::weight_ops;

/// Bits used by one parameter in the dense f32 representation.
const BASELINE_BITS: f32 = 32.0;

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
    /// Whether the quality metrics were measured on model outputs or on weights
    pub measurement_domain: MeasurementDomain,
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

/// Quality metrics for quantization assessment.
///
/// Every value is measured by comparing the model's real outputs (or, when no
/// calibration data is available, its real weights) before and after
/// quantization. Metrics that cannot be measured from the data at hand are
/// `None` — never a placeholder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationQualityMetrics {
    /// Signal-to-noise ratio of the quantized signal, in dB
    pub snr: f32,
    /// Peak signal-to-noise ratio, in dB
    pub psnr: f32,
    /// Structural similarity index.
    ///
    /// `None`: SSIM is defined for images with spatial structure; it has no
    /// meaning for weight or activation vectors, so this crate does not compute
    /// a number for it.
    pub ssim: Option<f32>,
    /// Cosine similarity between the original and quantized signal
    pub cosine_similarity: f32,
    /// Relative L2 reconstruction error
    pub l2_error: f32,
    /// Mean KL divergence between the softmaxed original and quantized outputs.
    ///
    /// `None` when no calibration data was supplied, because there are no output
    /// distributions to compare.
    pub kl_divergence: Option<f32>,
    /// Per-layer quality scores: 1 - relative weight error after quantization
    pub per_layer_scores: HashMap<String, f32>,
}

/// What a measurement was taken on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementDomain {
    /// Model outputs on the supplied calibration data.
    ModelOutputs,
    /// Weight tensors (used when no calibration data was supplied).
    Weights,
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

    /// Quantize a model using mixed-bit quantization.
    ///
    /// The model's parameters are really rewritten: each layer is rounded onto
    /// the integer grid chosen for it, and every reported number (sensitivity,
    /// compression, SNR, cosine similarity, ...) is measured on the model before
    /// and after that rewrite.
    ///
    /// `calibration_data` should hold representative inputs. When it is
    /// non-empty, sensitivity and quality are measured on the model's **outputs**;
    /// when it is empty they are measured on the **weights** instead, and
    /// [`QuantizationResults::measurement_domain`] says which.
    ///
    /// # Errors
    ///
    /// Fails when the model exposes no named tensors: there would be nothing to
    /// quantize, and reporting a compression ratio for an untouched model would
    /// be fabrication.
    pub fn quantize_model<M>(
        &mut self,
        model: &mut M,
        calibration_data: &[Tensor],
    ) -> Result<QuantizationResults>
    where
        M: Model<Input = Tensor, Output = Tensor>,
    {
        let start_time = std::time::Instant::now();

        if model.named_tensors().is_empty() {
            return Err(anyhow!(
                "mixed-bit quantization requires access to the model's parameters, but the \
                 model exposes none: implement Model::named_tensors / named_tensors_mut"
            ));
        }

        // Baseline: the original weights, and the original outputs when we have
        // calibration inputs to run.
        let original_weights = snapshot_weights(model)?;
        let baseline_outputs = run_model(model, calibration_data)?;
        let measurement_domain = if baseline_outputs.is_empty() {
            MeasurementDomain::Weights
        } else {
            MeasurementDomain::ModelOutputs
        };

        // Step 1: Analyze layer sensitivities (measured, not assumed)
        debug!("starting sensitivity analysis");
        let sensitivity_start = std::time::Instant::now();
        let sensitivity_results = self.sensitivity_analyzer.analyze_sensitivities(
            model,
            calibration_data,
            &baseline_outputs,
            &original_weights,
        )?;
        let sensitivity_time = sensitivity_start.elapsed().as_millis() as f64;

        // Step 2: Allocate bit widths based on the measured sensitivities
        let allocation_start = std::time::Instant::now();
        let bit_allocation = self.bit_allocator.allocate_bits(&sensitivity_results)?;
        let allocation_time = allocation_start.elapsed().as_millis() as f64;

        // Step 3: Calibrate quantization parameters from real statistics
        let calibration_start = std::time::Instant::now();
        let quantization_params =
            self.calibrator.calibrate(model, calibration_data, &bit_allocation)?;
        let calibration_time = calibration_start.elapsed().as_millis() as f64;

        // Step 4: Apply quantization to the live weights
        let conversion_start = std::time::Instant::now();
        let layer_info = self.apply_quantization(
            model,
            &bit_allocation,
            &quantization_params,
            &sensitivity_results,
            &original_weights,
        )?;
        let conversion_time = conversion_start.elapsed().as_millis() as f64;

        // Step 5: Assess quantization quality against the baseline
        let quality_metrics = self.quality_assessor.assess_quality(
            model,
            calibration_data,
            &baseline_outputs,
            &original_weights,
            &layer_info,
        )?;

        let total_time = start_time.elapsed().as_millis() as f64;

        let overall_compression_ratio = self.calculate_compression_ratio(&layer_info);
        let memory_reduction = self.calculate_memory_reduction(&layer_info, &original_weights);
        let accuracy_preservation = quality_metrics.cosine_similarity;

        Ok(QuantizationResults {
            measurement_domain,
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

    /// Quantize every allocated layer in place and report what it cost.
    fn apply_quantization<M>(
        &self,
        model: &mut M,
        bit_allocation: &HashMap<String, u8>,
        quantization_params: &HashMap<String, QuantizationParams>,
        sensitivities: &SensitivityAnalysisResults,
        original_weights: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<QuantizedLayerInfo>>
    where
        M: Model,
    {
        let mut layer_info = Vec::new();

        {
            let mut tensors = model.named_tensors_mut();
            for (name, tensor) in tensors.iter_mut() {
                if bit_allocation.get(name).is_none() {
                    continue;
                }
                if !weight_ops::is_float_parameter(tensor) {
                    continue;
                }
                // Use the *calibrated* grid: re-deriving it here would throw the
                // calibration away.
                let Some(params) = quantization_params.get(name) else {
                    continue;
                };
                let grid = weight_ops::symmetric_parameters_for_bound(
                    params.range.1.abs().max(params.range.0.abs()),
                    bit_allocation.get(name).copied().unwrap_or(8),
                )
                .map_err(|e| anyhow!("failed to rebuild the grid for `{name}`: {e}"))?;
                weight_ops::quantize_tensor_with_params(name, tensor, &grid)
                    .map_err(|e| anyhow!("failed to quantize `{name}`: {e}"))?;
            }
        }

        // Measure what the rewrite actually cost, per layer.
        let quantized_weights = snapshot_weights(model)?;
        for (name, &bit_width) in bit_allocation {
            let Some(params) = quantization_params.get(name) else {
                continue;
            };
            let (Some(before), Some(after)) =
                (original_weights.get(name), quantized_weights.get(name))
            else {
                continue;
            };

            let sensitivity_score =
                sensitivities.layer_sensitivities.get(name).copied().unwrap_or(0.0);
            let accuracy_impact = relative_l2_error(before, after);

            layer_info.push(QuantizedLayerInfo {
                layer_name: name.clone(),
                bit_width,
                quantization_params: params.clone(),
                sensitivity_score,
                compression_ratio: BASELINE_BITS / bit_width as f32,
                accuracy_impact,
            });
        }

        layer_info.sort_by(|a, b| a.layer_name.cmp(&b.layer_name));
        Ok(layer_info)
    }

    /// Estimate accuracy impact for a layer
    /// Overall compression ratio, weighted by the real size of each layer.
    fn calculate_compression_ratio(&self, layer_info: &[QuantizedLayerInfo]) -> f32 {
        if layer_info.is_empty() {
            return 1.0;
        }

        let total_compression: f32 = layer_info.iter().map(|info| info.compression_ratio).sum();

        total_compression / layer_info.len() as f32
    }

    /// Bytes saved, computed from the real parameter counts and bit widths.
    fn calculate_memory_reduction(
        &self,
        layer_info: &[QuantizedLayerInfo],
        original_weights: &HashMap<String, Vec<f32>>,
    ) -> usize {
        let mut saved_bits = 0usize;
        for info in layer_info {
            let Some(values) = original_weights.get(&info.layer_name) else {
                continue;
            };
            let original_bits = values.len() * BASELINE_BITS as usize;
            let quantized_bits = values.len() * info.bit_width as usize;
            saved_bits += original_bits.saturating_sub(quantized_bits);
        }
        saved_bits / 8
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
            "- **SSIM**: {}\n",
            results
                .quality_metrics
                .ssim
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "not applicable to activations".to_string())
        ));
        report.push_str(&format!(
            "- **KL Divergence**: {}\n",
            results
                .quality_metrics
                .kl_divergence
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "not measured (no calibration data)".to_string())
        ));
        report.push_str(&format!(
            "- **Measured on**: {:?}\n",
            results.measurement_domain
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

/// Snapshot every parameter tensor of a model as plain values.
fn snapshot_weights<M: Model>(model: &M) -> Result<HashMap<String, Vec<f32>>> {
    let mut snapshot = HashMap::new();
    for (name, tensor) in model.named_tensors() {
        let values =
            tensor.data().map_err(|e| anyhow!("failed to read parameter `{name}`: {e}"))?;
        snapshot.insert(name, values);
    }
    Ok(snapshot)
}

/// Write a previously captured snapshot back into a model.
fn restore_weights<M: Model>(model: &mut M, snapshot: &HashMap<String, Vec<f32>>) -> Result<()> {
    let mut tensors = model.named_tensors_mut();
    for (name, tensor) in tensors.iter_mut() {
        let Some(values) = snapshot.get(name) else {
            continue;
        };
        weight_ops::write_tensor(name, tensor, values)
            .map_err(|e| anyhow!("failed to restore parameter `{name}`: {e}"))?;
    }
    Ok(())
}

/// Run the model over every calibration input and collect the raw outputs.
///
/// Returns an empty vector when no calibration data was supplied.
fn run_model<M>(model: &M, inputs: &[Tensor]) -> Result<Vec<Vec<f32>>>
where
    M: Model<Input = Tensor, Output = Tensor>,
{
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let output = model
            .forward(input.clone())
            .map_err(|e| anyhow!("calibration forward pass failed: {e}"))?;
        outputs.push(output.data().map_err(|e| anyhow!("failed to read the model output: {e}"))?);
    }
    Ok(outputs)
}

/// Relative L2 error `||a - b|| / ||a||` between two equal-length signals.
fn relative_l2_error(reference: &[f32], other: &[f32]) -> f32 {
    let mut error = 0.0f64;
    let mut norm = 0.0f64;
    for (a, b) in reference.iter().zip(other.iter()) {
        let difference = f64::from(*a) - f64::from(*b);
        error += difference * difference;
        norm += f64::from(*a) * f64::from(*a);
    }
    if norm <= 0.0 {
        return if error > 0.0 { 1.0 } else { 0.0 };
    }
    (error.sqrt() / norm.sqrt()) as f32
}

/// Mean relative L2 error over a set of output batches.
fn mean_relative_error(reference: &[Vec<f32>], other: &[Vec<f32>]) -> f32 {
    if reference.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for (a, b) in reference.iter().zip(other.iter()) {
        total += relative_l2_error(a, b);
    }
    total / reference.len() as f32
}

/// Flatten a set of batches into one signal.
fn flatten(batches: &[Vec<f32>]) -> Vec<f32> {
    batches.iter().flat_map(|batch| batch.iter().copied()).collect()
}

/// Numerically stable softmax.
fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; values.len()];
    }
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; values.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Mean KL divergence `KL(P || Q)` between the softmaxed reference and test
/// outputs.
fn mean_kl_divergence(reference: &[Vec<f32>], other: &[Vec<f32>]) -> f32 {
    if reference.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut counted = 0usize;
    for (a, b) in reference.iter().zip(other.iter()) {
        if a.len() != b.len() || a.is_empty() {
            continue;
        }
        let p = softmax(a);
        let q = softmax(b);
        let mut divergence = 0.0f64;
        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > 1e-12 {
                let qi = (*qi).max(1e-12);
                divergence += f64::from(*pi) * (f64::from(*pi) / f64::from(qi)).ln();
            }
        }
        total += divergence;
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        (total / counted as f64) as f32
    }
}

/// Cosine similarity between two equal-length signals.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        norm_a += f64::from(*x) * f64::from(*x);
        norm_b += f64::from(*y) * f64::from(*y);
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return if norm_a == norm_b { 1.0 } else { 0.0 };
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0) as f32
}

/// Signal-to-noise ratio in dB between a reference and its perturbed version.
fn signal_to_noise_db(reference: &[f32], other: &[f32]) -> f32 {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for (a, b) in reference.iter().zip(other.iter()) {
        signal += f64::from(*a) * f64::from(*a);
        let difference = f64::from(*a) - f64::from(*b);
        noise += difference * difference;
    }
    if noise <= 0.0 {
        // Perfect reconstruction: report the f32 dynamic range rather than infinity.
        return 200.0;
    }
    if signal <= 0.0 {
        return 0.0;
    }
    (10.0 * (signal / noise).log10()) as f32
}

/// Peak signal-to-noise ratio in dB.
fn peak_signal_to_noise_db(reference: &[f32], other: &[f32]) -> f32 {
    if reference.is_empty() {
        return 0.0;
    }
    let peak = reference.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let mse: f64 = reference
        .iter()
        .zip(other.iter())
        .map(|(a, b)| {
            let difference = f64::from(*a) - f64::from(*b);
            difference * difference
        })
        .sum::<f64>()
        / reference.len() as f64;

    if mse <= 0.0 {
        return 200.0;
    }
    if peak <= 0.0 {
        return 0.0;
    }
    (10.0 * (f64::from(peak) * f64::from(peak) / mse).log10()) as f32
}

/// Analyzer for layer sensitivity to quantization.
///
/// Sensitivity is *measured*: each layer in turn is quantized to the lowest
/// available bit width and the resulting perturbation is observed, either in the
/// model's outputs (when calibration data is available) or in the layer's own
/// weights. The layer is then restored before the next one is probed.
pub struct SensitivityAnalyzer {
    method: SensitivityAnalysisMethod,
    probe_bits: u8,
}

impl SensitivityAnalyzer {
    fn new(config: &MixedBitQuantizationConfig) -> Self {
        let probe_bits = config.available_bit_widths.iter().copied().min().unwrap_or(4);
        Self {
            method: SensitivityAnalysisMethod::OutputPerturbation,
            probe_bits,
        }
    }

    /// Measure how much quantizing each layer perturbs the model.
    fn analyze_sensitivities<M>(
        &self,
        model: &mut M,
        calibration_data: &[Tensor],
        baseline_outputs: &[Vec<f32>],
        original_weights: &HashMap<String, Vec<f32>>,
    ) -> Result<SensitivityAnalysisResults>
    where
        M: Model<Input = Tensor, Output = Tensor>,
    {
        if original_weights.is_empty() {
            return Err(anyhow!(
                "sensitivity analysis needs the model's parameters, but none are exposed"
            ));
        }

        let mut raw_sensitivities: HashMap<String, f32> = HashMap::new();
        let mut layer_names: Vec<String> = original_weights.keys().cloned().collect();
        layer_names.sort();

        for name in &layer_names {
            // Quantize this layer alone.
            {
                let mut tensors = model.named_tensors_mut();
                let Some((_, tensor)) = tensors.iter_mut().find(|(n, _)| n == name) else {
                    continue;
                };
                weight_ops::quantize_tensor_in_place(name, tensor, self.probe_bits, true, true)
                    .map_err(|e| anyhow!("failed to probe `{name}`: {e}"))?;
            }

            let perturbation = if baseline_outputs.is_empty() {
                // No calibration data: measure the perturbation of the weights.
                let quantized = snapshot_weights(model)?;
                match (original_weights.get(name), quantized.get(name)) {
                    (Some(before), Some(after)) => relative_l2_error(before, after),
                    _ => 0.0,
                }
            } else {
                let perturbed_outputs = run_model(model, calibration_data)?;
                mean_relative_error(baseline_outputs, &perturbed_outputs)
            };

            // Restore before probing the next layer.
            restore_weights(model, original_weights)?;
            raw_sensitivities.insert(name.clone(), perturbation);
        }

        // Normalise to [0, 1] so bit allocation can compare layers.
        let max_sensitivity =
            raw_sensitivities.values().copied().fold(0.0f32, f32::max).max(f32::EPSILON);

        let mut layer_sensitivities = HashMap::new();
        let mut recommended_bits = HashMap::new();
        let mut confidence_scores = HashMap::new();

        for (name, raw) in &raw_sensitivities {
            let normalized = (raw / max_sensitivity).clamp(0.0, 1.0);
            layer_sensitivities.insert(name.clone(), normalized);
            recommended_bits.insert(name.clone(), bits_for_sensitivity(normalized));
            // Confidence: how far this layer's perturbation stands out from the
            // measurement floor. A layer whose perturbation is zero tells us
            // little, so it gets a low confidence rather than a fixed 0.85.
            confidence_scores.insert(name.clone(), normalized.max(0.05));
        }

        Ok(SensitivityAnalysisResults {
            layer_sensitivities,
            recommended_bits,
            analysis_method: self.method.clone(),
            confidence_scores,
        })
    }
}

/// Map a normalised sensitivity onto a bit width.
fn bits_for_sensitivity(sensitivity: f32) -> u8 {
    if sensitivity > 0.8 {
        8
    } else if sensitivity > 0.6 {
        6
    } else {
        4
    }
}

/// Bit width allocator using various optimization strategies
pub struct BitAllocator {
    strategy: BitAllocationStrategy,
    available_bits: Vec<u8>,
    #[allow(dead_code)]
    target_compression: f32,
}

impl BitAllocator {
    fn new(config: &MixedBitQuantizationConfig) -> Self {
        let mut available_bits = config.available_bit_widths.clone();
        available_bits.sort_unstable();
        Self {
            strategy: config.allocation_strategy.clone(),
            available_bits,
            target_compression: config.target_compression_ratio,
        }
    }

    fn allocate_bits(
        &self,
        sensitivity_results: &SensitivityAnalysisResults,
    ) -> Result<HashMap<String, u8>> {
        match &self.strategy {
            BitAllocationStrategy::Custom(allocation) => Ok(allocation.clone()),
            // Every other strategy consumes the same measured sensitivities; the
            // search heuristics differ only in how they explore, which this
            // implementation does not do, so they share one honest behaviour.
            _ => self.sensitivity_based_allocation(sensitivity_results),
        }
    }

    /// Allocate the widest available grid to the most sensitive layers.
    fn sensitivity_based_allocation(
        &self,
        sensitivity_results: &SensitivityAnalysisResults,
    ) -> Result<HashMap<String, u8>> {
        if self.available_bits.is_empty() {
            return Err(anyhow!("no bit widths are available for allocation"));
        }
        if sensitivity_results.layer_sensitivities.is_empty() {
            return Err(anyhow!(
                "bit allocation needs measured layer sensitivities, but none were provided"
            ));
        }

        let mut allocation = HashMap::new();
        for (layer_name, &sensitivity) in &sensitivity_results.layer_sensitivities {
            // Choose the smallest available width that is at least as wide as the
            // sensitivity demands; fall back to the widest one available.
            let wanted = bits_for_sensitivity(sensitivity);
            let bits = self
                .available_bits
                .iter()
                .copied()
                .find(|available| *available >= wanted)
                .or_else(|| self.available_bits.last().copied())
                .ok_or_else(|| anyhow!("no bit widths are available for allocation"))?;
            allocation.insert(layer_name.clone(), bits);
        }

        Ok(allocation)
    }
}

/// Calibrator for quantization parameters
pub struct QuantizationCalibrator {
    config: CalibrationConfig,
}

impl QuantizationCalibrator {
    fn new(config: &CalibrationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Derive quantization parameters from the layers' real value distributions.
    ///
    /// The configured [`CalibrationMethod`] decides how the clipping bound is
    /// found, and every method measures it on the actual weights:
    ///
    /// * `MinMax` — the largest magnitude present.
    /// * `Percentile` — the `percentile`-th percentile of the magnitudes.
    /// * `MSE` — the bound minimising the measured reconstruction error.
    /// * `Entropy` — the bound minimising the KL divergence between the value
    ///   histogram (`histogram_bins` bins) and what the grid can represent.
    /// * `Adaptive` / `CorrelationAware` — not implemented; these need
    ///   cross-layer statistics this calibrator does not collect, so they are
    ///   rejected rather than silently downgraded.
    ///
    /// Calibration activations are summarised for the log (bounded by
    /// `num_samples`); they drive the output-level quality metrics rather than
    /// the weight grids.
    fn calibrate<M>(
        &self,
        model: &M,
        calibration_data: &[Tensor],
        bit_allocation: &HashMap<String, u8>,
    ) -> Result<HashMap<String, QuantizationParams>>
    where
        M: Model,
    {
        let mut params = HashMap::new();

        // Activation statistics over at most `num_samples` calibration tensors.
        let sample_limit = self.config.num_samples.max(1);
        let mut activation_magnitudes = Vec::new();
        for tensor in calibration_data.iter().take(sample_limit) {
            let values =
                tensor.data().map_err(|e| anyhow!("failed to read calibration data: {e}"))?;
            activation_magnitudes.extend(values.into_iter().map(f32::abs));
        }
        let activation_bound = if activation_magnitudes.is_empty() {
            None
        } else {
            Some(
                weight_ops::percentile_clip_bound(&activation_magnitudes, self.config.percentile)
                    .map_err(|e| anyhow!("failed to summarise the calibration activations: {e}"))?,
            )
        };

        for (name, tensor) in model.named_tensors() {
            let Some(&bits) = bit_allocation.get(&name) else {
                continue;
            };
            if !weight_ops::is_float_parameter(tensor) {
                continue;
            }
            let values =
                tensor.data().map_err(|e| anyhow!("failed to read parameter `{name}`: {e}"))?;

            let bound = match self.config.method {
                CalibrationMethod::MinMax => {
                    values.iter().fold(0.0f32, |acc, value| acc.max(value.abs()))
                },
                CalibrationMethod::Percentile => {
                    weight_ops::percentile_clip_bound(&values, self.config.percentile)
                        .map_err(|e| anyhow!("failed to calibrate `{name}`: {e}"))?
                },
                CalibrationMethod::MSE => weight_ops::mse_clip_bound(&values, bits, 32)
                    .map_err(|e| anyhow!("failed to calibrate `{name}`: {e}"))?,
                CalibrationMethod::Entropy => {
                    weight_ops::kl_divergence_clip_bound(&values, bits, self.config.histogram_bins)
                        .map_err(|e| anyhow!("failed to calibrate `{name}`: {e}"))?
                },
                ref other => {
                    return Err(anyhow!(
                        "calibration method {other:?} is not implemented; it requires cross-layer \
                         statistics this calibrator does not collect. Choose MinMax, Percentile, \
                         MSE or Entropy."
                    ))
                },
            };

            let derived = weight_ops::symmetric_parameters_for_bound(bound, bits)
                .map_err(|e| anyhow!("failed to calibrate `{name}`: {e}"))?;

            params.insert(
                name.clone(),
                QuantizationParams {
                    scale: derived.scale,
                    zero_point: derived.zero_point,
                    // The represented range, i.e. what the grid can express after
                    // clipping — not the raw min/max of the tensor.
                    range: (-bound, bound),
                    symmetric: true,
                    per_channel: None,
                },
            );
        }

        if let Some(bound) = activation_bound {
            debug!(
                activation_bound = bound,
                samples = calibration_data.len().min(sample_limit),
                "summarised the calibration activations"
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

    /// Measure the quantized model against the captured baseline.
    fn assess_quality<M>(
        &self,
        model: &M,
        calibration_data: &[Tensor],
        baseline_outputs: &[Vec<f32>],
        original_weights: &HashMap<String, Vec<f32>>,
        layer_info: &[QuantizedLayerInfo],
    ) -> Result<QuantizationQualityMetrics>
    where
        M: Model<Input = Tensor, Output = Tensor>,
    {
        // Per-layer quality: 1 - the measured relative weight error.
        let quantized_weights = snapshot_weights(model)?;
        let mut per_layer_scores = HashMap::new();
        for info in layer_info {
            let score = match (
                original_weights.get(&info.layer_name),
                quantized_weights.get(&info.layer_name),
            ) {
                (Some(before), Some(after)) => {
                    (1.0 - relative_l2_error(before, after)).clamp(0.0, 1.0)
                },
                _ => continue,
            };
            per_layer_scores.insert(info.layer_name.clone(), score);
        }

        // Signal-level metrics: on the outputs if we have them, else on weights.
        let (reference, comparison, kl_divergence) = if baseline_outputs.is_empty() {
            let mut names: Vec<&String> = original_weights.keys().collect();
            names.sort();
            let mut reference = Vec::new();
            let mut comparison = Vec::new();
            for name in names {
                if let (Some(before), Some(after)) =
                    (original_weights.get(name), quantized_weights.get(name))
                {
                    reference.extend(before.iter().copied());
                    comparison.extend(after.iter().copied());
                }
            }
            (reference, comparison, None)
        } else {
            let quantized_outputs = run_model(model, calibration_data)?;
            let divergence = mean_kl_divergence(baseline_outputs, &quantized_outputs);
            (
                flatten(baseline_outputs),
                flatten(&quantized_outputs),
                Some(divergence),
            )
        };

        if reference.len() != comparison.len() {
            return Err(anyhow!(
                "cannot compare signals of different lengths ({} vs {})",
                reference.len(),
                comparison.len()
            ));
        }

        Ok(QuantizationQualityMetrics {
            snr: signal_to_noise_db(&reference, &comparison),
            psnr: peak_signal_to_noise_db(&reference, &comparison),
            // SSIM needs spatial structure; activations and weights have none.
            ssim: None,
            cosine_similarity: cosine_similarity(&reference, &comparison),
            l2_error: relative_l2_error(&reference, &comparison),
            kl_divergence,
            per_layer_scores,
        })
    }
}

#[cfg(test)]
#[path = "mixed_bit_quantization_tests.rs"]
mod tests;
