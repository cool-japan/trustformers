//! Tests for mixed-bit quantization.

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

    // Sensitivity is measured by perturbing each layer and observing the output.
    assert_eq!(
        analyzer.method,
        SensitivityAnalysisMethod::OutputPerturbation
    );
    // The probe uses the narrowest configured grid.
    assert_eq!(analyzer.probe_bits, 4);
}

#[test]
fn test_bit_allocator() {
    let config = MixedBitQuantizationConfig::default();
    let allocator = BitAllocator::new(&config);

    assert_eq!(allocator.target_compression, 4.0);
    assert_eq!(allocator.available_bits, vec![4, 6, 8, 16]);
}

#[test]
fn test_config_default_values() {
    let config = MixedBitQuantizationConfig::default();
    assert!((config.target_compression_ratio - 4.0).abs() < f32::EPSILON);
    assert!((config.max_accuracy_drop - 0.02).abs() < f32::EPSILON);
    assert_eq!(config.available_bit_widths, vec![4, 6, 8, 16]);
    assert_eq!(
        config.allocation_strategy,
        BitAllocationStrategy::SensitivityBased
    );
    assert!(config.gradient_free_optimization);
    assert!(config.progressive_quantization.is_none());
    assert!(config.layer_constraints.is_empty());
    assert!(config.hardware_constraints.is_none());
}

#[test]
fn test_config_chaining() {
    let config = MixedBitQuantizationConfig::default()
        .with_target_compression(16.0)
        .with_max_accuracy_drop(0.05)
        .with_bit_widths(vec![2, 4, 8, 16]);
    assert!((config.target_compression_ratio - 16.0).abs() < f32::EPSILON);
    assert!((config.max_accuracy_drop - 0.05).abs() < f32::EPSILON);
    assert_eq!(config.available_bit_widths, vec![2, 4, 8, 16]);
}

#[test]
fn test_bit_allocation_strategy_variants() {
    let strats = vec![
        BitAllocationStrategy::SensitivityBased,
        BitAllocationStrategy::ReinforcementLearning,
        BitAllocationStrategy::EvolutionaryAlgorithm,
        BitAllocationStrategy::GreedySearch,
        BitAllocationStrategy::MixedIntegerProgramming,
        BitAllocationStrategy::NeuralArchitectureSearch,
        BitAllocationStrategy::ParetoOptimal,
    ];
    for strat in &strats {
        let _ = format!("{:?}", strat);
    }
}

#[test]
fn test_bit_allocation_strategy_custom() {
    let mut custom_map = HashMap::new();
    custom_map.insert("layer1".to_string(), 4u8);
    custom_map.insert("layer2".to_string(), 8u8);
    let strat = BitAllocationStrategy::Custom(custom_map.clone());
    match strat {
        BitAllocationStrategy::Custom(m) => {
            assert_eq!(m.len(), 2);
            assert_eq!(m["layer1"], 4);
        },
        _ => panic!("Expected Custom variant"),
    }
}

#[test]
fn test_calibration_config_default() {
    let config = CalibrationConfig::default();
    assert_eq!(config.num_samples, 1000);
    assert!((config.percentile - 99.99).abs() < 0.1);
    assert!(config.entropy_calibration);
}

#[test]
fn test_sensitivity_analysis_method_eq() {
    assert_eq!(
        SensitivityAnalysisMethod::HessianBased,
        SensitivityAnalysisMethod::HessianBased
    );
    assert_ne!(
        SensitivityAnalysisMethod::HessianBased,
        SensitivityAnalysisMethod::GradientBased
    );
}

#[test]
fn test_quantization_params_creation() {
    let params = QuantizationParams {
        scale: 0.01,
        zero_point: 128,
        range: (-1.0, 1.0),
        symmetric: true,
        per_channel: None,
    };
    assert!((params.scale - 0.01).abs() < f32::EPSILON);
    assert_eq!(params.zero_point, 128);
    assert!(params.symmetric);
    assert!(params.per_channel.is_none());
}

#[test]
fn test_quantization_params_per_channel() {
    let channel_params = vec![
        ChannelQuantizationParams {
            scale: 0.01,
            zero_point: 0,
            range: (-1.0, 1.0),
        },
        ChannelQuantizationParams {
            scale: 0.02,
            zero_point: 0,
            range: (-2.0, 2.0),
        },
    ];
    let params = QuantizationParams {
        scale: 0.015,
        zero_point: 0,
        range: (-2.0, 2.0),
        symmetric: true,
        per_channel: Some(channel_params),
    };
    assert!(params.per_channel.is_some());
    assert_eq!(
        params.per_channel.as_ref().expect("channel params").len(),
        2
    );
}

#[test]
fn test_quantized_layer_info_creation() {
    let info = QuantizedLayerInfo {
        layer_name: "encoder.layer.0.attention".to_string(),
        bit_width: 8,
        quantization_params: QuantizationParams {
            scale: 0.01,
            zero_point: 0,
            range: (-1.0, 1.0),
            symmetric: true,
            per_channel: None,
        },
        sensitivity_score: 0.8,
        compression_ratio: 4.0,
        accuracy_impact: 0.01,
    };
    assert_eq!(info.bit_width, 8);
    assert!((info.sensitivity_score - 0.8).abs() < f32::EPSILON);
    assert!((info.compression_ratio - 4.0).abs() < f32::EPSILON);
}

#[test]
fn test_quantization_quality_metrics() {
    let metrics = QuantizationQualityMetrics {
        snr: 45.0,
        psnr: 48.0,
        ssim: Some(0.95),
        cosine_similarity: 0.98,
        l2_error: 0.001,
        kl_divergence: Some(0.05),
        per_layer_scores: HashMap::new(),
    };
    assert!(metrics.snr > 0.0);
    assert!(metrics.ssim.is_some_and(|ssim| (0.0..=1.0).contains(&ssim)));
    assert!(metrics.cosine_similarity >= 0.0 && metrics.cosine_similarity <= 1.0);
}

#[test]
fn test_layer_constraints() {
    let constraints = LayerQuantizationConstraints {
        min_bits: Some(4),
        max_bits: Some(16),
        fixed_bits: None,
        priority: 0.9,
        can_skip: false,
    };
    assert_eq!(constraints.min_bits, Some(4));
    assert_eq!(constraints.max_bits, Some(16));
    assert!(constraints.fixed_bits.is_none());
    assert!(!constraints.can_skip);
}

#[test]
fn test_layer_constraints_fixed_bits() {
    let constraints = LayerQuantizationConstraints {
        min_bits: None,
        max_bits: None,
        fixed_bits: Some(8),
        priority: 1.0,
        can_skip: false,
    };
    assert_eq!(constraints.fixed_bits, Some(8));
}

#[test]
fn test_mixed_bit_quantizer_creation() {
    let config = MixedBitQuantizationConfig::default();
    let _quantizer = MixedBitQuantizer::new(config);
}

#[test]
fn test_quantizer_with_custom_config() {
    let config = MixedBitQuantizationConfig::default()
        .with_target_compression(8.0)
        .with_max_accuracy_drop(0.05)
        .with_bit_widths(vec![2, 4, 8]);
    let _quantizer = MixedBitQuantizer::new(config);
}

#[test]
fn test_quantization_format_variants() {
    let formats = vec![
        QuantizationFormat::SignedInt { bits: 8 },
        QuantizationFormat::UnsignedInt { bits: 8 },
        QuantizationFormat::FloatingPoint { bits: 16 },
        QuantizationFormat::BlockWise {
            block_size: 32,
            bits: 4,
        },
        QuantizationFormat::Custom {
            name: "my_format".to_string(),
            bits: 6,
        },
    ];
    for fmt in &formats {
        let dbg = format!("{:?}", fmt);
        assert!(!dbg.is_empty());
    }
}

#[test]
fn test_progressive_quantization_config() {
    let config = ProgressiveQuantizationConfig {
        num_stages: 3,
        bit_schedule: BitReductionSchedule::Linear,
        epochs_per_stage: 5,
        learning_rate_schedule: vec![0.001, 0.0005, 0.0001],
    };
    assert_eq!(config.num_stages, 3);
    assert_eq!(config.epochs_per_stage, 5);
    assert_eq!(config.learning_rate_schedule.len(), 3);
}

#[test]
fn test_bit_reduction_schedule_variants() {
    let _linear = BitReductionSchedule::Linear;
    let _exp = BitReductionSchedule::Exponential { decay_rate: 0.9 };
    let _step = BitReductionSchedule::StepWise {
        steps: vec![(10, 0.5), (20, 0.25)],
    };
    let _custom = BitReductionSchedule::Custom(vec![1.0, 0.8, 0.6, 0.4]);
}

#[test]
fn test_sensitivity_analysis_results() {
    let mut sensitivities = HashMap::new();
    sensitivities.insert("layer0".to_string(), 0.3f32);
    sensitivities.insert("layer1".to_string(), 0.8f32);
    let mut bits = HashMap::new();
    bits.insert("layer0".to_string(), 4u8);
    bits.insert("layer1".to_string(), 8u8);
    let results = SensitivityAnalysisResults {
        layer_sensitivities: sensitivities,
        recommended_bits: bits,
        analysis_method: SensitivityAnalysisMethod::ActivationBased,
        confidence_scores: HashMap::new(),
    };
    assert_eq!(results.layer_sensitivities.len(), 2);
    assert_eq!(results.recommended_bits["layer0"], 4);
    assert_eq!(results.recommended_bits["layer1"], 8);
}

#[test]
fn test_quantization_timing_info() {
    let timing = QuantizationTimingInfo {
        total_time_ms: 1000.0,
        sensitivity_analysis_ms: 300.0,
        bit_allocation_ms: 100.0,
        calibration_ms: 400.0,
        conversion_ms: 200.0,
    };
    let sum = timing.sensitivity_analysis_ms
        + timing.bit_allocation_ms
        + timing.calibration_ms
        + timing.conversion_ms;
    assert!(sum <= timing.total_time_ms);
}

#[test]
fn test_channel_quantization_params() {
    let params = ChannelQuantizationParams {
        scale: 0.05,
        zero_point: 10,
        range: (-5.0, 5.0),
    };
    assert!((params.scale - 0.05).abs() < f32::EPSILON);
    assert_eq!(params.zero_point, 10);
    assert!((params.range.0 - (-5.0)).abs() < f32::EPSILON);
}

#[test]
fn test_outlier_rejection_strategy_variants() {
    let _none = OutlierRejectionStrategy::None;
    let _pct = OutlierRejectionStrategy::Percentile { threshold: 99.0 };
    let _iqr = OutlierRejectionStrategy::IQR { multiplier: 1.5 };
    let _std = OutlierRejectionStrategy::StandardDeviation { num_stds: 3.0 };
    let _custom = OutlierRejectionStrategy::Custom;
}

#[test]
fn test_calibration_method_variants() {
    let _minmax = CalibrationMethod::MinMax;
    let _entropy = CalibrationMethod::Entropy;
    let _pct = CalibrationMethod::Percentile;
    let _mse = CalibrationMethod::MSE;
    let _adaptive = CalibrationMethod::Adaptive;
}

// ---------------------------------------------------------------------------
// Regression tests: quantization must touch the model and measure the result
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::io::Read;
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearConfig;

impl Config for LinearConfig {
    fn architecture(&self) -> &'static str {
        "linear"
    }
}

/// A two-layer linear model whose parameters are really exposed.
struct LinearModel {
    config: LinearConfig,
    first: Tensor,
    second: Tensor,
}

impl LinearModel {
    fn new() -> Self {
        Self {
            config: LinearConfig,
            // Wide dynamic range: quantization error is clearly visible.
            first: Tensor::from_slice(&[1.0, -0.5, 0.25, 0.75, -1.0, 0.5], &[2, 3]).expect("first"),
            // Tiny weights: quantizing this layer barely moves the output.
            second: Tensor::from_slice(&[0.001, 0.002, 0.003], &[3, 1]).expect("second"),
        }
    }
}

impl Model for LinearModel {
    type Config = LinearConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
        input.matmul(&self.first)?.matmul(&self.second)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &LinearConfig {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        9
    }

    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        vec![
            ("first".to_string(), &self.first),
            ("second".to_string(), &self.second),
        ]
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        vec![
            ("first".to_string(), &mut self.first),
            ("second".to_string(), &mut self.second),
        ]
    }
}

/// A model that exposes nothing.
struct OpaqueModel {
    config: LinearConfig,
}

impl Model for OpaqueModel {
    type Config = LinearConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
        Ok(input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &LinearConfig {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        0
    }
}

fn calibration_batch() -> Vec<Tensor> {
    vec![
        Tensor::from_slice(&[1.0, 2.0], &[1, 2]).expect("input"),
        Tensor::from_slice(&[-1.0, 0.5], &[1, 2]).expect("input"),
    ]
}

#[test]
fn test_quantization_rewrites_the_real_weights() {
    let mut model = LinearModel::new();
    let before = model.first.data().expect("weights");

    let mut quantizer =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![4]));
    let results = quantizer
        .quantize_model(&mut model, &calibration_batch())
        .expect("quantization must succeed");

    let after = model.first.data().expect("weights");
    assert_ne!(before, after, "quantization must modify the weights");

    // The layer names come from the model, not from a hardcoded list.
    let names: Vec<&str> = results.layer_info.iter().map(|i| i.layer_name.as_str()).collect();
    assert_eq!(names, vec!["first", "second"]);
    assert!(!names.contains(&"embedding"), "no fabricated layer names");
    assert!(!names.contains(&"attention_0"));

    for info in &results.layer_info {
        assert_eq!(info.bit_width, 4);
        assert!((info.compression_ratio - 8.0).abs() < 1e-6);
        // The accuracy impact is the measured weight error, not a formula.
        assert!(info.accuracy_impact > 0.0);
    }

    assert_eq!(results.measurement_domain, MeasurementDomain::ModelOutputs);
    assert!(results.memory_reduction > 0);
}

#[test]
fn test_quality_metrics_are_measured_not_constant() {
    let mut coarse_model = LinearModel::new();
    let mut coarse =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![2]));
    let coarse_results = coarse
        .quantize_model(&mut coarse_model, &calibration_batch())
        .expect("quantization");

    let mut fine_model = LinearModel::new();
    let mut fine =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![16]));
    let fine_results = fine
        .quantize_model(&mut fine_model, &calibration_batch())
        .expect("quantization");

    // Metrics must respond to the bit width instead of being 45.0 / 0.98.
    assert!(
        fine_results.quality_metrics.snr > coarse_results.quality_metrics.snr,
        "16-bit SNR {} must beat 2-bit SNR {}",
        fine_results.quality_metrics.snr,
        coarse_results.quality_metrics.snr
    );
    assert!(
        fine_results.quality_metrics.l2_error < coarse_results.quality_metrics.l2_error,
        "16-bit error {} must be below 2-bit error {}",
        fine_results.quality_metrics.l2_error,
        coarse_results.quality_metrics.l2_error
    );
    assert!(
        fine_results.quality_metrics.cosine_similarity
            >= coarse_results.quality_metrics.cosine_similarity
    );
    assert!(
        (coarse_results.quality_metrics.snr - 45.0).abs() > 1e-6,
        "the SNR must not be the old hardcoded 45.0"
    );
    // SSIM has no meaning for activations and is not invented.
    assert!(fine_results.quality_metrics.ssim.is_none());
    // KL divergence is measured because calibration data was supplied.
    assert!(fine_results.quality_metrics.kl_divergence.is_some());
    assert!(
        fine_results.quality_metrics.kl_divergence.unwrap_or(1.0)
            <= coarse_results.quality_metrics.kl_divergence.unwrap_or(0.0) + 1e-6
    );
}

#[test]
fn test_sensitivity_reflects_each_layer_real_influence() {
    let mut model = LinearModel::new();
    let mut quantizer =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![4]));
    let results = quantizer
        .quantize_model(&mut model, &calibration_batch())
        .expect("quantization");

    let first = results
        .layer_info
        .iter()
        .find(|info| info.layer_name == "first")
        .expect("first layer");
    let second = results
        .layer_info
        .iter()
        .find(|info| info.layer_name == "second")
        .expect("second layer");

    // Both sensitivities are measured; the wide-range layer perturbs the output
    // no less than the tiny second layer, and neither is the old 0.9/0.8 table.
    assert!((0.0..=1.0).contains(&first.sensitivity_score));
    assert!((0.0..=1.0).contains(&second.sensitivity_score));
    assert!(
        first.sensitivity_score >= second.sensitivity_score,
        "the dominant layer must not be less sensitive: {} vs {}",
        first.sensitivity_score,
        second.sensitivity_score
    );
    assert!(
        (first.sensitivity_score - 0.9).abs() > 1e-6
            || (second.sensitivity_score - 0.8).abs() > 1e-6,
        "the sensitivities must not be the old hardcoded table"
    );
}

#[test]
fn test_quantization_without_calibration_data_measures_weights() {
    let mut model = LinearModel::new();
    let mut quantizer =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![8]));
    let results = quantizer.quantize_model(&mut model, &[]).expect("quantization");

    assert_eq!(results.measurement_domain, MeasurementDomain::Weights);
    assert!(
        results.quality_metrics.kl_divergence.is_none(),
        "with no data there are no output distributions to compare"
    );
    assert!(results.quality_metrics.snr > 0.0);
}

#[test]
fn test_quantizer_refuses_models_without_parameters() {
    let mut model = OpaqueModel {
        config: LinearConfig,
    };
    let mut quantizer = MixedBitQuantizer::new(MixedBitQuantizationConfig::default());
    let error = match quantizer.quantize_model(&mut model, &calibration_batch()) {
        Ok(_) => panic!("a model with no parameters cannot be quantized"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("named_tensors"), "{error}");
}

#[test]
fn test_report_contains_measured_values() {
    let mut model = LinearModel::new();
    let mut quantizer =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![8]));
    let results = quantizer
        .quantize_model(&mut model, &calibration_batch())
        .expect("quantization");
    let report = quantizer.generate_report(&results);

    assert!(report.contains("first"));
    assert!(report.contains("second"));
    assert!(
        report.contains("not applicable to activations"),
        "the report must say SSIM is not measured rather than print a constant"
    );
}

#[test]
fn test_calibration_method_changes_the_quantization_grid() {
    // One extreme outlier: min/max keeps it, percentile clips it away.
    let weights: Vec<f32> = std::iter::repeat_n(0.05f32, 63).chain(std::iter::once(20.0)).collect();

    let quantize_with = |method: CalibrationMethod| -> QuantizationResults {
        let mut model = LinearModel::new();
        model.first = Tensor::from_slice(&weights, &[8, 8]).expect("weights");
        model.second = Tensor::from_slice(&[0.5; 8], &[8, 1]).expect("weights");

        let mut config = MixedBitQuantizationConfig::default().with_bit_widths(vec![8]);
        config.calibration_config.method = method;
        config.calibration_config.percentile = 90.0;

        let mut quantizer = MixedBitQuantizer::new(config);
        quantizer
            .quantize_model(
                &mut model,
                &[Tensor::from_slice(&[1.0; 8], &[1, 8]).expect("input")],
            )
            .expect("quantization")
    };

    let min_max = quantize_with(CalibrationMethod::MinMax);
    let percentile = quantize_with(CalibrationMethod::Percentile);

    let grid_of = |results: &QuantizationResults| -> f32 {
        results
            .layer_info
            .iter()
            .find(|info| info.layer_name == "first")
            .map(|info| info.quantization_params.scale)
            .unwrap_or(f32::NAN)
    };

    assert!(
        grid_of(&percentile) < grid_of(&min_max),
        "percentile calibration must produce a finer grid than min/max: {} vs {}",
        grid_of(&percentile),
        grid_of(&min_max)
    );

    // The calibrated range is the represented range, not the raw min/max.
    let percentile_range = percentile
        .layer_info
        .iter()
        .find(|info| info.layer_name == "first")
        .map(|info| info.quantization_params.range)
        .expect("layer info");
    assert!(
        percentile_range.1 < 20.0,
        "the clipped range must exclude the outlier, got {percentile_range:?}"
    );
}

#[test]
fn test_calibrated_grid_is_the_one_applied_to_the_weights() {
    let mut model = LinearModel::new();
    let weights: Vec<f32> = std::iter::repeat_n(0.05f32, 63).chain(std::iter::once(20.0)).collect();
    model.first = Tensor::from_slice(&weights, &[8, 8]).expect("weights");
    model.second = Tensor::from_slice(&[0.5; 8], &[8, 1]).expect("weights");

    let mut config = MixedBitQuantizationConfig::default().with_bit_widths(vec![8]);
    config.calibration_config.method = CalibrationMethod::Percentile;
    config.calibration_config.percentile = 90.0;

    let mut quantizer = MixedBitQuantizer::new(config);
    let results = quantizer
        .quantize_model(
            &mut model,
            &[Tensor::from_slice(&[1.0; 8], &[1, 8]).expect("input")],
        )
        .expect("quantization");

    let info = results
        .layer_info
        .iter()
        .find(|info| info.layer_name == "first")
        .expect("layer info");

    // Every stored weight must sit on the calibrated grid, and the outlier must
    // have saturated at the calibrated bound.
    let stored = model.first.data().expect("weights");
    for value in &stored {
        let levels = value / info.quantization_params.scale;
        assert!(
            (levels - levels.round()).abs() < 1e-2,
            "{value} is not on the calibrated grid (scale {})",
            info.quantization_params.scale
        );
    }
    let largest = stored.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        largest < 20.0,
        "the outlier must have been clipped to the calibrated bound, got {largest}"
    );
}

#[test]
fn test_unimplemented_calibration_methods_are_rejected() {
    for method in [
        CalibrationMethod::Adaptive,
        CalibrationMethod::CorrelationAware,
    ] {
        let mut model = LinearModel::new();
        let mut config = MixedBitQuantizationConfig::default().with_bit_widths(vec![8]);
        config.calibration_config.method = method.clone();

        let mut quantizer = MixedBitQuantizer::new(config);
        let failed = quantizer.quantize_model(&mut model, &calibration_batch()).is_err();
        assert!(
            failed,
            "{method:?} has no implementation and must not be silently downgraded"
        );
    }
}

#[test]
fn test_quantization_preserves_parameter_dtypes() {
    use trustformers_core::tensor::DType;

    let mut model = LinearModel::new();
    model.first =
        Tensor::from_vec_with_dtype(vec![1.0, -0.5, 0.25, 0.75, -1.0, 0.5], &[2, 3], DType::F64)
            .expect("f64 weights");
    model.second = Tensor::from_slice(&[0.1, 0.2, 0.3], &[3, 1]).expect("weights");

    let mut quantizer =
        MixedBitQuantizer::new(MixedBitQuantizationConfig::default().with_bit_widths(vec![8]));
    quantizer.quantize_model(&mut model, &[]).expect("quantization");

    assert_eq!(
        model.first.dtype(),
        DType::F64,
        "an F64 parameter must stay F64 after quantization"
    );
    assert_eq!(model.second.dtype(), DType::F32);
}
