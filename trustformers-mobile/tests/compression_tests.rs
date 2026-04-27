//! Model compression tests for trustformers-mobile
//!
//! Tests quantization configs, pruning strategies, distillation configs,
//! and compression statistics without actual model weights.

use trustformers_mobile::compression::{
    CompressionSchedule, EarlyStoppingConfig, ProgressiveCompressionConfig,
    QualityPreservationConfig,
};
use trustformers_mobile::{
    CompressionConfig, DistillationConfig, DistillationStrategy, PruningStrategy, QualityMetric,
    QualityRecoveryStrategy, QuantizationPrecision, QuantizationStrategy,
};

fn make_int8_compression_config() -> CompressionConfig {
    CompressionConfig {
        target_compression_ratio: 0.25, // 4x compression
        quantization_strategy: QuantizationStrategy::Static(QuantizationPrecision::Int8),
        pruning_strategy: PruningStrategy::None,
        enable_distillation: false,
        distillation_config: None,
        progressive_compression: ProgressiveCompressionConfig {
            enabled: false,
            stages: 1,
            schedule: CompressionSchedule::Linear,
            validation_frequency: 100,
        },
        quality_preservation: QualityPreservationConfig {
            max_quality_loss: 0.05,
            quality_metrics: vec![QualityMetric::Perplexity],
            recovery_strategies: vec![QualityRecoveryStrategy::ReduceCompression],
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 5,
                min_improvement: 0.001,
                metric: QualityMetric::Perplexity,
            },
        },
        device_adaptive: false,
    }
}

fn make_int4_compression_config() -> CompressionConfig {
    CompressionConfig {
        target_compression_ratio: 0.125, // 8x compression
        quantization_strategy: QuantizationStrategy::Static(QuantizationPrecision::Int4),
        pruning_strategy: PruningStrategy::MagnitudeBased { sparsity: 0.3 },
        enable_distillation: true,
        distillation_config: Some(DistillationConfig {
            temperature: 4.0,
            distillation_weight: 0.7,
            hard_target_weight: 0.3,
            strategy: DistillationStrategy::OutputOnly,
            feature_matching: None,
        }),
        progressive_compression: ProgressiveCompressionConfig {
            enabled: true,
            stages: 3,
            schedule: CompressionSchedule::CosineAnnealing,
            validation_frequency: 50,
        },
        quality_preservation: QualityPreservationConfig {
            max_quality_loss: 0.10,
            quality_metrics: vec![QualityMetric::Accuracy],
            recovery_strategies: vec![QualityRecoveryStrategy::QualityFineTuning],
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 3,
                min_improvement: 0.001,
                metric: QualityMetric::Accuracy,
            },
        },
        device_adaptive: true,
    }
}

#[test]
fn test_quantization_precision_int8_config() {
    let config = make_int8_compression_config();
    assert_eq!(
        config.quantization_strategy,
        QuantizationStrategy::Static(QuantizationPrecision::Int8),
    );
}

#[test]
fn test_quantization_precision_int4_config() {
    let config = make_int4_compression_config();
    assert_eq!(
        config.quantization_strategy,
        QuantizationStrategy::Static(QuantizationPrecision::Int4),
    );
}

#[test]
fn test_int8_compression_ratio_approximately_4x() {
    let config = make_int8_compression_config();
    // INT8 vs FP32: target ratio ~0.25 (4x compression)
    assert!((config.target_compression_ratio - 0.25).abs() < 1e-6);
}

#[test]
fn test_int4_compression_ratio_approximately_8x() {
    let config = make_int4_compression_config();
    // INT4 vs FP32: target ratio ~0.125 (8x compression)
    assert!((config.target_compression_ratio - 0.125).abs() < 1e-6);
}

#[test]
fn test_pruning_strategy_none_variant() {
    let strategy = PruningStrategy::None;
    assert_eq!(strategy, PruningStrategy::None);
}

#[test]
fn test_pruning_strategy_magnitude_based_with_valid_sparsity() {
    let strategy = PruningStrategy::MagnitudeBased { sparsity: 0.5 };
    if let PruningStrategy::MagnitudeBased { sparsity } = strategy {
        assert!((sparsity - 0.5).abs() < 1e-6);
        assert!(
            sparsity > 0.0 && sparsity < 1.0,
            "sparsity must be in (0, 1)"
        );
    } else {
        panic!("expected MagnitudeBased variant");
    }
}

#[test]
fn test_pruning_strategy_structured_with_valid_ratio() {
    let strategy = PruningStrategy::Structured { ratio: 0.4 };
    if let PruningStrategy::Structured { ratio } = strategy {
        assert!(ratio > 0.0 && ratio < 1.0, "ratio must be in (0, 1)");
    } else {
        panic!("expected Structured variant");
    }
}

#[test]
fn test_distillation_config_temperature_positive() {
    let config = make_int4_compression_config();
    let distillation = config.distillation_config.expect("distillation config should exist");
    assert!(distillation.temperature > 0.0);
}

#[test]
fn test_distillation_config_weights_sum_to_one() {
    let config = make_int4_compression_config();
    let distillation = config.distillation_config.expect("distillation config should exist");
    let sum = distillation.distillation_weight + distillation.hard_target_weight;
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to ~1.0, got {sum}"
    );
}

#[test]
fn test_distillation_strategy_variants_exist() {
    let _output_only = DistillationStrategy::OutputOnly;
    let _feature = DistillationStrategy::FeatureLevel;
    let _attention = DistillationStrategy::AttentionTransfer;
    let _progressive = DistillationStrategy::Progressive;
    let _online = DistillationStrategy::Online;
}

#[test]
fn test_quantization_precision_variants_exist() {
    let _int1 = QuantizationPrecision::Int1;
    let _int2 = QuantizationPrecision::Int2;
    let _int4 = QuantizationPrecision::Int4;
    let _int8 = QuantizationPrecision::Int8;
    let _fp16 = QuantizationPrecision::FP16;
    let _bf16 = QuantizationPrecision::BF16;
    let _dynamic = QuantizationPrecision::Dynamic;
}

#[test]
fn test_quantization_strategy_variants_exist() {
    let _static_q = QuantizationStrategy::Static(QuantizationPrecision::Int8);
    let _dynamic = QuantizationStrategy::Dynamic;
    let _mixed = QuantizationStrategy::MixedPrecision;
    let _block = QuantizationStrategy::BlockWise;
    let _outlier = QuantizationStrategy::OutlierAware;
    let _adaptive = QuantizationStrategy::DeviceAdaptive;
}

#[test]
fn test_quality_preservation_max_loss_is_valid_fraction() {
    let config = make_int8_compression_config();
    let loss = config.quality_preservation.max_quality_loss;
    assert!(
        (0.0..=1.0).contains(&loss),
        "quality loss should be in [0,1], got {loss}"
    );
}

#[test]
fn test_quality_recovery_strategy_variants() {
    let _reduce = QualityRecoveryStrategy::ReduceCompression;
    let _increase = QualityRecoveryStrategy::IncreaseCapacity;
    let _finetune = QualityRecoveryStrategy::QualityFineTuning;
    let _rollback = QualityRecoveryStrategy::Rollback;
}

#[test]
fn test_compression_config_serialization_roundtrip() {
    let config = make_int8_compression_config();
    let json = serde_json::to_string(&config).expect("serialization should succeed");
    let restored: CompressionConfig =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert!((restored.target_compression_ratio - config.target_compression_ratio).abs() < 1e-6);
}

#[test]
fn test_compression_schedule_variants() {
    let _linear = CompressionSchedule::Linear;
    let _exponential = CompressionSchedule::Exponential;
    let _cosine = CompressionSchedule::CosineAnnealing;
    let _custom = CompressionSchedule::Custom;
}
