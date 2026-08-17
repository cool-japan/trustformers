//! Tests for the model compression pipeline.

use super::*;

#[test]
fn test_compression_config_default() {
    let config = CompressionConfig::default();
    assert_eq!(config.target_compression_ratio, 0.5);
    assert_eq!(config.strategies.len(), 1);
    assert!(
        !config.fine_tune,
        "fine-tuning needs a trainer, so it must be opt-in"
    );
    assert!(!config.progressive);
}

#[test]
fn test_simple_quantization_config() {
    let config = utils::simple_quantization_config(8);
    assert_eq!(config.strategies.len(), 1);

    if let CompressionStrategy::Quantization {
        bits,
        signed,
        symmetric,
    } = &config.strategies[0]
    {
        assert_eq!(*bits, 8);
        assert!(*signed);
        assert!(!*symmetric);
    } else {
        panic!("Expected Quantization strategy");
    }
}

#[test]
fn test_simple_pruning_config() {
    let config = utils::simple_pruning_config(0.5);
    assert_eq!(config.strategies.len(), 1);

    if let CompressionStrategy::UnstructuredPruning { sparsity, strategy } = &config.strategies[0] {
        assert_eq!(*sparsity, 0.5);
        assert!(matches!(strategy, PruningStrategy::Magnitude));
    } else {
        panic!("Expected UnstructuredPruning strategy");
    }
}

#[test]
fn test_combined_compression_config() {
    let config = utils::combined_compression_config(8, 0.3);
    assert_eq!(config.strategies.len(), 2);

    // First should be pruning
    if let CompressionStrategy::UnstructuredPruning { sparsity, .. } = &config.strategies[0] {
        assert_eq!(*sparsity, 0.3);
    } else {
        panic!("Expected UnstructuredPruning as first strategy");
    }

    // Second should be quantization
    if let CompressionStrategy::Quantization { bits, .. } = &config.strategies[1] {
        assert_eq!(*bits, 8);
    } else {
        panic!("Expected Quantization as second strategy");
    }
}

#[test]
fn test_progressive_compression_config() {
    let config = utils::progressive_compression_config(0.25, 3);
    assert_eq!(config.target_compression_ratio, 0.25);
    assert!(config.progressive);
    assert_eq!(config.progressive_stages, 3);
    assert_eq!(config.strategies.len(), 3);
}

#[test]
fn test_aggressive_compression_config() {
    let config = utils::aggressive_compression_config();
    assert_eq!(config.target_compression_ratio, 0.1);
    assert_eq!(config.strategies.len(), 5);
    assert!(config.fine_tune);
    assert_eq!(config.fine_tune_epochs, 5);
    assert_eq!(config.max_accuracy_drop, 0.05);
}

#[test]
fn test_estimate_compression_ratio() {
    let config = utils::simple_quantization_config(8);
    let ratio = utils::estimate_compression_ratio(&config);
    assert!((ratio - 0.25).abs() < 1e-6); // 8/32 = 0.25

    let pruning_config = utils::simple_pruning_config(0.5);
    let pruning_ratio = utils::estimate_compression_ratio(&pruning_config);
    assert!((pruning_ratio - 0.5).abs() < 1e-6); // 1 - 0.5 = 0.5
}

#[test]
fn test_compression_pipeline_creation() {
    let config = CompressionConfig::default();
    let pipeline = CompressionPipeline::new(config);
    assert!(pipeline.is_ok());

    let pipeline = pipeline.expect("operation failed");
    assert_eq!(pipeline.compression_stages.len(), 1);
    assert_eq!(pipeline.current_stage, 0);
}

#[test]
fn test_progressive_pipeline_creation() {
    let config = utils::progressive_compression_config(0.25, 3);
    let pipeline = CompressionPipeline::new(config);
    assert!(pipeline.is_ok());

    let pipeline = pipeline.expect("operation failed");
    assert_eq!(pipeline.compression_stages.len(), 3);
}

// ---------------------------------------------------------------------------
// Regression tests: compression must change the weights, not just the labels
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::io::Read;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyConfig;

impl Config for TinyConfig {
    fn architecture(&self) -> &'static str {
        "tiny"
    }
}

/// A model whose parameters are really exposed.
struct TinyModel {
    config: TinyConfig,
    weight: Tensor,
    bias: Tensor,
}

impl TinyModel {
    fn new(weight: &[f32], shape: &[usize], bias: &[f32]) -> Self {
        Self {
            config: TinyConfig,
            weight: Tensor::from_slice(weight, shape).expect("weight"),
            bias: Tensor::from_slice(bias, &[bias.len()]).expect("bias"),
        }
    }
}

impl Model for TinyModel {
    type Config = TinyConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        input.matmul(&self.weight)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &TinyConfig {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.weight.shape().iter().product::<usize>() + self.bias.shape()[0]
    }

    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        vec![
            ("weight".to_string(), &self.weight),
            ("bias".to_string(), &self.bias),
        ]
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        vec![
            ("weight".to_string(), &mut self.weight),
            ("bias".to_string(), &mut self.bias),
        ]
    }
}

/// A model that never exposes its parameters (the trait default).
struct OpaqueModel {
    config: TinyConfig,
}

impl Model for OpaqueModel {
    type Config = TinyConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        Ok(input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &TinyConfig {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        0
    }
}

fn sample_model() -> TinyModel {
    TinyModel::new(
        &[0.9, -0.05, 0.02, 0.8, -0.7, 0.01, 0.03, 0.6],
        &[4, 2],
        &[0.5, -0.4],
    )
}

#[test]
fn test_parameter_count_walks_the_real_tensors() {
    let compressed = CompressedModel::new(sample_model());
    assert_eq!(
        compressed.count_parameters().expect("parameter count"),
        10,
        "8 weights + 2 biases — not a hardcoded 1,000,000"
    );
    assert_eq!(compressed.parameter_count(), 10);
    assert_eq!(
        compressed.nonzero_parameter_count().expect("nonzero count"),
        10
    );
}

#[test]
fn test_pruning_actually_zeroes_weights() {
    let config = utils::simple_pruning_config(0.5);
    let pipeline = CompressionPipeline::new(config).expect("pipeline");

    let compressed = pipeline.compress(sample_model()).expect("compression");

    let remaining = compressed.nonzero_parameter_count().expect("nonzero count");
    assert_eq!(remaining, 5, "half of ten parameters must survive");

    let weights = compressed.model.weight.data().expect("weights");
    // The smallest magnitudes are the ones that were removed.
    assert_eq!(weights[1], 0.0);
    assert_eq!(weights[2], 0.0);
    assert_eq!(weights[5], 0.0);
    assert_eq!(weights[6], 0.0);
    assert!(weights[0] != 0.0 && weights[3] != 0.0 && weights[4] != 0.0);

    let analysis = compressed.analysis.as_ref().expect("analysis");
    assert_eq!(analysis.original_size, 10);
    assert_eq!(analysis.compressed_size, 5);
    assert!(
        (analysis.compression_ratio - 0.5).abs() < 1e-6,
        "got {}",
        analysis.compression_ratio
    );
    assert!(analysis.memory_reduction > 0);
    assert_eq!(analysis.layer_statistics["weight"].original_params, 8);
    assert!(analysis.layer_statistics["weight"].compressed_params < 8);
    assert!(analysis.latency_improvement.is_none());
}

#[test]
fn test_quantization_actually_rounds_weights() {
    let config = utils::simple_quantization_config(4);
    let pipeline = CompressionPipeline::new(config).expect("pipeline");

    let original = sample_model().weight.data().expect("weights");
    let compressed = pipeline.compress(sample_model()).expect("compression");
    let quantized = compressed.model.weight.data().expect("weights");

    assert_ne!(original, quantized, "quantization must modify the weights");
    assert!(compressed.is_quantized());

    // A 4-bit grid over this range has a coarse step: every value moved onto it.
    let params =
        weight_ops::derive_quantization_parameters(&original, 4, true, false).expect("parameters");
    for (before, after) in original.iter().zip(quantized.iter()) {
        assert!(
            (before - after).abs() <= params.scale * 1.5,
            "{before} -> {after} exceeds one quantization step"
        );
    }

    // 4-bit storage of 10 parameters is 5 bytes.
    assert_eq!(compressed.model_size_bytes().expect("size"), 5);
}

#[test]
fn test_pipeline_refuses_models_without_parameters() {
    let pipeline = CompressionPipeline::new(utils::simple_pruning_config(0.5)).expect("pipeline");
    let error = match pipeline.compress(OpaqueModel { config: TinyConfig }) {
        Ok(_) => panic!("a model with no named tensors cannot be compressed"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("named_tensors"), "{error}");
}

#[test]
fn test_weight_clustering_reduces_distinct_values() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::WeightClustering {
            num_clusters: 2,
            cluster_method: ClusteringMethod::KMeans,
        }],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let compressed = pipeline.compress(sample_model()).expect("compression");

    let weights = compressed.model.weight.data().expect("weights");
    let distinct: std::collections::BTreeSet<u32> = weights.iter().map(|v| v.to_bits()).collect();
    assert!(
        distinct.len() <= 2,
        "clustering to 2 centroids must leave at most 2 distinct values, got {}",
        distinct.len()
    );
}

#[test]
fn test_low_rank_decomposition_changes_the_matrix() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::LowRankDecomposition {
            decomposition_type: DecompositionType::SVD,
            rank_ratio: 0.5,
        }],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");

    let original = sample_model().weight.data().expect("weights");
    let compressed = pipeline.compress(sample_model()).expect("compression");
    let approximated = compressed.model.weight.data().expect("weights");

    assert_ne!(original, approximated, "the weights must be re-factorised");
    assert!(compressed
        .decomposition_config
        .as_ref()
        .expect("config")
        .layers_to_decompose
        .contains(&"weight".to_string()));
}

#[test]
fn test_unimplemented_decompositions_are_rejected() {
    for decomposition in [
        DecompositionType::Tucker,
        DecompositionType::CP,
        DecompositionType::NMF,
    ] {
        let config = CompressionConfig {
            strategies: vec![CompressionStrategy::LowRankDecomposition {
                decomposition_type: decomposition.clone(),
                rank_ratio: 0.5,
            }],
            ..Default::default()
        };
        let pipeline = CompressionPipeline::new(config).expect("pipeline");
        assert!(
            pipeline.compress(sample_model()).is_err(),
            "{decomposition:?} has no implementation and must not report success"
        );
    }
}

#[test]
fn test_huffman_coding_reports_a_measured_size() {
    let config = CompressionConfig {
        strategies: vec![
            CompressionStrategy::Quantization {
                bits: 8,
                signed: true,
                symmetric: true,
            },
            CompressionStrategy::HuffmanCoding { codebook_size: 256 },
        ],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let compressed = pipeline.compress(sample_model()).expect("compression");

    let encoded = compressed.encoded_size_bytes.expect("a measured encoded size");
    assert!(encoded > 0);
    assert_eq!(compressed.model_size_bytes().expect("size"), encoded);
    assert!(compressed.compression_techniques.contains(&"huffman_coding".to_string()));
}

#[test]
fn test_huffman_coding_requires_quantization_first() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::HuffmanCoding { codebook_size: 256 }],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let error = match pipeline.compress(sample_model()) {
        Ok(_) => panic!("Huffman coding without quantization must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("quantized"), "{error}");
}

#[test]
fn test_fine_tuning_without_a_trainer_is_an_error() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::Quantization {
            bits: 8,
            signed: true,
            symmetric: true,
        }],
        fine_tune: true,
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let error = match pipeline.compress(sample_model()) {
        Ok(_) => panic!("fine-tuning without a trainer must not silently succeed"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("CompressionTrainer"), "{error}");
}

#[test]
fn test_distillation_without_a_trainer_is_an_error() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::KnowledgeDistillation {
            teacher_model: "teacher".to_string(),
            temperature: 2.0,
            alpha: 0.5,
        }],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let error = match pipeline.compress(sample_model()) {
        Ok(_) => panic!("distillation without a trainer must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("CompressionTrainer"), "{error}");
}

#[test]
fn test_trainer_hooks_receive_the_real_model() {
    struct RecordingTrainer {
        fine_tuned: bool,
        distilled: bool,
    }

    impl CompressionTrainer<TinyModel> for RecordingTrainer {
        fn fine_tune(
            &mut self,
            model: &mut TinyModel,
            epochs: usize,
            _learning_rate: f32,
        ) -> Result<f32> {
            self.fine_tuned = true;
            assert_eq!(epochs, 3);
            // A trainer really can rewrite the weights.
            model.bias = Tensor::from_slice(&[0.0, 0.0], &[2])?;
            Ok(0.125)
        }

        fn distill(
            &mut self,
            student: &mut TinyModel,
            teacher_model: &str,
            _temperature: f32,
            _alpha: f32,
        ) -> Result<f32> {
            self.distilled = true;
            assert_eq!(teacher_model, "teacher");
            assert!(!student.named_tensors().is_empty());
            Ok(0.5)
        }
    }

    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::KnowledgeDistillation {
            teacher_model: "teacher".to_string(),
            temperature: 2.0,
            alpha: 0.5,
        }],
        fine_tune: true,
        fine_tune_epochs: 3,
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let mut trainer = RecordingTrainer {
        fine_tuned: false,
        distilled: false,
    };

    let compressed = pipeline
        .compress_with_trainer(sample_model(), &mut trainer)
        .expect("compression with trainer");

    assert!(trainer.fine_tuned);
    assert!(trainer.distilled);
    assert_eq!(compressed.distillation_loss, Some(0.5));
    assert_eq!(compressed.fine_tune_loss, Some(0.125));
    assert_eq!(compressed.model.bias.data().expect("bias"), vec![0.0, 0.0]);
}

#[test]
fn test_analyze_compression_requires_a_compressed_model() {
    let pipeline = CompressionPipeline::new(utils::simple_pruning_config(0.5)).expect("pipeline");
    let uncompressed = CompressedModel::new(sample_model());
    assert!(
        pipeline.analyze_compression(&uncompressed).is_err(),
        "an uncompressed model has no analysis to report"
    );

    let compressed = pipeline.compress(sample_model()).expect("compression");
    let analysis = pipeline.analyze_compression(&compressed).expect("analysis");
    assert!(analysis.original_size > analysis.compressed_size);
}

#[test]
fn test_structured_pruning_removes_whole_rows() {
    let config = CompressionConfig {
        strategies: vec![CompressionStrategy::StructuredPruning {
            pruning_ratio: 0.5,
            granularity: StructuredPruningGranularity::Neuron,
        }],
        ..Default::default()
    };
    let pipeline = CompressionPipeline::new(config).expect("pipeline");
    let compressed = pipeline.compress(sample_model()).expect("compression");

    let weights = compressed.model.weight.data().expect("weights");
    let zero_rows = (0..4)
        .filter(|row| weights[row * 2] == 0.0 && weights[row * 2 + 1] == 0.0)
        .count();
    assert_eq!(
        zero_rows, 2,
        "half of the four rows must be removed entirely"
    );
}
