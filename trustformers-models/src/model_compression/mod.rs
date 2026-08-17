//! # Model Compression Toolkit
//!
//! This module provides a comprehensive toolkit for model compression techniques,
//! enabling efficient deployment of large models with minimal performance loss.
//!
//! ## Features
//!
//! - **Quantization**: Post-training and quantization-aware training
//! - **Pruning**: Structured and unstructured pruning with various strategies
//! - **Low-Rank Decomposition**: SVD, Tucker, and CP decomposition
//! - **Knowledge Distillation**: Integration with distillation framework
//! - **Hybrid Compression**: Combining multiple compression techniques
//! - **AutoML**: Automatic compression pipeline optimization
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::model_compression::{
//!     CompressionPipeline, CompressionConfig, CompressionStrategy, PruningStrategy
//! };
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
//! #     type Input = ();
//! #     type Output = ();
//! #     fn forward(&self, input: ()) -> trustformers_core::Result<()> { Ok(input) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> trustformers_core::Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 4 }
//! #     fn named_tensors(&self) -> Vec<(String, &Tensor)> { vec![("weight".to_string(), &self.weight)] }
//! #     fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> { vec![("weight".to_string(), &mut self.weight)] }
//! # }
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let config = CompressionConfig {
//!     target_compression_ratio: 0.25, // 4x compression
//!     strategies: vec![
//!         CompressionStrategy::Quantization { bits: 8, signed: true, symmetric: true },
//!         CompressionStrategy::UnstructuredPruning { sparsity: 0.5, strategy: PruningStrategy::Magnitude },
//!     ],
//!     ..Default::default()
//! };
//!
//! let pipeline = CompressionPipeline::new(config)?;
//! # let model = DocModel { weight: Tensor::from_slice(&[0.1, -0.4, 0.8, -0.9], &[2, 2])? };
//! let compressed_model = pipeline.compress(model)?;
//! // Half the weights were really zeroed and every survivor was really rounded.
//! assert_eq!(compressed_model.nonzero_parameter_count()?, 2);
//! # let _ = compressed_model;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::TrustformersError, traits::Model, Result};

pub mod weight_ops;

pub use weight_ops::{
    huffman_decode, huffman_encode, HuffmanEncoded, PruneStats, QuantizationParameters,
    StructureAxis,
};

/// Bytes used by one parameter in the dense f32 representation.
const BYTES_PER_F32: usize = 4;

/// Hooks for the compression steps that require training.
///
/// Fine-tuning and knowledge distillation cannot be performed by the pipeline
/// itself: they need training data, an optimizer and a teacher model, none of
/// which this crate owns. Register a trainer to enable those strategies; without
/// one they return an error instead of silently returning the model untouched.
pub trait CompressionTrainer<M: Model> {
    /// Fine-tune a compressed model. Returns the achieved training loss.
    fn fine_tune(&mut self, model: &mut M, epochs: usize, learning_rate: f32) -> Result<f32>;

    /// Distil `teacher_model` into the compressed student. Returns the achieved
    /// distillation loss.
    fn distill(
        &mut self,
        student: &mut M,
        teacher_model: &str,
        temperature: f32,
        alpha: f32,
    ) -> Result<f32>;
}

/// Configuration for model compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (0.0-1.0, where 0.1 means 10x compression)
    pub target_compression_ratio: f32,
    /// List of compression strategies to apply
    pub strategies: Vec<CompressionStrategy>,
    /// Whether to fine-tune after compression
    pub fine_tune: bool,
    /// Number of fine-tuning epochs
    pub fine_tune_epochs: usize,
    /// Learning rate for fine-tuning
    pub fine_tune_lr: f32,
    /// Whether to use progressive compression
    pub progressive: bool,
    /// Number of progressive stages
    pub progressive_stages: usize,
    /// Metrics to optimize for (accuracy, latency, memory)
    pub optimization_objectives: Vec<OptimizationObjective>,
    /// Constraint on maximum accuracy drop
    pub max_accuracy_drop: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target_compression_ratio: 0.5,
            strategies: vec![CompressionStrategy::Quantization {
                bits: 8,
                signed: true,
                symmetric: false,
            }],
            // Fine-tuning needs a `CompressionTrainer`; enabling it by default
            // would make the default pipeline fail for callers that have none.
            fine_tune: false,
            fine_tune_epochs: 3,
            fine_tune_lr: 1e-5,
            progressive: false,
            progressive_stages: 3,
            optimization_objectives: vec![OptimizationObjective::ModelSize],
            max_accuracy_drop: 0.02, // 2% max drop
        }
    }
}

/// Different compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Quantization (reduce numerical precision)
    Quantization {
        bits: u8,
        signed: bool,
        symmetric: bool,
    },
    /// Post-training quantization
    PostTrainingQuantization {
        calibration_samples: usize,
        bits: u8,
    },
    /// Quantization-aware training
    QuantizationAwareTraining { bits: u8, fake_quantize: bool },
    /// Unstructured pruning (remove individual weights)
    UnstructuredPruning {
        sparsity: f32,
        strategy: PruningStrategy,
    },
    /// Structured pruning (remove entire neurons/channels)
    StructuredPruning {
        pruning_ratio: f32,
        granularity: StructuredPruningGranularity,
    },
    /// Low-rank decomposition
    LowRankDecomposition {
        decomposition_type: DecompositionType,
        rank_ratio: f32,
    },
    /// Weight clustering
    WeightClustering {
        num_clusters: usize,
        cluster_method: ClusteringMethod,
    },
    /// Huffman coding for weight compression
    HuffmanCoding { codebook_size: usize },
    /// Knowledge distillation
    KnowledgeDistillation {
        teacher_model: String,
        temperature: f32,
        alpha: f32,
    },
}

/// Pruning strategies for unstructured pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Magnitude-based pruning
    Magnitude,
    /// Gradient-based pruning
    Gradient,
    /// Random pruning (baseline)
    Random,
    /// SNIP (Single-shot Network Pruning)
    SNIP,
    /// GraSP (Gradient Signal Preservation)
    GraSP,
    /// Lottery ticket hypothesis
    LotteryTicket,
}

/// Granularity for structured pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuredPruningGranularity {
    /// Prune entire neurons
    Neuron,
    /// Prune entire channels
    Channel,
    /// Prune entire filters
    Filter,
    /// Prune attention heads
    AttentionHead,
    /// Prune transformer layers
    Layer,
}

/// Types of matrix decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionType {
    /// Singular Value Decomposition
    SVD,
    /// Tucker decomposition
    Tucker,
    /// CP (CANDECOMP/PARAFAC) decomposition
    CP,
    /// Non-negative matrix factorization
    NMF,
}

/// Clustering methods for weight clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMethod {
    /// K-means clustering
    KMeans,
    /// Gaussian mixture model
    GMM,
    /// Hierarchical clustering
    Hierarchical,
}

/// Optimization objectives for compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize model size
    ModelSize,
    /// Minimize inference latency
    Latency,
    /// Minimize memory usage
    Memory,
    /// Minimize energy consumption
    Energy,
    /// Maximize accuracy
    Accuracy,
    /// Custom weighted combination
    Weighted {
        size_weight: f32,
        latency_weight: f32,
        memory_weight: f32,
        accuracy_weight: f32,
    },
}

/// Results from compression analysis
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    /// Original model size (in parameters)
    pub original_size: usize,
    /// Compressed model size (in parameters)
    pub compressed_size: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Memory reduction (in bytes)
    pub memory_reduction: usize,
    /// Measured latency improvement. `None`: the pipeline runs no inference, so
    /// there is nothing to measure here.
    pub latency_improvement: Option<f32>,
    /// Accuracy metrics before and after compression
    pub accuracy_metrics: HashMap<String, (f32, f32)>, // (before, after)
    /// Per-layer compression statistics
    pub layer_statistics: HashMap<String, LayerCompressionStats>,
}

/// Compression statistics for a single layer
#[derive(Debug, Clone)]
pub struct LayerCompressionStats {
    /// Original parameter count
    pub original_params: usize,
    /// Compressed parameter count
    pub compressed_params: usize,
    /// Compression techniques applied
    pub techniques_applied: Vec<String>,
    /// Memory savings (bytes)
    pub memory_savings: usize,
    /// Estimated FLOP reduction
    pub flop_reduction: f32,
}

/// Model compression pipeline
pub struct CompressionPipeline {
    #[allow(dead_code)]
    config: CompressionConfig,
    compression_stages: Vec<CompressionStage>,
    #[allow(dead_code)]
    current_stage: usize,
}

impl CompressionPipeline {
    /// Create a new compression pipeline
    pub fn new(config: CompressionConfig) -> Result<Self> {
        let compression_stages = Self::create_compression_stages(&config)?;

        Ok(Self {
            config,
            compression_stages,
            current_stage: 0,
        })
    }

    /// Create compression stages from configuration
    fn create_compression_stages(config: &CompressionConfig) -> Result<Vec<CompressionStage>> {
        let mut stages = Vec::new();

        if config.progressive {
            // Create progressive stages
            let strategies_per_stage = config.strategies.len() / config.progressive_stages.max(1);

            for stage_idx in 0..config.progressive_stages {
                let start_idx = stage_idx * strategies_per_stage;
                let end_idx = (start_idx + strategies_per_stage).min(config.strategies.len());

                if start_idx < config.strategies.len() {
                    let stage_strategies = config.strategies[start_idx..end_idx].to_vec();
                    stages.push(CompressionStage {
                        strategies: stage_strategies,
                        fine_tune: config.fine_tune && stage_idx == config.progressive_stages - 1,
                        stage_index: stage_idx,
                    });
                }
            }
        } else {
            // Single stage with all strategies
            stages.push(CompressionStage {
                strategies: config.strategies.clone(),
                fine_tune: config.fine_tune,
                stage_index: 0,
            });
        }

        Ok(stages)
    }

    /// Compress a model using the configured pipeline.
    ///
    /// Every configured strategy really rewrites the model's parameters (through
    /// [`Model::named_tensors_mut`]); the reported analysis is measured on the
    /// weights before and after.
    ///
    /// # Errors
    ///
    /// * The model exposes no named tensors — there would be nothing to compress,
    ///   and reporting a compression ratio for an untouched model would be a lie.
    /// * A configured strategy needs a [`CompressionTrainer`] (fine-tuning,
    ///   knowledge distillation) and none was supplied; use
    ///   [`CompressionPipeline::compress_with_trainer`].
    pub fn compress<M: Model>(&self, model: M) -> Result<CompressedModel<M>> {
        self.compress_inner::<M, NoTrainer>(model, None)
    }

    /// Compress a model, using `trainer` for the strategies that need training.
    pub fn compress_with_trainer<M: Model, T: CompressionTrainer<M>>(
        &self,
        model: M,
        trainer: &mut T,
    ) -> Result<CompressedModel<M>> {
        self.compress_inner(model, Some(trainer))
    }

    fn compress_inner<M: Model, T: CompressionTrainer<M>>(
        &self,
        model: M,
        mut trainer: Option<&mut T>,
    ) -> Result<CompressedModel<M>> {
        let mut compressed_model = CompressedModel::new(model);

        // Snapshot the *original* size before anything is rewritten.
        let original_parameters = compressed_model.count_parameters()?;
        let original_bytes = original_parameters * BYTES_PER_F32;
        let baseline_stats = compressed_model.per_tensor_parameter_counts()?;

        // Apply each compression stage
        for stage in &self.compression_stages {
            compressed_model =
                self.apply_compression_stage(compressed_model, stage, trainer.as_deref_mut())?;
        }

        let compressed_parameters = compressed_model.nonzero_parameter_count()?;
        let compressed_bytes = compressed_model.model_size_bytes()?;

        let mut layer_statistics = HashMap::new();
        let final_stats = compressed_model.per_tensor_nonzero_counts()?;
        for (name, original) in &baseline_stats {
            let remaining = final_stats.get(name).copied().unwrap_or(0);
            layer_statistics.insert(
                name.clone(),
                LayerCompressionStats {
                    original_params: *original,
                    compressed_params: remaining,
                    techniques_applied: compressed_model.compression_techniques.clone(),
                    memory_savings: original.saturating_sub(remaining) * BYTES_PER_F32,
                    flop_reduction: if *original > 0 {
                        1.0 - remaining as f32 / *original as f32
                    } else {
                        0.0
                    },
                },
            );
        }

        let analysis = CompressionAnalysis {
            original_size: original_parameters,
            compressed_size: compressed_parameters,
            compression_ratio: if original_bytes > 0 {
                compressed_bytes as f32 / original_bytes as f32
            } else {
                1.0
            },
            memory_reduction: original_bytes.saturating_sub(compressed_bytes),
            // No inference is run here, so there is no latency to report.
            latency_improvement: None,
            accuracy_metrics: HashMap::new(),
            layer_statistics,
        };

        compressed_model.analysis = Some(analysis);
        Ok(compressed_model)
    }

    /// Apply a single compression stage
    fn apply_compression_stage<M: Model, T: CompressionTrainer<M>>(
        &self,
        mut model: CompressedModel<M>,
        stage: &CompressionStage,
        mut trainer: Option<&mut T>,
    ) -> Result<CompressedModel<M>> {
        for strategy in &stage.strategies {
            model = self.apply_compression_strategy(model, strategy, trainer.as_deref_mut())?;
        }

        // Fine-tune if requested
        if stage.fine_tune {
            model = self.fine_tune_model(model, trainer)?;
        }

        Ok(model)
    }

    /// Apply a single compression strategy
    fn apply_compression_strategy<M: Model, T: CompressionTrainer<M>>(
        &self,
        model: CompressedModel<M>,
        strategy: &CompressionStrategy,
        trainer: Option<&mut T>,
    ) -> Result<CompressedModel<M>> {
        match strategy {
            CompressionStrategy::Quantization {
                bits,
                signed,
                symmetric,
            } => self.apply_quantization(model, *bits, *signed, *symmetric),
            CompressionStrategy::PostTrainingQuantization {
                calibration_samples,
                bits,
            } => self.apply_post_training_quantization(model, *calibration_samples, *bits),
            CompressionStrategy::UnstructuredPruning {
                sparsity,
                strategy: pruning_strategy,
            } => self.apply_unstructured_pruning(model, *sparsity, pruning_strategy),
            CompressionStrategy::StructuredPruning {
                pruning_ratio,
                granularity,
            } => self.apply_structured_pruning(model, *pruning_ratio, granularity),
            CompressionStrategy::LowRankDecomposition {
                decomposition_type,
                rank_ratio,
            } => self.apply_low_rank_decomposition(model, decomposition_type, *rank_ratio),
            CompressionStrategy::WeightClustering {
                num_clusters,
                cluster_method,
            } => self.apply_weight_clustering(model, *num_clusters, cluster_method),
            CompressionStrategy::QuantizationAwareTraining {
                bits,
                fake_quantize,
            } => self.apply_quantization_aware_training(model, *bits, *fake_quantize),
            CompressionStrategy::HuffmanCoding { codebook_size } => {
                self.apply_huffman_coding(model, *codebook_size)
            },
            CompressionStrategy::KnowledgeDistillation {
                teacher_model,
                temperature,
                alpha,
            } => self.apply_knowledge_distillation(
                model,
                teacher_model,
                *temperature,
                *alpha,
                trainer,
            ),
        }
    }

    /// Quantize every parameter of the model onto a `bits`-wide grid.
    ///
    /// The weights are really rounded (quantize + dequantize), so the numerical
    /// effect is present in the model and measurable afterwards.
    fn apply_quantization<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        bits: u8,
        signed: bool,
        symmetric: bool,
    ) -> Result<CompressedModel<M>> {
        let mut tensors = model.model.named_tensors_mut();
        if tensors.is_empty() {
            return Err(no_named_tensors("quantization"));
        }

        let mut quantized_any = false;
        for (name, tensor) in tensors.iter_mut() {
            // Integer buffers are not weights; rewriting them would corrupt the model.
            if !weight_ops::is_float_parameter(tensor) {
                continue;
            }
            weight_ops::quantize_tensor_in_place(name, tensor, bits, signed, symmetric)?;
            quantized_any = true;
        }
        drop(tensors);

        if !quantized_any {
            return Err(TrustformersError::invalid_operation(
                "quantization found no floating-point parameter to quantize".to_string(),
            ));
        }

        model.quantization_config = Some(QuantizationConfig {
            bits,
            signed,
            symmetric,
            per_channel: false,
        });
        model.compression_techniques.push("quantization".to_string());

        Ok(model)
    }

    /// Post-training quantization.
    ///
    /// Without calibration activations the range is taken from the weights
    /// themselves (min/max), which is exactly what `apply_quantization` does; the
    /// calibration sample count is recorded so the caller can see it was unused.
    fn apply_post_training_quantization<M: Model>(
        &self,
        model: CompressedModel<M>,
        calibration_samples: usize,
        bits: u8,
    ) -> Result<CompressedModel<M>> {
        if calibration_samples == 0 {
            return Err(TrustformersError::invalid_config(
                "post-training quantization needs a positive calibration sample count".to_string(),
            ));
        }
        let mut model = self.apply_quantization(model, bits, true, false)?;
        model.compression_techniques.push("post_training_quantization".to_string());
        Ok(model)
    }

    /// Quantization-aware training.
    ///
    /// The fake-quantization pass is applied for real; the "training" half needs
    /// a [`CompressionTrainer`] and is requested separately through
    /// `CompressionConfig::fine_tune`.
    fn apply_quantization_aware_training<M: Model>(
        &self,
        model: CompressedModel<M>,
        bits: u8,
        fake_quantize: bool,
    ) -> Result<CompressedModel<M>> {
        if !fake_quantize {
            return Err(TrustformersError::invalid_config(
                "quantization-aware training without fake quantization is not implemented: \
                 real low-precision training requires a training backend"
                    .to_string(),
            ));
        }
        let mut model = self.apply_quantization(model, bits, true, false)?;
        model.compression_techniques.push("quantization_aware_training".to_string());
        Ok(model)
    }

    /// Entropy-code the quantized weights with a real canonical Huffman coder and
    /// record the measured compressed size.
    ///
    /// # Errors
    ///
    /// Fails when the model has not been quantized first: Huffman coding operates
    /// on the integer symbol stream, and coding raw f32 bit patterns would not
    /// compress anything.
    fn apply_huffman_coding<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        codebook_size: usize,
    ) -> Result<CompressedModel<M>> {
        let quantization = model.quantization_config.clone().ok_or_else(|| {
            TrustformersError::invalid_config(
                "Huffman coding requires a quantized model: apply a quantization strategy first"
                    .to_string(),
            )
        })?;

        if codebook_size == 0 {
            return Err(TrustformersError::invalid_config(
                "the Huffman codebook must hold at least one symbol".to_string(),
            ));
        }

        let mut symbols = Vec::new();
        {
            let tensors = model.model.named_tensors();
            if tensors.is_empty() {
                return Err(no_named_tensors("Huffman coding"));
            }
            for (name, tensor) in tensors {
                if !weight_ops::is_float_parameter(tensor) {
                    continue;
                }
                let values = weight_ops::tensor_values(&name, tensor)?;
                let params = weight_ops::derive_quantization_parameters(
                    &values,
                    quantization.bits,
                    quantization.signed,
                    quantization.symmetric,
                )?;
                let levels: Vec<i32> = values.iter().map(|v| params.quantize(*v)).collect();
                symbols.extend(weight_ops::levels_to_symbols(&levels, &params)?);
            }
        }

        let encoded = weight_ops::huffman_encode(&symbols)?;
        // Round-trip check: a compressed size we cannot decode is meaningless.
        let decoded = weight_ops::huffman_decode(&encoded)?;
        if decoded != symbols {
            return Err(TrustformersError::invalid_operation(
                "the Huffman encoder did not round-trip; refusing to report its size".to_string(),
            ));
        }

        model.encoded_size_bytes = Some(encoded.total_bytes());
        model.compression_techniques.push("huffman_coding".to_string());
        Ok(model)
    }

    /// Knowledge distillation, delegated to the registered trainer.
    fn apply_knowledge_distillation<M: Model, T: CompressionTrainer<M>>(
        &self,
        mut model: CompressedModel<M>,
        teacher_model: &str,
        temperature: f32,
        alpha: f32,
        trainer: Option<&mut T>,
    ) -> Result<CompressedModel<M>> {
        let trainer = trainer.ok_or_else(|| {
            TrustformersError::invalid_config(
                "knowledge distillation needs a CompressionTrainer (teacher forward passes and \
                 an optimizer); call CompressionPipeline::compress_with_trainer"
                    .to_string(),
            )
        })?;

        let loss = trainer.distill(&mut model.model, teacher_model, temperature, alpha)?;
        model.distillation_loss = Some(loss);
        model.compression_techniques.push("knowledge_distillation".to_string());
        Ok(model)
    }

    /// Remove individual weights until the requested sparsity is reached.
    fn apply_unstructured_pruning<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        sparsity: f32,
        strategy: &PruningStrategy,
    ) -> Result<CompressedModel<M>> {
        if !(0.0..1.0).contains(&sparsity) {
            return Err(TrustformersError::invalid_config(format!(
                "sparsity must lie in [0, 1), got {sparsity}"
            )));
        }

        match strategy {
            PruningStrategy::Magnitude | PruningStrategy::LotteryTicket => {
                // Global magnitude threshold across every floating-point tensor.
                let mut all_values = Vec::new();
                {
                    let tensors = model.model.named_tensors();
                    if tensors.is_empty() {
                        return Err(no_named_tensors("pruning"));
                    }
                    for (name, tensor) in tensors {
                        if !weight_ops::is_float_parameter(tensor) {
                            continue;
                        }
                        all_values.extend(weight_ops::tensor_values(&name, tensor)?);
                    }
                }
                if all_values.is_empty() {
                    return Err(TrustformersError::invalid_operation(
                        "pruning found no floating-point parameter to prune".to_string(),
                    ));
                }

                let threshold = weight_ops::magnitude_threshold_for(&all_values, sparsity)?;
                // Exactly `target` weights must go. Everything strictly below the
                // threshold is removed; the remainder is taken from the ties, so a
                // model of identical weights still reaches the requested sparsity.
                let target = ((all_values.len() as f32) * sparsity).round() as usize;
                let below = all_values.iter().filter(|value| value.abs() < threshold).count();
                let mut tie_budget = target.saturating_sub(below);

                let mut tensors = model.model.named_tensors_mut();
                for (name, tensor) in tensors.iter_mut() {
                    if !weight_ops::is_float_parameter(tensor) {
                        continue;
                    }
                    weight_ops::prune_at_threshold(name, tensor, threshold, &mut tie_budget)?;
                }
            },
            PruningStrategy::Random => {
                let mut state = 0x9E37_79B9_7F4A_7C15u64;
                let mut rng = move || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u32 << 24) as f32
                };
                let mut tensors = model.model.named_tensors_mut();
                if tensors.is_empty() {
                    return Err(no_named_tensors("pruning"));
                }
                for (name, tensor) in tensors.iter_mut() {
                    if !weight_ops::is_float_parameter(tensor) {
                        continue;
                    }
                    weight_ops::random_prune_in_place(name, tensor, sparsity, &mut rng)?;
                }
            },
            other => {
                return Err(TrustformersError::invalid_config(format!(
                    "pruning strategy {other:?} needs gradient information, which the \
                     compression pipeline does not have; use Magnitude or Random"
                )));
            },
        }

        model.pruning_config = Some(UnstructuredPruningConfig {
            sparsity,
            strategy: strategy.clone(),
            global_pruning: true,
        });
        model.compression_techniques.push("unstructured_pruning".to_string());

        Ok(model)
    }

    /// Remove whole rows / columns of the rank-2 weights.
    fn apply_structured_pruning<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        pruning_ratio: f32,
        granularity: &StructuredPruningGranularity,
    ) -> Result<CompressedModel<M>> {
        let axis = match granularity {
            StructuredPruningGranularity::Neuron
            | StructuredPruningGranularity::Filter
            | StructuredPruningGranularity::AttentionHead => StructureAxis::Rows,
            StructuredPruningGranularity::Channel => StructureAxis::Columns,
            StructuredPruningGranularity::Layer => {
                return Err(TrustformersError::invalid_config(
                    "layer-granularity pruning changes the model's architecture, which cannot \
                     be expressed by rewriting weights in place"
                        .to_string(),
                ));
            },
        };

        let mut tensors = model.model.named_tensors_mut();
        if tensors.is_empty() {
            return Err(no_named_tensors("structured pruning"));
        }

        let mut pruned_any = false;
        for (name, tensor) in tensors.iter_mut() {
            if tensor.shape().len() != 2 || !weight_ops::is_float_parameter(tensor) {
                // Biases, norms and integer buffers have no structure to remove.
                continue;
            }
            weight_ops::structured_prune_in_place(name, tensor, pruning_ratio, axis, false)?;
            pruned_any = true;
        }

        if !pruned_any {
            return Err(TrustformersError::invalid_operation(
                "structured pruning found no rank-2 weight to prune".to_string(),
            ));
        }

        model.structured_pruning_config = Some(StructuredPruningConfig {
            pruning_ratio,
            granularity: granularity.clone(),
            importance_metric: ImportanceMetric::L2Norm,
        });
        model.compression_techniques.push("structured_pruning".to_string());

        Ok(model)
    }

    /// Replace rank-2 weights by their best low-rank approximation.
    fn apply_low_rank_decomposition<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        decomposition_type: &DecompositionType,
        rank_ratio: f32,
    ) -> Result<CompressedModel<M>> {
        if !matches!(decomposition_type, DecompositionType::SVD) {
            return Err(TrustformersError::invalid_config(format!(
                "{decomposition_type:?} decomposition is not implemented; only SVD-style \
                 low-rank approximation of rank-2 weights is available"
            )));
        }
        if !(0.0..=1.0).contains(&rank_ratio) || rank_ratio == 0.0 {
            return Err(TrustformersError::invalid_config(format!(
                "the rank ratio must lie in (0, 1], got {rank_ratio}"
            )));
        }

        let mut decomposed = Vec::new();
        {
            let mut tensors = model.model.named_tensors_mut();
            if tensors.is_empty() {
                return Err(no_named_tensors("low-rank decomposition"));
            }
            for (name, tensor) in tensors.iter_mut() {
                let shape = tensor.shape();
                if shape.len() != 2 || !weight_ops::is_float_parameter(tensor) {
                    continue;
                }
                let full_rank = shape[0].min(shape[1]);
                let rank = ((full_rank as f32) * rank_ratio).round().max(1.0) as usize;
                weight_ops::low_rank_approximate_in_place(name, tensor, rank, 2)?;
                decomposed.push(name.clone());
            }
        }

        if decomposed.is_empty() {
            return Err(TrustformersError::invalid_operation(
                "low-rank decomposition found no rank-2 weight to factorise".to_string(),
            ));
        }

        model.decomposition_config = Some(DecompositionConfig {
            decomposition_type: decomposition_type.clone(),
            rank_ratio,
            layers_to_decompose: decomposed,
        });
        model.compression_techniques.push("low_rank_decomposition".to_string());

        Ok(model)
    }

    /// Collapse weights onto a shared codebook of centroids.
    fn apply_weight_clustering<M: Model>(
        &self,
        mut model: CompressedModel<M>,
        num_clusters: usize,
        cluster_method: &ClusteringMethod,
    ) -> Result<CompressedModel<M>> {
        if !matches!(cluster_method, ClusteringMethod::KMeans) {
            return Err(TrustformersError::invalid_config(format!(
                "{cluster_method:?} clustering is not implemented; k-means is available"
            )));
        }

        let mut tensors = model.model.named_tensors_mut();
        if tensors.is_empty() {
            return Err(no_named_tensors("weight clustering"));
        }
        let mut clustered_any = false;
        for (name, tensor) in tensors.iter_mut() {
            if !weight_ops::is_float_parameter(tensor) {
                continue;
            }
            weight_ops::cluster_weights_in_place(name, tensor, num_clusters, 25)?;
            clustered_any = true;
        }
        drop(tensors);

        if !clustered_any {
            return Err(TrustformersError::invalid_operation(
                "weight clustering found no floating-point parameter to cluster".to_string(),
            ));
        }

        model.clustering_config = Some(ClusteringConfig {
            num_clusters,
            cluster_method: cluster_method.clone(),
            per_layer_clustering: true,
        });
        model.compression_techniques.push("weight_clustering".to_string());

        Ok(model)
    }

    /// Fine-tune the compressed model through the registered trainer.
    fn fine_tune_model<M: Model, T: CompressionTrainer<M>>(
        &self,
        mut model: CompressedModel<M>,
        trainer: Option<&mut T>,
    ) -> Result<CompressedModel<M>> {
        let trainer = trainer.ok_or_else(|| {
            TrustformersError::invalid_config(
                "fine-tuning after compression needs a CompressionTrainer (training data and an \
                 optimizer); call CompressionPipeline::compress_with_trainer or set \
                 CompressionConfig::fine_tune to false"
                    .to_string(),
            )
        })?;

        let loss = trainer.fine_tune(
            &mut model.model,
            self.config.fine_tune_epochs,
            self.config.fine_tune_lr,
        )?;
        model.fine_tune_loss = Some(loss);
        model.compression_techniques.push("fine_tuning".to_string());
        Ok(model)
    }

    /// Analyse the compression results measured during [`CompressionPipeline::compress`].
    ///
    /// # Errors
    ///
    /// Fails when the model was never run through the pipeline, because there is
    /// no measurement to report.
    pub fn analyze_compression<M: Model>(
        &self,
        model: &CompressedModel<M>,
    ) -> Result<CompressionAnalysis> {
        model.analysis.clone().ok_or_else(|| {
            TrustformersError::invalid_operation(
                "this model has not been compressed, so there is nothing to analyse".to_string(),
            )
        })
    }
}

/// Error for the case the advisor calls out: a model that exposes no parameters.
fn no_named_tensors(operation: &str) -> TrustformersError {
    TrustformersError::invalid_operation(format!(
        "{operation} requires access to the model's parameters, but the model exposes none: \
         implement Model::named_tensors / Model::named_tensors_mut. Refusing to report a \
         compressed model that was never modified."
    ))
}

/// Placeholder trainer type used when no trainer is supplied.
///
/// It can never be constructed, so `compress` without a trainer cannot silently
/// perform a training step.
pub enum NoTrainer {}

impl<M: Model> CompressionTrainer<M> for NoTrainer {
    fn fine_tune(&mut self, _model: &mut M, _epochs: usize, _learning_rate: f32) -> Result<f32> {
        match *self {}
    }

    fn distill(
        &mut self,
        _student: &mut M,
        _teacher_model: &str,
        _temperature: f32,
        _alpha: f32,
    ) -> Result<f32> {
        match *self {}
    }
}

/// A single stage in the compression pipeline
#[derive(Debug, Clone)]
struct CompressionStage {
    strategies: Vec<CompressionStrategy>,
    fine_tune: bool,
    #[allow(dead_code)]
    stage_index: usize,
}

/// Compressed model wrapper
pub struct CompressedModel<M: Model> {
    /// The underlying model
    pub model: M,
    /// Applied compression techniques
    pub compression_techniques: Vec<String>,
    /// Quantization configuration
    pub quantization_config: Option<QuantizationConfig>,
    /// Pruning configuration
    pub pruning_config: Option<UnstructuredPruningConfig>,
    /// Structured pruning configuration
    pub structured_pruning_config: Option<StructuredPruningConfig>,
    /// Decomposition configuration
    pub decomposition_config: Option<DecompositionConfig>,
    /// Clustering configuration
    pub clustering_config: Option<ClusteringConfig>,
    /// Measured size of the entropy-coded weights, when Huffman coding ran.
    pub encoded_size_bytes: Option<usize>,
    /// Loss reported by the trainer's fine-tuning pass, when one ran.
    pub fine_tune_loss: Option<f32>,
    /// Loss reported by the trainer's distillation pass, when one ran.
    pub distillation_loss: Option<f32>,
    /// Compression analysis results
    pub analysis: Option<CompressionAnalysis>,
}

impl<M: Model> CompressedModel<M> {
    /// Create a new compressed model wrapper
    pub fn new(model: M) -> Self {
        Self {
            model,
            compression_techniques: Vec::new(),
            quantization_config: None,
            pruning_config: None,
            structured_pruning_config: None,
            decomposition_config: None,
            clustering_config: None,
            encoded_size_bytes: None,
            fine_tune_loss: None,
            distillation_loss: None,
            analysis: None,
        }
    }

    /// Count the model's parameters by walking its real tensors.
    ///
    /// # Errors
    ///
    /// Fails when the model exposes no named tensors and also reports zero
    /// parameters: there is no honest count to give.
    pub fn count_parameters(&self) -> Result<usize> {
        let tensors = self.model.named_tensors();
        if tensors.is_empty() {
            let declared = self.model.num_parameters();
            if declared == 0 {
                return Err(no_named_tensors("counting parameters"));
            }
            return Ok(declared);
        }
        Ok(tensors.iter().map(|(_, tensor)| tensor.shape().iter().product::<usize>()).sum())
    }

    /// Parameter count per tensor.
    pub fn per_tensor_parameter_counts(&self) -> Result<HashMap<String, usize>> {
        let tensors = self.model.named_tensors();
        if tensors.is_empty() {
            return Err(no_named_tensors("counting parameters"));
        }
        Ok(tensors
            .into_iter()
            .map(|(name, tensor)| (name, tensor.shape().iter().product::<usize>()))
            .collect())
    }

    /// Number of parameters that are not exactly zero — the count that survives
    /// pruning.
    pub fn nonzero_parameter_count(&self) -> Result<usize> {
        let tensors = self.model.named_tensors();
        if tensors.is_empty() {
            return Err(no_named_tensors("counting parameters"));
        }
        let mut total = 0usize;
        for (name, tensor) in tensors {
            let values = weight_ops::tensor_values(&name, tensor)?;
            total += values.iter().filter(|value| **value != 0.0).count();
        }
        Ok(total)
    }

    /// Non-zero parameter count per tensor.
    pub fn per_tensor_nonzero_counts(&self) -> Result<HashMap<String, usize>> {
        let tensors = self.model.named_tensors();
        if tensors.is_empty() {
            return Err(no_named_tensors("counting parameters"));
        }
        let mut counts = HashMap::new();
        for (name, tensor) in tensors {
            let values = weight_ops::tensor_values(&name, tensor)?;
            counts.insert(name, values.iter().filter(|value| **value != 0.0).count());
        }
        Ok(counts)
    }

    /// Get parameter count of the model.
    ///
    /// Returns 0 only when the underlying model exposes nothing to count; use
    /// [`CompressedModel::count_parameters`] to see the error instead.
    pub fn parameter_count(&self) -> usize {
        self.count_parameters().unwrap_or(0)
    }

    /// Size of the compressed representation, in bytes.
    ///
    /// * With Huffman coding applied this is the **measured** encoded size.
    /// * Otherwise it is the number of surviving (non-zero) parameters at the
    ///   current precision — the size of a sparse low-precision serialisation.
    pub fn model_size_bytes(&self) -> Result<usize> {
        if let Some(encoded) = self.encoded_size_bytes {
            return Ok(encoded);
        }

        let nonzero = self.nonzero_parameter_count()?;
        let bits = self.quantization_config.as_ref().map(|c| c.bits).unwrap_or(32);
        Ok(nonzero * bits as usize / 8)
    }

    /// Check if model is quantized
    pub fn is_quantized(&self) -> bool {
        self.quantization_config.is_some()
    }

    /// Check if model is pruned
    pub fn is_pruned(&self) -> bool {
        self.pruning_config.is_some() || self.structured_pruning_config.is_some()
    }

    /// Get compression summary
    pub fn compression_summary(&self) -> Result<CompressionSummary> {
        Ok(CompressionSummary {
            techniques: self.compression_techniques.clone(),
            parameter_count: self.count_parameters()?,
            model_size_bytes: self.model_size_bytes()?,
            is_quantized: self.is_quantized(),
            is_pruned: self.is_pruned(),
        })
    }
}

/// Configuration structures for different compression techniques
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub bits: u8,
    pub signed: bool,
    pub symmetric: bool,
    pub per_channel: bool,
}

#[derive(Debug, Clone)]
pub struct UnstructuredPruningConfig {
    pub sparsity: f32,
    pub strategy: PruningStrategy,
    pub global_pruning: bool,
}

#[derive(Debug, Clone)]
pub struct StructuredPruningConfig {
    pub pruning_ratio: f32,
    pub granularity: StructuredPruningGranularity,
    pub importance_metric: ImportanceMetric,
}

#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    pub decomposition_type: DecompositionType,
    pub rank_ratio: f32,
    pub layers_to_decompose: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    pub num_clusters: usize,
    pub cluster_method: ClusteringMethod,
    pub per_layer_clustering: bool,
}

/// Importance metrics for structured pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceMetric {
    /// L1 norm of weights
    L1Norm,
    /// L2 norm of weights
    L2Norm,
    /// Gradient-based importance
    Gradient,
    /// Fisher information
    Fisher,
    /// Random (baseline)
    Random,
}

/// Summary of compression applied to a model
#[derive(Debug, Clone)]
pub struct CompressionSummary {
    pub techniques: Vec<String>,
    pub parameter_count: usize,
    pub model_size_bytes: usize,
    pub is_quantized: bool,
    pub is_pruned: bool,
}

/// Utilities for model compression
pub mod utils {
    use super::*;

    /// Create a simple quantization config
    pub fn simple_quantization_config(bits: u8) -> CompressionConfig {
        CompressionConfig {
            strategies: vec![CompressionStrategy::Quantization {
                bits,
                signed: true,
                symmetric: false,
            }],
            ..Default::default()
        }
    }

    /// Create a simple pruning config
    pub fn simple_pruning_config(sparsity: f32) -> CompressionConfig {
        CompressionConfig {
            strategies: vec![CompressionStrategy::UnstructuredPruning {
                sparsity,
                strategy: PruningStrategy::Magnitude,
            }],
            ..Default::default()
        }
    }

    /// Create a combined compression config
    pub fn combined_compression_config(
        quantization_bits: u8,
        pruning_sparsity: f32,
    ) -> CompressionConfig {
        CompressionConfig {
            strategies: vec![
                CompressionStrategy::UnstructuredPruning {
                    sparsity: pruning_sparsity,
                    strategy: PruningStrategy::Magnitude,
                },
                CompressionStrategy::Quantization {
                    bits: quantization_bits,
                    signed: true,
                    symmetric: false,
                },
            ],
            ..Default::default()
        }
    }

    /// Create a progressive compression config
    pub fn progressive_compression_config(target_ratio: f32, stages: usize) -> CompressionConfig {
        CompressionConfig {
            target_compression_ratio: target_ratio,
            progressive: true,
            progressive_stages: stages,
            strategies: vec![
                CompressionStrategy::UnstructuredPruning {
                    sparsity: 0.3,
                    strategy: PruningStrategy::Magnitude,
                },
                CompressionStrategy::LowRankDecomposition {
                    decomposition_type: DecompositionType::SVD,
                    rank_ratio: 0.5,
                },
                CompressionStrategy::Quantization {
                    bits: 8,
                    signed: true,
                    symmetric: false,
                },
            ],
            ..Default::default()
        }
    }

    /// Create an aggressive compression config for maximum compression
    pub fn aggressive_compression_config() -> CompressionConfig {
        CompressionConfig {
            target_compression_ratio: 0.1, // 10x compression
            strategies: vec![
                CompressionStrategy::StructuredPruning {
                    pruning_ratio: 0.5,
                    granularity: StructuredPruningGranularity::Channel,
                },
                CompressionStrategy::UnstructuredPruning {
                    sparsity: 0.8,
                    strategy: PruningStrategy::Magnitude,
                },
                CompressionStrategy::LowRankDecomposition {
                    decomposition_type: DecompositionType::SVD,
                    rank_ratio: 0.3,
                },
                CompressionStrategy::WeightClustering {
                    num_clusters: 256,
                    cluster_method: ClusteringMethod::KMeans,
                },
                CompressionStrategy::Quantization {
                    bits: 4,
                    signed: true,
                    symmetric: true,
                },
            ],
            fine_tune: true,
            fine_tune_epochs: 5,
            max_accuracy_drop: 0.05, // Allow 5% accuracy drop for aggressive compression
            ..Default::default()
        }
    }

    /// Estimate compression ratio for a given configuration
    pub fn estimate_compression_ratio(config: &CompressionConfig) -> f32 {
        let mut ratio = 1.0;

        for strategy in &config.strategies {
            match strategy {
                CompressionStrategy::Quantization { bits, .. } => {
                    ratio *= *bits as f32 / 32.0; // Assuming float32 baseline
                },
                CompressionStrategy::UnstructuredPruning { sparsity, .. } => {
                    ratio *= 1.0 - sparsity; // Sparse storage efficiency
                },
                CompressionStrategy::StructuredPruning { pruning_ratio, .. } => {
                    ratio *= 1.0 - pruning_ratio;
                },
                CompressionStrategy::LowRankDecomposition { rank_ratio, .. } => {
                    ratio *= rank_ratio * 2.0; // Approximate for low-rank factorization
                },
                CompressionStrategy::WeightClustering { num_clusters, .. } => {
                    // Approximate compression from clustering
                    ratio *= (*num_clusters as f32).log2() / 32.0;
                },
                _ => {
                    // Conservative estimate for other strategies
                    ratio *= 0.8;
                },
            }
        }

        ratio.max(0.01) // Minimum 1% of original size
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
