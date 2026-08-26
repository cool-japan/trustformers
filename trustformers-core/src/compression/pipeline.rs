//! Compression Pipeline for combining multiple compression techniques

use crate::compression::pruning::{AutomaticPruner, Pruner};
use crate::compression::{distillation::DistillationConfig, pruning::PruningConfig};
use anyhow::{anyhow, Result};
use std::time::Instant;

/// Compression stage in the pipeline
#[derive(Debug, Clone)]
pub enum CompressionStage {
    /// Pruning stage
    Pruning {
        strategy: String,
        config: PruningConfig,
    },
    /// Quantization stage
    Quantization { bits: u8, symmetric: bool },
    /// Distillation stage
    Distillation {
        teacher_model: String,
        config: DistillationConfig,
    },
    /// Fine-tuning stage
    FineTuning { epochs: usize, learning_rate: f32 },
    /// Custom stage
    Custom {
        name: String,
        params: std::collections::HashMap<String, String>,
    },
}

/// Compression pipeline configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Pipeline stages to execute
    pub stages: Vec<CompressionStage>,
    /// Target compression ratio
    pub target_ratio: f32,
    /// Maximum acceptable accuracy loss
    pub max_accuracy_loss: f32,
    /// Whether to validate after each stage
    pub validate_stages: bool,
    /// Output directory for intermediate models
    pub output_dir: Option<std::path::PathBuf>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            stages: vec![],
            target_ratio: 10.0,
            max_accuracy_loss: 0.01,
            validate_stages: true,
            output_dir: None,
        }
    }
}

/// Result of compression pipeline
#[derive(Debug, Clone)]
pub struct CompressionResult<M>
where
    M: crate::traits::Model,
{
    /// Final compressed model
    pub model: M,
    /// Original model size in bytes
    pub original_size: usize,
    /// Compressed model size in bytes
    pub compressed_size: usize,
    /// Compression ratio achieved, measured as
    /// `original_size / compressed_size` where `compressed_size` counts only
    /// the non-zero parameters the compressed model still carries.
    pub compression_ratio: f32,
    /// Fraction of the original accuracy retained, when an accuracy evaluator
    /// was supplied to [`CompressionPipeline::compress_with_evaluator`].
    ///
    /// `None` means no evaluation was run; it is never an estimate.
    pub accuracy_retention: Option<f32>,
    /// Time taken for compression
    pub compression_time_seconds: u64,
    /// Stage-wise results
    pub stage_results: Vec<StageResult>,
}

#[derive(Debug, Clone)]
pub struct StageResult {
    /// Human-readable stage name.
    pub stage_name: String,
    /// Bytes of non-zero parameters after this stage.
    pub model_size: usize,
    /// Measured accuracy after this stage, when an evaluator was supplied.
    /// `None` means the stage was not evaluated.
    pub accuracy: Option<f32>,
    /// Wall clock seconds this stage took.
    pub time_seconds: u64,
}

/// Compression report
#[derive(Debug, Clone)]
pub struct CompressionReport {
    pub summary: String,
    pub detailed_metrics: std::collections::HashMap<String, f32>,
    pub recommendations: Vec<String>,
}

/// Measures a model's accuracy, so the pipeline can report a real retention
/// figure instead of an estimate.
type AccuracyEvaluator<'a, M> = dyn Fn(&M) -> Result<f32> + 'a;

/// Bytes occupied by the non-zero parameters of a model.
///
/// Pruning zeroes weights rather than removing them, so the honest "compressed
/// size" is the size a sparse format would need: 4 bytes per surviving f32.
/// Errors when the model exposes no named tensors, because then no size change
/// can be measured at all.
fn nonzero_parameter_bytes<M>(model: &M) -> Result<usize>
where
    M: crate::traits::Model,
{
    let tensors = model.named_tensors();
    if tensors.is_empty() {
        return Err(anyhow!(
            "compression needs weight access: this model exposes no tensors through \
             Model::named_tensors, so no size or compression ratio can be measured"
        ));
    }

    let mut nonzero = 0usize;
    for (_, tensor) in tensors {
        nonzero += tensor.data()?.iter().filter(|value| **value != 0.0).count();
    }
    Ok(nonzero * std::mem::size_of::<f32>())
}

/// Map a stage's strategy name onto a concrete pruning strategy.
///
/// Returns `None` for an unrecognised name, in which case the pruner's own
/// per-layer defaults are used.
fn pruning_strategy_by_name(
    strategy: &str,
    config: &PruningConfig,
) -> Option<Box<dyn crate::compression::pruning::PruningStrategy>> {
    use crate::compression::pruning::{MagnitudePruner, StructuredPruner};

    match strategy.to_ascii_lowercase().as_str() {
        "magnitude" | "unstructured" => {
            Some(Box::new(MagnitudePruner::new(config.target_sparsity)))
        },
        "structured" | "channel" => Some(Box::new(StructuredPruner::new(0))),
        _ => None,
    }
}

/// Main compression pipeline
pub struct CompressionPipeline {
    // Temporarily commented out due to trait object issues
    // stages: Vec<Box<dyn CompressionStageExecutor>>,
    config: CompressionConfig,
}

impl CompressionPipeline {
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            // stages: vec![], // Temporarily commented out
            config,
        }
    }

    /// Execute the compression pipeline without accuracy evaluation.
    ///
    /// `accuracy_retention` on the result stays `None`: nothing was measured.
    /// If [`CompressionConfig::validate_stages`] is set, this fails, because a
    /// stage cannot be validated against an accuracy budget that was never
    /// measured. Use [`Self::compress_with_evaluator`] to supply one.
    pub async fn compress<M>(&self, model: &M) -> Result<CompressionResult<M>>
    where
        M: crate::traits::Model + Clone,
    {
        self.run(model, None::<&AccuracyEvaluator<'_, M>>).await
    }

    /// Execute the compression pipeline, measuring accuracy after every stage.
    ///
    /// `evaluator` is called on the original model and after each stage; the
    /// reported `accuracy_retention` is the measured ratio of the final to the
    /// original accuracy.
    pub async fn compress_with_evaluator<M, F>(
        &self,
        model: &M,
        evaluator: F,
    ) -> Result<CompressionResult<M>>
    where
        M: crate::traits::Model + Clone,
        F: Fn(&M) -> Result<f32>,
    {
        self.run(model, Some(&evaluator)).await
    }

    async fn run<M>(
        &self,
        model: &M,
        evaluator: Option<&AccuracyEvaluator<'_, M>>,
    ) -> Result<CompressionResult<M>>
    where
        M: crate::traits::Model + Clone,
    {
        if self.config.validate_stages && evaluator.is_none() && !self.config.stages.is_empty() {
            return Err(anyhow!(
                "validate_stages is enabled but no accuracy evaluator was supplied: a stage \
                 cannot be checked against max_accuracy_loss without measuring accuracy. Call \
                 compress_with_evaluator, or set validate_stages = false."
            ));
        }

        let start_time = Instant::now();
        let mut current_model = model.clone();
        let original_size = nonzero_parameter_bytes(model)?;
        let baseline_accuracy = match evaluator {
            Some(evaluate) => Some(evaluate(model)?),
            None => None,
        };
        let mut stage_results = Vec::new();

        // Execute each stage in the pipeline
        for (stage_idx, stage) in self.config.stages.iter().enumerate() {
            let stage_start = Instant::now();
            let stage_name = self.get_stage_name(stage);

            tracing::info!(
                stage = stage_idx + 1,
                name = %stage_name,
                "executing compression stage"
            );

            // Apply the compression stage
            current_model = self.apply_compression_stage(&current_model, stage).await?;

            // Measure the real size of the compressed model.
            let stage_size = nonzero_parameter_bytes(&current_model)?;
            let stage_time = stage_start.elapsed().as_secs();

            let accuracy = match evaluator {
                Some(evaluate) => Some(evaluate(&current_model)?),
                None => None,
            };

            stage_results.push(StageResult {
                stage_name: stage_name.clone(),
                model_size: stage_size,
                accuracy,
                time_seconds: stage_time,
            });

            // Validate against the measured accuracy, never an estimate.
            if self.config.validate_stages {
                if let (Some(baseline), Some(current)) = (baseline_accuracy, accuracy) {
                    if baseline > 0.0 {
                        let retention = current / baseline;
                        let accuracy_loss = 1.0 - retention;
                        if accuracy_loss > self.config.max_accuracy_loss {
                            return Err(anyhow!(
                                "Stage '{}' exceeded maximum accuracy loss: {:.2}% > {:.2}%",
                                stage_name,
                                accuracy_loss * 100.0,
                                self.config.max_accuracy_loss * 100.0
                            ));
                        }
                    }
                }
            }

            // Save intermediate model if output directory is specified
            if let Some(ref output_dir) = self.config.output_dir {
                let model_path = output_dir.join(format!("model_stage_{}.bin", stage_idx + 1));
                return Err(anyhow!(
                    "CompressionConfig::output_dir is set ({}), but this pipeline cannot \
                     serialise an intermediate model: use crate::export or crate::checkpoint to \
                     write the returned model instead",
                    model_path.display()
                ));
            }
        }

        // Calculate final metrics from the real weights.
        let compressed_size = nonzero_parameter_bytes(&current_model)?;
        let compression_ratio = if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            0.0
        };
        let total_time = start_time.elapsed().as_secs();

        let accuracy_retention = match (baseline_accuracy, stage_results.last()) {
            (Some(baseline), Some(last)) if baseline > 0.0 => {
                last.accuracy.map(|accuracy| accuracy / baseline)
            },
            // No stages ran, but a baseline exists: nothing changed.
            (Some(_), None) => Some(1.0),
            _ => None,
        };

        if compression_ratio < self.config.target_ratio {
            tracing::warn!(
                target = self.config.target_ratio,
                achieved = compression_ratio,
                "target compression ratio not achieved"
            );
        }

        Ok(CompressionResult {
            model: current_model,
            original_size,
            compressed_size,
            compression_ratio,
            accuracy_retention,
            compression_time_seconds: total_time,
            stage_results,
        })
    }

    async fn apply_compression_stage<M>(&self, model: &M, stage: &CompressionStage) -> Result<M>
    where
        M: crate::traits::Model + Clone,
    {
        match stage {
            CompressionStage::Pruning { strategy, config } => {
                // Real pruning: rewrite the model's weights and let the pruner
                // report the sparsity it actually achieved.
                let mut pruner = AutomaticPruner::new();
                if let Some(named) = pruning_strategy_by_name(strategy, config) {
                    pruner = pruner.with_default_strategy(named);
                }
                let result = pruner.prune(model.clone(), config)?;
                tracing::info!(
                    strategy = %strategy,
                    sparsity = result.sparsity,
                    pruned_params = result.pruned_params,
                    "pruning stage applied"
                );
                Ok(result.model)
            },
            CompressionStage::Quantization { bits, symmetric } => Err(anyhow!(
                "quantization stage ({} bits, symmetric = {}) is not wired into the compression \
                 pipeline: the quantizers in crate::quantization operate on tensors, not on a \
                 generic Model, so no weight can be quantised here",
                bits,
                symmetric
            )),
            CompressionStage::Distillation { teacher_model, .. } => Err(anyhow!(
                "distillation stage (teacher '{}') is not implemented: training the student \
                 requires gradients, which are not available over the generic Model trait",
                teacher_model
            )),
            CompressionStage::FineTuning {
                epochs,
                learning_rate,
            } => Err(anyhow!(
                "fine-tuning stage ({} epochs, lr {}) is not implemented: it requires an \
                 optimizer and gradients, which are not available over the generic Model trait",
                epochs,
                learning_rate
            )),
            CompressionStage::Custom { name, .. } => Err(anyhow!(
                "custom compression stage '{}' has no registered executor",
                name
            )),
        }
    }

    fn get_stage_name(&self, stage: &CompressionStage) -> String {
        match stage {
            CompressionStage::Pruning { strategy, .. } => format!("Pruning ({})", strategy),
            CompressionStage::Quantization { bits, .. } => format!("Quantization ({}bit)", bits),
            CompressionStage::Distillation { .. } => "Distillation".to_string(),
            CompressionStage::FineTuning { .. } => "Fine-tuning".to_string(),
            CompressionStage::Custom { name, .. } => format!("Custom ({})", name),
        }
    }

    /// Generate compression report
    pub fn generate_report<M>(&self, result: &CompressionResult<M>) -> CompressionReport
    where
        M: crate::traits::Model,
    {
        let summary = format!(
            "Compression Summary:\n\
             - Original size: {} MB\n\
             - Compressed size: {} MB\n\
             - Compression ratio: {:.2}x\n\
             - Accuracy retention: {}\n\
             - Total time: {} seconds",
            result.original_size / 1_000_000,
            result.compressed_size / 1_000_000,
            result.compression_ratio,
            result
                .accuracy_retention
                .map(|value| format!("{:.2}%", value * 100.0))
                .unwrap_or_else(|| "not measured".to_string()),
            result.compression_time_seconds
        );

        let mut detailed_metrics = std::collections::HashMap::new();
        detailed_metrics.insert("compression_ratio".to_string(), result.compression_ratio);
        if let Some(retention) = result.accuracy_retention {
            detailed_metrics.insert("accuracy_retention".to_string(), retention);
        }
        detailed_metrics.insert(
            "size_reduction".to_string(),
            1.0 - (result.compressed_size as f32 / result.original_size as f32),
        );

        let recommendations = self.generate_recommendations(result);

        CompressionReport {
            summary,
            detailed_metrics,
            recommendations,
        }
    }

    // Temporarily commented out helper methods due to trait object issues
    /*
    fn execute_pruning<M>(&self, model: &M, strategy: &str, config: &PruningConfig) -> Result<M>
    where M: crate::traits::Model + Clone,
    {
        // Implementation would use actual pruning strategies
        Ok(model.clone())
    }

    fn execute_quantization<M>(&self, model: &M, bits: u8, symmetric: bool) -> Result<M>
    where M: crate::traits::Model + Clone,
    {
        // Implementation would use quantization module
        Ok(model.clone())
    }
    */

    // All helper methods temporarily commented out due to trait object issues
    /*
    async fn execute_distillation<M>(&self, model: &M, teacher_model: &str, config: &DistillationConfig) -> Result<M>
    where M: crate::traits::Model + Clone,
    {
        // Implementation would use distillation module
        Ok(model.clone())
    }

    fn execute_finetuning<M>(&self, model: &M, epochs: usize, learning_rate: f32) -> Result<M>
    where M: crate::traits::Model + Clone,
    {
        // Implementation would use training module
        Ok(model.clone())
    }

    fn execute_custom<M>(&self, model: &M, name: &str, params: &std::collections::HashMap<String, String>) -> Result<M>
    where M: crate::traits::Model + Clone,
    {
        // Implementation would use custom compression methods
        Ok(model.clone())
    }

    fn estimate_model_size<M>(&self, model: &M) -> usize
    where M: crate::traits::Model,
    {
        // Estimate based on parameter count and data type
        1_000_000 // Placeholder
    }

    fn evaluate_accuracy<M>(&self, model: &M) -> Result<f32>
    where M: crate::traits::Model,
    {
        // Would evaluate on validation set
        Ok(0.95)
    }
    */

    /// Check a stage's *measured* accuracy against the configured budget.
    ///
    /// A stage with no measurement cannot be validated; that is reported as an
    /// error rather than silently passing.
    #[allow(dead_code)]
    fn validate_stage_result(&self, result: &StageResult) -> Result<()> {
        let Some(accuracy) = result.accuracy else {
            return Err(anyhow!(
                "Stage {} was not evaluated, so it cannot be validated against max_accuracy_loss",
                result.stage_name
            ));
        };
        if accuracy < (1.0 - self.config.max_accuracy_loss) {
            return Err(anyhow!(
                "Stage {} resulted in too much accuracy loss: {:.2}%",
                result.stage_name,
                (1.0 - accuracy) * 100.0
            ));
        }
        Ok(())
    }

    fn generate_recommendations<M>(&self, result: &CompressionResult<M>) -> Vec<String>
    where
        M: crate::traits::Model,
    {
        let mut recommendations = Vec::new();

        if result.compression_ratio < self.config.target_ratio {
            recommendations.push(format!(
                "Target compression ratio {:.1}x not achieved. Consider more aggressive pruning or quantization.",
                self.config.target_ratio
            ));
        }

        match result.accuracy_retention {
            Some(retention) if retention < 0.95 => recommendations.push(
                "Significant accuracy loss detected. Consider using knowledge distillation or fine-tuning.".to_string()
            ),
            None => recommendations.push(
                "Accuracy was not measured. Run compress_with_evaluator to find out what the compression cost.".to_string()
            ),
            _ => {},
        }

        // Stage-specific recommendations
        for (i, stage_result) in result.stage_results.iter().enumerate() {
            if i > 0 {
                let prev_result = &result.stage_results[i - 1];
                let size_reduction =
                    1.0 - (stage_result.model_size as f32 / prev_result.model_size as f32);

                if size_reduction < 0.1 {
                    recommendations.push(format!(
                        "Stage '{}' achieved minimal size reduction ({:.1}%). Consider adjusting parameters.",
                        stage_result.stage_name,
                        size_reduction * 100.0
                    ));
                }
            }
        }

        recommendations
    }
}

/// Pipeline builder for easy configuration
pub struct PipelineBuilder {
    stages: Vec<CompressionStage>,
    config: CompressionConfig,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: CompressionConfig::default(),
        }
    }

    /// Add pruning stage
    pub fn add_pruning(mut self, sparsity: f32) -> Self {
        self.stages.push(CompressionStage::Pruning {
            strategy: "magnitude".to_string(),
            config: PruningConfig {
                target_sparsity: sparsity,
                ..Default::default()
            },
        });
        self
    }

    /// Add quantization stage
    pub fn add_quantization(mut self, bits: u8) -> Self {
        self.stages.push(CompressionStage::Quantization {
            bits,
            symmetric: true,
        });
        self
    }

    /// Add distillation stage
    pub fn add_distillation(mut self, teacher_model: String, temperature: f32) -> Self {
        self.stages.push(CompressionStage::Distillation {
            teacher_model,
            config: DistillationConfig {
                temperature,
                ..Default::default()
            },
        });
        self
    }

    /// Add fine-tuning stage
    pub fn add_finetuning(mut self, epochs: usize, learning_rate: f32) -> Self {
        self.stages.push(CompressionStage::FineTuning {
            epochs,
            learning_rate,
        });
        self
    }

    /// Set target compression ratio
    pub fn target_ratio(mut self, ratio: f32) -> Self {
        self.config.target_ratio = ratio;
        self
    }

    /// Set maximum accuracy loss
    pub fn max_accuracy_loss(mut self, loss: f32) -> Self {
        self.config.max_accuracy_loss = loss;
        self
    }

    /// Build the pipeline
    pub fn build(mut self) -> CompressionPipeline {
        self.config.stages = self.stages;
        CompressionPipeline::new(self.config)
    }
}

/// Trait for custom compression stage executors
#[allow(dead_code)]
trait CompressionStageExecutor: Send + Sync {
    fn execute<M>(&self, model: &M) -> Result<M>
    where
        M: crate::traits::Model;
    fn name(&self) -> &str;
}

// Mock implementation for demonstration
#[allow(dead_code)]
struct MockModel;

impl crate::traits::Model for MockModel {
    type Config = MockConfig;
    type Input = crate::tensor::Tensor;
    type Output = crate::tensor::Tensor;

    fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
        Ok(input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> crate::errors::Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &MockConfig
    }

    fn num_parameters(&self) -> usize {
        // Mock model with a reasonable parameter count for testing
        800_000
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
struct MockConfig;

impl crate::traits::Config for MockConfig {
    fn architecture(&self) -> &'static str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::traits::{Config, Model};
    use std::io::Read;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny"
        }
    }

    /// A model with real, addressable weights.
    #[derive(Debug, Clone)]
    struct TinyModel {
        config: TinyConfig,
        weight: Tensor,
    }

    impl TinyModel {
        fn new() -> Self {
            Self {
                config: TinyConfig,
                weight: Tensor::from_vec(
                    vec![0.9, -0.8, 0.05, -0.02, 0.7, -0.6, 0.01, -0.03],
                    &[2, 4],
                )
                .expect("from_vec failed"),
            }
        }
    }

    impl Model for TinyModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> crate::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            8
        }

        fn named_tensors(&self) -> Vec<(String, &Tensor)> {
            vec![("linear.weight".to_string(), &self.weight)]
        }

        fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
            vec![("linear.weight".to_string(), &mut self.weight)]
        }
    }

    /// A model that exposes no weights at all.
    #[derive(Debug, Clone)]
    struct OpaqueModel;

    impl Model for OpaqueModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> crate::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &TinyConfig
        }

        fn num_parameters(&self) -> usize {
            1_000_000
        }
    }

    /// Regression test: every stage used to be `Ok(model.clone())`, so the
    /// pipeline reported a 1.0x ratio with 98% "accuracy retention" for a model
    /// it had never touched. Pruning must now really zero weights and the size
    /// must really drop.
    #[tokio::test]
    async fn test_pruning_stage_really_compresses_the_model() {
        let pipeline = CompressionPipeline::new(CompressionConfig {
            stages: vec![CompressionStage::Pruning {
                strategy: "magnitude".to_string(),
                config: PruningConfig {
                    target_sparsity: 0.5,
                    ..Default::default()
                },
            }],
            target_ratio: 1.5,
            max_accuracy_loss: 1.0,
            validate_stages: false,
            output_dir: None,
        });

        let model = TinyModel::new();
        let result = pipeline.compress(&model).await.expect("compression failed");

        assert_eq!(result.original_size, 8 * 4, "8 non-zero f32 weights");
        assert!(
            result.compressed_size < result.original_size,
            "pruning must reduce the non-zero footprint: {} -> {}",
            result.original_size,
            result.compressed_size
        );
        assert!(
            result.compression_ratio > 1.0,
            "compression ratio must reflect a real change, got {}",
            result.compression_ratio
        );

        // The returned model's weights must actually contain zeros now.
        let weights = result.model.named_tensors()[0].1.data().expect("data failed");
        assert!(
            weights.contains(&0.0),
            "the compressed model must really carry pruned weights: {:?}",
            weights
        );

        // No evaluator was supplied, so no accuracy may be claimed.
        assert!(result.accuracy_retention.is_none());
        assert!(result.stage_results.iter().all(|stage| stage.accuracy.is_none()));
    }

    /// Accuracy retention must come from the supplied evaluator.
    #[tokio::test]
    async fn test_accuracy_retention_is_measured_not_estimated() {
        let pipeline = CompressionPipeline::new(CompressionConfig {
            stages: vec![CompressionStage::Pruning {
                strategy: "magnitude".to_string(),
                config: PruningConfig {
                    target_sparsity: 0.5,
                    ..Default::default()
                },
            }],
            target_ratio: 1.0,
            max_accuracy_loss: 1.0,
            validate_stages: true,
            output_dir: None,
        });

        let model = TinyModel::new();
        // "Accuracy" here is the fraction of surviving weights, so it really
        // changes when the model is pruned.
        let evaluator = |model: &TinyModel| -> Result<f32> {
            let data = model.weight.data()?;
            Ok(data.iter().filter(|value| **value != 0.0).count() as f32 / data.len() as f32)
        };

        let result = pipeline
            .compress_with_evaluator(&model, evaluator)
            .await
            .expect("compression failed");

        let retention = result.accuracy_retention.expect("an evaluator was supplied");
        assert!(
            retention < 1.0,
            "the evaluator saw a real change: {retention}"
        );
        // 0.98 was the hardcoded pruning estimate.
        assert!((retention - 0.98).abs() > 1e-6);
    }

    /// Stages that are not implemented must fail loudly.
    #[tokio::test]
    async fn test_unimplemented_stages_fail_instead_of_cloning() {
        for stage in [
            CompressionStage::Quantization {
                bits: 8,
                symmetric: true,
            },
            CompressionStage::Distillation {
                teacher_model: "teacher".to_string(),
                config: DistillationConfig::default(),
            },
            CompressionStage::FineTuning {
                epochs: 1,
                learning_rate: 1e-4,
            },
            CompressionStage::Custom {
                name: "mystery".to_string(),
                params: std::collections::HashMap::new(),
            },
        ] {
            let pipeline = CompressionPipeline::new(CompressionConfig {
                stages: vec![stage],
                validate_stages: false,
                ..Default::default()
            });
            let model = TinyModel::new();
            assert!(
                pipeline.compress(&model).await.is_err(),
                "an unimplemented stage must not silently return the input model"
            );
        }
    }

    /// A model with no weight access cannot be measured, so no ratio may be
    /// reported for it.
    #[tokio::test]
    async fn test_model_without_weight_access_is_refused() {
        let pipeline = CompressionPipeline::new(CompressionConfig {
            validate_stages: false,
            ..Default::default()
        });
        let error = pipeline
            .compress(&OpaqueModel)
            .await
            .expect_err("no size can be measured without named tensors");
        assert!(error.to_string().contains("named_tensors"));
    }

    /// Validation against an accuracy budget requires a measurement.
    #[tokio::test]
    async fn test_validate_stages_requires_an_evaluator() {
        let pipeline = CompressionPipeline::new(CompressionConfig {
            stages: vec![CompressionStage::Pruning {
                strategy: "magnitude".to_string(),
                config: PruningConfig::default(),
            }],
            validate_stages: true,
            ..Default::default()
        });
        let model = TinyModel::new();
        let error = pipeline
            .compress(&model)
            .await
            .expect_err("cannot validate an unmeasured accuracy");
        assert!(error.to_string().contains("no accuracy evaluator"));
    }
}
