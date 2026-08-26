//! Knowledge Distillation Implementation
//!
//! Train smaller student models from larger teacher models

use crate::tensor::Tensor;
use crate::traits::Model;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;

/// Distillation configuration
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss vs task loss
    pub alpha: f32,
    /// Learning rate for student model
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Layers to match between teacher and student
    pub matched_layers: HashMap<String, String>,
    /// Whether to use feature distillation
    pub use_feature_distillation: bool,
    /// Feature distillation weight
    pub feature_weight: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            learning_rate: 1e-4,
            epochs: 10,
            batch_size: 32,
            matched_layers: HashMap::new(),
            use_feature_distillation: false,
            feature_weight: 0.1,
        }
    }
}

/// Type alias for custom distillation loss function
pub type DistillationLossFn = Box<dyn Fn(&Tensor, &Tensor) -> f32 + Send + Sync>;

/// Distillation loss functions
pub enum DistillationLoss {
    /// Kullback-Leibler divergence
    KLDivergence,
    /// Mean squared error
    MSE,
    /// Cross entropy
    CrossEntropy,
    /// Custom loss function
    Custom(DistillationLossFn),
}

impl std::fmt::Debug for DistillationLoss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KLDivergence => write!(f, "KLDivergence"),
            Self::MSE => write!(f, "MSE"),
            Self::CrossEntropy => write!(f, "CrossEntropy"),
            Self::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

impl Clone for DistillationLoss {
    fn clone(&self) -> Self {
        match self {
            Self::KLDivergence => Self::KLDivergence,
            Self::MSE => Self::MSE,
            Self::CrossEntropy => Self::CrossEntropy,
            Self::Custom(_) => {
                // Custom loss functions cannot be cloned due to closure limitations
                // Return a sensible default (KL divergence is most common for distillation)
                tracing::warn!(
                    "Warning: Custom loss function cannot be cloned, falling back to KL divergence"
                );
                Self::KLDivergence
            },
        }
    }
}

/// Teacher model trait
pub trait TeacherModel: Model {
    /// Get intermediate features for distillation
    fn get_features(&self, layer_name: &str) -> Result<Tensor>;

    /// Get attention maps if available
    fn get_attention_maps(&self) -> Result<HashMap<String, Tensor>>;
}

/// Student model trait
pub trait StudentModel: Model {
    /// Set learning from teacher features
    fn set_feature_target(&mut self, layer_name: &str, features: &Tensor) -> Result<()>;

    /// Get intermediate features
    fn get_features(&self, layer_name: &str) -> Result<Tensor>;
}

/// Distillation strategy
pub enum DistillationStrategy {
    /// Response-based (output) distillation
    Response,
    /// Feature-based (intermediate) distillation
    Feature,
    /// Attention-based distillation
    Attention,
    /// Combined strategy
    Combined {
        response_weight: f32,
        feature_weight: f32,
        attention_weight: f32,
    },
}

/// Result of distillation
#[derive(Debug, Clone)]
pub struct DistillationResult<M>
where
    M: crate::traits::Model,
{
    pub student_model: M,
    pub final_loss: f32,
    pub accuracy_retention: f32,
    pub compression_ratio: f32,
    pub training_time_seconds: u64,
}

/// Main distiller interface
#[async_trait]
pub trait Distiller: Send + Sync {
    /// Distill knowledge from teacher to student
    async fn distill<T, S>(
        &self,
        teacher: &T,
        student: &S,
        config: &DistillationConfig,
    ) -> Result<S>
    where
        T: crate::traits::Model + Sync,
        S: crate::traits::Model + Send;

    /// Evaluate distillation quality
    fn evaluate<T, S>(&self, teacher: &T, student: &S) -> Result<f32>
    where
        T: crate::traits::Model,
        S: crate::traits::Model;
}

/// Standard knowledge distillation
pub struct KnowledgeDistiller {
    temperature: f32,
    loss_fn: DistillationLoss,
}

impl KnowledgeDistiller {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            loss_fn: DistillationLoss::KLDivergence,
        }
    }

    pub fn with_loss(mut self, loss_fn: DistillationLoss) -> Self {
        self.loss_fn = loss_fn;
        self
    }

    /// Temperature-softened softmax over a logits tensor.
    pub fn softmax_with_temperature(&self, logits: &Tensor) -> Result<Tensor> {
        let data = logits.data()?;
        let scaled: Vec<f32> = data.iter().map(|&x| x / self.temperature).collect();

        // Compute softmax
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let softmax: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        Ok(Tensor::from_vec(softmax, &logits.shape())?)
    }

    /// Distillation loss between student and teacher logits, using the
    /// configured loss function and the distiller's temperature.
    pub fn compute_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
    ) -> Result<f32> {
        let student_probs = self.softmax_with_temperature(student_logits)?;
        let teacher_probs = self.softmax_with_temperature(teacher_logits)?;

        match &self.loss_fn {
            DistillationLoss::KLDivergence => self.kl_divergence(&student_probs, &teacher_probs),
            DistillationLoss::MSE => self.mse_loss(&student_probs, &teacher_probs),
            DistillationLoss::CrossEntropy => self.cross_entropy(&student_probs, &teacher_probs),
            DistillationLoss::Custom(f) => Ok(f(&student_probs, &teacher_probs)),
        }
    }

    /// Kullback-Leibler divergence between two probability tensors.
    pub fn kl_divergence(&self, student: &Tensor, teacher: &Tensor) -> Result<f32> {
        let s_data = student.data()?;
        let t_data = teacher.data()?;

        if s_data.len() != t_data.len() {
            return Err(anyhow!("Tensor size mismatch"));
        }

        let kl = t_data
            .iter()
            .zip(s_data.iter())
            .map(
                |(&t, &s)| {
                    if t > 0.0 && s > 0.0 {
                        t * (t / s).ln()
                    } else {
                        0.0
                    }
                },
            )
            .sum::<f32>()
            * self.temperature
            * self.temperature;

        Ok(kl)
    }

    /// Mean squared error between two tensors.
    pub fn mse_loss(&self, student: &Tensor, teacher: &Tensor) -> Result<f32> {
        let s_data = student.data()?;
        let t_data = teacher.data()?;

        if s_data.len() != t_data.len() {
            return Err(anyhow!("Tensor size mismatch"));
        }

        let mse = s_data.iter().zip(t_data.iter()).map(|(&s, &t)| (s - t).powi(2)).sum::<f32>()
            / s_data.len() as f32;

        Ok(mse)
    }

    /// Cross entropy between a teacher distribution and student log-scores.
    pub fn cross_entropy(&self, student: &Tensor, teacher: &Tensor) -> Result<f32> {
        let s_data = student.data()?;
        let t_data = teacher.data()?;

        if s_data.len() != t_data.len() {
            return Err(anyhow!("Tensor size mismatch"));
        }

        let ce = -t_data
            .iter()
            .zip(s_data.iter())
            .map(|(&t, &s)| if s > 0.0 { t * s.ln() } else { 0.0 })
            .sum::<f32>();

        Ok(ce)
    }
}

#[async_trait]
impl Distiller for KnowledgeDistiller {
    /// Train a student model from a teacher.
    ///
    /// Not implemented: distillation requires back-propagating the distillation
    /// loss into the student's parameters, and `trustformers-core` exposes no
    /// autodiff path over an arbitrary `Model` (`named_tensors_mut` gives write
    /// access to the weights, but no gradients). Rather than run a loop over
    /// random tensors and report a decaying loss that has nothing to do with
    /// either model, this reports that the capability is missing.
    ///
    /// The pieces that *are* implementable without gradients are available and
    /// real: [`KnowledgeDistiller::compute_distillation_loss`] on actual logits,
    /// and [`KnowledgeDistiller::evaluate_on`], which runs both models over a
    /// validation set and measures their agreement.
    async fn distill<T, S>(
        &self,
        _teacher: &T,
        _student: &S,
        _config: &DistillationConfig,
    ) -> Result<S>
    where
        T: crate::traits::Model + Sync,
        S: crate::traits::Model + Send,
    {
        Err(anyhow!(
            "knowledge distillation training is not implemented: updating the student requires \
             gradients of the distillation loss with respect to its parameters, and no autodiff \
             path exists over the generic `Model` trait. Use `compute_distillation_loss` for the \
             loss and `evaluate_on` to measure teacher/student agreement."
        ))
    }

    /// Measure teacher/student agreement.
    ///
    /// Not implemented in this form: agreement can only be measured by running
    /// both models over data, and this signature carries none. Use
    /// [`KnowledgeDistiller::evaluate_on`].
    fn evaluate<T, S>(&self, _teacher: &T, _student: &S) -> Result<f32>
    where
        T: crate::traits::Model,
        S: crate::traits::Model,
    {
        Err(anyhow!(
            "evaluate() cannot measure agreement without data; call \
             evaluate_on(teacher, student, validation_inputs) instead"
        ))
    }
}

impl KnowledgeDistiller {
    /// Measure how closely the student reproduces the teacher on real data.
    ///
    /// Both models are run over every validation input and their outputs are
    /// compared after temperature-softened softmax. The returned agreement is
    /// `1 - mean total-variation distance` between the two distributions,
    /// averaged over the batch, so it is 1.0 for identical outputs and 0.0 for
    /// disjoint ones.
    ///
    /// Errors when `validation_inputs` is empty (there is nothing to measure)
    /// or when the two models produce differently shaped outputs.
    pub fn evaluate_on<T, S>(
        &self,
        teacher: &T,
        student: &S,
        validation_inputs: &[Tensor],
    ) -> Result<f32>
    where
        T: crate::traits::Model<Input = Tensor, Output = Tensor>,
        S: crate::traits::Model<Input = Tensor, Output = Tensor>,
    {
        if validation_inputs.is_empty() {
            return Err(anyhow!(
                "cannot evaluate teacher/student agreement without validation inputs"
            ));
        }

        let mut agreement_sum = 0.0f64;

        for (index, input) in validation_inputs.iter().enumerate() {
            let teacher_logits = teacher.forward(input.clone())?;
            let student_logits = student.forward(input.clone())?;

            if teacher_logits.shape() != student_logits.shape() {
                return Err(anyhow!(
                    "validation input {}: teacher output shape {:?} does not match student \
                     output shape {:?}",
                    index,
                    teacher_logits.shape(),
                    student_logits.shape()
                ));
            }

            let teacher_probs = self.softmax_with_temperature(&teacher_logits)?;
            let student_probs = self.softmax_with_temperature(&student_logits)?;
            let teacher_data = teacher_probs.data()?;
            let student_data = student_probs.data()?;

            // Total variation distance = 0.5 * L1 distance between distributions.
            let l1: f32 =
                teacher_data.iter().zip(student_data.iter()).map(|(t, s)| (t - s).abs()).sum();
            let total_variation = (0.5 * l1).clamp(0.0, 1.0);
            agreement_sum += (1.0 - total_variation) as f64;
        }

        Ok((agreement_sum / validation_inputs.len() as f64) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;
    use crate::tensor::Tensor;
    use crate::traits::{Config, Model};
    use std::io::Read;

    // Mock configuration for testing
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct MockConfig {
        hidden_size: usize,
    }

    impl MockConfig {
        fn new() -> Self {
            Self { hidden_size: 768 }
        }
    }

    impl Config for MockConfig {
        fn architecture(&self) -> &'static str {
            "mock-model"
        }
    }

    // Mock student model for testing
    #[derive(Debug, Clone)]
    struct MockStudentModel {
        #[allow(dead_code)]
        id: String,
        config: MockConfig,
    }

    impl MockStudentModel {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                config: MockConfig::new(),
            }
        }
    }

    impl Model for MockStudentModel {
        type Config = MockConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, _input: Self::Input) -> Result<Self::Output> {
            Tensor::zeros(&[1, 10])
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            1000
        }
    }

    // Mock teacher model for testing
    #[derive(Debug, Clone)]
    struct MockTeacherModel {
        #[allow(dead_code)]
        id: String,
        config: MockConfig,
    }

    impl MockTeacherModel {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                config: MockConfig::new(),
            }
        }
    }

    impl Model for MockTeacherModel {
        type Config = MockConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, _input: Self::Input) -> Result<Self::Output> {
            Tensor::ones(&[1, 10])
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            5000
        }
    }

    // A student whose logits are strongly peaked, so its softened distribution
    // is measurably different from a uniform teacher.
    #[derive(Debug, Clone)]
    struct DivergentStudentModel {
        config: MockConfig,
    }

    impl DivergentStudentModel {
        fn new() -> Self {
            Self {
                config: MockConfig::new(),
            }
        }
    }

    impl Model for DivergentStudentModel {
        type Config = MockConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, _input: Self::Input) -> Result<Self::Output> {
            let mut logits = vec![0.0f32; 10];
            logits[0] = 20.0;
            Tensor::from_vec(logits, &[1, 10])
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            1000
        }
    }

    /// Regression test: `distill` used to run a loop over `Tensor::randn`
    /// tensors, print a convincing decaying loss, and only then return an
    /// error claiming the "training loop completed successfully". It must now
    /// refuse up front and say why.
    #[tokio::test]
    async fn test_distill_reports_missing_autodiff_instead_of_faking_training() {
        let distiller = KnowledgeDistiller::new(3.0);
        let teacher = MockTeacherModel::new("teacher");
        let student = MockStudentModel::new("student");

        let config = DistillationConfig {
            epochs: 3,
            batch_size: 4,
            learning_rate: 0.01,
            ..Default::default()
        };

        let error = distiller
            .distill(&teacher, &student, &config)
            .await
            .expect_err("no training can happen without gradients");
        let message = error.to_string();
        assert!(
            message.contains("not implemented"),
            "unexpected message: {message}"
        );
        assert!(
            !message.contains("completed successfully"),
            "the error must not claim that training succeeded: {message}"
        );
    }

    /// Regression test: `evaluate` returned a hardcoded 0.95 agreement without
    /// touching either model.
    #[test]
    fn test_evaluate_without_data_is_refused() {
        let distiller = KnowledgeDistiller::new(3.0);
        let teacher = MockTeacherModel::new("teacher");
        let student = MockStudentModel::new("student");

        let error = distiller
            .evaluate(&teacher, &student)
            .expect_err("agreement cannot be measured without data");
        assert!(error.to_string().contains("evaluate_on"));
    }

    /// `evaluate_on` must run both models and report a measured agreement.
    #[test]
    fn test_evaluate_on_measures_real_agreement() {
        let distiller = KnowledgeDistiller::new(1.0);
        let teacher = MockTeacherModel::new("teacher");
        let student = MockStudentModel::new("student");
        let inputs = vec![
            Tensor::zeros(&[1, 10]).expect("zeros failed"),
            Tensor::ones(&[1, 10]).expect("ones failed"),
        ];

        // The mocks emit constant logits (ones vs zeros); after softmax both
        // are the uniform distribution, so the agreement is exactly 1.0.
        let agreement =
            distiller.evaluate_on(&teacher, &student, &inputs).expect("evaluation failed");
        assert!(
            (agreement - 1.0).abs() < 1e-5,
            "uniform-vs-uniform agreement should be 1.0, got {agreement}"
        );
        assert_ne!(agreement, 0.95, "the old hardcoded value must not survive");

        // A student whose outputs really differ must score lower.
        let divergent = DivergentStudentModel::new();
        let divergent_agreement =
            distiller.evaluate_on(&teacher, &divergent, &inputs).expect("evaluation failed");
        assert!(
            divergent_agreement < 0.9,
            "a peaked student vs a uniform teacher must disagree, got {divergent_agreement}"
        );
    }

    #[test]
    fn test_evaluate_on_requires_inputs() {
        let distiller = KnowledgeDistiller::new(3.0);
        let teacher = MockTeacherModel::new("teacher");
        let student = MockStudentModel::new("student");
        assert!(distiller.evaluate_on(&teacher, &student, &[]).is_err());
    }

    #[test]
    fn test_distillation_loss_computation() {
        let distiller = KnowledgeDistiller::new(3.0);

        let student_logits =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let teacher_logits =
            Tensor::from_vec(vec![1.5, 2.5, 3.5], &[1, 3]).expect("Tensor from_vec failed");

        let loss = distiller.compute_distillation_loss(&student_logits, &teacher_logits);
        assert!(loss.is_ok(), "Loss computation should succeed");

        let loss_value = loss.expect("operation failed in test");
        assert!(loss_value >= 0.0, "Loss should be non-negative");
    }

    // ── DistillationConfig tests ──

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert!((config.temperature - 3.0).abs() < 1e-6);
        assert!((config.alpha - 0.7).abs() < 1e-6);
        assert!((config.learning_rate - 1e-4).abs() < 1e-8);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
        assert!(config.matched_layers.is_empty());
        assert!(!config.use_feature_distillation);
        assert!((config.feature_weight - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_clone() {
        let config = DistillationConfig {
            epochs: 42,
            temperature: 5.0,
            ..DistillationConfig::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.epochs, 42);
        assert!((cloned.temperature - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_config_custom_layers() {
        let mut config = DistillationConfig::default();
        config
            .matched_layers
            .insert("teacher.layer_0".to_string(), "student.layer_0".to_string());
        assert_eq!(config.matched_layers.len(), 1);
    }

    // ── DistillationLoss tests ──

    #[test]
    fn test_distillation_loss_kl_debug() {
        let loss = DistillationLoss::KLDivergence;
        assert_eq!(format!("{:?}", loss), "KLDivergence");
    }

    #[test]
    fn test_distillation_loss_mse_debug() {
        let loss = DistillationLoss::MSE;
        assert_eq!(format!("{:?}", loss), "MSE");
    }

    #[test]
    fn test_distillation_loss_cross_entropy_debug() {
        let loss = DistillationLoss::CrossEntropy;
        assert_eq!(format!("{:?}", loss), "CrossEntropy");
    }

    #[test]
    fn test_distillation_loss_custom_debug() {
        let loss = DistillationLoss::Custom(Box::new(|_a, _b| 0.0));
        assert_eq!(format!("{:?}", loss), "Custom(<closure>)");
    }

    #[test]
    fn test_distillation_loss_clone_kl() {
        let loss = DistillationLoss::KLDivergence;
        let cloned = loss.clone();
        assert_eq!(format!("{:?}", cloned), "KLDivergence");
    }

    #[test]
    fn test_distillation_loss_clone_mse() {
        let loss = DistillationLoss::MSE;
        let cloned = loss.clone();
        assert_eq!(format!("{:?}", cloned), "MSE");
    }

    #[test]
    fn test_distillation_loss_clone_custom_fallback() {
        // Custom loss falls back to KLDivergence on clone
        let loss = DistillationLoss::Custom(Box::new(|_a, _b| 42.0));
        let cloned = loss.clone();
        assert_eq!(format!("{:?}", cloned), "KLDivergence");
    }

    // ── KnowledgeDistiller loss computation tests ──

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let distiller = KnowledgeDistiller::new(1.0);
        let logits =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        // Identical inputs should produce KL ~ 0
        let loss = distiller
            .compute_distillation_loss(&logits, &logits)
            .expect("loss computation failed");
        assert!(
            loss.abs() < 1e-4,
            "KL divergence of identical distributions should be ~0, got {}",
            loss
        );
    }

    #[test]
    fn test_mse_loss_identical() {
        let distiller = KnowledgeDistiller::new(1.0).with_loss(DistillationLoss::MSE);
        let logits =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let loss = distiller
            .compute_distillation_loss(&logits, &logits)
            .expect("loss computation failed");
        assert!(loss.abs() < 1e-6, "MSE of identical inputs should be 0");
    }

    #[test]
    fn test_cross_entropy_loss() {
        let distiller = KnowledgeDistiller::new(1.0).with_loss(DistillationLoss::CrossEntropy);
        let student =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let teacher =
            Tensor::from_vec(vec![1.5, 2.5, 3.5], &[1, 3]).expect("Tensor from_vec failed");
        let loss = distiller
            .compute_distillation_loss(&student, &teacher)
            .expect("loss computation failed");
        assert!(loss >= 0.0, "Cross entropy should be non-negative");
    }

    #[test]
    fn test_softmax_with_temperature_high_temp() {
        let distiller = KnowledgeDistiller::new(100.0);
        let logits =
            Tensor::from_vec(vec![1.0, 10.0, 1.0], &[1, 3]).expect("Tensor from_vec failed");
        let probs = distiller.softmax_with_temperature(&logits).expect("softmax failed");
        let data = probs.data().expect("data extraction failed");
        // High temperature -> near-uniform distribution
        let diff = (data[0] - data[1]).abs();
        assert!(
            diff < 0.1,
            "High temperature should produce near-uniform distribution, diff={}",
            diff
        );
    }

    #[test]
    fn test_softmax_with_temperature_low_temp() {
        let distiller = KnowledgeDistiller::new(0.01);
        let logits =
            Tensor::from_vec(vec![1.0, 10.0, 1.0], &[1, 3]).expect("Tensor from_vec failed");
        let probs = distiller.softmax_with_temperature(&logits).expect("softmax failed");
        let data = probs.data().expect("data extraction failed");
        // Low temperature -> peaked distribution
        assert!(
            data[1] > 0.99,
            "Low temperature should produce peaked distribution at max"
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let distiller = KnowledgeDistiller::new(3.0);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5])
            .expect("Tensor from_vec failed");
        let probs = distiller.softmax_with_temperature(&logits).expect("softmax failed");
        let data = probs.data().expect("data extraction failed");
        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_knowledge_distiller_with_loss() {
        let distiller = KnowledgeDistiller::new(2.0).with_loss(DistillationLoss::MSE);
        let student =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let teacher =
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[1, 3]).expect("Tensor from_vec failed");
        let loss = distiller
            .compute_distillation_loss(&student, &teacher)
            .expect("loss computation failed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_kl_divergence_size_mismatch() {
        let distiller = KnowledgeDistiller::new(1.0);
        let student = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor from_vec failed");
        let teacher =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        // Note: softmax is applied first, but the underlying kl_divergence checks size
        // The softmax produces same-size tensors as input, so the mismatch is caught in kl_divergence
        let result = distiller.kl_divergence(&student, &teacher);
        assert!(result.is_err());
    }

    #[test]
    fn test_mse_loss_size_mismatch() {
        let distiller = KnowledgeDistiller::new(1.0);
        let student = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor from_vec failed");
        let teacher =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let result = distiller.mse_loss(&student, &teacher);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_size_mismatch() {
        let distiller = KnowledgeDistiller::new(1.0);
        let student = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor from_vec failed");
        let teacher =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor from_vec failed");
        let result = distiller.cross_entropy(&student, &teacher);
        assert!(result.is_err());
    }

    // ── DistillationStrategy tests ──

    #[test]
    fn test_distillation_strategy_variants() {
        let _response = DistillationStrategy::Response;
        let _feature = DistillationStrategy::Feature;
        let _attention = DistillationStrategy::Attention;
        let _combined = DistillationStrategy::Combined {
            response_weight: 0.5,
            feature_weight: 0.3,
            attention_weight: 0.2,
        };
    }

    // ── Helper struct tests ──

    #[test]
    fn test_feature_distiller_creation() {
        let mut mappings = HashMap::new();
        mappings.insert("layer_0".to_string(), "student_layer_0".to_string());
        let _distiller = super::FeatureDistiller::new(mappings);
    }

    #[test]
    fn test_response_distiller_creation() {
        let _distiller = super::ResponseDistiller::new(2.0);
    }

    #[test]
    fn test_attention_distiller_creation() {
        let layers = vec!["attn.0".to_string(), "attn.1".to_string()];
        let _distiller = super::AttentionDistiller::new(layers);
    }

    #[test]
    fn test_layer_distiller_creation() {
        let pairs = vec![
            ("t.layer_0".to_string(), "s.layer_0".to_string()),
            ("t.layer_1".to_string(), "s.layer_1".to_string()),
        ];
        let _distiller = super::LayerDistiller::new(pairs);
    }

    #[test]
    fn test_hidden_state_distiller_creation() {
        let _distiller = super::HiddenStateDistiller::new(768, 384);
    }

    /// Regression test: this used to assert the hardcoded 0.95 agreement.
    #[test]
    fn test_evaluate_no_longer_returns_a_constant() {
        let distiller = KnowledgeDistiller::new(3.0);
        let teacher = MockTeacherModel::new("teacher");
        let student = MockStudentModel::new("student");
        assert!(
            distiller.evaluate(&teacher, &student).is_err(),
            "no agreement figure may be produced without data"
        );
    }
}

/// Feature-based distillation
pub struct FeatureDistiller {
    #[allow(dead_code)]
    layer_mappings: HashMap<String, String>,
}

impl FeatureDistiller {
    pub fn new(layer_mappings: HashMap<String, String>) -> Self {
        Self { layer_mappings }
    }
}

/// Response-based distillation (output matching)
pub struct ResponseDistiller {
    #[allow(dead_code)]
    temperature: f32,
}

impl ResponseDistiller {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

/// Attention-based distillation for transformers
pub struct AttentionDistiller {
    #[allow(dead_code)]
    attention_layers: Vec<String>,
}

impl AttentionDistiller {
    pub fn new(attention_layers: Vec<String>) -> Self {
        Self { attention_layers }
    }
}

/// Layer-wise distillation
pub struct LayerDistiller {
    #[allow(dead_code)]
    layer_pairs: Vec<(String, String)>,
}

impl LayerDistiller {
    pub fn new(layer_pairs: Vec<(String, String)>) -> Self {
        Self { layer_pairs }
    }
}

/// Hidden state distillation
pub struct HiddenStateDistiller {
    #[allow(dead_code)]
    hidden_size_teacher: usize,
    #[allow(dead_code)]
    hidden_size_student: usize,
}

impl HiddenStateDistiller {
    pub fn new(hidden_size_teacher: usize, hidden_size_student: usize) -> Self {
        Self {
            hidden_size_teacher,
            hidden_size_student,
        }
    }
}

// Mock implementation for demonstration
#[allow(dead_code)]
struct MockDistilledModel;

impl crate::traits::Model for MockDistilledModel {
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
        1_000_000
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
