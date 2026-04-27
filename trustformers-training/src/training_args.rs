use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use trustformers_core::errors::{invalid_config, Result};

/// Configuration arguments for training, closely matching HuggingFace's TrainingArguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingArguments {
    /// The output directory where the model predictions and checkpoints will be written.
    pub output_dir: PathBuf,

    /// Whether to overwrite the content of the output directory.
    pub overwrite_output_dir: bool,

    /// Whether to do evaluation during training
    pub do_eval: bool,

    /// Whether to do prediction on the test set
    pub do_predict: bool,

    /// Number of steps used for a linear warmup from 0 to learning_rate
    pub warmup_steps: usize,

    /// Ratio of total training steps used for a linear warmup from 0 to learning_rate
    pub warmup_ratio: f32,

    /// Learning rate for the optimizer
    pub learning_rate: f32,

    /// Weight decay coefficient for regularization
    pub weight_decay: f32,

    /// Beta1 hyperparameter for the Adam optimizer
    pub adam_beta1: f32,

    /// Beta2 hyperparameter for the Adam optimizer
    pub adam_beta2: f32,

    /// Epsilon hyperparameter for the Adam optimizer
    pub adam_epsilon: f32,

    /// Maximum gradient norm for gradient clipping
    pub max_grad_norm: f32,

    /// Total number of training epochs to perform
    pub num_train_epochs: f32,

    /// Total number of training steps to perform (overrides num_train_epochs if set)
    pub max_steps: Option<usize>,

    /// Number of updates steps to accumulate before performing a backward/update pass
    pub gradient_accumulation_steps: usize,

    /// Batch size per device during training
    pub per_device_train_batch_size: usize,

    /// Batch size per device during evaluation
    pub per_device_eval_batch_size: usize,

    /// Number of subprocesses to use for data loading
    pub dataloader_num_workers: usize,

    /// Whether to pin memory in data loaders
    pub dataloader_pin_memory: bool,

    /// How often to save the model checkpoint
    pub save_steps: usize,

    /// Maximum number of checkpoints to keep
    pub save_total_limit: Option<usize>,

    /// How often to log training metrics
    pub logging_steps: usize,

    /// How often to run evaluation
    pub eval_steps: usize,

    /// Whether to run evaluation at the end of training
    pub eval_at_end: bool,

    /// Random seed for initialization
    pub seed: u64,

    /// Whether to use 16-bit mixed precision training
    pub fp16: bool,

    /// Whether to use bfloat16 mixed precision training
    pub bf16: bool,

    /// The name of the metric to use to compare two different models
    pub metric_for_best_model: Option<String>,

    /// Whether the metric_for_best_model should be maximized or not
    pub greater_is_better: Option<bool>,

    /// How many evaluation calls to wait before stopping training
    pub early_stopping_patience: Option<usize>,

    /// Minimum change in the monitored metric to qualify as an improvement
    pub early_stopping_threshold: Option<f32>,

    /// Whether to load the best model found during training at the end of training
    pub load_best_model_at_end: bool,

    /// Strategy to adopt during evaluation
    pub evaluation_strategy: EvaluationStrategy,

    /// Strategy to adopt for saving checkpoints
    pub save_strategy: SaveStrategy,

    /// The logging directory to use
    pub logging_dir: Option<PathBuf>,

    /// Whether to run training
    pub do_train: bool,

    /// Resume training from a checkpoint
    pub resume_from_checkpoint: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvaluationStrategy {
    /// No evaluation during training
    No,
    /// Evaluate every eval_steps
    Steps,
    /// Evaluate at the end of each epoch
    Epoch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SaveStrategy {
    /// No saving during training
    No,
    /// Save every save_steps
    Steps,
    /// Save at the end of each epoch
    Epoch,
}

impl Default for TrainingArguments {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./results"),
            overwrite_output_dir: false,
            do_eval: false,
            do_predict: false,
            warmup_steps: 0,
            warmup_ratio: 0.0,
            learning_rate: 5e-5,
            weight_decay: 0.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            max_grad_norm: 1.0,
            num_train_epochs: 3.0,
            max_steps: None,
            gradient_accumulation_steps: 1,
            per_device_train_batch_size: 8,
            per_device_eval_batch_size: 8,
            dataloader_num_workers: 0,
            dataloader_pin_memory: false,
            save_steps: 500,
            save_total_limit: None,
            logging_steps: 10,
            eval_steps: 500,
            eval_at_end: true,
            seed: 42,
            fp16: false,
            bf16: false,
            metric_for_best_model: None,
            greater_is_better: None,
            early_stopping_patience: None,
            early_stopping_threshold: None,
            load_best_model_at_end: false,
            evaluation_strategy: EvaluationStrategy::No,
            save_strategy: SaveStrategy::Steps,
            logging_dir: None,
            do_train: true,
            resume_from_checkpoint: None,
        }
    }
}

impl TrainingArguments {
    /// Create a new TrainingArguments with the specified output directory
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            ..Default::default()
        }
    }

    /// Calculate the total number of training steps
    pub fn get_total_steps(&self, num_examples: usize) -> usize {
        if let Some(max_steps) = self.max_steps {
            max_steps
        } else {
            let steps_per_epoch = num_examples.div_ceil(self.per_device_train_batch_size);
            (self.num_train_epochs * steps_per_epoch as f32) as usize
        }
    }

    /// Calculate the effective batch size (accounting for gradient accumulation)
    pub fn get_effective_batch_size(&self) -> usize {
        self.per_device_train_batch_size * self.gradient_accumulation_steps
    }

    /// Calculate the number of warmup steps
    pub fn get_warmup_steps(&self, total_steps: usize) -> usize {
        if self.warmup_steps > 0 {
            self.warmup_steps
        } else {
            (self.warmup_ratio * total_steps as f32) as usize
        }
    }

    /// Validate the training arguments
    pub fn validate(&self) -> Result<()> {
        if self.per_device_train_batch_size == 0 {
            return Err(invalid_config(
                "per_device_train_batch_size",
                "must be greater than 0",
            ));
        }

        if self.per_device_eval_batch_size == 0 {
            return Err(invalid_config(
                "per_device_eval_batch_size",
                "must be greater than 0",
            ));
        }

        if self.gradient_accumulation_steps == 0 {
            return Err(invalid_config(
                "gradient_accumulation_steps",
                "must be greater than 0",
            ));
        }

        if self.learning_rate <= 0.0 {
            return Err(invalid_config("learning_rate", "must be positive"));
        }

        if self.num_train_epochs <= 0.0 && self.max_steps.is_none() {
            return Err(invalid_config(
                "training_schedule",
                "either num_train_epochs or max_steps must be positive",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn default_args() -> TrainingArguments {
        TrainingArguments::new("/tmp/trustformers_test_output")
    }

    // ──────────────────── Default values ────────────────────

    #[test]
    fn test_default_learning_rate() {
        let args = default_args();
        assert!((args.learning_rate - 5e-5).abs() < 1e-9);
    }

    #[test]
    fn test_default_num_epochs() {
        let args = default_args();
        assert!((args.num_train_epochs - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_batch_size() {
        let args = default_args();
        assert_eq!(args.per_device_train_batch_size, 8);
        assert_eq!(args.per_device_eval_batch_size, 8);
    }

    #[test]
    fn test_default_gradient_accumulation() {
        let args = default_args();
        assert_eq!(args.gradient_accumulation_steps, 1);
    }

    #[test]
    fn test_default_max_grad_norm() {
        let args = default_args();
        assert!((args.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_evaluation_strategy_is_no() {
        let args = default_args();
        assert_eq!(args.evaluation_strategy, EvaluationStrategy::No);
    }

    #[test]
    fn test_default_save_strategy_is_steps() {
        let args = default_args();
        assert_eq!(args.save_strategy, SaveStrategy::Steps);
    }

    #[test]
    fn test_new_sets_output_dir() {
        let args = TrainingArguments::new("/tmp/my_output_dir");
        assert_eq!(args.output_dir, PathBuf::from("/tmp/my_output_dir"));
    }

    // ──────────────────── get_total_steps ────────────────────

    #[test]
    fn test_get_total_steps_uses_max_steps_when_set() {
        let mut args = default_args();
        args.max_steps = Some(100);
        let total = args.get_total_steps(1000);
        assert_eq!(total, 100);
    }

    #[test]
    fn test_get_total_steps_from_epochs() {
        let args = default_args(); // 3 epochs, batch_size 8
                                   // 100 examples / 8 = 13 steps/epoch (ceil), 13 * 3 = 39
        let total = args.get_total_steps(100);
        assert_eq!(total, 39, "expected 39 total steps, got {}", total);
    }

    #[test]
    fn test_get_total_steps_exact_division() {
        let args = default_args(); // 3 epochs, batch_size 8
                                   // 16 examples / 8 = 2 steps/epoch; 2 * 3 = 6
        let total = args.get_total_steps(16);
        assert_eq!(total, 6);
    }

    // ──────────────────── get_effective_batch_size ────────────────────

    #[test]
    fn test_get_effective_batch_size_no_accumulation() {
        let args = default_args();
        assert_eq!(args.get_effective_batch_size(), 8);
    }

    #[test]
    fn test_get_effective_batch_size_with_accumulation() {
        let mut args = default_args();
        args.gradient_accumulation_steps = 4;
        assert_eq!(args.get_effective_batch_size(), 32);
    }

    // ──────────────────── get_warmup_steps ────────────────────

    #[test]
    fn test_get_warmup_steps_explicit_steps() {
        let mut args = default_args();
        args.warmup_steps = 50;
        args.warmup_ratio = 0.1; // should be ignored
        let ws = args.get_warmup_steps(1000);
        assert_eq!(ws, 50);
    }

    #[test]
    fn test_get_warmup_steps_from_ratio() {
        let mut args = default_args();
        args.warmup_steps = 0;
        args.warmup_ratio = 0.1;
        let ws = args.get_warmup_steps(1000);
        assert_eq!(ws, 100);
    }

    #[test]
    fn test_get_warmup_steps_zero_by_default() {
        let args = default_args();
        let ws = args.get_warmup_steps(500);
        assert_eq!(ws, 0);
    }

    // ──────────────────── validate ────────────────────

    #[test]
    fn test_validate_valid_defaults() {
        let args = default_args();
        assert!(args.validate().is_ok(), "default args should be valid");
    }

    #[test]
    fn test_validate_zero_batch_size_fails() {
        let mut args = default_args();
        args.per_device_train_batch_size = 0;
        assert!(
            args.validate().is_err(),
            "zero batch size should be invalid"
        );
    }

    #[test]
    fn test_validate_zero_eval_batch_size_fails() {
        let mut args = default_args();
        args.per_device_eval_batch_size = 0;
        assert!(
            args.validate().is_err(),
            "zero eval batch size should be invalid"
        );
    }

    #[test]
    fn test_validate_zero_gradient_accumulation_fails() {
        let mut args = default_args();
        args.gradient_accumulation_steps = 0;
        assert!(
            args.validate().is_err(),
            "zero gradient_accumulation_steps should be invalid"
        );
    }

    #[test]
    fn test_validate_zero_learning_rate_fails() {
        let mut args = default_args();
        args.learning_rate = 0.0;
        assert!(
            args.validate().is_err(),
            "zero learning rate should be invalid"
        );
    }

    #[test]
    fn test_validate_negative_learning_rate_fails() {
        let mut args = default_args();
        args.learning_rate = -0.001;
        assert!(
            args.validate().is_err(),
            "negative learning rate should be invalid"
        );
    }

    #[test]
    fn test_validate_max_steps_overrides_zero_epochs() {
        let mut args = default_args();
        args.num_train_epochs = 0.0;
        args.max_steps = Some(100);
        assert!(
            args.validate().is_ok(),
            "max_steps should compensate for 0 epochs"
        );
    }

    #[test]
    fn test_validate_zero_epochs_no_max_steps_fails() {
        let mut args = default_args();
        args.num_train_epochs = 0.0;
        args.max_steps = None;
        assert!(
            args.validate().is_err(),
            "zero epochs without max_steps should fail"
        );
    }

    // ──────────────────── Enums ────────────────────

    #[test]
    fn test_evaluation_strategy_equality() {
        assert_eq!(EvaluationStrategy::No, EvaluationStrategy::No);
        assert_eq!(EvaluationStrategy::Steps, EvaluationStrategy::Steps);
        assert_eq!(EvaluationStrategy::Epoch, EvaluationStrategy::Epoch);
        assert_ne!(EvaluationStrategy::No, EvaluationStrategy::Epoch);
    }

    #[test]
    fn test_save_strategy_equality() {
        assert_eq!(SaveStrategy::No, SaveStrategy::No);
        assert_eq!(SaveStrategy::Steps, SaveStrategy::Steps);
        assert_eq!(SaveStrategy::Epoch, SaveStrategy::Epoch);
        assert_ne!(SaveStrategy::No, SaveStrategy::Steps);
    }
}
