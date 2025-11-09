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
            let steps_per_epoch = (num_examples + self.per_device_train_batch_size - 1)
                / self.per_device_train_batch_size;
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
