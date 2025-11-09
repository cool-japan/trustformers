//! Configuration structures for RLHF training.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Overall RLHF training configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RLHFConfig {
    /// Configuration for supervised fine-tuning phase
    pub sft: SFTConfig,
    /// Configuration for reward model training
    pub reward_model: RewardModelConfig,
    /// Configuration for PPO training
    pub ppo: PPOConfig,
    /// Configuration for DPO training
    pub dpo: DPOConfig,
    /// Configuration for Constitutional AI
    pub constitutional: ConstitutionalConfig,
    /// General training settings
    pub general: GeneralConfig,
}

/// Supervised Fine-Tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFTConfig {
    /// Path to the SFT dataset
    pub dataset_path: PathBuf,
    /// Maximum sequence length
    pub max_length: usize,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Whether to use response templates
    pub use_response_template: bool,
    /// Response template format
    pub response_template: Option<String>,
}

/// Reward model training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardModelConfig {
    /// Path to preference comparison dataset
    pub dataset_path: PathBuf,
    /// Maximum sequence length
    pub max_length: usize,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Margin for ranking loss
    pub margin: f64,
    /// Whether to normalize rewards
    pub normalize_rewards: bool,
    /// Reward model architecture type
    pub model_type: RewardModelType,
    /// Hidden size for reward head
    pub reward_head_hidden_size: usize,
    /// Dropout rate
    pub dropout: f64,
}

/// PPO training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    /// Number of PPO epochs
    pub ppo_epochs: u32,
    /// Mini-batch size for PPO updates
    pub mini_batch_size: usize,
    /// Clipping parameter for PPO
    pub clip_param: f64,
    /// Value function coefficient
    pub vf_coef: f64,
    /// Entropy coefficient
    pub entropy_coef: f64,
    /// Learning rate for policy
    pub policy_lr: f64,
    /// Learning rate for value function
    pub value_lr: f64,
    /// KL divergence penalty coefficient
    pub kl_penalty: f64,
    /// Target KL divergence
    pub target_kl: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Number of generation steps per iteration
    pub generation_batch_size: usize,
    /// Maximum response length during generation
    pub max_response_length: usize,
    /// Temperature for sampling
    pub temperature: f64,
    /// Top-p for nucleus sampling
    pub top_p: f64,
}

/// Direct Preference Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPOConfig {
    /// Path to preference dataset
    pub dataset_path: PathBuf,
    /// Beta parameter for DPO loss
    pub beta: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: u32,
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Reference model for KL regularization
    pub reference_model_path: Option<PathBuf>,
    /// Whether to use reference-free DPO
    pub reference_free: bool,
    /// Label smoothing parameter
    pub label_smoothing: f64,
}

/// Constitutional AI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalConfig {
    /// Path to constitutional principles file
    pub principles_path: PathBuf,
    /// Number of constitutional training iterations
    pub num_iterations: u32,
    /// Learning rate for constitutional training
    pub learning_rate: f64,
    /// Batch size for constitutional training
    pub batch_size: usize,
    /// Whether to use critique and revision
    pub use_critique_revision: bool,
    /// Temperature for constitutional sampling
    pub temperature: f64,
    /// Penalty weight for principle violations
    pub violation_penalty: f64,
    /// Whether to use self-improvement
    pub self_improvement: bool,
}

/// General RLHF training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Model name or path
    pub model_name_or_path: String,
    /// Tokenizer name or path
    pub tokenizer_name_or_path: Option<String>,
    /// Output directory for checkpoints
    pub output_dir: PathBuf,
    /// Logging directory
    pub logging_dir: Option<PathBuf>,
    /// Device for training (cuda, cpu, mps)
    pub device: String,
    /// Number of devices for parallel training
    pub num_devices: usize,
    /// Random seed
    pub seed: u64,
    /// Whether to save intermediate checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint saving interval
    pub save_steps: usize,
    /// Evaluation interval
    pub eval_steps: usize,
    /// Whether to log to wandb
    pub use_wandb: bool,
    /// Wandb project name
    pub wandb_project: Option<String>,
    /// Maximum memory usage per device (GB)
    pub max_memory_per_device: Option<f64>,
}

/// Reward model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardModelType {
    /// Simple linear head on top of base model
    Linear,
    /// Multi-layer perceptron head
    MLP,
    /// Transformer head with attention
    Transformer,
    /// Ensemble of multiple heads
    Ensemble,
}

impl Default for SFTConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("data/sft_dataset.json"),
            max_length: 2048,
            epochs: 3,
            learning_rate: 5e-5,
            batch_size: 8,
            gradient_accumulation_steps: 1,
            use_response_template: true,
            response_template: Some("Human: {prompt}\n\nAssistant: {response}".to_string()),
        }
    }
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("data/reward_dataset.json"),
            max_length: 2048,
            epochs: 1,
            learning_rate: 1e-5,
            batch_size: 4,
            margin: 0.0,
            normalize_rewards: true,
            model_type: RewardModelType::Linear,
            reward_head_hidden_size: 512,
            dropout: 0.1,
        }
    }
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            ppo_epochs: 4,
            mini_batch_size: 4,
            clip_param: 0.2,
            vf_coef: 0.1,
            entropy_coef: 0.01,
            policy_lr: 1e-6,
            value_lr: 1e-5,
            kl_penalty: 0.1,
            target_kl: 0.01,
            max_grad_norm: 1.0,
            generation_batch_size: 64,
            max_response_length: 1024,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("data/dpo_dataset.json"),
            beta: 0.1,
            learning_rate: 1e-6,
            epochs: 1,
            batch_size: 4,
            max_length: 2048,
            reference_model_path: None,
            reference_free: false,
            label_smoothing: 0.0,
        }
    }
}

impl Default for ConstitutionalConfig {
    fn default() -> Self {
        Self {
            principles_path: PathBuf::from("data/constitutional_principles.json"),
            num_iterations: 3,
            learning_rate: 1e-6,
            batch_size: 4,
            use_critique_revision: true,
            temperature: 1.0,
            violation_penalty: 1.0,
            self_improvement: true,
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            model_name_or_path: "gpt2".to_string(),
            tokenizer_name_or_path: None,
            output_dir: PathBuf::from("./output"),
            logging_dir: Some(PathBuf::from("./logs")),
            device: "cpu".to_string(),
            num_devices: 1,
            seed: 42,
            save_checkpoints: true,
            save_steps: 1000,
            eval_steps: 500,
            use_wandb: false,
            wandb_project: None,
            max_memory_per_device: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rlhf_config_default() {
        let config = RLHFConfig::default();
        assert_eq!(config.sft.epochs, 3);
        assert_eq!(config.reward_model.model_type, RewardModelType::Linear);
        assert_eq!(config.ppo.clip_param, 0.2);
        assert_eq!(config.dpo.beta, 0.1);
    }

    #[test]
    fn test_reward_model_type_serialization() {
        let model_type = RewardModelType::MLP;
        let serialized = serde_json::to_string(&model_type).unwrap();
        let deserialized: RewardModelType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(model_type, deserialized);
    }

    #[test]
    fn test_config_validation() {
        let config = RLHFConfig::default();

        // Test that learning rates are reasonable
        assert!(config.sft.learning_rate > 0.0);
        assert!(config.ppo.policy_lr > 0.0);
        assert!(config.dpo.learning_rate > 0.0);

        // Test that batch sizes are positive
        assert!(config.sft.batch_size > 0);
        assert!(config.reward_model.batch_size > 0);
        assert!(config.ppo.mini_batch_size > 0);
    }
}
