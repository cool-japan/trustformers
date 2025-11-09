//! Training configuration structures and utilities.
//!
//! This module provides comprehensive configuration options for tokenizer training,
//! including basic training parameters and advanced configuration with validation,
//! checkpointing, and optimization features.

use serde::{Deserialize, Serialize};

/// Basic training configuration for tokenizer algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency threshold for including tokens
    pub min_frequency: usize,
    /// Special tokens to include in vocabulary
    pub special_tokens: Vec<String>,
    /// End-of-word suffix for subword tokenization
    pub end_of_word_suffix: String,
    /// Maximum input characters per word
    pub max_input_chars_per_word: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
            end_of_word_suffix: "##".to_string(),
            max_input_chars_per_word: 100,
        }
    }
}

/// Advanced training configuration with metrics and validation features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTrainingConfig {
    /// Base training configuration
    pub base_config: TrainingConfig,
    /// Fraction of data to use for validation
    pub validation_split: f64,
    /// Whether to enable comprehensive metrics tracking
    pub enable_metrics: bool,
    /// Whether to save training checkpoints
    pub save_checkpoints: bool,
    /// Directory for saving checkpoints
    pub checkpoint_dir: Option<String>,
    /// Maximum training time in seconds
    pub max_training_time: Option<f64>,
    /// Number of iterations without improvement before early stopping
    pub early_stopping_patience: usize,
    /// Minimum improvement threshold for continuing training
    pub min_improvement: f64,
}

impl Default for AdvancedTrainingConfig {
    fn default() -> Self {
        Self {
            base_config: TrainingConfig::default(),
            validation_split: 0.1,
            enable_metrics: true,
            save_checkpoints: false,
            checkpoint_dir: None,
            max_training_time: None,
            early_stopping_patience: 3,
            min_improvement: 0.01,
        }
    }
}

impl AdvancedTrainingConfig {
    /// Create a new advanced config from a base config.
    pub fn from_base_config(base_config: TrainingConfig) -> Self {
        Self {
            base_config,
            ..Default::default()
        }
    }

    /// Enable checkpointing with a specified directory.
    pub fn with_checkpointing<S: Into<String>>(mut self, checkpoint_dir: S) -> Self {
        self.save_checkpoints = true;
        self.checkpoint_dir = Some(checkpoint_dir.into());
        self
    }

    /// Set maximum training time.
    pub fn with_max_training_time(mut self, max_time: f64) -> Self {
        self.max_training_time = Some(max_time);
        self
    }

    /// Configure early stopping parameters.
    pub fn with_early_stopping(mut self, patience: usize, min_improvement: f64) -> Self {
        self.early_stopping_patience = patience;
        self.min_improvement = min_improvement;
        self
    }

    /// Set validation split fraction.
    pub fn with_validation_split(mut self, split: f64) -> Self {
        self.validation_split = split.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.base_config.vocab_size == 0 {
            return Err("Vocabulary size must be greater than 0".to_string());
        }

        if self.base_config.min_frequency == 0 {
            return Err("Minimum frequency must be greater than 0".to_string());
        }

        if self.validation_split < 0.0 || self.validation_split >= 1.0 {
            return Err("Validation split must be between 0.0 and 1.0".to_string());
        }

        if self.early_stopping_patience == 0 {
            return Err("Early stopping patience must be greater than 0".to_string());
        }

        if self.min_improvement < 0.0 {
            return Err("Minimum improvement must be non-negative".to_string());
        }

        if let Some(max_time) = self.max_training_time {
            if max_time <= 0.0 {
                return Err("Maximum training time must be positive".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.vocab_size, 30000);
        assert_eq!(config.min_frequency, 2);
        assert_eq!(config.special_tokens.len(), 5);
        assert_eq!(config.end_of_word_suffix, "##");
        assert_eq!(config.max_input_chars_per_word, 100);
    }

    #[test]
    fn test_default_advanced_config() {
        let config = AdvancedTrainingConfig::default();
        assert_eq!(config.validation_split, 0.1);
        assert!(config.enable_metrics);
        assert!(!config.save_checkpoints);
        assert_eq!(config.early_stopping_patience, 3);
        assert_eq!(config.min_improvement, 0.01);
    }

    #[test]
    fn test_advanced_config_builders() {
        let base_config = TrainingConfig {
            vocab_size: 50000,
            ..Default::default()
        };

        let advanced_config = AdvancedTrainingConfig::from_base_config(base_config)
            .with_checkpointing("/tmp/checkpoints")
            .with_max_training_time(3600.0)
            .with_early_stopping(5, 0.001)
            .with_validation_split(0.2);

        assert_eq!(advanced_config.base_config.vocab_size, 50000);
        assert!(advanced_config.save_checkpoints);
        assert_eq!(
            advanced_config.checkpoint_dir,
            Some("/tmp/checkpoints".to_string())
        );
        assert_eq!(advanced_config.max_training_time, Some(3600.0));
        assert_eq!(advanced_config.early_stopping_patience, 5);
        assert_eq!(advanced_config.min_improvement, 0.001);
        assert_eq!(advanced_config.validation_split, 0.2);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AdvancedTrainingConfig::default();
        assert!(config.validate().is_ok());

        config.base_config.vocab_size = 0;
        assert!(config.validate().is_err());

        config.base_config.vocab_size = 1000;
        config.validation_split = 1.5;
        assert!(config.validate().is_err());

        config.validation_split = 0.1;
        config.early_stopping_patience = 0;
        assert!(config.validate().is_err());

        config.early_stopping_patience = 3;
        config.min_improvement = -0.1;
        assert!(config.validate().is_err());

        config.min_improvement = 0.01;
        config.max_training_time = Some(-100.0);
        assert!(config.validate().is_err());

        config.max_training_time = Some(3600.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = AdvancedTrainingConfig {
            base_config: TrainingConfig {
                vocab_size: 50000,
                min_frequency: 5,
                special_tokens: vec!["<pad>".to_string(), "<unk>".to_string()],
                end_of_word_suffix: "##".to_string(),
                max_input_chars_per_word: 200,
            },
            validation_split: 0.2,
            enable_metrics: true,
            save_checkpoints: true,
            checkpoint_dir: Some("/tmp/checkpoints".to_string()),
            max_training_time: Some(3600.0),
            early_stopping_patience: 5,
            min_improvement: 0.001,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: AdvancedTrainingConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.base_config.vocab_size,
            deserialized.base_config.vocab_size
        );
        assert_eq!(config.validation_split, deserialized.validation_split);
        assert_eq!(config.enable_metrics, deserialized.enable_metrics);
        assert_eq!(config.save_checkpoints, deserialized.save_checkpoints);
        assert_eq!(config.checkpoint_dir, deserialized.checkpoint_dir);
        assert_eq!(config.max_training_time, deserialized.max_training_time);
        assert_eq!(
            config.early_stopping_patience,
            deserialized.early_stopping_patience
        );
        assert_eq!(config.min_improvement, deserialized.min_improvement);
    }
}
