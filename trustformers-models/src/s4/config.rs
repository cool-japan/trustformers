use serde::{Deserialize, Serialize};
use trustformers_core::{
    errors::{invalid_config, Result},
    traits::Config,
};

/// S4 (Structured State Space) model configuration
/// Reference: "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S4Config {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// State space dimension (d_state or N)
    pub d_state: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings (sequence length)
    pub max_position_embeddings: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Initializer range
    pub initializer_range: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Use bias in linear layers
    pub use_bias: bool,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
    /// Beginning of sequence token ID
    pub bos_token_id: u32,
    /// End of sequence token ID
    pub eos_token_id: u32,
    /// Whether to tie word embeddings
    pub tie_word_embeddings: bool,
    /// Model type identifier
    pub model_type: String,
    /// S4 specific parameters
    /// HiPPO matrix initialization method
    pub hippo_matrix: String,
    /// Discretization method for continuous-time system
    pub discretization: String,
    /// Time step for discretization
    pub dt: f32,
    /// Learning rate multiplier for state space parameters
    pub lr_mult: f32,
    /// Whether to use bidirectional processing
    pub bidirectional: bool,
    /// Postact function for S4 blocks
    pub postact: String,
    /// Transposed parameter initialization
    pub transposed: bool,
    /// Number of copies for multi-input multi-output
    pub n_ssm: Option<usize>,
}

impl Default for S4Config {
    fn default() -> Self {
        Self {
            d_model: 768,
            d_state: 64,
            n_layer: 12,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            dropout: 0.1,
            use_bias: true,
            pad_token_id: Some(50256),
            bos_token_id: 50256,
            eos_token_id: 50256,
            tie_word_embeddings: false,
            model_type: "s4".to_string(),
            hippo_matrix: "legs".to_string(),
            discretization: "zoh".to_string(), // Zero-order hold
            dt: 0.001,
            lr_mult: 1.0,
            bidirectional: false,
            postact: "glu".to_string(),
            transposed: true,
            n_ssm: None, // Auto-computed: d_model
        }
    }
}

impl S4Config {
    /// S4-Small configuration (similar to BERT-base)
    pub fn s4_small() -> Self {
        Self {
            d_model: 768,
            d_state: 64,
            n_layer: 12,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            ..Default::default()
        }
    }

    /// S4-Base configuration
    pub fn s4_base() -> Self {
        Self {
            d_model: 768,
            d_state: 64,
            n_layer: 12,
            vocab_size: 50257,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// S4-Large configuration
    pub fn s4_large() -> Self {
        Self {
            d_model: 1024,
            d_state: 64,
            n_layer: 24,
            vocab_size: 50257,
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }

    /// S4-Long configuration for very long sequences
    pub fn s4_long() -> Self {
        Self {
            d_model: 768,
            d_state: 64,
            n_layer: 6,
            vocab_size: 50257,
            max_position_embeddings: 16384,
            dt: 0.01, // Larger time step for longer sequences
            ..Default::default()
        }
    }

    /// Get the number of SSM copies (n_ssm) if not explicitly set
    pub fn get_n_ssm(&self) -> usize {
        self.n_ssm.unwrap_or(self.d_model)
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "s4-small" => Some(Self::s4_small()),
            "s4-base" => Some(Self::s4_base()),
            "s4-large" => Some(Self::s4_large()),
            "s4-long" => Some(Self::s4_long()),
            _ => None,
        }
    }

    /// Validate HiPPO matrix type
    pub fn validate_hippo_matrix(&self) -> Result<()> {
        match self.hippo_matrix.as_str() {
            "legs" | "legt" | "lagt" | "fourier" | "random" => Ok(()),
            _ => Err(trustformers_core::errors::invalid_config(
                "hippo_matrix",
                format!("Invalid HiPPO matrix type: {}", self.hippo_matrix),
            )),
        }
    }

    /// Validate discretization method
    pub fn validate_discretization(&self) -> Result<()> {
        match self.discretization.as_str() {
            "zoh" | "bilinear" | "euler" | "backward_euler" => Ok(()),
            _ => Err(trustformers_core::errors::invalid_config(
                "discretization",
                format!("Invalid discretization method: {}", self.discretization),
            )),
        }
    }
}

impl Config for S4Config {
    fn architecture(&self) -> &'static str {
        "s4"
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(invalid_config(
                "config_field",
                "d_model must be greater than 0",
            ));
        }
        if self.d_state == 0 {
            return Err(invalid_config(
                "config_field",
                "d_state must be greater than 0",
            ));
        }
        if self.n_layer == 0 {
            return Err(invalid_config(
                "config_field",
                "n_layer must be greater than 0",
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be greater than 0",
            ));
        }
        if self.dt <= 0.0 {
            return Err(invalid_config("config_field", "dt must be greater than 0"));
        }

        // Validate S4-specific parameters
        self.validate_hippo_matrix()?;
        self.validate_discretization()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = S4Config::default();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.d_state, 64);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.hippo_matrix, "legs");
        assert_eq!(config.discretization, "zoh");
    }

    #[test]
    fn test_n_ssm_computation() {
        let config = S4Config::default();
        assert_eq!(config.get_n_ssm(), 768); // d_model

        let config_with_n_ssm = S4Config {
            n_ssm: Some(128),
            ..Default::default()
        };
        assert_eq!(config_with_n_ssm.get_n_ssm(), 128);
    }

    #[test]
    fn test_predefined_configs() {
        let config_small = S4Config::s4_small();
        assert_eq!(config_small.d_model, 768);
        assert_eq!(config_small.n_layer, 12);

        let config_large = S4Config::s4_large();
        assert_eq!(config_large.d_model, 1024);
        assert_eq!(config_large.n_layer, 24);

        let config_long = S4Config::s4_long();
        assert_eq!(config_long.max_position_embeddings, 16384);
        assert_eq!(config_long.dt, 0.01);
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = S4Config::from_pretrained_name("s4-base");
        assert!(config.is_some());
        assert_eq!(config.expect("operation failed").d_model, 768);

        let config = S4Config::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_config_validation() {
        let config = S4Config::default();
        assert!(config.validate().is_ok());

        let invalid_config = S4Config {
            d_model: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_hippo = S4Config {
            hippo_matrix: "invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_hippo.validate().is_err());

        let invalid_discretization = S4Config {
            discretization: "invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_discretization.validate().is_err());
    }

    #[test]
    fn test_config_trait() {
        let config = S4Config::default();
        assert_eq!(config.architecture(), "s4");
        assert!(config.validate().is_ok());
    }
}
