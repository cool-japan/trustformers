use serde::{Deserialize, Serialize};
use trustformers_core::{
    errors::{invalid_config, Result},
    traits::Config,
};

/// Mamba model configuration
/// Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// State space dimension (d_state or N)
    pub d_state: usize,
    /// Local convolution width (d_conv)
    pub d_conv: usize,
    /// Expansion factor for inner dimension
    pub expand: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings (sequence length)
    pub max_position_embeddings: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Initializer range
    pub initializer_range: f32,
    /// Rescale prenorm residual
    pub rescale_prenorm_residual: bool,
    /// Use bias in linear layers
    pub use_bias: bool,
    /// Use conv bias
    pub use_conv_bias: bool,
    /// Time step rank (delta rank)
    pub dt_rank: Option<usize>,
    /// Time step minimum
    pub dt_min: f32,
    /// Time step maximum
    pub dt_max: f32,
    /// Time step initialization
    pub dt_init: String,
    /// Time step scale
    pub dt_scale: f32,
    /// Time step init floor
    pub dt_init_floor: f32,
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
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_layer: 24,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            initializer_range: 0.1,
            rescale_prenorm_residual: true,
            use_bias: false,
            use_conv_bias: true,
            dt_rank: None, // Auto-computed: ceil(d_model / 16)
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "random".to_string(),
            dt_scale: 1.0,
            dt_init_floor: 1e-4,
            pad_token_id: Some(0),
            bos_token_id: 0,
            eos_token_id: 0,
            tie_word_embeddings: true,
            model_type: "mamba".to_string(),
        }
    }
}

impl MambaConfig {
    /// Mamba-130M configuration
    pub fn mamba_130m() -> Self {
        Self {
            d_model: 768,
            n_layer: 24,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// Mamba-370M configuration
    pub fn mamba_370m() -> Self {
        Self {
            d_model: 1024,
            n_layer: 48,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// Mamba-790M configuration
    pub fn mamba_790m() -> Self {
        Self {
            d_model: 1536,
            n_layer: 48,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// Mamba-1.4B configuration
    pub fn mamba_1_4b() -> Self {
        Self {
            d_model: 2048,
            n_layer: 48,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// Mamba-2.8B configuration
    pub fn mamba_2_8b() -> Self {
        Self {
            d_model: 2560,
            n_layer: 64,
            vocab_size: 50280,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// Get the computed dt_rank (delta rank) if not explicitly set
    pub fn get_dt_rank(&self) -> usize {
        self.dt_rank.unwrap_or_else(|| {
            // Standard computation: ceil(d_model / 16)
            self.d_model.div_ceil(16)
        })
    }

    /// Get the inner dimension (d_inner)
    pub fn get_d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "state-spaces/mamba-130m" | "mamba-130m" => Some(Self::mamba_130m()),
            "state-spaces/mamba-370m" | "mamba-370m" => Some(Self::mamba_370m()),
            "state-spaces/mamba-790m" | "mamba-790m" => Some(Self::mamba_790m()),
            "state-spaces/mamba-1.4b" | "mamba-1.4b" => Some(Self::mamba_1_4b()),
            "state-spaces/mamba-2.8b" | "mamba-2.8b" => Some(Self::mamba_2_8b()),
            _ => None,
        }
    }
}

impl Config for MambaConfig {
    fn architecture(&self) -> &'static str {
        "mamba"
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(invalid_config(
                "config_field",
                "d_model must be greater than 0",
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MambaConfig::default();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.d_conv, 4);
        assert_eq!(config.expand, 2);
        assert_eq!(config.n_layer, 24);
    }

    #[test]
    fn test_dt_rank_computation() {
        let config = MambaConfig::default();
        assert_eq!(config.get_dt_rank(), 48); // ceil(768 / 16) = 48

        let config_with_dt_rank = MambaConfig {
            dt_rank: Some(32),
            ..Default::default()
        };
        assert_eq!(config_with_dt_rank.get_dt_rank(), 32);
    }

    #[test]
    fn test_d_inner_computation() {
        let config = MambaConfig::default();
        assert_eq!(config.get_d_inner(), 1536); // 768 * 2
    }

    #[test]
    fn test_predefined_configs() {
        let config_130m = MambaConfig::mamba_130m();
        assert_eq!(config_130m.d_model, 768);
        assert_eq!(config_130m.n_layer, 24);

        let config_2_8b = MambaConfig::mamba_2_8b();
        assert_eq!(config_2_8b.d_model, 2560);
        assert_eq!(config_2_8b.n_layer, 64);
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = MambaConfig::from_pretrained_name("state-spaces/mamba-130m");
        assert!(config.is_some());
        assert_eq!(config.expect("operation failed").d_model, 768);

        let config = MambaConfig::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_config_trait() {
        let config = MambaConfig::default();
        assert_eq!(config.architecture(), "mamba");
        assert!(config.validate().is_ok());
    }

    // ---- Mamba-2.8B specific defaults ----

    #[test]
    fn test_mamba_2_8b_d_model() {
        let config = MambaConfig::mamba_2_8b();
        assert_eq!(config.d_model, 2560, "Mamba-2.8B d_model must be 2560");
    }

    #[test]
    fn test_mamba_2_8b_n_layers() {
        let config = MambaConfig::mamba_2_8b();
        assert_eq!(config.n_layer, 64, "Mamba-2.8B must have 64 layers");
    }

    #[test]
    fn test_mamba_2_8b_d_state() {
        let config = MambaConfig::mamba_2_8b();
        assert_eq!(config.d_state, 16, "Mamba-2.8B d_state must be 16");
    }

    #[test]
    fn test_mamba_2_8b_expand() {
        let config = MambaConfig::mamba_2_8b();
        assert_eq!(config.expand, 2, "Mamba-2.8B expand must be 2");
    }

    // ---- dt_rank "auto" = ceil(d_model/16) ----

    #[test]
    fn test_dt_rank_auto_130m() {
        let config = MambaConfig::mamba_130m();
        // ceil(768 / 16) = 48
        assert_eq!(config.get_dt_rank(), 48);
    }

    #[test]
    fn test_dt_rank_auto_370m() {
        let config = MambaConfig::mamba_370m();
        // ceil(1024 / 16) = 64
        assert_eq!(config.get_dt_rank(), 64);
    }

    #[test]
    fn test_dt_rank_auto_790m() {
        let config = MambaConfig::mamba_790m();
        // ceil(1536 / 16) = 96
        assert_eq!(config.get_dt_rank(), 96);
    }

    #[test]
    fn test_dt_rank_auto_1_4b() {
        let config = MambaConfig::mamba_1_4b();
        // ceil(2048 / 16) = 128
        assert_eq!(config.get_dt_rank(), 128);
    }

    #[test]
    fn test_dt_rank_auto_2_8b() {
        let config = MambaConfig::mamba_2_8b();
        // ceil(2560 / 16) = 160
        assert_eq!(config.get_dt_rank(), 160);
    }

    #[test]
    fn test_dt_rank_explicit_overrides_auto() {
        let config = MambaConfig {
            d_model: 768,
            dt_rank: Some(100),
            ..Default::default()
        };
        assert_eq!(
            config.get_dt_rank(),
            100,
            "Explicit dt_rank should override auto"
        );
    }

    // ---- d_inner = expand * d_model ----

    #[test]
    fn test_d_inner_2_8b() {
        let config = MambaConfig::mamba_2_8b();
        assert_eq!(
            config.get_d_inner(),
            config.expand * config.d_model,
            "d_inner = expand * d_model"
        );
    }

    #[test]
    fn test_d_inner_370m() {
        let config = MambaConfig::mamba_370m();
        // 1024 * 2 = 2048
        assert_eq!(config.get_d_inner(), 2048);
    }

    // ---- d_conv = 4 (default SSM conv width) ----

    #[test]
    fn test_d_conv_default_is_4() {
        let config = MambaConfig::default();
        assert_eq!(config.d_conv, 4);
    }

    #[test]
    fn test_d_conv_preserved_in_presets() {
        for d_model in [768_usize, 1024, 1536, 2048, 2560] {
            let config = MambaConfig {
                d_model,
                ..Default::default()
            };
            assert_eq!(config.d_conv, 4, "d_conv must be 4 for d_model={}", d_model);
        }
    }

    // ---- conv_bias ----

    #[test]
    fn test_use_conv_bias_default_true() {
        let config = MambaConfig::default();
        assert!(
            config.use_conv_bias,
            "use_conv_bias must be true by default"
        );
    }

    #[test]
    fn test_use_bias_default_false() {
        let config = MambaConfig::default();
        assert!(
            !config.use_bias,
            "use_bias must be false by default (no linear bias)"
        );
    }

    // ---- Validation ----

    #[test]
    fn test_validation_fails_zero_d_model() {
        let config = MambaConfig {
            d_model: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_fails_zero_n_layer() {
        let config = MambaConfig {
            n_layer: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_fails_zero_vocab_size() {
        let config = MambaConfig {
            vocab_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_all_presets_validate() {
        for config in [
            MambaConfig::mamba_130m(),
            MambaConfig::mamba_370m(),
            MambaConfig::mamba_790m(),
            MambaConfig::mamba_1_4b(),
            MambaConfig::mamba_2_8b(),
        ] {
            assert!(
                config.validate().is_ok(),
                "Preset with d_model={} failed validation",
                config.d_model
            );
        }
    }

    // ---- from_pretrained short names ----

    #[test]
    fn test_from_pretrained_short_name_2_8b() {
        let config = MambaConfig::from_pretrained_name("mamba-2.8b");
        assert!(config.is_some());
        assert_eq!(config.expect("mamba-2.8b config").d_model, 2560);
    }

    #[test]
    fn test_from_pretrained_short_name_790m() {
        let config = MambaConfig::from_pretrained_name("mamba-790m");
        assert!(config.is_some());
        assert_eq!(config.expect("mamba-790m config").d_model, 1536);
    }

    // ---- Parameter ordering: larger model → larger d_model ----

    #[test]
    fn test_d_model_increases_with_model_size() {
        let configs = [
            MambaConfig::mamba_130m(),
            MambaConfig::mamba_370m(),
            MambaConfig::mamba_790m(),
            MambaConfig::mamba_1_4b(),
            MambaConfig::mamba_2_8b(),
        ];
        for i in 1..configs.len() {
            assert!(
                configs[i].d_model >= configs[i - 1].d_model,
                "d_model must be non-decreasing with model size"
            );
        }
    }

    // ---- Serialization ----

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = MambaConfig::mamba_2_8b();
        let json = serde_json::to_string(&config).expect("serialize MambaConfig");
        let restored: MambaConfig = serde_json::from_str(&json).expect("deserialize MambaConfig");
        assert_eq!(config.d_model, restored.d_model);
        assert_eq!(config.n_layer, restored.n_layer);
        assert_eq!(config.d_state, restored.d_state);
    }

    #[test]
    fn test_config_clone_preserves_fields() {
        let config = MambaConfig::mamba_2_8b();
        let cloned = config.clone();
        assert_eq!(config.d_model, cloned.d_model);
        assert_eq!(config.expand, cloned.expand);
        assert_eq!(config.d_conv, cloned.d_conv);
    }
}
