use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub activation_function: String,
    pub resid_pdrop: f32,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub model_type: String,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: None,
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt2".to_string(),
        }
    }
}

impl Config for Gpt2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.n_embd % self.n_head != 0 {
            return Err(invalid_config(
                "config_field",
                "n_embd must be divisible by n_head".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "GPT2"
    }
}

impl Gpt2Config {
    pub fn small() -> Self {
        Self::default()
    }

    pub fn medium() -> Self {
        Self {
            n_embd: 1024,
            n_head: 16,
            n_layer: 24,
            ..Self::default()
        }
    }

    pub fn large() -> Self {
        Self {
            n_embd: 1280,
            n_head: 20,
            n_layer: 36,
            ..Self::default()
        }
    }

    pub fn xl() -> Self {
        Self {
            n_embd: 1600,
            n_head: 25,
            n_layer: 48,
            ..Self::default()
        }
    }
}
