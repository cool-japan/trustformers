use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptJConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_positions: usize,
    pub rotary_dim: usize, // For RoPE (Rotary Position Embedding)
    pub activation_function: String,
    pub resid_pdrop: f32,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub use_cache: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub model_type: String,
}

impl Default for GptJConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50400,
            n_embd: 4096,
            n_layer: 28,
            n_head: 16,
            n_positions: 2048,
            rotary_dim: 64,
            activation_function: "gelu_new".to_string(),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gptj".to_string(),
        }
    }
}

impl Config for GptJConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.n_embd % self.n_head != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "n_embd must be divisible by n_head".to_string(),
                ),
            );
        }
        if self.rotary_dim > self.n_embd / self.n_head {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "rotary_dim cannot be larger than head_dim".to_string(),
                ),
            );
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "GPT-J"
    }
}

impl GptJConfig {
    /// GPT-J 6B configuration (the main publicly available model)
    pub fn gpt_j_6b() -> Self {
        Self {
            vocab_size: 50400,
            n_embd: 4096,
            n_layer: 28,
            n_head: 16,
            n_positions: 2048,
            rotary_dim: 64,
            activation_function: "gelu_new".to_string(),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gptj".to_string(),
        }
    }

    /// Create GPT-J configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("6b") {
            Self::gpt_j_6b()
        } else {
            // Default to 6B (the main model)
            Self::gpt_j_6b()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}
