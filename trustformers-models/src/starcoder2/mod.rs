//! # StarCoder2
//!
//! BigCode StarCoder2 (Lozhkov et al., 2024) is a family of open code-generation
//! models (3B / 7B / 15B parameters).
//!
//! ## Architectural highlights
//!
//! - **Multi-Query Attention** (near-MQA): `num_key_value_heads = 2` for all sizes.
//! - **SwiGLU FFN** with biases on all linear layers.
//! - **RoPE** positional embeddings (θ = 10 000).
//! - **Fill-In-the-Middle (FIM)** for code in-filling tasks.
//! - Optional sliding-window attention (unused in released checkpoints).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::starcoder2::{StarCoder2Config, StarCoder2ForCausalLM};
//!
//! let config = StarCoder2Config::small_test();
//! let model = StarCoder2ForCausalLM::new(config)?;
//! let logits = model.forward(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod fim;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::StarCoder2Config;
pub use fim::{format_fim_prompt, parse_fim_output, FimTokens};
pub use model::{
    StarCoder2Attention, StarCoder2DecoderLayer, StarCoder2ForCausalLM, StarCoder2MLP,
    StarCoder2Model, StarCoder2RmsNorm, StarCoder2RotaryEmbedding,
};
pub use tasks::{
    format_psm_prompt, format_spm_prompt, parse_fim_middle, StarCoder2ForSequenceClassification,
    StarCoder2TaskError,
};
