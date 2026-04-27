//! # Yi-1.5
//!
//! 01.AI Yi-1.5 (Young et al., 2024) is a family of bilingual language models
//! (6B / 9B / 34B) based on the LLaMA-2 decoder architecture with:
//!
//! - Extended vocabulary: 64 000 tokens.
//! - Grouped Query Attention: 4 KV heads (6B/9B) or 8 KV heads (34B).
//! - Tied embeddings: `lm_head.weight == embed_tokens.weight`.
//! - High RoPE base θ = 5 000 000 for long-context support.
//! - SwiGLU FFN, no bias.
//! - ChatML chat format (same as Qwen-2).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::yi::{YiConfig, YiForCausalLM};
//!
//! let config = YiConfig::small_test();
//! let model = YiForCausalLM::new(config)?;
//! let logits = model.forward(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod chat;
pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use chat::format_yi_chat;
pub use config::YiConfig;
pub use model::{
    YiAttention, YiDecoderLayer, YiForCausalLM, YiMLP, YiModel, YiRmsNorm, YiRotaryEmbedding,
};
pub use tasks::{
    format_chatml_prompt, format_simple_prompt, ChatMessage, YiForSequenceClassification,
    YiForTokenClassification, YiTaskError,
};
