//! OPT (Open Pre-trained Transformer) model family.
//!
//! Meta's OPT models are decoder-only transformers with:
//! - Learned positional embeddings (+ offset 2 convention).
//! - Pre-norm LayerNorm (in most variants).
//! - ReLU FFN without gating.
//! - Biases in all projections.
//!
//! # References
//!
//! Zhang et al. (2022) — "OPT: Open Pre-trained Transformer Language Models"
//! <https://arxiv.org/abs/2205.01068>

pub mod config;
pub mod model;
pub mod tasks;

pub use config::OptConfig;
pub use model::{
    OptAttention, OptDecoder, OptDecoderLayer, OptError, OptFeedForward, OptLayerNorm,
    OptLearnedPositionalEmbedding, OptLinear, OptModel,
};
pub use tasks::{format_completion_prompt, OptCausalLMOutput, OptForCausalLM};
