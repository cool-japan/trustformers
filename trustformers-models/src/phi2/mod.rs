//! # Phi-2
//!
//! Microsoft Phi-2 is a 2.7B-parameter small language model (Li et al., 2023)
//! notable for its strong performance relative to model size. Key architectural
//! properties:
//!
//! * **Parallel transformer blocks**: attention and MLP operate in parallel on
//!   the *same* layer-normalised input (unlike sequential residual stacking).
//! * **Rotary Position Embeddings (RoPE)** applied to Q and K projections.
//! * **GELU-activated MLP**: `fc1 (hidden→4×hidden) → GELU → fc2 (4×hidden→hidden)`.
//! * Standard Layer Normalisation (mean-variance, not RMS).
//! * No GQA — full Multi-Head Attention.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::phi2::{Phi2Config, Phi2ForCausalLM};
//!
//! let config = Phi2Config::small_test();
//! let model = Phi2ForCausalLM::new(config)?;
//! let logits = model.forward(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::Phi2Config;
pub use model::{
    Phi2Attention, Phi2DecoderLayer, Phi2ForCausalLM, Phi2LayerNorm, Phi2MLP, Phi2Model,
    Phi2RotaryEmbedding,
};
pub use tasks::{Phi2CausalLMOutput, Phi2ForCodeGeneration};
