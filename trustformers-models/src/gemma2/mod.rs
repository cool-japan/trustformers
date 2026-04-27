//! # Gemma-2 (Google's Second-Generation Open LLM)
//!
//! Gemma-2 is a family of open language models from Google DeepMind that introduces
//! several architectural improvements over the original Gemma:
//!
//! ## Key Architectural Innovations
//!
//! - **Alternating attention pattern**: Even layers use *local* (sliding window) attention;
//!   odd layers use *global* (full causal) attention. This balances efficiency and long-range
//!   capability.
//! - **Logit soft-capping**: Both attention scores and final LM logits are passed through
//!   `tanh(x / cap) * cap`, preventing extremely large values from destabilising training.
//! - **Grouped Query Attention (GQA)**: Fewer KV heads reduce memory bandwidth during inference.
//! - **Post-normalization**: RMSNorm is applied both *before* and *after* each residual add
//!   (pre-norm + post-norm), improving training stability.
//! - **GEGLU activation**: `gelu(gate) * up` in the MLP block.
//! - **Fixed 256-dim head size**: All Gemma-2 variants use `head_dim = 256` regardless of model
//!   size.
//!
//! ## Model Variants
//!
//! | Variant  | Params | Layers | Hidden | Heads (Q/KV) |
//! |----------|--------|--------|--------|--------------|
//! | 2B       | 2.6 B  | 26     | 2304   | 8 / 4        |
//! | 9B       | 9 B    | 42     | 3584   | 16 / 8       |
//! | 27B      | 27 B   | 46     | 4608   | 32 / 16      |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::gemma2::{Gemma2Config, Gemma2ForCausalLM};
//!
//! let config = Gemma2Config::gemma2_9b();
//! // For tests, use a tiny config instead:
//! // let config = Gemma2Config { hidden_size: 16, ..Default::default() };
//! let model = Gemma2ForCausalLM::new(config).expect("model creation");
//!
//! // Format a chat prompt
//! let prompt = trustformers_models::gemma2::format_chat_prompt("Hello!");
//! println!("{}", prompt);
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::Gemma2Config;
pub use model::{
    apply_soft_cap_inplace, geglu, gelu, soft_cap, Gemma2Attention, Gemma2DecoderLayer,
    Gemma2GegluMlp, Gemma2Model, Gemma2RmsNorm, Gemma2RotaryEmbedding,
};
pub use tasks::{format_chat_prompt, Gemma2Error, Gemma2ForCausalLM};
