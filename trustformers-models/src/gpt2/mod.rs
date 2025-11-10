//! # GPT-2 (Generative Pre-trained Transformer 2)
//!
//! GPT-2 is an autoregressive language model that uses a transformer decoder architecture.
//! It's designed for text generation and can be fine-tuned for various generation tasks.
//!
//! ## Architecture
//!
//! GPT-2 features:
//! - Transformer decoder blocks with causal (left-to-right) attention
//! - Byte Pair Encoding (BPE) tokenization
//! - Learned positional embeddings
//! - Layer normalization before each sub-block
//! - GELU activation function
//!
//! ## Model Variants
//!
//! Available configurations:
//! - **GPT-2 Small**: 124M parameters (12 layers, 768 hidden, 12 heads)
//! - **GPT-2 Medium**: 355M parameters (24 layers, 1024 hidden, 16 heads)
//! - **GPT-2 Large**: 774M parameters (36 layers, 1280 hidden, 20 heads)
//! - **GPT-2 XL**: 1.5B parameters (48 layers, 1600 hidden, 25 heads)
//!
//! ## Usage Examples
//!
//! ### Text Generation
//! ```rust,no_run
//! use trustformers_models::gpt2::{Gpt2LMHeadModel, Gpt2Config};
//! use trustformers_core::generation::{GenerationConfig, SamplingStrategy};
//!
//! let config = Gpt2Config::gpt2_medium();
//! let mut model = Gpt2LMHeadModel::new(config)?;
//! model.load_from_hub("gpt2-medium")?;
//!
//! // Generate text
//! let gen_config = GenerationConfig {
//!     max_length: 100,
//!     temperature: 0.8,
//!     top_p: 0.9,
//!     sampling_strategy: SamplingStrategy::TopPNucleus,
//!     ..Default::default()
//! };
//!
//! let generated_ids = model.generate(input_ids, gen_config)?;
//! ```
//!
//! ### Feature Extraction
//! ```rust,no_run
//! use trustformers_models::gpt2::{Gpt2Model, Gpt2Config};
//!
//! let config = Gpt2Config::gpt2_base();
//! let mut model = Gpt2Model::new(config)?;
//! model.load_from_hub("gpt2")?;
//!
//! // Extract hidden states
//! let outputs = model.forward(input_ids, None, None)?;
//! let hidden_states = outputs.last_hidden_state;
//! ```
//!
//! ### Text Completion
//! ```rust,no_run
//! use trustformers_models::gpt2::{Gpt2LMHeadModel, Gpt2Config};
//!
//! let config = Gpt2Config::gpt2_base();
//! let mut model = Gpt2LMHeadModel::new(config)?;
//! model.load_from_hub("gpt2")?;
//!
//! // Complete text with greedy decoding
//! let prompt = "The future of AI is";
//! let input_ids = tokenizer.encode(prompt)?;
//! let completed = model.generate_greedy(input_ids, max_length: 50)?;
//! ```
//!
//! ## Generation Strategies
//!
//! Supported decoding methods:
//! - **Greedy**: Select highest probability token at each step
//! - **Beam Search**: Explore multiple hypotheses
//! - **Top-K Sampling**: Sample from top K tokens
//! - **Top-P (Nucleus) Sampling**: Sample from cumulative probability mass
//! - **Temperature Scaling**: Control randomness
//!
//! ## Fine-tuning Applications
//!
//! GPT-2 can be fine-tuned for:
//! - Conversational AI
//! - Story generation
//! - Code completion
//! - Poetry and creative writing
//! - Domain-specific text generation
//!
//! ## Performance Optimization
//!
//! - Use KV-cache for faster generation
//! - Enable FlashAttention for memory efficiency
//! - Apply int8 quantization for deployment
//! - Implement batch generation for throughput
//!
//! ## Ethical Considerations
//!
//! When using GPT-2:
//! - Be aware of potential biases in generated text
//! - Implement content filtering for production use
//! - Consider the environmental impact of large models
//! - Respect OpenAI's responsible use guidelines

pub mod config;
pub mod model;
pub mod generation;

#[cfg(test)]
mod tests;

pub use config::Gpt2Config;
pub use generation::GenerativeModel;
pub use model::{Gpt2LMHeadModel, Gpt2Model};
