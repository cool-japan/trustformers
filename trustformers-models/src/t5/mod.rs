//! # T5 (Text-to-Text Transfer Transformer)
//!
//! T5 is a unified text-to-text transformer that treats every NLP task as a text generation
//! problem. It uses an encoder-decoder architecture with relative position embeddings.
//!
//! ## Architecture
//!
//! T5 features:
//! - Encoder-decoder transformer architecture
//! - Relative position bias instead of absolute position embeddings
//! - Pre-layer normalization
//! - No bias in linear layers (except layer norm)
//! - ReLU activation (unlike BERT/GPT's GELU)
//! - Dropout applied to feedforward layers, attention weights, and input embeddings
//!
//! ## Model Variants
//!
//! Available sizes:
//! - **T5-Small**: 60M parameters (6 layers, 512 hidden, 8 heads)
//! - **T5-Base**: 220M parameters (12 layers, 768 hidden, 12 heads)
//! - **T5-Large**: 770M parameters (24 layers, 1024 hidden, 16 heads)
//! - **T5-3B**: 3B parameters (24 layers, 1024 hidden, 32 heads)
//! - **T5-11B**: 11B parameters (24 layers, 1024 hidden, 128 heads)
//!
//! ## Text-to-Text Framework
//!
//! All tasks are framed as text-to-text:
//! - **Translation**: "translate English to French: Hello" → "Bonjour"
//! - **Summarization**: "summarize: [article]" → "[summary]"
//! - **Classification**: "sentiment: Great movie!" → "positive"
//! - **Question Answering**: "question: What is T5? context: [text]" → "[answer]"
//!
//! ## Usage Examples
//!
//! ### Translation
//! ```rust,no_run
//! use trustformers_models::t5::{T5ForConditionalGeneration, T5Config};
//!
//! let config = T5Config::t5_base();
//! let mut model = T5ForConditionalGeneration::new(config)?;
//! model.load_from_hub("t5-base")?;
//!
//! // Translate English to German
//! let input_text = "translate English to German: The house is wonderful.";
//! let input_ids = tokenizer.encode(input_text)?;
//!
//! let outputs = model.generate(input_ids, max_length: 50)?;
//! let translation = tokenizer.decode(outputs)?;
//! ```
//!
//! ### Summarization
//! ```rust,no_run
//! use trustformers_models::t5::{T5ForConditionalGeneration, T5Config};
//!
//! let config = T5Config::t5_small();
//! let mut model = T5ForConditionalGeneration::new(config)?;
//! model.load_from_hub("t5-small")?;
//!
//! // Summarize text
//! let article = "summarize: The tower is 324 metres (1,063 ft) tall, ...";
//! let input_ids = tokenizer.encode(article)?;
//!
//! let summary_ids = model.generate(input_ids, max_length: 150)?;
//! let summary = tokenizer.decode(summary_ids)?;
//! ```
//!
//! ### Custom Tasks
//! ```rust,no_run
//! use trustformers_models::t5::{T5ForConditionalGeneration, T5Config, T5Input};
//!
//! let config = T5Config::t5_base();
//! let mut model = T5ForConditionalGeneration::new(config)?;
//!
//! // Fine-tune for custom text-to-text task
//! let input = T5Input {
//!     input_ids,
//!     attention_mask: Some(attention_mask),
//!     decoder_input_ids: Some(decoder_input_ids),
//!     decoder_attention_mask: Some(decoder_attention_mask),
//!     labels: Some(labels),
//! };
//!
//! let outputs = model.forward(input)?;
//! let loss = outputs.loss.expect("operation failed");
//! ```
//!
//! ## Pre-training Objective
//!
//! T5 uses a denoising objective:
//! 1. Corrupt input by masking spans of tokens
//! 2. Replace spans with sentinel tokens
//! 3. Train model to reconstruct original text
//!
//! ## Task Prefixes
//!
//! Common task prefixes:
//! - `"translate [source] to [target]: "`
//! - `"summarize: "`
//! - `"question: "` (with `"context: "`)
//! - `"cola sentence: "` (grammatical acceptability)
//! - `"stsb sentence1: "` (semantic similarity)
//!
//! ## Performance Tips
//!
//! - Use task-specific prefixes for best results
//! - T5-Base offers good balance for most tasks
//! - Enable gradient checkpointing for large models
//! - Use beam search for translation/summarization
//! - Apply length penalties for generation control
//!
//! ## Advanced Features
//!
//! - **Relative Position Bias**: More efficient than absolute embeddings
//! - **Shared Embeddings**: Input/output embeddings are tied
//! - **No Bias Terms**: Simplified architecture
//! - **Mixture of Experts**: Available in Switch Transformers variant

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::T5Config;
pub use model::{T5ForConditionalGeneration, T5Input, T5LMOutput, T5Model, T5Output};
