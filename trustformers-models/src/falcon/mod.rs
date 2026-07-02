//! # Falcon - Technology Innovation Institute Language Models
//!
//! Falcon is a family of high-performance language models developed by TII.
//! These models use advanced architectural improvements for better efficiency.
//!
//! ## Architecture Features
//!
//! Falcon incorporates several key innovations:
//! - **Multi-Query Attention (MQA)**: Shared key-value heads for efficiency
//! - **ALiBi Positional Encoding**: Better extrapolation to longer sequences
//! - **Parallel Attention and MLP**: Faster computation
//! - **RefinedWeb Dataset**: High-quality training data
//! - **New Decoder Architecture**: In Falcon-180B for improved performance
//!
//! ## Model Variants
//!
//! Available configurations:
//! - **Falcon-7B**: 7B parameters with ALiBi, multi-query attention
//! - **Falcon-7B-Instruct**: Instruction-tuned version of 7B model
//! - **Falcon-40B**: 40B parameters with improved architecture
//! - **Falcon-40B-Instruct**: Instruction-tuned 40B model
//! - **Falcon-180B**: 180B parameters with new decoder architecture
//! - **Falcon-180B-Chat**: Chat-optimized version of 180B model
//!
//! ## Usage Examples
//!
//! ### Text Generation
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//! use trustformers_core::generation::{GenerationConfig, GenerationStrategy};
//!
//! let config = FalconConfig::falcon_7b();
//! let mut model = FalconForCausalLM::new(config)?;
//! model.load_from_hub("tiiuae/falcon-7b")?;
//!
//! let gen_config = GenerationConfig {
//!     strategy: GenerationStrategy::TopP { p: 0.95, temperature: 0.8 },
//!     max_new_tokens: Some(150),
//!     repetition_penalty: 1.1,
//!     ..Default::default()
//! };
//!
//! # let input_ids = trustformers_core::tensor::Tensor::randn(&[1, 10])?;
//! let max_new_tokens = gen_config.max_new_tokens.unwrap_or(150);
//! let generated = model.generate(input_ids, max_new_tokens)?;
//! # let _ = generated;
//! # Ok(())
//! # }
//! ```
//!
//! ### Instruction Following
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//!
//! let config = FalconConfig::falcon_7b_instruct();
//! let mut model = FalconForCausalLM::new(config)?;
//! model.load_from_hub("tiiuae/falcon-7b-instruct")?;
//!
//! // Use with instruction prompt
//! let instruction = "User: What are the benefits of renewable energy?\nFalcon:";
//! # let _ = instruction;
//! # let input_ids = trustformers_core::tensor::Tensor::randn(&[1, 10])?;
//!
//! let response = model.generate(input_ids, 500)?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! ### Large Model Inference
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//!
//! let config = FalconConfig {
//!     use_flash_attention: Some(true), // Enable FlashAttention
//!     ..FalconConfig::falcon_40b()
//! };
//!
//! let model = FalconForCausalLM::new(config)?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
//!
//! ## Key Features
//!
//! ### Multi-Query Attention
//! Reduces memory and computation by sharing key-value heads:
//! - Falcon-7B: 1 KV head for 71 query heads
//! - Falcon-40B: 8 KV heads for 128 query heads
//! - Significant speedup during generation
//!
//! ### ALiBi Positional Encoding
//! Attention with Linear Biases:
//! - No learned position embeddings
//! - Better extrapolation to longer sequences
//! - Used in Falcon-7B and Falcon-40B
//!
//! ### Parallel Architecture
//! Attention and MLP computed in parallel:
//! - Faster forward pass
//! - Better GPU utilization
//! - Maintains model quality
//!
//! ## Training Details
//!
//! - Trained on RefinedWeb (filtered CommonCrawl)
//! - Uses AdamW with cosine learning rate schedule
//! - Sequence length: 2048 tokens
//! - High-quality, curated training data
//!
//! ## Performance Tips
//!
//! - Use `use_flash_attention: true` for memory efficiency
//! - Enable gradient checkpointing for training
//! - Consider model sharding for very large models
//! - Use multi-query attention advantage during generation
//!
//! ## License Considerations
//!
//! Falcon models have specific licensing:
//! - Falcon-7B and 40B: Apache 2.0 for commercial use
//! - Falcon-180B: Custom license with some restrictions
//! - Check TII license terms before deployment

pub mod config;
pub mod model;
pub mod tasks;

pub use config::FalconConfig;
pub use model::{FalconAttention, FalconDecoderLayer, FalconForCausalLM, FalconMLP, FalconModel};
