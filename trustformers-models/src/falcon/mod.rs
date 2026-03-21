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
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//! use trustformers_core::generation::{GenerationConfig, SamplingStrategy};
//!
//! let config = FalconConfig::falcon_7b();
//! let mut model = FalconForCausalLM::new(config)?;
//! model.load_from_hub("tiiuae/falcon-7b")?;
//!
//! let gen_config = GenerationConfig {
//!     max_new_tokens: 150,
//!     temperature: 0.8,
//!     top_p: 0.95,
//!     repetition_penalty: 1.1,
//!     sampling_strategy: SamplingStrategy::TopPNucleus,
//!     ..Default::default()
//! };
//!
//! let generated = model.generate(input_ids, gen_config)?;
//! ```
//!
//! ### Instruction Following
//! ```rust,no_run
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//!
//! let config = FalconConfig::falcon_7b_instruct();
//! let mut model = FalconForCausalLM::new(config)?;
//! model.load_from_hub("tiiuae/falcon-7b-instruct")?;
//!
//! // Use with instruction prompt
//! let instruction = "User: What are the benefits of renewable energy?\nFalcon:";
//! let input_ids = tokenizer.encode(instruction)?;
//!
//! let response = model.generate(input_ids, max_length: 500)?;
//! ```
//!
//! ### Large Model Inference
//! ```rust,no_run
//! use trustformers_models::falcon::{FalconForCausalLM, FalconConfig};
//!
//! let config = FalconConfig {
//!     use_flash_attention: true,    // Enable FlashAttention
//!     gradient_checkpointing: true, // Save memory
//!     ..FalconConfig::falcon_40b()
//! };
//!
//! let mut model = FalconForCausalLM::new(config)?;
//! model.load_sharded("tiiuae/falcon-40b")?;  // Load in shards
//!
//! // Use tensor parallelism for large models
//! model.enable_tensor_parallel(4)?;
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

pub use config::FalconConfig;
pub use model::{FalconAttention, FalconDecoderLayer, FalconForCausalLM, FalconMLP, FalconModel};
