//! # LLaMA (Large Language Model Meta AI)
//!
//! LLaMA is a family of foundation language models designed for efficiency and performance.
//! It incorporates several architectural improvements over standard transformers.
//!
//! ## Architecture Innovations
//!
//! LLaMA introduces several key improvements:
//! - **RMSNorm**: Root Mean Square Layer Normalization for efficiency
//! - **SwiGLU activation**: Replaces ReLU in feed-forward network
//! - **Rotary Position Embeddings (RoPE)**: Better position encoding
//! - **No bias terms**: Simplified architecture
//! - **Grouped Query Attention (GQA)**: In larger models for efficiency
//!
//! ## Model Variants
//!
//! Available configurations:
//! - **LLaMA-7B**: 7B parameters (32 layers, 4096 hidden, 32 heads)
//! - **LLaMA-13B**: 13B parameters (40 layers, 5120 hidden, 40 heads)
//! - **LLaMA-30B**: 30B parameters (60 layers, 6656 hidden, 52 heads)
//! - **LLaMA-65B**: 65B parameters (80 layers, 8192 hidden, 64 heads)
//!
//! LLaMA 2 variants:
//! - **LLaMA 2-7B**: Enhanced 7B model with longer context
//! - **LLaMA 2-13B**: Improved 13B model
//! - **LLaMA 2-70B**: New larger variant with GQA
//!
//! ## Usage Examples
//!
//! ### Text Generation
//! ```rust,no_run
//! use trustformers_models::llama::{LlamaForCausalLM, LlamaConfig};
//! use trustformers_core::generation::{GenerationConfig, SamplingStrategy};
//!
//! let config = LlamaConfig::llama_7b();
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_from_hub("meta-llama/Llama-2-7b-hf")?;
//!
//! // Generate with advanced sampling
//! let gen_config = GenerationConfig {
//!     max_new_tokens: 200,
//!     temperature: 0.7,
//!     top_p: 0.9,
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
//! use trustformers_models::llama::{LlamaForCausalLM, LlamaConfig};
//!
//! let config = LlamaConfig::llama2_7b_chat();
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_from_hub("meta-llama/Llama-2-7b-chat-hf")?;
//!
//! // Format instruction with chat template
//! let instruction = "[INST] Explain quantum computing in simple terms. [/INST]";
//! let input_ids = tokenizer.encode(instruction)?;
//!
//! let response = model.generate(input_ids, max_length: 500)?;
//! ```
//!
//! ### Efficient Inference
//! ```rust,no_run
//! use trustformers_models::llama::{LlamaForCausalLM, LlamaConfig};
//!
//! let config = LlamaConfig {
//!     use_flash_attention: true,  // Enable FlashAttention
//!     rope_scaling: Some(2.0),    // Extended context
//!     ..LlamaConfig::llama_7b()
//! };
//!
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_quantized("llama-7b-4bit.gguf")?;  // Load quantized
//!
//! // Stream tokens for responsive UI
//! for token in model.generate_stream(input_ids)? {
//!     print!("{}", tokenizer.decode(&[token])?);
//! }
//! ```
//!
//! ## Key Components
//!
//! ### RMSNorm
//! More efficient normalization:
//! ```text
//! RMSNorm(x) = x * g / sqrt(mean(x²) + ε)
//! ```
//!
//! ### Rotary Position Embeddings
//! Encodes position information directly in attention:
//! - Relative position aware
//! - Extrapolates to longer sequences
//! - No learned embeddings needed
//!
//! ### SwiGLU Activation
//! Gated linear unit in FFN:
//! ```text
//! SwiGLU(x) = (xW₁ * σ(xW₃)) W₂
//! ```
//!
//! ## Training Details
//!
//! - Trained on 1-2 trillion tokens
//! - Uses AdamW optimizer with cosine schedule
//! - Gradient checkpointing for memory efficiency
//! - Mixed precision training (BF16)
//!
//! ## Performance Optimization
//!
//! - **KV-Cache**: Cache key-value pairs for generation
//! - **Flash Attention**: Fused attention kernels
//! - **Quantization**: 4-bit and 8-bit inference
//! - **Tensor Parallelism**: Split model across GPUs
//! - **Continuous Batching**: Dynamic batching for throughput
//!
//! ## Fine-tuning Tips
//!
//! - Use LoRA for parameter-efficient tuning
//! - Apply gradient checkpointing for memory
//! - Start with low learning rates (1e-5)
//! - Use instruction templates for chat models
//!
//! ## Safety Considerations
//!
//! When deploying LLaMA:
//! - Implement safety filters
//! - Monitor for harmful outputs
//! - Respect Meta's usage guidelines
//! - Consider compute requirements

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::LlamaConfig;
pub use model::{
    LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP, LlamaModel, RMSNorm,
    RotaryEmbedding,
};
