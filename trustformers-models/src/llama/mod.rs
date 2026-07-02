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
//! use trustformers_core::traits::Model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LlamaConfig::llama_7b();
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_from_path("path/to/llama-7b-weights")?;
//!
//! // Run a forward pass to obtain next-token logits
//! # let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
//! let output = model.forward(input_ids)?;
//! # let _ = output;
//! # Ok(())
//! # }
//! ```
//!
//! ### Instruction Following
//! ```rust,no_run
//! use trustformers_models::llama::{LlamaForCausalLM, LlamaConfig};
//! use trustformers_core::traits::Model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LlamaConfig::llama2_7b();
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_from_path("path/to/llama-2-7b-chat-weights")?;
//!
//! // Format instruction with chat template, then tokenize (tokenization not shown here)
//! let instruction = "[INST] Explain quantum computing in simple terms. [/INST]";
//! # let _ = instruction;
//! # let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
//!
//! let response = model.forward(input_ids)?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! ### Efficient Inference
//! ```rust,no_run
//! use trustformers_models::llama::{LlamaForCausalLM, LlamaConfig};
//! use trustformers_models::llama::config::RopeScaling;
//! use trustformers_core::traits::Model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LlamaConfig {
//!     // Extended context via linear RoPE scaling
//!     rope_scaling: Some(RopeScaling { scaling_type: "linear".to_string(), scaling_factor: 2.0 }),
//!     ..LlamaConfig::llama_7b()
//! };
//!
//! let mut model = LlamaForCausalLM::new(config)?;
//! model.load_from_path("llama-7b-weights")?;  // Load pretrained weights
//!
//! // Run inference
//! # let input_ids: Vec<u32> = vec![1, 2, 3];
//! let output = model.forward(input_ids)?;
//! # let _ = output;
//! # Ok(())
//! # }
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
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::LlamaConfig;
pub use model::{
    LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP, LlamaModel, RMSNorm,
    RotaryEmbedding,
};
