//! # Phi-3 (Microsoft's Small Language Model)
//!
//! Phi-3 is a family of small language models developed by Microsoft that achieve
//! impressive performance while being compact enough to run on mobile devices.
//!
//! ## Architecture Innovations
//!
//! Phi-3 incorporates several key improvements:
//! - **RMSNorm**: Root Mean Square Layer Normalization for efficiency
//! - **SwiGLU activation**: Gated linear units in feed-forward network
//! - **Rotary Position Embeddings (RoPE)**: Advanced position encoding
//! - **LongRope scaling**: Extended context support up to 128K tokens
//! - **Grouped Query Attention (GQA)**: In larger models for efficiency
//! - **Sliding Window Attention**: Optional local attention patterns
//!
//! ## Model Variants
//!
//! Available configurations:
//! - **Phi-3 Mini (3.8B)**: Compact model for mobile and edge devices
//! - **Phi-3 Small (7B)**: Balanced performance and efficiency
//! - **Phi-3 Medium (14B)**: Highest capability model in the family
//!
//! Each variant comes in multiple context lengths:
//! - **4K**: Standard context length for most applications
//! - **8K**: Extended context for small model
//! - **128K**: Very long context with LongRope scaling
//!
//! ## Usage Examples
//!
//! ### Text Generation with Phi-3 Mini
//! ```rust,no_run
//! use trustformers_models::phi3::{Phi3ForCausalLM, Phi3Config};
//! use trustformers_core::traits::Model;
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Phi3Config::phi3_mini_4k_instruct();
//! let model = Phi3ForCausalLM::new(config)?;
//!
//! // Run a forward pass to obtain next-token logits
//! # let input_ids = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5], &[5])?;
//! let output = model.forward(input_ids)?;
//! # let _ = output;
//! # Ok(())
//! # }
//! ```
//!
//! ### Instruction Following
//! ```rust,no_run
//! use trustformers_models::phi3::{Phi3ForCausalLM, Phi3Config};
//! use trustformers_core::traits::Model;
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Phi3Config::phi3_small_8k_instruct();
//! let model = Phi3ForCausalLM::new(config)?;
//!
//! // Format instruction with Phi-3 chat template, then tokenize (tokenization not shown here)
//! let instruction = "<|user|>\nExplain machine learning in simple terms.<|end|>\n<|assistant|>\n";
//! # let _ = instruction;
//! # let input_ids = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5], &[5])?;
//!
//! let response = model.forward(input_ids)?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! ### Long Context Processing
//! ```rust,no_run
//! use trustformers_models::phi3::{Phi3ForCausalLM, Phi3Config};
//! use trustformers_core::traits::Model;
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Use 128K context model for long documents
//! let config = Phi3Config::phi3_mini_128k_instruct();
//! let model = Phi3ForCausalLM::new(config)?;
//!
//! // Process long document (token ids shown here would normally come from a tokenizer)
//! # let long_input = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5], &[5])?;
//! let summary = model.forward(long_input)?;
//! # let _ = summary;
//! # Ok(())
//! # }
//! ```
//!
//! ### Efficient Mobile Deployment
//! ```rust,no_run
//! use trustformers_models::phi3::{Phi3ForCausalLM, Phi3Config};
//! use trustformers_core::traits::Model;
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Phi3Config {
//!     attention_dropout: 0.0,      // Disable dropout for inference
//!     ..Phi3Config::phi3_mini_4k_instruct()
//! };
//!
//! let model = Phi3ForCausalLM::new(config)?;
//!
//! // Optimized inference for mobile
//! # let input_ids = Tensor::from_vec_i64(vec![1, 2, 3], &[3])?;
//! let result = model.forward(input_ids)?;
//! # let _ = result;
//! # Ok(())
//! # }
//! ```
//!
//! ## Key Components
//!
//! ### RMSNorm
//! More efficient normalization than LayerNorm:
//! ```text
//! RMSNorm(x) = x * g / sqrt(mean(x²) + ε)
//! ```
//!
//! ### LongRope Scaling
//! Enables extended context through position embedding scaling:
//! - Handles context lengths up to 128K tokens
//! - Uses short and long scaling factors
//! - Maintains performance on shorter sequences
//!
//! ### SwiGLU Activation
//! Gated linear unit in feed-forward network:
//! ```text
//! SwiGLU(x) = (xW₁ ⊙ σ(xW₃)) W₂
//! ```
//! Where σ is SiLU activation and ⊙ is element-wise multiplication.
//!
//! ### Grouped Query Attention
//! Used in medium model for efficiency:
//! - Reduces memory usage during inference
//! - Maintains quality while improving speed
//! - Balances between MHA and MQA
//!
//! ## Training Details
//!
//! - Trained on high-quality filtered data
//! - Uses curriculum learning approach
//! - Incorporates safety training and alignment
//! - Optimized for instruction following
//!
//! ## Performance Characteristics
//!
//! - **Phi-3 Mini**: Best efficiency, mobile-friendly
//! - **Phi-3 Small**: Balanced performance/size
//! - **Phi-3 Medium**: Highest capability in family
//!
//! All models excel at:
//! - Instruction following
//! - Code generation
//! - Mathematical reasoning
//! - Common sense reasoning
//!
//! ## Mobile and Edge Optimization
//!
//! Phi-3 is specifically designed for deployment on resource-constrained devices:
//! - Efficient architecture reduces memory usage
//! - Supports quantization (4-bit, 8-bit)
//! - Optimized for CPU and mobile GPU inference
//! - Fast initialization and small model files
//!
//! ## Safety and Alignment
//!
//! Phi-3 models include safety features:
//! - Trained with safety datasets
//! - Reduced harmful output generation
//! - Built-in content filtering
//! - Aligned for helpful, harmless responses

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::Phi3Config;
pub use model::{
    Phi3Attention, Phi3DecoderLayer, Phi3ForCausalLM, Phi3MLP, Phi3Model, RMSNorm, RotaryEmbedding,
};
