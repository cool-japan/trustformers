//! # Claude (Anthropic's Constitutional AI)
//!
//! Claude is a family of large language models developed by Anthropic, designed with
//! Constitutional AI principles to be helpful, harmless, and honest.
//!
//! ## Constitutional AI Features
//!
//! Claude incorporates several key Constitutional AI innovations:
//! - **Harmlessness**: Built-in safety measures to prevent harmful outputs
//! - **Helpfulness**: Optimized to provide useful and accurate responses
//! - **Honesty**: Designed to acknowledge uncertainty and avoid hallucination
//! - **Self-supervision**: Constitutional AI training using AI feedback
//!
//! ## Architecture Innovations
//!
//! Claude uses an enhanced transformer architecture with:
//! - **Enhanced Attention**: Improved multi-head attention mechanisms
//! - **SwiGLU Activation**: Gated linear units in feed-forward networks
//! - **Layer Normalization**: Pre-norm architecture for stability
//! - **Rotary Position Embeddings**: Better position encoding
//! - **Grouped Query Attention**: Efficiency improvements in larger models
//!
//! ## Model Variants
//!
//! Available Claude model configurations:
//! - **Claude-1**: Original constitutional AI model
//! - **Claude-2**: Enhanced capabilities with longer context
//! - **Claude-2.1**: Improved version with 200K context length
//! - **Claude-3 Haiku**: Fastest, most cost-effective model
//! - **Claude-3 Sonnet**: Balanced performance and speed
//! - **Claude-3 Opus**: Most powerful model for complex tasks
//! - **Claude-3.5 Sonnet**: Enhanced Sonnet with improved capabilities
//!
//! ## Usage Examples
//!
//! ### Basic Text Generation
//! ```rust,no_run
//! use trustformers_models::claude::{ClaudeForCausalLM, ClaudeConfig};
//!
//! let config = ClaudeConfig::claude_3_sonnet();
//! let mut model = ClaudeForCausalLM::new(config)?;
//!
//! // Enable Constitutional AI features
//! let mut config = ClaudeConfig::claude_3_sonnet();
//! config.with_constitutional_ai(true)
//!       .with_constitutional_weights(1.2, 1.0, 1.1); // Emphasize harmlessness
//!
//! let model = ClaudeForCausalLM::new(config)?;
//! ```
//!
//! ### Constitutional AI Generation
//! ```rust,no_run
//! use trustformers_models::claude::{ClaudeForCausalLM, ClaudeConfig};
//!
//! let config = ClaudeConfig::claude_3_opus()
//!     .with_constitutional_ai(true)
//!     .with_constitutional_weights(1.5, 1.0, 1.2); // Safety-first configuration
//!
//! let model = ClaudeForCausalLM::new(config)?;
//!
//! // Generate with constitutional constraints
//! let output = model.generate_with_constitutional_ai(
//!     input_ids,
//!     max_new_tokens: 200,
//!     temperature: 0.7,
//!     top_p: 0.9,
//! )?;
//! ```
//!
//! ### Safety-Critical Applications
//! ```rust,no_run
//! use trustformers_models::claude::{ClaudeForCausalLM, ClaudeConfig};
//!
//! // Configuration for maximum safety
//! let mut config = ClaudeConfig::claude_3_sonnet();
//! config.with_constitutional_ai(true)
//!       .with_constitutional_weights(2.0, 0.8, 1.5); // High harmlessness weight
//!
//! let model = ClaudeForCausalLM::new(config)?;
//!
//! // Use for safety-critical applications like medical or legal advice
//! ```
//!
//! ## Constitutional AI Principles
//!
//! ### Harmlessness
//! - Refuses to generate harmful, illegal, or unethical content
//! - Implements safety filters and content policies
//! - Uses constitutional training to internalize safety principles
//!
//! ### Helpfulness
//! - Provides accurate and useful information
//! - Follows instructions while maintaining safety constraints
//! - Optimizes for user satisfaction within ethical bounds
//!
//! ### Honesty
//! - Acknowledges uncertainty and limitations
//! - Avoids hallucination and false information
//! - Provides sources and reasoning when possible
//!
//! ## Training Methodology
//!
//! Claude uses Constitutional AI training with:
//! - **Constitutional Training**: Training on a set of principles
//! - **AI Feedback**: Using AI systems to provide training feedback
//! - **RLHF**: Reinforcement Learning from Human Feedback
//! - **Self-Supervision**: Constitutional self-improvement
//!
//! ## Performance Optimization
//!
//! - **Efficient Attention**: Grouped-query attention for large models
//! - **Memory Optimization**: Optimized for long-context processing
//! - **Constitutional Caching**: Cache constitutional AI computations
//! - **Safety-Aware Inference**: Optimized safety checking
//!
//! ## Safety Considerations
//!
//! When deploying Claude models:
//! - Monitor constitutional AI weights and their effects
//! - Implement additional safety layers for high-risk applications
//! - Regularly evaluate harmlessness, helpfulness, and honesty
//! - Consider constitutional AI principles in fine-tuning
//!
//! ## Research Applications
//!
//! Claude is suitable for research in:
//! - Constitutional AI and AI safety
//! - Human-AI alignment
//! - Trustworthy AI systems
//! - AI ethics and governance

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{ClaudeConfig, RopeScaling};
pub use model::{
    ClaudeAttention, ClaudeDecoderLayer, ClaudeForCausalLM, ClaudeMLP, ClaudeModel, RotaryEmbedding,
};
