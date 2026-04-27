//! Task-specific wrappers for SD3 text encoders.

use crate::sd3::{
    config::{Sd3Config, Sd3ConfigError},
    model::{Sd3Error, Sd3TextEmbeddings, Sd3TextEncoderPipeline},
};

/// Top-level error type for SD3 text encoder operations.
#[derive(Debug, thiserror::Error)]
pub enum Sd3TaskError {
    #[error("Configuration error: {0}")]
    Config(#[from] Sd3ConfigError),
    #[error("Encoder error: {0}")]
    Encoder(#[from] Sd3Error),
    #[error("Empty token sequence provided")]
    EmptyTokens,
    #[error("Invalid output embeddings: expected pooled_dim={expected}, got {got}")]
    InvalidEmbeddingDim { expected: usize, got: usize },
}

/// The SD3 text encoder task wrapper.
///
/// Encapsulates the three-encoder pipeline and exposes a simple text encoding API
/// suitable for integration with the SD3 diffusion model (MMDiT).
pub struct Sd3TextEncoder {
    pipeline: Sd3TextEncoderPipeline,
}

impl Sd3TextEncoder {
    /// Create a new SD3TextEncoder from configuration.
    pub fn new(config: Sd3Config) -> Result<Self, Sd3TaskError> {
        config.validate()?;
        let pipeline = Sd3TextEncoderPipeline::new(config)?;
        Ok(Self { pipeline })
    }

    /// Encode token IDs and return combined SD3 text embeddings.
    ///
    /// This is the primary entry point for SD3 conditioning:
    ///   - Runs CLIP-L, CLIP-G, and T5-XXL encoders
    ///   - Returns pooled embeddings (global) + T5 embeddings (per-token)
    ///
    /// # Arguments
    /// * `token_ids` — token ID sequence (shared for all encoders in this simplified impl)
    ///
    /// # Returns
    /// [`Sd3TextEmbeddings`] with `.pooled_embeddings` and `.t5_embeddings`
    pub fn encode(&self, token_ids: &[u32]) -> Result<Sd3TextEmbeddings, Sd3TaskError> {
        if token_ids.is_empty() {
            return Err(Sd3TaskError::EmptyTokens);
        }
        let seq_len = token_ids.len();
        let embeddings = self.pipeline.encode_text(token_ids, seq_len)?;

        // Validate pooled embedding dimension
        let expected_pooled = self.pipeline.config().pooled_embedding_dim;
        if embeddings.pooled_embeddings.len() != expected_pooled {
            return Err(Sd3TaskError::InvalidEmbeddingDim {
                expected: expected_pooled,
                got: embeddings.pooled_embeddings.len(),
            });
        }

        Ok(embeddings)
    }

    /// Reference to the underlying pipeline.
    pub fn pipeline(&self) -> &Sd3TextEncoderPipeline {
        &self.pipeline
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &Sd3Config {
        self.pipeline.config()
    }
}
