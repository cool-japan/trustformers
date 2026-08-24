//! Config validation and pre-configured `ContextSummarizer` constructors.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use crate::pipeline::conversational::types::{SummarizationConfig, SummarizationStrategy};

use super::engine::ContextSummarizer;

// ================================================================================================
// ADDITIONAL HELPER FUNCTIONS
// ================================================================================================

/// Validate summarization configuration
pub fn validate_summarization_config(config: &SummarizationConfig) -> Result<()> {
    if config.target_length == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "Target length must be greater than 0".to_string(),
        ));
    }
    if config.trigger_threshold <= config.target_length {
        return Err(TrustformersError::invalid_input_simple(
            "Trigger threshold must be greater than target length".to_string(),
        ));
    }
    Ok(())
}

/// Create a default context summarizer
pub fn create_default_summarizer() -> ContextSummarizer {
    ContextSummarizer::new(SummarizationConfig::default())
}

/// Create a high-compression summarizer for memory-constrained environments
pub fn create_high_compression_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.target_length = 100;
    config.trigger_threshold = 500;
    config.strategy = SummarizationStrategy::Hybrid;
    ContextSummarizer::new(config)
}

/// Create a topic-focused extractive summarizer
pub fn create_extractive_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.strategy = SummarizationStrategy::Extractive;
    config.target_length = 300;
    ContextSummarizer::new(config)
}

/// Create an abstractive summarizer for detailed overviews
pub fn create_abstractive_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.strategy = SummarizationStrategy::Abstractive;
    config.target_length = 250;
    ContextSummarizer::new(config)
}
