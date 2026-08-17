// Generation module - refactored from generation.rs for better organization

pub mod beam_search;
pub mod cache;
pub mod cfg;
pub mod config;
pub mod constraints;
pub mod core;
pub mod diverse_sampling;
pub mod grammar;
pub mod json_schema;
pub mod logits_processing;
pub mod regex_constraint;
pub mod streaming;

// Re-export commonly used types
pub use beam_search::{
    beam_search_step, get_forbidden_tokens_for_ngram, BeamError, BeamHypothesis, BeamSearchConfig,
    BeamSearchDecoder, BeamState,
};
pub use cache::{Beam, KVCache};
pub use cfg::CFGGenerator;
pub use config::builder;
pub use config::{
    AssistedGenerationConfig, CFGConfig, GenerationConfig, GenerationStrategy,
    GuidedGenerationConfig, WatermarkingAlgorithm, WatermarkingConfig,
};
pub use constraints::{ConstraintValidator, GrammarValidator, JsonSchemaValidator};
pub use core::TextGenerator;
pub use diverse_sampling::{
    eta_sample, greedy_sample, min_p_sample, mirostat_sample, sample as diverse_sample,
    top_k_sample, top_p_sample, typical_sample, MirostatState, SamplingConfig, SamplingError,
    SamplingMethod,
};
pub use grammar::{Grammar, GrammarError, GrammarSymbol, ParseOutcome};
pub use json_schema::{JsonSchema, JsonSchemaError};
pub use logits_processing::{
    apply_repetition_penalty, apply_temperature, argmax, log_softmax, multinomial_sample,
    sample_index_from_cumulative, softmax, top_k_filter, top_p_filter,
};
pub use regex_constraint::RegexConstraint;
pub use streaming::{FinishReason, GenerationStream, GenerationStreamTrait, GenerationToken};

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod constraints_tests;
#[cfg(test)]
mod core_tests;
#[cfg(test)]
mod streaming_tests;
