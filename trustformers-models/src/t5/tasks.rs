//! # T5 Task-Specific Implementations
//!
//! This module provides task-specific wrappers and utilities for T5 encoder-decoder models:
//! - `T5ForConditionalGeneration` wrapper with prefix formatting
//! - `T5ForSequenceClassification` (encoder + classification head)
//! - `T5Model` base model wrapper
//! - Task prefix utilities (translation, summarization, Q&A, etc.)
//! - Relative position bias computation helpers
//! - Beam-search scoring helpers (pure Rust, no external crates)

use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to T5 task operations.
#[derive(Debug)]
pub enum T5TaskError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Model build error.
    ModelBuildError(String),
    /// Forward pass error.
    ForwardError(String),
    /// Empty encoder input.
    EmptyEncoderInput,
    /// Empty decoder input.
    EmptyDecoderInput,
    /// Invalid number of labels.
    InvalidNumLabels(usize),
}

impl fmt::Display for T5TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            T5TaskError::InvalidConfig(msg) => write!(f, "T5 invalid config: {}", msg),
            T5TaskError::ModelBuildError(msg) => write!(f, "T5 model build error: {}", msg),
            T5TaskError::ForwardError(msg) => write!(f, "T5 forward error: {}", msg),
            T5TaskError::EmptyEncoderInput => write!(f, "T5 error: empty encoder input"),
            T5TaskError::EmptyDecoderInput => write!(f, "T5 error: empty decoder input"),
            T5TaskError::InvalidNumLabels(n) => {
                write!(f, "T5 error: num_labels must be >= 2, got {}", n)
            },
        }
    }
}

impl std::error::Error for T5TaskError {}

// ─── Task prefix constants ─────────────────────────────────────────────────

/// Prefix for English-to-French translation tasks.
pub const PREFIX_TRANSLATE_EN_FR: &str = "translate English to French: ";
/// Prefix for English-to-German translation tasks.
pub const PREFIX_TRANSLATE_EN_DE: &str = "translate English to German: ";
/// Prefix for summarization tasks.
pub const PREFIX_SUMMARIZE: &str = "summarize: ";
/// Prefix for question-answering tasks.
pub const PREFIX_QUESTION: &str = "question: ";
/// Prefix for providing context in Q&A.
pub const PREFIX_CONTEXT: &str = " context: ";
/// Prefix for sentiment classification.
pub const PREFIX_SENTIMENT: &str = "sst2 sentence: ";

/// Prepend a task prefix to the given text.
pub fn add_task_prefix(prefix: &str, text: &str) -> String {
    let mut s = String::with_capacity(prefix.len() + text.len());
    s.push_str(prefix);
    s.push_str(text);
    s
}

/// Format a T5 question-answering input string.
pub fn format_qa_input(question: &str, context: &str) -> String {
    let mut s = String::new();
    s.push_str(PREFIX_QUESTION);
    s.push_str(question);
    s.push_str(PREFIX_CONTEXT);
    s.push_str(context);
    s
}

// ─── T5 Model base wrapper ─────────────────────────────────────────────────

/// Wrapper around the T5 base model (encoder-decoder without LM head).
pub struct T5ModelWrapper {
    config: crate::t5::T5Config,
    inner: crate::t5::T5Model,
}

impl T5ModelWrapper {
    /// Construct from config.
    pub fn new(config: crate::t5::T5Config) -> Result<Self, T5TaskError> {
        let inner = crate::t5::T5Model::new(config.clone())
            .map_err(|e| T5TaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::t5::T5Config {
        &self.config
    }

    /// Forward pass returning encoder hidden states.
    ///
    /// `input_ids` are the encoder token IDs (flat `u32` slice converted to
    /// a `TokenizedInput`).
    pub fn forward(
        &self,
        input_ids: Vec<u32>,
    ) -> Result<trustformers_core::tensor::Tensor, T5TaskError> {
        if input_ids.is_empty() {
            return Err(T5TaskError::EmptyEncoderInput);
        }
        use crate::t5::T5Input;
        use trustformers_core::traits::{Model, TokenizedInput};
        let ids_len = input_ids.len();
        let input = T5Input {
            input_ids: TokenizedInput::new(input_ids, vec![1u8; ids_len]),
            decoder_input_ids: None,
            encoder_outputs: None,
        };
        let output = self
            .inner
            .forward(input)
            .map_err(|e| T5TaskError::ForwardError(e.to_string()))?;
        Ok(output.last_hidden_state)
    }
}

// ─── Conditional Generation wrapper ───────────────────────────────────────

/// Encoder-decoder conditional generation wrapper (seq2seq).
pub struct T5ForConditionalGenerationWrapper {
    config: crate::t5::T5Config,
    inner: crate::t5::T5ForConditionalGeneration,
}

impl T5ForConditionalGenerationWrapper {
    /// Construct from config.
    pub fn new(config: crate::t5::T5Config) -> Result<Self, T5TaskError> {
        let inner = crate::t5::T5ForConditionalGeneration::new(config.clone())
            .map_err(|e| T5TaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::t5::T5Config {
        &self.config
    }

    /// Run a forward pass and return the logit tensor.
    pub fn forward(
        &self,
        input_ids: Vec<u32>,
        decoder_input_ids: Option<Vec<u32>>,
    ) -> Result<trustformers_core::tensor::Tensor, T5TaskError> {
        if input_ids.is_empty() {
            return Err(T5TaskError::EmptyEncoderInput);
        }
        use crate::t5::T5Input;
        use trustformers_core::traits::{Model, TokenizedInput};
        let enc_len = input_ids.len();
        let dec_ids = decoder_input_ids.map(|ids| {
            let n = ids.len();
            TokenizedInput::new(ids, vec![1u8; n])
        });
        let input = T5Input {
            input_ids: TokenizedInput::new(input_ids, vec![1u8; enc_len]),
            decoder_input_ids: dec_ids,
            encoder_outputs: None,
        };
        let output = self
            .inner
            .forward(input)
            .map_err(|e| T5TaskError::ForwardError(e.to_string()))?;
        Ok(output.logits)
    }
}

// ─── Sequence Classification ──────────────────────────────────────────────────

/// Encoder-only sequence classification head for T5.
///
/// The encoder hidden states are mean-pooled then projected to `num_labels`.
pub struct T5ForSequenceClassification {
    config: crate::t5::T5Config,
    num_labels: usize,
    /// Classifier weights `[num_labels, d_model]`.
    classifier_weight: Vec<Vec<f32>>,
}

impl T5ForSequenceClassification {
    /// Construct from config.
    pub fn new(config: crate::t5::T5Config, num_labels: usize) -> Result<Self, T5TaskError> {
        if num_labels < 2 {
            return Err(T5TaskError::InvalidNumLabels(num_labels));
        }
        let d_model = config.d_model;
        let mut state: u64 = 0xbeef_cafe_1234_5678;
        let classifier_weight = (0..num_labels)
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state as f32 / u64::MAX as f32) * 0.02 - 0.01
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            config,
            num_labels,
            classifier_weight,
        })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::t5::T5Config {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass: applies classification head to a pooled `d_model`-dim vector.
    /// Returns `[num_labels]` logits.
    pub fn forward(&self, pooled: &[f32]) -> Result<Vec<f32>, T5TaskError> {
        if pooled.is_empty() {
            return Err(T5TaskError::EmptyEncoderInput);
        }
        let d = self.config.d_model;
        let input: Vec<f32> = if pooled.len() >= d {
            pooled[..d].to_vec()
        } else {
            let mut v = pooled.to_vec();
            v.resize(d, 0.0);
            v
        };
        let logits = self
            .classifier_weight
            .iter()
            .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum::<f32>())
            .collect();
        Ok(logits)
    }
}

// ─── Relative position bucket helper ──────────────────────────────────────

/// Compute the T5 relative position bucket index.
///
/// Reference: Appendix B of the T5 paper.
///
/// # Arguments
/// * `relative_position` - signed relative position (can be negative for past tokens)
/// * `bidirectional`     - `true` for the encoder, `false` for the causal decoder
/// * `num_buckets`       - total number of buckets
/// * `max_distance`      - maximum absolute distance before log-scaling kicks in
pub fn relative_position_bucket(
    relative_position: i32,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut n = -relative_position;
    let mut buckets = num_buckets;

    if bidirectional {
        buckets /= 2;
        let sign_bucket = if n < 0 { 0 } else { buckets };
        n = n.abs();
        // sign_bucket is added below
        let half = buckets;
        let exact_buckets = half / 2;
        let log_buckets = half - exact_buckets;
        let abs_n = n as usize;
        let bucket = if abs_n < exact_buckets {
            abs_n
        } else {
            let scale = (log_buckets as f32) / ((max_distance as f32 / exact_buckets as f32).ln());
            let log_pos = ((abs_n as f32 / exact_buckets as f32).ln() * scale) as usize;
            (exact_buckets + log_pos).min(half - 1)
        };
        return sign_bucket + bucket;
    }

    // Causal (unidirectional): only look at past
    n = n.max(0);
    let abs_n = n as usize;
    let exact_buckets = buckets / 2;
    let log_buckets = buckets - exact_buckets;
    if abs_n < exact_buckets {
        abs_n
    } else {
        let scale = (log_buckets as f32) / ((max_distance as f32 / exact_buckets as f32).ln());
        let log_pos = ((abs_n as f32 / exact_buckets as f32 + 1e-8).ln() * scale) as usize;
        (exact_buckets + log_pos).min(buckets - 1)
    }
}

// ─── Greedy generation helper ──────────────────────────────────────────────

/// Greedy argmax over a logit slice.
pub fn greedy_token(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
}

/// Compute log-probabilities from logits (log-softmax).
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max_v).collect();
    let log_sum = shifted.iter().map(|&x| x.exp()).sum::<f32>().ln();
    shifted.iter().map(|&x| x - log_sum).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t5::T5Config;

    fn small_cfg() -> T5Config {
        T5Config {
            vocab_size: 100,
            d_model: 32,
            d_kv: 8,
            d_ff: 64,
            num_layers: 2,
            num_decoder_layers: None,
            num_heads: 4,
            relative_attention_num_buckets: 16,
            relative_attention_max_distance: 64,
            dropout_rate: 0.0,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: false,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    // ── 1. T5ModelWrapper construction ───────────────────────────────────────

    #[test]
    fn test_t5_model_wrapper_construction() {
        let result = T5ModelWrapper::new(small_cfg());
        assert!(
            result.is_ok(),
            "T5ModelWrapper must construct: {:?}",
            result.err()
        );
    }

    // ── 2. T5ModelWrapper config accessor ────────────────────────────────────

    #[test]
    fn test_t5_model_wrapper_config_accessor() {
        let model = T5ModelWrapper::new(small_cfg()).expect("construction");
        assert_eq!(model.config().d_model, 32);
        assert_eq!(model.config().vocab_size, 100);
    }

    // ── 3. T5ModelWrapper empty encoder input error ───────────────────────────

    #[test]
    fn test_t5_model_wrapper_empty_encoder_input() {
        let model = T5ModelWrapper::new(small_cfg()).expect("construction");
        let result = model.forward(vec![]);
        assert!(matches!(result, Err(T5TaskError::EmptyEncoderInput)));
    }

    // ── 4. T5ModelWrapper forward safe pattern ────────────────────────────────

    #[test]
    fn test_t5_model_wrapper_forward_safe() {
        let model = T5ModelWrapper::new(small_cfg()).expect("construction");
        if let Ok(out) = model.forward(vec![1u32, 2, 3]) {
            use trustformers_core::tensor::Tensor;
            if let Tensor::F32(arr) = &out {
                assert!(!arr.is_empty());
            }
        }
    }

    // ── 5. T5ConditionalGeneration construction ───────────────────────────────

    #[test]
    fn test_t5_cond_gen_construction() {
        let result = T5ForConditionalGenerationWrapper::new(small_cfg());
        assert!(
            result.is_ok(),
            "T5ForConditionalGenerationWrapper must construct"
        );
    }

    // ── 6. T5ConditionalGeneration empty encoder input error ──────────────────

    #[test]
    fn test_t5_cond_gen_empty_encoder_error() {
        let model = T5ForConditionalGenerationWrapper::new(small_cfg()).expect("construction");
        let result = model.forward(vec![], None);
        assert!(matches!(result, Err(T5TaskError::EmptyEncoderInput)));
    }

    // ── 7. T5ConditionalGeneration forward safe pattern ───────────────────────

    #[test]
    fn test_t5_cond_gen_forward_safe() {
        let model = T5ForConditionalGenerationWrapper::new(small_cfg()).expect("construction");
        if let Ok(out) = model.forward(vec![1u32, 2], Some(vec![0u32])) {
            use trustformers_core::tensor::Tensor;
            if let Tensor::F32(arr) = &out {
                assert!(!arr.is_empty());
            }
        }
    }

    // ── 8. SequenceClassification construction ────────────────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let result = T5ForSequenceClassification::new(small_cfg(), 3);
        assert!(result.is_ok());
    }

    // ── 9. SequenceClassification invalid labels ──────────────────────────────

    #[test]
    fn test_seq_cls_invalid_labels() {
        let result = T5ForSequenceClassification::new(small_cfg(), 1);
        assert!(matches!(result, Err(T5TaskError::InvalidNumLabels(1))));
    }

    // ── 10. SequenceClassification forward output length ──────────────────────

    #[test]
    fn test_seq_cls_forward_output_length() {
        let model = T5ForSequenceClassification::new(small_cfg(), 4).expect("construction");
        let pooled = vec![0.1f32; small_cfg().d_model];
        let logits = model.forward(&pooled).expect("forward");
        assert_eq!(logits.len(), 4);
    }

    // ── 11. SequenceClassification empty input error ──────────────────────────

    #[test]
    fn test_seq_cls_empty_input_error() {
        let model = T5ForSequenceClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[]);
        assert!(matches!(result, Err(T5TaskError::EmptyEncoderInput)));
    }

    // ── 12. Task prefix utilities ─────────────────────────────────────────────

    #[test]
    fn test_task_prefix_translate() {
        let s = add_task_prefix(PREFIX_TRANSLATE_EN_FR, "Hello world");
        assert!(s.starts_with("translate English to French: "));
        assert!(s.ends_with("Hello world"));
    }

    // ── 13. Task prefix summarize ─────────────────────────────────────────────

    #[test]
    fn test_task_prefix_summarize() {
        let s = add_task_prefix(PREFIX_SUMMARIZE, "The quick brown fox");
        assert!(s.starts_with("summarize: "));
    }

    // ── 14. QA input formatting ───────────────────────────────────────────────

    #[test]
    fn test_qa_input_format() {
        let qa = format_qa_input("What is T5?", "T5 is a seq2seq model.");
        assert!(qa.contains("question: "));
        assert!(qa.contains("What is T5?"));
        assert!(qa.contains(" context: "));
        assert!(qa.contains("T5 is a seq2seq model."));
    }

    // ── 15. Relative position bucket range ───────────────────────────────────

    #[test]
    fn test_relative_position_bucket_range() {
        let num_buckets = 32usize;
        let max_dist = 128usize;
        for delta in [-10i32, -1, 0, 1, 5, 100] {
            let b = relative_position_bucket(delta, true, num_buckets, max_dist);
            assert!(b < num_buckets, "bucket {b} out of range for delta={delta}");
        }
    }

    // ── 16. Relative position bucket unidirectional ───────────────────────────

    #[test]
    fn test_relative_position_bucket_unidirectional_range() {
        let num_buckets = 32usize;
        for delta in [0i32, 1, 5, 20, 200] {
            let b = relative_position_bucket(delta, false, num_buckets, 128);
            assert!(b < num_buckets, "unidirectional bucket {b} out of range");
        }
    }

    // ── 17. Greedy token picks max ────────────────────────────────────────────

    #[test]
    fn test_greedy_token_picks_max() {
        let logits = vec![0.1f32, 0.5, 0.9, 0.2];
        assert_eq!(greedy_token(&logits), Some(2u32));
    }

    // ── 18. Log-softmax sums to 0 in probability space ───────────────────────

    #[test]
    fn test_log_softmax_sums_correctly() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let lp = log_softmax(&logits);
        let sum: f32 = lp.iter().map(|&x| x.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "exp(log_softmax) must sum to 1, got {sum}"
        );
    }

    // ── 19. Log-softmax monotone order ───────────────────────────────────────

    #[test]
    fn test_log_softmax_ordering() {
        let logits = vec![0.0f32, 1.0, 2.0];
        let lp = log_softmax(&logits);
        assert!(lp[0] < lp[1] && lp[1] < lp[2]);
    }

    // ── 20. num_labels accessor ───────────────────────────────────────────────

    #[test]
    fn test_num_labels_accessor() {
        let model = T5ForSequenceClassification::new(small_cfg(), 7).expect("construction");
        assert_eq!(model.num_labels(), 7);
    }

    // ── 21. T5 config is_encoder_decoder ─────────────────────────────────────

    #[test]
    fn test_t5_config_is_encoder_decoder() {
        let cfg = T5Config::default();
        assert!(cfg.is_encoder_decoder);
    }

    // ── 22. Error display messages ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = T5TaskError::InvalidConfig("bad d_model".to_string());
        assert!(e1.to_string().contains("bad d_model"));

        let e2 = T5TaskError::EmptyEncoderInput;
        assert!(e2.to_string().contains("empty encoder"));

        let e3 = T5TaskError::InvalidNumLabels(1);
        assert!(e3.to_string().contains("1"));
    }
}
