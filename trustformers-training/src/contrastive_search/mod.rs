//! Contrastive Search Decoding
//!
//! This module implements contrastive search (Su et al., 2022), a decoding strategy
//! that balances model confidence with a degeneration penalty to produce diverse,
//! high-quality text without repetition.
//!
//! # Algorithm
//!
//! At each decoding step `t`, the selected token `x` maximises:
//!
//! ```text
//! score(x) = (1 - α) * p(x | context)
//!          - α * max_{v ∈ V_k} cos_sim(h_x, h_v)
//! ```
//!
//! where:
//! - `V_k` is the set of top-k candidates from the language model,
//! - `h_x` is the hidden-state / embedding of candidate `x`,
//! - `h_v` is the hidden-state of each previously generated token,
//! - `α` (`degeneration_penalty`) controls how strongly repetition is penalized.
//!
//! # Reference
//!
//! Su & Collier (2022): "A Contrastive Framework for Neural Text Generation"
//! <https://arxiv.org/abs/2202.06417>

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during contrastive search decoding.
#[derive(Debug, Clone, PartialEq)]
pub enum ContrastiveError {
    /// No candidate tokens were provided for this step.
    EmptyCandidates,
    /// Candidate embeddings have inconsistent dimensions.
    EmbeddingDimensionMismatch {
        /// Expected embedding dimension.
        expected: usize,
        /// Actual embedding dimension of the offending candidate.
        got: usize,
    },
    /// The decoder has already reached `max_length`.
    MaxLengthReached,
}

impl fmt::Display for ContrastiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContrastiveError::EmptyCandidates => {
                write!(f, "Contrastive search error: no candidates provided for this step")
            }
            ContrastiveError::EmbeddingDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Contrastive search error: embedding dimension mismatch — \
                     expected {expected}, got {got}"
                )
            }
            ContrastiveError::MaxLengthReached => {
                write!(
                    f,
                    "Contrastive search error: max_length has been reached; \
                     no more tokens can be generated"
                )
            }
        }
    }
}

impl std::error::Error for ContrastiveError {}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for contrastive search decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveSearchConfig {
    /// Number of top-k candidate tokens considered at each step (default: 5).
    pub top_k: usize,
    /// Degeneration penalty α ∈ [0, 1] (default: 0.6).
    pub degeneration_penalty: f32,
    /// Maximum number of tokens to generate (default: 128).
    pub max_length: usize,
    /// Temperature applied to logits before top-k selection (default: 1.0).
    pub temperature: f32,
    /// Token ID that triggers early stopping. `None` disables EOS (default: `None`).
    pub eos_token_id: Option<u32>,
    /// Minimum number of tokens to generate before EOS is allowed (default: 0).
    pub min_length: usize,
}

impl Default for ContrastiveSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            degeneration_penalty: 0.6,
            max_length: 128,
            temperature: 1.0,
            eos_token_id: None,
            min_length: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single candidate token from the language model's top-k output.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateToken {
    /// Token vocabulary index.
    pub token_id: u32,
    /// Probability of this token under the language model.
    pub prob: f32,
    /// Hidden state / embedding representation for this candidate at this position.
    pub embedding: Vec<f32>,
}

/// The decomposed contrastive score for a single candidate token.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveScore {
    /// The candidate token this score belongs to.
    pub token_id: u32,
    /// Model-confidence component: `(1 - α) * p(x)`.
    pub model_confidence: f32,
    /// Degeneration-penalty component: `α * max_cos_sim(h_x, context)`.
    pub degeneration_penalty: f32,
    /// Overall score: `model_confidence - degeneration_penalty`.
    pub total_score: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the cosine similarity between two embedding vectors.
///
/// Returns `0.0` if either vector has zero L2 norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: dimension mismatch");

    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute the contrastive score for a single candidate token.
///
/// # Formula
///
/// ```text
/// model_confidence   = (1 - α) * candidate.prob
/// degeneration       = α * max{ cos_sim(h_x, h_v) : h_v ∈ context_embeddings }
/// total_score        = model_confidence - degeneration
/// ```
///
/// If `context_embeddings` is empty, `degeneration` is set to `0.0` (no context
/// to repeat).
pub fn contrastive_score(
    candidate: &CandidateToken,
    context_embeddings: &[Vec<f32>],
    alpha: f32,
) -> ContrastiveScore {
    let model_confidence = (1.0 - alpha) * candidate.prob;

    let max_cos_sim = if context_embeddings.is_empty() {
        0.0_f32
    } else {
        context_embeddings
            .iter()
            .map(|ctx| cosine_similarity(&candidate.embedding, ctx))
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0) // clamp so penalty is non-negative
    };

    let degeneration_penalty = alpha * max_cos_sim;
    let total_score = model_confidence - degeneration_penalty;

    ContrastiveScore {
        token_id: candidate.token_id,
        model_confidence,
        degeneration_penalty,
        total_score,
    }
}

/// Select the best candidate token using contrastive scoring.
///
/// If `context_embeddings` is empty, the candidate with the highest probability
/// is returned (no degeneration penalty can be computed).
///
/// # Errors
///
/// Returns [`ContrastiveError::EmptyCandidates`] if `candidates` is empty.
/// Returns [`ContrastiveError::EmbeddingDimensionMismatch`] if any candidate
/// has a different embedding dimension from the first candidate.
pub fn select_contrastive_token(
    candidates: &[CandidateToken],
    context_embeddings: &[Vec<f32>],
    config: &ContrastiveSearchConfig,
) -> Result<CandidateToken, ContrastiveError> {
    if candidates.is_empty() {
        return Err(ContrastiveError::EmptyCandidates);
    }

    // Validate embedding dimensions are consistent
    let expected_dim = candidates[0].embedding.len();
    for cand in candidates.iter().skip(1) {
        if cand.embedding.len() != expected_dim {
            return Err(ContrastiveError::EmbeddingDimensionMismatch {
                expected: expected_dim,
                got: cand.embedding.len(),
            });
        }
    }

    // When there is no context, simply pick the highest-probability candidate
    if context_embeddings.is_empty() {
        let best = candidates
            .iter()
            .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal))
            .expect("candidates is non-empty");
        return Ok(best.clone());
    }

    let alpha = config.degeneration_penalty;

    let best = candidates
        .iter()
        .max_by(|a, b| {
            let sa = contrastive_score(a, context_embeddings, alpha);
            let sb = contrastive_score(b, context_embeddings, alpha);
            sa.total_score
                .partial_cmp(&sb.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("candidates is non-empty");

    Ok(best.clone())
}

// ─────────────────────────────────────────────────────────────────────────────
// ContrastiveDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful contrastive search decoder that accumulates the generated sequence
/// and context embeddings across steps.
#[derive(Debug, Clone)]
pub struct ContrastiveDecoder {
    /// Configuration for decoding.
    pub config: ContrastiveSearchConfig,
    /// Token IDs generated so far.
    pub generated_tokens: Vec<u32>,
    /// Hidden-state embeddings for each generated token.
    pub context_embeddings: Vec<Vec<f32>>,
    /// Whether an EOS token has been emitted.
    eos_emitted: bool,
}

impl ContrastiveDecoder {
    /// Create a new decoder with the given configuration.
    pub fn new(config: ContrastiveSearchConfig) -> Self {
        Self {
            config,
            generated_tokens: Vec::new(),
            context_embeddings: Vec::new(),
            eos_emitted: false,
        }
    }

    /// Reset the decoder state (generated tokens, context, EOS flag).
    pub fn reset(&mut self) {
        self.generated_tokens.clear();
        self.context_embeddings.clear();
        self.eos_emitted = false;
    }

    /// Perform one decoding step given a set of candidate tokens.
    ///
    /// 1. Selects the best token via contrastive scoring.
    /// 2. Appends the token ID and its embedding to the decoder state.
    /// 3. Checks for EOS (respecting `min_length`).
    ///
    /// # Returns
    ///
    /// The selected token ID.
    ///
    /// # Errors
    ///
    /// Returns [`ContrastiveError::MaxLengthReached`] if `is_finished()` was
    /// already true before this call.
    /// Propagates errors from `select_contrastive_token`.
    pub fn step(
        &mut self,
        candidates: Vec<CandidateToken>,
    ) -> Result<u32, ContrastiveError> {
        if self.is_finished() {
            return Err(ContrastiveError::MaxLengthReached);
        }

        let selected =
            select_contrastive_token(&candidates, &self.context_embeddings, &self.config)?;

        let token_id = selected.token_id;
        let embedding = selected.embedding.clone();

        self.generated_tokens.push(token_id);
        self.context_embeddings.push(embedding);

        // Check EOS condition
        let len = self.generated_tokens.len();
        if let Some(eos_id) = self.config.eos_token_id {
            if token_id == eos_id && len >= self.config.min_length {
                self.eos_emitted = true;
            }
        }

        Ok(token_id)
    }

    /// Returns `true` if decoding should stop (EOS emitted or max_length reached).
    pub fn is_finished(&self) -> bool {
        self.eos_emitted || self.generated_tokens.len() >= self.config.max_length
    }

    /// Return the sequence of generated token IDs.
    pub fn generated_sequence(&self) -> &[u32] {
        &self.generated_tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContrastiveGenerationOutput
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics from a complete contrastive decoding run.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveGenerationOutput {
    /// The generated token ID sequence (may include EOS).
    pub token_ids: Vec<u32>,
    /// Number of decoding steps taken.
    pub num_steps: usize,
    /// Whether generation ended because an EOS token was selected.
    pub stopped_by_eos: bool,
    /// Mean degeneration penalty across all steps.
    pub mean_degeneration_penalty: f32,
    /// Mean model confidence across all steps.
    pub mean_model_confidence: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Run contrastive decoding for multiple pre-computed candidate steps.
///
/// `candidates_per_step[t]` provides the candidate tokens for step `t`.  The
/// function drives a fresh [`ContrastiveDecoder`] through each step and
/// collects statistics.
///
/// Generation stops early if the decoder reaches EOS or `max_length` before
/// all steps are consumed.
///
/// # Errors
///
/// Propagates errors from [`ContrastiveDecoder::step`].
pub fn contrastive_decode(
    candidates_per_step: Vec<Vec<CandidateToken>>,
    config: &ContrastiveSearchConfig,
) -> Result<ContrastiveGenerationOutput, ContrastiveError> {
    let mut decoder = ContrastiveDecoder::new(config.clone());
    let mut scores_per_step: Vec<ContrastiveScore> = Vec::new();

    for candidates in candidates_per_step {
        if decoder.is_finished() {
            break;
        }

        // Compute scores before stepping (for statistics)
        let alpha = config.degeneration_penalty;
        let step_scores: Vec<ContrastiveScore> = candidates
            .iter()
            .map(|c| contrastive_score(c, &decoder.context_embeddings, alpha))
            .collect();

        let token_id = decoder.step(candidates)?;

        // Record score of the actually selected token
        if let Some(score) = step_scores.iter().find(|s| s.token_id == token_id) {
            scores_per_step.push(score.clone());
        }
    }

    let num_steps = decoder.generated_tokens.len();
    let stopped_by_eos = decoder.eos_emitted;

    let (mean_deg, mean_conf) = if scores_per_step.is_empty() {
        (0.0, 0.0)
    } else {
        let n = scores_per_step.len() as f32;
        let sum_deg: f32 = scores_per_step.iter().map(|s| s.degeneration_penalty).sum();
        let sum_conf: f32 = scores_per_step.iter().map(|s| s.model_confidence).sum();
        (sum_deg / n, sum_conf / n)
    };

    Ok(ContrastiveGenerationOutput {
        token_ids: decoder.generated_tokens,
        num_steps,
        stopped_by_eos,
        mean_degeneration_penalty: mean_deg,
        mean_model_confidence: mean_conf,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_candidate(token_id: u32, prob: f32, emb: Vec<f32>) -> CandidateToken {
        CandidateToken {
            token_id,
            prob,
            embedding: emb,
        }
    }

    // ── Config defaults ───────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = ContrastiveSearchConfig::default();
        assert_eq!(cfg.top_k, 5);
        assert!((cfg.degeneration_penalty - 0.6).abs() < 1e-8);
        assert_eq!(cfg.max_length, 128);
        assert!((cfg.temperature - 1.0).abs() < 1e-8);
        assert!(cfg.eos_token_id.is_none());
        assert_eq!(cfg.min_length, 0);
    }

    // ── cosine_similarity ────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "identical vectors should give 1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should give 0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&zero, &v)).abs() < 1e-8);
        assert!((cosine_similarity(&v, &zero)).abs() < 1e-8);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5, "opposite vectors should give -1.0, got {sim}");
    }

    // ── contrastive_score ────────────────────────────────────────────────

    #[test]
    fn test_contrastive_score_no_context() {
        let cand = make_candidate(42, 0.8, vec![1.0, 0.0]);
        let alpha = 0.6_f32;
        let score = contrastive_score(&cand, &[], alpha);

        let expected_conf = (1.0 - alpha) * 0.8;
        assert!((score.model_confidence - expected_conf).abs() < 1e-6);
        assert!((score.degeneration_penalty).abs() < 1e-8, "no context → no penalty");
        assert!((score.total_score - expected_conf).abs() < 1e-6);
        assert_eq!(score.token_id, 42);
    }

    #[test]
    fn test_contrastive_score_with_context_identical() {
        // Candidate embedding identical to context → max cos_sim = 1.0 → max penalty
        let emb = vec![1.0_f32, 0.0];
        let cand = make_candidate(1, 0.9, emb.clone());
        let context = vec![emb.clone()];
        let alpha = 0.6_f32;

        let score = contrastive_score(&cand, &context, alpha);

        let expected_conf = (1.0 - alpha) * 0.9;
        let expected_penalty = alpha * 1.0;
        assert!((score.model_confidence - expected_conf).abs() < 1e-5);
        assert!((score.degeneration_penalty - expected_penalty).abs() < 1e-5);
        assert!((score.total_score - (expected_conf - expected_penalty)).abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_score_penalizes_repetition() {
        let alpha = 0.6_f32;
        // Token 1: similar to context (high degeneration penalty)
        // Token 2: orthogonal to context (low penalty) but lower prob
        let context = vec![vec![1.0_f32, 0.0]];

        let cand_similar = make_candidate(1, 0.9, vec![1.0, 0.0]);
        let cand_ortho = make_candidate(2, 0.7, vec![0.0, 1.0]);

        let score_similar = contrastive_score(&cand_similar, &context, alpha);
        let score_ortho = contrastive_score(&cand_ortho, &context, alpha);

        // Orthogonal token should have lower degeneration penalty
        assert!(
            score_ortho.degeneration_penalty < score_similar.degeneration_penalty,
            "orthogonal penalty={} should < similar penalty={}",
            score_ortho.degeneration_penalty,
            score_similar.degeneration_penalty
        );
    }

    // ── select_contrastive_token ─────────────────────────────────────────

    #[test]
    fn test_select_empty_candidates() {
        let cfg = ContrastiveSearchConfig::default();
        let result = select_contrastive_token(&[], &[], &cfg);
        assert!(matches!(result, Err(ContrastiveError::EmptyCandidates)));
    }

    #[test]
    fn test_select_no_context_highest_prob() {
        let cfg = ContrastiveSearchConfig::default();
        let candidates = vec![
            make_candidate(1, 0.3, vec![1.0, 0.0]),
            make_candidate(2, 0.7, vec![0.0, 1.0]),
            make_candidate(3, 0.5, vec![0.5, 0.5]),
        ];
        let selected = select_contrastive_token(&candidates, &[], &cfg)
            .expect("selection failed");
        assert_eq!(selected.token_id, 2, "highest prob token should be selected with no context");
    }

    #[test]
    fn test_select_embedding_mismatch() {
        let cfg = ContrastiveSearchConfig::default();
        let candidates = vec![
            make_candidate(1, 0.5, vec![1.0, 0.0]),
            make_candidate(2, 0.5, vec![0.0, 1.0, 0.0]), // different dim
        ];
        let result = select_contrastive_token(&candidates, &[], &cfg);
        assert!(matches!(
            result,
            Err(ContrastiveError::EmbeddingDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_select_penalizes_repetition_in_context() {
        let mut cfg = ContrastiveSearchConfig::default();
        cfg.degeneration_penalty = 0.9; // high penalty

        // Context is in direction [1, 0]
        let context = vec![vec![1.0_f32, 0.0]];

        // token_id=1: prob=0.9 but embedding [1,0] → very similar to context
        // token_id=2: prob=0.5 but embedding [0,1] → orthogonal to context
        let candidates = vec![
            make_candidate(1, 0.9, vec![1.0, 0.0]),
            make_candidate(2, 0.5, vec![0.0, 1.0]),
        ];

        let selected = select_contrastive_token(&candidates, &context, &cfg)
            .expect("selection failed");

        // With high alpha, the non-repetitive token should win despite lower prob
        assert_eq!(
            selected.token_id, 2,
            "with high degeneration penalty, orthogonal token should be selected"
        );
    }

    // ── ContrastiveDecoder ───────────────────────────────────────────────

    #[test]
    fn test_decoder_step_basic() {
        let cfg = ContrastiveSearchConfig::default();
        let mut decoder = ContrastiveDecoder::new(cfg);

        let candidates = vec![
            make_candidate(10, 0.8, vec![1.0_f32, 0.0]),
            make_candidate(11, 0.2, vec![0.0_f32, 1.0]),
        ];
        let token = decoder.step(candidates).expect("step failed");
        // No context → highest prob = token 10
        assert_eq!(token, 10);
        assert_eq!(decoder.generated_sequence(), &[10]);
        assert_eq!(decoder.context_embeddings.len(), 1);
    }

    #[test]
    fn test_decoder_is_finished_on_eos() {
        let mut cfg = ContrastiveSearchConfig::default();
        cfg.eos_token_id = Some(99);
        cfg.max_length = 100;
        let mut decoder = ContrastiveDecoder::new(cfg);

        let candidates = vec![make_candidate(99, 1.0, vec![1.0_f32])];
        let token = decoder.step(candidates).expect("step failed");
        assert_eq!(token, 99);
        assert!(decoder.is_finished(), "EOS should mark decoder as finished");
        assert!(decoder.eos_emitted);
    }

    #[test]
    fn test_decoder_is_finished_on_max_length() {
        let mut cfg = ContrastiveSearchConfig::default();
        cfg.max_length = 2;
        let mut decoder = ContrastiveDecoder::new(cfg);

        for i in 0..2u32 {
            let candidates = vec![make_candidate(i, 1.0, vec![1.0_f32])];
            decoder.step(candidates).expect("step failed");
        }
        assert!(decoder.is_finished(), "max_length should mark decoder as finished");

        // Next step should error
        let candidates = vec![make_candidate(99, 1.0, vec![1.0_f32])];
        let result = decoder.step(candidates);
        assert!(matches!(result, Err(ContrastiveError::MaxLengthReached)));
    }

    #[test]
    fn test_decoder_reset() {
        let cfg = ContrastiveSearchConfig::default();
        let mut decoder = ContrastiveDecoder::new(cfg);

        let candidates = vec![make_candidate(5, 0.9, vec![1.0_f32])];
        decoder.step(candidates).expect("step failed");
        assert!(!decoder.generated_sequence().is_empty());

        decoder.reset();
        assert!(decoder.generated_sequence().is_empty());
        assert!(decoder.context_embeddings.is_empty());
        assert!(!decoder.is_finished());
    }

    // ── contrastive_decode ───────────────────────────────────────────────

    #[test]
    fn test_contrastive_decode_basic() {
        let cfg = ContrastiveSearchConfig::default();

        let steps: Vec<Vec<CandidateToken>> = (0..4)
            .map(|i| {
                vec![
                    make_candidate(i * 2, 0.7, vec![1.0_f32, 0.0]),
                    make_candidate(i * 2 + 1, 0.3, vec![0.0_f32, 1.0]),
                ]
            })
            .collect();

        let out = contrastive_decode(steps, &cfg).expect("decode failed");
        assert_eq!(out.num_steps, 4);
        assert_eq!(out.token_ids.len(), 4);
        assert!(!out.stopped_by_eos);
        assert!(out.mean_model_confidence >= 0.0);
        assert!(out.mean_degeneration_penalty >= 0.0);
    }

    #[test]
    fn test_contrastive_decode_stops_at_eos() {
        let mut cfg = ContrastiveSearchConfig::default();
        cfg.eos_token_id = Some(100);
        cfg.max_length = 20;

        // Steps: step 0 emits EOS immediately
        let steps: Vec<Vec<CandidateToken>> = vec![
            vec![make_candidate(100, 1.0, vec![1.0_f32])], // EOS
            vec![make_candidate(1, 1.0, vec![0.0_f32])],   // should not be reached
        ];

        let out = contrastive_decode(steps, &cfg).expect("decode failed");
        assert!(out.stopped_by_eos, "should have stopped at EOS");
        assert_eq!(out.num_steps, 1, "only EOS step should be taken");
        assert!(out.token_ids.contains(&100));
    }

    #[test]
    fn test_contrastive_decode_statistics() {
        let cfg = ContrastiveSearchConfig::default();

        let steps: Vec<Vec<CandidateToken>> = (0..3)
            .map(|i| {
                vec![make_candidate(i as u32, 0.8, vec![i as f32, 0.0])]
            })
            .collect();

        let out = contrastive_decode(steps, &cfg).expect("decode failed");
        // Statistics should be well-defined and non-negative
        assert!(out.mean_model_confidence >= 0.0);
        assert!(out.mean_degeneration_penalty >= 0.0);
        assert_eq!(out.num_steps, 3);
    }

    // ── Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = ContrastiveError::EmptyCandidates;
        assert!(e1.to_string().contains("candidates"));

        let e2 = ContrastiveError::EmbeddingDimensionMismatch { expected: 4, got: 3 };
        let msg = e2.to_string();
        assert!(msg.contains("4"), "message should mention expected dim: {msg}");
        assert!(msg.contains("3"), "message should mention actual dim: {msg}");

        let e3 = ContrastiveError::MaxLengthReached;
        assert!(e3.to_string().contains("max_length"));
    }

    // ── SimCSE / NT-Xent alignment & uniformity ─────────────────────────────

    /// Alignment = mean cosine similarity between positive pairs.
    /// Perfect alignment → 1.0.
    #[test]
    fn test_alignment_metric_perfect() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let alignment = cosine_similarity(&a, &b);
        assert!((alignment - 1.0).abs() < 1e-6, "perfect alignment should be 1.0, got {alignment}");
    }

    /// Alignment between orthogonal embeddings is 0.
    #[test]
    fn test_alignment_metric_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let alignment = cosine_similarity(&a, &b);
        assert!(alignment.abs() < 1e-6, "orthogonal alignment should be 0, got {alignment}");
    }

    /// Uniformity proxy: mean pairwise log-exp distance over a batch.
    /// Computed as log( mean_ij( exp(-2 * ||z_i - z_j||^2) ) ).
    /// For identical embeddings, all distances are 0 → exp(0)=1 → log(1)=0 (worst uniformity).
    #[test]
    fn test_uniformity_identical_embeddings_worst_case() {
        let embs = vec![
            vec![1.0_f32, 0.0],
            vec![1.0_f32, 0.0],
            vec![1.0_f32, 0.0],
        ];
        // Uniformity = log( mean_ij exp(-2 ||ei - ej||^2) ) — for identical vecs = log(1) = 0.
        let n = embs.len();
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist_sq: f32 = embs[i].iter().zip(embs[j].iter())
                        .map(|(a, b)| (a - b).powi(2)).sum();
                    sum += (-2.0 * dist_sq).exp();
                    count += 1;
                }
            }
        }
        let uniformity = (sum / count as f32).ln();
        assert!(
            (uniformity - 0.0).abs() < 1e-5,
            "identical embeddings uniformity should be 0.0 (log 1), got {uniformity}"
        );
    }

    /// Uniformity for spread embeddings is strongly negative (good uniformity).
    #[test]
    fn test_uniformity_spread_embeddings_better() {
        // Spread across 4 orthogonal directions.
        let embs = vec![
            vec![1.0_f32, 0.0],
            vec![-1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.0_f32, -1.0],
        ];
        let n = embs.len();
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist_sq: f32 = embs[i].iter().zip(embs[j].iter())
                        .map(|(a, b)| (a - b).powi(2)).sum();
                    sum += (-2.0 * dist_sq).exp();
                    count += 1;
                }
            }
        }
        let uniformity = (sum / count as f32).ln();
        assert!(uniformity < 0.0, "spread embeddings should have negative uniformity, got {uniformity}");
    }

    /// NT-Xent loss for a single positive pair over in-batch negatives.
    /// L = -log( exp(sim(z, z+)/τ) / Σ_k exp(sim(z, z_k)/τ) )
    #[test]
    fn test_nt_xent_loss_single_pair() {
        let tau = 0.07_f32;
        // Anchor z identical to z+, negatives orthogonal.
        let z     = vec![1.0_f32, 0.0];
        let z_pos = vec![1.0_f32, 0.0];
        let negatives: Vec<Vec<f32>> = vec![
            vec![0.0_f32, 1.0],
            vec![0.0_f32, -1.0],
        ];

        let sim_pos = cosine_similarity(&z, &z_pos) / tau;
        let neg_sims: Vec<f32> = negatives.iter()
            .map(|n| cosine_similarity(&z, n) / tau)
            .collect();

        // Numerically stable log-sum-exp.
        let all_sims: Vec<f32> = std::iter::once(sim_pos)
            .chain(neg_sims.iter().cloned())
            .collect();
        let max_s = all_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max_s + all_sims.iter().map(|s| (s - max_s).exp()).sum::<f32>().ln();
        let loss = -(sim_pos - log_sum_exp);

        assert!(loss >= 0.0, "NT-Xent loss must be non-negative, got {loss}");
        assert!(loss < 0.1, "perfect alignment should give near-zero loss, got {loss}");
    }

    /// NT-Xent loss increases when positive pair is dissimilar.
    #[test]
    fn test_nt_xent_loss_worse_for_dissimilar_positive() {
        let tau = 0.5_f32;
        let neg = vec![vec![0.0_f32, 1.0]];

        let compute_loss = |z: &[f32], z_pos: &[f32]| {
            let sim_pos = cosine_similarity(z, z_pos) / tau;
            let neg_sim = cosine_similarity(z, &neg[0]) / tau;
            let all = vec![sim_pos, neg_sim];
            let max_s = all.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_s + all.iter().map(|s| (s - max_s).exp()).sum::<f32>().ln();
            -(sim_pos - lse)
        };

        let loss_good = compute_loss(&[1.0_f32, 0.0], &[1.0_f32, 0.0]);  // identical
        let loss_bad  = compute_loss(&[1.0_f32, 0.0], &[-1.0_f32, 0.0]); // opposite
        assert!(
            loss_bad > loss_good,
            "dissimilar positive should give higher NT-Xent loss: bad={loss_bad} good={loss_good}"
        );
    }

    /// Temperature scaling: higher τ produces smaller score differences.
    #[test]
    fn test_contrastive_temperature_effect() {
        let context = vec![vec![1.0_f32, 0.0]];
        let cand = make_candidate(1, 0.8, vec![1.0_f32, 0.0]);

        // Two configs differing only in alpha (proxy for temperature effect on scores).
        let mut cfg_low = ContrastiveSearchConfig::default();
        cfg_low.degeneration_penalty = 0.1;
        let mut cfg_high = ContrastiveSearchConfig::default();
        cfg_high.degeneration_penalty = 0.9;

        let score_low  = contrastive_score(&cand, &context, cfg_low.degeneration_penalty);
        let score_high = contrastive_score(&cand, &context, cfg_high.degeneration_penalty);

        // Higher penalty → larger degeneration component.
        assert!(
            score_high.degeneration_penalty > score_low.degeneration_penalty,
            "higher alpha should increase penalty: {} vs {}",
            score_high.degeneration_penalty, score_low.degeneration_penalty
        );
    }

    /// Hard negative mining: the closest (most similar) candidate is the hard negative.
    #[test]
    fn test_hard_negative_mining_closest() {
        // Anchor embedding.
        let anchor = vec![1.0_f32, 0.0];
        // Candidates at various similarities.
        let candidates = vec![
            (0usize, vec![1.0_f32, 0.0]),    // sim=1.0 (positive / hard negative)
            (1usize, vec![0.7_f32, 0.7]),    // sim≈0.707
            (2usize, vec![0.0_f32, 1.0]),    // sim=0.0
            (3usize, vec![-1.0_f32, 0.0]),   // sim=-1.0 (easy negative)
        ];

        // Hard negative = max cosine similarity (excluding the positive at index 0).
        let hard_neg_idx = candidates.iter().skip(1)
            .max_by(|(_, a), (_, b)| {
                cosine_similarity(&anchor, a)
                    .partial_cmp(&cosine_similarity(&anchor, b))
                    .expect("valid float comparison")
            })
            .map(|(i, _)| *i)
            .expect("non-empty");

        // Candidate 1 has sim≈0.707 which is highest among the negatives.
        assert_eq!(hard_neg_idx, 1, "hard negative should be the most similar, got {hard_neg_idx}");
    }

    /// Contrastive accuracy: fraction of steps where the positive pair ranks highest.
    #[test]
    fn test_contrastive_accuracy_all_correct() {
        // Build a trivial scenario: 3 steps, each step the first candidate is the
        // designated "positive" and has the highest model confidence + lowest penalty.
        let cfg = ContrastiveSearchConfig { degeneration_penalty: 0.0, ..Default::default() };

        let steps: Vec<(u32, Vec<CandidateToken>)> = (0..3u32).map(|t| {
            // Positive: high prob; negatives: low prob. No context → highest prob wins.
            let positive_id = t * 10;
            let candidates = vec![
                make_candidate(positive_id,     0.9, vec![1.0_f32]),
                make_candidate(positive_id + 1, 0.1, vec![0.0_f32]),
            ];
            (positive_id, candidates)
        }).collect();

        let total = steps.len();
        let mut correct = 0usize;
        for (positive_id, candidates) in &steps {
            let selected = select_contrastive_token(candidates, &[], &cfg)
                .expect("selection should succeed");
            if selected.token_id == *positive_id {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total as f32;
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "all steps should be correct, accuracy={accuracy}"
        );
    }

    // ── min_length guard ─────────────────────────────────────────────────────

    /// EOS should not fire before min_length is reached.
    #[test]
    fn test_decoder_eos_respects_min_length() {
        let mut cfg = ContrastiveSearchConfig::default();
        cfg.eos_token_id = Some(99);
        cfg.min_length = 3;
        cfg.max_length = 20;
        let mut decoder = ContrastiveDecoder::new(cfg);

        // First two steps produce EOS token — should NOT trigger stop.
        for _ in 0..2 {
            let candidates = vec![make_candidate(99, 1.0, vec![1.0_f32])];
            decoder.step(candidates).expect("step failed");
            assert!(!decoder.is_finished(), "EOS before min_length should not finish decoder");
        }

        // Third step also produces EOS → now len == min_length → should finish.
        let candidates = vec![make_candidate(99, 1.0, vec![1.0_f32])];
        decoder.step(candidates).expect("step failed");
        assert!(decoder.is_finished(), "EOS at min_length should finish decoder");
    }

    // ── multi-step context accumulation ──────────────────────────────────────

    /// Verify that context embeddings accumulate correctly across steps.
    #[test]
    fn test_decoder_context_accumulates() {
        let cfg = ContrastiveSearchConfig::default();
        let mut decoder = ContrastiveDecoder::new(cfg);

        for step in 0..4u32 {
            let emb = vec![step as f32, 0.0_f32];
            let candidates = vec![make_candidate(step, 0.8, emb)];
            decoder.step(candidates).expect("step failed");
        }

        assert_eq!(decoder.context_embeddings.len(), 4, "context should have one entry per step");
        assert_eq!(decoder.generated_tokens.len(), 4, "should have generated 4 tokens");
    }

    // ── RepBERT margin loss proxy ─────────────────────────────────────────────

    /// RepBERT-style margin loss: max(0, margin - sim_pos + sim_neg).
    /// With a large margin, the loss should be positive when positives and negatives are close.
    #[test]
    fn test_repbert_margin_loss_violated() {
        let margin = 0.5_f32;
        let anchor = vec![1.0_f32, 0.0];
        let positive = vec![0.9_f32, 0.44]; // sim≈0.9
        let negative  = vec![0.8_f32, 0.6]; // sim≈0.8

        let sim_pos = cosine_similarity(&anchor, &positive);
        let sim_neg = cosine_similarity(&anchor, &negative);
        let loss = (margin - sim_pos + sim_neg).max(0.0);
        assert!(loss > 0.0, "margin loss should be violated: loss={loss}");
    }

    /// RepBERT-style margin loss: should be zero when positives are clearly more similar.
    #[test]
    fn test_repbert_margin_loss_satisfied() {
        let margin = 0.1_f32;
        let anchor = vec![1.0_f32, 0.0];
        let positive = vec![1.0_f32, 0.0];  // sim=1.0
        let negative  = vec![0.0_f32, 1.0]; // sim=0.0

        let sim_pos = cosine_similarity(&anchor, &positive);
        let sim_neg = cosine_similarity(&anchor, &negative);
        let loss = (margin - sim_pos + sim_neg).max(0.0);
        assert!(
            (loss - 0.0).abs() < 1e-5,
            "margin loss should be 0 when positive is clearly better, got {loss}"
        );
    }
}
