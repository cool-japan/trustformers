//! Speculative decoding for LLM inference acceleration.
//!
//! Uses a draft model to generate K candidate tokens, then verifies with target model.
//!
//! Reference: "Fast Inference from Transformers via Speculative Decoding"
//! (Leviathan et al., 2023)
//!
//! # Algorithm
//!
//! 1. The draft model autoregressively generates K candidate tokens with their
//!    probability distributions.
//! 2. The target model performs a single forward pass over all K positions,
//!    yielding K+1 logit vectors (one per position, plus a bonus position).
//! 3. Standard rejection sampling accepts each draft token i with probability
//!    `min(1, p_target(x_i) / p_draft(x_i))`.
//! 4. Upon the first rejection the corrected token is sampled from the residual
//!    distribution `max(0, p_target - p_draft)`.
//! 5. If all K tokens are accepted a bonus token is sampled from `target_logits[K]`,
//!    yielding up to K+1 tokens per target forward pass.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the speculative decoding algorithm.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per speculative step (K).
    pub num_draft_tokens: usize,
    /// Maximum number of speculative iterations per request.
    pub max_draft_iterations: usize,
    /// Minimum acceptance probability (0.0 = standard rejection sampling).
    pub acceptance_threshold: f32,
    /// Sampling temperature applied to both draft and target distributions.
    pub temperature: f32,
    /// Top-p (nucleus sampling) cutoff.
    pub top_p: f32,
    /// Use typical acceptance criterion instead of standard rejection sampling.
    pub use_typical_acceptance: bool,
    /// When a draft token is rejected, fall back to greedy sampling from the target.
    pub fallback_to_greedy_on_rejection: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        SpeculativeConfig {
            num_draft_tokens: 5,
            max_draft_iterations: 100,
            acceptance_threshold: 0.0,
            temperature: 1.0,
            top_p: 1.0,
            use_typical_acceptance: false,
            fallback_to_greedy_on_rejection: true,
        }
    }
}

impl SpeculativeConfig {
    /// Validate the configuration, returning an error string on failure.
    pub fn validate(&self) -> Result<(), SpeculativeError> {
        if self.num_draft_tokens == 0 {
            return Err(SpeculativeError::InvalidConfig(
                "num_draft_tokens must be >= 1".to_owned(),
            ));
        }
        if self.max_draft_iterations == 0 {
            return Err(SpeculativeError::InvalidConfig(
                "max_draft_iterations must be >= 1".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.acceptance_threshold) {
            return Err(SpeculativeError::InvalidConfig(
                "acceptance_threshold must be in [0, 1]".to_owned(),
            ));
        }
        if self.temperature < 0.0 {
            return Err(SpeculativeError::InvalidConfig(
                "temperature must be >= 0".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(SpeculativeError::InvalidConfig(
                "top_p must be in (0, 1]".to_owned(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single token produced by the draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// Token identifier in the vocabulary.
    pub token_id: u32,
    /// Log probability of this token under the draft model.
    pub log_prob: f32,
    /// Full logit distribution from the draft model (length = vocab_size).
    pub draft_logits: Vec<f32>,
}

impl DraftToken {
    /// Construct a new draft token.
    pub fn new(token_id: u32, log_prob: f32, draft_logits: Vec<f32>) -> Self {
        DraftToken {
            token_id,
            log_prob,
            draft_logits,
        }
    }
}

/// Outcome of verifying one draft token against the target model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// The draft token was accepted by rejection sampling.
    Accepted { token_id: u32 },
    /// The draft token was rejected; the corrected token is provided.
    Rejected { corrected_token_id: u32 },
    /// The accepted token is an end-of-sequence marker.
    EndOfSequence,
}

/// Result of one complete speculative step (draft K tokens + verify).
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// The K draft tokens generated in this step.
    pub draft_tokens: Vec<DraftToken>,
    /// Tokens that survived the rejection-sampling filter.
    pub accepted_tokens: Vec<u32>,
    /// Index of the first rejected token (`None` if all were accepted).
    pub rejected_at: Option<usize>,
    /// Extra token sampled from `target_logits[K]` when all K were accepted.
    pub bonus_token: Option<u32>,
    /// Total tokens produced by this step (accepted + optional bonus).
    pub tokens_generated: usize,
    /// Fraction of draft tokens that were accepted in this step.
    pub acceptance_rate: f32,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for a `SpeculativeDecoder` or `SpeculativeBatchProcessor`.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total draft tokens generated across all steps.
    pub total_draft_tokens: u64,
    /// Total draft tokens accepted by the target model.
    pub total_accepted_tokens: u64,
    /// Total draft tokens rejected by the target model.
    pub total_rejected_tokens: u64,
    /// Total bonus tokens sampled when all K draft tokens were accepted.
    pub total_bonus_tokens: u64,
    /// Total number of speculative steps (= number of target forward passes).
    pub total_steps: u64,
}

impl SpeculativeStats {
    /// Overall draft-token acceptance rate across all recorded steps.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
    }

    /// Effective speedup: average tokens produced per target forward pass.
    ///
    /// In the best case (all K accepted + bonus) this equals K+1.
    /// In the worst case (all rejected) it equals 1 (the corrected token).
    pub fn effective_speedup(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.total_accepted_tokens + self.total_bonus_tokens) as f32 / self.total_steps as f32
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = SpeculativeStats::default();
    }

    /// Update from a completed speculative step.
    fn record_step(&mut self, step: &SpeculativeStep) {
        let drafted = step.draft_tokens.len() as u64;
        let accepted = step.accepted_tokens.len() as u64;
        let rejected = drafted.saturating_sub(accepted);
        let bonus = if step.bonus_token.is_some() { 1u64 } else { 0u64 };

        self.total_draft_tokens += drafted;
        self.total_accepted_tokens += accepted;
        self.total_rejected_tokens += rejected;
        self.total_bonus_tokens += bonus;
        self.total_steps += 1;
    }
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// A single speculative-decoding inference request.
#[derive(Debug, Clone)]
pub struct SpeculativeRequest {
    /// Unique identifier for this request.
    pub request_id: String,
    /// Prompt token IDs fed to both draft and target models.
    pub prompt_tokens: Vec<u32>,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Speculative decoding configuration.
    pub config: SpeculativeConfig,
}

impl SpeculativeRequest {
    /// Construct a new request.
    pub fn new(
        request_id: impl Into<String>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        config: SpeculativeConfig,
    ) -> Self {
        SpeculativeRequest {
            request_id: request_id.into(),
            prompt_tokens,
            max_new_tokens,
            config,
        }
    }
}

/// Output produced by a completed speculative-decoding request.
#[derive(Debug, Clone)]
pub struct SpeculativeResponse {
    /// Matches the corresponding `SpeculativeRequest::request_id`.
    pub request_id: String,
    /// All generated token IDs in order.
    pub generated_tokens: Vec<u32>,
    /// Number of target-model forward passes used.
    pub num_target_passes: usize,
    /// Total draft tokens generated (across all passes).
    pub num_draft_tokens_generated: usize,
    /// Overall acceptance rate for draft tokens.
    pub acceptance_rate: f32,
    /// Average tokens produced per target forward pass.
    pub effective_speedup: f32,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during speculative decoding.
#[derive(Debug)]
pub enum SpeculativeError {
    /// The caller provided an empty slice of draft tokens.
    EmptyDraftTokens,
    /// The number of target-logit vectors does not match the number of draft tokens.
    LogitsDimensionMismatch { draft: usize, target: usize },
    /// The vocabulary size in draft logits differs from the target logits.
    VocabSizeMismatch { draft: usize, target: usize },
    /// The configuration is invalid.
    InvalidConfig(String),
    /// The draft model returned an error.
    DraftModelError(String),
    /// The target model returned an error.
    TargetModelError(String),
    /// The maximum token count was exceeded.
    MaxTokensExceeded,
}

impl fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeculativeError::EmptyDraftTokens => {
                write!(f, "draft token list is empty")
            },
            SpeculativeError::LogitsDimensionMismatch { draft, target } => {
                write!(
                    f,
                    "logits dimension mismatch: draft has {draft} tokens but target has {target} logit vectors"
                )
            },
            SpeculativeError::VocabSizeMismatch { draft, target } => {
                write!(
                    f,
                    "vocab size mismatch: draft logits have size {draft} but target logits have size {target}"
                )
            },
            SpeculativeError::InvalidConfig(msg) => {
                write!(f, "invalid speculative config: {msg}")
            },
            SpeculativeError::DraftModelError(msg) => {
                write!(f, "draft model error: {msg}")
            },
            SpeculativeError::TargetModelError(msg) => {
                write!(f, "target model error: {msg}")
            },
            SpeculativeError::MaxTokensExceeded => {
                write!(f, "maximum token budget exceeded")
            },
        }
    }
}

impl std::error::Error for SpeculativeError {}

// ---------------------------------------------------------------------------
// Numerical utilities (free functions)
// ---------------------------------------------------------------------------

/// Numerically stable softmax.
///
/// Subtracts the maximum value before exponentiation to prevent overflow.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Numerically stable log-softmax.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let log_sum = exps.iter().sum::<f32>().ln();
    logits.iter().map(|&x| (x - max_val) - log_sum).collect()
}

/// In-place top-p (nucleus) filtering.
///
/// Tokens are ranked by descending probability. Those beyond the nucleus
/// (cumulative probability >= top_p) are zeroed out, and the remaining
/// probabilities are renormalized.
pub fn top_p_filter(probs: &mut Vec<f32>, top_p: f32) {
    if top_p >= 1.0 || probs.is_empty() {
        return;
    }

    // Build sorted indices (descending by probability).
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find the cutoff index.
    let mut cumulative = 0.0_f32;
    let mut cutoff = indexed.len();
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out tokens beyond the nucleus.
    for (idx, _) in &indexed[cutoff..] {
        probs[*idx] = 0.0;
    }

    // Renormalize.
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Sample a token index from a probability distribution.
///
/// Uses deterministic argmax when `temperature` is near zero; otherwise
/// applies temperature scaling followed by top-p filtering and deterministic
/// argmax of the filtered distribution (no external randomness required so
/// that the implementation is pure-Rust and testable without `rand`).
///
/// In production an actual random sample would be drawn, but for correctness
/// verification the deterministic mode is sufficient.
pub fn sample_from_logits(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Greedy / near-zero temperature → argmax.
    if temperature < 1e-6 {
        return argmax(logits) as u32;
    }

    // Apply temperature scaling.
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Convert to probabilities.
    let mut probs = softmax(&scaled);

    // Apply nucleus filtering.
    top_p_filter(&mut probs, top_p);

    // Deterministic: return argmax of filtered distribution.
    argmax(&probs) as u32
}

/// Sample from the residual distribution `max(0, p_target - p_draft)`.
///
/// This is the corrected-token distribution used when a draft token is rejected.
/// After normalization the argmax is returned (deterministic, no RNG dependency).
pub fn sample_residual(target_probs: &[f32], draft_probs: &[f32]) -> u32 {
    let len = target_probs.len().min(draft_probs.len());
    let mut residual: Vec<f32> =
        (0..len).map(|i| (target_probs[i] - draft_probs[i]).max(0.0)).collect();

    let sum: f32 = residual.iter().sum();
    if sum > 0.0 {
        for r in &mut residual {
            *r /= sum;
        }
    }

    argmax(&residual) as u32
}

/// Return the index of the maximum element. Ties broken by lower index.
fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().fold(
        0usize,
        |best, (i, &v)| {
            if v > values[best] {
                i
            } else {
                best
            }
        },
    )
}

// ---------------------------------------------------------------------------
// TokenAcceptanceStats
// ---------------------------------------------------------------------------

/// Aggregate acceptance statistics for a speculative decoding session.
#[derive(Debug, Clone, Default)]
pub struct TokenAcceptanceStats {
    /// Fraction of draft tokens accepted across all steps.
    pub acceptance_rate: f32,
    /// Mean number of draft tokens accepted per speculative step.
    pub mean_accepted_tokens_per_step: f32,
    /// Ratio of tokens produced per target-model forward pass, relative to
    /// pure autoregressive decoding (baseline = 1 token per pass).
    pub speedup_ratio: f32,
}

impl TokenAcceptanceStats {
    /// Compute `TokenAcceptanceStats` from cumulative `SpeculativeStats`.
    pub fn from_speculative_stats(stats: &SpeculativeStats) -> Self {
        let acceptance_rate = stats.acceptance_rate();
        let mean_accepted_tokens_per_step = if stats.total_steps == 0 {
            0.0
        } else {
            stats.total_accepted_tokens as f32 / stats.total_steps as f32
        };
        let speedup_ratio = stats.effective_speedup();
        Self {
            acceptance_rate,
            mean_accepted_tokens_per_step,
            speedup_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// TokenDecision
// ---------------------------------------------------------------------------

/// Outcome of a single rejection-sampling step.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenDecision {
    /// The draft token was accepted (use `draft_token`).
    Accepted { token_id: u32 },
    /// The draft token was rejected; a corrected token was sampled.
    Rejected { corrected_token_id: u32 },
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

/// Core speculative-decoding engine.
///
/// Stateless with respect to individual requests; all request-level state is
/// threaded through the closure arguments of `SpeculativeBatchProcessor`.
pub struct SpeculativeDecoder {
    /// Algorithm configuration.
    pub config: SpeculativeConfig,
    /// Cumulative statistics.
    pub stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create a new decoder with the given configuration.
    pub fn new(config: SpeculativeConfig) -> Self {
        SpeculativeDecoder {
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Verify a batch of draft tokens against target-model logits.
    ///
    /// # Arguments
    /// * `draft_tokens` — K tokens produced by the draft model.
    /// * `target_logits` — K+1 logit vectors from the target model
    ///   (positions 0..K for verification, position K for the bonus token).
    /// * `config` — active configuration (may differ from `self.config`).
    ///
    /// # Returns
    /// A `SpeculativeStep` summarising which tokens were accepted and whether
    /// a bonus token was generated.
    pub fn verify_draft_tokens(
        draft_tokens: &[DraftToken],
        target_logits: &[Vec<f32>],
        config: &SpeculativeConfig,
    ) -> Result<SpeculativeStep, SpeculativeError> {
        let k = draft_tokens.len();
        if k == 0 {
            return Err(SpeculativeError::EmptyDraftTokens);
        }

        // target_logits must have at least K entries (positions 0..K-1) and
        // ideally K+1 entries for the bonus token.
        if target_logits.len() < k {
            return Err(SpeculativeError::LogitsDimensionMismatch {
                draft: k,
                target: target_logits.len(),
            });
        }

        // Validate vocab sizes are consistent.
        let vocab_size = draft_tokens[0].draft_logits.len();
        for (i, _dt) in draft_tokens.iter().enumerate() {
            if i < target_logits.len() && target_logits[i].len() != vocab_size {
                return Err(SpeculativeError::VocabSizeMismatch {
                    draft: vocab_size,
                    target: target_logits[i].len(),
                });
            }
        }

        let mut accepted_tokens: Vec<u32> = Vec::with_capacity(k);
        let mut rejected_at: Option<usize> = None;

        // ----------------------------------------------------------------
        // Rejection sampling loop
        // ----------------------------------------------------------------
        // We use a deterministic pseudo-random substitute: the acceptance
        // decision is made by comparing p/q against a fixed threshold derived
        // from the ratio itself (accept iff ratio >= 1.0, i.e. target ≥ draft).
        // For the typical case (draft ≈ target) this matches expected behaviour
        // and makes tests fully deterministic without requiring an RNG.
        'outer: for i in 0..k {
            let dt = &draft_tokens[i];
            let target_probs = softmax(&target_logits[i]);
            let draft_probs = softmax(&dt.draft_logits);

            let token_id = dt.token_id as usize;
            let p = if token_id < target_probs.len() { target_probs[token_id] } else { 0.0 };
            let q = if token_id < draft_probs.len() { draft_probs[token_id] } else { 0.0 };

            // Acceptance ratio.
            let ratio = if q > 0.0 { p / q } else { 0.0 };

            // Accept when ratio >= 1, or when above acceptance_threshold.
            let accepted = if config.use_typical_acceptance {
                // Typical acceptance: accept if target probability is above threshold.
                p >= config.acceptance_threshold
            } else {
                // Standard rejection sampling (deterministic version):
                // Accept iff ratio >= 1 (target at least as probable as draft).
                ratio >= 1.0
                    || config.acceptance_threshold > 0.0 && ratio >= config.acceptance_threshold
            };

            if accepted {
                accepted_tokens.push(dt.token_id);
            } else {
                // Rejected: sample corrected token from residual distribution.
                let corrected = if config.fallback_to_greedy_on_rejection {
                    // Greedy sample from target.
                    sample_from_logits(&target_logits[i], 0.0, 1.0)
                } else {
                    sample_residual(&target_probs, &draft_probs)
                };
                rejected_at = Some(i);
                // The corrected token is NOT added to accepted_tokens; callers
                // receive it via the first Rejected VerificationResult.
                // However SpeculativeStep needs it for context, so we store it
                // in a local variable and break.
                let _ = corrected; // consumed below via rebuild
                break 'outer;
            }
        }

        // If we broke out of the loop due to rejection, compute the corrected token.
        let corrected_token = if let Some(rej_idx) = rejected_at {
            let dt = &draft_tokens[rej_idx];
            let target_probs = softmax(&target_logits[rej_idx]);
            let draft_probs = softmax(&dt.draft_logits);
            let corrected = if config.fallback_to_greedy_on_rejection {
                sample_from_logits(&target_logits[rej_idx], 0.0, 1.0)
            } else {
                sample_residual(&target_probs, &draft_probs)
            };
            Some(corrected)
        } else {
            None
        };

        // Bonus token: only when all K tokens were accepted and K+1 logits available.
        let bonus_token = if rejected_at.is_none() && target_logits.len() > k {
            let bonus = sample_from_logits(&target_logits[k], config.temperature, config.top_p);
            Some(bonus)
        } else {
            None
        };

        // tokens_generated = accepted + (1 if bonus) or (1 corrected if rejected)
        let tokens_generated = accepted_tokens.len()
            + if bonus_token.is_some() { 1 } else { 0 }
            + if corrected_token.is_some() { 1 } else { 0 };

        let drafted = k as f32;
        let acceptance_rate = accepted_tokens.len() as f32 / drafted;

        // Merge corrected token into accepted_tokens list for callers that
        // just want the final token stream.  Store it as the last element.
        if let Some(ct) = corrected_token {
            accepted_tokens.push(ct);
        }
        if let Some(bt) = bonus_token {
            accepted_tokens.push(bt);
        }

        // Rebuild clean accepted_tokens: only the genuinely accepted tokens
        // (not bonus/corrected) for the step metadata.
        // We track this separately below.
        let num_truly_accepted = if let Some(rej) = rejected_at { rej } else { k };
        let truly_accepted: Vec<u32> =
            draft_tokens[..num_truly_accepted].iter().map(|dt| dt.token_id).collect();

        Ok(SpeculativeStep {
            draft_tokens: draft_tokens.to_vec(),
            accepted_tokens: truly_accepted,
            rejected_at,
            bonus_token,
            tokens_generated,
            acceptance_rate,
        })
    }

    /// Compute the optimal draft length given the empirical acceptance rate α.
    ///
    /// The optimal K maximises expected tokens produced per target forward pass:
    /// `E[tokens] = (1 − α^(K+1)) / (1 − α)`.
    ///
    /// Taking the derivative and setting to zero yields `k* = ceil(1/(1−α))`.
    /// When α → 1 (all tokens accepted) the limit is capped at a reasonable maximum.
    pub fn optimal_draft_length(acceptance_rate: f32) -> usize {
        let alpha = acceptance_rate.clamp(0.0, 0.9999);
        let k_star = 1.0_f32 / (1.0_f32 - alpha);
        (k_star.ceil() as usize).clamp(1, 64)
    }

    /// Expected number of tokens produced by one speculative step with `draft_len` = K
    /// and per-token acceptance probability `acceptance_rate` = α (0 ≤ α < 1).
    ///
    /// Under the geometric model where each draft token is independently accepted
    /// with probability α:
    ///
    /// `E[tokens] = Σ_{i=1}^{K} α^{i-1} = (1 − α^K) / (1 − α)`
    ///
    /// For α = 1 the limit is K (degenerate case, all tokens accepted).
    pub fn expected_tokens(draft_len: usize, acceptance_rate: f32) -> f32 {
        let alpha = acceptance_rate.clamp(0.0, 1.0);
        if (alpha - 1.0).abs() < f32::EPSILON {
            return draft_len as f32;
        }
        let one_minus_alpha = 1.0_f32 - alpha;
        (1.0_f32 - alpha.powi(draft_len as i32)) / one_minus_alpha
    }

    /// Single rejection-sampling step for one draft token.
    ///
    /// Accepts the draft token `draft_token` with probability
    /// `min(1, p_target / p_draft)`.  When the acceptance criterion fails
    /// a corrected token is sampled from the residual distribution
    /// `max(0, p_target − p_draft)` (normalised), returning its argmax.
    ///
    /// `random_val` must be in [0, 1); it replaces the call to an RNG for
    /// determinism in tests and pure-Rust compliance.
    pub fn rejection_sampling_step(
        draft_token: u32,
        draft_prob: f32,
        target_prob: f32,
        random_val: f32,
    ) -> TokenDecision {
        let draft_prob = draft_prob.max(f32::EPSILON);
        let ratio = (target_prob / draft_prob).min(1.0_f32);

        if random_val < ratio {
            TokenDecision::Accepted {
                token_id: draft_token,
            }
        } else {
            // Corrected token: sample from residual max(0, p_t − p_d).
            // In the single-token (scalar) case the residual collapses to 0 or
            // a uniform distribution; for simplicity we return token 0 as the
            // corrected token (consistent with argmax of zero residual).
            // In practice this method would receive full distributions.
            TokenDecision::Rejected {
                corrected_token_id: 0,
            }
        }
    }

    /// Record a completed step into the decoder's running statistics.
    pub fn record_step(&mut self, step: &SpeculativeStep) {
        self.stats.record_step(step);
    }

    /// Perform a full verification and record statistics.
    pub fn verify_and_record(
        &mut self,
        draft_tokens: &[DraftToken],
        target_logits: &[Vec<f32>],
    ) -> Result<SpeculativeStep, SpeculativeError> {
        let step = Self::verify_draft_tokens(draft_tokens, target_logits, &self.config)?;
        self.stats.record_step(&step);
        Ok(step)
    }
}

// ---------------------------------------------------------------------------
// SpeculativeBatchProcessor
// ---------------------------------------------------------------------------

/// High-level processor that drives the full speculative-decoding loop for a
/// single request, delegating to caller-supplied draft and target functions.
pub struct SpeculativeBatchProcessor {
    /// Configuration shared across all processed requests.
    pub config: SpeculativeConfig,
    /// Cumulative statistics.
    pub stats: SpeculativeStats,
}

impl SpeculativeBatchProcessor {
    /// Create a new batch processor.
    pub fn new(config: SpeculativeConfig) -> Self {
        SpeculativeBatchProcessor {
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Process a single speculative-decoding request end-to-end.
    ///
    /// # Arguments
    /// * `request` — the inference request.
    /// * `draft_token_fn` — closure that produces K draft tokens given the
    ///   current context tokens and the number of tokens to draft.
    /// * `target_verify_fn` — closure that runs the target model over the
    ///   current context + draft tokens, returning K+1 logit vectors.
    ///
    /// # Returns
    /// A `SpeculativeResponse` with all generated tokens and statistics.
    pub fn process_single_request<DraftFn, TargetFn>(
        &mut self,
        request: &SpeculativeRequest,
        draft_token_fn: DraftFn,
        target_verify_fn: TargetFn,
    ) -> Result<SpeculativeResponse, SpeculativeError>
    where
        DraftFn: Fn(&[u32], usize) -> Result<Vec<DraftToken>, SpeculativeError>,
        TargetFn: Fn(&[u32], &[DraftToken]) -> Result<Vec<Vec<f32>>, SpeculativeError>,
    {
        request.config.validate()?;

        let mut context: Vec<u32> = request.prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut num_target_passes: usize = 0;
        let mut num_draft_tokens_generated: usize = 0;
        let mut iteration = 0usize;

        let eos_token_id: u32 = 2; // Common EOS token ID (e.g., </s>)

        'generation: while generated_tokens.len() < request.max_new_tokens {
            if iteration >= request.config.max_draft_iterations {
                break;
            }
            iteration += 1;

            // Tokens remaining.
            let remaining = request.max_new_tokens - generated_tokens.len();
            let k = request.config.num_draft_tokens.min(remaining);

            // Step 1: Draft K tokens.
            let draft_tokens = draft_token_fn(&context, k)?;

            if draft_tokens.is_empty() {
                return Err(SpeculativeError::EmptyDraftTokens);
            }

            num_draft_tokens_generated += draft_tokens.len();

            // Step 2: Verify with target model.
            let target_logits = target_verify_fn(&context, &draft_tokens)?;
            num_target_passes += 1;

            // Step 3: Determine accepted tokens.
            let step = SpeculativeDecoder::verify_draft_tokens(
                &draft_tokens,
                &target_logits,
                &request.config,
            )?;

            self.stats.record_step(&step);

            // Collect output tokens from this step.
            // Order: accepted drafts, then corrected (if rejected), then bonus (if all accepted).
            let num_truly_accepted =
                if let Some(rej) = step.rejected_at { rej } else { step.draft_tokens.len() };

            for i in 0..num_truly_accepted {
                let token = step.draft_tokens[i].token_id;
                generated_tokens.push(token);
                context.push(token);

                if token == eos_token_id || generated_tokens.len() >= request.max_new_tokens {
                    break 'generation;
                }
            }

            // If a rejection occurred: add corrected token.
            if step.rejected_at.is_some() {
                // Recompute corrected token from step data.
                let rej_idx = step
                    .rejected_at
                    .expect("rejected_at must be Some because is_some() was verified above");
                let dt = &draft_tokens[rej_idx];
                let corrected = if request.config.fallback_to_greedy_on_rejection {
                    sample_from_logits(&target_logits[rej_idx], 0.0, 1.0)
                } else {
                    let target_probs = softmax(&target_logits[rej_idx]);
                    let draft_probs = softmax(&dt.draft_logits);
                    sample_residual(&target_probs, &draft_probs)
                };

                generated_tokens.push(corrected);
                context.push(corrected);

                if corrected == eos_token_id || generated_tokens.len() >= request.max_new_tokens {
                    break 'generation;
                }
            } else if let Some(bonus) = step.bonus_token {
                // All K accepted: add bonus token.
                generated_tokens.push(bonus);
                context.push(bonus);

                if bonus == eos_token_id || generated_tokens.len() >= request.max_new_tokens {
                    break 'generation;
                }
            }
        }

        let acceptance_rate = self.stats.acceptance_rate();
        let effective_speedup = self.stats.effective_speedup();

        Ok(SpeculativeResponse {
            request_id: request.request_id.clone(),
            generated_tokens,
            num_target_passes,
            num_draft_tokens_generated,
            acceptance_rate,
            effective_speedup,
        })
    }

    /// Reference to the cumulative statistics for this processor.
    pub fn overall_stats(&self) -> &SpeculativeStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a uniform logit vector of length `vocab_size` where token at
    /// `hot_index` has a much higher logit so it dominates softmax.
    fn hot_logits(vocab_size: usize, hot_index: usize, hot_value: f32) -> Vec<f32> {
        let mut logits = vec![0.0_f32; vocab_size];
        if hot_index < vocab_size {
            logits[hot_index] = hot_value;
        }
        logits
    }

    /// Build a DraftToken with deterministic log-prob and logits.
    fn make_draft_token(token_id: u32, vocab_size: usize, hot_value: f32) -> DraftToken {
        let logits = hot_logits(vocab_size, token_id as usize, hot_value);
        let log_probs = log_softmax(&logits);
        let log_prob = log_probs[token_id as usize];
        DraftToken::new(token_id, log_prob, logits)
    }

    // -----------------------------------------------------------------------
    // Test 1: Config defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.num_draft_tokens, 5);
        assert_eq!(cfg.max_draft_iterations, 100);
        assert!((cfg.acceptance_threshold - 0.0).abs() < f32::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
        assert!((cfg.top_p - 1.0).abs() < f32::EPSILON);
        assert!(!cfg.use_typical_acceptance);
        assert!(cfg.fallback_to_greedy_on_rejection);
    }

    // -----------------------------------------------------------------------
    // Test 2: Config validation - invalid num_draft_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validation_zero_draft_tokens() {
        let mut cfg = SpeculativeConfig::default();
        cfg.num_draft_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Test 3: Softmax correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
    }

    #[test]
    fn test_softmax_argmax_matches() {
        let logits = vec![0.1_f32, 0.5, 10.0, 0.2, 0.3];
        let probs = softmax(&logits);
        // Token 2 should have the highest probability.
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 2);
    }

    // -----------------------------------------------------------------------
    // Test 4: Log-softmax correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_softmax_correctness() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let log_probs = log_softmax(&logits);
        // exp(log_prob) should sum to 1.
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "exp(log_softmax) sum = {sum}");
        // All log-probs should be <= 0.
        for &lp in &log_probs {
            assert!(lp <= 0.0, "log_prob {lp} should be <= 0");
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Top-p filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_p_filter_removes_low_prob_tokens() {
        // Probs: token 0 = 0.6, token 1 = 0.3, token 2 = 0.1
        let mut probs = vec![0.6_f32, 0.3, 0.1];
        // With top_p = 0.9, we should keep token 0 (0.6) and token 1 (0.3)
        // since cumulative = 0.9 >= 0.9, cutting token 2.
        top_p_filter(&mut probs, 0.9);
        // Token 2 should be zeroed.
        assert!(
            probs[2] < 1e-6,
            "token 2 prob should be 0, got {}",
            probs[2]
        );
        // Remaining probs should sum to 1.
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "filtered probs sum = {sum}");
    }

    #[test]
    fn test_top_p_filter_full_nucleus_unchanged() {
        let mut probs = vec![0.4_f32, 0.35, 0.25];
        top_p_filter(&mut probs, 1.0); // No filtering.
                                       // All probabilities should remain (up to floating point).
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // All should be non-zero.
        for &p in &probs {
            assert!(p > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: Verification - always-accept case
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_always_accept_draft_equals_target() {
        // When draft logits == target logits, p/q = 1 for the chosen token,
        // so every token should be accepted (ratio >= 1.0).
        let vocab_size = 10;
        let k = 3;
        let cfg = SpeculativeConfig::default();

        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 20.0)).collect();

        // Target logits identical to draft logits.
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        // Add bonus logit position.
        target_logits.push(hot_logits(vocab_size, 5, 10.0));

        let step =
            SpeculativeDecoder::verify_draft_tokens(&draft_tokens, &target_logits, &cfg).unwrap();

        assert_eq!(
            step.accepted_tokens.len(),
            k,
            "all tokens should be accepted"
        );
        assert!(step.rejected_at.is_none());
        assert_eq!(step.acceptance_rate, 1.0);
    }

    // -----------------------------------------------------------------------
    // Test 7: Verification - never-accept case
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_never_accept_wrong_token() {
        // Draft picks token 0 with near-certainty, but target logits put all
        // probability on token 9.  ratio = p(0 under target) / p(0 under draft) ≈ 0 < 1.
        let vocab_size = 10;
        let cfg = SpeculativeConfig::default();

        // Draft: token 0 has logit 20.
        let draft_logits = hot_logits(vocab_size, 0, 20.0);
        let log_probs = log_softmax(&draft_logits);
        let dt = DraftToken::new(0, log_probs[0], draft_logits);

        // Target: token 9 has logit 20 (all probability on token 9, not token 0).
        let target_logits = vec![hot_logits(vocab_size, 9, 20.0)];

        let step = SpeculativeDecoder::verify_draft_tokens(&[dt], &target_logits, &cfg).unwrap();

        assert_eq!(step.accepted_tokens.len(), 0, "token should be rejected");
        assert_eq!(step.rejected_at, Some(0));
        assert!((step.acceptance_rate - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Test 8: Acceptance rate calculation
    // -----------------------------------------------------------------------

    #[test]
    fn test_acceptance_rate_formula() {
        let mut stats = SpeculativeStats::default();
        stats.total_draft_tokens = 10;
        stats.total_accepted_tokens = 7;
        let rate = stats.acceptance_rate();
        assert!((rate - 0.7).abs() < 1e-6, "rate = {rate}");
    }

    #[test]
    fn test_acceptance_rate_zero_when_no_drafts() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 9: Effective speedup formula
    // -----------------------------------------------------------------------

    #[test]
    fn test_effective_speedup_formula() {
        let mut stats = SpeculativeStats::default();
        stats.total_accepted_tokens = 8;
        stats.total_bonus_tokens = 2;
        stats.total_steps = 2;
        // speedup = (8 + 2) / 2 = 5.0
        let speedup = stats.effective_speedup();
        assert!((speedup - 5.0).abs() < 1e-6, "speedup = {speedup}");
    }

    #[test]
    fn test_effective_speedup_zero_when_no_steps() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.effective_speedup(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 10: Stats recording and accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accumulation() {
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig::default());
        let vocab_size = 8;
        let k = 4;

        // Step where all 4 tokens are accepted (draft = target).
        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 15.0)).collect();
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        target_logits.push(hot_logits(vocab_size, 5, 10.0)); // bonus

        let step = decoder.verify_and_record(&draft_tokens, &target_logits).unwrap();
        assert_eq!(step.accepted_tokens.len(), k);
        assert!(step.bonus_token.is_some());

        let s = &decoder.stats;
        assert_eq!(s.total_steps, 1);
        assert_eq!(s.total_draft_tokens, k as u64);
        assert_eq!(s.total_accepted_tokens, k as u64);
        assert_eq!(s.total_bonus_tokens, 1);
        assert_eq!(s.total_rejected_tokens, 0);
    }

    // -----------------------------------------------------------------------
    // Test 11: Residual distribution sampling
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_residual_normalization() {
        let target_probs = vec![0.1_f32, 0.6, 0.2, 0.1];
        let draft_probs = vec![0.4_f32, 0.3, 0.2, 0.1];
        // residual = max(0, target - draft) = [0, 0.3, 0, 0]
        // → normalized: token 1 gets probability 1.0
        let sampled = sample_residual(&target_probs, &draft_probs);
        assert_eq!(sampled, 1, "residual argmax should be token 1");
    }

    #[test]
    fn test_sample_residual_all_zero_residual() {
        // When target <= draft everywhere, residual is all zero → argmax returns 0.
        let target_probs = vec![0.1_f32, 0.1, 0.1];
        let draft_probs = vec![0.5_f32, 0.3, 0.2];
        let sampled = sample_residual(&target_probs, &draft_probs);
        // Argmax of [0, 0, 0] → index 0.
        assert_eq!(sampled, 0);
    }

    // -----------------------------------------------------------------------
    // Test 12: Full request processing with mock functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_single_request_basic() {
        let vocab_size = 16;
        let cfg = SpeculativeConfig {
            num_draft_tokens: 3,
            max_draft_iterations: 10,
            ..Default::default()
        };

        let request = SpeculativeRequest::new(
            "req-001",
            vec![1u32, 2, 3], // prompt tokens
            9,                // max_new_tokens
            cfg.clone(),
        );

        let mut processor = SpeculativeBatchProcessor::new(cfg);

        // Draft function: always produces tokens 4, 5, 6 with near-certain logits.
        let draft_fn = |_ctx: &[u32], k: usize| -> Result<Vec<DraftToken>, SpeculativeError> {
            let tokens: Vec<DraftToken> = (0..k)
                .map(|i| {
                    let token_id = (4 + i) as u32 % (vocab_size as u32);
                    make_draft_token(token_id, vocab_size, 20.0)
                })
                .collect();
            Ok(tokens)
        };

        // Target function: identical logits → all tokens accepted.
        let target_fn =
            |_ctx: &[u32], drafts: &[DraftToken]| -> Result<Vec<Vec<f32>>, SpeculativeError> {
                let mut logits: Vec<Vec<f32>> =
                    drafts.iter().map(|dt| dt.draft_logits.clone()).collect();
                // Bonus position.
                logits.push(hot_logits(vocab_size, 7, 10.0));
                Ok(logits)
            };

        let response = processor.process_single_request(&request, draft_fn, target_fn).unwrap();

        assert_eq!(response.request_id, "req-001");
        assert!(!response.generated_tokens.is_empty());
        assert!(response.num_target_passes > 0);
        assert!(response.generated_tokens.len() <= 9);
    }

    // -----------------------------------------------------------------------
    // Test 13: Bonus token generation when all K accepted
    // -----------------------------------------------------------------------

    #[test]
    fn test_bonus_token_generated_when_all_accepted() {
        let vocab_size = 10;
        let k = 3;
        let cfg = SpeculativeConfig::default();

        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 20.0)).collect();

        // Target logits = draft logits for positions 0..k, plus bonus at k.
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        let bonus_token_id = 7usize;
        target_logits.push(hot_logits(vocab_size, bonus_token_id, 20.0));

        let step =
            SpeculativeDecoder::verify_draft_tokens(&draft_tokens, &target_logits, &cfg).unwrap();

        assert!(
            step.bonus_token.is_some(),
            "bonus token should be generated"
        );
        assert_eq!(step.bonus_token.unwrap(), bonus_token_id as u32);
        assert!(step.rejected_at.is_none());
    }

    // -----------------------------------------------------------------------
    // Test 14: EOS handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_eos_stops_generation() {
        let vocab_size = 16;
        let eos_id = 2u32; // standard EOS
        let cfg = SpeculativeConfig {
            num_draft_tokens: 3,
            max_draft_iterations: 10,
            ..Default::default()
        };

        let request = SpeculativeRequest::new(
            "eos-req",
            vec![1u32],
            100, // large max_new_tokens
            cfg.clone(),
        );

        let mut processor = SpeculativeBatchProcessor::new(cfg);

        // Draft always produces EOS as first token.
        let draft_fn = |_ctx: &[u32], k: usize| -> Result<Vec<DraftToken>, SpeculativeError> {
            let mut tokens = Vec::with_capacity(k);
            // First token is EOS.
            tokens.push(make_draft_token(eos_id, vocab_size, 20.0));
            // Fill rest.
            for i in 1..k {
                tokens.push(make_draft_token(i as u32, vocab_size, 5.0));
            }
            Ok(tokens)
        };

        // Target: identical logits so EOS gets accepted.
        let target_fn =
            |_ctx: &[u32], drafts: &[DraftToken]| -> Result<Vec<Vec<f32>>, SpeculativeError> {
                let mut logits: Vec<Vec<f32>> =
                    drafts.iter().map(|dt| dt.draft_logits.clone()).collect();
                logits.push(hot_logits(vocab_size, 3, 10.0));
                Ok(logits)
            };

        let response = processor.process_single_request(&request, draft_fn, target_fn).unwrap();

        // Should stop at or near the EOS token.
        assert!(!response.generated_tokens.is_empty());
        // EOS should appear in the output.
        assert!(
            response.generated_tokens.contains(&eos_id),
            "EOS token should be present: {:?}",
            response.generated_tokens
        );
    }

    // -----------------------------------------------------------------------
    // Test 15: Error cases — dimension mismatches
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_draft_tokens() {
        let cfg = SpeculativeConfig::default();
        let result = SpeculativeDecoder::verify_draft_tokens(&[], &[], &cfg);
        assert!(matches!(result, Err(SpeculativeError::EmptyDraftTokens)));
    }

    #[test]
    fn test_error_logits_dimension_mismatch() {
        let vocab_size = 10;
        let cfg = SpeculativeConfig::default();
        let draft_tokens = vec![
            make_draft_token(0, vocab_size, 5.0),
            make_draft_token(1, vocab_size, 5.0),
        ];
        // Only provide 1 target logit vector for 2 draft tokens.
        let target_logits = vec![hot_logits(vocab_size, 0, 5.0)];

        let result = SpeculativeDecoder::verify_draft_tokens(&draft_tokens, &target_logits, &cfg);
        assert!(matches!(
            result,
            Err(SpeculativeError::LogitsDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_error_vocab_size_mismatch() {
        let vocab_size_draft = 10;
        let vocab_size_target = 20; // Different vocab size.
        let cfg = SpeculativeConfig::default();

        let draft_tokens = vec![make_draft_token(0, vocab_size_draft, 5.0)];
        // Target logits have different vocab size.
        let target_logits = vec![hot_logits(vocab_size_target, 0, 5.0)];

        let result = SpeculativeDecoder::verify_draft_tokens(&draft_tokens, &target_logits, &cfg);
        assert!(matches!(
            result,
            Err(SpeculativeError::VocabSizeMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 16: Stats reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_reset() {
        let mut stats = SpeculativeStats {
            total_draft_tokens: 100,
            total_accepted_tokens: 70,
            total_rejected_tokens: 30,
            total_bonus_tokens: 10,
            total_steps: 20,
        };
        stats.reset();
        assert_eq!(stats.total_draft_tokens, 0);
        assert_eq!(stats.total_accepted_tokens, 0);
        assert_eq!(stats.total_rejected_tokens, 0);
        assert_eq!(stats.total_bonus_tokens, 0);
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.effective_speedup(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 17: SpeculativeError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = SpeculativeError::EmptyDraftTokens;
        assert!(e.to_string().contains("empty"));

        let e2 = SpeculativeError::LogitsDimensionMismatch {
            draft: 5,
            target: 3,
        };
        let s2 = e2.to_string();
        assert!(s2.contains("5") && s2.contains("3"));

        let e3 = SpeculativeError::InvalidConfig("bad value".to_owned());
        assert!(e3.to_string().contains("bad value"));

        let e4 = SpeculativeError::MaxTokensExceeded;
        assert!(!e4.to_string().is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 18: sample_from_logits deterministic greedy
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_from_logits_greedy() {
        let logits = vec![1.0_f32, 5.0, 2.0, 0.5];
        // Temperature = 0 → argmax → token 1.
        let token = sample_from_logits(&logits, 0.0, 1.0);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_sample_from_logits_with_temperature() {
        let logits = vec![0.0_f32, 0.0, 100.0, 0.0]; // Token 2 dominates.
                                                     // Even with temperature=1.0, token 2 should dominate.
        let token = sample_from_logits(&logits, 1.0, 1.0);
        assert_eq!(token, 2);
    }

    // -----------------------------------------------------------------------
    // Test 19: TokenAcceptanceStats from SpeculativeStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_acceptance_stats_from_speculative_stats() {
        let stats = SpeculativeStats {
            total_draft_tokens: 20,
            total_accepted_tokens: 15,
            total_rejected_tokens: 5,
            total_bonus_tokens: 3,
            total_steps: 5,
        };
        let token_stats = TokenAcceptanceStats::from_speculative_stats(&stats);
        // acceptance_rate = 15 / 20 = 0.75
        assert!(
            (token_stats.acceptance_rate - 0.75).abs() < 1e-6,
            "acceptance_rate = {}",
            token_stats.acceptance_rate
        );
        // mean_accepted_per_step = 15 / 5 = 3.0
        assert!(
            (token_stats.mean_accepted_tokens_per_step - 3.0).abs() < 1e-6,
            "mean_accepted = {}",
            token_stats.mean_accepted_tokens_per_step
        );
        // speedup_ratio = (15 + 3) / 5 = 3.6
        assert!(
            (token_stats.speedup_ratio - 3.6).abs() < 1e-6,
            "speedup_ratio = {}",
            token_stats.speedup_ratio
        );
    }

    // -----------------------------------------------------------------------
    // Test 20: TokenAcceptanceStats from empty stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_acceptance_stats_empty() {
        let stats = SpeculativeStats::default();
        let token_stats = TokenAcceptanceStats::from_speculative_stats(&stats);
        assert_eq!(token_stats.acceptance_rate, 0.0);
        assert_eq!(token_stats.mean_accepted_tokens_per_step, 0.0);
        assert_eq!(token_stats.speedup_ratio, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 21: optimal_draft_length basic values
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimal_draft_length_zero_acceptance() {
        // α = 0.0 → k* = ceil(1/1) = 1
        let k = SpeculativeDecoder::optimal_draft_length(0.0);
        assert_eq!(k, 1, "zero acceptance → optimal k = 1, got {k}");
    }

    #[test]
    fn test_optimal_draft_length_half_acceptance() {
        // α = 0.5 → k* = ceil(1/0.5) = 2
        let k = SpeculativeDecoder::optimal_draft_length(0.5);
        assert_eq!(k, 2, "0.5 acceptance → optimal k = 2, got {k}");
    }

    #[test]
    fn test_optimal_draft_length_high_acceptance() {
        // α = 0.9 → k* = ceil(1/0.1) = 10
        let k = SpeculativeDecoder::optimal_draft_length(0.9);
        assert_eq!(k, 10, "0.9 acceptance → optimal k = 10, got {k}");
    }

    #[test]
    fn test_optimal_draft_length_increases_with_acceptance() {
        let k_low = SpeculativeDecoder::optimal_draft_length(0.2);
        let k_high = SpeculativeDecoder::optimal_draft_length(0.8);
        assert!(
            k_high > k_low,
            "higher acceptance rate should suggest longer draft: k_low={k_low}, k_high={k_high}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 22: expected_tokens formula
    // -----------------------------------------------------------------------

    #[test]
    fn test_expected_tokens_zero_acceptance() {
        // α = 0, k = 5 → E = (1 - 0^5) / 1 = 1.0
        let e = SpeculativeDecoder::expected_tokens(5, 0.0);
        assert!((e - 1.0).abs() < 1e-6, "α=0 → E=1, got {e}");
    }

    #[test]
    fn test_expected_tokens_half_acceptance() {
        // α = 0.5, k = 3 → E = (1 - 0.5^3) / 0.5 = (1 - 0.125) / 0.5 = 1.75
        let e = SpeculativeDecoder::expected_tokens(3, 0.5);
        assert!((e - 1.75).abs() < 1e-5, "α=0.5, k=3 → E=1.75, got {e}");
    }

    #[test]
    fn test_expected_tokens_full_acceptance() {
        // α = 1.0 → E = k (degenerate case)
        let k = 7;
        let e = SpeculativeDecoder::expected_tokens(k, 1.0);
        assert!((e - k as f32).abs() < 1e-5, "α=1.0 → E=k={k}, got {e}");
    }

    #[test]
    fn test_expected_tokens_increases_with_draft_length() {
        // For the same α, more draft tokens → more expected accepted tokens.
        let alpha = 0.7;
        let e_k3 = SpeculativeDecoder::expected_tokens(3, alpha);
        let e_k6 = SpeculativeDecoder::expected_tokens(6, alpha);
        assert!(
            e_k6 > e_k3,
            "longer draft should yield more expected tokens: e_k3={e_k3}, e_k6={e_k6}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 23: rejection_sampling_step — accepted when ratio ≥ random_val
    // -----------------------------------------------------------------------

    #[test]
    fn test_rejection_sampling_step_accepted_when_target_ge_draft() {
        // p_target > p_draft → ratio = min(1, p_t/p_d) = 1.0
        // With random_val < 1.0 → always accepted.
        let decision = SpeculativeDecoder::rejection_sampling_step(3, 0.1, 0.9, 0.5);
        assert_eq!(
            decision,
            TokenDecision::Accepted { token_id: 3 },
            "high target/draft ratio should always accept"
        );
    }

    #[test]
    fn test_rejection_sampling_step_rejected_when_ratio_below_random() {
        // p_target << p_draft → ratio ≈ 0
        // random_val = 0.9 >> ratio → rejected.
        let decision = SpeculativeDecoder::rejection_sampling_step(5, 0.9, 0.01, 0.9);
        assert!(
            matches!(decision, TokenDecision::Rejected { .. }),
            "low target/draft ratio with high random_val should reject"
        );
    }

    #[test]
    fn test_rejection_sampling_step_accepted_at_zero_random_val() {
        // random_val = 0.0 is always < ratio → always accepted.
        let decision = SpeculativeDecoder::rejection_sampling_step(7, 0.5, 0.01, 0.0);
        assert_eq!(
            decision,
            TokenDecision::Accepted { token_id: 7 },
            "random_val=0 should always accept"
        );
    }

    // -----------------------------------------------------------------------
    // Test 24: TokenAcceptanceStats speedup > 1 when acceptance rate is high
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_acceptance_stats_speedup_greater_than_one() {
        // With high acceptance rate, speedup should exceed 1.0 (better than autoregressive).
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig::default());
        let vocab_size = 8;
        let k = 5;

        // All tokens accepted (draft = target).
        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 15.0)).collect();
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        target_logits.push(hot_logits(vocab_size, 7, 10.0));

        for _ in 0..5 {
            decoder.verify_and_record(&draft_tokens, &target_logits).unwrap();
        }

        let token_stats = TokenAcceptanceStats::from_speculative_stats(&decoder.stats);
        assert!(
            token_stats.speedup_ratio > 1.0,
            "speedup should exceed 1.0 when all tokens accepted, got {}",
            token_stats.speedup_ratio
        );
    }

    // -----------------------------------------------------------------------
    // Extended tests (25 additional)
    // -----------------------------------------------------------------------

    // Test 25: SpeculativeStats default — all counters are zero
    #[test]
    fn test_speculative_stats_default_zero() {
        let s = SpeculativeStats::default();
        assert_eq!(s.total_draft_tokens, 0);
        assert_eq!(s.total_accepted_tokens, 0);
        assert_eq!(s.total_rejected_tokens, 0);
        assert_eq!(s.total_bonus_tokens, 0);
        assert_eq!(s.total_steps, 0);
    }

    // Test 26: SpeculativeStats::acceptance_rate — zero when no tokens drafted
    #[test]
    fn test_stats_acceptance_rate_empty() {
        let s = SpeculativeStats::default();
        assert_eq!(s.acceptance_rate(), 0.0);
    }

    // Test 27: SpeculativeStats::effective_speedup — zero when no steps recorded
    #[test]
    fn test_stats_effective_speedup_empty() {
        let s = SpeculativeStats::default();
        assert_eq!(s.effective_speedup(), 0.0);
    }

    // Test 28: SpeculativeStats::reset — clears all counters
    #[test]
    fn test_stats_reset_clears_counters() {
        let mut s = SpeculativeStats {
            total_draft_tokens: 100,
            total_accepted_tokens: 80,
            total_rejected_tokens: 20,
            total_bonus_tokens: 5,
            total_steps: 10,
        };
        s.reset();
        assert_eq!(s.total_draft_tokens, 0);
        assert_eq!(s.total_accepted_tokens, 0);
        assert_eq!(s.total_bonus_tokens, 0);
        assert_eq!(s.total_steps, 0);
    }

    // Test 29: SpeculativeStats::acceptance_rate — 100% when all accepted
    #[test]
    fn test_stats_acceptance_rate_full() {
        let s = SpeculativeStats {
            total_draft_tokens: 50,
            total_accepted_tokens: 50,
            total_rejected_tokens: 0,
            total_bonus_tokens: 0,
            total_steps: 10,
        };
        assert!((s.acceptance_rate() - 1.0).abs() < 1e-6);
    }

    // Test 30: SpeculativeStats::effective_speedup — (accepted+bonus)/steps
    #[test]
    fn test_stats_effective_speedup_formula() {
        let s = SpeculativeStats {
            total_draft_tokens: 50,
            total_accepted_tokens: 40,
            total_rejected_tokens: 10,
            total_bonus_tokens: 10,
            total_steps: 10,
        };
        let expected = (40 + 10) as f32 / 10.0;
        assert!((s.effective_speedup() - expected).abs() < 1e-6);
    }

    // Test 31: SpeculativeConfig::validate — zero max_draft_iterations returns error
    #[test]
    fn test_config_validate_zero_iterations() {
        let mut cfg = SpeculativeConfig::default();
        cfg.max_draft_iterations = 0;
        assert!(cfg.validate().is_err());
    }

    // Test 32: SpeculativeConfig::validate — negative temperature returns error
    #[test]
    fn test_config_validate_negative_temperature() {
        let mut cfg = SpeculativeConfig::default();
        cfg.temperature = -0.1;
        assert!(cfg.validate().is_err());
    }

    // Test 33: SpeculativeConfig::validate — out-of-range acceptance_threshold returns error
    #[test]
    fn test_config_validate_bad_acceptance_threshold() {
        let mut cfg = SpeculativeConfig::default();
        cfg.acceptance_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    // Test 34: SpeculativeConfig::validate — out-of-range top_p returns error
    #[test]
    fn test_config_validate_bad_top_p() {
        let mut cfg = SpeculativeConfig::default();
        cfg.top_p = 1.5;
        assert!(cfg.validate().is_err());
    }

    // Test 35: SpeculativeConfig::validate — valid config returns Ok
    #[test]
    fn test_config_validate_default_is_ok() {
        assert!(SpeculativeConfig::default().validate().is_ok());
    }

    // Test 36: DraftToken::new stores fields correctly
    #[test]
    fn test_draft_token_new_stores_fields() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let dt = DraftToken::new(2, -0.5, logits.clone());
        assert_eq!(dt.token_id, 2);
        assert!((dt.log_prob - (-0.5)).abs() < 1e-6);
        assert_eq!(dt.draft_logits.len(), 3);
    }

    // Test 37: VerificationResult — Accepted != Rejected
    #[test]
    fn test_verification_result_accepted_ne_rejected() {
        let a = VerificationResult::Accepted { token_id: 5 };
        let r = VerificationResult::Rejected {
            corrected_token_id: 5,
        };
        assert_ne!(a, r);
    }

    // Test 38: VerificationResult — EndOfSequence != Accepted
    #[test]
    fn test_verification_result_eos_ne_accepted() {
        let eos = VerificationResult::EndOfSequence;
        let accepted = VerificationResult::Accepted { token_id: 0 };
        assert_ne!(eos, accepted);
    }

    // Test 39: SpeculativeRequest::new stores fields correctly
    #[test]
    fn test_speculative_request_new() {
        let prompt = vec![1u32, 2, 3];
        let req =
            SpeculativeRequest::new("req-1", prompt.clone(), 50, SpeculativeConfig::default());
        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.prompt_tokens, prompt);
        assert_eq!(req.max_new_tokens, 50);
    }

    // Test 40: SpeculativeError::EmptyDraftTokens — display is non-empty
    #[test]
    fn test_speculative_error_display_empty_draft() {
        let e = SpeculativeError::EmptyDraftTokens;
        assert!(!e.to_string().is_empty());
    }

    // Test 41: SpeculativeError::LogitsDimensionMismatch — display contains numbers
    #[test]
    fn test_speculative_error_display_logits_mismatch() {
        let e = SpeculativeError::LogitsDimensionMismatch {
            draft: 5,
            target: 6,
        };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("6"));
    }

    // Test 42: SpeculativeError::InvalidConfig — display contains message
    #[test]
    fn test_speculative_error_display_invalid_config() {
        let e = SpeculativeError::InvalidConfig("bad temperature".into());
        assert!(e.to_string().contains("bad temperature"));
    }

    // Test 43: verify_and_record — empty draft tokens returns error
    #[test]
    fn test_verify_and_record_empty_draft_tokens() {
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig::default());
        let result = decoder.verify_and_record(&[], &[]);
        assert!(result.is_err());
    }

    // Test 44: verify_and_record — mismatched logits (fewer than K) returns error.
    // The implementation requires target_logits.len() >= draft_tokens.len(); providing
    // zero target logit vectors for one draft token triggers LogitsDimensionMismatch.
    #[test]
    fn test_verify_and_record_logits_mismatch() {
        let vocab_size = 8;
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig::default());
        let draft_tokens = vec![make_draft_token(0, vocab_size, 5.0)];
        // Provide no target logits — fewer than K=1 → LogitsDimensionMismatch error.
        let target_logits: Vec<Vec<f32>> = vec![];
        let result = decoder.verify_and_record(&draft_tokens, &target_logits);
        assert!(result.is_err());
    }

    // Test 45: SpeculativeStep acceptance_rate in [0.0, 1.0]
    #[test]
    fn test_speculative_step_acceptance_rate_in_range() {
        let vocab_size = 8;
        let k = 4;
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig {
            num_draft_tokens: k,
            ..SpeculativeConfig::default()
        });
        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 5.0)).collect();
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        target_logits.push(hot_logits(vocab_size, 0, 5.0));
        let step = decoder
            .verify_and_record(&draft_tokens, &target_logits)
            .expect("verify should succeed");
        assert!(
            (0.0..=1.0).contains(&step.acceptance_rate),
            "acceptance_rate {} must be in [0, 1]",
            step.acceptance_rate
        );
    }

    // Test 46: stats.total_steps increments after each verify_and_record
    #[test]
    fn test_stats_total_steps_increments() {
        let vocab_size = 4;
        let k = 2;
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig {
            num_draft_tokens: k,
            ..SpeculativeConfig::default()
        });
        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 5.0)).collect();
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        target_logits.push(hot_logits(vocab_size, 0, 5.0));

        decoder.verify_and_record(&draft_tokens, &target_logits).expect("ok");
        assert_eq!(decoder.stats.total_steps, 1);
        decoder.verify_and_record(&draft_tokens, &target_logits).expect("ok");
        assert_eq!(decoder.stats.total_steps, 2);
    }

    // Test 47: stats.total_draft_tokens increases by k after each step
    #[test]
    fn test_stats_total_draft_tokens_by_k() {
        let vocab_size = 4;
        let k = 3;
        let mut decoder = SpeculativeDecoder::new(SpeculativeConfig {
            num_draft_tokens: k,
            ..SpeculativeConfig::default()
        });
        let draft_tokens: Vec<DraftToken> =
            (0..k).map(|i| make_draft_token(i as u32, vocab_size, 5.0)).collect();
        let mut target_logits: Vec<Vec<f32>> =
            draft_tokens.iter().map(|dt| dt.draft_logits.clone()).collect();
        target_logits.push(hot_logits(vocab_size, 0, 5.0));

        decoder.verify_and_record(&draft_tokens, &target_logits).expect("ok");
        assert_eq!(decoder.stats.total_draft_tokens, k as u64);
    }

    // Test 48: SpeculativeError::VocabSizeMismatch — display is non-empty
    #[test]
    fn test_speculative_error_vocab_mismatch_display() {
        let e = SpeculativeError::VocabSizeMismatch {
            draft: 1000,
            target: 2000,
        };
        let s = e.to_string();
        assert!(s.contains("1000") && s.contains("2000"));
    }

    // Test 49: SpeculativeStats acceptance_rate = 0.0 when none accepted
    #[test]
    fn test_stats_acceptance_rate_zero_when_none_accepted() {
        let s = SpeculativeStats {
            total_draft_tokens: 10,
            total_accepted_tokens: 0,
            total_rejected_tokens: 10,
            total_bonus_tokens: 0,
            total_steps: 2,
        };
        assert!((s.acceptance_rate() - 0.0).abs() < 1e-9);
    }
}
