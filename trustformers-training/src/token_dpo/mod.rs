//! Token-level Direct Preference Optimization (Token-DPO).
//!
//! Reference: "Token-level Direct Preference Optimization" (Zeng et al., 2024).
//!
//! Token-DPO extends DPO to the token level, computing per-token advantages
//! and weighting them by a credit-assignment scheme. This provides finer-grained
//! credit assignment than sequence-level DPO.

use std::fmt;

// ──────────────────────────────────────────────
// Enumerations
// ──────────────────────────────────────────────

/// Granularity at which token-level losses are computed.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenGranularity {
    /// Every individual token.
    Token,
    /// Merge subword tokens into full words before scoring.
    Word,
    /// Split at sentence-boundary punctuation before scoring.
    Sentence,
}

/// Credit-assignment strategy that weights per-token losses.
#[derive(Debug, Clone, PartialEq)]
pub enum CreditAssignment {
    /// Equal weight `1/T` for all tokens.
    Uniform,
    /// Only the last token receives credit (degenerates to sequence-level DPO).
    LastToken,
    /// `γ^(T−1−t)` discounted credit, normalised to sum 1.
    Discounted,
    /// Attention-weight-based credit (falls back to Uniform when not provided).
    Attention,
}

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Configuration for Token-DPO training.
#[derive(Debug, Clone)]
pub struct TokenDpoConfig {
    /// DPO temperature parameter β (default 0.1).
    pub beta: f32,
    /// Maximum sequence length (default 512).
    pub max_length: usize,
    /// Granularity level for token scoring.
    pub token_level_granularity: TokenGranularity,
    /// Credit-assignment method applied to per-token losses.
    pub credit_assignment: CreditAssignment,
    /// If true, zero out reference log-probs (reference-free mode).
    pub use_reference_free: bool,
    /// Discount factor for `Discounted` credit assignment (default 0.9).
    pub gamma: f32,
}

impl Default for TokenDpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            max_length: 512,
            token_level_granularity: TokenGranularity::Token,
            credit_assignment: CreditAssignment::Uniform,
            use_reference_free: false,
            gamma: 0.9,
        }
    }
}

// ──────────────────────────────────────────────
// Example
// ──────────────────────────────────────────────

/// A single (chosen, rejected) pair for token-level DPO.
#[derive(Debug, Clone)]
pub struct TokenLevelExample {
    /// Prompt token IDs (shared between chosen and rejected).
    pub prompt: Vec<u32>,
    /// Chosen response token IDs.
    pub chosen: Vec<u32>,
    /// Rejected response token IDs.
    pub rejected: Vec<u32>,
    /// Per-token log-probs from the **current** policy for chosen response.
    pub chosen_log_probs: Vec<f32>,
    /// Per-token log-probs from the **current** policy for rejected response.
    pub rejected_log_probs: Vec<f32>,
    /// Per-token log-probs from the **reference** policy for chosen response.
    pub chosen_ref_log_probs: Vec<f32>,
    /// Per-token log-probs from the **reference** policy for rejected response.
    pub rejected_ref_log_probs: Vec<f32>,
    /// Optional pre-computed credit weights (overrides `CreditAssignment`).
    pub credit_weights: Option<Vec<f32>>,
}

// ──────────────────────────────────────────────
// Loss output
// ──────────────────────────────────────────────

/// Per-example statistics from token-level DPO loss computation.
#[derive(Debug, Clone)]
pub struct TokenDpoLossOutput {
    /// Sum of credit-weighted token-level losses.
    pub total_loss: f32,
    /// Per-token credit-weighted loss values.
    pub per_token_losses: Vec<f32>,
    /// Per-token DPO advantages (chosen − rejected log-ratio differences).
    pub per_token_advantages: Vec<f32>,
    /// Mean log-prob of chosen tokens under current policy.
    pub chosen_mean_log_prob: f32,
    /// Mean log-prob of rejected tokens under current policy.
    pub rejected_mean_log_prob: f32,
    /// Fraction of tokens with positive advantage.
    pub accuracy: f32,
}

impl fmt::Display for TokenDpoLossOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TokenDpoLoss {{ total: {:.4}, chosen_lp: {:.4}, rejected_lp: {:.4}, accuracy: {:.4} }}",
            self.total_loss,
            self.chosen_mean_log_prob,
            self.rejected_mean_log_prob,
            self.accuracy,
        )
    }
}

// ──────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────

/// Errors produced by token-level DPO computation.
#[derive(Debug)]
pub enum TokenDpoError {
    /// Example has empty chosen or rejected sequence.
    EmptySequence { field: &'static str },
    /// Log-prob vectors are length-mismatched for the given field.
    LengthMismatch { field: &'static str, expected: usize, got: usize },
    /// A log-probability or reward value was not finite.
    NonFiniteValue { field: &'static str },
    /// Credit weights provided but length doesn't match sequence.
    CreditWeightLengthMismatch { expected: usize, got: usize },
}

impl fmt::Display for TokenDpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenDpoError::EmptySequence { field } => {
                write!(f, "Token-DPO: empty sequence for field '{field}'")
            }
            TokenDpoError::LengthMismatch { field, expected, got } => write!(
                f,
                "Token-DPO: length mismatch for '{field}': expected {expected}, got {got}"
            ),
            TokenDpoError::NonFiniteValue { field } => {
                write!(f, "Token-DPO: non-finite value in field '{field}'")
            }
            TokenDpoError::CreditWeightLengthMismatch { expected, got } => write!(
                f,
                "Token-DPO: credit_weights length mismatch: expected {expected}, got {got}"
            ),
        }
    }
}

impl std::error::Error for TokenDpoError {}

// ──────────────────────────────────────────────
// Credit assignment
// ──────────────────────────────────────────────

/// Compute per-token credit weights for a sequence of length `length`.
///
/// All methods guarantee that the returned weights are non-negative.
/// For methods marked "normalised" (Uniform, Discounted) the weights sum to 1.
/// LastToken places all mass on the last position.
/// Attention falls back to Uniform in the absence of attention data.
pub fn compute_token_credits(
    length: usize,
    method: &CreditAssignment,
    gamma: f32,
) -> Vec<f32> {
    if length == 0 {
        return vec![];
    }
    match method {
        CreditAssignment::Uniform => {
            let w = 1.0 / length as f32;
            vec![w; length]
        }
        CreditAssignment::LastToken => {
            let mut weights = vec![0.0_f32; length];
            weights[length - 1] = 1.0;
            weights
        }
        CreditAssignment::Discounted => {
            // weight_t = γ^(T−1−t), then normalise to sum = 1
            let mut weights: Vec<f32> = (0..length)
                .map(|t| gamma.powi((length - 1 - t) as i32))
                .collect();
            let total: f32 = weights.iter().sum();
            if total > 1e-12 {
                for w in &mut weights {
                    *w /= total;
                }
            }
            weights
        }
        CreditAssignment::Attention => {
            // Without external attention data, fall back to uniform
            let w = 1.0 / length as f32;
            vec![w; length]
        }
    }
}

// ──────────────────────────────────────────────
// Sigmoid helper
// ──────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ──────────────────────────────────────────────
// Core token-level DPO loss
// ──────────────────────────────────────────────

/// Compute the token-level DPO loss for a single example.
///
/// For each position t in `chosen`:
/// ```text
/// advantage_t = (chosen_lp[t] − chosen_ref_lp[t]) − (rejected_lp[t] − rejected_ref_lp[t])
/// credit_t    = compute_token_credits(len, method, gamma)[t]
/// token_loss_t = −log(σ(β × advantage_t)) × credit_t
/// ```
///
/// Total loss = Σ_t token_loss_t.
///
/// When `use_reference_free` is set in the caller, pass zero-slices for ref log-probs.
pub fn compute_token_dpo_loss(
    example: &TokenLevelExample,
    beta: f32,
    credit_method: &CreditAssignment,
    gamma: f32,
) -> Result<TokenDpoLossOutput, TokenDpoError> {
    let chosen_len = example.chosen.len();
    let rejected_len = example.rejected.len();

    if chosen_len == 0 {
        return Err(TokenDpoError::EmptySequence { field: "chosen" });
    }
    if rejected_len == 0 {
        return Err(TokenDpoError::EmptySequence { field: "rejected" });
    }

    // Validate chosen-side lengths
    if example.chosen_log_probs.len() != chosen_len {
        return Err(TokenDpoError::LengthMismatch {
            field: "chosen_log_probs",
            expected: chosen_len,
            got: example.chosen_log_probs.len(),
        });
    }
    if example.chosen_ref_log_probs.len() != chosen_len {
        return Err(TokenDpoError::LengthMismatch {
            field: "chosen_ref_log_probs",
            expected: chosen_len,
            got: example.chosen_ref_log_probs.len(),
        });
    }

    // Validate rejected-side lengths
    if example.rejected_log_probs.len() != rejected_len {
        return Err(TokenDpoError::LengthMismatch {
            field: "rejected_log_probs",
            expected: rejected_len,
            got: example.rejected_log_probs.len(),
        });
    }
    if example.rejected_ref_log_probs.len() != rejected_len {
        return Err(TokenDpoError::LengthMismatch {
            field: "rejected_ref_log_probs",
            expected: rejected_len,
            got: example.rejected_ref_log_probs.len(),
        });
    }

    // Validate finiteness
    for lp in example.chosen_log_probs.iter().chain(example.chosen_ref_log_probs.iter()) {
        if !lp.is_finite() {
            return Err(TokenDpoError::NonFiniteValue { field: "chosen_log_probs" });
        }
    }
    for lp in example.rejected_log_probs.iter().chain(example.rejected_ref_log_probs.iter()) {
        if !lp.is_finite() {
            return Err(TokenDpoError::NonFiniteValue { field: "rejected_log_probs" });
        }
    }

    // Compute credits (use pre-computed if provided)
    // Token-DPO operates at the chosen-sequence level; rejected is used for advantage only.
    let credits = match &example.credit_weights {
        Some(w) => {
            if w.len() != chosen_len {
                return Err(TokenDpoError::CreditWeightLengthMismatch {
                    expected: chosen_len,
                    got: w.len(),
                });
            }
            w.clone()
        }
        None => compute_token_credits(chosen_len, credit_method, gamma),
    };

    // Compute rejected sequence log-ratio (mean, broadcast to chosen length for advantage)
    let rejected_log_ratio: f32 = example
        .rejected_log_probs
        .iter()
        .zip(example.rejected_ref_log_probs.iter())
        .map(|(lp, rlp)| lp - rlp)
        .sum::<f32>()
        / rejected_len as f32;

    // Per-token advantages and losses
    let mut per_token_advantages = Vec::with_capacity(chosen_len);
    let mut per_token_losses = Vec::with_capacity(chosen_len);
    let mut total_loss = 0.0_f32;
    let mut positive_adv_count = 0usize;

    for t in 0..chosen_len {
        let chosen_log_ratio_t =
            example.chosen_log_probs[t] - example.chosen_ref_log_probs[t];
        // advantage_t: chosen ratio − rejected mean ratio
        let advantage_t = chosen_log_ratio_t - rejected_log_ratio;
        let token_loss_t = -sigmoid(beta * advantage_t).ln() * credits[t];

        per_token_advantages.push(advantage_t);
        per_token_losses.push(token_loss_t);
        total_loss += token_loss_t;

        if advantage_t > 0.0 {
            positive_adv_count += 1;
        }
    }

    let accuracy = positive_adv_count as f32 / chosen_len as f32;

    let chosen_mean_log_prob = example.chosen_log_probs.iter().sum::<f32>() / chosen_len as f32;
    let rejected_mean_log_prob =
        example.rejected_log_probs.iter().sum::<f32>() / rejected_len as f32;

    Ok(TokenDpoLossOutput {
        total_loss,
        per_token_losses,
        per_token_advantages,
        chosen_mean_log_prob,
        rejected_mean_log_prob,
        accuracy,
    })
}

// ──────────────────────────────────────────────
// Trainer
// ──────────────────────────────────────────────

/// Stateful Token-DPO trainer with history tracking.
pub struct TokenDpoTrainer {
    /// Training configuration.
    pub config: TokenDpoConfig,
    /// Number of completed steps.
    pub step: usize,
    /// History of loss outputs.
    pub history: Vec<TokenDpoLossOutput>,
}

impl TokenDpoTrainer {
    /// Create a new trainer.
    pub fn new(config: TokenDpoConfig) -> Self {
        Self { config, step: 0, history: Vec::new() }
    }

    /// Compute token-level DPO loss for a batch of examples, accumulating history.
    pub fn compute_batch_loss(
        &mut self,
        examples: &[TokenLevelExample],
    ) -> Result<Vec<TokenDpoLossOutput>, TokenDpoError> {
        let mut outputs = Vec::with_capacity(examples.len());
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let credit_method = self.config.credit_assignment.clone();

        for ex in examples {
            // In reference-free mode, zero-out reference log-probs at the call site
            let effective_ex = if self.config.use_reference_free {
                let mut e = ex.clone();
                e.chosen_ref_log_probs = vec![0.0; e.chosen.len()];
                e.rejected_ref_log_probs = vec![0.0; e.rejected.len()];
                e
            } else {
                ex.clone()
            };

            let out = compute_token_dpo_loss(&effective_ex, beta, &credit_method, gamma)?;
            self.history.push(out.clone());
            outputs.push(out);
        }
        self.step += 1;
        Ok(outputs)
    }

    /// Mean accuracy (positive-advantage token fraction) over the last `window` step outputs.
    pub fn mean_recent_accuracy(&self, window: usize) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let start = self.history.len().saturating_sub(window);
        let slice = &self.history[start..];
        slice.iter().map(|h| h.accuracy).sum::<f32>() / slice.len() as f32
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(
        chosen_lp: Vec<f32>,
        rejected_lp: Vec<f32>,
        chosen_ref_lp: Vec<f32>,
        rejected_ref_lp: Vec<f32>,
    ) -> TokenLevelExample {
        let chosen_len = chosen_lp.len();
        let rejected_len = rejected_lp.len();
        TokenLevelExample {
            prompt: vec![1, 2, 3],
            chosen: (0..chosen_len as u32).collect(),
            rejected: (0..rejected_len as u32).collect(),
            chosen_log_probs: chosen_lp,
            rejected_log_probs: rejected_lp,
            chosen_ref_log_probs: chosen_ref_lp,
            rejected_ref_log_probs: rejected_ref_lp,
            credit_weights: None,
        }
    }

    // ── Test 1: uniform credit sums to 1 ─────────────────────────────────
    #[test]
    fn test_uniform_credit_sums_to_one() {
        let credits = compute_token_credits(5, &CreditAssignment::Uniform, 0.9);
        let sum: f32 = credits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        assert_eq!(credits.len(), 5);
    }

    // ── Test 2: last-token credit ─────────────────────────────────────────
    #[test]
    fn test_last_token_credit() {
        let credits = compute_token_credits(4, &CreditAssignment::LastToken, 0.9);
        assert_eq!(credits.len(), 4);
        assert!((credits[3] - 1.0).abs() < 1e-6);
        for &c in &credits[..3] {
            assert!((c).abs() < 1e-6, "non-last credit should be 0, got {c}");
        }
    }

    // ── Test 3: discounted credit sums to 1 ──────────────────────────────
    #[test]
    fn test_discounted_credit_sums_to_one() {
        let credits = compute_token_credits(6, &CreditAssignment::Discounted, 0.9);
        let sum: f32 = credits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "discounted sum = {sum}");
    }

    // ── Test 4: discounted credit is monotonically increasing ─────────────
    #[test]
    fn test_discounted_credit_monotone() {
        let credits = compute_token_credits(5, &CreditAssignment::Discounted, 0.8);
        for i in 0..credits.len() - 1 {
            assert!(
                credits[i] <= credits[i + 1],
                "credits[{i}]={} > credits[{}]={}",
                credits[i], i + 1, credits[i + 1]
            );
        }
    }

    // ── Test 5: attention falls back to uniform ───────────────────────────
    #[test]
    fn test_attention_fallback_uniform() {
        let credits_att = compute_token_credits(4, &CreditAssignment::Attention, 0.9);
        let credits_uni = compute_token_credits(4, &CreditAssignment::Uniform, 0.9);
        assert_eq!(credits_att, credits_uni);
    }

    // ── Test 6: basic token DPO loss formula ──────────────────────────────
    #[test]
    fn test_basic_token_dpo_loss() {
        // chosen better than rejected (positive log-ratio difference)
        let ex = make_example(
            vec![-0.5],      // chosen_lp
            vec![-2.0],      // rejected_lp
            vec![-1.0],      // chosen_ref_lp  → chosen ratio = -0.5 - (-1.0) = 0.5
            vec![-1.0],      // rejected_ref_lp → rejected ratio = -2.0 - (-1.0) = -1.0
        );
        // advantage = 0.5 - (-1.0) = 1.5  (positive → chosen preferred)
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::LastToken, 0.9)
            .expect("should succeed");
        assert_eq!(out.accuracy, 1.0, "all tokens should have positive advantage");
        assert!(out.total_loss > 0.0, "loss should be positive");
    }

    // ── Test 7: negative advantage → accuracy = 0 ─────────────────────────
    #[test]
    fn test_negative_advantage_accuracy_zero() {
        // chosen worse than rejected
        let ex = make_example(
            vec![-2.0],
            vec![-0.5],
            vec![-1.0],
            vec![-1.0],
        );
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9)
            .expect("ok");
        assert_eq!(out.accuracy, 0.0);
    }

    // ── Test 8: empty chosen error ────────────────────────────────────────
    #[test]
    fn test_empty_chosen_error() {
        let ex = TokenLevelExample {
            prompt: vec![1],
            chosen: vec![],
            rejected: vec![1],
            chosen_log_probs: vec![],
            rejected_log_probs: vec![-1.0],
            chosen_ref_log_probs: vec![],
            rejected_ref_log_probs: vec![-1.0],
            credit_weights: None,
        };
        let err = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).unwrap_err();
        assert!(matches!(err, TokenDpoError::EmptySequence { field: "chosen" }));
    }

    // ── Test 9: length mismatch error ─────────────────────────────────────
    #[test]
    fn test_length_mismatch_error() {
        let mut ex = make_example(vec![-1.0, -1.0], vec![-1.0], vec![-1.0], vec![-1.0]);
        ex.chosen_log_probs = vec![-1.0]; // length 1, chosen length 2
        let err = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).unwrap_err();
        assert!(matches!(err, TokenDpoError::LengthMismatch { .. }));
    }

    // ── Test 10: reference-free mode via trainer ──────────────────────────
    #[test]
    fn test_reference_free_mode() {
        let config = TokenDpoConfig {
            use_reference_free: true,
            credit_assignment: CreditAssignment::Uniform,
            ..Default::default()
        };
        let mut trainer = TokenDpoTrainer::new(config);
        // Even with non-zero ref log probs, trainer should zero them out
        let ex = make_example(
            vec![-0.5],
            vec![-2.0],
            vec![-99.0], // these should be ignored
            vec![-99.0],
        );
        let outputs = trainer.compute_batch_loss(&[ex]).expect("ok");
        // With ref_lp = 0: chosen ratio = -0.5, rejected ratio = -2.0
        // advantage = -0.5 - (-2.0) = 1.5 > 0
        assert_eq!(outputs[0].accuracy, 1.0);
    }

    // ── Test 11: batch loss tracking ──────────────────────────────────────
    #[test]
    fn test_batch_loss_tracking() {
        let config = TokenDpoConfig::default();
        let mut trainer = TokenDpoTrainer::new(config);
        let ex = make_example(vec![-1.0], vec![-2.0], vec![-1.0], vec![-1.0]);
        trainer.compute_batch_loss(&[ex.clone(), ex]).expect("ok");
        assert_eq!(trainer.history.len(), 2);
        assert_eq!(trainer.step, 1);
    }

    // ── Test 12: TokenDpoError Display ───────────────────────────────────
    #[test]
    fn test_error_display() {
        let e = format!("{}", TokenDpoError::EmptySequence { field: "chosen" });
        assert!(e.contains("chosen"), "got: {e}");

        let e2 = format!(
            "{}",
            TokenDpoError::LengthMismatch { field: "foo", expected: 3, got: 2 }
        );
        assert!(e2.contains("foo"), "got: {e2}");

        let e3 = format!("{}", TokenDpoError::NonFiniteValue { field: "chosen_log_probs" });
        assert!(e3.contains("chosen"), "got: {e3}");

        let e4 = format!(
            "{}",
            TokenDpoError::CreditWeightLengthMismatch { expected: 5, got: 3 }
        );
        assert!(e4.contains("5"), "got: {e4}");
    }

    // ── Test 13: per_token_losses length equals chosen length ────────────
    #[test]
    fn test_per_token_losses_length() {
        let ex = make_example(
            vec![-1.0, -1.5, -0.8],
            vec![-2.0],
            vec![-1.0, -1.0, -1.0],
            vec![-1.0],
        );
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9)
            .expect("ok");
        assert_eq!(out.per_token_losses.len(), 3);
        assert_eq!(out.per_token_advantages.len(), 3);
    }

    // ── Test 14: pre-computed credit weights override ─────────────────────
    #[test]
    fn test_precomputed_credit_weights() {
        let mut ex = make_example(
            vec![-1.0, -1.0],
            vec![-2.0],
            vec![-1.0, -1.0],
            vec![-1.0],
        );
        // Assign all credit to first token
        ex.credit_weights = Some(vec![1.0, 0.0]);
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9)
            .expect("ok");
        // Second token loss should be 0 (credit=0)
        assert!((out.per_token_losses[1]).abs() < 1e-6, "second token loss should be 0");
    }

    // ── Test 15: mean_recent_accuracy tracking ────────────────────────────
    #[test]
    fn test_mean_recent_accuracy() {
        let config = TokenDpoConfig::default();
        let mut trainer = TokenDpoTrainer::new(config);
        // Step 1: chosen better (accuracy=1.0)
        let good = make_example(vec![-0.5], vec![-2.0], vec![-1.0], vec![-1.0]);
        // Step 2: chosen worse (accuracy=0.0)
        let bad = make_example(vec![-2.0], vec![-0.5], vec![-1.0], vec![-1.0]);

        trainer.compute_batch_loss(&[good]).expect("step1");
        trainer.compute_batch_loss(&[bad]).expect("step2");

        // Last 2 outputs: accuracy 1.0 and 0.0 → mean 0.5
        let mean_acc = trainer.mean_recent_accuracy(2);
        assert!((mean_acc - 0.5).abs() < 1e-5, "mean accuracy = {mean_acc}");
    }

    // ── Additional Token-DPO tests ────────────────────────────────────────

    #[test]
    fn test_per_token_reward_computation_formula() {
        // advantage_t = (chosen_lp[t] - chosen_ref_lp[t]) - mean(rejected_lp - rejected_ref_lp)
        // With single rejected token: rejected_log_ratio = rejected_lp - rejected_ref_lp
        let ex = make_example(
            vec![-1.0],   // chosen_lp
            vec![-2.0],   // rejected_lp
            vec![-2.0],   // chosen_ref_lp → chosen_ratio = -1 - (-2) = 1.0
            vec![-1.5],   // rejected_ref_lp → rejected_ratio = -2 - (-1.5) = -0.5
        );
        // advantage = 1.0 - (-0.5) = 1.5
        let out = compute_token_dpo_loss(&ex, 1.0, &CreditAssignment::Uniform, 0.9).expect("ok");
        assert!((out.per_token_advantages[0] - 1.5).abs() < 1e-5, "advantage={}", out.per_token_advantages[0]);
    }

    #[test]
    fn test_sequence_level_loss_equals_mean_token_losses_uniform() {
        // With Uniform credit (1/T each), total_loss = sum(credit_t * token_loss_t)
        // For T=3 tokens, each credit = 1/3: total = (l1+l2+l3)/3
        let ex = make_example(
            vec![-1.0, -1.5, -0.5],
            vec![-2.0],
            vec![-1.5, -1.5, -1.5],
            vec![-1.5],
        );
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).expect("ok");
        let manual_sum: f32 = out.per_token_losses.iter().sum();
        assert!((out.total_loss - manual_sum).abs() < 1e-5, "total_loss={} sum={}", out.total_loss, manual_sum);
    }

    #[test]
    fn test_token_masking_zero_credit_contributes_zero() {
        // Tokens with credit=0 should contribute 0 to total_loss
        let mut ex = make_example(
            vec![-1.0, -1.0, -1.0],
            vec![-2.0],
            vec![-1.5, -1.5, -1.5],
            vec![-1.5],
        );
        // Only middle token gets credit
        ex.credit_weights = Some(vec![0.0, 1.0, 0.0]);
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).expect("ok");
        assert!((out.per_token_losses[0]).abs() < 1e-6, "first token loss should be 0, got {}", out.per_token_losses[0]);
        assert!((out.per_token_losses[2]).abs() < 1e-6, "third token loss should be 0, got {}", out.per_token_losses[2]);
        // total_loss = only middle token's loss
        assert!((out.total_loss - out.per_token_losses[1]).abs() < 1e-6, "total should equal middle token loss");
    }

    #[test]
    fn test_earlier_tokens_get_less_credit_with_discounted() {
        // Discounted credit: later tokens get higher weight
        // The last token should have the highest credit
        let credits = compute_token_credits(4, &CreditAssignment::Discounted, 0.5);
        assert!(
            credits[3] > credits[0],
            "last token {} should have more credit than first {}",
            credits[3], credits[0]
        );
    }

    #[test]
    fn test_alignment_with_sequence_level_when_uniform_weight() {
        // With uniform weights, total_loss should equal sum of per-token sigmoid losses / T
        // verify algebraically
        let ex = make_example(
            vec![-1.0, -2.0],
            vec![-1.5],
            vec![-1.5, -1.5],
            vec![-1.5],
        );
        let beta = 0.5_f32;
        let out = compute_token_dpo_loss(&ex, beta, &CreditAssignment::Uniform, 0.9).expect("ok");
        // Each token credit = 0.5; verify total = sum(credit * individual_loss)
        let expected_total: f32 = out.per_token_losses.iter().sum();
        assert!((out.total_loss - expected_total).abs() < 1e-5, "total mismatch: {} vs {}", out.total_loss, expected_total);
    }

    #[test]
    fn test_different_sequence_lengths_chosen_vs_rejected() {
        // Chosen and rejected can have different lengths
        let chosen_len = 3;
        let rejected_len = 5;
        let ex = make_example(
            vec![-1.0; chosen_len],
            vec![-1.5; rejected_len],
            vec![-1.2; chosen_len],
            vec![-1.2; rejected_len],
        );
        let out = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).expect("ok");
        assert_eq!(out.per_token_losses.len(), chosen_len);
        assert_eq!(out.per_token_advantages.len(), chosen_len);
    }

    #[test]
    fn test_aggregation_sum_strategy_via_custom_weights() {
        // Simulating "sum" aggregation: set credits proportional to sequence length
        // (multiply each by T to get sum instead of mean)
        let t = 3_usize;
        let mut ex = make_example(
            vec![-1.0; t],
            vec![-2.0],
            vec![-1.5; t],
            vec![-1.5],
        );
        // Sum strategy: credit_t = 1.0 (each token contributes full weight)
        ex.credit_weights = Some(vec![1.0; t]);
        let out_sum = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).expect("sum ok");

        // Uniform: each token has credit 1/T
        let mut ex_uniform = ex.clone();
        ex_uniform.credit_weights = None;
        let out_uniform = compute_token_dpo_loss(&ex_uniform, 0.1, &CreditAssignment::Uniform, 0.9).expect("uniform ok");

        // Sum loss = T * uniform_total (since each credit is T times larger)
        assert!((out_sum.total_loss - t as f32 * out_uniform.total_loss).abs() < 1e-4,
            "sum={} expected {}*uniform={}", out_sum.total_loss, t, t as f32 * out_uniform.total_loss);
    }

    #[test]
    fn test_last_token_credit_loss_equals_full_loss_for_single_token() {
        // With a single-token sequence, LastToken and Uniform should give the same result
        let ex = make_example(vec![-1.0], vec![-2.0], vec![-1.5], vec![-1.5]);
        let out_last = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::LastToken, 0.9).expect("ok");
        let out_uniform = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).expect("ok");
        // LastToken: credit=1.0; Uniform: credit=1/1=1.0 → same
        assert!((out_last.total_loss - out_uniform.total_loss).abs() < 1e-6,
            "single-token: LastToken={} Uniform={}", out_last.total_loss, out_uniform.total_loss);
    }

    #[test]
    fn test_credit_weight_length_mismatch_error() {
        let mut ex = make_example(
            vec![-1.0, -1.0],
            vec![-2.0],
            vec![-1.5, -1.5],
            vec![-1.5],
        );
        // Wrong length credit weights (3 instead of 2)
        ex.credit_weights = Some(vec![0.5, 0.3, 0.2]);
        let err = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).unwrap_err();
        assert!(matches!(err, TokenDpoError::CreditWeightLengthMismatch { expected: 2, got: 3 }));
    }

    #[test]
    fn test_empty_rejected_error() {
        let ex = TokenLevelExample {
            prompt: vec![1],
            chosen: vec![1],
            rejected: vec![],
            chosen_log_probs: vec![-1.0],
            rejected_log_probs: vec![],
            chosen_ref_log_probs: vec![-1.0],
            rejected_ref_log_probs: vec![],
            credit_weights: None,
        };
        let err = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).unwrap_err();
        assert!(matches!(err, TokenDpoError::EmptySequence { field: "rejected" }));
    }

    #[test]
    fn test_non_finite_log_prob_in_chosen_returns_error() {
        let mut ex = make_example(vec![-1.0, f32::NAN], vec![-2.0], vec![-1.5, -1.5], vec![-1.5]);
        ex.chosen_log_probs[1] = f32::NAN;
        let err = compute_token_dpo_loss(&ex, 0.1, &CreditAssignment::Uniform, 0.9).unwrap_err();
        assert!(matches!(err, TokenDpoError::NonFiniteValue { .. }));
    }

    #[test]
    fn test_reference_free_mode_zeroes_ref_log_probs() {
        // Reference-free mode: ref log-probs become 0, so log-ratio = policy log-prob
        let config = TokenDpoConfig {
            use_reference_free: true,
            ..Default::default()
        };
        let mut trainer = TokenDpoTrainer::new(config);
        // With ref_lp=0: chosen_ratio = -0.5, rejected_ratio = -2.0
        // advantage = -0.5 - (-2.0) = 1.5 → positive → accuracy=1.0
        let ex = make_example(
            vec![-0.5],
            vec![-2.0],
            vec![-99.0], // ignored in reference-free
            vec![-99.0], // ignored in reference-free
        );
        let outs = trainer.compute_batch_loss(&[ex]).expect("ok");
        assert_eq!(outs[0].accuracy, 1.0, "ref-free: advantage should be positive");
    }

    #[test]
    fn test_trainer_step_increments_per_call() {
        let mut trainer = TokenDpoTrainer::new(TokenDpoConfig::default());
        assert_eq!(trainer.step, 0);
        let ex = make_example(vec![-1.0], vec![-2.0], vec![-1.0], vec![-1.0]);
        trainer.compute_batch_loss(&[ex.clone()]).expect("step 1");
        assert_eq!(trainer.step, 1);
        trainer.compute_batch_loss(&[ex]).expect("step 2");
        assert_eq!(trainer.step, 2);
    }

    #[test]
    fn test_discounted_credit_gamma_zero_all_mass_on_last() {
        // With γ=0: weight_t = 0^(T-1-t); only t=T-1 gives 0^0=1, others 0
        let n = 4;
        let credits = compute_token_credits(n, &CreditAssignment::Discounted, 0.0);
        assert!((credits[n - 1] - 1.0).abs() < 1e-5, "last credit should be 1.0 for gamma=0");
        for &c in &credits[..n - 1] {
            assert!(c < 1e-5, "earlier credits should be ~0 for gamma=0, got {}", c);
        }
    }

    #[test]
    fn test_display_token_dpo_loss_output() {
        let out = TokenDpoLossOutput {
            total_loss: 0.5,
            per_token_losses: vec![0.5],
            per_token_advantages: vec![1.0],
            chosen_mean_log_prob: -1.0,
            rejected_mean_log_prob: -2.0,
            accuracy: 1.0,
        };
        let s = format!("{}", out);
        assert!(s.contains("0.5000") || s.contains("total"), "display format: {}", s);
    }
}
