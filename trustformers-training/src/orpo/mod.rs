//! ORPO — Odds Ratio Preference Optimization (2024).
//!
//! ORPO trains the base model and aligns it simultaneously without a reference
//! model, by combining an SFT (NLL) loss with an odds-ratio-based preference
//! penalty in a single objective.
//!
//! Reference: Hong et al., "ORPO: Monolithic Preference Optimization without
//! Reference Model", arXiv:2403.07691, 2024.

use std::fmt;

// ──────────────────────────────────────────────
// Error types
// ──────────────────────────────────────────────

/// Errors that can arise during ORPO loss computation.
#[derive(Debug, thiserror::Error)]
pub enum OrpoError {
    /// The sequence has no tokens.
    #[error("Empty sequence: log_probs slice is empty")]
    EmptySequence,
    /// All token positions are masked (label = -100).
    #[error("All tokens masked (label=-100)")]
    AllMasked,
    /// `log_probs` and `labels` have different lengths.
    #[error("Length mismatch: log_probs={lp} labels={lb}")]
    LengthMismatch { lp: usize, lb: usize },
    /// A numerical instability was detected during computation.
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    /// The configuration contains invalid values.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ──────────────────────────────────────────────
// Loss variant
// ──────────────────────────────────────────────

/// Variant of the ORPO odds-ratio loss term.
#[derive(Debug, Clone, PartialEq)]
pub enum OrpoLossVariant {
    /// Original log odds-ratio loss: `-log σ(log_odds(chosen, rejected))`.
    Original,
    /// Explicit log odds-ratio — identical to `Original`; kept for API clarity.
    LogOddsRatio,
    /// Sigmoid approximation with numerically-stable log-sum-exp form.
    SigmoidApprox,
    /// Margin ORPO: penalises pairs where the margin between rewards is below
    /// `margin`.  `L_OR = -log σ(log_odds - margin)`.
    MarginOrpo { margin: f64 },
}

impl Default for OrpoLossVariant {
    fn default() -> Self {
        OrpoLossVariant::Original
    }
}

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Configuration for ORPO training.
#[derive(Debug, Clone)]
pub struct OrpoConfig {
    /// λ — weight for the odds-ratio penalty (default 0.1).
    pub lambda: f64,
    /// β — temperature applied to log-odds before the sigmoid (default 1.0).
    /// Values > 1 sharpen the preference signal.
    pub beta: f64,
    /// Label-smoothing factor applied to the SFT NLL loss (default 0.0).
    pub label_smoothing: f64,
    /// Loss variant selector.
    pub loss_type: OrpoLossVariant,
    /// Whether the model is reference-free (always `true` for ORPO).
    pub reference_free: bool,
    /// Maximum total sequence length (prompt + completion).
    pub max_length: usize,
    /// Maximum length of the completion portion.
    pub max_completion_length: usize,
}

impl Default for OrpoConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            beta: 1.0,
            label_smoothing: 0.0,
            loss_type: OrpoLossVariant::Original,
            reference_free: true,
            max_length: 1024,
            max_completion_length: 256,
        }
    }
}

impl OrpoConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), OrpoError> {
        if self.lambda < 0.0 {
            return Err(OrpoError::InvalidConfig(format!(
                "lambda must be >= 0, got {}",
                self.lambda
            )));
        }
        if self.beta <= 0.0 {
            return Err(OrpoError::InvalidConfig(format!(
                "beta must be > 0, got {}",
                self.beta
            )));
        }
        if !(0.0..1.0).contains(&self.label_smoothing) {
            return Err(OrpoError::InvalidConfig(format!(
                "label_smoothing must be in [0, 1), got {}",
                self.label_smoothing
            )));
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────
// SequenceLogProbs
// ──────────────────────────────────────────────

/// Per-token log-probabilities for a single sequence.
///
/// Tokens whose label is `-100` are treated as padding and excluded from all
/// aggregation operations.
#[derive(Debug, Clone)]
pub struct SequenceLogProbs {
    /// Per-token log-probabilities (aligned with `labels`).
    pub log_probs: Vec<f64>,
    /// Token IDs; `-100` means "ignore this position".
    pub labels: Vec<i64>,
    /// Total number of tokens (including masked ones).
    pub length: usize,
}

impl SequenceLogProbs {
    /// Construct a new `SequenceLogProbs`.
    pub fn new(log_probs: Vec<f64>, labels: Vec<i64>) -> Self {
        let length = log_probs.len();
        Self { log_probs, labels, length }
    }

    /// Sum of log-probabilities at positions where `label != -100`.
    pub fn sum_log_probs(&self) -> f64 {
        self.log_probs
            .iter()
            .zip(self.labels.iter())
            .filter(|(_, &lbl)| lbl != -100)
            .map(|(&lp, _)| lp)
            .sum()
    }

    /// Mean log-probability at non-masked positions.
    ///
    /// Returns `0.0` when all tokens are masked.
    pub fn mean_log_prob(&self) -> f64 {
        let n = self.num_valid_tokens();
        if n == 0 {
            return 0.0;
        }
        self.sum_log_probs() / n as f64
    }

    /// Number of non-masked tokens (i.e., positions where `label != -100`).
    pub fn num_valid_tokens(&self) -> usize {
        self.labels.iter().filter(|&&lbl| lbl != -100).count()
    }

    /// Log-odds ratio of the chosen sequence over the rejected sequence.
    ///
    /// ```text
    /// p_chosen   = exp(mean_log_prob(chosen))
    /// p_rejected = exp(mean_log_prob(rejected))
    ///
    /// log_odds = log(p_c / (1 - p_c)) - log(p_r / (1 - p_r))
    /// ```
    ///
    /// Probabilities are clamped to `(ε, 1−ε)` to avoid log(0) or log of
    /// negative numbers.
    pub fn log_odds(chosen: &SequenceLogProbs, rejected: &SequenceLogProbs) -> f64 {
        const EPS: f64 = 1e-10;
        let clamp = |p: f64| p.max(EPS).min(1.0 - EPS);

        let pc = clamp(chosen.mean_log_prob().exp());
        let pr = clamp(rejected.mean_log_prob().exp());

        let odds_chosen = pc / (1.0 - pc);
        let odds_rejected = pr / (1.0 - pr);

        odds_chosen.ln() - odds_rejected.ln()
    }
}

// ──────────────────────────────────────────────
// Core free functions
// ──────────────────────────────────────────────

/// Compute the log odds ratio for a chosen vs rejected sequence from raw
/// mean log-probability slices.
///
/// Each slice `chosen_log_probs` / `rejected_log_probs` is a per-token
/// log-probability (already masked / filtered by the caller).
///
/// The odds ratio is:
/// ```text
/// p   = exp(mean(log_probs))        (clamped to (ε, 1-ε))
/// log_odds(chosen)   = log(p_c / (1 - p_c))
/// log_odds(rejected) = log(p_r / (1 - p_r))
/// ratio = log_odds(chosen) - log_odds(rejected)
/// ```
///
/// Returns `Err(OrpoError::EmptyLogProbs)` if either slice is empty.
pub fn compute_log_odds_ratio(
    chosen_log_probs: &[f64],
    rejected_log_probs: &[f64],
) -> Result<f64, OrpoError> {
    if chosen_log_probs.is_empty() {
        return Err(OrpoError::EmptySequence);
    }
    if rejected_log_probs.is_empty() {
        return Err(OrpoError::EmptySequence);
    }

    const EPS: f64 = 1e-10;
    let clamp = |p: f64| p.max(EPS).min(1.0 - EPS);

    let mean_c: f64 = chosen_log_probs.iter().sum::<f64>() / chosen_log_probs.len() as f64;
    let mean_r: f64 = rejected_log_probs.iter().sum::<f64>() / rejected_log_probs.len() as f64;

    let pc = clamp(mean_c.exp());
    let pr = clamp(mean_r.exp());

    // Clamp odds to avoid log(0)
    let odds_c = (pc / (1.0 - pc)).max(EPS);
    let odds_r = (pr / (1.0 - pr)).max(EPS);

    let log_or = odds_c.ln() - odds_r.ln();

    if !log_or.is_finite() {
        return Err(OrpoError::NumericalInstability(format!(
            "log odds ratio is non-finite: pc={pc}, pr={pr}"
        )));
    }

    Ok(log_or)
}

/// Output of `compute_orpo_loss`.
#[derive(Debug, Clone)]
pub struct OrpoLossOutput {
    /// Combined SFT + OR loss.
    pub total_loss: f64,
    /// SFT (NLL) component.
    pub nll_loss: f64,
    /// Odds-ratio penalty component.
    pub or_loss: f64,
    /// Reward for the chosen sequence (mean log prob).
    pub chosen_reward: f64,
    /// Reward for the rejected sequence (mean log prob).
    pub rejected_reward: f64,
    /// Log odds ratio (chosen − rejected in odds space).
    pub odds_ratio: f64,
    /// 1.0 if chosen_reward > rejected_reward, else 0.0.
    pub accuracy: f64,
}

/// Compute the full ORPO loss for a preference pair.
///
/// `L_ORPO = nll_loss + λ × L_OR`
///
/// where:
/// - `nll_loss`  = provided directly (caller computes from chosen sequence)
/// - `L_OR`      = determined by `config.loss_type`
pub fn compute_orpo_loss(
    chosen_log_probs: &[f64],
    rejected_log_probs: &[f64],
    nll_loss: f64,
    config: &OrpoConfig,
) -> Result<OrpoLossOutput, OrpoError> {
    config.validate()?;

    if chosen_log_probs.is_empty() {
        return Err(OrpoError::EmptySequence);
    }
    if rejected_log_probs.is_empty() {
        return Err(OrpoError::EmptySequence);
    }

    let log_or = compute_log_odds_ratio(chosen_log_probs, rejected_log_probs)?;
    let scaled_log_or = config.beta * log_or;

    let or_loss = match &config.loss_type {
        OrpoLossVariant::Original | OrpoLossVariant::LogOddsRatio => {
            // L_OR = -log σ(log_odds)
            stable_log_sigmoid(scaled_log_or)
        }
        OrpoLossVariant::SigmoidApprox => {
            // Numerically stable using softplus: log(1 + exp(-x))
            softplus(-scaled_log_or)
        }
        OrpoLossVariant::MarginOrpo { margin } => {
            // L_OR = -log σ(log_odds - margin)
            stable_log_sigmoid(scaled_log_or - margin)
        }
    };

    let total_loss = nll_loss + config.lambda * or_loss;

    let mean_c = chosen_log_probs.iter().sum::<f64>() / chosen_log_probs.len() as f64;
    let mean_r = rejected_log_probs.iter().sum::<f64>() / rejected_log_probs.len() as f64;

    let accuracy = if mean_c > mean_r { 1.0 } else { 0.0 };

    Ok(OrpoLossOutput {
        total_loss,
        nll_loss,
        or_loss,
        chosen_reward: mean_c,
        rejected_reward: mean_r,
        odds_ratio: log_or,
        accuracy,
    })
}

/// Numerically stable `-log σ(x)` = `log(1 + exp(-x))`.
#[inline]
fn stable_log_sigmoid(x: f64) -> f64 {
    // -log σ(x) = log(1 + exp(-x))  — equivalent to softplus(-x)
    softplus(-x)
}

/// Numerically stable `log(1 + exp(x))` (softplus).
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // asymptotic: log(1+exp(x)) ≈ x for large x
    } else if x < -20.0 {
        x.exp() // asymptotic: log(1+exp(x)) ≈ exp(x) for very negative x
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}

// ──────────────────────────────────────────────
// OrpoStats
// ──────────────────────────────────────────────

/// Statistics aggregated over one training iteration (multiple pairs).
#[derive(Debug, Clone)]
pub struct OrpoStats {
    /// Iteration index (0-based).
    pub iteration: usize,
    /// Mean OR loss component.
    pub mean_or_loss: f64,
    /// Mean NLL loss component.
    pub mean_nll_loss: f64,
    /// Mean preference accuracy (fraction where chosen > rejected).
    pub mean_accuracy: f64,
    /// Mean reward for chosen responses.
    pub mean_chosen_reward: f64,
    /// Mean reward for rejected responses.
    pub mean_rejected_reward: f64,
    /// Mean log odds ratio.
    pub mean_odds_ratio: f64,
}

// ──────────────────────────────────────────────
// OrpoTrainer
// ──────────────────────────────────────────────

/// Stateful ORPO trainer that tracks per-iteration statistics.
pub struct OrpoTrainer {
    config: OrpoConfig,
    stats: Vec<OrpoStats>,
    iteration: usize,
    // Accumulator for the current iteration
    acc_or_loss: f64,
    acc_nll_loss: f64,
    acc_accuracy: f64,
    acc_chosen_reward: f64,
    acc_rejected_reward: f64,
    acc_odds_ratio: f64,
    acc_count: usize,
}

impl OrpoTrainer {
    /// Create a new `OrpoTrainer` with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `OrpoError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: OrpoConfig) -> Result<Self, OrpoError> {
        config.validate()?;
        Ok(Self {
            config,
            stats: Vec::new(),
            iteration: 0,
            acc_or_loss: 0.0,
            acc_nll_loss: 0.0,
            acc_accuracy: 0.0,
            acc_chosen_reward: 0.0,
            acc_rejected_reward: 0.0,
            acc_odds_ratio: 0.0,
            acc_count: 0,
        })
    }

    /// Process one preference pair and return the loss breakdown.
    ///
    /// The caller provides `nll_loss` (negative log-likelihood on the chosen
    /// completion) because the actual forward-pass NLL depends on model
    /// infrastructure that is outside this crate.
    pub fn train_step(
        &mut self,
        chosen_log_probs: &[f64],
        rejected_log_probs: &[f64],
        nll_loss: f64,
    ) -> Result<OrpoLossOutput, OrpoError> {
        let out = compute_orpo_loss(chosen_log_probs, rejected_log_probs, nll_loss, &self.config)?;

        self.acc_or_loss += out.or_loss;
        self.acc_nll_loss += out.nll_loss;
        self.acc_accuracy += out.accuracy;
        self.acc_chosen_reward += out.chosen_reward;
        self.acc_rejected_reward += out.rejected_reward;
        self.acc_odds_ratio += out.odds_ratio;
        self.acc_count += 1;

        Ok(out)
    }

    /// Flush the accumulated statistics for the current iteration and start a
    /// new one.  Returns `None` if no steps have been performed since the last
    /// flush.
    pub fn end_iteration(&mut self) -> Option<OrpoStats> {
        if self.acc_count == 0 {
            return None;
        }
        let n = self.acc_count as f64;
        let stats = OrpoStats {
            iteration: self.iteration,
            mean_or_loss: self.acc_or_loss / n,
            mean_nll_loss: self.acc_nll_loss / n,
            mean_accuracy: self.acc_accuracy / n,
            mean_chosen_reward: self.acc_chosen_reward / n,
            mean_rejected_reward: self.acc_rejected_reward / n,
            mean_odds_ratio: self.acc_odds_ratio / n,
        };
        self.stats.push(stats.clone());
        // Reset accumulators
        self.acc_or_loss = 0.0;
        self.acc_nll_loss = 0.0;
        self.acc_accuracy = 0.0;
        self.acc_chosen_reward = 0.0;
        self.acc_rejected_reward = 0.0;
        self.acc_odds_ratio = 0.0;
        self.acc_count = 0;
        self.iteration += 1;
        Some(stats)
    }

    /// Return a summary (averaged over all recorded iterations) or `None` if
    /// no iterations have been completed yet.
    pub fn stats_summary(&self) -> Option<OrpoStats> {
        if self.stats.is_empty() {
            return None;
        }
        let n = self.stats.len() as f64;
        Some(OrpoStats {
            iteration: self.stats.len().saturating_sub(1),
            mean_or_loss: self.stats.iter().map(|s| s.mean_or_loss).sum::<f64>() / n,
            mean_nll_loss: self.stats.iter().map(|s| s.mean_nll_loss).sum::<f64>() / n,
            mean_accuracy: self.stats.iter().map(|s| s.mean_accuracy).sum::<f64>() / n,
            mean_chosen_reward: self.stats.iter().map(|s| s.mean_chosen_reward).sum::<f64>() / n,
            mean_rejected_reward: self.stats.iter().map(|s| s.mean_rejected_reward).sum::<f64>() / n,
            mean_odds_ratio: self.stats.iter().map(|s| s.mean_odds_ratio).sum::<f64>() / n,
        })
    }

    /// Reference to all stored iteration stats.
    pub fn all_stats(&self) -> &[OrpoStats] {
        &self.stats
    }

    /// Current iteration index (number of completed iterations).
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Access the trainer's configuration.
    pub fn config(&self) -> &OrpoConfig {
        &self.config
    }
}

// ──────────────────────────────────────────────
// Legacy types (kept for API compatibility)
// ──────────────────────────────────────────────

/// Aggregated loss statistics for one ORPO update step (legacy interface).
#[derive(Debug, Clone)]
pub struct OrpoLossResult {
    /// Combined SFT + OR loss.
    pub total_loss: f64,
    /// SFT (NLL) component.
    pub sft_loss: f64,
    /// Odds-ratio penalty component.
    pub or_loss: f64,
    /// log-odds ratio (positive ⟹ chosen preferred).
    pub log_odds_ratio: f64,
    /// Mean log-probability of the chosen sequence.
    pub chosen_mean_log_prob: f64,
    /// Mean log-probability of the rejected sequence.
    pub rejected_mean_log_prob: f64,
    /// 1.0 if chosen_prob > rejected_prob, else 0.0.
    pub accuracy: f64,
}

impl fmt::Display for OrpoLossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrpoLossResult {{ total_loss: {:.4}, sft_loss: {:.4}, or_loss: {:.4}, \
             log_odds_ratio: {:.4}, chosen_mlp: {:.4}, rejected_mlp: {:.4}, accuracy: {:.4} }}",
            self.total_loss,
            self.sft_loss,
            self.or_loss,
            self.log_odds_ratio,
            self.chosen_mean_log_prob,
            self.rejected_mean_log_prob,
            self.accuracy,
        )
    }
}

/// Computes the ORPO training objective for a preference pair (legacy API).
pub struct OrpoLoss {
    config: OrpoConfig,
    step: u64,
}

impl OrpoLoss {
    /// Create a new loss computer from the supplied configuration.
    pub fn new(config: OrpoConfig) -> Self {
        Self { config, step: 0 }
    }

    /// Negative log-likelihood (SFT) loss over non-masked tokens.
    pub fn nll_loss(&self, seq: &SequenceLogProbs) -> f64 {
        let mean_lp = seq.mean_log_prob();
        let raw_nll = -mean_lp;

        if self.config.label_smoothing > 0.0 {
            let ls = self.config.label_smoothing;
            (1.0 - ls) * raw_nll + ls * (raw_nll + 1.0)
        } else {
            raw_nll
        }
    }

    /// Odds-ratio loss component.
    ///
    /// `L_OR = −log(σ(β × log_odds(chosen, rejected)))`
    pub fn or_loss(&self, chosen: &SequenceLogProbs, rejected: &SequenceLogProbs) -> f64 {
        let log_odds = SequenceLogProbs::log_odds(chosen, rejected);
        let scaled = self.config.beta * log_odds;
        stable_log_sigmoid(scaled)
    }

    /// Full ORPO loss for one preference pair.
    ///
    /// `L_ORPO = L_SFT + λ × L_OR`
    pub fn compute_loss(
        &mut self,
        chosen: &SequenceLogProbs,
        rejected: &SequenceLogProbs,
    ) -> Result<OrpoLossResult, OrpoError> {
        // Validate chosen
        if chosen.log_probs.is_empty() {
            return Err(OrpoError::EmptySequence);
        }
        if chosen.log_probs.len() != chosen.labels.len() {
            return Err(OrpoError::LengthMismatch {
                lp: chosen.log_probs.len(),
                lb: chosen.labels.len(),
            });
        }
        if chosen.num_valid_tokens() == 0 {
            return Err(OrpoError::AllMasked);
        }

        // Validate rejected
        if rejected.log_probs.is_empty() {
            return Err(OrpoError::EmptySequence);
        }
        if rejected.log_probs.len() != rejected.labels.len() {
            return Err(OrpoError::LengthMismatch {
                lp: rejected.log_probs.len(),
                lb: rejected.labels.len(),
            });
        }
        if rejected.num_valid_tokens() == 0 {
            return Err(OrpoError::AllMasked);
        }

        let sft_loss = self.nll_loss(chosen);
        let or_loss = self.or_loss(chosen, rejected);
        let total_loss = sft_loss + self.config.lambda * or_loss;

        let log_odds_ratio = SequenceLogProbs::log_odds(chosen, rejected);
        let chosen_mean_log_prob = chosen.mean_log_prob();
        let rejected_mean_log_prob = rejected.mean_log_prob();

        let accuracy = if chosen_mean_log_prob > rejected_mean_log_prob { 1.0 } else { 0.0 };

        self.step += 1;

        Ok(OrpoLossResult {
            total_loss,
            sft_loss,
            or_loss,
            log_odds_ratio,
            chosen_mean_log_prob,
            rejected_mean_log_prob,
            accuracy,
        })
    }

    /// Number of `compute_loss` calls completed so far.
    pub fn step(&self) -> u64 {
        self.step
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: sequence where every label is valid (not -100)
    fn seq(log_probs: Vec<f64>) -> SequenceLogProbs {
        let n = log_probs.len();
        let labels = (0..n as i64).collect();
        SequenceLogProbs::new(log_probs, labels)
    }

    // Helper: sequence with some masked positions
    fn seq_masked(log_probs: Vec<f64>, labels: Vec<i64>) -> SequenceLogProbs {
        SequenceLogProbs::new(log_probs, labels)
    }

    // ── Test 1: sum_log_probs with no masking ─────────────────────────────
    #[test]
    fn test_sum_log_probs_no_mask() {
        let s = seq(vec![-1.0, -2.0, -3.0]);
        let expected = -6.0;
        assert!((s.sum_log_probs() - expected).abs() < 1e-10, "expected {expected}, got {}", s.sum_log_probs());
    }

    // ── Test 2: sum_log_probs with masking ────────────────────────────────
    #[test]
    fn test_sum_log_probs_with_mask() {
        // labels: [0, -100, 2] → only positions 0 and 2 contribute
        let s = seq_masked(vec![-1.0, -99.0, -3.0], vec![0, -100, 2]);
        let expected = -4.0;
        assert!((s.sum_log_probs() - expected).abs() < 1e-10, "expected {expected}, got {}", s.sum_log_probs());
    }

    // ── Test 3: mean_log_prob with no masking ─────────────────────────────
    #[test]
    fn test_mean_log_prob_no_mask() {
        let s = seq(vec![-1.0, -2.0, -3.0]);
        let expected = -2.0;
        assert!((s.mean_log_prob() - expected).abs() < 1e-10, "expected {expected}, got {}", s.mean_log_prob());
    }

    // ── Test 4: num_valid_tokens with masking ─────────────────────────────
    #[test]
    fn test_num_valid_tokens_with_mask() {
        let s = seq_masked(
            vec![-1.0, -2.0, -3.0, -4.0],
            vec![0, -100, 2, -100],
        );
        assert_eq!(s.num_valid_tokens(), 2);
    }

    // ── Test 5: log_odds direction (chosen better → positive) ─────────────
    #[test]
    fn test_log_odds_direction_chosen_better() {
        let chosen = seq(vec![-0.5, -0.5]);   // mean = -0.5
        let rejected = seq(vec![-2.0, -2.0]); // mean = -2.0
        let lo = SequenceLogProbs::log_odds(&chosen, &rejected);
        assert!(lo > 0.0, "log_odds should be positive when chosen is better, got {lo}");
    }

    // ── Test 6: nll_loss basic ────────────────────────────────────────────
    #[test]
    fn test_nll_loss_basic() {
        let config = OrpoConfig { label_smoothing: 0.0, ..Default::default() };
        let loss_fn = OrpoLoss::new(config);
        let s = seq(vec![-1.0, -2.0, -3.0]); // mean = -2.0
        let nll = loss_fn.nll_loss(&s);
        let expected = 2.0; // -mean_log_prob
        assert!((nll - expected).abs() < 1e-10, "expected {expected}, got {nll}");
    }

    // ── Test 7: nll_loss with masking ─────────────────────────────────────
    #[test]
    fn test_nll_loss_with_masking() {
        let config = OrpoConfig { label_smoothing: 0.0, ..Default::default() };
        let loss_fn = OrpoLoss::new(config);
        // only positions 0 and 2 contribute: mean = (-1 + -3)/2 = -2
        let s = seq_masked(vec![-1.0, -99.0, -3.0], vec![0, -100, 2]);
        let nll = loss_fn.nll_loss(&s);
        let expected = 2.0;
        assert!((nll - expected).abs() < 1e-10, "expected {expected}, got {nll}");
    }

    // ── Test 8: or_loss is positive ───────────────────────────────────────
    #[test]
    fn test_or_loss_positive() {
        let config = OrpoConfig::default();
        let loss_fn = OrpoLoss::new(config);
        // sigmoid of any real number is in (0,1), so -log(sigmoid) > 0
        let chosen = seq(vec![-0.5, -0.5]);
        let rejected = seq(vec![-2.0, -2.0]);
        let or_loss = loss_fn.or_loss(&chosen, &rejected);
        assert!(or_loss > 0.0, "or_loss should always be positive, got {or_loss}");
    }

    // ── Test 9: compute_loss structure (total = sft + lambda * or) ────────
    #[test]
    fn test_compute_loss_decomposition() {
        let config = OrpoConfig { lambda: 0.1, label_smoothing: 0.0, ..Default::default() };
        let mut loss_fn = OrpoLoss::new(config);
        let chosen = seq(vec![-0.5, -0.5]);
        let rejected = seq(vec![-2.0, -2.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");

        let expected_total = result.sft_loss + 0.1 * result.or_loss;
        assert!(
            (result.total_loss - expected_total).abs() < 1e-10,
            "total_loss={} but sft+lambda*or={}",
            result.total_loss, expected_total
        );
    }

    // ── Test 10: accuracy = 1.0 when chosen is better ─────────────────────
    #[test]
    fn test_accuracy_one_when_chosen_better() {
        let mut loss_fn = OrpoLoss::new(OrpoConfig::default());
        let chosen = seq(vec![-0.5, -0.5]);   // mean = -0.5
        let rejected = seq(vec![-2.0, -2.0]); // mean = -2.0
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!((result.accuracy - 1.0).abs() < 1e-10, "expected accuracy=1.0, got {}", result.accuracy);
    }

    // ── Test 11: accuracy = 0.0 when rejected is better ───────────────────
    #[test]
    fn test_accuracy_zero_when_rejected_better() {
        let mut loss_fn = OrpoLoss::new(OrpoConfig::default());
        let chosen = seq(vec![-3.0, -3.0]);   // mean = -3.0 (worse)
        let rejected = seq(vec![-0.5, -0.5]); // mean = -0.5 (better)
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!((result.accuracy - 0.0).abs() < 1e-10, "expected accuracy=0.0, got {}", result.accuracy);
    }

    // ── Test 12: lambda=0 → only SFT loss ────────────────────────────────
    #[test]
    fn test_lambda_zero_only_sft() {
        let config = OrpoConfig { lambda: 0.0, label_smoothing: 0.0, ..Default::default() };
        let mut loss_fn = OrpoLoss::new(config);
        let chosen = seq(vec![-1.0, -2.0]);
        let rejected = seq(vec![-3.0, -4.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!(
            (result.total_loss - result.sft_loss).abs() < 1e-10,
            "With lambda=0, total_loss should equal sft_loss"
        );
    }

    // ── Test 13: config defaults ──────────────────────────────────────────
    #[test]
    fn test_config_defaults() {
        let cfg = OrpoConfig::default();
        assert!((cfg.lambda - 0.1).abs() < 1e-10);
        assert!((cfg.beta - 1.0).abs() < 1e-10);
        assert!((cfg.label_smoothing - 0.0).abs() < 1e-10);
        assert_eq!(cfg.max_length, 1024);
        assert_eq!(cfg.max_completion_length, 256);
        assert!(cfg.reference_free);
    }

    // ── Test 14: OrpoLossResult display ───────────────────────────────────
    #[test]
    fn test_orpo_loss_result_display() {
        let result = OrpoLossResult {
            total_loss: 1.2345,
            sft_loss: 1.0,
            or_loss: 2.345,
            log_odds_ratio: 0.5,
            chosen_mean_log_prob: -1.0,
            rejected_mean_log_prob: -2.0,
            accuracy: 1.0,
        };
        let s = format!("{result}");
        assert!(s.contains("total_loss"), "display should contain 'total_loss'");
        assert!(s.contains("OrpoLossResult"), "display should contain struct name");
        assert!(s.contains("1.2345"), "display should contain the total_loss value");
    }

    // ── Test 15a: empty sequence error ────────────────────────────────────
    #[test]
    fn test_empty_sequence_error() {
        let mut loss_fn = OrpoLoss::new(OrpoConfig::default());
        let empty = SequenceLogProbs::new(vec![], vec![]);
        let valid = seq(vec![-1.0]);
        let err = loss_fn.compute_loss(&empty, &valid).unwrap_err();
        assert!(matches!(err, OrpoError::EmptySequence));
    }

    // ── Test 15b: all-masked error ─────────────────────────────────────────
    #[test]
    fn test_all_masked_error() {
        let mut loss_fn = OrpoLoss::new(OrpoConfig::default());
        let all_masked = seq_masked(vec![-1.0, -2.0], vec![-100, -100]);
        let valid = seq(vec![-1.0]);
        let err = loss_fn.compute_loss(&all_masked, &valid).unwrap_err();
        assert!(matches!(err, OrpoError::AllMasked));
    }

    // ── Test 16: compute_log_odds_ratio direction ──────────────────────────
    #[test]
    fn test_compute_log_odds_ratio_positive_when_chosen_better() {
        // chosen mean = -0.5, rejected mean = -2.0 → chosen is better
        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];
        let lor = compute_log_odds_ratio(&chosen, &rejected).expect("should not fail");
        assert!(lor > 0.0, "log odds ratio should be positive when chosen is better, got {lor}");
    }

    // ── Test 17: compute_log_odds_ratio negative when rejected better ──────
    #[test]
    fn test_compute_log_odds_ratio_negative_when_rejected_better() {
        let chosen = vec![-3.0, -3.0];
        let rejected = vec![-0.5, -0.5];
        let lor = compute_log_odds_ratio(&chosen, &rejected).expect("should not fail");
        assert!(lor < 0.0, "log odds ratio should be negative when rejected is better, got {lor}");
    }

    // ── Test 18: compute_log_odds_ratio empty error ────────────────────────
    #[test]
    fn test_compute_log_odds_ratio_empty_error() {
        let err = compute_log_odds_ratio(&[], &[-1.0]).unwrap_err();
        assert!(matches!(err, OrpoError::EmptySequence));
        let err2 = compute_log_odds_ratio(&[-1.0], &[]).unwrap_err();
        assert!(matches!(err2, OrpoError::EmptySequence));
    }

    // ── Test 19: compute_orpo_loss odds ratio > 0 when chosen better ──────
    #[test]
    fn test_compute_orpo_loss_odds_ratio_positive() {
        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];
        let cfg = OrpoConfig::default();
        let out = compute_orpo_loss(&chosen, &rejected, 0.5, &cfg).expect("should succeed");
        assert!(out.odds_ratio > 0.0, "odds_ratio should be > 0 when chosen is better");
    }

    // ── Test 20: compute_orpo_loss total_loss = nll + lambda * or ─────────
    #[test]
    fn test_compute_orpo_loss_decomposition() {
        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];
        let nll = 1.234;
        let cfg = OrpoConfig { lambda: 0.2, ..Default::default() };
        let out = compute_orpo_loss(&chosen, &rejected, nll, &cfg).expect("ok");
        let expected = out.nll_loss + 0.2 * out.or_loss;
        assert!((out.total_loss - expected).abs() < 1e-10,
            "total_loss={} expected={}", out.total_loss, expected);
    }

    // ── Test 21: OrpoLossVariant::MarginOrpo increases loss ───────────────
    #[test]
    fn test_margin_orpo_increases_loss() {
        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];
        let cfg_no_margin = OrpoConfig {
            loss_type: OrpoLossVariant::Original,
            ..Default::default()
        };
        let cfg_with_margin = OrpoConfig {
            loss_type: OrpoLossVariant::MarginOrpo { margin: 1.0 },
            ..Default::default()
        };
        let out_no = compute_orpo_loss(&chosen, &rejected, 1.0, &cfg_no_margin).expect("ok");
        let out_m = compute_orpo_loss(&chosen, &rejected, 1.0, &cfg_with_margin).expect("ok");
        // Margin shifts the sigmoid input down → larger -log σ → larger or_loss
        assert!(out_m.or_loss >= out_no.or_loss,
            "margin orpo or_loss {} should be >= original {}", out_m.or_loss, out_no.or_loss);
    }

    // ── Test 22: SigmoidApprox variant is numerically close to Original ───
    #[test]
    fn test_sigmoid_approx_close_to_original() {
        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];
        let cfg_orig = OrpoConfig { loss_type: OrpoLossVariant::Original, ..Default::default() };
        let cfg_approx = OrpoConfig { loss_type: OrpoLossVariant::SigmoidApprox, ..Default::default() };
        let out_orig = compute_orpo_loss(&chosen, &rejected, 1.0, &cfg_orig).expect("ok");
        let out_approx = compute_orpo_loss(&chosen, &rejected, 1.0, &cfg_approx).expect("ok");
        // Both forms should agree closely since they implement the same function
        assert!((out_orig.or_loss - out_approx.or_loss).abs() < 1e-8,
            "Original {} vs SigmoidApprox {}", out_orig.or_loss, out_approx.or_loss);
    }

    // ── Test 23: OrpoTrainer accumulates stats correctly ──────────────────
    #[test]
    fn test_orpo_trainer_accumulates_stats() {
        let cfg = OrpoConfig::default();
        let mut trainer = OrpoTrainer::new(cfg).expect("valid config");

        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];

        trainer.train_step(&chosen, &rejected, 1.0).expect("step 1 ok");
        trainer.train_step(&chosen, &rejected, 0.8).expect("step 2 ok");

        let stats = trainer.end_iteration().expect("should have stats");
        assert_eq!(stats.iteration, 0);
        // accuracy = 1.0 for both steps
        assert!((stats.mean_accuracy - 1.0).abs() < 1e-10);
        // mean nll = (1.0 + 0.8) / 2 = 0.9
        assert!((stats.mean_nll_loss - 0.9).abs() < 1e-10);
    }

    // ── Test 24: OrpoTrainer end_iteration returns None with no steps ─────
    #[test]
    fn test_orpo_trainer_end_iteration_no_steps() {
        let cfg = OrpoConfig::default();
        let mut trainer = OrpoTrainer::new(cfg).expect("valid config");
        assert!(trainer.end_iteration().is_none());
    }

    // ── Test 25: OrpoTrainer stats_summary averages multiple iterations ───
    #[test]
    fn test_orpo_trainer_stats_summary() {
        let cfg = OrpoConfig::default();
        let mut trainer = OrpoTrainer::new(cfg).expect("valid config");

        let chosen = vec![-0.5, -0.5];
        let rejected = vec![-2.0, -2.0];

        // Iteration 0
        trainer.train_step(&chosen, &rejected, 2.0).expect("ok");
        trainer.end_iteration();

        // Iteration 1
        trainer.train_step(&chosen, &rejected, 0.0).expect("ok");
        trainer.end_iteration();

        let summary = trainer.stats_summary().expect("should have summary");
        // Both iterations have accuracy=1.0
        assert!((summary.mean_accuracy - 1.0).abs() < 1e-10);
        // mean_nll = (2.0 + 0.0) / 2 = 1.0 over iterations
        assert!((summary.mean_nll_loss - 1.0).abs() < 1e-10);
    }

    // ── Test 26: config validate rejects bad lambda ────────────────────────
    #[test]
    fn test_config_validate_bad_lambda() {
        let cfg = OrpoConfig { lambda: -0.1, ..Default::default() };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, OrpoError::InvalidConfig(_)));
    }

    // ── Test 27: config validate rejects bad beta ──────────────────────────
    #[test]
    fn test_config_validate_bad_beta() {
        let cfg = OrpoConfig { beta: 0.0, ..Default::default() };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, OrpoError::InvalidConfig(_)));
    }

    // ── Test 28: OrpoTrainer new rejects invalid config ───────────────────
    #[test]
    fn test_orpo_trainer_rejects_invalid_config() {
        let cfg = OrpoConfig { lambda: -1.0, ..Default::default() };
        let result = OrpoTrainer::new(cfg);
        assert!(result.is_err(), "should reject invalid config");
    }

    // ── Test 29: softplus numerical stability for large values ────────────
    #[test]
    fn test_softplus_large_values() {
        // For large x, softplus(x) ≈ x
        let val = softplus(100.0);
        assert!((val - 100.0).abs() < 1.0, "softplus(100) should be ~100, got {val}");
        // For very negative x, softplus(x) ≈ 0
        let val_neg = softplus(-100.0);
        assert!(val_neg < 1e-10, "softplus(-100) should be ~0, got {val_neg}");
    }

    // ── Test 30: log_odds_ratio symmetry ──────────────────────────────────
    #[test]
    fn test_log_odds_ratio_symmetry() {
        // Swapping chosen and rejected negates the log odds ratio
        let a = vec![-0.5, -0.5];
        let b = vec![-2.0, -2.0];
        let lor_ab = compute_log_odds_ratio(&a, &b).expect("ok");
        let lor_ba = compute_log_odds_ratio(&b, &a).expect("ok");
        assert!((lor_ab + lor_ba).abs() < 1e-10,
            "log_odds(a,b) + log_odds(b,a) should be 0, got {lor_ab} + {lor_ba}");
    }
}
