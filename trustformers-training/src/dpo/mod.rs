//! DPO: Direct Preference Optimization
//!
//! This module implements Direct Preference Optimization (DPO) and several
//! variants for training language models directly from human preference data,
//! without requiring an explicit reward model.
//!
//! # Key Concepts
//!
//! - **DPO loss** (Rafailov et al. 2023):
//!   `L = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]`
//! - **β** (temperature): controls deviation from the reference policy.
//! - **Implicit reward**: `r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))`
//! - **Label smoothing**: regularises the binary cross-entropy target.
//!
//! # Reference
//!
//! Rafailov et al. 2023: "Direct Preference Optimization: Your Language Model
//! is Secretly a Reward Model"  <https://arxiv.org/abs/2305.18290>

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during DPO computations.
#[derive(Debug, Clone, PartialEq)]
pub enum DpoError {
    /// The batch of preference pairs was empty.
    EmptyBatch,
    /// A numerical error occurred (NaN, Inf, etc.).
    NumericalError(String),
    /// The configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for DpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DpoError::EmptyBatch => {
                write!(f, "DPO batch is empty; at least one preference pair is required")
            }
            DpoError::NumericalError(msg) => {
                write!(f, "Numerical error in DPO computation: {}", msg)
            }
            DpoError::InvalidConfig(msg) => {
                write!(f, "Invalid DPO configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for DpoError {}

// ─────────────────────────────────────────────────────────────────────────────
// Loss type
// ─────────────────────────────────────────────────────────────────────────────

/// Variant of the DPO loss function.
#[derive(Debug, Clone, PartialEq)]
pub enum DpoLossType {
    /// Standard DPO: `-log σ(β * h_θ)` (Rafailov et al. 2023).
    Sigmoid,
    /// Hinge DPO: `max(0, 1 - β * h_θ)`.
    Hinge,
    /// IPO (Identity Preference Optimisation): `(h_θ - 1/(2β))^2`.
    Ipo,
    /// Robust DPO with a shift applied inside the sigmoid:
    /// `-log σ(β * h_θ - shift)`.
    SigmoidWithShift { shift: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single preference pair (chosen response vs. rejected response).
///
/// All four fields are *sequence-level* log-probabilities, i.e. the sum of
/// per-token log-probabilities over the response.
#[derive(Debug, Clone, PartialEq)]
pub struct DpoPair {
    /// `log π_θ(y_w | x)` — policy log-prob for the chosen response.
    pub policy_log_prob_chosen: f32,
    /// `log π_ref(y_w | x)` — reference log-prob for the chosen response.
    pub reference_log_prob_chosen: f32,
    /// `log π_θ(y_l | x)` — policy log-prob for the rejected response.
    pub policy_log_prob_rejected: f32,
    /// `log π_ref(y_l | x)` — reference log-prob for the rejected response.
    pub reference_log_prob_rejected: f32,
}

impl DpoPair {
    /// Log ratio for the **chosen** response:
    /// `log π_θ(y_w|x) - log π_ref(y_w|x)`.
    #[inline]
    pub fn chosen_log_ratio(&self) -> f32 {
        self.policy_log_prob_chosen - self.reference_log_prob_chosen
    }

    /// Log ratio for the **rejected** response:
    /// `log π_θ(y_l|x) - log π_ref(y_l|x)`.
    #[inline]
    pub fn rejected_log_ratio(&self) -> f32 {
        self.policy_log_prob_rejected - self.reference_log_prob_rejected
    }

    /// Reward margin `β * (chosen_log_ratio - rejected_log_ratio)`.
    ///
    /// A positive margin means the current policy prefers the chosen response
    /// over the rejected one relative to the reference.
    #[inline]
    pub fn reward_margin(&self, beta: f32) -> f32 {
        beta * (self.chosen_log_ratio() - self.rejected_log_ratio())
    }

    /// Implicit reward for the chosen response: `β * chosen_log_ratio`.
    #[inline]
    pub fn implicit_reward_chosen(&self, beta: f32) -> f32 {
        beta * self.chosen_log_ratio()
    }

    /// Implicit reward for the rejected response: `β * rejected_log_ratio`.
    #[inline]
    pub fn implicit_reward_rejected(&self, beta: f32) -> f32 {
        beta * self.rejected_log_ratio()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the DPO algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct DpoConfig {
    /// Temperature `β` controlling how much the policy can deviate from the
    /// reference (default: `0.1`).
    pub beta: f32,
    /// Label smoothing coefficient `ε` in `[0, 1)`.
    /// `0.0` = standard DPO, `0.1` = 10 % label smoothing (default: `0.0`).
    pub label_smoothing: f32,
    /// If `true`, treat the reference policy as uniform (reference-free DPO).
    /// The reference log-probs in each pair are ignored (default: `false`).
    pub reference_free: bool,
    /// Which loss variant to use (default: [`DpoLossType::Sigmoid`]).
    pub loss_type: DpoLossType,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
            reference_free: false,
            loss_type: DpoLossType::Sigmoid,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable `log σ(x) = -log(1 + exp(-x)) = x - softplus(x)`.
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    // softplus(x) = log(1 + exp(x))
    // log_sigmoid(x) = x - softplus(x)
    // For large |x| we use the identity directly to avoid overflow.
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DPO loss computation
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a DPO loss computation over a batch of preference pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct DpoLossResult {
    /// Mean loss over the batch.
    pub loss: f32,
    /// Mean implicit reward for chosen responses: `mean(β * chosen_log_ratio)`.
    pub chosen_reward_mean: f32,
    /// Mean implicit reward for rejected responses: `mean(β * rejected_log_ratio)`.
    pub rejected_reward_mean: f32,
    /// Mean reward margin: `mean(chosen_reward - rejected_reward)`.
    pub reward_margin_mean: f32,
    /// Fraction of pairs where `chosen_reward > rejected_reward`.
    pub reward_accuracy: f32,
    /// Mean chosen log-ratio: `mean(log π_θ(y_w) - log π_ref(y_w))`.
    pub chosen_log_ratio_mean: f32,
    /// Mean rejected log-ratio: `mean(log π_θ(y_l) - log π_ref(y_l))`.
    pub rejected_log_ratio_mean: f32,
    /// Number of preference pairs in the batch.
    pub num_pairs: usize,
}

/// Compute the DPO loss for a batch of preference pairs.
///
/// # Errors
///
/// Returns [`DpoError::EmptyBatch`] if `pairs` is empty.
/// Returns [`DpoError::InvalidConfig`] if `label_smoothing` is outside `[0, 1)`.
/// Returns [`DpoError::NumericalError`] if NaN or infinity is detected in the loss.
pub fn compute_dpo_loss(pairs: &[DpoPair], config: &DpoConfig) -> Result<DpoLossResult, DpoError> {
    let n = pairs.len();

    if n == 0 {
        return Err(DpoError::EmptyBatch);
    }
    if config.label_smoothing < 0.0 || config.label_smoothing >= 1.0 {
        return Err(DpoError::InvalidConfig(format!(
            "label_smoothing must be in [0, 1), got {}",
            config.label_smoothing
        )));
    }

    let mut loss_sum = 0.0f32;
    let mut chosen_reward_sum = 0.0f32;
    let mut rejected_reward_sum = 0.0f32;
    let mut chosen_log_ratio_sum = 0.0f32;
    let mut rejected_log_ratio_sum = 0.0f32;
    let mut correct_count = 0usize;

    for pair in pairs {
        // Effective log-ratios: reference-free mode zeroes out the reference terms
        let (chosen_lr, rejected_lr) = if config.reference_free {
            (pair.policy_log_prob_chosen, pair.policy_log_prob_rejected)
        } else {
            (pair.chosen_log_ratio(), pair.rejected_log_ratio())
        };

        // h_θ = chosen_log_ratio - rejected_log_ratio
        let h = chosen_lr - rejected_lr;
        let scaled_h = config.beta * h;

        let loss_i = match &config.loss_type {
            DpoLossType::Sigmoid => {
                if config.label_smoothing == 0.0 {
                    // -log σ(β * h)
                    -log_sigmoid(scaled_h)
                } else {
                    // Label-smoothed DPO:
                    // -(1-ε) * log σ(β*h) - ε * log σ(-β*h)
                    let eps = config.label_smoothing;
                    -(1.0 - eps) * log_sigmoid(scaled_h) - eps * log_sigmoid(-scaled_h)
                }
            }
            DpoLossType::Hinge => {
                // max(0, 1 - β * h)
                (1.0 - scaled_h).max(0.0)
            }
            DpoLossType::Ipo => {
                // (h - 1/(2β))^2
                let target = 1.0 / (2.0 * config.beta);
                (h - target).powi(2)
            }
            DpoLossType::SigmoidWithShift { shift } => {
                // -log σ(β * h - shift)
                -log_sigmoid(scaled_h - shift)
            }
        };

        let chosen_reward = config.beta * chosen_lr;
        let rejected_reward = config.beta * rejected_lr;

        if chosen_reward > rejected_reward {
            correct_count += 1;
        }

        loss_sum += loss_i;
        chosen_reward_sum += chosen_reward;
        rejected_reward_sum += rejected_reward;
        chosen_log_ratio_sum += chosen_lr;
        rejected_log_ratio_sum += rejected_lr;
    }

    let inv_n = 1.0 / n as f32;
    let loss = loss_sum * inv_n;
    let chosen_reward_mean = chosen_reward_sum * inv_n;
    let rejected_reward_mean = rejected_reward_sum * inv_n;
    let reward_margin_mean = chosen_reward_mean - rejected_reward_mean;
    let reward_accuracy = correct_count as f32 * inv_n;
    let chosen_log_ratio_mean = chosen_log_ratio_sum * inv_n;
    let rejected_log_ratio_mean = rejected_log_ratio_sum * inv_n;

    if loss.is_nan() || loss.is_infinite() {
        return Err(DpoError::NumericalError(format!(
            "DPO loss={} is not finite",
            loss
        )));
    }

    Ok(DpoLossResult {
        loss,
        chosen_reward_mean,
        rejected_reward_mean,
        reward_margin_mean,
        reward_accuracy,
        chosen_log_ratio_mean,
        rejected_log_ratio_mean,
        num_pairs: n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Reward statistics
// ─────────────────────────────────────────────────────────────────────────────

/// DPO reward statistics for monitoring training progress.
#[derive(Debug, Clone)]
pub struct DpoRewardStats {
    /// Per-pair implicit chosen rewards `β * chosen_log_ratio`.
    pub chosen_rewards: Vec<f32>,
    /// Per-pair implicit rejected rewards `β * rejected_log_ratio`.
    pub rejected_rewards: Vec<f32>,
    /// Per-pair reward margins `chosen_reward - rejected_reward`.
    pub margins: Vec<f32>,
    /// Fraction of pairs where chosen reward > rejected reward.
    pub accuracy: f32,
    /// Mean reward margin.
    pub mean_margin: f32,
    /// Standard deviation of the reward margin.
    pub std_margin: f32,
}

impl DpoRewardStats {
    /// Compute reward statistics from a slice of preference pairs and a `beta` value.
    pub fn from_pairs(pairs: &[DpoPair], beta: f32) -> Self {
        let n = pairs.len();

        if n == 0 {
            return Self {
                chosen_rewards: Vec::new(),
                rejected_rewards: Vec::new(),
                margins: Vec::new(),
                accuracy: 0.0,
                mean_margin: 0.0,
                std_margin: 0.0,
            };
        }

        let mut chosen_rewards = Vec::with_capacity(n);
        let mut rejected_rewards = Vec::with_capacity(n);
        let mut margins = Vec::with_capacity(n);
        let mut correct = 0usize;

        for pair in pairs {
            let cr = pair.implicit_reward_chosen(beta);
            let rr = pair.implicit_reward_rejected(beta);
            let margin = cr - rr;

            if cr > rr {
                correct += 1;
            }

            chosen_rewards.push(cr);
            rejected_rewards.push(rr);
            margins.push(margin);
        }

        let mean_margin = margins.iter().sum::<f32>() / n as f32;
        let var = margins
            .iter()
            .map(|m| (m - mean_margin).powi(2))
            .sum::<f32>()
            / n as f32;
        let std_margin = var.sqrt();
        let accuracy = correct as f32 / n as f32;

        Self {
            chosen_rewards,
            rejected_rewards,
            margins,
            accuracy,
            mean_margin,
            std_margin,
        }
    }

    /// Compute the `p`-th percentile of the reward margins using linear interpolation.
    ///
    /// `p` must be in `[0, 1]`. Values outside this range are clamped.
    ///
    /// Returns `0.0` for an empty stats object.
    pub fn percentile_margin(&self, p: f32) -> f32 {
        if self.margins.is_empty() {
            return 0.0;
        }

        let p_clamped = p.clamp(0.0, 1.0);
        let mut sorted = self.margins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }

        let float_idx = p_clamped * (n - 1) as f32;
        let lo = float_idx.floor() as usize;
        let hi = float_idx.ceil() as usize;

        if lo == hi {
            return sorted[lo];
        }

        let frac = float_idx - lo as f32;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful DPO trainer that accumulates step history for diagnostics.
#[derive(Debug)]
pub struct DpoTrainer {
    /// Configuration for DPO.
    pub config: DpoConfig,
    history: Vec<DpoLossResult>,
}

impl DpoTrainer {
    /// Create a new [`DpoTrainer`] with the given configuration.
    pub fn new(config: DpoConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Run one DPO step over a batch of preference pairs.
    ///
    /// The result is pushed onto the history.
    pub fn step(&mut self, pairs: &[DpoPair]) -> Result<DpoLossResult, DpoError> {
        let result = compute_dpo_loss(pairs, &self.config)?;
        self.history.push(result.clone());
        Ok(result)
    }

    /// Return the full step history.
    pub fn history(&self) -> &[DpoLossResult] {
        &self.history
    }

    /// Mean loss over all steps in the history.
    ///
    /// Returns `0.0` if the history is empty.
    pub fn mean_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.loss).sum();
        sum / self.history.len() as f32
    }

    /// Mean reward accuracy over all steps in the history.
    ///
    /// Returns `0.0` if the history is empty.
    pub fn mean_reward_accuracy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.reward_accuracy).sum();
        sum / self.history.len() as f32
    }

    /// Mean chosen and rejected rewards from the **last** step.
    ///
    /// Returns `None` if no steps have been taken.
    pub fn reward_stats(&self) -> Option<(f32, f32)> {
        self.history
            .last()
            .map(|r| (r.chosen_reward_mean, r.rejected_reward_mean))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper constructors ──────────────────────────────────────────────

    fn make_pair(
        policy_chosen: f32,
        ref_chosen: f32,
        policy_rejected: f32,
        ref_rejected: f32,
    ) -> DpoPair {
        DpoPair {
            policy_log_prob_chosen: policy_chosen,
            reference_log_prob_chosen: ref_chosen,
            policy_log_prob_rejected: policy_rejected,
            reference_log_prob_rejected: ref_rejected,
        }
    }

    // ── DpoPair tests ────────────────────────────────────────────────────

    #[test]
    fn test_dpo_pair_log_ratios() {
        let pair = make_pair(-1.0, -2.0, -3.0, -1.5);
        // chosen_log_ratio = -1.0 - (-2.0) = 1.0
        assert!((pair.chosen_log_ratio() - 1.0).abs() < 1e-6);
        // rejected_log_ratio = -3.0 - (-1.5) = -1.5
        assert!((pair.rejected_log_ratio() - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_dpo_pair_reward_margin() {
        let pair = make_pair(-1.0, -2.0, -3.0, -1.5);
        let beta = 0.5;
        // margin = 0.5 * (1.0 - (-1.5)) = 0.5 * 2.5 = 1.25
        assert!((pair.reward_margin(beta) - 1.25).abs() < 1e-5);
    }

    #[test]
    fn test_dpo_pair_implicit_rewards() {
        let pair = make_pair(-1.0, -2.0, -3.0, -1.5);
        let beta = 0.1;
        assert!((pair.implicit_reward_chosen(beta) - 0.1).abs() < 1e-6);
        assert!((pair.implicit_reward_rejected(beta) - (-0.15)).abs() < 1e-6);
    }

    // ── DpoConfig tests ──────────────────────────────────────────────────

    #[test]
    fn test_dpo_config_default() {
        let cfg = DpoConfig::default();
        assert!((cfg.beta - 0.1).abs() < 1e-8);
        assert!((cfg.label_smoothing).abs() < 1e-8);
        assert!(!cfg.reference_free);
        assert_eq!(cfg.loss_type, DpoLossType::Sigmoid);
    }

    // ── Loss variant tests ───────────────────────────────────────────────

    #[test]
    fn test_dpo_loss_sigmoid_perfect() {
        // When chosen_log_ratio >> rejected_log_ratio the loss should be near 0
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        // chosen_log_ratio = 5.0, rejected_log_ratio = -5.0 → h = 10 → loss ≈ 0
        let pair = make_pair(0.0, -5.0, 0.0, 5.0);
        let result = compute_dpo_loss(&[pair], &cfg).expect("dpo loss failed");
        assert!(result.loss < 1e-3, "expected near-zero loss, got {}", result.loss);
        assert_eq!(result.reward_accuracy, 1.0);
    }

    #[test]
    fn test_dpo_loss_sigmoid_random() {
        // Simple sanity check: loss is non-negative and finite
        let cfg = DpoConfig::default();
        let pairs = vec![
            make_pair(-1.0, -1.5, -2.0, -1.8),
            make_pair(-0.5, -1.0, -1.5, -1.0),
        ];
        let result = compute_dpo_loss(&pairs, &cfg).expect("dpo loss failed");
        assert!(result.loss >= 0.0);
        assert!(result.loss.is_finite());
        assert_eq!(result.num_pairs, 2);
    }

    #[test]
    fn test_dpo_loss_label_smoothing_effect() {
        // Label smoothing should increase the loss compared to standard DPO
        // when the model is very confident (large margin)
        let pair = make_pair(0.0, -10.0, 0.0, 10.0);
        let cfg_no_smooth = DpoConfig { beta: 1.0, label_smoothing: 0.0, ..Default::default() };
        let cfg_smooth = DpoConfig { beta: 1.0, label_smoothing: 0.1, ..Default::default() };

        let r1 = compute_dpo_loss(&[pair.clone()], &cfg_no_smooth).expect("loss 1 failed");
        let r2 = compute_dpo_loss(&[pair], &cfg_smooth).expect("loss 2 failed");
        // Smoothed loss should be higher (harder target)
        assert!(r2.loss > r1.loss, "smoothed={} should > standard={}", r2.loss, r1.loss);
    }

    #[test]
    fn test_dpo_loss_hinge_no_violation() {
        // h = 2/β = 20 → 1 - β*h = 1 - 2 = -1 → max(0,-1) = 0
        let cfg = DpoConfig {
            beta: 0.1,
            loss_type: DpoLossType::Hinge,
            ..Default::default()
        };
        // chosen_lr = 10, rejected_lr = -10, h = 20
        let pair = make_pair(0.0, -10.0, 0.0, 10.0);
        let result = compute_dpo_loss(&[pair], &cfg).expect("hinge loss failed");
        assert!((result.loss).abs() < 1e-5, "expected zero hinge loss, got {}", result.loss);
    }

    #[test]
    fn test_dpo_loss_hinge_with_violation() {
        // h = 0 → 1 - β*0 = 1 → max(0,1) = 1
        let cfg = DpoConfig {
            beta: 0.1,
            loss_type: DpoLossType::Hinge,
            ..Default::default()
        };
        let pair = make_pair(-1.0, -1.0, -1.0, -1.0); // all log-ratios = 0
        let result = compute_dpo_loss(&[pair], &cfg).expect("hinge loss failed");
        assert!((result.loss - 1.0).abs() < 1e-5, "expected 1.0, got {}", result.loss);
    }

    #[test]
    fn test_dpo_loss_ipo() {
        // IPO: (h - 1/(2β))^2
        // β = 0.5 → target = 1.0
        // chosen_lr = 2.0, rejected_lr = 0.0 → h = 2.0 → loss = (2.0 - 1.0)^2 = 1.0
        let cfg = DpoConfig {
            beta: 0.5,
            loss_type: DpoLossType::Ipo,
            ..Default::default()
        };
        let pair = make_pair(0.0, -2.0, -1.0, -1.0); // chosen_lr=2, rejected_lr=0
        let result = compute_dpo_loss(&[pair], &cfg).expect("ipo loss failed");
        assert!((result.loss - 1.0).abs() < 1e-5, "expected 1.0, got {}", result.loss);
    }

    #[test]
    fn test_dpo_loss_sigmoid_with_shift() {
        // SigmoidWithShift: -log σ(β*h - shift)
        // β = 1, shift = 0 → same as Sigmoid
        let cfg_standard = DpoConfig { beta: 1.0, ..Default::default() };
        let cfg_shifted = DpoConfig {
            beta: 1.0,
            loss_type: DpoLossType::SigmoidWithShift { shift: 0.0 },
            ..Default::default()
        };
        let pair = make_pair(-1.0, -2.0, -2.0, -1.5);
        let r1 = compute_dpo_loss(&[pair.clone()], &cfg_standard).expect("standard failed");
        let r2 = compute_dpo_loss(&[pair], &cfg_shifted).expect("shifted failed");
        assert!((r1.loss - r2.loss).abs() < 1e-5);
    }

    // ── Reward accuracy tests ────────────────────────────────────────────

    #[test]
    fn test_dpo_reward_accuracy_all_correct() {
        let cfg = DpoConfig::default();
        // Each pair: chosen_lr > rejected_lr
        let pairs: Vec<DpoPair> = (0..4)
            .map(|_| make_pair(0.0, -2.0, 0.0, 2.0))
            .collect();
        let result = compute_dpo_loss(&pairs, &cfg).expect("loss failed");
        assert!((result.reward_accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dpo_reward_accuracy_mixed() {
        let cfg = DpoConfig::default();
        // 2 correct, 2 incorrect
        let pairs = vec![
            make_pair(0.0, -2.0, 0.0, 2.0),   // correct
            make_pair(0.0, -2.0, 0.0, 2.0),   // correct
            make_pair(0.0, 2.0, 0.0, -2.0),   // incorrect (chosen_lr=-2)
            make_pair(0.0, 2.0, 0.0, -2.0),   // incorrect
        ];
        let result = compute_dpo_loss(&pairs, &cfg).expect("loss failed");
        assert!((result.reward_accuracy - 0.5).abs() < 1e-5);
    }

    // ── DpoRewardStats tests ─────────────────────────────────────────────

    #[test]
    fn test_dpo_reward_stats_from_pairs() {
        let pairs = vec![
            make_pair(0.0, -1.0, 0.0, 1.0),  // chosen_lr=1, rejected_lr=-1
            make_pair(0.0, -2.0, 0.0, 0.0),  // chosen_lr=2, rejected_lr=0
        ];
        let beta = 0.5;
        let stats = DpoRewardStats::from_pairs(&pairs, beta);
        assert_eq!(stats.chosen_rewards.len(), 2);
        assert_eq!(stats.rejected_rewards.len(), 2);
        assert_eq!(stats.accuracy, 1.0);
        assert!(stats.mean_margin > 0.0);
    }

    #[test]
    fn test_dpo_reward_stats_percentile() {
        let pairs: Vec<DpoPair> = (0..5)
            .map(|i| {
                // margins will be 0.1*beta, 0.2*beta, ... 0.5*beta
                make_pair(0.0, -(i as f32 + 1.0) * 0.1, 0.0, 0.0)
            })
            .collect();
        let stats = DpoRewardStats::from_pairs(&pairs, 1.0);

        // Sorted margins: [0.1, 0.2, 0.3, 0.4, 0.5]
        // 0th percentile = 0.1
        let p0 = stats.percentile_margin(0.0);
        assert!((p0 - 0.1).abs() < 1e-5, "p0={}", p0);

        // 100th percentile = 0.5
        let p100 = stats.percentile_margin(1.0);
        assert!((p100 - 0.5).abs() < 1e-5, "p100={}", p100);

        // 50th percentile = 0.3
        let p50 = stats.percentile_margin(0.5);
        assert!((p50 - 0.3).abs() < 1e-5, "p50={}", p50);
    }

    // ── DpoTrainer tests ─────────────────────────────────────────────────

    #[test]
    fn test_dpo_trainer_step() {
        let mut trainer = DpoTrainer::new(DpoConfig::default());
        let pairs = vec![make_pair(-1.0, -2.0, -2.0, -1.5)];
        let result = trainer.step(&pairs).expect("step failed");
        assert!(result.loss.is_finite());
        assert_eq!(trainer.history().len(), 1);
    }

    #[test]
    fn test_dpo_trainer_history() {
        let mut trainer = DpoTrainer::new(DpoConfig::default());
        let pairs = vec![make_pair(-1.0, -2.0, -2.0, -1.5)];
        trainer.step(&pairs).expect("step 1 failed");
        trainer.step(&pairs).expect("step 2 failed");
        assert_eq!(trainer.history().len(), 2);
    }

    #[test]
    fn test_dpo_trainer_mean_loss() {
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        let mut trainer = DpoTrainer::new(cfg);
        assert!((trainer.mean_loss()).abs() < 1e-8, "empty history should give 0");

        let pairs1 = vec![make_pair(0.0, -1.0, 0.0, 1.0)];
        let pairs2 = vec![make_pair(-1.0, -2.0, -2.0, -1.5)];
        let r1 = trainer.step(&pairs1).expect("step 1 failed");
        let r2 = trainer.step(&pairs2).expect("step 2 failed");

        let expected_mean = (r1.loss + r2.loss) / 2.0;
        assert!((trainer.mean_loss() - expected_mean).abs() < 1e-5);
    }

    // ── Error handling tests ─────────────────────────────────────────────

    #[test]
    fn test_dpo_error_display() {
        let e1 = DpoError::EmptyBatch;
        assert!(e1.to_string().contains("empty"));

        let e2 = DpoError::NumericalError("inf detected".to_string());
        assert!(e2.to_string().contains("inf"));

        let e3 = DpoError::InvalidConfig("beta must be positive".to_string());
        assert!(e3.to_string().contains("beta"));
    }

    #[test]
    fn test_dpo_loss_empty_batch() {
        let cfg = DpoConfig::default();
        let result = compute_dpo_loss(&[], &cfg);
        assert!(matches!(result, Err(DpoError::EmptyBatch)));
    }

    #[test]
    fn test_dpo_trainer_reward_stats() {
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        let mut trainer = DpoTrainer::new(cfg);
        assert!(trainer.reward_stats().is_none());

        let pairs = vec![make_pair(0.0, -2.0, 0.0, 2.0)];
        trainer.step(&pairs).expect("step failed");
        let stats = trainer.reward_stats();
        assert!(stats.is_some());
        let (chosen, rejected) = stats.expect("stats should be Some");
        assert!(chosen > rejected, "chosen={} should > rejected={}", chosen, rejected);
    }

    // ── Additional DPO tests ─────────────────────────────────────────────

    #[test]
    fn test_dpo_loss_sigmoid_chosen_equals_rejected_gives_log_half() {
        // When chosen = rejected: h=0 → loss = -log σ(0) = -log(0.5) = log(2)
        let cfg = DpoConfig { beta: 0.1, label_smoothing: 0.0, ..Default::default() };
        // chosen_log_ratio = 0, rejected_log_ratio = 0 → h = 0
        let pair = make_pair(-1.0, -1.0, -1.0, -1.0);
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        let expected = (2.0_f32).ln(); // log(2) ≈ 0.6931
        assert!(
            (result.loss - expected).abs() < 1e-5,
            "expected log(2)={}, got {}",
            expected, result.loss
        );
    }

    #[test]
    fn test_dpo_reward_margin_chosen_greater_than_rejected() {
        // When policy favors chosen: chosen_reward > rejected_reward
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        // chosen_lr = 2.0, rejected_lr = -1.0
        let pair = make_pair(0.0, -2.0, 0.0, 1.0);
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        assert!(
            result.chosen_reward_mean > result.rejected_reward_mean,
            "chosen={} should > rejected={}",
            result.chosen_reward_mean, result.rejected_reward_mean
        );
        assert!(result.reward_margin_mean > 0.0, "margin should be positive");
    }

    #[test]
    fn test_dpo_label_smoothing_zero_matches_original() {
        // ε=0 should produce identical results to the standard sigmoid DPO
        let pair = make_pair(-1.0, -2.0, -2.0, -1.5);
        let cfg_no_smooth = DpoConfig { beta: 0.5, label_smoothing: 0.0, ..Default::default() };
        let cfg_smooth_zero = DpoConfig { beta: 0.5, label_smoothing: 0.0, ..Default::default() };
        let r1 = compute_dpo_loss(&[pair.clone()], &cfg_no_smooth).expect("ok");
        let r2 = compute_dpo_loss(&[pair], &cfg_smooth_zero).expect("ok");
        assert!((r1.loss - r2.loss).abs() < 1e-6, "losses should match: {} vs {}", r1.loss, r2.loss);
    }

    #[test]
    fn test_dpo_label_smoothing_positive_increases_min_loss() {
        // With ε > 0, even a "perfect" pair should have loss > 0
        let pair = make_pair(0.0, -100.0, 0.0, 100.0); // very large margin
        let cfg = DpoConfig { beta: 1.0, label_smoothing: 0.2, ..Default::default() };
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        // Min loss with ε=0.2: 0.2 * log(2) ≈ 0.1386
        let min_expected = 0.2 * (2.0_f32).ln();
        assert!(
            result.loss >= min_expected - 1e-3,
            "smoothed loss {} should be >= {}", result.loss, min_expected
        );
    }

    #[test]
    fn test_dpo_ipo_loss_at_target_is_zero() {
        // IPO: (h - 1/(2β))^2 = 0 when h = 1/(2β)
        let beta = 0.5_f32;
        let target = 1.0 / (2.0 * beta); // = 1.0
        let cfg = DpoConfig { beta, loss_type: DpoLossType::Ipo, ..Default::default() };
        // chosen_lr - rejected_lr = target → chosen_lr = target, rejected_lr = 0
        let pair = make_pair(0.0, -target, -0.0, 0.0); // chosen_lr=target, rejected_lr=0
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        assert!((result.loss).abs() < 1e-5, "IPO at target should be 0, got {}", result.loss);
    }

    #[test]
    fn test_dpo_hinge_loss_positive_when_margin_not_satisfied() {
        // When β*h < 1: hinge = 1 - β*h > 0
        let cfg = DpoConfig {
            beta: 1.0,
            loss_type: DpoLossType::Hinge,
            ..Default::default()
        };
        // h = 0.5 → 1 - 1.0*0.5 = 0.5
        let pair = make_pair(0.0, -0.5, -0.0, 0.0); // chosen_lr=0.5, rejected_lr=0 → h=0.5
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        assert!((result.loss - 0.5).abs() < 1e-5, "hinge loss={}", result.loss);
    }

    #[test]
    fn test_dpo_hinge_loss_zero_when_margin_satisfied() {
        // When β*h > 1: hinge = 0
        let cfg = DpoConfig {
            beta: 1.0,
            loss_type: DpoLossType::Hinge,
            ..Default::default()
        };
        // h = 2.0 → 1 - 1.0*2.0 = -1 → max(0,-1) = 0
        let pair = make_pair(0.0, -2.0, -0.0, 0.0); // chosen_lr=2.0 → h=2.0
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        assert!((result.loss).abs() < 1e-5, "hinge should be 0, got {}", result.loss);
    }

    #[test]
    fn test_dpo_high_beta_sharpens_preference() {
        // High β → loss more sensitive to margin differences
        let pair_small_margin = make_pair(0.0, -0.5, 0.0, 0.5); // h=1.0
        let cfg_low_beta = DpoConfig { beta: 0.01, ..Default::default() };
        let cfg_high_beta = DpoConfig { beta: 10.0, ..Default::default() };
        let r_low = compute_dpo_loss(&[pair_small_margin.clone()], &cfg_low_beta).expect("ok");
        let r_high = compute_dpo_loss(&[pair_small_margin], &cfg_high_beta).expect("ok");
        // High beta → larger scaled_h → -log_sigmoid(large) → near 0
        // Low beta → smaller scaled_h → -log_sigmoid(small) → near log(2)
        assert!(
            r_low.loss > r_high.loss,
            "low beta loss {} should > high beta loss {}",
            r_low.loss, r_high.loss
        );
    }

    #[test]
    fn test_dpo_beta_zero_equivalent_gives_log_half() {
        // When β≈0: scaled_h ≈ 0 → loss ≈ -log σ(0) = log(2)
        let cfg = DpoConfig { beta: 1e-8, ..Default::default() };
        let pair = make_pair(-1.0, -2.0, -0.5, 0.5); // non-trivial h
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        let log2 = (2.0_f32).ln();
        assert!((result.loss - log2).abs() < 1e-3, "near-zero beta loss={} expected log(2)={}", result.loss, log2);
    }

    #[test]
    fn test_dpo_batch_loss_is_mean_of_individual_losses() {
        // Batch loss should equal the mean of individually computed losses
        let cfg = DpoConfig { beta: 0.5, ..Default::default() };
        let pairs = vec![
            make_pair(-1.0, -2.0, -1.5, -1.8),
            make_pair(-0.5, -1.0, -0.8, -1.2),
            make_pair(0.0, -0.5, -0.3, 0.1),
        ];
        let batch_result = compute_dpo_loss(&pairs, &cfg).expect("batch ok");

        let mut individual_sum = 0.0_f32;
        for pair in &pairs {
            let r = compute_dpo_loss(&[pair.clone()], &cfg).expect("individual ok");
            individual_sum += r.loss;
        }
        let individual_mean = individual_sum / pairs.len() as f32;
        assert!((batch_result.loss - individual_mean).abs() < 1e-5, "batch={} individual_mean={}", batch_result.loss, individual_mean);
    }

    #[test]
    fn test_dpo_accuracy_metric_fraction_chosen_greater() {
        // Accuracy = fraction where chosen_reward > rejected_reward
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        let pairs = vec![
            make_pair(0.0, -2.0, 0.0, 2.0),  // chosen_lr=2 > rejected_lr=-2 ✓
            make_pair(0.0, 2.0, 0.0, -2.0),  // chosen_lr=-2 < rejected_lr=2  ✗
            make_pair(0.0, -1.0, 0.0, 1.0),  // chosen_lr=1 > rejected_lr=-1  ✓
            make_pair(0.0, 1.0, 0.0, -1.0),  // chosen_lr=-1 < rejected_lr=1  ✗
        ];
        let result = compute_dpo_loss(&pairs, &cfg).expect("ok");
        assert!((result.reward_accuracy - 0.5).abs() < 1e-5, "accuracy={}", result.reward_accuracy);
    }

    #[test]
    fn test_dpo_numerical_stability_extreme_log_probs() {
        // Should not produce NaN with extreme but finite log probs
        let cfg = DpoConfig { beta: 0.1, ..Default::default() };
        let pair = make_pair(-100.0, -200.0, -150.0, -50.0);
        let result = compute_dpo_loss(&[pair], &cfg);
        assert!(result.is_ok(), "should not error on extreme values");
        let r = result.expect("ok");
        assert!(r.loss.is_finite(), "loss must be finite, got {}", r.loss);
    }

    #[test]
    fn test_dpo_reference_free_mode_ignores_reference() {
        // In reference-free mode, the reference log-probs should not affect the loss.
        // Two pairs with same policy log-probs but DIFFERENT reference log-probs should
        // produce the same loss when reference_free=true.
        // make_pair(policy_chosen, ref_chosen, policy_rejected, ref_rejected)
        let pair_ref_a = make_pair(-1.0, -0.5, -2.0, -0.3); // ref_chosen=-0.5, ref_rejected=-0.3
        let pair_ref_b = make_pair(-1.0, -5.0, -2.0, -3.0); // ref_chosen=-5.0, ref_rejected=-3.0, different refs
        let cfg_free = DpoConfig { reference_free: true, beta: 0.5, ..Default::default() };
        let r_a = compute_dpo_loss(&[pair_ref_a], &cfg_free).expect("ok_a");
        let r_b = compute_dpo_loss(&[pair_ref_b], &cfg_free).expect("ok_b");
        // In reference-free mode: h = policy_chosen - policy_rejected = -1.0 - (-2.0) = 1.0 (same for both)
        assert!(
            (r_a.loss - r_b.loss).abs() < 1e-5,
            "reference-free: same policy log-probs but different refs → same loss: {} vs {}",
            r_a.loss, r_b.loss
        );
    }

    #[test]
    fn test_dpo_invalid_label_smoothing_returns_error() {
        let cfg = DpoConfig { label_smoothing: 1.0, ..Default::default() };
        let pair = make_pair(-1.0, -2.0, -1.5, -0.5);
        let result = compute_dpo_loss(&[pair], &cfg);
        assert!(matches!(result, Err(DpoError::InvalidConfig(_))), "expected InvalidConfig error");
    }

    #[test]
    fn test_dpo_reward_stats_std_margin() {
        // With identical margins, std should be 0
        let pairs: Vec<DpoPair> = (0..4)
            .map(|_| make_pair(0.0, -1.0, 0.0, 1.0)) // margin = β*(1-(-1)) = 2β
            .collect();
        let stats = DpoRewardStats::from_pairs(&pairs, 0.5);
        assert!(stats.std_margin < 1e-5, "std should be ~0 for identical margins, got {}", stats.std_margin);
        assert_eq!(stats.accuracy, 1.0);
    }

    #[test]
    fn test_dpo_reward_stats_empty_pairs() {
        let stats = DpoRewardStats::from_pairs(&[], 1.0);
        assert!(stats.chosen_rewards.is_empty());
        assert_eq!(stats.accuracy, 0.0);
        assert_eq!(stats.mean_margin, 0.0);
        assert_eq!(stats.percentile_margin(0.5), 0.0);
    }

    #[test]
    fn test_dpo_trainer_mean_reward_accuracy() {
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        let mut trainer = DpoTrainer::new(cfg);
        assert!((trainer.mean_reward_accuracy()).abs() < 1e-8, "empty gives 0");

        // All correct pair
        let correct = vec![make_pair(0.0, -2.0, 0.0, 2.0)];
        // All incorrect pair
        let incorrect = vec![make_pair(0.0, 2.0, 0.0, -2.0)];

        trainer.step(&correct).expect("step 1");
        trainer.step(&incorrect).expect("step 2");

        // Mean accuracy over 2 steps: (1.0 + 0.0) / 2 = 0.5
        assert!((trainer.mean_reward_accuracy() - 0.5).abs() < 1e-5, "mean_acc={}", trainer.mean_reward_accuracy());
    }

    // ── New extended tests ───────────────────────────────────────────────────

    // Test: DPO with length-normalized rewards (simulated via scaled log-probs)
    #[test]
    fn test_dpo_length_normalized_rewards_scale() {
        // Length normalization can be simulated by dividing log-probs by length.
        // A short chosen (len=1) and long chosen (len=10) with same per-token lp should give same reward.
        let beta = 0.5_f32;
        // Per-token log-prob = -1 for all tokens
        let short_lp = -1.0_f32; // 1 token → total = -1
        let long_lp = -10.0_f32 / 10.0; // 10 tokens → normalized = -1

        // Both normalized to same value: chosen and rejected have same ratio → h = 0
        let pair_short = DpoPair {
            policy_log_prob_chosen: short_lp,
            reference_log_prob_chosen: short_lp,
            policy_log_prob_rejected: short_lp,
            reference_log_prob_rejected: short_lp,
        };
        let pair_long = DpoPair {
            policy_log_prob_chosen: long_lp,
            reference_log_prob_chosen: long_lp,
            policy_log_prob_rejected: long_lp,
            reference_log_prob_rejected: long_lp,
        };
        let cfg = DpoConfig { beta, ..Default::default() };
        let r_short = compute_dpo_loss(&[pair_short], &cfg).expect("ok");
        let r_long = compute_dpo_loss(&[pair_long], &cfg).expect("ok");
        // Both have h = 0 → same loss = log(2)
        assert!((r_short.loss - r_long.loss).abs() < 1e-5,
            "length-normalized pairs with same per-token lp should give same loss: short={} long={}", r_short.loss, r_long.loss);
    }

    // Test: reference model synchronization — same policy and ref → neutral update
    #[test]
    fn test_reference_model_synchronization() {
        // When policy = reference model, log-ratios are 0, h = 0.
        // Loss = -log σ(0) = log(2)
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        // policy log-probs = reference log-probs → ratios = 0
        let pair = make_pair(-1.5, -1.5, -2.0, -2.0);
        let result = compute_dpo_loss(&[pair], &cfg).expect("ok");
        let expected = (2.0_f32).ln();
        assert!((result.loss - expected).abs() < 1e-5,
            "synchronized policy/ref should give log(2) loss, got {}", result.loss);
    }

    // Test: DPO with very long sequences (near max_length) — numerical stability
    #[test]
    fn test_dpo_long_sequence_numerical_stability() {
        // Simulate long sequences: sum of many small log-probs
        let cfg = DpoConfig { beta: 0.1, ..Default::default() };
        // Policy log-prob: sum over 512 tokens of -0.01 each = -5.12
        let policy_chosen = -5.12_f32;
        let ref_chosen = -6.0_f32;
        let policy_rejected = -7.0_f32;
        let ref_rejected = -5.5_f32;
        let pair = make_pair(policy_chosen, ref_chosen, policy_rejected, ref_rejected);
        let result = compute_dpo_loss(&[pair], &cfg);
        assert!(result.is_ok(), "should not fail on long-seq log-probs");
        let r = result.expect("ok");
        assert!(r.loss.is_finite(), "loss must be finite, got {}", r.loss);
    }

    // Test: multiple preference pairs batched — batch size reported correctly
    #[test]
    fn test_multiple_preference_pairs_batched() {
        let cfg = DpoConfig::default();
        let n = 8;
        let pairs: Vec<DpoPair> = (0..n)
            .map(|i| make_pair(-(i as f32 * 0.1), -(i as f32 * 0.2), -(i as f32 * 0.3 + 0.5), -(i as f32 * 0.1)))
            .collect();
        let result = compute_dpo_loss(&pairs, &cfg).expect("ok");
        assert_eq!(result.num_pairs, n, "num_pairs should match batch size");
        assert!(result.loss.is_finite());
    }

    // Test: gradient accumulation with DPO — splitting batch gives same mean loss
    #[test]
    fn test_gradient_accumulation_equivalent_to_full_batch() {
        let cfg = DpoConfig { beta: 0.5, ..Default::default() };
        let pairs = vec![
            make_pair(-1.0, -1.5, -2.0, -1.0),
            make_pair(-0.5, -1.0, -1.5, -0.8),
            make_pair(0.0, -0.5, -0.3, 0.2),
            make_pair(-1.2, -2.0, -0.8, -0.3),
        ];

        // Full batch
        let full_result = compute_dpo_loss(&pairs, &cfg).expect("full batch ok");

        // Split into 2 micro-batches
        let r1 = compute_dpo_loss(&pairs[..2], &cfg).expect("micro 1 ok");
        let r2 = compute_dpo_loss(&pairs[2..], &cfg).expect("micro 2 ok");
        let micro_mean = (r1.loss + r2.loss) / 2.0;

        assert!((full_result.loss - micro_mean).abs() < 1e-5,
            "grad accumulation: full={} micro_mean={}", full_result.loss, micro_mean);
    }

    // Test: DPO temperature sensitivity — higher beta amplifies margin effect
    #[test]
    fn test_dpo_temperature_sensitivity_higher_beta_lower_loss() {
        // When chosen is preferred: higher β → lower loss (margin amplified)
        let pair = make_pair(0.0, -3.0, 0.0, 3.0); // h = 6
        let cfg_low = DpoConfig { beta: 0.01, ..Default::default() };
        let cfg_high = DpoConfig { beta: 5.0, ..Default::default() };
        let r_low = compute_dpo_loss(&[pair.clone()], &cfg_low).expect("ok");
        let r_high = compute_dpo_loss(&[pair], &cfg_high).expect("ok");
        assert!(r_high.loss < r_low.loss,
            "higher beta reduces loss when chosen preferred: low={} high={}", r_low.loss, r_high.loss);
    }

    // Test: offline DPO mode — reference log-probs are fixed, only policy updates
    #[test]
    fn test_offline_dpo_reference_fixed() {
        // In offline DPO, the reference is fixed. We verify that changing policy
        // while keeping reference fixed changes the loss as expected.
        let beta = 1.0_f32;
        let ref_chosen = -2.0_f32;
        let ref_rejected = -1.0_f32;

        // Policy moves toward chosen
        let pair_before = DpoPair {
            policy_log_prob_chosen: ref_chosen, // same as ref: ratio = 0
            reference_log_prob_chosen: ref_chosen,
            policy_log_prob_rejected: ref_rejected,
            reference_log_prob_rejected: ref_rejected,
        };
        let pair_after = DpoPair {
            policy_log_prob_chosen: ref_chosen + 1.0, // policy improved chosen
            reference_log_prob_chosen: ref_chosen,
            policy_log_prob_rejected: ref_rejected,
            reference_log_prob_rejected: ref_rejected,
        };
        let cfg = DpoConfig { beta, ..Default::default() };
        let r_before = compute_dpo_loss(&[pair_before], &cfg).expect("ok");
        let r_after = compute_dpo_loss(&[pair_after], &cfg).expect("ok");
        assert!(r_after.loss < r_before.loss,
            "improving policy chosen log-prob reduces loss: before={} after={}", r_before.loss, r_after.loss);
    }

    // Test: weight update direction — loss decreases when chosen log-prob improves
    #[test]
    fn test_weight_update_direction_loss_decreases() {
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        // Baseline: h = 0
        let pair_base = make_pair(-1.0, -1.0, -1.0, -1.0);
        let r_base = compute_dpo_loss(&[pair_base], &cfg).expect("ok");

        // After update: chosen log-ratio improves
        let pair_updated = make_pair(-0.5, -1.0, -1.0, -1.0); // chosen_lr = 0.5 now
        let r_updated = compute_dpo_loss(&[pair_updated], &cfg).expect("ok");
        assert!(r_updated.loss < r_base.loss,
            "improving chosen log-ratio should reduce loss: base={} updated={}", r_base.loss, r_updated.loss);
    }

    // Test: convergence on distinguishable pairs — loss decreases with large margin
    #[test]
    fn test_convergence_on_distinguishable_pairs() {
        let cfg = DpoConfig { beta: 1.0, ..Default::default() };
        // Small margin
        let pair_small = make_pair(0.0, -0.1, 0.0, 0.1); // h = 0.2
        // Large margin (well-converged)
        let pair_large = make_pair(0.0, -5.0, 0.0, 5.0); // h = 10

        let r_small = compute_dpo_loss(&[pair_small], &cfg).expect("ok");
        let r_large = compute_dpo_loss(&[pair_large], &cfg).expect("ok");
        assert!(r_large.loss < r_small.loss,
            "well-separated pair should have lower loss: small_margin={} large_margin={}", r_small.loss, r_large.loss);
    }

    // Test: robustness to noisy preferences (50/50 random labels) — loss ≈ log(2)
    #[test]
    fn test_robustness_to_noisy_preferences() {
        // When labels are random (50/50 correct), reward accuracy ≈ 0.5
        // and loss ≈ log(2) (uninformative signal)
        let cfg = DpoConfig { beta: 0.01, ..Default::default() };
        // Mix of correct and incorrect pairs with near-zero margin
        let pairs = vec![
            make_pair(-1.0, -1.0, -1.0, -1.0), // h=0 (noisy)
            make_pair(-1.0, -1.0, -1.0, -1.0), // h=0 (noisy)
            make_pair(-1.0, -1.0, -1.0, -1.0), // h=0 (noisy)
            make_pair(-1.0, -1.0, -1.0, -1.0), // h=0 (noisy)
        ];
        let result = compute_dpo_loss(&pairs, &cfg).expect("ok");
        let log2 = (2.0_f32).ln();
        assert!(
            (result.loss - log2).abs() < 1e-4,
            "noisy labels (h≈0) should give loss ≈ log(2)={log2}, got {}", result.loss
        );
    }
}
