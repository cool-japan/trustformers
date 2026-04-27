//! SimPO — Simple Preference Optimization (2024).
//!
//! SimPO uses length-normalized average log probability as a reward signal,
//! adds a target margin γ, and eliminates the need for a reference model.
//!
//! Reference: Meng et al., "SimPO: Simple Preference Optimization with a
//! Reference-Free Reward", arXiv:2405.14734, 2024.

use std::fmt;

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Loss types supported by SimPO.
#[derive(Debug, Clone, PartialEq)]
pub enum SimpoLossType {
    /// Original SimPO sigmoid loss: −log(σ(β*(r_w − r_l) − γ))
    /// Alias: `Sigmoid`.
    Basic,
    /// Label-smoothed variant: mixes target distribution with uniform.
    LabelSmoothed,
    /// Margin-only hinge: max(0, γ − β*(r_w − r_l))
    /// Alias: `Hinge` / `MarginOnly`.
    MarginOnly,
    /// Sigmoid (DPO-style): −log(σ(reward_diff)) — same as Basic
    Sigmoid,
    /// Hinge: max(0, −reward_diff) — same as MarginOnly
    Hinge,
    /// IPO variant: (reward_diff / β − 1/(2β))²
    IPO,
}

/// Configuration for SimPO training.
#[derive(Debug, Clone)]
pub struct SimpoConfig {
    /// DPO-style temperature β (default 2.5).
    pub beta: f64,
    /// Target reward margin γ (default 1.4).
    pub gamma: f64,
    /// Label-smoothing factor (default 0.0).
    pub label_smoothing: f64,
    /// Which loss variant to use.
    pub loss_type: SimpoLossType,
}

impl Default for SimpoConfig {
    fn default() -> Self {
        Self {
            beta: 2.5,
            gamma: 1.4,
            label_smoothing: 0.0,
            loss_type: SimpoLossType::Sigmoid,
        }
    }
}

// ──────────────────────────────────────────────
// SimpoReward
// ──────────────────────────────────────────────

/// A sequence's length-normalised log-probability reward.
#[derive(Debug, Clone)]
pub struct SimpoReward {
    /// Per-token log-probabilities.
    pub log_probs: Vec<f64>,
    /// Number of tokens (== log_probs.len()).
    pub length: usize,
}

impl SimpoReward {
    /// Construct a new `SimpoReward`.
    pub fn new(log_probs: Vec<f64>) -> Self {
        let length = log_probs.len();
        Self { log_probs, length }
    }

    /// Length-normalised reward: `sum(log_probs) / length`.
    ///
    /// Returns `0.0` for an empty sequence.
    pub fn reward(&self) -> f64 {
        if self.length == 0 {
            return 0.0;
        }
        self.sum_log_probs() / self.length as f64
    }

    /// Sum of all per-token log-probabilities.
    pub fn sum_log_probs(&self) -> f64 {
        self.log_probs.iter().sum()
    }
}

// ──────────────────────────────────────────────
// SimpoLoss
// ──────────────────────────────────────────────

/// Computes the SimPO training objective for a preference pair.
pub struct SimpoLoss {
    config: SimpoConfig,
    step: u64,
}

impl SimpoLoss {
    /// Create a new loss computer from the supplied configuration.
    pub fn new(config: SimpoConfig) -> Self {
        Self { config, step: 0 }
    }

    /// SimPO reward difference:
    ///
    /// `β × (r_chosen − r_rejected) − γ`
    ///
    /// where `r = reward()` (length-normalised log prob).
    pub fn reward_diff(&self, chosen: &SimpoReward, rejected: &SimpoReward) -> f64 {
        self.config.beta * (chosen.reward() - rejected.reward()) - self.config.gamma
    }

    /// Compute SimPO loss for a single preference pair.
    ///
    /// - **Sigmoid**: `−log(σ(reward_diff))`
    /// - **Hinge**:   `max(0, −reward_diff)`
    /// - **IPO**:     `(reward_diff/β − 1/(2β))²`
    pub fn compute_loss(
        &mut self,
        chosen: &SimpoReward,
        rejected: &SimpoReward,
    ) -> Result<SimpoLossResult, SimpoError> {
        if chosen.length == 0 || rejected.length == 0 {
            return Err(SimpoError::EmptySequence);
        }

        let cr = chosen.reward();
        let rr = rejected.reward();

        if cr.is_nan() || rr.is_nan() {
            return Err(SimpoError::NanReward);
        }

        let rd = self.reward_diff(chosen, rejected);

        let loss = match self.config.loss_type {
            SimpoLossType::Basic | SimpoLossType::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-rd).exp());
                // label smoothing: −[(1−ls)·log σ(rd) + ls·log σ(−rd)]
                if self.config.label_smoothing > 0.0 {
                    let ls = self.config.label_smoothing;
                    let sig_neg = 1.0 / (1.0 + rd.exp());
                    -(1.0 - ls) * sigmoid.ln() - ls * sig_neg.max(1e-12).ln()
                } else {
                    -sigmoid.ln()
                }
            }
            SimpoLossType::LabelSmoothed => {
                // Explicit label-smoothed sigmoid loss
                let ls = self.config.label_smoothing.max(0.1); // default 0.1 if unset
                let sigmoid = 1.0 / (1.0 + (-rd).exp());
                let sig_neg = 1.0 / (1.0 + rd.exp());
                -(1.0 - ls) * sigmoid.ln() - ls * sig_neg.max(1e-12).ln()
            }
            SimpoLossType::MarginOnly | SimpoLossType::Hinge => {
                // max(0, γ − β*(r_w − r_l))  — i.e., max(0, -rd)
                (-rd).max(0.0)
            }
            SimpoLossType::IPO => {
                // (reward_diff/beta - 1/(2*beta))^2
                let inner = rd / self.config.beta - 1.0 / (2.0 * self.config.beta);
                inner * inner
            }
        };

        let accuracy = if cr > rr { 1.0 } else { 0.0 };

        self.step += 1;

        Ok(SimpoLossResult {
            loss,
            chosen_reward: cr,
            rejected_reward: rr,
            reward_diff: rd,
            accuracy,
            loss_type: self.config.loss_type.clone(),
        })
    }

    /// Number of `compute_loss` calls completed so far.
    pub fn step(&self) -> u64 {
        self.step
    }
}

// ──────────────────────────────────────────────
// SimpoLossResult
// ──────────────────────────────────────────────

/// Loss statistics for a single SimPO preference pair.
#[derive(Debug, Clone)]
pub struct SimpoLossResult {
    /// Scalar loss value.
    pub loss: f64,
    /// Length-normalised reward for the chosen sequence.
    pub chosen_reward: f64,
    /// Length-normalised reward for the rejected sequence.
    pub rejected_reward: f64,
    /// `β × (r_chosen − r_rejected) − γ`.
    pub reward_diff: f64,
    /// 1.0 if r_chosen > r_rejected, else 0.0.
    pub accuracy: f64,
    /// Which loss variant was used.
    pub loss_type: SimpoLossType,
}

impl fmt::Display for SimpoLossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lt = match self.loss_type {
            SimpoLossType::Basic => "basic",
            SimpoLossType::LabelSmoothed => "label_smoothed",
            SimpoLossType::MarginOnly => "margin_only",
            SimpoLossType::Sigmoid => "sigmoid",
            SimpoLossType::Hinge => "hinge",
            SimpoLossType::IPO => "ipo",
        };
        write!(
            f,
            "SimpoLossResult {{ loss: {:.4}, chosen_reward: {:.4}, rejected_reward: {:.4}, \
             reward_diff: {:.4}, accuracy: {:.4}, loss_type: {lt} }}",
            self.loss, self.chosen_reward, self.rejected_reward,
            self.reward_diff, self.accuracy,
        )
    }
}

// ──────────────────────────────────────────────
// Standalone helpers
// ──────────────────────────────────────────────

/// Compute the SimPO reward for a completion: mean per-token log-probability.
///
/// `reward(y) = (1/|y|) × Σ log π_θ(y_t | x, y_{<t})`
///
/// Returns 0.0 for an empty slice.
pub fn compute_simpo_reward(log_probs: &[f32]) -> f32 {
    if log_probs.is_empty() {
        return 0.0;
    }
    log_probs.iter().sum::<f32>() / log_probs.len() as f32
}

/// Output of the standalone SimPO loss computation.
#[derive(Debug, Clone)]
pub struct SimpoLossOutput {
    /// Scalar loss value.
    pub loss: f32,
    /// Length-normalised reward for the chosen completion.
    pub chosen_reward: f32,
    /// Length-normalised reward for the rejected completion.
    pub rejected_reward: f32,
    /// Reward margin: `chosen_reward − rejected_reward`.
    pub margin: f32,
    /// 1.0 if chosen_reward > rejected_reward, else 0.0.
    pub accuracy: f32,
}

/// Compute SimPO loss from per-token log-probabilities.
///
/// Uses `SimpoConfig::loss_type` to select the loss variant:
/// - **Basic / Sigmoid**: `−log σ(β*(r_w − r_l) − γ)`
/// - **LabelSmoothed**: label-smoothed sigmoid
/// - **MarginOnly / Hinge**: `max(0, γ − β*(r_w − r_l))`
/// - **IPO**: `(β*(r_w−r_l)−γ)/β − 1/(2β))²`
pub fn compute_simpo_loss(
    chosen_log_probs: &[f32],
    rejected_log_probs: &[f32],
    config: &SimpoConfig,
) -> Result<SimpoLossOutput, SimpoError> {
    if chosen_log_probs.is_empty() || rejected_log_probs.is_empty() {
        return Err(SimpoError::EmptySequence);
    }

    let r_w = compute_simpo_reward(chosen_log_probs);
    let r_l = compute_simpo_reward(rejected_log_probs);

    if r_w.is_nan() || r_l.is_nan() {
        return Err(SimpoError::NanReward);
    }

    let beta = config.beta as f32;
    let gamma = config.gamma as f32;
    let rd = beta * (r_w - r_l) - gamma;

    let loss = match config.loss_type {
        SimpoLossType::Basic | SimpoLossType::Sigmoid => {
            let sigmoid = 1.0 / (1.0 + (-rd).exp());
            if config.label_smoothing > 0.0 {
                let ls = config.label_smoothing as f32;
                let sig_neg = 1.0 / (1.0 + rd.exp());
                -(1.0 - ls) * sigmoid.ln() - ls * sig_neg.max(1e-12).ln()
            } else {
                -sigmoid.ln()
            }
        }
        SimpoLossType::LabelSmoothed => {
            let ls = (config.label_smoothing as f32).max(0.1);
            let sigmoid = 1.0 / (1.0 + (-rd).exp());
            let sig_neg = 1.0 / (1.0 + rd.exp());
            -(1.0 - ls) * sigmoid.ln() - ls * sig_neg.max(1e-12).ln()
        }
        SimpoLossType::MarginOnly | SimpoLossType::Hinge => (-rd).max(0.0),
        SimpoLossType::IPO => {
            let inner = rd / beta - 1.0 / (2.0 * beta);
            inner * inner
        }
    };

    Ok(SimpoLossOutput {
        loss,
        chosen_reward: r_w,
        rejected_reward: r_l,
        margin: r_w - r_l,
        accuracy: if r_w > r_l { 1.0 } else { 0.0 },
    })
}

// ──────────────────────────────────────────────
// SimpoError
// ──────────────────────────────────────────────

/// Errors that can arise during SimPO loss computation.
#[derive(Debug, thiserror::Error)]
pub enum SimpoError {
    /// One or both sequences have no tokens.
    #[error("Empty sequence")]
    EmptySequence,
    /// A reward value is NaN.
    #[error("Invalid reward: NaN")]
    NanReward,
}

// ──────────────────────────────────────────────
// SimpoTrainer
// ──────────────────────────────────────────────

/// Batch trainer that averages SimPO loss and accuracy over a batch of
/// preference pairs.
pub struct SimpoTrainer {
    loss_fn: SimpoLoss,
    history: Vec<SimpoLossResult>,
}

impl SimpoTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: SimpoConfig) -> Self {
        Self {
            loss_fn: SimpoLoss::new(config),
            history: Vec::new(),
        }
    }

    /// Process a batch of preference pairs, returning batch-level statistics.
    ///
    /// Each element of `batch` is `(chosen, rejected)`.  The individual
    /// `SimpoLossResult`s are appended to `history`.
    pub fn train_step(
        &mut self,
        batch: &[(SimpoReward, SimpoReward)],
    ) -> Result<SimpoTrainResult, SimpoError> {
        if batch.is_empty() {
            return Err(SimpoError::EmptySequence);
        }

        let mut total_loss = 0.0_f64;
        let mut total_accuracy = 0.0_f64;

        for (chosen, rejected) in batch {
            let result = self.loss_fn.compute_loss(chosen, rejected)?;
            total_loss += result.loss;
            total_accuracy += result.accuracy;
            self.history.push(result);
        }

        let n = batch.len() as f64;
        let step = self.loss_fn.step();

        Ok(SimpoTrainResult {
            mean_loss: total_loss / n,
            mean_accuracy: total_accuracy / n,
            batch_size: batch.len(),
            step,
        })
    }

    /// Full history of individual `SimpoLossResult`s from all training steps.
    pub fn history(&self) -> &[SimpoLossResult] {
        &self.history
    }

    /// Mean accuracy across all entries in `history`.
    pub fn mean_accuracy(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|r| r.accuracy).sum();
        sum / self.history.len() as f64
    }
}

// ──────────────────────────────────────────────
// SimpoTrainResult
// ──────────────────────────────────────────────

/// Batch-level training statistics from one `train_step` call.
#[derive(Debug, Clone)]
pub struct SimpoTrainResult {
    /// Mean loss over the batch.
    pub mean_loss: f64,
    /// Mean accuracy over the batch.
    pub mean_accuracy: f64,
    /// Number of preference pairs in the batch.
    pub batch_size: usize,
    /// Cumulative number of `compute_loss` calls so far.
    pub step: u64,
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a SimpoReward from a slice of log-probs
    fn reward(lps: &[f64]) -> SimpoReward {
        SimpoReward::new(lps.to_vec())
    }

    // ── Test 1: SimpoReward reward() = sum/length ──────────────────────────
    #[test]
    fn test_simpo_reward_value() {
        let r = reward(&[-1.0, -2.0, -3.0]); // sum=-6, len=3 → reward=-2
        assert!((r.reward() - (-2.0)).abs() < 1e-10, "expected -2.0, got {}", r.reward());
    }

    // ── Test 2: reward_diff direction ─────────────────────────────────────
    #[test]
    fn test_reward_diff_direction() {
        let config = SimpoConfig { beta: 1.0, gamma: 0.0, ..Default::default() };
        let loss_fn = SimpoLoss::new(config);
        let chosen = reward(&[-0.5]);     // reward = -0.5
        let rejected = reward(&[-2.0]);   // reward = -2.0
        let rd = loss_fn.reward_diff(&chosen, &rejected);
        // beta * (-0.5 - (-2.0)) - 0 = 1.5
        assert!((rd - 1.5).abs() < 1e-10, "expected 1.5, got {rd}");
    }

    // ── Test 3: Sigmoid loss > 0 ──────────────────────────────────────────
    #[test]
    fn test_sigmoid_loss_positive() {
        let config = SimpoConfig { loss_type: SimpoLossType::Sigmoid, ..Default::default() };
        let mut loss_fn = SimpoLoss::new(config);
        let chosen = reward(&[-0.5]);
        let rejected = reward(&[-2.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!(result.loss > 0.0, "sigmoid loss should be positive, got {}", result.loss);
    }

    // ── Test 4: Hinge loss = 0 when chosen >> rejected ────────────────────
    #[test]
    fn test_hinge_loss_zero_when_chosen_dominant() {
        let config = SimpoConfig {
            beta: 1.0,
            gamma: 0.0,
            loss_type: SimpoLossType::Hinge,
            ..Default::default()
        };
        let mut loss_fn = SimpoLoss::new(config);
        // chosen reward = 0, rejected reward = -100 → reward_diff = 100 >> 0
        let chosen = reward(&[0.0]);
        let rejected = reward(&[-100.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        // hinge = max(0, -reward_diff) = max(0, -100) = 0
        assert!((result.loss - 0.0).abs() < 1e-10, "hinge loss should be 0, got {}", result.loss);
    }

    // ── Test 5: IPO loss ──────────────────────────────────────────────────
    #[test]
    fn test_ipo_loss_value() {
        let config = SimpoConfig {
            beta: 2.0,
            gamma: 0.0,
            loss_type: SimpoLossType::IPO,
            ..Default::default()
        };
        let mut loss_fn = SimpoLoss::new(config);
        // chosen reward = -1, rejected reward = -1 → reward_diff = 0
        // IPO: (0/2 - 1/(2*2))^2 = (-0.25)^2 = 0.0625
        let chosen = reward(&[-1.0]);
        let rejected = reward(&[-1.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        let expected = 0.0625;
        assert!(
            (result.loss - expected).abs() < 1e-10,
            "expected IPO loss={expected}, got {}",
            result.loss
        );
    }

    // ── Test 6: accuracy = 1.0 when chosen better ─────────────────────────
    #[test]
    fn test_accuracy_one_chosen_better() {
        let mut loss_fn = SimpoLoss::new(SimpoConfig::default());
        let chosen = reward(&[-0.5]);
        let rejected = reward(&[-2.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!((result.accuracy - 1.0).abs() < 1e-10, "expected 1.0, got {}", result.accuracy);
    }

    // ── Test 7: accuracy = 0.0 when rejected better ──────────────────────
    #[test]
    fn test_accuracy_zero_rejected_better() {
        let mut loss_fn = SimpoLoss::new(SimpoConfig::default());
        let chosen = reward(&[-3.0]);   // worse
        let rejected = reward(&[-0.5]); // better
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!((result.accuracy - 0.0).abs() < 1e-10, "expected 0.0, got {}", result.accuracy);
    }

    // ── Test 8: gamma effect (higher gamma → higher loss) ─────────────────
    #[test]
    fn test_gamma_effect() {
        let config_lo = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Sigmoid, ..Default::default() };
        let config_hi = SimpoConfig { beta: 1.0, gamma: 5.0, loss_type: SimpoLossType::Sigmoid, ..Default::default() };
        let mut lf_lo = SimpoLoss::new(config_lo);
        let mut lf_hi = SimpoLoss::new(config_hi);
        let chosen = reward(&[-1.0]);
        let rejected = reward(&[-2.0]);
        let lo_result = lf_lo.compute_loss(&chosen, &rejected).expect("should not fail");
        let hi_result = lf_hi.compute_loss(&chosen, &rejected).expect("should not fail");
        assert!(
            hi_result.loss > lo_result.loss,
            "higher gamma should yield higher loss: lo={} hi={}",
            lo_result.loss, hi_result.loss
        );
    }

    // ── Test 9: beta scaling ──────────────────────────────────────────────
    #[test]
    fn test_beta_scaling() {
        // With Sigmoid loss, larger beta amplifies reward_diff → lower loss when chosen is better
        let config_lo = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Sigmoid, ..Default::default() };
        let config_hi = SimpoConfig { beta: 10.0, gamma: 0.0, loss_type: SimpoLossType::Sigmoid, ..Default::default() };
        let mut lf_lo = SimpoLoss::new(config_lo);
        let mut lf_hi = SimpoLoss::new(config_hi);
        let chosen = reward(&[-0.5]);   // reward = -0.5
        let rejected = reward(&[-2.0]); // reward = -2.0
        let lo_result = lf_lo.compute_loss(&chosen, &rejected).expect("should not fail");
        let hi_result = lf_hi.compute_loss(&chosen, &rejected).expect("should not fail");
        // Higher beta → larger reward_diff → sigmoid closer to 1 → -log(sigmoid) smaller
        assert!(
            hi_result.loss < lo_result.loss,
            "higher beta should yield lower sigmoid loss when chosen better: lo={} hi={}",
            lo_result.loss, hi_result.loss
        );
    }

    // ── Test 10: trainer train_step mean ──────────────────────────────────
    #[test]
    fn test_trainer_train_step_mean() {
        let mut trainer = SimpoTrainer::new(SimpoConfig::default());
        let batch = vec![
            (reward(&[-0.5]), reward(&[-2.0])),
            (reward(&[-0.6]), reward(&[-2.1])),
        ];
        let result = trainer.train_step(&batch).expect("should not fail");
        assert_eq!(result.batch_size, 2);
        assert!(result.mean_loss > 0.0, "mean loss should be positive");
        assert!(result.mean_accuracy >= 0.0 && result.mean_accuracy <= 1.0);
    }

    // ── Test 11: trainer mean_accuracy ───────────────────────────────────
    #[test]
    fn test_trainer_mean_accuracy_all_correct() {
        let mut trainer = SimpoTrainer::new(SimpoConfig::default());
        let batch = vec![
            (reward(&[-0.5]), reward(&[-2.0])),
            (reward(&[-0.3]), reward(&[-3.0])),
        ];
        let _ = trainer.train_step(&batch).expect("should not fail");
        let acc = trainer.mean_accuracy();
        assert!((acc - 1.0).abs() < 1e-10, "expected mean_accuracy=1.0, got {acc}");
    }

    // ── Test 12: history length ───────────────────────────────────────────
    #[test]
    fn test_trainer_history_length() {
        let mut trainer = SimpoTrainer::new(SimpoConfig::default());
        let batch1 = vec![(reward(&[-0.5]), reward(&[-2.0]))];
        let batch2 = vec![
            (reward(&[-0.3]), reward(&[-3.0])),
            (reward(&[-0.7]), reward(&[-1.5])),
        ];
        let _ = trainer.train_step(&batch1).expect("should not fail");
        let _ = trainer.train_step(&batch2).expect("should not fail");
        assert_eq!(trainer.history().len(), 3, "history should have 3 entries total");
    }

    // ── Test 13: SimpoLossResult display ─────────────────────────────────
    #[test]
    fn test_simpo_loss_result_display() {
        let result = SimpoLossResult {
            loss: 0.4321,
            chosen_reward: -0.5,
            rejected_reward: -2.0,
            reward_diff: 1.0,
            accuracy: 1.0,
            loss_type: SimpoLossType::Sigmoid,
        };
        let s = format!("{result}");
        assert!(s.contains("SimpoLossResult"), "display should contain struct name");
        assert!(s.contains("0.4321"), "display should contain loss value");
        assert!(s.contains("sigmoid"), "display should contain loss type");
    }

    // ── Test 14: empty sequence error ────────────────────────────────────
    #[test]
    fn test_empty_sequence_error() {
        let mut loss_fn = SimpoLoss::new(SimpoConfig::default());
        let empty = SimpoReward::new(vec![]);
        let valid = reward(&[-1.0]);
        let err = loss_fn.compute_loss(&empty, &valid).unwrap_err();
        assert!(matches!(err, SimpoError::EmptySequence));
    }

    // ── Test 15: NaN reward error ─────────────────────────────────────────
    #[test]
    fn test_nan_reward_error() {
        let mut loss_fn = SimpoLoss::new(SimpoConfig::default());
        let nan_reward = SimpoReward { log_probs: vec![f64::NAN], length: 1 };
        let valid = reward(&[-1.0]);
        let err = loss_fn.compute_loss(&nan_reward, &valid).unwrap_err();
        assert!(matches!(err, SimpoError::NanReward));
    }

    // ── Test 16: compute_simpo_reward mean log-prob ───────────────────────
    #[test]
    fn test_compute_simpo_reward_value() {
        // mean([-1, -2, -3]) = -2
        let r = compute_simpo_reward(&[-1.0f32, -2.0, -3.0]);
        assert!((r - (-2.0f32)).abs() < 1e-5, "expected -2.0, got {r}");
    }

    // ── Test 17: compute_simpo_reward empty → 0.0 ─────────────────────────
    #[test]
    fn test_compute_simpo_reward_empty() {
        let r = compute_simpo_reward(&[]);
        assert_eq!(r, 0.0);
    }

    // ── Test 18: compute_simpo_loss basic numerical value ─────────────────
    #[test]
    fn test_compute_simpo_loss_basic_value() {
        // chosen lps = [-1.0] → r_w = -1.0
        // rejected lps = [-2.0] → r_l = -2.0
        // β=1, γ=0 → rd = -1 - (-2) = 1.0
        // loss = -log(σ(1.0)) = -log(1/(1+e^{-1})) ≈ 0.3133
        let config = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Basic, label_smoothing: 0.0 };
        let result = compute_simpo_loss(&[-1.0f32], &[-2.0f32], &config).expect("ok");
        let expected = -(1.0f32 / (1.0 + (-1.0f32).exp())).ln();
        assert!((result.loss - expected).abs() < 1e-5, "expected {expected}, got {}", result.loss);
    }

    // ── Test 19: compute_simpo_loss margin only (hinge) ──────────────────
    #[test]
    fn test_compute_simpo_loss_margin_only_zero() {
        // chosen >> rejected → hinge loss = 0
        let config = SimpoConfig {
            beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::MarginOnly, label_smoothing: 0.0,
        };
        let result = compute_simpo_loss(&[0.0f32], &[-100.0f32], &config).expect("ok");
        assert!(result.loss < 1e-5, "margin_only loss should be 0, got {}", result.loss);
    }

    // ── Test 20: compute_simpo_loss margin = chosen - rejected ────────────
    #[test]
    fn test_compute_simpo_loss_margin_field() {
        let config = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Basic, label_smoothing: 0.0 };
        let result = compute_simpo_loss(&[-1.0f32], &[-3.0f32], &config).expect("ok");
        // r_w = -1, r_l = -3, margin = 2
        assert!((result.margin - 2.0f32).abs() < 1e-5, "expected margin=2.0, got {}", result.margin);
    }

    // ── Test 21: compute_simpo_loss accuracy ──────────────────────────────
    #[test]
    fn test_compute_simpo_loss_accuracy_correct() {
        let config = SimpoConfig::default();
        let result = compute_simpo_loss(&[-0.5f32], &[-2.0f32], &config).expect("ok");
        assert!((result.accuracy - 1.0f32).abs() < 1e-5, "expected accuracy=1.0");
    }

    // ── Test 22: compute_simpo_loss empty chosen error ────────────────────
    #[test]
    fn test_compute_simpo_loss_empty_error() {
        let config = SimpoConfig::default();
        let err = compute_simpo_loss(&[], &[-1.0f32], &config).unwrap_err();
        assert!(matches!(err, SimpoError::EmptySequence));
    }

    // ── Test 23: Basic variant == Sigmoid variant ─────────────────────────
    #[test]
    fn test_basic_equals_sigmoid_variant() {
        let cfg_basic = SimpoConfig { loss_type: SimpoLossType::Basic, beta: 2.0, gamma: 0.5, label_smoothing: 0.0 };
        let cfg_sig   = SimpoConfig { loss_type: SimpoLossType::Sigmoid, beta: 2.0, gamma: 0.5, label_smoothing: 0.0 };
        let r1 = compute_simpo_loss(&[-1.0f32], &[-2.0f32], &cfg_basic).expect("ok");
        let r2 = compute_simpo_loss(&[-1.0f32], &[-2.0f32], &cfg_sig).expect("ok");
        assert!((r1.loss - r2.loss).abs() < 1e-5, "Basic and Sigmoid should produce same loss");
    }

    // ── Test 24: LabelSmoothed loss > Basic loss (increased uncertainty) ──
    #[test]
    fn test_label_smoothed_vs_basic() {
        let cfg_basic = SimpoConfig { loss_type: SimpoLossType::Basic, beta: 1.0, gamma: 0.0, label_smoothing: 0.0 };
        let cfg_ls    = SimpoConfig { loss_type: SimpoLossType::LabelSmoothed, beta: 1.0, gamma: 0.0, label_smoothing: 0.1 };
        // chosen much better than rejected → basic loss very small
        // label smoothed adds penalty from the negative direction
        let r_basic = compute_simpo_loss(&[0.0f32], &[-10.0f32], &cfg_basic).expect("ok");
        let r_ls    = compute_simpo_loss(&[0.0f32], &[-10.0f32], &cfg_ls).expect("ok");
        assert!(r_ls.loss >= r_basic.loss - 1e-4,
            "label-smoothed loss should be >= basic loss, got basic={} ls={}", r_basic.loss, r_ls.loss);
    }

    // ── Test 25: MarginOnly equals Hinge variant ──────────────────────────
    #[test]
    fn test_margin_only_equals_hinge() {
        let cfg_mo = SimpoConfig { loss_type: SimpoLossType::MarginOnly, beta: 1.0, gamma: 0.5, label_smoothing: 0.0 };
        let cfg_h  = SimpoConfig { loss_type: SimpoLossType::Hinge, beta: 1.0, gamma: 0.5, label_smoothing: 0.0 };
        let r_mo = compute_simpo_loss(&[-1.0f32], &[-2.0f32], &cfg_mo).expect("ok");
        let r_h  = compute_simpo_loss(&[-1.0f32], &[-2.0f32], &cfg_h).expect("ok");
        assert!((r_mo.loss - r_h.loss).abs() < 1e-5, "MarginOnly and Hinge should produce same loss");
    }

    // ── Test 26: gamma = 0 → SimpoLoss matches compute_simpo_loss ────────
    #[test]
    fn test_standalone_matches_struct_zero_gamma() {
        let config = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Basic, label_smoothing: 0.0 };
        let chosen_lps_f32  = [-1.0f32, -1.5, -2.0];
        let rejected_lps_f32 = [-3.0f32, -2.5, -2.0];

        let standalone = compute_simpo_loss(&chosen_lps_f32, &rejected_lps_f32, &config).expect("ok");

        // Verify chosen_reward matches compute_simpo_reward
        let expected_r_w = compute_simpo_reward(&chosen_lps_f32);
        assert!((standalone.chosen_reward - expected_r_w).abs() < 1e-5);
    }

    // ── Test 27: SimpoLossOutput fields correct for IPO ──────────────────
    #[test]
    fn test_compute_simpo_loss_ipo_value() {
        // β=2, γ=0, r_w=r_l → rd=0 → IPO = (0/2 - 1/(2*2))^2 = 0.0625
        let config = SimpoConfig { beta: 2.0, gamma: 0.0, loss_type: SimpoLossType::IPO, label_smoothing: 0.0 };
        let result = compute_simpo_loss(&[-1.0f32], &[-1.0f32], &config).expect("ok");
        assert!((result.loss - 0.0625f32).abs() < 1e-5, "IPO loss expected 0.0625, got {}", result.loss);
    }

    // ── New extended tests ─────────────────────────────────────────────────

    // Test: SimPO reward = mean log-prob (length-normalized)
    #[test]
    fn test_simpo_reward_is_mean_log_prob() {
        // reward = sum(log_probs) / length
        let lps = [-1.0_f64, -2.0, -3.0, -4.0]; // sum=-10, len=4 → mean=-2.5
        let r = reward(&lps);
        assert!((r.reward() - (-2.5)).abs() < 1e-10, "expected -2.5, got {}", r.reward());
    }

    // Test: longer sequences don't inherently have lower reward (length normalization ensures this)
    #[test]
    fn test_longer_sequences_not_penalized_by_length_normalization() {
        // Short sequence: 2 tokens, each log-prob = -1.0 → reward = -1.0
        let short_r = reward(&[-1.0_f64, -1.0]);
        // Long sequence: 8 tokens, each log-prob = -1.0 → reward = -1.0 (same per-token)
        let long_r = reward(&[-1.0_f64, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        assert!(
            (short_r.reward() - long_r.reward()).abs() < 1e-10,
            "short and long sequences with same per-token lp should have same reward: short={} long={}",
            short_r.reward(), long_r.reward()
        );
    }

    // Test: margin γ — loss is 0 when reward gap > γ (for MarginOnly variant)
    #[test]
    fn test_margin_gamma_loss_zero_when_gap_exceeds_gamma() {
        // MarginOnly: loss = max(0, -reward_diff) where reward_diff = β*(r_w-r_l) - γ
        // r_w = -0.5, r_l = -5.0, β = 1, γ = 1.0
        // reward_diff = 1*(−0.5 − (−5.0)) − 1.0 = 4.5 − 1.0 = 3.5 > 0 → loss = 0
        let config = SimpoConfig {
            beta: 1.0,
            gamma: 1.0,
            loss_type: SimpoLossType::MarginOnly,
            label_smoothing: 0.0,
        };
        let mut loss_fn = SimpoLoss::new(config);
        let chosen = reward(&[-0.5_f64]);
        let rejected = reward(&[-5.0_f64]);
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("ok");
        assert!(result.loss < 1e-10, "MarginOnly loss should be 0 when gap > gamma, got {}", result.loss);
    }

    // Test: accuracy metric = 1.0 when chosen >> rejected
    #[test]
    fn test_accuracy_one_when_chosen_greatly_exceeds_rejected() {
        let mut loss_fn = SimpoLoss::new(SimpoConfig::default());
        let chosen = reward(&[0.0_f64]); // reward = 0
        let rejected = reward(&[-100.0_f64]); // reward = -100
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("ok");
        assert!((result.accuracy - 1.0).abs() < 1e-10, "accuracy should be 1.0, got {}", result.accuracy);
    }

    // Test: SimPO vs DPO — SimPO doesn't need reference model log-probs
    #[test]
    fn test_simpo_does_not_require_reference_log_probs() {
        // SimPO only needs chosen and rejected log-probs (no ref model)
        // We verify the SimpoReward structure has no ref_log_probs field
        let r = SimpoReward::new(vec![-1.0_f64, -2.0]);
        // Only has log_probs and length — no reference log-probs
        assert_eq!(r.log_probs.len(), 2);
        assert_eq!(r.length, 2);
        // Reward is computed purely from policy log-probs
        assert!((r.reward() - (-1.5)).abs() < 1e-10, "reward should be mean of log_probs");
    }

    // Test: identical sequences produce margin=0 edge case
    #[test]
    fn test_identical_sequences_margin_zero() {
        let lps = vec![-1.0_f64, -2.0, -1.5];
        let mut loss_fn = SimpoLoss::new(SimpoConfig { beta: 1.0, gamma: 0.0, ..Default::default() });
        let chosen = reward(&lps);
        let rejected = reward(&lps); // identical
        let result = loss_fn.compute_loss(&chosen, &rejected).expect("ok");
        // r_w = r_l → reward_diff = β*(r_w-r_l) - γ = -γ = 0 (since gamma=0)
        assert!(result.reward_diff.abs() < 1e-10, "reward_diff should be 0 for identical sequences, got {}", result.reward_diff);
        // loss = -log(σ(0)) = log(2)
        let expected_loss = (2.0_f64).ln();
        assert!((result.loss - expected_loss).abs() < 1e-9, "loss for identical seqs should be log(2), got {}", result.loss);
    }

    // Test: SimPO Basic == Sigmoid variant (same loss)
    #[test]
    fn test_basic_and_sigmoid_variants_produce_same_loss() {
        let cfg_basic = SimpoConfig { loss_type: SimpoLossType::Basic, beta: 1.5, gamma: 0.5, label_smoothing: 0.0 };
        let cfg_sig = SimpoConfig { loss_type: SimpoLossType::Sigmoid, beta: 1.5, gamma: 0.5, label_smoothing: 0.0 };
        let mut lf_basic = SimpoLoss::new(cfg_basic);
        let mut lf_sig = SimpoLoss::new(cfg_sig);
        let chosen = reward(&[-0.5_f64, -1.0]);
        let rejected = reward(&[-2.0_f64, -2.5]);
        let r_basic = lf_basic.compute_loss(&chosen, &rejected).expect("ok");
        let r_sig = lf_sig.compute_loss(&chosen, &rejected).expect("ok");
        assert!((r_basic.loss - r_sig.loss).abs() < 1e-10,
            "Basic and Sigmoid should produce same loss: basic={} sig={}", r_basic.loss, r_sig.loss);
    }

    // Test: batch size independence — mean loss same for batch of 1 vs N identical pairs
    #[test]
    fn test_batch_size_independence_mean_loss() {
        let config = SimpoConfig { beta: 1.0, gamma: 0.5, loss_type: SimpoLossType::Sigmoid, label_smoothing: 0.0 };

        // Single pair
        let mut trainer_1 = SimpoTrainer::new(config.clone());
        let chosen_r = SimpoReward::new(vec![-1.0_f64]);
        let rejected_r = SimpoReward::new(vec![-2.0_f64]);
        let r1 = trainer_1.train_step(&[(chosen_r.clone(), rejected_r.clone())]).expect("ok");

        // Batch of 5 identical pairs
        let mut trainer_5 = SimpoTrainer::new(config);
        let batch_5 = vec![
            (chosen_r.clone(), rejected_r.clone()),
            (chosen_r.clone(), rejected_r.clone()),
            (chosen_r.clone(), rejected_r.clone()),
            (chosen_r.clone(), rejected_r.clone()),
            (chosen_r, rejected_r),
        ];
        let r5 = trainer_5.train_step(&batch_5).expect("ok");

        assert!((r1.mean_loss - r5.mean_loss).abs() < 1e-10,
            "mean loss should be same for batch of 1 vs 5 identical pairs: r1={} r5={}", r1.mean_loss, r5.mean_loss);
    }

    // Test: numerical stability with very negative log-probs
    #[test]
    fn test_numerical_stability_very_negative_log_probs() {
        // Very negative log-probs (but with reasonable reward difference) shouldn't cause NaN.
        // We use chosen < rejected so the sigmoid argument doesn't overflow f32.
        // rd = beta * (r_w - r_l) - gamma; we keep |rd| < 20 to avoid overflow.
        // r_w = mean([-10, -12]) = -11; r_l = mean([-8, -9]) = -8.5
        // rd = 2.5*(-11 - (-8.5)) - 1.4 = 2.5*(-2.5) - 1.4 = -6.25 - 1.4 = -7.65 (small enough)
        let config = SimpoConfig::default();
        let mut loss_fn = SimpoLoss::new(config.clone());
        let chosen = reward(&[-10.0_f64, -12.0]);
        let rejected = reward(&[-8.0_f64, -9.0]);
        let result = loss_fn.compute_loss(&chosen, &rejected);
        assert!(result.is_ok(), "should not fail with very negative log-probs");
        let r = result.expect("ok");
        assert!(r.loss.is_finite(), "loss must be finite, got {}", r.loss);

        // Also test standalone function with same bounded inputs
        let r2 = compute_simpo_loss(
            &[-10.0f32, -12.0],
            &[-8.0f32, -9.0],
            &config,
        ).expect("ok");
        assert!(r2.loss.is_finite(), "standalone loss must be finite");
    }

    // Test: gradient direction — decreasing loss when chosen reward increases
    #[test]
    fn test_gradient_direction_loss_decreases_with_chosen_reward_increase() {
        let config = SimpoConfig { beta: 1.0, gamma: 0.0, loss_type: SimpoLossType::Sigmoid, label_smoothing: 0.0 };
        let mut loss_fn = SimpoLoss::new(config);

        // Baseline: chosen reward = -2.0
        let chosen_base = reward(&[-2.0_f64]);
        let rejected = reward(&[-3.0_f64]);
        let r_base = loss_fn.compute_loss(&chosen_base, &rejected).expect("ok");

        // After gradient step: chosen reward improves to -0.5
        let chosen_improved = reward(&[-0.5_f64]);
        let r_improved = loss_fn.compute_loss(&chosen_improved, &rejected).expect("ok");

        assert!(r_improved.loss < r_base.loss,
            "loss should decrease as chosen reward increases: base={} improved={}", r_base.loss, r_improved.loss);
    }
}
