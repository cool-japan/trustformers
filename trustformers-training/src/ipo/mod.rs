//! IPO: Identity Preference Optimization
//! Reference: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
//! (Azar et al., 2024)
//!
//! Unlike DPO which optimizes a surrogate MLE objective that can overfit, IPO directly
//! optimizes the preference oracle. IPO uses a squared L2 loss instead of logistic loss
//! (DPO) to prevent reward overoptimization.
//!
//! ## Loss Formulation
//!
//! For a pair (x, y_w, y_l):
//!   `h_θ(x, y_w, y_l) = (log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))`
//!
//! IPO loss:
//!   `L_IPO = E[(h_θ(x, y_w, y_l) - 1/(2β))²]`
//!
//! The 1/(2β) target is derived from the optimal solution to the KL-constrained optimization
//! problem. Higher β means the model should stay closer to the reference (smaller target gap).
//!
//! ## Comparison with DPO
//!
//! DPO loss: `L_DPO = -E[log σ(β * h_θ(x, y_w, y_l))]`
//!
//! IPO replaces the logistic loss with squared L2 loss, which prevents the unbounded
//! reward margin growth that DPO can suffer from.

use std::fmt;

/// A preference pair for IPO training.
///
/// Each pair consists of a prompt x with a chosen (preferred) response y_w
/// and a rejected response y_l, along with their log probabilities under
/// both the policy and reference models.
#[derive(Debug, Clone)]
pub struct IpoPair {
    /// Log probability of preferred completion under policy model: log π_θ(y_w|x)
    pub policy_log_prob_chosen: f32,
    /// Log probability of preferred completion under reference model: log π_ref(y_w|x)
    pub reference_log_prob_chosen: f32,
    /// Log probability of rejected completion under policy model: log π_θ(y_l|x)
    pub policy_log_prob_rejected: f32,
    /// Log probability of rejected completion under reference model: log π_ref(y_l|x)
    pub reference_log_prob_rejected: f32,
}

impl IpoPair {
    /// Compute `h_θ(x, y_w, y_l)` — the log-ratio difference between chosen and rejected.
    ///
    /// `h_θ = (log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))`
    ///
    /// Positive h_θ indicates the policy prefers the chosen over rejected.
    /// The IPO objective pushes h_θ toward 1/(2β).
    pub fn h_theta(&self) -> f32 {
        let chosen_log_ratio = self.policy_log_prob_chosen - self.reference_log_prob_chosen;
        let rejected_log_ratio = self.policy_log_prob_rejected - self.reference_log_prob_rejected;
        chosen_log_ratio - rejected_log_ratio
    }

    /// Compute the DPO-style log-ratio (same as h_theta, used for comparison).
    ///
    /// This is the same quantity used in DPO's logistic loss, provided here
    /// to enable direct comparison between IPO and DPO on the same data.
    pub fn dpo_log_ratio(&self) -> f32 {
        self.h_theta()
    }
}

/// IPO configuration parameters.
#[derive(Debug, Clone)]
pub struct IpoConfig {
    /// KL penalty strength β — higher β means stronger constraint toward reference model.
    /// The target h_θ value is 1/(2β), so higher β leads to a smaller target gap.
    pub beta: f32,
    /// Label smoothing coefficient ε for robustness.
    ///
    /// When ε > 0, the target is smoothed: target = (1 - ε) / (2β).
    /// This can prevent overconfidence in labels.
    /// Default: 0.0 (no smoothing)
    pub label_smoothing: f32,
}

impl Default for IpoConfig {
    fn default() -> Self {
        Self { beta: 0.1, label_smoothing: 0.0 }
    }
}

/// Result of an IPO loss computation for a batch of pairs.
#[derive(Debug, Clone)]
pub struct IpoLossResult {
    /// Mean IPO loss over the batch: E[(h_θ - 1/(2β))²]
    pub loss: f32,
    /// Mean h_θ(x, y_w, y_l) over the batch
    pub mean_h_theta: f32,
    /// Target value 1/(2β) (or smoothed version)
    pub target: f32,
    /// Mean squared deviation from target: E[(h_θ - target)²]
    pub mean_squared_deviation: f32,
    /// Fraction of pairs where the policy prefers chosen over rejected (h_θ > 0)
    pub preference_accuracy: f32,
    /// Number of pairs in the batch
    pub num_pairs: usize,
}

/// Compute the sigmoid function: σ(x) = 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the IPO loss for a batch of preference pairs.
///
/// ## Algorithm
///
/// 1. For each pair, compute h_θ = (chosen_log_ratio) - (rejected_log_ratio)
/// 2. Compute target = (1 - label_smoothing) / (2β)
/// 3. For each pair: `loss_i = (h_θ_i - target)²`
/// 4. Return mean loss over the batch
///
/// ## Errors
///
/// Returns `IpoError::EmptyBatch` if no pairs are provided.
/// Returns `IpoError::InvalidBeta` if β ≤ 0 (target would be undefined or infinite).
/// Returns `IpoError::NumericalError` if NaN or Inf values are encountered.
pub fn compute_ipo_loss(pairs: &[IpoPair], config: &IpoConfig) -> Result<IpoLossResult, IpoError> {
    if pairs.is_empty() {
        return Err(IpoError::EmptyBatch);
    }
    if config.beta <= 0.0 {
        return Err(IpoError::InvalidBeta(format!(
            "beta must be positive, got {}",
            config.beta
        )));
    }

    // Target: 1/(2β) with optional label smoothing
    // With smoothing ε: target = (1 - ε) / (2β)
    let target = (1.0 - config.label_smoothing) / (2.0 * config.beta);

    let h_thetas: Vec<f32> = pairs.iter().map(|p| p.h_theta()).collect();

    let squared_deviations: Vec<f32> = h_thetas.iter().map(|&h| (h - target).powi(2)).collect();

    let mean_h_theta = h_thetas.iter().sum::<f32>() / (h_thetas.len() as f32);
    let mean_squared_deviation =
        squared_deviations.iter().sum::<f32>() / (squared_deviations.len() as f32);

    // IPO loss = mean squared deviation
    let loss = mean_squared_deviation;

    if loss.is_nan() || loss.is_infinite() {
        return Err(IpoError::NumericalError(format!("IPO loss is not finite: {loss}")));
    }

    // Preference accuracy: fraction where h_θ > 0 (policy prefers chosen)
    let num_preferred = h_thetas.iter().filter(|&&h| h > 0.0).count();
    let preference_accuracy = num_preferred as f32 / h_thetas.len() as f32;

    Ok(IpoLossResult {
        loss,
        mean_h_theta,
        target,
        mean_squared_deviation,
        preference_accuracy,
        num_pairs: pairs.len(),
    })
}

/// Comparison of IPO and DPO behavior on the same set of pairs.
///
/// Useful for analyzing how the two objectives differ on the same data,
/// particularly regarding reward overoptimization risk.
#[derive(Debug, Clone)]
pub struct IpoDpoComparison {
    /// IPO loss: E[(h_θ - 1/(2β))²]
    pub ipo_loss: f32,
    /// DPO loss: -mean(log σ(β * h_θ))
    pub dpo_loss: f32,
    /// Mean h_θ under IPO computation
    pub ipo_h_theta_mean: f32,
    /// Mean h_θ under DPO computation (same data, same metric)
    pub dpo_h_theta_mean: f32,
    /// Whether IPO and DPO agree on which direction is preferred (both give h_θ > 0 or both < 0)
    pub preference_agreement: bool,
}

/// Compare IPO vs DPO behavior on the same set of pairs.
///
/// Computes both losses on the same data to enable side-by-side analysis
/// of how the two objectives behave.
///
/// DPO loss uses: `-mean(log σ(β * h_θ(x, y_w, y_l)))`
///
/// ## Errors
///
/// Returns `IpoError::EmptyBatch` if no pairs are provided.
/// Returns `IpoError::InvalidBeta` if β ≤ 0.
pub fn compare_ipo_dpo(
    pairs: &[IpoPair],
    config: &IpoConfig,
) -> Result<IpoDpoComparison, IpoError> {
    if pairs.is_empty() {
        return Err(IpoError::EmptyBatch);
    }
    if config.beta <= 0.0 {
        return Err(IpoError::InvalidBeta(format!(
            "beta must be positive, got {}",
            config.beta
        )));
    }

    let h_thetas: Vec<f32> = pairs.iter().map(|p| p.h_theta()).collect();

    // IPO loss
    let target = (1.0 - config.label_smoothing) / (2.0 * config.beta);
    let ipo_loss =
        h_thetas.iter().map(|&h| (h - target).powi(2)).sum::<f32>() / h_thetas.len() as f32;

    // DPO loss: -mean(log σ(β * h_θ))
    let dpo_loss = h_thetas
        .iter()
        .map(|&h| {
            let logit = config.beta * h;
            let sig = sigmoid(logit);
            // Clamp to avoid log(0)
            let sig_clamped = sig.max(1e-7);
            -sig_clamped.ln()
        })
        .sum::<f32>()
        / h_thetas.len() as f32;

    let ipo_h_theta_mean = h_thetas.iter().sum::<f32>() / h_thetas.len() as f32;
    // dpo_h_theta_mean is the same (same data, same h_theta computation)
    let dpo_h_theta_mean = ipo_h_theta_mean;

    // Agreement: both should prefer the same direction (same sign of mean h_theta)
    let preference_agreement =
        (ipo_h_theta_mean > 0.0) == (dpo_h_theta_mean > 0.0)
        || (ipo_h_theta_mean == 0.0 && dpo_h_theta_mean == 0.0);

    Ok(IpoDpoComparison {
        ipo_loss,
        dpo_loss,
        ipo_h_theta_mean,
        dpo_h_theta_mean,
        preference_agreement,
    })
}

/// IPO trainer that accumulates preference pairs across steps and tracks training history.
///
/// Provides convergence checking and history analysis to monitor training progress.
#[derive(Debug, Clone)]
pub struct IpoTrainer {
    /// Configuration for IPO loss computation.
    pub config: IpoConfig,
    history: Vec<IpoLossResult>,
}

impl IpoTrainer {
    /// Create a new IPO trainer with the given configuration.
    pub fn new(config: IpoConfig) -> Self {
        Self { config, history: Vec::new() }
    }

    /// Compute the IPO loss for a batch of pairs and record the result.
    ///
    /// The result is appended to the trainer's history for later analysis.
    pub fn step(&mut self, pairs: &[IpoPair]) -> Result<IpoLossResult, IpoError> {
        let result = compute_ipo_loss(pairs, &self.config)?;
        self.history.push(result.clone());
        Ok(result)
    }

    /// Return all historical IPO loss results across training steps.
    pub fn history(&self) -> &[IpoLossResult] {
        &self.history
    }

    /// Compute the mean loss over all recorded history.
    ///
    /// Returns 0.0 if no history is available.
    pub fn mean_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.loss).sum();
        sum / (self.history.len() as f32)
    }

    /// Check whether training has converged.
    ///
    /// Returns `true` if the last 3 losses are all within 0.01 of each other.
    /// Returns `false` if fewer than 3 steps have been recorded.
    pub fn convergence_check(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let last_three = &self.history[self.history.len() - 3..];
        let max_loss = last_three.iter().map(|r| r.loss).fold(f32::NEG_INFINITY, f32::max);
        let min_loss = last_three.iter().map(|r| r.loss).fold(f32::INFINITY, f32::min);
        (max_loss - min_loss) < 0.01
    }
}

/// Errors that can occur during IPO loss computation.
#[derive(Debug, Clone)]
pub enum IpoError {
    /// The batch of pairs was empty.
    EmptyBatch,
    /// The beta parameter was invalid (must be positive).
    InvalidBeta(String),
    /// A numerical error (NaN or Inf) was encountered during computation.
    NumericalError(String),
}

impl fmt::Display for IpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpoError::EmptyBatch => write!(f, "IPO error: batch is empty"),
            IpoError::InvalidBeta(msg) => write!(f, "IPO error: invalid beta — {msg}"),
            IpoError::NumericalError(msg) => write!(f, "IPO numerical error: {msg}"),
        }
    }
}

impl std::error::Error for IpoError {}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Configuration tests ──────────────────────────────────────────────────

    #[test]
    fn test_ipo_config_default() {
        let config = IpoConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.label_smoothing, 0.0);
    }

    // ── IpoPair / h_theta tests ───────────────────────────────────────────────

    #[test]
    fn test_ipo_pair_h_theta_positive() {
        // Policy favors chosen over rejected (higher log_ratio for chosen)
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0, // chosen log_ratio = 1.0
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0, // rejected log_ratio = -1.0
        };
        // h_theta = 1.0 - (-1.0) = 2.0
        assert!(
            (pair.h_theta() - 2.0).abs() < 1e-6,
            "Expected h_theta=2.0, got {}",
            pair.h_theta()
        );
    }

    #[test]
    fn test_ipo_pair_h_theta_negative() {
        // Policy favors rejected over chosen (negative h_theta)
        let pair = IpoPair {
            policy_log_prob_chosen: -3.0,
            reference_log_prob_chosen: -2.0, // chosen log_ratio = -1.0
            policy_log_prob_rejected: -1.0,
            reference_log_prob_rejected: -2.0, // rejected log_ratio = 1.0
        };
        // h_theta = -1.0 - 1.0 = -2.0
        assert!(
            (pair.h_theta() - (-2.0)).abs() < 1e-6,
            "Expected h_theta=-2.0, got {}",
            pair.h_theta()
        );
    }

    #[test]
    fn test_ipo_pair_h_theta_zero() {
        // Policy treats chosen and rejected the same
        let pair = IpoPair {
            policy_log_prob_chosen: -2.0,
            reference_log_prob_chosen: -2.0, // chosen log_ratio = 0.0
            policy_log_prob_rejected: -1.5,
            reference_log_prob_rejected: -1.5, // rejected log_ratio = 0.0
        };
        assert!(
            pair.h_theta().abs() < 1e-6,
            "Expected h_theta=0.0, got {}",
            pair.h_theta()
        );
    }

    #[test]
    fn test_ipo_pair_dpo_log_ratio() {
        // dpo_log_ratio should be the same as h_theta
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        };
        assert_eq!(pair.h_theta(), pair.dpo_log_ratio());
    }

    // ── compute_ipo_loss tests ────────────────────────────────────────────────

    #[test]
    fn test_ipo_loss_single_pair() {
        // h_theta = 2.0, beta=0.1, target = 1/(2*0.1) = 5.0
        // loss = (2.0 - 5.0)^2 = 9.0
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        };
        let config = IpoConfig { beta: 0.1, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");

        assert_eq!(result.num_pairs, 1);
        let expected_target = 5.0f32; // 1/(2*0.1)
        assert!(
            (result.target - expected_target).abs() < 1e-5,
            "Expected target {expected_target}, got {}",
            result.target
        );
        let expected_loss = (2.0f32 - 5.0).powi(2);
        assert!(
            (result.loss - expected_loss).abs() < 1e-5,
            "Expected loss {expected_loss}, got {}",
            result.loss
        );
    }

    #[test]
    fn test_ipo_loss_batch() {
        // Multiple pairs, verify mean is computed correctly
        let pairs = vec![
            IpoPair {
                // h_theta = (0 - (-1)) - (-1 - (-1)) = 1 - 0 = 1.0
                policy_log_prob_chosen: -1.0,
                reference_log_prob_chosen: -1.0,
                policy_log_prob_rejected: -2.0,
                reference_log_prob_rejected: -1.0,
            },
            IpoPair {
                // h_theta = ((-1) - (-2)) - ((-2) - (-1)) = 1 - (-1) = 2.0
                policy_log_prob_chosen: -1.0,
                reference_log_prob_chosen: -2.0,
                policy_log_prob_rejected: -2.0,
                reference_log_prob_rejected: -1.0,
            },
        ];
        let config = IpoConfig { beta: 0.5, label_smoothing: 0.0 };
        // target = 1/(2*0.5) = 1.0
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        assert_eq!(result.num_pairs, 2);
        assert!((result.target - 1.0).abs() < 1e-5, "Expected target=1.0");
        // h_thetas: [1.0, 2.0], mean=1.5
        assert!(
            (result.mean_h_theta - 1.5).abs() < 1e-5,
            "Expected mean_h_theta=1.5, got {}",
            result.mean_h_theta
        );
        // losses: (1-1)^2=0, (2-1)^2=1 => mean = 0.5
        assert!(
            (result.loss - 0.5).abs() < 1e-5,
            "Expected loss=0.5, got {}",
            result.loss
        );
    }

    #[test]
    fn test_ipo_loss_target_is_half_over_beta() {
        for &beta in &[0.1f32, 0.5, 1.0, 2.0, 5.0] {
            let pairs = vec![IpoPair {
                policy_log_prob_chosen: -1.0,
                reference_log_prob_chosen: -2.0,
                policy_log_prob_rejected: -2.0,
                reference_log_prob_rejected: -1.0,
            }];
            let config = IpoConfig { beta, label_smoothing: 0.0 };
            let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
            let expected_target = 1.0 / (2.0 * beta);
            assert!(
                (result.target - expected_target).abs() < 1e-5,
                "For beta={beta}, expected target={expected_target}, got {}",
                result.target
            );
        }
    }

    #[test]
    fn test_ipo_loss_label_smoothing() {
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -2.0,
            reference_log_prob_rejected: -1.0,
        };
        let config_no_smooth = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let config_smooth = IpoConfig { beta: 1.0, label_smoothing: 0.1 };

        let result_no_smooth =
            compute_ipo_loss(&[pair.clone()], &config_no_smooth).expect("should succeed");
        let result_smooth = compute_ipo_loss(&[pair], &config_smooth).expect("should succeed");

        // target_no_smooth = 1/(2*1) = 0.5
        // target_smooth = (1 - 0.1)/(2*1) = 0.45
        assert!(
            (result_no_smooth.target - 0.5).abs() < 1e-5,
            "Expected 0.5, got {}",
            result_no_smooth.target
        );
        assert!(
            (result_smooth.target - 0.45).abs() < 1e-5,
            "Expected 0.45, got {}",
            result_smooth.target
        );
        // The losses should differ
        assert!(
            (result_no_smooth.loss - result_smooth.loss).abs() > 1e-6,
            "Different smoothing should produce different losses"
        );
    }

    #[test]
    fn test_ipo_loss_higher_beta_lower_target() {
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -1.5,
            policy_log_prob_rejected: -2.0,
            reference_log_prob_rejected: -1.5,
        };
        let config_low_beta = IpoConfig { beta: 0.1, ..IpoConfig::default() };
        let config_high_beta = IpoConfig { beta: 2.0, ..IpoConfig::default() };

        let result_low = compute_ipo_loss(&[pair.clone()], &config_low_beta).expect("should succeed");
        let result_high = compute_ipo_loss(&[pair], &config_high_beta).expect("should succeed");

        // Higher beta => smaller target (1/(2β))
        assert!(
            result_high.target < result_low.target,
            "Higher beta should give lower target: {} vs {}",
            result_high.target,
            result_low.target
        );
    }

    #[test]
    fn test_ipo_preference_accuracy_all_correct() {
        // All pairs have positive h_theta (policy correctly prefers chosen)
        let pairs: Vec<IpoPair> = (0..5)
            .map(|i| IpoPair {
                policy_log_prob_chosen: -1.0 - i as f32 * 0.1,
                reference_log_prob_chosen: -2.0 - i as f32 * 0.1,
                policy_log_prob_rejected: -3.0,
                reference_log_prob_rejected: -2.0,
            })
            .collect();
        let config = IpoConfig::default();
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        assert!(
            (result.preference_accuracy - 1.0).abs() < 1e-6,
            "Expected accuracy=1.0, got {}",
            result.preference_accuracy
        );
    }

    #[test]
    fn test_ipo_preference_accuracy_mixed() {
        let pairs = vec![
            IpoPair {
                // h_theta > 0 (correct)
                policy_log_prob_chosen: -1.0,
                reference_log_prob_chosen: -2.0,
                policy_log_prob_rejected: -3.0,
                reference_log_prob_rejected: -2.0,
            },
            IpoPair {
                // h_theta < 0 (incorrect)
                policy_log_prob_chosen: -3.0,
                reference_log_prob_chosen: -2.0,
                policy_log_prob_rejected: -1.0,
                reference_log_prob_rejected: -2.0,
            },
        ];
        let config = IpoConfig::default();
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        assert!(
            (result.preference_accuracy - 0.5).abs() < 1e-6,
            "Expected accuracy=0.5, got {}",
            result.preference_accuracy
        );
    }

    // ── compare_ipo_dpo tests ─────────────────────────────────────────────────

    #[test]
    fn test_ipo_compare_ipo_dpo() {
        let pairs = vec![
            IpoPair {
                policy_log_prob_chosen: -1.0,
                reference_log_prob_chosen: -2.0,
                policy_log_prob_rejected: -3.0,
                reference_log_prob_rejected: -2.0,
            },
            IpoPair {
                policy_log_prob_chosen: -1.5,
                reference_log_prob_chosen: -2.5,
                policy_log_prob_rejected: -2.5,
                reference_log_prob_rejected: -1.5,
            },
        ];
        let config = IpoConfig { beta: 0.5, ..IpoConfig::default() };
        let comparison = compare_ipo_dpo(&pairs, &config).expect("should succeed");

        // Both losses should be positive and finite
        assert!(comparison.ipo_loss >= 0.0, "IPO loss should be non-negative");
        assert!(comparison.dpo_loss >= 0.0, "DPO loss should be non-negative");
        assert!(comparison.ipo_loss.is_finite(), "IPO loss should be finite");
        assert!(comparison.dpo_loss.is_finite(), "DPO loss should be finite");

        // h_theta means should match
        assert!(
            (comparison.ipo_h_theta_mean - comparison.dpo_h_theta_mean).abs() < 1e-6,
            "IPO and DPO should compute the same h_theta"
        );

        // Both have same direction (positive h_theta), so they should agree
        assert!(comparison.preference_agreement, "IPO and DPO should agree on preference direction");
    }

    // ── IpoTrainer tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ipo_trainer_step() {
        let config = IpoConfig::default();
        let mut trainer = IpoTrainer::new(config);

        let pairs = vec![IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        }];

        let result = trainer.step(&pairs).expect("step should succeed");
        assert_eq!(result.num_pairs, 1);
        assert_eq!(trainer.history().len(), 1);
    }

    #[test]
    fn test_ipo_trainer_history() {
        let config = IpoConfig::default();
        let mut trainer = IpoTrainer::new(config);

        // Initially no history
        assert_eq!(trainer.history().len(), 0);
        assert_eq!(trainer.mean_loss(), 0.0);

        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        };

        trainer.step(&[pair.clone()]).expect("step 1");
        trainer.step(&[pair.clone()]).expect("step 2");
        trainer.step(&[pair]).expect("step 3");

        assert_eq!(trainer.history().len(), 3);
        // All steps identical → mean equals individual step loss
        let step_loss = trainer.history()[0].loss;
        assert!(
            (trainer.mean_loss() - step_loss).abs() < 1e-6,
            "Mean should equal individual step loss for identical steps"
        );
    }

    #[test]
    fn test_ipo_trainer_convergence() {
        let config = IpoConfig::default();
        let mut trainer = IpoTrainer::new(config);

        // Fewer than 3 steps → no convergence
        assert!(!trainer.convergence_check());

        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        };

        // 3 identical steps → converged (all losses identical, difference = 0 < 0.01)
        trainer.step(&[pair.clone()]).expect("step 1");
        trainer.step(&[pair.clone()]).expect("step 2");
        trainer.step(&[pair]).expect("step 3");
        assert!(trainer.convergence_check(), "3 identical steps should be converged");

        // Now add a step with very different loss
        let diverging_pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -100.0, // very large h_theta
            policy_log_prob_rejected: -100.0,
            reference_log_prob_rejected: -1.0,
        };
        trainer.step(&[diverging_pair]).expect("diverging step");
        assert!(!trainer.convergence_check(), "Diverging loss should not be converged");
    }

    // ── Error display tests ───────────────────────────────────────────────────

    #[test]
    fn test_ipo_error_display() {
        let empty = IpoError::EmptyBatch;
        let invalid_beta = IpoError::InvalidBeta("must be positive".to_string());
        let numerical = IpoError::NumericalError("Inf detected".to_string());

        assert!(empty.to_string().contains("empty"), "EmptyBatch should mention 'empty'");
        assert!(
            invalid_beta.to_string().contains("beta"),
            "InvalidBeta should mention 'beta'"
        );
        assert!(
            numerical.to_string().contains("Inf detected"),
            "NumericalError should contain the message"
        );
    }

    // ── Edge case tests ───────────────────────────────────────────────────────

    #[test]
    fn test_ipo_loss_empty_batch() {
        let config = IpoConfig::default();
        let result = compute_ipo_loss(&[], &config);
        assert!(
            matches!(result, Err(IpoError::EmptyBatch)),
            "Empty batch should return EmptyBatch error"
        );
    }

    #[test]
    fn test_ipo_loss_invalid_beta() {
        let pair = IpoPair {
            policy_log_prob_chosen: -1.0,
            reference_log_prob_chosen: -2.0,
            policy_log_prob_rejected: -3.0,
            reference_log_prob_rejected: -2.0,
        };
        let config_zero = IpoConfig { beta: 0.0, ..IpoConfig::default() };
        let config_neg = IpoConfig { beta: -1.0, ..IpoConfig::default() };

        assert!(
            matches!(compute_ipo_loss(&[pair.clone()], &config_zero), Err(IpoError::InvalidBeta(_))),
            "Zero beta should return InvalidBeta error"
        );
        assert!(
            matches!(compute_ipo_loss(&[pair], &config_neg), Err(IpoError::InvalidBeta(_))),
            "Negative beta should return InvalidBeta error"
        );
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn make_pair_with_h_theta(h_theta: f32) -> IpoPair {
        IpoPair {
            policy_log_prob_chosen: h_theta,
            reference_log_prob_chosen: 0.0,
            policy_log_prob_rejected: 0.0,
            reference_log_prob_rejected: 0.0,
        }
    }

    #[test]
    fn test_ipo_loss_formula_direct() {
        let pair = make_pair_with_h_theta(3.0);
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
        let expected_target = 0.5_f32;
        let expected_loss = (3.0_f32 - expected_target).powi(2);
        assert!((result.target - expected_target).abs() < 1e-5,
            "Expected target {expected_target}, got {}", result.target);
        assert!((result.loss - expected_loss).abs() < 1e-5,
            "Expected loss {expected_loss}, got {}", result.loss);
    }

    #[test]
    fn test_ipo_optimal_point_zero_loss() {
        // beta=0.5, target=1.0; h_theta=1.0 → loss = 0
        let pair = make_pair_with_h_theta(1.0);
        let config = IpoConfig { beta: 0.5, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
        assert!(result.loss < 1e-5, "Loss should be ~0 at optimal point, got {}", result.loss);
    }

    #[test]
    fn test_ipo_loss_always_non_negative() {
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        for &h in &[-5.0_f32, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0] {
            let pair = make_pair_with_h_theta(h);
            let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
            assert!(result.loss >= 0.0,
                "Loss should be non-negative for h_theta={h}: loss={}", result.loss);
        }
    }

    #[test]
    fn test_ipo_gradient_direction_below_target() {
        let pair = make_pair_with_h_theta(0.1);
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
        assert!(result.loss > 0.0, "Loss should be positive when h_theta < target");
        let expected = (0.1_f32 - 0.5).powi(2);
        assert!((result.loss - expected).abs() < 1e-5,
            "Expected {expected}, got {}", result.loss);
    }

    #[test]
    fn test_ipo_gradient_direction_above_target() {
        let pair = make_pair_with_h_theta(2.0);
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
        assert!(result.loss > 0.0, "Loss should be positive when h_theta > target");
        let expected = (2.0_f32 - 0.5).powi(2);
        assert!((result.loss - expected).abs() < 1e-5,
            "Expected {expected}, got {}", result.loss);
    }

    #[test]
    fn test_ipo_tau_strictness_lower_beta_larger_target() {
        let pair = make_pair_with_h_theta(1.0);
        let config_low = IpoConfig { beta: 0.1, label_smoothing: 0.0 };
        let config_high = IpoConfig { beta: 2.0, label_smoothing: 0.0 };
        let r_low = compute_ipo_loss(&[pair.clone()], &config_low).expect("ok");
        let r_high = compute_ipo_loss(&[pair], &config_high).expect("ok");
        assert!(r_low.target > r_high.target,
            "Lower beta should give larger target: low={}, high={}", r_low.target, r_high.target);
    }

    #[test]
    fn test_ipo_identical_chosen_rejected_zero_h_theta() {
        let pair = make_pair_with_h_theta(0.0);
        let beta = 2.0_f32;
        let config = IpoConfig { beta, label_smoothing: 0.0 };
        let result = compute_ipo_loss(&[pair], &config).expect("should succeed");
        let expected_target = 1.0 / (2.0 * beta);
        let expected_loss = expected_target.powi(2);
        assert!((result.loss - expected_loss).abs() < 1e-5,
            "Expected loss {expected_loss}, got {}", result.loss);
    }

    #[test]
    fn test_ipo_batch_averaging() {
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let h_values = [0.0_f32, 1.0, 2.0, 3.0];
        let target = 0.5_f32;
        let pairs: Vec<IpoPair> = h_values.iter().map(|&h| make_pair_with_h_theta(h)).collect();
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        let expected_loss: f32 = h_values.iter().map(|&h| (h - target).powi(2)).sum::<f32>()
            / h_values.len() as f32;
        assert!((result.loss - expected_loss).abs() < 1e-5,
            "Expected mean loss {expected_loss}, got {}", result.loss);
    }

    #[test]
    fn test_ipo_label_smoothing_reduces_target() {
        let pair = make_pair_with_h_theta(1.0);
        let config_no_smooth = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let config_smooth = IpoConfig { beta: 1.0, label_smoothing: 0.2 };
        let r1 = compute_ipo_loss(&[pair.clone()], &config_no_smooth).expect("ok");
        let r2 = compute_ipo_loss(&[pair], &config_smooth).expect("ok");
        assert!(r2.target < r1.target,
            "Label smoothing should reduce target: no_smooth={}, smooth={}", r1.target, r2.target);
        let expected_smooth_target = (1.0 - 0.2_f32) / (2.0 * 1.0_f32);
        assert!((r2.target - expected_smooth_target).abs() < 1e-5,
            "Expected smoothed target {expected_smooth_target}, got {}", r2.target);
    }

    #[test]
    fn test_ipo_vs_dpo_both_positive_finite() {
        let pair = make_pair_with_h_theta(5.0);
        let config = IpoConfig { beta: 0.5, label_smoothing: 0.0 };
        let comparison = compare_ipo_dpo(&[pair], &config).expect("should succeed");
        assert!(comparison.ipo_loss >= 0.0, "IPO loss should be non-negative");
        assert!(comparison.dpo_loss >= 0.0, "DPO loss should be non-negative");
        assert!(comparison.ipo_loss.is_finite(), "IPO loss should be finite");
        assert!(comparison.dpo_loss.is_finite(), "DPO loss should be finite");
        // target=1.0; IPO loss = (5-1)^2 = 16.0
        assert!((comparison.ipo_loss - 16.0).abs() < 1e-4,
            "Expected IPO loss 16.0, got {}", comparison.ipo_loss);
    }

    #[test]
    fn test_ipo_trainer_records_all_steps() {
        let config = IpoConfig::default();
        let mut trainer = IpoTrainer::new(config);
        let pair = make_pair_with_h_theta(1.0);
        for _ in 0..5 {
            trainer.step(&[pair.clone()]).expect("step failed");
        }
        assert_eq!(trainer.history().len(), 5, "Should have 5 history entries");
    }

    #[test]
    fn test_ipo_trainer_mean_loss_correct() {
        let config = IpoConfig { beta: 1.0, label_smoothing: 0.0 };
        let mut trainer = IpoTrainer::new(config);
        // target=0.5; losses: (0-0.5)^2=0.25, (0.5-0.5)^2=0, (2-0.5)^2=2.25
        trainer.step(&[make_pair_with_h_theta(0.0)]).expect("step 1");
        trainer.step(&[make_pair_with_h_theta(0.5)]).expect("step 2");
        trainer.step(&[make_pair_with_h_theta(2.0)]).expect("step 3");
        let expected_mean = (0.25 + 0.0 + 2.25) / 3.0;
        let mean = trainer.mean_loss();
        assert!((mean - expected_mean).abs() < 1e-5,
            "Expected mean_loss {expected_mean}, got {mean}");
    }

    #[test]
    fn test_ipo_preference_accuracy_zero() {
        let config = IpoConfig::default();
        let pairs: Vec<IpoPair> = vec![-1.0_f32, -2.0, -0.1, -5.0]
            .into_iter()
            .map(make_pair_with_h_theta)
            .collect();
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        assert_eq!(result.preference_accuracy, 0.0,
            "All negative h_theta should give accuracy=0, got {}", result.preference_accuracy);
    }

    #[test]
    fn test_ipo_large_batch_loss_finite() {
        let config = IpoConfig { beta: 0.5, label_smoothing: 0.0 };
        let pairs: Vec<IpoPair> = (0..100)
            .map(|i| make_pair_with_h_theta((i as f32 - 50.0) * 0.1))
            .collect();
        let result = compute_ipo_loss(&pairs, &config).expect("should succeed");
        assert!(result.loss.is_finite(), "Large batch loss should be finite");
        assert!(result.loss >= 0.0, "Large batch loss should be non-negative");
        assert_eq!(result.num_pairs, 100);
    }

    #[test]
    fn test_ipo_loss_symmetric_around_target() {
        let beta = 1.0_f32;
        let config = IpoConfig { beta, label_smoothing: 0.0 };
        let target = 1.0 / (2.0 * beta);
        let d = 1.5_f32;
        let pair_above = make_pair_with_h_theta(target + d);
        let pair_below = make_pair_with_h_theta(target - d);
        let r_above = compute_ipo_loss(&[pair_above], &config).expect("ok");
        let r_below = compute_ipo_loss(&[pair_below], &config).expect("ok");
        assert!((r_above.loss - r_below.loss).abs() < 1e-5,
            "Loss should be symmetric around target: above={}, below={}",
            r_above.loss, r_below.loss);
    }
}
