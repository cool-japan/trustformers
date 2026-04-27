//! KTO: Kahneman-Tversky Optimization
//! Reference: "KTO: Model Alignment as Prospect Theoretic Optimization" (Ethayarajh et al., 2024)
//!
//! KTO is based on prospect theory, which observes that humans are more sensitive to losses
//! than to gains of the same magnitude. Unlike DPO, KTO does NOT require pairwise preference
//! data — it works with single-turn (prompt, completion, is_preferred: bool) examples.
//!
//! ## Loss Formulation
//!
//! For a preferred example y_w:
//!   `L_w = 1 - sigmoid(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - z_ref))`
//!
//! For a rejected example y_l:
//!   `L_l = 1 - sigmoid(β * (z_ref - (log π_θ(y_l|x) - log π_ref(y_l|x))))`
//!
//! Where `z_ref = KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref]` (estimated over the batch).
//!
//! Total loss: `L_KTO = λ_w * E[L_w] + λ_l * E[L_l]`

use std::fmt;

/// A single training example for KTO (no pairwise requirement).
///
/// KTO works with individual examples labeled as preferred or rejected,
/// unlike DPO which requires paired (chosen, rejected) data.
#[derive(Debug, Clone)]
pub struct KtoExample {
    /// Log probability of completion under the policy model: log π_θ(y|x)
    pub policy_log_prob: f32,
    /// Log probability of completion under the reference model: log π_ref(y|x)
    pub reference_log_prob: f32,
    /// Whether this completion is preferred (desirable) or rejected (undesirable)
    pub is_preferred: bool,
}

/// KTO configuration parameters.
#[derive(Debug, Clone)]
pub struct KtoConfig {
    /// Temperature parameter β — controls the strength of deviation from the reference model.
    /// Higher β means the model is penalized more for diverging from the reference.
    pub beta: f32,
    /// Weight λ_w for preferred (desirable) examples. Default: 1.0
    pub lambda_preferred: f32,
    /// Weight λ_l for rejected (undesirable) examples. Default: 1.0
    pub lambda_rejected: f32,
    /// Small epsilon for numerical stability in edge cases.
    pub eps: f32,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            lambda_preferred: 1.0,
            lambda_rejected: 1.0,
            eps: 1e-8,
        }
    }
}

/// Estimated KL divergence between policy and reference model, computed from a batch.
///
/// `KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref]`
///
/// This estimate is used as z_ref in the KTO loss formulation.
#[derive(Debug, Clone)]
pub struct KlEstimate {
    /// The estimated KL divergence value.
    pub value: f32,
    /// Number of examples used to compute the estimate.
    pub num_examples: usize,
}

impl KlEstimate {
    /// Compute KL estimate as mean of (policy_log_prob - ref_log_prob) over the entire batch.
    ///
    /// Uses all examples (both preferred and rejected) for the estimate, since
    /// z_ref should reflect the overall policy-reference divergence.
    ///
    /// Returns a zero estimate when the batch is empty.
    pub fn from_batch(examples: &[KtoExample]) -> Self {
        if examples.is_empty() {
            return Self { value: 0.0, num_examples: 0 };
        }
        let sum: f32 = examples
            .iter()
            .map(|ex| ex.policy_log_prob - ex.reference_log_prob)
            .sum();
        let value = sum / (examples.len() as f32);
        Self { value, num_examples: examples.len() }
    }
}

/// Result of a KTO loss computation for a batch of examples.
#[derive(Debug, Clone)]
pub struct KtoLossResult {
    /// Total KTO loss: λ_w * E[L_w] + λ_l * E[L_l]
    pub total_loss: f32,
    /// Loss component from preferred examples: E[L_w]
    pub preferred_loss: f32,
    /// Loss component from rejected examples: E[L_l]
    pub rejected_loss: f32,
    /// Number of preferred examples in the batch
    pub num_preferred: usize,
    /// Number of rejected examples in the batch
    pub num_rejected: usize,
    /// Estimated KL divergence used as z_ref
    pub kl_estimate: f32,
    /// Mean log-ratio (policy - reference) for preferred examples
    pub mean_preferred_log_ratio: f32,
    /// Mean log-ratio (policy - reference) for rejected examples
    pub mean_rejected_log_ratio: f32,
}

/// Compute the sigmoid function: σ(x) = 1 / (1 + exp(-x)).
///
/// Numerically stable for both large positive and large negative inputs.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the KTO loss for a batch of examples.
///
/// ## Algorithm
///
/// 1. Estimate z_ref = KL(π_θ || π_ref) as the mean log-ratio over all examples
/// 2. For each preferred example: `L_w = 1 - sigmoid(β * (log_ratio - z_ref))`
/// 3. For each rejected example: `L_l = 1 - sigmoid(β * (z_ref - log_ratio))`
/// 4. Total loss = `λ_w * mean(L_w) + λ_l * mean(L_l)`
///
/// ## Errors
///
/// Returns `KtoError::EmptyBatch` if no examples are provided.
/// Returns `KtoError::NoPreferredExamples` if there are no preferred examples.
/// Returns `KtoError::NoRejectedExamples` if there are no rejected examples.
/// Returns `KtoError::NumericalError` if NaN or Inf values are encountered.
pub fn compute_kto_loss(
    examples: &[KtoExample],
    config: &KtoConfig,
) -> Result<KtoLossResult, KtoError> {
    if examples.is_empty() {
        return Err(KtoError::EmptyBatch);
    }

    let preferred: Vec<&KtoExample> =
        examples.iter().filter(|ex| ex.is_preferred).collect();
    let rejected: Vec<&KtoExample> =
        examples.iter().filter(|ex| !ex.is_preferred).collect();

    if preferred.is_empty() {
        return Err(KtoError::NoPreferredExamples);
    }
    if rejected.is_empty() {
        return Err(KtoError::NoRejectedExamples);
    }

    // Step 1: Estimate z_ref from all examples
    let kl_est = KlEstimate::from_batch(examples);
    let z_ref = kl_est.value;

    // Step 2: Compute preferred losses and mean log-ratio
    let preferred_log_ratios: Vec<f32> = preferred
        .iter()
        .map(|ex| ex.policy_log_prob - ex.reference_log_prob)
        .collect();

    let preferred_losses: Vec<f32> = preferred_log_ratios
        .iter()
        .map(|&lr| 1.0 - sigmoid(config.beta * (lr - z_ref)))
        .collect();

    let mean_preferred_log_ratio =
        preferred_log_ratios.iter().sum::<f32>() / (preferred_log_ratios.len() as f32);
    let preferred_loss_mean =
        preferred_losses.iter().sum::<f32>() / (preferred_losses.len() as f32);

    // Step 3: Compute rejected losses and mean log-ratio
    let rejected_log_ratios: Vec<f32> = rejected
        .iter()
        .map(|ex| ex.policy_log_prob - ex.reference_log_prob)
        .collect();

    let rejected_losses: Vec<f32> = rejected_log_ratios
        .iter()
        .map(|&lr| 1.0 - sigmoid(config.beta * (z_ref - lr)))
        .collect();

    let mean_rejected_log_ratio =
        rejected_log_ratios.iter().sum::<f32>() / (rejected_log_ratios.len() as f32);
    let rejected_loss_mean =
        rejected_losses.iter().sum::<f32>() / (rejected_losses.len() as f32);

    // Step 4: Combine with weights
    let total_loss = config.lambda_preferred * preferred_loss_mean
        + config.lambda_rejected * rejected_loss_mean;

    // Numerical stability check
    if total_loss.is_nan() || total_loss.is_infinite() {
        return Err(KtoError::NumericalError(format!(
            "Total loss is not finite: {total_loss}"
        )));
    }

    Ok(KtoLossResult {
        total_loss,
        preferred_loss: preferred_loss_mean,
        rejected_loss: rejected_loss_mean,
        num_preferred: preferred.len(),
        num_rejected: rejected.len(),
        kl_estimate: z_ref,
        mean_preferred_log_ratio,
        mean_rejected_log_ratio,
    })
}

/// KTO trainer that accumulates examples across steps and tracks training history.
///
/// Provides convenient methods for running multiple training steps and inspecting
/// the evolution of losses and KL estimates over time.
#[derive(Debug, Clone)]
pub struct KtoTrainer {
    /// Configuration for KTO loss computation.
    pub config: KtoConfig,
    history: Vec<KtoLossResult>,
}

impl KtoTrainer {
    /// Create a new KTO trainer with the given configuration.
    pub fn new(config: KtoConfig) -> Self {
        Self { config, history: Vec::new() }
    }

    /// Compute the KTO loss for a batch of examples and record the result.
    ///
    /// The result is appended to the trainer's history for later analysis.
    pub fn step(&mut self, examples: &[KtoExample]) -> Result<KtoLossResult, KtoError> {
        let result = compute_kto_loss(examples, &self.config)?;
        self.history.push(result.clone());
        Ok(result)
    }

    /// Return all historical KTO loss results across training steps.
    pub fn history(&self) -> &[KtoLossResult] {
        &self.history
    }

    /// Compute the mean total loss over all recorded history.
    ///
    /// Returns 0.0 if no history is available.
    pub fn mean_total_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.total_loss).sum();
        sum / (self.history.len() as f32)
    }

    /// Compute the mean KL estimate over all recorded history.
    ///
    /// Returns 0.0 if no history is available.
    pub fn mean_kl_estimate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.kl_estimate).sum();
        sum / (self.history.len() as f32)
    }
}

/// Errors that can occur during KTO loss computation.
#[derive(Debug, Clone)]
pub enum KtoError {
    /// The batch of examples was empty.
    EmptyBatch,
    /// The batch contained no preferred (desirable) examples.
    NoPreferredExamples,
    /// The batch contained no rejected (undesirable) examples.
    NoRejectedExamples,
    /// A numerical error (NaN or Inf) was encountered during computation.
    NumericalError(String),
}

impl fmt::Display for KtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KtoError::EmptyBatch => write!(f, "KTO error: batch is empty"),
            KtoError::NoPreferredExamples => {
                write!(f, "KTO error: batch contains no preferred examples")
            }
            KtoError::NoRejectedExamples => {
                write!(f, "KTO error: batch contains no rejected examples")
            }
            KtoError::NumericalError(msg) => {
                write!(f, "KTO numerical error: {msg}")
            }
        }
    }
}

impl std::error::Error for KtoError {}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Configuration tests ──────────────────────────────────────────────────

    #[test]
    fn test_kto_config_default() {
        let config = KtoConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.lambda_preferred, 1.0);
        assert_eq!(config.lambda_rejected, 1.0);
        assert_eq!(config.eps, 1e-8);
    }

    #[test]
    fn test_kto_config_custom() {
        let config = KtoConfig {
            beta: 0.5,
            lambda_preferred: 2.0,
            lambda_rejected: 0.5,
            eps: 1e-6,
        };
        assert_eq!(config.beta, 0.5);
        assert_eq!(config.lambda_preferred, 2.0);
        assert_eq!(config.lambda_rejected, 0.5);
        assert_eq!(config.eps, 1e-6);
    }

    // ── KlEstimate tests ─────────────────────────────────────────────────────

    #[test]
    fn test_kl_estimate_empty() {
        let est = KlEstimate::from_batch(&[]);
        assert_eq!(est.value, 0.0);
        assert_eq!(est.num_examples, 0);
    }

    #[test]
    fn test_kl_estimate_single_preferred() {
        let examples = vec![KtoExample {
            policy_log_prob: -1.0,
            reference_log_prob: -2.0,
            is_preferred: true,
        }];
        let est = KlEstimate::from_batch(&examples);
        // KL ≈ policy - ref = -1.0 - (-2.0) = 1.0
        assert!((est.value - 1.0).abs() < 1e-6, "Expected 1.0, got {}", est.value);
        assert_eq!(est.num_examples, 1);
    }

    #[test]
    fn test_kl_estimate_single_rejected() {
        let examples = vec![KtoExample {
            policy_log_prob: -3.0,
            reference_log_prob: -1.0,
            is_preferred: false,
        }];
        let est = KlEstimate::from_batch(&examples);
        // KL ≈ policy - ref = -3.0 - (-1.0) = -2.0
        assert!((est.value - (-2.0)).abs() < 1e-6, "Expected -2.0, got {}", est.value);
        assert_eq!(est.num_examples, 1);
    }

    #[test]
    fn test_kl_estimate_batch() {
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -3.0,
                reference_log_prob: -3.0,
                is_preferred: false,
            },
            KtoExample {
                policy_log_prob: -2.0,
                reference_log_prob: -4.0,
                is_preferred: true,
            },
        ];
        let est = KlEstimate::from_batch(&examples);
        // log_ratios: 1.0, 0.0, 2.0 => mean = 1.0
        assert!((est.value - 1.0).abs() < 1e-6, "Expected 1.0, got {}", est.value);
        assert_eq!(est.num_examples, 3);
    }

    #[test]
    fn test_kl_estimate_zero_when_equal() {
        // When policy == reference for all examples, KL should be 0
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.5,
                reference_log_prob: -1.5,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -2.0,
                reference_log_prob: -2.0,
                is_preferred: false,
            },
        ];
        let est = KlEstimate::from_batch(&examples);
        assert!(est.value.abs() < 1e-6, "Expected 0.0, got {}", est.value);
    }

    // ── compute_kto_loss tests ────────────────────────────────────────────────

    #[test]
    fn test_kto_loss_all_preferred() {
        // All-preferred batch should return NoRejectedExamples error
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -1.5,
                reference_log_prob: -2.5,
                is_preferred: true,
            },
        ];
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config);
        assert!(
            matches!(result, Err(KtoError::NoRejectedExamples)),
            "Expected NoRejectedExamples, got {:?}",
            result
        );
    }

    #[test]
    fn test_kto_loss_all_rejected() {
        // All-rejected batch should return NoPreferredExamples error
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: false,
            },
            KtoExample {
                policy_log_prob: -1.5,
                reference_log_prob: -2.5,
                is_preferred: false,
            },
        ];
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config);
        assert!(
            matches!(result, Err(KtoError::NoPreferredExamples)),
            "Expected NoPreferredExamples, got {:?}",
            result
        );
    }

    #[test]
    fn test_kto_loss_mixed_batch() {
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -3.0,
                reference_log_prob: -2.0,
                is_preferred: false,
            },
        ];
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        // Verify structure: num counts
        assert_eq!(result.num_preferred, 1);
        assert_eq!(result.num_rejected, 1);
        // Loss values should be in [0, 1] since sigmoid is in (0, 1)
        // and L = 1 - sigmoid(...) is also in (0, 1)
        assert!(
            result.preferred_loss >= 0.0 && result.preferred_loss <= 1.0,
            "preferred_loss out of range: {}",
            result.preferred_loss
        );
        assert!(
            result.rejected_loss >= 0.0 && result.rejected_loss <= 1.0,
            "rejected_loss out of range: {}",
            result.rejected_loss
        );
        assert!(result.total_loss.is_finite(), "total_loss should be finite");
    }

    #[test]
    fn test_kto_loss_beta_effect() {
        // With very high beta and policy >> reference for preferred,
        // L_w should be very small (well-trained on preferred)
        let examples_high_ratio = vec![
            KtoExample {
                policy_log_prob: 0.0,
                reference_log_prob: -10.0, // log_ratio = 10.0 for preferred
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -10.0,
                reference_log_prob: 0.0, // log_ratio = -10.0 for rejected
                is_preferred: false,
            },
        ];
        let config_high_beta = KtoConfig { beta: 5.0, ..KtoConfig::default() };
        let config_low_beta = KtoConfig { beta: 0.01, ..KtoConfig::default() };

        let result_high = compute_kto_loss(&examples_high_ratio, &config_high_beta)
            .expect("should succeed");
        let result_low =
            compute_kto_loss(&examples_high_ratio, &config_low_beta).expect("should succeed");

        // High beta should push both losses closer to 0 (well-separated distributions)
        assert!(
            result_high.preferred_loss < result_low.preferred_loss,
            "High beta should produce lower preferred loss when log_ratio >> z_ref"
        );
    }

    #[test]
    fn test_kto_loss_lambda_weights() {
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -1.5,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -2.0,
                reference_log_prob: -1.0,
                is_preferred: false,
            },
        ];
        let config_equal = KtoConfig {
            lambda_preferred: 1.0,
            lambda_rejected: 1.0,
            ..KtoConfig::default()
        };
        let config_preferred_heavy = KtoConfig {
            lambda_preferred: 2.0,
            lambda_rejected: 0.5,
            ..KtoConfig::default()
        };

        let result_equal = compute_kto_loss(&examples, &config_equal).expect("should succeed");
        let result_heavy =
            compute_kto_loss(&examples, &config_preferred_heavy).expect("should succeed");

        // The totals should differ due to different weights
        assert!(
            (result_equal.total_loss - result_heavy.total_loss).abs() > 1e-6,
            "Different lambda weights should produce different total losses"
        );

        // Verify the weighted sum manually
        let expected_heavy = 2.0 * result_equal.preferred_loss + 0.5 * result_equal.rejected_loss;
        assert!(
            (result_heavy.total_loss - expected_heavy).abs() < 1e-5,
            "Expected {expected_heavy}, got {}",
            result_heavy.total_loss
        );
    }

    #[test]
    fn test_kto_loss_symmetric_kl() {
        // When KL = 0 (policy == reference everywhere),
        // preferred and rejected losses should both be 0.5 (sigmoid(0) = 0.5, L = 0.5)
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -1.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -2.0,
                reference_log_prob: -2.0,
                is_preferred: false,
            },
        ];
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");

        // z_ref = 0, log_ratio = 0 for both
        // L_w = 1 - sigmoid(beta * (0 - 0)) = 1 - sigmoid(0) = 1 - 0.5 = 0.5
        // L_l = 1 - sigmoid(beta * (0 - 0)) = 0.5
        assert!(
            (result.preferred_loss - 0.5).abs() < 1e-5,
            "Expected 0.5, got {}",
            result.preferred_loss
        );
        assert!(
            (result.rejected_loss - 0.5).abs() < 1e-5,
            "Expected 0.5, got {}",
            result.rejected_loss
        );
        assert!(
            (result.total_loss - 1.0).abs() < 1e-5,
            "Expected 1.0, got {}",
            result.total_loss
        );
    }

    // ── KtoTrainer tests ─────────────────────────────────────────────────────

    #[test]
    fn test_kto_trainer_step() {
        let config = KtoConfig::default();
        let mut trainer = KtoTrainer::new(config);

        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -3.0,
                reference_log_prob: -1.5,
                is_preferred: false,
            },
        ];
        let result = trainer.step(&examples).expect("step should succeed");
        assert_eq!(result.num_preferred, 1);
        assert_eq!(result.num_rejected, 1);
        assert_eq!(trainer.history().len(), 1);
    }

    #[test]
    fn test_kto_trainer_history() {
        let config = KtoConfig::default();
        let mut trainer = KtoTrainer::new(config);

        let make_examples = |policy_pref: f32, ref_pref: f32, policy_rej: f32, ref_rej: f32| {
            vec![
                KtoExample {
                    policy_log_prob: policy_pref,
                    reference_log_prob: ref_pref,
                    is_preferred: true,
                },
                KtoExample {
                    policy_log_prob: policy_rej,
                    reference_log_prob: ref_rej,
                    is_preferred: false,
                },
            ]
        };

        trainer.step(&make_examples(-1.0, -2.0, -3.0, -1.0)).expect("step 1");
        trainer.step(&make_examples(-0.5, -1.5, -2.5, -1.5)).expect("step 2");
        trainer.step(&make_examples(-0.8, -1.8, -2.0, -1.0)).expect("step 3");

        assert_eq!(trainer.history().len(), 3);
        // All losses should be finite
        for result in trainer.history() {
            assert!(result.total_loss.is_finite(), "all history losses should be finite");
        }
    }

    #[test]
    fn test_kto_trainer_mean_loss() {
        let config = KtoConfig::default();
        let mut trainer = KtoTrainer::new(config);

        // Initially zero history
        assert_eq!(trainer.mean_total_loss(), 0.0);
        assert_eq!(trainer.mean_kl_estimate(), 0.0);

        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -2.0,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -3.0,
                reference_log_prob: -1.5,
                is_preferred: false,
            },
        ];

        trainer.step(&examples).expect("step 1");
        trainer.step(&examples).expect("step 2");

        // Mean loss over two identical steps should equal the individual step loss
        let first_loss = trainer.history()[0].total_loss;
        let mean = trainer.mean_total_loss();
        assert!(
            (mean - first_loss).abs() < 1e-6,
            "Mean of identical steps should equal individual step loss: {mean} vs {first_loss}"
        );
    }

    // ── Error display tests ───────────────────────────────────────────────────

    #[test]
    fn test_kto_error_display() {
        let empty = KtoError::EmptyBatch;
        let no_pref = KtoError::NoPreferredExamples;
        let no_rej = KtoError::NoRejectedExamples;
        let numerical = KtoError::NumericalError("NaN detected".to_string());

        assert!(empty.to_string().contains("empty"), "EmptyBatch should mention 'empty'");
        assert!(
            no_pref.to_string().contains("preferred"),
            "NoPreferredExamples should mention 'preferred'"
        );
        assert!(
            no_rej.to_string().contains("rejected"),
            "NoRejectedExamples should mention 'rejected'"
        );
        assert!(
            numerical.to_string().contains("NaN detected"),
            "NumericalError should contain the message"
        );
    }

    // ── Sigmoid function tests ────────────────────────────────────────────────

    #[test]
    fn test_kto_sigmoid_values() {
        // sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");

        // sigmoid(large positive) ≈ 1.0
        assert!(
            sigmoid(100.0) > 0.9999,
            "sigmoid(100) should be close to 1.0, got {}",
            sigmoid(100.0)
        );

        // sigmoid(large negative) ≈ 0.0
        assert!(
            sigmoid(-100.0) < 1e-4,
            "sigmoid(-100) should be close to 0.0, got {}",
            sigmoid(-100.0)
        );

        // sigmoid(-x) = 1 - sigmoid(x)
        for x in &[-2.0f32, -1.0, 0.5, 1.5, 3.0] {
            let s_pos = sigmoid(*x);
            let s_neg = sigmoid(-x);
            assert!(
                (s_pos + s_neg - 1.0).abs() < 1e-6,
                "sigmoid({x}) + sigmoid(-{x}) should be 1.0: {s_pos} + {s_neg}"
            );
        }

        // Known value: sigmoid(1.0) ≈ 0.7311
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(
            (sigmoid(1.0) - expected).abs() < 1e-6,
            "sigmoid(1.0) should be {expected}"
        );
    }

    // ── Edge case tests ───────────────────────────────────────────────────────

    #[test]
    fn test_kto_loss_empty_batch() {
        let config = KtoConfig::default();
        let result = compute_kto_loss(&[], &config);
        assert!(
            matches!(result, Err(KtoError::EmptyBatch)),
            "Empty batch should return EmptyBatch error"
        );
    }

    #[test]
    fn test_kto_loss_mean_log_ratios() {
        // Verify mean log ratios are computed correctly
        let examples = vec![
            KtoExample {
                policy_log_prob: -1.0,
                reference_log_prob: -3.0, // preferred log_ratio = 2.0
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -5.0,
                reference_log_prob: -3.0, // rejected log_ratio = -2.0
                is_preferred: false,
            },
        ];
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config).expect("should succeed");

        assert!(
            (result.mean_preferred_log_ratio - 2.0).abs() < 1e-5,
            "Expected preferred log_ratio 2.0, got {}",
            result.mean_preferred_log_ratio
        );
        assert!(
            (result.mean_rejected_log_ratio - (-2.0)).abs() < 1e-5,
            "Expected rejected log_ratio -2.0, got {}",
            result.mean_rejected_log_ratio
        );
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn pref(policy: f32, reference: f32) -> KtoExample {
        KtoExample { policy_log_prob: policy, reference_log_prob: reference, is_preferred: true }
    }

    fn rej(policy: f32, reference: f32) -> KtoExample {
        KtoExample { policy_log_prob: policy, reference_log_prob: reference, is_preferred: false }
    }

    // 1. Symmetric: preferred with log_ratio=d and rejected with log_ratio=-d give same loss (z_ref=0)
    #[test]
    fn test_kto_desirable_undesirable_symmetry() {
        let d = 2.0_f32;
        // z_ref = (d + (-d)) / 2 = 0
        let examples = vec![pref(d, 0.0), rej(-d, 0.0)];
        let config = KtoConfig { beta: 0.5, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!((result.preferred_loss - result.rejected_loss).abs() < 1e-5,
            "Symmetric inputs should give equal preferred/rejected losses: pref={}, rej={}",
            result.preferred_loss, result.rejected_loss);
    }

    // 2. Higher z_ref increases preferred loss
    #[test]
    fn test_kto_positive_z_ref_increases_preferred_loss() {
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let examples_a = vec![pref(1.0, 0.0), rej(0.0, 0.0)]; // z_ref = 0.5
        let examples_b = vec![pref(1.0, 0.0), rej(2.0, 0.0)]; // z_ref = 1.5
        let r_a = compute_kto_loss(&examples_a, &config).expect("ok");
        let r_b = compute_kto_loss(&examples_b, &config).expect("ok");
        assert!(r_b.preferred_loss > r_a.preferred_loss,
            "Higher z_ref should increase preferred loss: a={}, b={}", r_a.preferred_loss, r_b.preferred_loss);
    }

    // 3. Very good preferred output: L_w → 0
    #[test]
    fn test_kto_preferred_loss_near_zero_for_good_output() {
        let examples = vec![pref(20.0, 0.0), rej(-20.0, 0.0)]; // z_ref = 0
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!(result.preferred_loss < 0.01,
            "Very good preferred output should have loss near 0, got {}", result.preferred_loss);
    }

    // 4. Policy correctly avoids bad output: L_l → 0
    #[test]
    fn test_kto_rejected_loss_near_zero_when_policy_avoids_bad() {
        let examples = vec![pref(20.0, 0.0), rej(-20.0, 0.0)]; // z_ref = 0
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!(result.rejected_loss < 0.01,
            "Policy correctly avoiding bad output should have loss near 0, got {}", result.rejected_loss);
    }

    // 5. Very bad preferred output: L_w → 1
    #[test]
    fn test_kto_preferred_loss_near_one_for_bad_preferred() {
        // z_ref = (-20 + 0) / 2 = -10; L_w = 1 - sigmoid(1*(-20-(-10))) = 1 - sigmoid(-10) ≈ 1
        let examples = vec![pref(-20.0, 0.0), rej(0.0, 0.0)];
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!(result.preferred_loss > 0.9,
            "Bad preferred output should have loss near 1, got {}", result.preferred_loss);
    }

    // 6. High beta makes decision boundary sharper
    #[test]
    fn test_kto_high_beta_sharp_boundary() {
        // log_ratio = 0.1 > z_ref=0 → preferred should benefit from high beta
        let examples = vec![pref(0.1, 0.0), rej(-0.1, 0.0)];
        let config_high = KtoConfig { beta: 10.0, ..KtoConfig::default() };
        let config_low = KtoConfig { beta: 0.1, ..KtoConfig::default() };
        let r_high = compute_kto_loss(&examples, &config_high).expect("ok");
        let r_low = compute_kto_loss(&examples, &config_low).expect("ok");
        assert!(r_high.preferred_loss < r_low.preferred_loss,
            "High beta should give lower preferred loss for log_ratio > z_ref: high={}, low={}",
            r_high.preferred_loss, r_low.preferred_loss);
    }

    // 7. z_ref estimation from 4-example minibatch
    #[test]
    fn test_kto_z_ref_estimation_from_minibatch() {
        let examples = vec![
            pref(1.0, 0.0),   // log_ratio = 1.0
            pref(3.0, 1.0),   // log_ratio = 2.0
            rej(-1.0, 0.0),   // log_ratio = -1.0
            rej(0.0, 1.0),    // log_ratio = -1.0
        ];
        // z_ref = (1.0 + 2.0 + (-1.0) + (-1.0)) / 4 = 0.25
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!((result.kl_estimate - 0.25).abs() < 1e-5,
            "Expected z_ref=0.25, got {}", result.kl_estimate);
    }

    // 8. Unpaired batch: 3 preferred, 1 rejected
    #[test]
    fn test_kto_unpaired_batch_3_pref_1_rej() {
        let examples = vec![
            pref(-1.0, -2.0),
            pref(-1.5, -2.5),
            pref(-0.5, -1.5),
            rej(-3.0, -1.0),
        ];
        let config = KtoConfig::default();
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert_eq!(result.num_preferred, 3);
        assert_eq!(result.num_rejected, 1);
        assert!(result.total_loss.is_finite());
    }

    // 9. Batch aggregate: total = lambda_w * mean(L_w) + lambda_l * mean(L_l)
    #[test]
    fn test_kto_batch_aggregate_is_weighted_mean() {
        let examples = vec![
            pref(-1.0, -2.0),
            pref(-2.0, -1.0),
            rej(-3.0, -1.0),
            rej(-0.5, -2.0),
        ];
        let config = KtoConfig { lambda_preferred: 2.0, lambda_rejected: 0.5, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        let expected = 2.0 * result.preferred_loss + 0.5 * result.rejected_loss;
        assert!((result.total_loss - expected).abs() < 1e-5,
            "Expected {expected}, got {}", result.total_loss);
    }

    // 10. lambda_preferred=0 → total = lambda_rejected * rejected_loss
    #[test]
    fn test_kto_lambda_preferred_zero() {
        let examples = vec![pref(-1.0, -2.0), rej(-3.0, -1.0)];
        let config = KtoConfig { lambda_preferred: 0.0, lambda_rejected: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!((result.total_loss - result.rejected_loss).abs() < 1e-5,
            "With lambda_preferred=0, total should equal rejected_loss");
    }

    // 11. Trainer: 4 steps → history has 4 entries
    #[test]
    fn test_kto_trainer_four_steps() {
        let config = KtoConfig::default();
        let mut trainer = KtoTrainer::new(config);
        let examples = vec![pref(-1.0, -2.0), rej(-3.0, -1.0)];
        for _ in 0..4 {
            trainer.step(&examples).expect("step failed");
        }
        assert_eq!(trainer.history().len(), 4);
    }

    // 12. mean_kl_estimate matches manual mean
    #[test]
    fn test_kto_trainer_mean_kl_estimate_correct() {
        let config = KtoConfig::default();
        let mut trainer = KtoTrainer::new(config);
        // batch A: z_ref = (1.0 + (-1.0))/2 = 0.0
        trainer.step(&[pref(1.0, 0.0), rej(-1.0, 0.0)]).expect("step 1");
        // batch B: z_ref = (2.0 + 0.0)/2 = 1.0
        trainer.step(&[pref(2.0, 0.0), rej(0.0, 0.0)]).expect("step 2");
        let mean_kl = trainer.mean_kl_estimate();
        assert!((mean_kl - 0.5).abs() < 1e-5,
            "Expected mean_kl=0.5, got {mean_kl}");
    }

    // 13. KL estimate uses both preferred and rejected examples
    #[test]
    fn test_kto_kl_estimate_uses_all_examples() {
        let config = KtoConfig::default();
        let mixed = vec![pref(2.0, 0.0), rej(-2.0, 0.0)];
        let result = compute_kto_loss(&mixed, &config).expect("ok");
        // z_ref = (2 + (-2)) / 2 = 0
        assert!((result.kl_estimate - 0.0).abs() < 1e-5,
            "KL estimate should average all examples: got {}", result.kl_estimate);
    }

    // 14. preferred_loss and rejected_loss always in [0, 1]
    #[test]
    fn test_kto_loss_in_zero_one_range() {
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let test_cases: Vec<Vec<KtoExample>> = vec![
            vec![pref(0.0, 0.0), rej(0.0, 0.0)],
            vec![pref(10.0, 0.0), rej(-10.0, 0.0)],
            vec![pref(-10.0, 0.0), rej(10.0, 0.0)],
            vec![pref(1.0, 2.0), rej(3.0, 1.0)],
        ];
        for examples in test_cases {
            let result = compute_kto_loss(&examples, &config).expect("should succeed");
            assert!(result.preferred_loss >= 0.0 && result.preferred_loss <= 1.0,
                "preferred_loss out of [0,1]: {}", result.preferred_loss);
            assert!(result.rejected_loss >= 0.0 && result.rejected_loss <= 1.0,
                "rejected_loss out of [0,1]: {}", result.rejected_loss);
        }
    }

    // 15. Rejected loss near 1 when policy strongly prefers bad output
    #[test]
    fn test_kto_rejected_loss_near_one_when_policy_prefers_bad() {
        // z_ref = (0 + 20)/2 = 10; L_l = 1 - sigmoid(1*(10-20)) = 1 - sigmoid(-10) ≈ 1
        let examples = vec![pref(0.0, 0.0), rej(20.0, 0.0)];
        let config = KtoConfig { beta: 1.0, ..KtoConfig::default() };
        let result = compute_kto_loss(&examples, &config).expect("should succeed");
        assert!(result.rejected_loss > 0.9,
            "Policy preferring bad output should give high rejected loss, got {}", result.rejected_loss);
    }
}
