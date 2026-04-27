//! PPO: Proximal Policy Optimization for RLHF
//!
//! This module implements Proximal Policy Optimization with Generalized Advantage Estimation
//! (GAE) for Reinforcement Learning from Human Feedback (RLHF).
//!
//! # Key Concepts
//!
//! - **Policy gradient** with clipped surrogate objective:
//!   `min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)`
//! - **GAE** (Generalized Advantage Estimation):
//!   `A_t = Σ (γλ)^l δ_{t+l}` where `δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`
//! - **Value function loss**: `(V_θ(s) - V_target)^2`
//! - **Entropy bonus**: `-β_entropy * H(π_θ)`
//! - **Combined loss**: `L = L_policy + c_vf * L_value + c_entropy * L_entropy`
//!
//! # Reference
//!
//! Schulman et al. 2017: "Proximal Policy Optimization Algorithms"
//! <https://arxiv.org/abs/1707.06347>

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during PPO computations.
#[derive(Debug, Clone, PartialEq)]
pub enum PpoError {
    /// The batch of steps was empty.
    EmptyBatch,
    /// The number of steps and advantages/value_targets do not match.
    LengthMismatch { steps: usize, advantages: usize },
    /// A numerical error occurred (NaN, Inf, etc.).
    NumericalError(String),
}

impl fmt::Display for PpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PpoError::EmptyBatch => write!(f, "PPO batch is empty; at least one step is required"),
            PpoError::LengthMismatch { steps, advantages } => write!(
                f,
                "Length mismatch: {} steps but {} advantages/value_targets",
                steps, advantages
            ),
            PpoError::NumericalError(msg) => write!(f, "Numerical error in PPO computation: {}", msg),
        }
    }
}

impl std::error::Error for PpoError {}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single time step in a PPO episode.
///
/// Each step records the log-probabilities of the action taken under both the
/// current (new) policy and the old (reference) policy, the value-function
/// estimate, the extrinsic reward, and whether the episode terminated here.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoStep {
    /// Log probability of the action under the current policy `π_θ(a|s)`.
    pub log_prob: f32,
    /// Log probability of the action under the old / reference policy `π_θ_old(a|s)`.
    pub old_log_prob: f32,
    /// Value estimate `V(s_t)` from the value function head.
    pub value: f32,
    /// Extrinsic reward `r_t` received at this time step.
    pub reward: f32,
    /// Whether this is the last step in the episode (terminal state).
    pub is_terminal: bool,
}

impl PpoStep {
    /// Probability ratio `r_t = exp(log_prob - old_log_prob)`.
    ///
    /// A ratio of 1 means the policies are identical for this action.
    /// Values > 1 mean the current policy assigns higher probability.
    #[inline]
    pub fn ratio(&self) -> f32 {
        (self.log_prob - self.old_log_prob).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the PPO algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoConfig {
    /// Clipping threshold `ε` for the surrogate objective (default: `0.2`).
    pub clip_epsilon: f32,
    /// Discount factor `γ` for future rewards (default: `0.99`).
    pub discount_gamma: f32,
    /// GAE lambda `λ` controlling the bias-variance trade-off (default: `0.95`).
    pub gae_lambda: f32,
    /// Coefficient `c_vf` for the value function loss (default: `0.5`).
    pub value_coeff: f32,
    /// Coefficient `c_entropy` for the entropy bonus (default: `0.01`).
    pub entropy_coeff: f32,
    /// Maximum gradient norm for clipping — informational only (default: `0.5`).
    pub max_grad_norm: f32,
    /// Whether to normalize advantages to zero mean and unit variance (default: `true`).
    pub normalize_advantages: bool,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2,
            discount_gamma: 0.99,
            gae_lambda: 0.95,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            max_grad_norm: 0.5,
            normalize_advantages: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GAE computation
// ─────────────────────────────────────────────────────────────────────────────

/// The result of computing Generalized Advantage Estimation for an episode.
#[derive(Debug, Clone)]
pub struct GaeResult {
    /// Per-step advantage estimates `A_t`.
    pub advantages: Vec<f32>,
    /// Per-step value targets `V_target_t = A_t + V(s_t)`.
    pub value_targets: Vec<f32>,
    /// Bootstrap value used for the state just after the last step.
    pub last_value: f32,
}

/// Compute Generalized Advantage Estimation (GAE) from a sequence of steps.
///
/// The algorithm iterates in **reverse** order, accumulating the GAE:
/// ```text
/// δ_t  = r_t + γ * V(s_{t+1}) - V(s_t)
/// GAE_t = δ_t + γ * λ * GAE_{t+1}
/// ```
/// At terminal steps `is_terminal == true`, the bootstrap value is treated as 0.
///
/// `last_value` is `V(s_T)` used to bootstrap the final non-terminal state.
///
/// Returns [`GaeResult`] with per-step advantages and value targets.
pub fn compute_gae(steps: &[PpoStep], last_value: f32, config: &PpoConfig) -> GaeResult {
    let n = steps.len();
    let mut advantages = vec![0.0f32; n];
    let mut value_targets = vec![0.0f32; n];

    let mut gae = 0.0f32;

    for i in (0..n).rev() {
        let next_value = if steps[i].is_terminal {
            // Terminal state: no future value
            0.0
        } else if i + 1 < n {
            steps[i + 1].value
        } else {
            // Last step, not terminal — use bootstrap
            last_value
        };

        let td_residual = steps[i].reward + config.discount_gamma * next_value - steps[i].value;

        if steps[i].is_terminal {
            // Reset GAE at episode boundaries
            gae = td_residual;
        } else {
            gae = td_residual + config.discount_gamma * config.gae_lambda * gae;
        }

        advantages[i] = gae;
        value_targets[i] = advantages[i] + steps[i].value;
    }

    GaeResult {
        advantages,
        value_targets,
        last_value,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PPO loss computation
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a single PPO loss computation.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoLossResult {
    /// Policy (actor) loss — already negated so that gradient descent minimises it.
    pub policy_loss: f32,
    /// Value function (critic) loss: `0.5 * mean((V_θ - V_target)²)`.
    pub value_loss: f32,
    /// Entropy loss proxy: `mean(-log_prob)`.
    ///
    /// The total loss includes `+ entropy_coeff * entropy_loss`, which means the
    /// optimizer is incentivised to keep log_prob low (i.e., entropy high).
    pub entropy_loss: f32,
    /// Weighted combination: `policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss`.
    pub total_loss: f32,
    /// Fraction of steps where the probability ratio was clipped.
    pub clip_fraction: f32,
    /// Mean probability ratio `r_t` over the batch.
    pub mean_ratio: f32,
    /// Mean advantage `A_t` over the batch (after optional normalisation).
    pub mean_advantage: f32,
    /// Approximate KL divergence: `E[old_log_prob - log_prob]`.
    pub approx_kl: f32,
}

/// Compute the PPO loss for a batch of steps and their pre-computed advantages.
///
/// # Errors
///
/// Returns [`PpoError::EmptyBatch`] if `steps` is empty.
/// Returns [`PpoError::LengthMismatch`] if slice lengths differ.
/// Returns [`PpoError::NumericalError`] if NaN or infinity is detected.
pub fn compute_ppo_loss(
    steps: &[PpoStep],
    advantages: &[f32],
    value_targets: &[f32],
    config: &PpoConfig,
) -> Result<PpoLossResult, PpoError> {
    let n = steps.len();

    if n == 0 {
        return Err(PpoError::EmptyBatch);
    }
    if advantages.len() != n {
        return Err(PpoError::LengthMismatch {
            steps: n,
            advantages: advantages.len(),
        });
    }
    if value_targets.len() != n {
        return Err(PpoError::LengthMismatch {
            steps: n,
            advantages: value_targets.len(),
        });
    }

    // ── 1. Optionally normalise advantages ────────────────────────────────
    let norm_advantages: Vec<f32> = if config.normalize_advantages {
        let mean = advantages.iter().sum::<f32>() / n as f32;
        let var = advantages
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let std = (var + 1e-8).sqrt();
        advantages.iter().map(|a| (a - mean) / std).collect()
    } else {
        advantages.to_vec()
    };

    // ── 2. Per-step quantities ─────────────────────────────────────────────
    let mut policy_loss_sum = 0.0f32;
    let mut value_loss_sum = 0.0f32;
    let mut entropy_loss_sum = 0.0f32;
    let mut clip_count = 0usize;
    let mut ratio_sum = 0.0f32;
    let mut kl_sum = 0.0f32;

    for (i, step) in steps.iter().enumerate() {
        let ratio = step.ratio();
        let adv = norm_advantages[i];

        // Clipped surrogate: min(r * A, clip(r, 1-ε, 1+ε) * A)
        let clip_lo = 1.0 - config.clip_epsilon;
        let clip_hi = 1.0 + config.clip_epsilon;
        let clipped_ratio = ratio.clamp(clip_lo, clip_hi);

        let obj_unclipped = ratio * adv;
        let obj_clipped = clipped_ratio * adv;
        // We *minimise* the negative objective: loss = -min(obj, clipped_obj)
        let policy_term = -obj_unclipped.min(obj_clipped);

        // Value loss: 0.5 * (V - V_target)^2
        let value_term = 0.5 * (step.value - value_targets[i]).powi(2);

        // Entropy proxy: -log_prob  (negative because higher entropy ↔ lower |log_prob|)
        let entropy_term = -step.log_prob;

        // Approximate KL: E[old_log_prob - log_prob]
        let kl_term = step.old_log_prob - step.log_prob;

        // Clip-fraction tracking
        if (ratio - 1.0).abs() > config.clip_epsilon {
            clip_count += 1;
        }

        policy_loss_sum += policy_term;
        value_loss_sum += value_term;
        entropy_loss_sum += entropy_term;
        ratio_sum += ratio;
        kl_sum += kl_term;
    }

    let inv_n = 1.0 / n as f32;
    let policy_loss = policy_loss_sum * inv_n;
    let value_loss = value_loss_sum * inv_n;
    let entropy_loss = entropy_loss_sum * inv_n;
    let clip_fraction = clip_count as f32 * inv_n;
    let mean_ratio = ratio_sum * inv_n;
    let mean_advantage = norm_advantages.iter().sum::<f32>() * inv_n;
    let approx_kl = kl_sum * inv_n;

    let total_loss =
        policy_loss + config.value_coeff * value_loss + config.entropy_coeff * entropy_loss;

    // Numerical sanity check
    if total_loss.is_nan() || total_loss.is_infinite() {
        return Err(PpoError::NumericalError(format!(
            "total_loss={} is not finite (policy={}, value={}, entropy={})",
            total_loss, policy_loss, value_loss, entropy_loss
        )));
    }

    Ok(PpoLossResult {
        policy_loss,
        value_loss,
        entropy_loss,
        total_loss,
        clip_fraction,
        mean_ratio,
        mean_advantage,
        approx_kl,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful PPO trainer that accumulates update history for diagnostics.
#[derive(Debug)]
pub struct PpoTrainer {
    /// Configuration for PPO.
    pub config: PpoConfig,
    history: Vec<PpoLossResult>,
    total_steps: usize,
}

impl PpoTrainer {
    /// Create a new [`PpoTrainer`] with the given configuration.
    pub fn new(config: PpoConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            total_steps: 0,
        }
    }

    /// Run one PPO update step from a collected episode.
    ///
    /// Computes GAE advantages, then the PPO loss.
    /// The result is pushed onto the history.
    ///
    /// # Parameters
    ///
    /// - `steps`: sequence of `(s_t, a_t, r_t)` data from rollout collection.
    /// - `last_value`: `V(s_T)` — bootstrap value for the state just after the last step.
    pub fn update(
        &mut self,
        steps: &[PpoStep],
        last_value: f32,
    ) -> Result<PpoLossResult, PpoError> {
        let gae = compute_gae(steps, last_value, &self.config);
        let loss =
            compute_ppo_loss(steps, &gae.advantages, &gae.value_targets, &self.config)?;
        self.history.push(loss.clone());
        self.total_steps += steps.len();
        Ok(loss)
    }

    /// Return the full update history.
    pub fn history(&self) -> &[PpoLossResult] {
        &self.history
    }

    /// Total number of environment steps seen across all updates.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Mean policy loss over all updates in the history.
    ///
    /// Returns `0.0` if the history is empty.
    pub fn mean_policy_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.policy_loss).sum();
        sum / self.history.len() as f32
    }

    /// Mean approximate KL divergence over all updates in the history.
    ///
    /// Returns `0.0` if the history is empty.
    pub fn mean_approx_kl(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|r| r.approx_kl).sum();
        sum / self.history.len() as f32
    }

    /// Check whether early stopping should be triggered.
    ///
    /// Returns `true` if the most recent update's `approx_kl` exceeds `kl_threshold`.
    pub fn should_early_stop(&self, kl_threshold: f32) -> bool {
        self.history
            .last()
            .map(|r| r.approx_kl > kl_threshold)
            .unwrap_or(false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper constructors ──────────────────────────────────────────────

    fn make_step(log_prob: f32, old_log_prob: f32, value: f32, reward: f32, is_terminal: bool) -> PpoStep {
        PpoStep { log_prob, old_log_prob, value, reward, is_terminal }
    }

    fn no_norm_config() -> PpoConfig {
        PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        }
    }

    // ── PpoStep tests ────────────────────────────────────────────────────

    #[test]
    fn test_ppo_step_ratio_unchanged() {
        // Same log_prob → ratio = 1
        let step = make_step(-1.0, -1.0, 0.5, 1.0, false);
        let ratio = step.ratio();
        assert!((ratio - 1.0).abs() < 1e-6, "expected ratio ≈ 1, got {}", ratio);
    }

    #[test]
    fn test_ppo_step_ratio_increased() {
        // log_prob > old_log_prob → ratio > 1
        let step = make_step(-0.5, -1.5, 0.5, 1.0, false);
        let ratio = step.ratio();
        // exp(-0.5 - (-1.5)) = exp(1.0) ≈ 2.718
        let expected = (1.0_f32).exp();
        assert!((ratio - expected).abs() < 1e-5, "expected ratio ≈ {}, got {}", expected, ratio);
    }

    // ── PpoConfig tests ──────────────────────────────────────────────────

    #[test]
    fn test_ppo_config_default() {
        let cfg = PpoConfig::default();
        assert!((cfg.clip_epsilon - 0.2).abs() < 1e-8);
        assert!((cfg.discount_gamma - 0.99).abs() < 1e-8);
        assert!((cfg.gae_lambda - 0.95).abs() < 1e-8);
        assert!((cfg.value_coeff - 0.5).abs() < 1e-8);
        assert!((cfg.entropy_coeff - 0.01).abs() < 1e-8);
        assert!((cfg.max_grad_norm - 0.5).abs() < 1e-8);
        assert!(cfg.normalize_advantages);
    }

    // ── GAE tests ────────────────────────────────────────────────────────

    #[test]
    fn test_gae_single_step_terminal() {
        // Terminal step: no bootstrap
        // δ_0 = r + γ * 0 - V = 1.0 + 0 - 0.5 = 0.5
        let cfg = no_norm_config();
        let steps = vec![make_step(0.0, 0.0, 0.5, 1.0, true)];
        let res = compute_gae(&steps, 0.0, &cfg);
        assert_eq!(res.advantages.len(), 1);
        assert!((res.advantages[0] - 0.5).abs() < 1e-5);
        assert!((res.value_targets[0] - (0.5 + 0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_gae_single_step_non_terminal() {
        // Non-terminal single step bootstraps from last_value
        // δ = r + γ * last_value - V = 1.0 + 0.99 * 2.0 - 0.5 = 2.48
        let cfg = no_norm_config();
        let steps = vec![make_step(0.0, 0.0, 0.5, 1.0, false)];
        let last_value = 2.0;
        let res = compute_gae(&steps, last_value, &cfg);
        let expected = 1.0 + 0.99 * 2.0 - 0.5;
        assert!((res.advantages[0] - expected).abs() < 1e-4);
        assert!((res.value_targets[0] - (expected + 0.5)).abs() < 1e-4);
        assert!((res.last_value - last_value).abs() < 1e-8);
    }

    #[test]
    fn test_gae_multi_step() {
        // Three non-terminal steps, last_value = 0 (terminal boundary)
        let cfg = PpoConfig {
            discount_gamma: 1.0,
            gae_lambda: 1.0,
            normalize_advantages: false,
            ..Default::default()
        };
        // rewards = [1, 1, 1], values = [0, 0, 0], last_value = 0
        // δ_2 = 1 + 1*0 - 0 = 1   => GAE_2 = 1
        // δ_1 = 1 + 1*0 - 0 = 1   => GAE_1 = 1 + 1*1*1 = 2
        // δ_0 = 1 + 1*0 - 0 = 1   => GAE_0 = 1 + 1*1*2 = 3
        let steps: Vec<PpoStep> = (0..3).map(|_| make_step(0.0, 0.0, 0.0, 1.0, false)).collect();
        let res = compute_gae(&steps, 0.0, &cfg);
        assert!((res.advantages[0] - 3.0).abs() < 1e-4, "advantages[0]={}", res.advantages[0]);
        assert!((res.advantages[1] - 2.0).abs() < 1e-4);
        assert!((res.advantages[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_gae_discount_effect() {
        // With γ < 1 future rewards are discounted
        let cfg = PpoConfig {
            discount_gamma: 0.5,
            gae_lambda: 1.0,
            normalize_advantages: false,
            ..Default::default()
        };
        // Two steps, values = [0, 0], rewards = [0, 1], last_value = 0
        // δ_1 = 1 + 0.5*0 - 0 = 1   => GAE_1 = 1
        // δ_0 = 0 + 0.5*0 - 0 = 0   => GAE_0 = 0 + 0.5*1*1 = 0.5
        let steps = vec![
            make_step(0.0, 0.0, 0.0, 0.0, false),
            make_step(0.0, 0.0, 0.0, 1.0, false),
        ];
        let res = compute_gae(&steps, 0.0, &cfg);
        assert!((res.advantages[1] - 1.0).abs() < 1e-5);
        assert!((res.advantages[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gae_lambda_effect() {
        // λ = 0 collapses to 1-step TD residuals
        let cfg = PpoConfig {
            discount_gamma: 1.0,
            gae_lambda: 0.0,
            normalize_advantages: false,
            ..Default::default()
        };
        // values=[1,1], rewards=[0,0], last_value=1
        // δ_1 = 0 + 1*1 - 1 = 0  => GAE_1 = 0
        // δ_0 = 0 + 1*1 - 1 = 0  => GAE_0 = 0 + 1*0*0 = 0
        let steps = vec![
            make_step(0.0, 0.0, 1.0, 0.0, false),
            make_step(0.0, 0.0, 1.0, 0.0, false),
        ];
        let res = compute_gae(&steps, 1.0, &cfg);
        assert!((res.advantages[0]).abs() < 1e-5);
        assert!((res.advantages[1]).abs() < 1e-5);
    }

    #[test]
    fn test_gae_value_targets() {
        // value_targets[t] = advantages[t] + values[t]
        let cfg = no_norm_config();
        let steps = vec![
            make_step(0.0, 0.0, 1.0, 2.0, false),
            make_step(0.0, 0.0, 1.5, 3.0, true),
        ];
        let res = compute_gae(&steps, 0.0, &cfg);
        for (i, step) in steps.iter().enumerate() {
            let expected_vt = res.advantages[i] + step.value;
            assert!(
                (res.value_targets[i] - expected_vt).abs() < 1e-5,
                "value_targets[{}]: expected {}, got {}",
                i,
                expected_vt,
                res.value_targets[i]
            );
        }
    }

    // ── PPO loss tests ───────────────────────────────────────────────────

    #[test]
    fn test_ppo_loss_no_clipping() {
        // ratio = 1 for all steps (same policy): clip should not activate
        let cfg = no_norm_config();
        let steps: Vec<PpoStep> = (0..4)
            .map(|i| make_step(-1.0, -1.0, 0.5, i as f32, false))
            .collect();
        let adv = vec![1.0f32; 4];
        let vt = vec![1.0f32; 4];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ppo loss failed");
        assert!((res.clip_fraction).abs() < 1e-8, "no clips expected, got {}", res.clip_fraction);
        assert!((res.mean_ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ppo_loss_with_clipping() {
        // Large ratio → should be clipped
        let cfg = no_norm_config();
        // ratio = exp(2.0) ≈ 7.39, well above 1 + 0.2
        let step = make_step(0.0, -2.0, 0.5, 1.0, false);
        let adv = vec![1.0f32];
        let vt = vec![1.0f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ppo loss failed");
        // Policy loss should be clipped: -min(r*A, clip*A) = -(1.2 * 1.0) = -1.2
        assert!(
            (res.policy_loss - (-1.2)).abs() < 1e-5,
            "expected policy_loss ≈ -1.2, got {}",
            res.policy_loss
        );
    }

    #[test]
    fn test_ppo_loss_clip_fraction() {
        // 2 out of 4 steps are clipped
        let cfg = no_norm_config();
        let steps = vec![
            make_step(0.0, -2.0, 0.5, 1.0, false),   // ratio ≈ 7.39, clipped
            make_step(-2.0, 0.0, 0.5, 1.0, false),   // ratio ≈ 0.135, clipped
            make_step(-1.0, -1.0, 0.5, 1.0, false),  // ratio = 1, not clipped
            make_step(-1.1, -1.0, 0.5, 1.0, false),  // ratio ≈ 0.905, not clipped (|r-1|=0.095 < 0.2)
        ];
        let adv = vec![1.0f32; 4];
        let vt = vec![1.0f32; 4];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ppo loss failed");
        assert!((res.clip_fraction - 0.5).abs() < 1e-5, "expected 0.5, got {}", res.clip_fraction);
    }

    #[test]
    fn test_ppo_loss_normalize_advantages() {
        // When normalize_advantages=true the mean_advantage should be near 0
        let cfg = PpoConfig { normalize_advantages: true, ..Default::default() };
        let steps: Vec<PpoStep> = (0..5)
            .map(|_| make_step(-1.0, -1.0, 0.5, 1.0, false))
            .collect();
        // Non-trivial advantages
        let adv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vt = vec![0.5f32; 5];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ppo loss failed");
        // After normalisation mean should be ≈ 0
        assert!(
            res.mean_advantage.abs() < 1e-5,
            "expected mean_advantage ≈ 0, got {}",
            res.mean_advantage
        );
    }

    #[test]
    fn test_ppo_loss_approx_kl() {
        // approx_kl = mean(old_log_prob - log_prob)
        let cfg = no_norm_config();
        // log_prob = -1.0, old_log_prob = -0.5 → kl = -0.5 - (-1.0) = 0.5
        let steps: Vec<PpoStep> = (0..3)
            .map(|_| make_step(-1.0, -0.5, 0.5, 1.0, false))
            .collect();
        let adv = vec![1.0f32; 3];
        let vt = vec![1.0f32; 3];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ppo loss failed");
        assert!((res.approx_kl - 0.5).abs() < 1e-5, "expected 0.5, got {}", res.approx_kl);
    }

    // ── PpoTrainer tests ─────────────────────────────────────────────────

    #[test]
    fn test_ppo_trainer_update() {
        let cfg = PpoConfig { normalize_advantages: false, ..Default::default() };
        let mut trainer = PpoTrainer::new(cfg);
        let steps: Vec<PpoStep> = (0..4)
            .map(|_| make_step(-1.0, -1.0, 0.5, 1.0, false))
            .collect();
        let result = trainer.update(&steps, 0.5).expect("update failed");
        assert!(result.total_loss.is_finite());
        assert_eq!(trainer.total_steps(), 4);
    }

    #[test]
    fn test_ppo_trainer_history() {
        let cfg = no_norm_config();
        let mut trainer = PpoTrainer::new(cfg);
        let steps: Vec<PpoStep> = (0..2)
            .map(|_| make_step(-1.0, -1.0, 0.5, 1.0, true))
            .collect();
        trainer.update(&steps, 0.0).expect("update 1 failed");
        trainer.update(&steps, 0.0).expect("update 2 failed");
        assert_eq!(trainer.history().len(), 2);
        assert_eq!(trainer.total_steps(), 4);
    }

    #[test]
    fn test_ppo_trainer_early_stop() {
        let cfg = no_norm_config();
        let mut trainer = PpoTrainer::new(cfg);
        // Ensure no stop before any update
        assert!(!trainer.should_early_stop(0.01));

        // Create steps where old_log_prob >> log_prob → large KL
        let steps: Vec<PpoStep> = (0..3)
            .map(|_| make_step(-5.0, -0.1, 0.5, 1.0, false))
            .collect();
        trainer.update(&steps, 0.0).expect("update failed");
        // approx_kl = mean(old_log_prob - log_prob) = -0.1 - (-5.0) = 4.9
        assert!(trainer.should_early_stop(0.01), "expected early stop with high KL");
        assert!(!trainer.should_early_stop(10.0), "should not stop with very high threshold");
    }

    // ── Error handling tests ─────────────────────────────────────────────

    #[test]
    fn test_ppo_error_display() {
        let e1 = PpoError::EmptyBatch;
        assert!(e1.to_string().contains("empty"));

        let e2 = PpoError::LengthMismatch { steps: 5, advantages: 3 };
        let s = e2.to_string();
        assert!(s.contains("5") && s.contains("3"), "got: {}", s);

        let e3 = PpoError::NumericalError("NaN detected".to_string());
        assert!(e3.to_string().contains("NaN"));
    }

    #[test]
    fn test_ppo_loss_empty_batch() {
        let cfg = PpoConfig::default();
        let result = compute_ppo_loss(&[], &[], &[], &cfg);
        assert!(matches!(result, Err(PpoError::EmptyBatch)));
    }

    #[test]
    fn test_ppo_loss_length_mismatch() {
        let cfg = PpoConfig::default();
        let steps = vec![make_step(-1.0, -1.0, 0.5, 1.0, false)];
        let result = compute_ppo_loss(&steps, &[1.0, 2.0], &[1.0, 2.0], &cfg);
        assert!(matches!(result, Err(PpoError::LengthMismatch { .. })));
    }

    // ── Additional GAE tests ─────────────────────────────────────────────

    #[test]
    fn test_gae_gamma_one_reduces_to_cumulative_return() {
        // With γ=1.0 and λ=1.0, values=0: GAE_t = sum of future rewards
        let cfg = PpoConfig {
            discount_gamma: 1.0,
            gae_lambda: 1.0,
            normalize_advantages: false,
            ..Default::default()
        };
        // rewards = [2, 3, 5], values = [0,0,0], last_value = 0
        // GAE_2 = 5, GAE_1 = 3+5=8, GAE_0 = 2+3+5=10
        let steps = vec![
            make_step(0.0, 0.0, 0.0, 2.0, false),
            make_step(0.0, 0.0, 0.0, 3.0, false),
            make_step(0.0, 0.0, 0.0, 5.0, false),
        ];
        let res = compute_gae(&steps, 0.0, &cfg);
        assert!((res.advantages[0] - 10.0).abs() < 1e-4, "GAE_0={}", res.advantages[0]);
        assert!((res.advantages[1] - 8.0).abs() < 1e-4, "GAE_1={}", res.advantages[1]);
        assert!((res.advantages[2] - 5.0).abs() < 1e-4, "GAE_2={}", res.advantages[2]);
    }

    #[test]
    fn test_gae_lambda_zero_reduces_to_td_error() {
        // With λ=0 GAE_t = δ_t (one-step TD residual)
        let cfg = PpoConfig {
            discount_gamma: 0.99,
            gae_lambda: 0.0,
            normalize_advantages: false,
            ..Default::default()
        };
        // values=[2,3], rewards=[1,1], last_value=4
        // δ_1 = 1 + 0.99*4 - 3 = 1 + 3.96 - 3 = 1.96
        // δ_0 = 1 + 0.99*3 - 2 = 1 + 2.97 - 2 = 1.97
        // With λ=0: GAE_0=δ_0, GAE_1=δ_1
        let steps = vec![
            make_step(0.0, 0.0, 2.0, 1.0, false),
            make_step(0.0, 0.0, 3.0, 1.0, false),
        ];
        let res = compute_gae(&steps, 4.0, &cfg);
        let expected_1 = 1.0 + 0.99 * 4.0 - 3.0;
        let expected_0 = 1.0 + 0.99 * 3.0 - 2.0;
        assert!((res.advantages[1] - expected_1).abs() < 1e-4, "GAE_1={} expected={}", res.advantages[1], expected_1);
        assert!((res.advantages[0] - expected_0).abs() < 1e-4, "GAE_0={} expected={}", res.advantages[0], expected_0);
    }

    #[test]
    fn test_gae_with_known_numerical_values() {
        // Exact hand-computed test
        // γ=0.9, λ=0.8, values=[1.0, 2.0], rewards=[1.0, 2.0], last_value=3.0
        // δ_1 = 2.0 + 0.9*3.0 - 2.0 = 2.7
        // GAE_1 = δ_1 = 2.7 (last step; no further GAE accumulation from after)
        // δ_0 = 1.0 + 0.9*2.0 - 1.0 = 1.8
        // GAE_0 = δ_0 + 0.9*0.8*GAE_1 = 1.8 + 0.72*2.7 = 1.8 + 1.944 = 3.744
        let cfg = PpoConfig {
            discount_gamma: 0.9,
            gae_lambda: 0.8,
            normalize_advantages: false,
            ..Default::default()
        };
        let steps = vec![
            make_step(0.0, 0.0, 1.0, 1.0, false),
            make_step(0.0, 0.0, 2.0, 2.0, false),
        ];
        let res = compute_gae(&steps, 3.0, &cfg);
        let expected_1 = 2.0 + 0.9 * 3.0 - 2.0;
        let expected_0 = 1.0 + 0.9 * 2.0 - 1.0 + 0.9 * 0.8 * expected_1;
        assert!((res.advantages[1] - expected_1).abs() < 1e-4, "got {}", res.advantages[1]);
        assert!((res.advantages[0] - expected_0).abs() < 1e-4, "got {}", res.advantages[0]);
    }

    // ── Additional clipped-surrogate tests ──────────────────────────────

    #[test]
    fn test_clipped_objective_ratio_below_lower_bound() {
        // When ratio < 1-ε and advantage > 0: -min(ratio*A, clipped*A) = -ratio*A
        // because ratio*A < clipped*A, so min selects the smaller (ratio*A)
        let eps = 0.2_f32;
        let cfg = PpoConfig {
            clip_epsilon: eps,
            normalize_advantages: false,
            ..Default::default()
        };
        // ratio = exp(-3) ≈ 0.04979, well below 1-0.2=0.8
        let step = make_step(-3.0, 0.0, 0.5, 1.0, false);
        let adv = vec![1.0_f32];
        let vt = vec![1.0_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        // -min(ratio*1, 0.8*1) = -ratio = -exp(-3) ≈ -0.04979
        let expected_ratio = (-3.0_f32).exp();
        assert!((res.policy_loss - (-expected_ratio)).abs() < 1e-5, "policy_loss={} expected={}", res.policy_loss, -expected_ratio);
        // The step IS clipped (|ratio-1| = |0.0498-1| = 0.95 > 0.2)
        assert_eq!(res.clip_fraction, 1.0, "should be clipped");
    }

    #[test]
    fn test_clipped_objective_ratio_above_upper_bound() {
        // ratio > 1+ε: clipped to 1+ε, policy loss = -(1+ε)*A when A>0
        let eps = 0.2_f32;
        let cfg = PpoConfig {
            clip_epsilon: eps,
            normalize_advantages: false,
            ..Default::default()
        };
        // ratio = exp(3) ≈ 20.09, well above 1.2
        let step = make_step(3.0, 0.0, 0.5, 1.0, false);
        let adv = vec![1.0_f32];
        let vt = vec![1.0_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        // -min(20.09*1, 1.2*1) = -1.2
        assert!((res.policy_loss - (-1.2)).abs() < 1e-4, "policy_loss={}", res.policy_loss);
    }

    #[test]
    fn test_clipped_objective_ratio_in_range() {
        // ratio in (1-ε, 1+ε): no clipping, policy loss = -ratio * A
        let eps = 0.2_f32;
        let cfg = PpoConfig {
            clip_epsilon: eps,
            normalize_advantages: false,
            ..Default::default()
        };
        // ratio = exp(0.1) ≈ 1.105, inside (0.8, 1.2)
        let step = make_step(0.1, 0.0, 0.5, 1.0, false);
        let adv = vec![1.0_f32];
        let vt = vec![0.0_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        let expected_ratio = (0.1_f32).exp();
        let expected_policy = -(expected_ratio * 1.0);
        assert!((res.policy_loss - expected_policy).abs() < 1e-4, "policy_loss={} expected={}", res.policy_loss, expected_policy);
    }

    // ── Value loss tests ─────────────────────────────────────────────────

    #[test]
    fn test_value_loss_mse_formula() {
        // value_loss = 0.5 * mean((V - V_target)^2)
        // Single step: V=1.0, V_target=3.0 → 0.5*(1-3)^2 = 2.0
        let cfg = PpoConfig {
            normalize_advantages: false,
            value_coeff: 1.0,
            entropy_coeff: 0.0,
            ..Default::default()
        };
        let step = make_step(-1.0, -1.0, 1.0, 0.0, false);
        let adv = vec![0.0_f32];
        let vt = vec![3.0_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        assert!((res.value_loss - 2.0).abs() < 1e-5, "value_loss={}", res.value_loss);
    }

    #[test]
    fn test_value_loss_zero_when_perfect() {
        // V = V_target → value_loss = 0
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let step = make_step(-1.0, -1.0, 2.5, 0.0, false);
        let adv = vec![0.0_f32];
        let vt = vec![2.5_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        assert!((res.value_loss).abs() < 1e-6, "value_loss should be 0, got {}", res.value_loss);
    }

    // ── Entropy bonus tests ──────────────────────────────────────────────

    #[test]
    fn test_entropy_loss_equals_negative_mean_log_prob() {
        // entropy_loss = mean(-log_prob)
        let cfg = PpoConfig {
            normalize_advantages: false,
            entropy_coeff: 1.0,
            ..Default::default()
        };
        // log_probs = [-1.0, -2.0] → entropy_loss = (1.0+2.0)/2 = 1.5
        let steps = vec![
            make_step(-1.0, -1.0, 0.5, 0.0, false),
            make_step(-2.0, -2.0, 0.5, 0.0, false),
        ];
        let adv = vec![0.0_f32; 2];
        let vt = vec![0.5_f32; 2];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ok");
        assert!((res.entropy_loss - 1.5).abs() < 1e-5, "entropy_loss={}", res.entropy_loss);
    }

    #[test]
    fn test_entropy_uniform_distribution_n_tokens() {
        // For uniform over n tokens: log_prob = -log(n) per token
        // entropy_loss = mean(-log_prob) = log(n)
        let n = 4_usize;
        let log_prob = -(n as f32).ln();
        let cfg = PpoConfig {
            normalize_advantages: false,
            entropy_coeff: 1.0,
            ..Default::default()
        };
        let steps: Vec<PpoStep> = (0..n).map(|_| make_step(log_prob, log_prob, 0.5, 0.0, false)).collect();
        let adv = vec![0.0_f32; n];
        let vt = vec![0.5_f32; n];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ok");
        let expected = (n as f32).ln();
        assert!((res.entropy_loss - expected).abs() < 1e-4, "entropy_loss={} expected ln({n})={}", res.entropy_loss, expected);
    }

    // ── KL divergence tests ──────────────────────────────────────────────

    #[test]
    fn test_approx_kl_zero_for_identical_distributions() {
        // When log_prob == old_log_prob, approx_kl = 0
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let steps: Vec<PpoStep> = (0..3).map(|_| make_step(-1.5, -1.5, 0.5, 1.0, false)).collect();
        let adv = vec![1.0_f32; 3];
        let vt = vec![1.0_f32; 3];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ok");
        assert!((res.approx_kl).abs() < 1e-6, "kl should be 0, got {}", res.approx_kl);
    }

    #[test]
    fn test_approx_kl_positive_when_diverged() {
        // When old_log_prob > log_prob: KL = old - new > 0
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let steps: Vec<PpoStep> = (0..4).map(|_| make_step(-3.0, -1.0, 0.5, 1.0, false)).collect();
        let adv = vec![1.0_f32; 4];
        let vt = vec![1.0_f32; 4];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ok");
        // approx_kl = mean(-1.0 - (-3.0)) = 2.0
        assert!((res.approx_kl - 2.0).abs() < 1e-5, "kl={}", res.approx_kl);
    }

    // ── Policy gradient direction test ───────────────────────────────────

    #[test]
    fn test_policy_gradient_direction_matches_reward() {
        // When ratio=1 (unchanged policy) and advantage>0, policy_loss<0
        // meaning gradient update increases chosen actions' log probs
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let steps: Vec<PpoStep> = (0..4).map(|_| make_step(-1.0, -1.0, 0.5, 1.0, false)).collect();
        let pos_adv = vec![1.0_f32; 4];
        let neg_adv = vec![-1.0_f32; 4];
        let vt = vec![0.5_f32; 4];
        let pos_res = compute_ppo_loss(&steps, &pos_adv, &vt, &cfg).expect("ok");
        let neg_res = compute_ppo_loss(&steps, &neg_adv, &vt, &cfg).expect("ok");
        // Positive advantage → negative policy loss (gradient goes up)
        assert!(pos_res.policy_loss < 0.0, "policy_loss should be negative for positive advantage, got {}", pos_res.policy_loss);
        // Negative advantage → positive policy loss
        assert!(neg_res.policy_loss > 0.0, "policy_loss should be positive for negative advantage, got {}", neg_res.policy_loss);
    }

    // ── Trainer accumulation tests ───────────────────────────────────────

    #[test]
    fn test_ppo_trainer_accumulates_total_steps() {
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let mut trainer = PpoTrainer::new(cfg);
        let steps_a: Vec<PpoStep> = (0..3).map(|_| make_step(-1.0, -1.0, 0.5, 1.0, false)).collect();
        let steps_b: Vec<PpoStep> = (0..5).map(|_| make_step(-1.0, -1.0, 0.5, 1.0, true)).collect();
        trainer.update(&steps_a, 0.5).expect("update a");
        trainer.update(&steps_b, 0.0).expect("update b");
        assert_eq!(trainer.total_steps(), 8, "expected 3+5=8 total steps");
        assert_eq!(trainer.history().len(), 2);
    }

    #[test]
    fn test_ppo_trainer_mean_policy_loss_empty() {
        let trainer = PpoTrainer::new(PpoConfig::default());
        assert!((trainer.mean_policy_loss()).abs() < 1e-8, "empty should return 0");
    }

    #[test]
    fn test_ppo_trainer_mean_policy_loss_multiple_updates() {
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let mut trainer = PpoTrainer::new(cfg);
        let steps: Vec<PpoStep> = vec![make_step(-1.0, -1.0, 0.5, 1.0, false)];
        let r1 = trainer.update(&steps, 0.0).expect("update 1");
        let r2 = trainer.update(&steps, 0.0).expect("update 2");
        let expected = (r1.policy_loss + r2.policy_loss) / 2.0;
        assert!((trainer.mean_policy_loss() - expected).abs() < 1e-5, "mean policy loss mismatch");
    }

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_ppo_loss_zero_advantages() {
        // Zero advantages → policy_loss = 0 (no gradient signal)
        let cfg = PpoConfig {
            normalize_advantages: false,
            entropy_coeff: 0.0,
            ..Default::default()
        };
        let steps: Vec<PpoStep> = (0..4).map(|_| make_step(-1.0, -1.0, 0.5, 0.0, false)).collect();
        let adv = vec![0.0_f32; 4];
        let vt = vec![0.5_f32; 4];
        let res = compute_ppo_loss(&steps, &adv, &vt, &cfg).expect("ok");
        assert!((res.policy_loss).abs() < 1e-6, "policy_loss should be 0, got {}", res.policy_loss);
    }

    #[test]
    fn test_ppo_loss_single_token() {
        // Single step should work correctly
        let cfg = PpoConfig {
            normalize_advantages: false,
            ..Default::default()
        };
        let step = make_step(-1.0, -1.0, 1.0, 2.0, true);
        let adv = vec![0.5_f32];
        let vt = vec![1.5_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("single token ok");
        assert!(res.total_loss.is_finite());
        assert_eq!(res.clip_fraction, 0.0); // ratio=1, not clipped
    }

    #[test]
    fn test_ppo_loss_very_large_ratio_still_clipped() {
        // Very large ratio (e.g., exp(100)) should still be clipped to 1+ε
        let cfg = PpoConfig {
            clip_epsilon: 0.2,
            normalize_advantages: false,
            entropy_coeff: 0.0,
            value_coeff: 0.0,
            ..Default::default()
        };
        let step = make_step(100.0, 0.0, 0.5, 1.0, false);
        let adv = vec![1.0_f32];
        let vt = vec![0.5_f32];
        let res = compute_ppo_loss(&[step], &adv, &vt, &cfg).expect("ok");
        // -min(exp(100), 1.2) * 1.0 = -1.2
        assert!((res.policy_loss - (-1.2)).abs() < 1e-4, "policy_loss={}", res.policy_loss);
        assert_eq!(res.clip_fraction, 1.0);
    }

    #[test]
    fn test_gae_empty_steps() {
        // GAE on empty slice should produce empty result
        let cfg = PpoConfig::default();
        let res = compute_gae(&[], 1.0, &cfg);
        assert!(res.advantages.is_empty());
        assert!(res.value_targets.is_empty());
    }
}
