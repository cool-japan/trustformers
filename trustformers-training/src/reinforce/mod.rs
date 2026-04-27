//! REINFORCE (Williams, 1992) with variance reduction techniques.
//!
//! REINFORCE is the foundational policy gradient algorithm. This implementation
//! supports multiple baseline methods for variance reduction, including:
//! - Running EMA baseline
//! - Self-Critical Sequence Training (SCST, Rennie et al. 2017)
//! - Value function baseline
//!
//! Entropy regularization is included to encourage exploration.

use std::fmt;

// ──────────────────────────────────────────────
// Enumerations
// ──────────────────────────────────────────────

/// Method used to compute the advantage baseline.
#[derive(Debug, Clone, PartialEq)]
pub enum BaselineMethod {
    /// No baseline — pure REINFORCE (high variance).
    None,
    /// Exponential moving average of past rewards.
    RunningMean,
    /// Greedy-decode reward used as baseline (Self-Critical Sequence Training).
    SelfCritical,
    /// Learned value-function baseline (value provided externally via `baseline_reward`).
    ValueFunction,
}

/// Reward normalization strategy applied before advantage computation.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardNormalization {
    /// Identity — no normalization.
    None,
    /// Z-score: (r − μ) / (σ + 1e-8).
    Zscore,
    /// Clamp to `[min_reward, max_reward]`.
    Clip { min_reward: f32, max_reward: f32 },
    /// Element-wise tanh.
    Tanh,
}

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Configuration for REINFORCE training.
#[derive(Debug, Clone)]
pub struct ReinforceConfig {
    /// Adam/SGD learning rate (default 1e-5).
    pub learning_rate: f32,
    /// Baseline method for variance reduction.
    pub baseline_method: BaselineMethod,
    /// Number of response samples per prompt (default 4).
    pub num_samples: usize,
    /// Entropy regularization coefficient (default 0.01).
    pub entropy_coeff: f32,
    /// Maximum gradient norm for clipping (default 1.0).
    pub max_grad_norm: f32,
    /// Reward normalization strategy applied before loss.
    pub reward_normalizer: RewardNormalization,
    /// If true, subtract mean and divide by std of advantages (default true).
    pub whiten_rewards: bool,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            baseline_method: BaselineMethod::RunningMean,
            num_samples: 4,
            entropy_coeff: 0.01,
            max_grad_norm: 1.0,
            reward_normalizer: RewardNormalization::None,
            whiten_rewards: true,
        }
    }
}

// ──────────────────────────────────────────────
// Example
// ──────────────────────────────────────────────

/// A single (prompt, response, reward) training example for REINFORCE.
#[derive(Debug, Clone)]
pub struct ReinforceExample {
    /// Prompt token IDs.
    pub prompt: Vec<u32>,
    /// Sampled response token IDs.
    pub response: Vec<u32>,
    /// Scalar reward from the reward model / environment.
    pub reward: f32,
    /// Per-token log-probabilities under the current policy (len == response.len()).
    pub log_probs: Vec<f32>,
    /// Optional greedy/value baseline reward for variance reduction.
    pub baseline_reward: Option<f32>,
}

// ──────────────────────────────────────────────
// Loss output
// ──────────────────────────────────────────────

/// Aggregated statistics from one REINFORCE update.
#[derive(Debug, Clone)]
pub struct ReinforceLossOutput {
    /// policy_loss − entropy_bonus.
    pub total_loss: f32,
    /// Mean per-token policy gradient loss (−log_prob × advantage).
    pub policy_loss: f32,
    /// Entropy regularization term (positive value; subtracted from total).
    pub entropy_loss: f32,
    /// Mean reward across examples.
    pub mean_reward: f32,
    /// Mean advantage across examples.
    pub mean_advantage: f32,
    /// Number of examples.
    pub num_examples: usize,
    /// Total number of response tokens.
    pub num_tokens: usize,
}

impl fmt::Display for ReinforceLossOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReinforceLoss {{ total: {:.4}, policy: {:.4}, entropy: {:.4}, \
             reward: {:.4}, advantage: {:.4}, examples: {}, tokens: {} }}",
            self.total_loss,
            self.policy_loss,
            self.entropy_loss,
            self.mean_reward,
            self.mean_advantage,
            self.num_examples,
            self.num_tokens,
        )
    }
}

// ──────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────

/// Errors that can arise during REINFORCE loss computation.
#[derive(Debug)]
pub enum ReinforceError {
    /// The example slice was empty.
    EmptyBatch,
    /// `log_probs` length does not match `response` length for the given example index.
    LengthMismatch(usize),
    /// A reward or log-probability was not finite (NaN or Inf).
    NonFiniteValue { example_idx: usize, field: &'static str },
}

impl fmt::Display for ReinforceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReinforceError::EmptyBatch => write!(f, "REINFORCE: batch contains no examples"),
            ReinforceError::LengthMismatch(idx) => write!(
                f,
                "REINFORCE: example {idx}: log_probs.len() != response.len()"
            ),
            ReinforceError::NonFiniteValue { example_idx, field } => write!(
                f,
                "REINFORCE: example {example_idx}: non-finite value in field '{field}'"
            ),
        }
    }
}

impl std::error::Error for ReinforceError {}

// ──────────────────────────────────────────────
// Reward normalization
// ──────────────────────────────────────────────

/// Normalize a slice of rewards according to the chosen strategy.
pub fn normalize_rewards(rewards: &[f32], method: &RewardNormalization) -> Vec<f32> {
    match method {
        RewardNormalization::None => rewards.to_vec(),
        RewardNormalization::Zscore => {
            let n = rewards.len() as f32;
            if n == 0.0 {
                return vec![];
            }
            let mean = rewards.iter().sum::<f32>() / n;
            let variance =
                rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
            let std = variance.sqrt();
            let denom = std + 1e-8;
            rewards.iter().map(|r| (r - mean) / denom).collect()
        }
        RewardNormalization::Clip { min_reward, max_reward } => rewards
            .iter()
            .map(|r| r.max(*min_reward).min(*max_reward))
            .collect(),
        RewardNormalization::Tanh => rewards.iter().map(|r| r.tanh()).collect(),
    }
}

// ──────────────────────────────────────────────
// Entropy helper
// ──────────────────────────────────────────────

/// Approximate per-token entropy contribution from its log-probability.
///
/// For a categorical distribution, the entropy contribution of the chosen token
/// is approximated as `−log_prob` (the self-information).
pub fn compute_entropy(log_prob: f32) -> f32 {
    -log_prob
}

// ──────────────────────────────────────────────
// Whitening helper
// ──────────────────────────────────────────────

fn whiten(values: &[f32]) -> Vec<f32> {
    let n = values.len() as f32;
    if n == 0.0 {
        return vec![];
    }
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let denom = std + 1e-8;
    values.iter().map(|v| (v - mean) / denom).collect()
}

// ──────────────────────────────────────────────
// Core loss function
// ──────────────────────────────────────────────

/// Compute the REINFORCE loss for a batch of examples.
///
/// Steps:
/// 1. Validate inputs.
/// 2. Normalize rewards.
/// 3. Compute baselines from the caller-supplied baseline rewards (if any).
/// 4. Compute per-example advantage = reward − baseline.
/// 5. Optionally whiten advantages.
/// 6. Per-token loss = −log_prob × advantage.
/// 7. Entropy bonus = entropy_coeff × mean(−log_prob).
/// 8. Total = mean_token_loss − entropy_bonus.
pub fn compute_reinforce_loss(
    examples: &[ReinforceExample],
    config: &ReinforceConfig,
) -> Result<ReinforceLossOutput, ReinforceError> {
    if examples.is_empty() {
        return Err(ReinforceError::EmptyBatch);
    }

    // Validate
    for (idx, ex) in examples.iter().enumerate() {
        if ex.log_probs.len() != ex.response.len() {
            return Err(ReinforceError::LengthMismatch(idx));
        }
        if !ex.reward.is_finite() {
            return Err(ReinforceError::NonFiniteValue {
                example_idx: idx,
                field: "reward",
            });
        }
        for lp in &ex.log_probs {
            if !lp.is_finite() {
                return Err(ReinforceError::NonFiniteValue {
                    example_idx: idx,
                    field: "log_probs",
                });
            }
        }
    }

    // Normalize rewards
    let raw_rewards: Vec<f32> = examples.iter().map(|e| e.reward).collect();
    let rewards = normalize_rewards(&raw_rewards, &config.reward_normalizer);

    // Compute baselines per example
    let baselines: Vec<f32> = examples
        .iter()
        .map(|e| match &config.baseline_method {
            BaselineMethod::None => 0.0,
            BaselineMethod::RunningMean => 0.0, // trainer updates baseline separately
            BaselineMethod::SelfCritical => e.baseline_reward.unwrap_or(0.0),
            BaselineMethod::ValueFunction => e.baseline_reward.unwrap_or(0.0),
        })
        .collect();

    // Raw advantages
    let raw_advantages: Vec<f32> = rewards
        .iter()
        .zip(baselines.iter())
        .map(|(r, b)| r - b)
        .collect();

    // Optionally whiten
    let advantages = if config.whiten_rewards {
        whiten(&raw_advantages)
    } else {
        raw_advantages
    };

    // Compute per-token losses and entropy
    let mut total_policy_loss = 0.0_f32;
    let mut total_entropy = 0.0_f32;
    let mut total_tokens = 0usize;

    for (ex, adv) in examples.iter().zip(advantages.iter()) {
        let n = ex.log_probs.len();
        if n == 0 {
            continue;
        }
        for lp in &ex.log_probs {
            total_policy_loss += -lp * adv;
            total_entropy += compute_entropy(*lp);
        }
        total_tokens += n;
    }

    let mean_policy_loss = if total_tokens > 0 {
        total_policy_loss / total_tokens as f32
    } else {
        0.0
    };
    let mean_entropy = if total_tokens > 0 {
        total_entropy / total_tokens as f32
    } else {
        0.0
    };
    let entropy_bonus = config.entropy_coeff * mean_entropy;
    let total_loss = mean_policy_loss - entropy_bonus;

    let mean_reward = raw_rewards.iter().sum::<f32>() / raw_rewards.len() as f32;
    let mean_advantage = advantages.iter().sum::<f32>() / advantages.len() as f32;

    Ok(ReinforceLossOutput {
        total_loss,
        policy_loss: mean_policy_loss,
        entropy_loss: entropy_bonus,
        mean_reward,
        mean_advantage,
        num_examples: examples.len(),
        num_tokens: total_tokens,
    })
}

// ──────────────────────────────────────────────
// Baseline tracker
// ──────────────────────────────────────────────

/// Maintains and updates a running baseline for variance reduction.
#[derive(Debug, Clone)]
pub struct ReinforceBaseline {
    /// Baseline method used.
    pub method: BaselineMethod,
    /// Current EMA estimate of the mean reward.
    pub running_mean: f32,
    /// EMA decay factor (default 0.99).
    pub ema_decay: f32,
    /// Number of update calls so far.
    pub count: usize,
}

impl ReinforceBaseline {
    /// Construct a new baseline tracker.
    pub fn new(method: BaselineMethod) -> Self {
        Self {
            method,
            running_mean: 0.0,
            ema_decay: 0.99,
            count: 0,
        }
    }

    /// Compute per-example baseline values.
    ///
    /// - `None` → all zeros.
    /// - `RunningMean` → current EMA (broadcast to all examples).
    /// - `SelfCritical` / `ValueFunction` → each example's `baseline_reward` or 0.0.
    pub fn compute_baseline(&self, examples: &[ReinforceExample]) -> Vec<f32> {
        match &self.method {
            BaselineMethod::None => vec![0.0; examples.len()],
            BaselineMethod::RunningMean => vec![self.running_mean; examples.len()],
            BaselineMethod::SelfCritical => examples
                .iter()
                .map(|e| e.baseline_reward.unwrap_or(0.0))
                .collect(),
            BaselineMethod::ValueFunction => examples
                .iter()
                .map(|e| e.baseline_reward.unwrap_or(0.0))
                .collect(),
        }
    }

    /// Update the EMA running mean with a new batch of rewards.
    pub fn update(&mut self, rewards: &[f32]) {
        if rewards.is_empty() {
            return;
        }
        let batch_mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        if self.count == 0 {
            self.running_mean = batch_mean;
        } else {
            self.running_mean = self.ema_decay * self.running_mean + (1.0 - self.ema_decay) * batch_mean;
        }
        self.count += 1;
    }
}

// ──────────────────────────────────────────────
// Trainer
// ──────────────────────────────────────────────

/// Stateful REINFORCE trainer that integrates baseline tracking with loss computation.
pub struct ReinforceTrainer {
    /// Training configuration.
    pub config: ReinforceConfig,
    /// Baseline tracker.
    pub baseline: ReinforceBaseline,
    /// Number of completed training steps.
    pub step: usize,
    /// History of loss outputs for analysis.
    pub history: Vec<ReinforceLossOutput>,
}

impl ReinforceTrainer {
    /// Create a new trainer from the given configuration.
    pub fn new(config: ReinforceConfig) -> Self {
        let baseline = ReinforceBaseline::new(config.baseline_method.clone());
        Self {
            config,
            baseline,
            step: 0,
            history: Vec::new(),
        }
    }

    /// Compute the REINFORCE loss, updating the baseline and recording history.
    ///
    /// For `RunningMean` baseline the current EMA is injected as `baseline_reward`
    /// before loss computation, then the EMA is updated after.
    pub fn compute_loss(
        &mut self,
        mut examples: Vec<ReinforceExample>,
    ) -> Result<ReinforceLossOutput, ReinforceError> {
        // For RunningMean, inject the current running mean as baseline reward
        if self.config.baseline_method == BaselineMethod::RunningMean {
            let bm = self.baseline.running_mean;
            for ex in &mut examples {
                ex.baseline_reward = Some(bm);
            }
        }

        // Create a temporary config that uses SelfCritical (reads baseline_reward directly)
        // when method is RunningMean, so compute_reinforce_loss sees the injected value.
        let effective_config = if self.config.baseline_method == BaselineMethod::RunningMean {
            let mut c = self.config.clone();
            c.baseline_method = BaselineMethod::SelfCritical;
            c
        } else {
            self.config.clone()
        };

        let output = compute_reinforce_loss(&examples, &effective_config)?;

        // Update baseline with raw rewards
        let rewards: Vec<f32> = examples.iter().map(|e| e.reward).collect();
        self.baseline.update(&rewards);

        self.step += 1;
        self.history.push(output.clone());
        Ok(output)
    }

    /// Mean reward over the most recent `window` steps.
    pub fn mean_recent_reward(&self, window: usize) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let start = self.history.len().saturating_sub(window);
        let slice = &self.history[start..];
        slice.iter().map(|h| h.mean_reward).sum::<f32>() / slice.len() as f32
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(reward: f32, log_probs: Vec<f32>, baseline: Option<f32>) -> ReinforceExample {
        let n = log_probs.len();
        ReinforceExample {
            prompt: vec![1, 2, 3],
            response: (0..n as u32).collect(),
            reward,
            log_probs,
            baseline_reward: baseline,
        }
    }

    // ── Test 1: basic loss with no baseline ───────────────────────────────
    #[test]
    fn test_basic_loss_no_baseline() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let examples = vec![make_example(1.0, vec![-1.0, -2.0], None)];
        let out = compute_reinforce_loss(&examples, &config).expect("should succeed");
        // advantage = reward - 0 = 1.0
        // token losses: -(-1.0)*1.0 = 1.0, -(-2.0)*1.0 = 2.0 → mean = 1.5
        assert!((out.policy_loss - 1.5).abs() < 1e-5, "policy_loss={}", out.policy_loss);
        assert_eq!(out.num_examples, 1);
        assert_eq!(out.num_tokens, 2);
    }

    // ── Test 2: self-critical baseline ────────────────────────────────────
    #[test]
    fn test_self_critical_baseline() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::SelfCritical,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // reward=2.0, baseline=1.0 → advantage=1.0
        let examples = vec![make_example(2.0, vec![-1.0], Some(1.0))];
        let out = compute_reinforce_loss(&examples, &config).expect("should succeed");
        // token loss = -(-1.0)*1.0 = 1.0
        assert!((out.policy_loss - 1.0).abs() < 1e-5, "got {}", out.policy_loss);
    }

    // ── Test 3: value function baseline ───────────────────────────────────
    #[test]
    fn test_value_function_baseline() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::ValueFunction,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // reward=5.0, baseline=3.0 → advantage=2.0
        let examples = vec![make_example(5.0, vec![-2.0], Some(3.0))];
        let out = compute_reinforce_loss(&examples, &config).expect("should succeed");
        // token loss = -(-2.0)*2.0 = 4.0
        assert!((out.policy_loss - 4.0).abs() < 1e-5, "got {}", out.policy_loss);
    }

    // ── Test 4: reward normalization Zscore ───────────────────────────────
    #[test]
    fn test_normalize_zscore() {
        let rewards = vec![1.0_f32, 2.0, 3.0];
        let normalized = normalize_rewards(&rewards, &RewardNormalization::Zscore);
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be 0, got {mean}");
    }

    // ── Test 5: reward normalization Clip ─────────────────────────────────
    #[test]
    fn test_normalize_clip() {
        let rewards = vec![-5.0_f32, 0.5, 10.0];
        let normalized = normalize_rewards(
            &rewards,
            &RewardNormalization::Clip { min_reward: -1.0, max_reward: 1.0 },
        );
        assert!((normalized[0] - (-1.0)).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    // ── Test 6: reward normalization Tanh ────────────────────────────────
    #[test]
    fn test_normalize_tanh() {
        let rewards = vec![0.0_f32, 1.0, -1.0];
        let normalized = normalize_rewards(&rewards, &RewardNormalization::Tanh);
        assert!((normalized[0]).abs() < 1e-6);
        assert!((normalized[1] - 1.0_f32.tanh()).abs() < 1e-6);
        assert!((normalized[2] - (-1.0_f32).tanh()).abs() < 1e-6);
    }

    // ── Test 7: entropy bonus reduces total loss ──────────────────────────
    #[test]
    fn test_entropy_reduces_loss() {
        let base_config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let entropy_config = ReinforceConfig {
            entropy_coeff: 0.1,
            ..base_config.clone()
        };
        let examples = vec![make_example(1.0, vec![-1.0], None)];
        let out_base = compute_reinforce_loss(&examples, &base_config).expect("base");
        let out_entropy = compute_reinforce_loss(&examples, &entropy_config).expect("entropy");
        assert!(
            out_entropy.total_loss < out_base.total_loss,
            "entropy bonus should reduce loss: {} vs {}",
            out_entropy.total_loss, out_base.total_loss
        );
    }

    // ── Test 8: whitening normalizes advantages ───────────────────────────
    #[test]
    fn test_whitening() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: true,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // Two examples with very different rewards — whitening should normalize them
        let examples = vec![
            make_example(0.0, vec![-1.0], None),
            make_example(100.0, vec![-1.0], None),
        ];
        let out = compute_reinforce_loss(&examples, &config).expect("should succeed");
        // After whitening mean_advantage ≈ 0
        assert!(
            out.mean_advantage.abs() < 1e-4,
            "mean advantage should be ≈ 0 after whitening, got {}",
            out.mean_advantage
        );
    }

    // ── Test 9: empty batch error ─────────────────────────────────────────
    #[test]
    fn test_empty_batch_error() {
        let config = ReinforceConfig::default();
        let err = compute_reinforce_loss(&[], &config).unwrap_err();
        assert!(matches!(err, ReinforceError::EmptyBatch));
    }

    // ── Test 10: length mismatch error ────────────────────────────────────
    #[test]
    fn test_length_mismatch_error() {
        let config = ReinforceConfig::default();
        let mut ex = make_example(1.0, vec![-1.0, -2.0], None);
        ex.response = vec![1]; // len 1 vs log_probs len 2
        let err = compute_reinforce_loss(&[ex], &config).unwrap_err();
        assert!(matches!(err, ReinforceError::LengthMismatch(0)));
    }

    // ── Test 11: running mean baseline in trainer ─────────────────────────
    #[test]
    fn test_running_mean_baseline_updates() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::RunningMean,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let mut trainer = ReinforceTrainer::new(config);
        assert_eq!(trainer.baseline.count, 0);

        let examples = vec![make_example(2.0, vec![-1.0], None)];
        trainer.compute_loss(examples).expect("step 1");
        assert_eq!(trainer.baseline.count, 1);
        assert!((trainer.baseline.running_mean - 2.0).abs() < 1e-5);
    }

    // ── Test 12: trainer history and mean_recent_reward ───────────────────
    #[test]
    fn test_trainer_history() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let mut trainer = ReinforceTrainer::new(config);
        for reward in [1.0_f32, 2.0, 3.0] {
            let examples = vec![make_example(reward, vec![-1.0], None)];
            trainer.compute_loss(examples).expect("step");
        }
        assert_eq!(trainer.step, 3);
        assert_eq!(trainer.history.len(), 3);
        let recent = trainer.mean_recent_reward(2);
        // last 2 rewards: 2.0, 3.0 → mean = 2.5
        assert!((recent - 2.5).abs() < 1e-5, "recent mean = {recent}");
    }

    // ── Test 13: compute_entropy is non-negative for valid log probs ───────
    #[test]
    fn test_compute_entropy_non_negative() {
        // For negative log probs (probabilities in (0,1]), entropy = -log_prob >= 0
        for lp in [-0.1_f32, -1.0, -5.0, -0.001] {
            let e = compute_entropy(lp);
            assert!(e >= 0.0, "entropy should be >= 0 for lp={lp}, got {e}");
        }
    }

    // ── Test 14: ReinforceBaseline compute_baseline for None ──────────────
    #[test]
    fn test_baseline_none_returns_zeros() {
        let bl = ReinforceBaseline::new(BaselineMethod::None);
        let examples = vec![
            make_example(1.0, vec![-1.0], None),
            make_example(2.0, vec![-1.0], None),
        ];
        let baselines = bl.compute_baseline(&examples);
        assert_eq!(baselines, vec![0.0, 0.0]);
    }

    // ── Test 15: ReinforceError Display ───────────────────────────────────
    #[test]
    fn test_error_display() {
        let e1 = format!("{}", ReinforceError::EmptyBatch);
        assert!(e1.contains("no examples"), "got: {e1}");

        let e2 = format!("{}", ReinforceError::LengthMismatch(3));
        assert!(e2.contains('3'), "got: {e2}");

        let e3 = format!(
            "{}",
            ReinforceError::NonFiniteValue { example_idx: 0, field: "reward" }
        );
        assert!(e3.contains("reward"), "got: {e3}");
    }

    // ── Additional REINFORCE tests ────────────────────────────────────────

    #[test]
    fn test_return_computation_gamma_one_undiscounted() {
        // With γ=1 (no discounting), policy gradient = G_t * ∇log π_θ
        // The advantage equals total sum of future rewards when baseline=0
        // For a single example: advantage = reward, loss = -log_prob * reward
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // reward=3.0, log_probs=[-1.0] → loss = -(-1.0)*3.0 = 3.0
        let examples = vec![make_example(3.0, vec![-1.0], None)];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        assert!((out.policy_loss - 3.0).abs() < 1e-5, "policy_loss={}", out.policy_loss);
        assert!((out.mean_reward - 3.0).abs() < 1e-5, "mean_reward={}", out.mean_reward);
    }

    #[test]
    fn test_return_computation_gamma_zero_immediate_reward_only() {
        // With γ=0 (only immediate reward), the gradient comes from the immediate reward
        // The algorithm doesn't have an internal γ discounting per se; reward normalization
        // simulates different settings. We verify that loss scales linearly with reward.
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let ex1 = make_example(1.0, vec![-1.0], None);
        let ex2 = make_example(2.0, vec![-1.0], None);
        let r1 = compute_reinforce_loss(&[ex1], &config).expect("ok1");
        let r2 = compute_reinforce_loss(&[ex2], &config).expect("ok2");
        // r2 policy_loss should be 2x r1 (reward doubled)
        assert!((r2.policy_loss / r1.policy_loss - 2.0).abs() < 1e-5, "ratio={}", r2.policy_loss / r1.policy_loss);
    }

    #[test]
    fn test_baseline_subtraction_reduces_variance_conceptually() {
        // With zero advantage (reward == baseline), policy_loss should be 0
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::ValueFunction,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // reward=5.0, baseline=5.0 → advantage=0 → loss=0
        let examples = vec![make_example(5.0, vec![-1.0], Some(5.0))];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        assert!((out.policy_loss).abs() < 1e-5, "zero advantage → zero policy loss, got {}", out.policy_loss);
    }

    #[test]
    fn test_actor_critic_advantage_computation() {
        // Advantage A_t = G_t - V(s_t): positive when reward exceeds value estimate
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::ValueFunction,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // reward=3.0, V=1.0 → advantage=2.0 → loss = -(-1.0)*2.0 = 2.0
        let examples = vec![make_example(3.0, vec![-1.0], Some(1.0))];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        assert!((out.policy_loss - 2.0).abs() < 1e-5, "policy_loss={}", out.policy_loss);
    }

    #[test]
    fn test_gradient_zero_when_rewards_equal_baseline() {
        // All rewards equal to baseline → zero advantages → zero policy gradient
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::SelfCritical,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let examples = vec![
            make_example(2.0, vec![-1.0, -2.0], Some(2.0)),
            make_example(2.0, vec![-0.5, -1.5], Some(2.0)),
        ];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        assert!((out.policy_loss).abs() < 1e-5, "gradient should be 0 when all advantages=0, got {}", out.policy_loss);
    }

    #[test]
    fn test_normalization_of_returns_zero_mean_unit_variance() {
        // After Zscore normalization, mean≈0
        let rewards = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_rewards(&rewards, &RewardNormalization::Zscore);
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-5, "mean after zscore should be ~0, got {}", mean);
        // variance ≈ 1
        let var = normalized.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / normalized.len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "variance after zscore should be ~1, got {}", var);
    }

    #[test]
    fn test_entropy_regularization_with_nonzero_coeff() {
        // entropy_bonus = entropy_coeff * mean(-log_prob)
        // For log_prob=-2.0: entropy_loss = 2.0; with coeff=0.5 → bonus=1.0
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.5,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let examples = vec![make_example(0.0, vec![-2.0], None)];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        // policy_loss = -(-2.0)*0.0 = 0.0; entropy_bonus = 0.5*2.0 = 1.0
        // total_loss = 0.0 - 1.0 = -1.0
        assert!((out.entropy_loss - 1.0).abs() < 1e-5, "entropy_loss={}", out.entropy_loss);
        assert!((out.total_loss - (-1.0)).abs() < 1e-5, "total_loss={}", out.total_loss);
    }

    #[test]
    fn test_multi_episode_batch_aggregation() {
        // Multiple examples: policy_loss = mean over all tokens of all examples
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        // ex1: reward=1, log_probs=[-1,-1] → 2 tokens, loss_per_token=1.0 each
        // ex2: reward=2, log_probs=[-1,-1,-1] → 3 tokens, loss_per_token=2.0 each
        // total = 2*1.0 + 3*2.0 = 8.0; mean = 8.0/5 = 1.6
        let examples = vec![
            make_example(1.0, vec![-1.0, -1.0], None),
            make_example(2.0, vec![-1.0, -1.0, -1.0], None),
        ];
        let out = compute_reinforce_loss(&examples, &config).expect("ok");
        assert!((out.policy_loss - 1.6).abs() < 1e-5, "policy_loss={}", out.policy_loss);
        assert_eq!(out.num_tokens, 5);
        assert_eq!(out.num_examples, 2);
    }

    #[test]
    fn test_reward_scaling_invariance() {
        // Scaling rewards by constant c scales policy_loss by c (when no normalization)
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let scale = 3.0_f32;
        let ex1 = make_example(1.0, vec![-1.0], None);
        let ex2 = make_example(scale, vec![-1.0], None);
        let r1 = compute_reinforce_loss(&[ex1], &config).expect("ok1");
        let r2 = compute_reinforce_loss(&[ex2], &config).expect("ok2");
        assert!((r2.policy_loss / r1.policy_loss - scale).abs() < 1e-5, "ratio={}", r2.policy_loss / r1.policy_loss);
    }

    #[test]
    fn test_non_finite_reward_returns_error() {
        let config = ReinforceConfig::default();
        let ex = make_example(f32::NAN, vec![-1.0], None);
        let err = compute_reinforce_loss(&[ex], &config).unwrap_err();
        assert!(matches!(err, ReinforceError::NonFiniteValue { .. }));
    }

    #[test]
    fn test_non_finite_log_prob_returns_error() {
        let config = ReinforceConfig::default();
        let ex = make_example(1.0, vec![f32::INFINITY], None);
        let err = compute_reinforce_loss(&[ex], &config).unwrap_err();
        assert!(matches!(err, ReinforceError::NonFiniteValue { .. }));
    }

    #[test]
    fn test_baseline_running_mean_after_multiple_updates() {
        // EMA update: first call sets to batch_mean; subsequent calls decay
        let mut bl = ReinforceBaseline::new(BaselineMethod::RunningMean);
        bl.update(&[10.0_f32]);
        assert!((bl.running_mean - 10.0).abs() < 1e-5, "first update should set mean to batch mean");
        // Second update: ema = 0.99*10 + 0.01*0 = 9.9
        bl.update(&[0.0_f32]);
        assert!((bl.running_mean - 9.9).abs() < 1e-5, "second update={}", bl.running_mean);
    }

    #[test]
    fn test_reinforce_trainer_step_counter() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let mut trainer = ReinforceTrainer::new(config);
        assert_eq!(trainer.step, 0);
        let examples = vec![make_example(1.0, vec![-1.0], None)];
        trainer.compute_loss(examples.clone()).expect("step 1");
        trainer.compute_loss(examples.clone()).expect("step 2");
        assert_eq!(trainer.step, 2, "step counter should be 2 after 2 compute_loss calls");
    }

    #[test]
    fn test_reinforce_trainer_mean_recent_reward_window() {
        let config = ReinforceConfig {
            baseline_method: BaselineMethod::None,
            whiten_rewards: false,
            entropy_coeff: 0.0,
            reward_normalizer: RewardNormalization::None,
            ..Default::default()
        };
        let mut trainer = ReinforceTrainer::new(config);
        for r in [10.0_f32, 20.0, 30.0, 40.0, 50.0] {
            let examples = vec![make_example(r, vec![-1.0], None)];
            trainer.compute_loss(examples).expect("step");
        }
        // Last 3 rewards: 30, 40, 50 → mean = 40
        let mean = trainer.mean_recent_reward(3);
        assert!((mean - 40.0).abs() < 1e-4, "mean_recent={}", mean);
        // Window larger than history → mean of all: (10+20+30+40+50)/5=30
        let mean_all = trainer.mean_recent_reward(100);
        assert!((mean_all - 30.0).abs() < 1e-4, "mean_all={}", mean_all);
    }

    #[test]
    fn test_entropy_helper_grows_with_more_uniform_distribution() {
        // More uniform (smaller |log_prob|) → lower entropy contribution per token
        // but sum over more tokens → total higher for uniform
        // entropy(lp) = -lp; more negative lp → more entropy
        let high_entropy_lp = -3.0_f32;  // p = exp(-3) ≈ 0.05, spread out
        let low_entropy_lp = -0.01_f32;  // p = exp(-0.01) ≈ 0.99, concentrated
        let e_high = compute_entropy(high_entropy_lp);
        let e_low = compute_entropy(low_entropy_lp);
        assert!(e_high > e_low, "higher |log_prob| → more entropy: {} vs {}", e_high, e_low);
    }
}
