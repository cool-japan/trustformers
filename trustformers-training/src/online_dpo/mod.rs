//! Online DPO: Online Direct Preference Optimization
//!
//! This module implements Online DPO, which generates preference pairs on-the-fly
//! during training using a reward model, instead of relying on a static preference
//! dataset.
//!
//! # Key Concepts
//!
//! - At each training step, K responses are sampled for each prompt.
//! - A reward model scores each response.
//! - The best and worst responses form a preference pair.
//! - Pairs with a reward gap below `hard_pair_threshold` are discarded.
//! - A running mean baseline is optionally subtracted from rewards.
//!
//! # References
//!
//! - "Online DPO: Online Direct Preference Optimization with Fast-Slow Chaser"
//! - "RLHF Workflow: From Reward Modeling to Online RLHF"

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during Online DPO computations.
#[derive(Debug, Clone, PartialEq)]
pub enum OnlineDpoError {
    /// The batch of preference pairs was empty.
    EmptyBatch,
    /// Mismatched slice lengths were provided.
    LengthMismatch {
        /// Name describing what was expected.
        field: String,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// A numerical error occurred (NaN, Inf, etc.).
    NumericalError(String),
    /// The configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for OnlineDpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnlineDpoError::EmptyBatch => {
                write!(
                    f,
                    "Online DPO batch is empty; at least one preference pair is required"
                )
            }
            OnlineDpoError::LengthMismatch {
                field,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Online DPO length mismatch in {field}: expected {expected}, got {got}"
                )
            }
            OnlineDpoError::NumericalError(msg) => {
                write!(f, "Numerical error in Online DPO computation: {msg}")
            }
            OnlineDpoError::InvalidConfig(msg) => {
                write!(f, "Invalid Online DPO configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for OnlineDpoError {}

// ─────────────────────────────────────────────────────────────────────────────
// Loss type
// ─────────────────────────────────────────────────────────────────────────────

/// Variant of the Online DPO loss function.
#[derive(Debug, Clone, PartialEq)]
pub enum OnlineDpoLossType {
    /// Standard DPO sigmoid loss: `-log σ(β * margin)`.
    Sigmoid,
    /// Hinge loss variant: `max(0, 1 - β * margin)`.
    Hinge,
    /// Robust/conservative DPO with label smoothing (label_smoothing = 0.1):
    /// `-log σ(β * margin) * (1 - 0.1) + 0.1 * log(2)`.
    Robust,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Online DPO training.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineDpoConfig {
    /// KL penalty coefficient β (default: 0.1).
    pub beta: f32,
    /// Number of responses to generate per prompt (default: 2).
    pub num_responses_per_prompt: usize,
    /// Constant baseline to subtract from rewards (default: 0.0).
    /// Used only when `use_running_mean_baseline` is false.
    pub reward_baseline: f32,
    /// Maximum response token length (default: 512).
    pub max_response_length: usize,
    /// Sampling temperature for response generation (default: 0.7).
    pub temperature: f32,
    /// Whether to subtract the EMA running mean from rewards (default: true).
    pub use_running_mean_baseline: bool,
    /// EMA decay factor for the running mean (default: 0.99).
    pub ema_decay: f32,
    /// Minimum reward gap required to form a preference pair (default: 0.1).
    pub hard_pair_threshold: f32,
    /// Maximum number of preference pairs to keep per batch (default: 32).
    pub max_pairs_per_batch: usize,
    /// Which loss variant to use (default: [`OnlineDpoLossType::Sigmoid`]).
    pub loss_type: OnlineDpoLossType,
}

impl Default for OnlineDpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            num_responses_per_prompt: 2,
            reward_baseline: 0.0,
            max_response_length: 512,
            temperature: 0.7,
            use_running_mean_baseline: true,
            ema_decay: 0.99,
            hard_pair_threshold: 0.1,
            max_pairs_per_batch: 32,
            loss_type: OnlineDpoLossType::Sigmoid,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single online preference pair generated from reward-ranked responses.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineResponsePair {
    /// Tokenized prompt.
    pub prompt: Vec<u32>,
    /// Tokenized chosen (higher-reward) response.
    pub chosen: Vec<u32>,
    /// Tokenized rejected (lower-reward) response.
    pub rejected: Vec<u32>,
    /// Raw reward for the chosen response.
    pub chosen_reward: f32,
    /// Raw reward for the rejected response.
    pub rejected_reward: f32,
    /// `chosen_reward - rejected_reward`, always ≥ 0.
    pub reward_gap: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable `log σ(x) = -log(1 + exp(-x))`.
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0_f32 + (-x).exp()).ln()
    } else {
        x - (1.0_f32 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineDpoSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Selects and filters preference pairs from reward-ranked generated responses.
#[derive(Debug, Clone)]
pub struct OnlineDpoSelector {
    /// EMA running mean of rewards, used as a baseline when configured.
    pub running_mean_reward: f32,
    /// Configuration reference.
    pub config: OnlineDpoConfig,
}

impl OnlineDpoSelector {
    /// Create a new selector with the given configuration.
    pub fn new(config: OnlineDpoConfig) -> Self {
        Self {
            running_mean_reward: 0.0,
            config,
        }
    }

    /// Create preference pairs from a batch of prompts, responses, and rewards.
    ///
    /// For each prompt `i`, the response with the highest reward becomes `chosen`
    /// and the response with the lowest reward becomes `rejected`. A pair is only
    /// created if `reward_gap >= config.hard_pair_threshold`.
    ///
    /// The resulting pairs are sorted by `reward_gap` (descending) and at most
    /// `config.max_pairs_per_batch` pairs are returned.
    pub fn create_pairs(
        prompts: &[Vec<u32>],
        responses: &[Vec<Vec<u32>>],
        rewards: &[Vec<f32>],
        config: &OnlineDpoConfig,
    ) -> Vec<OnlineResponsePair> {
        debug_assert_eq!(
            prompts.len(),
            responses.len(),
            "prompts and responses length must match"
        );
        debug_assert_eq!(
            prompts.len(),
            rewards.len(),
            "prompts and rewards length must match"
        );

        let mut pairs: Vec<OnlineResponsePair> = Vec::new();

        for ((prompt, resp_list), reward_list) in
            prompts.iter().zip(responses.iter()).zip(rewards.iter())
        {
            if resp_list.is_empty() || reward_list.is_empty() {
                continue;
            }
            if resp_list.len() != reward_list.len() {
                continue;
            }

            // Find best (max) and worst (min) by reward
            let mut best_idx = 0usize;
            let mut worst_idx = 0usize;
            let mut best_reward = reward_list[0];
            let mut worst_reward = reward_list[0];

            for (idx, &r) in reward_list.iter().enumerate() {
                if r > best_reward {
                    best_reward = r;
                    best_idx = idx;
                }
                if r < worst_reward {
                    worst_reward = r;
                    worst_idx = idx;
                }
            }

            let gap = best_reward - worst_reward;
            if gap < config.hard_pair_threshold {
                continue;
            }

            pairs.push(OnlineResponsePair {
                prompt: prompt.clone(),
                chosen: resp_list[best_idx].clone(),
                rejected: resp_list[worst_idx].clone(),
                chosen_reward: best_reward,
                rejected_reward: worst_reward,
                reward_gap: gap,
            });
        }

        // Sort descending by reward_gap
        pairs.sort_by(|a, b| {
            b.reward_gap
                .partial_cmp(&a.reward_gap)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max_pairs_per_batch
        pairs.truncate(config.max_pairs_per_batch);
        pairs
    }

    /// Update the EMA running mean with a slice of new reward values.
    ///
    /// Uses the update rule:
    /// `running_mean = ema_decay * running_mean + (1 - ema_decay) * mean(rewards)`
    pub fn update_running_mean(&mut self, rewards: &[f32]) {
        if rewards.is_empty() {
            return;
        }
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let decay = self.config.ema_decay;
        self.running_mean_reward = decay * self.running_mean_reward + (1.0 - decay) * mean;
    }

    /// Apply the configured baseline to a single reward value.
    ///
    /// If `use_running_mean_baseline` is true, subtracts `running_mean_reward`;
    /// otherwise subtracts `config.reward_baseline`.
    pub fn apply_baseline(&self, reward: f32) -> f32 {
        if self.config.use_running_mean_baseline {
            reward - self.running_mean_reward
        } else {
            reward - self.config.reward_baseline
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineDpoBatchOutput
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics from one Online DPO batch.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineDpoBatchOutput {
    /// Mean loss over all preference pairs in the batch.
    pub total_loss: f32,
    /// Mean chosen reward (after baseline subtraction).
    pub mean_chosen_reward: f32,
    /// Mean rejected reward (after baseline subtraction).
    pub mean_rejected_reward: f32,
    /// Mean reward gap across pairs.
    pub mean_reward_gap: f32,
    /// Fraction of pairs where the DPO margin (log-ratio diff) > 0.
    pub accuracy: f32,
    /// Number of preference pairs processed.
    pub num_pairs: usize,
    /// The β used for this batch (copied from config).
    pub effective_beta: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineDpoLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless loss computation helpers for Online DPO.
pub struct OnlineDpoLoss;

impl OnlineDpoLoss {
    /// Compute the scalar loss for a single preference pair.
    ///
    /// # Arguments
    ///
    /// * `chosen_log_probs` — policy log-prob for the chosen response.
    /// * `rejected_log_probs` — policy log-prob for the rejected response.
    /// * `ref_chosen_log_probs` — reference log-prob for the chosen response.
    /// * `ref_rejected_log_probs` — reference log-prob for the rejected response.
    /// * `beta` — KL penalty weight.
    /// * `loss_type` — which loss variant to apply.
    ///
    /// The margin is:
    /// `(chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp)`
    pub fn compute_loss(
        chosen_log_probs: f32,
        rejected_log_probs: f32,
        ref_chosen_log_probs: f32,
        ref_rejected_log_probs: f32,
        beta: f32,
        loss_type: OnlineDpoLossType,
    ) -> f32 {
        let margin = (chosen_log_probs - ref_chosen_log_probs)
            - (rejected_log_probs - ref_rejected_log_probs);

        match loss_type {
            OnlineDpoLossType::Sigmoid => -log_sigmoid(beta * margin),
            OnlineDpoLossType::Hinge => (1.0 - beta * margin).max(0.0),
            OnlineDpoLossType::Robust => {
                // label_smoothing = 0.1
                let ls = 0.1_f32;
                -log_sigmoid(beta * margin) * (1.0 - ls) + ls * 2.0_f32.ln()
            }
        }
    }

    /// Compute the loss for a single [`OnlineResponsePair`] given policy and
    /// reference log-probabilities.
    pub fn compute_pair_loss(
        pair: &OnlineResponsePair,
        chosen_lp: f32,
        rejected_lp: f32,
        ref_chosen_lp: f32,
        ref_rejected_lp: f32,
        config: &OnlineDpoConfig,
    ) -> f32 {
        let _ = pair; // pair metadata not needed for the scalar loss
        Self::compute_loss(
            chosen_lp,
            rejected_lp,
            ref_chosen_lp,
            ref_rejected_lp,
            config.beta,
            config.loss_type.clone(),
        )
    }

    /// Compute the aggregated loss and statistics for a batch of preference pairs.
    ///
    /// # Errors
    ///
    /// Returns [`OnlineDpoError::EmptyBatch`] if `pairs` is empty.
    /// Returns [`OnlineDpoError::LengthMismatch`] if any log-prob slice length
    /// does not equal `pairs.len()`.
    /// Returns [`OnlineDpoError::NumericalError`] if the mean loss is non-finite.
    pub fn compute_batch_loss(
        pairs: &[OnlineResponsePair],
        chosen_lps: &[f32],
        rejected_lps: &[f32],
        ref_chosen_lps: &[f32],
        ref_rejected_lps: &[f32],
        config: &OnlineDpoConfig,
    ) -> Result<OnlineDpoBatchOutput, OnlineDpoError> {
        let n = pairs.len();
        if n == 0 {
            return Err(OnlineDpoError::EmptyBatch);
        }

        // Validate slice lengths
        let check = |name: &str, len: usize| -> Result<(), OnlineDpoError> {
            if len != n {
                Err(OnlineDpoError::LengthMismatch {
                    field: name.to_string(),
                    expected: n,
                    got: len,
                })
            } else {
                Ok(())
            }
        };
        check("chosen_lps", chosen_lps.len())?;
        check("rejected_lps", rejected_lps.len())?;
        check("ref_chosen_lps", ref_chosen_lps.len())?;
        check("ref_rejected_lps", ref_rejected_lps.len())?;

        let mut loss_sum = 0.0_f32;
        let mut chosen_reward_sum = 0.0_f32;
        let mut rejected_reward_sum = 0.0_f32;
        let mut reward_gap_sum = 0.0_f32;
        let mut margin_positive_count = 0usize;

        for i in 0..n {
            let margin = (chosen_lps[i] - ref_chosen_lps[i])
                - (rejected_lps[i] - ref_rejected_lps[i]);

            let loss_i = Self::compute_loss(
                chosen_lps[i],
                rejected_lps[i],
                ref_chosen_lps[i],
                ref_rejected_lps[i],
                config.beta,
                config.loss_type.clone(),
            );

            if margin > 0.0 {
                margin_positive_count += 1;
            }

            loss_sum += loss_i;
            chosen_reward_sum += pairs[i].chosen_reward;
            rejected_reward_sum += pairs[i].rejected_reward;
            reward_gap_sum += pairs[i].reward_gap;
        }

        let inv_n = 1.0 / n as f32;
        let total_loss = loss_sum * inv_n;

        if !total_loss.is_finite() {
            return Err(OnlineDpoError::NumericalError(format!(
                "Online DPO mean loss={total_loss} is not finite"
            )));
        }

        Ok(OnlineDpoBatchOutput {
            total_loss,
            mean_chosen_reward: chosen_reward_sum * inv_n,
            mean_rejected_reward: rejected_reward_sum * inv_n,
            mean_reward_gap: reward_gap_sum * inv_n,
            accuracy: margin_positive_count as f32 * inv_n,
            num_pairs: n,
            effective_beta: config.beta,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineDpoTrainer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful Online DPO trainer that accumulates batch statistics.
#[derive(Debug)]
pub struct OnlineDpoTrainer {
    /// Configuration for Online DPO.
    pub config: OnlineDpoConfig,
    /// The pair selector (maintains EMA running mean).
    pub selector: OnlineDpoSelector,
    /// Number of `process_batch` calls completed.
    pub step: usize,
    /// History of batch outputs for diagnostics.
    pub history: Vec<OnlineDpoBatchOutput>,
}

impl OnlineDpoTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: OnlineDpoConfig) -> Self {
        let selector = OnlineDpoSelector::new(config.clone());
        Self {
            config,
            selector,
            step: 0,
            history: Vec::new(),
        }
    }

    /// Process a single batch of online preference pairs and accumulate history.
    ///
    /// # Arguments
    ///
    /// * `pairs` — online preference pairs (typically from `OnlineDpoSelector::create_pairs`).
    /// * `chosen_lps` — policy log-probs for chosen responses, one per pair.
    /// * `rejected_lps` — policy log-probs for rejected responses, one per pair.
    /// * `ref_chosen_lps` — reference log-probs for chosen responses.
    /// * `ref_rejected_lps` — reference log-probs for rejected responses.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`OnlineDpoLoss::compute_batch_loss`].
    pub fn process_batch(
        &mut self,
        pairs: Vec<OnlineResponsePair>,
        chosen_lps: Vec<f32>,
        rejected_lps: Vec<f32>,
        ref_chosen_lps: Vec<f32>,
        ref_rejected_lps: Vec<f32>,
    ) -> Result<OnlineDpoBatchOutput, OnlineDpoError> {
        let output = OnlineDpoLoss::compute_batch_loss(
            &pairs,
            &chosen_lps,
            &rejected_lps,
            &ref_chosen_lps,
            &ref_rejected_lps,
            &self.config,
        )?;

        // Update EMA running mean from all rewards in this batch
        let all_rewards: Vec<f32> = pairs
            .iter()
            .flat_map(|p| [p.chosen_reward, p.rejected_reward])
            .collect();
        self.selector.update_running_mean(&all_rewards);

        self.step += 1;
        self.history.push(output.clone());
        Ok(output)
    }

    /// Return the full batch history.
    pub fn history(&self) -> &[OnlineDpoBatchOutput] {
        &self.history
    }

    /// Mean loss over all recorded batches. Returns 0.0 for empty history.
    pub fn mean_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|o| o.total_loss).sum();
        sum / self.history.len() as f32
    }

    /// Mean accuracy over all recorded batches. Returns 0.0 for empty history.
    pub fn mean_accuracy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|o| o.accuracy).sum();
        sum / self.history.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_pair(
        chosen_reward: f32,
        rejected_reward: f32,
    ) -> OnlineResponsePair {
        OnlineResponsePair {
            prompt: vec![1, 2, 3],
            chosen: vec![10, 11],
            rejected: vec![20, 21],
            chosen_reward,
            rejected_reward,
            reward_gap: chosen_reward - rejected_reward,
        }
    }

    // ── Config default ───────────────────────────────────────────────────

    #[test]
    fn test_online_dpo_config_default() {
        let cfg = OnlineDpoConfig::default();
        assert!((cfg.beta - 0.1).abs() < 1e-8);
        assert_eq!(cfg.num_responses_per_prompt, 2);
        assert!((cfg.reward_baseline).abs() < 1e-8);
        assert_eq!(cfg.max_response_length, 512);
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
        assert!(cfg.use_running_mean_baseline);
        assert!((cfg.ema_decay - 0.99).abs() < 1e-8);
        assert!((cfg.hard_pair_threshold - 0.1).abs() < 1e-8);
        assert_eq!(cfg.max_pairs_per_batch, 32);
        assert_eq!(cfg.loss_type, OnlineDpoLossType::Sigmoid);
    }

    // ── Loss computation ─────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_loss_perfect_margin() {
        // Very large positive margin → loss near 0
        let loss = OnlineDpoLoss::compute_loss(
            0.0,  // chosen_lp
            0.0,  // rejected_lp
            -10.0, // ref_chosen_lp  → chosen_ratio = +10
            10.0,  // ref_rejected_lp → rejected_ratio = -10
            0.1,
            OnlineDpoLossType::Sigmoid,
        );
        // margin = 10 - (-10) = 20, scaled = 2.0, -log_sigmoid(2.0) ≈ 0.127
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    #[test]
    fn test_hinge_loss_no_violation() {
        // margin = 20, beta*margin = 2 > 1 → max(0, 1-2) = 0
        let loss = OnlineDpoLoss::compute_loss(
            0.0,
            0.0,
            -10.0,
            10.0,
            0.1,
            OnlineDpoLossType::Hinge,
        );
        assert!((loss).abs() < 1e-5, "expected 0 hinge loss, got {loss}");
    }

    #[test]
    fn test_hinge_loss_with_violation() {
        // margin = 0, beta*margin = 0 → max(0, 1-0) = 1
        let loss = OnlineDpoLoss::compute_loss(
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            OnlineDpoLossType::Hinge,
        );
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0 hinge loss, got {loss}");
    }

    #[test]
    fn test_robust_loss_higher_than_sigmoid() {
        // Robust adds a positive label_smoothing * log(2) term, so it should
        // be >= the sigmoid loss when margin is large and positive
        let chosen_lp = 0.0;
        let rejected_lp = 0.0;
        let ref_chosen = -5.0;
        let ref_rejected = 5.0;
        let beta = 0.1;

        let sigmoid_loss = OnlineDpoLoss::compute_loss(
            chosen_lp,
            rejected_lp,
            ref_chosen,
            ref_rejected,
            beta,
            OnlineDpoLossType::Sigmoid,
        );
        let robust_loss = OnlineDpoLoss::compute_loss(
            chosen_lp,
            rejected_lp,
            ref_chosen,
            ref_rejected,
            beta,
            OnlineDpoLossType::Robust,
        );
        // Robust = sigmoid_part * 0.9 + 0.1 * ln(2)
        // When sigmoid_part is near 0 (large margin), robust ≈ 0.1 * ln(2) > sigmoid ≈ 0
        assert!(
            robust_loss >= sigmoid_loss - 1e-5,
            "robust={robust_loss} should >= sigmoid={sigmoid_loss}"
        );
    }

    // ── Batch loss ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_loss_empty() {
        let cfg = OnlineDpoConfig::default();
        let result = OnlineDpoLoss::compute_batch_loss(&[], &[], &[], &[], &[], &cfg);
        assert!(matches!(result, Err(OnlineDpoError::EmptyBatch)));
    }

    #[test]
    fn test_batch_loss_length_mismatch() {
        let cfg = OnlineDpoConfig::default();
        let pairs = vec![make_pair(1.0, 0.5)];
        let result =
            OnlineDpoLoss::compute_batch_loss(&pairs, &[-1.0, -2.0], &[-1.0], &[-1.0], &[-1.0], &cfg);
        assert!(matches!(result, Err(OnlineDpoError::LengthMismatch { .. })));
    }

    #[test]
    fn test_batch_loss_basic() {
        let cfg = OnlineDpoConfig::default();
        let pairs = vec![make_pair(1.5, 0.5), make_pair(2.0, 1.0)];
        let chosen_lps = vec![-0.5_f32, -0.3];
        let rejected_lps = vec![-1.5_f32, -1.2];
        let ref_chosen = vec![-1.0_f32, -0.8];
        let ref_rejected = vec![-0.8_f32, -0.6];

        let out = OnlineDpoLoss::compute_batch_loss(
            &pairs,
            &chosen_lps,
            &rejected_lps,
            &ref_chosen,
            &ref_rejected,
            &cfg,
        )
        .expect("batch loss failed");

        assert!(out.total_loss.is_finite());
        assert!(out.total_loss >= 0.0);
        assert_eq!(out.num_pairs, 2);
        assert!((out.effective_beta - 0.1).abs() < 1e-8);
    }

    // ── Selector ─────────────────────────────────────────────────────────

    #[test]
    fn test_create_pairs_basic() {
        let cfg = OnlineDpoConfig::default();
        let prompts = vec![vec![1u32, 2, 3]];
        let responses = vec![vec![vec![10u32, 11], vec![20u32, 21]]];
        let rewards = vec![vec![0.8_f32, 0.3]]; // gap = 0.5 > threshold 0.1

        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].chosen_reward - 0.8).abs() < 1e-6);
        assert!((pairs[0].rejected_reward - 0.3).abs() < 1e-6);
        assert!((pairs[0].reward_gap - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_create_pairs_gap_too_small() {
        let mut cfg = OnlineDpoConfig::default();
        cfg.hard_pair_threshold = 0.5;
        let prompts = vec![vec![1u32, 2]];
        let responses = vec![vec![vec![10u32], vec![20u32]]];
        let rewards = vec![vec![0.8_f32, 0.7]]; // gap = 0.1 < threshold 0.5

        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_create_pairs_max_pairs() {
        let mut cfg = OnlineDpoConfig::default();
        cfg.max_pairs_per_batch = 2;

        let prompts: Vec<Vec<u32>> = (0..5).map(|i| vec![i as u32]).collect();
        let responses: Vec<Vec<Vec<u32>>> = (0..5)
            .map(|i| vec![vec![i as u32 * 10], vec![i as u32 * 10 + 1]])
            .collect();
        let rewards: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![(i as f32 + 1.0) * 0.3, (i as f32) * 0.1])
            .collect();

        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert!(pairs.len() <= 2);
    }

    // ── Running mean ─────────────────────────────────────────────────────

    #[test]
    fn test_update_running_mean() {
        let cfg = OnlineDpoConfig::default(); // ema_decay = 0.99
        let mut sel = OnlineDpoSelector::new(cfg);
        assert!((sel.running_mean_reward).abs() < 1e-8);

        sel.update_running_mean(&[1.0, 1.0]); // mean = 1.0
        // new_running = 0.99 * 0 + 0.01 * 1.0 = 0.01
        assert!((sel.running_mean_reward - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_apply_baseline_running_mean() {
        let cfg = OnlineDpoConfig::default(); // use_running_mean_baseline = true
        let mut sel = OnlineDpoSelector::new(cfg);
        sel.running_mean_reward = 0.5;
        assert!((sel.apply_baseline(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_baseline_fixed() {
        let mut cfg = OnlineDpoConfig::default();
        cfg.use_running_mean_baseline = false;
        cfg.reward_baseline = 0.3;
        let sel = OnlineDpoSelector::new(cfg);
        assert!((sel.apply_baseline(1.0) - 0.7).abs() < 1e-6);
    }

    // ── Trainer ──────────────────────────────────────────────────────────

    #[test]
    fn test_trainer_process_batch() {
        let cfg = OnlineDpoConfig::default();
        let mut trainer = OnlineDpoTrainer::new(cfg);

        let pairs = vec![make_pair(1.5, 0.5)];
        let result = trainer.process_batch(
            pairs,
            vec![-0.5],
            vec![-1.5],
            vec![-1.0],
            vec![-0.8],
        );
        assert!(result.is_ok());
        assert_eq!(trainer.step, 1);
        assert_eq!(trainer.history().len(), 1);
    }

    #[test]
    fn test_trainer_mean_loss_empty() {
        let trainer = OnlineDpoTrainer::new(OnlineDpoConfig::default());
        assert!((trainer.mean_loss()).abs() < 1e-8);
    }

    #[test]
    fn test_error_display() {
        let e1 = OnlineDpoError::EmptyBatch;
        assert!(e1.to_string().contains("empty"));

        let e2 = OnlineDpoError::LengthMismatch {
            field: "chosen_lps".to_string(),
            expected: 3,
            got: 2,
        };
        assert!(e2.to_string().contains("chosen_lps"));

        let e3 = OnlineDpoError::NumericalError("NaN detected".to_string());
        assert!(e3.to_string().contains("NaN"));

        let e4 = OnlineDpoError::InvalidConfig("beta must be > 0".to_string());
        assert!(e4.to_string().contains("beta"));
    }

    // ── Additional Online DPO tests ───────────────────────────────────────

    #[test]
    fn test_reference_model_update_ema_schedule() {
        // EMA running mean converges toward the true mean over multiple updates
        let cfg = OnlineDpoConfig { ema_decay: 0.9, ..Default::default() };
        let mut sel = OnlineDpoSelector::new(cfg);
        // All rewards = 5.0; after enough updates, running mean → 5.0
        for _ in 0..200 {
            sel.update_running_mean(&[5.0]);
        }
        assert!((sel.running_mean_reward - 5.0).abs() < 0.01, "running_mean={}", sel.running_mean_reward);
    }

    #[test]
    fn test_create_pairs_selects_best_and_worst() {
        // The chosen response is the one with highest reward, rejected with lowest
        let cfg = OnlineDpoConfig::default();
        let prompts = vec![vec![1u32]];
        let responses = vec![vec![
            vec![10u32], // reward 0.2
            vec![20u32], // reward 0.9 ← best
            vec![30u32], // reward 0.1 ← worst
        ]];
        let rewards = vec![vec![0.2_f32, 0.9, 0.1]];
        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].chosen_reward - 0.9).abs() < 1e-6);
        assert!((pairs[0].rejected_reward - 0.1).abs() < 1e-6);
        // chosen response should be [20], rejected [30]
        assert_eq!(pairs[0].chosen, vec![20u32]);
        assert_eq!(pairs[0].rejected, vec![30u32]);
    }

    #[test]
    fn test_rejection_sampling_gap_threshold_filtering() {
        // Only pairs with reward_gap >= hard_pair_threshold should be kept
        let mut cfg = OnlineDpoConfig::default();
        cfg.hard_pair_threshold = 0.5;
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| vec![i as u32]).collect();
        let responses: Vec<Vec<Vec<u32>>> = (0..4)
            .map(|i| vec![vec![i as u32 * 10], vec![i as u32 * 10 + 1]])
            .collect();
        let rewards: Vec<Vec<f32>> = vec![
            vec![1.0, 0.4],  // gap=0.6 >= 0.5 ✓
            vec![0.8, 0.4],  // gap=0.4 < 0.5  ✗
            vec![1.0, 0.0],  // gap=1.0 >= 0.5 ✓
            vec![0.6, 0.2],  // gap=0.4 < 0.5  ✗
        ];
        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert_eq!(pairs.len(), 2, "only 2 pairs should pass the threshold");
        for p in &pairs {
            assert!(p.reward_gap >= 0.5, "gap {} < threshold", p.reward_gap);
        }
    }

    #[test]
    fn test_pairs_sorted_by_reward_gap_descending() {
        // Pairs should be sorted descending by reward_gap
        let cfg = OnlineDpoConfig { max_pairs_per_batch: 10, ..Default::default() };
        let prompts: Vec<Vec<u32>> = (0..3).map(|i| vec![i as u32]).collect();
        let responses: Vec<Vec<Vec<u32>>> = (0..3)
            .map(|i| vec![vec![i as u32 * 2], vec![i as u32 * 2 + 1]])
            .collect();
        // gaps: 0.9, 0.3, 0.6
        let rewards: Vec<Vec<f32>> = vec![
            vec![1.0, 0.1],
            vec![0.8, 0.5],
            vec![0.9, 0.3],
        ];
        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        for i in 0..pairs.len().saturating_sub(1) {
            assert!(
                pairs[i].reward_gap >= pairs[i + 1].reward_gap,
                "pairs not sorted: {}[{}] < {}[{}]",
                pairs[i].reward_gap, i, pairs[i + 1].reward_gap, i + 1
            );
        }
    }

    #[test]
    fn test_moving_average_of_reference_model_with_single_value() {
        // EMA update: new = decay * old + (1-decay) * batch_mean
        // Starting from 0: new = 0.99 * 0.0 + 0.01 * 7.0 = 0.07
        let cfg = OnlineDpoConfig { ema_decay: 0.99, ..Default::default() };
        let mut sel = OnlineDpoSelector::new(cfg);
        sel.update_running_mean(&[7.0]);
        let expected = 0.01 * 7.0; // (1-0.99) * 7.0 = 0.07
        assert!((sel.running_mean_reward - expected).abs() < 1e-5, "first update={} expected={}", sel.running_mean_reward, expected);
    }

    #[test]
    fn test_online_accuracy_high_when_model_converging() {
        // High accuracy means chosen logit >> rejected logit, loss is small
        let cfg = OnlineDpoConfig { beta: 0.1, ..Default::default() };
        let pairs = vec![make_pair(2.0, 0.5)];
        // policy: chosen much more likely than rejected, reference equal
        let chosen_lps = vec![-0.1_f32];
        let rejected_lps = vec![-5.0_f32];
        let ref_chosen_lps = vec![-1.0_f32];
        let ref_rejected_lps = vec![-1.0_f32];
        let out = OnlineDpoLoss::compute_batch_loss(
            &pairs, &chosen_lps, &rejected_lps, &ref_chosen_lps, &ref_rejected_lps, &cfg
        ).expect("ok");
        // margin = (-0.1 - (-1.0)) - (-5.0 - (-1.0)) = 0.9 - (-4.0) = 4.9 > 0
        assert_eq!(out.accuracy, 1.0, "accuracy should be 1.0 when chosen >> rejected");
        assert!(out.total_loss < (2.0_f32).ln(), "loss should be less than log(2) for positive margin");
    }

    #[test]
    fn test_reward_model_threshold_filtering_multiple_prompts() {
        // With a high threshold, very close reward responses are filtered out
        let mut cfg = OnlineDpoConfig::default();
        cfg.hard_pair_threshold = 1.0;
        let prompts = vec![vec![1u32], vec![2u32]];
        let responses = vec![
            vec![vec![1u32], vec![2u32]],
            vec![vec![3u32], vec![4u32]],
        ];
        let rewards = vec![
            vec![0.6_f32, 0.1],   // gap=0.5 < 1.0 → filtered
            vec![2.0_f32, 0.0],   // gap=2.0 >= 1.0 → kept
        ];
        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].reward_gap - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_loss_zero_margin_gives_log_two() {
        // margin=0 → -log_sigmoid(0) = log(2)
        let loss = OnlineDpoLoss::compute_loss(
            0.0, 0.0, 0.0, 0.0, 0.1, OnlineDpoLossType::Sigmoid
        );
        let expected = (2.0_f32).ln();
        assert!((loss - expected).abs() < 1e-5, "loss={} expected log(2)={}", loss, expected);
    }

    #[test]
    fn test_hinge_loss_exactly_at_boundary() {
        // When beta*margin = 1.0: hinge = max(0, 1-1) = 0
        // margin = 1/beta = 10.0 when beta=0.1
        let loss = OnlineDpoLoss::compute_loss(
            0.0,   // chosen_lp
            0.0,   // rejected_lp
            -5.0,  // ref_chosen_lp → chosen_ratio = 5.0
            5.0,   // ref_rejected_lp → rejected_ratio = -5.0
            0.1,   // beta
            OnlineDpoLossType::Hinge,
        );
        // margin = 5.0 - (-5.0) = 10.0; beta*margin = 0.1*10 = 1.0; max(0, 0) = 0
        assert!((loss).abs() < 1e-5, "hinge at boundary should be 0, got {}", loss);
    }

    #[test]
    fn test_robust_loss_formula_numerically() {
        // Robust: (1-0.1)*(-log_sigmoid(beta*margin)) + 0.1*ln(2)
        let beta = 0.1_f32;
        let chosen_lp = 0.0_f32;
        let rejected_lp = 0.0_f32;
        let ref_chosen = -1.0_f32;
        let ref_rejected = 1.0_f32;
        // margin = (0-(-1)) - (0-1) = 1 - (-1) = 2.0
        // beta*margin = 0.2
        // log_sigmoid(0.2) = -ln(1+exp(-0.2))
        let scaled_margin = beta * 2.0;
        let ls = 0.1_f32;
        let log_sig = -(1.0_f32 + (-scaled_margin).exp()).ln();
        let expected_robust = -(1.0 - ls) * log_sig + ls * (2.0_f32).ln();
        let actual_robust = OnlineDpoLoss::compute_loss(
            chosen_lp, rejected_lp, ref_chosen, ref_rejected, beta, OnlineDpoLossType::Robust
        );
        assert!((actual_robust - expected_robust).abs() < 1e-5, "robust={} expected={}", actual_robust, expected_robust);
    }

    #[test]
    fn test_batch_loss_all_correct_margin_accuracy_one() {
        // All pairs have positive margin → accuracy = 1.0
        let cfg = OnlineDpoConfig::default();
        let pairs = vec![
            make_pair(2.0, 0.5),
            make_pair(3.0, 1.0),
        ];
        // chosen is much more likely than rejected compared to reference
        let chosen_lps = vec![-0.1_f32, -0.2];
        let rejected_lps = vec![-5.0_f32, -4.0];
        let ref_chosen = vec![-1.0_f32, -1.0];
        let ref_rejected = vec![-1.0_f32, -1.0];
        let out = OnlineDpoLoss::compute_batch_loss(
            &pairs, &chosen_lps, &rejected_lps, &ref_chosen, &ref_rejected, &cfg
        ).expect("ok");
        assert_eq!(out.accuracy, 1.0, "accuracy={}", out.accuracy);
    }

    #[test]
    fn test_trainer_history_and_mean_loss_multiple_batches() {
        let cfg = OnlineDpoConfig::default();
        let mut trainer = OnlineDpoTrainer::new(cfg);
        let pairs = vec![make_pair(1.5, 0.5)];

        let r1 = trainer.process_batch(
            pairs.clone(), vec![-0.5], vec![-1.5], vec![-1.0], vec![-0.8]
        ).expect("batch 1");
        let r2 = trainer.process_batch(
            pairs, vec![-0.3], vec![-1.2], vec![-0.8], vec![-0.6]
        ).expect("batch 2");

        assert_eq!(trainer.history().len(), 2);
        assert_eq!(trainer.step, 2);
        let expected_mean = (r1.total_loss + r2.total_loss) / 2.0;
        assert!((trainer.mean_loss() - expected_mean).abs() < 1e-5, "mean_loss mismatch");
    }

    #[test]
    fn test_apply_baseline_zero_running_mean() {
        // With running_mean=0 and use_running_mean_baseline=true, baseline is identity shift by 0
        let cfg = OnlineDpoConfig::default();
        let sel = OnlineDpoSelector::new(cfg);
        assert!((sel.running_mean_reward).abs() < 1e-8);
        let reward = 2.5_f32;
        assert!((sel.apply_baseline(reward) - reward).abs() < 1e-6);
    }

    #[test]
    fn test_online_dpo_mean_accuracy_multiple_steps() {
        let cfg = OnlineDpoConfig::default();
        let mut trainer = OnlineDpoTrainer::new(cfg);
        assert!((trainer.mean_accuracy()).abs() < 1e-8, "empty history → 0");

        let pairs = vec![make_pair(1.0, 0.0)];
        // Step 1: all correct (accuracy=1.0)
        trainer.process_batch(
            pairs.clone(), vec![-0.1], vec![-5.0], vec![-1.0], vec![-1.0]
        ).expect("batch 1");
        // Step 2: all wrong (negative margin)
        trainer.process_batch(
            pairs.clone(), vec![-5.0], vec![-0.1], vec![-1.0], vec![-1.0]
        ).expect("batch 2");

        let mean_acc = trainer.mean_accuracy();
        assert!((mean_acc - 0.5).abs() < 1e-5, "mean_acc={}", mean_acc);
    }

    #[test]
    fn test_create_pairs_empty_responses_skipped() {
        let cfg = OnlineDpoConfig::default();
        let prompts = vec![vec![1u32], vec![2u32]];
        let responses = vec![
            vec![],          // empty → skipped
            vec![vec![10u32], vec![20u32]], // gap=0.5 ≥ 0.1 → kept
        ];
        let rewards = vec![
            vec![],
            vec![0.8_f32, 0.3],
        ];
        let pairs = OnlineDpoSelector::create_pairs(&prompts, &responses, &rewards, &cfg);
        assert_eq!(pairs.len(), 1, "empty response list should be skipped");
    }
}
