//! RAFT: Reward rAnked Fine-Tuning
//!
//! Reference: "RAFT: Reward rAnked Finetuning for Generative Foundation Model Alignment"
//! (Dong et al., 2023)
//!
//! RAFT generates K responses per prompt, scores them with a reward model, keeps
//! only the top-ranked ones, and uses those for supervised fine-tuning (SFT).
//! This iteratively steers the model toward higher-reward outputs without requiring
//! pairwise preference data or a separate value model.
//!
//! ## Algorithm
//!
//! For each iteration:
//! 1. For each prompt x_i, sample K responses: {y_i^1, ..., y_i^K}
//! 2. Score each response with a reward model r(x_i, y_i^j)
//! 3. Keep the top `ceil(K * top_k_fraction)` responses per prompt
//! 4. Fine-tune the policy on the selected (prompt, response) pairs via SFT

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during RAFT operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RaftError {
    /// The dataset is empty; at least one prompt is required.
    EmptyDataset,
    /// No responses passed the selection threshold.
    NoResponsesSelected,
    /// Not enough samples were provided relative to the configuration.
    InsufficientSamples { required: usize, got: usize },
    /// The configuration has an invalid field value.
    InvalidConfig(String),
    /// Reward normalization failed (e.g., zero standard deviation on a non-trivial input).
    RewardNormalizationFailed(String),
}

impl fmt::Display for RaftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RaftError::EmptyDataset => {
                write!(f, "RAFT error: dataset is empty; at least one prompt is required")
            }
            RaftError::NoResponsesSelected => {
                write!(
                    f,
                    "RAFT error: no responses were selected; \
                     try lowering reward_threshold or increasing top_k_fraction"
                )
            }
            RaftError::InsufficientSamples { required, got } => {
                write!(
                    f,
                    "RAFT error: insufficient samples — required {required}, got {got}"
                )
            }
            RaftError::InvalidConfig(msg) => {
                write!(f, "RAFT invalid config: {msg}")
            }
            RaftError::RewardNormalizationFailed(msg) => {
                write!(f, "RAFT reward normalization failed: {msg}")
            }
        }
    }
}

impl std::error::Error for RaftError {}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for RAFT (Reward rAnked Fine-Tuning).
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Number of candidate responses to generate per prompt (K). Default: 8
    pub num_samples_per_prompt: usize,
    /// Fraction of responses to keep after ranking (top-K selection). Default: 0.25
    pub top_k_fraction: f32,
    /// Optional minimum reward; responses below this threshold are discarded
    /// regardless of `top_k_fraction`. Default: None
    pub reward_threshold: Option<f32>,
    /// Sampling temperature used when generating responses. Default: 0.7
    pub temperature_sampling: f32,
    /// Total number of RAFT iterations. Default: 10
    pub num_iterations: usize,
    /// Mini-batch size for SFT updates. Default: 32
    pub batch_size: usize,
    /// Learning rate for SFT fine-tuning. Default: 1e-5
    pub learning_rate: f32,
    /// Whether to normalize rewards to mean=0, std=1 before selection. Default: true
    pub normalize_rewards: bool,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            num_samples_per_prompt: 8,
            top_k_fraction: 0.25,
            reward_threshold: None,
            temperature_sampling: 0.7,
            num_iterations: 10,
            batch_size: 32,
            learning_rate: 1e-5,
            normalize_rewards: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// A model-generated response paired with its reward-model score.
#[derive(Debug, Clone)]
pub struct ScoredResponse {
    /// Token ids of the generated response.
    pub response: Vec<u32>,
    /// Scalar reward assigned by the reward model.
    pub reward: f32,
    /// Index into the originating prompt dataset.
    pub prompt_idx: usize,
}

/// Dataset of prompts for RAFT training.
#[derive(Debug, Clone)]
pub struct RaftDataset {
    /// Prompts represented as token id sequences.
    pub prompts: Vec<Vec<u32>>,
}

impl RaftDataset {
    /// Number of prompts in the dataset.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Whether the dataset contains no prompts.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// A batch ready for supervised fine-tuning after RAFT selection.
///
/// Contains the selected (prompt, response) pairs together with reward statistics
/// for logging and monitoring.
#[derive(Debug, Clone)]
pub struct RaftBatch {
    /// Selected (prompt_tokens, response_tokens) pairs for SFT.
    pub selected_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    /// Total number of (prompt, response) candidates that were scored.
    pub total_generated: usize,
    /// Number of responses kept after selection.
    pub total_selected: usize,
    /// Mean reward over **all** generated candidates.
    pub mean_reward_all: f32,
    /// Mean reward over the **selected** candidates.
    pub mean_reward_selected: f32,
    /// The reward threshold that was actually applied during selection.
    /// For top-K-only selection, this is the reward of the last kept response.
    pub reward_threshold_used: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// RAFT selector
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless helper that implements the RAFT selection and batch-building logic.
pub struct RaftSelector {
    /// RAFT hyper-parameters.
    pub config: RaftConfig,
}

impl RaftSelector {
    /// Create a new selector with the given configuration.
    pub fn new(config: RaftConfig) -> Self {
        Self { config }
    }

    /// Select the top responses from a flat list of scored responses.
    ///
    /// Steps:
    /// 1. Sort responses by reward descending.
    /// 2. Keep the top `ceil(len * top_k_fraction)` responses.
    /// 3. If `config.reward_threshold` is set, additionally discard any response
    ///    whose reward falls below the threshold.
    ///
    /// Returns an empty vec if the input is empty.
    pub fn select_top_responses(
        responses: &[ScoredResponse],
        config: &RaftConfig,
    ) -> Vec<ScoredResponse> {
        if responses.is_empty() {
            return Vec::new();
        }

        // Sort descending by reward
        let mut sorted: Vec<ScoredResponse> = responses.to_vec();
        sorted.sort_by(|a, b| b.reward.partial_cmp(&a.reward).unwrap_or(std::cmp::Ordering::Equal));

        // Determine top-K count
        let keep_n = ceil_frac(sorted.len(), config.top_k_fraction);
        let keep_n = keep_n.max(1); // always keep at least 1

        let mut selected: Vec<ScoredResponse> = sorted.into_iter().take(keep_n).collect();

        // Apply reward threshold filter if configured
        if let Some(threshold) = config.reward_threshold {
            selected.retain(|r| r.reward >= threshold);
        }

        selected
    }

    /// Normalize a mutable slice of `ScoredResponse` to mean=0, std=1.
    ///
    /// If the slice has fewer than 2 elements or the standard deviation is
    /// effectively zero (< 1e-8), rewards are left unchanged to avoid division
    /// by zero.
    pub fn normalize_rewards(responses: &mut [ScoredResponse]) {
        if responses.len() < 2 {
            return;
        }

        let n = responses.len() as f32;
        let mean = responses.iter().map(|r| r.reward).sum::<f32>() / n;
        let variance =
            responses.iter().map(|r| (r.reward - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-8 {
            // All rewards identical — normalization would divide by zero; skip.
            return;
        }

        for r in responses.iter_mut() {
            r.reward = (r.reward - mean) / std_dev;
        }
    }

    /// Build a `RaftBatch` from per-prompt scored responses.
    ///
    /// For each prompt, the top responses are selected according to `config`.
    /// The resulting (prompt, response) pairs are collected into a `RaftBatch`.
    ///
    /// # Arguments
    /// * `prompts` — the original prompt token sequences.
    /// * `scored_responses` — one `Vec<ScoredResponse>` per prompt.
    /// * `config` — RAFT hyper-parameters.
    ///
    /// # Errors
    /// * `RaftError::EmptyDataset` if `prompts` is empty.
    /// * `RaftError::InsufficientSamples` if any prompt has zero responses.
    /// * `RaftError::NoResponsesSelected` if no responses survive selection across all prompts.
    pub fn build_sft_batch(
        prompts: &[Vec<u32>],
        scored_responses: &[Vec<ScoredResponse>],
        config: &RaftConfig,
    ) -> Result<RaftBatch, RaftError> {
        if prompts.is_empty() {
            return Err(RaftError::EmptyDataset);
        }

        let mut selected_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        let mut total_generated = 0_usize;
        let mut total_selected = 0_usize;
        let mut sum_reward_all = 0.0_f32;
        let mut sum_reward_selected = 0.0_f32;
        let mut min_selected_reward = f32::INFINITY;

        for (i, prompt) in prompts.iter().enumerate() {
            let responses = scored_responses
                .get(i)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            if responses.is_empty() {
                return Err(RaftError::InsufficientSamples {
                    required: config.num_samples_per_prompt,
                    got: 0,
                });
            }

            total_generated += responses.len();
            sum_reward_all += responses.iter().map(|r| r.reward).sum::<f32>();

            let selected = Self::select_top_responses(responses, config);
            for resp in &selected {
                sum_reward_selected += resp.reward;
                if resp.reward < min_selected_reward {
                    min_selected_reward = resp.reward;
                }
                selected_pairs.push((prompt.clone(), resp.response.clone()));
            }
            total_selected += selected.len();
        }

        if total_selected == 0 {
            return Err(RaftError::NoResponsesSelected);
        }

        let mean_reward_all = sum_reward_all / (total_generated as f32);
        let mean_reward_selected = sum_reward_selected / (total_selected as f32);
        let reward_threshold_used = if min_selected_reward == f32::INFINITY {
            0.0
        } else {
            min_selected_reward
        };

        Ok(RaftBatch {
            selected_pairs,
            total_generated,
            total_selected,
            mean_reward_all,
            mean_reward_selected,
            reward_threshold_used,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iteration statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single RAFT iteration.
#[derive(Debug, Clone)]
pub struct RaftIterationStats {
    /// Which RAFT iteration these stats belong to (0-indexed).
    pub iteration: usize,
    /// Number of unique prompts processed.
    pub num_prompts: usize,
    /// Total number of (prompt, response) candidates generated.
    pub total_generated: usize,
    /// Number of responses kept after selection.
    pub total_selected: usize,
    /// Fraction of generated responses that were selected: `total_selected / total_generated`.
    pub selection_rate: f32,
    /// Mean reward over all generated candidates.
    pub mean_reward_all: f32,
    /// Mean reward over the selected candidates.
    pub mean_reward_selected: f32,
    /// Reward gain: `mean_reward_selected - mean_reward_all`.
    pub reward_gain: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// RAFT trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful RAFT trainer that records per-iteration statistics and tracks progress.
#[derive(Debug, Clone)]
pub struct RaftTrainer {
    /// RAFT hyper-parameters.
    pub config: RaftConfig,
    /// Current RAFT iteration index (0-indexed).
    pub iteration: usize,
    /// History of per-iteration statistics.
    pub history: Vec<RaftIterationStats>,
}

impl RaftTrainer {
    /// Create a new RAFT trainer at iteration 0.
    pub fn new(config: RaftConfig) -> Self {
        Self { config, iteration: 0, history: Vec::new() }
    }

    /// Record the result of a completed RAFT iteration.
    ///
    /// The provided `RaftBatch` is converted to `RaftIterationStats`, appended to
    /// history, and the iteration counter is incremented.
    ///
    /// # Errors
    /// Returns `RaftError::NoResponsesSelected` if the batch contains no selected pairs.
    pub fn process_iteration(
        &mut self,
        batch: RaftBatch,
    ) -> Result<RaftIterationStats, RaftError> {
        if batch.total_selected == 0 {
            return Err(RaftError::NoResponsesSelected);
        }

        let num_prompts = batch.selected_pairs.len();
        let selection_rate = if batch.total_generated == 0 {
            0.0
        } else {
            batch.total_selected as f32 / batch.total_generated as f32
        };

        let reward_gain = batch.mean_reward_selected - batch.mean_reward_all;

        let stats = RaftIterationStats {
            iteration: self.iteration,
            num_prompts,
            total_generated: batch.total_generated,
            total_selected: batch.total_selected,
            selection_rate,
            mean_reward_all: batch.mean_reward_all,
            mean_reward_selected: batch.mean_reward_selected,
            reward_gain,
        };

        self.history.push(stats.clone());
        self.iteration += 1;
        Ok(stats)
    }

    /// Sum of all selected examples across every recorded iteration.
    pub fn total_selected_examples(&self) -> usize {
        self.history.iter().map(|s| s.total_selected).sum()
    }

    /// Compute a simple linear trend in mean reward improvement across iterations.
    ///
    /// Returns the mean `reward_gain` over all recorded iterations, which serves
    /// as a proxy for how much the selection step is improving reward each time.
    /// Returns 0.0 if no history is available.
    pub fn mean_reward_improvement(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.reward_gain).sum();
        sum / (self.history.len() as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `ceil(n * fraction)`, clamped to `[0, n]`.
#[inline]
fn ceil_frac(n: usize, fraction: f32) -> usize {
    if n == 0 || fraction <= 0.0 {
        return 0;
    }
    if fraction >= 1.0 {
        return n;
    }
    let raw = (n as f32 * fraction).ceil() as usize;
    raw.min(n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_raft_config_defaults() {
        let cfg = RaftConfig::default();
        assert_eq!(cfg.num_samples_per_prompt, 8);
        assert!((cfg.top_k_fraction - 0.25).abs() < 1e-6);
        assert!(cfg.reward_threshold.is_none());
        assert!((cfg.temperature_sampling - 0.7).abs() < 1e-6);
        assert_eq!(cfg.num_iterations, 10);
        assert_eq!(cfg.batch_size, 32);
        assert!((cfg.learning_rate - 1e-5).abs() < 1e-10);
        assert!(cfg.normalize_rewards);
    }

    #[test]
    fn test_raft_config_custom() {
        let cfg = RaftConfig {
            num_samples_per_prompt: 4,
            top_k_fraction: 0.5,
            reward_threshold: Some(0.8),
            temperature_sampling: 1.0,
            num_iterations: 5,
            batch_size: 16,
            learning_rate: 2e-5,
            normalize_rewards: false,
        };
        assert_eq!(cfg.num_samples_per_prompt, 4);
        assert!((cfg.top_k_fraction - 0.5).abs() < 1e-6);
        assert_eq!(cfg.reward_threshold, Some(0.8));
        assert!(!cfg.normalize_rewards);
    }

    // ── RaftDataset helpers ───────────────────────────────────────────────────

    #[test]
    fn test_raft_dataset_len_is_empty() {
        let empty = RaftDataset { prompts: vec![] };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let one = RaftDataset { prompts: vec![vec![1, 2, 3]] };
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    // ── Top-K selection ───────────────────────────────────────────────────────

    fn make_responses(rewards: &[f32]) -> Vec<ScoredResponse> {
        rewards
            .iter()
            .enumerate()
            .map(|(i, &r)| ScoredResponse {
                response: vec![i as u32],
                reward: r,
                prompt_idx: 0,
            })
            .collect()
    }

    #[test]
    fn test_select_top_responses_basic() {
        let responses = make_responses(&[0.1, 0.9, 0.4, 0.7, 0.2]);
        let config = RaftConfig { top_k_fraction: 0.4, ..RaftConfig::default() };
        // ceil(5 * 0.4) = ceil(2.0) = 2
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert_eq!(selected.len(), 2);
        // Top 2 should be 0.9 and 0.7
        assert!((selected[0].reward - 0.9).abs() < 1e-6);
        assert!((selected[1].reward - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_select_top_responses_all_fraction_one() {
        let responses = make_responses(&[0.3, 0.1, 0.8]);
        let config = RaftConfig { top_k_fraction: 1.0, ..RaftConfig::default() };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_select_top_responses_empty() {
        let config = RaftConfig::default();
        let selected = RaftSelector::select_top_responses(&[], &config);
        assert!(selected.is_empty());
    }

    // ── Reward threshold filtering ────────────────────────────────────────────

    #[test]
    fn test_reward_threshold_filtering() {
        let responses = make_responses(&[0.1, 0.5, 0.9, 0.3, 0.8]);
        let config = RaftConfig {
            top_k_fraction: 1.0, // keep all by fraction
            reward_threshold: Some(0.45),
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        // Only rewards >= 0.45 survive: 0.5, 0.9, 0.8
        assert_eq!(selected.len(), 3);
        for r in &selected {
            assert!(r.reward >= 0.45, "reward {} should be >= 0.45", r.reward);
        }
    }

    #[test]
    fn test_threshold_eliminates_all_but_at_least_one_via_fraction() {
        // top_k_fraction keeps top-1, threshold is below all values → 1 survives
        let responses = make_responses(&[0.3, 0.7, 0.5]);
        let config = RaftConfig {
            top_k_fraction: 0.1, // ceil(3 * 0.1) = 1
            reward_threshold: Some(0.0),
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert_eq!(selected.len(), 1);
        assert!((selected[0].reward - 0.7).abs() < 1e-6);
    }

    // ── Reward normalization ──────────────────────────────────────────────────

    #[test]
    fn test_reward_normalization_mean_zero_std_one() {
        let mut responses = make_responses(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        RaftSelector::normalize_rewards(&mut responses);

        let n = responses.len() as f32;
        let mean = responses.iter().map(|r| r.reward).sum::<f32>() / n;
        let var = responses.iter().map(|r| (r.reward - mean).powi(2)).sum::<f32>() / n;
        let std_dev = var.sqrt();

        assert!(mean.abs() < 1e-5, "mean should be ≈ 0, got {mean}");
        assert!((std_dev - 1.0).abs() < 1e-5, "std_dev should be ≈ 1, got {std_dev}");
    }

    #[test]
    fn test_reward_normalization_identical_rewards_unchanged() {
        // All same reward → std = 0, normalization should be a no-op
        let mut responses = make_responses(&[3.0, 3.0, 3.0]);
        RaftSelector::normalize_rewards(&mut responses);
        for r in &responses {
            assert!((r.reward - 3.0).abs() < 1e-6, "identical rewards should stay unchanged");
        }
    }

    #[test]
    fn test_reward_normalization_single_element_unchanged() {
        let mut responses = make_responses(&[5.0]);
        RaftSelector::normalize_rewards(&mut responses);
        assert!((responses[0].reward - 5.0).abs() < 1e-6);
    }

    // ── SFT batch building ────────────────────────────────────────────────────

    #[test]
    fn test_build_sft_batch_basic() {
        let prompts = vec![vec![1_u32, 2], vec![3_u32, 4]];
        let scored_responses = vec![
            make_responses(&[0.2, 0.8, 0.5, 0.1]),
            make_responses(&[0.9, 0.3, 0.6, 0.4]),
        ];
        let config = RaftConfig {
            top_k_fraction: 0.25, // keep top 1 of 4 per prompt
            reward_threshold: None,
            normalize_rewards: false,
            ..RaftConfig::default()
        };

        let batch = RaftSelector::build_sft_batch(&prompts, &scored_responses, &config)
            .expect("should succeed");

        assert_eq!(batch.total_generated, 8);
        assert_eq!(batch.total_selected, 2); // 1 per prompt
        assert_eq!(batch.selected_pairs.len(), 2);
    }

    #[test]
    fn test_build_sft_batch_empty_prompts() {
        let result = RaftSelector::build_sft_batch(&[], &[], &RaftConfig::default());
        assert!(matches!(result, Err(RaftError::EmptyDataset)));
    }

    #[test]
    fn test_build_sft_batch_empty_responses_for_prompt() {
        let prompts = vec![vec![1_u32]];
        let scored_responses: Vec<Vec<ScoredResponse>> = vec![vec![]]; // zero responses
        let result =
            RaftSelector::build_sft_batch(&prompts, &scored_responses, &RaftConfig::default());
        assert!(matches!(
            result,
            Err(RaftError::InsufficientSamples { required: _, got: 0 })
        ));
    }

    // ── Selection rate ────────────────────────────────────────────────────────

    #[test]
    fn test_selection_rate_computation() {
        let batch = RaftBatch {
            selected_pairs: vec![(vec![1], vec![2])],
            total_generated: 8,
            total_selected: 2,
            mean_reward_all: 0.5,
            mean_reward_selected: 0.9,
            reward_threshold_used: 0.7,
        };
        let config = RaftConfig::default();
        let mut trainer = RaftTrainer::new(config);
        let stats = trainer.process_iteration(batch).expect("should succeed");

        assert!((stats.selection_rate - 0.25).abs() < 1e-6, "2/8 = 0.25, got {}", stats.selection_rate);
    }

    // ── Reward gain ───────────────────────────────────────────────────────────

    #[test]
    fn test_reward_gain_computation() {
        let batch = RaftBatch {
            selected_pairs: vec![(vec![1], vec![2])],
            total_generated: 4,
            total_selected: 1,
            mean_reward_all: 0.5,
            mean_reward_selected: 0.9,
            reward_threshold_used: 0.8,
        };
        let config = RaftConfig::default();
        let mut trainer = RaftTrainer::new(config);
        let stats = trainer.process_iteration(batch).expect("should succeed");

        assert!(
            (stats.reward_gain - 0.4).abs() < 1e-5,
            "reward_gain = 0.9 - 0.5 = 0.4, got {}",
            stats.reward_gain
        );
    }

    // ── Trainer iteration tracking ────────────────────────────────────────────

    #[test]
    fn test_raft_trainer_iteration_tracking() {
        let mut trainer = RaftTrainer::new(RaftConfig::default());
        assert_eq!(trainer.iteration, 0);
        assert!(trainer.history.is_empty());

        let batch1 = RaftBatch {
            selected_pairs: vec![(vec![1], vec![2])],
            total_generated: 8,
            total_selected: 2,
            mean_reward_all: 0.4,
            mean_reward_selected: 0.8,
            reward_threshold_used: 0.6,
        };
        trainer.process_iteration(batch1).expect("iter 0");
        assert_eq!(trainer.iteration, 1);
        assert_eq!(trainer.history.len(), 1);

        let batch2 = RaftBatch {
            selected_pairs: vec![(vec![3], vec![4])],
            total_generated: 8,
            total_selected: 2,
            mean_reward_all: 0.5,
            mean_reward_selected: 0.9,
            reward_threshold_used: 0.7,
        };
        trainer.process_iteration(batch2).expect("iter 1");
        assert_eq!(trainer.iteration, 2);
        assert_eq!(trainer.history.len(), 2);
    }

    #[test]
    fn test_raft_trainer_total_selected_examples() {
        let mut trainer = RaftTrainer::new(RaftConfig::default());

        for i in 0..3_usize {
            let batch = RaftBatch {
                selected_pairs: vec![(vec![i as u32], vec![i as u32 + 1])],
                total_generated: 8,
                total_selected: (i + 1) * 2, // 2, 4, 6
                mean_reward_all: 0.5,
                mean_reward_selected: 0.8,
                reward_threshold_used: 0.6,
            };
            trainer.process_iteration(batch).expect("should succeed");
        }

        // 2 + 4 + 6 = 12
        assert_eq!(trainer.total_selected_examples(), 12);
    }

    #[test]
    fn test_raft_mean_reward_improvement() {
        let mut trainer = RaftTrainer::new(RaftConfig::default());
        assert_eq!(trainer.mean_reward_improvement(), 0.0);

        let rewards_all = [0.3, 0.4, 0.5];
        let rewards_sel = [0.7, 0.8, 0.9];

        for i in 0..3 {
            let batch = RaftBatch {
                selected_pairs: vec![(vec![i as u32], vec![i as u32 + 1])],
                total_generated: 8,
                total_selected: 2,
                mean_reward_all: rewards_all[i],
                mean_reward_selected: rewards_sel[i],
                reward_threshold_used: 0.6,
            };
            trainer.process_iteration(batch).expect("iter");
        }

        // reward_gains: 0.4, 0.4, 0.4 → mean = 0.4
        let improvement = trainer.mean_reward_improvement();
        assert!(
            (improvement - 0.4).abs() < 1e-5,
            "expected 0.4, got {improvement}"
        );
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_process_iteration_empty_batch_error() {
        let mut trainer = RaftTrainer::new(RaftConfig::default());
        let empty_batch = RaftBatch {
            selected_pairs: vec![],
            total_generated: 4,
            total_selected: 0,
            mean_reward_all: 0.5,
            mean_reward_selected: 0.0,
            reward_threshold_used: 0.0,
        };
        let result = trainer.process_iteration(empty_batch);
        assert!(matches!(result, Err(RaftError::NoResponsesSelected)));
    }

    #[test]
    fn test_raft_error_display() {
        let e1 = RaftError::EmptyDataset;
        let e2 = RaftError::NoResponsesSelected;
        let e3 = RaftError::InsufficientSamples { required: 8, got: 3 };
        let e4 = RaftError::InvalidConfig("bad fraction".to_string());
        let e5 = RaftError::RewardNormalizationFailed("zero std".to_string());

        assert!(e1.to_string().contains("empty"));
        assert!(e2.to_string().contains("selected"));
        assert!(e3.to_string().contains("8"));
        assert!(e3.to_string().contains("3"));
        assert!(e4.to_string().contains("bad fraction"));
        assert!(e5.to_string().contains("zero std"));
    }

    // ── New tests ─────────────────────────────────────────────────────────────

    // Test: ranking — top-k selection correctly orders by reward descending
    #[test]
    fn test_ranking_top_k_ordered_by_reward() {
        let rewards_input = [0.3_f32, 0.9, 0.1, 0.7, 0.5];
        let responses = make_responses(&rewards_input);
        let config = RaftConfig { top_k_fraction: 0.4, reward_threshold: None, ..RaftConfig::default() };
        // ceil(5 * 0.4) = 2, should pick 0.9 and 0.7
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert_eq!(selected.len(), 2);
        assert!((selected[0].reward - 0.9).abs() < 1e-6, "first should be 0.9, got {}", selected[0].reward);
        assert!((selected[1].reward - 0.7).abs() < 1e-6, "second should be 0.7, got {}", selected[1].reward);
    }

    // Test: reward threshold filtering removes below-threshold entries
    #[test]
    fn test_reward_threshold_filters_low_rewards() {
        let responses = make_responses(&[0.1, 0.9, 0.5, 0.8, 0.2]);
        let config = RaftConfig {
            top_k_fraction: 1.0,
            reward_threshold: Some(0.5),
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        // Only >= 0.5: 0.9, 0.8, 0.5
        assert_eq!(selected.len(), 3);
        for r in &selected {
            assert!(r.reward >= 0.5, "all selected should be >= 0.5, got {}", r.reward);
        }
    }

    // Test: SFT loss training on selected samples — selected mean_reward > total mean_reward
    #[test]
    fn test_sft_selected_mean_reward_greater_than_all() {
        let prompts = vec![vec![1_u32]];
        let responses = make_responses(&[0.1, 0.2, 0.9, 0.3, 0.8]);
        let scored_responses = vec![responses];
        let config = RaftConfig {
            top_k_fraction: 0.4,
            reward_threshold: None,
            normalize_rewards: false,
            ..RaftConfig::default()
        };
        let batch = RaftSelector::build_sft_batch(&prompts, &scored_responses, &config).expect("ok");
        assert!(
            batch.mean_reward_selected > batch.mean_reward_all,
            "selected mean reward {} should > all mean reward {}",
            batch.mean_reward_selected, batch.mean_reward_all
        );
    }

    // Test: higher reward → higher priority for selection (SFT trains on good samples)
    #[test]
    fn test_higher_reward_selected_over_lower() {
        let responses = make_responses(&[0.1, 0.9, 0.5]);
        let config = RaftConfig {
            top_k_fraction: 0.34, // ceil(3 * 0.34) = ceil(1.02) = 2
            reward_threshold: None,
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        // Top 2: 0.9 and 0.5
        let min_selected = selected.iter().map(|r| r.reward).fold(f32::INFINITY, f32::min);
        let max_not_selected = 0.1_f32; // the lowest reward that wasn't selected
        assert!(
            min_selected > max_not_selected,
            "lowest selected reward {min_selected} should > highest unselected {max_not_selected}"
        );
    }

    // Test: rejection threshold — all samples below threshold are discarded
    #[test]
    fn test_rejection_threshold_discards_all_below() {
        let responses = make_responses(&[0.1, 0.2, 0.3]);
        let config = RaftConfig {
            top_k_fraction: 1.0,
            reward_threshold: Some(0.5), // all below this
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert!(selected.is_empty(), "all rewards below threshold, nothing should be selected");
    }

    // Test: batch reward normalization changes reward values to mean=0, std=1
    #[test]
    fn test_batch_reward_normalization_properties() {
        let mut responses = make_responses(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        RaftSelector::normalize_rewards(&mut responses);
        let n = responses.len() as f32;
        let mean = responses.iter().map(|r| r.reward).sum::<f32>() / n;
        let var = responses.iter().map(|r| (r.reward - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 1e-4, "mean should be ≈ 0, got {mean}");
        assert!((var.sqrt() - 1.0).abs() < 1e-4, "std should be ≈ 1, got {}", var.sqrt());
    }

    // Test: reward scaling — verify normalized scores are in reasonable range
    #[test]
    fn test_reward_scaling_range_after_normalization() {
        // With small N and known values, normalized scores should span roughly [-2, 2]
        let mut responses = make_responses(&[0.0, 1.0, 2.0, 3.0]);
        RaftSelector::normalize_rewards(&mut responses);
        for r in &responses {
            assert!(
                r.reward.abs() <= 3.0,
                "normalized reward {} should be within reasonable range [-3, 3]", r.reward
            );
        }
    }

    // Test: RAFT iteration — count of selected vs discarded samples
    #[test]
    fn test_raft_iteration_selected_vs_discarded_count() {
        let prompts = vec![vec![1_u32], vec![2_u32]];
        let scored_responses = vec![
            make_responses(&[0.1, 0.5, 0.9, 0.3]), // 4 responses
            make_responses(&[0.2, 0.8, 0.4, 0.6]), // 4 responses
        ];
        let config = RaftConfig {
            top_k_fraction: 0.5, // keep top 2 of 4 per prompt
            reward_threshold: None,
            normalize_rewards: false,
            ..RaftConfig::default()
        };
        let batch = RaftSelector::build_sft_batch(&prompts, &scored_responses, &config).expect("ok");
        assert_eq!(batch.total_generated, 8, "8 total responses");
        assert_eq!(batch.total_selected, 4, "4 selected (top-2 from each prompt)");
        let discarded = batch.total_generated - batch.total_selected;
        assert_eq!(discarded, 4, "4 discarded");
    }

    // Test: with equal rewards — all or random selection (at least 1 must survive)
    #[test]
    fn test_equal_rewards_selection_at_least_one_survives() {
        let responses = make_responses(&[0.5, 0.5, 0.5, 0.5]);
        let config = RaftConfig {
            top_k_fraction: 0.25,
            reward_threshold: None,
            ..RaftConfig::default()
        };
        let selected = RaftSelector::select_top_responses(&responses, &config);
        assert!(!selected.is_empty(), "at least one should survive selection with equal rewards");
        // ceil(4 * 0.25) = 1
        assert_eq!(selected.len(), 1, "expected 1 selected, got {}", selected.len());
    }

    // Test: empty batch handling after threshold filtering
    #[test]
    fn test_empty_batch_after_threshold_filtering_returns_error() {
        let prompts = vec![vec![1_u32]];
        // All rewards below threshold
        let scored_responses = vec![make_responses(&[0.1, 0.2, 0.3])];
        let config = RaftConfig {
            top_k_fraction: 1.0,
            reward_threshold: Some(1.0), // nothing survives
            normalize_rewards: false,
            ..RaftConfig::default()
        };
        let result = RaftSelector::build_sft_batch(&prompts, &scored_responses, &config);
        assert!(matches!(result, Err(RaftError::NoResponsesSelected)), "should fail with NoResponsesSelected");
    }

    // Test: ceil_frac correctness
    #[test]
    fn test_ceil_frac_correctness() {
        // ceil(10 * 0.25) = ceil(2.5) = 3
        assert_eq!(ceil_frac(10, 0.25), 3);
        // ceil(4 * 0.25) = ceil(1.0) = 1
        assert_eq!(ceil_frac(4, 0.25), 1);
        // ceil(0 * anything) = 0
        assert_eq!(ceil_frac(0, 0.5), 0);
        // fraction >= 1 returns n
        assert_eq!(ceil_frac(5, 1.0), 5);
        assert_eq!(ceil_frac(5, 2.0), 5);
    }

    // Test: ScoredResponse prompt_idx is preserved
    #[test]
    fn test_scored_response_prompt_idx_preserved() {
        let resp = ScoredResponse {
            response: vec![1, 2, 3],
            reward: 0.7,
            prompt_idx: 42,
        };
        assert_eq!(resp.prompt_idx, 42);
    }

    // Test: build_sft_batch reward_threshold_used equals min selected reward
    #[test]
    fn test_build_sft_batch_reward_threshold_used_is_min_selected() {
        let prompts = vec![vec![1_u32]];
        let scored_responses = vec![make_responses(&[0.2, 0.5, 0.8, 0.9])];
        let config = RaftConfig {
            top_k_fraction: 0.5, // keep top 2: 0.9 and 0.8
            reward_threshold: None,
            normalize_rewards: false,
            ..RaftConfig::default()
        };
        let batch = RaftSelector::build_sft_batch(&prompts, &scored_responses, &config).expect("ok");
        // Min of top-2 selected rewards: 0.8
        assert!((batch.reward_threshold_used - 0.8).abs() < 1e-5,
            "threshold_used should equal min selected reward, got {}", batch.reward_threshold_used);
    }

    // Test: RaftDataset clone
    #[test]
    fn test_raft_dataset_clone() {
        let ds = RaftDataset { prompts: vec![vec![1, 2], vec![3, 4]] };
        let clone = ds.clone();
        assert_eq!(clone.len(), ds.len());
        assert_eq!(clone.prompts[0], ds.prompts[0]);
    }

    // Test: trainer history accumulates correctly over multiple iterations
    #[test]
    fn test_raft_trainer_history_accumulation() {
        let mut trainer = RaftTrainer::new(RaftConfig::default());
        for i in 0..5_usize {
            let batch = RaftBatch {
                selected_pairs: vec![(vec![i as u32], vec![i as u32 + 1])],
                total_generated: 8,
                total_selected: 2,
                mean_reward_all: 0.5,
                mean_reward_selected: 0.8,
                reward_threshold_used: 0.6,
            };
            trainer.process_iteration(batch).expect("ok");
        }
        assert_eq!(trainer.history.len(), 5);
        assert_eq!(trainer.iteration, 5);
        for (i, stat) in trainer.history.iter().enumerate() {
            assert_eq!(stat.iteration, i, "iteration index should match history index");
        }
    }
}
