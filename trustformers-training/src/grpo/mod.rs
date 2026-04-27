//! Group Relative Policy Optimization (GRPO) — DeepSeek-R1's training algorithm.
//!
//! GRPO eliminates the value model by computing group-relative advantages across
//! G responses sampled for the same prompt. The policy is optimized with a clipped
//! objective similar to PPO, augmented by a KL penalty against a frozen reference policy.

use std::fmt;

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Configuration for GRPO training.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// G: number of responses sampled per prompt (default 8).
    pub group_size: usize,
    /// PPO clipping parameter ε (default 0.2).
    pub epsilon: f64,
    /// KL regularization coefficient β (default 0.01).
    pub beta: f64,
    /// Sampling temperature (default 1.0).
    pub temperature: f64,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Total number of training iterations.
    pub num_iterations: u64,
    /// Whether to z-score–normalize advantages within a group (default true).
    pub advantage_normalization: bool,
    /// Whether to add a KL(policy ‖ ref_policy) penalty term (default true).
    pub use_kl_penalty: bool,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            epsilon: 0.2,
            beta: 0.01,
            temperature: 1.0,
            max_new_tokens: 512,
            num_iterations: 1000,
            advantage_normalization: true,
            use_kl_penalty: true,
        }
    }
}

// ──────────────────────────────────────────────
// GroupResponse
// ──────────────────────────────────────────────

/// A single response within a GRPO group.
#[derive(Debug, Clone)]
pub struct GroupResponse {
    /// Index of this response within its group.
    pub response_id: usize,
    /// Token ids of the response.
    pub tokens: Vec<u32>,
    /// Per-token log-probabilities under the current policy.
    pub log_probs: Vec<f64>,
    /// Per-token log-probabilities under the frozen reference policy.
    pub ref_log_probs: Vec<f64>,
    /// Scalar reward assigned by the reward model.
    pub reward: f64,
    /// Relative advantage computed by [`ResponseGroup::compute_advantages`].
    pub advantage: f64,
}

impl GroupResponse {
    /// Construct a new `GroupResponse` with advantage initialised to 0.
    pub fn new(
        response_id: usize,
        tokens: Vec<u32>,
        log_probs: Vec<f64>,
        ref_log_probs: Vec<f64>,
        reward: f64,
    ) -> Self {
        Self {
            response_id,
            tokens,
            log_probs,
            ref_log_probs,
            reward,
            advantage: 0.0,
        }
    }

    /// Sum of per-token log-probabilities (sequence log-likelihood under policy).
    pub fn sequence_log_prob(&self) -> f64 {
        self.log_probs.iter().sum()
    }

    /// Mean per-token KL divergence: E_t[log π(t) − log π_ref(t)].
    pub fn kl_divergence(&self) -> f64 {
        if self.log_probs.is_empty() {
            return 0.0;
        }
        let n = self.log_probs.len() as f64;
        self.log_probs
            .iter()
            .zip(self.ref_log_probs.iter())
            .map(|(lp, rlp)| lp - rlp)
            .sum::<f64>()
            / n
    }

    /// Number of tokens in this response.
    pub fn response_length(&self) -> usize {
        self.tokens.len()
    }
}

// ──────────────────────────────────────────────
// ResponseGroup
// ──────────────────────────────────────────────

/// A group of G responses generated for the same prompt.
#[derive(Debug, Clone)]
pub struct ResponseGroup {
    /// The prompt text that generated all responses.
    pub prompt: String,
    /// The G individual responses.
    pub responses: Vec<GroupResponse>,
}

impl ResponseGroup {
    /// Create a new group.
    pub fn new(prompt: String, responses: Vec<GroupResponse>) -> Self {
        Self { prompt, responses }
    }

    /// Compute group-relative advantages and write them into each response.
    ///
    /// `A_i = (r_i − mean(r)) / (std(r) + ε)`.
    /// If all rewards are identical the advantages are all set to 0.
    pub fn compute_advantages(&mut self) {
        if self.responses.is_empty() {
            return;
        }
        let mean = self.mean_reward();
        let std = self.std_reward();
        let denom = std + 1e-8;
        // Guard: if rewards are effectively constant set all to 0.
        if std < 1e-12 {
            for resp in &mut self.responses {
                resp.advantage = 0.0;
            }
        } else {
            for resp in &mut self.responses {
                resp.advantage = (resp.reward - mean) / denom;
            }
        }
    }

    /// Mean reward across the group.
    pub fn mean_reward(&self) -> f64 {
        if self.responses.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.responses.iter().map(|r| r.reward).sum();
        sum / self.responses.len() as f64
    }

    /// Population standard deviation of rewards across the group.
    pub fn std_reward(&self) -> f64 {
        if self.responses.is_empty() {
            return 0.0;
        }
        let mean = self.mean_reward();
        let variance: f64 = self
            .responses
            .iter()
            .map(|r| (r.reward - mean).powi(2))
            .sum::<f64>()
            / self.responses.len() as f64;
        variance.sqrt()
    }

    /// Return a reference to the response with the highest reward, if any.
    pub fn best_response(&self) -> Option<&GroupResponse> {
        self.responses
            .iter()
            .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ──────────────────────────────────────────────
// GrpoLoss
// ──────────────────────────────────────────────

/// Computes the GRPO training objective for a response group.
pub struct GrpoLoss {
    config: GrpoConfig,
    step: u64,
}

impl GrpoLoss {
    /// Create a new loss computer from the supplied configuration.
    pub fn new(config: GrpoConfig) -> Self {
        Self { config, step: 0 }
    }

    /// Clipped policy-gradient loss for a single token.
    ///
    /// ```text
    /// ratio          = exp(log_prob − old_log_prob)
    /// clipped_ratio  = clip(ratio, 1−ε, 1+ε)
    /// loss           = −min(ratio × A, clipped_ratio × A)
    /// ```
    pub fn policy_gradient_loss(
        &self,
        log_prob: f64,
        old_log_prob: f64,
        advantage: f64,
    ) -> f64 {
        let ratio = (log_prob - old_log_prob).exp();
        let clipped_ratio =
            ratio.max(1.0 - self.config.epsilon).min(1.0 + self.config.epsilon);
        let unclipped = ratio * advantage;
        let clipped = clipped_ratio * advantage;
        -unclipped.min(clipped)
    }

    /// KL penalty contribution for a single token: `β × (log_prob − ref_log_prob)`.
    pub fn kl_penalty(&self, log_prob: f64, ref_log_prob: f64) -> f64 {
        self.config.beta * (log_prob - ref_log_prob)
    }

    /// Compute the full GRPO loss for a response group.
    ///
    /// For every (response, token) pair:
    /// - `pg_loss  = policy_gradient_loss(log_prob, ref_log_prob, advantage)`
    /// - `kl       = kl_penalty(log_prob, ref_log_prob)` iff `use_kl_penalty`
    /// - `token_loss = pg_loss + kl`
    ///
    /// Response loss = mean over tokens; Group loss = mean over responses.
    pub fn compute_loss(
        &mut self,
        group: &ResponseGroup,
    ) -> Result<GrpoLossResult, GrpoError> {
        if group.responses.is_empty() {
            return Err(GrpoError::EmptyGroup);
        }

        // Validate advantages have been computed (non-zero or all rewards equal).
        // We detect "not computed" by checking length mismatches first.
        for (idx, resp) in group.responses.iter().enumerate() {
            if resp.log_probs.len() != resp.ref_log_probs.len() {
                return Err(GrpoError::LengthMismatch(idx));
            }
        }

        // Check that advantages look set: if every response has advantage==0.0 AND
        // rewards are NOT all equal, then compute_advantages was probably not called.
        let all_advantages_zero = group.responses.iter().all(|r| r.advantage == 0.0);
        let rewards_all_equal = {
            let first = group.responses[0].reward;
            group.responses.iter().all(|r| (r.reward - first).abs() < 1e-12)
        };
        if all_advantages_zero && !rewards_all_equal {
            return Err(GrpoError::AdvantagesNotComputed);
        }

        let mut total_pg_loss = 0.0_f64;
        let mut total_kl = 0.0_f64;
        let mut total_clipped_tokens = 0_u64;
        let mut total_tokens = 0_u64;
        let mut response_lengths = Vec::with_capacity(group.responses.len());

        for resp in &group.responses {
            if resp.log_probs.is_empty() {
                response_lengths.push(0);
                continue;
            }

            let n = resp.log_probs.len();
            let mut resp_pg = 0.0_f64;
            let mut resp_kl = 0.0_f64;
            let mut resp_clipped = 0_u64;

            for i in 0..n {
                let lp = resp.log_probs[i];
                let rlp = resp.ref_log_probs[i];
                let adv = resp.advantage;

                // Policy gradient loss
                let ratio = (lp - rlp).exp();
                let clipped_ratio =
                    ratio.max(1.0 - self.config.epsilon).min(1.0 + self.config.epsilon);
                let was_clipped = (ratio - clipped_ratio).abs() > 1e-12;
                if was_clipped {
                    resp_clipped += 1;
                }
                resp_pg += -(ratio * adv).min(clipped_ratio * adv);

                // KL penalty
                if self.config.use_kl_penalty {
                    resp_kl += self.config.beta * (lp - rlp);
                }
            }

            total_pg_loss += resp_pg / n as f64;
            total_kl += resp_kl / n as f64;
            total_clipped_tokens += resp_clipped;
            total_tokens += n as u64;
            response_lengths.push(n);
        }

        let num_responses = group.responses.len() as f64;
        let mean_pg_loss = total_pg_loss / num_responses;
        let mean_kl = total_kl / num_responses;
        let total_loss = mean_pg_loss + mean_kl;

        let clip_fraction = if total_tokens > 0 {
            total_clipped_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };

        let mean_reward = group.mean_reward();
        let mean_advantage = if group.responses.is_empty() {
            0.0
        } else {
            group.responses.iter().map(|r| r.advantage).sum::<f64>()
                / group.responses.len() as f64
        };

        self.step += 1;

        Ok(GrpoLossResult {
            total_loss,
            policy_gradient_loss: mean_pg_loss,
            kl_penalty: mean_kl,
            mean_reward,
            mean_advantage,
            clip_fraction,
            response_lengths,
        })
    }

    /// Number of `compute_loss` calls completed so far.
    pub fn step(&self) -> u64 {
        self.step
    }
}

// ──────────────────────────────────────────────
// GrpoLossResult
// ──────────────────────────────────────────────

/// Aggregated loss statistics for one GRPO update step.
#[derive(Debug, Clone)]
pub struct GrpoLossResult {
    /// Combined policy-gradient + KL loss.
    pub total_loss: f64,
    /// Mean policy-gradient component.
    pub policy_gradient_loss: f64,
    /// Mean KL-penalty component.
    pub kl_penalty: f64,
    /// Mean reward across the group.
    pub mean_reward: f64,
    /// Mean advantage across the group.
    pub mean_advantage: f64,
    /// Fraction of tokens where the importance ratio was clipped.
    pub clip_fraction: f64,
    /// Token lengths of each response in the group.
    pub response_lengths: Vec<usize>,
}

impl fmt::Display for GrpoLossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GrpoLossResult {{ total_loss: {:.4}, pg_loss: {:.4}, kl: {:.4}, \
             mean_reward: {:.4}, mean_advantage: {:.4}, clip_fraction: {:.4}, \
             response_lengths: {:?} }}",
            self.total_loss,
            self.policy_gradient_loss,
            self.kl_penalty,
            self.mean_reward,
            self.mean_advantage,
            self.clip_fraction,
            self.response_lengths,
        )
    }
}

// ──────────────────────────────────────────────
// GrpoError
// ──────────────────────────────────────────────

/// Errors that can arise during GRPO loss computation.
#[derive(Debug, thiserror::Error)]
pub enum GrpoError {
    /// The response group contains no responses.
    #[error("Empty response group")]
    EmptyGroup,
    /// `log_probs` and `ref_log_probs` have different lengths for the given response.
    #[error("Mismatched log prob lengths in response {0}")]
    LengthMismatch(usize),
    /// `compute_advantages` has not been called on the group yet.
    #[error("Advantages not computed — call compute_advantages first")]
    AdvantagesNotComputed,
}

// ──────────────────────────────────────────────
// GrpoSample
// ──────────────────────────────────────────────

/// A single GRPO training sample: one prompt with G completions and rewards.
#[derive(Debug, Clone)]
pub struct GrpoSample {
    /// Prompt token ids.
    pub prompt: Vec<u32>,
    /// G completions, each a sequence of token ids.
    pub completions: Vec<Vec<u32>>,
    /// Scalar reward per completion (same length as `completions`).
    pub rewards: Vec<f32>,
}

impl GrpoSample {
    /// Create a new sample, validating that completions and rewards have equal length.
    pub fn new(
        prompt: Vec<u32>,
        completions: Vec<Vec<u32>>,
        rewards: Vec<f32>,
    ) -> Result<Self, GrpoError> {
        if completions.len() != rewards.len() {
            return Err(GrpoError::LengthMismatch(0));
        }
        Ok(Self { prompt, completions, rewards })
    }

    /// Number of completions in this sample.
    pub fn group_size(&self) -> usize {
        self.completions.len()
    }
}

// ──────────────────────────────────────────────
// GrpoStats
// ──────────────────────────────────────────────

/// Per-iteration statistics for a GRPO training run.
#[derive(Debug, Clone)]
pub struct GrpoStats {
    /// Current training iteration (0-indexed).
    pub iteration: usize,
    /// Mean reward across all completions in the group.
    pub mean_reward: f32,
    /// Standard deviation of rewards across the group.
    pub std_reward: f32,
    /// Policy gradient component of the loss.
    pub policy_loss: f32,
    /// Mean KL divergence per token.
    pub kl_divergence: f32,
    /// Fraction of tokens where the importance ratio was clipped.
    pub clip_fraction: f32,
}

impl fmt::Display for GrpoStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GrpoStats {{ iter={}, mean_reward={:.4}, std_reward={:.4}, \
             policy_loss={:.4}, kl={:.4}, clip_frac={:.4} }}",
            self.iteration,
            self.mean_reward,
            self.std_reward,
            self.policy_loss,
            self.kl_divergence,
            self.clip_fraction,
        )
    }
}

// ──────────────────────────────────────────────
// GrpoLossOutput (f32 API)
// ──────────────────────────────────────────────

/// Output of the `compute_grpo_loss` standalone function.
#[derive(Debug, Clone)]
pub struct GrpoLossOutput {
    /// Policy gradient loss (negated clipped ratio objective).
    pub policy_loss: f32,
    /// KL divergence penalty component: β × mean(log π_θ − log π_ref).
    pub kl_loss: f32,
    /// Combined: policy_loss + kl_loss.
    pub total_loss: f32,
    /// Mean reward across the group.
    pub mean_reward: f32,
    /// Mean normalised advantage across the group.
    pub mean_advantage: f32,
}

// ──────────────────────────────────────────────
// Standalone helpers (f32 API)
// ──────────────────────────────────────────────

/// Compute group-relative normalised advantages from a slice of raw rewards.
///
/// `A_i = (r_i − μ_g) / (σ_g + ε)`
///
/// When all rewards are identical (σ_g < ε) every advantage is set to 0.
pub fn compute_group_advantages(rewards: &[f32], epsilon: f32) -> Vec<f32> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let variance = rewards.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    let std = variance.sqrt();
    if std < epsilon {
        return vec![0.0f32; rewards.len()];
    }
    let denom = std + epsilon;
    rewards.iter().map(|r| (r - mean) / denom).collect()
}

/// Configuration for the standalone `compute_grpo_loss` function.
///
/// This mirrors `GrpoConfig` with `f32` fields for convenience in tight loops.
#[derive(Debug, Clone)]
pub struct GrpoLossConfig {
    /// Number of completions per group (informational; not enforced here).
    pub group_size: usize,
    /// KL regularisation coefficient β.
    pub beta: f32,
    /// Advantage normalisation epsilon ε (used in advantage computation).
    pub epsilon: f32,
    /// Maximum generated sequence length (informational).
    pub max_completion_length: usize,
    /// PPO clipping range ε_clip.
    pub clip_range: f32,
    /// KL penalty coefficient (may differ from β when using separate schedules).
    pub kl_coef: f32,
}

impl Default for GrpoLossConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            beta: 0.01,
            epsilon: 1e-8,
            max_completion_length: 512,
            clip_range: 0.2,
            kl_coef: 0.01,
        }
    }
}

/// Compute the GRPO loss for one group of completions.
///
/// All three slices must have the same length (one element per completion in
/// the group).  Each element is the **mean per-token** log-prob for that
/// completion.
///
/// Algorithm:
/// 1. Compute group advantages via `compute_group_advantages`.
/// 2. For each completion:
///    - ratio = exp(log_prob − ref_log_prob)
///    - L_clip = min(ratio × A, clip(ratio, 1−ε_clip, 1+ε_clip) × A)
///    - kl = log_prob − ref_log_prob  (per-token estimate)
/// 3. policy_loss = −mean(L_clip), kl_loss = β × mean(kl)
/// 4. total_loss = policy_loss + kl_loss
pub fn compute_grpo_loss(
    log_probs: &[f32],
    ref_log_probs: &[f32],
    advantages: &[f32],
    config: &GrpoLossConfig,
) -> Result<GrpoLossOutput, GrpoError> {
    if log_probs.is_empty() {
        return Err(GrpoError::EmptyGroup);
    }
    if log_probs.len() != ref_log_probs.len() || log_probs.len() != advantages.len() {
        return Err(GrpoError::LengthMismatch(0));
    }

    let n = log_probs.len() as f32;
    let eps = config.clip_range;

    let mut pg_sum = 0.0_f32;
    let mut kl_sum = 0.0_f32;

    for i in 0..log_probs.len() {
        let lp = log_probs[i];
        let rlp = ref_log_probs[i];
        let adv = advantages[i];

        let ratio = (lp - rlp).exp();
        let clipped = ratio.max(1.0 - eps).min(1.0 + eps);
        let l_unclip = ratio * adv;
        let l_clip = clipped * adv;
        pg_sum += -l_unclip.min(l_clip);

        kl_sum += lp - rlp;
    }

    let policy_loss = pg_sum / n;
    let kl_loss = config.kl_coef * (kl_sum / n);
    let total_loss = policy_loss + kl_loss;

    let mean_reward = 0.0f32; // caller computes from raw rewards
    let mean_advantage = advantages.iter().sum::<f32>() / n;

    Ok(GrpoLossOutput {
        policy_loss,
        kl_loss,
        total_loss,
        mean_reward,
        mean_advantage,
    })
}

// ──────────────────────────────────────────────
// GrpoTrainer
// ──────────────────────────────────────────────

/// Stateful GRPO trainer that accumulates per-iteration statistics.
pub struct GrpoTrainer {
    config: GrpoLossConfig,
    iteration: usize,
    history: Vec<GrpoStats>,
}

impl GrpoTrainer {
    /// Create a new trainer from the given configuration.
    pub fn new(config: GrpoLossConfig) -> Self {
        Self { config, iteration: 0, history: Vec::new() }
    }

    /// Process one batch of GRPO samples and return aggregated statistics.
    ///
    /// For each sample the group advantages are computed from the sample's
    /// rewards, then the GRPO loss is computed using the mean per-token
    /// log-prob as the per-completion summary.  Results are averaged across
    /// the batch.
    pub fn train_step(
        &mut self,
        samples: &[GrpoSample],
        log_probs: &[Vec<f32>],
        ref_log_probs: &[Vec<f32>],
    ) -> Result<GrpoStats, GrpoError> {
        if samples.is_empty() {
            return Err(GrpoError::EmptyGroup);
        }
        if samples.len() != log_probs.len() || samples.len() != ref_log_probs.len() {
            return Err(GrpoError::LengthMismatch(0));
        }

        let mut total_policy_loss = 0.0f32;
        let mut total_kl = 0.0f32;
        let mut total_mean_reward = 0.0f32;
        let mut total_std_reward = 0.0f32;
        let mut total_clip_frac = 0.0f32;

        for (idx, sample) in samples.iter().enumerate() {
            let g = sample.group_size();
            if g == 0 {
                continue;
            }

            let lps = &log_probs[idx];
            let rlps = &ref_log_probs[idx];

            if lps.len() != g || rlps.len() != g {
                return Err(GrpoError::LengthMismatch(idx));
            }

            let advantages = compute_group_advantages(&sample.rewards, self.config.epsilon);

            let result = compute_grpo_loss(lps, rlps, &advantages, &self.config)?;

            let n = g as f32;
            let mean_r = sample.rewards.iter().sum::<f32>() / n;
            let var_r = sample.rewards.iter().map(|r| (r - mean_r) * (r - mean_r)).sum::<f32>() / n;
            let std_r = var_r.sqrt();

            // Clip fraction: count completions where ratio is outside [1-ε, 1+ε]
            let mut clipped = 0usize;
            for i in 0..g {
                let ratio = (lps[i] - rlps[i]).exp();
                if ratio < 1.0 - self.config.clip_range || ratio > 1.0 + self.config.clip_range {
                    clipped += 1;
                }
            }
            let clip_frac = clipped as f32 / n;

            total_policy_loss += result.policy_loss;
            total_kl += result.kl_loss;
            total_mean_reward += mean_r;
            total_std_reward += std_r;
            total_clip_frac += clip_frac;
        }

        let m = samples.len() as f32;
        let stats = GrpoStats {
            iteration: self.iteration,
            mean_reward: total_mean_reward / m,
            std_reward: total_std_reward / m,
            policy_loss: total_policy_loss / m,
            kl_divergence: total_kl / m,
            clip_fraction: total_clip_frac / m,
        };

        self.history.push(stats.clone());
        self.iteration += 1;

        Ok(stats)
    }

    /// Full history of statistics from all training steps.
    pub fn history(&self) -> &[GrpoStats] {
        &self.history
    }

    /// Current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_response(id: usize, lp: Vec<f64>, rlp: Vec<f64>, reward: f64) -> GroupResponse {
        let tokens: Vec<u32> = (0..lp.len() as u32).collect();
        GroupResponse::new(id, tokens, lp, rlp, reward)
    }

    // ── Test 1: sequence_log_prob ──────────────────────────────────────────
    #[test]
    fn test_sequence_log_prob() {
        let resp = make_response(0, vec![-1.0, -2.0, -3.0], vec![-1.0, -1.0, -1.0], 1.0);
        let slp = resp.sequence_log_prob();
        assert!((slp - (-6.0)).abs() < 1e-10, "expected -6.0, got {}", slp);
    }

    // ── Test 2: kl_divergence ─────────────────────────────────────────────
    #[test]
    fn test_kl_divergence() {
        // log_prob = [-1, -2, -3], ref_log_prob = [-1, -1, -1]
        // kl = mean(0, -1, -2) = -1
        let resp = make_response(0, vec![-1.0, -2.0, -3.0], vec![-1.0, -1.0, -1.0], 1.0);
        let kl = resp.kl_divergence();
        assert!((kl - (-1.0)).abs() < 1e-10, "expected -1.0, got {}", kl);
    }

    // ── Test 3: compute_advantages mean ───────────────────────────────────
    #[test]
    fn test_compute_advantages_mean() {
        let r1 = make_response(0, vec![-0.5], vec![-0.5], 1.0);
        let r2 = make_response(1, vec![-0.5], vec![-0.5], 3.0);
        let r3 = make_response(2, vec![-0.5], vec![-0.5], 5.0);
        let mut group = ResponseGroup::new("prompt".into(), vec![r1, r2, r3]);
        group.compute_advantages();
        // mean = 3, std = sqrt((4+0+4)/3) = sqrt(8/3)
        let mean_adv: f64 =
            group.responses.iter().map(|r| r.advantage).sum::<f64>() / 3.0;
        assert!(mean_adv.abs() < 1e-9, "mean advantage should be ≈ 0, got {}", mean_adv);
    }

    // ── Test 4: compute_advantages std ────────────────────────────────────
    #[test]
    fn test_compute_advantages_std() {
        let r1 = make_response(0, vec![-0.5], vec![-0.5], 0.0);
        let r2 = make_response(1, vec![-0.5], vec![-0.5], 2.0);
        let mut group = ResponseGroup::new("prompt".into(), vec![r1, r2]);
        group.compute_advantages();
        // mean=1, std=1  → advantages = [-1, 1]
        assert!((group.responses[0].advantage - (-1.0)).abs() < 1e-6);
        assert!((group.responses[1].advantage - 1.0).abs() < 1e-6);
    }

    // ── Test 5: equal rewards → zero advantage ────────────────────────────
    #[test]
    fn test_equal_rewards_zero_advantage() {
        let responses: Vec<GroupResponse> = (0..4)
            .map(|i| make_response(i, vec![-0.5], vec![-0.5], 2.0))
            .collect();
        let mut group = ResponseGroup::new("prompt".into(), responses);
        group.compute_advantages();
        for resp in &group.responses {
            assert_eq!(resp.advantage, 0.0);
        }
    }

    // ── Test 6: best_response ─────────────────────────────────────────────
    #[test]
    fn test_best_response() {
        let r1 = make_response(0, vec![-0.5], vec![-0.5], 0.5);
        let r2 = make_response(1, vec![-0.5], vec![-0.5], 3.0);
        let r3 = make_response(2, vec![-0.5], vec![-0.5], 1.5);
        let group = ResponseGroup::new("prompt".into(), vec![r1, r2, r3]);
        let best = group.best_response().expect("should have best");
        assert_eq!(best.response_id, 1);
        assert!((best.reward - 3.0).abs() < 1e-10);
    }

    // ── Test 7: policy_gradient_loss clipping (ratio > 1+ε) ───────────────
    #[test]
    fn test_pg_loss_clipped_high() {
        let config = GrpoConfig { epsilon: 0.2, ..Default::default() };
        let loss_fn = GrpoLoss::new(config);
        // log_prob = 0, old_log_prob = -2 → ratio = e^2 ≈ 7.39 >> 1.2
        // advantage > 0 so clipping should kick in
        let log_prob = 0.0_f64;
        let old_log_prob = -2.0_f64;
        let advantage = 1.0_f64;
        let ratio = (log_prob - old_log_prob).exp(); // ≈ 7.39
        assert!(ratio > 1.2, "ratio should exceed 1+ε");
        let loss = loss_fn.policy_gradient_loss(log_prob, old_log_prob, advantage);
        // clipped_ratio = 1.2, so loss = -(1.2 * 1.0) = -1.2
        assert!((loss - (-1.2)).abs() < 1e-6, "clipped loss should be -1.2, got {}", loss);
    }

    // ── Test 8: policy_gradient_loss unclipped ────────────────────────────
    #[test]
    fn test_pg_loss_unclipped() {
        let config = GrpoConfig { epsilon: 0.2, ..Default::default() };
        let loss_fn = GrpoLoss::new(config);
        // ratio ≈ 1 (no change), advantage = 0.5
        let log_prob = -1.0_f64;
        let old_log_prob = -1.0_f64;
        let advantage = 0.5_f64;
        let loss = loss_fn.policy_gradient_loss(log_prob, old_log_prob, advantage);
        // ratio = 1.0, unclipped = 1.0 * 0.5 = 0.5, loss = -0.5
        assert!((loss - (-0.5)).abs() < 1e-10, "expected -0.5, got {}", loss);
    }

    // ── Test 9: kl_penalty beta scaling ───────────────────────────────────
    #[test]
    fn test_kl_penalty_beta_scaling() {
        let config = GrpoConfig { beta: 0.05, ..Default::default() };
        let loss_fn = GrpoLoss::new(config);
        let kl = loss_fn.kl_penalty(-1.0, -1.5);
        // 0.05 * (-1.0 - (-1.5)) = 0.05 * 0.5 = 0.025
        assert!((kl - 0.025).abs() < 1e-10, "expected 0.025, got {}", kl);
    }

    // ── Test 10: compute_loss basic ───────────────────────────────────────
    #[test]
    fn test_compute_loss_basic() {
        let config = GrpoConfig::default();
        let mut loss_fn = GrpoLoss::new(config);
        let r1 = make_response(0, vec![-1.0, -1.0], vec![-1.0, -1.0], 1.0);
        let r2 = make_response(1, vec![-1.0, -1.0], vec![-1.0, -1.0], 3.0);
        let mut group = ResponseGroup::new("p".into(), vec![r1, r2]);
        group.compute_advantages();
        let result = loss_fn.compute_loss(&group).expect("should not fail");
        assert!(result.total_loss.is_finite(), "total_loss should be finite");
        assert_eq!(result.response_lengths, vec![2, 2]);
        assert_eq!(loss_fn.step(), 1);
    }

    // ── Test 11: compute_loss clip_fraction ───────────────────────────────
    #[test]
    fn test_compute_loss_clip_fraction() {
        let config = GrpoConfig { epsilon: 0.2, use_kl_penalty: false, ..Default::default() };
        let mut loss_fn = GrpoLoss::new(config);
        // log_prob = 0, ref_log_prob = -3 → ratio = e^3 ≈ 20 >> 1.2 → all clipped
        let r1 = make_response(0, vec![0.0, 0.0], vec![-3.0, -3.0], 2.0);
        let r2 = make_response(1, vec![0.0, 0.0], vec![-3.0, -3.0], 4.0);
        let mut group = ResponseGroup::new("p".into(), vec![r1, r2]);
        group.compute_advantages();
        let result = loss_fn.compute_loss(&group).expect("should not fail");
        // All 4 tokens should be clipped for responses with advantage > 0
        // Response 1 (advantage < 0): ratio clipped below too
        assert!(result.clip_fraction >= 0.0 && result.clip_fraction <= 1.0);
    }

    // ── Test 12: GrpoLossResult display ───────────────────────────────────
    #[test]
    fn test_grpo_loss_result_display() {
        let result = GrpoLossResult {
            total_loss: 1.23,
            policy_gradient_loss: 1.0,
            kl_penalty: 0.23,
            mean_reward: 2.5,
            mean_advantage: 0.0,
            clip_fraction: 0.1,
            response_lengths: vec![10, 12],
        };
        let s = format!("{}", result);
        assert!(s.contains("total_loss"), "display should contain 'total_loss'");
        assert!(s.contains("1.2300"), "display should contain formatted loss");
    }

    // ── Test 13: empty group error ────────────────────────────────────────
    #[test]
    fn test_empty_group_error() {
        let mut loss_fn = GrpoLoss::new(GrpoConfig::default());
        let group = ResponseGroup::new("p".into(), vec![]);
        let err = loss_fn.compute_loss(&group).unwrap_err();
        assert!(matches!(err, GrpoError::EmptyGroup));
    }

    // ── Test 14: config defaults ──────────────────────────────────────────
    #[test]
    fn test_config_defaults() {
        let cfg = GrpoConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.epsilon - 0.2).abs() < 1e-10);
        assert!((cfg.beta - 0.01).abs() < 1e-10);
        assert!((cfg.temperature - 1.0).abs() < 1e-10);
        assert_eq!(cfg.max_new_tokens, 512);
        assert_eq!(cfg.num_iterations, 1000);
        assert!(cfg.advantage_normalization);
        assert!(cfg.use_kl_penalty);
    }

    // ── Test 15: use_kl_penalty=false ────────────────────────────────────
    #[test]
    fn test_no_kl_penalty() {
        let config_kl = GrpoConfig { use_kl_penalty: true, ..Default::default() };
        let config_no_kl = GrpoConfig { use_kl_penalty: false, ..Default::default() };
        let mut loss_kl = GrpoLoss::new(config_kl);
        let mut loss_no_kl = GrpoLoss::new(config_no_kl);

        let build_group = || {
            let r1 = make_response(0, vec![-1.0], vec![-2.0], 1.0);
            let r2 = make_response(1, vec![-1.0], vec![-2.0], 3.0);
            let mut g = ResponseGroup::new("p".into(), vec![r1, r2]);
            g.compute_advantages();
            g
        };

        let res_kl = loss_kl.compute_loss(&build_group()).expect("should not fail");
        let res_no_kl = loss_no_kl.compute_loss(&build_group()).expect("should not fail");

        // With KL penalty, kl term is non-zero; without, it is exactly 0
        assert!((res_no_kl.kl_penalty).abs() < 1e-10, "kl_penalty should be 0.0");
        // The total losses should differ when ref_log_probs differ from log_probs
        // (they do: -1.0 vs -2.0)
        assert!(
            (res_kl.total_loss - res_no_kl.total_loss).abs() > 1e-10,
            "losses should differ when KL penalty is toggled"
        );
    }

    // ── Test 16: compute_group_advantages mean ≈ 0 ───────────────────────
    #[test]
    fn test_compute_group_advantages_mean_zero() {
        let rewards = [1.0f32, 3.0, 5.0];
        let advs = compute_group_advantages(&rewards, 1e-8);
        assert_eq!(advs.len(), 3);
        let mean: f32 = advs.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5, "mean advantage should be ≈ 0, got {mean}");
    }

    // ── Test 17: compute_group_advantages equal rewards → zeros ─────────
    #[test]
    fn test_compute_group_advantages_equal_rewards() {
        let rewards = [2.0f32; 5];
        let advs = compute_group_advantages(&rewards, 1e-8);
        for a in &advs {
            assert!(*a == 0.0, "all-equal rewards → zero advantages");
        }
    }

    // ── Test 18: compute_group_advantages known values ────────────────────
    #[test]
    fn test_compute_group_advantages_known_values() {
        // rewards = [0, 2] → mean=1, std=1 → advantages = [-1, 1]
        let rewards = [0.0f32, 2.0];
        let advs = compute_group_advantages(&rewards, 1e-8);
        assert!((advs[0] - (-1.0f32)).abs() < 1e-5, "expected -1, got {}", advs[0]);
        assert!((advs[1] - 1.0f32).abs() < 1e-5, "expected 1, got {}", advs[1]);
    }

    // ── Test 19: compute_grpo_loss unclipped positive advantage ──────────
    #[test]
    fn test_compute_grpo_loss_unclipped() {
        // ratio = exp(0) = 1, advantage = 1.0 → policy_loss = -1.0
        let log_probs = [0.0f32];
        let ref_log_probs = [0.0f32];
        let advantages = [1.0f32];
        let config = GrpoLossConfig { clip_range: 0.2, kl_coef: 0.0, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config)
            .expect("should not fail");
        // loss = -min(1*1, clip(1,0.8,1.2)*1) = -min(1,1) = -1
        assert!((result.policy_loss - (-1.0f32)).abs() < 1e-5, "expected -1.0, got {}", result.policy_loss);
    }

    // ── Test 20: compute_grpo_loss clipped ratio ──────────────────────────
    #[test]
    fn test_compute_grpo_loss_clipped_ratio() {
        // log_prob=0, ref=-5 → ratio=e^5 ≈148 >> 1.2 → clipped at 1.2
        // advantage=1 → L_clip = 1.2, policy_loss = -1.2
        let log_probs = [0.0f32];
        let ref_log_probs = [-5.0f32];
        let advantages = [1.0f32];
        let config = GrpoLossConfig { clip_range: 0.2, kl_coef: 0.0, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config)
            .expect("should not fail");
        assert!((result.policy_loss - (-1.2f32)).abs() < 1e-5, "clipped policy_loss expected -1.2, got {}", result.policy_loss);
    }

    // ── Test 21: compute_grpo_loss KL component ───────────────────────────
    #[test]
    fn test_compute_grpo_loss_kl_component() {
        // log_prob=-1, ref_log_prob=-2 → kl_per_token = 1
        // kl_loss = kl_coef * 1 = 0.05
        let log_probs = [-1.0f32];
        let ref_log_probs = [-2.0f32];
        let advantages = [0.0f32];
        let config = GrpoLossConfig { kl_coef: 0.05, clip_range: 0.2, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config)
            .expect("should not fail");
        assert!((result.kl_loss - 0.05f32).abs() < 1e-5, "kl_loss expected 0.05, got {}", result.kl_loss);
    }

    // ── Test 22: compute_grpo_loss total_loss = policy + kl ─────────────
    #[test]
    fn test_compute_grpo_loss_total_equals_sum() {
        let log_probs = [-1.0f32, -1.5];
        let ref_log_probs = [-1.0f32, -1.0];
        let advantages = [0.5f32, -0.5];
        let config = GrpoLossConfig::default();
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config)
            .expect("should not fail");
        let expected = result.policy_loss + result.kl_loss;
        assert!((result.total_loss - expected).abs() < 1e-5,
            "total_loss should equal policy_loss + kl_loss");
    }

    // ── Test 23: GrpoSample construction ─────────────────────────────────
    #[test]
    fn test_grpo_sample_new() {
        let sample = GrpoSample::new(
            vec![1, 2, 3],
            vec![vec![4, 5], vec![6, 7]],
            vec![1.0f32, 2.0],
        ).expect("should construct");
        assert_eq!(sample.group_size(), 2);
        assert_eq!(sample.prompt.len(), 3);
    }

    // ── Test 24: GrpoSample length mismatch error ─────────────────────────
    #[test]
    fn test_grpo_sample_length_mismatch() {
        let err = GrpoSample::new(
            vec![1],
            vec![vec![2, 3], vec![4, 5]],
            vec![1.0f32],  // only 1 reward for 2 completions
        ).unwrap_err();
        assert!(matches!(err, GrpoError::LengthMismatch(_)));
    }

    // ── Test 25: GrpoTrainer produces stats ───────────────────────────────
    #[test]
    fn test_grpo_trainer_train_step() {
        let config = GrpoLossConfig::default();
        let mut trainer = GrpoTrainer::new(config);

        let samples = vec![
            GrpoSample::new(vec![1], vec![vec![2], vec![3]], vec![1.0f32, 3.0])
                .expect("ok"),
        ];
        let log_probs = vec![vec![-1.0f32, -1.0]];
        let ref_log_probs = vec![vec![-1.0f32, -1.5]];

        let stats = trainer.train_step(&samples, &log_probs, &ref_log_probs)
            .expect("should not fail");

        assert_eq!(stats.iteration, 0);
        assert!(stats.mean_reward.is_finite());
        assert_eq!(trainer.iteration(), 1);
        assert_eq!(trainer.history().len(), 1);
    }

    // ── Test 26: GrpoStats display ────────────────────────────────────────
    #[test]
    fn test_grpo_stats_display() {
        let stats = GrpoStats {
            iteration: 5,
            mean_reward: 1.23,
            std_reward: 0.45,
            policy_loss: -0.67,
            kl_divergence: 0.01,
            clip_fraction: 0.15,
        };
        let s = format!("{stats}");
        assert!(s.contains("iter=5"), "display should contain iteration");
        assert!(s.contains("1.2300"), "display should contain mean_reward");
    }

    // ── Test 27: compute_grpo_loss empty slice error ──────────────────────
    #[test]
    fn test_compute_grpo_loss_empty_error() {
        let config = GrpoLossConfig::default();
        let err = compute_grpo_loss(&[], &[], &[], &config).unwrap_err();
        assert!(matches!(err, GrpoError::EmptyGroup));
    }

    // ── Test 28: gradient direction (lower log_prob for positive adv) ──────
    #[test]
    fn test_grpo_gradient_direction_positive_advantage() {
        // positive advantage → loss should decrease as log_prob increases toward ref
        // ratio > 1 and advantage > 0 → would be clipped → loss = -(1+ε)*adv
        let lp_high = [0.5f32];
        let lp_low  = [-0.5f32];
        let ref_lp  = [0.0f32];
        let advs    = [1.0f32];
        let config = GrpoLossConfig { clip_range: 0.2, kl_coef: 0.0, ..Default::default() };

        let loss_high = compute_grpo_loss(&lp_high, &ref_lp, &advs, &config)
            .expect("ok").total_loss;
        let loss_low = compute_grpo_loss(&lp_low, &ref_lp, &advs, &config)
            .expect("ok").total_loss;

        // lp_high is further above ref → more clipped → same clipped loss; but
        // lp_low has ratio < 1, so unclipped. The high-ratio case is clipped more:
        assert!(
            loss_high <= loss_low + 1e-4,
            "higher ratio with positive adv should have ≤ loss (clipped): high={loss_high} low={loss_low}"
        );
    }

    // ── New extended tests ────────────────────────────────────────────────────

    // Test: group advantage normalization — mean ≈ 0, std ≈ 1 after normalization
    #[test]
    fn test_group_advantage_normalization_mean_zero_std_one() {
        let rewards = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let advs = compute_group_advantages(&rewards, 1e-8);
        let n = advs.len() as f32;
        let mean = advs.iter().sum::<f32>() / n;
        let var = advs.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n;
        let std = var.sqrt();
        assert!(mean.abs() < 1e-4, "mean advantage should be ≈ 0, got {mean}");
        // std is close to 1 (the normalization denominator is std + eps, so not exactly 1)
        assert!(std > 0.9 && std <= 1.0 + 1e-3, "std should be ≈ 1, got {std}");
    }

    // Test: GRPO loss with all same rewards — std=0 edge case → advantages all zero
    #[test]
    fn test_grpo_all_same_rewards_std_zero_edge_case() {
        let rewards = [2.0f32; 6];
        let advs = compute_group_advantages(&rewards, 1e-8);
        for a in &advs {
            assert!(*a == 0.0, "all-equal rewards → zero advantages, got {a}");
        }
        // Loss should still work (just all zero advantages)
        let log_probs = vec![-1.0f32; 6];
        let ref_log_probs = vec![-1.0f32; 6];
        let config = GrpoLossConfig { kl_coef: 0.0, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advs, &config).expect("ok");
        assert!(result.policy_loss.is_finite(), "loss should be finite even with zero advantages");
    }

    // Test: clip fraction — ratio outside [1-ε, 1+ε]
    #[test]
    fn test_clip_fraction_ratio_outside_bounds() {
        // log_prob much higher than ref → ratio >> 1+ε → should be clipped
        let log_probs = vec![5.0f32; 4];    // ratio = e^10 >> 1.2
        let ref_log_probs = vec![-5.0f32; 4];
        let advantages = vec![1.0f32; 4];
        let config = GrpoLossConfig { clip_range: 0.2, kl_coef: 0.0, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config).expect("ok");
        // All tokens have ratio >> 1.2, all should be clipped
        // The clipped loss should equal -(1+eps)*adv for all tokens
        let expected_pg = -(1.0 + 0.2) * 1.0_f32; // -(1+ε)*A
        assert!((result.policy_loss - expected_pg).abs() < 1e-4,
            "fully clipped policy loss should be -(1+ε)*A={expected_pg}, got {}", result.policy_loss);
    }

    // Test: KL divergence component is 0 when log_probs = ref_log_probs
    #[test]
    fn test_kl_divergence_zero_when_identical_log_probs() {
        let log_probs = vec![-1.0f32, -2.0, -0.5];
        let ref_log_probs = log_probs.clone();
        let advantages = vec![1.0f32, -0.5, 0.3];
        let config = GrpoLossConfig { kl_coef: 1.0, clip_range: 0.2, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config).expect("ok");
        assert!(result.kl_loss.abs() < 1e-6, "KL should be 0 when policy = ref, got {}", result.kl_loss);
    }

    // Test: total loss = policy_loss + β * kl_loss
    #[test]
    fn test_total_loss_equals_policy_plus_beta_kl() {
        let log_probs = vec![-1.0f32, -0.5, -1.5];
        let ref_log_probs = vec![-1.5f32, -1.0, -2.0];
        let advantages = vec![0.5f32, -0.3, 1.0];
        let config = GrpoLossConfig { kl_coef: 0.05, clip_range: 0.2, ..Default::default() };
        let result = compute_grpo_loss(&log_probs, &ref_log_probs, &advantages, &config).expect("ok");
        let expected_total = result.policy_loss + result.kl_loss;
        assert!((result.total_loss - expected_total).abs() < 1e-5,
            "total_loss should equal policy_loss + kl_loss: {} vs {}", result.total_loss, expected_total);
    }

    // Test: group_size=1 degenerate case — single response, advantage=0
    #[test]
    fn test_grpo_group_size_one_degenerate() {
        let rewards = [3.0f32];
        let advs = compute_group_advantages(&rewards, 1e-8);
        assert_eq!(advs.len(), 1);
        assert_eq!(advs[0], 0.0, "single response has std=0, advantage should be 0");
    }

    // Test: GrpoSample construction and group_size()
    #[test]
    fn test_grpo_sample_construction_and_group_size() {
        let sample = GrpoSample::new(
            vec![1_u32, 2, 3],
            vec![vec![4_u32, 5], vec![6_u32, 7], vec![8_u32, 9]],
            vec![1.0f32, 2.0, 3.0],
        ).expect("ok");
        assert_eq!(sample.group_size(), 3);
        assert_eq!(sample.prompt.len(), 3);
        assert_eq!(sample.completions.len(), 3);
        assert_eq!(sample.rewards.len(), 3);
    }

    // Test: multi-group batch — verify group independence (each group normalized independently)
    #[test]
    fn test_multi_group_batch_group_independence() {
        // Two groups with different reward distributions should have independent advantages
        let rewards_g1 = [1.0f32, 3.0]; // mean=2, std=1
        let rewards_g2 = [10.0f32, 20.0]; // mean=15, std=5
        let advs_g1 = compute_group_advantages(&rewards_g1, 1e-8);
        let advs_g2 = compute_group_advantages(&rewards_g2, 1e-8);
        // Group 1 advantage for lower reward = -1
        assert!((advs_g1[0] - (-1.0_f32)).abs() < 1e-4, "g1 low adv={}", advs_g1[0]);
        // Group 2 advantage for lower reward = -1
        assert!((advs_g2[0] - (-1.0_f32)).abs() < 1e-4, "g2 low adv={}", advs_g2[0]);
        // Cross-group: g2 rewards don't affect g1 normalization
        assert!((advs_g1[1] - 1.0_f32).abs() < 1e-4, "g1 high adv should be 1.0");
    }

    // Test: advantage direction — positive reward → positive advantage relative to mean
    #[test]
    fn test_advantage_direction_positive_reward() {
        let rewards = [0.0f32, 5.0, 10.0]; // mean=5
        let advs = compute_group_advantages(&rewards, 1e-8);
        // reward=10 > mean=5 → positive advantage
        assert!(advs[2] > 0.0, "above-mean reward should have positive advantage, got {}", advs[2]);
        // reward=0 < mean=5 → negative advantage
        assert!(advs[0] < 0.0, "below-mean reward should have negative advantage, got {}", advs[0]);
        // reward=5 = mean → zero advantage
        assert!(advs[1].abs() < 1e-5, "at-mean reward should have ≈0 advantage, got {}", advs[1]);
    }

    // Test: trainer history accumulation — multiple train_step calls
    #[test]
    fn test_grpo_trainer_history_accumulation() {
        let config = GrpoLossConfig::default();
        let mut trainer = GrpoTrainer::new(config);

        for _ in 0..3 {
            let samples = vec![
                GrpoSample::new(vec![1], vec![vec![2], vec![3]], vec![1.0f32, 2.0]).expect("ok"),
            ];
            let log_probs = vec![vec![-1.0f32, -1.0]];
            let ref_log_probs = vec![vec![-1.0f32, -1.0]];
            trainer.train_step(&samples, &log_probs, &ref_log_probs).expect("ok");
        }

        assert_eq!(trainer.history().len(), 3, "should have 3 entries");
        assert_eq!(trainer.iteration(), 3, "iteration should be 3");
        for (i, stat) in trainer.history().iter().enumerate() {
            assert_eq!(stat.iteration, i, "iteration index mismatch at {i}");
        }
    }
}
