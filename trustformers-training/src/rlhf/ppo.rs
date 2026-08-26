//! Proximal Policy Optimization (PPO) implementation for language models.

use crate::rlhf::{PPOConfig, RLHFMetrics};
use anyhow::{anyhow, Result};
use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use std::collections::HashMap;

/// PPO trainer for language models
#[derive(Debug)]
pub struct PPOTrainer {
    config: PPOConfig,
    policy_model: Option<PolicyModel>,
    value_model: Option<ValueModel>,
    reference_model: Option<PolicyModel>,
    optimizer: PPOOptimizer,
    statistics: PPOStatistics,
}

/// Key of the token-embedding matrix, shape `[vocab_size, hidden_size]`.
pub const PARAM_EMBEDDING: &str = "embedding";
/// Key of the optional hidden transform, shape `[hidden_size, hidden_size]`.
pub const PARAM_HIDDEN_WEIGHT: &str = "hidden.weight";
/// Key of the output projection, shape `[hidden_size, vocab_size]`.
pub const PARAM_OUTPUT_WEIGHT: &str = "output.weight";
/// Key of the optional output bias, shape `[1, vocab_size]`.
pub const PARAM_OUTPUT_BIAS: &str = "output.bias";
/// Key of the value head, shape `[hidden_size, 1]`.
pub const PARAM_VALUE_HEAD: &str = "value_head";
/// Key of the optional value-head bias, shape `[1, 1]`.
pub const PARAM_VALUE_BIAS: &str = "value_head.bias";

/// Recency decay used when pooling a token sequence into a single context vector.
pub(crate) const CONTEXT_DECAY: f32 = 0.9;

/// Deterministic 64-bit mixer (splitmix64) used to seed parameter initialisation.
///
/// A PRNG with an explicit seed keeps model initialisation reproducible across runs, which
/// is what makes "two different weight sets produce different logits" a testable property.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw a uniform sample in `(-limit, limit)` from the given PRNG state.
fn uniform_symmetric(state: &mut u64, limit: f32) -> f32 {
    let bits = splitmix64(state);
    // 53-bit mantissa -> [0, 1)
    let unit = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
    ((unit as f32) * 2.0 - 1.0) * limit
}

/// Build a Xavier-uniform matrix of the requested shape from a deterministic seed.
fn xavier_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let limit = (6.0f32 / (rows as f32 + cols as f32)).sqrt();
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| uniform_symmetric(&mut state, limit))
}

/// Exponentially-decayed causal pooling of a token sequence into one context vector.
///
/// `ctx = Σ_i decay^(n-1-i) · E[t_i] / Σ_i decay^(n-1-i)`
///
/// The most recent token therefore carries the largest weight, which is the causal-LM prior
/// this lightweight policy encodes. Every component of the result depends on the *actual*
/// embedding rows, so changing the weights changes the context.
pub(crate) fn pool_context(
    embedding: &Array2<f32>,
    sequence: &[u32],
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if sequence.is_empty() {
        return Err(anyhow!("cannot encode an empty token sequence"));
    }
    let vocab_size = embedding.nrows();
    let mut ctx = vec![0.0f32; hidden_size];
    let mut weight_sum = 0.0f32;
    let n = sequence.len();
    for (i, &token) in sequence.iter().enumerate() {
        let token_idx = token as usize;
        if token_idx >= vocab_size {
            return Err(anyhow!(
                "token id {token_idx} is outside the model vocabulary [0, {vocab_size})"
            ));
        }
        let w = CONTEXT_DECAY.powi((n - 1 - i) as i32);
        weight_sum += w;
        for j in 0..hidden_size {
            ctx[j] += w * embedding[[token_idx, j]];
        }
    }
    if weight_sum > 0.0 {
        for value in ctx.iter_mut() {
            *value /= weight_sum;
        }
    }
    Ok(ctx)
}

/// Fetch a parameter matrix, validating its shape.
fn require_param<'a>(
    parameters: &'a HashMap<String, Array2<f32>>,
    key: &str,
    expected: (usize, usize),
    model_id: &str,
) -> Result<&'a Array2<f32>> {
    let matrix = parameters.get(key).ok_or_else(|| {
        anyhow!(
            "model '{model_id}' is missing the required parameter '{key}' \
             (expected shape {:?}); initialise it with the model's constructor or load \
             pretrained weights before running a forward pass",
            expected
        )
    })?;
    if matrix.dim() != expected {
        return Err(anyhow!(
            "model '{model_id}' parameter '{key}' has shape {:?}, expected {:?}",
            matrix.dim(),
            expected
        ));
    }
    Ok(matrix)
}

/// Policy model for language generation.
///
/// # Architecture
///
/// A single-layer causal bag-of-context language model, fully specified by `parameters`:
///
/// | key | shape | role |
/// |-----|-------|------|
/// | [`PARAM_EMBEDDING`] | `[vocab_size, hidden_size]` | token embeddings |
/// | [`PARAM_HIDDEN_WEIGHT`] (optional) | `[hidden_size, hidden_size]` | `tanh` hidden transform |
/// | [`PARAM_OUTPUT_WEIGHT`] | `[hidden_size, vocab_size]` | output projection |
/// | [`PARAM_OUTPUT_BIAS`] (optional) | `[1, vocab_size]` | output bias |
///
/// `logits(seq) = tanh(pool(seq) · W_hidden) · W_out + b`, where `pool` is the recency-decayed
/// average of the embedded tokens. Every logit is a function of the stored weights — there is
/// no closed-form fallback — so a model with no parameters returns an error rather than a
/// fabricated distribution.
#[derive(Debug, Clone)]
pub struct PolicyModel {
    /// Model identifier
    pub model_id: String,
    /// Parameter matrices keyed by the `PARAM_*` constants.
    pub parameters: HashMap<String, Array2<f32>>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl PolicyModel {
    /// Create a policy with Xavier-uniform parameters drawn from `seed`.
    ///
    /// Two different seeds produce two genuinely different models.
    pub fn new_initialized(
        model_id: impl Into<String>,
        vocab_size: usize,
        hidden_size: usize,
        seed: u64,
    ) -> Result<Self> {
        if vocab_size == 0 || hidden_size == 0 {
            return Err(anyhow!(
                "vocab_size and hidden_size must both be non-zero (got {vocab_size}, {hidden_size})"
            ));
        }
        let mut parameters = HashMap::new();
        parameters.insert(
            PARAM_EMBEDDING.to_string(),
            xavier_matrix(vocab_size, hidden_size, seed),
        );
        parameters.insert(
            PARAM_HIDDEN_WEIGHT.to_string(),
            xavier_matrix(hidden_size, hidden_size, seed ^ 0x5DEE_CE66),
        );
        parameters.insert(
            PARAM_OUTPUT_WEIGHT.to_string(),
            xavier_matrix(hidden_size, vocab_size, seed ^ 0x1234_5678_9ABC),
        );
        parameters.insert(
            PARAM_OUTPUT_BIAS.to_string(),
            Array2::zeros((1, vocab_size)),
        );
        Ok(Self {
            model_id: model_id.into(),
            parameters,
            vocab_size,
            hidden_size,
        })
    }

    /// Encode a token sequence into the model's hidden representation.
    pub fn encode(&self, sequence: &[u32]) -> Result<Vec<f32>> {
        let embedding = require_param(
            &self.parameters,
            PARAM_EMBEDDING,
            (self.vocab_size, self.hidden_size),
            &self.model_id,
        )?;
        let ctx = pool_context(embedding, sequence, self.hidden_size)?;

        match self.parameters.get(PARAM_HIDDEN_WEIGHT) {
            Some(hidden) => {
                if hidden.dim() != (self.hidden_size, self.hidden_size) {
                    return Err(anyhow!(
                        "model '{}' parameter '{PARAM_HIDDEN_WEIGHT}' has shape {:?}, expected {:?}",
                        self.model_id,
                        hidden.dim(),
                        (self.hidden_size, self.hidden_size)
                    ));
                }
                let mut out = vec![0.0f32; self.hidden_size];
                for j in 0..self.hidden_size {
                    let mut acc = 0.0f32;
                    for k in 0..self.hidden_size {
                        acc += ctx[k] * hidden[[k, j]];
                    }
                    out[j] = acc.tanh();
                }
                Ok(out)
            },
            None => Ok(ctx),
        }
    }

    /// Compute the next-token logits for `sequence` over the **full** vocabulary.
    pub fn logits(&self, sequence: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.encode(sequence)?;
        let output = require_param(
            &self.parameters,
            PARAM_OUTPUT_WEIGHT,
            (self.hidden_size, self.vocab_size),
            &self.model_id,
        )?;

        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..self.hidden_size {
                acc += hidden[k] * output[[k, v]];
            }
            *logit = acc;
        }

        if let Some(bias) = self.parameters.get(PARAM_OUTPUT_BIAS) {
            if bias.dim() != (1, self.vocab_size) {
                return Err(anyhow!(
                    "model '{}' parameter '{PARAM_OUTPUT_BIAS}' has shape {:?}, expected {:?}",
                    self.model_id,
                    bias.dim(),
                    (1, self.vocab_size)
                ));
            }
            for (v, logit) in logits.iter_mut().enumerate() {
                *logit += bias[[0, v]];
            }
        }

        Ok(logits)
    }
}

/// Value model for estimating state values.
///
/// # Architecture
///
/// | key | shape | role |
/// |-----|-------|------|
/// | [`PARAM_EMBEDDING`] | `[vocab_size, hidden_size]` | token embeddings |
/// | [`PARAM_VALUE_HEAD`] | `[hidden_size, 1]` | scalar projection |
/// | [`PARAM_VALUE_BIAS`] (optional) | `[1, 1]` | bias |
///
/// `value(seq) = tanh(pool(seq) · w_v + b_v)`, bounded to `(-1, 1)`.
#[derive(Debug, Clone)]
pub struct ValueModel {
    /// Model identifier
    pub model_id: String,
    /// Parameter matrices keyed by the `PARAM_*` constants.
    pub parameters: HashMap<String, Array2<f32>>,
    /// Hidden size
    pub hidden_size: usize,
}

impl ValueModel {
    /// Create a value model with Xavier-uniform parameters drawn from `seed`.
    pub fn new_initialized(
        model_id: impl Into<String>,
        vocab_size: usize,
        hidden_size: usize,
        seed: u64,
    ) -> Result<Self> {
        if vocab_size == 0 || hidden_size == 0 {
            return Err(anyhow!(
                "vocab_size and hidden_size must both be non-zero (got {vocab_size}, {hidden_size})"
            ));
        }
        let mut parameters = HashMap::new();
        parameters.insert(
            PARAM_EMBEDDING.to_string(),
            xavier_matrix(vocab_size, hidden_size, seed ^ 0xA5A5_A5A5),
        );
        parameters.insert(
            PARAM_VALUE_HEAD.to_string(),
            xavier_matrix(hidden_size, 1, seed ^ 0x5A5A_5A5A),
        );
        parameters.insert(PARAM_VALUE_BIAS.to_string(), Array2::zeros((1, 1)));
        Ok(Self {
            model_id: model_id.into(),
            parameters,
            hidden_size,
        })
    }

    /// Estimate the value of a token sequence, in `(-1, 1)`.
    pub fn value(&self, sequence: &[u32]) -> Result<f32> {
        let embedding = self.parameters.get(PARAM_EMBEDDING).ok_or_else(|| {
            anyhow!(
                "value model '{}' is missing the required parameter '{PARAM_EMBEDDING}'",
                self.model_id
            )
        })?;
        if embedding.ncols() != self.hidden_size {
            return Err(anyhow!(
                "value model '{}' parameter '{PARAM_EMBEDDING}' has {} columns, expected {}",
                self.model_id,
                embedding.ncols(),
                self.hidden_size
            ));
        }
        let ctx = pool_context(embedding, sequence, self.hidden_size)?;

        let head = require_param(
            &self.parameters,
            PARAM_VALUE_HEAD,
            (self.hidden_size, 1),
            &self.model_id,
        )?;
        let mut value = 0.0f32;
        for k in 0..self.hidden_size {
            value += ctx[k] * head[[k, 0]];
        }
        if let Some(bias) = self.parameters.get(PARAM_VALUE_BIAS) {
            if bias.dim() != (1, 1) {
                return Err(anyhow!(
                    "value model '{}' parameter '{PARAM_VALUE_BIAS}' has shape {:?}, expected (1, 1)",
                    self.model_id,
                    bias.dim()
                ));
            }
            value += bias[[0, 0]];
        }
        Ok(value.tanh())
    }
}

/// PPO-specific optimizer
#[derive(Debug)]
pub struct PPOOptimizer {
    /// Learning rate for policy
    pub policy_lr: f64,
    /// Learning rate for value function
    pub value_lr: f64,
    /// Momentum parameter
    pub momentum: f64,
    /// Weight decay
    pub weight_decay: f64,
}

/// PPO training statistics
#[derive(Debug, Default)]
pub struct PPOStatistics {
    /// Total number of steps
    pub total_steps: usize,
    /// Policy losses over time
    pub policy_losses: Vec<f32>,
    /// Value losses over time
    pub value_losses: Vec<f32>,
    /// KL divergences over time
    pub kl_divergences: Vec<f32>,
    /// Rewards over time
    pub rewards: Vec<f32>,
    /// Advantage estimates
    pub advantages: Vec<f32>,
    /// Clip fractions
    pub clip_fractions: Vec<f32>,
}

/// Experience batch for PPO training
#[derive(Debug, Clone)]
pub struct ExperienceBatch {
    /// Input sequences (batch_size, seq_len)
    pub sequences: Array2<u32>,
    /// Action probabilities from current policy
    pub action_probs: Array2<f32>,
    /// Action probabilities from old policy
    pub old_action_probs: Array2<f32>,
    /// Rewards for each sequence
    pub rewards: Array1<f32>,
    /// Value estimates
    pub values: Array1<f32>,
    /// Advantage estimates
    pub advantages: Array1<f32>,
    /// Returns (rewards-to-go)
    pub returns: Array1<f32>,
}

/// PPO training step result
#[derive(Debug, Clone)]
pub struct PPOStepResult {
    /// Policy loss
    pub policy_loss: f32,
    /// Value loss
    pub value_loss: f32,
    /// KL divergence
    pub kl_divergence: f32,
    /// Entropy
    pub entropy: f32,
    /// Clip fraction
    pub clip_fraction: f32,
    /// Explained variance
    pub explained_variance: f32,
}

impl PPOTrainer {
    /// Create a new PPO trainer
    pub fn new(config: PPOConfig) -> Result<Self> {
        let optimizer = PPOOptimizer {
            policy_lr: config.policy_lr,
            value_lr: config.value_lr,
            momentum: 0.9,
            weight_decay: 0.01,
        };

        Ok(Self {
            config,
            policy_model: None,
            value_model: None,
            reference_model: None,
            optimizer,
            statistics: PPOStatistics::default(),
        })
    }

    /// Initialize models
    pub fn initialize_models(
        &mut self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        reference_model: Option<PolicyModel>,
    ) -> Result<()> {
        self.policy_model = Some(policy_model);
        self.value_model = Some(value_model);
        self.reference_model = reference_model;
        Ok(())
    }

    /// Generate responses using the policy model
    pub fn generate_responses(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> Result<Vec<GenerationResult>> {
        let policy_model = self
            .policy_model
            .as_ref()
            .ok_or_else(|| anyhow!("Policy model not initialized"))?;

        let value_model = self
            .value_model
            .as_ref()
            .ok_or_else(|| anyhow!("Value model not initialized"))?;

        let mut results = Vec::new();

        for prompt in prompts {
            // Tokenize the prompt
            let prompt_tokens = self.tokenize(prompt)?;

            // Generate tokens using the policy model
            let generated_tokens =
                self.sample_tokens_with_model(policy_model, &prompt_tokens, max_length)?;

            // Create full sequence (prompt + generated)
            let full_sequence = [prompt_tokens.clone(), generated_tokens.clone()].concat();

            // Detokenize the generated part
            let response = self.detokenize(&generated_tokens)?;

            // Calculate log probabilities for the generated tokens
            let log_probs = self.calculate_log_probs_with_model(
                policy_model,
                &prompt_tokens,
                &generated_tokens,
            )?;

            // Calculate value estimate for the full sequence
            let value = self.calculate_value_with_model(value_model, &full_sequence)?;

            results.push(GenerationResult {
                prompt: prompt.clone(),
                response,
                tokens: generated_tokens,
                log_probs,
                value,
            });
        }

        Ok(results)
    }

    /// Calculate advantages using Generalized Advantage Estimation (GAE)
    pub fn calculate_advantages(
        &self,
        rewards: &Array1<f32>,
        values: &Array1<f32>,
        gamma: f32,
        lambda: f32,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let n = rewards.len();
        let mut advantages = Array1::zeros(n);
        let mut returns = Array1::zeros(n);

        let mut gae = 0.0f32;

        // Calculate GAE backwards
        for i in (0..n).rev() {
            let delta = if i == n - 1 {
                rewards[i] - values[i]
            } else {
                rewards[i] + (gamma * values[i + 1]) - values[i]
            };

            gae = delta + (gamma * lambda * gae);
            advantages[i] = gae;
            returns[i] = advantages[i] + values[i];
        }

        Ok((advantages, returns))
    }

    /// Perform a PPO training step
    pub fn training_step(&mut self, batch: &ExperienceBatch) -> Result<PPOStepResult> {
        // Calculate policy loss with clipping
        let policy_loss = self.calculate_policy_loss(batch)?;

        // Calculate value loss
        let value_loss = self.calculate_value_loss(batch)?;

        // Calculate KL divergence
        let kl_divergence = self.calculate_kl_divergence(batch)?;

        // Calculate entropy
        let entropy = self.calculate_entropy(batch)?;

        // Calculate clip fraction
        let clip_fraction = self.calculate_clip_fraction(batch)?;

        // Calculate explained variance
        let explained_variance = self.calculate_explained_variance(batch)?;

        // Update statistics
        self.statistics.total_steps += 1;
        self.statistics.policy_losses.push(policy_loss);
        self.statistics.value_losses.push(value_loss);
        self.statistics.kl_divergences.push(kl_divergence);
        self.statistics.clip_fractions.push(clip_fraction);

        Ok(PPOStepResult {
            policy_loss,
            value_loss,
            kl_divergence,
            entropy,
            clip_fraction,
            explained_variance,
        })
    }

    /// Calculate policy loss with PPO clipping
    fn calculate_policy_loss(&self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.advantages.len();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Calculate probability ratio
            let ratio = batch.action_probs[[i, 0]] / batch.old_action_probs[[i, 0]];

            // Clipped objective
            let clip_ratio = ratio.clamp(
                1.0 - self.config.clip_param as f32,
                1.0 + self.config.clip_param as f32,
            );

            let obj1 = ratio * batch.advantages[i];
            let obj2 = clip_ratio * batch.advantages[i];

            total_loss -= obj1.min(obj2);
        }

        Ok(total_loss / batch_size as f32)
    }

    /// Calculate value function loss
    fn calculate_value_loss(&self, batch: &ExperienceBatch) -> Result<f32> {
        let mut total_loss = 0.0;
        let batch_size = batch.values.len();

        for i in 0..batch_size {
            let value_loss = (batch.values[i] - batch.returns[i]).powi(2);
            total_loss += value_loss;
        }

        Ok(total_loss / batch_size as f32)
    }

    /// Calculate KL divergence between current and old policy
    fn calculate_kl_divergence(&self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.action_probs.shape()[0];
        let mut total_kl = 0.0;

        for i in 0..batch_size {
            let old_prob = batch.old_action_probs[[i, 0]];
            let new_prob = batch.action_probs[[i, 0]];

            if old_prob > 0.0 && new_prob > 0.0 {
                total_kl += old_prob * (old_prob / new_prob).ln();
            }
        }

        Ok(total_kl / batch_size as f32)
    }

    /// Calculate entropy of the policy
    fn calculate_entropy(&self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.action_probs.shape()[0];
        let mut total_entropy = 0.0;

        for i in 0..batch_size {
            let prob = batch.action_probs[[i, 0]];
            if prob > 0.0 {
                total_entropy -= prob * prob.ln();
            }
        }

        Ok(total_entropy / batch_size as f32)
    }

    /// Calculate fraction of clipped samples
    fn calculate_clip_fraction(&self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.advantages.len();
        let mut clipped_count = 0;

        for i in 0..batch_size {
            let ratio = batch.action_probs[[i, 0]] / batch.old_action_probs[[i, 0]];

            if ratio < (1.0 - self.config.clip_param as f32)
                || ratio > (1.0 + self.config.clip_param as f32)
            {
                clipped_count += 1;
            }
        }

        Ok(clipped_count as f32 / batch_size as f32)
    }

    /// Calculate explained variance of the value function
    fn calculate_explained_variance(&self, batch: &ExperienceBatch) -> Result<f32> {
        let y_true = &batch.returns;
        let y_pred = &batch.values;

        let y_true_mean = y_true.mean().unwrap_or(0.0);
        let y_pred_mean = y_pred.mean().unwrap_or(0.0);

        let mut var_y = 0.0;
        let mut var_pred = 0.0;

        for i in 0..y_true.len() {
            var_y += (y_true[i] - y_true_mean).powi(2);
            var_pred += (y_pred[i] - y_pred_mean).powi(2);
        }

        if var_y == 0.0 {
            return Ok(0.0);
        }

        Ok(1.0 - (var_pred / var_y))
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> &PPOStatistics {
        &self.statistics
    }

    /// Convert training statistics to RLHF metrics
    pub fn to_rlhf_metrics(&self) -> RLHFMetrics {
        let latest_policy_loss = self.statistics.policy_losses.last().copied();
        let latest_value_loss = self.statistics.value_losses.last().copied();
        let latest_kl = self.statistics.kl_divergences.last().copied().unwrap_or(0.0);
        let avg_reward = self.statistics.rewards.iter().sum::<f32>()
            / self.statistics.rewards.len().max(1) as f32;

        RLHFMetrics {
            phase: crate::rlhf::RLHFPhase::PPO,
            policy_loss: latest_policy_loss,
            value_loss: latest_value_loss,
            reward_accuracy: None,
            kl_divergence: latest_kl,
            avg_reward,
            ppo_objective: latest_policy_loss,
            advantages: self.statistics.advantages.clone(),
            response_lengths: Vec::new(),
            constitutional_violations: None,
        }
    }

    // Helper methods (proper implementations)

    /// Tokenize text using a simple whitespace tokenizer with special tokens
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let mut token_map = HashMap::new();
        let mut current_id = 0u32;

        // Add special tokens
        token_map.insert("<pad>".to_string(), current_id);
        current_id += 1;
        token_map.insert("<bos>".to_string(), current_id);
        current_id += 1;
        token_map.insert("<eos>".to_string(), current_id);
        current_id += 1;
        token_map.insert("<unk>".to_string(), current_id);

        // Hashing tokenizer: ids must stay inside the policy model's vocabulary, otherwise
        // the embedding lookup in `PolicyModel::encode` would be out of bounds.
        const NUM_SPECIAL_TOKENS: u32 = 4;
        let vocab_size = self
            .policy_model
            .as_ref()
            .map(|m| m.vocab_size as u32)
            .unwrap_or(50_000)
            .max(NUM_SPECIAL_TOKENS + 1);
        let hash_space = vocab_size - NUM_SPECIAL_TOKENS;

        // Simple word-based tokenization
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();
        tokens.push(1); // <bos>

        for word in words {
            if let Some(&id) = token_map.get(word) {
                tokens.push(id);
            } else {
                // Simple hash-based ID generation for unknown words
                let id = (word.len() as u32)
                    .wrapping_mul(31)
                    .wrapping_add(word.chars().map(|c| c as u32).fold(0u32, u32::wrapping_add))
                    % hash_space
                    + NUM_SPECIAL_TOKENS;
                tokens.push(id);
            }
        }

        tokens.push(2); // <eos>
        Ok(tokens)
    }

    /// Detokenize tokens back to text
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &token in tokens {
            match token {
                0 => result.push_str("<pad>"),
                1 => result.push_str("<bos>"),
                2 => result.push_str("<eos>"),
                3 => result.push_str("<unk>"),
                _ => {
                    // Simple hash-based word reconstruction (simplified)
                    let word = format!("word_{}", token);
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(&word);
                },
            }
        }

        Ok(result)
    }

    /// Sample tokens using the policy model with temperature sampling
    fn sample_tokens_with_model(
        &self,
        model: &PolicyModel,
        prompt_tokens: &[u32],
        max_length: usize,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_sequence = prompt_tokens.to_vec();

        // Temperature for sampling
        let temperature = 0.8f32;

        for _ in 0..max_length {
            // Get logits from model (simplified matrix multiplication)
            let logits = self.get_model_logits(model, &current_sequence)?;

            // Apply temperature scaling
            let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

            // Convert to probabilities using softmax
            let max_logit = scaled_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> =
                scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

            // Sample from the distribution
            let token = self.sample_from_distribution(&probabilities)?;

            // Check for end-of-sequence token
            if token == 2 {
                break;
            }

            generated_tokens.push(token);
            current_sequence.push(token);

            // Prevent infinite generation
            if generated_tokens.len() >= 100 {
                break;
            }
        }

        Ok(generated_tokens)
    }

    /// Get next-token logits for `sequence` from the policy model's real parameters.
    ///
    /// This is a straight delegation to [`PolicyModel::logits`]: the full vocabulary is
    /// scored from the stored embedding / hidden / output matrices, so different weights
    /// give different logits and a model without parameters errors out.
    fn get_model_logits(&self, model: &PolicyModel, sequence: &[u32]) -> Result<Vec<f32>> {
        model.logits(sequence)
    }

    /// Sample from a probability distribution
    fn sample_from_distribution(&self, probabilities: &[f32]) -> Result<u32> {
        let mut cumulative = 0.0;
        let random_value = fastrand::f32(); // Simple random number

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probabilities.len() - 1) as u32)
    }

    /// Calculate log probabilities with the model
    fn calculate_log_probs_with_model(
        &self,
        model: &PolicyModel,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
    ) -> Result<Array1<f32>> {
        let mut log_probs = Vec::new();
        let mut current_sequence = prompt_tokens.to_vec();

        for &token in generated_tokens {
            // Get logits for current sequence
            let logits = self.get_model_logits(model, &current_sequence)?;

            // Convert to log probabilities
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let log_sum_exp = max_logit + sum_exp.ln();

            // Get log probability for the actual token
            let token_logit = logits.get(token as usize).copied().unwrap_or(f32::NEG_INFINITY);
            let log_prob = token_logit - log_sum_exp;

            log_probs.push(log_prob);
            current_sequence.push(token);
        }

        Ok(Array1::from_vec(log_probs))
    }

    /// Calculate value estimate using the value model
    fn calculate_value_with_model(
        &self,
        value_model: &ValueModel,
        sequence: &[u32],
    ) -> Result<f32> {
        value_model.value(sequence)
    }
}

/// Result of text generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Original prompt
    pub prompt: String,
    /// Generated response
    pub response: String,
    /// Generated tokens
    pub tokens: Vec<u32>,
    /// Log probabilities for each token
    pub log_probs: Array1<f32>,
    /// Value estimate for the generated sequence
    pub value: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_trainer_creation() {
        let config = PPOConfig::default();
        let trainer = PPOTrainer::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_advantage_calculation() {
        let config = PPOConfig::default();
        let trainer = PPOTrainer::new(config).expect("operation failed in test");

        let rewards = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let values = Array1::from_vec(vec![0.5, 1.5, 2.5]);

        let result = trainer.calculate_advantages(&rewards, &values, 0.99f32, 0.95f32);
        assert!(result.is_ok());

        let (advantages, returns) = result.expect("operation failed in test");
        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);
    }

    // ── Real policy / value forward pass ─────────────────────────────────────

    #[test]
    fn test_policy_logits_depend_on_the_model_weights() {
        // Regression: `get_model_logits` used to be a fixed function of the token ids only,
        // so two entirely different weight sets produced byte-identical logits.
        let a = PolicyModel::new_initialized("a", 32, 8, 1).expect("model a");
        let b = PolicyModel::new_initialized("b", 32, 8, 999).expect("model b");
        let sequence = [1u32, 5, 9, 2];

        let logits_a = a.logits(&sequence).expect("logits a");
        let logits_b = b.logits(&sequence).expect("logits b");
        assert_eq!(logits_a.len(), 32);
        assert_eq!(logits_b.len(), 32);

        let max_diff = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "different weights must give different logits (max diff {max_diff})"
        );
    }

    #[test]
    fn test_policy_logits_depend_on_the_input_sequence() {
        let model = PolicyModel::new_initialized("p", 32, 8, 7).expect("model");
        let one = model.logits(&[3u32, 4, 5]).expect("logits one");
        let two = model.logits(&[9u32, 10, 11]).expect("logits two");
        let max_diff =
            one.iter().zip(two.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "different inputs must give different logits"
        );
    }

    #[test]
    fn test_policy_logits_cover_the_whole_vocabulary() {
        // Regression: the old projection stopped at `vocab_size.min(1000)`, leaving every
        // token above index 999 pinned at exactly 0.0.
        let model = PolicyModel::new_initialized("wide", 1200, 4, 11).expect("model");
        let logits = model.logits(&[1u32, 2, 3]).expect("logits");
        assert_eq!(logits.len(), 1200);
        let tail_nonzero = logits[1000..].iter().any(|v| v.abs() > 1e-9);
        assert!(
            tail_nonzero,
            "logits beyond index 999 must be produced by the projection, not left at zero"
        );
    }

    #[test]
    fn test_policy_logits_error_when_parameters_are_missing() {
        let model = PolicyModel {
            model_id: "empty".to_string(),
            parameters: HashMap::new(),
            vocab_size: 16,
            hidden_size: 4,
        };
        let err = model.logits(&[1u32, 2]).expect_err("must not fabricate logits");
        assert!(
            err.to_string().contains(PARAM_EMBEDDING),
            "error should name the missing parameter, got: {err}"
        );
    }

    #[test]
    fn test_policy_logits_reject_out_of_vocabulary_tokens() {
        let model = PolicyModel::new_initialized("p", 16, 4, 3).expect("model");
        assert!(model.logits(&[99u32]).is_err());
    }

    #[test]
    fn test_policy_logits_are_hand_checkable_without_a_hidden_layer() {
        // With no hidden transform and a single token, the context is exactly that token's
        // embedding row, so the logit is a plain dot product with the output column.
        let mut parameters = HashMap::new();
        let mut embedding = Array2::zeros((2, 2));
        embedding[[1, 0]] = 2.0;
        embedding[[1, 1]] = 3.0;
        let mut output = Array2::zeros((2, 2));
        output[[0, 1]] = 1.0;
        output[[1, 1]] = 10.0;
        parameters.insert(PARAM_EMBEDDING.to_string(), embedding);
        parameters.insert(PARAM_OUTPUT_WEIGHT.to_string(), output);

        let model = PolicyModel {
            model_id: "hand".to_string(),
            parameters,
            vocab_size: 2,
            hidden_size: 2,
        };
        let logits = model.logits(&[1u32]).expect("logits");
        assert!((logits[0] - 0.0).abs() < 1e-6, "logit0 = {}", logits[0]);
        // 2*1 + 3*10 = 32
        assert!((logits[1] - 32.0).abs() < 1e-5, "logit1 = {}", logits[1]);
    }

    #[test]
    fn test_value_model_depends_on_weights_and_input() {
        let a = ValueModel::new_initialized("va", 32, 8, 2).expect("value a");
        let b = ValueModel::new_initialized("vb", 32, 8, 4242).expect("value b");
        let seq = [1u32, 6, 7];
        let va = a.value(&seq).expect("value a");
        let vb = b.value(&seq).expect("value b");
        assert!(
            (va - vb).abs() > 1e-6,
            "different value weights must give different values ({va} vs {vb})"
        );
        assert!(
            va.abs() <= 1.0 && vb.abs() <= 1.0,
            "tanh output must be bounded"
        );

        let other = a.value(&[2u32, 3, 4]).expect("value other");
        assert!(
            (va - other).abs() > 1e-9,
            "different inputs must give different values"
        );
    }

    #[test]
    fn test_value_model_errors_without_parameters() {
        let vm = ValueModel {
            model_id: "empty".to_string(),
            parameters: HashMap::new(),
            hidden_size: 4,
        };
        assert!(vm.value(&[1u32]).is_err());
    }

    #[test]
    fn test_generate_responses_uses_the_real_models() {
        let cfg = PPOConfig::default();
        let mut trainer = PPOTrainer::new(cfg).expect("trainer creation failed");
        let policy = PolicyModel::new_initialized("p", 64, 8, 21).expect("policy");
        let value = ValueModel::new_initialized("v", 64, 8, 22).expect("value");
        trainer.initialize_models(policy, value, None).expect("initialize failed");

        let results = trainer
            .generate_responses(&["hello world".to_string()], 5)
            .expect("generation failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].value.is_finite());
        assert_eq!(results[0].log_probs.len(), results[0].tokens.len());
        for lp in results[0].log_probs.iter() {
            assert!(lp.is_finite() && *lp <= 0.0, "log prob out of range: {lp}");
        }
    }

    #[test]
    fn test_tokenize_stays_inside_the_model_vocabulary() {
        let cfg = PPOConfig::default();
        let mut trainer = PPOTrainer::new(cfg).expect("trainer creation failed");
        let policy = PolicyModel::new_initialized("p", 64, 8, 5).expect("policy");
        let value = ValueModel::new_initialized("v", 64, 8, 6).expect("value");
        trainer.initialize_models(policy, value, None).expect("initialize failed");

        let tokens = trainer
            .tokenize("a much longer sentence with many distinct words here")
            .expect("tokenize failed");
        assert!(
            tokens.iter().all(|&t| (t as usize) < 64),
            "every token id must fit the model vocabulary: {tokens:?}"
        );
    }

    #[test]
    fn test_policy_model_creation() {
        let model = PolicyModel {
            model_id: "test_model".to_string(),
            parameters: HashMap::new(),
            vocab_size: 50000,
            hidden_size: 768,
        };

        assert_eq!(model.vocab_size, 50000);
        assert_eq!(model.hidden_size, 768);
    }

    #[test]
    fn test_experience_batch() {
        let batch = ExperienceBatch {
            sequences: Array2::zeros((4, 10)),
            action_probs: Array2::ones((4, 1)),
            old_action_probs: Array2::ones((4, 1)),
            rewards: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            values: Array1::from_vec(vec![0.5, 1.5, 2.5, 3.5]),
            advantages: Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]),
            returns: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
        };

        assert_eq!(batch.rewards.len(), 4);
        assert_eq!(batch.sequences.shape(), &[4, 10]);
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ppo_config_default_clip_param_range() {
        let cfg = PPOConfig::default();
        assert!(
            cfg.clip_param > 0.0 && cfg.clip_param < 1.0,
            "clip_param should be in (0,1), got {}",
            cfg.clip_param
        );
    }

    #[test]
    fn test_ppo_config_default_positive_lr() {
        let cfg = PPOConfig::default();
        assert!(cfg.policy_lr > 0.0);
        assert!(cfg.value_lr > 0.0);
    }

    #[test]
    fn test_ppo_config_default_positive_coefficients() {
        let cfg = PPOConfig::default();
        assert!(cfg.vf_coef > 0.0);
        assert!(cfg.entropy_coef >= 0.0);
        assert!(cfg.kl_penalty >= 0.0);
    }

    #[test]
    fn test_policy_model_empty_params() {
        let model = PolicyModel {
            model_id: "empty".to_string(),
            parameters: HashMap::new(),
            vocab_size: 1000,
            hidden_size: 64,
        };
        assert!(model.parameters.is_empty());
        assert_eq!(model.vocab_size, 1000);
    }

    #[test]
    fn test_value_model_creation() {
        let vm = ValueModel {
            model_id: "value_net".to_string(),
            parameters: HashMap::new(),
            hidden_size: 256,
        };
        assert_eq!(vm.hidden_size, 256);
    }

    #[test]
    fn test_ppo_statistics_default() {
        let stats = PPOStatistics::default();
        assert_eq!(stats.total_steps, 0);
        assert!(stats.policy_losses.is_empty());
        assert!(stats.value_losses.is_empty());
        assert!(stats.kl_divergences.is_empty());
    }

    #[test]
    fn test_advantage_calculation_single_step() {
        let cfg = PPOConfig::default();
        let trainer = PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let rewards = Array1::from_vec(vec![1.0]);
        let values = Array1::from_vec(vec![0.0]);
        let result = trainer.calculate_advantages(&rewards, &values, 0.99, 0.95);
        assert!(result.is_ok());
        let (adv, ret) = result.unwrap_or_else(|_| panic!("advantage calculation failed"));
        assert_eq!(adv.len(), 1);
        assert_eq!(ret.len(), 1);
        // For single step: delta = 1.0 - 0.0, GAE = delta
        assert!((adv[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_advantage_returns_are_sum_of_adv_and_value() {
        let cfg = PPOConfig::default();
        let trainer = PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let rewards = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let values = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let (adv, ret) = trainer
            .calculate_advantages(&rewards, &values, 0.99, 0.95)
            .unwrap_or_else(|_| panic!("advantage calculation failed"));
        for i in 0..3 {
            let expected_ret = adv[i] + values[i];
            assert!(
                (ret[i] - expected_ret).abs() < 1e-4,
                "return[{}] should equal advantage+value",
                i
            );
        }
    }

    #[test]
    fn test_ppo_step_result_structure() {
        let result = PPOStepResult {
            policy_loss: 0.5,
            value_loss: 0.3,
            kl_divergence: 0.01,
            entropy: 1.5,
            clip_fraction: 0.1,
            explained_variance: 0.8,
        };
        assert!(result.policy_loss >= 0.0);
        assert!(result.clip_fraction >= 0.0 && result.clip_fraction <= 1.0);
    }

    #[test]
    fn test_experience_batch_advantages_aligned_with_rewards() {
        let n = 8;
        let mut s = 42u64;
        fn lcg(s: &mut u64) -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s % 1000) as f32 / 1000.0
        }
        let rewards: Vec<f32> = (0..n).map(|_| lcg(&mut s)).collect();
        let values: Vec<f32> = (0..n).map(|_| lcg(&mut s) * 0.5).collect();
        let adv: Vec<f32> = (0..n).map(|_| lcg(&mut s) - 0.5).collect();
        let ret: Vec<f32> = adv.iter().zip(values.iter()).map(|(a, v)| a + v).collect();

        let batch = ExperienceBatch {
            sequences: Array2::zeros((n, 5)),
            action_probs: Array2::ones((n, 1)),
            old_action_probs: Array2::ones((n, 1)),
            rewards: Array1::from_vec(rewards),
            values: Array1::from_vec(values),
            advantages: Array1::from_vec(adv),
            returns: Array1::from_vec(ret),
        };
        assert_eq!(batch.rewards.len(), n);
        assert_eq!(batch.advantages.len(), n);
    }

    #[test]
    fn test_ppo_trainer_training_step() {
        let cfg = PPOConfig::default();
        let mut trainer =
            PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let batch = ExperienceBatch {
            sequences: Array2::zeros((2, 5)),
            action_probs: Array2::ones((2, 1)) * 0.6,
            old_action_probs: Array2::ones((2, 1)) * 0.5,
            rewards: Array1::from_vec(vec![1.0, 2.0]),
            values: Array1::from_vec(vec![0.5, 1.5]),
            advantages: Array1::from_vec(vec![0.5, 0.5]),
            returns: Array1::from_vec(vec![1.0, 2.0]),
        };
        let result = trainer.training_step(&batch);
        assert!(result.is_ok(), "training step should succeed");
    }

    #[test]
    fn test_ppo_statistics_accumulate_after_step() {
        let cfg = PPOConfig::default();
        let mut trainer =
            PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let batch = ExperienceBatch {
            sequences: Array2::zeros((2, 5)),
            action_probs: Array2::ones((2, 1)),
            old_action_probs: Array2::ones((2, 1)),
            rewards: Array1::from_vec(vec![1.0, 1.0]),
            values: Array1::from_vec(vec![0.5, 0.5]),
            advantages: Array1::from_vec(vec![0.5, 0.5]),
            returns: Array1::from_vec(vec![1.0, 1.0]),
        };
        trainer.training_step(&batch).unwrap_or(PPOStepResult {
            policy_loss: 0.0,
            value_loss: 0.0,
            kl_divergence: 0.0,
            entropy: 0.0,
            clip_fraction: 0.0,
            explained_variance: 0.0,
        });
        assert_eq!(trainer.statistics.total_steps, 1);
        assert!(!trainer.statistics.policy_losses.is_empty());
    }

    #[test]
    fn test_advantage_high_gamma_larger_returns() {
        let cfg = PPOConfig::default();
        let trainer = PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let rewards = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let values = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let (_, ret_high) = trainer
            .calculate_advantages(&rewards, &values, 0.99, 0.95)
            .unwrap_or_else(|_| panic!("failed"));
        let (_, ret_low) = trainer
            .calculate_advantages(&rewards, &values, 0.1, 0.95)
            .unwrap_or_else(|_| panic!("failed"));
        // With high gamma, future rewards matter more so first return should be higher
        assert!(
            ret_high[0] > ret_low[0],
            "higher gamma should give larger return at t=0"
        );
    }

    #[test]
    fn test_ppo_trainer_initialize_models() {
        let cfg = PPOConfig::default();
        let mut trainer =
            PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let policy = PolicyModel {
            model_id: "p".to_string(),
            parameters: HashMap::new(),
            vocab_size: 1000,
            hidden_size: 64,
        };
        let value = ValueModel {
            model_id: "v".to_string(),
            parameters: HashMap::new(),
            hidden_size: 64,
        };
        let result = trainer.initialize_models(policy, value, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ppo_trainer_generate_without_models_fails() {
        let cfg = PPOConfig::default();
        let trainer = PPOTrainer::new(cfg).unwrap_or_else(|_| panic!("trainer creation failed"));
        let result = trainer.generate_responses(&["hello".to_string()], 10);
        assert!(
            result.is_err(),
            "generate should fail without initialized models"
        );
    }

    #[test]
    fn test_ppo_optimizer_structure() {
        let opt = PPOOptimizer {
            policy_lr: 1e-4,
            value_lr: 1e-4,
            momentum: 0.9,
            weight_decay: 0.01,
        };
        assert!(opt.policy_lr > 0.0);
        assert!(opt.momentum > 0.0 && opt.momentum < 1.0);
    }

    #[test]
    fn test_ppo_config_target_kl_positive() {
        let cfg = PPOConfig::default();
        assert!(cfg.target_kl > 0.0, "target_kl should be positive");
    }

    #[test]
    fn test_experience_batch_returns_values_check() {
        // returns should be >= values (when advantages >= 0) or <= values (when adv <= 0)
        let batch = ExperienceBatch {
            sequences: Array2::zeros((3, 4)),
            action_probs: Array2::ones((3, 1)),
            old_action_probs: Array2::ones((3, 1)),
            rewards: Array1::from_vec(vec![1.0, 1.0, 1.0]),
            values: Array1::from_vec(vec![0.5, 0.5, 0.5]),
            advantages: Array1::from_vec(vec![0.5, 0.5, 0.5]), // all positive
            returns: Array1::from_vec(vec![1.0, 1.0, 1.0]),
        };
        for i in 0..3 {
            // returns = advantages + values = 0.5 + 0.5 = 1.0
            assert!((batch.returns[i] - (batch.advantages[i] + batch.values[i])).abs() < 1e-5);
        }
    }
}
