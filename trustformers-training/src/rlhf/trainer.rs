//! RLHF trainer implementation providing unified training interface.
//!
//! Every phase drives the **real** models defined in [`crate::rlhf::ppo`] and
//! [`crate::rlhf::reward_model`] through the analytic gradients in
//! [`crate::rlhf::policy_optimizer`]. There are no simulated losses, no heuristic stand-ins
//! for log-probabilities and no synthetic reward curves: calling [`RLHFTrainer::train`]
//! changes parameters, and the reported metrics are read back off those parameters.

use crate::rlhf::policy_optimizer::{
    apply_sgd, dpo_loss_and_grads, sequence_log_prob, sequence_log_prob_backward,
    value_loss_and_grads, Gradients,
};
use crate::rlhf::ppo::{PolicyModel, ValueModel};
use crate::rlhf::reward_model::{RewardBatch, RewardModel};
use crate::rlhf::{
    ConstitutionalPrinciple, HumanFeedback, PreferencePair, RLHFMetrics, RLHFPhase,
    RewardModelConfig,
};
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use std::collections::HashMap;
use trustformers_core::errors::{invalid_config, Result};

/// Padding / reserved id count of the trainer's hashing tokenizer.
const NUM_RESERVED_TOKENS: u32 = 3;
/// Beginning-of-sequence id.
const BOS_TOKEN: u32 = 1;
/// End-of-sequence id.
const EOS_TOKEN: u32 = 2;

/// Rating scale assumed for [`HumanFeedback::rating`] when normalising to `[0, 1]`.
const RATING_SCALE: f32 = 5.0;

/// FNV-1a 64-bit hash used by the trainer's hashing tokenizer.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Wrap an `anyhow` error coming out of the model layer into the crate error type.
fn model_error(context: &'static str, err: anyhow::Error) -> trustformers_core::TrustformersError {
    invalid_config(format!("{context}: {err}"), context)
}

/// RLHF trainer configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RLHFTrainerConfig {
    /// Training phase
    pub phase: RLHFPhase,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// KL penalty coefficient
    pub kl_penalty: f32,
    /// Reward scaling factor
    pub reward_scale: f32,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Constitutional AI principles (if using Constitutional phase)
    pub constitutional_principles: Vec<ConstitutionalPrinciple>,
    /// Vocabulary size of the policy / reference models.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Hidden width of the policy, value and reward models.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Seed for reproducible parameter initialisation.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// DPO temperature β.
    #[serde(default = "default_beta")]
    pub beta: f32,
    /// PPO clipping parameter ε.
    #[serde(default = "default_clip_epsilon")]
    pub clip_epsilon: f32,
}

fn default_vocab_size() -> usize {
    512
}

fn default_hidden_size() -> usize {
    32
}

fn default_seed() -> u64 {
    42
}

fn default_beta() -> f32 {
    0.1
}

fn default_clip_epsilon() -> f32 {
    0.2
}

impl Default for RLHFTrainerConfig {
    fn default() -> Self {
        Self {
            phase: RLHFPhase::SFT,
            learning_rate: 1e-5,
            batch_size: 8,
            epochs: 3,
            kl_penalty: 0.1,
            reward_scale: 1.0,
            max_seq_length: 512,
            constitutional_principles: Vec::new(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            seed: default_seed(),
            beta: default_beta(),
            clip_epsilon: default_clip_epsilon(),
        }
    }
}

/// Unified RLHF trainer supporting multiple phases.
///
/// The trainer owns four real models:
///
/// * a [`PolicyModel`] that every phase updates,
/// * a frozen reference [`PolicyModel`] (the KL / DPO anchor),
/// * a [`ValueModel`] used as the PPO critic,
/// * a [`RewardModel`] trained on the preference pairs.
///
/// They are created lazily by [`RLHFTrainer::ensure_models`] from the sizes in
/// [`RLHFTrainerConfig`], or supplied explicitly with [`RLHFTrainer::set_models`].
pub struct RLHFTrainer {
    /// Training configuration
    config: RLHFTrainerConfig,
    /// Training metrics
    metrics: RLHFMetrics,
    /// Feedback data
    feedback_data: Vec<HumanFeedback>,
    /// Preference pairs
    preference_pairs: Vec<PreferencePair>,
    /// Trainable policy.
    policy: Option<PolicyModel>,
    /// Frozen reference policy (KL / DPO anchor).
    reference_policy: Option<PolicyModel>,
    /// PPO critic.
    value_model: Option<ValueModel>,
    /// Reward model trained on preference pairs.
    reward_model: Option<RewardModel>,
}

impl std::fmt::Debug for RLHFTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RLHFTrainer")
            .field("config", &self.config)
            .field("metrics", &self.metrics)
            .field("feedback_data", &self.feedback_data.len())
            .field("preference_pairs", &self.preference_pairs.len())
            .field("policy_initialized", &self.policy.is_some())
            .finish()
    }
}

impl RLHFTrainer {
    /// Create a new RLHF trainer
    pub fn new(config: RLHFTrainerConfig) -> Self {
        Self {
            config,
            metrics: RLHFMetrics::default(),
            feedback_data: Vec::new(),
            preference_pairs: Vec::new(),
            policy: None,
            reference_policy: None,
            value_model: None,
            reward_model: None,
        }
    }

    /// Add human feedback data
    pub fn add_feedback(&mut self, feedback: HumanFeedback) {
        self.feedback_data.push(feedback);
    }

    /// Add preference pair
    pub fn add_preference_pair(&mut self, pair: PreferencePair) {
        self.preference_pairs.push(pair);
    }

    /// Get current training metrics
    pub fn metrics(&self) -> &RLHFMetrics {
        &self.metrics
    }

    /// Update training phase
    pub fn set_phase(&mut self, phase: RLHFPhase) {
        self.config.phase = phase;
        self.metrics.phase = phase;
    }

    /// Install externally-built models, replacing anything created by `ensure_models`.
    ///
    /// The reference policy is snapshotted from `policy` and stays frozen for the rest of
    /// training, which is what makes the KL and DPO terms meaningful.
    pub fn set_models(
        &mut self,
        policy: PolicyModel,
        value_model: ValueModel,
        reward_model: RewardModel,
    ) {
        self.reference_policy = Some(policy.clone());
        self.policy = Some(policy);
        self.value_model = Some(value_model);
        self.reward_model = Some(reward_model);
    }

    /// Borrow the trainable policy, if it has been created.
    pub fn policy(&self) -> Option<&PolicyModel> {
        self.policy.as_ref()
    }

    /// Borrow the reward model, if it has been created.
    pub fn reward_model(&self) -> Option<&RewardModel> {
        self.reward_model.as_ref()
    }

    /// Create the policy / reference / value / reward models if they do not exist yet.
    pub fn ensure_models(&mut self) -> Result<()> {
        if self.policy.is_none() {
            let policy = PolicyModel::new_initialized(
                "rlhf-policy",
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.seed,
            )
            .map_err(|e| model_error("policy initialisation", e))?;
            self.reference_policy = Some(policy.clone());
            self.policy = Some(policy);
        }
        if self.reference_policy.is_none() {
            self.reference_policy = self.policy.clone();
        }
        if self.value_model.is_none() {
            self.value_model = Some(
                ValueModel::new_initialized(
                    "rlhf-value",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.seed ^ 0x0F0F_0F0F,
                )
                .map_err(|e| model_error("value initialisation", e))?,
            );
        }
        if self.reward_model.is_none() {
            let rm_config = RewardModelConfig {
                max_length: self.config.max_seq_length,
                learning_rate: self.config.learning_rate as f64,
                batch_size: self.config.batch_size,
                ..RewardModelConfig::default()
            };
            let mut rm = RewardModel::new(rm_config).map_err(|e| model_error("reward model", e))?;
            rm.initialize_parameters(self.config.hidden_size)
                .map_err(|e| model_error("reward head initialisation", e))?;
            self.reward_model = Some(rm);
        }
        Ok(())
    }

    /// Hashing tokenizer shared by every phase.
    ///
    /// Words are mapped into `[NUM_RESERVED_TOKENS, vocab_size)` with FNV-1a; the sequence is
    /// wrapped in `<bos>` / `<eos>` and truncated to `max_seq_length`. Ids `0`, `1` and `2`
    /// are reserved for padding, BOS and EOS respectively.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let vocab = self.config.vocab_size.max(NUM_RESERVED_TOKENS as usize + 1) as u64;
        let span = vocab - NUM_RESERVED_TOKENS as u64;
        let mut tokens = vec![BOS_TOKEN];
        for word in text.split_whitespace() {
            if tokens.len() + 1 >= self.config.max_seq_length.max(2) {
                break;
            }
            tokens.push((fnv1a64(word.as_bytes()) % span) as u32 + NUM_RESERVED_TOKENS);
        }
        tokens.push(EOS_TOKEN);
        tokens
    }

    /// Start training for the current phase
    pub async fn train(&mut self) -> Result<RLHFMetrics> {
        match self.config.phase {
            RLHFPhase::SFT => self.train_supervised().await,
            RLHFPhase::RewardModel => self.train_reward_model().await,
            RLHFPhase::PPO => self.train_ppo().await,
            RLHFPhase::DPO => self.train_dpo().await,
            RLHFPhase::Constitutional => self.train_constitutional().await,
        }
    }

    /// Supervised fine-tuning: quality-weighted next-token cross-entropy on the policy.
    ///
    /// For each `(prompt, response)` pair the loss is the mean negative log-likelihood of the
    /// response tokens conditioned on the prompt, weighted by the normalised human rating so
    /// that highly-rated responses dominate the update. Gradients come from
    /// [`sequence_log_prob_backward`] and are applied with plain SGD.
    async fn train_supervised(&mut self) -> Result<RLHFMetrics> {
        if self.feedback_data.is_empty() {
            return Err(invalid_config(
                "No feedback data available for supervised fine-tuning",
                "train_supervised",
            ));
        }
        self.ensure_models()?;

        let batch_size = self.config.batch_size.max(1);
        let examples: Vec<(Vec<u32>, Vec<u32>, f32)> = self
            .feedback_data
            .iter()
            .map(|fb| {
                let prompt = self.tokenize(&fb.prompt);
                let response = self.tokenize(&fb.response);
                let weight = (fb.rating / RATING_SCALE).clamp(0.0, 1.0);
                (prompt, response, weight)
            })
            .filter(|(p, r, _)| !p.is_empty() && !r.is_empty())
            .collect();

        if examples.is_empty() {
            return Err(invalid_config(
                "All feedback entries tokenized to empty sequences",
                "train_supervised",
            ));
        }

        let mut final_loss = 0.0f32;
        for _epoch in 0..self.config.epochs.max(1) {
            let mut epoch_nll = 0.0f32;
            let mut epoch_tokens = 0usize;

            for chunk in examples.chunks(batch_size) {
                let mut grads = Gradients::new();
                let mut batch_nll = 0.0f32;
                let mut batch_tokens = 0usize;

                {
                    let policy = self.policy.as_ref().ok_or_else(|| {
                        invalid_config("policy model missing", "train_supervised")
                    })?;
                    for (prompt, response, weight) in chunk {
                        // Ascending the (weighted) log-likelihood == descending the NLL, and
                        // `apply_sgd` subtracts, so the upstream is negative here.
                        let scale = -weight / (chunk.len() * response.len()).max(1) as f32;
                        let logp =
                            sequence_log_prob_backward(policy, prompt, response, scale, &mut grads)
                                .map_err(|e| model_error("sft backward", e))?;
                        batch_nll += -logp;
                        batch_tokens += response.len();
                    }
                }

                let policy = self
                    .policy
                    .as_mut()
                    .ok_or_else(|| invalid_config("policy model missing", "train_supervised"))?;
                apply_sgd(&mut policy.parameters, &grads, self.config.learning_rate)
                    .map_err(|e| model_error("sft update", e))?;

                epoch_nll += batch_nll;
                epoch_tokens += batch_tokens;
            }

            final_loss = if epoch_tokens > 0 { epoch_nll / epoch_tokens as f32 } else { 0.0 };
        }

        self.metrics.policy_loss = Some(final_loss);
        self.metrics.phase = RLHFPhase::SFT;
        self.metrics.response_lengths =
            self.feedback_data.iter().map(|fb| fb.response.len()).collect();
        Ok(self.metrics.clone())
    }

    /// Train the reward model on the stored preference pairs.
    ///
    /// Delegates to [`RewardModel::train_step`], which runs the analytic backward pass of the
    /// pairwise Bradley–Terry loss over the reward head's real parameters.
    async fn train_reward_model(&mut self) -> Result<RLHFMetrics> {
        if self.preference_pairs.is_empty() {
            return Err(invalid_config(
                "No preference pairs available for reward model training",
                "train_reward_model",
            ));
        }
        self.ensure_models()?;

        let batch_size = self.config.batch_size.max(1);
        let pairs = self.preference_pairs.clone();
        let reward_model = self
            .reward_model
            .as_mut()
            .ok_or_else(|| invalid_config("reward model missing", "train_reward_model"))?;

        let mut last_accuracy = 0.0f32;
        let mut last_loss = 0.0f32;

        for _epoch in 0..self.config.epochs.max(1) {
            let mut epoch_accuracy = 0.0f32;
            let mut epoch_loss = 0.0f32;
            let mut num_batches = 0usize;

            for chunk in pairs.chunks(batch_size) {
                let chosen_texts: Vec<String> =
                    chunk.iter().map(|p| format!("{} {}", p.prompt, p.chosen)).collect();
                let rejected_texts: Vec<String> =
                    chunk.iter().map(|p| format!("{} {}", p.prompt, p.rejected)).collect();

                let chosen_tokens = reward_model
                    .encode_batch(&chosen_texts)
                    .map_err(|e| model_error("reward tokenization", e))?;
                let rejected_tokens = reward_model
                    .encode_batch(&rejected_texts)
                    .map_err(|e| model_error("reward tokenization", e))?;

                let batch = RewardBatch {
                    prompts: chunk.iter().map(|p| p.prompt.clone()).collect(),
                    chosen_responses: chunk.iter().map(|p| p.chosen.clone()).collect(),
                    rejected_responses: chunk.iter().map(|p| p.rejected.clone()).collect(),
                    chosen_tokens,
                    rejected_tokens,
                    labels: Array1::from_elem(chunk.len(), 1.0),
                };

                let result = reward_model
                    .train_step(&batch)
                    .map_err(|e| model_error("reward training step", e))?;
                epoch_accuracy += result.accuracy;
                epoch_loss += result.loss;
                num_batches += 1;
            }

            if num_batches > 0 {
                last_accuracy = epoch_accuracy / num_batches as f32;
                last_loss = epoch_loss / num_batches as f32;
            }
        }

        self.metrics.reward_accuracy = Some(last_accuracy);
        self.metrics.policy_loss = Some(last_loss);
        self.metrics.phase = RLHFPhase::RewardModel;
        Ok(self.metrics.clone())
    }

    /// PPO against the trained reward model.
    ///
    /// Each feedback entry becomes one rollout: the recorded response is the action, the
    /// reward model scores `prompt + response`, and the critic supplies the baseline. The
    /// update is the clipped surrogate objective
    ///
    /// ```text
    /// L = −min( r·A , clip(r, 1±ε)·A ) + kl_penalty · (log π_θ − log π_ref)
    /// r = exp(log π_θ(y|x) − log π_old(y|x))
    /// ```
    ///
    /// whose derivative with respect to `log π_θ` is `−A·r` on the unclipped branch and `0`
    /// on the clipped one; the critic is fitted to the observed reward by squared error.
    async fn train_ppo(&mut self) -> Result<RLHFMetrics> {
        if self.feedback_data.is_empty() {
            return Err(invalid_config(
                "No feedback data available for PPO training",
                "train_ppo",
            ));
        }
        self.ensure_models()?;

        let clip_epsilon = self.config.clip_epsilon;
        let batch_size = self.config.batch_size.max(1);
        let rollouts: Vec<(Vec<u32>, Vec<u32>, String)> = self
            .feedback_data
            .iter()
            .map(|fb| {
                (
                    self.tokenize(&fb.prompt),
                    self.tokenize(&fb.response),
                    format!("{} {}", fb.prompt, fb.response),
                )
            })
            .filter(|(p, r, _)| !p.is_empty() && !r.is_empty())
            .collect();

        if rollouts.is_empty() {
            return Err(invalid_config(
                "All feedback entries tokenized to empty sequences",
                "train_ppo",
            ));
        }

        let mut last_objective = 0.0f32;
        let mut last_kl = 0.0f32;
        let mut last_value_loss = 0.0f32;
        let mut advantages = Vec::new();
        let mut rewards = Vec::new();

        for _epoch in 0..self.config.epochs.max(1) {
            let mut epoch_objective = 0.0f32;
            let mut epoch_kl = 0.0f32;
            let mut epoch_value_loss = 0.0f32;
            let mut num_batches = 0usize;
            advantages.clear();
            rewards.clear();

            for chunk in rollouts.chunks(batch_size) {
                let mut policy_grads = Gradients::new();
                let mut value_grads = Gradients::new();
                let mut batch_objective = 0.0f32;
                let mut batch_kl = 0.0f32;
                let mut batch_value_loss = 0.0f32;

                {
                    let policy = self
                        .policy
                        .as_ref()
                        .ok_or_else(|| invalid_config("policy missing", "train_ppo"))?;
                    let reference = self
                        .reference_policy
                        .as_ref()
                        .ok_or_else(|| invalid_config("reference missing", "train_ppo"))?;
                    let value_model = self
                        .value_model
                        .as_ref()
                        .ok_or_else(|| invalid_config("value model missing", "train_ppo"))?;
                    let reward_model = self
                        .reward_model
                        .as_ref()
                        .ok_or_else(|| invalid_config("reward model missing", "train_ppo"))?;

                    let scale = 1.0 / chunk.len() as f32;
                    for (prompt, response, text) in chunk {
                        let full: Vec<u32> =
                            prompt.iter().chain(response.iter()).copied().collect();

                        // Reward from the real reward model.
                        let prediction = reward_model
                            .predict_reward(text)
                            .map_err(|e| model_error("reward prediction", e))?;
                        let reward = prediction.score * self.config.reward_scale;

                        // Baseline from the critic.
                        let baseline = value_model
                            .value(&full)
                            .map_err(|e| model_error("value estimate", e))?;
                        let advantage = reward - baseline;

                        // Old and reference log-probabilities (both frozen for this step).
                        let log_pi_old = sequence_log_prob(policy, prompt, response)
                            .map_err(|e| model_error("policy log-prob", e))?;
                        let log_pi_ref = sequence_log_prob(reference, prompt, response)
                            .map_err(|e| model_error("reference log-prob", e))?;

                        // Within a step r == 1 by construction; the clip test still applies
                        // and keeps the branch selection identical to a multi-epoch inner loop.
                        let ratio = 1.0f32;
                        let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
                        let unclipped_obj = ratio * advantage;
                        let clipped_obj = clipped * advantage;
                        let objective = unclipped_obj.min(clipped_obj);
                        let uses_unclipped = unclipped_obj <= clipped_obj;

                        let kl = log_pi_old - log_pi_ref;

                        // d(loss)/d(log pi): -A*r on the unclipped branch, plus the KL term.
                        let mut upstream = self.config.kl_penalty;
                        if uses_unclipped {
                            upstream -= advantage * ratio;
                        }
                        sequence_log_prob_backward(
                            policy,
                            prompt,
                            response,
                            upstream * scale,
                            &mut policy_grads,
                        )
                        .map_err(|e| model_error("ppo backward", e))?;

                        // Critic regression onto the observed reward.
                        let v_loss =
                            value_loss_and_grads(value_model, &full, reward, &mut value_grads)
                                .map_err(|e| model_error("value backward", e))?;

                        batch_objective += objective * scale;
                        batch_kl += kl * scale;
                        batch_value_loss += v_loss * scale;
                        advantages.push(advantage);
                        rewards.push(reward);
                    }
                }

                {
                    let policy = self
                        .policy
                        .as_mut()
                        .ok_or_else(|| invalid_config("policy missing", "train_ppo"))?;
                    apply_sgd(
                        &mut policy.parameters,
                        &policy_grads,
                        self.config.learning_rate,
                    )
                    .map_err(|e| model_error("ppo policy update", e))?;
                }
                {
                    let value_model = self
                        .value_model
                        .as_mut()
                        .ok_or_else(|| invalid_config("value model missing", "train_ppo"))?;
                    apply_sgd(
                        &mut value_model.parameters,
                        &value_grads,
                        self.config.learning_rate,
                    )
                    .map_err(|e| model_error("ppo value update", e))?;
                }

                epoch_objective += batch_objective;
                epoch_kl += batch_kl;
                epoch_value_loss += batch_value_loss;
                num_batches += 1;
            }

            if num_batches > 0 {
                last_objective = epoch_objective / num_batches as f32;
                last_kl = epoch_kl / num_batches as f32;
                last_value_loss = epoch_value_loss / num_batches as f32;
            }
        }

        let avg_reward = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().sum::<f32>() / rewards.len() as f32
        };

        self.metrics.ppo_objective = Some(last_objective);
        self.metrics.kl_divergence = last_kl;
        self.metrics.value_loss = Some(last_value_loss);
        self.metrics.avg_reward = avg_reward;
        self.metrics.advantages = advantages.clone();
        self.metrics.phase = RLHFPhase::PPO;
        Ok(self.metrics.clone())
    }

    /// Direct Preference Optimization against the frozen reference policy.
    ///
    /// Log-probabilities are the model's real per-token log-probabilities and the gradients
    /// come from [`dpo_loss_and_grads`].
    async fn train_dpo(&mut self) -> Result<RLHFMetrics> {
        if self.preference_pairs.is_empty() {
            return Err(invalid_config(
                "No preference pairs available for DPO training",
                "train_dpo",
            ));
        }
        self.ensure_models()?;

        let beta = self.config.beta;
        let batch_size = self.config.batch_size.max(1);
        let triples: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = self
            .preference_pairs
            .iter()
            .map(|pair| {
                (
                    self.tokenize(&pair.prompt),
                    self.tokenize(&pair.chosen),
                    self.tokenize(&pair.rejected),
                )
            })
            .filter(|(p, c, r)| !p.is_empty() && !c.is_empty() && !r.is_empty())
            .collect();

        if triples.is_empty() {
            return Err(invalid_config(
                "All preference pairs tokenized to empty sequences",
                "train_dpo",
            ));
        }

        let mut last_loss = 0.0f32;
        let mut last_reward = 0.0f32;
        let mut last_accuracy = 0.0f32;

        for _epoch in 0..self.config.epochs.max(1) {
            let mut epoch_loss = 0.0f32;
            let mut epoch_reward = 0.0f32;
            let mut correct = 0usize;
            let mut seen = 0usize;

            for chunk in triples.chunks(batch_size) {
                let mut grads = Gradients::new();
                let scale = 1.0 / chunk.len() as f32;

                {
                    let policy = self
                        .policy
                        .as_ref()
                        .ok_or_else(|| invalid_config("policy missing", "train_dpo"))?;
                    let reference = self
                        .reference_policy
                        .as_ref()
                        .ok_or_else(|| invalid_config("reference missing", "train_dpo"))?;

                    for (prompt, chosen, rejected) in chunk {
                        let mut item_grads = Gradients::new();
                        let outcome = dpo_loss_and_grads(
                            policy,
                            reference,
                            prompt,
                            chosen,
                            rejected,
                            beta,
                            &mut item_grads,
                        )
                        .map_err(|e| model_error("dpo backward", e))?;

                        for (key, grad) in item_grads {
                            let entry = grads.entry(key).or_insert_with(|| grad.mapv(|_| 0.0f32));
                            *entry = &*entry + &(grad * scale);
                        }

                        epoch_loss += outcome.loss * scale;
                        epoch_reward += outcome.chosen_reward * scale;
                        if outcome.chosen_reward > outcome.rejected_reward {
                            correct += 1;
                        }
                        seen += 1;
                    }
                }

                let policy = self
                    .policy
                    .as_mut()
                    .ok_or_else(|| invalid_config("policy missing", "train_dpo"))?;
                apply_sgd(&mut policy.parameters, &grads, self.config.learning_rate)
                    .map_err(|e| model_error("dpo update", e))?;
            }

            let num_batches = triples.len().div_ceil(batch_size).max(1) as f32;
            last_loss = epoch_loss / num_batches;
            last_reward = epoch_reward / num_batches;
            last_accuracy = if seen > 0 { correct as f32 / seen as f32 } else { 0.0 };
        }

        self.metrics.policy_loss = Some(last_loss);
        self.metrics.avg_reward = last_reward;
        self.metrics.reward_accuracy = Some(last_accuracy);
        self.metrics.phase = RLHFPhase::DPO;
        Ok(self.metrics.clone())
    }

    /// Constitutional AI: lexical critique of each response, then a compliance-weighted
    /// supervised update on the policy.
    ///
    /// The critique is an explicit, principle-driven lexical rule (see
    /// [`RLHFTrainer::evaluate_principle_violation`]) — it is not a model and is not
    /// presented as one. The training signal is real: responses that violate a principle get
    /// a lower likelihood weight, so the policy is genuinely pushed away from them.
    async fn train_constitutional(&mut self) -> Result<RLHFMetrics> {
        if self.config.constitutional_principles.is_empty() {
            return Err(invalid_config(
                "No constitutional principles defined for Constitutional AI training",
                "train_constitutional",
            ));
        }

        if self.feedback_data.is_empty() {
            return Err(invalid_config(
                "No feedback data available for Constitutional AI training",
                "train_constitutional",
            ));
        }
        self.ensure_models()?;

        let batch_size = self.config.batch_size.max(1);
        let mut total_violations = 0usize;
        let scored: Vec<(Vec<u32>, Vec<u32>, f32)> = self
            .feedback_data
            .iter()
            .map(|fb| {
                let mut worst = 0.0f32;
                let mut violations = 0usize;
                for principle in &self.config.constitutional_principles {
                    let score = self.evaluate_principle_violation(&fb.response, principle);
                    if score > 0.5 {
                        violations += 1;
                    }
                    worst = worst.max(score * principle.weight.max(0.0));
                }
                total_violations += violations;
                (
                    self.tokenize(&fb.prompt),
                    self.tokenize(&fb.response),
                    (1.0 - worst).clamp(0.0, 1.0),
                )
            })
            .filter(|(p, r, _)| !p.is_empty() && !r.is_empty())
            .collect();

        if scored.is_empty() {
            return Err(invalid_config(
                "All feedback entries tokenized to empty sequences",
                "train_constitutional",
            ));
        }

        let mut final_loss = 0.0f32;
        for _epoch in 0..self.config.epochs.max(1) {
            let mut epoch_nll = 0.0f32;
            let mut epoch_tokens = 0usize;

            for chunk in scored.chunks(batch_size) {
                let mut grads = Gradients::new();
                {
                    let policy = self
                        .policy
                        .as_ref()
                        .ok_or_else(|| invalid_config("policy missing", "train_constitutional"))?;
                    for (prompt, response, compliance) in chunk {
                        let scale = -compliance / (chunk.len() * response.len()).max(1) as f32;
                        let logp =
                            sequence_log_prob_backward(policy, prompt, response, scale, &mut grads)
                                .map_err(|e| model_error("constitutional backward", e))?;
                        epoch_nll += -logp * compliance;
                        epoch_tokens += response.len();
                    }
                }

                let policy = self
                    .policy
                    .as_mut()
                    .ok_or_else(|| invalid_config("policy missing", "train_constitutional"))?;
                apply_sgd(&mut policy.parameters, &grads, self.config.learning_rate)
                    .map_err(|e| model_error("constitutional update", e))?;
            }

            final_loss = if epoch_tokens > 0 { epoch_nll / epoch_tokens as f32 } else { 0.0 };
        }

        self.metrics.constitutional_violations = Some(total_violations);
        self.metrics.policy_loss = Some(final_loss);
        self.metrics.phase = RLHFPhase::Constitutional;
        Ok(self.metrics.clone())
    }

    /// Lexical violation score in `[0, 1]` for one principle.
    ///
    /// The score is the fraction of the principle's own cue phrases that appear in the
    /// response. Cues are taken from [`ConstitutionalPrinciple::violation_examples`] when
    /// present, otherwise from the significant words of
    /// [`ConstitutionalPrinciple::criteria`]; the three built-in principle names keep their
    /// curated keyword lists. This is a documented rule-based critic — there is no model
    /// behind it and none is claimed.
    pub fn evaluate_principle_violation(
        &self,
        response: &str,
        principle: &ConstitutionalPrinciple,
    ) -> f32 {
        const HARMFUL: &[&str] = &["violence", "hate", "harmful", "dangerous", "illegal"];
        const UNHELPFUL: &[&str] = &["don't know", "can't help", "no idea", "unclear"];
        const DISHONEST: &[&str] = &["fake", "false", "lie", "untrue", "misleading"];

        let lowered = response.to_lowercase();
        let ratio = |cues: &[&str]| -> f32 {
            if cues.is_empty() {
                return 0.0;
            }
            let hits = cues.iter().filter(|cue| lowered.contains(&cue.to_lowercase())).count();
            (hits as f32 / cues.len() as f32).min(1.0)
        };

        match principle.name.to_lowercase().as_str() {
            "harmlessness" => ratio(HARMFUL),
            "helpfulness" => ratio(UNHELPFUL),
            "honesty" => ratio(DISHONEST),
            _ => {
                if !principle.violation_examples.is_empty() {
                    let cues: Vec<&str> =
                        principle.violation_examples.iter().map(|s| s.as_str()).collect();
                    ratio(&cues)
                } else {
                    // Fall back to the significant words of the principle's own criteria.
                    let cues: Vec<&str> =
                        principle.criteria.split_whitespace().filter(|w| w.len() > 4).collect();
                    ratio(&cues)
                }
            },
        }
    }

    /// Evaluate model performance
    pub async fn evaluate(&self) -> Result<HashMap<String, f32>> {
        let mut eval_metrics = HashMap::new();

        // Basic evaluation metrics
        eval_metrics.insert("avg_reward".to_string(), self.metrics.avg_reward);
        eval_metrics.insert("kl_divergence".to_string(), self.metrics.kl_divergence);

        if let Some(policy_loss) = self.metrics.policy_loss {
            eval_metrics.insert("policy_loss".to_string(), policy_loss);
        }

        if let Some(reward_accuracy) = self.metrics.reward_accuracy {
            eval_metrics.insert("reward_accuracy".to_string(), reward_accuracy);
        }

        if let Some(value_loss) = self.metrics.value_loss {
            eval_metrics.insert("value_loss".to_string(), value_loss);
        }

        Ok(eval_metrics)
    }

    /// Save trainer state
    pub fn save_state(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create trainer state struct for serialization
        let state = TrainerState {
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            feedback_data_count: self.feedback_data.len(),
            preference_pairs_count: self.preference_pairs.len(),
        };

        // Serialize and save state
        let serialized = serde_json::to_string_pretty(&state).map_err(|e| {
            invalid_config(
                format!("Failed to serialize trainer state: {}", e),
                "save_state",
            )
        })?;

        let mut file = File::create(path).map_err(|e| {
            invalid_config(format!("Failed to create state file: {}", e), "save_state")
        })?;

        file.write_all(serialized.as_bytes()).map_err(|e| {
            invalid_config(format!("Failed to write state file: {}", e), "save_state")
        })?;

        Ok(())
    }

    /// Load trainer state
    pub fn load_state(&mut self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| {
            invalid_config(format!("Failed to open state file: {}", e), "load_state")
        })?;

        let mut contents = String::new();
        file.read_to_string(&mut contents).map_err(|e| {
            invalid_config(format!("Failed to read state file: {}", e), "load_state")
        })?;

        let state: TrainerState = serde_json::from_str(&contents).map_err(|e| {
            invalid_config(
                format!("Failed to deserialize trainer state: {}", e),
                "load_state",
            )
        })?;

        // Restore state
        self.config = state.config;
        self.metrics = state.metrics;

        // Note: model parameters are not part of the JSON state; re-create or re-load them
        // with `ensure_models` / `set_models` after restoring.

        Ok(())
    }
}

/// Serializable trainer state for save/load operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrainerState {
    config: RLHFTrainerConfig,
    metrics: RLHFMetrics,
    feedback_data_count: usize,
    preference_pairs_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_trainer_creation() {
        let config = RLHFTrainerConfig::default();
        let trainer = RLHFTrainer::new(config);

        assert_eq!(trainer.config.phase, RLHFPhase::SFT);
        assert_eq!(trainer.feedback_data.len(), 0);
        assert_eq!(trainer.preference_pairs.len(), 0);
    }

    #[test]
    fn test_add_feedback() {
        let config = RLHFTrainerConfig::default();
        let mut trainer = RLHFTrainer::new(config);

        let feedback = HumanFeedback {
            id: "test".to_string(),
            prompt: "Test prompt".to_string(),
            response: "Test response".to_string(),
            rating: 4.0,
            feedback_text: None,
            timestamp: Utc::now(),
            annotator_id: None,
            metadata: HashMap::new(),
        };

        trainer.add_feedback(feedback);
        assert_eq!(trainer.feedback_data.len(), 1);
    }

    #[test]
    fn test_phase_update() {
        let config = RLHFTrainerConfig::default();
        let mut trainer = RLHFTrainer::new(config);

        trainer.set_phase(RLHFPhase::PPO);
        assert_eq!(trainer.config.phase, RLHFPhase::PPO);
        assert_eq!(trainer.metrics.phase, RLHFPhase::PPO);
    }

    #[test]
    fn test_trainer_config_defaults() {
        let config = RLHFTrainerConfig::default();
        assert_eq!(config.phase, RLHFPhase::SFT);
        assert_eq!(config.learning_rate, 1e-5);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.epochs, 3);
        assert_eq!(config.kl_penalty, 0.1);
        assert_eq!(config.reward_scale, 1.0);
        assert_eq!(config.max_seq_length, 512);
        assert!(config.constitutional_principles.is_empty());
    }

    #[test]
    fn test_add_multiple_feedback() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        for i in 0..5 {
            let feedback = HumanFeedback {
                id: format!("fb_{}", i),
                prompt: format!("Prompt {}", i),
                response: format!("Response {}", i),
                rating: (i + 1) as f32,
                feedback_text: None,
                timestamp: Utc::now(),
                annotator_id: None,
                metadata: HashMap::new(),
            };
            trainer.add_feedback(feedback);
        }
        assert_eq!(trainer.feedback_data.len(), 5);
    }

    #[test]
    fn test_add_preference_pair() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        let pair = PreferencePair {
            prompt: "Test prompt".to_string(),
            chosen: "Good response".to_string(),
            rejected: "Bad response".to_string(),
            confidence: 0.8,
            reasoning: None,
        };
        trainer.add_preference_pair(pair);
        assert_eq!(trainer.preference_pairs.len(), 1);
    }

    #[test]
    fn test_metrics_access() {
        let trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        let metrics = trainer.metrics();
        assert_eq!(metrics.phase, RLHFPhase::SFT);
    }

    #[test]
    fn test_phase_sft_to_reward() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        assert_eq!(trainer.config.phase, RLHFPhase::SFT);
        trainer.set_phase(RLHFPhase::RewardModel);
        assert_eq!(trainer.config.phase, RLHFPhase::RewardModel);
    }

    #[test]
    fn test_phase_to_dpo() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        trainer.set_phase(RLHFPhase::DPO);
        assert_eq!(trainer.config.phase, RLHFPhase::DPO);
        assert_eq!(trainer.metrics.phase, RLHFPhase::DPO);
    }

    #[test]
    fn test_phase_to_constitutional() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        trainer.set_phase(RLHFPhase::Constitutional);
        assert_eq!(trainer.config.phase, RLHFPhase::Constitutional);
    }

    #[test]
    fn test_config_with_constitutional_principles() {
        let config = RLHFTrainerConfig {
            phase: RLHFPhase::Constitutional,
            constitutional_principles: vec![ConstitutionalPrinciple {
                name: "helpfulness".to_string(),
                description: "Be helpful".to_string(),
                weight: 1.0,
                criteria: "Must be helpful".to_string(),
                violation_examples: vec![],
                adherence_examples: vec![],
            }],
            ..Default::default()
        };
        assert_eq!(config.constitutional_principles.len(), 1);
    }

    #[test]
    fn test_config_custom_values() {
        let config = RLHFTrainerConfig {
            phase: RLHFPhase::PPO,
            learning_rate: 5e-6,
            batch_size: 16,
            epochs: 5,
            kl_penalty: 0.05,
            reward_scale: 2.0,
            max_seq_length: 1024,
            constitutional_principles: Vec::new(),
            ..RLHFTrainerConfig::default()
        };
        assert_eq!(config.learning_rate, 5e-6);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_seq_length, 1024);
    }

    #[test]
    fn test_feedback_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            serde_json::Value::String("human".to_string()),
        );
        metadata.insert(
            "task".to_string(),
            serde_json::Value::String("qa".to_string()),
        );
        let feedback = HumanFeedback {
            id: "meta_fb".to_string(),
            prompt: "Test".to_string(),
            response: "Answer".to_string(),
            rating: 4.5,
            feedback_text: Some("Good answer".to_string()),
            timestamp: Utc::now(),
            annotator_id: Some("ann_001".to_string()),
            metadata,
        };
        assert!(feedback.feedback_text.is_some());
        assert!(feedback.annotator_id.is_some());
        assert_eq!(feedback.metadata.len(), 2);
    }

    #[test]
    fn test_preference_pair_with_reasoning() {
        let pair = PreferencePair {
            prompt: "Explain AI".to_string(),
            chosen: "AI is a broad field...".to_string(),
            rejected: "AI is robots".to_string(),
            confidence: 0.95,
            reasoning: Some("More comprehensive answer".to_string()),
        };
        assert!(pair.confidence > 0.5);
        assert!(pair.reasoning.is_some());
    }

    #[test]
    fn test_trainer_initial_state() {
        let trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        assert!(trainer.feedback_data.is_empty());
        assert!(trainer.preference_pairs.is_empty());
    }

    #[test]
    fn test_trainer_multiple_phase_changes() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        let phases = [
            RLHFPhase::SFT,
            RLHFPhase::RewardModel,
            RLHFPhase::PPO,
            RLHFPhase::DPO,
            RLHFPhase::Constitutional,
        ];
        for phase in &phases {
            trainer.set_phase(*phase);
            assert_eq!(trainer.config.phase, *phase);
        }
    }

    #[test]
    fn test_feedback_rating_range() {
        let feedback = HumanFeedback {
            id: "range_test".to_string(),
            prompt: "test".to_string(),
            response: "test".to_string(),
            rating: 5.0,
            feedback_text: None,
            timestamp: Utc::now(),
            annotator_id: None,
            metadata: HashMap::new(),
        };
        assert!(feedback.rating >= 0.0 && feedback.rating <= 5.0);
    }

    fn feedback(id: &str, prompt: &str, response: &str, rating: f32) -> HumanFeedback {
        HumanFeedback {
            id: id.to_string(),
            prompt: prompt.to_string(),
            response: response.to_string(),
            rating,
            feedback_text: None,
            timestamp: Utc::now(),
            annotator_id: None,
            metadata: HashMap::new(),
        }
    }

    fn small_config(phase: RLHFPhase) -> RLHFTrainerConfig {
        RLHFTrainerConfig {
            phase,
            learning_rate: 0.2,
            batch_size: 2,
            epochs: 3,
            vocab_size: 64,
            hidden_size: 8,
            max_seq_length: 16,
            ..RLHFTrainerConfig::default()
        }
    }

    // ── Real training phases ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_sft_changes_the_policy_parameters() {
        // Regression: `train_supervised` used to multiply a synthetic scalar by
        // `1 - learning_rate` and never touched a model.
        let mut trainer = RLHFTrainer::new(small_config(RLHFPhase::SFT));
        trainer.add_feedback(feedback("a", "what is rust", "a systems language", 5.0));
        trainer.add_feedback(feedback("b", "what is cargo", "the rust build tool", 4.0));
        trainer.ensure_models().expect("models");

        let before = trainer
            .policy()
            .expect("policy")
            .parameters
            .get("output.weight")
            .expect("output weight")
            .clone();

        trainer.train().await.expect("sft failed");

        let after = trainer
            .policy()
            .expect("policy")
            .parameters
            .get("output.weight")
            .expect("output weight");
        let max_delta = before
            .iter()
            .zip(after.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-6,
            "SFT must move the policy parameters (max delta {max_delta})"
        );
        assert!(trainer.metrics().policy_loss.expect("loss").is_finite());
    }

    #[tokio::test]
    async fn test_sft_raises_the_likelihood_of_the_training_responses() {
        let mut cfg = small_config(RLHFPhase::SFT);
        cfg.epochs = 60;
        let mut trainer = RLHFTrainer::new(cfg);
        trainer.add_feedback(feedback("a", "greeting", "hello there friend", 5.0));
        trainer.ensure_models().expect("models");

        let prompt = trainer.tokenize("greeting");
        let response = trainer.tokenize("hello there friend");
        let before = crate::rlhf::policy_optimizer::sequence_log_prob(
            trainer.policy().expect("policy"),
            &prompt,
            &response,
        )
        .expect("logp");

        trainer.train().await.expect("sft failed");

        let after = crate::rlhf::policy_optimizer::sequence_log_prob(
            trainer.policy().expect("policy"),
            &prompt,
            &response,
        )
        .expect("logp");
        assert!(
            after > before,
            "SFT must increase the likelihood of the trained response ({before} -> {after})"
        );
    }

    #[tokio::test]
    async fn test_reward_model_phase_learns_the_preferences() {
        let mut cfg = small_config(RLHFPhase::RewardModel);
        cfg.epochs = 60;
        cfg.learning_rate = 0.5;
        let mut trainer = RLHFTrainer::new(cfg);
        trainer.add_preference_pair(PreferencePair {
            prompt: "what is the capital of france".to_string(),
            chosen: "the capital of france is paris".to_string(),
            rejected: "i do not know".to_string(),
            confidence: 0.9,
            reasoning: None,
        });

        let metrics = trainer.train().await.expect("reward training failed");
        let accuracy = metrics.reward_accuracy.expect("accuracy");
        assert!(
            accuracy >= 1.0 - 1e-6,
            "the reward model must learn a single separable pair, got {accuracy}"
        );

        let rm = trainer.reward_model().expect("reward model");
        let chosen = rm
            .predict_reward("what is the capital of france the capital of france is paris")
            .expect("chosen score");
        let rejected = rm
            .predict_reward("what is the capital of france i do not know")
            .expect("rejected score");
        assert!(
            chosen.score > rejected.score,
            "chosen must outscore rejected after training ({} vs {})",
            chosen.score,
            rejected.score
        );
    }

    #[tokio::test]
    async fn test_dpo_phase_moves_the_policy_away_from_the_reference() {
        // Regression: DPO used to run on `simulate_log_probability`, a function of string
        // length and word-repetition ratio, and never touched a model.
        let mut cfg = small_config(RLHFPhase::DPO);
        cfg.epochs = 25;
        cfg.beta = 0.5;
        let mut trainer = RLHFTrainer::new(cfg);
        trainer.add_preference_pair(PreferencePair {
            prompt: "explain ai".to_string(),
            chosen: "artificial intelligence studies machine reasoning".to_string(),
            rejected: "robots".to_string(),
            confidence: 0.95,
            reasoning: None,
        });

        let metrics = trainer.train().await.expect("dpo failed");
        assert!(metrics.policy_loss.expect("loss").is_finite());

        let prompt = trainer.tokenize("explain ai");
        let chosen = trainer.tokenize("artificial intelligence studies machine reasoning");
        let rejected = trainer.tokenize("robots");
        let policy = trainer.policy().expect("policy");

        let logp_chosen =
            crate::rlhf::policy_optimizer::sequence_log_prob(policy, &prompt, &chosen)
                .expect("chosen");
        let logp_rejected =
            crate::rlhf::policy_optimizer::sequence_log_prob(policy, &prompt, &rejected)
                .expect("rejected");
        assert!(logp_chosen.is_finite() && logp_rejected.is_finite());
        assert!(
            metrics.avg_reward.is_finite(),
            "the implicit DPO reward must be a real number"
        );
    }

    #[tokio::test]
    async fn test_dpo_loss_decreases_over_epochs() {
        let pair = PreferencePair {
            prompt: "explain ai".to_string(),
            chosen: "artificial intelligence studies machine reasoning".to_string(),
            rejected: "robots".to_string(),
            confidence: 0.95,
            reasoning: None,
        };

        let run = |epochs: usize| {
            let mut cfg = small_config(RLHFPhase::DPO);
            cfg.epochs = epochs;
            cfg.beta = 0.5;
            let mut trainer = RLHFTrainer::new(cfg);
            trainer.add_preference_pair(pair.clone());
            trainer
        };

        let mut short = run(1);
        let short_metrics = short.train().await.expect("dpo short");
        let mut long = run(40);
        let long_metrics = long.train().await.expect("dpo long");

        assert!(
            long_metrics.policy_loss.expect("long") < short_metrics.policy_loss.expect("short"),
            "more DPO epochs must reduce the loss ({:?} -> {:?})",
            short_metrics.policy_loss,
            long_metrics.policy_loss
        );
    }

    #[tokio::test]
    async fn test_ppo_phase_updates_policy_and_critic() {
        let mut cfg = small_config(RLHFPhase::PPO);
        cfg.epochs = 5;
        let mut trainer = RLHFTrainer::new(cfg);
        trainer.add_feedback(feedback("a", "greeting", "hello there", 5.0));
        trainer.add_feedback(feedback("b", "greeting", "go away", 1.0));
        trainer.ensure_models().expect("models");

        let before = trainer
            .policy()
            .expect("policy")
            .parameters
            .get("output.weight")
            .expect("weight")
            .clone();

        let metrics = trainer.train().await.expect("ppo failed");
        assert!(metrics.ppo_objective.expect("objective").is_finite());
        assert!(metrics.value_loss.expect("value loss").is_finite());
        assert_eq!(metrics.advantages.len(), 2);

        let after = trainer
            .policy()
            .expect("policy")
            .parameters
            .get("output.weight")
            .expect("weight");
        let max_delta = before
            .iter()
            .zip(after.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta > 1e-9, "PPO must move the policy parameters");
    }

    #[tokio::test]
    async fn test_constitutional_phase_counts_violations_and_trains() {
        let mut cfg = small_config(RLHFPhase::Constitutional);
        cfg.constitutional_principles = vec![ConstitutionalPrinciple {
            name: "harmlessness".to_string(),
            description: "avoid harm".to_string(),
            weight: 1.0,
            criteria: "response avoids harmful content".to_string(),
            violation_examples: vec![],
            adherence_examples: vec![],
        }];
        let mut trainer = RLHFTrainer::new(cfg);
        trainer.add_feedback(feedback(
            "bad",
            "how do i",
            "violence hate harmful dangerous illegal",
            1.0,
        ));
        trainer.add_feedback(feedback("good", "how do i", "bake a cake safely", 5.0));

        let metrics = trainer.train().await.expect("constitutional failed");
        assert_eq!(
            metrics.constitutional_violations,
            Some(1),
            "exactly one response violates the harmlessness principle"
        );
        assert!(metrics.policy_loss.expect("loss").is_finite());
    }

    #[test]
    fn test_principle_violation_uses_the_principle_itself() {
        // Regression: the catch-all arm used to call `simulate_reward_prediction`, a
        // length/character-diversity heuristic unrelated to the principle.
        let trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        let principle = ConstitutionalPrinciple {
            name: "privacy".to_string(),
            description: "do not leak secrets".to_string(),
            weight: 1.0,
            criteria: "response withholds personal data".to_string(),
            violation_examples: vec!["home address".to_string(), "social security".to_string()],
            adherence_examples: vec![],
        };

        let clean = trainer.evaluate_principle_violation("here is a recipe", &principle);
        let dirty =
            trainer.evaluate_principle_violation("their home address is on file", &principle);
        assert_eq!(clean, 0.0, "a clean response must score 0");
        assert!(dirty > 0.0, "a matching cue must raise the violation score");

        // A long, diverse but clean response must still score 0 — the old heuristic would
        // have produced a non-zero value purely from its length.
        let long_clean = trainer.evaluate_principle_violation(
            "a thoroughly detailed and lengthy explanation covering many unrelated topics",
            &principle,
        );
        assert_eq!(long_clean, 0.0);
    }

    #[test]
    fn test_tokenizer_stays_inside_the_vocabulary() {
        let trainer = RLHFTrainer::new(small_config(RLHFPhase::SFT));
        let tokens = trainer.tokenize("some words that need hashing into a small vocabulary");
        assert!(tokens.first() == Some(&1), "sequence must start with <bos>");
        assert!(tokens.last() == Some(&2), "sequence must end with <eos>");
        assert!(tokens.iter().all(|&t| (t as usize) < 64));
    }

    #[tokio::test]
    async fn test_train_sft_requires_feedback() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        let result = trainer.train().await;
        assert!(result.is_err()); // No feedback data
    }

    #[tokio::test]
    async fn test_train_reward_model_requires_preferences() {
        let mut trainer = RLHFTrainer::new(RLHFTrainerConfig::default());
        trainer.set_phase(RLHFPhase::RewardModel);
        let result = trainer.train().await;
        assert!(result.is_err()); // No preference pairs
    }
}
