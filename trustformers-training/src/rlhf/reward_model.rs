//! Reward model implementation for RLHF training.

use crate::rlhf::{HumanFeedback, PreferencePair, RewardModelConfig, RewardModelType};
use anyhow::Result;
use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reward model for scoring text generations.
///
/// # Architecture
///
/// Text is turned into a fixed-width feature vector with the *hashing trick*
/// (see [`RewardModel::featurize`]) and a trainable head maps that vector to a scalar
/// reward. The head is one of [`RewardModelType`]:
///
/// | type | parameters | forward |
/// |------|------------|---------|
/// | `Linear` | `reward_head.weight [d,1]`, `reward_head.bias [1,1]` | `x·W + b` |
/// | `MLP` | `reward_head.fc1.{weight,bias}`, `reward_head.fc2.{weight,bias}` | `relu(x·W₁+b₁)·W₂+b₂` |
/// | `Transformer` | `reward_head.attention.weight [d,d]`, `reward_head.output.weight [d,1]` | `tanh(x·A)·O` |
/// | `Ensemble` | `reward_head_{0,1,2}.{weight,bias}` | mean of three linear heads |
///
/// where `d = base_model_size` as passed to [`RewardModel::initialize_parameters`].
/// [`RewardModel::train_step`] runs a real backward pass over these parameters, so training
/// provably changes what [`RewardModel::predict_reward`] returns.
#[derive(Debug)]
pub struct RewardModel {
    config: RewardModelConfig,
    model_type: RewardModelType,
    parameters: HashMap<String, Array2<f32>>,
    /// Feature dimensionality `d`, set by `initialize_parameters`.
    feature_size: usize,
    tokenizer: Option<RewardTokenizer>,
    training_data: Vec<PreferencePair>,
    statistics: RewardModelStatistics,
}

/// Hashing tokenizer for the reward model.
///
/// Words are mapped to ids by an FNV-1a hash modulo `vocab_size`; no vocabulary file is
/// needed and the id space is bounded, which keeps the downstream feature hashing exact.
#[derive(Debug, Clone)]
pub struct RewardTokenizer {
    vocab_size: usize,
    max_length: usize,
}

/// FNV-1a 64-bit hash — used both for tokenization and for feature hashing.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Deterministic splitmix64 step for reproducible parameter initialisation.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Xavier-uniform matrix built from a deterministic seed.
fn xavier(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let limit = (6.0f32 / (rows as f32 + cols as f32)).sqrt();
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        let bits = splitmix64(&mut state);
        let unit = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
        ((unit as f32) * 2.0 - 1.0) * limit
    })
}

/// Training statistics for reward model
#[derive(Debug, Default)]
pub struct RewardModelStatistics {
    /// Training accuracy over time
    pub accuracies: Vec<f32>,
    /// Training losses over time
    pub losses: Vec<f32>,
    /// Validation accuracies
    pub val_accuracies: Vec<f32>,
    /// Validation losses
    pub val_losses: Vec<f32>,
    /// Number of training steps
    pub training_steps: usize,
    /// Average reward scores
    pub avg_reward_scores: Vec<f32>,
}

/// Reward prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPrediction {
    /// Input text
    pub text: String,
    /// Predicted reward score
    pub score: f32,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f32,
    /// Individual component scores (if using ensemble)
    pub component_scores: Vec<f32>,
    /// Attention weights (if available)
    pub attention_weights: Option<Array2<f32>>,
}

/// Training batch for reward model
#[derive(Debug, Clone)]
pub struct RewardBatch {
    /// Prompt texts
    pub prompts: Vec<String>,
    /// Response texts (chosen)
    pub chosen_responses: Vec<String>,
    /// Response texts (rejected)
    pub rejected_responses: Vec<String>,
    /// Tokenized chosen sequences
    pub chosen_tokens: Array2<u32>,
    /// Tokenized rejected sequences
    pub rejected_tokens: Array2<u32>,
    /// Preference labels (1.0 for chosen, 0.0 for rejected)
    pub labels: Array1<f32>,
}

/// Reward model training step result
#[derive(Debug, Clone)]
pub struct RewardTrainingResult {
    /// Training loss
    pub loss: f32,
    /// Training accuracy
    pub accuracy: f32,
    /// Average reward for chosen responses
    pub avg_chosen_reward: f32,
    /// Average reward for rejected responses
    pub avg_rejected_reward: f32,
    /// Reward margin (chosen - rejected)
    pub reward_margin: f32,
}

impl RewardModel {
    /// Create a new reward model
    pub fn new(config: RewardModelConfig) -> Result<Self> {
        let tokenizer = RewardTokenizer {
            vocab_size: 50000, // Default vocabulary size
            max_length: config.max_length,
        };

        Ok(Self {
            model_type: config.model_type,
            config,
            parameters: HashMap::new(),
            feature_size: 0,
            tokenizer: Some(tokenizer),
            training_data: Vec::new(),
            statistics: RewardModelStatistics::default(),
        })
    }

    /// Initialize model parameters for a `base_model_size`-dimensional feature space.
    ///
    /// Parameters are drawn from a seeded Xavier-uniform distribution rather than zeroed:
    /// a zero-initialised MLP head has identically zero gradients and can never learn.
    pub fn initialize_parameters(&mut self, base_model_size: usize) -> Result<()> {
        if base_model_size == 0 {
            return Err(anyhow::anyhow!(
                "base_model_size must be non-zero for the reward head"
            ));
        }
        self.feature_size = base_model_size;
        match self.model_type {
            RewardModelType::Linear => {
                self.initialize_linear_head(base_model_size)?;
            },
            RewardModelType::MLP => {
                self.initialize_mlp_head(base_model_size)?;
            },
            RewardModelType::Transformer => {
                self.initialize_transformer_head(base_model_size)?;
            },
            RewardModelType::Ensemble => {
                self.initialize_ensemble_heads(base_model_size)?;
            },
        }
        Ok(())
    }

    /// Load training data from preference pairs
    pub fn load_training_data(&mut self, preference_pairs: Vec<PreferencePair>) -> Result<()> {
        self.training_data = preference_pairs;
        Ok(())
    }

    /// Load training data from human feedback
    pub fn load_from_human_feedback(&mut self, feedback: Vec<HumanFeedback>) -> Result<()> {
        // Convert human feedback to preference pairs
        let mut pairs = Vec::new();

        // Group feedback by prompt and create pairs based on ratings
        let mut feedback_by_prompt: HashMap<String, Vec<&HumanFeedback>> = HashMap::new();

        for fb in &feedback {
            feedback_by_prompt.entry(fb.prompt.clone()).or_default().push(fb);
        }

        for (prompt, prompt_feedback) in feedback_by_prompt {
            // Sort by rating descending
            let mut sorted_feedback = prompt_feedback;
            sorted_feedback.sort_by(|a, b| {
                b.rating.partial_cmp(&a.rating).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Create pairs between high and low rated responses
            for i in 0..sorted_feedback.len() {
                for j in (i + 1)..sorted_feedback.len() {
                    if sorted_feedback[i].rating > sorted_feedback[j].rating {
                        pairs.push(PreferencePair {
                            prompt: prompt.clone(),
                            chosen: sorted_feedback[i].response.clone(),
                            rejected: sorted_feedback[j].response.clone(),
                            confidence: (sorted_feedback[i].rating - sorted_feedback[j].rating)
                                / 5.0,
                            reasoning: None,
                        });
                    }
                }
            }
        }

        self.training_data = pairs;
        Ok(())
    }

    /// Train the reward model on preference data
    pub fn train_step(&mut self, batch: &RewardBatch) -> Result<RewardTrainingResult> {
        // Forward pass for chosen responses
        let chosen_rewards = self.forward(&batch.chosen_tokens)?;

        // Forward pass for rejected responses
        let rejected_rewards = self.forward(&batch.rejected_tokens)?;

        // Calculate ranking loss
        let loss = self.calculate_ranking_loss(&chosen_rewards, &rejected_rewards)?;

        // Calculate accuracy (how often chosen > rejected)
        let accuracy = self.calculate_accuracy(&chosen_rewards, &rejected_rewards)?;

        // Calculate statistics
        let avg_chosen_reward = chosen_rewards.mean().unwrap_or(0.0);
        let avg_rejected_reward = rejected_rewards.mean().unwrap_or(0.0);
        let reward_margin = avg_chosen_reward - avg_rejected_reward;

        // Update statistics
        self.statistics.training_steps += 1;
        self.statistics.losses.push(loss);
        self.statistics.accuracies.push(accuracy);
        self.statistics.avg_reward_scores.push(avg_chosen_reward);

        // Real backward pass: analytic gradients of the pairwise ranking loss followed by an
        // SGD update of the head parameters.
        self.backward_pass(&batch.chosen_tokens, &batch.rejected_tokens)?;

        Ok(RewardTrainingResult {
            loss,
            accuracy,
            avg_chosen_reward,
            avg_rejected_reward,
            reward_margin,
        })
    }

    /// Predict reward for a single text
    pub fn predict_reward(&self, text: &str) -> Result<RewardPrediction> {
        let tokens = self.tokenize(text)?;
        let tokens_array = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;

        let rewards = self.forward(&tokens_array)?;
        let score = rewards[0];

        // Calculate confidence based on model uncertainty (simplified)
        let confidence = self.calculate_confidence(score)?;

        Ok(RewardPrediction {
            text: text.to_string(),
            score,
            confidence,
            component_scores: vec![score], // Single model for now
            attention_weights: None,
        })
    }

    /// Predict rewards for multiple texts
    pub fn predict_batch(&self, texts: &[String]) -> Result<Vec<RewardPrediction>> {
        let mut predictions = Vec::new();

        for text in texts {
            predictions.push(self.predict_reward(text)?);
        }

        Ok(predictions)
    }

    /// Compare two responses and return preference probability
    pub fn compare_responses(
        &self,
        prompt: &str,
        response_a: &str,
        response_b: &str,
    ) -> Result<f32> {
        let full_text_a = format!("{} {}", prompt, response_a);
        let full_text_b = format!("{} {}", prompt, response_b);

        let pred_a = self.predict_reward(&full_text_a)?;
        let pred_b = self.predict_reward(&full_text_b)?;

        // Convert to probability using sigmoid
        let score_diff = pred_a.score - pred_b.score;
        let preference_prob = 1.0 / (1.0 + (-score_diff).exp());

        Ok(preference_prob)
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> &RewardModelStatistics {
        &self.statistics
    }

    /// Tokenize and right-pad a set of texts into a `[batch, max_len]` id matrix.
    ///
    /// Padding uses id `0`, which [`RewardModel::featurize`] skips, so short and long texts
    /// in the same batch are scored on their content alone.
    pub fn encode_batch(&self, texts: &[String]) -> Result<Array2<u32>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let rows: Vec<Vec<u32>> =
            texts.iter().map(|t| self.tokenize(t)).collect::<Result<Vec<_>>>()?;
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
        let mut flat = Vec::with_capacity(rows.len() * width);
        for mut row in rows {
            row.resize(width, 0);
            flat.extend(row);
        }
        Ok(Array2::from_shape_vec((texts.len(), width), flat)?)
    }

    // Private helper methods

    fn initialize_linear_head(&mut self, base_size: usize) -> Result<()> {
        self.parameters.insert(
            "reward_head.weight".to_string(),
            xavier(base_size, 1, 0x5EED_0001),
        );
        self.parameters.insert("reward_head.bias".to_string(), Array2::zeros((1, 1)));
        Ok(())
    }

    fn initialize_mlp_head(&mut self, base_size: usize) -> Result<()> {
        let hidden_size = self.config.reward_head_hidden_size;
        if hidden_size == 0 {
            return Err(anyhow::anyhow!(
                "reward_head_hidden_size must be non-zero for an MLP reward head"
            ));
        }

        self.parameters.insert(
            "reward_head.fc1.weight".to_string(),
            xavier(base_size, hidden_size, 0x5EED_0002),
        );
        self.parameters.insert(
            "reward_head.fc1.bias".to_string(),
            Array2::zeros((1, hidden_size)),
        );
        self.parameters.insert(
            "reward_head.fc2.weight".to_string(),
            xavier(hidden_size, 1, 0x5EED_0003),
        );
        self.parameters
            .insert("reward_head.fc2.bias".to_string(), Array2::zeros((1, 1)));
        Ok(())
    }

    fn initialize_transformer_head(&mut self, base_size: usize) -> Result<()> {
        self.parameters.insert(
            "reward_head.attention.weight".to_string(),
            xavier(base_size, base_size, 0x5EED_0004),
        );
        self.parameters.insert(
            "reward_head.output.weight".to_string(),
            xavier(base_size, 1, 0x5EED_0005),
        );
        Ok(())
    }

    fn initialize_ensemble_heads(&mut self, base_size: usize) -> Result<()> {
        // Initialize multiple heads for ensemble; distinct seeds keep them decorrelated.
        for i in 0..3 {
            self.parameters.insert(
                format!("reward_head_{}.weight", i),
                xavier(base_size, 1, 0x5EED_0010 + i as u64),
            );
            self.parameters.insert(format!("reward_head_{}.bias", i), Array2::zeros((1, 1)));
        }
        Ok(())
    }

    /// Fetch a parameter matrix or fail with a message naming it.
    fn param(&self, key: &str) -> Result<&Array2<f32>> {
        self.parameters.get(key).ok_or_else(|| {
            anyhow::anyhow!(
                "reward model parameter '{key}' is missing; call initialize_parameters() \
                 (or load pretrained weights) before scoring"
            )
        })
    }

    /// Feature-hash a token sequence into a `feature_size`-dimensional vector.
    ///
    /// This is the standard hashing trick: token `t` contributes `±1` to bucket
    /// `h(t) mod d`, the sign coming from an independent bit of the same hash so that
    /// collisions cancel in expectation instead of accumulating. Padding (`0`) is skipped and
    /// the result is L2-normalised, which makes the reward scale independent of length.
    pub fn featurize(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let dim = self.feature_size;
        if dim == 0 {
            return Err(anyhow::anyhow!(
                "reward model has no feature space; call initialize_parameters() first"
            ));
        }
        let mut features = vec![0.0f32; dim];
        for &token in tokens {
            if token == 0 {
                continue; // padding
            }
            let hash = fnv1a64(&token.to_le_bytes());
            let bucket = (hash % dim as u64) as usize;
            let sign = if (hash >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            features[bucket] += sign;
        }
        let norm = features.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in features.iter_mut() {
                *v /= norm;
            }
        }
        Ok(features)
    }

    /// Score one feature vector with the configured head.
    fn score_features(&self, features: &[f32]) -> Result<f32> {
        let d = self.feature_size;
        match self.model_type {
            RewardModelType::Linear => {
                let w = self.param("reward_head.weight")?;
                let b = self.param("reward_head.bias")?;
                let mut s = b[[0, 0]];
                for k in 0..d {
                    s += features[k] * w[[k, 0]];
                }
                Ok(s)
            },
            RewardModelType::MLP => {
                let w1 = self.param("reward_head.fc1.weight")?;
                let b1 = self.param("reward_head.fc1.bias")?;
                let w2 = self.param("reward_head.fc2.weight")?;
                let b2 = self.param("reward_head.fc2.bias")?;
                let h = w1.ncols();
                let mut s = b2[[0, 0]];
                for j in 0..h {
                    let mut z = b1[[0, j]];
                    for k in 0..d {
                        z += features[k] * w1[[k, j]];
                    }
                    s += z.max(0.0) * w2[[j, 0]];
                }
                Ok(s)
            },
            RewardModelType::Transformer => {
                let a = self.param("reward_head.attention.weight")?;
                let o = self.param("reward_head.output.weight")?;
                let mut s = 0.0f32;
                for j in 0..d {
                    let mut u = 0.0f32;
                    for k in 0..d {
                        u += features[k] * a[[k, j]];
                    }
                    s += u.tanh() * o[[j, 0]];
                }
                Ok(s)
            },
            RewardModelType::Ensemble => {
                let mut total = 0.0f32;
                for i in 0..3 {
                    let w = self.param(&format!("reward_head_{}.weight", i))?;
                    let b = self.param(&format!("reward_head_{}.bias", i))?;
                    let mut s = b[[0, 0]];
                    for k in 0..d {
                        s += features[k] * w[[k, 0]];
                    }
                    total += s;
                }
                Ok(total / 3.0)
            },
        }
    }

    /// Accumulate `d(score)/d(parameters)` scaled by `upstream` into `grads`.
    ///
    /// These are exact analytic gradients of [`RewardModel::score_features`] for the
    /// configured head, so the SGD step in [`RewardModel::apply_gradients`] moves the
    /// parameters in the direction that actually reduces the ranking loss.
    fn accumulate_head_gradients(
        &self,
        features: &[f32],
        upstream: f32,
        grads: &mut HashMap<String, Array2<f32>>,
    ) -> Result<()> {
        let d = self.feature_size;
        match self.model_type {
            RewardModelType::Linear => {
                let gw = grads
                    .entry("reward_head.weight".to_string())
                    .or_insert_with(|| Array2::zeros((d, 1)));
                for k in 0..d {
                    gw[[k, 0]] += features[k] * upstream;
                }
                let gb = grads
                    .entry("reward_head.bias".to_string())
                    .or_insert_with(|| Array2::zeros((1, 1)));
                gb[[0, 0]] += upstream;
            },
            RewardModelType::MLP => {
                let w1 = self.param("reward_head.fc1.weight")?.clone();
                let b1 = self.param("reward_head.fc1.bias")?.clone();
                let w2 = self.param("reward_head.fc2.weight")?.clone();
                let h = w1.ncols();

                // Pre-activations of the hidden layer, needed for the relu gate.
                let mut pre = vec![0.0f32; h];
                for (j, slot) in pre.iter_mut().enumerate() {
                    let mut z = b1[[0, j]];
                    for k in 0..d {
                        z += features[k] * w1[[k, j]];
                    }
                    *slot = z;
                }

                {
                    let gw2 = grads
                        .entry("reward_head.fc2.weight".to_string())
                        .or_insert_with(|| Array2::zeros((h, 1)));
                    for j in 0..h {
                        gw2[[j, 0]] += pre[j].max(0.0) * upstream;
                    }
                }
                {
                    let gb2 = grads
                        .entry("reward_head.fc2.bias".to_string())
                        .or_insert_with(|| Array2::zeros((1, 1)));
                    gb2[[0, 0]] += upstream;
                }
                {
                    let gw1 = grads
                        .entry("reward_head.fc1.weight".to_string())
                        .or_insert_with(|| Array2::zeros((d, h)));
                    for j in 0..h {
                        if pre[j] <= 0.0 {
                            continue; // relu gate is closed
                        }
                        let dz = w2[[j, 0]] * upstream;
                        for k in 0..d {
                            gw1[[k, j]] += features[k] * dz;
                        }
                    }
                }
                {
                    let gb1 = grads
                        .entry("reward_head.fc1.bias".to_string())
                        .or_insert_with(|| Array2::zeros((1, h)));
                    for j in 0..h {
                        if pre[j] > 0.0 {
                            gb1[[0, j]] += w2[[j, 0]] * upstream;
                        }
                    }
                }
            },
            RewardModelType::Transformer => {
                let a = self.param("reward_head.attention.weight")?.clone();
                let o = self.param("reward_head.output.weight")?.clone();

                let mut u = vec![0.0f32; d];
                for (j, slot) in u.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for k in 0..d {
                        acc += features[k] * a[[k, j]];
                    }
                    *slot = acc.tanh();
                }

                {
                    let go = grads
                        .entry("reward_head.output.weight".to_string())
                        .or_insert_with(|| Array2::zeros((d, 1)));
                    for j in 0..d {
                        go[[j, 0]] += u[j] * upstream;
                    }
                }
                {
                    let ga = grads
                        .entry("reward_head.attention.weight".to_string())
                        .or_insert_with(|| Array2::zeros((d, d)));
                    for j in 0..d {
                        let dpre = o[[j, 0]] * upstream * (1.0 - u[j] * u[j]);
                        for k in 0..d {
                            ga[[k, j]] += features[k] * dpre;
                        }
                    }
                }
            },
            RewardModelType::Ensemble => {
                let share = upstream / 3.0;
                for i in 0..3 {
                    {
                        let gw = grads
                            .entry(format!("reward_head_{}.weight", i))
                            .or_insert_with(|| Array2::zeros((d, 1)));
                        for k in 0..d {
                            gw[[k, 0]] += features[k] * share;
                        }
                    }
                    {
                        let gb = grads
                            .entry(format!("reward_head_{}.bias", i))
                            .or_insert_with(|| Array2::zeros((1, 1)));
                        gb[[0, 0]] += share;
                    }
                }
            },
        }
        Ok(())
    }

    /// Apply one SGD step: `p -= learning_rate * grad`.
    fn apply_gradients(&mut self, grads: &HashMap<String, Array2<f32>>) -> Result<()> {
        let lr = self.config.learning_rate as f32;
        for (key, grad) in grads {
            let param = self.parameters.get_mut(key).ok_or_else(|| {
                anyhow::anyhow!("gradient computed for unknown parameter '{key}'")
            })?;
            if param.dim() != grad.dim() {
                return Err(anyhow::anyhow!(
                    "gradient for '{key}' has shape {:?} but the parameter has shape {:?}",
                    grad.dim(),
                    param.dim()
                ));
            }
            *param -= &(grad * lr);
        }
        Ok(())
    }

    /// Score a batch of tokenized sequences with the real head parameters.
    fn forward(&self, tokens: &Array2<u32>) -> Result<Array1<f32>> {
        let batch_size = tokens.shape()[0];
        let mut rewards = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let row: Vec<u32> = tokens.row(i).iter().copied().collect();
            let features = self.featurize(&row)?;
            let score = self.score_features(&features)?;
            if !score.is_finite() {
                return Err(anyhow::anyhow!(
                    "reward model produced a non-finite score for batch element {i}"
                ));
            }
            rewards[i] = score;
        }
        Ok(rewards)
    }

    fn calculate_ranking_loss(&self, chosen: &Array1<f32>, rejected: &Array1<f32>) -> Result<f32> {
        let mut total_loss = 0.0;
        let batch_size = chosen.len();

        for i in 0..batch_size {
            // Ranking loss: -log(sigmoid(chosen - rejected))
            let score_diff = chosen[i] - rejected[i] - self.config.margin as f32;
            let sigmoid = 1.0 / (1.0 + (-score_diff).exp());
            total_loss -= sigmoid.ln();
        }

        Ok(total_loss / batch_size as f32)
    }

    fn calculate_accuracy(&self, chosen: &Array1<f32>, rejected: &Array1<f32>) -> Result<f32> {
        let mut correct = 0;
        let batch_size = chosen.len();

        for i in 0..batch_size {
            if chosen[i] > rejected[i] {
                correct += 1;
            }
        }

        Ok(correct as f32 / batch_size as f32)
    }

    fn calculate_confidence(&self, score: f32) -> Result<f32> {
        // Simple confidence based on score magnitude
        Ok((score.abs() / 2.0).min(1.0))
    }

    /// Backward pass for the pairwise Bradley–Terry ranking loss.
    ///
    /// With `d = s_chosen − s_rejected − margin` and `L = −log σ(d)`:
    ///
    /// ```text
    /// dL/dd          = −(1 − σ(d))
    /// dL/ds_chosen   =  dL/dd
    /// dL/ds_rejected = −dL/dd
    /// ```
    ///
    /// Those upstream derivatives are pushed through the head with
    /// [`RewardModel::accumulate_head_gradients`], averaged over the batch and applied with
    /// [`RewardModel::apply_gradients`]. Training therefore genuinely changes what
    /// [`RewardModel::forward`] returns.
    fn backward_pass(
        &mut self,
        chosen_tokens: &Array2<u32>,
        rejected_tokens: &Array2<u32>,
    ) -> Result<()> {
        let batch_size = chosen_tokens.shape()[0];
        if batch_size == 0 {
            return Err(anyhow::anyhow!(
                "cannot train a reward model on an empty batch"
            ));
        }
        if rejected_tokens.shape()[0] != batch_size {
            return Err(anyhow::anyhow!(
                "chosen/rejected batches disagree: {} vs {}",
                batch_size,
                rejected_tokens.shape()[0]
            ));
        }

        let margin = self.config.margin as f32;
        let scale = 1.0 / batch_size as f32;
        let mut grads: HashMap<String, Array2<f32>> = HashMap::new();

        for i in 0..batch_size {
            let chosen_row: Vec<u32> = chosen_tokens.row(i).iter().copied().collect();
            let rejected_row: Vec<u32> = rejected_tokens.row(i).iter().copied().collect();

            let chosen_features = self.featurize(&chosen_row)?;
            let rejected_features = self.featurize(&rejected_row)?;

            let s_chosen = self.score_features(&chosen_features)?;
            let s_rejected = self.score_features(&rejected_features)?;

            let diff = s_chosen - s_rejected - margin;
            let sigma = 1.0 / (1.0 + (-diff).exp());
            let dl_dd = -(1.0 - sigma) * scale;

            self.accumulate_head_gradients(&chosen_features, dl_dd, &mut grads)?;
            self.accumulate_head_gradients(&rejected_features, -dl_dd, &mut grads)?;
        }

        self.apply_gradients(&grads)
    }

    /// Hashing tokenizer: whitespace words are mapped into `[1, vocab_size)` with FNV-1a.
    ///
    /// Id `0` is reserved for padding and is therefore never produced.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let vocab_size = self.tokenizer.as_ref().map(|t| t.vocab_size).unwrap_or(50_000).max(2);
        let max_length =
            self.tokenizer.as_ref().map(|t| t.max_length).unwrap_or(self.config.max_length);
        let tokens: Vec<u32> = text
            .split_whitespace()
            .map(|word| (fnv1a64(word.as_bytes()) % (vocab_size as u64 - 1)) as u32 + 1)
            .take(max_length.max(1))
            .collect();
        if tokens.is_empty() {
            // An all-whitespace input still needs a well-defined (empty-content) sequence.
            return Ok(vec![0]);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_from(model: &RewardModel, text: &str) -> Array2<u32> {
        let t = model.tokenize(text).expect("tokenize failed");
        let len = t.len();
        Array2::from_shape_vec((1, len), t).expect("shape failed")
    }

    // ── Real forward / backward ───────────────────────────────────────────────

    #[test]
    fn test_forward_uses_head_parameters_not_sequence_length() {
        // Regression: the old forward scored by `seq_len / 100 + mean(token % 10) / 10 - 0.5`
        // and never read `self.parameters`. Zeroing the head must now zero the score.
        let mut model = RewardModel::new(RewardModelConfig::default()).expect("model");
        model.initialize_parameters(32).expect("init");

        let tokens = tokens_from(&model, "the quick brown fox");
        let before = model.forward(&tokens).expect("forward")[0];

        for (_, p) in model.parameters.iter_mut() {
            p.fill(0.0);
        }
        let after = model.forward(&tokens).expect("forward")[0];

        assert!(
            after.abs() < 1e-6,
            "with all-zero head parameters the score must be 0, got {after}"
        );
        assert!(
            (before - after).abs() > 1e-9,
            "the score must depend on the head parameters"
        );
    }

    #[test]
    fn test_forward_matches_a_hand_computed_linear_head() {
        let mut model = RewardModel::new(RewardModelConfig::default()).expect("model");
        model.initialize_parameters(8).expect("init");

        let tokens = tokens_from(&model, "alpha beta gamma");
        let row: Vec<u32> = tokens.row(0).iter().copied().collect();
        let features = model.featurize(&row).expect("featurize");

        let w = model.parameters.get("reward_head.weight").expect("weight");
        let b = model.parameters.get("reward_head.bias").expect("bias");
        let expected: f32 = b[[0, 0]] + (0..8).map(|k| features[k] * w[[k, 0]]).sum::<f32>();

        let got = model.forward(&tokens).expect("forward")[0];
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_featurize_is_l2_normalised_and_content_dependent() {
        let mut model = RewardModel::new(RewardModelConfig::default()).expect("model");
        model.initialize_parameters(16).expect("init");

        let a = model.featurize(&[7, 9, 11]).expect("featurize a");
        let b = model.featurize(&[7, 9, 12]).expect("featurize b");
        let norm = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "features must be unit norm, got {norm}"
        );
        assert!(
            a != b,
            "different token sequences must give different features"
        );

        let padded = model.featurize(&[7, 9, 11, 0, 0]).expect("featurize padded");
        assert_eq!(a, padded, "padding must not change the features");
    }

    #[test]
    fn test_train_step_changes_the_model_output() {
        // Regression: `backward_pass` was `Ok(())`, so training could never change anything.
        let config = RewardModelConfig {
            learning_rate: 0.5,
            ..RewardModelConfig::default()
        };
        let mut model = RewardModel::new(config).expect("model");
        model.initialize_parameters(24).expect("init");

        let chosen = tokens_from(&model, "a genuinely helpful and accurate answer");
        let rejected = tokens_from(&model, "no");
        // Both batches must have the same width for Array2 construction below.
        let chosen_vec: Vec<u32> = chosen.row(0).iter().copied().collect();
        let mut rejected_vec: Vec<u32> = rejected.row(0).iter().copied().collect();
        rejected_vec.resize(chosen_vec.len(), 0);
        let width = chosen_vec.len();
        let batch = RewardBatch {
            prompts: vec!["q".into()],
            chosen_responses: vec!["good".into()],
            rejected_responses: vec!["bad".into()],
            chosen_tokens: Array2::from_shape_vec((1, width), chosen_vec.clone())
                .expect("chosen shape"),
            rejected_tokens: Array2::from_shape_vec((1, width), rejected_vec.clone())
                .expect("rejected shape"),
            labels: Array1::from_vec(vec![1.0]),
        };

        let before = model.forward(&batch.chosen_tokens).expect("forward")[0];
        let first = model.train_step(&batch).expect("train_step");
        let after = model.forward(&batch.chosen_tokens).expect("forward")[0];

        assert!(
            (before - after).abs() > 1e-6,
            "train_step must change the model output ({before} -> {after})"
        );

        // Repeated steps must reduce the ranking loss on the same batch.
        let mut last = first.loss;
        for _ in 0..40 {
            let out = model.train_step(&batch).expect("train_step");
            last = out.loss;
        }
        assert!(
            last < first.loss,
            "the ranking loss must decrease with training ({} -> {last})",
            first.loss
        );
        assert!(
            model.forward(&batch.chosen_tokens).expect("forward")[0]
                > model.forward(&batch.rejected_tokens).expect("forward")[0],
            "after training the chosen response must score higher"
        );
    }

    #[test]
    fn test_train_step_reduces_loss_for_every_head_type() {
        for model_type in [
            RewardModelType::Linear,
            RewardModelType::MLP,
            RewardModelType::Transformer,
            RewardModelType::Ensemble,
        ] {
            let config = RewardModelConfig {
                model_type,
                learning_rate: 0.5,
                reward_head_hidden_size: 8,
                ..RewardModelConfig::default()
            };
            let mut model = RewardModel::new(config).expect("model");
            model.initialize_parameters(16).expect("init");

            let batch = RewardBatch {
                prompts: vec!["q".into()],
                chosen_responses: vec!["good".into()],
                rejected_responses: vec!["bad".into()],
                chosen_tokens: Array2::from_shape_vec((1, 4), vec![3u32, 11, 27, 5])
                    .expect("chosen"),
                rejected_tokens: Array2::from_shape_vec((1, 4), vec![9u32, 2, 14, 0])
                    .expect("rejected"),
                labels: Array1::from_vec(vec![1.0]),
            };

            let first = model.train_step(&batch).expect("train_step").loss;
            let mut last = first;
            for _ in 0..60 {
                last = model.train_step(&batch).expect("train_step").loss;
            }
            assert!(
                last < first,
                "{model_type:?} head must learn: loss {first} -> {last}"
            );
        }
    }

    #[test]
    fn test_forward_errors_before_initialization() {
        let model = RewardModel::new(RewardModelConfig::default()).expect("model");
        let tokens = Array2::from_shape_vec((1, 3), vec![1u32, 2, 3]).expect("shape");
        assert!(
            model.forward(&tokens).is_err(),
            "scoring without parameters must error instead of inventing a reward"
        );
    }

    #[test]
    fn test_tokenize_never_emits_padding_for_real_words() {
        let model = RewardModel::new(RewardModelConfig::default()).expect("model");
        let tokens = model.tokenize("hello there world").expect("tokenize");
        assert_eq!(tokens.len(), 3);
        assert!(
            tokens.iter().all(|&t| t != 0),
            "id 0 is reserved for padding"
        );
        let again = model.tokenize("hello there world").expect("tokenize");
        assert_eq!(tokens, again, "tokenization must be deterministic");
    }

    #[test]
    fn test_reward_model_creation() {
        let config = RewardModelConfig::default();
        let model = RewardModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_reward_prediction() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("operation failed in test");
        model.initialize_parameters(768).expect("operation failed in test");

        let prediction = model.predict_reward("This is a test response");
        assert!(prediction.is_ok());

        let pred = prediction.expect("operation failed in test");
        assert!(!pred.text.is_empty());
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_preference_pair_conversion() {
        let feedback = vec![
            HumanFeedback {
                id: "1".to_string(),
                prompt: "What is AI?".to_string(),
                response: "AI is artificial intelligence".to_string(),
                rating: 5.0,
                feedback_text: None,
                timestamp: chrono::Utc::now(),
                annotator_id: None,
                metadata: HashMap::new(),
            },
            HumanFeedback {
                id: "2".to_string(),
                prompt: "What is AI?".to_string(),
                response: "AI is bad".to_string(),
                rating: 2.0,
                feedback_text: None,
                timestamp: chrono::Utc::now(),
                annotator_id: None,
                metadata: HashMap::new(),
            },
        ];

        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("operation failed in test");
        let result = model.load_from_human_feedback(feedback);
        assert!(result.is_ok());
        assert_eq!(model.training_data.len(), 1);
    }

    #[test]
    fn test_response_comparison() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("operation failed in test");
        model.initialize_parameters(768).expect("operation failed in test");

        let preference = model.compare_responses(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "I don't know.",
        );

        assert!(preference.is_ok());
        let prob = preference.expect("operation failed in test");
        assert!((0.0..=1.0).contains(&prob));
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_reward_model_config_default() {
        let cfg = RewardModelConfig::default();
        assert!(cfg.max_length > 0);
        assert!(cfg.epochs > 0);
        assert!(cfg.batch_size > 0);
    }

    #[test]
    fn test_reward_prediction_score_in_valid_range() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(768).expect("init failed");
        let pred = model.predict_reward("hello world").expect("prediction failed");
        // score may be any real but should be finite
        assert!(pred.score.is_finite());
    }

    #[test]
    fn test_reward_prediction_confidence_range() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(768).expect("init failed");
        let pred = model.predict_reward("test").expect("prediction failed");
        assert!(
            pred.confidence >= 0.0 && pred.confidence <= 1.0,
            "confidence should be in [0,1], got {}",
            pred.confidence
        );
    }

    #[test]
    fn test_reward_model_predict_batch_empty() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(64).expect("init failed");
        let result = model.predict_batch(&[]);
        assert!(result.is_ok());
        let preds = result.expect("predict_batch should succeed");
        assert!(preds.is_empty());
    }

    #[test]
    fn test_reward_model_predict_batch_multiple_texts() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(64).expect("init failed");
        let texts: Vec<String> = vec!["text one".into(), "text two".into(), "text three".into()];
        let preds = model.predict_batch(&texts).expect("predict_batch failed");
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_reward_model_compare_responses_returns_probability() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(128).expect("init failed");
        let p = model
            .compare_responses("prompt", "response a", "response b")
            .expect("compare failed");
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_reward_model_statistics_initially_empty() {
        let config = RewardModelConfig::default();
        let model = RewardModel::new(config).expect("model creation failed");
        let stats = model.get_statistics();
        assert_eq!(stats.training_steps, 0);
        assert!(stats.losses.is_empty());
        assert!(stats.accuracies.is_empty());
    }

    #[test]
    fn test_reward_training_result_structure() {
        let result = RewardTrainingResult {
            loss: 0.5,
            accuracy: 0.8,
            avg_chosen_reward: 1.2,
            avg_rejected_reward: -0.3,
            reward_margin: 1.5,
        };
        assert!(result.reward_margin > 0.0, "chosen should exceed rejected");
        assert!(
            (result.reward_margin - (result.avg_chosen_reward - result.avg_rejected_reward)).abs()
                < 1e-5
        );
    }

    #[test]
    fn test_reward_prediction_component_scores_nonempty() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(64).expect("init failed");
        let pred = model.predict_reward("hello").expect("predict failed");
        assert!(!pred.component_scores.is_empty());
    }

    #[test]
    fn test_reward_model_load_training_data_empty() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        let result = model.load_training_data(vec![]);
        assert!(result.is_ok(), "loading empty data should succeed");
        assert_eq!(model.training_data.len(), 0);
    }

    #[test]
    fn test_reward_model_mlp_type_initializes() {
        let config = RewardModelConfig {
            model_type: RewardModelType::MLP,
            ..RewardModelConfig::default()
        };
        let mut model = RewardModel::new(config).expect("model creation failed");
        let result = model.initialize_parameters(128);
        assert!(result.is_ok(), "MLP init should succeed");
        assert!(!model.parameters.is_empty());
    }

    #[test]
    fn test_reward_model_statistics_after_step_nonempty() {
        use scirs2_core::ndarray::{Array1, Array2};
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).expect("model creation failed");
        model.initialize_parameters(64).expect("init failed");

        let batch = RewardBatch {
            prompts: vec!["q".into()],
            chosen_responses: vec!["good".into()],
            rejected_responses: vec!["bad".into()],
            chosen_tokens: Array2::zeros((1, 5)),
            rejected_tokens: Array2::zeros((1, 5)),
            labels: Array1::from_vec(vec![1.0]),
        };
        let result = model.train_step(&batch);
        assert!(result.is_ok(), "train_step should succeed");
        let stats = model.get_statistics();
        assert_eq!(stats.training_steps, 1);
    }
}
