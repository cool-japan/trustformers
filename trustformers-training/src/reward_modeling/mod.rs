//! Reward model training for RLHF.
//!
//! Two complementary pieces:
//!
//! * [`RmRewardModel`] — a self-contained **bag-of-words** reward model: a learned linear
//!   head over hashed token features, trained on preference pairs with the real
//!   Bradley-Terry / regression gradients. It has no transformer backbone, and does not
//!   claim one; `RmRewardModelConfig::num_layers` and `dropout_rate` are carried for
//!   interoperability and are explicitly unused.
//! * [`training`] — the pieces for a reward head on top of a **real encoder**:
//!   [`pool_hidden_states`], [`compute_reward_score`] and [`batch_reward_loss`] take hidden
//!   states you produce with an actual model.
//!
//! Plus [`PreferenceDataset`] and friends for managing human preference data.

pub mod training;
pub use training::{
    batch_reward_loss, compute_reward_score, pool_hidden_states, reward_loss, PoolingType,
    RewardLossType as TrainingRewardLossType, RewardModelConfig as RmTrainingConfig,
    RewardModelTrainingExample, RewardTrainError, RewardTrainingOutput,
};

use std::fmt;

// ──────────────────────────────────────────────
// PreferencePair
// ──────────────────────────────────────────────

/// A single human preference datapoint: a prompt with a chosen (preferred) and
/// a rejected response.
#[derive(Debug, Clone)]
pub struct RmPreferencePair {
    /// The prompt that elicited both responses.
    pub prompt: String,
    /// The preferred response.
    pub chosen: String,
    /// The rejected response.
    pub rejected: String,
    /// Optional gold score for the chosen response.
    pub chosen_score: Option<f64>,
    /// Optional gold score for the rejected response.
    pub rejected_score: Option<f64>,
}

impl RmPreferencePair {
    /// Construct a preference pair without gold scores.
    pub fn new(
        prompt: impl Into<String>,
        chosen: impl Into<String>,
        rejected: impl Into<String>,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            chosen: chosen.into(),
            rejected: rejected.into(),
            chosen_score: None,
            rejected_score: None,
        }
    }

    /// Score margin (chosen_score − rejected_score), available only when both
    /// gold scores are present.
    pub fn margin(&self) -> Option<f64> {
        match (self.chosen_score, self.rejected_score) {
            (Some(c), Some(r)) => Some(c - r),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────
// RewardLossType
// ──────────────────────────────────────────────

/// Supported loss functions for reward model training.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardLossType {
    /// Standard Bradley-Terry: `−log σ(r_chosen − r_rejected)`.
    BradleyTerry,
    /// Bradley-Terry with a required margin:
    /// `−log σ(r_chosen − r_rejected − margin)`.
    BradleyTerryWithMargin,
    /// Regression: `(r_chosen − 1)² + (r_rejected + 1)²`.
    Regression,
}

// ──────────────────────────────────────────────
// RewardModelConfig
// ──────────────────────────────────────────────

/// Configuration for the simplified reward model.
#[derive(Debug, Clone)]
pub struct RmRewardModelConfig {
    /// Dimensionality of the hidden token representation.
    pub hidden_size: usize,
    /// Number of transformer layers (informational; not used in the simplified model).
    pub num_layers: usize,
    /// Dropout rate (informational).
    pub dropout_rate: f64,
    /// Minimum reward margin applied in `BradleyTerryWithMargin` loss.
    pub margin: f64,
    /// Which loss function to use during training.
    pub loss_type: RewardLossType,
    /// Whether to z-score–normalise rewards before computing loss statistics.
    pub normalize_rewards: bool,
}

impl Default for RmRewardModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            dropout_rate: 0.1,
            margin: 0.0,
            loss_type: RewardLossType::BradleyTerry,
            normalize_rewards: false,
        }
    }
}

// ──────────────────────────────────────────────
// RewardModel
// ──────────────────────────────────────────────

/// A bag-of-words reward model: a learned linear head over hashed token features.
///
/// This is a *real*, if small, model — not a stand-in. Text is turned into a
/// `hidden_size`-dimensional feature vector with the signed hashing trick (see
/// [`RmRewardModel::embed`]) and scored by a linear head, so the score is a genuine
/// function of the model's parameters and [`RmRewardModel::train_step`] genuinely changes
/// it. It carries no contextual/transformer backbone; for that, score pooled hidden states
/// from a real encoder with [`compute_reward_score`] instead.
///
/// Its predecessor hashed the whole string into `[-2, 2]` and ignored `reward_head`
/// entirely, so training was impossible and every "reward" was an artefact of the hash.
pub struct RmRewardModel {
    config: RmRewardModelConfig,
    /// Linear reward head weights (length = hidden_size).
    reward_head: Vec<f64>,
    /// Bias of the linear reward head.
    bias: f64,
}

impl RmRewardModel {
    /// Construct a new reward model with a zero-initialised head.
    ///
    /// A zero head scores every text `bias` (i.e. `0.0`); it is
    /// [`train_step`](Self::train_step) that gives the model its preferences. The linear
    /// objective is convex, so a zero start is not a symmetry problem.
    pub fn new(config: RmRewardModelConfig) -> Self {
        let hidden_size = config.hidden_size.max(1);
        Self {
            config,
            reward_head: vec![0.0; hidden_size],
            bias: 0.0,
        }
    }

    /// The model's current head weights.
    pub fn reward_head(&self) -> &[f64] {
        &self.reward_head
    }

    /// The model's current bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Overwrite the head weights.
    ///
    /// # Errors
    ///
    /// `weights.len() != config.hidden_size`.
    pub fn set_reward_head(&mut self, weights: Vec<f64>) -> Result<(), RewardError> {
        if weights.len() != self.reward_head.len() {
            return Err(RewardError::DimensionMismatch {
                expected: self.reward_head.len(),
                actual: weights.len(),
            });
        }
        self.reward_head = weights;
        Ok(())
    }

    /// Feature vector for `text`: L2-normalised signed token-hash counts.
    ///
    /// Tokens are maximal runs of alphanumeric characters, lowercased. Each token `t` is
    /// hashed once with FNV-1a; the low bits choose the feature index and the top bit its
    /// sign — the standard signed hashing trick, whose collisions cancel in expectation
    /// instead of accumulating. The vector is L2-normalised so that a long response cannot
    /// out-score a short one purely by length.
    pub fn embed(&self, text: &str) -> Vec<f64> {
        let dim = self.reward_head.len();
        let mut features = vec![0.0f64; dim];
        for token in text.split(|c: char| !c.is_alphanumeric()).filter(|token| !token.is_empty()) {
            let hash = fnv1a(token.to_lowercase().as_bytes());
            let index = (hash % dim as u64) as usize;
            let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
            features[index] += sign;
        }
        let norm = features.iter().map(|f| f * f).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for feature in &mut features {
                *feature /= norm;
            }
        }
        features
    }

    /// Reward for `text`: `reward_head · embed(text) + bias`.
    pub fn score(&self, text: &str) -> f64 {
        let features = self.embed(text);
        let dot: f64 = features.iter().zip(self.reward_head.iter()).map(|(f, w)| f * w).sum();
        dot + self.bias
    }

    /// Score a slice of texts.
    pub fn score_batch(&self, texts: &[&str]) -> Vec<f64> {
        texts.iter().map(|t| self.score(t)).collect()
    }

    /// One full-batch gradient-descent step on the configured loss.
    ///
    /// Because the score is linear in the parameters, `∂L/∂w = ∂L/∂r · φ(text)` with
    /// `φ = embed(text)`, and the exact gradient of each supported loss is:
    ///
    /// ```text
    /// BradleyTerry(+Margin):  ∂L/∂r_chosen = −σ(−d),  ∂L/∂r_rejected = +σ(−d)
    ///                          with d = r_chosen − r_rejected (− margin)
    /// Regression:             ∂L/∂r_chosen = 2(r_chosen − 1),
    ///                          ∂L/∂r_rejected = 2(r_rejected + 1)
    /// ```
    ///
    /// The bias gradient cancels for the Bradley-Terry losses (they depend only on the
    /// score *difference*), which is why the bias only moves under `Regression`.
    ///
    /// Returns the loss statistics measured **before** the update, so a caller can watch
    /// them fall across steps.
    ///
    /// # Errors
    ///
    /// `pairs` is empty, `learning_rate` is not finite and positive, or a score is
    /// non-finite.
    pub fn train_step(
        &mut self,
        pairs: &[RmPreferencePair],
        learning_rate: f64,
    ) -> Result<RewardLossResult, RewardError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(RewardError::InvalidHyperparameter(format!(
                "learning_rate must be finite and positive, got {learning_rate}"
            )));
        }
        let stats = self.compute_loss(pairs)?;

        let dim = self.reward_head.len();
        let mut grad_head = vec![0.0f64; dim];
        let mut grad_bias = 0.0f64;

        for pair in pairs {
            let chosen_features = self.embed(&pair.chosen);
            let rejected_features = self.embed(&pair.rejected);
            let chosen_score = self.score(&pair.chosen);
            let rejected_score = self.score(&pair.rejected);

            let (d_chosen, d_rejected) = match self.config.loss_type {
                RewardLossType::BradleyTerry => {
                    let diff = chosen_score - rejected_score;
                    let g = sigmoid(-diff);
                    (-g, g)
                },
                RewardLossType::BradleyTerryWithMargin => {
                    let diff = chosen_score - rejected_score - self.config.margin;
                    let g = sigmoid(-diff);
                    (-g, g)
                },
                RewardLossType::Regression => {
                    (2.0 * (chosen_score - 1.0), 2.0 * (rejected_score + 1.0))
                },
            };

            for i in 0..dim {
                grad_head[i] += d_chosen * chosen_features[i] + d_rejected * rejected_features[i];
            }
            grad_bias += d_chosen + d_rejected;
        }

        let scale = learning_rate / pairs.len() as f64;
        for i in 0..dim {
            self.reward_head[i] -= scale * grad_head[i];
        }
        self.bias -= scale * grad_bias;

        if !self.reward_head.iter().all(|w| w.is_finite()) || !self.bias.is_finite() {
            return Err(RewardError::InvalidScore);
        }
        Ok(stats)
    }

    /// Run `epochs` full-batch [`train_step`](Self::train_step)s, returning the statistics
    /// measured at the start of each one.
    pub fn train(
        &mut self,
        pairs: &[RmPreferencePair],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<Vec<RewardLossResult>, RewardError> {
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            history.push(self.train_step(pairs, learning_rate)?);
        }
        Ok(history)
    }

    /// Compute the configured loss over a batch of preference pairs.
    pub fn compute_loss(
        &self,
        pairs: &[RmPreferencePair],
    ) -> Result<RewardLossResult, RewardError> {
        if pairs.is_empty() {
            return Err(RewardError::EmptyBatch);
        }

        let mut total_loss = 0.0_f64;
        let mut total_chosen = 0.0_f64;
        let mut total_rejected = 0.0_f64;
        let mut correct = 0_usize;
        let mut total_margin = 0.0_f64;

        for pair in pairs {
            let rc = self.score(&pair.chosen);
            let rr = self.score(&pair.rejected);

            if !rc.is_finite() || !rr.is_finite() {
                return Err(RewardError::InvalidScore);
            }

            let loss = match self.config.loss_type {
                RewardLossType::BradleyTerry => {
                    let diff = rc - rr;
                    -sigmoid(diff).ln()
                },
                RewardLossType::BradleyTerryWithMargin => {
                    let diff = rc - rr - self.config.margin;
                    -sigmoid(diff).ln()
                },
                RewardLossType::Regression => (rc - 1.0).powi(2) + (rr + 1.0).powi(2),
            };

            total_loss += loss;
            total_chosen += rc;
            total_rejected += rr;
            total_margin += rc - rr;
            if rc > rr {
                correct += 1;
            }
        }

        let n = pairs.len() as f64;
        Ok(RewardLossResult {
            mean_loss: total_loss / n,
            mean_chosen_score: total_chosen / n,
            mean_rejected_score: total_rejected / n,
            accuracy: correct as f64 / n,
            mean_margin: total_margin / n,
            batch_size: pairs.len(),
        })
    }

    /// Fraction of preference pairs where `score(chosen) > score(rejected)`.
    pub fn pairwise_accuracy(&self, pairs: &[RmPreferencePair]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }
        let correct =
            pairs.iter().filter(|p| self.score(&p.chosen) > self.score(&p.rejected)).count();
        correct as f64 / pairs.len() as f64
    }

    /// Z-score normalise a vector of reward scores.
    ///
    /// If variance is zero (all scores identical) the input is returned unchanged.
    pub fn normalize_rewards(&self, scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        if std < 1e-12 {
            return scores.to_vec();
        }
        scores.iter().map(|s| (s - mean) / std).collect()
    }
}

/// Logistic sigmoid: σ(x) = 1 / (1 + e^{-x}).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 64-bit FNV-1a, used to place a token in the feature vector.
///
/// FNV-1a mixes every byte into both the low bits (which pick the feature index) and the
/// high bit (which picks the sign), which a multiplicative hash like djb2 does not.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ──────────────────────────────────────────────
// BradleyTerryModel
// ──────────────────────────────────────────────

/// Bradley-Terry preference model for reward comparisons.
///
/// Models the probability that response `y_w` is preferred over `y_l` as:
/// `P(y_w > y_l) = σ((r_w − r_l) / temperature)`
#[derive(Debug, Clone)]
pub struct BradleyTerryModel {
    /// Temperature scaling factor (default 1.0).
    pub temperature: f32,
}

impl BradleyTerryModel {
    /// Create a new Bradley-Terry model with the given temperature.
    pub fn new(temperature: f32) -> Result<Self, RewardError> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(RewardError::InvalidScore);
        }
        Ok(Self { temperature })
    }

    /// Create with default temperature = 1.0.
    pub fn default_temperature() -> Self {
        Self { temperature: 1.0 }
    }

    /// Compute the preference probability P(y_w > y_l).
    ///
    /// `P = σ((r_w − r_l) / T)`
    pub fn preference_probability(&self, reward_w: f32, reward_l: f32) -> f32 {
        let diff = (reward_w - reward_l) / self.temperature;
        1.0 / (1.0 + (-diff).exp())
    }

    /// Compute the Bradley-Terry loss for a preference pair.
    ///
    /// `BT_loss = −log P(y_w > y_l) = −log σ((r_w − r_l) / T)`
    pub fn bt_loss(&self, reward_w: f32, reward_l: f32) -> f32 {
        let p = self.preference_probability(reward_w, reward_l);
        -p.max(1e-12).ln()
    }

    /// Compute BT loss over a batch of (reward_w, reward_l) pairs.
    pub fn batch_bt_loss(&self, pairs: &[(f32, f32)]) -> Result<f32, RewardError> {
        if pairs.is_empty() {
            return Err(RewardError::EmptyBatch);
        }
        let total: f32 = pairs.iter().map(|&(rw, rl)| self.bt_loss(rw, rl)).sum();
        Ok(total / pairs.len() as f32)
    }
}

// ──────────────────────────────────────────────
// RewardNormalizer
// ──────────────────────────────────────────────

/// Online normalizer for reward scores using Welford's algorithm.
///
/// Maintains running mean and variance so that rewards can be normalised to
/// approximately zero mean / unit variance at any point during training.
#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    /// Current estimate of the mean.
    pub mean: f32,
    /// Current estimate of the standard deviation.
    pub std: f32,
    /// Optional clipping range `(min, max)` applied after normalisation.
    pub clip_range: Option<(f32, f32)>,
    /// Number of samples seen so far (Welford state).
    count: u64,
    /// Running sum of squared deviations from the mean (Welford M2).
    m2: f32,
}

impl RewardNormalizer {
    /// Create a normalizer initialised from a non-empty slice of samples.
    pub fn new_from_samples(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std: 1.0,
                clip_range: None,
                count: 0,
                m2: 0.0,
            };
        }
        let n = samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / n;
        let variance = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
        let std = variance.sqrt().max(1e-8);
        Self {
            mean,
            std,
            clip_range: None,
            count: samples.len() as u64,
            m2: variance * n,
        }
    }

    /// Create a normalizer with no prior samples.
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            clip_range: None,
            count: 0,
            m2: 0.0,
        }
    }

    /// Normalise a single reward: `(reward − mean) / std`, then clip.
    pub fn normalize(&self, reward: f32) -> f32 {
        let normed = (reward - self.mean) / self.std.max(1e-8);
        match self.clip_range {
            Some((lo, hi)) => normed.max(lo).min(hi),
            None => normed,
        }
    }

    /// Update the running statistics with a new reward observation (Welford).
    pub fn update(&mut self, new_reward: f32) {
        self.count += 1;
        let delta = new_reward - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = new_reward - self.mean;
        self.m2 += delta * delta2;
        if self.count >= 2 {
            let variance = self.m2 / self.count as f32;
            self.std = variance.sqrt().max(1e-8);
        }
    }

    /// Number of samples seen so far.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for RewardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────
// margin_ranking_loss
// ──────────────────────────────────────────────

/// Margin ranking loss: `max(0, margin − (r_w − r_l))`.
///
/// Encourages `r_w − r_l ≥ margin` by penalising pairs where the gap is
/// smaller than the target margin.
pub fn margin_ranking_loss(reward_w: f32, reward_l: f32, margin: f32) -> f32 {
    (margin - (reward_w - reward_l)).max(0.0)
}

// ──────────────────────────────────────────────
// RewardLossResult
// ──────────────────────────────────────────────

/// Aggregated result of a reward model loss computation over a batch.
#[derive(Debug, Clone)]
pub struct RewardLossResult {
    /// Mean loss over the batch.
    pub mean_loss: f64,
    /// Mean reward assigned to chosen responses.
    pub mean_chosen_score: f64,
    /// Mean reward assigned to rejected responses.
    pub mean_rejected_score: f64,
    /// Pairwise accuracy: fraction where `score(chosen) > score(rejected)`.
    pub accuracy: f64,
    /// Mean reward margin `r_chosen − r_rejected`.
    pub mean_margin: f64,
    /// Number of preference pairs in the batch.
    pub batch_size: usize,
}

impl fmt::Display for RewardLossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RewardLossResult {{ mean_loss: {:.4}, chosen: {:.4}, rejected: {:.4}, \
             accuracy: {:.4}, mean_margin: {:.4}, batch_size: {} }}",
            self.mean_loss,
            self.mean_chosen_score,
            self.mean_rejected_score,
            self.accuracy,
            self.mean_margin,
            self.batch_size,
        )
    }
}

// ──────────────────────────────────────────────
// RewardError
// ──────────────────────────────────────────────

/// Errors that can arise during reward model operations.
#[derive(Debug, thiserror::Error)]
pub enum RewardError {
    /// The batch contains no preference pairs.
    #[error("Empty batch")]
    EmptyBatch,
    /// A computed score is NaN or infinite.
    #[error("Invalid score: NaN or Inf")]
    InvalidScore,
    /// A supplied vector does not match the model's `hidden_size`.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Length the model requires.
        expected: usize,
        /// Length that was supplied.
        actual: usize,
    },
    /// A training hyper-parameter is outside its valid range.
    #[error("Invalid hyper-parameter: {0}")]
    InvalidHyperparameter(String),
}

// ──────────────────────────────────────────────
// PreferenceDataset
// ──────────────────────────────────────────────

/// A collection of [`RmPreferencePair`]s with dataset-management utilities.
pub struct PreferenceDataset {
    pairs: Vec<RmPreferencePair>,
}

impl PreferenceDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Append a preference pair to the dataset.
    pub fn add(&mut self, pair: RmPreferencePair) {
        self.pairs.push(pair);
    }

    /// Number of preference pairs in the dataset.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns `true` if the dataset contains no pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Deterministic train / validation split.
    ///
    /// The first `floor(n * train_fraction)` pairs go to training; the rest to
    /// validation.  `train_fraction` is clamped to `[0, 1]`.
    pub fn split(&self, train_fraction: f64) -> (Self, Self) {
        let fraction = train_fraction.clamp(0.0, 1.0);
        let train_size = (self.pairs.len() as f64 * fraction).floor() as usize;
        let train = Self {
            pairs: self.pairs[..train_size].to_vec(),
        };
        let val = Self {
            pairs: self.pairs[train_size..].to_vec(),
        };
        (train, val)
    }

    /// Compute summary statistics over the dataset.
    pub fn statistics(&self) -> DatasetStats {
        let total_pairs = self.pairs.len();

        let avg_chosen_length = if total_pairs == 0 {
            0.0
        } else {
            self.pairs.iter().map(|p| p.chosen.len() as f64).sum::<f64>() / total_pairs as f64
        };

        let avg_rejected_length = if total_pairs == 0 {
            0.0
        } else {
            self.pairs.iter().map(|p| p.rejected.len() as f64).sum::<f64>() / total_pairs as f64
        };

        let pairs_with_scores = self
            .pairs
            .iter()
            .filter(|p| p.chosen_score.is_some() && p.rejected_score.is_some())
            .count();

        let mean_score_margin = if pairs_with_scores == 0 {
            None
        } else {
            let sum: f64 = self.pairs.iter().filter_map(|p| p.margin()).sum();
            Some(sum / pairs_with_scores as f64)
        };

        DatasetStats {
            total_pairs,
            avg_chosen_length,
            avg_rejected_length,
            pairs_with_scores,
            mean_score_margin,
        }
    }
}

impl Default for PreferenceDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────
// DatasetStats
// ──────────────────────────────────────────────

/// Summary statistics for a [`PreferenceDataset`].
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of preference pairs.
    pub total_pairs: usize,
    /// Average character length of chosen responses.
    pub avg_chosen_length: f64,
    /// Average character length of rejected responses.
    pub avg_rejected_length: f64,
    /// Number of pairs for which gold scores are available.
    pub pairs_with_scores: usize,
    /// Mean gold score margin (chosen − rejected), or `None` if unavailable.
    pub mean_score_margin: Option<f64>,
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> RmRewardModel {
        RmRewardModel::new(RmRewardModelConfig::default())
    }

    fn pair(chosen: &str, rejected: &str) -> RmPreferencePair {
        RmPreferencePair::new("prompt", chosen, rejected)
    }

    // ── Test 1: PreferencePair margin ────────────────────────────────────
    #[test]
    fn test_preference_pair_margin() {
        let mut p = pair("good", "bad");
        assert!(p.margin().is_none(), "no scores → margin should be None");
        p.chosen_score = Some(0.9);
        p.rejected_score = Some(0.3);
        let m = p.margin().expect("should have margin");
        assert!((m - 0.6).abs() < 1e-10);
    }

    // ── Test 2: score is finite ───────────────────────────────────────────
    #[test]
    fn test_score_is_finite() {
        let model = default_model();
        let s = model.score("Hello, world!");
        assert!(s.is_finite());
        assert!(
            (-2.0..=2.0).contains(&s),
            "score {} should be in [-2, 2]",
            s
        );
    }

    // ── Test 3: score_batch length ────────────────────────────────────────
    #[test]
    fn test_score_batch_length() {
        let model = default_model();
        let texts = ["a", "bb", "ccc", "dddd"];
        let scores = model.score_batch(&texts);
        assert_eq!(scores.len(), texts.len());
    }

    // ── Test 4: BradleyTerry loss > 0 ────────────────────────────────────
    #[test]
    fn test_bradley_terry_loss_positive() {
        let model = default_model();
        let pairs = vec![pair("response A chosen", "response B rejected")];
        let result = model.compute_loss(&pairs).expect("should succeed");
        assert!(
            result.mean_loss > 0.0,
            "BT loss should be positive, got {}",
            result.mean_loss
        );
    }

    // ── Test 5: BradleyTerryWithMargin larger when margin > 0 ────────────
    #[test]
    fn test_bt_with_margin_larger() {
        let config_no_margin = RmRewardModelConfig {
            loss_type: RewardLossType::BradleyTerry,
            margin: 0.0,
            ..Default::default()
        };
        let config_with_margin = RmRewardModelConfig {
            loss_type: RewardLossType::BradleyTerryWithMargin,
            margin: 1.0,
            ..Default::default()
        };
        let model_no = RmRewardModel::new(config_no_margin);
        let model_yes = RmRewardModel::new(config_with_margin);
        let pairs = vec![pair("response A chosen", "response B rejected")];
        let loss_no = model_no.compute_loss(&pairs).expect("ok").mean_loss;
        let loss_yes = model_yes.compute_loss(&pairs).expect("ok").mean_loss;
        // Subtracting a positive margin from the diff makes the loss >= the no-margin loss
        assert!(
            loss_yes >= loss_no,
            "margin loss ({}) should be >= no-margin loss ({})",
            loss_yes,
            loss_no
        );
    }

    // ── Test 6: Regression loss ───────────────────────────────────────────
    #[test]
    fn test_regression_loss() {
        let config = RmRewardModelConfig {
            loss_type: RewardLossType::Regression,
            ..Default::default()
        };
        let model = RmRewardModel::new(config);
        let pairs = vec![pair("response A chosen", "response B rejected")];
        let result = model.compute_loss(&pairs).expect("ok");
        assert!(result.mean_loss >= 0.0);
        assert!(result.mean_loss.is_finite());
    }

    /// A small preference set whose signal is a single distinguishing word.
    fn learnable_pairs() -> Vec<RmPreferencePair> {
        vec![
            pair("this answer is helpful", "this answer is harmful"),
            pair("a helpful and clear reply", "a harmful and clear reply"),
            pair(
                "helpful guidance for the user",
                "harmful guidance for the user",
            ),
        ]
    }

    // ── Test 7: pairwise_accuracy (chosen scores higher → 1.0) ───────────
    #[test]
    fn test_pairwise_accuracy_all_correct() {
        // Train first so the ordering is a property of the *model*, not of an accident of
        // the hash: an untrained (zero) head scores everything identically.
        let mut model = default_model();
        let pairs = learnable_pairs();
        model.train(&pairs, 1.0, 200).expect("training should succeed");
        let acc = model.pairwise_accuracy(&pairs);
        assert!(
            (acc - 1.0).abs() < 1e-10,
            "after training every chosen response should score higher, got {acc}"
        );
    }

    // ── Regression: the score must be a function of the model's parameters ──
    #[test]
    fn test_score_depends_on_the_reward_head() {
        // The old implementation hashed the text and ignored `reward_head` entirely, so
        // this assertion could not hold for any pair of weight vectors.
        let mut model = default_model();
        let text = "a helpful answer";
        let features = model.embed(text);
        assert_eq!(features.len(), RmRewardModelConfig::default().hidden_size);

        assert!(
            model.score(text).abs() < 1e-12,
            "a zero head scores every text at the bias (0.0)"
        );

        let head_a: Vec<f64> = features.iter().map(|f| f * 3.0).collect();
        model.set_reward_head(head_a).expect("dimension matches");
        let score_a = model.score(text);

        let head_b: Vec<f64> = features.iter().map(|f| f * -7.0).collect();
        model.set_reward_head(head_b).expect("dimension matches");
        let score_b = model.score(text);

        assert!(
            (score_a - score_b).abs() > 1e-6,
            "different weights must give different scores, got {score_a} and {score_b}"
        );
        // score = w·φ with w = 3φ and |φ| = 1, so the score is exactly 3.
        assert!((score_a - 3.0).abs() < 1e-9, "expected 3.0, got {score_a}");
        assert!((score_b + 7.0).abs() < 1e-9, "expected -7.0, got {score_b}");

        assert!(matches!(
            model.set_reward_head(vec![0.0; 3]),
            Err(RewardError::DimensionMismatch { .. })
        ));
    }

    // ── Regression: training must actually change the model ──────────────
    #[test]
    fn test_training_reduces_the_loss_and_learns_the_preference() {
        let mut model = default_model();
        let pairs = learnable_pairs();

        let before = model.compute_loss(&pairs).expect("loss");
        // A zero head cannot separate anything: diff = 0 → −ln σ(0) = ln 2.
        assert!((before.mean_loss - std::f64::consts::LN_2).abs() < 1e-9);
        assert!(before.accuracy < 1e-10);

        let history = model.train(&pairs, 1.0, 150).expect("training should succeed");
        assert_eq!(history.len(), 150);
        assert!(
            history[0].mean_loss > history[history.len() - 1].mean_loss,
            "loss must fall over training: {} -> {}",
            history[0].mean_loss,
            history[history.len() - 1].mean_loss
        );

        let after = model.compute_loss(&pairs).expect("loss");
        assert!(
            after.mean_loss < before.mean_loss,
            "loss must improve: {} -> {}",
            before.mean_loss,
            after.mean_loss
        );
        assert!(
            (after.accuracy - 1.0).abs() < 1e-10,
            "the model should separate every pair, got {}",
            after.accuracy
        );
        assert!(
            after.mean_margin > 0.0,
            "chosen responses must score above rejected ones"
        );
        // A word that never appeared keeps a zero weight — the head learned from data,
        // not from a hash of the whole string.
        let unseen = model.embed("zzzz");
        let unseen_score: f64 =
            unseen.iter().zip(model.reward_head().iter()).map(|(f, w)| f * w).sum();
        assert!(unseen_score.abs() < 1.0);
    }

    // ── embed(): tokenised, order-free and L2-normalised ─────────────────
    #[test]
    fn test_embed_is_tokenised_and_normalised() {
        let model = default_model();

        let features = model.embed("Helpful, helpful; HELPFUL");
        let norm = features.iter().map(|f| f * f).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "embedding must be L2-normalised"
        );
        // Case and punctuation are not part of the token.
        assert_eq!(model.embed("helpful helpful helpful"), features);

        // Bag of words: order does not matter, content does.
        assert_eq!(model.embed("alpha beta"), model.embed("beta alpha"));
        assert_ne!(model.embed("alpha beta"), model.embed("alpha gamma"));

        // An empty text has no tokens, hence the zero vector (norm 0, not NaN).
        assert!(model.embed("   ").iter().all(|f| *f == 0.0));
    }

    // ── train_step rejects a nonsensical learning rate ───────────────────
    #[test]
    fn test_train_step_rejects_bad_hyperparameters() {
        let mut model = default_model();
        let pairs = learnable_pairs();
        assert!(matches!(
            model.train_step(&pairs, 0.0),
            Err(RewardError::InvalidHyperparameter(_))
        ));
        assert!(matches!(
            model.train_step(&pairs, f64::NAN),
            Err(RewardError::InvalidHyperparameter(_))
        ));
        assert!(matches!(
            model.train_step(&[], 0.1),
            Err(RewardError::EmptyBatch)
        ));
    }

    // ── Regression drives absolute scores to ±1; Bradley-Terry only the gap ──
    #[test]
    fn test_regression_loss_targets_absolute_scores() {
        let pairs = learnable_pairs();

        let mut regression = RmRewardModel::new(RmRewardModelConfig {
            loss_type: RewardLossType::Regression,
            ..Default::default()
        });
        regression.train(&pairs, 0.5, 400).expect("train");
        let stats = regression.compute_loss(&pairs).expect("loss");
        assert!(
            (stats.mean_chosen_score - 1.0).abs() < 0.5,
            "regression pulls chosen scores towards +1, got {}",
            stats.mean_chosen_score
        );
        assert!(
            (stats.mean_rejected_score + 1.0).abs() < 0.5,
            "regression pulls rejected scores towards -1, got {}",
            stats.mean_rejected_score
        );

        // Bradley-Terry only maximises the *difference*: it is invariant to a constant
        // shift, so it never targets a particular level and keeps widening the margin.
        let mut bt = default_model();
        bt.train(&pairs, 1.0, 400).expect("train");
        let bt_stats = bt.compute_loss(&pairs).expect("loss");
        assert!(
            bt_stats.mean_margin > stats.mean_margin,
            "unbounded BT margin ({}) should exceed the regression margin ({})",
            bt_stats.mean_margin,
            stats.mean_margin
        );
        assert!(
            bt.bias().abs() < 1e-12,
            "the BT gradient of the bias cancels, so the bias must stay put"
        );
    }

    // ── Test 8: normalize_rewards mean≈0 / std≈1 ─────────────────────────
    #[test]
    fn test_normalize_rewards() {
        let model = default_model();
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normed = model.normalize_rewards(&scores);
        let n = normed.len() as f64;
        let mean = normed.iter().sum::<f64>() / n;
        let var = normed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-9, "mean should be ≈ 0, got {}", mean);
        assert!(
            (var.sqrt() - 1.0).abs() < 1e-9,
            "std should be ≈ 1, got {}",
            var.sqrt()
        );
    }

    // ── Test 9: RewardLossResult display ─────────────────────────────────
    #[test]
    fn test_reward_loss_result_display() {
        let result = RewardLossResult {
            mean_loss: 0.693,
            mean_chosen_score: 1.5,
            mean_rejected_score: -1.5,
            accuracy: 0.8,
            mean_margin: 3.0,
            batch_size: 10,
        };
        let s = format!("{}", result);
        assert!(
            s.contains("mean_loss"),
            "display should contain 'mean_loss'"
        );
        assert!(
            s.contains("0.6930"),
            "display should contain formatted loss"
        );
    }

    // ── Test 10: empty batch error ────────────────────────────────────────
    #[test]
    fn test_empty_batch_error() {
        let model = default_model();
        let err = model.compute_loss(&[]).unwrap_err();
        assert!(matches!(err, RewardError::EmptyBatch));
    }

    // ── Test 11: dataset add/len/split (correct sizes) ────────────────────
    #[test]
    fn test_dataset_add_len_split() {
        let mut ds = PreferenceDataset::new();
        for i in 0..10 {
            ds.add(pair(&format!("chosen {}", i), &format!("rejected {}", i)));
        }
        assert_eq!(ds.len(), 10);
        let (train, val) = ds.split(0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    // ── Test 12: DatasetStats ─────────────────────────────────────────────
    #[test]
    fn test_dataset_stats() {
        let mut ds = PreferenceDataset::new();
        let mut p1 = pair("hello world", "hi");
        p1.chosen_score = Some(1.0);
        p1.rejected_score = Some(0.0);
        let p2 = pair("foo bar baz", "qux");
        ds.add(p1);
        ds.add(p2);
        let stats = ds.statistics();
        assert_eq!(stats.total_pairs, 2);
        assert_eq!(stats.pairs_with_scores, 1);
        assert!(stats.mean_score_margin.is_some());
        assert!((stats.mean_score_margin.expect("some") - 1.0).abs() < 1e-10);
        // avg_chosen_length = mean of "hello world"(11) and "foo bar baz"(11) = 11
        assert!((stats.avg_chosen_length - 11.0).abs() < 1e-10);
    }

    // ── Test 13: PreferencePair no score ─────────────────────────────────
    #[test]
    fn test_preference_pair_no_score() {
        let p = pair("chosen", "rejected");
        assert!(p.chosen_score.is_none());
        assert!(p.rejected_score.is_none());
        assert!(p.margin().is_none());
    }

    // ── Test 14: accuracy is 1.0 in the trained order and 0.0 when reversed ──
    #[test]
    fn test_accuracy_is_inverted_when_the_pairs_are_swapped() {
        let mut model = default_model();
        let pairs = learnable_pairs();
        model.train(&pairs, 1.0, 200).expect("training should succeed");

        let reversed: Vec<RmPreferencePair> = pairs
            .iter()
            .map(|p| RmPreferencePair::new(&p.prompt, &p.rejected, &p.chosen))
            .collect();

        assert!((model.pairwise_accuracy(&pairs) - 1.0).abs() < 1e-10);
        assert!(
            model.pairwise_accuracy(&reversed) < 1e-10,
            "swapping chosen/rejected must invert the accuracy"
        );
    }

    // ── Test 15: default config ────────────────────────────────────────────
    #[test]
    fn test_default_config() {
        let cfg = RmRewardModelConfig::default();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_layers, 12);
        assert!((cfg.dropout_rate - 0.1).abs() < 1e-10);
        assert!((cfg.margin - 0.0).abs() < 1e-10);
        assert_eq!(cfg.loss_type, RewardLossType::BradleyTerry);
        assert!(!cfg.normalize_rewards);
    }

    // ── Test 16: BradleyTerryModel preference_probability ────────────────
    #[test]
    fn test_bt_preference_probability() {
        let bt = BradleyTerryModel::new(1.0).expect("valid temperature");
        // P(r_w=1 > r_l=0) = σ(1) = 1/(1+e^{-1}) ≈ 0.7311
        let p = bt.preference_probability(1.0, 0.0);
        let expected = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((p - expected).abs() < 1e-5, "expected {expected}, got {p}");
    }

    // ── Test 17: BradleyTerryModel equal rewards → P = 0.5 ───────────────
    #[test]
    fn test_bt_equal_rewards_half_probability() {
        let bt = BradleyTerryModel::default_temperature();
        let p = bt.preference_probability(2.0, 2.0);
        assert!((p - 0.5f32).abs() < 1e-5, "equal rewards → P=0.5, got {p}");
    }

    // ── Test 18: BradleyTerryModel bt_loss ───────────────────────────────
    #[test]
    fn test_bt_loss_value() {
        let bt = BradleyTerryModel::default_temperature();
        // BT loss = -log P(r_w=2 > r_l=0) = -log σ(2)
        let loss = bt.bt_loss(2.0, 0.0);
        let expected = -(1.0f32 / (1.0 + (-2.0f32).exp())).ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "expected {expected}, got {loss}"
        );
    }

    // ── Test 19: BradleyTerryModel bt_loss positive ───────────────────────
    #[test]
    fn test_bt_loss_always_positive() {
        let bt = BradleyTerryModel::default_temperature();
        let loss1 = bt.bt_loss(1.0, 0.0);
        let loss2 = bt.bt_loss(-1.0, 2.0);
        assert!(loss1 > 0.0, "BT loss should be positive");
        assert!(
            loss2 > 0.0,
            "BT loss should be positive even for bad ordering"
        );
    }

    // ── Test 20: BradleyTerryModel invalid temperature ───────────────────
    #[test]
    fn test_bt_invalid_temperature() {
        let err = BradleyTerryModel::new(0.0).unwrap_err();
        assert!(matches!(err, RewardError::InvalidScore));
        let err2 = BradleyTerryModel::new(-1.0).unwrap_err();
        assert!(matches!(err2, RewardError::InvalidScore));
    }

    // ── Test 21: BradleyTerryModel batch_bt_loss ─────────────────────────
    #[test]
    fn test_bt_batch_bt_loss() {
        let bt = BradleyTerryModel::default_temperature();
        let pairs = vec![(1.0f32, 0.0), (2.0, -1.0), (0.5, -0.5)];
        let mean_loss = bt.batch_bt_loss(&pairs).expect("ok");
        assert!(mean_loss > 0.0, "batch BT loss should be positive");
        assert!(mean_loss.is_finite(), "batch BT loss should be finite");
    }

    // ── Test 22: RewardNormalizer new_from_samples mean=0 std=1 ──────────
    #[test]
    fn test_reward_normalizer_from_samples() {
        let samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let norm = RewardNormalizer::new_from_samples(&samples);
        // mean should be 3.0
        assert!(
            (norm.mean - 3.0).abs() < 1e-5,
            "expected mean=3.0, got {}",
            norm.mean
        );
        // population std = sqrt(mean_sq_deviation) = sqrt(2.0)
        let expected_std = 2.0_f32.sqrt();
        assert!(
            (norm.std - expected_std).abs() < 1e-4,
            "expected std={expected_std}, got {}",
            norm.std
        );
    }

    // ── Test 23: RewardNormalizer normalize ───────────────────────────────
    #[test]
    fn test_reward_normalizer_normalize() {
        let samples = vec![0.0f32, 2.0]; // mean=1, std=1
        let norm = RewardNormalizer::new_from_samples(&samples);
        let n0 = norm.normalize(0.0);
        let n2 = norm.normalize(2.0);
        assert!((n0 - (-1.0f32)).abs() < 1e-4, "expected -1.0, got {n0}");
        assert!((n2 - 1.0f32).abs() < 1e-4, "expected 1.0, got {n2}");
    }

    // ── Test 24: RewardNormalizer normalize with clip ─────────────────────
    #[test]
    fn test_reward_normalizer_clip() {
        let samples = vec![0.0f32, 2.0, 4.0];
        let mut norm = RewardNormalizer::new_from_samples(&samples);
        norm.clip_range = Some((-1.0, 1.0));
        let far_out = norm.normalize(100.0);
        assert!(
            (far_out - 1.0f32).abs() < 1e-5,
            "clipped at max=1.0, got {far_out}"
        );
    }

    // ── Test 25: RewardNormalizer online update convergence ───────────────
    #[test]
    fn test_reward_normalizer_online_update() {
        let mut norm = RewardNormalizer::new();
        // Feed in values 1..=5
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0] {
            norm.update(v);
        }
        assert_eq!(norm.count(), 5);
        assert!(
            (norm.mean - 3.0).abs() < 1e-4,
            "online mean should be 3.0, got {}",
            norm.mean
        );
    }

    // ── Test 26: margin_ranking_loss zero when margin satisfied ───────────
    #[test]
    fn test_margin_ranking_loss_zero() {
        // r_w - r_l = 2.0 >= margin = 1.0 → loss = 0
        let loss = margin_ranking_loss(2.0, 0.0, 1.0);
        assert_eq!(loss, 0.0, "loss should be 0 when margin satisfied");
    }

    // ── Test 27: margin_ranking_loss positive when margin violated ─────────
    #[test]
    fn test_margin_ranking_loss_positive() {
        // r_w - r_l = 0.5, margin = 2.0 → loss = 1.5
        let loss = margin_ranking_loss(1.0, 0.5, 2.0);
        assert!(
            (loss - 1.5f32).abs() < 1e-5,
            "expected loss=1.5, got {loss}"
        );
    }

    // ── New tests 28-50 ───────────────────────────────────────────────────

    // 28. Bradley-Terry P(A>B) = σ(r_A - r_B) — verify formula
    #[test]
    fn test_bt_preference_probability_formula() {
        let bt = BradleyTerryModel::new(1.0).expect("valid");
        // P(r_w=2 > r_l=0) = σ(2)
        let p = bt.preference_probability(2.0, 0.0);
        let expected = 1.0f32 / (1.0 + (-2.0f32).exp());
        assert!((p - expected).abs() < 1e-5, "P={p}, expected {expected}");
    }

    // 29. P(A>B) + P(B>A) = 1 (complementary probabilities)
    #[test]
    fn test_bt_complementary_probabilities() {
        let bt = BradleyTerryModel::default_temperature();
        let r_a = 1.5f32;
        let r_b = 0.3f32;
        let p_ab = bt.preference_probability(r_a, r_b);
        let p_ba = bt.preference_probability(r_b, r_a);
        assert!(
            (p_ab + p_ba - 1.0f32).abs() < 1e-5,
            "P(A>B) + P(B>A) should be 1.0, got {}",
            p_ab + p_ba
        );
    }

    // 30. margin_ranking_loss with margin=0 is always non-negative
    #[test]
    fn test_margin_ranking_loss_margin_zero_nonneg() {
        // margin=0 means any positive diff gives 0 loss; negative diff gives positive loss
        let loss_pos = margin_ranking_loss(2.0, 0.5, 0.0);
        let loss_neg = margin_ranking_loss(0.5, 2.0, 0.0);
        assert!(
            loss_pos >= 0.0,
            "margin=0, positive diff → non-negative loss, got {loss_pos}"
        );
        assert!(
            loss_neg > 0.0,
            "margin=0, negative diff → positive loss, got {loss_neg}"
        );
    }

    // 31. pairwise_accuracy returns value in [0, 1]
    #[test]
    fn test_pairwise_accuracy_in_range() {
        let model = default_model();
        let pairs: Vec<RmPreferencePair> = (0..10)
            .map(|i| pair(&format!("chosen_{i}"), &format!("rejected_{i}")))
            .collect();
        let acc = model.pairwise_accuracy(&pairs);
        assert!(
            (0.0..=1.0).contains(&acc),
            "accuracy should be in [0,1], got {acc}"
        );
    }

    // 32. normalize_rewards: mean≈0 and std≈1 after z-score normalisation
    #[test]
    fn test_normalize_rewards_z_score() {
        let model = default_model();
        let scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let normed = model.normalize_rewards(&scores);
        let n = normed.len() as f64;
        let mean = normed.iter().sum::<f64>() / n;
        let var = normed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-9, "z-score mean should be ≈0, got {mean}");
        assert!(
            (var.sqrt() - 1.0).abs() < 1e-9,
            "z-score std should be ≈1, got {}",
            var.sqrt()
        );
    }

    // 33. normalize_rewards with identical inputs returns same vector (zero variance)
    #[test]
    fn test_normalize_rewards_identical_inputs_unchanged() {
        let model = default_model();
        let scores = vec![3.0, 3.0, 3.0, 3.0];
        let normed = model.normalize_rewards(&scores);
        for (orig, norm) in scores.iter().zip(normed.iter()) {
            assert!(
                (orig - norm).abs() < 1e-10,
                "identical scores should be returned as-is"
            );
        }
    }

    // 34. Ensemble reward: average of multiple scores via score_batch
    #[test]
    fn test_ensemble_reward_average() {
        let model = default_model();
        let texts = ["alpha", "beta", "gamma", "delta"];
        let scores = model.score_batch(&texts);
        assert_eq!(scores.len(), 4);
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        // average should be in [-2, 2]
        assert!(
            (-2.0..=2.0).contains(&avg),
            "ensemble average should be in [-2,2], got {avg}"
        );
    }

    // 35. A trained model separates texts by content, not by length
    #[test]
    fn test_trained_model_separates_texts_by_content() {
        let mut model = default_model();
        model.train(&learnable_pairs(), 1.0, 200).expect("train");

        // The distinguishing token dominates, whatever the surrounding length.
        let short = model.score("helpful");
        let long_text = model.score("harmful harmful harmful harmful harmful harmful");
        assert!(
            short > long_text,
            "the learned preference must survive a length difference \
             (helpful={short}, harmful={long_text})"
        );

        // Length alone is not a signal: repeating the same token leaves the L2-normalised
        // embedding — and therefore the score — unchanged.
        let once = model.score("helpful");
        let thrice = model.score("helpful helpful helpful");
        assert!(
            (once - thrice).abs() < 1e-9,
            "repetition must not inflate the reward ({once} vs {thrice})"
        );
    }

    // 36. score() is always finite, for any input
    #[test]
    fn test_score_is_always_finite() {
        let mut model = default_model();
        model.train(&learnable_pairs(), 1.0, 50).expect("train");
        let texts = [
            "",
            "a",
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "12345",
            "!@#$%^&*()",
            "UPPER CASE",
            "mixed Case With Numbers 123",
        ];
        for text in &texts {
            let s = model.score(text);
            assert!(s.is_finite(), "score for '{text}' = {s} must be finite");
        }
        // A text with no tokens has a zero embedding, so its score is exactly the bias.
        assert!((model.score("!@#$%^&*()") - model.bias()).abs() < 1e-12);
    }

    // 37. BradleyTerryModel: higher T → probability closer to 0.5
    #[test]
    fn test_bt_higher_temperature_closer_to_half() {
        let bt_low = BradleyTerryModel::new(0.1).expect("valid");
        let bt_high = BradleyTerryModel::new(10.0).expect("valid");
        let r_w = 2.0f32;
        let r_l = 0.0f32;
        let p_low = bt_low.preference_probability(r_w, r_l);
        let p_high = bt_high.preference_probability(r_w, r_l);
        // Higher T → p closer to 0.5
        assert!(
            (p_high - 0.5f32).abs() < (p_low - 0.5f32).abs(),
            "higher T ({}) should give prob closer to 0.5 than lower T ({}): p_high={p_high}, p_low={p_low}",
            10.0, 0.1
        );
    }

    // 38. RewardNormalizer update multiple times → count matches
    #[test]
    fn test_reward_normalizer_update_count() {
        let mut norm = RewardNormalizer::new();
        for i in 1..=10 {
            norm.update(i as f32);
        }
        assert_eq!(norm.count(), 10, "count should be 10 after 10 updates");
    }

    // 39. RewardNormalizer with no samples: mean=0, std=1
    #[test]
    fn test_reward_normalizer_empty_samples() {
        let norm = RewardNormalizer::new_from_samples(&[]);
        assert_eq!(norm.mean, 0.0);
        assert!((norm.std - 1.0).abs() < 1e-6, "default std should be 1.0");
        assert_eq!(norm.count(), 0);
    }

    // 40. margin_ranking_loss: diff exactly equals margin → loss = 0
    #[test]
    fn test_margin_ranking_loss_exact_margin() {
        // r_w - r_l = 1.5 = margin → loss = max(0, 0) = 0
        let loss = margin_ranking_loss(2.0, 0.5, 1.5);
        assert!(
            (loss).abs() < 1e-5,
            "exact margin should give loss=0, got {loss}"
        );
    }

    // 41. BradleyTerryModel::batch_bt_loss empty batch → RewardError::EmptyBatch
    #[test]
    fn test_bt_batch_bt_loss_empty_batch() {
        let bt = BradleyTerryModel::default_temperature();
        let err = bt.batch_bt_loss(&[]).expect_err("should fail on empty");
        assert!(matches!(err, RewardError::EmptyBatch));
    }

    // 42. RmRewardModel::score reproducibility: same input → same score
    #[test]
    fn test_score_reproducibility() {
        let model = default_model();
        let text = "The same text should always score the same";
        let s1 = model.score(text);
        let s2 = model.score(text);
        assert!(
            (s1 - s2).abs() < 1e-15,
            "score should be deterministic: {s1} vs {s2}"
        );
    }

    // 43. RewardLossResult display contains accuracy
    #[test]
    fn test_reward_loss_result_display_contains_accuracy() {
        let result = RewardLossResult {
            mean_loss: 0.5,
            mean_chosen_score: 1.0,
            mean_rejected_score: -1.0,
            accuracy: 0.75,
            mean_margin: 2.0,
            batch_size: 4,
        };
        let s = format!("{result}");
        assert!(
            s.contains("accuracy") || s.contains("0.7500"),
            "display should contain accuracy info: {s}"
        );
    }

    // 44. PreferenceDataset::is_empty on empty dataset → true
    #[test]
    fn test_preference_dataset_is_empty() {
        let ds = PreferenceDataset::new();
        assert!(ds.is_empty(), "new dataset should be empty");
        assert_eq!(ds.len(), 0);
    }

    // 45. PreferenceDataset split with fraction=0 → all in val, none in train
    #[test]
    fn test_preference_dataset_split_fraction_zero() {
        let mut ds = PreferenceDataset::new();
        for i in 0..5 {
            ds.add(pair(&format!("c{i}"), &format!("r{i}")));
        }
        let (train, val) = ds.split(0.0);
        assert_eq!(train.len(), 0, "train should be empty with fraction=0");
        assert_eq!(val.len(), 5, "val should contain all 5 with fraction=0");
    }

    // 46. PreferenceDataset split with fraction=1 → all in train, none in val
    #[test]
    fn test_preference_dataset_split_fraction_one() {
        let mut ds = PreferenceDataset::new();
        for i in 0..6 {
            ds.add(pair(&format!("c{i}"), &format!("r{i}")));
        }
        let (train, val) = ds.split(1.0);
        assert_eq!(train.len(), 6, "train should contain all 6 with fraction=1");
        assert_eq!(val.len(), 0, "val should be empty with fraction=1");
    }

    // 47. DatasetStats mean_score_margin=None when no scores available
    #[test]
    fn test_dataset_stats_no_score_margin_is_none() {
        let mut ds = PreferenceDataset::new();
        ds.add(pair("chosen_a", "rejected_a"));
        ds.add(pair("chosen_b", "rejected_b"));
        let stats = ds.statistics();
        assert!(
            stats.mean_score_margin.is_none(),
            "no scores → mean_score_margin should be None"
        );
        assert_eq!(stats.pairs_with_scores, 0);
    }

    // 48. RmRewardModelConfig normalize_rewards field
    #[test]
    fn test_rm_config_normalize_rewards_field() {
        let cfg = RmRewardModelConfig {
            normalize_rewards: true,
            ..Default::default()
        };
        assert!(
            cfg.normalize_rewards,
            "normalize_rewards should be settable"
        );
        let cfg_default = RmRewardModelConfig::default();
        assert!(
            !cfg_default.normalize_rewards,
            "default normalize_rewards should be false"
        );
    }

    // 49. RewardNormalizer clip_range lower bound
    #[test]
    fn test_reward_normalizer_clip_lower_bound() {
        let samples = vec![0.0f32, 10.0];
        let mut norm = RewardNormalizer::new_from_samples(&samples);
        norm.clip_range = Some((-1.0, 1.0));
        // Very low value should be clipped to lower bound
        let clipped = norm.normalize(-1000.0);
        assert!(
            (clipped - (-1.0f32)).abs() < 1e-5,
            "should clip at -1.0, got {clipped}"
        );
    }

    // 50. BT loss with very large reward difference → loss approaches 0
    #[test]
    fn test_bt_loss_large_diff_approaches_zero() {
        let bt = BradleyTerryModel::default_temperature();
        // r_w >> r_l → P(A>B) → 1 → loss → 0
        let loss = bt.bt_loss(100.0, -100.0);
        assert!(
            loss < 1e-5,
            "BT loss with huge margin should approach 0, got {loss}"
        );
    }
}
