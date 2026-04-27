//! Reward model training infrastructure.
//!
//! Implements preference-based reward model training with multiple loss functions
//! and pooling strategies, following the approaches in InstructGPT, Anthropic HH,
//! and related RLHF literature.

use std::fmt;

// ──────────────────────────────────────────────
// PoolingType
// ──────────────────────────────────────────────

/// Strategy for pooling a sequence of hidden states into a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    /// Use the hidden state of the last token (common for decoder-only models).
    LastToken,
    /// Average all token hidden states equally.
    Mean,
    /// Element-wise maximum over the sequence dimension.
    Max,
    /// Weighted average where later tokens receive higher weight.
    WeightedMean,
}

// ──────────────────────────────────────────────
// RewardLossType
// ──────────────────────────────────────────────

/// Loss function variant for reward model training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardLossType {
    /// Bradley-Terry preference model: `−log σ(r_chosen − r_rejected)`.
    BradleyTerry,
    /// Hinge loss: `max(0, margin − (r_chosen − r_rejected))`.
    Hinge,
    /// InstructGPT-style: Bradley-Terry with optional label smoothing.
    InstructGpt,
    /// MSE regression against a scalar label.
    Regression,
}

// ──────────────────────────────────────────────
// RewardModelConfig
// ──────────────────────────────────────────────

/// Configuration for reward model training.
#[derive(Debug, Clone)]
pub struct RewardModelConfig {
    /// Dimensionality of the hidden representation.
    pub hidden_size: usize,
    /// Number of output labels (1 = scalar reward).
    pub num_labels: usize,
    /// Loss function variant.
    pub loss_type: RewardLossType,
    /// Margin for `Hinge` loss.
    pub margin: f32,
    /// Label smoothing coefficient for `InstructGpt` loss.
    pub label_smoothing: f32,
    /// Coefficient penalising deviation of mean reward from zero.
    pub center_rewards_coefficient: f32,
    /// How to pool the sequence of hidden states into one vector.
    pub pooling_type: PoolingType,
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_labels: 1,
            loss_type: RewardLossType::BradleyTerry,
            margin: 0.0,
            label_smoothing: 0.0,
            center_rewards_coefficient: 0.0,
            pooling_type: PoolingType::LastToken,
        }
    }
}

// ──────────────────────────────────────────────
// RewardModelTrainingExample
// ──────────────────────────────────────────────

/// One training example for the reward model.
///
/// `chosen_hidden` and `rejected_hidden` are flat row-major buffers of shape
/// `[seq_len, hidden_size]`.
#[derive(Debug, Clone)]
pub struct RewardModelTrainingExample {
    /// Flat hidden states for the chosen (preferred) response.
    pub chosen_hidden: Vec<f32>,
    /// Flat hidden states for the rejected response.
    pub rejected_hidden: Vec<f32>,
    /// Sequence length for `chosen_hidden`.
    pub chosen_seq_len: usize,
    /// Sequence length for `rejected_hidden`.
    pub rejected_seq_len: usize,
    /// Optional scalar label used in `Regression` mode.
    pub scalar_label: Option<f32>,
}

// ──────────────────────────────────────────────
// RewardTrainingOutput
// ──────────────────────────────────────────────

/// Aggregated statistics from a batch reward training step.
#[derive(Debug, Clone)]
pub struct RewardTrainingOutput {
    /// Mean loss over the batch (including center-rewards penalty if enabled).
    pub total_loss: f32,
    /// Mean reward score for chosen responses.
    pub mean_chosen_score: f32,
    /// Mean reward score for rejected responses.
    pub mean_rejected_score: f32,
    /// Fraction of examples where `score_chosen > score_rejected`.
    pub accuracy: f32,
    /// Mean `score_chosen − score_rejected`.
    pub mean_margin: f32,
}

// ──────────────────────────────────────────────
// RewardTrainError
// ──────────────────────────────────────────────

/// Errors that can arise during reward model training.
#[derive(Debug)]
pub enum RewardTrainError {
    /// The training batch contains no examples.
    EmptyBatch,
    /// Hidden state buffer has wrong length given seq_len × hidden_size.
    HiddenSizeMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements found.
        actual: usize,
    },
    /// A computed score is NaN or infinite.
    InvalidScore,
    /// Regression loss requires a scalar label but none was provided.
    MissingScalarLabel,
    /// hidden_size is zero, which is not valid.
    ZeroHiddenSize,
    /// Sequence length is zero.
    ZeroSeqLen,
}

impl fmt::Display for RewardTrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "Reward training batch is empty"),
            Self::HiddenSizeMismatch { expected, actual } => write!(
                f,
                "Hidden state size mismatch: expected {} elements, got {}",
                expected, actual
            ),
            Self::InvalidScore => write!(f, "Computed reward score is NaN or Inf"),
            Self::MissingScalarLabel => {
                write!(f, "Regression loss requires scalar_label but it is None")
            }
            Self::ZeroHiddenSize => write!(f, "hidden_size must be > 0"),
            Self::ZeroSeqLen => write!(f, "Sequence length must be > 0"),
        }
    }
}

impl std::error::Error for RewardTrainError {}

// ──────────────────────────────────────────────
// sigmoid helper
// ──────────────────────────────────────────────

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

// ──────────────────────────────────────────────
// pool_hidden_states
// ──────────────────────────────────────────────

/// Pool a flat `[seq_len × hidden_size]` buffer into a `[hidden_size]` vector.
///
/// # Errors
/// Returns [`RewardTrainError::ZeroSeqLen`] when `seq_len == 0`,
/// [`RewardTrainError::ZeroHiddenSize`] when `hidden_size == 0`, and
/// [`RewardTrainError::HiddenSizeMismatch`] when the buffer length does not
/// equal `seq_len * hidden_size`.
pub fn pool_hidden_states(
    hidden: &[f32],
    seq_len: usize,
    hidden_size: usize,
    pooling: PoolingType,
) -> Result<Vec<f32>, RewardTrainError> {
    if seq_len == 0 {
        return Err(RewardTrainError::ZeroSeqLen);
    }
    if hidden_size == 0 {
        return Err(RewardTrainError::ZeroHiddenSize);
    }
    let expected = seq_len * hidden_size;
    if hidden.len() != expected {
        return Err(RewardTrainError::HiddenSizeMismatch {
            expected,
            actual: hidden.len(),
        });
    }

    match pooling {
        PoolingType::LastToken => {
            let start = (seq_len - 1) * hidden_size;
            Ok(hidden[start..start + hidden_size].to_vec())
        }

        PoolingType::Mean => {
            let mut out = vec![0.0_f32; hidden_size];
            for t in 0..seq_len {
                let offset = t * hidden_size;
                for (i, val) in out.iter_mut().enumerate() {
                    *val += hidden[offset + i];
                }
            }
            let n = seq_len as f32;
            for val in out.iter_mut() {
                *val /= n;
            }
            Ok(out)
        }

        PoolingType::Max => {
            let mut out = vec![f32::NEG_INFINITY; hidden_size];
            for t in 0..seq_len {
                let offset = t * hidden_size;
                for (i, val) in out.iter_mut().enumerate() {
                    let candidate = hidden[offset + i];
                    if candidate > *val {
                        *val = candidate;
                    }
                }
            }
            Ok(out)
        }

        PoolingType::WeightedMean => {
            // weight[t] = t / (seq_len - 1) for t = 0 .. seq_len-1
            // When seq_len == 1 weight = 1.0 to avoid division by zero.
            let mut out = vec![0.0_f32; hidden_size];
            let weight_sum: f32 = if seq_len == 1 {
                1.0
            } else {
                // sum of 0/(n-1) + 1/(n-1) + ... + (n-1)/(n-1) = n/2
                (seq_len as f32) / 2.0
            };

            for t in 0..seq_len {
                let weight = if seq_len == 1 {
                    1.0_f32
                } else {
                    t as f32 / (seq_len - 1) as f32
                };
                let offset = t * hidden_size;
                for (i, val) in out.iter_mut().enumerate() {
                    *val += weight * hidden[offset + i];
                }
            }
            for val in out.iter_mut() {
                *val /= weight_sum;
            }
            Ok(out)
        }
    }
}

// ──────────────────────────────────────────────
// compute_reward_score
// ──────────────────────────────────────────────

/// Compute a scalar reward score as the dot product of a pooled hidden state
/// with a reward head weight vector.
///
/// `reward_head` has shape `[hidden_size]` (the bias-free linear projection
/// onto a scalar), matching the linearised `[hidden_size, 1]` weight matrix.
///
/// # Errors
/// Returns [`RewardTrainError::HiddenSizeMismatch`] when the slices differ in
/// length, and [`RewardTrainError::InvalidScore`] when the result is non-finite.
pub fn compute_reward_score(
    hidden_state: &[f32],
    reward_head: &[f32],
    hidden_size: usize,
) -> Result<f32, RewardTrainError> {
    if hidden_state.len() != hidden_size || reward_head.len() != hidden_size {
        return Err(RewardTrainError::HiddenSizeMismatch {
            expected: hidden_size,
            actual: hidden_state.len().min(reward_head.len()),
        });
    }
    let score: f32 = hidden_state
        .iter()
        .zip(reward_head.iter())
        .map(|(h, w)| h * w)
        .sum();
    if !score.is_finite() {
        return Err(RewardTrainError::InvalidScore);
    }
    Ok(score)
}

// ──────────────────────────────────────────────
// reward_loss (single pair)
// ──────────────────────────────────────────────

/// Compute the loss for a single (chosen, rejected) reward score pair.
///
/// For `Regression`, `scalar_label` must be `Some`; otherwise
/// [`RewardTrainError::MissingScalarLabel`] is returned.
pub fn reward_loss(
    chosen_score: f32,
    rejected_score: f32,
    config: &RewardModelConfig,
    scalar_label: Option<f32>,
) -> Result<f32, RewardTrainError> {
    let loss = match config.loss_type {
        RewardLossType::BradleyTerry => {
            let diff = chosen_score - rejected_score;
            -sigmoid_f32(diff).ln()
        }

        RewardLossType::Hinge => {
            let diff = chosen_score - rejected_score;
            (config.margin - diff).max(0.0)
        }

        RewardLossType::InstructGpt => {
            let diff = chosen_score - rejected_score;
            let s = config.label_smoothing;
            // InstructGPT: −(1 − s) log σ(diff) + s log 2
            -(1.0 - s) * sigmoid_f32(diff).ln() + s * 2.0_f32.ln()
        }

        RewardLossType::Regression => {
            let label = scalar_label.ok_or(RewardTrainError::MissingScalarLabel)?;
            (chosen_score - label).powi(2)
        }
    };

    if !loss.is_finite() {
        return Err(RewardTrainError::InvalidScore);
    }
    Ok(loss)
}

// ──────────────────────────────────────────────
// batch_reward_loss
// ──────────────────────────────────────────────

/// Compute reward model loss over a batch of training examples.
///
/// The reward head is a flat `[hidden_size]` vector (linearised `[hidden_size, 1]`
/// weight matrix — a bias-free linear projection to a scalar).
///
/// An optional center-rewards penalty `config.center_rewards_coefficient * mean²`
/// is added when `config.center_rewards_coefficient > 0`.
///
/// # Errors
/// See [`RewardTrainError`] variants.
pub fn batch_reward_loss(
    examples: &[RewardModelTrainingExample],
    reward_head: &[f32],
    config: &RewardModelConfig,
) -> Result<RewardTrainingOutput, RewardTrainError> {
    if examples.is_empty() {
        return Err(RewardTrainError::EmptyBatch);
    }

    let hidden_size = config.hidden_size;
    let mut total_loss = 0.0_f32;
    let mut sum_chosen = 0.0_f32;
    let mut sum_rejected = 0.0_f32;
    let mut sum_margin = 0.0_f32;
    let mut correct = 0_usize;

    for ex in examples {
        // Pool chosen hidden states
        let pooled_chosen = pool_hidden_states(
            &ex.chosen_hidden,
            ex.chosen_seq_len,
            hidden_size,
            config.pooling_type,
        )?;
        // Pool rejected hidden states
        let pooled_rejected = pool_hidden_states(
            &ex.rejected_hidden,
            ex.rejected_seq_len,
            hidden_size,
            config.pooling_type,
        )?;

        let score_chosen = compute_reward_score(&pooled_chosen, reward_head, hidden_size)?;
        let score_rejected = compute_reward_score(&pooled_rejected, reward_head, hidden_size)?;

        let loss = reward_loss(score_chosen, score_rejected, config, ex.scalar_label)?;

        total_loss += loss;
        sum_chosen += score_chosen;
        sum_rejected += score_rejected;
        sum_margin += score_chosen - score_rejected;
        if score_chosen > score_rejected {
            correct += 1;
        }
    }

    let n = examples.len() as f32;
    let mean_chosen = sum_chosen / n;
    let mean_rejected = sum_rejected / n;

    // Center-rewards penalty: penalise if the mean reward deviates from 0.
    let center_penalty = if config.center_rewards_coefficient > 0.0 {
        let mean_all = (sum_chosen + sum_rejected) / (2.0 * n);
        config.center_rewards_coefficient * mean_all.powi(2)
    } else {
        0.0
    };

    Ok(RewardTrainingOutput {
        total_loss: total_loss / n + center_penalty,
        mean_chosen_score: mean_chosen,
        mean_rejected_score: mean_rejected,
        accuracy: correct as f32 / n,
        mean_margin: sum_margin / n,
    })
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a trivial training example with uniform hidden states.
    fn make_example(
        seq_len: usize,
        hidden_size: usize,
        chosen_val: f32,
        rejected_val: f32,
        scalar_label: Option<f32>,
    ) -> RewardModelTrainingExample {
        RewardModelTrainingExample {
            chosen_hidden: vec![chosen_val; seq_len * hidden_size],
            rejected_hidden: vec![rejected_val; seq_len * hidden_size],
            chosen_seq_len: seq_len,
            rejected_seq_len: seq_len,
            scalar_label,
        }
    }

    /// Config shortcut.
    fn bt_config(hidden_size: usize) -> RewardModelConfig {
        RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::BradleyTerry,
            ..Default::default()
        }
    }

    // ── Test 1: BradleyTerry loss formula ─────────────────────────────────
    #[test]
    fn test_bradley_terry_formula() {
        // chosen_score = 1.0, rejected_score = 0.0
        // loss = -log(σ(1.0)) = -log(1/(1+e^{-1})) ≈ 0.3133
        let config = bt_config(4);
        let loss = reward_loss(1.0, 0.0, &config, None).expect("bt loss");
        let expected = -(sigmoid_f32(1.0).ln());
        assert!((loss - expected).abs() < 1e-6, "BT loss = {}, expected {}", loss, expected);
    }

    // ── Test 2: BradleyTerry loss is positive ─────────────────────────────
    #[test]
    fn test_bradley_terry_positive() {
        let config = bt_config(4);
        let loss = reward_loss(2.0, -1.0, &config, None).expect("bt loss");
        assert!(loss > 0.0, "BT loss should be > 0, got {}", loss);
    }

    // ── Test 3: Hinge loss at margin ──────────────────────────────────────
    #[test]
    fn test_hinge_loss_formula() {
        let config = RewardModelConfig {
            hidden_size: 4,
            loss_type: RewardLossType::Hinge,
            margin: 1.0,
            ..Default::default()
        };
        // diff = 0.5  → max(0, 1.0 − 0.5) = 0.5
        let loss = reward_loss(1.0, 0.5, &config, None).expect("hinge loss");
        assert!((loss - 0.5).abs() < 1e-6, "Hinge = {}", loss);

        // diff = 2.0 → max(0, 1.0 − 2.0) = 0
        let loss_zero = reward_loss(3.0, 1.0, &config, None).expect("hinge zero");
        assert!(loss_zero.abs() < 1e-6, "Hinge should be 0, got {}", loss_zero);
    }

    // ── Test 4: InstructGPT label smoothing ───────────────────────────────
    #[test]
    fn test_instructgpt_smoothing() {
        // With s=0 it should match BradleyTerry exactly.
        let config_bt = bt_config(4);
        let config_ig = RewardModelConfig {
            hidden_size: 4,
            loss_type: RewardLossType::InstructGpt,
            label_smoothing: 0.0,
            ..Default::default()
        };
        let chosen = 1.5_f32;
        let rejected = 0.5_f32;
        let loss_bt = reward_loss(chosen, rejected, &config_bt, None).expect("bt");
        let loss_ig = reward_loss(chosen, rejected, &config_ig, None).expect("ig");
        assert!(
            (loss_bt - loss_ig).abs() < 1e-6,
            "IG with s=0 should equal BT: bt={} ig={}",
            loss_bt,
            loss_ig
        );

        // With smoothing > 0, the loss should differ.
        let config_smooth = RewardModelConfig {
            hidden_size: 4,
            loss_type: RewardLossType::InstructGpt,
            label_smoothing: 0.1,
            ..Default::default()
        };
        let loss_smooth = reward_loss(chosen, rejected, &config_smooth, None).expect("smooth");
        assert!(
            (loss_smooth - loss_bt).abs() > 1e-6,
            "Smoothed IG should differ from BT"
        );
    }

    // ── Test 5: LastToken pooling shape ───────────────────────────────────
    #[test]
    fn test_last_token_pooling_shape() {
        let hidden_size = 8;
        let seq_len = 5;
        let hidden: Vec<f32> = (0..(seq_len * hidden_size) as i32).map(|x| x as f32).collect();
        let pooled =
            pool_hidden_states(&hidden, seq_len, hidden_size, PoolingType::LastToken)
                .expect("pool");
        assert_eq!(pooled.len(), hidden_size, "LastToken pool length should be hidden_size");
        // Last token starts at (seq_len-1)*hidden_size
        let start = (seq_len - 1) * hidden_size;
        for (i, &v) in pooled.iter().enumerate() {
            assert!(
                (v - (start + i) as f32).abs() < 1e-6,
                "LastToken[{}] = {} expected {}",
                i,
                v,
                (start + i) as f32
            );
        }
    }

    // ── Test 6: Mean pooling correctness ──────────────────────────────────
    #[test]
    fn test_mean_pooling() {
        // 2 tokens, hidden_size=2: [[1,2],[3,4]] → mean = [2,3]
        let hidden = vec![1.0_f32, 2.0, 3.0, 4.0];
        let pooled = pool_hidden_states(&hidden, 2, 2, PoolingType::Mean).expect("mean pool");
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6, "mean[0] = {}", pooled[0]);
        assert!((pooled[1] - 3.0).abs() < 1e-6, "mean[1] = {}", pooled[1]);
    }

    // ── Test 7: reward score dot product ──────────────────────────────────
    #[test]
    fn test_reward_score_dot_product() {
        // h = [1, 2, 3], w = [0.5, 0.5, 0.5] → dot = 3.0
        let h = vec![1.0_f32, 2.0, 3.0];
        let w = vec![0.5_f32, 0.5, 0.5];
        let score = compute_reward_score(&h, &w, 3).expect("score");
        assert!((score - 3.0).abs() < 1e-6, "dot product = {}", score);
    }

    // ── Test 8: accuracy metric in batch_reward_loss ──────────────────────
    #[test]
    fn test_accuracy_metric() {
        // chosen_val > 0, rejected_val < 0 → after dot with positive weights,
        // chosen_score > rejected_score in every example.
        let hidden_size = 4;
        let config = bt_config(hidden_size);
        let reward_head = vec![1.0_f32; hidden_size];

        let examples: Vec<_> = (0..5)
            .map(|_| make_example(2, hidden_size, 1.0, -1.0, None))
            .collect();

        let out = batch_reward_loss(&examples, &reward_head, &config).expect("batch");
        assert!(
            (out.accuracy - 1.0).abs() < 1e-6,
            "All chosen should score higher: acc = {}",
            out.accuracy
        );
    }

    // ── Test 9: batch loss (multi-example) ────────────────────────────────
    #[test]
    fn test_batch_loss_multi_example() {
        let hidden_size = 2;
        let config = bt_config(hidden_size);
        let reward_head = vec![1.0_f32; hidden_size];

        let examples = vec![
            make_example(1, hidden_size, 1.0, 0.0, None),
            make_example(1, hidden_size, 0.5, -0.5, None),
        ];

        let out = batch_reward_loss(&examples, &reward_head, &config).expect("batch");
        assert!(out.total_loss > 0.0, "total_loss should be > 0, got {}", out.total_loss);
        assert!(out.mean_chosen_score > out.mean_rejected_score);
    }

    // ── Test 10: center_rewards_coefficient effect ────────────────────────
    #[test]
    fn test_center_rewards_penalty() {
        let hidden_size = 2;
        let reward_head = vec![1.0_f32; hidden_size];

        let config_no_penalty = RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::BradleyTerry,
            center_rewards_coefficient: 0.0,
            ..Default::default()
        };
        let config_with_penalty = RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::BradleyTerry,
            center_rewards_coefficient: 1.0,
            ..Default::default()
        };

        // Chosen val=2.0, rejected val=0.0 → mean reward ≠ 0 → penalty adds loss.
        let examples = vec![make_example(1, hidden_size, 2.0, 0.0, None)];

        let out_no = batch_reward_loss(&examples, &reward_head, &config_no_penalty).expect("no");
        let out_with =
            batch_reward_loss(&examples, &reward_head, &config_with_penalty).expect("with");

        assert!(
            out_with.total_loss > out_no.total_loss,
            "Center penalty should increase loss: {} vs {}",
            out_with.total_loss,
            out_no.total_loss
        );
    }

    // ── Test 11: empty batch error ────────────────────────────────────────
    #[test]
    fn test_empty_batch_error() {
        let config = bt_config(4);
        let reward_head = vec![0.0_f32; 4];
        let err = batch_reward_loss(&[], &reward_head, &config).unwrap_err();
        assert!(
            matches!(err, RewardTrainError::EmptyBatch),
            "Expected EmptyBatch error"
        );
    }

    // ── Test 12: regression loss requires scalar_label ────────────────────
    #[test]
    fn test_regression_requires_label() {
        let config = RewardModelConfig {
            hidden_size: 4,
            loss_type: RewardLossType::Regression,
            ..Default::default()
        };
        let err = reward_loss(1.0, 0.5, &config, None).unwrap_err();
        assert!(matches!(err, RewardTrainError::MissingScalarLabel));
    }

    // ── Test 13: hidden size mismatch error ───────────────────────────────
    #[test]
    fn test_hidden_size_mismatch_error() {
        // seq_len=2, hidden_size=4 → expect 8 elements, provide 6
        let hidden = vec![0.0_f32; 6];
        let err = pool_hidden_states(&hidden, 2, 4, PoolingType::Mean).unwrap_err();
        assert!(matches!(err, RewardTrainError::HiddenSizeMismatch { .. }));
    }

    // ── Test 14: Max pooling ──────────────────────────────────────────────
    #[test]
    fn test_max_pooling() {
        // 2 tokens, hidden_size=2: [[1,4],[3,2]] → max = [3,4]
        let hidden = vec![1.0_f32, 4.0, 3.0, 2.0];
        let pooled = pool_hidden_states(&hidden, 2, 2, PoolingType::Max).expect("max pool");
        assert!((pooled[0] - 3.0).abs() < 1e-6, "max[0] = {}", pooled[0]);
        assert!((pooled[1] - 4.0).abs() < 1e-6, "max[1] = {}", pooled[1]);
    }

    // ── Test 15: WeightedMean pooling ─────────────────────────────────────
    #[test]
    fn test_weighted_mean_pooling() {
        // 3 tokens, hidden_size=1: values [1.0, 2.0, 3.0]
        // weights = [0/2, 1/2, 2/2] = [0, 0.5, 1.0], weight_sum = 1.5
        // weighted_sum = 0*1 + 0.5*2 + 1.0*3 = 4.0
        // result = 4.0 / 1.5 ≈ 2.667
        let hidden = vec![1.0_f32, 2.0, 3.0];
        let pooled = pool_hidden_states(&hidden, 3, 1, PoolingType::WeightedMean).expect("wm");
        let expected = 4.0_f32 / 1.5_f32;
        assert!(
            (pooled[0] - expected).abs() < 1e-5,
            "WeightedMean = {} expected {}",
            pooled[0],
            expected
        );
    }

    // ── New tests 16-34 ──────────────────────────────────────────────────

    // 16. Training loop metrics: batch_reward_loss returns all RewardTrainingOutput fields
    #[test]
    fn test_training_loop_all_fields_set() {
        let hidden_size = 4;
        let config = bt_config(hidden_size);
        let reward_head = vec![1.0_f32; hidden_size];
        let examples = vec![
            make_example(2, hidden_size, 1.0, -1.0, None),
            make_example(2, hidden_size, 0.5, -0.5, None),
        ];
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("out");
        assert!(out.total_loss.is_finite(), "total_loss should be finite");
        assert!(out.mean_chosen_score.is_finite(), "mean_chosen_score should be finite");
        assert!(out.mean_rejected_score.is_finite(), "mean_rejected_score should be finite");
        assert!(out.accuracy >= 0.0 && out.accuracy <= 1.0, "accuracy in [0,1]");
        assert!(out.mean_margin.is_finite(), "mean_margin should be finite");
    }

    // 17. Validation reward ranking accuracy = 1.0 when all chosen > rejected
    #[test]
    fn test_validation_ranking_accuracy_all_correct() {
        let hidden_size = 2;
        let config = bt_config(hidden_size);
        let reward_head = vec![1.0_f32; hidden_size];

        // chosen_val=2.0 > rejected_val=0.0 → score_chosen > score_rejected always
        let examples: Vec<_> = (0..8)
            .map(|_| make_example(1, hidden_size, 2.0, 0.0, None))
            .collect();

        let out = batch_reward_loss(&examples, &reward_head, &config).expect("out");
        assert!(
            (out.accuracy - 1.0).abs() < 1e-5,
            "all chosen > rejected → accuracy = 1.0, got {}",
            out.accuracy
        );
    }

    // 18. Checkpoint selection: higher accuracy is the better checkpoint
    #[test]
    fn test_checkpoint_selection_higher_accuracy_wins() {
        let hidden_size = 2;
        let reward_head = vec![1.0_f32; hidden_size];

        // Batch A: all chosen > rejected → accuracy=1.0
        let examples_a: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, 1.0, -1.0, None))
            .collect();
        // Batch B: all chosen < rejected (reversed) → accuracy=0.0
        let examples_b: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, -1.0, 1.0, None))
            .collect();

        let out_a = batch_reward_loss(&examples_a, &reward_head, &bt_config(hidden_size)).expect("a");
        let out_b = batch_reward_loss(&examples_b, &reward_head, &bt_config(hidden_size)).expect("b");

        assert!(out_a.accuracy > out_b.accuracy,
            "batch A accuracy ({}) should be > batch B accuracy ({})",
            out_a.accuracy, out_b.accuracy);
    }

    // 19. Learning rate schedule simulation: loss decreases when margin is large
    #[test]
    fn test_loss_lower_with_large_margin() {
        let hidden_size = 2;
        let reward_head = vec![1.0_f32; hidden_size];
        let config = bt_config(hidden_size);

        // Large margin: chosen=10, rejected=-10
        let examples_large: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, 10.0, -10.0, None))
            .collect();
        // Small margin: chosen=0.1, rejected=-0.1
        let examples_small: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, 0.1, -0.1, None))
            .collect();

        let out_large = batch_reward_loss(&examples_large, &reward_head, &config).expect("large");
        let out_small = batch_reward_loss(&examples_small, &reward_head, &config).expect("small");

        assert!(
            out_large.total_loss < out_small.total_loss,
            "larger margin should give lower BT loss: {} vs {}",
            out_large.total_loss, out_small.total_loss
        );
    }

    // 20. Hinge loss: zero when margin satisfied (diff >= margin)
    #[test]
    fn test_hinge_loss_zero_when_margin_satisfied() {
        let config = RewardModelConfig {
            hidden_size: 2,
            loss_type: RewardLossType::Hinge,
            margin: 1.0,
            ..Default::default()
        };
        // chosen_score - rejected_score = 2.0*2 - 0.0*2 = 4.0 >= margin=1.0
        let reward_head = vec![1.0_f32; 2];
        let examples = vec![make_example(1, 2, 2.0, 0.0, None)];
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("hinge");
        assert!(out.total_loss.abs() < 1e-5,
            "Hinge loss should be 0 when diff > margin, got {}", out.total_loss);
    }

    // 21. Hinge loss: positive when margin not satisfied
    #[test]
    fn test_hinge_loss_positive_when_margin_violated() {
        let config = RewardModelConfig {
            hidden_size: 2,
            loss_type: RewardLossType::Hinge,
            margin: 5.0, // large margin
            ..Default::default()
        };
        let reward_head = vec![0.1_f32; 2]; // small reward head
        let examples = vec![make_example(1, 2, 1.0, 0.0, None)];
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("hinge");
        assert!(out.total_loss > 0.0,
            "Hinge loss should be > 0 when margin not satisfied, got {}", out.total_loss);
    }

    // 22. Regression loss: (chosen_score - label)^2
    #[test]
    fn test_regression_loss_known_value() {
        // hidden=[1,1], reward_head=[1,1], seq=1 → pooled=[1,1] → score=2.0
        // label=3.0, regression_loss = (2.0 - 3.0)^2 = 1.0
        let config = RewardModelConfig {
            hidden_size: 2,
            loss_type: RewardLossType::Regression,
            ..Default::default()
        };
        let reward_head = vec![1.0_f32; 2];
        let example = RewardModelTrainingExample {
            chosen_hidden: vec![1.0_f32; 2],
            rejected_hidden: vec![0.0_f32; 2], // not used in regression
            chosen_seq_len: 1,
            rejected_seq_len: 1,
            scalar_label: Some(3.0),
        };
        let out = batch_reward_loss(&[example], &reward_head, &config).expect("regression");
        // chosen_score = 1*1 + 1*1 = 2.0; (2.0 - 3.0)^2 = 1.0
        assert!((out.total_loss - 1.0).abs() < 1e-5,
            "regression loss (2-3)^2 = 1.0, got {}", out.total_loss);
    }

    // 23. InstructGPT loss with smoothing=0.1 differs from BT loss
    #[test]
    fn test_instructgpt_smoothing_differs_from_bt() {
        let bt_config_val = bt_config(2);
        let ig_config = RewardModelConfig {
            hidden_size: 2,
            loss_type: RewardLossType::InstructGpt,
            label_smoothing: 0.1,
            ..Default::default()
        };
        let reward_head = vec![1.0_f32; 2];
        let examples = vec![make_example(1, 2, 1.0, 0.0, None)];

        let out_bt = batch_reward_loss(&examples, &reward_head, &bt_config_val).expect("bt");
        let out_ig = batch_reward_loss(&examples, &reward_head, &ig_config).expect("ig");

        assert!(
            (out_bt.total_loss - out_ig.total_loss).abs() > 1e-5,
            "InstructGPT loss with smoothing should differ from BT: bt={} ig={}",
            out_bt.total_loss, out_ig.total_loss
        );
    }

    // 24. pool_hidden_states: zero seq_len → ZeroSeqLen error
    #[test]
    fn test_pool_zero_seq_len_error() {
        let hidden = vec![1.0_f32; 4];
        let err = pool_hidden_states(&hidden, 0, 4, PoolingType::Mean).expect_err("zero seq");
        assert!(matches!(err, RewardTrainError::ZeroSeqLen));
    }

    // 25. pool_hidden_states: zero hidden_size → ZeroHiddenSize error
    #[test]
    fn test_pool_zero_hidden_size_error() {
        let hidden = vec![1.0_f32; 4];
        let err = pool_hidden_states(&hidden, 1, 0, PoolingType::Mean).expect_err("zero hidden");
        assert!(matches!(err, RewardTrainError::ZeroHiddenSize));
    }

    // 26. WeightedMean pooling with single token equals that token
    #[test]
    fn test_weighted_mean_single_token() {
        let hidden = vec![3.0_f32, 7.0_f32, 1.5_f32];
        let pooled = pool_hidden_states(&hidden, 1, 3, PoolingType::WeightedMean).expect("wm");
        assert_eq!(pooled.len(), 3);
        for (p, h) in pooled.iter().zip(hidden.iter()) {
            assert!((p - h).abs() < 1e-5,
                "single-token WeightedMean should equal the token itself: {p} vs {h}");
        }
    }

    // 27. compute_reward_score with all-zero hidden state → score = 0
    #[test]
    fn test_reward_score_zero_hidden() {
        let hidden = vec![0.0_f32; 4];
        let reward_head = vec![1.0_f32; 4];
        let score = compute_reward_score(&hidden, &reward_head, 4).expect("score");
        assert!((score).abs() < 1e-6, "zero hidden → score=0, got {score}");
    }

    // 28. compute_reward_score size mismatch → HiddenSizeMismatch error
    #[test]
    fn test_reward_score_size_mismatch() {
        let hidden = vec![1.0_f32; 3]; // 3 elements
        let reward_head = vec![1.0_f32; 4]; // 4 elements
        let err = compute_reward_score(&hidden, &reward_head, 4).expect_err("mismatch");
        assert!(matches!(err, RewardTrainError::HiddenSizeMismatch { .. }));
    }

    // 29. batch_reward_loss with regression loss and scalar_label
    #[test]
    fn test_batch_regression_with_label() {
        let hidden_size = 2;
        let config = RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::Regression,
            ..Default::default()
        };
        let reward_head = vec![0.5_f32; hidden_size]; // small weights
        let examples = vec![
            make_example(1, hidden_size, 1.0, 0.0, Some(0.5)),
            make_example(1, hidden_size, 2.0, 0.0, Some(1.5)),
        ];
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("regression batch");
        assert!(out.total_loss >= 0.0, "regression loss should be non-negative");
        assert!(out.total_loss.is_finite(), "regression loss should be finite");
    }

    // 30. center_rewards penalty is zero when chosen and rejected are symmetric
    #[test]
    fn test_center_rewards_zero_for_symmetric() {
        let hidden_size = 2;
        let config_no_penalty = RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::BradleyTerry,
            center_rewards_coefficient: 0.0,
            ..Default::default()
        };
        let config_with_penalty = RewardModelConfig {
            hidden_size,
            loss_type: RewardLossType::BradleyTerry,
            center_rewards_coefficient: 10.0,
            ..Default::default()
        };
        let reward_head = vec![1.0_f32; hidden_size];
        // Symmetric: chosen=+v, rejected=-v → mean=(v+(-v))/2=0 → center penalty≈0
        let examples = vec![
            make_example(1, hidden_size, 1.0, -1.0, None),
            make_example(1, hidden_size, -1.0, 1.0, None),
        ];
        let out_no = batch_reward_loss(&examples, &reward_head, &config_no_penalty).expect("no");
        let out_with = batch_reward_loss(&examples, &reward_head, &config_with_penalty).expect("with");
        // Center reward = 0 when mean is 0 → both losses should be equal
        assert!(
            (out_no.total_loss - out_with.total_loss).abs() < 1e-4,
            "symmetric rewards → center penalty≈0: no={} with={}",
            out_no.total_loss, out_with.total_loss
        );
    }

    // 31. RewardTrainError display messages contain expected strings
    #[test]
    fn test_reward_train_error_display_messages() {
        let e = RewardTrainError::EmptyBatch;
        assert!(e.to_string().to_lowercase().contains("empty"), "EmptyBatch display: {}", e);

        let e = RewardTrainError::HiddenSizeMismatch { expected: 8, actual: 4 };
        let s = e.to_string();
        assert!(s.contains("8") && s.contains("4"), "HiddenSizeMismatch display: {s}");

        let e = RewardTrainError::InvalidScore;
        assert!(e.to_string().to_lowercase().contains("nan") || e.to_string().to_lowercase().contains("inf"),
            "InvalidScore display: {}", e);

        let e = RewardTrainError::MissingScalarLabel;
        assert!(e.to_string().to_lowercase().contains("label"), "MissingScalarLabel display: {}", e);

        let e = RewardTrainError::ZeroHiddenSize;
        assert!(e.to_string().to_lowercase().contains("hidden") || e.to_string().to_lowercase().contains("size"),
            "ZeroHiddenSize display: {}", e);

        let e = RewardTrainError::ZeroSeqLen;
        assert!(e.to_string().to_lowercase().contains("seq") || e.to_string().to_lowercase().contains("len"),
            "ZeroSeqLen display: {}", e);
    }

    // 32. batch_reward_loss mean_margin is positive when chosen consistently higher
    #[test]
    fn test_batch_loss_mean_margin_positive() {
        let hidden_size = 2;
        let reward_head = vec![1.0_f32; hidden_size];
        let config = bt_config(hidden_size);
        let examples: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, 2.0, -1.0, None))
            .collect();
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("out");
        assert!(out.mean_margin > 0.0,
            "mean_margin should be positive when chosen > rejected, got {}", out.mean_margin);
    }

    // 33. RewardModelConfig default has num_labels=1
    #[test]
    fn test_reward_model_config_default_num_labels() {
        let cfg = RewardModelConfig::default();
        assert_eq!(cfg.num_labels, 1, "default num_labels should be 1");
    }

    // 34. batch_reward_loss accuracy=0.0 when all rejected > chosen
    #[test]
    fn test_batch_accuracy_zero_when_all_rejected_higher() {
        let hidden_size = 2;
        let reward_head = vec![1.0_f32; hidden_size];
        let config = bt_config(hidden_size);
        // chosen_val < rejected_val → after dot product, score_chosen < score_rejected
        let examples: Vec<_> = (0..4)
            .map(|_| make_example(1, hidden_size, -2.0, 2.0, None))
            .collect();
        let out = batch_reward_loss(&examples, &reward_head, &config).expect("out");
        assert!((out.accuracy).abs() < 1e-5,
            "accuracy should be 0 when all rejected score higher, got {}", out.accuracy);
    }
}
