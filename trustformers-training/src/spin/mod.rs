//! SPIN: Self-Play Fine-Tuning
//!
//! Reference: "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"
//! (Chen et al., 2024)
//!
//! SPIN trains a model to distinguish its own generations from human-written data,
//! without needing preference pairs. The main player (π_θ) is trained to maximize:
//!   `log P(y_real | x) - log P(y_generated | x)`
//! where `y_generated` comes from the previous iteration's policy (π_θ_t).
//!
//! ## Loss Formulation
//!
//! For a real/generated pair (y_real, y_gen) given prompt x:
//!   `L_SPIN = -log σ(β * (log π_θ(y_real|x) - log π_θ(y_gen|x)))`
//!
//! This is analogous to DPO but `y_gen` is from the model itself — no reference model needed.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during SPIN computations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpinError {
    /// The batch was empty.
    EmptyBatch,
    /// Mismatch between expected and actual number of logit sequences.
    LogitsDimensionMismatch { expected: usize, got: usize },
    /// The vocab size is inconsistent with the logits provided.
    VocabSizeMismatch,
    /// The configuration has an invalid field value.
    InvalidConfig(String),
}

impl fmt::Display for SpinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpinError::EmptyBatch => {
                write!(f, "SPIN error: batch is empty; at least one example is required")
            }
            SpinError::LogitsDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "SPIN error: logits dimension mismatch — expected {expected}, got {got}"
                )
            }
            SpinError::VocabSizeMismatch => {
                write!(f, "SPIN error: vocab size is inconsistent with provided logits")
            }
            SpinError::InvalidConfig(msg) => {
                write!(f, "SPIN invalid config: {msg}")
            }
        }
    }
}

impl std::error::Error for SpinError {}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SPIN (Self-Play Fine-Tuning).
#[derive(Debug, Clone)]
pub struct SpinConfig {
    /// Regularization weight λ controlling the overall loss scale. Default: 0.1
    pub lambda: f32,
    /// Logistic loss temperature β. Higher β means sharper separation. Default: 0.1
    pub beta: f32,
    /// Maximum sequence length for truncation. Default: 512
    pub max_length: usize,
    /// Number of SPIN self-play iterations. Default: 3
    pub iterations: usize,
    /// Number of model-generated responses per prompt. Default: 1
    pub num_generated_per_prompt: usize,
    /// Label smoothing applied to the binary cross-entropy target. Default: 0.0
    pub label_smoothing: f32,
    /// Whether to normalize log-probs by sequence length. Default: true
    pub use_length_normalization: bool,
}

impl Default for SpinConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            beta: 0.1,
            max_length: 512,
            iterations: 3,
            num_generated_per_prompt: 1,
            label_smoothing: 0.0,
            use_length_normalization: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single SPIN training example.
///
/// Each example pairs a human-written response with a model-generated one for
/// the same prompt. The model-generated response comes from the **previous**
/// SPIN iteration's policy (π_θ_t), acting as the "opponent."
#[derive(Debug, Clone)]
pub struct SpinExample {
    /// Input token ids (the prompt / conditioning context).
    pub prompt: Vec<u32>,
    /// Human-written response token ids (the "real" y).
    pub real_response: Vec<u32>,
    /// Model-generated response token ids from the previous iteration (the "fake" y).
    pub generated_response: Vec<u32>,
}

/// A batch of SPIN examples used for a single gradient update.
#[derive(Debug, Clone)]
pub struct SpinBatch {
    /// The individual (prompt, real, generated) triples.
    pub examples: Vec<SpinExample>,
}

impl SpinBatch {
    /// Number of examples in the batch.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the batch contains no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss output
// ─────────────────────────────────────────────────────────────────────────────

/// Detailed output from a SPIN loss computation over a batch.
#[derive(Debug, Clone)]
pub struct SpinLossOutput {
    /// Mean SPIN loss across the batch.
    pub total_loss: f32,
    /// Mean log-probability of real responses (monitoring metric).
    pub real_log_probs_mean: f32,
    /// Mean log-probability of generated responses (monitoring metric).
    pub gen_log_probs_mean: f32,
    /// Probability margin: `real_log_probs_mean - gen_log_probs_mean`.
    /// Positive margin indicates the model assigns higher probability to real data.
    pub prob_margin: f32,
    /// Fraction of examples where real log-prob > generated log-prob.
    /// Should increase during training as the model learns to prefer real data.
    pub accuracy: f32,
    /// Number of examples in the batch.
    pub batch_size: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// SPIN loss computation
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless SPIN loss computation utilities.
pub struct SpinLoss;

impl SpinLoss {
    /// Compute the log-probability of a label sequence given a flat logits array.
    ///
    /// `logits` is a flat array of shape `[seq_len * vocab_size]` where each
    /// consecutive `vocab_size` elements correspond to one time step.
    ///
    /// Applies log-softmax over `vocab_size` at each position and sums the
    /// log-probabilities for each label token.  Optionally divides by sequence
    /// length to normalize across different response lengths.
    ///
    /// Returns `f32::NEG_INFINITY` if `labels` is empty.
    pub fn compute_log_probs(
        logits: &[f32],
        labels: &[u32],
        vocab_size: usize,
        length_normalize: bool,
    ) -> f32 {
        if labels.is_empty() || vocab_size == 0 {
            return f32::NEG_INFINITY;
        }

        let seq_len = labels.len();
        let mut total_log_prob = 0.0_f32;

        for (t, &label) in labels.iter().enumerate() {
            let start = t * vocab_size;
            let end = start + vocab_size;

            // Bounds check: if logits don't cover this position, treat as -inf
            if end > logits.len() {
                return f32::NEG_INFINITY;
            }

            let step_logits = &logits[start..end];

            // Numerically stable log-softmax: subtract max before exp
            let max_logit = step_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = step_logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let log_sum_exp = max_logit + exp_sum.ln();

            let label_idx = label as usize;
            if label_idx >= vocab_size {
                return f32::NEG_INFINITY;
            }

            let log_prob = step_logits[label_idx] - log_sum_exp;
            total_log_prob += log_prob;
        }

        if length_normalize {
            total_log_prob / (seq_len as f32)
        } else {
            total_log_prob
        }
    }

    /// Compute the SPIN loss for a single (real, generated) log-prob pair.
    ///
    /// `L_SPIN = -log σ(β * (real_log_probs - gen_log_probs))`
    ///
    /// This is the same logistic loss as DPO, but the "opponent" is the model's
    /// own previous generation rather than a human-annotated rejected response.
    pub fn compute_spin_loss(real_log_probs: f32, gen_log_probs: f32, config: &SpinConfig) -> f32 {
        let margin = config.beta * (real_log_probs - gen_log_probs);
        // -log σ(x) = log(1 + exp(-x)) — numerically stable form
        softplus(-margin)
    }

    /// Compute the SPIN loss over a full batch.
    ///
    /// # Arguments
    /// * `batch` — the SPIN batch with example metadata (used only for length).
    /// * `logits_real` — one flat `[seq_len * vocab_size]` logit vector per example for the real response.
    /// * `logits_gen`  — one flat `[seq_len * vocab_size]` logit vector per example for the generated response.
    /// * `vocab_size`  — vocabulary size.
    /// * `config`      — SPIN hyper-parameters.
    ///
    /// # Errors
    /// Returns `SpinError::EmptyBatch` if the batch is empty.
    /// Returns `SpinError::LogitsDimensionMismatch` if logit count ≠ batch size.
    pub fn compute_batch_loss(
        batch: &SpinBatch,
        logits_real: &[Vec<f32>],
        logits_gen: &[Vec<f32>],
        vocab_size: usize,
        config: &SpinConfig,
    ) -> Result<SpinLossOutput, SpinError> {
        let n = batch.len();
        if n == 0 {
            return Err(SpinError::EmptyBatch);
        }
        if logits_real.len() != n {
            return Err(SpinError::LogitsDimensionMismatch {
                expected: n,
                got: logits_real.len(),
            });
        }
        if logits_gen.len() != n {
            return Err(SpinError::LogitsDimensionMismatch {
                expected: n,
                got: logits_gen.len(),
            });
        }

        let mut total_loss = 0.0_f32;
        let mut sum_real_lp = 0.0_f32;
        let mut sum_gen_lp = 0.0_f32;
        let mut correct = 0usize;

        for i in 0..n {
            let example = &batch.examples[i];
            let real_labels = &example.real_response;
            let gen_labels = &example.generated_response;

            let real_lp = Self::compute_log_probs(
                &logits_real[i],
                real_labels,
                vocab_size,
                config.use_length_normalization,
            );
            let gen_lp = Self::compute_log_probs(
                &logits_gen[i],
                gen_labels,
                vocab_size,
                config.use_length_normalization,
            );

            let loss = Self::compute_spin_loss(real_lp, gen_lp, config);
            total_loss += loss;
            sum_real_lp += real_lp;
            sum_gen_lp += gen_lp;
            if real_lp > gen_lp {
                correct += 1;
            }
        }

        let n_f = n as f32;
        let real_log_probs_mean = sum_real_lp / n_f;
        let gen_log_probs_mean = sum_gen_lp / n_f;

        Ok(SpinLossOutput {
            total_loss: total_loss / n_f,
            real_log_probs_mean,
            gen_log_probs_mean,
            prob_margin: real_log_probs_mean - gen_log_probs_mean,
            accuracy: correct as f32 / n_f,
            batch_size: n,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iteration statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single SPIN iteration (epoch over the dataset).
#[derive(Debug, Clone)]
pub struct SpinIterationStats {
    /// Which SPIN iteration these stats belong to (0-indexed).
    pub iteration: usize,
    /// Mean loss over all batches in this iteration.
    pub mean_loss: f32,
    /// Mean accuracy (fraction where real > generated) over all batches.
    pub mean_accuracy: f32,
    /// Mean probability margin (real_lp_mean - gen_lp_mean) over all batches.
    pub mean_prob_margin: f32,
    /// Total number of examples processed in this iteration.
    pub num_examples: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// SPIN trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful SPIN trainer that tracks the current self-play iteration and history.
///
/// In each SPIN iteration:
/// 1. Generate responses using the current policy (done outside this struct).
/// 2. Compute SPIN loss via `compute_loss`.
/// 3. Record iteration statistics via `record_iteration`.
/// 4. Advance to the next iteration via `advance_iteration`.
#[derive(Debug, Clone)]
pub struct SpinTrainer {
    /// SPIN hyper-parameters.
    pub config: SpinConfig,
    /// Current self-play iteration index (0-indexed).
    pub iteration: usize,
    /// Per-iteration aggregated statistics (one entry per completed iteration).
    pub history: Vec<SpinIterationStats>,
}

impl SpinTrainer {
    /// Create a new SPIN trainer at iteration 0.
    pub fn new(config: SpinConfig) -> Self {
        Self { config, iteration: 0, history: Vec::new() }
    }

    /// Compute the SPIN batch loss using this trainer's configuration.
    ///
    /// Delegates to `SpinLoss::compute_batch_loss`.
    pub fn compute_loss(
        &self,
        batch: &SpinBatch,
        logits_real: &[Vec<f32>],
        logits_gen: &[Vec<f32>],
        vocab_size: usize,
    ) -> Result<SpinLossOutput, SpinError> {
        SpinLoss::compute_batch_loss(batch, logits_real, logits_gen, vocab_size, &self.config)
    }

    /// Record aggregated statistics for the current (or any) iteration.
    pub fn record_iteration(&mut self, stats: SpinIterationStats) {
        self.history.push(stats);
    }

    /// Advance to the next SPIN iteration.
    pub fn advance_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Return the current SPIN iteration index.
    pub fn current_iteration(&self) -> usize {
        self.iteration
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// For large positive x, returns x directly to avoid overflow.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_spin_config_defaults() {
        let cfg = SpinConfig::default();
        assert_eq!(cfg.lambda, 0.1);
        assert_eq!(cfg.beta, 0.1);
        assert_eq!(cfg.max_length, 512);
        assert_eq!(cfg.iterations, 3);
        assert_eq!(cfg.num_generated_per_prompt, 1);
        assert_eq!(cfg.label_smoothing, 0.0);
        assert!(cfg.use_length_normalization);
    }

    #[test]
    fn test_spin_config_custom() {
        let cfg = SpinConfig {
            lambda: 0.5,
            beta: 0.2,
            max_length: 256,
            iterations: 5,
            num_generated_per_prompt: 4,
            label_smoothing: 0.1,
            use_length_normalization: false,
        };
        assert_eq!(cfg.lambda, 0.5);
        assert_eq!(cfg.beta, 0.2);
        assert_eq!(cfg.max_length, 256);
        assert_eq!(cfg.iterations, 5);
        assert_eq!(cfg.num_generated_per_prompt, 4);
        assert_eq!(cfg.label_smoothing, 0.1);
        assert!(!cfg.use_length_normalization);
    }

    // ── SpinBatch helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_spin_batch_len_is_empty() {
        let empty = SpinBatch { examples: vec![] };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let one = SpinBatch {
            examples: vec![SpinExample {
                prompt: vec![1],
                real_response: vec![2],
                generated_response: vec![3],
            }],
        };
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    // ── compute_log_probs (single token) ─────────────────────────────────────

    #[test]
    fn test_compute_log_probs_single_token_uniform() {
        // Uniform logits: log_prob of any token = log(1/vocab_size)
        let vocab_size = 4_usize;
        let logits = vec![0.0_f32; vocab_size]; // uniform
        let labels = vec![2_u32];

        // Without length normalization: should be log(0.25) = -ln(4)
        let lp = SpinLoss::compute_log_probs(&logits, &labels, vocab_size, false);
        let expected = -(vocab_size as f32).ln();
        assert!((lp - expected).abs() < 1e-5, "Expected {expected}, got {lp}");
    }

    #[test]
    fn test_compute_log_probs_single_token_peaked() {
        // Very high logit at token 1, low elsewhere: log_prob(1) ≈ 0
        let vocab_size = 4_usize;
        let logits = vec![-1000.0_f32, 1000.0, -1000.0, -1000.0];
        let labels = vec![1_u32];

        let lp = SpinLoss::compute_log_probs(&logits, &labels, vocab_size, false);
        // log_prob should be very close to 0
        assert!(lp > -0.01, "peaked logit should give log_prob ≈ 0, got {lp}");
    }

    #[test]
    fn test_compute_log_probs_length_normalization() {
        let vocab_size = 4_usize;
        // 2-token sequence, all uniform logits
        let logits = vec![0.0_f32; vocab_size * 2];
        let labels = vec![0_u32, 1_u32];

        let lp_unnorm = SpinLoss::compute_log_probs(&logits, &labels, vocab_size, false);
        let lp_norm = SpinLoss::compute_log_probs(&logits, &labels, vocab_size, true);

        // Normalized should be half of unnormalized (2 tokens)
        assert!(
            (lp_norm - lp_unnorm / 2.0).abs() < 1e-5,
            "length normalized should be unnorm/2: {lp_norm} vs {}", lp_unnorm / 2.0
        );
    }

    #[test]
    fn test_compute_log_probs_empty_labels() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let lp = SpinLoss::compute_log_probs(&logits, &[], 3, false);
        assert_eq!(lp, f32::NEG_INFINITY, "empty labels should give NEG_INFINITY");
    }

    // ── SPIN loss formula ─────────────────────────────────────────────────────

    #[test]
    fn test_spin_loss_zero_margin_equals_log2() {
        // When margin = 0: L = -log σ(0) = -log(0.5) = log(2)
        let config = SpinConfig::default();
        let loss = SpinLoss::compute_spin_loss(0.0, 0.0, &config);
        let expected = 2.0_f32.ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "SPIN loss at zero margin should equal log(2) ≈ {expected}, got {loss}"
        );
    }

    #[test]
    fn test_spin_loss_positive_margin_less_than_log2() {
        // When real > gen (positive margin), loss < log(2)
        let config = SpinConfig { beta: 1.0, ..SpinConfig::default() };
        let loss = SpinLoss::compute_spin_loss(2.0, 0.0, &config);
        let log2 = 2.0_f32.ln();
        assert!(loss < log2, "positive margin should reduce loss below log(2): {loss} vs {log2}");
    }

    #[test]
    fn test_spin_loss_large_positive_margin_near_zero() {
        // Very large positive margin → loss ≈ 0
        let config = SpinConfig { beta: 1.0, ..SpinConfig::default() };
        let loss = SpinLoss::compute_spin_loss(100.0, 0.0, &config);
        assert!(loss < 1e-6, "huge positive margin should give loss ≈ 0, got {loss}");
    }

    #[test]
    fn test_spin_loss_beta_scaling() {
        // Higher β amplifies the margin, reducing loss faster
        let cfg_low = SpinConfig { beta: 0.1, ..SpinConfig::default() };
        let cfg_high = SpinConfig { beta: 2.0, ..SpinConfig::default() };
        let margin_real = 3.0_f32;
        let margin_gen = 0.0_f32;

        let loss_low = SpinLoss::compute_spin_loss(margin_real, margin_gen, &cfg_low);
        let loss_high = SpinLoss::compute_spin_loss(margin_real, margin_gen, &cfg_high);
        assert!(
            loss_high < loss_low,
            "higher beta should give lower loss for positive margin: {loss_high} vs {loss_low}"
        );
    }

    // ── compute_batch_loss ────────────────────────────────────────────────────

    fn make_uniform_logits(seq_len: usize, vocab_size: usize) -> Vec<f32> {
        vec![0.0_f32; seq_len * vocab_size]
    }

    fn make_batch_with_n(n: usize, real_len: usize, gen_len: usize) -> SpinBatch {
        let examples = (0..n)
            .map(|_| SpinExample {
                prompt: vec![0_u32],
                real_response: (0..real_len as u32).collect(),
                generated_response: (0..gen_len as u32).collect(),
            })
            .collect();
        SpinBatch { examples }
    }

    #[test]
    fn test_batch_loss_single_example_uniform() {
        let vocab_size = 4_usize;
        let real_len = 2_usize;
        let gen_len = 2_usize;
        let batch = make_batch_with_n(1, real_len, gen_len);

        // Uniform logits give identical log-probs for real and gen → margin = 0 → loss = log(2)
        let logits_real = vec![make_uniform_logits(real_len, vocab_size)];
        let logits_gen = vec![make_uniform_logits(gen_len, vocab_size)];

        let config = SpinConfig { beta: 1.0, ..SpinConfig::default() };
        let out = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, vocab_size, &config)
            .expect("should succeed");

        let expected_loss = 2.0_f32.ln();
        assert!(
            (out.total_loss - expected_loss).abs() < 1e-5,
            "uniform logits → loss = log(2): {}, got {}",
            expected_loss,
            out.total_loss
        );
        assert_eq!(out.batch_size, 1);
    }

    #[test]
    fn test_batch_loss_empty_batch_error() {
        let empty = SpinBatch { examples: vec![] };
        let config = SpinConfig::default();
        let result = SpinLoss::compute_batch_loss(&empty, &[], &[], 4, &config);
        assert!(matches!(result, Err(SpinError::EmptyBatch)));
    }

    #[test]
    fn test_batch_loss_dimension_mismatch_real() {
        let batch = make_batch_with_n(2, 2, 2);
        let logits_real = vec![make_uniform_logits(2, 4)]; // only 1, not 2
        let logits_gen = vec![make_uniform_logits(2, 4), make_uniform_logits(2, 4)];
        let config = SpinConfig::default();
        let result = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, 4, &config);
        assert!(
            matches!(result, Err(SpinError::LogitsDimensionMismatch { expected: 2, got: 1 })),
            "got: {result:?}"
        );
    }

    #[test]
    fn test_batch_loss_dimension_mismatch_gen() {
        let batch = make_batch_with_n(2, 2, 2);
        let logits_real = vec![make_uniform_logits(2, 4), make_uniform_logits(2, 4)];
        let logits_gen = vec![make_uniform_logits(2, 4)]; // only 1
        let config = SpinConfig::default();
        let result = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, 4, &config);
        assert!(
            matches!(result, Err(SpinError::LogitsDimensionMismatch { expected: 2, got: 1 })),
            "got: {result:?}"
        );
    }

    // ── accuracy metric ───────────────────────────────────────────────────────

    #[test]
    fn test_accuracy_all_correct() {
        // Real logits strongly prefer token 0; gen logits prefer token 1.
        // So real_lp > gen_lp for all examples → accuracy = 1.0
        let vocab_size = 2_usize;
        let n = 3;
        let batch = make_batch_with_n(n, 1, 1);

        // Real: high logit at label 0 (index 0)
        let real_logits_step = vec![100.0_f32, -100.0];
        // Gen: high logit at label 0 (index 0) but reversed → lower overall
        let gen_logits_step = vec![-100.0_f32, 100.0]; // label 0 gets -100 logit

        let logits_real = vec![real_logits_step.clone(); n];
        let logits_gen = vec![gen_logits_step.clone(); n];

        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let out = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, vocab_size, &config)
            .expect("should succeed");

        assert!(
            (out.accuracy - 1.0).abs() < 1e-6,
            "expected accuracy 1.0, got {}",
            out.accuracy
        );
    }

    // ── probability margin ────────────────────────────────────────────────────

    #[test]
    fn test_prob_margin_equals_diff_of_means() {
        let vocab_size = 4_usize;
        let batch = make_batch_with_n(2, 2, 2);
        let logits_real = vec![make_uniform_logits(2, vocab_size); 2];
        let logits_gen = vec![make_uniform_logits(2, vocab_size); 2];
        let config = SpinConfig::default();

        let out = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, vocab_size, &config)
            .expect("should succeed");

        assert!(
            (out.prob_margin - (out.real_log_probs_mean - out.gen_log_probs_mean)).abs() < 1e-6,
            "prob_margin should equal real_mean - gen_mean"
        );
    }

    // ── trainer iteration tracking ────────────────────────────────────────────

    #[test]
    fn test_spin_trainer_iteration_tracking() {
        let mut trainer = SpinTrainer::new(SpinConfig::default());
        assert_eq!(trainer.current_iteration(), 0);

        trainer.advance_iteration();
        assert_eq!(trainer.current_iteration(), 1);

        trainer.advance_iteration();
        assert_eq!(trainer.current_iteration(), 2);
    }

    // ── history recording ─────────────────────────────────────────────────────

    #[test]
    fn test_spin_trainer_history_recording() {
        let mut trainer = SpinTrainer::new(SpinConfig::default());
        assert!(trainer.history.is_empty());

        trainer.record_iteration(SpinIterationStats {
            iteration: 0,
            mean_loss: 0.693,
            mean_accuracy: 0.5,
            mean_prob_margin: 0.0,
            num_examples: 100,
        });
        trainer.advance_iteration();

        trainer.record_iteration(SpinIterationStats {
            iteration: 1,
            mean_loss: 0.45,
            mean_accuracy: 0.7,
            mean_prob_margin: 1.2,
            num_examples: 100,
        });

        assert_eq!(trainer.history.len(), 2);
        assert!((trainer.history[0].mean_loss - 0.693).abs() < 1e-3);
        assert!((trainer.history[1].mean_accuracy - 0.7).abs() < 1e-6);
        assert_eq!(trainer.current_iteration(), 1);
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_spin_error_display() {
        let e1 = SpinError::EmptyBatch;
        let e2 = SpinError::LogitsDimensionMismatch { expected: 4, got: 2 };
        let e3 = SpinError::VocabSizeMismatch;
        let e4 = SpinError::InvalidConfig("bad beta".to_string());

        assert!(e1.to_string().contains("empty"));
        assert!(e2.to_string().contains("4"));
        assert!(e2.to_string().contains("2"));
        assert!(e3.to_string().contains("vocab"));
        assert!(e4.to_string().contains("bad beta"));
    }

    // ── trainer compute_loss delegation ──────────────────────────────────────

    #[test]
    fn test_spin_trainer_compute_loss_delegation_ext() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: true, ..SpinConfig::default() };
        let trainer = SpinTrainer::new(config);
        let vocab_size = 4_usize;
        let batch = make_batch_with_n(1, 2, 2);
        let logits_real = vec![make_uniform_logits(2, vocab_size)];
        let logits_gen = vec![make_uniform_logits(2, vocab_size)];

        let out = trainer
            .compute_loss(&batch, &logits_real, &logits_gen, vocab_size)
            .expect("should succeed");

        // Uniform logits → margin = 0 → loss = log(2)
        let expected = 2.0_f32.ln();
        assert!(
            (out.total_loss - expected).abs() < 1e-5,
            "trainer delegation should match direct call: {} vs {}",
            out.total_loss,
            expected
        );
    }

    // ── New tests ─────────────────────────────────────────────────────────────

    // Test: self-play loss approaches 0 when real log-prob >> gen log-prob
    #[test]
    fn test_self_play_loss_approaches_zero_when_real_dominates() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let loss = SpinLoss::compute_spin_loss(0.0_f32, -100.0_f32, &config);
        assert!(loss < 1e-5, "loss should approach 0 when real >> gen, got {loss}");
    }

    // Test: SPIN loss is large when generated dominates over real
    #[test]
    fn test_self_play_loss_large_when_gen_dominates() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let loss = SpinLoss::compute_spin_loss(-100.0_f32, 0.0_f32, &config);
        assert!(loss > 50.0, "loss should be very large when gen >> real, got {loss}");
    }

    // Test: SPIN reward — real data is distinguishable from generated data
    #[test]
    fn test_spin_reward_generated_distinguishable_from_real() {
        let vocab_size = 2_usize;
        let n = 4;
        let batch = make_batch_with_n(n, 1, 1);
        let real_logits = vec![vec![10.0_f32, -10.0]; n];
        let gen_logits = vec![vec![-10.0_f32, 10.0]; n];
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let out = SpinLoss::compute_batch_loss(&batch, &real_logits, &gen_logits, vocab_size, &config)
            .expect("ok");
        assert!(out.prob_margin > 0.0, "real should have higher log-prob than gen, margin={}", out.prob_margin);
        assert!(out.accuracy > 0.5, "accuracy should be high when real dominates");
    }

    // Test: discriminator predicts real=high, generated=low
    #[test]
    fn test_discriminator_predicts_real_high_gen_low() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let correct_loss = SpinLoss::compute_spin_loss(0.0_f32, -5.0_f32, &config);
        let wrong_loss = SpinLoss::compute_spin_loss(-5.0_f32, 0.0_f32, &config);
        assert!(
            correct_loss < wrong_loss,
            "correct discrimination should give lower loss: correct={correct_loss}, wrong={wrong_loss}"
        );
    }

    // Test: player 0 (LM) update direction — increasing real log-prob decreases loss
    #[test]
    fn test_player_zero_update_direction_prefers_real() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let loss_before = SpinLoss::compute_spin_loss(0.0_f32, 0.0_f32, &config);
        let loss_after = SpinLoss::compute_spin_loss(2.0_f32, 0.0_f32, &config);
        assert!(
            loss_after < loss_before,
            "improving real log-prob should decrease loss: before={loss_before}, after={loss_after}"
        );
    }

    // Test: player 1 (discriminator) update — decreasing gen log-prob decreases loss
    #[test]
    fn test_player_one_update_direction_distinguishes_gen() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let loss_hard = SpinLoss::compute_spin_loss(0.0_f32, 0.0_f32, &config);
        let loss_easy = SpinLoss::compute_spin_loss(0.0_f32, -5.0_f32, &config);
        assert!(
            loss_easy < loss_hard,
            "distinguishable gen reduces loss: easy={loss_easy}, hard={loss_hard}"
        );
    }

    // Test: convergence — as generated improves, prob_margin decreases
    #[test]
    fn test_spin_convergence_prob_margin_decreases() {
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let vocab_size = 2_usize;
        let batch = make_batch_with_n(4, 1, 1);
        let real_logits = vec![vec![10.0_f32, -10.0]; 4];
        // Iteration 0: gen very far from real
        let gen_logits_iter0 = vec![vec![-10.0_f32, 10.0]; 4];
        // Iteration 1: gen closer to real
        let gen_logits_iter1 = vec![vec![5.0_f32, -5.0]; 4];
        let out0 = SpinLoss::compute_batch_loss(&batch, &real_logits, &gen_logits_iter0, vocab_size, &config).expect("ok");
        let out1 = SpinLoss::compute_batch_loss(&batch, &real_logits, &gen_logits_iter1, vocab_size, &config).expect("ok");
        assert!(
            out1.prob_margin < out0.prob_margin,
            "margin should decrease as gen improves: iter0={}, iter1={}", out0.prob_margin, out1.prob_margin
        );
    }

    // Test: SPR (Single Play Round) — one full iteration cycle
    #[test]
    fn test_single_play_round_one_iteration() {
        let mut trainer = SpinTrainer::new(SpinConfig::default());
        assert_eq!(trainer.current_iteration(), 0);
        let vocab_size = 4_usize;
        let batch = make_batch_with_n(2, 3, 3);
        let logits_real = vec![make_uniform_logits(3, vocab_size); 2];
        let logits_gen = vec![make_uniform_logits(3, vocab_size); 2];
        let out = trainer.compute_loss(&batch, &logits_real, &logits_gen, vocab_size).expect("ok");
        trainer.record_iteration(SpinIterationStats {
            iteration: 0,
            mean_loss: out.total_loss,
            mean_accuracy: out.accuracy,
            mean_prob_margin: out.prob_margin,
            num_examples: out.batch_size,
        });
        trainer.advance_iteration();
        assert_eq!(trainer.current_iteration(), 1);
        assert_eq!(trainer.history.len(), 1);
        assert!((trainer.history[0].mean_loss - out.total_loss).abs() < 1e-6);
    }

    // Test: batch aggregation — mean loss over N identical examples = single example loss
    #[test]
    fn test_batch_aggregation_mean_loss_invariant() {
        let vocab_size = 4_usize;
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let batch1 = make_batch_with_n(1, 2, 2);
        let logits_real_1 = vec![make_uniform_logits(2, vocab_size)];
        let logits_gen_1 = vec![make_uniform_logits(2, vocab_size)];
        let out1 = SpinLoss::compute_batch_loss(&batch1, &logits_real_1, &logits_gen_1, vocab_size, &config).expect("ok");
        let n = 5_usize;
        let batch5 = make_batch_with_n(n, 2, 2);
        let logits_real_5 = vec![make_uniform_logits(2, vocab_size); n];
        let logits_gen_5 = vec![make_uniform_logits(2, vocab_size); n];
        let out5 = SpinLoss::compute_batch_loss(&batch5, &logits_real_5, &logits_gen_5, vocab_size, &config).expect("ok");
        assert!(
            (out1.total_loss - out5.total_loss).abs() < 1e-5,
            "batch of 5 identical examples should give same mean loss: 1={}, 5={}", out1.total_loss, out5.total_loss
        );
    }

    // Test: sampling temperature effect — peaked logits give higher log-prob for chosen token
    #[test]
    fn test_sampling_temperature_peaked_vs_uniform() {
        let vocab_size = 4_usize;
        let labels = vec![0_u32];
        // "Cold" (peaked) logits: very high at token 0
        let logits_cold = vec![100.0_f32, 0.0, 0.0, 0.0];
        // "Warm" (uniform) logits
        let logits_warm = vec![0.0_f32; vocab_size];
        let lp_cold = SpinLoss::compute_log_probs(&logits_cold, &labels, vocab_size, false);
        let lp_warm = SpinLoss::compute_log_probs(&logits_warm, &labels, vocab_size, false);
        assert!(
            lp_cold > lp_warm,
            "peaked (cold) logits should give higher log-prob: cold={lp_cold}, warm={lp_warm}"
        );
    }

    // Test: length normalization makes short and long sequences comparable
    #[test]
    fn test_length_normalization_equalizes_short_long() {
        let vocab_size = 4_usize;
        let logits_short = vec![0.0_f32; vocab_size];
        let labels_short = vec![0_u32];
        let logits_long = vec![0.0_f32; vocab_size * 4];
        let labels_long = vec![0_u32; 4];
        let lp_short_norm = SpinLoss::compute_log_probs(&logits_short, &labels_short, vocab_size, true);
        let lp_long_norm = SpinLoss::compute_log_probs(&logits_long, &labels_long, vocab_size, true);
        // With normalization: same per-token log-prob → equal values
        assert!(
            (lp_short_norm - lp_long_norm).abs() < 1e-5,
            "length-normalized log-probs should be equal for uniform logits: short={lp_short_norm}, long={lp_long_norm}"
        );
        // Without normalization: longer sequence is more negative
        let lp_short_raw = SpinLoss::compute_log_probs(&logits_short, &labels_short, vocab_size, false);
        let lp_long_raw = SpinLoss::compute_log_probs(&logits_long, &labels_long, vocab_size, false);
        assert!(lp_long_raw < lp_short_raw, "raw long seq more negative: short={lp_short_raw}, long={lp_long_raw}");
    }

    // Test: numerical — known logprob difference produces expected loss value
    #[test]
    fn test_numerical_known_logprob_difference_produces_expected_loss() {
        // beta=1, real_lp=0, gen_lp=-2 → margin=2 → loss = ln(1 + e^{-2})
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let loss = SpinLoss::compute_spin_loss(0.0_f32, -2.0_f32, &config);
        let expected = (1.0_f32 + (-2.0_f32).exp()).ln();
        assert!((loss - expected).abs() < 1e-5, "expected {expected}, got {loss}");
    }

    // Test: accuracy = 0.0 when all generated > real
    #[test]
    fn test_accuracy_all_incorrect() {
        let vocab_size = 2_usize;
        let n = 3;
        let batch = make_batch_with_n(n, 1, 1);
        // Real labels are token 0, but logits strongly prefer token 1 → low real log-prob
        let real_logits = vec![vec![-100.0_f32, 100.0]; n];
        // Gen labels are token 0, logits strongly prefer token 0 → high gen log-prob
        let gen_logits = vec![vec![100.0_f32, -100.0]; n];
        let config = SpinConfig { beta: 1.0, use_length_normalization: false, ..SpinConfig::default() };
        let out = SpinLoss::compute_batch_loss(&batch, &real_logits, &gen_logits, vocab_size, &config).expect("ok");
        assert!(out.accuracy < 1e-6, "accuracy should be 0.0 when gen always has higher log-prob, got {}", out.accuracy);
    }

    // Test: softplus large-positive numerically stable
    #[test]
    fn test_softplus_large_positive_is_stable() {
        let result = softplus(100.0_f32);
        assert!(result.is_finite() && (result - 100.0_f32).abs() < 1.0, "softplus(100) ≈ 100, got {result}");
    }

    // Test: softplus at zero = log(2)
    #[test]
    fn test_softplus_at_zero_is_log2() {
        let result = softplus(0.0_f32);
        assert!((result - (2.0_f32).ln()).abs() < 1e-6, "softplus(0)=log(2), got {result}");
    }

    // Test: config iterations and num_generated_per_prompt are properly stored
    #[test]
    fn test_config_iterations_and_generated_per_prompt() {
        let config = SpinConfig { iterations: 5, num_generated_per_prompt: 4, ..SpinConfig::default() };
        assert_eq!(config.iterations, 5);
        assert_eq!(config.num_generated_per_prompt, 4);
    }

    // Test: batch_size field in output matches batch input
    #[test]
    fn test_batch_size_field_matches_input() {
        let vocab_size = 4_usize;
        let n = 7;
        let batch = make_batch_with_n(n, 2, 2);
        let logits_real = vec![make_uniform_logits(2, vocab_size); n];
        let logits_gen = vec![make_uniform_logits(2, vocab_size); n];
        let config = SpinConfig::default();
        let out = SpinLoss::compute_batch_loss(&batch, &logits_real, &logits_gen, vocab_size, &config).expect("ok");
        assert_eq!(out.batch_size, n, "batch_size should match input batch size");
    }
}
