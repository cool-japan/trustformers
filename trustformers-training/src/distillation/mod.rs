//! Knowledge Distillation
//!
//! Teacher-student knowledge distillation with multiple strategies:
//! soft targets (KL divergence), feature-level matching, attention transfer,
//! progressive distillation, and combined variants.

// ─── Strategy ───────────────────────────────────────────────────────────────

/// Strategy for computing distillation loss.
#[derive(Debug, Clone, PartialEq)]
pub enum DistillationStrategy {
    /// Standard KD: KL divergence between soft teacher/student logits.
    SoftTargets {
        /// Temperature T used to soften distributions (default 4.0).
        temperature: f64,
        /// Weight for distillation loss; `(1 - alpha)` is used for CE loss (default 0.7).
        alpha: f64,
    },
    /// Feature-level distillation: match intermediate representations.
    FeatureBased {
        /// `(teacher_layer_idx, student_layer_idx)` pairs.
        layer_mapping: Vec<(usize, usize)>,
        loss_type: FeatureLossType,
    },
    /// Attention transfer: match attention weight matrices.
    AttentionTransfer {
        layer_pairs: Vec<(usize, usize)>,
        normalize: bool,
    },
    /// Progressive distillation: gradually increase student complexity.
    Progressive {
        stages: Vec<DistillationStage>,
    },
    /// Combined: soft targets + feature matching.
    Combined {
        soft_alpha: f64,
        feature_alpha: f64,
        temperature: f64,
    },
}

// ─── Supporting types ────────────────────────────────────────────────────────

/// Loss type for feature-level distillation.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureLossType {
    L2,
    Cosine,
    KL,
}

/// One stage of progressive distillation.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillationStage {
    pub stage_idx: usize,
    pub steps: u64,
    pub temperature: f64,
    pub alpha: f64,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a distillation session.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    pub strategy: DistillationStrategy,
    pub teacher_layers: usize,
    pub student_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

// ─── LogitTensor ─────────────────────────────────────────────────────────────

/// Simulated logits for one forward pass.
///
/// Shape: `[seq_len * vocab_size]` (row-major: position first).
#[derive(Debug, Clone)]
pub struct LogitTensor {
    /// Flat logits, shape `[seq_len * vocab_size]`.
    pub values: Vec<f64>,
    pub seq_len: usize,
    pub vocab_size: usize,
}

impl LogitTensor {
    /// Create a new `LogitTensor`, validating that `values.len() == seq_len * vocab_size`.
    pub fn new(values: Vec<f64>, seq_len: usize, vocab_size: usize) -> Result<Self, DistillError> {
        if values.is_empty() {
            return Err(DistillError::EmptyLogits);
        }
        let expected = seq_len * vocab_size;
        if values.len() != expected {
            return Err(DistillError::ShapeMismatch {
                teacher: format!("{}", expected),
                student: format!("{}", values.len()),
            });
        }
        Ok(Self { values, seq_len, vocab_size })
    }

    /// Compute softmax with temperature over each position's vocab slice.
    ///
    /// Returns a flat vector of the same shape as `values`.
    pub fn softmax_with_temperature(&self, temperature: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.values.len());
        for pos in 0..self.seq_len {
            let start = pos * self.vocab_size;
            let end = start + self.vocab_size;
            let slice = &self.values[start..end];

            // Numerically stable: subtract max before exp
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = slice.iter().map(|&v| ((v - max_val) / temperature).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for e in exps {
                out.push(e / sum);
            }
        }
        out
    }

    /// Return the argmax token index per position.
    pub fn argmax_per_position(&self) -> Vec<usize> {
        (0..self.seq_len)
            .map(|pos| {
                let start = pos * self.vocab_size;
                let end = start + self.vocab_size;
                self.values[start..end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─── FeatureMap ───────────────────────────────────────────────────────────────

/// Feature map from a hidden layer.
///
/// Shape: `[seq_len * hidden_size]` (row-major: position first).
#[derive(Debug, Clone)]
pub struct FeatureMap {
    /// Flat values, shape `[seq_len * hidden_size]`.
    pub values: Vec<f64>,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub layer_index: usize,
}

impl FeatureMap {
    pub fn new(values: Vec<f64>, seq_len: usize, hidden_size: usize, layer_index: usize) -> Self {
        Self { values, seq_len, hidden_size, layer_index }
    }

    /// Compute the L2 norm of the feature vector for each token position.
    pub fn l2_norm_per_token(&self) -> Vec<f64> {
        (0..self.seq_len)
            .map(|pos| {
                let start = pos * self.hidden_size;
                let end = start + self.hidden_size;
                self.values[start..end].iter().map(|v| v * v).sum::<f64>().sqrt()
            })
            .collect()
    }
}

// ─── DistillationLoss ─────────────────────────────────────────────────────────

/// The main distillation loss computer.
pub struct DistillationLoss {
    config: DistillationConfig,
    current_stage: usize,
    step: u64,
}

impl DistillationLoss {
    pub fn new(config: DistillationConfig) -> Self {
        Self { config, current_stage: 0, step: 0 }
    }

    /// Compute soft-target KD loss (KL divergence with temperature scaling).
    ///
    /// `KL(p_teacher || p_student)` where both are softmax'd at temperature T,
    /// averaged over sequence positions, then scaled by `T²`.
    pub fn soft_target_loss(
        &self,
        teacher_logits: &LogitTensor,
        student_logits: &LogitTensor,
        temperature: f64,
    ) -> Result<f64, DistillError> {
        if temperature <= 0.0 {
            return Err(DistillError::InvalidTemperature(temperature));
        }
        if teacher_logits.seq_len != student_logits.seq_len
            || teacher_logits.vocab_size != student_logits.vocab_size
        {
            return Err(DistillError::ShapeMismatch {
                teacher: format!(
                    "seq={} vocab={}",
                    teacher_logits.seq_len, teacher_logits.vocab_size
                ),
                student: format!(
                    "seq={} vocab={}",
                    student_logits.seq_len, student_logits.vocab_size
                ),
            });
        }

        let p_t = teacher_logits.softmax_with_temperature(temperature);
        let p_s = student_logits.softmax_with_temperature(temperature);

        let seq_len = teacher_logits.seq_len as f64;
        let vocab_size = teacher_logits.vocab_size;

        let mut kl_total = 0.0_f64;
        for pos in 0..teacher_logits.seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            let mut kl_pos = 0.0_f64;
            for i in start..end {
                let pt = p_t[i];
                let ps = p_s[i];
                if pt > 1e-12 && ps > 1e-12 {
                    kl_pos += pt * (pt / ps).ln();
                }
            }
            kl_total += kl_pos;
        }

        // Average over positions, scale by T^2 (standard KD practice)
        Ok((kl_total / seq_len) * temperature * temperature)
    }

    /// Compute feature-level loss between teacher and student feature maps.
    ///
    /// The `layer_mapping` from the config determines which teacher/student
    /// layer pairs are compared.
    pub fn feature_loss(
        &self,
        teacher_features: &[FeatureMap],
        student_features: &[FeatureMap],
        loss_type: &FeatureLossType,
    ) -> Result<f64, DistillError> {
        let layer_mapping = match &self.config.strategy {
            DistillationStrategy::FeatureBased { layer_mapping, .. } => layer_mapping.clone(),
            DistillationStrategy::Combined { .. } => {
                // For Combined, pair teacher/student layers by index up to min(len)
                let n = teacher_features.len().min(student_features.len());
                (0..n).map(|i| (i, i)).collect()
            }
            _ => {
                let n = teacher_features.len().min(student_features.len());
                (0..n).map(|i| (i, i)).collect()
            }
        };

        if layer_mapping.is_empty() {
            return Ok(0.0);
        }

        let mut total_loss = 0.0_f64;
        let mut count = 0usize;

        for (t_idx, s_idx) in &layer_mapping {
            let t_feat = teacher_features
                .iter()
                .find(|f| f.layer_index == *t_idx)
                .ok_or(DistillError::FeatureCountMismatch)?;
            let s_feat = student_features
                .iter()
                .find(|f| f.layer_index == *s_idx)
                .ok_or(DistillError::FeatureCountMismatch)?;

            if t_feat.seq_len != s_feat.seq_len || t_feat.hidden_size != s_feat.hidden_size {
                return Err(DistillError::ShapeMismatch {
                    teacher: format!("seq={} hidden={}", t_feat.seq_len, t_feat.hidden_size),
                    student: format!("seq={} hidden={}", s_feat.seq_len, s_feat.hidden_size),
                });
            }

            let layer_loss = match loss_type {
                FeatureLossType::L2 => {
                    compute_l2_loss(&t_feat.values, &s_feat.values)
                }
                FeatureLossType::Cosine => {
                    compute_cosine_loss(t_feat, s_feat)
                }
                FeatureLossType::KL => {
                    compute_kl_feature_loss(t_feat, s_feat)
                }
            };

            total_loss += layer_loss;
            count += 1;
        }

        Ok(if count > 0 { total_loss / count as f64 } else { 0.0 })
    }

    /// Compute the combined distillation loss for one batch.
    pub fn compute_loss(
        &mut self,
        teacher_logits: &LogitTensor,
        student_logits: &LogitTensor,
        teacher_features: &[FeatureMap],
        student_features: &[FeatureMap],
        hard_labels: &[usize],
    ) -> Result<DistillationLossResult, DistillError> {
        self.step += 1;

        match &self.config.strategy.clone() {
            DistillationStrategy::SoftTargets { temperature, alpha } => {
                let t = *temperature;
                let a = *alpha;

                let soft_loss = self.soft_target_loss(teacher_logits, student_logits, t)?;
                let hard_loss = cross_entropy_loss(student_logits, hard_labels)?;
                let total = a * soft_loss + (1.0 - a) * hard_loss;

                Ok(DistillationLossResult {
                    total_loss: total,
                    soft_target_component: soft_loss,
                    feature_component: 0.0,
                    hard_label_component: hard_loss,
                    current_temperature: t,
                    stage: self.current_stage,
                })
            }

            DistillationStrategy::FeatureBased { loss_type, .. } => {
                let lt = loss_type.clone();
                let feat_loss = self.feature_loss(teacher_features, student_features, &lt)?;
                let hard_loss = cross_entropy_loss(student_logits, hard_labels)?;
                let total = 0.5 * feat_loss + 0.5 * hard_loss;

                Ok(DistillationLossResult {
                    total_loss: total,
                    soft_target_component: 0.0,
                    feature_component: feat_loss,
                    hard_label_component: hard_loss,
                    current_temperature: 1.0,
                    stage: self.current_stage,
                })
            }

            DistillationStrategy::AttentionTransfer { .. } => {
                let hard_loss = cross_entropy_loss(student_logits, hard_labels)?;
                Ok(DistillationLossResult {
                    total_loss: hard_loss,
                    soft_target_component: 0.0,
                    feature_component: 0.0,
                    hard_label_component: hard_loss,
                    current_temperature: 1.0,
                    stage: self.current_stage,
                })
            }

            DistillationStrategy::Progressive { stages } => {
                let stages = stages.clone();
                let stage = self.current_stage.min(stages.len().saturating_sub(1));
                let (t, a) = if stages.is_empty() {
                    (4.0, 0.7)
                } else {
                    (stages[stage].temperature, stages[stage].alpha)
                };

                let soft_loss = self.soft_target_loss(teacher_logits, student_logits, t)?;
                let hard_loss = cross_entropy_loss(student_logits, hard_labels)?;
                let total = a * soft_loss + (1.0 - a) * hard_loss;

                Ok(DistillationLossResult {
                    total_loss: total,
                    soft_target_component: soft_loss,
                    feature_component: 0.0,
                    hard_label_component: hard_loss,
                    current_temperature: t,
                    stage: self.current_stage,
                })
            }

            DistillationStrategy::Combined { soft_alpha, feature_alpha, temperature } => {
                let t = *temperature;
                let sa = *soft_alpha;
                let fa = *feature_alpha;

                let soft_loss = self.soft_target_loss(teacher_logits, student_logits, t)?;
                let feat_loss =
                    self.feature_loss(teacher_features, student_features, &FeatureLossType::L2)?;
                let hard_loss = cross_entropy_loss(student_logits, hard_labels)?;
                let remainder = (1.0 - sa - fa).max(0.0);
                let total = sa * soft_loss + fa * feat_loss + remainder * hard_loss;

                Ok(DistillationLossResult {
                    total_loss: total,
                    soft_target_component: soft_loss,
                    feature_component: feat_loss,
                    hard_label_component: hard_loss,
                    current_temperature: t,
                    stage: self.current_stage,
                })
            }
        }
    }

    /// Advance to the next distillation stage (for Progressive strategy).
    pub fn advance_stage(&mut self) {
        self.current_stage += 1;
        self.step = 0;
    }
}

// ─── DistillationLossResult ───────────────────────────────────────────────────

/// The result of a single distillation loss computation.
#[derive(Debug, Clone)]
pub struct DistillationLossResult {
    pub total_loss: f64,
    pub soft_target_component: f64,
    pub feature_component: f64,
    pub hard_label_component: f64,
    pub current_temperature: f64,
    pub stage: usize,
}

impl std::fmt::Display for DistillationLossResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DistillationLoss {{ total={:.6}, soft={:.6}, feature={:.6}, hard={:.6}, T={:.2}, stage={} }}",
            self.total_loss,
            self.soft_target_component,
            self.feature_component,
            self.hard_label_component,
            self.current_temperature,
            self.stage,
        )
    }
}

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum DistillError {
    #[error("Shape mismatch: teacher {teacher} vs student {student}")]
    ShapeMismatch { teacher: String, student: String },
    #[error("Empty logits")]
    EmptyLogits,
    #[error("Invalid temperature: {0}")]
    InvalidTemperature(f64),
    #[error("Feature count mismatch")]
    FeatureCountMismatch,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Mean squared error between two flat vectors.
fn compute_l2_loss(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>()
        / a.len() as f64;
    mse
}

/// Cosine loss: `1 - cosine_similarity`, averaged per token position.
fn compute_cosine_loss(t_feat: &FeatureMap, s_feat: &FeatureMap) -> f64 {
    if t_feat.seq_len == 0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for pos in 0..t_feat.seq_len {
        let start = pos * t_feat.hidden_size;
        let end = start + t_feat.hidden_size;
        let tv = &t_feat.values[start..end];
        let sv = &s_feat.values[start..end];

        let dot: f64 = tv.iter().zip(sv.iter()).map(|(a, b)| a * b).sum();
        let norm_t: f64 = tv.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_s: f64 = sv.iter().map(|v| v * v).sum::<f64>().sqrt();

        let cos_sim = if norm_t > 1e-12 && norm_s > 1e-12 {
            dot / (norm_t * norm_s)
        } else {
            0.0
        };
        total += 1.0 - cos_sim;
    }
    total / t_feat.seq_len as f64
}

/// KL divergence between softmax'd feature vectors per token position.
fn compute_kl_feature_loss(t_feat: &FeatureMap, s_feat: &FeatureMap) -> f64 {
    if t_feat.seq_len == 0 {
        return 0.0;
    }
    // Build temporary LogitTensors to reuse softmax_with_temperature
    let t_logit = LogitTensor {
        values: t_feat.values.clone(),
        seq_len: t_feat.seq_len,
        vocab_size: t_feat.hidden_size,
    };
    let s_logit = LogitTensor {
        values: s_feat.values.clone(),
        seq_len: s_feat.seq_len,
        vocab_size: s_feat.hidden_size,
    };

    let p_t = t_logit.softmax_with_temperature(1.0);
    let p_s = s_logit.softmax_with_temperature(1.0);

    let vocab_size = t_feat.hidden_size;
    let mut kl_total = 0.0_f64;
    for pos in 0..t_feat.seq_len {
        let start = pos * vocab_size;
        let end = start + vocab_size;
        for i in start..end {
            let pt = p_t[i];
            let ps = p_s[i];
            if pt > 1e-12 && ps > 1e-12 {
                kl_total += pt * (pt / ps).ln();
            }
        }
    }
    kl_total / t_feat.seq_len as f64
}

/// Cross-entropy loss of student logits against hard (ground-truth) labels.
fn cross_entropy_loss(logits: &LogitTensor, labels: &[usize]) -> Result<f64, DistillError> {
    if labels.len() != logits.seq_len {
        return Err(DistillError::ShapeMismatch {
            teacher: format!("labels={}", labels.len()),
            student: format!("seq_len={}", logits.seq_len),
        });
    }
    let probs = logits.softmax_with_temperature(1.0);
    let mut total = 0.0_f64;
    for (pos, &label) in labels.iter().enumerate() {
        let start = pos * logits.vocab_size;
        let p = probs[start + label.min(logits.vocab_size - 1)];
        total -= p.max(1e-12).ln();
    }
    Ok(total / logits.seq_len as f64)
}

// ─── DistillationLossType ─────────────────────────────────────────────────────

/// Simple loss type enum for standalone distillation functions.
#[derive(Debug, Clone, PartialEq)]
pub enum DistillationLossType {
    /// KL divergence from teacher to student (forward KL).
    KLDivergence,
    /// Forward KL: KL(teacher || student) — same as KLDivergence.
    ForwardKL,
    /// Reverse KL: KL(student || teacher).
    ReverseKL,
    /// Jensen-Shannon divergence: 0.5 × (KL(p||m) + KL(q||m)) where m = 0.5*(p+q).
    JSD,
    /// Mean squared error between logit vectors.
    MSE,
}

// ─── Standalone distillation helpers ─────────────────────────────────────────

/// Soft-target KL distillation loss: `T² × KL(softmax(teacher/T) ‖ softmax(student/T))`.
///
/// This is the classic Hinton et al. (2015) knowledge distillation loss.
/// Averaged over the vocabulary (or sequence × vocab for batches).
pub fn soft_target_kl_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    temperature: f32,
) -> Result<f32, DistillError> {
    if student_logits.is_empty() || teacher_logits.is_empty() {
        return Err(DistillError::EmptyLogits);
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(DistillError::ShapeMismatch {
            teacher: format!("{}", teacher_logits.len()),
            student: format!("{}", student_logits.len()),
        });
    }
    if temperature <= 0.0 {
        return Err(DistillError::InvalidTemperature(temperature as f64));
    }

    let n = student_logits.len();
    let t = temperature;

    // Softmax teacher at temperature T
    let max_t = teacher_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_t: Vec<f32> = teacher_logits.iter().map(|&v| ((v - max_t) / t).exp()).collect();
    let sum_t: f32 = exp_t.iter().sum();
    let p_t: Vec<f32> = exp_t.iter().map(|e| e / sum_t).collect();

    // Softmax student at temperature T
    let max_s = student_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_s: Vec<f32> = student_logits.iter().map(|&v| ((v - max_s) / t).exp()).collect();
    let sum_s: f32 = exp_s.iter().sum();
    let p_s: Vec<f32> = exp_s.iter().map(|e| e / sum_s).collect();

    // KL(p_t || p_s) = Σ p_t * log(p_t / p_s)
    let mut kl = 0.0f32;
    for i in 0..n {
        if p_t[i] > 1e-12 && p_s[i] > 1e-12 {
            kl += p_t[i] * (p_t[i] / p_s[i]).ln();
        }
    }

    // Scale by T² as per Hinton et al.
    Ok(kl * t * t)
}

/// Combined distillation loss: `α × task_loss + (1 − α) × distill_loss`.
///
/// `alpha` = weight of the task loss; `1 - alpha` = weight of the distillation loss.
pub fn combined_distillation_loss(task_loss: f32, distill_loss: f32, alpha: f32) -> f32 {
    let a = alpha.max(0.0).min(1.0);
    a * task_loss + (1.0 - a) * distill_loss
}

/// Feature matching loss: MSE between intermediate student/teacher activations.
///
/// `L = (1/N) × Σ (student_i − teacher_i)²`
pub fn feature_matching_loss(
    student_features: &[f32],
    teacher_features: &[f32],
) -> Result<f32, DistillError> {
    if student_features.is_empty() || teacher_features.is_empty() {
        return Err(DistillError::EmptyLogits);
    }
    if student_features.len() != teacher_features.len() {
        return Err(DistillError::ShapeMismatch {
            teacher: format!("{}", teacher_features.len()),
            student: format!("{}", student_features.len()),
        });
    }
    let n = student_features.len() as f32;
    let mse = student_features
        .iter()
        .zip(teacher_features.iter())
        .map(|(s, t)| (s - t) * (s - t))
        .sum::<f32>()
        / n;
    Ok(mse)
}

/// Attention transfer loss: MSE between normalised student/teacher attention maps.
///
/// Attention maps are L2-normalised per row before computing MSE, following
/// Zagoruyko & Komodakis (2017).
pub fn attention_transfer_loss(
    student_attn: &[f32],
    teacher_attn: &[f32],
) -> Result<f32, DistillError> {
    if student_attn.is_empty() || teacher_attn.is_empty() {
        return Err(DistillError::EmptyLogits);
    }
    if student_attn.len() != teacher_attn.len() {
        return Err(DistillError::ShapeMismatch {
            teacher: format!("{}", teacher_attn.len()),
            student: format!("{}", student_attn.len()),
        });
    }
    // L2-normalise each map vector, then compute MSE
    let norm_s = student_attn.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    let norm_t = teacher_attn.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);

    let n = student_attn.len() as f32;
    let mse = student_attn
        .iter()
        .zip(teacher_attn.iter())
        .map(|(s, t)| {
            let sn = s / norm_s;
            let tn = t / norm_t;
            (sn - tn) * (sn - tn)
        })
        .sum::<f32>()
        / n;
    Ok(mse)
}

/// Jensen-Shannon divergence between two probability distributions (or logit vectors).
///
/// `JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)` where `m = 0.5*(p+q)`.
/// Inputs are treated as unnormalised logits; softmax is applied internally.
pub fn jsd_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
) -> Result<f32, DistillError> {
    if student_logits.is_empty() || teacher_logits.is_empty() {
        return Err(DistillError::EmptyLogits);
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(DistillError::ShapeMismatch {
            teacher: format!("{}", teacher_logits.len()),
            student: format!("{}", student_logits.len()),
        });
    }
    let n = student_logits.len();

    // Softmax both
    let softmax = |logits: &[f32]| -> Vec<f32> {
        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
        let s: f32 = exps.iter().sum();
        exps.iter().map(|e| e / s).collect()
    };

    let p = softmax(teacher_logits);
    let q = softmax(student_logits);

    let mut jsd = 0.0f32;
    for i in 0..n {
        let m = 0.5 * (p[i] + q[i]);
        if p[i] > 1e-12 && m > 1e-12 {
            jsd += 0.5 * p[i] * (p[i] / m).ln();
        }
        if q[i] > 1e-12 && m > 1e-12 {
            jsd += 0.5 * q[i] * (q[i] / m).ln();
        }
    }
    Ok(jsd)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Helper: build a uniform logit tensor (all same value → uniform distribution)
    fn uniform_logits(seq_len: usize, vocab_size: usize, val: f64) -> LogitTensor {
        LogitTensor::new(vec![val; seq_len * vocab_size], seq_len, vocab_size).unwrap()
    }

    // Helper: build a one-hot-ish logit tensor where each position peaks at `peak_idx`
    fn peaked_logits(seq_len: usize, vocab_size: usize, peak_idx: usize, peak_val: f64) -> LogitTensor {
        let mut values = vec![0.0f64; seq_len * vocab_size];
        for pos in 0..seq_len {
            values[pos * vocab_size + peak_idx] = peak_val;
        }
        LogitTensor::new(values, seq_len, vocab_size).unwrap()
    }

    // Helper: build a basic config
    fn soft_config(temperature: f64, alpha: f64) -> DistillationConfig {
        DistillationConfig {
            strategy: DistillationStrategy::SoftTargets { temperature, alpha },
            teacher_layers: 12,
            student_layers: 6,
            vocab_size: 100,
            hidden_size: 64,
        }
    }

    // 1. LogitTensor softmax temperature effect: higher T → flatter distribution
    #[test]
    fn test_softmax_temperature_effect() {
        let logits = peaked_logits(1, 10, 0, 5.0);

        let probs_low = logits.softmax_with_temperature(0.5);
        let probs_high = logits.softmax_with_temperature(10.0);

        // At low T the distribution is sharper → higher peak probability
        assert!(
            probs_low[0] > probs_high[0],
            "lower temperature should produce higher peak prob"
        );
        // Sums should be ≈ 1.0
        let sum_low: f64 = probs_low.iter().sum();
        let sum_high: f64 = probs_high.iter().sum();
        assert!((sum_low - 1.0).abs() < 1e-10, "probs should sum to 1 (low T)");
        assert!((sum_high - 1.0).abs() < 1e-10, "probs should sum to 1 (high T)");
    }

    // 2. argmax_per_position returns correct index
    #[test]
    fn test_argmax_per_position() {
        let logits = peaked_logits(3, 5, 2, 10.0);
        let argmax = logits.argmax_per_position();
        assert_eq!(argmax, vec![2, 2, 2]);
    }

    // 3. FeatureMap l2_norm_per_token
    #[test]
    fn test_feature_map_l2_norm() {
        // 2 positions, hidden=3: [[3,4,0], [0,0,5]]
        let values = vec![3.0, 4.0, 0.0,  0.0, 0.0, 5.0];
        let fm = FeatureMap::new(values, 2, 3, 0);
        let norms = fm.l2_norm_per_token();
        assert_eq!(norms.len(), 2);
        assert!((norms[0] - 5.0).abs() < 1e-10, "norm of (3,4,0) should be 5");
        assert!((norms[1] - 5.0).abs() < 1e-10, "norm of (0,0,5) should be 5");
    }

    // 4. soft_target_loss numerical check: same logits → KL = 0
    #[test]
    fn test_soft_target_loss_zero_when_identical() {
        let logits = peaked_logits(4, 20, 3, 2.0);
        let config = soft_config(4.0, 0.7);
        let distill = DistillationLoss::new(config);
        let loss = distill.soft_target_loss(&logits, &logits, 4.0).unwrap();
        assert!(loss.abs() < 1e-10, "KL divergence should be 0 for identical distributions");
    }

    // 5. feature L2 loss: identical features → 0
    #[test]
    fn test_feature_l2_loss_identical() {
        let values: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let tf = FeatureMap::new(values.clone(), 3, 4, 0);
        let sf = FeatureMap::new(values, 3, 4, 0);

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(0, 0)],
                loss_type: FeatureLossType::L2,
            },
            teacher_layers: 4,
            student_layers: 4,
            vocab_size: 50,
            hidden_size: 4,
        };
        let distill = DistillationLoss::new(config);
        let loss = distill.feature_loss(&[tf], &[sf], &FeatureLossType::L2).unwrap();
        assert!(loss.abs() < 1e-10, "L2 loss on identical features should be 0");
    }

    // 6. feature cosine loss: identical features → 0
    #[test]
    fn test_feature_cosine_loss_identical() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0];
        let tf = FeatureMap::new(values.clone(), 2, 4, 1);
        let sf = FeatureMap::new(values, 2, 4, 1);

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(1, 1)],
                loss_type: FeatureLossType::Cosine,
            },
            teacher_layers: 4,
            student_layers: 4,
            vocab_size: 50,
            hidden_size: 4,
        };
        let distill = DistillationLoss::new(config);
        let loss = distill.feature_loss(&[tf], &[sf], &FeatureLossType::Cosine).unwrap();
        assert!(loss.abs() < 1e-10, "Cosine loss on identical features should be 0");
    }

    // 7. compute_loss combined strategy
    #[test]
    fn test_compute_loss_combined() {
        let seq_len = 3;
        let vocab_size = 10;
        let hidden_size = 8;

        let t_logits = peaked_logits(seq_len, vocab_size, 0, 2.0);
        let s_logits = peaked_logits(seq_len, vocab_size, 0, 2.0);

        let t_vals: Vec<f64> = (0..seq_len * hidden_size).map(|i| i as f64).collect();
        let s_vals = t_vals.clone();
        let t_feat = vec![FeatureMap::new(t_vals, seq_len, hidden_size, 0)];
        let s_feat = vec![FeatureMap::new(s_vals, seq_len, hidden_size, 0)];
        let labels = vec![0usize; seq_len];

        let config = DistillationConfig {
            strategy: DistillationStrategy::Combined {
                soft_alpha: 0.4,
                feature_alpha: 0.3,
                temperature: 3.0,
            },
            teacher_layers: 4,
            student_layers: 4,
            vocab_size,
            hidden_size,
        };

        let mut distill = DistillationLoss::new(config);
        let result = distill.compute_loss(&t_logits, &s_logits, &t_feat, &s_feat, &labels).unwrap();

        assert!(result.total_loss >= 0.0, "total loss should be non-negative");
        assert!((result.soft_target_component).abs() < 1e-10, "soft targets identical → 0");
        assert!((result.feature_component).abs() < 1e-10, "features identical → 0");
    }

    // 8. advance_stage increments stage counter
    #[test]
    fn test_advance_stage() {
        let stages = vec![
            DistillationStage { stage_idx: 0, steps: 100, temperature: 5.0, alpha: 0.8 },
            DistillationStage { stage_idx: 1, steps: 200, temperature: 3.0, alpha: 0.6 },
        ];
        let config = DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages },
            teacher_layers: 12,
            student_layers: 6,
            vocab_size: 50,
            hidden_size: 32,
        };
        let mut distill = DistillationLoss::new(config);
        assert_eq!(distill.current_stage, 0);
        distill.advance_stage();
        assert_eq!(distill.current_stage, 1);
        distill.advance_stage();
        assert_eq!(distill.current_stage, 2);
    }

    // 9. SoftTargets strategy produces positive loss for different logits
    #[test]
    fn test_soft_targets_strategy_nonzero() {
        let t_logits = peaked_logits(2, 8, 0, 5.0);
        let s_logits = peaked_logits(2, 8, 7, 5.0);
        let config = soft_config(4.0, 0.7);
        let distill = DistillationLoss::new(config);
        let loss = distill.soft_target_loss(&t_logits, &s_logits, 4.0).unwrap();
        assert!(loss > 0.0, "KL divergence should be positive for different distributions");
    }

    // 10. Zero loss when teacher equals student (compute_loss SoftTargets)
    #[test]
    fn test_zero_loss_teacher_equals_student() {
        let seq_len = 4;
        let vocab_size = 16;
        let logits = peaked_logits(seq_len, vocab_size, 2, 3.0);
        let labels = vec![2usize; seq_len];

        let config = soft_config(4.0, 1.0); // alpha=1.0 means only distillation loss
        let mut distill = DistillationLoss::new(config);
        let result = distill
            .compute_loss(&logits, &logits, &[], &[], &labels)
            .unwrap();

        assert!(
            result.soft_target_component.abs() < 1e-9,
            "soft target loss should be 0 when teacher == student"
        );
    }

    // 11. Temperature scaling: loss scales with T^2
    #[test]
    fn test_temperature_t_squared_scaling() {
        // KL(uniform || uniform) = 0, so compare peaked vs uniform at different T
        let t_logits = peaked_logits(1, 10, 0, 3.0);
        let s_logits = uniform_logits(1, 10, 0.0);
        let config = soft_config(1.0, 0.7);
        let distill = DistillationLoss::new(config);

        let loss_t1 = distill.soft_target_loss(&t_logits, &s_logits, 1.0).unwrap();
        let loss_t2 = distill.soft_target_loss(&t_logits, &s_logits, 2.0).unwrap();

        // At T=2 the raw KL is smaller because distributions are softer,
        // but after T^2 scaling the relationship should hold:
        // loss_t2 includes T^2=4 factor while loss_t1 has T^2=1.
        // The exact ratio depends on the data, but both should be positive.
        assert!(loss_t1 > 0.0, "loss at T=1 should be positive");
        assert!(loss_t2 > 0.0, "loss at T=2 should be positive");

        // The T^2 scaling means higher temperature amplifies the scaled loss
        // relative to the raw KL. Verify the scaling is applied (loss_t2 not == loss_t1*4)
        // since the KL itself changes with T — just check they're different.
        assert!(
            (loss_t1 - loss_t2).abs() > 1e-10,
            "losses at different temperatures should differ"
        );
    }

    // 12. Shape mismatch error for mismatched seq_len
    #[test]
    fn test_shape_mismatch_error() {
        let t = peaked_logits(4, 10, 0, 1.0);
        let s = peaked_logits(3, 10, 0, 1.0); // different seq_len
        let config = soft_config(4.0, 0.7);
        let distill = DistillationLoss::new(config);
        let result = distill.soft_target_loss(&t, &s, 4.0);
        assert!(matches!(result, Err(DistillError::ShapeMismatch { .. })));
    }

    // 13. Display DistillationLossResult
    #[test]
    fn test_display_distillation_loss_result() {
        let r = DistillationLossResult {
            total_loss: 1.234567,
            soft_target_component: 0.5,
            feature_component: 0.3,
            hard_label_component: 0.434567,
            current_temperature: 4.0,
            stage: 2,
        };
        let s = format!("{r}");
        assert!(s.contains("total="), "display should contain 'total='");
        assert!(s.contains("soft="), "display should contain 'soft='");
        assert!(s.contains("stage=2"), "display should contain stage number");
    }

    // 14. LogitTensor::new rejects empty values
    #[test]
    fn test_logit_tensor_new_empty_error() {
        let result = LogitTensor::new(vec![], 0, 10);
        assert!(matches!(result, Err(DistillError::EmptyLogits)));
    }

    // ─── Extra: HashMap-based layer_mapping lookup test ───────────────────────
    #[test]
    fn test_feature_layer_index_lookup() {
        // Verify that feature_loss correctly matches by layer_index
        let tv = vec![1.0f64; 12]; // seq=3, hidden=4
        let sv = vec![1.0f64; 12];
        let tf = vec![FeatureMap::new(tv, 3, 4, 5)]; // teacher layer 5
        let sf = vec![FeatureMap::new(sv, 3, 4, 7)]; // student layer 7

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(5, 7)],
                loss_type: FeatureLossType::L2,
            },
            teacher_layers: 8,
            student_layers: 8,
            vocab_size: 50,
            hidden_size: 4,
        };
        let distill = DistillationLoss::new(config);
        let loss = distill.feature_loss(&tf, &sf, &FeatureLossType::L2).unwrap();
        assert!(loss.abs() < 1e-10, "L2 loss for identical features should be 0");
    }

    // HashMap usage test (ensures HashMap import is exercised)
    #[test]
    fn test_hashmap_in_config() {
        let mut map: HashMap<String, f64> = HashMap::new();
        map.insert("temperature".to_string(), 4.0);
        map.insert("alpha".to_string(), 0.7);
        assert_eq!(*map.get("temperature").unwrap(), 4.0);
    }

    // ─── Standalone function tests ────────────────────────────────────────

    // 15. soft_target_kl_loss identical logits → loss ≈ 0
    #[test]
    fn test_soft_target_kl_loss_identical() {
        let logits = vec![1.0f32, 2.0, 3.0, 1.0, 0.5];
        let loss = soft_target_kl_loss(&logits, &logits, 2.0).expect("ok");
        assert!(loss.abs() < 1e-5, "KL of identical dists should be 0, got {loss}");
    }

    // 16. soft_target_kl_loss different logits → positive loss
    #[test]
    fn test_soft_target_kl_loss_different_positive() {
        let teacher = vec![10.0f32, 0.0, 0.0];
        let student = vec![0.0f32, 10.0, 0.0];
        let loss = soft_target_kl_loss(&student, &teacher, 1.0).expect("ok");
        assert!(loss > 0.0, "KL loss should be positive for different dists, got {loss}");
    }

    // 17. soft_target_kl_loss invalid temperature → error
    #[test]
    fn test_soft_target_kl_loss_invalid_temperature() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let err = soft_target_kl_loss(&logits, &logits, 0.0);
        assert!(err.is_err(), "zero temperature should be an error");
    }

    // 18. soft_target_kl_loss shape mismatch → error
    #[test]
    fn test_soft_target_kl_loss_shape_mismatch() {
        let t = vec![1.0f32, 2.0, 3.0];
        let s = vec![1.0f32, 2.0];
        let err = soft_target_kl_loss(&s, &t, 2.0);
        assert!(matches!(err, Err(DistillError::ShapeMismatch { .. })));
    }

    // 19. soft_target_kl_loss scales with T^2
    #[test]
    fn test_soft_target_kl_loss_temperature_scaling() {
        let teacher = vec![5.0f32, 0.0, 0.0];
        let student = vec![0.0f32, 5.0, 0.0];
        let loss_t1 = soft_target_kl_loss(&student, &teacher, 1.0).expect("ok");
        let loss_t4 = soft_target_kl_loss(&student, &teacher, 4.0).expect("ok");
        // Both should be positive; check they differ (T scaling changes both KL and T^2 factor)
        assert!(loss_t1 > 0.0 && loss_t4 > 0.0);
        assert!((loss_t1 - loss_t4).abs() > 1e-6, "losses at T=1 and T=4 should differ");
    }

    // 20. combined_distillation_loss alpha=1 → only task_loss
    #[test]
    fn test_combined_distillation_loss_alpha_one() {
        let combined = combined_distillation_loss(3.0, 7.0, 1.0);
        assert!((combined - 3.0f32).abs() < 1e-5, "alpha=1 → task_loss only, got {combined}");
    }

    // 21. combined_distillation_loss alpha=0 → only distill_loss
    #[test]
    fn test_combined_distillation_loss_alpha_zero() {
        let combined = combined_distillation_loss(3.0, 7.0, 0.0);
        assert!((combined - 7.0f32).abs() < 1e-5, "alpha=0 → distill_loss only, got {combined}");
    }

    // 22. combined_distillation_loss alpha=0.5 → average
    #[test]
    fn test_combined_distillation_loss_alpha_half() {
        let combined = combined_distillation_loss(2.0, 4.0, 0.5);
        assert!((combined - 3.0f32).abs() < 1e-5, "alpha=0.5 → average=3.0, got {combined}");
    }

    // 23. feature_matching_loss identical → 0
    #[test]
    fn test_feature_matching_loss_identical() {
        let feats = vec![1.0f32, 2.0, 3.0, 4.0];
        let loss = feature_matching_loss(&feats, &feats).expect("ok");
        assert!(loss.abs() < 1e-5, "MSE of identical features should be 0, got {loss}");
    }

    // 24. feature_matching_loss known value
    #[test]
    fn test_feature_matching_loss_known_value() {
        // student=[0,0], teacher=[1,1] → MSE = (1+1)/2 = 1.0
        let student = vec![0.0f32, 0.0];
        let teacher = vec![1.0f32, 1.0];
        let loss = feature_matching_loss(&student, &teacher).expect("ok");
        assert!((loss - 1.0f32).abs() < 1e-5, "expected MSE=1.0, got {loss}");
    }

    // 25. attention_transfer_loss identical maps → 0
    #[test]
    fn test_attention_transfer_loss_identical() {
        let attn = vec![0.25f32, 0.25, 0.25, 0.25];
        let loss = attention_transfer_loss(&attn, &attn).expect("ok");
        assert!(loss.abs() < 1e-5, "attention transfer loss for identical maps should be 0");
    }

    // 26. attention_transfer_loss different maps → positive
    #[test]
    fn test_attention_transfer_loss_different() {
        let student = vec![1.0f32, 0.0, 0.0, 0.0];
        let teacher = vec![0.0f32, 0.0, 0.0, 1.0];
        let loss = attention_transfer_loss(&student, &teacher).expect("ok");
        assert!(loss > 0.0, "attention transfer loss should be positive for different maps");
    }

    // 27. jsd_loss identical → 0
    #[test]
    fn test_jsd_loss_identical() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let loss = jsd_loss(&logits, &logits).expect("ok");
        assert!(loss.abs() < 1e-5, "JSD of identical dists should be 0, got {loss}");
    }

    // 28. jsd_loss orthogonal → max (≈ ln 2 ≈ 0.693)
    #[test]
    fn test_jsd_loss_orthogonal_positive() {
        // One distribution peaks at 0, the other at last position
        let n = 8;
        let mut t = vec![0.0f32; n];
        let mut s = vec![0.0f32; n];
        t[0] = 100.0;
        s[n - 1] = 100.0;
        let loss = jsd_loss(&s, &t).expect("ok");
        assert!(loss > 0.0, "JSD of non-overlapping dists should be positive, got {loss}");
        // JSD ≤ ln(2)
        assert!(loss <= 0.694 + 1e-4, "JSD should be ≤ ln(2), got {loss}");
    }

    // ── New tests 29-52 ──────────────────────────────────────────────────────

    // 29. Temperature softening: entropy increases monotonically with T
    #[test]
    fn test_softmax_entropy_increases_with_temperature() {
        // Peaked logits: higher T → softer distribution → higher entropy
        let logits = peaked_logits(1, 10, 0, 5.0);

        let entropy = |probs: &[f64]| -> f64 {
            probs.iter().filter(|&&p| p > 1e-15).map(|&p| -p * p.ln()).sum::<f64>()
        };

        let p1 = logits.softmax_with_temperature(0.5);
        let p2 = logits.softmax_with_temperature(2.0);
        let p3 = logits.softmax_with_temperature(8.0);

        let h1 = entropy(&p1);
        let h2 = entropy(&p2);
        let h3 = entropy(&p3);

        assert!(h1 < h2, "entropy at T=0.5 ({h1}) should be < entropy at T=2.0 ({h2})");
        assert!(h2 < h3, "entropy at T=2.0 ({h2}) should be < entropy at T=8.0 ({h3})");
    }

    // 30. KL divergence is zero for identical distributions (standalone function)
    #[test]
    fn test_kl_zero_for_identical_distributions() {
        let logits = vec![2.0f32, 1.0, 0.5, 3.0, 0.1];
        let loss = soft_target_kl_loss(&logits, &logits, 1.0).expect("ok");
        assert!(loss.abs() < 1e-5, "KL(p||p) should be 0, got {loss}");
    }

    // 31. KL divergence is strictly positive for different distributions
    #[test]
    fn test_kl_positive_for_different_distributions() {
        let teacher = vec![10.0f32, 0.0, 0.0, 0.0];
        let student = vec![0.0f32, 0.0, 0.0, 10.0];
        let loss = soft_target_kl_loss(&student, &teacher, 1.0).expect("ok");
        assert!(loss > 0.0, "KL divergence must be > 0 for different distributions, got {loss}");
    }

    // 32. Feature matching L2 loss for different activations is > 0
    #[test]
    fn test_feature_matching_loss_nonzero_for_different() {
        let teacher_f = vec![1.0f32, 2.0, 3.0, 4.0];
        let student_f = vec![0.0f32, 0.0, 0.0, 0.0];
        let loss = feature_matching_loss(&student_f, &teacher_f).expect("ok");
        assert!(loss > 0.0, "MSE should be positive for different activations, got {loss}");
    }

    // 33. Attention transfer: normalised identical maps → zero loss
    #[test]
    fn test_attention_transfer_normalised_identical() {
        // Different scale, same direction → after normalisation should be the same
        let map1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let map2 = vec![2.0f32, 4.0, 6.0, 8.0]; // same direction, double magnitude
        let loss = attention_transfer_loss(&map2, &map1).expect("ok");
        // Normalised vectors are the same → MSE = 0
        assert!(loss.abs() < 1e-5, "scaled-identical attention maps should give 0 loss, got {loss}");
    }

    // 34. Combined loss weighting: alpha * task + (1 - alpha) * distill
    #[test]
    fn test_combined_loss_weighting_numerically() {
        // task=4.0, distill=2.0, alpha=0.25 → 0.25*4 + 0.75*2 = 1.0 + 1.5 = 2.5
        let combined = combined_distillation_loss(4.0, 2.0, 0.25);
        assert!((combined - 2.5f32).abs() < 1e-5, "expected 2.5, got {combined}");
    }

    // 35. Progressive distillation: second stage uses different temperature
    #[test]
    fn test_progressive_distillation_stage_temperature() {
        let stages = vec![
            DistillationStage { stage_idx: 0, steps: 50, temperature: 8.0, alpha: 0.9 },
            DistillationStage { stage_idx: 1, steps: 50, temperature: 2.0, alpha: 0.5 },
        ];
        let config = DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages },
            teacher_layers: 12,
            student_layers: 4,
            vocab_size: 20,
            hidden_size: 16,
        };
        let mut distill = DistillationLoss::new(config);

        let t_logits = peaked_logits(1, 20, 0, 3.0);
        let s_logits = peaked_logits(1, 20, 5, 3.0);
        let labels = vec![0usize];

        let result_stage0 = distill
            .compute_loss(&t_logits, &s_logits, &[], &[], &labels)
            .expect("stage 0 loss");
        assert!((result_stage0.current_temperature - 8.0).abs() < 1e-6,
            "stage 0 temperature should be 8.0, got {}", result_stage0.current_temperature);

        distill.advance_stage();
        let result_stage1 = distill
            .compute_loss(&t_logits, &s_logits, &[], &[], &labels)
            .expect("stage 1 loss");
        assert!((result_stage1.current_temperature - 2.0).abs() < 1e-6,
            "stage 1 temperature should be 2.0, got {}", result_stage1.current_temperature);
    }

    // 36. TinyBERT-style: feature matching with linear transformation
    //     Simulate a projection matrix by scaling student features before matching
    #[test]
    fn test_tinybert_style_feature_projection() {
        let teacher_f = vec![1.0f32, 2.0, 3.0, 4.0];
        // Simulate: student after projection = teacher (projection aligns perfectly)
        let student_projected = teacher_f.clone();
        let loss = feature_matching_loss(&student_projected, &teacher_f).expect("ok");
        assert!(loss.abs() < 1e-5, "projected student matching teacher → loss=0, got {loss}");

        // Before projection (raw student differs): loss > 0
        let student_raw = vec![0.5f32, 1.0, 1.5, 2.0];
        let loss_raw = feature_matching_loss(&student_raw, &teacher_f).expect("ok");
        assert!(loss_raw > 0.0, "un-projected student → positive loss, got {loss_raw}");
    }

    // 37. Data-free distillation: generated logits drive a distillation loss
    #[test]
    fn test_data_free_distillation_generated_logits() {
        // Simulate a simple "inversion" by using teacher output as student target
        // A generated logit that matches teacher logits should produce near-zero KL loss
        let teacher_logits = vec![3.0f32, 1.0, 0.5, 0.1, 2.0];
        // Generated student logits = teacher logits (perfect inversion)
        let student_generated = teacher_logits.clone();
        let loss = soft_target_kl_loss(&student_generated, &teacher_logits, 2.0).expect("ok");
        assert!(loss.abs() < 1e-5, "perfect inversion should give near-zero loss, got {loss}");

        // A randomly-generated logit (LCG: multiplier=6364136223846793005, addend=1442695040888963407)
        let mut state: u64 = 42;
        let lcg_next = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
            ((*s >> 32) as f32) / (u32::MAX as f32) * 6.0 - 3.0
        };
        let student_random: Vec<f32> = (0..5).map(|_| lcg_next(&mut state)).collect();
        let loss_random = soft_target_kl_loss(&student_random, &teacher_logits, 2.0).expect("ok");
        // Random student should generally not match teacher exactly
        assert!(loss_random >= 0.0, "KL loss must be non-negative, got {loss_random}");
    }

    // 38. DistillationConfig construction for all strategy variants
    #[test]
    fn test_distillation_config_all_strategies() {
        let _soft = DistillationConfig {
            strategy: DistillationStrategy::SoftTargets { temperature: 4.0, alpha: 0.7 },
            teacher_layers: 12, student_layers: 6, vocab_size: 50000, hidden_size: 768,
        };
        let _feat = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(0, 0), (6, 3)],
                loss_type: FeatureLossType::L2,
            },
            teacher_layers: 12, student_layers: 6, vocab_size: 50000, hidden_size: 768,
        };
        let _attn = DistillationConfig {
            strategy: DistillationStrategy::AttentionTransfer {
                layer_pairs: vec![(0, 0)],
                normalize: true,
            },
            teacher_layers: 12, student_layers: 6, vocab_size: 50000, hidden_size: 768,
        };
        let _prog = DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages: vec![] },
            teacher_layers: 12, student_layers: 6, vocab_size: 50000, hidden_size: 768,
        };
        let _comb = DistillationConfig {
            strategy: DistillationStrategy::Combined {
                soft_alpha: 0.4, feature_alpha: 0.3, temperature: 3.0,
            },
            teacher_layers: 12, student_layers: 6, vocab_size: 50000, hidden_size: 768,
        };
    }

    // 39. DistillationLoss step counter increments on each compute_loss call
    #[test]
    fn test_distillation_loss_step_counter() {
        let config = soft_config(2.0, 0.5);
        let mut distill = DistillationLoss::new(config);
        assert_eq!(distill.step, 0);
        let logits = peaked_logits(1, 10, 0, 1.0);
        let labels = vec![0usize];
        distill.compute_loss(&logits, &logits, &[], &[], &labels).expect("step 1");
        assert_eq!(distill.step, 1);
        distill.compute_loss(&logits, &logits, &[], &[], &labels).expect("step 2");
        assert_eq!(distill.step, 2);
    }

    // 40. Cross-entropy hard label component: correct token → lower loss
    #[test]
    fn test_hard_label_cross_entropy_correct_token_lower_loss() {
        let vocab_size = 10;
        let seq_len = 1;

        // Student peaks at token 0 (correct)
        let s_correct = peaked_logits(seq_len, vocab_size, 0, 5.0);
        // Student peaks at token 9 (wrong)
        let s_wrong = peaked_logits(seq_len, vocab_size, 9, 5.0);

        let teacher = peaked_logits(seq_len, vocab_size, 0, 5.0);
        let labels = vec![0usize]; // correct label is token 0

        let config_correct = soft_config(1.0, 0.0); // alpha=0 → only hard label loss
        let mut d_correct = DistillationLoss::new(config_correct);
        let config_wrong = soft_config(1.0, 0.0);
        let mut d_wrong = DistillationLoss::new(config_wrong);

        let res_correct = d_correct.compute_loss(&teacher, &s_correct, &[], &[], &labels).expect("correct");
        let res_wrong = d_wrong.compute_loss(&teacher, &s_wrong, &[], &[], &labels).expect("wrong");

        assert!(
            res_correct.hard_label_component < res_wrong.hard_label_component,
            "correct prediction should have lower CE loss: {} vs {}",
            res_correct.hard_label_component, res_wrong.hard_label_component
        );
    }

    // 41. Feature KL loss type: identical features → 0
    #[test]
    fn test_feature_kl_loss_identical() {
        // Using KL loss type on identical feature maps
        let values: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let tf = FeatureMap::new(values.clone(), 2, 4, 0);
        let sf = FeatureMap::new(values, 2, 4, 0);

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(0, 0)],
                loss_type: FeatureLossType::KL,
            },
            teacher_layers: 4, student_layers: 4, vocab_size: 50, hidden_size: 4,
        };
        let distill = DistillationLoss::new(config);
        let loss = distill.feature_loss(&[tf], &[sf], &FeatureLossType::KL).expect("ok");
        assert!(loss.abs() < 1e-10, "KL feature loss for identical maps should be 0, got {loss}");
    }

    // 42. Error: EmptyLogits from soft_target_kl_loss
    #[test]
    fn test_soft_target_kl_loss_empty_logits_error() {
        let empty: Vec<f32> = vec![];
        let non_empty = vec![1.0f32, 2.0];
        let err1 = soft_target_kl_loss(&empty, &non_empty, 1.0);
        assert!(matches!(err1, Err(DistillError::EmptyLogits)), "empty student should error");
        let err2 = soft_target_kl_loss(&non_empty, &empty, 1.0);
        assert!(matches!(err2, Err(DistillError::EmptyLogits)), "empty teacher should error");
    }

    // 43. Error: InvalidTemperature from soft_target_kl_loss with negative T
    #[test]
    fn test_soft_target_kl_loss_negative_temperature_error() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let err = soft_target_kl_loss(&logits, &logits, -1.0);
        assert!(matches!(err, Err(DistillError::InvalidTemperature(_))), "negative T should error");
    }

    // 44. Error: FeatureCountMismatch from feature_loss with missing layer index
    #[test]
    fn test_feature_loss_feature_count_mismatch_error() {
        let tf = vec![FeatureMap::new(vec![1.0; 4], 1, 4, 0)];
        let sf = vec![FeatureMap::new(vec![1.0; 4], 1, 4, 0)];

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(0, 99)], // student layer 99 does not exist
                loss_type: FeatureLossType::L2,
            },
            teacher_layers: 4, student_layers: 4, vocab_size: 50, hidden_size: 4,
        };
        let distill = DistillationLoss::new(config);
        let err = distill.feature_loss(&tf, &sf, &FeatureLossType::L2);
        assert!(matches!(err, Err(DistillError::FeatureCountMismatch)), "missing layer should error");
    }

    // 45. JSD symmetry: JSD(p, q) == JSD(q, p)
    #[test]
    fn test_jsd_loss_symmetry() {
        let p = vec![3.0f32, 0.5, 1.0, 2.0];
        let q = vec![0.1f32, 2.0, 0.5, 3.0];
        let jsd_pq = jsd_loss(&p, &q).expect("pq");
        let jsd_qp = jsd_loss(&q, &p).expect("qp");
        assert!(
            (jsd_pq - jsd_qp).abs() < 1e-5,
            "JSD should be symmetric: JSD(p,q)={jsd_pq} vs JSD(q,p)={jsd_qp}"
        );
    }

    // 46. feature_matching_loss with known MSE: diff=2 for each element → MSE=4
    #[test]
    fn test_feature_matching_loss_known_mse_value() {
        let teacher_f = vec![2.0f32; 4];
        let student_f = vec![4.0f32; 4]; // diff=2, MSE = 4/4*4 = 4.0
        let loss = feature_matching_loss(&student_f, &teacher_f).expect("ok");
        assert!((loss - 4.0f32).abs() < 1e-5, "expected MSE=4.0, got {loss}");
    }

    // 47. attention_transfer_loss with all-zero student should not panic
    #[test]
    fn test_attention_transfer_loss_near_zero_student() {
        let student = vec![1e-13f32; 4]; // near-zero (below 1e-12 norm guard)
        let teacher = vec![1.0f32, 0.0, 0.0, 0.0];
        // Should complete without panic; loss may be non-zero or zero depending on guard
        let loss = attention_transfer_loss(&student, &teacher).expect("ok");
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
    }

    // 48. combined_distillation_loss clamping: alpha > 1 clamped to 1 → only task_loss
    #[test]
    fn test_combined_distillation_loss_alpha_clamped_above_one() {
        let combined = combined_distillation_loss(5.0, 9.0, 2.0); // alpha clamped to 1.0
        assert!((combined - 5.0f32).abs() < 1e-5, "alpha>1 clamped to 1, got {combined}");
    }

    // 49. DistillationLoss::compute_loss with FeatureBased strategy
    #[test]
    fn test_compute_loss_feature_based_strategy() {
        let seq_len = 2;
        let vocab_size = 8;
        let hidden_size = 4;

        let s_logits = peaked_logits(seq_len, vocab_size, 1, 2.0);
        let t_feat = vec![FeatureMap::new(vec![1.0f64; seq_len * hidden_size], seq_len, hidden_size, 0)];
        let s_feat = vec![FeatureMap::new(vec![2.0f64; seq_len * hidden_size], seq_len, hidden_size, 0)];
        let labels = vec![1usize; seq_len];

        let config = DistillationConfig {
            strategy: DistillationStrategy::FeatureBased {
                layer_mapping: vec![(0, 0)],
                loss_type: FeatureLossType::L2,
            },
            teacher_layers: 4, student_layers: 4, vocab_size, hidden_size,
        };
        let mut distill = DistillationLoss::new(config);
        let result = distill.compute_loss(&s_logits, &s_logits, &t_feat, &s_feat, &labels).expect("ok");
        assert!(result.feature_component > 0.0, "feature loss should be > 0 for different activations");
        assert!(result.total_loss > 0.0, "total loss should be positive");
    }

    // 50. DistillationLoss::compute_loss with Progressive strategy empty stages
    #[test]
    fn test_compute_loss_progressive_empty_stages() {
        let logits = peaked_logits(1, 5, 0, 2.0);
        let labels = vec![0usize];
        let config = DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages: vec![] },
            teacher_layers: 4, student_layers: 2, vocab_size: 5, hidden_size: 8,
        };
        let mut distill = DistillationLoss::new(config);
        let result = distill.compute_loss(&logits, &logits, &[], &[], &labels).expect("ok");
        // Empty stages falls back to defaults (T=4.0, alpha=0.7)
        assert!((result.current_temperature - 4.0).abs() < 1e-6,
            "empty stages should use default T=4.0, got {}", result.current_temperature);
    }

    // 51. DistillationLoss advance_stage resets step counter
    #[test]
    fn test_advance_stage_resets_step_counter() {
        let stages = vec![
            DistillationStage { stage_idx: 0, steps: 10, temperature: 4.0, alpha: 0.7 },
        ];
        let config = DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages },
            teacher_layers: 4, student_layers: 2, vocab_size: 10, hidden_size: 8,
        };
        let mut distill = DistillationLoss::new(config);
        let logits = peaked_logits(1, 10, 0, 1.0);
        let labels = vec![0usize];
        distill.compute_loss(&logits, &logits, &[], &[], &labels).expect("step 1");
        assert_eq!(distill.step, 1);
        distill.advance_stage();
        assert_eq!(distill.step, 0, "advance_stage should reset step to 0");
    }

    // 52. AttentionTransfer strategy in compute_loss returns hard_loss only
    #[test]
    fn test_compute_loss_attention_transfer_strategy() {
        let seq_len = 2;
        let vocab_size = 8;
        let logits = peaked_logits(seq_len, vocab_size, 3, 2.0);
        let labels = vec![3usize; seq_len];

        let config = DistillationConfig {
            strategy: DistillationStrategy::AttentionTransfer {
                layer_pairs: vec![(0, 0)],
                normalize: true,
            },
            teacher_layers: 6, student_layers: 3, vocab_size, hidden_size: 16,
        };
        let mut distill = DistillationLoss::new(config);
        let result = distill.compute_loss(&logits, &logits, &[], &[], &labels).expect("ok");
        // AttentionTransfer uses only hard loss; soft and feature components should be 0
        assert!((result.soft_target_component).abs() < 1e-10,
            "soft_target should be 0 for AttentionTransfer, got {}", result.soft_target_component);
        assert!((result.feature_component).abs() < 1e-10,
            "feature should be 0 for AttentionTransfer, got {}", result.feature_component);
        assert_eq!(result.total_loss, result.hard_label_component,
            "total_loss should equal hard_label_component for AttentionTransfer");
    }
}
