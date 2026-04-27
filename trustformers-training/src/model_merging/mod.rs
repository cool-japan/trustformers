//! Model Merging
//!
//! Implements SLERP, TIES, DARE, and linear interpolation for combining
//! fine-tuned models at the weight level.

pub mod merge_algorithms;

use std::collections::HashMap;

// ─── MergeMethod ─────────────────────────────────────────────────────────────

/// Merging method selector.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeMethod {
    /// Spherical Linear Interpolation between two models.
    Slerp {
        /// Interpolation parameter: 0.0 = model_a, 1.0 = model_b.
        t: f64,
    },
    /// TIES-Merging: Trim, Elect Sign, Merge.
    Ties {
        /// Fraction of parameters to keep per layer (e.g. 0.5).
        density: f64,
        /// Scaling factor applied to the merged task vector.
        lambda: f64,
    },
    /// DARE: Drop And REscale.
    Dare {
        /// Fraction of delta weights to drop (e.g. 0.9).
        drop_rate: f64,
        /// Rescale survivors by `1 / (1 - drop_rate)`.
        rescale: bool,
    },
    /// Linear interpolation: simple weighted average of model weights.
    Linear {
        /// One weight per model; normalised to sum to 1.0 internally.
        weights: Vec<f64>,
    },
}

// ─── ModelWeights ─────────────────────────────────────────────────────────────

/// A model's weights as flat vectors per named layer.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub layers: HashMap<String, Vec<f64>>,
}

impl ModelWeights {
    pub fn new() -> Self {
        Self { layers: HashMap::new() }
    }

    pub fn add_layer(&mut self, name: impl Into<String>, weights: Vec<f64>) {
        self.layers.insert(name.into(), weights);
    }

    /// All layer names, sorted for determinism.
    pub fn layer_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.layers.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Total number of parameters across all layers.
    pub fn total_params(&self) -> usize {
        self.layers.values().map(|v| v.len()).sum()
    }

    /// Compute task vector: `self - base` for every layer shared with `base`.
    pub fn task_vector(&self, base: &ModelWeights) -> Result<TaskVector, MergeError> {
        let mut deltas = HashMap::new();
        for name in self.layer_names() {
            let self_w = &self.layers[name];
            let base_w = base.layers.get(name).ok_or_else(|| MergeError::LayerMismatch(name.to_string()))?;
            if self_w.len() != base_w.len() {
                return Err(MergeError::ShapeMismatch {
                    layer: name.to_string(),
                    a: self_w.len(),
                    b: base_w.len(),
                });
            }
            let delta: Vec<f64> = self_w.iter().zip(base_w.iter()).map(|(a, b)| a - b).collect();
            deltas.insert(name.to_string(), delta);
        }
        Ok(TaskVector { deltas })
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

// ─── TaskVector ──────────────────────────────────────────────────────────────

/// Task vector: the delta weights `fine_tuned - base`.
#[derive(Debug, Clone)]
pub struct TaskVector {
    pub deltas: HashMap<String, Vec<f64>>,
}

impl TaskVector {
    /// Scale all delta values by `factor`.
    pub fn scale(&self, factor: f64) -> Self {
        let deltas = self.deltas.iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|x| x * factor).collect()))
            .collect();
        Self { deltas }
    }

    /// Element-wise sum of two task vectors (layers must match).
    pub fn add(&self, other: &TaskVector) -> Result<Self, MergeError> {
        let mut deltas = HashMap::new();
        for name in sorted_keys(&self.deltas) {
            let a = &self.deltas[name];
            let b = other.deltas.get(name).ok_or_else(|| MergeError::LayerMismatch(name.to_string()))?;
            if a.len() != b.len() {
                return Err(MergeError::ShapeMismatch {
                    layer: name.to_string(),
                    a: a.len(),
                    b: b.len(),
                });
            }
            deltas.insert(name.to_string(), a.iter().zip(b.iter()).map(|(x, y)| x + y).collect());
        }
        Ok(Self { deltas })
    }

    /// Apply this task vector to base weights: `base + self`.
    pub fn apply_to(&self, base: &ModelWeights) -> Result<ModelWeights, MergeError> {
        let mut result = ModelWeights::new();
        for (name, delta) in &self.deltas {
            let base_w = base.layers.get(name).ok_or_else(|| MergeError::LayerMismatch(name.clone()))?;
            if base_w.len() != delta.len() {
                return Err(MergeError::ShapeMismatch {
                    layer: name.clone(),
                    a: base_w.len(),
                    b: delta.len(),
                });
            }
            let merged: Vec<f64> = base_w.iter().zip(delta.iter()).map(|(b, d)| b + d).collect();
            result.add_layer(name.clone(), merged);
        }
        Ok(result)
    }

    /// Total L2 norm across all delta vectors.
    pub fn magnitude(&self) -> f64 {
        let sum_sq: f64 = self.deltas.values().flat_map(|v| v.iter()).map(|x| x * x).sum();
        sum_sq.sqrt()
    }
}

// ─── ModelMerger ─────────────────────────────────────────────────────────────

pub struct ModelMerger {
    method: MergeMethod,
}

impl ModelMerger {
    pub fn new(method: MergeMethod) -> Self {
        Self { method }
    }

    // ── SLERP ────────────────────────────────────────────────────────────────

    /// Merge two models using Spherical Linear Interpolation.
    ///
    /// Computes task vectors `ta = model_a - base` and `tb = model_b - base`,
    /// performs SLERP between them per layer, then applies to base.
    /// Falls back to linear interpolation when vectors are parallel (theta ≈ 0).
    pub fn merge_slerp(
        base: &ModelWeights,
        model_a: &ModelWeights,
        model_b: &ModelWeights,
        t: f64,
    ) -> Result<ModelWeights, MergeError> {
        let ta = model_a.task_vector(base)?;
        let tb = model_b.task_vector(base)?;

        let layer_names = {
            let mut names: Vec<String> = ta.deltas.keys().cloned().collect();
            names.sort_unstable();
            names
        };

        let mut result_tv = TaskVector { deltas: HashMap::new() };

        for name in &layer_names {
            let va = ta.deltas.get(name).ok_or_else(|| MergeError::LayerMismatch(name.clone()))?;
            let vb = tb.deltas.get(name).ok_or_else(|| MergeError::LayerMismatch(name.clone()))?;

            if va.len() != vb.len() {
                return Err(MergeError::ShapeMismatch {
                    layer: name.clone(),
                    a: va.len(),
                    b: vb.len(),
                });
            }

            let slerped = slerp_vectors(va, vb, t);
            result_tv.deltas.insert(name.clone(), slerped);
        }

        result_tv.apply_to(base)
    }

    // ── TIES ─────────────────────────────────────────────────────────────────

    /// Merge using TIES-Merging (Trim, Elect Sign, Merge).
    ///
    /// Steps:
    /// 1. Compute task vectors.
    /// 2. Trim: keep top-`density` fraction by absolute value.
    /// 3. Elect sign: per parameter, use the sign whose total magnitude is greatest.
    /// 4. Merge: average same-sign parameters.
    /// 5. Scale by `lambda` and apply to base.
    pub fn merge_ties(
        base: &ModelWeights,
        models: &[&ModelWeights],
        density: f64,
        lambda: f64,
    ) -> Result<ModelWeights, MergeError> {
        if models.is_empty() {
            return Err(MergeError::EmptyModels);
        }

        // 1. Compute task vectors
        let task_vectors: Vec<TaskVector> = models.iter()
            .map(|m| m.task_vector(base))
            .collect::<Result<Vec<_>, _>>()?;

        let layer_names: Vec<String> = {
            let mut names: Vec<String> = task_vectors[0].deltas.keys().cloned().collect();
            names.sort_unstable();
            names
        };

        let mut merged_tv = TaskVector { deltas: HashMap::new() };

        for name in &layer_names {
            let vecs: Vec<&Vec<f64>> = task_vectors.iter()
                .map(|tv| tv.deltas.get(name).ok_or_else(|| MergeError::LayerMismatch(name.clone())))
                .collect::<Result<Vec<_>, _>>()?;

            let param_len = vecs[0].len();

            // 2. Trim: zero out bottom (1 - density) fraction by |value| per model
            let trimmed: Vec<Vec<f64>> = vecs.iter().map(|v| trim_by_density(v, density)).collect();

            // 3. Elect sign: for each param position choose majority sign by summed abs
            let elected_signs = elect_signs(&trimmed, param_len);

            // 4. Average same-sign contributions
            let merged_layer = merge_same_sign(&trimmed, &elected_signs, param_len);

            // Verify shape consistency
            for (i, v) in vecs.iter().enumerate() {
                if v.len() != param_len {
                    return Err(MergeError::ShapeMismatch {
                        layer: name.clone(),
                        a: param_len,
                        b: v.len(),
                    });
                }
                let _ = i;
            }

            merged_tv.deltas.insert(name.clone(), merged_layer);
        }

        // 5. Scale by lambda
        let scaled_tv = merged_tv.scale(lambda);
        scaled_tv.apply_to(base)
    }

    // ── DARE ─────────────────────────────────────────────────────────────────

    /// Merge using DARE (Drop And REscale).
    ///
    /// Deterministic drop: keep parameter at index `i` if `i % round(1/drop_rate) == 0`.
    /// When `drop_rate == 0.0` all parameters are kept.
    pub fn merge_dare(
        base: &ModelWeights,
        model: &ModelWeights,
        drop_rate: f64,
        rescale: bool,
    ) -> Result<ModelWeights, MergeError> {
        if !(0.0..1.0).contains(&drop_rate) {
            return Err(MergeError::InvalidWeight(drop_rate));
        }

        let tv = model.task_vector(base)?;
        let layer_names: Vec<String> = {
            let mut names: Vec<String> = tv.deltas.keys().cloned().collect();
            names.sort_unstable();
            names
        };

        let rescale_factor = if rescale && drop_rate > 0.0 { 1.0 / (1.0 - drop_rate) } else { 1.0 };

        // Stride: keep every `stride`-th parameter (1-indexed offset)
        let stride = if drop_rate > 0.0 {
            (1.0 / drop_rate).round() as usize
        } else {
            1 // keep all
        };

        let mut dare_tv = TaskVector { deltas: HashMap::new() };

        for name in &layer_names {
            let delta = tv.deltas.get(name).ok_or_else(|| MergeError::LayerMismatch(name.clone()))?;
            let kept: Vec<f64> = delta.iter().enumerate().map(|(i, &v)| {
                if drop_rate == 0.0 || (stride > 0 && i % stride == 0) {
                    v * rescale_factor
                } else {
                    0.0
                }
            }).collect();
            dare_tv.deltas.insert(name.clone(), kept);
        }

        dare_tv.apply_to(base)
    }

    // ── Linear ───────────────────────────────────────────────────────────────

    /// Merge models using a simple weighted average.
    pub fn merge_linear(models: &[&ModelWeights], weights: &[f64]) -> Result<ModelWeights, MergeError> {
        if models.is_empty() {
            return Err(MergeError::EmptyModels);
        }
        if models.len() != weights.len() {
            return Err(MergeError::WeightCountMismatch {
                models: models.len(),
                weights: weights.len(),
            });
        }
        for &w in weights {
            if w < 0.0 || !w.is_finite() {
                return Err(MergeError::InvalidWeight(w));
            }
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum < 1e-12 {
            return Err(MergeError::InvalidWeight(weight_sum));
        }
        let norm_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        // Gather all layer names present in ALL models
        let layer_names: Vec<String> = {
            let mut names: Vec<String> = models[0].layers.keys().cloned().collect();
            names.sort_unstable();
            names
        };

        let mut result = ModelWeights::new();

        for name in &layer_names {
            let param_len = models[0].layers.get(name)
                .ok_or_else(|| MergeError::LayerMismatch(name.clone()))?.len();

            let mut combined = vec![0.0f64; param_len];

            for (model, &w) in models.iter().zip(norm_weights.iter()) {
                let layer = model.layers.get(name)
                    .ok_or_else(|| MergeError::LayerMismatch(name.clone()))?;
                if layer.len() != param_len {
                    return Err(MergeError::ShapeMismatch {
                        layer: name.clone(),
                        a: param_len,
                        b: layer.len(),
                    });
                }
                for (c, &v) in combined.iter_mut().zip(layer.iter()) {
                    *c += w * v;
                }
            }
            result.add_layer(name.clone(), combined);
        }

        Ok(result)
    }

    // ── Generic dispatch ─────────────────────────────────────────────────────

    /// Dispatch merge based on this merger's configured method.
    ///
    /// - `Slerp` / `Ties` / `Dare` require exactly one base model (first) + remaining.
    /// - `Linear` treats all models equally (no base).
    pub fn merge(
        &self,
        base: &ModelWeights,
        models: &[&ModelWeights],
    ) -> Result<ModelWeights, MergeError> {
        match &self.method {
            MergeMethod::Slerp { t } => {
                if models.len() < 2 {
                    return Err(MergeError::EmptyModels);
                }
                Self::merge_slerp(base, models[0], models[1], *t)
            }
            MergeMethod::Ties { density, lambda } => {
                Self::merge_ties(base, models, *density, *lambda)
            }
            MergeMethod::Dare { drop_rate, rescale } => {
                if models.is_empty() {
                    return Err(MergeError::EmptyModels);
                }
                Self::merge_dare(base, models[0], *drop_rate, *rescale)
            }
            MergeMethod::Linear { weights } => {
                Self::merge_linear(models, weights)
            }
        }
    }
}

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    #[error("Layer mismatch: layer '{0}' not found in all models")]
    LayerMismatch(String),
    #[error("Shape mismatch for layer '{layer}': {a} vs {b}")]
    ShapeMismatch { layer: String, a: usize, b: usize },
    #[error("Empty models list")]
    EmptyModels,
    #[error("Invalid weight: {0}")]
    InvalidWeight(f64),
    #[error("Weight count mismatch: {models} models but {weights} weights")]
    WeightCountMismatch { models: usize, weights: usize },
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Sorted key list for deterministic iteration over a HashMap.
fn sorted_keys<V>(map: &HashMap<String, V>) -> Vec<&str> {
    let mut keys: Vec<&str> = map.keys().map(|s| s.as_str()).collect();
    keys.sort_unstable();
    keys
}

/// SLERP between two flat vectors.
///
/// Treats each vector as a direction in parameter space.
/// Falls back to linear interpolation when vectors are nearly parallel (theta < 1e-6).
fn slerp_vectors(va: &[f64], vb: &[f64], t: f64) -> Vec<f64> {
    let norm_a: f64 = va.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = vb.iter().map(|x| x * x).sum::<f64>().sqrt();

    // If either vector is a zero vector, fall back to linear
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return va.iter().zip(vb.iter()).map(|(a, b)| (1.0 - t) * a + t * b).collect();
    }

    let dot: f64 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum::<f64>();
    let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta.abs() < 1e-6 {
        // Nearly parallel: linear interpolation
        return va.iter().zip(vb.iter()).map(|(a, b)| (1.0 - t) * a + t * b).collect();
    }

    let sin_theta = theta.sin();
    let w_a = ((1.0 - t) * theta).sin() / sin_theta;
    let w_b = (t * theta).sin() / sin_theta;

    va.iter().zip(vb.iter()).map(|(a, b)| w_a * a + w_b * b).collect()
}

/// Trim a weight vector: keep top-`density` fraction by absolute value,
/// zero out the rest.
fn trim_by_density(v: &[f64], density: f64) -> Vec<f64> {
    if v.is_empty() || density >= 1.0 {
        return v.to_vec();
    }
    if density <= 0.0 {
        return vec![0.0f64; v.len()];
    }

    // Determine threshold: keep the top `keep_k` elements by |value|
    let keep_k = ((v.len() as f64 * density).ceil() as usize).max(1).min(v.len());

    // Sort absolute values descending to find threshold
    let mut abs_sorted: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    abs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = abs_sorted[keep_k - 1];

    v.iter().map(|&x| if x.abs() >= threshold { x } else { 0.0 }).collect()
}

/// For each parameter position, elect the sign with the greatest summed absolute value.
/// Returns +1.0 or -1.0 per position.
fn elect_signs(trimmed: &[Vec<f64>], param_len: usize) -> Vec<f64> {
    (0..param_len).map(|i| {
        let pos_mass: f64 = trimmed.iter()
            .map(|v| if *v.get(i).unwrap_or(&0.0) > 0.0 { v[i].abs() } else { 0.0 })
            .sum();
        let neg_mass: f64 = trimmed.iter()
            .map(|v| if *v.get(i).unwrap_or(&0.0) < 0.0 { v[i].abs() } else { 0.0 })
            .sum();
        if pos_mass >= neg_mass { 1.0 } else { -1.0 }
    }).collect()
}

/// Average contributions that agree with the elected sign.
fn merge_same_sign(trimmed: &[Vec<f64>], elected_signs: &[f64], param_len: usize) -> Vec<f64> {
    (0..param_len).map(|i| {
        let sign = *elected_signs.get(i).unwrap_or(&1.0);
        let same_sign: Vec<f64> = trimmed.iter()
            .filter_map(|v| {
                let x = *v.get(i).unwrap_or(&0.0);
                if x * sign > 0.0 { Some(x) } else { None }
            })
            .collect();
        if same_sign.is_empty() {
            0.0
        } else {
            same_sign.iter().sum::<f64>() / same_sign.len() as f64
        }
    }).collect()
}

// ─── Standalone public free functions ────────────────────────────────────────
//
// These operate on plain `&[f32]` slices and do not require the full
// `ModelWeights` / `ModelMerger` infrastructure.  They implement the same
// algorithms at a lower level for use cases where callers work directly with
// flat weight buffers.

// ── FNV-1a helper for DARE ────────────────────────────────────────────────────

#[inline]
fn fnv1a_dare(seed: u64, index: usize) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in &seed.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for &byte in &(index as u64).to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── SLERP ─────────────────────────────────────────────────────────────────────

/// Spherical Linear Interpolation between two weight vectors on the hypersphere.
///
/// ```text
/// Ω = arccos(dot(w0, w1) / (|w0| * |w1|))
/// slerp(t, w0, w1) = sin((1-t)*Ω)/sin(Ω) * w0 + sin(t*Ω)/sin(Ω) * w1
/// ```
///
/// Falls back to linear interpolation when either vector is near-zero or vectors
/// are nearly parallel.
///
/// # Errors
///
/// Returns `MergeError::ShapeMismatch` when slices have different lengths.
pub fn slerp_merge(weights_a: &[f32], weights_b: &[f32], t: f32) -> Result<Vec<f32>, MergeError> {
    safe_slerp(weights_a, weights_b, t, 1e-6)
}

/// SLERP with explicit epsilon for the parallel-fallback threshold.
///
/// When `|Ω| < eps`, falls back to linear interpolation (`lerp`).
///
/// # Errors
///
/// Returns `MergeError::ShapeMismatch` when slices have different lengths.
pub fn safe_slerp(
    weights_a: &[f32],
    weights_b: &[f32],
    t: f32,
    eps: f32,
) -> Result<Vec<f32>, MergeError> {
    if weights_a.len() != weights_b.len() {
        return Err(MergeError::ShapeMismatch {
            layer: "slerp".to_string(),
            a: weights_a.len(),
            b: weights_b.len(),
        });
    }

    let t64 = t as f64;
    let eps64 = eps as f64;

    let norm_a: f64 = weights_a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = weights_b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();

    // Zero-vector: fall back to lerp
    if norm_a < eps64 || norm_b < eps64 {
        let out = weights_a.iter().zip(weights_b.iter())
            .map(|(&a, &b)| ((1.0 - t64) * a as f64 + t64 * b as f64) as f32)
            .collect();
        return Ok(out);
    }

    let dot: f64 = weights_a.iter().zip(weights_b.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();

    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < eps64 {
        // Nearly parallel: lerp fallback
        let out = weights_a.iter().zip(weights_b.iter())
            .map(|(&a, &b)| ((1.0 - t64) * a as f64 + t64 * b as f64) as f32)
            .collect();
        return Ok(out);
    }

    let sin_omega = omega.sin();
    let w_a = ((1.0 - t64) * omega).sin() / sin_omega;
    let w_b = (t64 * omega).sin() / sin_omega;

    let out = weights_a.iter().zip(weights_b.iter())
        .map(|(&a, &b)| (w_a * a as f64 + w_b * b as f64) as f32)
        .collect();
    Ok(out)
}

// ── TIES ──────────────────────────────────────────────────────────────────────

/// Configuration for the TIES merge algorithm.
#[derive(Debug, Clone)]
pub struct TiesConfig {
    /// Fraction of parameters to **keep** by magnitude (e.g. 0.2 keeps the top 20%).
    pub top_k_fraction: f32,
    /// Scaling factor applied to the merged task vector before adding to the base.
    pub lambda: f32,
}

impl Default for TiesConfig {
    fn default() -> Self {
        Self { top_k_fraction: 0.2, lambda: 1.0 }
    }
}

/// Step 1 of TIES — trim the task vector, keeping only the top-`top_k_fraction`
/// parameters by absolute magnitude.  Everything else is set to zero.
///
/// `task_vector = finetuned - base`
pub fn ties_trim(task_vector: &[f32], _base_vector: &[f32], top_k_fraction: f32) -> Vec<f32> {
    if task_vector.is_empty() {
        return Vec::new();
    }
    let keep_k = ((task_vector.len() as f32 * top_k_fraction).ceil() as usize)
        .max(1)
        .min(task_vector.len());

    let mut abs_sorted: Vec<f32> = task_vector.iter().map(|x| x.abs()).collect();
    // Sort descending to find the keep threshold
    abs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = abs_sorted[keep_k - 1];

    task_vector.iter().map(|&v| if v.abs() >= threshold { v } else { 0.0 }).collect()
}

/// Step 2 of TIES — elect consensus sign per parameter position.
///
/// For each position the sign whose summed absolute mass is greatest wins.
/// Returns a vector of +1.0 or -1.0.
pub fn ties_elect_sign(task_vectors: &[Vec<f32>]) -> Vec<f32> {
    if task_vectors.is_empty() {
        return Vec::new();
    }
    let param_len = task_vectors[0].len();
    (0..param_len).map(|i| {
        let pos_mass: f32 = task_vectors.iter()
            .map(|v| if v.get(i).copied().unwrap_or(0.0) > 0.0 { v[i].abs() } else { 0.0 })
            .sum();
        let neg_mass: f32 = task_vectors.iter()
            .map(|v| if v.get(i).copied().unwrap_or(0.0) < 0.0 { v[i].abs() } else { 0.0 })
            .sum();
        if pos_mass >= neg_mass { 1.0 } else { -1.0 }
    }).collect()
}

/// Step 3 of TIES — disjoint merge: average only parameters whose sign agrees
/// with the elected sign at each position.
pub fn ties_disjoint_merge(task_vectors: &[Vec<f32>], elected_signs: &[f32]) -> Vec<f32> {
    if task_vectors.is_empty() || elected_signs.is_empty() {
        return Vec::new();
    }
    let param_len = elected_signs.len();
    (0..param_len).map(|i| {
        let sign = *elected_signs.get(i).unwrap_or(&1.0);
        let agreeing: Vec<f32> = task_vectors.iter()
            .filter_map(|v| {
                let x = v.get(i).copied().unwrap_or(0.0);
                if x * sign > 0.0 { Some(x) } else { None }
            })
            .collect();
        if agreeing.is_empty() {
            0.0
        } else {
            agreeing.iter().sum::<f32>() / agreeing.len() as f32
        }
    }).collect()
}

/// Full TIES merge pipeline on flat weight slices.
///
/// Steps: Trim → Elect Sign → Disjoint Merge → apply to base.
///
/// # Errors
///
/// Returns `MergeError::EmptyModels` when `task_vectors` is empty.
/// Returns `MergeError::ShapeMismatch` when any task vector length differs from `base`.
pub fn ties_merge_slices(
    base: &[f32],
    task_vectors: &[Vec<f32>],
    config: &TiesConfig,
) -> Result<Vec<f32>, MergeError> {
    if task_vectors.is_empty() {
        return Err(MergeError::EmptyModels);
    }
    for (idx, tv) in task_vectors.iter().enumerate() {
        if tv.len() != base.len() {
            return Err(MergeError::ShapeMismatch {
                layer: format!("task_vector[{}]", idx),
                a: base.len(),
                b: tv.len(),
            });
        }
    }

    // Dummy base vector for ties_trim signature
    let dummy_base = vec![0.0f32; base.len()];

    // Step 1: trim each task vector
    let trimmed: Vec<Vec<f32>> = task_vectors.iter()
        .map(|tv| ties_trim(tv, &dummy_base, config.top_k_fraction))
        .collect();

    // Step 2: elect signs
    let elected_signs = ties_elect_sign(&trimmed);

    // Step 3: disjoint merge
    let merged_delta = ties_disjoint_merge(&trimmed, &elected_signs);

    // Apply to base with lambda scaling
    let result: Vec<f32> = base.iter().zip(merged_delta.iter())
        .map(|(&b, &d)| b + config.lambda * d)
        .collect();

    Ok(result)
}

// ── DARE ──────────────────────────────────────────────────────────────────────

/// Configuration for DARE (Drop And REscale).
#[derive(Debug, Clone)]
pub struct DareConfig {
    /// Probability of zeroing each delta element.  Must be in `[0, 1)`.
    pub drop_rate: f32,
    /// Whether to rescale surviving deltas by `1 / (1 - drop_rate)`.
    pub rescale: bool,
    /// Seed for reproducible deterministic dropping via FNV hash.
    pub seed: u64,
}

impl Default for DareConfig {
    fn default() -> Self {
        Self { drop_rate: 0.9, rescale: true, seed: 0 }
    }
}

/// Sparsify a task vector using DARE.
///
/// Each element is dropped (set to 0) with probability `config.drop_rate`.
/// When `config.rescale` is `true`, surviving elements are multiplied by
/// `1 / (1 - drop_rate)` to preserve the expected value.
///
/// The drop decision is deterministic: element at position `i` is dropped iff
/// `FNV(seed, i) mod 2^53 / 2^53 < drop_rate`.
pub fn dare_sparsify(task_vector: &[f32], config: &DareConfig) -> Vec<f32> {
    let rescale = if config.rescale && config.drop_rate > 0.0 && config.drop_rate < 1.0 {
        1.0 / (1.0 - config.drop_rate)
    } else {
        1.0
    };

    task_vector.iter().enumerate().map(|(i, &v)| {
        let h = fnv1a_dare(config.seed, i);
        let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
        if unit < config.drop_rate as f64 {
            0.0
        } else {
            v * rescale
        }
    }).collect()
}

/// Full DARE merge: apply sparsified delta to base.
///
/// `result = base + dare_sparsify(finetuned - base, config)`
///
/// # Errors
///
/// Returns `MergeError::ShapeMismatch` when `base` and `fine_tuned` lengths differ.
/// Returns `MergeError::InvalidWeight` when `drop_rate >= 1.0`.
pub fn dare_merge_slices(
    base: &[f32],
    fine_tuned: &[f32],
    config: &DareConfig,
) -> Result<Vec<f32>, MergeError> {
    if base.len() != fine_tuned.len() {
        return Err(MergeError::ShapeMismatch {
            layer: "dare".to_string(),
            a: base.len(),
            b: fine_tuned.len(),
        });
    }
    if config.drop_rate >= 1.0 {
        return Err(MergeError::InvalidWeight(config.drop_rate as f64));
    }

    let task_vector: Vec<f32> = base.iter().zip(fine_tuned.iter()).map(|(&b, &f)| f - b).collect();
    let sparse_delta = dare_sparsify(&task_vector, config);
    let result: Vec<f32> = base.iter().zip(sparse_delta.iter()).map(|(&b, &d)| b + d).collect();
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a ModelWeights with one layer
    fn single_layer(name: &str, values: Vec<f64>) -> ModelWeights {
        let mut m = ModelWeights::new();
        m.add_layer(name, values);
        m
    }

    // Helper: two-layer model
    fn two_layer_model(a_vals: Vec<f64>, b_vals: Vec<f64>) -> ModelWeights {
        let mut m = ModelWeights::new();
        m.add_layer("layer_a", a_vals);
        m.add_layer("layer_b", b_vals);
        m
    }

    // 1. task_vector computation
    #[test]
    fn test_task_vector_computation() {
        let base = single_layer("w", vec![1.0, 2.0, 3.0]);
        let fine = single_layer("w", vec![2.0, 3.0, 4.0]);
        let tv = fine.task_vector(&base).unwrap();
        assert_eq!(tv.deltas["w"], vec![1.0, 1.0, 1.0]);
    }

    // 2. task_vector scale
    #[test]
    fn test_task_vector_scale() {
        let base = single_layer("w", vec![0.0, 0.0, 0.0]);
        let fine = single_layer("w", vec![2.0, 4.0, 6.0]);
        let tv = fine.task_vector(&base).unwrap();
        let scaled = tv.scale(0.5);
        assert_eq!(scaled.deltas["w"], vec![1.0, 2.0, 3.0]);
    }

    // 3. task_vector apply_to
    #[test]
    fn test_task_vector_apply_to() {
        let base = single_layer("w", vec![10.0, 20.0, 30.0]);
        let fine = single_layer("w", vec![12.0, 22.0, 32.0]);
        let tv = fine.task_vector(&base).unwrap();
        let result = tv.apply_to(&base).unwrap();
        assert_eq!(result.layers["w"], vec![12.0, 22.0, 32.0]);
    }

    // 4. merge_linear with two models, equal weights
    #[test]
    fn test_merge_linear_two_equal() {
        let m1 = single_layer("w", vec![0.0, 0.0, 0.0]);
        let m2 = single_layer("w", vec![4.0, 8.0, 12.0]);
        let result = ModelMerger::merge_linear(&[&m1, &m2], &[1.0, 1.0]).unwrap();
        let w = &result.layers["w"];
        assert!((w[0] - 2.0).abs() < 1e-10);
        assert!((w[1] - 4.0).abs() < 1e-10);
        assert!((w[2] - 6.0).abs() < 1e-10);
    }

    // 5. merge_linear with three models, custom weights
    #[test]
    fn test_merge_linear_three_models() {
        let m1 = single_layer("w", vec![6.0]);
        let m2 = single_layer("w", vec![0.0]);
        let m3 = single_layer("w", vec![3.0]);
        // weights 2:1:1 → (2*6 + 1*0 + 1*3) / 4 = 15/4 = 3.75
        let result = ModelMerger::merge_linear(&[&m1, &m2, &m3], &[2.0, 1.0, 1.0]).unwrap();
        assert!((result.layers["w"][0] - 3.75).abs() < 1e-10);
    }

    // 6. merge_linear wrong weight count returns error
    #[test]
    fn test_merge_linear_wrong_weight_count() {
        let m1 = single_layer("w", vec![1.0]);
        let m2 = single_layer("w", vec![2.0]);
        let err = ModelMerger::merge_linear(&[&m1, &m2], &[1.0]).unwrap_err();
        assert!(matches!(err, MergeError::WeightCountMismatch { .. }));
    }

    // 7. merge_slerp with parallel task vectors (t=0.5) → linear interpolation fallback
    #[test]
    fn test_merge_slerp_parallel_vectors() {
        // model_a and model_b have identical task vectors (parallel) → slerp falls back to lerp
        let base = single_layer("w", vec![0.0, 0.0]);
        let ma = single_layer("w", vec![2.0, 2.0]);
        let mb = single_layer("w", vec![2.0, 2.0]);
        let result = ModelMerger::merge_slerp(&base, &ma, &mb, 0.5).unwrap();
        // midpoint of identical vectors = same vector
        assert!((result.layers["w"][0] - 2.0).abs() < 1e-10);
        assert!((result.layers["w"][1] - 2.0).abs() < 1e-10);
    }

    // 8. merge_slerp extreme t=0 → model_a, t=1 → model_b
    #[test]
    fn test_merge_slerp_extremes() {
        let base = single_layer("w", vec![0.0, 0.0, 0.0]);
        let ma = single_layer("w", vec![3.0, 0.0, 0.0]);
        let mb = single_layer("w", vec![0.0, 4.0, 0.0]);

        let at_t0 = ModelMerger::merge_slerp(&base, &ma, &mb, 0.0).unwrap();
        let at_t1 = ModelMerger::merge_slerp(&base, &ma, &mb, 1.0).unwrap();

        // t=0 → weight of model_a → [3,0,0]
        assert!((at_t0.layers["w"][0] - 3.0).abs() < 1e-8, "t=0 should reproduce model_a");
        assert!(at_t0.layers["w"][1].abs() < 1e-8);

        // t=1 → weight of model_b → [0,4,0]
        assert!(at_t1.layers["w"][0].abs() < 1e-8);
        assert!((at_t1.layers["w"][1] - 4.0).abs() < 1e-8, "t=1 should reproduce model_b");
    }

    // 9. merge_ties with density=1.0 keeps all parameters
    #[test]
    fn test_merge_ties_density_1() {
        let base = single_layer("w", vec![0.0, 0.0, 0.0]);
        let m1 = single_layer("w", vec![2.0, -1.0, 3.0]);
        let m2 = single_layer("w", vec![4.0, -2.0, 1.0]);

        let result = ModelMerger::merge_ties(&base, &[&m1, &m2], 1.0, 1.0).unwrap();
        let w = &result.layers["w"];
        // With density=1.0 all params kept, signs elected, lambda=1
        // Pos 0: both positive → avg(2,4)=3 → sign positive
        // Pos 1: both negative → avg(-1,-2)=-1.5
        // Pos 2: both positive → avg(3,1)=2
        assert!(w[0] > 0.0, "pos 0 should be positive");
        assert!(w[1] < 0.0, "pos 1 should be negative");
        assert!(w[2] > 0.0, "pos 2 should be positive");
    }

    // 10. merge_ties sign election: conflicting signs, majority wins
    #[test]
    fn test_merge_ties_sign_election() {
        let base = single_layer("w", vec![0.0]);
        // m1 has large positive delta, m2 small negative delta
        let m1 = single_layer("w", vec![10.0]);
        let m2 = single_layer("w", vec![-1.0]);
        let m3 = single_layer("w", vec![8.0]);

        let result = ModelMerger::merge_ties(&base, &[&m1, &m2, &m3], 1.0, 1.0).unwrap();
        // Positive mass: 10+8=18, negative mass: 1 → positive sign elected
        assert!(result.layers["w"][0] > 0.0, "majority positive sign should win");
    }

    // 11. merge_dare with drop_rate=0.0 keeps all parameters unchanged
    #[test]
    fn test_merge_dare_drop_zero() {
        let base = single_layer("w", vec![1.0, 2.0, 3.0]);
        let model = single_layer("w", vec![2.0, 4.0, 6.0]);
        let result = ModelMerger::merge_dare(&base, &model, 0.0, false).unwrap();
        assert_eq!(result.layers["w"], vec![2.0, 4.0, 6.0]);
    }

    // 12. merge_dare with rescale: survivors amplified by 1/(1-drop_rate)
    #[test]
    fn test_merge_dare_rescale() {
        let base = single_layer("w", vec![0.0, 0.0, 0.0, 0.0]);
        // delta = [1,1,1,1]
        let model = single_layer("w", vec![1.0, 1.0, 1.0, 1.0]);
        // drop_rate=0.5 → stride=2 → keep indices 0,2; drop 1,3
        // rescale_factor = 1/(1-0.5) = 2.0
        let result = ModelMerger::merge_dare(&base, &model, 0.5, true).unwrap();
        let w = &result.layers["w"];
        assert!((w[0] - 2.0).abs() < 1e-10, "kept param at idx 0 should be rescaled");
        assert!(w[1].abs() < 1e-10, "dropped param at idx 1 should be 0");
        assert!((w[2] - 2.0).abs() < 1e-10, "kept param at idx 2 should be rescaled");
        assert!(w[3].abs() < 1e-10, "dropped param at idx 3 should be 0");
    }

    // 13. layer mismatch error
    #[test]
    fn test_layer_mismatch_error() {
        let base = single_layer("layer_x", vec![1.0, 2.0]);
        let fine = single_layer("layer_y", vec![3.0, 4.0]); // different name
        let err = fine.task_vector(&base).unwrap_err();
        assert!(matches!(err, MergeError::LayerMismatch(_)));
    }

    // 14. shape mismatch error
    #[test]
    fn test_shape_mismatch_error() {
        let base = single_layer("w", vec![1.0, 2.0, 3.0]);
        let fine = single_layer("w", vec![1.0, 2.0]); // different length
        let err = fine.task_vector(&base).unwrap_err();
        assert!(matches!(err, MergeError::ShapeMismatch { .. }));
    }

    // 15. magnitude calculation
    #[test]
    fn test_magnitude_calculation() {
        let base = single_layer("w", vec![0.0, 0.0, 0.0, 0.0]);
        // delta = [3, 4, 0, 0] → magnitude = 5
        let fine = single_layer("w", vec![3.0, 4.0, 0.0, 0.0]);
        let tv = fine.task_vector(&base).unwrap();
        assert!((tv.magnitude() - 5.0).abs() < 1e-10);
    }

    // ─── Additional: total_params and layer_names ──────────────────────────────
    #[test]
    fn test_total_params_and_layer_names() {
        let m = two_layer_model(vec![1.0, 2.0, 3.0], vec![4.0, 5.0]);
        assert_eq!(m.total_params(), 5);
        let names = m.layer_names();
        // sorted: layer_a before layer_b
        assert_eq!(names, vec!["layer_a", "layer_b"]);
    }

    // ─── merge via generic dispatch ───────────────────────────────────────────
    #[test]
    fn test_merge_dispatch_linear() {
        let m1 = single_layer("w", vec![0.0, 0.0]);
        let m2 = single_layer("w", vec![2.0, 4.0]);
        let base = single_layer("w", vec![0.0, 0.0]); // unused for linear
        let merger = ModelMerger::new(MergeMethod::Linear { weights: vec![1.0, 1.0] });
        let result = merger.merge(&base, &[&m1, &m2]).unwrap();
        assert!((result.layers["w"][0] - 1.0).abs() < 1e-10);
        assert!((result.layers["w"][1] - 2.0).abs() < 1e-10);
    }

    // ── New standalone SLERP / TIES / DARE tests ──────────────────────────────

    // 18. slerp_merge t=0 returns model_a
    #[test]
    fn test_slerp_merge_t0_returns_a() {
        let a = vec![3.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 4.0, 0.0];
        let result = slerp_merge(&a, &b, 0.0).expect("ok");
        for (&r, &expected) in result.iter().zip(a.iter()) {
            assert!((r - expected).abs() < 1e-5, "t=0 should equal a");
        }
    }

    // 19. slerp_merge t=1 returns model_b
    #[test]
    fn test_slerp_merge_t1_returns_b() {
        let a = vec![3.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 4.0, 0.0];
        let result = slerp_merge(&a, &b, 1.0).expect("ok");
        for (&r, &expected) in result.iter().zip(b.iter()) {
            assert!((r - expected).abs() < 1e-5, "t=1 should equal b");
        }
    }

    // 20. slerp_merge shape mismatch returns error
    #[test]
    fn test_slerp_merge_shape_mismatch() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0]; // different length
        let err = slerp_merge(&a, &b, 0.5).unwrap_err();
        assert!(matches!(err, MergeError::ShapeMismatch { .. }));
    }

    // 21. slerp_merge at t=0.5 between orthogonal unit vectors gives 45-degree result
    #[test]
    fn test_slerp_merge_orthogonal_midpoint() {
        // [1,0] and [0,1] at omega=π/2 → midpoint is [cos(π/4), sin(π/4)]
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let result = slerp_merge(&a, &b, 0.5).expect("ok");
        let expected = (std::f32::consts::PI / 4.0).cos(); // ≈ 0.7071
        assert!((result[0] - expected).abs() < 1e-5);
        assert!((result[1] - expected).abs() < 1e-5);
    }

    // 22. safe_slerp with eps=1.0 forces lerp (all omega < 1.0 treated as parallel)
    #[test]
    fn test_safe_slerp_large_eps_forces_lerp() {
        let a = vec![2.0f32, 0.0];
        let b = vec![4.0f32, 0.0];
        // vectors are parallel, omega≈0 → lerp
        let result = safe_slerp(&a, &b, 0.5, 1e-6).expect("ok");
        // lerp(0.5, [2,0], [4,0]) = [3, 0]
        assert!((result[0] - 3.0).abs() < 1e-5, "lerp fallback: {}", result[0]);
    }

    // 23. ties_trim keeps top-k fraction by magnitude
    #[test]
    fn test_ties_trim_keeps_top_fraction() {
        let tv = vec![1.0f32, 4.0, 2.0, 3.0]; // magnitudes: 1,4,2,3
        let base = vec![0.0f32; 4];
        // top 50% = top 2 → keep magnitudes 3 and 4
        let trimmed = ties_trim(&tv, &base, 0.5);
        // positions: 0→trim(1<3), 1→keep(4), 2→trim(2<3), 3→keep(3)
        assert_eq!(trimmed[0], 0.0, "mag 1 should be trimmed");
        assert!((trimmed[1] - 4.0).abs() < 1e-6, "mag 4 should be kept");
        assert_eq!(trimmed[2], 0.0, "mag 2 should be trimmed");
        assert!((trimmed[3] - 3.0).abs() < 1e-6, "mag 3 should be kept");
    }

    // 24. ties_elect_sign majority positive
    #[test]
    fn test_ties_elect_sign_majority_positive() {
        let tvs = vec![
            vec![5.0f32],
            vec![-1.0f32],
            vec![3.0f32],
        ];
        let signs = ties_elect_sign(&tvs);
        assert!((signs[0] - 1.0).abs() < 1e-6, "majority positive should elect +1");
    }

    // 25. ties_elect_sign majority negative
    #[test]
    fn test_ties_elect_sign_majority_negative() {
        let tvs = vec![
            vec![-10.0f32],
            vec![1.0f32],
            vec![-2.0f32],
        ];
        let signs = ties_elect_sign(&tvs);
        assert!((signs[0] - (-1.0)).abs() < 1e-6, "majority negative should elect -1");
    }

    // 26. ties_disjoint_merge averages only same-sign contributions
    #[test]
    fn test_ties_disjoint_merge_same_sign_only() {
        let tvs = vec![
            vec![6.0f32],  // positive
            vec![-1.0f32], // negative (disagrees with elected +1)
            vec![4.0f32],  // positive
        ];
        let elected = vec![1.0f32]; // positive sign elected
        let merged = ties_disjoint_merge(&tvs, &elected);
        // Average of [6, 4] = 5
        assert!((merged[0] - 5.0).abs() < 1e-5, "disjoint merge should avg same-sign: {}", merged[0]);
    }

    // 27. ties_merge_slices pipeline correctness
    #[test]
    fn test_ties_merge_slices_pipeline() {
        let base = vec![0.0f32; 2];
        let task_vectors = vec![
            vec![4.0f32, -2.0],
            vec![6.0f32, -3.0],
        ];
        let cfg = TiesConfig { top_k_fraction: 1.0, lambda: 1.0 };
        let result = ties_merge_slices(&base, &task_vectors, &cfg).expect("ok");
        // Both positive at pos 0: avg = 5.0; both negative at pos 1: avg = -2.5
        assert!((result[0] - 5.0).abs() < 1e-5, "pos 0 should be 5, got {}", result[0]);
        assert!((result[1] - (-2.5)).abs() < 1e-5, "pos 1 should be -2.5, got {}", result[1]);
    }

    // 28. ties_merge_slices empty returns error
    #[test]
    fn test_ties_merge_slices_empty_error() {
        let base = vec![1.0f32];
        let cfg = TiesConfig::default();
        let err = ties_merge_slices(&base, &[], &cfg).unwrap_err();
        assert!(matches!(err, MergeError::EmptyModels));
    }

    // 29. dare_sparsify: drop_rate=0 keeps everything
    #[test]
    fn test_dare_sparsify_no_drop() {
        let tv = vec![1.0f32, 2.0, 3.0, 4.0];
        let cfg = DareConfig { drop_rate: 0.0, rescale: false, seed: 0 };
        let sparse = dare_sparsify(&tv, &cfg);
        assert_eq!(sparse, tv, "drop_rate=0 should keep all elements");
    }

    // 30. dare_sparsify: rescale=true amplifies survivors by 1/(1-drop_rate)
    #[test]
    fn test_dare_sparsify_rescale_amplifies() {
        // Use seed that we know drops/keeps specific positions (we test invariant not positions)
        let n = 1000usize;
        let tv = vec![1.0f32; n];
        let cfg = DareConfig { drop_rate: 0.5, rescale: true, seed: 0 };
        let cfg_no_rescale = DareConfig { drop_rate: 0.5, rescale: false, seed: 0 };
        let sparse = dare_sparsify(&tv, &cfg);
        let sparse_no = dare_sparsify(&tv, &cfg_no_rescale);
        // For each kept element (non-zero in no_rescale), rescaled version should be 2x
        for (&r, &no) in sparse.iter().zip(sparse_no.iter()) {
            if no.abs() > 1e-7 {
                assert!((r - 2.0 * no).abs() < 1e-5, "rescaled {} should be 2x no-rescale {}", r, no);
            }
        }
    }

    // 31. dare_merge_slices: drop_rate=0, rescale=false → result equals finetuned
    #[test]
    fn test_dare_merge_slices_no_drop_equals_finetuned() {
        let base = vec![1.0f32, 2.0, 3.0];
        let fine = vec![2.0f32, 4.0, 6.0];
        let cfg = DareConfig { drop_rate: 0.0, rescale: false, seed: 0 };
        let result = dare_merge_slices(&base, &fine, &cfg).expect("ok");
        for (&r, &f) in result.iter().zip(fine.iter()) {
            assert!((r - f).abs() < 1e-6, "no-drop should equal finetuned: {} vs {}", r, f);
        }
    }

    // 32. dare_merge_slices: shape mismatch error
    #[test]
    fn test_dare_merge_slices_shape_mismatch() {
        let base = vec![0.0f32; 3];
        let fine = vec![1.0f32; 4]; // different length
        let cfg = DareConfig::default();
        let err = dare_merge_slices(&base, &fine, &cfg).unwrap_err();
        assert!(matches!(err, MergeError::ShapeMismatch { .. }));
    }

    // 33. dare_merge_slices: drop_rate=1.0 returns error
    #[test]
    fn test_dare_merge_slices_drop_rate_1_error() {
        let base = vec![0.0f32; 3];
        let fine = vec![1.0f32; 3];
        let cfg = DareConfig { drop_rate: 1.0, rescale: false, seed: 0 };
        let err = dare_merge_slices(&base, &fine, &cfg).unwrap_err();
        assert!(matches!(err, MergeError::InvalidWeight(_)));
    }

    // 34. dare_sparsify determinism: same seed, same result
    #[test]
    fn test_dare_sparsify_deterministic() {
        let tv: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let cfg = DareConfig { drop_rate: 0.7, rescale: true, seed: 42 };
        let r1 = dare_sparsify(&tv, &cfg);
        let r2 = dare_sparsify(&tv, &cfg);
        assert_eq!(r1, r2, "dare_sparsify must be deterministic");
    }

    // 35. ties_merge_slices with lambda=0 returns base unchanged
    #[test]
    fn test_ties_merge_slices_lambda_zero_returns_base() {
        let base = vec![1.0f32, 2.0, 3.0];
        let task_vectors = vec![vec![5.0f32, 10.0, 15.0]];
        let cfg = TiesConfig { top_k_fraction: 1.0, lambda: 0.0 };
        let result = ties_merge_slices(&base, &task_vectors, &cfg).expect("ok");
        for (&r, &b) in result.iter().zip(base.iter()) {
            assert!((r - b).abs() < 1e-6, "lambda=0 should return base unchanged");
        }
    }
}
