//! # Vision Transformer (ViT) Task-Specific Implementations
//!
//! This module provides task-specific utilities for ViT image models, including:
//! - Image classification helpers (top-k logit filtering, argmax prediction)
//! - Feature extraction utilities (mean pooling, CLS-token extraction)
//! - Patch grid geometry calculations
//! - LayerNorm and GELU helpers used in the ViT encoder
//! - Attention mask computation for image patches
//! - Multi-label classification scoring

use std::fmt;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by ViT task-specific operations.
#[derive(Debug)]
pub enum ViTTaskError {
    /// Empty input patches or features.
    EmptyInput,
    /// Image size is not divisible by patch size.
    InvalidPatchSize {
        image_size: usize,
        patch_size: usize,
    },
    /// Top-k value exceeds the number of classes.
    TopKTooLarge { k: usize, num_classes: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Forward pass failed.
    ForwardError(String),
    /// Unexpected batch size.
    BatchSizeMismatch { expected: usize, got: usize },
}

impl fmt::Display for ViTTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViTTaskError::EmptyInput => write!(f, "ViT task error: empty input"),
            ViTTaskError::InvalidPatchSize {
                image_size,
                patch_size,
            } => write!(
                f,
                "ViT task error: image_size={image_size} not divisible by patch_size={patch_size}"
            ),
            ViTTaskError::TopKTooLarge { k, num_classes } => write!(
                f,
                "ViT task error: top_k={k} exceeds num_classes={num_classes}"
            ),
            ViTTaskError::InvalidConfig(msg) => {
                write!(f, "ViT task error: invalid config: {msg}")
            },
            ViTTaskError::ForwardError(msg) => {
                write!(f, "ViT task error: forward error: {msg}")
            },
            ViTTaskError::BatchSizeMismatch { expected, got } => write!(
                f,
                "ViT task error: batch size mismatch — expected {expected}, got {got}"
            ),
        }
    }
}

impl std::error::Error for ViTTaskError {}

// ─── GELU ─────────────────────────────────────────────────────────────────────

/// GELU activation (tanh approximation) used in the ViT MLP blocks.
///
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Apply GELU element-wise to a slice.
pub fn gelu_vec(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| gelu(x)).collect()
}

// ─── LayerNorm ────────────────────────────────────────────────────────────────

/// Apply LayerNorm to a flat slice (unit weight, zero bias).
///
/// `output[i] = (input[i] - mean) / sqrt(var + eps)`
pub fn layer_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let denom = (var + eps).sqrt();
    input.iter().map(|&x| (x - mean) / denom).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return exps;
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Sigmoid ──────────────────────────────────────────────────────────────────

/// Element-wise sigmoid for multi-label classification.
pub fn sigmoid_vec(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

// ─── Top-k prediction ─────────────────────────────────────────────────────────

/// Return the indices of the top-k class logits (sorted by descending score).
///
/// # Errors
///
/// Returns [`ViTTaskError::TopKTooLarge`] when `k > num_classes`.
pub fn top_k_predictions(logits: &[f32], k: usize) -> Result<Vec<usize>, ViTTaskError> {
    let n = logits.len();
    if k > n {
        return Err(ViTTaskError::TopKTooLarge { k, num_classes: n });
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().take(k).map(|(i, _)| i).collect())
}

/// Return the single predicted class index (argmax).
pub fn predict_class(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

// ─── Patch geometry ───────────────────────────────────────────────────────────

/// Compute the total number of patches for a square image.
///
/// # Errors
///
/// Returns [`ViTTaskError::InvalidPatchSize`] if the image size is not divisible
/// by the patch size or either dimension is zero.
pub fn num_patches(image_size: usize, patch_size: usize) -> Result<usize, ViTTaskError> {
    if patch_size == 0 {
        return Err(ViTTaskError::InvalidConfig(
            "patch_size must be > 0".to_string(),
        ));
    }
    if !image_size.is_multiple_of(patch_size) {
        return Err(ViTTaskError::InvalidPatchSize {
            image_size,
            patch_size,
        });
    }
    let per_side = image_size / patch_size;
    Ok(per_side * per_side)
}

/// Compute the sequence length including the class token (if used).
pub fn vit_seq_length(
    image_size: usize,
    patch_size: usize,
    use_class_token: bool,
) -> Result<usize, ViTTaskError> {
    let patches = num_patches(image_size, patch_size)?;
    Ok(if use_class_token { patches + 1 } else { patches })
}

// ─── Feature pooling ──────────────────────────────────────────────────────────

/// Pool a flat `[seq_len * hidden_size]` feature map by averaging all tokens.
///
/// # Errors
///
/// Returns [`ViTTaskError::EmptyInput`] when the slice is empty.
pub fn mean_pool(features: &[f32], hidden_size: usize) -> Result<Vec<f32>, ViTTaskError> {
    if features.is_empty() || hidden_size == 0 {
        return Err(ViTTaskError::EmptyInput);
    }
    let seq_len = features.len() / hidden_size;
    if seq_len == 0 {
        return Err(ViTTaskError::EmptyInput);
    }
    let mut pooled = vec![0.0f32; hidden_size];
    for t in 0..seq_len {
        let start = t * hidden_size;
        for (i, val) in features[start..start + hidden_size].iter().enumerate() {
            pooled[i] += val;
        }
    }
    let n = seq_len as f32;
    for v in &mut pooled {
        *v /= n;
    }
    Ok(pooled)
}

/// Extract the CLS token (first token) from a flat `[seq_len * hidden_size]` map.
///
/// # Errors
///
/// Returns [`ViTTaskError::EmptyInput`] when the slice is empty.
pub fn cls_pool(features: &[f32], hidden_size: usize) -> Result<Vec<f32>, ViTTaskError> {
    if features.is_empty() || hidden_size == 0 {
        return Err(ViTTaskError::EmptyInput);
    }
    Ok(features[..hidden_size].to_vec())
}

// ─── ViTModel task head ───────────────────────────────────────────────────────

/// Feature extraction head for ViT.
///
/// Wraps mean-pooling and CLS-token extraction for use downstream.
/// All weights are zero-initialised (test-only).
pub struct ViTForFeatureExtraction {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Whether to use the CLS token (vs. mean pooling).
    pub use_class_token: bool,
}

impl ViTForFeatureExtraction {
    /// Create a feature extraction head.
    pub fn new(hidden_size: usize, use_class_token: bool) -> Result<Self, ViTTaskError> {
        if hidden_size == 0 {
            return Err(ViTTaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            hidden_size,
            use_class_token,
        })
    }

    /// Pool the encoder hidden states to a fixed-size feature vector.
    ///
    /// `hidden_states` is flat `[seq_len * hidden_size]`.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, ViTTaskError> {
        if self.use_class_token {
            cls_pool(hidden_states, self.hidden_size)
        } else {
            mean_pool(hidden_states, self.hidden_size)
        }
    }
}

// ─── ViTForImageClassification task head ─────────────────────────────────────

/// Image classification head for ViT.
///
/// Projects the pooled CLS representation to `[num_labels]` logits.
/// All weights are zero-initialised for test use.
pub struct ViTTaskClassifier {
    /// Number of output classes.
    pub num_labels: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Whether to use the CLS token (vs. mean pooling).
    pub use_class_token: bool,
    /// Weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias `[num_labels]`.
    bias: Vec<f32>,
}

impl ViTTaskClassifier {
    /// Create a new image classification head.
    pub fn new(
        hidden_size: usize,
        num_labels: usize,
        use_class_token: bool,
    ) -> Result<Self, ViTTaskError> {
        if hidden_size == 0 {
            return Err(ViTTaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(ViTTaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            hidden_size,
            use_class_token,
            weight: vec![vec![0.0f32; hidden_size]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Project pooled representation to class logits.
    ///
    /// `hidden_states` is flat `[seq_len * hidden_size]`.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, ViTTaskError> {
        let pooled = if self.use_class_token {
            cls_pool(hidden_states, self.hidden_size)?
        } else {
            mean_pool(hidden_states, self.hidden_size)?
        };
        let logits: Vec<f32> = self
            .weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| row.iter().zip(pooled.iter()).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        Ok(logits)
    }
}

// ─── Multi-label classification ───────────────────────────────────────────────

/// Threshold multi-label predictions from sigmoid outputs.
///
/// Returns the set of class indices where `sigmoid(logit) >= threshold`.
pub fn multi_label_predict(logits: &[f32], threshold: f32) -> Vec<usize> {
    sigmoid_vec(logits)
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| if p >= threshold { Some(i) } else { None })
        .collect()
}

// ─── Attention bias ───────────────────────────────────────────────────────────

/// Build a full-attention bias mask (all zeros = all pairs visible).
///
/// ViT uses full bidirectional attention with no causal masking.
pub fn vit_attention_bias(seq_len: usize) -> Vec<f32> {
    vec![0.0f32; seq_len * seq_len]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_vit_config_base ──────────────────────────────────────────────

    #[test]
    fn test_vit_config_base() {
        use crate::vit::config::ViTConfig;
        use trustformers_core::traits::Config;
        let cfg = ViTConfig::base();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_labels, 1000);
        assert!(cfg.validate().is_ok());
    }

    // ── test_vit_config_tiny ──────────────────────────────────────────────

    #[test]
    fn test_vit_config_tiny() {
        use crate::vit::config::ViTConfig;
        let cfg = ViTConfig::tiny();
        assert_eq!(cfg.hidden_size, 192);
        assert_eq!(cfg.num_attention_heads, 3);
    }

    // ── test_vit_config_large ─────────────────────────────────────────────

    #[test]
    fn test_vit_config_large() {
        use crate::vit::config::ViTConfig;
        let cfg = ViTConfig::large();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
    }

    // ── test_vit_config_seq_length ────────────────────────────────────────

    #[test]
    fn test_vit_config_seq_length() {
        use crate::vit::config::ViTConfig;
        let cfg = ViTConfig::base();
        // 224/16 = 14 patches per side → 196 patches + 1 CLS = 197
        assert_eq!(cfg.num_patches(), 196);
        assert_eq!(cfg.seq_length(), 197);
    }

    // ── test_vit_gelu_zero ────────────────────────────────────────────────

    #[test]
    fn test_vit_gelu_zero() {
        assert!(gelu(0.0).abs() < 1e-6);
    }

    // ── test_vit_gelu_large_positive ─────────────────────────────────────

    #[test]
    fn test_vit_gelu_large_positive() {
        let v = gelu(10.0);
        assert!((v - 10.0).abs() < 1e-3, "gelu(10) ≈ 10, got {v}");
    }

    // ── test_vit_layer_norm_zero_mean ─────────────────────────────────────

    #[test]
    fn test_vit_layer_norm_zero_mean() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = layer_norm(&x, 1e-12);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "layer_norm mean should be ~0, got {mean}"
        );
    }

    // ── test_vit_layer_norm_unit_variance ────────────────────────────────

    #[test]
    fn test_vit_layer_norm_unit_variance() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0];
        let out = layer_norm(&x, 1e-12);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "variance should be ~1, got {var}");
    }

    // ── test_vit_softmax ──────────────────────────────────────────────────

    #[test]
    fn test_vit_softmax() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    // ── test_vit_num_patches ──────────────────────────────────────────────

    #[test]
    fn test_vit_num_patches() {
        assert_eq!(num_patches(224, 16).expect("patches"), 196);
        assert_eq!(num_patches(32, 16).expect("patches"), 4);
    }

    // ── test_vit_num_patches_invalid ─────────────────────────────────────

    #[test]
    fn test_vit_num_patches_invalid() {
        assert!(matches!(
            num_patches(225, 16),
            Err(ViTTaskError::InvalidPatchSize { .. })
        ));
    }

    // ── test_vit_seq_length_with_cls ─────────────────────────────────────

    #[test]
    fn test_vit_seq_length_with_cls() {
        let seq = vit_seq_length(224, 16, true).expect("seq_length");
        assert_eq!(seq, 197);
    }

    // ── test_vit_seq_length_without_cls ──────────────────────────────────

    #[test]
    fn test_vit_seq_length_without_cls() {
        let seq = vit_seq_length(224, 16, false).expect("seq_length");
        assert_eq!(seq, 196);
    }

    // ── test_vit_top_k_predictions ────────────────────────────────────────

    #[test]
    fn test_vit_top_k_predictions() {
        let logits = vec![0.1f32, 0.9, 0.5, 0.7, 0.3];
        let top3 = top_k_predictions(&logits, 3).expect("top_k");
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0], 1); // highest: 0.9
        assert_eq!(top3[1], 3); // second: 0.7
        assert_eq!(top3[2], 2); // third:  0.5
    }

    // ── test_vit_top_k_too_large ──────────────────────────────────────────

    #[test]
    fn test_vit_top_k_too_large() {
        let logits = vec![1.0f32; 3];
        assert!(matches!(
            top_k_predictions(&logits, 10),
            Err(ViTTaskError::TopKTooLarge {
                k: 10,
                num_classes: 3
            })
        ));
    }

    // ── test_vit_predict_class ────────────────────────────────────────────

    #[test]
    fn test_vit_predict_class() {
        let logits = vec![0.1f32, 0.9, 0.5];
        assert_eq!(predict_class(&logits), Some(1));
    }

    // ── test_vit_mean_pool ────────────────────────────────────────────────

    #[test]
    fn test_vit_mean_pool() {
        // seq_len=2, hidden=3
        let features = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
        let pooled = mean_pool(&features, 3).expect("mean_pool");
        assert_eq!(pooled.len(), 3);
        assert!(
            (pooled[0] - 2.0).abs() < 1e-5,
            "mean[0] should be 2.0, got {}",
            pooled[0]
        );
        assert!((pooled[1] - 3.0).abs() < 1e-5);
        assert!((pooled[2] - 4.0).abs() < 1e-5);
    }

    // ── test_vit_cls_pool ─────────────────────────────────────────────────

    #[test]
    fn test_vit_cls_pool() {
        let features = vec![1.0f32, 2.0, 3.0, 99.0, 99.0, 99.0];
        let cls = cls_pool(&features, 3).expect("cls_pool");
        assert_eq!(cls, vec![1.0, 2.0, 3.0]);
    }

    // ── test_vit_feature_extraction_cls ──────────────────────────────────

    #[test]
    fn test_vit_feature_extraction_cls() {
        let head = ViTForFeatureExtraction::new(4, true).expect("feat ext");
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 9.0, 9.0, 9.0, 9.0];
        let pooled = head.forward(&features).expect("forward");
        assert_eq!(pooled, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── test_vit_feature_extraction_mean ─────────────────────────────────

    #[test]
    fn test_vit_feature_extraction_mean() {
        let head = ViTForFeatureExtraction::new(2, false).expect("feat ext");
        let features = vec![0.0f32, 4.0, 2.0, 2.0]; // seq_len=2
        let pooled = head.forward(&features).expect("forward");
        assert!((pooled[0] - 1.0).abs() < 1e-5);
        assert!((pooled[1] - 3.0).abs() < 1e-5);
    }

    // ── test_vit_classifier_construction ────────────────────────────────

    #[test]
    fn test_vit_classifier_construction() {
        let head = ViTTaskClassifier::new(64, 1000, true);
        assert!(head.is_ok());
        let h = head.expect("classifier");
        assert_eq!(h.num_labels, 1000);
        assert_eq!(h.hidden_size, 64);
    }

    // ── test_vit_classifier_forward ──────────────────────────────────────

    #[test]
    fn test_vit_classifier_forward() {
        let head = ViTTaskClassifier::new(8, 5, true).expect("classifier");
        // seq_len=4, hidden=8 — CLS token is index 0
        let features = vec![0.1f32; 32];
        let logits = head.forward(&features).expect("forward");
        assert_eq!(logits.len(), 5);
    }

    // ── test_vit_multi_label_predict ─────────────────────────────────────

    #[test]
    fn test_vit_multi_label_predict() {
        // High positive logits → sigmoid ≈ 1.0 → all above threshold 0.5
        let logits = vec![5.0f32, -5.0, 5.0];
        let labels = multi_label_predict(&logits, 0.5);
        assert!(labels.contains(&0));
        assert!(!labels.contains(&1)); // sigmoid(-5) ≈ 0.007 < 0.5
        assert!(labels.contains(&2));
    }

    // ── test_vit_attention_bias_shape ────────────────────────────────────

    #[test]
    fn test_vit_attention_bias_shape() {
        let bias = vit_attention_bias(197);
        assert_eq!(bias.len(), 197 * 197);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    // ── test_vit_error_display ────────────────────────────────────────────

    #[test]
    fn test_vit_error_display() {
        let e1 = ViTTaskError::EmptyInput;
        assert!(e1.to_string().contains("empty"));

        let e2 = ViTTaskError::InvalidPatchSize {
            image_size: 225,
            patch_size: 16,
        };
        assert!(e2.to_string().contains("225") && e2.to_string().contains("16"));

        let e3 = ViTTaskError::TopKTooLarge {
            k: 5,
            num_classes: 3,
        };
        assert!(e3.to_string().contains("5") && e3.to_string().contains("3"));

        let e4 = ViTTaskError::InvalidConfig("bad".to_string());
        assert!(e4.to_string().contains("bad"));
    }

    // ── test_vit_lcg_varied_classifiers ──────────────────────────────────

    #[test]
    fn test_vit_lcg_varied_classifiers() {
        let mut state = 61u64;
        for _ in 0..6 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let hidden = ((state % 4) + 2) as usize * 8;
            let num_labels = ((state >> 4) % 5 + 2) as usize;
            let head = ViTTaskClassifier::new(hidden, num_labels, true).expect("cls head");
            let features: Vec<f32> = (0..hidden * 4).map(|i| i as f32 * 0.01).collect();
            let logits = head.forward(&features).expect("forward");
            assert_eq!(logits.len(), num_labels);
        }
    }

    // ── test_vit_sigmoid_bounds ───────────────────────────────────────────

    #[test]
    fn test_vit_sigmoid_bounds() {
        let logits = vec![-100.0f32, 0.0, 100.0];
        let probs = sigmoid_vec(&logits);
        assert!(probs[0] < 0.01, "sigmoid(-100) ≈ 0");
        assert!((probs[1] - 0.5).abs() < 1e-5, "sigmoid(0) = 0.5");
        assert!(probs[2] > 0.99, "sigmoid(100) ≈ 1");
    }
}
