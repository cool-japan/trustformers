//! Low-Rank Adaptation (LoRA) for efficient fine-tuning.
//!
//! LoRA adds trainable rank decomposition matrices to frozen pretrained weights.
//! For a weight matrix W ∈ R^{d×k}, LoRA learns:
//!   ΔW = B × A,  where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
//!
//! During forward pass: h = W·x + (B·A·x) × (alpha/rank)
//!
//! # Example
//! ```rust,ignore
//! use trustformers::finetuning::{LoraConfig, LoraLinear, LoraBias};
//! let config = LoraConfig::builder()
//!     .rank(8)
//!     .alpha(16.0)
//!     .dropout(0.1)
//!     .target_modules(vec!["query", "value"])
//!     .build()
//!     .expect("valid config");
//! assert_eq!(config.scale(), 2.0);
//! ```

use crate::error::{Result, TrustformersError};
use tracing::debug;
use trustformers_core::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// LoraBias
// ─────────────────────────────────────────────────────────────────────────────

/// Controls which bias parameters are updated during LoRA fine-tuning.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LoraBias {
    /// Do not train any bias parameters (recommended default).
    #[default]
    None,
    /// Train all bias parameters.
    All,
    /// Train only the bias parameters of LoRA layers.
    LoraOnly,
}

impl std::fmt::Display for LoraBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoraBias::None => write!(f, "none"),
            LoraBias::All => write!(f, "all"),
            LoraBias::LoraOnly => write!(f, "lora_only"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoraConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for LoRA fine-tuning.
///
/// # Example
/// ```rust,ignore
/// use trustformers::finetuning::LoraConfig;
/// let config = LoraConfig::new(8, 16.0);
/// assert_eq!(config.scale(), 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the decomposition matrices (r).
    pub rank: usize,
    /// Scaling factor alpha (LoRA scale = alpha / rank).
    pub alpha: f32,
    /// Dropout probability applied to the LoRA output path (must be in [0, 1)).
    pub dropout: f32,
    /// Module name substrings to apply LoRA to (matched by substring containment).
    pub target_modules: Vec<String>,
    /// Whether to merge LoRA weights into the base weight after training.
    pub merge_weights: bool,
    /// Bias configuration: controls which bias params receive gradient updates.
    pub bias: LoraBias,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["query".to_string(), "value".to_string()],
            merge_weights: false,
            bias: LoraBias::None,
        }
    }
}

impl LoraConfig {
    /// Create a minimal LoRA config with the given rank and alpha.
    ///
    /// # Example
    /// ```rust,ignore
    /// use trustformers::finetuning::LoraConfig;
    /// let cfg = LoraConfig::new(4, 8.0);
    /// assert_eq!(cfg.rank, 4);
    /// assert_eq!(cfg.alpha, 8.0);
    /// ```
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    /// Return a builder for fluent construction.
    pub fn builder() -> LoraConfigBuilder {
        LoraConfigBuilder::default()
    }

    /// Compute the LoRA scaling factor: `alpha / rank`.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Validate the configuration, returning an error if any constraint is violated.
    ///
    /// # Errors
    /// Returns `TrustformersError::InvalidInput` if rank is zero, alpha is non-positive,
    /// or dropout is out of `[0, 1)`.
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(TrustformersError::InvalidInput {
                message: "LoRA rank must be > 0".to_string(),
                parameter: Some("rank".to_string()),
                expected: Some("rank >= 1".to_string()),
                received: Some(self.rank.to_string()),
                suggestion: Some("Use rank = 8 as a common starting point".to_string()),
            });
        }
        if self.alpha <= 0.0 {
            return Err(TrustformersError::InvalidInput {
                message: "LoRA alpha must be > 0".to_string(),
                parameter: Some("alpha".to_string()),
                expected: Some("alpha > 0".to_string()),
                received: Some(self.alpha.to_string()),
                suggestion: Some("Use alpha = 16.0 as a common starting point".to_string()),
            });
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(TrustformersError::InvalidInput {
                message: "LoRA dropout must be in [0, 1)".to_string(),
                parameter: Some("dropout".to_string()),
                expected: Some("[0.0, 1.0)".to_string()),
                received: Some(self.dropout.to_string()),
                suggestion: Some("Use dropout = 0.0 to disable or 0.1 for regularization".to_string()),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoraConfigBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`LoraConfig`].
///
/// # Example
/// ```rust,ignore
/// use trustformers::finetuning::{LoraConfig, LoraBias};
/// let cfg = LoraConfig::builder()
///     .rank(16)
///     .alpha(32.0)
///     .dropout(0.05)
///     .target_modules(vec!["q_proj", "v_proj"])
///     .bias(LoraBias::None)
///     .build()
///     .unwrap();
/// assert_eq!(cfg.rank, 16);
/// ```
#[derive(Debug, Default)]
pub struct LoraConfigBuilder {
    rank: Option<usize>,
    alpha: Option<f32>,
    dropout: Option<f32>,
    target_modules: Option<Vec<String>>,
    merge_weights: Option<bool>,
    bias: Option<LoraBias>,
}

impl LoraConfigBuilder {
    /// Set the rank of the low-rank decomposition.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Set the alpha scaling factor.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Set the dropout probability for the LoRA path.
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = Some(dropout);
        self
    }

    /// Set the target module name substrings.
    pub fn target_modules(mut self, modules: Vec<impl Into<String>>) -> Self {
        self.target_modules = Some(modules.into_iter().map(Into::into).collect());
        self
    }

    /// Set whether to merge weights after training.
    pub fn merge_weights(mut self, merge: bool) -> Self {
        self.merge_weights = Some(merge);
        self
    }

    /// Set the bias training mode.
    pub fn bias(mut self, bias: LoraBias) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Build the [`LoraConfig`], validating all parameters.
    ///
    /// # Errors
    /// Returns an error if rank is zero, alpha is non-positive, or dropout is outside `[0, 1)`.
    pub fn build(self) -> Result<LoraConfig> {
        let config = LoraConfig {
            rank: self.rank.unwrap_or(8),
            alpha: self.alpha.unwrap_or(16.0),
            dropout: self.dropout.unwrap_or(0.0),
            target_modules: self
                .target_modules
                .unwrap_or_else(|| vec!["query".to_string(), "value".to_string()]),
            merge_weights: self.merge_weights.unwrap_or(false),
            bias: self.bias.unwrap_or(LoraBias::None),
        };
        config.validate()?;
        Ok(config)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoraLinear
// ─────────────────────────────────────────────────────────────────────────────

/// A LoRA-adapted linear layer.
///
/// Wraps an existing weight matrix W with trainable low-rank matrices A and B:
///   - A ∈ R^{rank × in_features}  (initialised with Kaiming-uniform / Gaussian noise)
///   - B ∈ R^{out_features × rank} (initialised to zeros)
///
/// Forward: `h = W·x + scale × B·(A·x)`
///
/// # Example
/// ```rust,ignore
/// use trustformers::finetuning::{LoraConfig, LoraLinear};
/// let config = LoraConfig::new(4, 8.0);
/// let layer = LoraLinear::new(64, 128, 4, &config).unwrap();
/// assert_eq!(layer.in_features, 64);
/// assert_eq!(layer.out_features, 128);
/// ```
#[derive(Debug)]
pub struct LoraLinear {
    /// Frozen base weight W ∈ R^{out_features × in_features}.
    pub base_weight: Tensor,
    /// Optional frozen base bias.
    pub base_bias: Option<Tensor>,
    /// LoRA A matrix ∈ R^{rank × in_features}.
    pub lora_a: Tensor,
    /// LoRA B matrix ∈ R^{out_features × rank}.
    pub lora_b: Tensor,
    /// Scaling factor: alpha / rank.
    pub scale: f32,
    /// Dropout probability for the LoRA output path.
    pub dropout: f32,
    /// Whether the LoRA weights are currently merged into the base weight.
    pub merged: bool,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
}

impl LoraLinear {
    /// Create a new LoRA adapter for a linear layer.
    ///
    /// The base weight is initialised to zeros (representing an unloaded frozen weight).
    /// `lora_a` is initialised with small Gaussian noise; `lora_b` is initialised to zeros
    /// so that the adapter contributes zero to the output at initialisation.
    ///
    /// # Arguments
    /// * `in_features`  – number of input features
    /// * `out_features` – number of output features
    /// * `rank`         – low-rank dimension r
    /// * `config`       – the full `LoraConfig` (for scale and dropout)
    ///
    /// # Errors
    /// Returns an error if tensor allocation fails or the config is invalid.
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        config: &LoraConfig,
    ) -> Result<Self> {
        config.validate()?;

        if rank > in_features || rank > out_features {
            return Err(TrustformersError::InvalidInput {
                message: format!(
                    "LoRA rank ({rank}) must be <= min(in_features={in_features}, out_features={out_features})"
                ),
                parameter: Some("rank".to_string()),
                expected: Some(format!("<= {}", in_features.min(out_features))),
                received: Some(rank.to_string()),
                suggestion: Some("Choose a smaller rank or larger layer dimensions".to_string()),
            });
        }

        // Base weight: zeros (will be loaded from pretrained checkpoint)
        let base_weight = Tensor::zeros(&[out_features, in_features])
            .map_err(TrustformersError::Core)?;

        // lora_a: small random noise ~ N(0, 0.02) for Kaiming-like initialisation
        let lora_a = {
            let raw = Tensor::randn(&[rank, in_features])
                .map_err(TrustformersError::Core)?;
            // Scale by 1/sqrt(in_features) to approximate Kaiming uniform magnitude
            let scale_factor = (in_features as f32).sqrt().recip();
            raw.mul_scalar(scale_factor)
                .map_err(TrustformersError::Core)?
        };

        // lora_b: zeros so initial ΔW = B·A = 0
        let lora_b = Tensor::zeros(&[out_features, rank])
            .map_err(TrustformersError::Core)?;

        debug!(
            rank = rank,
            in_features = in_features,
            out_features = out_features,
            scale = config.scale(),
            "LoraLinear created"
        );

        Ok(Self {
            base_weight,
            base_bias: None,
            lora_a,
            lora_b,
            scale: config.scale(),
            dropout: config.dropout,
            merged: false,
            in_features,
            out_features,
        })
    }

    /// Forward pass: `h = W·x + scale × B·(A·x)`.
    ///
    /// `input` must have shape `[*, in_features]` where `*` is any batch dimension.
    ///
    /// # Errors
    /// Returns an error if the input shape is incompatible or a matmul fails.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let last_dim = *input_shape
            .last()
            .ok_or_else(|| TrustformersError::InvalidInput {
                message: "Input tensor has no dimensions".to_string(),
                parameter: Some("input".to_string()),
                expected: Some(format!("[*, {}]", self.in_features)),
                received: Some("[]".to_string()),
                suggestion: None,
            })?;

        if last_dim != self.in_features && !self.merged {
            return Err(TrustformersError::InvalidInput {
                message: format!(
                    "Input last dim {last_dim} != in_features {}",
                    self.in_features
                ),
                parameter: Some("input".to_string()),
                expected: Some(format!("[*, {}]", self.in_features)),
                received: Some(format!("{input_shape:?}")),
                suggestion: None,
            });
        }

        // Base linear: W·x  (W is [out, in], x is [in, *] via transpose convention)
        // Flatten to 2D for matmul: [batch, in] × [in, out] = [batch, out]
        let w_t = self
            .base_weight
            .transpose(0, 1)
            .map_err(TrustformersError::Core)?;
        let base_out = input
            .matmul(&w_t)
            .map_err(TrustformersError::Core)?;

        if self.merged {
            // When merged the base weight already contains ΔW
            return Ok(base_out);
        }

        // LoRA path: scale × B·(A·x)
        // A is [rank, in], so A^T is [in, rank]
        let a_t = self
            .lora_a
            .transpose(0, 1)
            .map_err(TrustformersError::Core)?;
        let ax = input
            .matmul(&a_t)
            .map_err(TrustformersError::Core)?;

        // B is [out, rank], so B^T is [rank, out]
        let b_t = self
            .lora_b
            .transpose(0, 1)
            .map_err(TrustformersError::Core)?;
        let bax = ax.matmul(&b_t).map_err(TrustformersError::Core)?;

        let scaled_bax = bax
            .mul_scalar(self.scale)
            .map_err(TrustformersError::Core)?;

        base_out
            .add(&scaled_bax)
            .map_err(TrustformersError::Core)
    }

    /// Merge LoRA weights into the base weight for inference speed-up.
    ///
    /// Computes `W' = W + scale × B·A` and stores the result in `base_weight`.
    /// After merging, `forward()` uses only the base weight (no LoRA path).
    ///
    /// # Errors
    /// Returns an error if already merged or if a tensor operation fails.
    pub fn merge_weights(&mut self) -> Result<()> {
        if self.merged {
            return Err(TrustformersError::InvalidInput {
                message: "LoRA weights are already merged".to_string(),
                parameter: None,
                expected: Some("merged == false".to_string()),
                received: Some("merged == true".to_string()),
                suggestion: Some("Call unmerge_weights() first".to_string()),
            });
        }

        // delta = scale × B · A  — shapes: [out, rank] × [rank, in] = [out, in]
        let ba = self
            .lora_b
            .matmul(&self.lora_a)
            .map_err(TrustformersError::Core)?;
        let delta = ba
            .mul_scalar(self.scale)
            .map_err(TrustformersError::Core)?;

        self.base_weight = self
            .base_weight
            .add(&delta)
            .map_err(TrustformersError::Core)?;
        self.merged = true;

        debug!(scale = self.scale, "LoRA weights merged into base weight");
        Ok(())
    }

    /// Unmerge LoRA weights from the base weight.
    ///
    /// Computes `W = W' - scale × B·A`, reversing a previous `merge_weights()` call.
    ///
    /// # Errors
    /// Returns an error if not currently merged or if a tensor operation fails.
    pub fn unmerge_weights(&mut self) -> Result<()> {
        if !self.merged {
            return Err(TrustformersError::InvalidInput {
                message: "LoRA weights are not merged; nothing to unmerge".to_string(),
                parameter: None,
                expected: Some("merged == true".to_string()),
                received: Some("merged == false".to_string()),
                suggestion: Some("Call merge_weights() first".to_string()),
            });
        }

        let ba = self
            .lora_b
            .matmul(&self.lora_a)
            .map_err(TrustformersError::Core)?;
        let delta = ba
            .mul_scalar(self.scale)
            .map_err(TrustformersError::Core)?;

        self.base_weight = self
            .base_weight
            .sub(&delta)
            .map_err(TrustformersError::Core)?;
        self.merged = false;

        debug!("LoRA weights unmerged from base weight");
        Ok(())
    }

    /// Returns references to the trainable LoRA parameter tensors.
    ///
    /// The base weight and bias are frozen and not included.
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Count the total number of trainable LoRA parameters.
    pub fn num_trainable_params(&self) -> usize {
        // A: rank × in_features  +  B: out_features × rank
        self.lora_a.len() + self.lora_b.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoraConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_lora_config_defaults() {
        let cfg = LoraConfig::default();
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.scale(), 2.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_lora_config_new() {
        let cfg = LoraConfig::new(4, 8.0);
        assert_eq!(cfg.rank, 4);
        assert_eq!(cfg.alpha, 8.0);
        assert_eq!(cfg.scale(), 2.0);
    }

    #[test]
    fn test_lora_config_scale_calculation() {
        let cfg = LoraConfig::new(16, 32.0);
        assert!((cfg.scale() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_config_validate_zero_rank() {
        let cfg = LoraConfig::new(0, 8.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lora_config_validate_negative_alpha() {
        let cfg = LoraConfig::new(4, -1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lora_config_validate_bad_dropout() {
        let mut cfg = LoraConfig::new(4, 8.0);
        cfg.dropout = 1.0;
        assert!(cfg.validate().is_err());

        cfg.dropout = -0.1;
        assert!(cfg.validate().is_err());
    }

    // ── LoraConfigBuilder ─────────────────────────────────────────────────────

    #[test]
    fn test_lora_builder_full() {
        let cfg = LoraConfig::builder()
            .rank(16)
            .alpha(32.0)
            .dropout(0.05)
            .target_modules(vec!["q_proj", "v_proj"])
            .bias(LoraBias::None)
            .build()
            .expect("valid config");

        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha, 32.0);
        assert!((cfg.dropout - 0.05).abs() < 1e-6);
        assert_eq!(cfg.target_modules, vec!["q_proj", "v_proj"]);
        assert_eq!(cfg.bias, LoraBias::None);
    }

    #[test]
    fn test_lora_builder_invalid_rank_fails() {
        let result = LoraConfig::builder().rank(0).build();
        assert!(result.is_err());
    }

    // ── LoraLinear ────────────────────────────────────────────────────────────

    #[test]
    fn test_lora_linear_creation() {
        let config = LoraConfig::new(4, 8.0);
        let layer = LoraLinear::new(32, 64, 4, &config).expect("LoraLinear::new");

        assert_eq!(layer.in_features, 32);
        assert_eq!(layer.out_features, 64);
        assert!((layer.scale - 2.0).abs() < 1e-6);
        assert!(!layer.merged);

        // A: [rank=4, in=32]  B: [out=64, rank=4]
        assert_eq!(layer.lora_a.shape(), vec![4, 32]);
        assert_eq!(layer.lora_b.shape(), vec![64, 4]);
        // Base weight: [out=64, in=32]
        assert_eq!(layer.base_weight.shape(), vec![64, 32]);
    }

    #[test]
    fn test_lora_linear_trainable_parameters() {
        let config = LoraConfig::new(4, 8.0);
        let layer = LoraLinear::new(16, 32, 4, &config).expect("LoraLinear::new");
        let params = layer.trainable_parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_lora_linear_num_trainable_params() {
        let config = LoraConfig::new(4, 8.0);
        let layer = LoraLinear::new(16, 32, 4, &config).expect("LoraLinear::new");
        // A: 4×16 = 64, B: 32×4 = 128 → total = 192
        assert_eq!(layer.num_trainable_params(), 64 + 128);
    }

    #[test]
    fn test_lora_linear_rank_larger_than_features_fails() {
        let config = LoraConfig::new(100, 8.0);
        let result = LoraLinear::new(8, 16, 100, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_merge_unmerge_cycle() {
        let config = LoraConfig::new(4, 8.0);
        let mut layer = LoraLinear::new(8, 8, 4, &config).expect("LoraLinear::new");
        assert!(!layer.merged);

        layer.merge_weights().expect("merge");
        assert!(layer.merged);

        // Merging twice should fail
        assert!(layer.merge_weights().is_err());

        layer.unmerge_weights().expect("unmerge");
        assert!(!layer.merged);

        // Unmerging twice should fail
        assert!(layer.unmerge_weights().is_err());
    }

    #[test]
    fn test_lora_linear_forward_shape() {
        let config = LoraConfig::new(4, 8.0);
        let layer = LoraLinear::new(16, 32, 4, &config).expect("LoraLinear::new");

        // Input: [batch=2, in=16]
        let input = Tensor::zeros(&[2, 16]).expect("zeros");
        let output = layer.forward(&input).expect("forward");

        // Output should be [2, 32]
        assert_eq!(output.shape(), vec![2, 32]);
    }

    #[test]
    fn test_lora_bias_display() {
        assert_eq!(LoraBias::None.to_string(), "none");
        assert_eq!(LoraBias::All.to_string(), "all");
        assert_eq!(LoraBias::LoraOnly.to_string(), "lora_only");
    }
}
