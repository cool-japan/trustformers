//! Bottleneck Adapter for parameter-efficient fine-tuning.
//!
//! Inserts small trainable modules after each transformer sub-layer:
//!
//! ```text
//! x  →  [LayerNorm]  →  Down-project (d → r)  →  Activation  →  Up-project (r → d)  →  + x
//! ```
//!
//! where `r << d` controls the adapter capacity.
//!
//! # Example
//! ```rust,ignore
//! use trustformers::finetuning::{AdapterConfig, AdapterActivation, BottleneckAdapter};
//! let config = AdapterConfig {
//!     hidden_size: 768,
//!     bottleneck_size: 64,
//!     activation: AdapterActivation::Gelu,
//!     ..Default::default()
//! };
//! let adapter = BottleneckAdapter::new(config).unwrap();
//! ```

use crate::error::{Result, TrustformersError};
use tracing::debug;
use trustformers_core::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// AdapterActivation
// ─────────────────────────────────────────────────────────────────────────────

/// Activation function used inside the bottleneck adapter.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AdapterActivation {
    /// Rectified Linear Unit.
    Relu,
    /// Gaussian Error Linear Unit (recommended for BERT-like models).
    #[default]
    Gelu,
    /// Sigmoid-weighted Linear Unit (used in LLaMA/PaLM family).
    Silu,
    /// Hyperbolic tangent.
    Tanh,
}

impl std::fmt::Display for AdapterActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterActivation::Relu => write!(f, "relu"),
            AdapterActivation::Gelu => write!(f, "gelu"),
            AdapterActivation::Silu => write!(f, "silu"),
            AdapterActivation::Tanh => write!(f, "tanh"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdapterConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`BottleneckAdapter`].
///
/// # Example
/// ```rust,ignore
/// use trustformers::finetuning::{AdapterConfig, AdapterActivation};
/// let config = AdapterConfig {
///     hidden_size: 768,
///     bottleneck_size: 64,
///     activation: AdapterActivation::Relu,
///     ..Default::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Hidden dimension (input = output size of the adapter).
    pub hidden_size: usize,
    /// Bottleneck dimension r (should be << hidden_size).
    pub bottleneck_size: usize,
    /// Activation function in the bottleneck.
    pub activation: AdapterActivation,
    /// Dropout probability after the adapter output (before residual add).
    pub dropout: f32,
    /// Whether to prepend a layer normalisation before the adapter.
    pub use_layer_norm: bool,
    /// Initial weight scale for the residual connection (1.0 = standard residual).
    pub residual_scale: f32,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            bottleneck_size: 64,
            activation: AdapterActivation::Gelu,
            dropout: 0.0,
            use_layer_norm: true,
            residual_scale: 1.0,
        }
    }
}

impl AdapterConfig {
    /// Validate configuration constraints.
    ///
    /// # Errors
    /// Returns an error if `bottleneck_size` is zero, `hidden_size` is zero,
    /// dropout is outside `[0, 1)`, or `bottleneck_size >= hidden_size`.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(TrustformersError::InvalidInput {
                message: "AdapterConfig: hidden_size must be > 0".to_string(),
                parameter: Some("hidden_size".to_string()),
                expected: Some("> 0".to_string()),
                received: Some(self.hidden_size.to_string()),
                suggestion: Some("Use 768 for BERT-base".to_string()),
            });
        }
        if self.bottleneck_size == 0 {
            return Err(TrustformersError::InvalidInput {
                message: "AdapterConfig: bottleneck_size must be > 0".to_string(),
                parameter: Some("bottleneck_size".to_string()),
                expected: Some("> 0".to_string()),
                received: Some(self.bottleneck_size.to_string()),
                suggestion: Some("Use 64 as a common starting value".to_string()),
            });
        }
        if self.bottleneck_size >= self.hidden_size {
            return Err(TrustformersError::InvalidInput {
                message: format!(
                    "AdapterConfig: bottleneck_size ({}) must be < hidden_size ({})",
                    self.bottleneck_size, self.hidden_size
                ),
                parameter: Some("bottleneck_size".to_string()),
                expected: Some(format!("< {}", self.hidden_size)),
                received: Some(self.bottleneck_size.to_string()),
                suggestion: Some("The adapter bottleneck should be much smaller than the hidden dimension".to_string()),
            });
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(TrustformersError::InvalidInput {
                message: "AdapterConfig: dropout must be in [0, 1)".to_string(),
                parameter: Some("dropout".to_string()),
                expected: Some("[0.0, 1.0)".to_string()),
                received: Some(self.dropout.to_string()),
                suggestion: None,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BottleneckAdapter
// ─────────────────────────────────────────────────────────────────────────────

/// Bottleneck adapter module for parameter-efficient fine-tuning.
///
/// Implements the Houlsby-style adapter:
/// ```text
/// h = x + residual_scale × up_proj( activation( down_proj( LayerNorm(x) ) ) )
/// ```
///
/// Only `down_proj`, `down_bias`, `up_proj`, `up_bias`, and (if enabled)
/// `layer_norm_weight` / `layer_norm_bias` are trainable.
///
/// # Example
/// ```rust,ignore
/// use trustformers::finetuning::{AdapterConfig, BottleneckAdapter};
/// let config = AdapterConfig { hidden_size: 32, bottleneck_size: 8, ..Default::default() };
/// let adapter = BottleneckAdapter::new(config).unwrap();
/// assert_eq!(adapter.num_trainable_params(), 8*32 + 8 + 32*8 + 32 + 32 + 32);
/// ```
#[derive(Debug)]
pub struct BottleneckAdapter {
    /// Down-projection weight: hidden_size → bottleneck_size.
    pub down_proj: Tensor,
    /// Down-projection bias: shape [bottleneck_size].
    pub down_bias: Tensor,
    /// Up-projection weight: bottleneck_size → hidden_size.
    pub up_proj: Tensor,
    /// Up-projection bias: shape [hidden_size].
    pub up_bias: Tensor,
    /// Layer-norm scale (gamma), shape [hidden_size] — present if `use_layer_norm`.
    pub layer_norm_weight: Option<Tensor>,
    /// Layer-norm bias (beta), shape [hidden_size] — present if `use_layer_norm`.
    pub layer_norm_bias: Option<Tensor>,
    /// The configuration this adapter was built from.
    pub config: AdapterConfig,
}

impl BottleneckAdapter {
    /// Create a new bottleneck adapter.
    ///
    /// Projection weights are initialised with small Gaussian noise;
    /// biases and layer-norm parameters are initialised to zeros/ones respectively.
    ///
    /// # Errors
    /// Returns an error if the config fails validation or tensor allocation fails.
    pub fn new(config: AdapterConfig) -> Result<Self> {
        config.validate()?;

        let h = config.hidden_size;
        let r = config.bottleneck_size;
        let scale = (h as f32).sqrt().recip();

        // Down projection: [bottleneck, hidden]
        let down_proj = {
            let raw = Tensor::randn(&[r, h]).map_err(TrustformersError::Core)?;
            raw.mul_scalar(scale).map_err(TrustformersError::Core)?
        };
        let down_bias = Tensor::zeros(&[r]).map_err(TrustformersError::Core)?;

        // Up projection: [hidden, bottleneck] — initialised to zeros for identity start
        let up_proj = Tensor::zeros(&[h, r]).map_err(TrustformersError::Core)?;
        let up_bias = Tensor::zeros(&[h]).map_err(TrustformersError::Core)?;

        // Optional layer norm: gamma = ones, beta = zeros
        let (layer_norm_weight, layer_norm_bias) = if config.use_layer_norm {
            let gamma = Tensor::ones(&[h]).map_err(TrustformersError::Core)?;
            let beta = Tensor::zeros(&[h]).map_err(TrustformersError::Core)?;
            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        debug!(
            hidden_size = h,
            bottleneck_size = r,
            activation = %config.activation,
            use_layer_norm = config.use_layer_norm,
            "BottleneckAdapter created"
        );

        Ok(Self {
            down_proj,
            down_bias,
            up_proj,
            up_bias,
            layer_norm_weight,
            layer_norm_bias,
            config,
        })
    }

    /// Forward pass: `output = x + residual_scale × adapter(x)`.
    ///
    /// The adapter sub-function is:
    /// `adapter(x) = up_proj( activation( down_proj( LN(x) ) + down_bias ) ) + up_bias`
    ///
    /// `hidden_states` must have shape `[batch, seq_len, hidden_size]` or
    /// `[*, hidden_size]` more generally.
    ///
    /// # Errors
    /// Returns an error if tensor shapes are incompatible or an operation fails.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let last_dim = *shape
            .last()
            .ok_or_else(|| TrustformersError::InvalidInput {
                message: "hidden_states has no dimensions".to_string(),
                parameter: Some("hidden_states".to_string()),
                expected: Some(format!("[*, {}]", self.config.hidden_size)),
                received: Some("[]".to_string()),
                suggestion: None,
            })?;

        if last_dim != self.config.hidden_size {
            return Err(TrustformersError::InvalidInput {
                message: format!(
                    "hidden_states last dim {last_dim} != hidden_size {}",
                    self.config.hidden_size
                ),
                parameter: Some("hidden_states".to_string()),
                expected: Some(format!("[*, {}]", self.config.hidden_size)),
                received: Some(format!("{shape:?}")),
                suggestion: None,
            });
        }

        // Optional layer-norm
        let normed = if self.config.use_layer_norm {
            // Approximate layer norm: normalise over last dimension
            hidden_states
                .layer_norm(-1, 1e-5)
                .map_err(TrustformersError::Core)?
        } else {
            hidden_states.clone()
        };

        // Down projection: [..., hidden] × [hidden, bottleneck] = [..., bottleneck]
        let down_t = self
            .down_proj
            .transpose(0, 1)
            .map_err(TrustformersError::Core)?;
        let down_out = normed.matmul(&down_t).map_err(TrustformersError::Core)?;
        // Add bias (broadcast over batch dims)
        let down_out = down_out
            .add(&self.down_bias)
            .map_err(TrustformersError::Core)?;

        // Activation
        let activated = self.apply_activation(&down_out)?;

        // Up projection: [..., bottleneck] × [bottleneck, hidden] = [..., hidden]
        let up_t = self
            .up_proj
            .transpose(0, 1)
            .map_err(TrustformersError::Core)?;
        let up_out = activated.matmul(&up_t).map_err(TrustformersError::Core)?;
        let up_out = up_out
            .add(&self.up_bias)
            .map_err(TrustformersError::Core)?;

        // Residual connection: x + scale * adapter(x)
        let scaled = up_out
            .mul_scalar(self.config.residual_scale)
            .map_err(TrustformersError::Core)?;
        hidden_states
            .add(&scaled)
            .map_err(TrustformersError::Core)
    }

    /// Apply the configured activation function to a tensor.
    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        match self.config.activation {
            AdapterActivation::Relu => input.relu().map_err(TrustformersError::Core),
            AdapterActivation::Gelu => input.gelu().map_err(TrustformersError::Core),
            AdapterActivation::Silu => input.silu().map_err(TrustformersError::Core),
            AdapterActivation::Tanh => input.tanh().map_err(TrustformersError::Core),
        }
    }

    /// Returns references to all trainable parameter tensors.
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![
            &self.down_proj,
            &self.down_bias,
            &self.up_proj,
            &self.up_bias,
        ];
        if let Some(ref w) = self.layer_norm_weight {
            params.push(w);
        }
        if let Some(ref b) = self.layer_norm_bias {
            params.push(b);
        }
        params
    }

    /// Count the total number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let h = self.config.hidden_size;
        let r = self.config.bottleneck_size;
        // down_proj [r, h] + down_bias [r] + up_proj [h, r] + up_bias [h]
        let core = r * h + r + h * r + h;
        // layer_norm_weight [h] + layer_norm_bias [h]
        let ln = if self.config.use_layer_norm { 2 * h } else { 0 };
        core + ln
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AdapterConfig {
        AdapterConfig {
            hidden_size: 32,
            bottleneck_size: 8,
            activation: AdapterActivation::Relu,
            dropout: 0.0,
            use_layer_norm: false,
            residual_scale: 1.0,
        }
    }

    #[test]
    fn test_adapter_config_default_valid() {
        let cfg = AdapterConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_adapter_config_zero_hidden_fails() {
        let mut cfg = small_config();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_adapter_config_zero_bottleneck_fails() {
        let mut cfg = small_config();
        cfg.bottleneck_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_adapter_config_bottleneck_ge_hidden_fails() {
        let mut cfg = small_config();
        cfg.bottleneck_size = cfg.hidden_size;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_adapter_config_bad_dropout_fails() {
        let mut cfg = small_config();
        cfg.dropout = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_bottleneck_adapter_creation() {
        let adapter = BottleneckAdapter::new(small_config()).expect("create adapter");
        assert_eq!(adapter.down_proj.shape(), vec![8, 32]);
        assert_eq!(adapter.up_proj.shape(), vec![32, 8]);
        assert_eq!(adapter.down_bias.shape(), vec![8]);
        assert_eq!(adapter.up_bias.shape(), vec![32]);
        assert!(adapter.layer_norm_weight.is_none());
    }

    #[test]
    fn test_bottleneck_adapter_with_layer_norm() {
        let mut cfg = small_config();
        cfg.use_layer_norm = true;
        let adapter = BottleneckAdapter::new(cfg).expect("create adapter");
        assert!(adapter.layer_norm_weight.is_some());
        assert!(adapter.layer_norm_bias.is_some());
    }

    #[test]
    fn test_adapter_forward_output_shape() {
        let adapter = BottleneckAdapter::new(small_config()).expect("create adapter");
        let input = Tensor::zeros(&[2, 32]).expect("zeros");
        let output = adapter.forward(&input).expect("forward");
        assert_eq!(output.shape(), vec![2, 32]);
    }

    #[test]
    fn test_adapter_forward_wrong_dim_fails() {
        let adapter = BottleneckAdapter::new(small_config()).expect("create adapter");
        let bad_input = Tensor::zeros(&[2, 16]).expect("zeros");
        assert!(adapter.forward(&bad_input).is_err());
    }

    #[test]
    fn test_adapter_num_trainable_params_no_ln() {
        let cfg = AdapterConfig {
            hidden_size: 32,
            bottleneck_size: 8,
            use_layer_norm: false,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(cfg).expect("create adapter");
        // down [8×32] + down_bias [8] + up [32×8] + up_bias [32]
        let expected = 8 * 32 + 8 + 32 * 8 + 32;
        assert_eq!(adapter.num_trainable_params(), expected);
    }

    #[test]
    fn test_adapter_num_trainable_params_with_ln() {
        let cfg = AdapterConfig {
            hidden_size: 32,
            bottleneck_size: 8,
            use_layer_norm: true,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(cfg).expect("create adapter");
        let expected = 8 * 32 + 8 + 32 * 8 + 32 + 2 * 32;
        assert_eq!(adapter.num_trainable_params(), expected);
    }

    #[test]
    fn test_adapter_trainable_parameters_count_no_ln() {
        let adapter = BottleneckAdapter::new(small_config()).expect("create adapter");
        assert_eq!(adapter.trainable_parameters().len(), 4);
    }

    #[test]
    fn test_adapter_activation_display() {
        assert_eq!(AdapterActivation::Relu.to_string(), "relu");
        assert_eq!(AdapterActivation::Gelu.to_string(), "gelu");
        assert_eq!(AdapterActivation::Silu.to_string(), "silu");
        assert_eq!(AdapterActivation::Tanh.to_string(), "tanh");
    }

    #[test]
    fn test_adapter_gelu_activation() {
        let cfg = AdapterConfig {
            hidden_size: 16,
            bottleneck_size: 4,
            activation: AdapterActivation::Gelu,
            use_layer_norm: false,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(cfg).expect("create adapter");
        let input = Tensor::zeros(&[1, 16]).expect("zeros");
        assert!(adapter.forward(&input).is_ok());
    }
}
