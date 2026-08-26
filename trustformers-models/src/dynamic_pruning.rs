//! Dynamic token pruning for efficient transformer inference.
//!
//! This module implements various dynamic token pruning strategies that can reduce
//! computational costs during inference by selectively processing only the most
//! important tokens. This is particularly useful for long sequences where many
//! tokens may be less relevant to the final output.
//!
//! # Strategies Implemented
//!
//! - **Attention-based pruning**: Prune tokens with low attention scores
//! - **Confidence-based pruning**: Prune tokens with high prediction confidence
//! - **Layer-wise adaptive pruning**: Different pruning rates per layer
//! - **Progressive pruning**: Gradually increase pruning through layers
//! - **Learned pruning**: Use learned gates to determine token importance
//!
//! # Example
//!
//! ```
//! use trustformers_models::dynamic_pruning::{
//!     DynamicPruner, AttentionBasedPruningConfig
//! };
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AttentionBasedPruningConfig {
//!     attention_threshold: 0.1,
//!     min_tokens_ratio: 0.3,
//!     ..Default::default()
//! };
//!
//! let pruner = DynamicPruner::attention_based(config);
//! # let hidden_states = Tensor::randn(&[1, 4, 8])?;
//! # let attention_scores = Tensor::randn(&[1, 2, 4, 4])?;
//! let result = pruner.prune_tokens(&hidden_states, Some(&attention_scores), None, None)?;
//! let pruned_tokens = result.pruned_hidden_states;
//! # let _ = pruned_tokens;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};

/// Numerically stable softmax statistics for a single score vector.
///
/// Returns `(max_probability, normalized_entropy)` where the Shannon entropy
/// `H = -∑ p·ln p` is divided by `ln(n)` so that it always lands in `[0, 1]`
/// (0 = one class dominates, 1 = uniform). For `n <= 1` the entropy is 0 and the
/// probability is 1, which is the mathematically correct degenerate case.
fn softmax_statistics(values: &[f32]) -> (f32, f32) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (1.0, 0.0);
    }

    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return (0.0, 0.0);
    }
    let exps: Vec<f32> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return (0.0, 0.0);
    }

    let mut max_prob = 0.0f32;
    let mut entropy = 0.0f32;
    for &e in &exps {
        let p = e / sum;
        if p > max_prob {
            max_prob = p;
        }
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    let normalized_entropy = (entropy / (n as f32).ln()).clamp(0.0, 1.0);
    (max_prob.clamp(0.0, 1.0), normalized_entropy)
}

/// Read a rank-3 tensor as `(dim0, dim1, dim2, data)` with a validated buffer.
fn tensor_3d_view(tensor: &Tensor, what: &str) -> Result<(usize, usize, usize, Vec<f32>)> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(TrustformersError::invalid_operation(format!(
            "{what} must be rank 3 [batch, seq, features], got shape {shape:?}"
        )));
    }
    let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
    if d0 == 0 || d1 == 0 || d2 == 0 {
        return Err(TrustformersError::invalid_operation(format!(
            "{what} must not have a zero-sized dimension (shape {shape:?})"
        )));
    }
    let data = tensor.data()?;
    if data.len() != d0 * d1 * d2 {
        return Err(TrustformersError::invalid_operation(format!(
            "{what} buffer holds {} values but shape {:?} implies {}",
            data.len(),
            shape,
            d0 * d1 * d2
        )));
    }
    Ok((d0, d1, d2, data))
}

/// Dynamic token pruning strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Prune based on attention scores
    AttentionBased,
    /// Prune based on prediction confidence
    ConfidenceBased,
    /// Learned pruning with trainable gates
    LearnedGates,
    /// Layer-wise adaptive pruning
    LayerAdaptive,
    /// Progressive pruning through layers
    Progressive,
    /// Hybrid approach combining multiple strategies
    Hybrid(Vec<PruningStrategy>),
}

/// Configuration for attention-based pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBasedPruningConfig {
    /// Minimum attention score threshold for keeping tokens
    pub attention_threshold: f32,
    /// Minimum ratio of tokens to keep (0.0 to 1.0)
    pub min_tokens_ratio: f32,
    /// Maximum ratio of tokens to prune (0.0 to 1.0)
    pub max_pruning_ratio: f32,
    /// Use attention variance for adaptive thresholding
    pub use_adaptive_threshold: bool,
    /// Attention head to use for pruning (-1 for average across heads)
    pub attention_head_index: i32,
    /// Number of tokens to always keep (important positions)
    pub keep_top_k: usize,
}

impl Default for AttentionBasedPruningConfig {
    fn default() -> Self {
        Self {
            attention_threshold: 0.1,
            min_tokens_ratio: 0.3,
            max_pruning_ratio: 0.7,
            use_adaptive_threshold: true,
            attention_head_index: -1, // Use average
            keep_top_k: 1,            // Keep CLS token
        }
    }
}

/// Configuration for confidence-based pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBasedPruningConfig {
    /// Confidence threshold for pruning (tokens above this are pruned)
    pub confidence_threshold: f32,
    /// Use entropy as confidence measure
    pub use_entropy: bool,
    /// Minimum tokens to keep
    pub min_tokens_ratio: f32,
    /// Look-ahead window for prediction confidence
    pub lookahead_window: usize,
}

impl Default for ConfidenceBasedPruningConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            use_entropy: true,
            min_tokens_ratio: 0.3,
            lookahead_window: 5,
        }
    }
}

/// Configuration for learned gate pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedGatePruningConfig {
    /// Hidden dimension for gate network
    pub gate_hidden_dim: usize,
    /// Temperature for Gumbel softmax
    pub temperature: f32,
    /// Sparsity regularization weight
    pub sparsity_weight: f32,
    /// Use straight-through estimator
    pub use_straight_through: bool,
}

impl Default for LearnedGatePruningConfig {
    fn default() -> Self {
        Self {
            gate_hidden_dim: 64,
            temperature: 1.0,
            sparsity_weight: 0.01,
            use_straight_through: true,
        }
    }
}

/// Configuration for layer-wise adaptive pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAdaptivePruningConfig {
    /// Pruning ratios per layer
    pub layer_pruning_ratios: Vec<f32>,
    /// Base pruning ratio if not specified per layer
    pub base_pruning_ratio: f32,
    /// Adaptation factor based on layer depth
    pub depth_adaptation_factor: f32,
}

impl Default for LayerAdaptivePruningConfig {
    fn default() -> Self {
        Self {
            layer_pruning_ratios: vec![],
            base_pruning_ratio: 0.3,
            depth_adaptation_factor: 1.1, // Increase pruning in deeper layers
        }
    }
}

/// Configuration for progressive pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressivePruningConfig {
    /// Initial pruning ratio in early layers
    pub initial_pruning_ratio: f32,
    /// Final pruning ratio in late layers
    pub final_pruning_ratio: f32,
    /// Progression schedule
    pub progression_schedule: ProgressionSchedule,
}

/// Progression schedule for pruning through layers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProgressionSchedule {
    Linear,
    Exponential,
    Cosine,
    Custom(Vec<f32>),
}

impl Default for ProgressivePruningConfig {
    fn default() -> Self {
        Self {
            initial_pruning_ratio: 0.1,
            final_pruning_ratio: 0.5,
            progression_schedule: ProgressionSchedule::Linear,
        }
    }
}

/// Token importance scores and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenImportance {
    /// Importance score for each token
    pub importance_scores: Vec<f32>,
    /// Original token indices
    pub token_indices: Vec<usize>,
    /// Pruning decision for each token (true = keep, false = prune)
    pub keep_mask: Vec<bool>,
    /// Reason for pruning/keeping each token
    pub pruning_reasons: Vec<PruningReason>,
}

/// Reason for pruning decision
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PruningReason {
    LowAttention,
    HighConfidence,
    LearnedGate,
    LayerPolicy,
    AlwaysKeep,
    MinimumRatio,
}

/// Result of token pruning operation
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// Pruned hidden states
    pub pruned_hidden_states: Tensor,
    /// Attention mask for pruned tokens
    pub pruned_attention_mask: Tensor,
    /// Token importance information
    pub token_importance: TokenImportance,
    /// Number of tokens before pruning
    pub original_length: usize,
    /// Number of tokens after pruning
    pub pruned_length: usize,
    /// Compression ratio (pruned_length / original_length)
    pub compression_ratio: f32,
}

/// Learned gate network for token pruning
#[derive(Debug, Clone)]
pub struct LearnedGateNetwork {
    /// Linear layer for gate computation
    pub gate_linear: Tensor, // Weight matrix
    pub gate_bias: Tensor, // Bias vector
    /// Projection from the gate hidden state down to a single gate logit.
    ///
    /// This is a *parameter* of the network: it is created once in
    /// [`LearnedGateNetwork::new`] and reused by every forward pass, so repeated
    /// calls on the same input are deterministic.
    pub gate_output_weights: Tensor,
    config: LearnedGatePruningConfig,
}

impl LearnedGateNetwork {
    /// Create new learned gate network
    pub fn new(input_dim: usize, config: LearnedGatePruningConfig) -> Result<Self> {
        // Initialize gate network weights
        let gate_linear = Tensor::randn(&[input_dim, config.gate_hidden_dim])?;
        let gate_bias = Tensor::zeros(&[config.gate_hidden_dim])?;
        let gate_output_weights = Tensor::randn(&[config.gate_hidden_dim, 1])?;

        Ok(Self {
            gate_linear,
            gate_bias,
            gate_output_weights,
            config,
        })
    }

    /// Compute gate probabilities for tokens
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch_size, seq_len, hidden_dim]
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];
        let hidden_dim = hidden_states.shape()[2];

        // Reshape for linear transformation
        let reshaped = hidden_states.reshape(&[batch_size * seq_len, hidden_dim])?;

        // Gate computation: hidden -> gate_hidden -> 1
        let gate_hidden = reshaped.matmul(&self.gate_linear)?.add(&self.gate_bias)?;
        let gate_activated = gate_hidden.tanh()?; // Activation

        // Output gate: gate_hidden -> 1 (binary decision)
        let gate_logits = gate_activated.matmul(&self.gate_output_weights)?;

        // Apply Gumbel softmax for differentiable discrete decisions
        let gate_probs = if self.config.use_straight_through {
            self.gumbel_softmax(&gate_logits)?
        } else {
            gate_logits.sigmoid()?
        };

        // Reshape back to [batch_size, seq_len, 1]
        gate_probs.reshape(&[batch_size, seq_len, 1])
    }

    /// Gumbel softmax for differentiable discrete sampling
    fn gumbel_softmax(&self, logits: &Tensor) -> Result<Tensor> {
        // Add Gumbel noise for sampling
        let gumbel_noise = self.sample_gumbel(logits.shape())?;
        let noisy_logits = logits.add(&gumbel_noise)?;

        // Apply softmax with temperature
        let scaled_logits = noisy_logits.scalar_div(self.config.temperature)?;
        scaled_logits.sigmoid()
    }

    /// Sample from Gumbel distribution
    fn sample_gumbel(&self, shape: Vec<usize>) -> Result<Tensor> {
        // G = -log(-log(U)) where U ~ Uniform(0,1)
        // Use randn and transform to uniform via sigmoid to get [0,1] range
        let normal = Tensor::randn(&shape)?;
        let uniform = normal.sigmoid()?;
        let eps = 1e-7;

        // Clamp uniform to avoid log(0)
        let eps_tensor = Tensor::ones(&shape)?.scalar_mul(eps)?;
        let clamped = uniform.add(&eps_tensor)?;
        let log_uniform = clamped.log()?;
        let neg_log_uniform = log_uniform.scalar_mul(-1.0)?;
        let log_neg_log_uniform = neg_log_uniform.log()?;
        log_neg_log_uniform.scalar_mul(-1.0)
    }
}

/// Main dynamic token pruner
#[derive(Debug, Clone)]
pub struct DynamicPruner {
    strategy: PruningStrategy,
    attention_config: Option<AttentionBasedPruningConfig>,
    confidence_config: Option<ConfidenceBasedPruningConfig>,
    learned_gate_config: Option<LearnedGatePruningConfig>,
    layer_adaptive_config: Option<LayerAdaptivePruningConfig>,
    progressive_config: Option<ProgressivePruningConfig>,
    gate_network: Option<LearnedGateNetwork>,
}

impl DynamicPruner {
    /// Create attention-based pruner
    pub fn attention_based(config: AttentionBasedPruningConfig) -> Self {
        Self {
            strategy: PruningStrategy::AttentionBased,
            attention_config: Some(config),
            confidence_config: None,
            learned_gate_config: None,
            layer_adaptive_config: None,
            progressive_config: None,
            gate_network: None,
        }
    }

    /// Create confidence-based pruner
    pub fn confidence_based(config: ConfidenceBasedPruningConfig) -> Self {
        Self {
            strategy: PruningStrategy::ConfidenceBased,
            attention_config: None,
            confidence_config: Some(config),
            learned_gate_config: None,
            layer_adaptive_config: None,
            progressive_config: None,
            gate_network: None,
        }
    }

    /// Create learned gate pruner
    pub fn learned_gates(input_dim: usize, config: LearnedGatePruningConfig) -> Result<Self> {
        let gate_network = LearnedGateNetwork::new(input_dim, config.clone())?;

        Ok(Self {
            strategy: PruningStrategy::LearnedGates,
            attention_config: None,
            confidence_config: None,
            learned_gate_config: Some(config),
            layer_adaptive_config: None,
            progressive_config: None,
            gate_network: Some(gate_network),
        })
    }

    /// Create layer-adaptive pruner
    pub fn layer_adaptive(config: LayerAdaptivePruningConfig) -> Self {
        Self {
            strategy: PruningStrategy::LayerAdaptive,
            attention_config: None,
            confidence_config: None,
            learned_gate_config: None,
            layer_adaptive_config: Some(config),
            progressive_config: None,
            gate_network: None,
        }
    }

    /// Create progressive pruner
    pub fn progressive(config: ProgressivePruningConfig) -> Self {
        Self {
            strategy: PruningStrategy::Progressive,
            attention_config: None,
            confidence_config: None,
            learned_gate_config: None,
            layer_adaptive_config: None,
            progressive_config: Some(config),
            gate_network: None,
        }
    }

    /// Prune tokens based on the configured strategy.
    ///
    /// Confidence-based pruning falls back to hidden-state statistics; use
    /// [`DynamicPruner::prune_tokens_with_logits`] when the model's per-token
    /// logits are available, so that confidence is the real predictive confidence.
    pub fn prune_tokens(
        &self,
        hidden_states: &Tensor,
        attention_scores: Option<&Tensor>,
        layer_index: Option<usize>,
        total_layers: Option<usize>,
    ) -> Result<PruningResult> {
        self.prune_tokens_with_logits(
            hidden_states,
            attention_scores,
            None,
            layer_index,
            total_layers,
        )
    }

    /// Prune tokens using the configured strategy, optionally informed by the
    /// model's per-token `logits` (shape `[batch, seq, num_classes]`).
    pub fn prune_tokens_with_logits(
        &self,
        hidden_states: &Tensor,
        attention_scores: Option<&Tensor>,
        logits: Option<&Tensor>,
        layer_index: Option<usize>,
        total_layers: Option<usize>,
    ) -> Result<PruningResult> {
        match &self.strategy {
            PruningStrategy::AttentionBased => {
                let config = self.attention_config.as_ref().ok_or_else(|| {
                    TrustformersError::invalid_config(
                        "AttentionBased strategy requires attention_config".to_string(),
                    )
                })?;
                self.attention_based_pruning(hidden_states, attention_scores, config)
            },
            PruningStrategy::ConfidenceBased => {
                let config = self.confidence_config.as_ref().ok_or_else(|| {
                    TrustformersError::invalid_config(
                        "ConfidenceBased strategy requires confidence_config".to_string(),
                    )
                })?;
                self.confidence_based_pruning(hidden_states, logits, config)
            },
            PruningStrategy::LearnedGates => {
                let config = self.learned_gate_config.as_ref().ok_or_else(|| {
                    TrustformersError::invalid_config(
                        "LearnedGates strategy requires learned_gate_config".to_string(),
                    )
                })?;
                self.learned_gate_pruning(hidden_states, config)
            },
            PruningStrategy::LayerAdaptive => {
                let config = self.layer_adaptive_config.as_ref().ok_or_else(|| {
                    TrustformersError::invalid_config(
                        "LayerAdaptive strategy requires layer_adaptive_config".to_string(),
                    )
                })?;
                self.layer_adaptive_pruning(hidden_states, layer_index.unwrap_or(0), config)
            },
            PruningStrategy::Progressive => {
                let config = self.progressive_config.as_ref().ok_or_else(|| {
                    TrustformersError::invalid_config(
                        "Progressive strategy requires progressive_config".to_string(),
                    )
                })?;
                self.progressive_pruning(
                    hidden_states,
                    layer_index.unwrap_or(0),
                    total_layers.unwrap_or(12),
                    config,
                )
            },
            PruningStrategy::Hybrid(strategies) => self.hybrid_pruning(
                hidden_states,
                strategies,
                attention_scores,
                logits,
                layer_index,
                total_layers,
            ),
        }
    }

    /// Attention-based token pruning
    fn attention_based_pruning(
        &self,
        hidden_states: &Tensor,
        attention_scores: Option<&Tensor>,
        config: &AttentionBasedPruningConfig,
    ) -> Result<PruningResult> {
        let attention_scores = attention_scores.ok_or_else(|| {
            TrustformersError::invalid_operation(
                "Attention scores required for attention-based pruning".to_string(),
            )
        })?;

        let seq_len = hidden_states.shape()[1];
        let _hidden_dim = hidden_states.shape()[2];

        // Extract attention scores for the specified head or average across heads
        let attention_weights = if config.attention_head_index >= 0 {
            // Use specific attention head
            let head_idx = config.attention_head_index as usize;
            attention_scores.slice(1, head_idx, head_idx + 1)?
        } else {
            // Average across all attention heads
            // Take mean across heads dimension (dim 1)
            let sum = attention_scores.sum(Some(vec![1]), false)?;
            let num_heads = attention_scores.shape()[1] as f32;
            sum.scalar_div(num_heads)? // Average over head dimension
        };

        // Compute importance scores for each token
        let importance_scores = self.compute_attention_importance(&attention_weights, config)?;

        if importance_scores.len() != seq_len {
            return Err(TrustformersError::invalid_operation(format!(
                "Attention scores describe {} key positions but the hidden states have \
                 sequence length {}",
                importance_scores.len(),
                seq_len
            )));
        }

        // Determine which tokens to keep
        let (keep_mask, pruning_reasons) = self.determine_tokens_to_keep(
            &importance_scores,
            config.min_tokens_ratio,
            config.max_pruning_ratio,
            config.keep_top_k,
        )?;

        // Apply pruning to hidden states
        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    /// Confidence-based token pruning
    fn confidence_based_pruning(
        &self,
        hidden_states: &Tensor,
        logits: Option<&Tensor>,
        config: &ConfidenceBasedPruningConfig,
    ) -> Result<PruningResult> {
        let seq_len = hidden_states.shape()[1];
        let _hidden_dim = hidden_states.shape()[2];

        // Compute confidence scores (simplified - in practice would use model predictions)
        let confidence_scores = self.compute_confidence_scores(hidden_states, logits, config)?;

        // Convert confidence to importance (higher confidence = lower importance for pruning)
        let importance_scores: Vec<f32> = confidence_scores
            .iter()
            .map(|&conf| 1.0 - conf) // Invert confidence
            .collect();

        // Determine which tokens to keep
        let (keep_mask, pruning_reasons) = self.determine_tokens_to_keep(
            &importance_scores,
            config.min_tokens_ratio,
            1.0 - config.min_tokens_ratio,
            1, // Keep at least one token
        )?;

        // Apply pruning
        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    /// Learned gate-based token pruning
    fn learned_gate_pruning(
        &self,
        hidden_states: &Tensor,
        _config: &LearnedGatePruningConfig,
    ) -> Result<PruningResult> {
        let gate_network = self.gate_network.as_ref().ok_or_else(|| {
            TrustformersError::invalid_config(
                "LearnedGates strategy requires gate_network to be initialized".to_string(),
            )
        })?;

        // Compute gate probabilities
        let gate_probs = gate_network.forward(hidden_states)?;

        // Convert probabilities to importance scores
        let seq_len = hidden_states.shape()[1];

        // Extract importance scores (assuming single batch for simplicity)
        let importance_scores = self.extract_gate_scores(&gate_probs)?;

        // Determine which tokens to keep based on gate decisions
        let threshold = 0.5; // Gate threshold
        let keep_mask: Vec<bool> =
            importance_scores.iter().map(|&score| score > threshold).collect();

        let pruning_reasons = vec![PruningReason::LearnedGate; seq_len];

        // Apply pruning
        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    /// Layer-adaptive token pruning
    fn layer_adaptive_pruning(
        &self,
        hidden_states: &Tensor,
        layer_index: usize,
        config: &LayerAdaptivePruningConfig,
    ) -> Result<PruningResult> {
        let seq_len = hidden_states.shape()[1];

        // Determine pruning ratio for this layer
        let pruning_ratio = if layer_index < config.layer_pruning_ratios.len() {
            config.layer_pruning_ratios[layer_index]
        } else {
            // Use base ratio with depth adaptation
            config.base_pruning_ratio * (config.depth_adaptation_factor.powi(layer_index as i32))
        };

        // Simple importance scoring (could be more sophisticated)
        let importance_scores = self.compute_simple_importance(hidden_states)?;

        // Determine tokens to keep based on layer-specific ratio
        let min_tokens_ratio = 1.0 - pruning_ratio.min(0.9); // Keep at least 10%
        let (keep_mask, pruning_reasons) =
            self.determine_tokens_to_keep(&importance_scores, min_tokens_ratio, pruning_ratio, 1)?;

        // Apply pruning
        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    /// Progressive token pruning through layers
    fn progressive_pruning(
        &self,
        hidden_states: &Tensor,
        layer_index: usize,
        total_layers: usize,
        config: &ProgressivePruningConfig,
    ) -> Result<PruningResult> {
        let seq_len = hidden_states.shape()[1];

        // Calculate progressive pruning ratio
        let progress = layer_index as f32 / (total_layers - 1) as f32;
        let pruning_ratio = match &config.progression_schedule {
            ProgressionSchedule::Linear => {
                config.initial_pruning_ratio
                    + (config.final_pruning_ratio - config.initial_pruning_ratio) * progress
            },
            ProgressionSchedule::Exponential => {
                config.initial_pruning_ratio
                    * (config.final_pruning_ratio / config.initial_pruning_ratio).powf(progress)
            },
            ProgressionSchedule::Cosine => {
                config.initial_pruning_ratio
                    + (config.final_pruning_ratio - config.initial_pruning_ratio)
                        * (1.0 - (std::f32::consts::PI * progress).cos())
                        / 2.0
            },
            ProgressionSchedule::Custom(ratios) => {
                if layer_index < ratios.len() {
                    ratios[layer_index]
                } else {
                    config.final_pruning_ratio
                }
            },
        };

        // Compute importance and apply progressive pruning
        let importance_scores = self.compute_simple_importance(hidden_states)?;
        let min_tokens_ratio = 1.0 - pruning_ratio.min(0.9);
        let (keep_mask, pruning_reasons) =
            self.determine_tokens_to_keep(&importance_scores, min_tokens_ratio, pruning_ratio, 1)?;

        // Apply pruning
        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    /// Hybrid pruning combining multiple strategies
    fn hybrid_pruning(
        &self,
        hidden_states: &Tensor,
        strategies: &[PruningStrategy],
        attention_scores: Option<&Tensor>,
        logits: Option<&Tensor>,
        layer_index: Option<usize>,
        total_layers: Option<usize>,
    ) -> Result<PruningResult> {
        // For simplicity, combine strategies by averaging their importance scores
        let seq_len = hidden_states.shape()[1];
        let mut combined_importance = vec![0.0; seq_len];
        let mut valid_strategies = 0;

        for strategy in strategies {
            let temp_pruner = match strategy {
                PruningStrategy::AttentionBased => {
                    if let Some(config) = &self.attention_config {
                        DynamicPruner::attention_based(config.clone())
                    } else {
                        continue;
                    }
                },
                PruningStrategy::ConfidenceBased => {
                    if let Some(config) = &self.confidence_config {
                        DynamicPruner::confidence_based(config.clone())
                    } else {
                        continue;
                    }
                },
                _ => continue, // Skip complex strategies for now
            };

            if let Ok(result) = temp_pruner.prune_tokens_with_logits(
                hidden_states,
                attention_scores,
                logits,
                layer_index,
                total_layers,
            ) {
                for (i, &score) in result.token_importance.importance_scores.iter().enumerate() {
                    combined_importance[i] += score;
                }
                valid_strategies += 1;
            }
        }

        // Average the importance scores
        if valid_strategies > 0 {
            for score in &mut combined_importance {
                *score /= valid_strategies as f32;
            }
        }

        // Apply combined decision
        let (keep_mask, pruning_reasons) = self.determine_tokens_to_keep(
            &combined_importance,
            0.3, // Default min ratio
            0.7, // Default max pruning
            1,   // Keep top 1
        )?;

        let pruned_hidden_states = self.apply_pruning_mask(hidden_states, &keep_mask)?;
        let pruned_attention_mask =
            self.create_attention_mask(&keep_mask, hidden_states.shape()[0])?;

        let original_length = seq_len;
        let pruned_length = keep_mask.iter().filter(|&&x| x).count();
        let compression_ratio = pruned_length as f32 / original_length as f32;

        Ok(PruningResult {
            pruned_hidden_states,
            pruned_attention_mask,
            token_importance: TokenImportance {
                importance_scores: combined_importance,
                token_indices: (0..seq_len).collect(),
                keep_mask,
                pruning_reasons,
            },
            original_length,
            pruned_length,
            compression_ratio,
        })
    }

    // Helper methods

    /// Compute per-token importance from **real** attention weights.
    ///
    /// The importance of token `j` is the attention it *receives*, i.e. the
    /// column sum `∑_i A[i, j]` of the attention matrix, averaged over the batch
    /// and (if still present) the head dimension. This is the standard
    /// "attention-received" saliency used by token-pruning methods such as
    /// PoWER-BERT and LTP.
    ///
    /// Accepted shapes: `[batch, seq_q, seq_k]` or `[batch, heads, seq_q, seq_k]`.
    /// The returned vector has `seq_k` entries.
    fn compute_attention_importance(
        &self,
        attention_weights: &Tensor,
        config: &AttentionBasedPruningConfig,
    ) -> Result<Vec<f32>> {
        // attention_weights shape: [batch_size, seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
        let shape = attention_weights.shape();
        let (batch_size, num_heads, seq_q, seq_k) = match shape.len() {
            3 => (shape[0], 1usize, shape[1], shape[2]),
            4 => (shape[0], shape[1], shape[2], shape[3]),
            other => {
                return Err(TrustformersError::invalid_operation(format!(
                    "Attention weights must be rank 3 [batch, seq, seq] or rank 4 \
                     [batch, heads, seq, seq], got rank {other} (shape {shape:?})"
                )))
            },
        };

        let data = attention_weights.data()?;
        let expected = batch_size * num_heads * seq_q * seq_k;
        if data.len() != expected {
            return Err(TrustformersError::invalid_operation(format!(
                "Attention weight buffer holds {} values but shape {:?} implies {}",
                data.len(),
                shape,
                expected
            )));
        }
        if seq_k == 0 || seq_q == 0 || batch_size == 0 {
            return Err(TrustformersError::invalid_operation(
                "Attention weights must not have a zero-sized dimension".to_string(),
            ));
        }

        // Column-wise sum: how much attention each key position receives.
        let mut importance_scores = vec![0.0f32; seq_k];
        for b in 0..batch_size {
            for h in 0..num_heads {
                let head_offset = (b * num_heads + h) * seq_q * seq_k;
                for q in 0..seq_q {
                    let row_offset = head_offset + q * seq_k;
                    for (k, score) in importance_scores.iter_mut().enumerate() {
                        *score += data[row_offset + k];
                    }
                }
            }
        }

        let averaging_factor = (batch_size * num_heads) as f32;
        for score in &mut importance_scores {
            *score /= averaging_factor;
        }

        let seq_len = seq_k;

        // Apply adaptive thresholding if enabled
        if config.use_adaptive_threshold {
            let mean: f32 = importance_scores.iter().sum::<f32>() / seq_len as f32;
            let variance: f32 =
                importance_scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / seq_len as f32;
            let std_dev = variance.sqrt();

            // Normalize scores using z-score normalization
            for score in &mut importance_scores {
                *score = (*score - mean) / (std_dev + 1e-8);
                // Apply sigmoid to get values between 0 and 1
                *score = 1.0 / (1.0 + (-*score).exp());
            }
        }

        Ok(importance_scores)
    }

    /// Compute per-token confidence from **real** activations.
    ///
    /// * When `logits` is supplied (shape `[batch, seq, num_classes]`) the score is
    ///   the model's real predictive confidence at that position: the maximum
    ///   softmax probability, or `1 - H(p)/ln(C)` when `config.use_entropy` is set.
    /// * When no logits are available the same statistics are computed over the
    ///   token's hidden-state vector. This is an explicit *representation
    ///   uncertainty* proxy — a token whose feature distribution is strongly
    ///   peaked is treated as confidently encoded — and it is computed from the
    ///   real hidden states, never from the token position.
    ///
    /// Scores are averaged over the batch dimension.
    fn compute_confidence_scores(
        &self,
        hidden_states: &Tensor,
        logits: Option<&Tensor>,
        config: &ConfidenceBasedPruningConfig,
    ) -> Result<Vec<f32>> {
        let (source, what) = match logits {
            Some(logits) => (logits, "Confidence logits"),
            None => (hidden_states, "Hidden states"),
        };
        let (batch_size, seq_len, feature_dim, data) = tensor_3d_view(source, what)?;

        if logits.is_some() && seq_len != hidden_states.shape()[1] {
            return Err(TrustformersError::invalid_operation(format!(
                "Logits sequence length {} does not match hidden-state sequence length {}",
                seq_len,
                hidden_states.shape()[1]
            )));
        }

        let mut confidence_scores = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let mut accumulated = 0.0f32;
            for batch in 0..batch_size {
                let offset = (batch * seq_len + token) * feature_dim;
                let slice = &data[offset..offset + feature_dim];
                let (max_prob, normalized_entropy) = softmax_statistics(slice);
                accumulated += if config.use_entropy { 1.0 - normalized_entropy } else { max_prob };
            }
            confidence_scores.push((accumulated / batch_size as f32).clamp(0.0, 1.0));
        }

        // Apply lookahead smoothing if specified
        if config.lookahead_window > 1 {
            let window = config.lookahead_window.min(seq_len);
            let mut smoothed_scores = confidence_scores.clone();

            for i in 0..seq_len {
                let start = i.saturating_sub(window / 2);
                let end = (i + window / 2 + 1).min(seq_len);
                let window_avg: f32 =
                    confidence_scores[start..end].iter().sum::<f32>() / (end - start) as f32;
                smoothed_scores[i] = (confidence_scores[i] + window_avg) / 2.0;
            }

            confidence_scores = smoothed_scores;
        }

        Ok(confidence_scores)
    }

    /// Per-token importance as the **real** L2 norm of the token's hidden state,
    /// averaged over the batch and rescaled so the most important token scores 1.
    ///
    /// The magnitude of a token's representation is the standard cheap saliency
    /// signal used by magnitude-based token pruning.
    fn compute_simple_importance(&self, hidden_states: &Tensor) -> Result<Vec<f32>> {
        let (batch_size, seq_len, hidden_dim, data) =
            tensor_3d_view(hidden_states, "Hidden states")?;

        let mut importance_scores = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let mut norm_sum = 0.0f32;
            for batch in 0..batch_size {
                let offset = (batch * seq_len + token) * hidden_dim;
                let norm_squared: f32 =
                    data[offset..offset + hidden_dim].iter().map(|&v| v * v).sum();
                norm_sum += norm_squared.sqrt();
            }
            importance_scores.push(norm_sum / batch_size as f32);
        }

        // Normalize to [0, 1] range so downstream thresholds are scale-free.
        let max_score = importance_scores.iter().copied().fold(0.0, f32::max);
        if max_score > 0.0 {
            for score in &mut importance_scores {
                *score /= max_score;
            }
        }

        Ok(importance_scores)
    }

    /// Read the **real** gate probabilities produced by [`LearnedGateNetwork`].
    ///
    /// Accepts `[batch, seq, 1]` (the network's output shape) or `[batch, seq, k]`,
    /// in which case the `k` gate outputs are averaged. Scores are averaged over
    /// the batch dimension.
    fn extract_gate_scores(&self, gate_probs: &Tensor) -> Result<Vec<f32>> {
        let (batch_size, seq_len, gate_dim, data) = tensor_3d_view(gate_probs, "Gate outputs")?;

        let mut scores = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let mut accumulated = 0.0f32;
            for batch in 0..batch_size {
                let offset = (batch * seq_len + token) * gate_dim;
                let gate_sum: f32 = data[offset..offset + gate_dim].iter().sum();
                accumulated += gate_sum / gate_dim as f32;
            }
            scores.push(accumulated / batch_size as f32);
        }

        Ok(scores)
    }

    fn determine_tokens_to_keep(
        &self,
        importance_scores: &[f32],
        min_tokens_ratio: f32,
        max_pruning_ratio: f32,
        keep_top_k: usize,
    ) -> Result<(Vec<bool>, Vec<PruningReason>)> {
        let seq_len = importance_scores.len();
        let min_tokens = ((seq_len as f32) * min_tokens_ratio).ceil() as usize;
        let max_tokens_to_prune = ((seq_len as f32) * max_pruning_ratio).floor() as usize;
        let max_tokens_to_keep = seq_len - max_tokens_to_prune;

        // Create indexed scores for sorting
        let mut indexed_scores: Vec<(usize, f32)> =
            importance_scores.iter().enumerate().map(|(i, &score)| (i, score)).collect();

        // Sort by importance (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut keep_mask = vec![false; seq_len];
        let mut pruning_reasons = vec![PruningReason::LowAttention; seq_len];

        // Keep top-k most important tokens
        for i in 0..keep_top_k.min(seq_len) {
            let (idx, _) = indexed_scores[i];
            keep_mask[idx] = true;
            pruning_reasons[idx] = PruningReason::AlwaysKeep;
        }

        // Keep additional tokens up to min_tokens or max_tokens_to_keep
        let tokens_to_keep = min_tokens.clamp(keep_top_k, max_tokens_to_keep);
        for i in keep_top_k..tokens_to_keep.min(seq_len) {
            let (idx, _) = indexed_scores[i];
            keep_mask[idx] = true;
            pruning_reasons[idx] = PruningReason::MinimumRatio;
        }

        Ok((keep_mask, pruning_reasons))
    }

    /// Gather the kept tokens out of `hidden_states` along the sequence dimension.
    ///
    /// This is a real gather: `pruned[b, new_idx, :] = hidden_states[b, orig_idx, :]`
    /// for every position whose `keep_mask` entry is `true`, preserving the original
    /// ordering of the kept tokens.
    fn apply_pruning_mask(&self, hidden_states: &Tensor, keep_mask: &[bool]) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim, data) =
            tensor_3d_view(hidden_states, "Hidden states")?;

        if keep_mask.len() != seq_len {
            return Err(TrustformersError::invalid_operation(format!(
                "Keep mask has {} entries but the hidden states have sequence length {}",
                keep_mask.len(),
                seq_len
            )));
        }

        // Count tokens to keep
        let kept_tokens: Vec<usize> = keep_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        let new_seq_len = kept_tokens.len();

        if new_seq_len == 0 {
            return Err(TrustformersError::invalid_operation(
                "Cannot prune all tokens".to_string(),
            ));
        }

        // Gather the surviving token vectors into a compact buffer.
        let mut pruned = Vec::with_capacity(batch_size * new_seq_len * hidden_dim);
        for batch in 0..batch_size {
            for &orig_idx in &kept_tokens {
                let offset = (batch * seq_len + orig_idx) * hidden_dim;
                pruned.extend_from_slice(&data[offset..offset + hidden_dim]);
            }
        }

        Tensor::from_slice(&pruned, &[batch_size, new_seq_len, hidden_dim])
    }

    /// Build the attention mask that matches a pruned sequence.
    ///
    /// Every surviving token is visible (`1.0`); the mask has one row per batch
    /// element so it can be broadcast against the pruned hidden states.
    fn create_attention_mask(&self, keep_mask: &[bool], batch_size: usize) -> Result<Tensor> {
        let new_seq_len = keep_mask.iter().filter(|&&keep| keep).count();

        if new_seq_len == 0 {
            return Err(TrustformersError::invalid_operation(
                "Cannot build an attention mask for an empty sequence".to_string(),
            ));
        }

        // Create attention mask for kept tokens (all ones)
        Tensor::ones(&[batch_size.max(1), new_seq_len])
    }
}

/// Pruning statistics and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStatistics {
    /// Average compression ratio across all layers
    pub avg_compression_ratio: f32,
    /// Compression ratios per layer
    pub layer_compression_ratios: Vec<f32>,
    /// Total computational savings (estimated)
    pub computational_savings: f32,
    /// Memory savings (estimated)
    pub memory_savings: f32,
    /// Token distribution by pruning reason
    pub pruning_reason_distribution: HashMap<PruningReason, usize>,
}

impl PruningStatistics {
    /// Create new statistics from pruning results
    pub fn from_results(results: &[PruningResult]) -> Self {
        let mut layer_compression_ratios = Vec::new();
        let mut total_compression = 0.0;
        let mut pruning_reason_distribution = HashMap::new();

        for result in results {
            layer_compression_ratios.push(result.compression_ratio);
            total_compression += result.compression_ratio;

            // Count pruning reasons
            for reason in &result.token_importance.pruning_reasons {
                *pruning_reason_distribution.entry(reason.clone()).or_insert(0) += 1;
            }
        }

        let avg_compression_ratio =
            if !results.is_empty() { total_compression / results.len() as f32 } else { 1.0 };

        // Estimate computational and memory savings
        let computational_savings = 1.0 - avg_compression_ratio.powi(2); // Quadratic due to attention
        let memory_savings = 1.0 - avg_compression_ratio;

        Self {
            avg_compression_ratio,
            layer_compression_ratios,
            computational_savings,
            memory_savings,
            pruning_reason_distribution,
        }
    }

    /// Render this report as the human-readable text [`Self::print_report`]
    /// and [`Self::log_report`] both emit.
    fn report_string(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Dynamic Token Pruning Report ===\n");
        report.push_str(&format!(
            "Average Compression Ratio: {:.3}\n",
            self.avg_compression_ratio
        ));
        report.push_str(&format!(
            "Computational Savings: {:.1}%\n",
            self.computational_savings * 100.0
        ));
        report.push_str(&format!(
            "Memory Savings: {:.1}%\n",
            self.memory_savings * 100.0
        ));
        report.push_str("\nLayer-wise Compression:\n");
        for (i, ratio) in self.layer_compression_ratios.iter().enumerate() {
            report.push_str(&format!("  Layer {}: {:.3}\n", i, ratio));
        }
        report.push_str("\nPruning Reason Distribution:");
        for (reason, count) in &self.pruning_reason_distribution {
            report.push_str(&format!("\n  {:?}: {}", reason, count));
        }
        report
    }

    /// Write `Self::report_string` to stdout.
    ///
    /// This is an explicit, caller-initiated escape hatch for binaries and
    /// examples; nothing on the pruning path writes to stdout on its own.
    /// Library callers should prefer [`Self::log_report`], which routes the
    /// same report through `tracing` so the host application controls the
    /// sink.
    pub fn print_report(&self) {
        println!("{}", self.report_string());
    }

    /// Emit `Self::report_string` at `info` level through `tracing`.
    pub fn log_report(&self) {
        tracing::info!("{}", self.report_string());
    }

    /// Get efficiency metrics
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            throughput_improvement: 1.0 / self.avg_compression_ratio,
            latency_reduction: self.computational_savings,
            memory_reduction: self.memory_savings,
            quality_preservation: self.estimate_quality_preservation(),
        }
    }

    fn estimate_quality_preservation(&self) -> f32 {
        // Estimate how much model quality is preserved based on compression ratio
        // Higher compression = lower quality preservation (roughly)
        let base_preservation = self.avg_compression_ratio.powf(0.5);

        // Adjust based on pruning strategy quality
        let strategy_bonus =
            if self.pruning_reason_distribution.contains_key(&PruningReason::LowAttention) {
                0.1 // Attention-based pruning preserves quality better
            } else {
                0.0
            };

        (base_preservation + strategy_bonus).min(1.0)
    }
}

/// Efficiency metrics from pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub throughput_improvement: f32,
    pub latency_reduction: f32,
    pub memory_reduction: f32,
    pub quality_preservation: f32,
}

/// Early exit mechanisms for adaptive computation
/// This complements dynamic pruning by allowing models to exit early when confident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitConfig {
    /// Confidence threshold for early exit
    pub confidence_threshold: f32,
    /// Minimum layer to allow early exit
    pub min_exit_layer: usize,
    /// Maximum number of early exit points
    pub max_exit_points: usize,
    /// Use patience mechanism (wait N layers before exit)
    pub use_patience: bool,
    /// Patience window size
    pub patience_window: usize,
    /// Entropy threshold for uncertainty-based exit
    pub entropy_threshold: f32,
}

impl Default for EarlyExitConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            min_exit_layer: 6, // Don't exit too early
            max_exit_points: 4,
            use_patience: true,
            patience_window: 3,
            entropy_threshold: 0.1,
        }
    }
}

/// Early exit point information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitPoint {
    pub layer_index: usize,
    pub confidence: f32,
    pub entropy: f32,
    pub should_exit: bool,
    pub exit_reason: ExitReason,
}

/// Reason for early exit decision
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExitReason {
    HighConfidence,
    LowEntropy,
    Patience,
    ForcedExit,
    NoExit,
}

/// Early exit mechanism for adaptive computation time
pub struct EarlyExitController {
    config: EarlyExitConfig,
    exit_classifiers: Vec<Linear>,
    patience_counters: HashMap<usize, usize>, // batch_item -> patience_count
}

/// Simple linear classifier for early exit
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<Self> {
        let weight = Tensor::randn(&[input_dim, output_dim])?;
        let bias = if use_bias { Some(Tensor::zeros(&[output_dim])?) } else { None };

        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.matmul(&self.weight)?;
        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}

impl EarlyExitController {
    pub fn new(config: EarlyExitConfig, hidden_dim: usize, num_classes: usize) -> Result<Self> {
        let mut exit_classifiers = Vec::new();

        // Create exit classifiers for each potential exit point
        for _ in 0..config.max_exit_points {
            exit_classifiers.push(Linear::new(hidden_dim, num_classes, true)?);
        }

        Ok(Self {
            config,
            exit_classifiers,
            patience_counters: HashMap::new(),
        })
    }

    /// Determine if model should exit early at given layer
    pub fn should_exit(
        &mut self,
        hidden_states: &Tensor,
        layer_index: usize,
        batch_indices: &[usize],
    ) -> Result<Vec<EarlyExitPoint>> {
        let batch_size = hidden_states.shape()[0];
        let mut exit_points = Vec::new();

        // Don't exit before minimum layer
        if layer_index < self.config.min_exit_layer {
            return Ok(vec![
                EarlyExitPoint {
                    layer_index,
                    confidence: 0.0,
                    entropy: f32::INFINITY,
                    should_exit: false,
                    exit_reason: ExitReason::NoExit,
                };
                batch_size
            ]);
        }

        // Get predictions from exit classifier
        let classifier_idx =
            (layer_index - self.config.min_exit_layer).min(self.exit_classifiers.len() - 1);
        let logits = self.exit_classifiers[classifier_idx].forward(hidden_states)?;

        // Compute confidence and entropy for each batch item
        for i in 0..batch_size {
            let (confidence, entropy) = self.compute_confidence_entropy(&logits, i)?;

            let batch_idx = batch_indices.get(i).copied().unwrap_or(i);
            let patience_count = self.patience_counters.get(&batch_idx).copied().unwrap_or(0);

            let (should_exit, exit_reason) =
                self.make_exit_decision(confidence, entropy, patience_count, layer_index);

            // Update patience counter
            if should_exit {
                self.patience_counters.remove(&batch_idx);
            } else if confidence > self.config.confidence_threshold * 0.8 {
                // Increment patience if we're getting close to threshold
                self.patience_counters.insert(batch_idx, patience_count + 1);
            }

            exit_points.push(EarlyExitPoint {
                layer_index,
                confidence,
                entropy,
                should_exit,
                exit_reason,
            });
        }

        Ok(exit_points)
    }

    /// Compute `(confidence, entropy)` for one batch element from the **real**
    /// exit-classifier logits.
    ///
    /// The logits row for `batch_idx` is softmaxed over the class dimension;
    /// `confidence` is the maximum class probability and `entropy` is the Shannon
    /// entropy `-∑ p·ln p` in nats. Rank-3 logits `[batch, seq, classes]` are mean
    /// pooled over the sequence dimension first, which is the usual way a
    /// sequence-level exit head scores a whole sequence.
    fn compute_confidence_entropy(&self, logits: &Tensor, batch_idx: usize) -> Result<(f32, f32)> {
        let shape = logits.shape();
        let data = logits.data()?;

        let (batch_size, seq_len, num_classes) = match shape.len() {
            2 => (shape[0], 1usize, shape[1]),
            3 => (shape[0], shape[1], shape[2]),
            other => {
                return Err(TrustformersError::invalid_operation(format!(
                    "Exit-classifier logits must be rank 2 [batch, classes] or rank 3 \
                     [batch, seq, classes], got rank {other} (shape {shape:?})"
                )))
            },
        };

        if data.len() != batch_size * seq_len * num_classes {
            return Err(TrustformersError::invalid_operation(format!(
                "Logit buffer holds {} values but shape {:?} implies {}",
                data.len(),
                shape,
                batch_size * seq_len * num_classes
            )));
        }
        if batch_idx >= batch_size {
            return Err(TrustformersError::invalid_operation(format!(
                "Batch index {batch_idx} is out of range for a batch of {batch_size}"
            )));
        }
        if num_classes == 0 {
            return Err(TrustformersError::invalid_operation(
                "Exit-classifier logits must have at least one class".to_string(),
            ));
        }

        // Mean-pool the sequence dimension (a no-op for rank-2 logits).
        let mut pooled = vec![0.0f32; num_classes];
        for position in 0..seq_len {
            let offset = (batch_idx * seq_len + position) * num_classes;
            for (class, value) in pooled.iter_mut().enumerate() {
                *value += data[offset + class];
            }
        }
        for value in &mut pooled {
            *value /= seq_len as f32;
        }

        // Numerically stable softmax over the class dimension.
        let max = pooled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Err(TrustformersError::invalid_operation(
                "Exit-classifier logits contain no finite values".to_string(),
            ));
        }
        let exps: Vec<f32> = pooled.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(TrustformersError::invalid_operation(
                "Softmax of the exit-classifier logits is degenerate".to_string(),
            ));
        }

        let mut confidence = 0.0f32;
        let mut entropy = 0.0f32;
        for &e in &exps {
            let p = e / sum;
            if p > confidence {
                confidence = p;
            }
            if p > 1e-12 {
                entropy -= p * p.ln();
            }
        }

        Ok((confidence.clamp(0.0, 1.0), entropy.max(0.0)))
    }

    fn make_exit_decision(
        &self,
        confidence: f32,
        entropy: f32,
        patience_count: usize,
        _layer_index: usize,
    ) -> (bool, ExitReason) {
        // High confidence exit
        if confidence >= self.config.confidence_threshold {
            return (true, ExitReason::HighConfidence);
        }

        // Low entropy exit
        if entropy <= self.config.entropy_threshold {
            return (true, ExitReason::LowEntropy);
        }

        // Patience-based exit
        if self.config.use_patience && patience_count >= self.config.patience_window {
            return (true, ExitReason::Patience);
        }

        // No exit
        (false, ExitReason::NoExit)
    }

    /// Reset patience counters (e.g., for new sequences)
    pub fn reset_patience(&mut self) {
        self.patience_counters.clear();
    }

    /// Get exit statistics
    pub fn get_exit_statistics(&self, exit_history: &[Vec<EarlyExitPoint>]) -> EarlyExitStatistics {
        let mut total_exits = 0;
        let mut layer_exit_counts = HashMap::new();
        let mut reason_counts = HashMap::new();
        let mut total_samples = 0;

        for layer_exits in exit_history {
            for exit_point in layer_exits {
                total_samples += 1;
                if exit_point.should_exit {
                    total_exits += 1;
                    *layer_exit_counts.entry(exit_point.layer_index).or_insert(0) += 1;
                    *reason_counts.entry(exit_point.exit_reason.clone()).or_insert(0) += 1;
                }
            }
        }

        let exit_rate =
            if total_samples > 0 { total_exits as f32 / total_samples as f32 } else { 0.0 };

        let avg_exit_layer = if total_exits > 0 {
            layer_exit_counts
                .iter()
                .map(|(&layer, &count)| layer as f32 * count as f32)
                .sum::<f32>()
                / total_exits as f32
        } else {
            0.0
        };

        // The number of layers actually observed in this history is the honest
        // denominator for "how much of the network did we skip".
        let observed_layers = exit_history.len();

        EarlyExitStatistics {
            exit_rate,
            avg_exit_layer,
            layer_exit_counts,
            reason_counts,
            computational_savings: Self::estimate_computational_savings(
                exit_rate,
                avg_exit_layer,
                observed_layers,
            ),
        }
    }

    /// Fraction of layer executions saved by early exits, relative to the number
    /// of layers actually observed in the recorded history.
    fn estimate_computational_savings(
        exit_rate: f32,
        avg_exit_layer: f32,
        observed_layers: usize,
    ) -> f32 {
        if observed_layers == 0 {
            return 0.0;
        }
        let total_layers = observed_layers as f32;
        let layers_saved = (total_layers - avg_exit_layer).max(0.0);
        let savings_per_exit = layers_saved / total_layers;
        (exit_rate * savings_per_exit).clamp(0.0, 1.0)
    }
}

/// Statistics for early exit behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitStatistics {
    pub exit_rate: f32,
    pub avg_exit_layer: f32,
    pub layer_exit_counts: HashMap<usize, usize>,
    pub reason_counts: HashMap<ExitReason, usize>,
    pub computational_savings: f32,
}

/// Combined pruning and early exit controller
pub struct AdaptiveComputationController {
    pruner: DynamicPruner,
    early_exit: EarlyExitController,
    adaptive_config: AdaptiveComputationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveComputationConfig {
    /// Enable both pruning and early exit
    pub use_both_strategies: bool,
    /// Prioritize early exit over pruning
    pub prioritize_early_exit: bool,
    /// Computational budget (0.0 to 1.0)
    pub computation_budget: f32,
    /// Quality threshold to maintain
    pub quality_threshold: f32,
}

impl Default for AdaptiveComputationConfig {
    fn default() -> Self {
        Self {
            use_both_strategies: true,
            prioritize_early_exit: false,
            computation_budget: 0.5,
            quality_threshold: 0.9,
        }
    }
}

impl AdaptiveComputationController {
    pub fn new(
        pruner: DynamicPruner,
        early_exit: EarlyExitController,
        config: AdaptiveComputationConfig,
    ) -> Self {
        Self {
            pruner,
            early_exit,
            adaptive_config: config,
        }
    }

    /// Make adaptive computation decisions
    pub fn adaptive_forward(
        &mut self,
        hidden_states: &Tensor,
        attention_scores: Option<&Tensor>,
        layer_index: usize,
        total_layers: usize,
        batch_indices: &[usize],
    ) -> Result<AdaptiveComputationResult> {
        // Check for early exit first if prioritized
        let exit_points = if self.adaptive_config.prioritize_early_exit {
            Some(self.early_exit.should_exit(hidden_states, layer_index, batch_indices)?)
        } else {
            None
        };

        // Apply pruning if not exiting
        let pruning_result = if exit_points
            .as_ref()
            .map(|eps| eps.iter().any(|ep| ep.should_exit))
            .unwrap_or(false)
        {
            None // Skip pruning if exiting
        } else {
            Some(self.pruner.prune_tokens(
                hidden_states,
                attention_scores,
                Some(layer_index),
                Some(total_layers),
            )?)
        };

        // Check for early exit if not already done. Errors are propagated rather
        // than swallowed: a failed exit head must not look like "do not exit".
        let exit_points = match exit_points {
            Some(points) => points,
            None => match pruning_result {
                Some(ref pruned) => self.early_exit.should_exit(
                    &pruned.pruned_hidden_states,
                    layer_index,
                    batch_indices,
                )?,
                None => self.early_exit.should_exit(hidden_states, layer_index, batch_indices)?,
            },
        };

        let should_continue = !exit_points.iter().any(|ep| ep.should_exit);

        Ok(AdaptiveComputationResult {
            pruning_result,
            exit_points,
            should_continue,
            computation_used: self.estimate_computation_used(layer_index, total_layers),
        })
    }

    fn estimate_computation_used(&self, layer_index: usize, total_layers: usize) -> f32 {
        (layer_index + 1) as f32 / total_layers as f32
    }
}

/// Result of adaptive computation decision
#[derive(Debug)]
pub struct AdaptiveComputationResult {
    pub pruning_result: Option<PruningResult>,
    pub exit_points: Vec<EarlyExitPoint>,
    pub should_continue: bool,
    pub computation_used: f32,
}

#[cfg(test)]
#[path = "dynamic_pruning_tests.rs"]
mod tests;
