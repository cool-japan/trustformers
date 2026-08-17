use super::config::HierarchicalConfig;
use super::utils::{
    aggregate_hierarchical_features, build_hierarchy, create_tree_mask, HierarchicalOutput,
};
use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear, MultiHeadAttention},
    tensor::Tensor,
    traits::Layer,
};

/// Hierarchical attention layer
pub struct HierarchicalAttention {
    config: HierarchicalConfig,
    attention_layers: Vec<MultiHeadAttention>,
    norm_layers: Vec<LayerNorm>,
    projection_layers: Vec<Linear>,
}

impl HierarchicalAttention {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let mut attention_layers = Vec::new();
        let mut norm_layers = Vec::new();
        let mut projection_layers = Vec::new();

        for level in 0..config.num_levels {
            let hidden_size = config.get_hidden_size(level);

            attention_layers.push(MultiHeadAttention::new(
                hidden_size,
                config.num_heads,
                config.attention_dropout,
                true, // use_bias
            )?);

            norm_layers.push(LayerNorm::new(vec![hidden_size], config.layer_norm_eps)?);
            projection_layers.push(Linear::new(hidden_size, hidden_size, true));
        }

        Ok(Self {
            config,
            attention_layers,
            norm_layers,
            projection_layers,
        })
    }
}

impl Layer for HierarchicalAttention {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let _seq_len = input.shape()[1];
        let target_shape = input.shape();

        // Build hierarchical representation
        let hierarchy = build_hierarchy(
            input,
            self.config.num_levels,
            self.config.reduction_factor,
            self.config.reduction_method.clone(),
        )?;

        let mut level_outputs = Vec::new();

        // Process each level
        for (level, level_input) in hierarchy.iter().enumerate() {
            let normed_input = self.norm_layers[level].forward(level_input.clone())?;
            let attn_output = self.attention_layers[level].forward(normed_input)?;
            let projected = self.projection_layers[level].forward(attn_output)?;
            level_outputs.push(projected);
        }

        // Aggregate outputs
        let output = aggregate_hierarchical_features(
            level_outputs.clone(),
            &self.config.aggregation_method,
            &target_shape,
        )?;

        Ok(HierarchicalOutput {
            output,
            level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl HierarchicalAttention {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;
        for layer in &self.attention_layers {
            total += layer.parameter_count();
        }
        for norm in &self.norm_layers {
            total += norm.parameter_count();
        }
        for proj in &self.projection_layers {
            total += proj.parameter_count();
        }
        total
    }
}

/// Hierarchical encoder layer
pub struct HierarchicalEncoder {
    config: HierarchicalConfig,
    layers: Vec<HierarchicalLayer>,
}

impl HierarchicalEncoder {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let mut layers = Vec::new();

        for _ in 0..config.num_layers_per_level {
            layers.push(HierarchicalLayer::new(config.clone())?);
        }

        Ok(Self { config, layers })
    }
}

impl Layer for HierarchicalEncoder {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = input;
        let mut all_level_outputs = Vec::new();

        for layer in &self.layers {
            let output = layer.forward(hidden_states)?;
            hidden_states = output.output;
            all_level_outputs.extend(output.level_outputs);
        }

        Ok(HierarchicalOutput {
            output: hidden_states,
            level_outputs: all_level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl HierarchicalEncoder {
    /// Configuration this encoder was built from.
    pub fn config(&self) -> &HierarchicalConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }
}

/// Single hierarchical layer
pub struct HierarchicalLayer {
    config: HierarchicalConfig,
    hierarchical_attention: HierarchicalAttention,
    feed_forward: HierarchicalFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl HierarchicalLayer {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let hierarchical_attention = HierarchicalAttention::new(config.clone())?;
        let feed_forward = HierarchicalFeedForward::new(config.clone())?;
        let norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            config,
            hierarchical_attention,
            feed_forward,
            norm1,
            norm2,
        })
    }
}

impl Layer for HierarchicalLayer {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let residual = input.clone();

        // Hierarchical attention
        let normed_input = self.norm1.forward(input)?;
        let attn_output = self.hierarchical_attention.forward(normed_input)?;
        let hidden_states = residual.add(&attn_output.output)?;

        let residual = hidden_states.clone();

        // Feed forward
        let normed_input = self.norm2.forward(hidden_states)?;
        let ff_output = self.feed_forward.forward(normed_input)?;
        let hidden_states = residual.add(&ff_output.output)?;

        Ok(HierarchicalOutput {
            output: hidden_states,
            level_outputs: attn_output.level_outputs,
            attention_weights: attn_output.attention_weights,
            hierarchical_positions: attn_output.hierarchical_positions,
        })
    }
}

impl HierarchicalLayer {
    /// Configuration this layer was built from.
    pub fn config(&self) -> &HierarchicalConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        self.hierarchical_attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.norm1.parameter_count()
            + self.norm2.parameter_count()
    }
}

/// Hierarchical feed-forward network
pub struct HierarchicalFeedForward {
    config: HierarchicalConfig,
    level_layers: Vec<Vec<Linear>>,
}

impl HierarchicalFeedForward {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let mut level_layers = Vec::new();

        for level in 0..config.num_levels {
            let hidden_size = config.get_hidden_size(level);
            let intermediate_size = config.intermediate_size;

            let mut layers = Vec::new();
            layers.push(Linear::new(hidden_size, intermediate_size, true));
            layers.push(Linear::new(intermediate_size, hidden_size, true));

            level_layers.push(layers);
        }

        Ok(Self {
            config,
            level_layers,
        })
    }
}

impl Layer for HierarchicalFeedForward {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Build hierarchy
        let hierarchy = build_hierarchy(
            input.clone(),
            self.config.num_levels,
            self.config.reduction_factor,
            self.config.reduction_method.clone(),
        )?;

        let mut level_outputs = Vec::new();

        // Process each level
        for (level, level_input) in hierarchy.iter().enumerate() {
            let intermediate = self.level_layers[level][0].forward(level_input.clone())?;
            let activated = intermediate.gelu()?;
            let output = self.level_layers[level][1].forward(activated)?;
            level_outputs.push(output);
        }

        // Aggregate outputs
        let target_shape = input.shape();
        let output = aggregate_hierarchical_features(
            level_outputs.clone(),
            &self.config.aggregation_method,
            &target_shape,
        )?;

        Ok(HierarchicalOutput {
            output,
            level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl HierarchicalFeedForward {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;
        for level_layers in &self.level_layers {
            for layer in level_layers {
                total += layer.parameter_count();
            }
        }
        total
    }
}

/// Pyramid layer for pyramid transformers
pub struct PyramidLayer {
    config: HierarchicalConfig,
    down_layers: Vec<Linear>,
    up_layers: Vec<Linear>,
    skip_connections: Vec<Linear>,
}

impl PyramidLayer {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let mut down_layers = Vec::new();
        let mut up_layers = Vec::new();
        let mut skip_connections = Vec::new();

        for level in 0..config.num_levels - 1 {
            let curr_size = config.get_hidden_size(level);
            let next_size = config.get_hidden_size(level + 1);

            down_layers.push(Linear::new(curr_size, next_size, true));
            up_layers.push(Linear::new(next_size, curr_size, true));

            if config.pyramid_config.as_ref().is_some_and(|c| c.skip_connections) {
                skip_connections.push(Linear::new(curr_size, curr_size, true));
            }
        }

        Ok(Self {
            config,
            down_layers,
            up_layers,
            skip_connections,
        })
    }
}

impl Layer for PyramidLayer {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut level_outputs = Vec::new();
        let mut current = input.clone();
        let mut skip_features = Vec::new();

        // Downward pass
        for level in 0..self.config.num_levels - 1 {
            if !self.skip_connections.is_empty() {
                skip_features.push(self.skip_connections[level].forward(current.clone())?);
            }

            current = self.down_layers[level].forward(current)?;
            level_outputs.push(current.clone());
        }

        // Upward pass
        for level in (0..self.config.num_levels - 1).rev() {
            current = self.up_layers[level].forward(current)?;

            if !skip_features.is_empty() && level < skip_features.len() {
                current = current.add(&skip_features[level])?;
            }
        }

        Ok(HierarchicalOutput {
            output: current,
            level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl PyramidLayer {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;
        for layer in &self.down_layers {
            total += layer.parameter_count();
        }
        for layer in &self.up_layers {
            total += layer.parameter_count();
        }
        for layer in &self.skip_connections {
            total += layer.parameter_count();
        }
        total
    }
}

/// Tree attention layer
pub struct TreeAttention {
    config: HierarchicalConfig,
    attention: MultiHeadAttention,
    tree_mask: Tensor,
}

impl TreeAttention {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_heads,
            config.attention_dropout,
            true, // use_bias
        )?;

        let tree_mask = if let Some(tree_config) = &config.tree_config {
            create_tree_mask(
                config.max_seq_lengths[0],
                tree_config.branching_factor,
                &tree_config.tree_construction,
            )?
        } else {
            Tensor::zeros(&[config.max_seq_lengths[0], config.max_seq_lengths[0]])?
        };

        Ok(Self {
            config,
            attention,
            tree_mask,
        })
    }
}

impl Layer for TreeAttention {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let _seq_len = input.shape()[1];

        // Apply tree-structured attention with tree mask
        let masked_output =
            self.attention.forward_self_attention(&input, Some(&self.tree_mask), false)?;

        Ok(HierarchicalOutput {
            output: masked_output,
            level_outputs: vec![],
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl TreeAttention {
    /// Configuration this layer was built from.
    pub fn config(&self) -> &HierarchicalConfig {
        &self.config
    }

    /// The tree-structured attention mask applied by this layer.
    pub fn tree_mask(&self) -> &Tensor {
        &self.tree_mask
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
    }
}

/// Nested transformer layer
///
/// Applies inner attention, outer attention and a position-wise feed-forward
/// network, each pre-normed with its own [`LayerNorm`] and wrapped in a residual
/// connection.
pub struct NestedTransformerLayer {
    config: HierarchicalConfig,
    outer_attention: MultiHeadAttention,
    inner_attention: MultiHeadAttention,
    /// Up-projection `hidden_size -> intermediate_size` of the feed-forward block.
    feed_forward: Linear,
    /// Down-projection `intermediate_size -> hidden_size`, so the block's output
    /// can be added back onto the residual stream.
    feed_forward_out: Linear,
    /// One pre-norm per sub-block: inner attention, outer attention, feed-forward.
    norm_layers: Vec<LayerNorm>,
}

impl NestedTransformerLayer {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let outer_attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_heads,
            config.attention_dropout,
            true, // use_bias
        )?;

        let inner_attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_heads,
            config.attention_dropout,
            true, // use_bias
        )?;

        let feed_forward = Linear::new(config.hidden_size, config.intermediate_size, true);
        let feed_forward_out = Linear::new(config.intermediate_size, config.hidden_size, true);

        let norm_layers = vec![
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
        ];

        Ok(Self {
            config,
            outer_attention,
            inner_attention,
            feed_forward,
            feed_forward_out,
            norm_layers,
        })
    }
}

impl Layer for NestedTransformerLayer {
    type Input = Tensor;
    type Output = HierarchicalOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let residual = input.clone();

        // Inner attention
        let normed_input = self.norm_layers[0].forward(input)?;
        let inner_output = self.inner_attention.forward(normed_input)?;
        let hidden_states = residual.add(&inner_output)?;

        let residual = hidden_states.clone();

        // Outer attention
        let normed_input = self.norm_layers[1].forward(hidden_states)?;
        let outer_output = self.outer_attention.forward(normed_input)?;
        let hidden_states = residual.add(&outer_output)?;

        let residual = hidden_states.clone();

        // Position-wise feed-forward network: down(gelu(up(x))) + residual.
        let normed_input = self.norm_layers[2].forward(hidden_states)?;
        let intermediate = self.feed_forward.forward(normed_input)?;
        let activated = intermediate.gelu()?;
        let ff_output = self.feed_forward_out.forward(activated)?;
        let hidden_states = residual.add(&ff_output)?;

        Ok(HierarchicalOutput {
            output: hidden_states,
            level_outputs: vec![inner_output, outer_output],
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl NestedTransformerLayer {
    /// Configuration this layer was built from.
    pub fn config(&self) -> &HierarchicalConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let mut total = self.outer_attention.parameter_count()
            + self.inner_attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.feed_forward_out.parameter_count();

        for norm in &self.norm_layers {
            total += norm.parameter_count();
        }

        total
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::hierarchical::config::HierarchicalConfig;

    /// A model small enough to build repeatedly inside a test run.
    fn tiny_config() -> HierarchicalConfig {
        HierarchicalConfig {
            hidden_size: 32,
            num_levels: 2,
            num_heads: 4,
            reduction_factor: 2,
            num_layers_per_level: 1,
            intermediate_size: 64,
            dropout: 0.0,
            attention_dropout: 0.0,
            max_seq_lengths: vec![16, 8],
            ..HierarchicalConfig::default()
        }
    }

    fn ramp_input(seq: usize, hidden: usize) -> Result<Tensor> {
        Tensor::from_vec(
            (0..seq * hidden).map(|i| (i as f32 * 0.05).sin()).collect(),
            &[1, seq, hidden],
        )
    }

    /// The nested layer's feed-forward network must actually be applied.
    ///
    /// A previous revision constructed a single `Linear(hidden -> intermediate)`,
    /// counted it in `parameter_count` and never called it, so the layer was two
    /// stacked attentions with no position-wise network at all. Zeroing the
    /// down-projection is a no-op against that code and changes the output here.
    #[test]
    fn test_nested_layer_applies_its_feed_forward() -> Result<()> {
        let config = tiny_config();
        let mut layer = NestedTransformerLayer::new(config.clone())?;
        let input = ramp_input(4, config.hidden_size)?;

        let with_ffn = layer.forward(input.clone())?.output.data()?;

        // Zeroing the down-projection removes the whole feed-forward contribution.
        layer.feed_forward_out.set_weight(Tensor::zeros(&[
            config.hidden_size,
            config.intermediate_size,
        ])?)?;
        layer.feed_forward_out.set_bias(Tensor::zeros(&[config.hidden_size])?)?;
        let without_ffn = layer.forward(input)?.output.data()?;

        assert_eq!(with_ffn.len(), without_ffn.len());
        assert!(
            with_ffn.iter().zip(without_ffn.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "zeroing the feed-forward down-projection changed nothing — the FFN is unused"
        );
        assert!(with_ffn.iter().all(|v| v.is_finite()));

        Ok(())
    }

    /// `parameter_count` must equal the weights the layer really applies.
    #[test]
    fn test_nested_layer_parameter_count_matches_the_applied_weights() -> Result<()> {
        let config = tiny_config();
        let layer = NestedTransformerLayer::new(config.clone())?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        // up: [intermediate, hidden] + bias, down: [hidden, intermediate] + bias.
        let feed_forward =
            (hidden * intermediate + intermediate) + (intermediate * hidden + hidden);
        // Three pre-norms, each a weight and a bias of width `hidden`.
        let norms = 3 * (2 * hidden);
        let attention =
            layer.inner_attention.parameter_count() + layer.outer_attention.parameter_count();

        assert_eq!(layer.norm_layers.len(), 3);
        assert_eq!(layer.parameter_count(), attention + feed_forward + norms);

        Ok(())
    }

    /// The nested layer keeps the model width and produces a real signal.
    #[test]
    fn test_nested_layer_output_shape_and_signal() -> Result<()> {
        let config = tiny_config();
        let layer = NestedTransformerLayer::new(config.clone())?;
        let out = layer.forward(ramp_input(6, config.hidden_size)?)?;

        assert_eq!(out.output.shape(), vec![1, 6, config.hidden_size]);
        let data = out.output.data()?;
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().any(|v| v.abs() > 1e-6));

        Ok(())
    }
}
