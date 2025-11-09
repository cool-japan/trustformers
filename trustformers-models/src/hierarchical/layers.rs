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
    #[allow(dead_code)]
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
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }
}

/// Single hierarchical layer
pub struct HierarchicalLayer {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
    }
}

/// Nested transformer layer
pub struct NestedTransformerLayer {
    #[allow(dead_code)]
    config: HierarchicalConfig,
    outer_attention: MultiHeadAttention,
    inner_attention: MultiHeadAttention,
    feed_forward: Linear,
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

        let norm_layers = vec![
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
        ];

        Ok(Self {
            config,
            outer_attention,
            inner_attention,
            feed_forward,
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

        Ok(HierarchicalOutput {
            output: hidden_states,
            level_outputs: vec![inner_output, outer_output],
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
}

impl NestedTransformerLayer {
    pub fn parameter_count(&self) -> usize {
        let mut total = self.outer_attention.parameter_count()
            + self.inner_attention.parameter_count()
            + self.feed_forward.parameter_count();

        for norm in &self.norm_layers {
            total += norm.parameter_count();
        }

        total
    }
}
