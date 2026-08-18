use crate::bert::config::BertConfig;
use crate::weight_loading::checkpoint::WeightBinder;
use scirs2_core::ndarray::s; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::{tensor_op_error, Result};
use trustformers_core::layers::{FeedForward, LayerNorm, MultiHeadAttention};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

/// Checkpoint sub-paths for one encoder layer.
///
/// BERT and DistilBERT share this encoder implementation but spell their
/// parameters differently (`attention.self.query` versus `attention.q_lin`, and
/// so on). Keeping the spelling in one table means the binder, and the tests that
/// build fixtures for it, are driven by the same source.
#[derive(Debug, Clone)]
pub struct BertLayerNames {
    /// Prefix of the encoder stack, e.g. `encoder.layer.`.
    pub layer_stack_prefix: &'static str,
    /// Query projection, relative to the layer prefix.
    pub query: &'static str,
    /// Key projection.
    pub key: &'static str,
    /// Value projection.
    pub value: &'static str,
    /// Attention output projection.
    pub attention_output: &'static str,
    /// Layer norm applied after the attention residual.
    pub attention_norm: &'static str,
    /// First feed-forward projection (hidden -> intermediate).
    pub intermediate: &'static str,
    /// Second feed-forward projection (intermediate -> hidden).
    pub feed_forward_output: &'static str,
    /// Layer norm applied after the feed-forward residual.
    pub output_norm: &'static str,
}

impl BertLayerNames {
    /// HuggingFace `BertModel` parameter spelling.
    pub const fn bert() -> Self {
        Self {
            layer_stack_prefix: "encoder.layer.",
            query: "attention.self.query",
            key: "attention.self.key",
            value: "attention.self.value",
            attention_output: "attention.output.dense",
            attention_norm: "attention.output.LayerNorm",
            intermediate: "intermediate.dense",
            feed_forward_output: "output.dense",
            output_norm: "output.LayerNorm",
        }
    }

    /// HuggingFace `DistilBertModel` parameter spelling.
    pub const fn distilbert() -> Self {
        Self {
            layer_stack_prefix: "transformer.layer.",
            query: "attention.q_lin",
            key: "attention.k_lin",
            value: "attention.v_lin",
            attention_output: "attention.out_lin",
            attention_norm: "sa_layer_norm",
            intermediate: "ffn.lin1",
            feed_forward_output: "ffn.lin2",
            output_norm: "output_layer_norm",
        }
    }
}

/// Take a `[out, in]` weight and its `[out]` bias from the checkpoint.
///
/// Both HuggingFace `nn.Linear` and this crate's [`trustformers_core::layers::Linear`]
/// store the weight as `[out_features, in_features]`, so no transposition is
/// involved; the shapes are checked instead of assumed.
///
/// Returns `None` for a tensor the checkpoint does not hold — the binder records
/// it, so the caller leaves the parameter untouched rather than substituting one.
fn take_linear(
    binder: &mut WeightBinder<'_>,
    name: &str,
    weight_shape: [usize; 2],
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let weight = binder.take_shaped(&format!("{name}.weight"), &weight_shape)?;
    let bias = binder.take_shaped(&format!("{name}.bias"), &[weight_shape[0]])?;
    Ok((weight, bias))
}

/// Copy a layer-norm weight/bias pair from the checkpoint.
fn bind_layer_norm(
    binder: &mut WeightBinder<'_>,
    name: &str,
    hidden_size: usize,
    norm: &mut LayerNorm,
) -> Result<()> {
    if let Some(weight) = binder.take_shaped(&format!("{name}.weight"), &[hidden_size])? {
        norm.set_weight(weight)?;
    }
    if let Some(bias) = binder.take_shaped(&format!("{name}.bias"), &[hidden_size])? {
        norm.set_bias(bias)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct BertEmbeddings {
    word_embeddings: trustformers_core::layers::Embedding,
    position_embeddings: trustformers_core::layers::Embedding,
    token_type_embeddings: trustformers_core::layers::Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout_prob: f32,
    device: Device,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id as usize),
                device,
            )?,
            position_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.max_position_embeddings,
                config.hidden_size,
                None,
                device,
            )?,
            token_type_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.type_vocab_size,
                config.hidden_size,
                None,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout_prob: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input_ids: Vec<u32>, token_type_ids: Option<Vec<u32>>) -> Result<Tensor> {
        let seq_length = input_ids.len();
        let position_ids: Vec<u32> = (0..seq_length as u32).collect();

        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let mut embeddings = word_embeddings.add(&position_embeddings)?;

        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeddings)?;
        }

        self.layer_norm.forward(embeddings)
    }

    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.token_type_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Copy the embedding tables and their layer norm out of a checkpoint.
    ///
    /// `has_token_type_embeddings` is false for DistilBERT, which has no segment
    /// embedding table; the local table then keeps its zero-equivalent role and
    /// is not requested from the checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when a tensor exists but has the wrong shape. Absent tensors are
    /// recorded on the binder and reported together by
    /// [`WeightBinder::finish`](crate::weight_loading::checkpoint::WeightBinder::finish).
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        config: &BertConfig,
        has_token_type_embeddings: bool,
    ) -> Result<()> {
        let hidden = config.hidden_size;

        if let Some(weight) = binder.take_shaped(
            "embeddings.word_embeddings.weight",
            &[config.vocab_size, hidden],
        )? {
            self.word_embeddings.set_weight(weight)?;
        }
        if let Some(weight) = binder.take_shaped(
            "embeddings.position_embeddings.weight",
            &[config.max_position_embeddings, hidden],
        )? {
            self.position_embeddings.set_weight(weight)?;
        }
        if has_token_type_embeddings {
            if let Some(weight) = binder.take_shaped(
                "embeddings.token_type_embeddings.weight",
                &[config.type_vocab_size, hidden],
            )? {
                self.token_type_embeddings.set_weight(weight)?;
            }
        }
        bind_layer_norm(binder, "embeddings.LayerNorm", hidden, &mut self.layer_norm)
    }

    /// Append the embedding parameters under the names
    /// [`BertEmbeddings::load_weights`] binds.
    ///
    /// `has_token_type_embeddings` must match the value passed to
    /// `load_weights`: DistilBERT has no `token_type_embeddings`, and listing a
    /// table its checkpoints never contain would make a round-trip through
    /// `named_tensors` produce a file that no longer loads.
    pub fn collect_named_parameters<'a>(
        &'a self,
        has_token_type_embeddings: bool,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        self.word_embeddings
            .collect_named_parameters("embeddings.word_embeddings", into);
        self.position_embeddings
            .collect_named_parameters("embeddings.position_embeddings", into);
        if has_token_type_embeddings {
            self.token_type_embeddings
                .collect_named_parameters("embeddings.token_type_embeddings", into);
        }
        self.layer_norm.collect_named_parameters("embeddings.LayerNorm", into);
    }

    /// Mutable counterpart of [`BertEmbeddings::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        has_token_type_embeddings: bool,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        self.word_embeddings
            .collect_named_parameters_mut("embeddings.word_embeddings", into);
        self.position_embeddings
            .collect_named_parameters_mut("embeddings.position_embeddings", into);
        if has_token_type_embeddings {
            self.token_type_embeddings
                .collect_named_parameters_mut("embeddings.token_type_embeddings", into);
        }
        self.layer_norm.collect_named_parameters_mut("embeddings.LayerNorm", into);
    }
}

#[derive(Debug, Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: FeedForward,
    output_layer_norm: LayerNorm,
    device: Device,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::new_with_device(config, device)?,
            intermediate: FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            ),
            output_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.intermediate.parameter_count()
            + self.output_layer_norm.parameter_count()
    }

    /// Copy one encoder layer's parameters out of a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        layer_prefix: &str,
        names: &BertLayerNames,
        config: &BertConfig,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        self.attention.load_weights(binder, layer_prefix, names, config)?;

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.intermediate),
            [intermediate, hidden],
        )?;
        if let Some(weight) = weight {
            self.intermediate.set_dense_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.intermediate.set_dense_bias(bias)?;
        }

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.feed_forward_output),
            [hidden, intermediate],
        )?;
        if let Some(weight) = weight {
            self.intermediate.set_output_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.intermediate.set_output_bias(bias)?;
        }

        bind_layer_norm(
            binder,
            &format!("{layer_prefix}{}", names.output_norm),
            hidden,
            &mut self.output_layer_norm,
        )
    }

    /// Append this encoder layer's parameters, in the order
    /// [`BertLayer::load_weights`] binds them.
    pub fn collect_named_parameters<'a>(
        &'a self,
        layer_prefix: &str,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        self.attention.collect_named_parameters(layer_prefix, names, into);
        self.intermediate.collect_named_parameters(
            layer_prefix.trim_end_matches('.'),
            names.intermediate,
            names.feed_forward_output,
            into,
        );
        self.output_layer_norm
            .collect_named_parameters(&format!("{layer_prefix}{}", names.output_norm), into);
    }

    /// Mutable counterpart of [`BertLayer::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        layer_prefix: &str,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let feed_forward_prefix = layer_prefix.trim_end_matches('.').to_string();
        let norm_name = format!("{layer_prefix}{}", names.output_norm);
        self.attention.collect_named_parameters_mut(layer_prefix, names, into);
        self.intermediate.collect_named_parameters_mut(
            &feed_forward_prefix,
            names.intermediate,
            names.feed_forward_output,
            into,
        );
        self.output_layer_norm.collect_named_parameters_mut(&norm_name, into);
    }
}

impl Layer for BertLayer {
    type Input = (Tensor, Option<Tensor>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, attention_mask) = input;

        let attention_output = self.attention.forward((hidden_states.clone(), attention_mask))?;
        let intermediate_output = self.intermediate.forward(attention_output.clone())?;

        let layer_output = intermediate_output.add(&attention_output)?;
        self.output_layer_norm.forward(layer_output)
    }
}

#[derive(Debug, Clone)]
pub struct BertAttention {
    self_attention: MultiHeadAttention,
    output_layer_norm: LayerNorm,
    device: Device,
}

impl BertAttention {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new_with_device(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
                device,
            )?,
            output_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input: (Tensor, Option<Tensor>)) -> Result<Tensor> {
        let (hidden_states, attention_mask) = input;

        let attention_output = self.self_attention.forward_self_attention(
            &hidden_states,
            attention_mask.as_ref(),
            false, // causal
        )?;
        let output = attention_output.add(&hidden_states)?;
        self.output_layer_norm.forward(output)
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attention.parameter_count() + self.output_layer_norm.parameter_count()
    }

    /// Copy the four attention projections and the post-attention norm.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        layer_prefix: &str,
        names: &BertLayerNames,
        config: &BertConfig,
    ) -> Result<()> {
        let hidden = config.hidden_size;

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.query),
            [hidden, hidden],
        )?;
        if let Some(weight) = weight {
            self.self_attention.set_query_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.self_attention.set_query_bias(bias)?;
        }

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.key),
            [hidden, hidden],
        )?;
        if let Some(weight) = weight {
            self.self_attention.set_key_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.self_attention.set_key_bias(bias)?;
        }

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.value),
            [hidden, hidden],
        )?;
        if let Some(weight) = weight {
            self.self_attention.set_value_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.self_attention.set_value_bias(bias)?;
        }

        let (weight, bias) = take_linear(
            binder,
            &format!("{layer_prefix}{}", names.attention_output),
            [hidden, hidden],
        )?;
        if let Some(weight) = weight {
            self.self_attention.set_out_proj_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.self_attention.set_out_proj_bias(bias)?;
        }

        bind_layer_norm(
            binder,
            &format!("{layer_prefix}{}", names.attention_norm),
            hidden,
            &mut self.output_layer_norm,
        )
    }

    /// Append the four projections and the post-attention norm, spelled with the
    /// same [`BertLayerNames`] table [`BertAttention::load_weights`] uses.
    ///
    /// `layer_prefix` already ends in a `.` (`encoder.layer.3.`), matching the
    /// loader's convention, so the sub-paths are concatenated directly.
    pub fn collect_named_parameters<'a>(
        &'a self,
        layer_prefix: &str,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        self.self_attention.projections().collect_named_parameters(
            "",
            [
                &format!("{layer_prefix}{}", names.query),
                &format!("{layer_prefix}{}", names.key),
                &format!("{layer_prefix}{}", names.value),
                &format!("{layer_prefix}{}", names.attention_output),
            ],
            into,
        );
        self.output_layer_norm
            .collect_named_parameters(&format!("{layer_prefix}{}", names.attention_norm), into);
    }

    /// Mutable counterpart of [`BertAttention::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        layer_prefix: &str,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let norm_name = format!("{layer_prefix}{}", names.attention_norm);
        let projection_names = [
            format!("{layer_prefix}{}", names.query),
            format!("{layer_prefix}{}", names.key),
            format!("{layer_prefix}{}", names.value),
            format!("{layer_prefix}{}", names.attention_output),
        ];
        self.self_attention.projections_mut().collect_named_parameters_mut(
            "",
            [
                &projection_names[0],
                &projection_names[1],
                &projection_names[2],
                &projection_names[3],
            ],
            into,
        );
        self.output_layer_norm.collect_named_parameters_mut(&norm_name, into);
    }
}

#[derive(Debug, Clone)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
    device: Device,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new_with_device(config, device)?);
        }
        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: Tensor, attention_mask: Option<Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states;

        for layer in &self.layers {
            hidden_states = layer.forward((hidden_states, attention_mask.clone()))?;
        }

        Ok(hidden_states)
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Copy every encoder layer's parameters out of a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        names: &BertLayerNames,
        config: &BertConfig,
    ) -> Result<()> {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_prefix = format!("{}{index}.", names.layer_stack_prefix);
            layer.load_weights(binder, &layer_prefix, names, config)?;
        }
        Ok(())
    }

    /// Append every encoder layer's parameters, in layer-index order.
    pub fn collect_named_parameters<'a>(
        &'a self,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_prefix = format!("{}{index}.", names.layer_stack_prefix);
            layer.collect_named_parameters(&layer_prefix, names, into);
        }
    }

    /// Mutable counterpart of [`BertEncoder::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        names: &BertLayerNames,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_prefix = format!("{}{index}.", names.layer_stack_prefix);
            layer.collect_named_parameters_mut(&layer_prefix, names, into);
        }
    }
}

#[derive(Debug, Clone)]
pub struct BertPooler {
    dense: trustformers_core::layers::Linear,
    device: Device,
}

impl BertPooler {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: trustformers_core::layers::Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                true,
                device,
            ),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
    }

    /// The pooler's dense projection.
    pub fn dense(&self) -> &trustformers_core::layers::Linear {
        &self.dense
    }

    /// Append the pooler projection under `pooler.dense.…`.
    pub fn collect_named_parameters<'a>(&'a self, into: &mut Vec<(String, &'a Tensor)>) {
        self.dense.collect_named_parameters("pooler.dense", into);
    }

    /// Mutable counterpart of [`BertPooler::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        self.dense.collect_named_parameters_mut("pooler.dense", into);
    }

    /// Copy the pooler projection out of a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        config: &BertConfig,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        let (weight, bias) = take_linear(binder, "pooler.dense", [hidden, hidden])?;
        if let Some(weight) = weight {
            self.dense.set_weight(weight)?;
        }
        if let Some(bias) = bias {
            self.dense.set_bias(bias)?;
        }
        Ok(())
    }
}

impl Layer for BertPooler {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                // Input shape is [seq_len, hidden_size] (2D)
                // We want to extract the first token: [1, hidden_size]
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        format!(
                            "BertPooler expects 2D input, got {} dimensions",
                            shape.len()
                        ),
                    ));
                }

                // Extract first token and keep it 2D: [1, hidden_size]
                let first_token = arr.slice(s![0..1, ..]).to_owned().into_dyn();
                let pooled = self.dense.forward(Tensor::F32(first_token))?;
                trustformers_core::ops::activations::tanh(&pooled)
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type for pooler".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bert::config::BertConfig;
    use trustformers_core::device::Device;
    use trustformers_core::traits::Layer;

    // --- LCG for deterministic pseudo-random data ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_f32(state: &mut u64) -> f32 {
        let v = lcg_next(state);
        ((v >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn small_config() -> BertConfig {
        BertConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        }
    }

    // --- BertEmbeddings ---

    #[test]
    fn test_bert_embeddings_new_cpu() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        assert_eq!(embeddings.device(), Device::CPU);
    }

    #[test]
    fn test_bert_embeddings_forward_output_shape() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let seq_len = 8usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = embeddings
            .forward(input_ids, None)
            .expect("BertEmbeddings forward must succeed");
        let shape = output.shape();
        assert_eq!(shape[0], seq_len, "First dim must equal seq_len");
        assert_eq!(
            shape[1], cfg.hidden_size,
            "Second dim must equal hidden_size"
        );
    }

    #[test]
    fn test_bert_embeddings_with_token_type_ids() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let seq_len = 6usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let token_type_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let output = embeddings
            .forward(input_ids, Some(token_type_ids))
            .expect("BertEmbeddings with token_type_ids must succeed");
        let shape = output.shape();
        assert_eq!(shape[0], seq_len);
        assert_eq!(shape[1], cfg.hidden_size);
    }

    #[test]
    fn test_bert_embeddings_parameter_count_positive() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let params = embeddings.parameter_count();
        assert!(params > 0, "Embedding parameter count must be positive");
    }

    #[test]
    fn test_bert_embeddings_parameter_count_includes_all_tables() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        // word + position + token_type + layer_norm (2 * hidden_size)
        let min_expected = cfg.vocab_size * cfg.hidden_size
            + cfg.max_position_embeddings * cfg.hidden_size
            + cfg.type_vocab_size * cfg.hidden_size;
        assert!(
            embeddings.parameter_count() >= min_expected,
            "parameter_count must cover at least word+position+token_type tables"
        );
    }

    // --- BertAttention ---

    #[test]
    fn test_bert_attention_new_cpu() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        assert_eq!(attn.device(), Device::CPU);
    }

    #[test]
    fn test_bert_attention_output_shape() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        let batch = 1usize;
        let seq_len = 4usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 12345;
        // MultiHeadAttention forward_self_attention expects 3D: [batch, seq_len, hidden]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = attn.forward((input, None)).expect("BertAttention forward must succeed");
        let shape = output.shape();
        // Output shape must have hidden_size as last dim
        assert_eq!(*shape.last().expect("shape must not be empty"), hidden_size);
    }

    #[test]
    fn test_bert_attention_parameter_count_positive() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        assert!(attn.parameter_count() > 0);
    }

    // --- BertLayer ---

    #[test]
    fn test_bert_layer_new_cpu() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        assert_eq!(layer.device(), Device::CPU);
    }

    #[test]
    fn test_bert_layer_output_shape() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        let batch = 1usize;
        let seq_len = 5usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 99999;
        // BertLayer attention expects 3D: [batch, seq_len, hidden]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = layer.forward((input, None)).expect("BertLayer forward must succeed");
        let shape = output.shape();
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            hidden_size,
            "BertLayer must preserve hidden_size in last dim"
        );
        assert!(
            shape.contains(&seq_len),
            "BertLayer must preserve seq_len in output shape"
        );
    }

    #[test]
    fn test_bert_layer_parameter_count_positive() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        assert!(layer.parameter_count() > 0);
    }

    // --- BertEncoder ---

    #[test]
    fn test_bert_encoder_new() {
        let cfg = small_config();
        let encoder = BertEncoder::new(&cfg).expect("BertEncoder::new must succeed");
        assert_eq!(encoder.device(), Device::CPU);
    }

    #[test]
    fn test_bert_encoder_output_shape() {
        let cfg = small_config();
        let encoder = BertEncoder::new(&cfg).expect("BertEncoder::new must succeed");
        let batch = 1usize;
        let seq_len = 4usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 777;
        // Encoder expects 3D input: [batch, seq_len, hidden_size]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = encoder.forward(input, None).expect("BertEncoder forward must succeed");
        let shape = output.shape();
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            hidden_size,
            "Encoder output must have hidden_size in last dim"
        );
        assert!(
            shape.contains(&seq_len),
            "Encoder output must preserve seq_len"
        );
    }

    #[test]
    fn test_bert_encoder_parameter_count_scales_with_layers() {
        let cfg2 = small_config();
        let cfg4 = BertConfig {
            num_hidden_layers: 4,
            ..small_config()
        };
        let enc2 = BertEncoder::new(&cfg2).expect("BertEncoder 2 layers must succeed");
        let enc4 = BertEncoder::new(&cfg4).expect("BertEncoder 4 layers must succeed");
        assert!(
            enc4.parameter_count() > enc2.parameter_count(),
            "More layers must lead to more parameters"
        );
    }

    // --- BertPooler ---

    #[test]
    fn test_bert_pooler_new() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        assert_eq!(pooler.device(), Device::CPU);
    }

    #[test]
    fn test_bert_pooler_output_shape() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        // BertPooler expects exactly 2D input: [seq_len, hidden_size]
        // It extracts the first token (CLS) and applies dense+tanh
        let seq_len = 6usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 54321;
        let data: Vec<f32> = (0..seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, hidden_size]).expect("Tensor creation must succeed");
        let output = pooler.forward(input).expect("BertPooler forward must succeed");
        let shape = output.shape();
        // Pooler returns [1, hidden_size] (the CLS token linear transform)
        assert!(
            shape.contains(&hidden_size),
            "BertPooler output must contain hidden_size dimension, got {:?}",
            shape
        );
    }

    #[test]
    fn test_bert_pooler_output_tanh_bounded() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        // BertPooler expects 2D input [seq_len, hidden_size]
        let seq_len = 3usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 11111;
        let data: Vec<f32> = (0..seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, hidden_size]).expect("Tensor creation must succeed");
        let output = pooler.forward(input).expect("BertPooler forward must succeed with 2D input");
        // tanh output must be in (-1, 1)
        if let trustformers_core::tensor::Tensor::F32(arr) = &output {
            for &v in arr.iter() {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "BertPooler tanh output must be in [-1, 1], got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_bert_pooler_parameter_count_positive() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        assert!(pooler.parameter_count() > 0);
    }

    // --- Attention head size property ---

    #[test]
    fn test_attention_head_size_equals_hidden_div_heads() {
        let cfg = small_config();
        // hidden=64, heads=8 -> head_size=8
        let head_size = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(
            head_size, 8,
            "head_size must be hidden_size / num_attention_heads"
        );
    }

    // --- Linear weight orientation ---
    //
    // HuggingFace stores `nn.Linear.weight` as `[out_features, in_features]` and
    // so does `trustformers_core::layers::Linear`, whose `forward` multiplies by
    // `W^T`. `take_linear` therefore binds the checkpoint tensor without
    // transposing it -- a claim that only a *non-square* projection can falsify,
    // because a square weight round-trips identically either way round.

    /// Config whose feed-forward block is deliberately non-square (2 -> 3 -> 2).
    fn orientation_config() -> BertConfig {
        BertConfig {
            vocab_size: 4,
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            intermediate_size: 3,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 4,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(false),
            classifier_dropout: None,
        }
    }

    #[test]
    fn feed_forward_weights_are_bound_out_by_in_and_match_a_hand_computed_reference() {
        use crate::weight_loading::checkpoint::Checkpoint;
        use crate::weight_loading::test_support::{build_safetensors, F32Tensor};
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let cfg = orientation_config();
        let names = BertLayerNames::bert();

        // W1 is [intermediate, hidden] = [3, 2]; W2 is [hidden, intermediate] = [2, 3].
        let w1 = [[1.0f32, 2.0], [0.0, -1.0], [0.5, 0.5]];
        let b1 = [0.0f32, 1.0, -1.0];
        let w2 = [[1.0f32, 0.0, -2.0], [0.5, 1.0, 0.25]];
        let b2 = [0.25f32, -0.5];

        let bytes = build_safetensors(&[
            F32Tensor::new(
                &format!("encoder.layer.0.{}.weight", names.intermediate),
                &[3, 2],
                w1.iter().flatten().copied().collect(),
            ),
            F32Tensor::new(
                &format!("encoder.layer.0.{}.bias", names.intermediate),
                &[3],
                b1.to_vec(),
            ),
            F32Tensor::new(
                &format!("encoder.layer.0.{}.weight", names.feed_forward_output),
                &[2, 3],
                w2.iter().flatten().copied().collect(),
            ),
            F32Tensor::new(
                &format!("encoder.layer.0.{}.bias", names.feed_forward_output),
                &[2],
                b2.to_vec(),
            ),
        ]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("fixture must parse");
        let mut binder = checkpoint.binder("");

        let mut layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        // The attention and layer-norm parameters are absent on purpose: the
        // binder records them as missing rather than failing, which keeps this
        // test focused on the feed-forward block.
        layer
            .load_weights(&mut binder, "encoder.layer.0.", &names, &cfg)
            .expect("binding the feed-forward block must succeed");

        let x = [1.0f32, 2.0];
        let input = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), x.to_vec()).expect("input must build"),
        );
        let output = layer.intermediate.forward(input).expect("feed-forward must run");
        let actual: Vec<f32> = match &output {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 output, got {other:?}"),
        };

        // Reference: y = W2 * gelu(W1 * x + b1) + b2, with every weight indexed
        // as `[out][in]`. Reading either weight as `[in][out]` gives a different
        // vector (and, for W1, a dimension mismatch).
        let pre: Vec<f32> =
            (0..3).map(|o| (0..2).map(|i| w1[o][i] * x[i]).sum::<f32>() + b1[o]).collect();
        assert_eq!(pre, vec![5.0, -1.0, 0.5]);
        let activated = trustformers_core::ops::activations::gelu(&Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), pre).expect("activation input must build"),
        ))
        .expect("gelu must run");
        let h: Vec<f32> = match &activated {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 activation, got {other:?}"),
        };
        let expected: Vec<f32> =
            (0..2).map(|o| (0..3).map(|i| w2[o][i] * h[i]).sum::<f32>() + b2[o]).collect();

        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "feed-forward output {actual:?} does not match the reference {expected:?}"
            );
        }
    }

    #[test]
    fn a_transposed_linear_weight_is_rejected_rather_than_silently_accepted() {
        use crate::weight_loading::checkpoint::Checkpoint;
        use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

        let cfg = orientation_config();
        let names = BertLayerNames::bert();

        // `[hidden, intermediate]` instead of `[intermediate, hidden]`: the same
        // values, the wrong way round. A loader that transposed on the way in
        // would accept this and quietly compute a different function.
        let bytes = build_safetensors(&[F32Tensor::ramp(
            &format!("encoder.layer.0.{}.weight", names.intermediate),
            &[2, 3],
            1.0,
        )]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("fixture must parse");
        let mut binder = checkpoint.binder("");

        let mut layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        let err = layer
            .load_weights(&mut binder, "encoder.layer.0.", &names, &cfg)
            .expect_err("a transposed weight must be a hard error");
        let message = err.to_string();
        assert!(
            message.contains("[2, 3]") && message.contains("[3, 2]"),
            "the error must name both the actual and the expected shape: {message}"
        );
    }
}
