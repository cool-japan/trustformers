#![allow(dead_code)]

use crate::albert::config::AlbertConfig;
use crate::common::ActivationType;
use crate::weight_loading::binding::{bind_linear, take_norm_bias, take_norm_weight};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{Embedding, LayerNorm, Linear};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlbertEmbeddings {
    pub(crate) word_embeddings: Embedding,
    pub(crate) position_embeddings: Embedding,
    pub(crate) token_type_embeddings: Embedding,
    pub(crate) layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    pub(crate) embedding_hidden_mapping_in: Linear,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertTransformerGroup {
    pub(crate) albert_layers: Vec<AlbertLayer>,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlbertLayer {
    pub(crate) attention: AlbertAttention,
    pub(crate) ffn: AlbertFeedForward,
    pub(crate) attention_output: AlbertAttentionOutput,
    pub(crate) ffn_output: AlbertFFNOutput,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlbertAttention {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) dense: Linear,
    #[allow(dead_code)]
    pub(crate) layer_norm: LayerNorm,
    dropout: f32,
    num_attention_heads: usize,
    attention_head_size: usize,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertAttentionOutput {
    pub(crate) dense: Linear,
    pub(crate) layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertFeedForward {
    pub(crate) dense: Linear,
    intermediate_act_fn: ActivationType,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertFFNOutput {
    pub(crate) dense: Linear,
    pub(crate) layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlbertModel {
    config: AlbertConfig,
    embeddings: AlbertEmbeddings,
    encoder: AlbertTransformer,
    pooler: Option<AlbertPooler>,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlbertTransformer {
    #[allow(dead_code)]
    embedding_hidden_mapping_in: Linear,
    pub(crate) albert_layer_groups: Vec<AlbertTransformerGroup>,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertPooler {
    pub(crate) dense: Linear,
    activation: ActivationType,
    device: Device,
}

#[derive(Debug)]
pub struct AlbertModelOutput {
    pub last_hidden_state: Tensor,
    pub pooler_output: Option<Tensor>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl AlbertEmbeddings {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.embedding_size,
            Some(config.pad_token_id as usize),
            device,
        )?;
        let position_embeddings = Embedding::new_with_device(
            config.max_position_embeddings,
            config.embedding_size,
            None,
            device,
        )?;
        let token_type_embeddings = Embedding::new_with_device(
            config.type_vocab_size,
            config.embedding_size,
            None,
            device,
        )?;
        let layer_norm =
            LayerNorm::new_with_device(vec![config.embedding_size], config.layer_norm_eps, device)?;
        let embedding_hidden_mapping_in =
            Linear::new_with_device(config.embedding_size, config.hidden_size, true, device);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            embedding_hidden_mapping_in,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, input_ids: Vec<u32>, token_type_ids: Option<Vec<u32>>) -> Result<Tensor> {
        let seq_length = input_ids.len();
        let position_ids: Vec<u32> = (0..seq_length as u32).collect();

        let token_type_ids = token_type_ids.unwrap_or_else(|| vec![0; seq_length]);

        let inputs_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = inputs_embeds.add(&position_embeddings)?.add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(embeddings)?;

        let hidden_states = self.embedding_hidden_mapping_in.forward(embeddings)?;

        Ok(hidden_states)
    }
}

impl AlbertAttention {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        Ok(Self {
            query: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            key: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            value: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.attention_probs_dropout_prob,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states.clone())?;
        let key_layer = self.key.forward(hidden_states.clone())?;
        let value_layer = self.value.forward(hidden_states)?;

        let context_layer =
            self.compute_attention(query_layer, key_layer, value_layer, attention_mask)?;
        let attention_output = self.dense.forward(context_layer)?;

        Ok(attention_output)
    }

    fn compute_attention(
        &self,
        _query: Tensor,
        _key: Tensor,
        value: Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        Ok(value)
    }
}

impl AlbertAttentionOutput {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, input_tensor: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.add(&input_tensor)?;
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

impl AlbertFeedForward {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        // Parse the activation string once at construction. Unsupported
        // identifiers are rejected here (previously this error was raised on
        // every forward pass).
        let intermediate_act_fn = ActivationType::try_from(config.hidden_act.as_str())?;
        Ok(Self {
            dense: Linear::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                true,
                device,
            ),
            intermediate_act_fn,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;

        let hidden_states = self.intermediate_act_fn.apply(&hidden_states)?;

        Ok(hidden_states)
    }
}

impl AlbertFFNOutput {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(
                config.intermediate_size,
                config.hidden_size,
                true,
                device,
            ),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, input_tensor: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.add(&input_tensor)?;
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

impl AlbertLayer {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: AlbertAttention::new_with_device(config, device)?,
            ffn: AlbertFeedForward::new_with_device(config, device)?,
            attention_output: AlbertAttentionOutput::new_with_device(config, device)?,
            ffn_output: AlbertFFNOutput::new_with_device(config, device)?,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states.clone(), attention_mask)?;
        let attention_output = self.attention_output.forward(attention_output, hidden_states)?;

        let ffn_output = self.ffn.forward(attention_output.clone())?;
        let layer_output = self.ffn_output.forward(ffn_output, attention_output)?;

        Ok(layer_output)
    }
}

impl AlbertTransformerGroup {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let mut albert_layers = Vec::new();
        for _ in 0..config.inner_group_num {
            albert_layers.push(AlbertLayer::new_with_device(config, device)?);
        }

        Ok(Self {
            albert_layers,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states;

        for layer in &self.albert_layers {
            hidden_states = layer.forward(hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

impl AlbertTransformer {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let embedding_hidden_mapping_in =
            Linear::new_with_device(config.embedding_size, config.hidden_size, true, device);

        let mut albert_layer_groups = Vec::new();
        for _ in 0..config.num_hidden_groups {
            albert_layer_groups.push(AlbertTransformerGroup::new_with_device(config, device)?);
        }

        Ok(Self {
            embedding_hidden_mapping_in,
            albert_layer_groups,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states;

        let layers_per_group =
            self.albert_layer_groups.len() / self.albert_layer_groups.len().max(1);

        for layer_group_idx in 0..self.albert_layer_groups.len() {
            let layer_group = &self.albert_layer_groups[layer_group_idx];

            for _ in 0..layers_per_group {
                hidden_states = layer_group.forward(hidden_states, attention_mask)?;
            }
        }

        Ok(hidden_states)
    }
}

impl AlbertPooler {
    fn new(config: &AlbertConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Self {
        Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            activation: ActivationType::Tanh,
            device,
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let first_token_tensor = hidden_states.select_first_token()?;
        let pooled_output = self.dense.forward(first_token_tensor)?;

        let pooled_output = self.activation.apply(&pooled_output)?;

        Ok(pooled_output)
    }
}

impl AlbertModel {
    pub fn new(config: AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: AlbertConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = AlbertEmbeddings::new_with_device(&config, device)?;
        let encoder = AlbertTransformer::new_with_device(&config, device)?;
        let pooler = Some(AlbertPooler::new_with_device(&config, device));

        Ok(Self {
            config,
            embeddings,
            encoder,
            pooler,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for AlbertModel {
    type Config = AlbertConfig;
    type Input = TokenizedInput;
    type Output = AlbertModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.embeddings.forward(input.input_ids, input.token_type_ids)?;
        let last_hidden_state = self.encoder.forward(hidden_states, None)?;

        let pooler_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(last_hidden_state.clone())?)
        } else {
            None
        };

        Ok(AlbertModelOutput {
            last_hidden_state,
            pooler_output,
            hidden_states: None,
            attentions: None,
        })
    }

    /// Load a HuggingFace ALBERT checkpoint (safetensors or `torch.save`).
    ///
    /// A previous revision was `Ok(())` — the reader was never touched, so every
    /// "load" left the model randomly initialised while reporting success. See
    /// [`AlbertModel::load_from_checkpoint`] for the name map.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Count embedding parameters
        total += self.embeddings.word_embeddings.parameter_count();
        total += self.embeddings.position_embeddings.parameter_count();
        total += self.embeddings.token_type_embeddings.parameter_count();
        total += self.embeddings.layer_norm.parameter_count();
        total += self.embeddings.embedding_hidden_mapping_in.parameter_count();

        // Count encoder parameters
        total += self.encoder.embedding_hidden_mapping_in.parameter_count();

        // Count parameters in each layer group
        for group in &self.encoder.albert_layer_groups {
            for layer in &group.albert_layers {
                // Attention parameters
                total += layer.attention.query.parameter_count();
                total += layer.attention.key.parameter_count();
                total += layer.attention.value.parameter_count();
                total += layer.attention.dense.parameter_count();
                total += layer.attention.layer_norm.parameter_count();

                // Feed-forward parameters
                total += layer.ffn.dense.parameter_count();

                // Attention output parameters
                total += layer.attention_output.dense.parameter_count();
                total += layer.attention_output.layer_norm.parameter_count();

                // FFN output parameters
                total += layer.ffn_output.dense.parameter_count();
                total += layer.ffn_output.layer_norm.parameter_count();
            }
        }

        // Count pooler parameters if present
        if let Some(ref pooler) = self.pooler {
            total += pooler.dense.parameter_count();
        }

        total
    }
}

impl AlbertModel {
    /// Checkpoint namespaces an ALBERT encoder legitimately does not consume.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &[
        "predictions.",
        "cls.",
        "classifier.",
        "qa_outputs.",
        "sop_classifier.",
    ];

    /// Non-parameter buffers HuggingFace stores alongside ALBERT's weights.
    pub(crate) const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] =
        &["embeddings.position_ids", "embeddings.token_type_ids"];

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`AlbertModel::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// ALBERT's distinguishing feature is **cross-layer parameter sharing**: the
    /// checkpoint stores `num_hidden_groups` groups of `inner_group_num` layers,
    /// and the `num_hidden_layers` runtime layers reuse them. The tensor names
    /// therefore carry a *group* index, not a layer index
    /// (`encoder.albert_layer_groups.0.albert_layers.0.…`), and this loader
    /// binds exactly the groups the checkpoint holds. Its factorised embedding
    /// (`embedding_size` ≠ `hidden_size`) also means the up-projection
    /// `encoder.embedding_hidden_mapping_in` is a real parameter rather than an
    /// identity.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like an ALBERT checkpoint, when a
    /// tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix =
            checkpoint.detect_prefix(&["", "albert."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let config = self.config.clone();
        let embedding_size = config.embedding_size;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        // ── Embeddings (at `embedding_size`, not `hidden_size`) ──────────────
        if let Some(w) = binder.take_shaped(
            "embeddings.word_embeddings.weight",
            &[config.vocab_size, embedding_size],
        )? {
            self.embeddings.word_embeddings.set_weight(w)?;
        }
        if let Some(w) = binder.take_shaped(
            "embeddings.position_embeddings.weight",
            &[config.max_position_embeddings, embedding_size],
        )? {
            self.embeddings.position_embeddings.set_weight(w)?;
        }
        if let Some(w) = binder.take_shaped(
            "embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, embedding_size],
        )? {
            self.embeddings.token_type_embeddings.set_weight(w)?;
        }
        if let Some(w) = binder.take_shaped("embeddings.LayerNorm.weight", &[embedding_size])? {
            self.embeddings.layer_norm.set_weight(w)?;
        }
        if let Some(b) = binder.take_shaped("embeddings.LayerNorm.bias", &[embedding_size])? {
            self.embeddings.layer_norm.set_bias(b)?;
        }

        // ── Factorised embedding up-projection ───────────────────────────────
        bind_linear(
            &mut binder,
            "encoder.embedding_hidden_mapping_in",
            hidden,
            embedding_size,
            true,
            &mut self.encoder.embedding_hidden_mapping_in,
        )?;

        // ── Shared layer groups ──────────────────────────────────────────────
        for (group_index, group) in self.encoder.albert_layer_groups.iter_mut().enumerate() {
            for (layer_index, layer) in group.albert_layers.iter_mut().enumerate() {
                let base = format!(
                    "encoder.albert_layer_groups.{group_index}.albert_layers.{layer_index}"
                );
                bind_linear(
                    &mut binder,
                    &format!("{base}.attention.query"),
                    hidden,
                    hidden,
                    true,
                    &mut layer.attention.query,
                )?;
                bind_linear(
                    &mut binder,
                    &format!("{base}.attention.key"),
                    hidden,
                    hidden,
                    true,
                    &mut layer.attention.key,
                )?;
                bind_linear(
                    &mut binder,
                    &format!("{base}.attention.value"),
                    hidden,
                    hidden,
                    true,
                    &mut layer.attention.value,
                )?;
                bind_linear(
                    &mut binder,
                    &format!("{base}.attention.dense"),
                    hidden,
                    hidden,
                    true,
                    &mut layer.attention_output.dense,
                )?;
                if let Some(w) =
                    take_norm_weight(&mut binder, &format!("{base}.attention.LayerNorm"), hidden)?
                {
                    layer.attention_output.layer_norm.set_weight(w)?;
                }
                if let Some(b) =
                    take_norm_bias(&mut binder, &format!("{base}.attention.LayerNorm"), hidden)?
                {
                    layer.attention_output.layer_norm.set_bias(b)?;
                }

                bind_linear(
                    &mut binder,
                    &format!("{base}.ffn"),
                    intermediate,
                    hidden,
                    true,
                    &mut layer.ffn.dense,
                )?;
                bind_linear(
                    &mut binder,
                    &format!("{base}.ffn_output"),
                    hidden,
                    intermediate,
                    true,
                    &mut layer.ffn_output.dense,
                )?;
                if let Some(w) = take_norm_weight(
                    &mut binder,
                    &format!("{base}.full_layer_layer_norm"),
                    hidden,
                )? {
                    layer.ffn_output.layer_norm.set_weight(w)?;
                }
                if let Some(b) = take_norm_bias(
                    &mut binder,
                    &format!("{base}.full_layer_layer_norm"),
                    hidden,
                )? {
                    layer.ffn_output.layer_norm.set_bias(b)?;
                }
            }
        }

        // ── Pooler (optional in HuggingFace exports) ─────────────────────────
        if checkpoint.contains(&format!("{prefix}pooler.weight")) {
            let pooler = self.pooler.as_mut().ok_or_else(|| {
                TrustformersError::model_error(
                    "the checkpoint carries a pooler but this model has none".to_string(),
                )
            })?;
            bind_linear(
                &mut binder,
                "pooler",
                hidden,
                hidden,
                true,
                &mut pooler.dense,
            )?;
        } else {
            // Do not leave a randomly-initialised projection pretending to be
            // trained when the checkpoint has none.
            self.pooler = None;
        }

        binder.finish(UnusedTensors::new(
            Self::ALLOWED_UNUSED_PREFIXES,
            Self::ALLOWED_UNUSED_SUFFIXES,
        ))
    }
}
