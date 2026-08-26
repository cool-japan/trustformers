use crate::bert::layers::{BertEncoder, BertLayerNames};
use crate::distilbert::config::DistilBertConfig;
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors, WeightBinder};
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct DistilBertModel {
    config: DistilBertConfig,
    embeddings: DistilBertEmbeddings,
    transformer: BertEncoder,
    device: Device,
}

impl DistilBertModel {
    pub fn new(config: DistilBertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DistilBertConfig, device: Device) -> Result<Self> {
        let embeddings = DistilBertEmbeddings::new_with_device(&config, device)?;

        let bert_config = Self::as_bert_config(&config);
        let transformer = BertEncoder::new_with_device(&bert_config, device)?;

        Ok(Self {
            config,
            embeddings,
            transformer,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The equivalent BERT encoder configuration.
    ///
    /// DistilBERT's transformer stack is a BERT encoder; only the parameter
    /// spelling and the absence of segment embeddings differ.
    fn as_bert_config(config: &DistilBertConfig) -> crate::bert::config::BertConfig {
        crate::bert::config::BertConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: config.hidden_act.clone(),
            hidden_dropout_prob: config.hidden_dropout_prob,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: 1, // DistilBERT doesn't use token type embeddings
            initializer_range: config.initializer_range,
            layer_norm_eps: config.layer_norm_eps,
            pad_token_id: config.pad_token_id,
            position_embedding_type: config.position_embedding_type.clone(),
            use_cache: config.use_cache,
            classifier_dropout: config.classifier_dropout,
        }
    }

    /// Checkpoint namespaces a base DistilBERT encoder legitimately leaves unused.
    ///
    /// HuggingFace ships the masked-LM head (`vocab_transform`,
    /// `vocab_layer_norm`, `vocab_projector`) and task heads in the same file as
    /// the encoder.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &[
        "vocab_transform.",
        "vocab_layer_norm.",
        "vocab_projector.",
        "classifier.",
        "pre_classifier.",
        "qa_outputs.",
    ];

    /// Non-parameter buffers HuggingFace stores alongside DistilBERT's weights.
    ///
    /// `position_ids` is a registered buffer rather than a learnable parameter
    /// and appears under whatever task prefix the checkpoint uses, so it is
    /// matched by suffix.
    pub(crate) const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] =
        &["embeddings.position_ids", "embeddings.token_type_ids"];

    /// The unused-tensor policy for a bare DistilBERT encoder load.
    pub(crate) fn unused_tensor_policy() -> UnusedTensors<'static> {
        UnusedTensors::new(Self::ALLOWED_UNUSED_PREFIXES, Self::ALLOWED_UNUSED_SUFFIXES)
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// A previous revision ignored the reader entirely and returned `Ok(())`, so
    /// a caller loading a real checkpoint kept a randomly-initialised model and
    /// was told the load had succeeded.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the checkpoint is not a
    /// DistilBERT checkpoint, when a tensor has the wrong shape, or when any
    /// parameter is missing.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// # Errors
    ///
    /// See [`DistilBertModel::load_pretrained_report`].
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix =
            checkpoint.detect_prefix(&["", "distilbert."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);
        let bert_config = Self::as_bert_config(&self.config);
        let names = BertLayerNames::distilbert();

        self.embeddings.load_weights(&mut binder, &self.config)?;
        self.transformer.load_weights(&mut binder, &names, &bert_config)?;

        binder.finish(Self::unused_tensor_policy())
    }

    pub fn forward_with_embeddings(
        &self,
        input_ids: Vec<u32>,
        attention_mask: Option<Vec<u8>>,
    ) -> Result<DistilBertModelOutput> {
        let embeddings = self.embeddings.forward(input_ids.clone())?;

        // The embedding stack produces `[seq_len, hidden_size]`, but the shared
        // BERT encoder's attention needs an explicit batch axis. Without this
        // reshape the first `split_heads` call fails on a 2-D input.
        let embeddings = match embeddings {
            Tensor::F32(arr) => {
                let reshaped = arr
                    .to_shape(IxDyn(&[1, input_ids.len(), self.config.hidden_size]))
                    .map_err(|e| {
                        trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                    })?
                    .to_owned();
                Tensor::F32(reshaped)
            },
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Unsupported tensor type in embeddings",
                        "DistilBertModel::forward_with_embeddings",
                    ),
                )
            },
        };

        let attention_mask_tensor = if let Some(mask) = attention_mask {
            let mask_f32: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
            let shape = vec![1, 1, 1, mask_f32.len()];
            Some(Tensor::F32(
                ArrayD::from_shape_vec(IxDyn(&shape), mask_f32).map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?,
            ))
        } else {
            None
        };

        let hidden_states = self.transformer.forward(embeddings, attention_mask_tensor)?;

        Ok(DistilBertModelOutput {
            last_hidden_state: hidden_states,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DistilBertEmbeddings {
    word_embeddings: trustformers_core::layers::Embedding,
    position_embeddings: trustformers_core::layers::Embedding,
    layer_norm: trustformers_core::layers::LayerNorm,
    dropout: f32,
    device: Device,
}

impl DistilBertEmbeddings {
    pub fn new(config: &DistilBertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DistilBertConfig, device: Device) -> Result<Self> {
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
            layer_norm: trustformers_core::layers::LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of parameters in the embedding tables and their layer norm.
    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Copy the embedding tables and their layer norm out of a checkpoint.
    ///
    /// DistilBERT has no segment (token type) table, so none is requested.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        config: &DistilBertConfig,
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
        if let Some(weight) = binder.take_shaped("embeddings.LayerNorm.weight", &[hidden])? {
            self.layer_norm.set_weight(weight)?;
        }
        if let Some(bias) = binder.take_shaped("embeddings.LayerNorm.bias", &[hidden])? {
            self.layer_norm.set_bias(bias)?;
        }
        Ok(())
    }
}

impl Layer for DistilBertEmbeddings {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, _inputs: Self::Input) -> Result<Self::Output> {
        Err(trustformers_core::errors::TrustformersError::model_error(
            "DistilBertEmbeddings requires special forward method with input_ids".to_string(),
        ))
    }
}

impl DistilBertEmbeddings {
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let seq_length = input_ids.len();
        let position_ids: Vec<u32> = (0..seq_length as u32).collect();

        let inputs_embeds = self.word_embeddings.forward_ids(&input_ids)?;
        let position_embeds = self.position_embeddings.forward_ids(&position_ids)?;

        let embeddings = inputs_embeds.add(&position_embeds)?;
        let embeddings = self.layer_norm.forward(embeddings)?;
        embeddings.dropout(self.dropout)
    }
}

#[derive(Debug)]
pub struct DistilBertModelOutput {
    pub last_hidden_state: Tensor,
}

impl Model for DistilBertModel {
    type Config = DistilBertConfig;
    type Input = TokenizedInput;
    type Output = DistilBertModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_with_embeddings(input.input_ids, Some(input.attention_mask))
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        // Summed from the constituent layers rather than reported as a constant:
        // the previous implementation returned a hardcoded 1_000_000 regardless
        // of the configuration.
        self.embeddings.parameter_count() + self.transformer.parameter_count()
    }
}
