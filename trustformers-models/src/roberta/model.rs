use crate::bert::layers::{BertEncoder, BertLayerNames, BertPooler};
use crate::bert::model::BertModel;
use crate::roberta::config::RobertaConfig;
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, WeightBinder};
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct RobertaModel {
    config: RobertaConfig,
    embeddings: RobertaEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    device: Device,
}

impl RobertaModel {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RobertaConfig, device: Device) -> Result<Self> {
        let embeddings = RobertaEmbeddings::new_with_device(&config, device)?;

        let bert_config = crate::bert::config::BertConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: config.hidden_act.clone(),
            hidden_dropout_prob: config.hidden_dropout_prob,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: config.type_vocab_size,
            initializer_range: config.initializer_range,
            layer_norm_eps: config.layer_norm_eps,
            pad_token_id: config.pad_token_id,
            position_embedding_type: config.position_embedding_type.clone(),
            use_cache: config.use_cache,
            classifier_dropout: config.classifier_dropout,
        };

        let encoder = BertEncoder::new_with_device(&bert_config, device)?;
        let pooler = Some(BertPooler::new_with_device(&bert_config, device)?);

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

    pub fn forward_with_embeddings(
        &self,
        input_ids: Vec<u32>,
        attention_mask: Option<Vec<u8>>,
        token_type_ids: Option<Vec<u32>>,
    ) -> Result<RobertaModelOutput> {
        let embeddings = self.embeddings.forward(input_ids.clone(), token_type_ids)?;

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

        let encoder_output = self.encoder.forward(embeddings, attention_mask_tensor)?;

        let pooler_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(encoder_output.clone())?)
        } else {
            None
        };

        Ok(RobertaModelOutput {
            last_hidden_state: encoder_output,
            pooler_output,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RobertaEmbeddings {
    word_embeddings: trustformers_core::layers::Embedding,
    position_embeddings: trustformers_core::layers::Embedding,
    token_type_embeddings: trustformers_core::layers::Embedding,
    layer_norm: trustformers_core::layers::LayerNorm,
    dropout: f32,
    padding_idx: usize,
    device: Device,
}

impl RobertaEmbeddings {
    pub fn new(config: &RobertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &RobertaConfig, device: Device) -> Result<Self> {
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
            layer_norm: trustformers_core::layers::LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            padding_idx: config.pad_token_id as usize,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn create_position_ids_from_input_ids(&self, input_ids: &[u32]) -> Vec<u32> {
        let mut position_ids = Vec::new();
        let mut pos = self.padding_idx as u32 + 1;

        for &token_id in input_ids {
            if token_id == self.padding_idx as u32 {
                position_ids.push(self.padding_idx as u32);
            } else {
                position_ids.push(pos);
                pos += 1;
            }
        }

        position_ids
    }
}

impl Layer for RobertaEmbeddings {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, _inputs: Self::Input) -> Result<Self::Output> {
        Err(trustformers_core::errors::TrustformersError::model_error(
            "RobertaEmbeddings requires special forward method with input_ids and token_type_ids"
                .to_string(),
        ))
    }
}

impl RobertaEmbeddings {
    pub fn forward(&self, input_ids: Vec<u32>, token_type_ids: Option<Vec<u32>>) -> Result<Tensor> {
        let position_ids = self.create_position_ids_from_input_ids(&input_ids);

        let inputs_embeds = self.word_embeddings.forward_ids(&input_ids)?;
        let position_embeds = self.position_embeddings.forward_ids(&position_ids)?;

        let token_type_embeds = if let Some(token_type_ids) = token_type_ids {
            self.token_type_embeddings.forward_ids(&token_type_ids)?
        } else {
            let zero_token_types = vec![0u32; input_ids.len()];
            self.token_type_embeddings.forward_ids(&zero_token_types)?
        };

        let embeddings = inputs_embeds.add(&position_embeds)?.add(&token_type_embeds)?;
        let embeddings = self.layer_norm.forward(embeddings)?;
        embeddings.dropout(self.dropout)
    }
}

#[derive(Debug)]
pub struct RobertaModelOutput {
    pub last_hidden_state: Tensor,
    pub pooler_output: Option<Tensor>,
}

impl Model for RobertaModel {
    type Config = RobertaConfig;
    type Input = TokenizedInput;
    type Output = RobertaModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_with_embeddings(
            input.input_ids,
            Some(input.attention_mask),
            input.token_type_ids,
        )
    }

    /// Load a HuggingFace RoBERTa checkpoint (safetensors or `torch.save`).
    ///
    /// A previous revision was `Ok(())` — the reader was never touched, so every
    /// "load" left the model randomly initialised while reporting success. See
    /// [`RobertaModel::load_from_checkpoint`] for the name map.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        // Calculate total parameters for RoBERTa model
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_layers = self.config.num_hidden_layers;

        // Embedding layer: vocab_size * hidden_size + position embeddings + type embeddings
        let embedding_params = vocab_size * hidden_size
            + self.config.max_position_embeddings * hidden_size
            + self.config.type_vocab_size * hidden_size;

        // Each transformer layer
        let attention_params = 4 * hidden_size * hidden_size; // q, k, v, o projections
        let mlp_params = hidden_size * intermediate_size + intermediate_size * hidden_size; // down, up
        let norm_params = 2 * hidden_size; // attention norm + mlp norm
        let layer_params = attention_params + mlp_params + norm_params;

        embedding_params + (num_layers * layer_params)
    }
}

impl RobertaModel {
    /// Build the BERT-shaped config the shared encoder layers were created with.
    fn bert_config(&self) -> crate::bert::config::BertConfig {
        crate::bert::config::BertConfig {
            vocab_size: self.config.vocab_size,
            hidden_size: self.config.hidden_size,
            num_hidden_layers: self.config.num_hidden_layers,
            num_attention_heads: self.config.num_attention_heads,
            intermediate_size: self.config.intermediate_size,
            hidden_act: self.config.hidden_act.clone(),
            hidden_dropout_prob: self.config.hidden_dropout_prob,
            attention_probs_dropout_prob: self.config.attention_probs_dropout_prob,
            max_position_embeddings: self.config.max_position_embeddings,
            type_vocab_size: self.config.type_vocab_size,
            initializer_range: self.config.initializer_range,
            layer_norm_eps: self.config.layer_norm_eps,
            pad_token_id: self.config.pad_token_id,
            position_embedding_type: self.config.position_embedding_type.clone(),
            use_cache: self.config.use_cache,
            classifier_dropout: self.config.classifier_dropout,
        }
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`RobertaModel::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// RoBERTa uses BERT's encoder tensor layout verbatim
    /// (`encoder.layer.{i}.attention.self.query.…`), differing only in the
    /// wrapper prefix (`roberta.` rather than `bert.`) and in having no
    /// meaningful segment embeddings. Both encoder and pooler are therefore
    /// bound through the shared BERT layer loaders rather than a second copy of
    /// the same name map.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like a RoBERTa checkpoint, when a
    /// tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix =
            checkpoint.detect_prefix(&["", "roberta."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);
        let bert_config = self.bert_config();
        let names = BertLayerNames::bert();

        self.embeddings.load_weights(&mut binder, &self.config)?;
        self.encoder.load_weights(&mut binder, &names, &bert_config)?;

        // The pooler is optional in HuggingFace exports
        // (`add_pooling_layer=False`). When the checkpoint has none, drop ours
        // rather than leaving a randomly-initialised projection in place
        // pretending to be trained.
        if checkpoint.contains(&format!("{prefix}pooler.dense.weight")) {
            let pooler = match self.pooler.as_mut() {
                Some(pooler) => pooler,
                None => {
                    self.pooler = Some(BertPooler::new_with_device(&bert_config, self.device)?);
                    self.pooler.as_mut().ok_or_else(|| {
                        TrustformersError::model_error(
                            "RoBERTa pooler could not be created for the checkpoint".to_string(),
                        )
                    })?
                },
            };
            pooler.load_weights(&mut binder, &bert_config)?;
        } else {
            self.pooler = None;
        }

        binder.finish(BertModel::unused_tensor_policy())
    }
}

impl RobertaEmbeddings {
    /// Copy the embedding tables and their layer norm out of a checkpoint.
    ///
    /// RoBERTa always ships `token_type_embeddings` even though its
    /// `type_vocab_size` is 1, so the table is bound like any other parameter.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        config: &RobertaConfig,
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
        if let Some(weight) = binder.take_shaped(
            "embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, hidden],
        )? {
            self.token_type_embeddings.set_weight(weight)?;
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
