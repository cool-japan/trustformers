use crate::bert::layers::BertEncoder;
use crate::distilbert::config::DistilBertConfig;
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

        // Convert to BERT config for reusing BertEncoder
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
            type_vocab_size: 1, // DistilBERT doesn't use token type embeddings
            initializer_range: config.initializer_range,
            layer_norm_eps: config.layer_norm_eps,
            pad_token_id: config.pad_token_id,
            position_embedding_type: config.position_embedding_type.clone(),
            use_cache: config.use_cache,
            classifier_dropout: config.classifier_dropout,
        };

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

    pub fn forward_with_embeddings(
        &self,
        input_ids: Vec<u32>,
        attention_mask: Option<Vec<u8>>,
    ) -> Result<DistilBertModelOutput> {
        let embeddings = self.embeddings.forward(input_ids.clone())?;

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

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        // Calculate approximate parameters for DistilBERT
        1000000 // Placeholder
    }
}
