use crate::bert::layers::{BertEncoder, BertPooler};
use crate::roberta::config::RobertaConfig;
use std::io::Read;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct RobertaModel {
    config: RobertaConfig,
    embeddings: RobertaEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
}

impl RobertaModel {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let embeddings = RobertaEmbeddings::new(&config)?;

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

        let encoder = BertEncoder::new(&bert_config)?;
        let pooler = Some(BertPooler::new(&bert_config)?);

        Ok(Self {
            config,
            embeddings,
            encoder,
            pooler,
        })
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
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), mask_f32).map_err(|e| {
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
}

impl RobertaEmbeddings {
    pub fn new(config: &RobertaConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: trustformers_core::layers::Embedding::new(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id as usize),
            )?,
            position_embeddings: trustformers_core::layers::Embedding::new(
                config.max_position_embeddings,
                config.hidden_size,
                None,
            )?,
            token_type_embeddings: trustformers_core::layers::Embedding::new(
                config.type_vocab_size,
                config.hidden_size,
                None,
            )?,
            layer_norm: trustformers_core::layers::LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?,
            dropout: config.hidden_dropout_prob,
            padding_idx: config.pad_token_id as usize,
        })
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

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
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
