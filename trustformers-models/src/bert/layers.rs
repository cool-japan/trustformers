use crate::bert::config::BertConfig;
use trustformers_core::errors::{tensor_op_error, Result};
use trustformers_core::layers::{FeedForward, LayerNorm, MultiHeadAttention};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

#[derive(Debug, Clone)]
pub struct BertEmbeddings {
    word_embeddings: trustformers_core::layers::Embedding,
    position_embeddings: trustformers_core::layers::Embedding,
    token_type_embeddings: trustformers_core::layers::Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout_prob: f32,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self> {
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
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout_prob: config.hidden_dropout_prob,
        })
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
}

#[derive(Debug, Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: FeedForward,
    output_layer_norm: LayerNorm,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::new(config)?,
            intermediate: FeedForward::new(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_dropout_prob,
            )?,
            output_layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.intermediate.parameter_count()
            + self.output_layer_norm.parameter_count()
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
}

impl BertAttention {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
            )?,
            output_layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
        })
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
}

#[derive(Debug, Clone)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config)?);
        }
        Ok(Self { layers })
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
}

#[derive(Debug, Clone)]
pub struct BertPooler {
    dense: trustformers_core::layers::Linear,
}

impl BertPooler {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            dense: trustformers_core::layers::Linear::new(
                config.hidden_size,
                config.hidden_size,
                true,
            ),
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
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
                let first_token = arr.slice(ndarray::s![0..1, ..]).to_owned().into_dyn();
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
