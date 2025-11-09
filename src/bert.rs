use tch::{nn, Tensor, Kind, Device};
use crate::models::{TransformerConfig, TransformerEncoder, LayerNormTrait, LayerNorm, RMSNorm};
use crate::utils::{PositionalEncoding, PositionalEncodingType};
use std::collections::HashMap;

#[derive(Clone)]
pub struct BertConfig {
    pub vocab_size: i64,
    pub d_model: i64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub d_ff: i64,
    pub max_seq_len: i64,
    pub dropout: f64,
    pub layer_norm_eps: f64,
    pub use_rms_norm: bool,
    pub num_token_types: i64,
    pub pad_token_id: i64,
    pub cls_token_id: i64,
    pub sep_token_id: i64,
    pub mask_token_id: i64,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 3072,
            max_seq_len: 512,
            dropout: 0.1,
            layer_norm_eps: 1e-12,
            use_rms_norm: false,
            num_token_types: 2,
            pad_token_id: 0,
            cls_token_id: 101,
            sep_token_id: 102,
            mask_token_id: 103,
        }
    }
}

pub struct BertModel {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: TransformerEncoder,
    pooler: Option<BertPooler>,
}

pub struct BertEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: Box<dyn LayerNormTrait>,
    dropout: nn::Dropout,
}

pub struct BertPooler {
    dense: nn::Linear,
    activation: nn::func_t,
}

impl BertModel {
    pub fn new(vs: &nn::Path, config: BertConfig, use_pooler: bool) -> Self {
        let embeddings = BertEmbeddings::new(&(vs / "embeddings"), &config);

        let transformer_config = TransformerConfig {
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            d_ff: config.d_ff,
            max_seq_len: config.max_seq_len,
            dropout: config.dropout,
            layer_norm_eps: config.layer_norm_eps,
            use_rms_norm: config.use_rms_norm,
            activation_type: crate::layers::ActivationType::GELU,
            use_glu: false,
            use_swiglu: false,
            positional_encoding: PositionalEncodingType::Learned,
            use_alibi: false,
            pre_norm: false,
        };

        let encoder = TransformerEncoder::new(&(vs / "encoder"), &transformer_config);

        let pooler = if use_pooler {
            Some(BertPooler::new(&(vs / "pooler"), config.d_model))
        } else {
            None
        };

        Self {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   token_type_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>) -> BertOutput {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids);
        let encoder_output = self.encoder.forward(&embedding_output, attention_mask);

        let pooled_output = if let Some(pooler) = &self.pooler {
            Some(pooler.forward(&encoder_output))
        } else {
            None
        };

        BertOutput {
            last_hidden_state: encoder_output,
            pooled_output,
        }
    }
}

impl BertEmbeddings {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let word_embeddings = nn::embedding(vs / "word_embeddings", config.vocab_size, config.d_model, Default::default());
        let position_embeddings = nn::embedding(vs / "position_embeddings", config.max_seq_len, config.d_model, Default::default());
        let token_type_embeddings = nn::embedding(vs / "token_type_embeddings", config.num_token_types, config.d_model, Default::default());

        let layer_norm: Box<dyn LayerNormTrait> = if config.use_rms_norm {
            Box::new(RMSNorm::new(vs / "LayerNorm", vec![config.d_model], config.layer_norm_eps))
        } else {
            Box::new(LayerNorm::new(vs / "LayerNorm", vec![config.d_model], config.layer_norm_eps, true))
        };

        let dropout = nn::dropout(vs / "dropout", config.dropout);

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Tensor {
        let seq_len = input_ids.size()[1];
        let position_ids = Tensor::arange(seq_len, (Kind::Int64, input_ids.device()));

        let word_embeds = input_ids.apply(&self.word_embeddings);
        let position_embeds = position_ids.apply(&self.position_embeddings);

        let token_type_embeds = if let Some(token_type_ids) = token_type_ids {
            token_type_ids.apply(&self.token_type_embeddings)
        } else {
            let token_type_ids = Tensor::zeros_like(input_ids);
            token_type_ids.apply(&self.token_type_embeddings)
        };

        let embeddings = word_embeds + position_embeds.unsqueeze(0).expand_as(&word_embeds) + token_type_embeds;
        let embeddings = self.layer_norm.forward(&embeddings);
        embeddings.apply(&self.dropout)
    }
}

impl BertPooler {
    fn new(vs: &nn::Path, d_model: i64) -> Self {
        let dense = nn::linear(vs / "dense", d_model, d_model, Default::default());
        Self {
            dense,
            activation: nn::func_t::new(|xs| xs.tanh()),
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let first_token_tensor = hidden_states.narrow(1, 0, 1).squeeze_dim(1);
        let pooled_output = first_token_tensor.apply(&self.dense);
        self.activation.forward_t(&pooled_output, false)
    }
}

pub struct BertOutput {
    pub last_hidden_state: Tensor,
    pub pooled_output: Option<Tensor>,
}

pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertLMPredictionHead,
}

pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: nn::Linear,
    bias: Tensor,
}

pub struct BertPredictionHeadTransform {
    dense: nn::Linear,
    layer_norm: Box<dyn LayerNormTrait>,
}

impl BertForMaskedLM {
    pub fn new(vs: &nn::Path, config: BertConfig) -> Self {
        let bert = BertModel::new(&(vs / "bert"), config.clone(), false);
        let cls = BertLMPredictionHead::new(&(vs / "cls"), &config);

        Self { bert, cls }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   token_type_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>,
                   labels: Option<&Tensor>) -> MaskedLMOutput {
        let bert_output = self.bert.forward(input_ids, token_type_ids, attention_mask);
        let prediction_scores = self.cls.forward(&bert_output.last_hidden_state);

        let loss = if let Some(labels) = labels {
            let loss_fct = nn::CrossEntropyLoss::new();
            let active_loss = labels.ne(-100);
            let active_logits = prediction_scores.view([-1, self.bert.config.vocab_size]);
            let active_labels = labels.view([-1]);

            let active_logits = active_logits.masked_select(&active_loss.view([-1]).unsqueeze(1).expand_as(&active_logits));
            let active_labels = active_labels.masked_select(&active_loss.view([-1]));

            Some(active_logits.cross_entropy_for_logits(&active_labels))
        } else {
            None
        };

        MaskedLMOutput {
            loss,
            logits: prediction_scores,
            hidden_states: bert_output.last_hidden_state,
        }
    }
}

impl BertLMPredictionHead {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let transform = BertPredictionHeadTransform::new(&(vs / "predictions" / "transform"), config);
        let decoder = nn::linear(vs / "predictions" / "decoder", config.d_model, config.vocab_size, Default::default());
        let bias = vs.var("predictions/decoder/bias", &[config.vocab_size], nn::Init::Const(0.0));

        Self {
            transform,
            decoder,
            bias,
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let transformed = self.transform.forward(hidden_states);
        let decoded = transformed.apply(&self.decoder) + &self.bias;
        decoded
    }
}

impl BertPredictionHeadTransform {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let dense = nn::linear(vs / "dense", config.d_model, config.d_model, Default::default());

        let layer_norm: Box<dyn LayerNormTrait> = if config.use_rms_norm {
            Box::new(RMSNorm::new(vs / "LayerNorm", vec![config.d_model], config.layer_norm_eps))
        } else {
            Box::new(LayerNorm::new(vs / "LayerNorm", vec![config.d_model], config.layer_norm_eps, true))
        };

        Self { dense, layer_norm }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_states = hidden_states.apply(&self.dense);
        let hidden_states = hidden_states.gelu("none");
        self.layer_norm.forward(&hidden_states)
    }
}

pub struct MaskedLMOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Tensor,
}

pub struct BertForNextSentencePrediction {
    bert: BertModel,
    cls: BertOnlyNSPHead,
}

pub struct BertOnlyNSPHead {
    seq_relationship: nn::Linear,
}

impl BertForNextSentencePrediction {
    pub fn new(vs: &nn::Path, config: BertConfig) -> Self {
        let bert = BertModel::new(&(vs / "bert"), config.clone(), true);
        let cls = BertOnlyNSPHead::new(&(vs / "cls"), config.d_model);

        Self { bert, cls }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   token_type_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>,
                   labels: Option<&Tensor>) -> NextSentencePredictionOutput {
        let bert_output = self.bert.forward(input_ids, token_type_ids, attention_mask);
        let pooled_output = bert_output.pooled_output.expect("Pooled output required for NSP");
        let seq_relationship_scores = self.cls.forward(&pooled_output);

        let loss = if let Some(labels) = labels {
            Some(seq_relationship_scores.cross_entropy_for_logits(labels))
        } else {
            None
        };

        NextSentencePredictionOutput {
            loss,
            logits: seq_relationship_scores,
            hidden_states: bert_output.last_hidden_state,
        }
    }
}

impl BertOnlyNSPHead {
    fn new(vs: &nn::Path, d_model: i64) -> Self {
        let seq_relationship = nn::linear(vs / "seq_relationship", d_model, 2, Default::default());
        Self { seq_relationship }
    }

    fn forward(&self, pooled_output: &Tensor) -> Tensor {
        pooled_output.apply(&self.seq_relationship)
    }
}

pub struct NextSentencePredictionOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Tensor,
}

pub struct BertForPreTraining {
    bert: BertModel,
    cls: BertPreTrainingHeads,
}

pub struct BertPreTrainingHeads {
    predictions: BertLMPredictionHead,
    seq_relationship: nn::Linear,
}

impl BertForPreTraining {
    pub fn new(vs: &nn::Path, config: BertConfig) -> Self {
        let bert = BertModel::new(&(vs / "bert"), config.clone(), true);
        let cls = BertPreTrainingHeads::new(&(vs / "cls"), &config);

        Self { bert, cls }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   token_type_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>,
                   masked_lm_labels: Option<&Tensor>,
                   next_sentence_labels: Option<&Tensor>) -> PreTrainingOutput {
        let bert_output = self.bert.forward(input_ids, token_type_ids, attention_mask);
        let pooled_output = bert_output.pooled_output.expect("Pooled output required for pre-training");

        let (prediction_scores, seq_relationship_scores) = self.cls.forward(&bert_output.last_hidden_state, &pooled_output);

        let masked_lm_loss = if let Some(labels) = masked_lm_labels {
            let loss_fct = nn::CrossEntropyLoss::new();
            let active_loss = labels.ne(-100);
            let active_logits = prediction_scores.view([-1, self.bert.config.vocab_size]);
            let active_labels = labels.view([-1]);

            let active_logits = active_logits.masked_select(&active_loss.view([-1]).unsqueeze(1).expand_as(&active_logits));
            let active_labels = active_labels.masked_select(&active_loss.view([-1]));

            Some(active_logits.cross_entropy_for_logits(&active_labels))
        } else {
            None
        };

        let next_sentence_loss = if let Some(labels) = next_sentence_labels {
            Some(seq_relationship_scores.cross_entropy_for_logits(labels))
        } else {
            None
        };

        let total_loss = match (masked_lm_loss.as_ref(), next_sentence_loss.as_ref()) {
            (Some(mlm), Some(nsp)) => Some(mlm + nsp),
            (Some(mlm), None) => Some(mlm.shallow_clone()),
            (None, Some(nsp)) => Some(nsp.shallow_clone()),
            (None, None) => None,
        };

        PreTrainingOutput {
            loss: total_loss,
            prediction_logits: prediction_scores,
            seq_relationship_logits: seq_relationship_scores,
            hidden_states: bert_output.last_hidden_state,
        }
    }
}

impl BertPreTrainingHeads {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let predictions = BertLMPredictionHead::new(vs, config);
        let seq_relationship = nn::linear(vs / "seq_relationship", config.d_model, 2, Default::default());

        Self {
            predictions,
            seq_relationship,
        }
    }

    fn forward(&self, sequence_output: &Tensor, pooled_output: &Tensor) -> (Tensor, Tensor) {
        let prediction_scores = self.predictions.forward(sequence_output);
        let seq_relationship_scores = pooled_output.apply(&self.seq_relationship);
        (prediction_scores, seq_relationship_scores)
    }
}

pub struct PreTrainingOutput {
    pub loss: Option<Tensor>,
    pub prediction_logits: Tensor,
    pub seq_relationship_logits: Tensor,
    pub hidden_states: Tensor,
}

pub fn create_masked_lm_labels(input_ids: &Tensor, mask_token_id: i64, vocab_size: i64, mask_prob: f64) -> (Tensor, Tensor) {
    let mut masked_input = input_ids.shallow_clone();
    let mut labels = Tensor::full_like(input_ids, -100);

    let mask = Tensor::rand_like(input_ids, (Kind::Float, input_ids.device())).lt(mask_prob);

    // 80% of the time, replace with [MASK]
    let mask_indices = mask.logical_and(&Tensor::rand_like(&mask, (Kind::Float, mask.device())).lt(0.8));
    masked_input = masked_input.masked_fill(&mask_indices, mask_token_id);

    // 10% of the time, replace with random token
    let random_indices = mask.logical_and(&Tensor::rand_like(&mask, (Kind::Float, mask.device())).ge(0.8)).logical_and(&Tensor::rand_like(&mask, (Kind::Float, mask.device())).lt(0.9));
    let random_tokens = Tensor::randint(vocab_size, &random_indices.size(), (Kind::Int64, random_indices.device()));
    masked_input = masked_input.where_self(&random_indices.logical_not(), &random_tokens);

    // 10% of the time, keep original token

    labels = labels.where_self(&mask.logical_not(), input_ids);

    (masked_input, labels)
}