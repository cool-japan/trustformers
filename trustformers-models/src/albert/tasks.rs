use crate::albert::config::AlbertConfig;
use crate::albert::model::AlbertModel;
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct AlbertForSequenceClassification {
    albert: AlbertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertForTokenClassification {
    albert: AlbertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertForQuestionAnswering {
    albert: AlbertModel,
    qa_outputs: Linear,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertForMaskedLM {
    albert: AlbertModel,
    predictions: AlbertMLMHead,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct AlbertMLMHead {
    dense: Linear,
    layer_norm: trustformers_core::layers::LayerNorm,
    decoder: Linear,
    bias: Tensor,
    device: Device,
}

#[derive(Debug)]
pub struct AlbertSequenceClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Vec<Tensor>>,
}

#[derive(Debug)]
pub struct AlbertTokenClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Vec<Tensor>>,
}

#[derive(Debug)]
pub struct AlbertForQuestionAnsweringOutput {
    pub start_logits: Tensor,
    pub end_logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Vec<Tensor>>,
}

#[derive(Debug)]
pub struct AlbertMaskedLMOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Vec<Tensor>>,
}

impl AlbertForSequenceClassification {
    pub fn new(config: AlbertConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: AlbertConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let albert = AlbertModel::new_with_device(config.clone(), device)?;
        let _classifier_dropout =
            config.classifier_dropout_prob.unwrap_or(config.hidden_dropout_prob);
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            albert,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for AlbertForSequenceClassification {
    type Config = AlbertConfig;
    type Input = TokenizedInput;
    type Output = AlbertSequenceClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let albert_output = self.albert.forward(input)?;

        let pooled_output = albert_output.pooler_output.ok_or_else(|| {
            trustformers_core::errors::TrustformersError::model_error(
                "Pooler output is required for sequence classification".to_string(),
            )
        })?;

        let logits = self.classifier.forward(pooled_output)?;

        Ok(AlbertSequenceClassifierOutput {
            logits,
            hidden_states: Some(albert_output.last_hidden_state),
            attentions: albert_output.attentions,
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.albert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.albert.get_config()
    }

    fn num_parameters(&self) -> usize {
        let config = self.albert.get_config();

        // Embeddings: word + position + token_type embeddings
        let embedding_params = config.vocab_size * config.embedding_size
            + config.max_position_embeddings * config.embedding_size
            + config.type_vocab_size * config.embedding_size
            + config.embedding_size * 2; // LayerNorm (gamma + beta)

        // Embedding projection
        let projection_params = config.embedding_size * config.hidden_size + config.hidden_size;

        // ALBERT uses parameter sharing, so we only count unique layer groups
        // Each layer group has inner_group_num layers
        let attention_params_per_layer =
            4 * (config.hidden_size * config.hidden_size + config.hidden_size); // Q,K,V,dense + biases
        let ffn_params_per_layer = config.hidden_size * config.intermediate_size + config.intermediate_size // FFN in
            + config.intermediate_size * config.hidden_size + config.hidden_size; // FFN out
        let layer_norm_params = 4 * config.hidden_size; // 2 LayerNorms per layer, 2 params each

        let params_per_layer =
            attention_params_per_layer + ffn_params_per_layer + layer_norm_params;
        let encoder_params = config.num_hidden_groups * config.inner_group_num * params_per_layer;

        // Pooler (if exists)
        let pooler_params = config.hidden_size * config.hidden_size + config.hidden_size;

        // Classifier head
        let classifier_params = config.hidden_size * self.num_labels + self.num_labels;

        embedding_params + projection_params + encoder_params + pooler_params + classifier_params
    }
}

impl AlbertForTokenClassification {
    pub fn new(config: AlbertConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: AlbertConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let albert = AlbertModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            albert,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for AlbertForTokenClassification {
    type Config = AlbertConfig;
    type Input = TokenizedInput;
    type Output = AlbertTokenClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let albert_output = self.albert.forward(input)?;
        let logits = self.classifier.forward(albert_output.last_hidden_state.clone())?;

        Ok(AlbertTokenClassifierOutput {
            logits,
            hidden_states: Some(albert_output.last_hidden_state),
            attentions: albert_output.attentions,
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.albert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.albert.get_config()
    }

    fn num_parameters(&self) -> usize {
        let config = self.albert.get_config();

        // Base model params (same calculation as above)
        let embedding_params = config.vocab_size * config.embedding_size
            + config.max_position_embeddings * config.embedding_size
            + config.type_vocab_size * config.embedding_size
            + config.embedding_size * 2;

        let projection_params = config.embedding_size * config.hidden_size + config.hidden_size;

        let params_per_layer = 4 * (config.hidden_size * config.hidden_size + config.hidden_size)
            + config.hidden_size * config.intermediate_size
            + config.intermediate_size
            + config.intermediate_size * config.hidden_size
            + config.hidden_size
            + 4 * config.hidden_size;

        let encoder_params = config.num_hidden_groups * config.inner_group_num * params_per_layer;
        let pooler_params = config.hidden_size * config.hidden_size + config.hidden_size;

        // Token classification head
        let classifier_params = config.hidden_size * self.num_labels + self.num_labels;

        embedding_params + projection_params + encoder_params + pooler_params + classifier_params
    }
}

impl AlbertForQuestionAnswering {
    pub fn new(config: AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: AlbertConfig, device: Device) -> Result<Self> {
        let albert = AlbertModel::new_with_device(config.clone(), device)?;
        let qa_outputs = Linear::new_with_device(config.hidden_size, 2, true, device);

        Ok(Self {
            albert,
            qa_outputs,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for AlbertForQuestionAnswering {
    type Config = AlbertConfig;
    type Input = TokenizedInput;
    type Output = AlbertForQuestionAnsweringOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let albert_output = self.albert.forward(input)?;
        let logits = self.qa_outputs.forward(albert_output.last_hidden_state.clone())?;

        let _batch_size = logits.shape()[0];
        let _sequence_length = logits.shape()[1];

        let start_logits = logits.slice(2, 0, 1)?;
        let end_logits = logits.slice(2, 1, 2)?;

        Ok(AlbertForQuestionAnsweringOutput {
            start_logits,
            end_logits,
            hidden_states: Some(albert_output.last_hidden_state),
            attentions: albert_output.attentions,
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.albert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.albert.get_config()
    }

    fn num_parameters(&self) -> usize {
        let config = self.albert.get_config();

        // Base model params
        let embedding_params = config.vocab_size * config.embedding_size
            + config.max_position_embeddings * config.embedding_size
            + config.type_vocab_size * config.embedding_size
            + config.embedding_size * 2;

        let projection_params = config.embedding_size * config.hidden_size + config.hidden_size;

        let params_per_layer = 4 * (config.hidden_size * config.hidden_size + config.hidden_size)
            + config.hidden_size * config.intermediate_size
            + config.intermediate_size
            + config.intermediate_size * config.hidden_size
            + config.hidden_size
            + 4 * config.hidden_size;

        let encoder_params = config.num_hidden_groups * config.inner_group_num * params_per_layer;
        let pooler_params = config.hidden_size * config.hidden_size + config.hidden_size;

        // QA head: 2 outputs (start and end logits)
        let qa_params = config.hidden_size * 2 + 2;

        embedding_params + projection_params + encoder_params + pooler_params + qa_params
    }
}

impl AlbertMLMHead {
    fn new(config: &AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let dense =
            Linear::new_with_device(config.hidden_size, config.embedding_size, true, device);
        let layer_norm = trustformers_core::layers::LayerNorm::new_with_device(
            vec![config.embedding_size],
            config.layer_norm_eps,
            device,
        )?;
        let decoder =
            Linear::new_with_device(config.embedding_size, config.vocab_size, false, device);
        let bias = Tensor::zeros(&[config.vocab_size])?;

        Ok(Self {
            dense,
            layer_norm,
            decoder,
            bias,
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = match "gelu" {
            "gelu" => trustformers_core::ops::activations::gelu(&hidden_states)?,
            "relu" => trustformers_core::ops::activations::relu(&hidden_states)?,
            _ => hidden_states,
        };
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        let hidden_states = self.decoder.forward(hidden_states)?;
        let hidden_states = hidden_states.add(&self.bias)?;

        Ok(hidden_states)
    }
}

impl AlbertForMaskedLM {
    pub fn new(config: AlbertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: AlbertConfig, device: Device) -> Result<Self> {
        let albert = AlbertModel::new_with_device(config.clone(), device)?;
        let predictions = AlbertMLMHead::new_with_device(&config, device)?;

        Ok(Self {
            albert,
            predictions,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for AlbertForMaskedLM {
    type Config = AlbertConfig;
    type Input = TokenizedInput;
    type Output = AlbertMaskedLMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let albert_output = self.albert.forward(input)?;
        let logits = self.predictions.forward(albert_output.last_hidden_state.clone())?;

        Ok(AlbertMaskedLMOutput {
            logits,
            hidden_states: Some(albert_output.last_hidden_state),
            attentions: albert_output.attentions,
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.albert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.albert.get_config()
    }

    fn num_parameters(&self) -> usize {
        let config = self.albert.get_config();

        // Base model params
        let embedding_params = config.vocab_size * config.embedding_size
            + config.max_position_embeddings * config.embedding_size
            + config.type_vocab_size * config.embedding_size
            + config.embedding_size * 2;

        let projection_params = config.embedding_size * config.hidden_size + config.hidden_size;

        let params_per_layer = 4 * (config.hidden_size * config.hidden_size + config.hidden_size)
            + config.hidden_size * config.intermediate_size
            + config.intermediate_size
            + config.intermediate_size * config.hidden_size
            + config.hidden_size
            + 4 * config.hidden_size;

        let encoder_params = config.num_hidden_groups * config.inner_group_num * params_per_layer;
        let pooler_params = config.hidden_size * config.hidden_size + config.hidden_size;

        // MLM head: dense + layer_norm + decoder
        let mlm_head_params = config.hidden_size * config.embedding_size + config.embedding_size // dense
            + config.embedding_size * 2 // layer_norm
            + config.embedding_size * config.vocab_size + config.vocab_size; // decoder + bias

        embedding_params + projection_params + encoder_params + pooler_params + mlm_head_params
    }
}
