use crate::distilbert::config::DistilBertConfig;
use crate::distilbert::model::DistilBertModel;
use std::io::Read;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct DistilBertForSequenceClassification {
    distilbert: DistilBertModel,
    pre_classifier: Linear,
    classifier: Linear,
    dropout: f32,
    #[allow(dead_code)]
    num_labels: usize,
}

impl DistilBertForSequenceClassification {
    pub fn new(config: DistilBertConfig, num_labels: usize) -> Result<Self> {
        let distilbert = DistilBertModel::new(config.clone())?;
        let pre_classifier = Linear::new(config.hidden_size, config.hidden_size, true);
        let classifier = Linear::new(config.hidden_size, num_labels, true);
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            distilbert,
            pre_classifier,
            classifier,
            dropout,
            num_labels,
        })
    }
}

#[derive(Debug)]
pub struct SequenceClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for DistilBertForSequenceClassification {
    type Config = DistilBertConfig;
    type Input = TokenizedInput;
    type Output = SequenceClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let distilbert_output = self.distilbert.forward(input)?;

        // Use the first token ([CLS]) for classification
        let hidden_state = distilbert_output.last_hidden_state.select_first_token()?;
        let pooled_output = self.pre_classifier.forward(hidden_state)?;
        let pooled_output = trustformers_core::ops::activations::relu(&pooled_output)?;
        let pooled_output = pooled_output.dropout(self.dropout)?;
        let logits = self.classifier.forward(pooled_output)?;

        Ok(SequenceClassifierOutput {
            logits,
            hidden_states: Some(distilbert_output.last_hidden_state),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.distilbert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.distilbert.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model
        self.distilbert.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct DistilBertForMaskedLM {
    distilbert: DistilBertModel,
    vocab_transform: Linear,
    vocab_layer_norm: trustformers_core::layers::LayerNorm,
    vocab_projector: Linear,
}

impl DistilBertForMaskedLM {
    pub fn new(config: DistilBertConfig) -> Result<Self> {
        let distilbert = DistilBertModel::new(config.clone())?;
        let vocab_transform = Linear::new(config.hidden_size, config.hidden_size, true);
        let vocab_layer_norm = trustformers_core::layers::LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps,
        )?;
        let vocab_projector = Linear::new(config.hidden_size, config.vocab_size, true);

        Ok(Self {
            distilbert,
            vocab_transform,
            vocab_layer_norm,
            vocab_projector,
        })
    }
}

#[derive(Debug)]
pub struct MaskedLMOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for DistilBertForMaskedLM {
    type Config = DistilBertConfig;
    type Input = TokenizedInput;
    type Output = MaskedLMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let distilbert_output = self.distilbert.forward(input)?;
        let hidden_states = distilbert_output.last_hidden_state.clone();

        let prediction_logits = self.vocab_transform.forward(hidden_states.clone())?;
        let prediction_logits = trustformers_core::ops::activations::gelu(&prediction_logits)?;
        let prediction_logits = self.vocab_layer_norm.forward(prediction_logits)?;
        let prediction_logits = self.vocab_projector.forward(prediction_logits)?;

        Ok(MaskedLMOutput {
            logits: prediction_logits,
            hidden_states: Some(hidden_states),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.distilbert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.distilbert.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model
        self.distilbert.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct DistilBertForTokenClassification {
    distilbert: DistilBertModel,
    classifier: Linear,
    dropout: f32,
    #[allow(dead_code)]
    num_labels: usize,
}

impl DistilBertForTokenClassification {
    pub fn new(config: DistilBertConfig, num_labels: usize) -> Result<Self> {
        let distilbert = DistilBertModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            distilbert,
            classifier,
            dropout,
            num_labels,
        })
    }
}

#[derive(Debug)]
pub struct TokenClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for DistilBertForTokenClassification {
    type Config = DistilBertConfig;
    type Input = TokenizedInput;
    type Output = TokenClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let distilbert_output = self.distilbert.forward(input)?;
        let sequence_output = distilbert_output.last_hidden_state;

        let sequence_output = sequence_output.dropout(self.dropout)?;
        let logits = self.classifier.forward(sequence_output.clone())?;

        Ok(TokenClassifierOutput {
            logits,
            hidden_states: Some(sequence_output),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.distilbert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.distilbert.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model
        self.distilbert.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct DistilBertForQuestionAnswering {
    distilbert: DistilBertModel,
    qa_outputs: Linear,
    dropout: f32,
}

impl DistilBertForQuestionAnswering {
    pub fn new(config: DistilBertConfig) -> Result<Self> {
        let distilbert = DistilBertModel::new(config.clone())?;
        let qa_outputs = Linear::new(config.hidden_size, 2, true);
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            distilbert,
            qa_outputs,
            dropout,
        })
    }
}

#[derive(Debug)]
pub struct QuestionAnsweringOutput {
    pub start_logits: Tensor,
    pub end_logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for DistilBertForQuestionAnswering {
    type Config = DistilBertConfig;
    type Input = TokenizedInput;
    type Output = QuestionAnsweringOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let distilbert_output = self.distilbert.forward(input)?;
        let sequence_output = distilbert_output.last_hidden_state;

        let sequence_output = sequence_output.dropout(self.dropout)?;
        let logits = self.qa_outputs.forward(sequence_output.clone())?;

        let split_logits = logits.split(logits.shape().len() - 1, 1)?;
        if split_logits.len() != 2 {
            return Err(trustformers_core::errors::TrustformersError::model_error(
                "Expected 2 QA outputs (start and end), got different number".to_string(),
            ));
        }

        let start_logits = split_logits[0].clone();
        let end_logits = split_logits[1].clone();

        Ok(QuestionAnsweringOutput {
            start_logits,
            end_logits,
            hidden_states: Some(sequence_output),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.distilbert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.distilbert.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model
        self.distilbert.num_parameters()
    }
}
