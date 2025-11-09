use crate::roberta::config::RobertaConfig;
use crate::roberta::model::RobertaModel;
use std::io::Read;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct RobertaForSequenceClassification {
    roberta: RobertaModel,
    classifier: RobertaClassificationHead,
    #[allow(dead_code)]
    num_labels: usize,
}

impl RobertaForSequenceClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let classifier = RobertaClassificationHead::new(&config, num_labels)?;

        Ok(Self {
            roberta,
            classifier,
            num_labels,
        })
    }
}

#[derive(Debug, Clone)]
struct RobertaClassificationHead {
    dense: Linear,
    dropout: f32,
    out_proj: Linear,
}

impl RobertaClassificationHead {
    fn new(config: &RobertaConfig, num_labels: usize) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size, true),
            dropout: config.classifier_dropout.unwrap_or(config.hidden_dropout_prob),
            out_proj: Linear::new(config.hidden_size, num_labels, true),
        })
    }

    fn forward(&self, features: Tensor) -> Result<Tensor> {
        let x = features.select_first_token()?;
        let x = self.dense.forward(x)?;
        let x = trustformers_core::ops::activations::tanh(&x)?;
        let x = x.dropout(self.dropout)?;
        self.out_proj.forward(x)
    }
}

#[derive(Debug)]
pub struct SequenceClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for RobertaForSequenceClassification {
    type Config = RobertaConfig;
    type Input = TokenizedInput;
    type Output = SequenceClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let roberta_output = self.roberta.forward(input)?;
        let logits = self.classifier.forward(roberta_output.last_hidden_state.clone())?;

        Ok(SequenceClassifierOutput {
            logits,
            hidden_states: Some(roberta_output.last_hidden_state),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.roberta.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.roberta.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model or provide reasonable default
        self.roberta.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct RobertaForMaskedLM {
    roberta: RobertaModel,
    lm_head: RobertaLMHead,
}

impl RobertaForMaskedLM {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let lm_head = RobertaLMHead::new(&config)?;

        Ok(Self { roberta, lm_head })
    }
}

#[derive(Debug, Clone)]
struct RobertaLMHead {
    dense: Linear,
    layer_norm: trustformers_core::layers::LayerNorm,
    decoder: Linear,
}

impl RobertaLMHead {
    fn new(config: &RobertaConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size, true),
            layer_norm: trustformers_core::layers::LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?,
            decoder: Linear::new(config.hidden_size, config.vocab_size, true),
        })
    }

    fn forward(&self, features: Tensor) -> Result<Tensor> {
        let x = self.dense.forward(features)?;
        let x = trustformers_core::ops::activations::gelu(&x)?;
        let x = self.layer_norm.forward(x)?;
        self.decoder.forward(x)
    }
}

#[derive(Debug)]
pub struct MaskedLMOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for RobertaForMaskedLM {
    type Config = RobertaConfig;
    type Input = TokenizedInput;
    type Output = MaskedLMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let roberta_output = self.roberta.forward(input)?;
        let prediction_scores = self.lm_head.forward(roberta_output.last_hidden_state.clone())?;

        Ok(MaskedLMOutput {
            logits: prediction_scores,
            hidden_states: Some(roberta_output.last_hidden_state),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.roberta.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.roberta.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model or provide reasonable default
        self.roberta.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct RobertaForTokenClassification {
    roberta: RobertaModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl RobertaForTokenClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            roberta,
            classifier,
            num_labels,
        })
    }
}

#[derive(Debug)]
pub struct TokenClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for RobertaForTokenClassification {
    type Config = RobertaConfig;
    type Input = TokenizedInput;
    type Output = TokenClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let roberta_output = self.roberta.forward(input)?;
        let sequence_output = roberta_output.last_hidden_state;

        let logits = self.classifier.forward(sequence_output.clone())?;

        Ok(TokenClassifierOutput {
            logits,
            hidden_states: Some(sequence_output),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.roberta.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.roberta.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model or provide reasonable default
        self.roberta.num_parameters()
    }
}

#[derive(Debug, Clone)]
pub struct RobertaForQuestionAnswering {
    roberta: RobertaModel,
    qa_outputs: Linear,
}

impl RobertaForQuestionAnswering {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let qa_outputs = Linear::new(config.hidden_size, 2, true);

        Ok(Self {
            roberta,
            qa_outputs,
        })
    }
}

#[derive(Debug)]
pub struct QuestionAnsweringOutput {
    pub start_logits: Tensor,
    pub end_logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for RobertaForQuestionAnswering {
    type Config = RobertaConfig;
    type Input = TokenizedInput;
    type Output = QuestionAnsweringOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let roberta_output = self.roberta.forward(input)?;
        let sequence_output = roberta_output.last_hidden_state;

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
        self.roberta.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.roberta.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Delegate to underlying model or provide reasonable default
        self.roberta.num_parameters()
    }
}
