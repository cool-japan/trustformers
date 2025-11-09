use crate::bert::config::BertConfig;
use crate::bert::model::BertModel;
use std::io::Read;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl BertForSequenceClassification {
    pub fn new(config: BertConfig, num_labels: usize) -> Result<Self> {
        let bert = BertModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            bert,
            classifier,
            num_labels,
        })
    }
}

#[derive(Debug)]
pub struct SequenceClassifierOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for BertForSequenceClassification {
    type Config = BertConfig;
    type Input = TokenizedInput;
    type Output = SequenceClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let bert_output = self.bert.forward(input)?;

        let pooled_output = bert_output.pooler_output.ok_or_else(|| {
            trustformers_core::errors::TrustformersError::model_error(
                "BertForSequenceClassification requires pooler output".to_string(),
            )
        })?;

        let logits = self.classifier.forward(pooled_output)?;

        Ok(SequenceClassifierOutput {
            logits,
            hidden_states: Some(bert_output.last_hidden_state),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.bert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.classifier.parameter_count()
    }
}

#[derive(Debug, Clone)]
pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertLMHead,
}

impl BertForMaskedLM {
    pub fn new(config: BertConfig) -> Result<Self> {
        let bert = BertModel::new(config.clone())?;
        let cls = BertLMHead::new(&config)?;

        Ok(Self { bert, cls })
    }
}

#[derive(Debug, Clone)]
struct BertLMHead {
    dense: Linear,
    layer_norm: trustformers_core::layers::LayerNorm,
    decoder: Linear,
}

impl BertLMHead {
    fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size, true),
            layer_norm: trustformers_core::layers::LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?,
            decoder: Linear::new(config.hidden_size, config.vocab_size, true),
        })
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = trustformers_core::ops::activations::gelu(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        self.decoder.forward(hidden_states)
    }

    fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
            + self.layer_norm.parameter_count()
            + self.decoder.parameter_count()
    }
}

#[derive(Debug)]
pub struct MaskedLMOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for BertForMaskedLM {
    type Config = BertConfig;
    type Input = TokenizedInput;
    type Output = MaskedLMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let bert_output = self.bert.forward(input)?;
        let prediction_scores = self.cls.forward(bert_output.last_hidden_state.clone())?;

        Ok(MaskedLMOutput {
            logits: prediction_scores,
            hidden_states: Some(bert_output.last_hidden_state),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.bert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.cls.parameter_count()
    }
}

#[derive(Debug, Clone)]
pub struct BertForTokenClassification {
    bert: BertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl BertForTokenClassification {
    pub fn new(config: BertConfig, num_labels: usize) -> Result<Self> {
        let bert = BertModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            bert,
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

impl Model for BertForTokenClassification {
    type Config = BertConfig;
    type Input = TokenizedInput;
    type Output = TokenClassifierOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let bert_output = self.bert.forward(input)?;
        let sequence_output = bert_output.last_hidden_state;

        let logits = self.classifier.forward(sequence_output.clone())?;

        Ok(TokenClassifierOutput {
            logits,
            hidden_states: Some(sequence_output),
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.bert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.classifier.parameter_count()
    }
}

#[derive(Debug, Clone)]
pub struct BertForQuestionAnswering {
    bert: BertModel,
    qa_outputs: Linear,
}

impl BertForQuestionAnswering {
    pub fn new(config: BertConfig) -> Result<Self> {
        let bert = BertModel::new(config.clone())?;
        // QA outputs has 2 classes: start and end positions
        let qa_outputs = Linear::new(config.hidden_size, 2, true);

        Ok(Self { bert, qa_outputs })
    }
}

#[derive(Debug)]
pub struct QuestionAnsweringOutput {
    pub start_logits: Tensor,
    pub end_logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for BertForQuestionAnswering {
    type Config = BertConfig;
    type Input = TokenizedInput;
    type Output = QuestionAnsweringOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let bert_output = self.bert.forward(input)?;
        let sequence_output = bert_output.last_hidden_state;

        let logits = self.qa_outputs.forward(sequence_output.clone())?;

        // Split logits into start and end logits along the last dimension (dimension with size 2)
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
        self.bert.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.qa_outputs.parameter_count()
    }
}
