#![allow(dead_code)]

use crate::bert::config::BertConfig;
use crate::bert::model::BertModel;
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport};
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

/// Bind a `[out, in]` head projection and its bias from a checkpoint.
///
/// A head that the checkpoint does not carry is legitimate — a pretrained
/// backbone ships without a fine-tuned head — but it is recorded in
/// [`LoadReport::missing`] so the caller can see that the layer kept its random
/// initialisation. A head that *is* present must reach the model: silently
/// leaving it behind is exactly the failure this crate is being cleaned of.
///
/// # Errors
///
/// Fails when a tensor is present with the wrong shape.
fn bind_head_linear(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    name: &str,
    weight_shape: [usize; 2],
    layer: &mut Linear,
) -> Result<()> {
    let weight_name = format!("{name}.weight");
    match checkpoint.take_shaped(&weight_name, &weight_shape)? {
        Some(weight) => {
            layer.set_weight(weight)?;
            report.mark_loaded(&weight_name);
        },
        None => report.note_absent(&weight_name),
    }

    let bias_name = format!("{name}.bias");
    match checkpoint.take_shaped(&bias_name, &[weight_shape[0]])? {
        Some(bias) => {
            layer.set_bias(bias)?;
            report.mark_loaded(&bias_name);
        },
        None => report.note_absent(&bias_name),
    }
    Ok(())
}

/// Bind a head layer norm from a checkpoint, recording what was found.
///
/// # Errors
///
/// Fails when a tensor is present with the wrong shape.
fn bind_head_layer_norm(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    name: &str,
    hidden_size: usize,
    norm: &mut trustformers_core::layers::LayerNorm,
) -> Result<()> {
    let weight_name = format!("{name}.weight");
    match checkpoint.take_shaped(&weight_name, &[hidden_size])? {
        Some(weight) => {
            norm.set_weight(weight)?;
            report.mark_loaded(&weight_name);
        },
        None => report.note_absent(&weight_name),
    }

    let bias_name = format!("{name}.bias");
    match checkpoint.take_shaped(&bias_name, &[hidden_size])? {
        Some(bias) => {
            norm.set_bias(bias)?;
            report.mark_loaded(&bias_name);
        },
        None => report.note_absent(&bias_name),
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl BertForSequenceClassification {
    pub fn new(config: BertConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(config: BertConfig, num_labels: usize, device: Device) -> Result<Self> {
        let bert = BertModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            bert,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The classification head's weight matrix.
    pub fn classifier_weight(&self) -> &Tensor {
        self.classifier.weight()
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
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.classifier.parameter_count()
    }
}

impl BertForSequenceClassification {
    /// Load the encoder and, when the checkpoint carries one, the classifier head.
    ///
    /// A previous revision delegated straight to `BertModel::load_pretrained`,
    /// which loads only the encoder. A fine-tuned checkpoint's `classifier.*`
    /// tensors were therefore dropped and inference ran through a
    /// `Tensor::randn` classifier while `load_pretrained` reported success.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when the head is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.bert.load_from_checkpoint(&checkpoint)?;
        let hidden = self.bert.get_config().hidden_size;
        bind_head_linear(
            &checkpoint,
            &mut report,
            "classifier",
            [self.num_labels, hidden],
            &mut self.classifier,
        )?;
        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertLMHead,
    device: Device,
}

impl BertForMaskedLM {
    pub fn new(config: BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: BertConfig, device: Device) -> Result<Self> {
        let bert = BertModel::new_with_device(config.clone(), device)?;
        let cls = BertLMHead::new_with_device(&config, device)?;

        Ok(Self { bert, cls, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[derive(Debug, Clone)]
struct BertLMHead {
    dense: Linear,
    layer_norm: trustformers_core::layers::LayerNorm,
    decoder: Linear,
    device: Device,
}

impl BertLMHead {
    fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            layer_norm: trustformers_core::layers::LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            decoder: Linear::new_with_device(config.hidden_size, config.vocab_size, true, device),
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
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

    /// Bind the masked-LM head from a checkpoint, if it carries one.
    ///
    /// # Errors
    ///
    /// Fails when a head tensor is present with the wrong shape.
    fn load_weights(
        &mut self,
        checkpoint: &Checkpoint,
        report: &mut LoadReport,
        config: &BertConfig,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        bind_head_linear(
            checkpoint,
            report,
            "cls.predictions.transform.dense",
            [hidden, hidden],
            &mut self.dense,
        )?;
        bind_head_layer_norm(
            checkpoint,
            report,
            "cls.predictions.transform.LayerNorm",
            hidden,
            &mut self.layer_norm,
        )?;

        // The decoder is usually tied to the input embedding table, in which case
        // the checkpoint stores only its bias under `cls.predictions.bias`.
        let decoder_weight = "cls.predictions.decoder.weight";
        match checkpoint.take_shaped(decoder_weight, &[config.vocab_size, hidden])? {
            Some(weight) => {
                self.decoder.set_weight(weight)?;
                report.mark_loaded(decoder_weight);
            },
            None => {
                let tied = [
                    "bert.embeddings.word_embeddings.weight",
                    "embeddings.word_embeddings.weight",
                ]
                .into_iter()
                .find(|name| checkpoint.contains(name));
                match tied {
                    Some(name) => {
                        if let Some(weight) =
                            checkpoint.take_shaped(name, &[config.vocab_size, hidden])?
                        {
                            self.decoder.set_weight(weight)?;
                        }
                    },
                    None => report.note_absent(decoder_weight),
                }
            },
        }

        for bias_name in ["cls.predictions.decoder.bias", "cls.predictions.bias"] {
            if let Some(bias) = checkpoint.take_shaped(bias_name, &[config.vocab_size])? {
                self.decoder.set_bias(bias)?;
                report.mark_loaded(bias_name);
                return Ok(());
            }
        }
        report.note_absent("cls.predictions.bias");
        Ok(())
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
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.cls.parameter_count()
    }
}

impl BertForMaskedLM {
    /// Load the encoder and, when the checkpoint carries one, the MLM head.
    ///
    /// HuggingFace stores the head as `cls.predictions.transform.dense.*`,
    /// `cls.predictions.transform.LayerNorm.*` and `cls.predictions.decoder.*`
    /// (with `cls.predictions.bias` as the decoder bias when the decoder itself
    /// is tied to the embedding table).
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when a head tensor is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.bert.load_from_checkpoint(&checkpoint)?;
        let config = self.bert.get_config().clone();
        self.cls.load_weights(&checkpoint, &mut report, &config)?;
        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct BertForTokenClassification {
    bert: BertModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl BertForTokenClassification {
    pub fn new(config: BertConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(config: BertConfig, num_labels: usize, device: Device) -> Result<Self> {
        let bert = BertModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            bert,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.classifier.parameter_count()
    }
}

impl BertForTokenClassification {
    /// Load the encoder and, when present, the token-classification head.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when the head is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.bert.load_from_checkpoint(&checkpoint)?;
        let hidden = self.bert.get_config().hidden_size;
        bind_head_linear(
            &checkpoint,
            &mut report,
            "classifier",
            [self.num_labels, hidden],
            &mut self.classifier,
        )?;
        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct BertForQuestionAnswering {
    bert: BertModel,
    qa_outputs: Linear,
    device: Device,
}

impl BertForQuestionAnswering {
    pub fn new(config: BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: BertConfig, device: Device) -> Result<Self> {
        let bert = BertModel::new_with_device(config.clone(), device)?;
        // QA outputs has 2 classes: start and end positions
        let qa_outputs = Linear::new_with_device(config.hidden_size, 2, true, device);

        Ok(Self {
            bert,
            qa_outputs,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.bert.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.bert.num_parameters() + self.qa_outputs.parameter_count()
    }
}

impl BertForQuestionAnswering {
    /// Load the encoder and, when present, the span-prediction head.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when the head is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.bert.load_from_checkpoint(&checkpoint)?;
        let hidden = self.bert.get_config().hidden_size;
        bind_head_linear(
            &checkpoint,
            &mut report,
            "qa_outputs",
            [2, hidden],
            &mut self.qa_outputs,
        )?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::{Model, TokenizedInput};

    /// Tiny BertConfig for fast tests (no pooler issue workaround: we test
    /// tasks that directly use the last_hidden_state).
    fn tiny_config() -> BertConfig {
        BertConfig {
            vocab_size: 256,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 128,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(false),
            classifier_dropout: None,
        }
    }

    fn make_input(seq_len: usize) -> TokenizedInput {
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let attention_mask: Vec<u8> = vec![1u8; seq_len];
        TokenizedInput::new(input_ids, attention_mask)
    }

    // --- BertForTokenClassification ---

    #[test]
    fn test_token_classification_new() {
        let cfg = tiny_config();
        let model = BertForTokenClassification::new(cfg, 5)
            .expect("BertForTokenClassification::new must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    #[test]
    fn test_token_classification_output_shape() {
        let cfg = tiny_config();
        let num_labels = 5usize;
        let seq_len = 6usize;
        let model = BertForTokenClassification::new(cfg, num_labels)
            .expect("BertForTokenClassification::new must succeed");
        let output = model
            .forward(make_input(seq_len))
            .expect("BertForTokenClassification forward must succeed");
        let shape = output.logits.shape();
        // last_hidden_state is [1, seq_len, hidden_size]; classifier produces [1, seq_len, num_labels]
        // but since last_hidden_state feeds directly as 3D, the shape should end with num_labels
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            num_labels,
            "final logits dim must equal num_labels"
        );
    }

    #[test]
    fn test_token_classification_hidden_states_present() {
        let cfg = tiny_config();
        let model = BertForTokenClassification::new(cfg, 3)
            .expect("BertForTokenClassification::new must succeed");
        let output = model.forward(make_input(4)).expect("forward must succeed");
        assert!(
            output.hidden_states.is_some(),
            "hidden_states must be returned"
        );
    }

    #[test]
    fn test_token_classification_num_parameters_positive() {
        let cfg = tiny_config();
        let model = BertForTokenClassification::new(cfg, 4)
            .expect("BertForTokenClassification::new must succeed");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_token_classification_get_config() {
        let cfg = tiny_config();
        let model = BertForTokenClassification::new(cfg.clone(), 2)
            .expect("BertForTokenClassification::new must succeed");
        let c = model.get_config();
        assert_eq!(c.vocab_size, cfg.vocab_size);
    }

    // --- BertForMaskedLM ---

    #[test]
    fn test_masked_lm_new() {
        let cfg = tiny_config();
        let model = BertForMaskedLM::new(cfg).expect("BertForMaskedLM::new must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    #[test]
    fn test_masked_lm_output_last_dim_is_vocab_size() {
        let cfg = tiny_config();
        let vocab_size = cfg.vocab_size;
        let model = BertForMaskedLM::new(cfg).expect("BertForMaskedLM::new must succeed");
        let output = model.forward(make_input(4)).expect("BertForMaskedLM forward must succeed");
        let shape = output.logits.shape();
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            vocab_size,
            "BertForMaskedLM final logits dim must equal vocab_size"
        );
    }

    #[test]
    fn test_masked_lm_output_seq_len_preserved() {
        let cfg = tiny_config();
        let seq_len = 5usize;
        let model = BertForMaskedLM::new(cfg).expect("BertForMaskedLM::new must succeed");
        let output = model
            .forward(make_input(seq_len))
            .expect("BertForMaskedLM forward must succeed");
        let shape = output.logits.shape();
        // shape is [1, seq_len, vocab_size] or [seq_len, vocab_size]
        // sequence dimension must contain seq_len
        assert!(
            shape.contains(&seq_len),
            "seq_len must appear in BertForMaskedLM logits shape, got {:?}",
            shape
        );
    }

    #[test]
    fn test_masked_lm_num_parameters_positive() {
        let cfg = tiny_config();
        let model = BertForMaskedLM::new(cfg).expect("BertForMaskedLM::new must succeed");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_masked_lm_hidden_states_present() {
        let cfg = tiny_config();
        let model = BertForMaskedLM::new(cfg).expect("BertForMaskedLM::new must succeed");
        let output = model.forward(make_input(3)).expect("forward must succeed");
        assert!(
            output.hidden_states.is_some(),
            "hidden_states must be returned"
        );
    }

    // --- BertForQuestionAnswering ---

    #[test]
    fn test_qa_new() {
        let cfg = tiny_config();
        let model =
            BertForQuestionAnswering::new(cfg).expect("BertForQuestionAnswering::new must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    #[test]
    fn test_qa_start_logits_shape() {
        let cfg = tiny_config();
        let seq_len = 6usize;
        let model =
            BertForQuestionAnswering::new(cfg).expect("BertForQuestionAnswering::new must succeed");
        let output = model
            .forward(make_input(seq_len))
            .expect("BertForQuestionAnswering forward must succeed");
        // start_logits shape must contain seq_len
        let shape = output.start_logits.shape();
        assert!(
            shape.contains(&seq_len),
            "start_logits must cover seq_len positions, got shape {:?}",
            shape
        );
    }

    #[test]
    fn test_qa_end_logits_shape_matches_start() {
        let cfg = tiny_config();
        let seq_len = 6usize;
        let model =
            BertForQuestionAnswering::new(cfg).expect("BertForQuestionAnswering::new must succeed");
        let output = model
            .forward(make_input(seq_len))
            .expect("BertForQuestionAnswering forward must succeed");
        assert_eq!(
            output.start_logits.shape(),
            output.end_logits.shape(),
            "start_logits and end_logits must have the same shape"
        );
    }

    #[test]
    fn test_qa_num_parameters_positive() {
        let cfg = tiny_config();
        let model =
            BertForQuestionAnswering::new(cfg).expect("BertForQuestionAnswering::new must succeed");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_qa_hidden_states_present() {
        let cfg = tiny_config();
        let model =
            BertForQuestionAnswering::new(cfg).expect("BertForQuestionAnswering::new must succeed");
        let output = model.forward(make_input(4)).expect("forward must succeed");
        assert!(
            output.hidden_states.is_some(),
            "hidden_states must be returned"
        );
    }

    #[test]
    fn test_qa_get_config() {
        let cfg = tiny_config();
        let model = BertForQuestionAnswering::new(cfg.clone())
            .expect("BertForQuestionAnswering::new must succeed");
        let c = model.get_config();
        assert_eq!(c.hidden_size, cfg.hidden_size);
    }
}
