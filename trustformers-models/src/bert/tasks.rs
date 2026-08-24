#![allow(dead_code)]

use crate::bert::config::BertConfig;
use crate::bert::model::BertModel;
use crate::weight_loading::binding::{bind_head_layer_norm, bind_head_linear};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport};
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

/// The checkpoint namespace HuggingFace nests the encoder under for every BERT
/// task model.
///
/// `BertModel` itself exports its parameters unprefixed; `BertForMaskedLM`,
/// `BertForSequenceClassification` and friends write `bert.embeddings.…`. The
/// loader detects either spelling, but the *published* names have to commit to
/// one, and the task models' is `bert.`.
const BACKBONE_NAMESPACE: &str = "bert";

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

    /// Publish the encoder under `bert.` plus the classification head.
    ///
    /// These are the names [`BertForSequenceClassification::load_pretrained_report`]
    /// binds: the encoder is nested under `bert.` (which
    /// [`BertModel::load_from_checkpoint`] detects) and the head is
    /// `classifier.{weight,bias}` at the root, exactly as HuggingFace writes it.
    ///
    /// Without this override the default empty `named_tensors` made every
    /// exporter refuse the model, so a fine-tuned classifier could be loaded but
    /// never written back out.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters(BACKBONE_NAMESPACE, &mut tensors);
        self.classifier.collect_named_parameters("classifier", &mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters_mut(BACKBONE_NAMESPACE, &mut tensors);
        self.classifier.collect_named_parameters_mut("classifier", &mut tensors);
        tensors
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

    /// Publish the masked-LM head under HuggingFace's `cls.predictions.…` names.
    ///
    /// # Why the bias is not `cls.predictions.decoder.bias`
    ///
    /// HuggingFace declares the decoder with `bias=False` and then aliases
    /// `decoder.bias` to a separate `cls.predictions.bias` parameter, so a real
    /// checkpoint carries *both* names for one tensor. `named_tensors` must not
    /// publish a duplicate, so the canonical `cls.predictions.bias` is the one
    /// emitted — and [`BertLMHead::load_weights`] accepts either spelling on the
    /// way back in.
    fn collect_named_parameters<'a>(&'a self, into: &mut Vec<(String, &'a Tensor)>) {
        self.dense.collect_named_parameters("cls.predictions.transform.dense", into);
        self.layer_norm
            .collect_named_parameters("cls.predictions.transform.LayerNorm", into);
        into.push((
            "cls.predictions.decoder.weight".to_string(),
            self.decoder.weight(),
        ));
        if let Some(bias) = self.decoder.bias() {
            into.push(("cls.predictions.bias".to_string(), bias));
        }
    }

    /// Mutable counterpart of [`BertLMHead::collect_named_parameters`].
    fn collect_named_parameters_mut<'a>(&'a mut self, into: &mut Vec<(String, &'a mut Tensor)>) {
        self.dense.collect_named_parameters_mut("cls.predictions.transform.dense", into);
        self.layer_norm
            .collect_named_parameters_mut("cls.predictions.transform.LayerNorm", into);
        // One call, because two successive `*_mut()` accessors would each borrow
        // all of `self.decoder`.
        let (weight, bias) = self.decoder.parameters_mut();
        into.push(("cls.predictions.decoder.weight".to_string(), weight));
        if let Some(bias) = bias {
            into.push(("cls.predictions.bias".to_string(), bias));
        }
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

    /// Publish the encoder under `bert.` plus the `cls.predictions.…` MLM head.
    ///
    /// See [`BertLMHead::collect_named_parameters`] for why the decoder bias is
    /// published once, as `cls.predictions.bias`.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters(BACKBONE_NAMESPACE, &mut tensors);
        self.cls.collect_named_parameters(&mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters_mut(BACKBONE_NAMESPACE, &mut tensors);
        self.cls.collect_named_parameters_mut(&mut tensors);
        tensors
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

    /// Publish the encoder under `bert.` plus the token-classification head,
    /// which HuggingFace also spells `classifier.{weight,bias}`.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters(BACKBONE_NAMESPACE, &mut tensors);
        self.classifier.collect_named_parameters("classifier", &mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters_mut(BACKBONE_NAMESPACE, &mut tensors);
        self.classifier.collect_named_parameters_mut("classifier", &mut tensors);
        tensors
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

    /// Publish the encoder under `bert.` plus the span-prediction head, which
    /// HuggingFace names `qa_outputs.{weight,bias}`.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters(BACKBONE_NAMESPACE, &mut tensors);
        self.qa_outputs.collect_named_parameters("qa_outputs", &mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::new();
        self.bert.collect_named_parameters_mut(BACKBONE_NAMESPACE, &mut tensors);
        self.qa_outputs.collect_named_parameters_mut("qa_outputs", &mut tensors);
        tensors
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
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};
    use std::collections::{BTreeMap, HashSet};
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

    // --- Model::named_tensors ---
    //
    // Every test below fails against the previous revision, where the four task
    // wrappers inherited `Model::named_tensors`' empty default: the encoder and
    // head were unreachable by name, so no exporter, checkpoint writer or
    // compressor could touch a fine-tuned BERT.

    /// A config small enough that a full round trip through a real safetensors
    /// container runs in milliseconds.
    fn round_trip_config() -> BertConfig {
        BertConfig {
            vocab_size: 12,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 8,
            ..tiny_config()
        }
    }

    fn published_names<M: Model>(model: &M) -> Vec<String> {
        model.named_tensors().into_iter().map(|(name, _)| name).collect()
    }

    /// Overwrite every published parameter with a ramp whose base depends on the
    /// tensor's position and name, so two same-shaped tensors never hold the
    /// same values and a mix-up shows up as a value mismatch.
    fn install_known_weights<M: Model>(model: &mut M, seed: f32) {
        for (index, (name, parameter)) in model.named_tensors_mut().into_iter().enumerate() {
            let base = seed + (index as f32) * 13.0 + (name.len() as f32) * 0.25;
            let values: Vec<f32> =
                (0..parameter.len()).map(|offset| base + (offset as f32) * 0.03125).collect();
            let shape = parameter.shape();
            *parameter =
                Tensor::from_vec(values, &shape).expect("a regenerated tensor keeps its shape");
        }
    }

    fn snapshot<M: Model>(model: &M) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        model
            .named_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name,
                    (
                        tensor.shape(),
                        tensor.to_vec_f32().expect("test fixtures are all F32"),
                    ),
                )
            })
            .collect()
    }

    /// Turn a model's published parameters into a real safetensors byte stream
    /// keyed by exactly the names it published.
    fn checkpoint_from_published<M: Model>(model: &M) -> Vec<u8> {
        let tensors: Vec<F32Tensor> = model
            .named_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                F32Tensor::new(
                    &name,
                    &tensor.shape(),
                    tensor.to_vec_f32().expect("test fixtures are all F32"),
                )
            })
            .collect();
        build_safetensors(&tensors)
    }

    #[test]
    fn task_heads_publish_the_encoder_under_the_bert_namespace() {
        let config = round_trip_config();
        let sequence = BertForSequenceClassification::new(config.clone(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        let token = BertForTokenClassification::new(config.clone(), 3)
            .expect("BertForTokenClassification::new must succeed");
        let qa = BertForQuestionAnswering::new(config.clone())
            .expect("BertForQuestionAnswering::new must succeed");
        let masked = BertForMaskedLM::new(config).expect("BertForMaskedLM::new must succeed");

        for (label, names) in [
            ("sequence_classification", published_names(&sequence)),
            ("token_classification", published_names(&token)),
            ("question_answering", published_names(&qa)),
            ("masked_lm", published_names(&masked)),
        ] {
            assert!(
                !names.is_empty(),
                "{label}: a model that publishes nothing cannot be exported"
            );
            let unique: HashSet<&String> = names.iter().collect();
            assert_eq!(
                unique.len(),
                names.len(),
                "{label}: published names must be unique"
            );
            for required in [
                "bert.embeddings.word_embeddings.weight",
                "bert.encoder.layer.0.attention.self.query.weight",
                "bert.encoder.layer.0.output.LayerNorm.bias",
                "bert.pooler.dense.weight",
            ] {
                assert!(
                    names.iter().any(|name| name == required),
                    "{label}: missing encoder parameter '{required}'"
                );
            }
            assert!(
                !names.iter().any(|name| name.starts_with("embeddings.")),
                "{label}: the encoder must appear only under the `bert.` namespace"
            );
        }
    }

    #[test]
    fn task_heads_publish_their_huggingface_head_names() {
        let config = round_trip_config();

        let sequence = BertForSequenceClassification::new(config.clone(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        let sequence_names: HashSet<String> = published_names(&sequence).into_iter().collect();
        assert!(sequence_names.contains("classifier.weight"));
        assert!(sequence_names.contains("classifier.bias"));

        let token = BertForTokenClassification::new(config.clone(), 3)
            .expect("BertForTokenClassification::new must succeed");
        let token_names: HashSet<String> = published_names(&token).into_iter().collect();
        assert!(token_names.contains("classifier.weight"));

        let qa = BertForQuestionAnswering::new(config.clone())
            .expect("BertForQuestionAnswering::new must succeed");
        let qa_names: HashSet<String> = published_names(&qa).into_iter().collect();
        assert!(qa_names.contains("qa_outputs.weight"));
        assert!(qa_names.contains("qa_outputs.bias"));
        assert!(
            !qa_names.contains("classifier.weight"),
            "the span head is `qa_outputs`, not `classifier`"
        );

        let masked = BertForMaskedLM::new(config).expect("BertForMaskedLM::new must succeed");
        let masked_names: HashSet<String> = published_names(&masked).into_iter().collect();
        for required in [
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.transform.LayerNorm.bias",
            "cls.predictions.decoder.weight",
            "cls.predictions.bias",
        ] {
            assert!(
                masked_names.contains(required),
                "masked LM head missing '{required}'"
            );
        }
        assert!(
            !masked_names.contains("cls.predictions.decoder.bias"),
            "the decoder bias is tied to `cls.predictions.bias` and must be published once"
        );
    }

    #[test]
    fn named_tensors_mut_reaches_the_live_head() {
        let mut model = BertForSequenceClassification::new(round_trip_config(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        install_known_weights(&mut model, 0.5);
        let before = snapshot(&model);

        for (name, parameter) in model.named_tensors_mut() {
            if name == "classifier.bias" {
                *parameter = Tensor::from_vec(vec![7.0; parameter.len()], &parameter.shape())
                    .expect("shape preserved");
            }
        }

        let after = snapshot(&model);
        assert_eq!(
            after.get("classifier.bias").map(|(_, values)| values.clone()),
            Some(vec![7.0; 3]),
            "a write through named_tensors_mut must reach the live head"
        );
        assert_eq!(
            after.get("bert.embeddings.word_embeddings.weight"),
            before.get("bert.embeddings.word_embeddings.weight"),
            "an unrelated parameter must not change"
        );
        assert_eq!(
            model.classifier_weight().to_vec_f32().expect("F32"),
            before
                .get("classifier.weight")
                .map(|(_, values)| values.clone())
                .expect("classifier weight is published"),
            "the published tensor and the layer's own accessor must be the same parameter"
        );
    }

    /// The strongest form of the naming claim: write a **real safetensors
    /// checkpoint** keyed by exactly the names each task model publishes, load it
    /// back through `Model::load_pretrained`, and check every value arrives.
    ///
    /// A published name the loader does not bind would either be rejected as an
    /// unrecognised tensor or leave the parameter untouched, and either way this
    /// test goes red.
    #[test]
    fn published_names_load_back_through_load_pretrained() {
        let config = round_trip_config();

        // Sequence classification: encoder under `bert.` plus `classifier.*`.
        let mut source = BertForSequenceClassification::new(config.clone(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        install_known_weights(&mut source, 4.25);
        let expected = snapshot(&source);
        let bytes = checkpoint_from_published(&source);

        let mut loaded = BertForSequenceClassification::new(config.clone(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        install_known_weights(&mut loaded, 900.0);
        assert_ne!(
            snapshot(&loaded),
            expected,
            "the destination must start out different, or the test proves nothing"
        );
        loaded
            .load_pretrained(&mut std::io::Cursor::new(bytes))
            .expect("a checkpoint keyed by named_tensors' own names must load");
        assert_eq!(snapshot(&loaded), expected);

        // Question answering: the same encoder plus `qa_outputs.*`.
        let mut qa_source = BertForQuestionAnswering::new(config.clone())
            .expect("BertForQuestionAnswering::new must succeed");
        install_known_weights(&mut qa_source, 6.5);
        let qa_expected = snapshot(&qa_source);
        let qa_bytes = checkpoint_from_published(&qa_source);

        let mut qa_loaded = BertForQuestionAnswering::new(config.clone())
            .expect("BertForQuestionAnswering::new must succeed");
        install_known_weights(&mut qa_loaded, 700.0);
        qa_loaded
            .load_pretrained(&mut std::io::Cursor::new(qa_bytes))
            .expect("the QA head's published names must load");
        assert_eq!(snapshot(&qa_loaded), qa_expected);

        // Masked LM: the `cls.predictions.*` head, including the tied bias name.
        let mut mlm_source =
            BertForMaskedLM::new(config.clone()).expect("BertForMaskedLM::new must succeed");
        install_known_weights(&mut mlm_source, 8.75);
        let mlm_expected = snapshot(&mlm_source);
        let mlm_bytes = checkpoint_from_published(&mlm_source);

        let mut mlm_loaded =
            BertForMaskedLM::new(config).expect("BertForMaskedLM::new must succeed");
        install_known_weights(&mut mlm_loaded, 500.0);
        mlm_loaded
            .load_pretrained(&mut std::io::Cursor::new(mlm_bytes))
            .expect("the masked-LM head's published names must load");
        assert_eq!(snapshot(&mlm_loaded), mlm_expected);
    }

    /// End-to-end proof that a fine-tuned task model is now exportable: push a
    /// `BertForSequenceClassification` through the **real** GGUF exporter and
    /// re-read the file with the **real** reader.
    ///
    /// Against the previous revision the exporter refused outright ("exposes no
    /// named tensors"), because the task wrappers published nothing.
    #[test]
    fn sequence_classification_round_trips_through_the_real_gguf_exporter() {
        use trustformers_core::export::gguf_format::read_gguf_file;
        use trustformers_core::export::{ExportConfig, ExportFormat, GGUFExporter, ModelExporter};

        let directory = std::env::temp_dir().join("trustformers_bert_task_gguf_round_trip");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("scratch directory must be creatable");
        let output = directory.join("tiny_bert_sequence_classification");

        let mut model = BertForSequenceClassification::new(round_trip_config(), 3)
            .expect("BertForSequenceClassification::new must succeed");
        install_known_weights(&mut model, 2.75);
        let expected = snapshot(&model);

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&model, &config).expect("GGUF export must succeed");

        let path = output.with_extension("gguf");
        let parsed = read_gguf_file(&path).expect("the real GGUF reader must parse our own output");
        assert_eq!(
            parsed.tensors.len(),
            expected.len(),
            "every published parameter must reach the file"
        );
        for (name, (shape, values)) in &expected {
            let (info, _) =
                parsed.tensor(name).unwrap_or_else(|| panic!("'{name}' missing from the file"));
            // GGUF stores dimensions fastest-varying first, i.e. reversed.
            let file_shape: Vec<usize> =
                info.dimensions.iter().rev().map(|&dimension| dimension as usize).collect();
            assert_eq!(
                &file_shape, shape,
                "shape of '{name}' changed on the way out"
            );
            assert_eq!(
                &parsed.tensor_f32(name).expect("tensor data must decode"),
                values,
                "values of '{name}' changed"
            );
        }

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// `num_parameters` counts what `named_tensors` publishes — if the two
    /// disagree, one of them is describing a model that does not exist.
    #[test]
    fn published_parameters_account_for_num_parameters() {
        let config = round_trip_config();
        let models: Vec<(&str, usize, usize)> = {
            let sequence = BertForSequenceClassification::new(config.clone(), 3)
                .expect("BertForSequenceClassification::new must succeed");
            let token = BertForTokenClassification::new(config.clone(), 3)
                .expect("BertForTokenClassification::new must succeed");
            let qa = BertForQuestionAnswering::new(config.clone())
                .expect("BertForQuestionAnswering::new must succeed");
            let masked = BertForMaskedLM::new(config).expect("BertForMaskedLM::new must succeed");
            vec![
                (
                    "sequence_classification",
                    sequence.num_parameters(),
                    sequence.named_tensors().iter().map(|(_, t)| t.len()).sum(),
                ),
                (
                    "token_classification",
                    token.num_parameters(),
                    token.named_tensors().iter().map(|(_, t)| t.len()).sum(),
                ),
                (
                    "question_answering",
                    qa.num_parameters(),
                    qa.named_tensors().iter().map(|(_, t)| t.len()).sum(),
                ),
                (
                    "masked_lm",
                    masked.num_parameters(),
                    masked.named_tensors().iter().map(|(_, t)| t.len()).sum(),
                ),
            ]
        };

        for (label, counted, published) in models {
            assert_eq!(
                counted, published,
                "{label}: num_parameters() and named_tensors() disagree"
            );
        }
    }
}
