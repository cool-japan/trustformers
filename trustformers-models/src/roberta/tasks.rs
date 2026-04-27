#![allow(dead_code)]
#![allow(unused_imports)]

use crate::roberta::config::RobertaConfig;
use crate::roberta::model::RobertaModel;
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::layers::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RobertaForSequenceClassification {
    roberta: RobertaModel,
    classifier: RobertaClassificationHead,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl RobertaForSequenceClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: RobertaConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let roberta = RobertaModel::new_with_device(config.clone(), device)?;
        let classifier = RobertaClassificationHead::new_with_device(&config, num_labels, device)?;

        Ok(Self {
            roberta,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[derive(Debug, Clone)]
struct RobertaClassificationHead {
    dense: Linear,
    dropout: f32,
    out_proj: Linear,
    device: Device,
}

impl RobertaClassificationHead {
    fn new(config: &RobertaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    fn new_with_device(config: &RobertaConfig, num_labels: usize, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            dropout: config.classifier_dropout.unwrap_or(config.hidden_dropout_prob),
            out_proj: Linear::new_with_device(config.hidden_size, num_labels, true, device),
            device,
        })
    }

    fn device(&self) -> Device {
        self.device
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
#[allow(dead_code)]
pub struct RobertaForMaskedLM {
    roberta: RobertaModel,
    lm_head: RobertaLMHead,
    device: Device,
}

impl RobertaForMaskedLM {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RobertaConfig, device: Device) -> Result<Self> {
        let roberta = RobertaModel::new_with_device(config.clone(), device)?;
        let lm_head = RobertaLMHead::new_with_device(&config, device)?;

        Ok(Self {
            roberta,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[derive(Debug, Clone)]
struct RobertaLMHead {
    dense: Linear,
    layer_norm: trustformers_core::layers::LayerNorm,
    decoder: Linear,
    device: Device,
}

impl RobertaLMHead {
    fn new(config: &RobertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &RobertaConfig, device: Device) -> Result<Self> {
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
    device: Device,
}

impl RobertaForTokenClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: RobertaConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let roberta = RobertaModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            roberta,
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
    device: Device,
}

impl RobertaForQuestionAnswering {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RobertaConfig, device: Device) -> Result<Self> {
        let roberta = RobertaModel::new_with_device(config.clone(), device)?;
        let qa_outputs = Linear::new_with_device(config.hidden_size, 2, true, device);

        Ok(Self {
            roberta,
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

#[cfg(test)]
mod tests {
    use crate::roberta::config::RobertaConfig;
    use trustformers_core::traits::Config;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    fn make_small_roberta_config() -> RobertaConfig {
        RobertaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 64,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        }
    }

    #[test]
    fn test_roberta_config_default_fields() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.vocab_size, 50265);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.pad_token_id, 1);
        assert_eq!(cfg.bos_token_id, 0);
        assert_eq!(cfg.eos_token_id, 2);
    }

    #[test]
    fn test_roberta_config_base_validates() {
        let cfg = RobertaConfig::roberta_base();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_roberta_config_large_validates() {
        let cfg = RobertaConfig::roberta_large();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_architecture_name() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.architecture(), "RoBERTa");
    }

    #[test]
    fn test_hidden_not_divisible_by_heads_fails() {
        let cfg = RobertaConfig {
            hidden_size: 100,
            num_attention_heads: 12,
            ..RobertaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_small_config_validates() {
        let cfg = make_small_roberta_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_small_config_hidden_divisible_by_heads() {
        let cfg = make_small_roberta_config();
        assert_eq!(cfg.hidden_size % cfg.num_attention_heads, 0);
    }

    #[test]
    fn test_position_embedding_type_default() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.position_embedding_type.as_deref(), Some("absolute"));
    }

    #[test]
    fn test_use_cache_default() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.use_cache, Some(true));
    }

    #[test]
    fn test_classifier_dropout_default_none() {
        let cfg = RobertaConfig::default();
        assert!(cfg.classifier_dropout.is_none());
    }

    #[test]
    fn test_type_vocab_size_default() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.type_vocab_size, 1);
    }

    #[test]
    fn test_large_has_more_layers_than_base() {
        let base = RobertaConfig::roberta_base();
        let large = RobertaConfig::roberta_large();
        assert!(large.num_hidden_layers > base.num_hidden_layers);
    }

    #[test]
    fn test_large_has_more_heads_than_base() {
        let base = RobertaConfig::roberta_base();
        let large = RobertaConfig::roberta_large();
        assert!(large.num_attention_heads > base.num_attention_heads);
    }

    #[test]
    fn test_large_has_larger_hidden_size() {
        let base = RobertaConfig::roberta_base();
        let large = RobertaConfig::roberta_large();
        assert!(large.hidden_size > base.hidden_size);
    }

    #[test]
    fn test_layer_norm_eps_default() {
        let cfg = RobertaConfig::default();
        assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_initializer_range_default() {
        let cfg = RobertaConfig::default();
        assert!((cfg.initializer_range - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_defaults() {
        let cfg = RobertaConfig::default();
        assert!((cfg.hidden_dropout_prob - 0.1).abs() < 1e-6);
        assert!((cfg.attention_probs_dropout_prob - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_max_position_embeddings_default() {
        let cfg = RobertaConfig::default();
        assert_eq!(cfg.max_position_embeddings, 514);
    }

    #[test]
    fn test_intermediate_size_base() {
        let cfg = RobertaConfig::roberta_base();
        assert_eq!(cfg.intermediate_size, 3072);
    }

    #[test]
    fn test_lcg_values_in_range() {
        let mut rng = Lcg::new(271828);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
