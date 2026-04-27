use crate::electra::config::ElectraConfig;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

pub struct ElectraForTokenClassification {
    pub electra: crate::electra::model::ElectraDiscriminator,
    pub classifier: trustformers_core::layers::linear::Linear,
    pub dropout: f32,
    pub num_labels: usize,
    device: Device,
}

impl ElectraForTokenClassification {
    pub fn new(config: ElectraConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: ElectraConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            electra: crate::electra::model::ElectraDiscriminator::new_with_device(&config, device)?,
            classifier: trustformers_core::layers::linear::Linear::new_with_device(
                config.discriminator_hidden_size,
                num_labels,
                true,
                device,
            ),
            dropout,
            num_labels,
            device,
        })
    }

    pub fn from_pretrained(model_name: &str, num_labels: usize) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config, num_labels)
    }

    pub fn from_pretrained_with_device(
        model_name: &str,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, num_labels, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        let hidden_states =
            self.electra.forward(input_ids, token_type_ids, position_ids, attention_mask)?;

        // Apply dropout
        let hidden_states = hidden_states * (1.0 - self.dropout);

        // Token classification head
        let classifier_input = Tensor::F32(hidden_states.into_dyn());
        let logits = self.classifier.forward(classifier_input)?;
        let logits = match logits {
            Tensor::F32(arr) => arr.into_dimensionality::<Ix3>().map_err(|e| {
                trustformers_core::errors::TrustformersError::shape_error(e.to_string())
            })?,
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from classifier",
                        "classifier",
                    ),
                )
            },
        };

        Ok(logits)
    }
}

pub struct ElectraForQuestionAnswering {
    pub electra: crate::electra::model::ElectraDiscriminator,
    pub qa_outputs: trustformers_core::layers::linear::Linear,
    pub dropout: f32,
    device: Device,
}

impl ElectraForQuestionAnswering {
    pub fn new(config: ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ElectraConfig, device: Device) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            electra: crate::electra::model::ElectraDiscriminator::new_with_device(&config, device)?,
            qa_outputs: trustformers_core::layers::linear::Linear::new_with_device(
                config.discriminator_hidden_size,
                2,
                true,
                device,
            ), // start and end logits
            dropout,
            device,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn from_pretrained_with_device(model_name: &str, device: Device) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let hidden_states =
            self.electra.forward(input_ids, token_type_ids, position_ids, attention_mask)?;

        // Apply dropout
        let hidden_states = hidden_states * (1.0 - self.dropout);

        // QA head
        let qa_input = Tensor::F32(hidden_states.into_dyn());
        let logits = self.qa_outputs.forward(qa_input)?;
        let logits = match logits {
            Tensor::F32(arr) => arr.into_dimensionality::<Ix3>().map_err(|e| {
                trustformers_core::errors::TrustformersError::shape_error(e.to_string())
            })?,
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from QA outputs",
                        "qa_outputs",
                    ),
                )
            },
        };

        // Split into start and end logits
        let start_logits = logits.slice(s![.., .., 0]).to_owned();
        let end_logits = logits.slice(s![.., .., 1]).to_owned();

        Ok((start_logits, end_logits))
    }
}

pub struct ElectraForMultipleChoice {
    pub electra: crate::electra::model::ElectraDiscriminator,
    pub classifier: trustformers_core::layers::linear::Linear,
    pub dropout: f32,
    device: Device,
}

impl ElectraForMultipleChoice {
    pub fn new(config: ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ElectraConfig, device: Device) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            electra: crate::electra::model::ElectraDiscriminator::new_with_device(&config, device)?,
            classifier: trustformers_core::layers::linear::Linear::new_with_device(
                config.discriminator_hidden_size,
                1,
                true,
                device,
            ),
            dropout,
            device,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn from_pretrained_with_device(model_name: &str, device: Device) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array2<f32>> {
        let hidden_states =
            self.electra.forward(input_ids, token_type_ids, position_ids, attention_mask)?;

        // Use [CLS] token representation (first token)
        let cls_hidden = hidden_states.slice(s![0, 0, ..]).to_owned();

        // Apply dropout
        let cls_hidden = cls_hidden * (1.0 - self.dropout);

        // Classification head
        let cls_input = Tensor::F32(cls_hidden.insert_axis(Axis(0)).into_dyn());
        let logits = self.classifier.forward(cls_input)?;
        let logits = match logits {
            Tensor::F32(arr) => arr.into_dimensionality::<Ix2>().map_err(|e| {
                trustformers_core::errors::TrustformersError::shape_error(e.to_string())
            })?,
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from classifier",
                        "classifier",
                    ),
                )
            },
        };

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electra::model::{ElectraForPreTraining, ElectraForSequenceClassification};
    use trustformers_core::traits::Config;
    // Array1 already imported via scirs2_core at top

    #[test]
    fn test_electra_sequence_classification() {
        let config = ElectraConfig::small();
        let model = ElectraForSequenceClassification::new(config, 2).expect("operation failed");

        let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]); // [CLS] I love ELECTRA [SEP]
        let result = model.forward(&input_ids, None, None, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, 2]);
    }

    #[test]
    fn test_electra_token_classification() {
        let config = ElectraConfig::small();
        let model = ElectraForTokenClassification::new(config, 9).expect("operation failed"); // BIO tagging

        let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
        let result = model.forward(&input_ids, None, None, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, input_ids.len(), 9]);
    }

    #[test]
    fn test_electra_question_answering() {
        let config = ElectraConfig::small();
        let model = ElectraForQuestionAnswering::new(config).expect("operation failed");

        let input_ids = Array1::from_vec(vec![
            101, 2054, 2003, 7570, 1029, 102, 7570, 2003, 2307, 102,
        ]);
        let result = model.forward(&input_ids, None, None, None);

        assert!(result.is_ok());
        let (start_logits, end_logits) = result.expect("operation failed");
        assert_eq!(start_logits.shape(), &[1, input_ids.len()]);
        assert_eq!(end_logits.shape(), &[1, input_ids.len()]);
    }

    #[test]
    fn test_electra_pretraining() {
        let config = ElectraConfig::small();
        let model = ElectraForPreTraining::new(config.clone()).expect("operation failed");

        let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
        let result = model.forward(&input_ids, None, None, None);

        assert!(result.is_ok());
        let (generator_logits, discriminator_logits) = result.expect("operation failed");
        assert_eq!(
            generator_logits.shape(),
            &[1, input_ids.len(), config.vocab_size]
        );
        assert_eq!(discriminator_logits.shape(), &[1, input_ids.len(), 1]);
    }

    // ---- Config: small / base / large sizes ----

    #[test]
    fn test_electra_small_discriminator_hidden_size() {
        let config = ElectraConfig::small();
        assert_eq!(config.discriminator_hidden_size, 256);
    }

    #[test]
    fn test_electra_base_discriminator_hidden_size() {
        let config = ElectraConfig::base();
        assert_eq!(config.discriminator_hidden_size, 768);
    }

    #[test]
    fn test_electra_large_discriminator_hidden_size() {
        let config = ElectraConfig::large();
        assert_eq!(config.discriminator_hidden_size, 1024);
    }

    #[test]
    fn test_electra_config_validation_ok() {
        for config in [
            ElectraConfig::small(),
            ElectraConfig::base(),
            ElectraConfig::large(),
        ] {
            assert!(config.validate().is_ok(), "Config validation must pass");
        }
    }

    #[test]
    fn test_electra_config_architecture_name() {
        let config = ElectraConfig::small();
        assert_eq!(config.architecture(), "ELECTRA");
    }

    // ---- Discriminator RTD output shape ----

    #[test]
    fn test_discriminator_output_binary_per_token() {
        // ElectraForPreTraining discriminator logits: [1, seq_len, 1]
        let config = ElectraConfig::small();
        let model =
            ElectraForPreTraining::new(config.clone()).expect("ElectraForPreTraining construction");
        let seq_len = 8_usize;
        // LCG for token IDs in [101, 2000]
        let mut seed: u64 = 99;
        let token_ids: Vec<u32> = (0..seq_len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                101 + (seed % 1899) as u32
            })
            .collect();
        let input_ids = Array1::from_vec(token_ids);
        let (_, disc_logits) = model
            .forward(&input_ids, None, None, None)
            .expect("discriminator forward failed");
        // Shape [1, seq_len, 1]
        assert_eq!(disc_logits.shape(), &[1, seq_len, 1]);
    }

    #[test]
    fn test_generator_output_vocab_size_axis() {
        let config = ElectraConfig::small();
        let seq_len = 5_usize;
        let model =
            ElectraForPreTraining::new(config.clone()).expect("ElectraForPreTraining construction");
        let input_ids = Array1::from_vec(vec![101u32, 2054, 2003, 7570, 102]);
        let (gen_logits, _) =
            model.forward(&input_ids, None, None, None).expect("generator forward failed");
        // Shape [1, seq_len, vocab_size]
        assert_eq!(gen_logits.shape()[2], config.vocab_size);
        assert_eq!(gen_logits.shape()[1], seq_len);
    }

    // ---- Token classification: BIO tagging per-token ----

    #[test]
    fn test_token_classification_seq_len_preserved() {
        let config = ElectraConfig::small();
        let num_labels = 5_usize;
        let model =
            ElectraForTokenClassification::new(config, num_labels).expect("construction failed");
        let seq_len = 7_usize;
        // LCG-generated token IDs
        let mut seed: u64 = 1234;
        let tokens: Vec<u32> = (0..seq_len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                100 + (seed % 500) as u32
            })
            .collect();
        let input_ids = Array1::from_vec(tokens);
        let logits = model
            .forward(&input_ids, None, None, None)
            .expect("token classification forward failed");
        assert_eq!(logits.shape(), &[1, seq_len, num_labels]);
    }

    #[test]
    fn test_token_classification_label_dimension() {
        // For binary replaced-token detection, use 2 labels
        let config = ElectraConfig::small();
        let model = ElectraForTokenClassification::new(config, 2).expect("construction failed");
        let input_ids = Array1::from_vec(vec![101u32, 100, 200, 102]);
        let logits = model.forward(&input_ids, None, None, None).expect("forward pass failed");
        assert_eq!(logits.shape()[2], 2, "Binary label dimension must be 2");
    }

    // ---- Binary cross-entropy loss shape (per-token) ----

    /// Compute per-token binary cross-entropy from discriminator logits.
    fn bce_per_token(logits: &scirs2_core::ndarray::Array3<f32>, labels: &[f32]) -> Vec<f32> {
        let seq_len = logits.shape()[1];
        (0..seq_len)
            .map(|t| {
                let logit = logits[[0, t, 0]];
                let p = 1.0 / (1.0 + (-logit).exp()); // sigmoid
                let label = labels[t];
                -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
            })
            .collect()
    }

    #[test]
    fn test_bce_loss_nonnegative() {
        let config = ElectraConfig::small();
        let model = ElectraForPreTraining::new(config).expect("ElectraForPreTraining construction");
        let input_ids = Array1::from_vec(vec![101u32, 2054, 2003, 7570, 102]);
        let (_, disc_logits) = model
            .forward(&input_ids, None, None, None)
            .expect("discriminator forward failed");
        // Binary labels: alternating 0/1 (LCG-like pattern)
        let labels = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0];
        let losses = bce_per_token(&disc_logits, &labels);
        for (t, &loss) in losses.iter().enumerate() {
            assert!(loss >= 0.0, "BCE loss at token {} is negative: {}", t, loss);
        }
    }

    #[test]
    fn test_bce_loss_length_matches_seq_len() {
        let config = ElectraConfig::small();
        let seq_len = 6_usize;
        let model = ElectraForPreTraining::new(config).expect("ElectraForPreTraining construction");
        let input_ids = Array1::from_vec(vec![101u32, 1, 2, 3, 4, 102]);
        let (_, disc_logits) = model
            .forward(&input_ids, None, None, None)
            .expect("discriminator forward failed");
        let labels = vec![0.0_f32; seq_len];
        let losses = bce_per_token(&disc_logits, &labels);
        assert_eq!(losses.len(), seq_len);
    }

    // ---- Accuracy computation (per-token binary) ----

    /// Compute per-token binary accuracy from discriminator logits (threshold at 0).
    fn binary_accuracy(logits: &scirs2_core::ndarray::Array3<f32>, labels: &[u32]) -> f32 {
        let seq_len = logits.shape()[1];
        let correct: usize = (0..seq_len)
            .filter(|&t| {
                let pred = if logits[[0, t, 0]] >= 0.0 { 1u32 } else { 0u32 };
                pred == labels[t]
            })
            .count();
        correct as f32 / seq_len as f32
    }

    #[test]
    fn test_binary_accuracy_in_range() {
        let config = ElectraConfig::small();
        let model = ElectraForPreTraining::new(config).expect("ElectraForPreTraining construction");
        let input_ids = Array1::from_vec(vec![101u32, 2054, 2003, 7570, 102]);
        let (_, disc_logits) = model
            .forward(&input_ids, None, None, None)
            .expect("discriminator forward failed");
        let labels = vec![0u32, 1, 0, 1, 0];
        let acc = binary_accuracy(&disc_logits, &labels);
        assert!((0.0..=1.0).contains(&acc), "Accuracy {} out of [0,1]", acc);
    }

    // ---- QA output start/end logits consistency ----

    #[test]
    fn test_qa_start_end_same_seq_len() {
        let config = ElectraConfig::small();
        let model = ElectraForQuestionAnswering::new(config).expect("QA construction failed");
        let input_ids = Array1::from_vec(vec![101u32, 2054, 2003, 7570, 1029, 102]);
        let (start_logits, end_logits) =
            model.forward(&input_ids, None, None, None).expect("QA forward failed");
        assert_eq!(
            start_logits.shape()[1],
            end_logits.shape()[1],
            "Start and end logits must have same seq length"
        );
    }

    #[test]
    fn test_qa_output_batch_dim_one() {
        let config = ElectraConfig::small();
        let model = ElectraForQuestionAnswering::new(config).expect("QA construction failed");
        let input_ids = Array1::from_vec(vec![101u32, 100, 102]);
        let (start_logits, end_logits) =
            model.forward(&input_ids, None, None, None).expect("QA forward failed");
        assert_eq!(start_logits.shape()[0], 1);
        assert_eq!(end_logits.shape()[0], 1);
    }

    // ---- from_pretrained_name routing ----

    #[test]
    fn test_from_pretrained_name_small() {
        let config = ElectraConfig::from_pretrained_name("google/electra-small-discriminator");
        assert_eq!(config.discriminator_hidden_size, 256);
    }

    #[test]
    fn test_from_pretrained_name_large() {
        let config = ElectraConfig::from_pretrained_name("google/electra-large-discriminator");
        assert_eq!(config.discriminator_hidden_size, 1024);
    }

    #[test]
    fn test_from_pretrained_name_default_to_base() {
        let config = ElectraConfig::from_pretrained_name("unknown-model");
        assert_eq!(config.discriminator_hidden_size, 768);
    }

    // ---- Generator hidden size is smaller than discriminator ----

    #[test]
    fn test_generator_smaller_than_discriminator() {
        for config in [
            ElectraConfig::small(),
            ElectraConfig::base(),
            ElectraConfig::large(),
        ] {
            assert!(
                config.generator_hidden_size <= config.discriminator_hidden_size,
                "Generator must be ≤ discriminator in size"
            );
        }
    }
}
