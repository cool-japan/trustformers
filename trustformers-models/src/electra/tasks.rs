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
}
