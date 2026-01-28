use crate::deberta::config::DebertaConfig;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::Result;
use trustformers_core::layers::linear::Linear;
use trustformers_core::ops::activations::gelu;
use trustformers_core::traits::Layer;

pub struct DebertaForTokenClassification {
    pub deberta: crate::deberta::model::DebertaModel,
    pub classifier: Linear,
    pub dropout: f32,
    pub num_labels: usize,
    device: Device,
}

impl DebertaForTokenClassification {
    pub fn new(config: DebertaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: DebertaConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            deberta: crate::deberta::model::DebertaModel::new_with_device(config.clone(), device)?,
            classifier: Linear::new_with_device(config.hidden_size, num_labels, true, device),
            dropout,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str, num_labels: usize) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config, num_labels)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        let hidden_states = self.deberta.forward(input_ids, attention_mask)?;

        // Apply dropout
        let hidden_states = hidden_states * (1.0 - self.dropout);

        // Token classification head
        let classifier_input = trustformers_core::tensor::Tensor::F32(hidden_states.into_dyn());
        let logits = self.classifier.forward(classifier_input)?;
        let logits = match logits {
            trustformers_core::tensor::Tensor::F32(arr) => {
                arr.into_dimensionality::<Ix3>().map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?
            },
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

pub struct DebertaForQuestionAnswering {
    pub deberta: crate::deberta::model::DebertaModel,
    pub qa_outputs: Linear,
    pub dropout: f32,
    device: Device,
}

impl DebertaForQuestionAnswering {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DebertaConfig, device: Device) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            deberta: crate::deberta::model::DebertaModel::new_with_device(config.clone(), device)?,
            qa_outputs: Linear::new_with_device(config.hidden_size, 2, true, device), // start and end logits
            dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let hidden_states = self.deberta.forward(input_ids, attention_mask)?;

        // Apply dropout
        let hidden_states = hidden_states * (1.0 - self.dropout);

        // QA head
        let qa_input = trustformers_core::tensor::Tensor::F32(hidden_states.into_dyn());
        let logits = self.qa_outputs.forward(qa_input)?;
        let logits = match logits {
            trustformers_core::tensor::Tensor::F32(arr) => {
                arr.into_dimensionality::<Ix3>().map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?
            },
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from qa_outputs",
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

pub struct DebertaForMultipleChoice {
    pub deberta: crate::deberta::model::DebertaModel,
    pub pooler: Linear,
    pub classifier: Linear,
    pub dropout: f32,
    device: Device,
}

impl DebertaForMultipleChoice {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DebertaConfig, device: Device) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            deberta: crate::deberta::model::DebertaModel::new_with_device(config.clone(), device)?,
            pooler: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            classifier: Linear::new_with_device(config.hidden_size, 1, true, device),
            dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array2<f32>> {
        let hidden_states = self.deberta.forward(input_ids, attention_mask)?;

        // Use [CLS] token representation (first token)
        let cls_hidden = hidden_states.slice(s![0, 0, ..]).to_owned();

        // Pooler
        let pooler_input =
            trustformers_core::tensor::Tensor::F32(cls_hidden.insert_axis(Axis(0)).into_dyn());
        let pooled_output = self.pooler.forward(pooler_input)?;
        let pooled_output = match pooled_output {
            trustformers_core::tensor::Tensor::F32(arr) => {
                arr.into_dimensionality::<Ix2>().map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?
            },
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from pooler",
                        "pooler",
                    ),
                )
            },
        };
        let pooled_tensor = trustformers_core::tensor::Tensor::F32(pooled_output.into_dyn());
        let pooled_output = gelu(&pooled_tensor)?;
        let pooled_output = match pooled_output {
            trustformers_core::tensor::Tensor::F32(arr) => {
                arr.into_dimensionality::<Ix2>().map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?
            },
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Expected F32 tensor from gelu",
                        "gelu",
                    ),
                )
            },
        };

        // Apply dropout
        let pooled_output = pooled_output * (1.0 - self.dropout);

        // Classification head
        let classifier_input = trustformers_core::tensor::Tensor::F32(pooled_output.into_dyn());
        let logits = self.classifier.forward(classifier_input)?;
        let logits = match logits {
            trustformers_core::tensor::Tensor::F32(arr) => {
                arr.into_dimensionality::<Ix2>().map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?
            },
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
    use crate::deberta::model::{DebertaForMaskedLM, DebertaForSequenceClassification};
    // Array1 already imported via scirs2_core at top

    #[test]
    fn test_deberta_sequence_classification() {
        let config = DebertaConfig::base();
        let model = DebertaForSequenceClassification::new(config, 2).expect("operation failed");

        let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]); // [CLS] tokens [SEP]
        let result = model.forward(&input_ids, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, 2]);
    }

    #[test]
    fn test_deberta_token_classification() {
        let config = DebertaConfig::base();
        let model = DebertaForTokenClassification::new(config, 9).expect("operation failed"); // BIO tagging

        let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]);
        let result = model.forward(&input_ids, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, input_ids.len(), 9]);
    }

    #[test]
    fn test_deberta_question_answering() {
        let config = DebertaConfig::base();
        let model = DebertaForQuestionAnswering::new(config).expect("operation failed");

        let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2, 4, 5, 6, 7, 2]);
        let result = model.forward(&input_ids, None);

        assert!(result.is_ok());
        let (start_logits, end_logits) = result.expect("operation failed");
        assert_eq!(start_logits.shape(), &[1, input_ids.len()]);
        assert_eq!(end_logits.shape(), &[1, input_ids.len()]);
    }

    #[test]
    fn test_deberta_masked_lm() {
        let config = DebertaConfig::base();
        let model = DebertaForMaskedLM::new(config.clone()).expect("operation failed");

        let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]);
        let result = model.forward(&input_ids, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, input_ids.len(), config.vocab_size]);
    }

    #[test]
    fn test_deberta_multiple_choice() {
        let config = DebertaConfig::base();
        let model = DebertaForMultipleChoice::new(config).expect("operation failed");

        let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]);
        let result = model.forward(&input_ids, None);

        assert!(result.is_ok());
        let logits = result.expect("operation failed");
        assert_eq!(logits.shape(), &[1, 1]);
    }
}
