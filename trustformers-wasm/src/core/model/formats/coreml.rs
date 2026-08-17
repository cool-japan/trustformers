//! Core ML model format: detection and metadata only.
//!
//! There is no Core ML protobuf weight parser in this crate. Earlier code
//! pretended to parse Core ML's `NeuralNetwork` protobuf structure and
//! returned `WasmTensor::randn(...)` for each "detected" layer — i.e. pure
//! noise dressed up with a "✅ Loaded N layers" success log. `load_weights`
//! now returns a structured `UnsupportedFormat` error instead.

use super::ModelFormatParser;
use crate::core::model::config::ModelFormat;
use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;

/// Core ML model parser: format detection/metadata only (see module docs).
pub struct CoreMLParser;

impl ModelFormatParser for CoreMLParser {
    fn can_parse(&self, data: &[u8]) -> bool {
        data.len() > 8
            && (data.starts_with(b"\x08\x01")
                || data.starts_with(b"MLMODEL")
                || self.check_mlmodel_signature(data))
    }

    fn parse_metadata(&self, data: &[u8]) -> Result<HashMap<String, String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "Core ML".to_string());
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        if self.is_neural_network_model(data) {
            metadata.insert("model_type".to_string(), "neural_network".to_string());
        }

        let complexity = if data.len() > 50_000_000 {
            "large"
        } else if data.len() > 10_000_000 {
            "medium"
        } else {
            "small"
        };
        metadata.insert("complexity".to_string(), complexity.to_string());

        Ok(metadata)
    }

    fn load_weights(&self, data: &[u8]) -> Result<Vec<(String, WasmTensor)>, String> {
        Err(format!(
            "UnsupportedFormat: Core ML weight loading is not implemented in this WASM build \
             ({} bytes received). No protobuf parser exists for the Core ML NeuralNetwork \
             format here; convert the model to SafeTensors instead of loading it directly.",
            data.len()
        ))
    }

    fn get_format(&self) -> ModelFormat {
        ModelFormat::CoreML
    }
}

impl CoreMLParser {
    fn check_mlmodel_signature(&self, data: &[u8]) -> bool {
        data.windows(8).any(|window| window == b"mlmodel\0" || window == b"CoreML\0\0")
    }

    fn is_neural_network_model(&self, data: &[u8]) -> bool {
        String::from_utf8_lossy(data).contains("neuralNetwork")
            || data.windows(12).any(|w| w == b"neuralNetwork")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_weights_errors_instead_of_fabricating() {
        let parser = CoreMLParser;
        let data = b"MLMODELsome-fake-payload-with-neuralNetwork-marker".to_vec();
        assert!(parser.can_parse(&data));
        let err = parser.load_weights(&data).expect_err("must error, never fabricate tensors");
        assert!(err.contains("UnsupportedFormat"));
    }

    #[test]
    fn test_get_format() {
        assert_eq!(CoreMLParser.get_format(), ModelFormat::CoreML);
    }
}
