//! Model checkpoint format detection and parsing.

mod coreml;
mod safetensors;
mod tensorrt;
mod tflite;

pub use coreml::CoreMLParser;
pub use safetensors::{parse_safetensors, ParsedTensor, SafeTensorsParser};
pub use tensorrt::TensorRTParser;
pub use tflite::TensorFlowLiteParser;

use crate::core::model::config::ModelFormat;
use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;

/// Model format detection result.
#[derive(Debug, Clone)]
pub struct FormatDetectionResult {
    pub format: ModelFormat,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

/// Model format parser trait for different formats.
///
/// `load_weights` returns *named* tensors: `(tensor_name, tensor)` pairs.
/// A format that cannot actually recover real weights (no working parser
/// exists for it in this build) must return `Err` — never a placeholder
/// tensor standing in for real data. See [`CoreMLParser`] and
/// [`TensorFlowLiteParser`] for the canonical "unsupported format" shape.
///
/// Errors are plain `String`s, not `JsValue`: constructing a `JsValue`
/// (even a bare `JsValue::from_str`) unconditionally panics on non-wasm32
/// targets, which would make every parser's error path untestable
/// natively. `JsValue` conversion happens once, at the `#[wasm_bindgen]`
/// boundary in `WasmModel`.
pub trait ModelFormatParser {
    fn can_parse(&self, data: &[u8]) -> bool;
    fn parse_metadata(&self, data: &[u8]) -> Result<HashMap<String, String>, String>;
    fn load_weights(&self, data: &[u8]) -> Result<Vec<(String, WasmTensor)>, String>;
    fn get_format(&self) -> ModelFormat;
}

/// Model format detector and parser manager.
pub struct ModelFormatManager {
    parsers: Vec<std::boxed::Box<dyn ModelFormatParser>>,
}

impl ModelFormatManager {
    pub fn new() -> Self {
        let parsers: Vec<std::boxed::Box<dyn ModelFormatParser>> = std::vec![
            std::boxed::Box::new(SafeTensorsParser),
            std::boxed::Box::new(TensorRTParser::new()),
            std::boxed::Box::new(CoreMLParser),
            std::boxed::Box::new(TensorFlowLiteParser),
        ];

        Self { parsers }
    }

    /// Detect model format from binary data.
    pub fn detect_format(&self, data: &[u8]) -> Option<FormatDetectionResult> {
        for parser in &self.parsers {
            if parser.can_parse(data) {
                let metadata = parser.parse_metadata(data).unwrap_or_default();
                return Some(FormatDetectionResult {
                    format: parser.get_format(),
                    confidence: 0.9,
                    metadata,
                });
            }
        }
        self.detect_format_by_heuristics(data)
    }

    /// Load model using the appropriate parser, returning named tensors.
    pub fn load_model(
        &self,
        data: &[u8],
        format: Option<ModelFormat>,
    ) -> Result<Vec<(String, WasmTensor)>, String> {
        let target_format = match format {
            Some(f) => f,
            None => match self.detect_format(data) {
                Some(result) => result.format,
                None => return Err("Unsupported model format".to_string()),
            },
        };

        for parser in &self.parsers {
            if parser.get_format() == target_format {
                return parser.load_weights(data);
            }
        }

        Err(format!("No parser available for format: {target_format:?}"))
    }

    fn detect_format_by_heuristics(&self, data: &[u8]) -> Option<FormatDetectionResult> {
        let mut metadata = HashMap::new();
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        if data.starts_with(b"ONNX") || data.windows(3).any(|w| w == [0x08, 0x01, 0x12]) {
            return Some(FormatDetectionResult {
                format: ModelFormat::Onnx,
                confidence: 0.7,
                metadata,
            });
        }

        if data.starts_with(b"GGUF") || data.windows(4).any(|w| w == [0x47, 0x47, 0x55, 0x46]) {
            return Some(FormatDetectionResult {
                format: ModelFormat::Gguf,
                confidence: 0.8,
                metadata,
            });
        }

        None
    }
}

impl Default for ModelFormatManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_by_heuristics_onnx() {
        let manager = ModelFormatManager::new();
        let data = b"ONNX-fake-payload-that-is-definitely-not-a-flatbuffer".to_vec();
        let result = manager.detect_format(&data).expect("should detect onnx");
        assert_eq!(result.format, ModelFormat::Onnx);
    }

    #[test]
    fn test_detect_format_by_heuristics_gguf() {
        let manager = ModelFormatManager::new();
        let data = b"GGUF-fake-payload-that-is-definitely-not-a-flatbuffer".to_vec();
        let result = manager.detect_format(&data).expect("should detect gguf");
        assert_eq!(result.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_unknown_returns_none() {
        let manager = ModelFormatManager::new();
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        assert!(manager.detect_format(&data).is_none());
    }

    #[test]
    fn test_load_model_unsupported_format_errors_not_dummy() {
        let manager = ModelFormatManager::new();
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0];
        let err = manager.load_model(&data, None);
        assert!(
            err.is_err(),
            "unknown data must error, not silently succeed"
        );
    }
}
