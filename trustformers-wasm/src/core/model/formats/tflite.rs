//! TensorFlow Lite model format: detection and metadata only.
//!
//! There is no FlatBuffers-based TFLite weight parser in this crate.
//! Earlier code estimated a plausible tensor *count* from the file size and
//! then filled each one with `WasmTensor::zeros(...)`, logging a fabricated
//! "✅ Loaded N tensors" success message. `load_weights` now returns a
//! structured `UnsupportedFormat` error instead.

use super::ModelFormatParser;
use crate::core::model::config::ModelFormat;
use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;

/// TensorFlow Lite parser: format detection/metadata only (see module docs).
pub struct TensorFlowLiteParser;

impl ModelFormatParser for TensorFlowLiteParser {
    fn can_parse(&self, data: &[u8]) -> bool {
        data.len() > 16
            && (data.starts_with(b"TFL3")
                || data[12..16] == [0x54, 0x46, 0x4C, 0x33]
                || self.check_flatbuffer_signature(data))
    }

    fn parse_metadata(&self, data: &[u8]) -> Result<HashMap<String, String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "TensorFlow Lite".to_string());
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        if self.is_quantized_model(data) {
            metadata.insert("quantization".to_string(), "int8".to_string());
        } else {
            metadata.insert("quantization".to_string(), "float32".to_string());
        }

        Ok(metadata)
    }

    fn load_weights(&self, data: &[u8]) -> Result<Vec<(String, WasmTensor)>, String> {
        Err(format!(
            "UnsupportedFormat: TensorFlow Lite weight loading is not implemented in this WASM \
             build ({} bytes received). No FlatBuffers parser exists for the TFLite schema \
             here; convert the model to SafeTensors instead of loading it directly.",
            data.len()
        ))
    }

    fn get_format(&self) -> ModelFormat {
        ModelFormat::TensorFlowLite
    }
}

impl TensorFlowLiteParser {
    /// Loose FlatBuffers structural check: a valid FlatBuffer's first four
    /// bytes are a little-endian `u32` offset to the root table, which
    /// must be non-zero and point somewhere inside the buffer. This is
    /// intentionally still a heuristic (FlatBuffers has no magic number by
    /// default), but requiring the offset to be in-bounds rejects
    /// essentially all plain-text/ASCII payloads, unlike the previous
    /// "every byte in data[4..8] happens to be < 128" check (true for
    /// almost any short ASCII string, causing false-positive TFLite
    /// detection on e.g. `b"ONNX..."`/`b"GGUF..."` payloads).
    fn check_flatbuffer_signature(&self, data: &[u8]) -> bool {
        if data.len() < 8 {
            return false;
        }
        let root_offset = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        root_offset > 0 && root_offset < data.len()
    }

    fn is_quantized_model(&self, data: &[u8]) -> bool {
        String::from_utf8_lossy(data).contains("quantization")
            || data.windows(4).any(|w| w == b"INT8" || w == b"UINT8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_weights_errors_instead_of_fabricating() {
        let parser = TensorFlowLiteParser;
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"TFL3");
        assert!(parser.can_parse(&data));
        let err = parser.load_weights(&data).expect_err("must error, never fabricate tensors");
        assert!(err.contains("UnsupportedFormat"));
    }

    #[test]
    fn test_get_format() {
        assert_eq!(
            TensorFlowLiteParser.get_format(),
            ModelFormat::TensorFlowLite
        );
    }

    #[test]
    fn test_flatbuffer_heuristic_rejects_ascii_text() {
        // Regression test for the old "every byte < 128" heuristic, which
        // matched almost any printable-ASCII payload as TFLite.
        let parser = TensorFlowLiteParser;
        let data = b"ONNX-fake-payload-that-is-definitely-not-a-flatbuffer".to_vec();
        assert!(
            !parser.can_parse(&data),
            "plain ASCII text must not be detected as TFLite"
        );
    }

    #[test]
    fn test_flatbuffer_heuristic_accepts_in_bounds_root_offset() {
        let parser = TensorFlowLiteParser;
        let mut data = vec![0u8; 32];
        // A plausible root offset of 8, well within the 32-byte buffer.
        data[0..4].copy_from_slice(&8u32.to_le_bytes());
        assert!(parser.can_parse(&data));
    }
}
