//! NNEF (Neural Network Exchange Format) export surface.
//!
//! NNEF is a Khronos standard for representing neural networks. An NNEF package is
//! an *operation graph* (`graph.nnef`) plus the binary tensors it references.
//!
//! # Why no NNEF package is written
//!
//! [`Model`] exposes parameters (via [`Model::named_tensors`]) but not topology:
//! there is no way to learn which operations connect which tensors. Earlier
//! revisions of this exporter papered over that by emitting a fixed 768-wide,
//! 12-block transformer graph plus weight files filled with `(i % 256) as u8`,
//! regardless of the model handed to it. That artifact described a model that did
//! not exist, so it is no longer produced: [`NNEFExporter::export`] now returns a
//! structured [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation).
//!
//! Use the GGUF or GGML exporters to write the model's real parameters; they are
//! tensor containers and need no topology.

use crate::errors::unsupported_operation;
use crate::export::{ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::traits::Model;
use anyhow::{anyhow, Result};

/// Explanation attached to every refusal to write an NNEF package.
pub const NNEF_UNSUPPORTED_REASON: &str =
    "an NNEF package requires the model's operation graph, which the `Model` trait \
     does not expose (`named_tensors` yields parameters only). TrustformeRS will not \
     emit a synthesized graph under a real model's name.";

/// NNEF exporter for TrustformeRS models
#[derive(Clone)]
pub struct NNEFExporter {
    version: String,
    extensions: Vec<String>,
}

impl NNEFExporter {
    /// Create a new NNEF exporter
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            extensions: vec!["KHR_enable_fragment_definitions".to_string()],
        }
    }

    /// Create NNEF exporter with custom configuration
    pub fn with_config(version: String, extensions: Vec<String>) -> Self {
        Self {
            version,
            extensions,
        }
    }

    /// The NNEF version this exporter would declare.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The NNEF extensions this exporter would declare.
    pub fn extensions(&self) -> &[String] {
        &self.extensions
    }

    /// Derive the declared input shape from the export configuration.
    ///
    /// This is purely a function of [`ExportConfig`]; it makes no claim about the
    /// model's actual layout.
    pub fn get_input_shape(&self, config: &ExportConfig) -> Vec<i64> {
        // Infer model type from context and configuration
        let batch_size = config.batch_size.unwrap_or(1) as i64;

        // Check if this is a vision model based on configuration hints
        if let Some(ref input_shape) = config.input_shape {
            if input_shape.len() == 4 {
                // Vision model (NCHW format): [batch, channels, height, width]
                return input_shape.iter().map(|&x| x as i64).collect();
            } else if input_shape.len() == 3 && input_shape[2] > 50 {
                // Vision model (HWC format): [height, width, channels]
                return vec![
                    batch_size,
                    input_shape[2] as i64,
                    input_shape[0] as i64,
                    input_shape[1] as i64,
                ];
            }
        }

        // Check for audio/signal processing models
        if config.sequence_length.unwrap_or(512) > 8192 {
            // Likely audio or long sequence model
            return vec![batch_size, config.sequence_length.unwrap_or(16000) as i64];
        }

        // Default to NLP model with token IDs
        let sequence_length = config.sequence_length.unwrap_or(512) as i64;

        // Check if this might be a multimodal model
        if let Some(ref task_type) = config.task_type {
            if task_type.to_lowercase().contains("multimodal")
                || task_type.to_lowercase().contains("vision")
            {
                // Vision-language model might have multiple inputs
                return vec![batch_size, sequence_length, 3, 224, 224]; // [batch, seq_len, channels, height, width]
            }
        }

        // Standard NLP model: [batch_size, sequence_length]
        vec![batch_size, sequence_length]
    }

    /// Derive the declared output shape from the export configuration.
    ///
    /// This is purely a function of [`ExportConfig`]; it makes no claim about the
    /// model's actual layout.
    pub fn get_output_shape(&self, config: &ExportConfig) -> Vec<i64> {
        let batch_size = config.batch_size.unwrap_or(1) as i64;
        let sequence_length = config.sequence_length.unwrap_or(512) as i64;

        // Infer output shape based on task type or configuration
        if let Some(ref task_type) = config.task_type {
            match task_type.to_lowercase().as_str() {
                "classification" | "text-classification" => {
                    // Classification: [batch_size, num_classes]
                    let num_classes = config.vocab_size.unwrap_or(2) as i64; // Binary classification default
                    vec![batch_size, num_classes]
                },
                "token-classification" | "ner" => {
                    // Token classification: [batch_size, sequence_length, num_labels]
                    let num_labels = config.vocab_size.unwrap_or(9) as i64; // Common NER label count
                    vec![batch_size, sequence_length, num_labels]
                },
                "question-answering" | "qa" => {
                    // QA model: [batch_size, sequence_length, 2] for start/end positions
                    vec![batch_size, sequence_length, 2]
                },
                "image-classification" => {
                    // Vision classification: [batch_size, num_classes]
                    let num_classes = config.vocab_size.unwrap_or(1000) as i64; // ImageNet default
                    vec![batch_size, num_classes]
                },
                "object-detection" => {
                    // Object detection: [batch_size, num_detections, 6] (x, y, w, h, confidence, class)
                    vec![batch_size, 100, 6] // Default 100 detections
                },
                "generation" | "text-generation" | "causal-lm" => {
                    // Language generation: [batch_size, sequence_length, vocab_size]
                    let vocab_size = config.vocab_size.unwrap_or(50257) as i64; // GPT-2 vocab size
                    vec![batch_size, sequence_length, vocab_size]
                },
                "masked-lm" | "mlm" => {
                    // Masked language modeling: [batch_size, sequence_length, vocab_size]
                    let vocab_size = config.vocab_size.unwrap_or(30522) as i64; // BERT vocab size
                    vec![batch_size, sequence_length, vocab_size]
                },
                "embedding" | "feature-extraction" => {
                    // Feature extraction: [batch_size, sequence_length, hidden_size]
                    let hidden_size = 768; // Common transformer hidden size
                    vec![batch_size, sequence_length, hidden_size]
                },
                "similarity" | "sentence-similarity" => {
                    // Sentence similarity: [batch_size, hidden_size]
                    let hidden_size = 768;
                    vec![batch_size, hidden_size]
                },
                _ => {
                    // Default to hidden states output
                    vec![batch_size, sequence_length, 768]
                },
            }
        } else {
            // Infer from input shape if no task type specified
            let input_shape = self.get_input_shape(config);
            match input_shape.len() {
                2 => {
                    // 2D input likely means text: output hidden states
                    vec![batch_size, sequence_length, 768]
                },
                3 => {
                    // 3D input might be embeddings: preserve or add vocab projection
                    vec![batch_size, sequence_length, 768]
                },
                4 => {
                    // 4D input likely vision: output classification logits
                    vec![batch_size, 1000] // ImageNet classes
                },
                _ => {
                    // Default fallback
                    vec![batch_size, sequence_length, 768]
                },
            }
        }
    }

    /// Convert export precision to the corresponding NNEF scalar data type name.
    pub fn precision_to_dtype(&self, precision: ExportPrecision) -> &'static str {
        match precision {
            ExportPrecision::FP32 => "real32",
            ExportPrecision::FP16 => "real16",
            ExportPrecision::INT8 => "integer8",
            ExportPrecision::INT4 => "integer4",
        }
    }

    /// Validate an NNEF export configuration.
    pub fn validate_config(&self, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::NNEF {
            return Err(anyhow!(
                "Invalid format for NNEF exporter: {:?}",
                config.format
            ));
        }

        // Check precision support
        match config.precision {
            ExportPrecision::FP32 | ExportPrecision::FP16 => {},
            ExportPrecision::INT8 | ExportPrecision::INT4 => {
                if config.quantization.is_none() {
                    return Err(anyhow!(
                        "Quantization config required for integer precision"
                    ));
                }
            },
        }

        Ok(())
    }
}

impl ModelExporter for NNEFExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no NNEF package is written.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.validate_config(config)?;
        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing topology is a limitation of the `Model` trait.
        let _tensors = crate::export::collect_model_tensors(model)?;
        Err(unsupported_operation("NNEF graph export", NNEF_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::NNEF]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::NNEF {
            return Err(anyhow!("NNEF exporter only supports NNEF format"));
        }
        Ok(())
    }
}

impl Default for NNEFExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::export::test_support::TestModel;

    #[test]
    fn test_nnef_exporter_creation() {
        let exporter = NNEFExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats, vec![ExportFormat::NNEF]);
    }

    #[test]
    fn test_nnef_exporter_with_config() {
        let exporter = NNEFExporter::with_config(
            "1.0".to_string(),
            vec!["KHR_enable_fragment_definitions".to_string()],
        );
        assert_eq!(exporter.version, "1.0");
        assert_eq!(exporter.extensions.len(), 1);
    }

    #[test]
    fn test_precision_to_dtype() {
        let exporter = NNEFExporter::new();
        assert_eq!(exporter.precision_to_dtype(ExportPrecision::FP32), "real32");
        assert_eq!(exporter.precision_to_dtype(ExportPrecision::FP16), "real16");
        assert_eq!(
            exporter.precision_to_dtype(ExportPrecision::INT8),
            "integer8"
        );
        assert_eq!(
            exporter.precision_to_dtype(ExportPrecision::INT4),
            "integer4"
        );
    }

    #[test]
    fn test_input_output_shapes() {
        let exporter = NNEFExporter::new();
        let config = ExportConfig {
            format: ExportFormat::NNEF,
            batch_size: Some(2),
            sequence_length: Some(128),
            ..Default::default()
        };

        let input_shape = exporter.get_input_shape(&config);
        assert_eq!(input_shape, vec![2, 128]);

        let output_shape = exporter.get_output_shape(&config);
        assert_eq!(output_shape, vec![2, 128, 768]);
    }

    /// Regression test for the exporter that used to write a fixed 12-block
    /// transformer graph plus `(i % 256) as u8` weight files for any model.
    #[test]
    fn export_refuses_to_write_a_synthesized_package() {
        let dir = std::env::temp_dir().join("trustformers_nnef_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let exporter = NNEFExporter::new();
        let model = TestModel::with_seed(3.0);
        let config = ExportConfig {
            format: ExportFormat::NNEF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("must not fabricate a graph");
        assert!(
            err.to_string().contains("Unsupported operation"),
            "expected UnsupportedOperation, got: {err}"
        );
        assert!(
            !output.with_extension("nnef").exists(),
            "no NNEF package directory may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_reports_missing_weights_before_missing_topology() {
        let exporter = NNEFExporter::new();
        let model = TestModel::empty();
        let config = ExportConfig {
            format: ExportFormat::NNEF,
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("no weights, no export");
        assert!(
            err.to_string().contains("named_tensors"),
            "expected the missing-weights diagnostic, got: {err}"
        );
    }

    #[test]
    fn test_validate_config_success() {
        let exporter = NNEFExporter::new();
        let config = ExportConfig {
            format: ExportFormat::NNEF,
            precision: ExportPrecision::FP32,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_format() {
        let exporter = NNEFExporter::new();
        let config = ExportConfig {
            format: ExportFormat::ONNX,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_model_success() {
        let exporter = NNEFExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::NNEF).is_ok());
    }

    #[test]
    fn test_validate_model_wrong_format() {
        let exporter = NNEFExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::ONNX).is_err());
    }
}
