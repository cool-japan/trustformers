//! Legacy GGML export surface.
//!
//! # Why no `.ggml` file is written
//!
//! A legacy GGML file is a fixed header of architecture hyper-parameters
//! (`n_vocab`, `n_ctx`, `n_embd`, `n_head`, `n_layer`, `n_rot`, `ftype`),
//! **followed by the full tokenizer vocabulary**, and only then the tensors. The
//! [`Model`] trait supplies none of that: [`Model::named_tensors`] yields parameters,
//! and there is no vocabulary or head count to be had.
//!
//! Earlier revisions filled the gap by hard-coding GPT-2's numbers
//! (`n_vocab = 50257`, `n_embd = 768`, `n_layer = 12`), emitting a vocabulary of
//! `token_0 … token_50256`, and generating every weight from `thread_rng()`. That
//! artifact described a model that did not exist, so it is no longer produced:
//! [`GGMLExporter::export`] returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation).
//!
//! GGML has in any case been superseded by GGUF, which is self-describing and needs
//! no vocabulary. Use [`super::gguf::GGUFExporter`], which writes the model's real
//! parameters.

use super::{collect_model_tensors, ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::errors::unsupported_operation;
use crate::traits::Model;
use anyhow::{anyhow, Result};

/// `"ggml"` in ASCII, little endian.
pub const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Version tag of the legacy GGML container.
pub const GGML_VERSION: u32 = 1;

/// Explanation attached to every refusal to write a GGML file.
pub const GGML_UNSUPPORTED_REASON: &str =
    "a legacy GGML file begins with architecture hyper-parameters and the complete \
     tokenizer vocabulary, neither of which the `Model` trait exposes; TrustformeRS \
     will not substitute hard-coded GPT-2 numbers and a generated vocabulary. Export \
     to GGUF instead, which is self-describing and carries the model's real tensors.";

/// GGML tensor element types (`ggml_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl GGMLType {
    /// Map an export precision onto the tensor type used for weights.
    pub fn from_precision(precision: ExportPrecision) -> Self {
        match precision {
            ExportPrecision::FP32 => GGMLType::F32,
            ExportPrecision::FP16 => GGMLType::F16,
            ExportPrecision::INT8 => GGMLType::Q8_0,
            ExportPrecision::INT4 => GGMLType::Q4_0,
        }
    }

    /// Number of elements packed into one storage block.
    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 => 1,
            GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::Q8K => 256,
            _ => 32,
        }
    }

    /// Size in bytes of one storage block.
    pub fn type_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,
            GGMLType::Q4_1 => 20,
            GGMLType::Q5_0 => 22,
            GGMLType::Q5_1 => 24,
            GGMLType::Q8_0 => 34,
            GGMLType::Q8_1 => 36,
            GGMLType::Q2K => 84,
            GGMLType::Q3K => 110,
            GGMLType::Q4K => 144,
            GGMLType::Q5K => 176,
            GGMLType::Q6K => 210,
            GGMLType::Q8K => 292,
        }
    }
}

/// One tensor of a GGML container.
#[derive(Debug, Clone)]
pub struct GGMLTensor {
    pub name: String,
    pub tensor_type: GGMLType,
    pub dimensions: Vec<usize>,
    pub data: Vec<u8>,
}

/// GGML exporter.
#[derive(Clone, Debug, Default)]
pub struct GGMLExporter {
    quantization_enabled: bool,
}

impl GGMLExporter {
    /// Create a GGML exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request quantized weights. Recorded but currently unreachable, because
    /// [`GGMLExporter::export`] never writes a file — see the module docs.
    pub fn with_quantization(mut self, enabled: bool) -> Self {
        self.quantization_enabled = enabled;
        self
    }

    /// Whether quantization was requested.
    pub fn quantization_enabled(&self) -> bool {
        self.quantization_enabled
    }
}

impl ModelExporter for GGMLExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no `.ggml` file is written.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::GGML {
            return Err(anyhow!("GGMLExporter only supports GGML format"));
        }

        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing vocabulary is a limitation of the `Model` trait.
        let _tensors = collect_model_tensors(model)?;
        Err(unsupported_operation("legacy GGML export", GGML_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::GGML]
    }

    fn validate_model<M: Model>(&self, model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::GGML {
            return Err(anyhow!("GGMLExporter only supports GGML format"));
        }
        collect_model_tensors(model)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::test_support::TestModel;

    #[test]
    fn test_ggml_exporter_creation() {
        let exporter = GGMLExporter::new();
        assert!(!exporter.quantization_enabled());
        assert!(exporter.with_quantization(true).quantization_enabled());
    }

    #[test]
    fn test_ggml_type_conversion() {
        assert_eq!(GGMLType::from_precision(ExportPrecision::FP32) as u32, 0);
        assert_eq!(GGMLType::from_precision(ExportPrecision::FP16) as u32, 1);
        assert_eq!(GGMLType::from_precision(ExportPrecision::INT8) as u32, 8);
        assert_eq!(GGMLType::from_precision(ExportPrecision::INT4) as u32, 2);
    }

    #[test]
    fn test_ggml_block_layout_matches_llama_cpp() {
        assert_eq!(GGMLType::F32.type_size(), 4);
        assert_eq!(GGMLType::F16.type_size(), 2);
        assert_eq!(GGMLType::Q8_0.block_size(), 32);
        assert_eq!(GGMLType::Q8_0.type_size(), 34);
        assert_eq!(GGMLType::Q4_0.type_size(), 18);
    }

    #[test]
    fn test_supported_formats() {
        assert_eq!(
            GGMLExporter::new().supported_formats(),
            vec![ExportFormat::GGML]
        );
    }

    #[test]
    fn test_ggml_constants() {
        assert_eq!(GGML_MAGIC.to_le_bytes(), *b"lmgg");
        assert_eq!(GGML_VERSION, 1);
    }

    /// Regression test for the exporter that used to write a hard-coded GPT-2
    /// header, a `token_0 … token_50256` vocabulary and `thread_rng()` weights.
    #[test]
    fn export_refuses_to_write_a_synthesized_container() {
        let dir = std::env::temp_dir().join("trustformers_ggml_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let config = ExportConfig {
            format: ExportFormat::GGML,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = GGMLExporter::new()
            .export(&TestModel::with_seed(1.0), &config)
            .expect_err("must not fabricate a container");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
        assert!(
            !output.with_extension("ggml").exists(),
            "no .ggml may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_reports_missing_weights_first() {
        let config = ExportConfig {
            format: ExportFormat::GGML,
            ..Default::default()
        };
        let err = GGMLExporter::new()
            .export(&TestModel::empty(), &config)
            .expect_err("no weights, no export");
        assert!(err.to_string().contains("named_tensors"), "{err}");
    }
}
