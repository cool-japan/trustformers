//! GGUF export.
//!
//! GGUF is a tensor container (header + metadata + tensor directory + data). It
//! carries no operation graph, so a model's *real* parameters — obtained from
//! [`Model::named_tensors`](crate::traits::Model::named_tensors) — are all that is
//! needed to write a faithful file.
//!
//! The binary format itself lives in [`super::gguf_format`]; this module only turns
//! a [`Model`] plus an [`ExportConfig`] into a [`GGUFPayload`].
//!
//! Earlier revisions of this exporter ignored the model entirely and wrote a
//! hard-coded GPT-2 skeleton whose weights were `sin(i * 0.001)` and whose
//! vocabulary was `token_0 … token_50256`. Nothing of the sort is produced now: a
//! model that exposes no named tensors makes [`GGUFExporter::export`] fail.

use super::gguf_format::{
    payload_from_tensors, write_gguf_file, GGUFPayload, GGUF_DEFAULT_ALIGNMENT,
};
use super::{collect_model_tensors, ExportConfig, ExportFormat, ModelExporter};
use crate::traits::{Config, Model};
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

pub use super::gguf_format::{
    GGUFHeader, GGUFTensorInfo, GGUFTensorType, GGUFValue, GGUFValueType, GGUF_MAGIC, GGUF_VERSION,
};

/// GGUF exporter for TrustformeRS models.
#[derive(Clone, Debug)]
pub struct GGUFExporter {
    alignment: usize,
}

impl Default for GGUFExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GGUFExporter {
    /// Create an exporter with the default 32-byte tensor alignment.
    pub fn new() -> Self {
        Self {
            alignment: GGUF_DEFAULT_ALIGNMENT as usize,
        }
    }

    /// Override the `general.alignment` used for the tensor data section.
    ///
    /// The value must be a power of two; anything else is rejected at export time.
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// The alignment this exporter will declare.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Round an offset up to this exporter's alignment.
    pub fn align_offset(&self, offset: u64) -> u64 {
        super::gguf_format::align_up(offset, self.alignment as u64)
    }

    /// Build the in-memory GGUF payload for `model`.
    ///
    /// Every tensor written comes from [`Model::named_tensors`]; the metadata is
    /// derived from the model's configuration and the caller's [`ExportConfig`].
    pub fn build_payload<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<GGUFPayload> {
        if !self.alignment.is_power_of_two() || self.alignment == 0 {
            return Err(anyhow!(
                "GGUF alignment must be a positive power of two, got {}",
                self.alignment
            ));
        }

        let tensors = collect_model_tensors(model)?;
        let tensor_type = GGUFTensorType::from_precision(config.precision);
        let architecture = model.get_config().architecture();

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GGUFValue::String(architecture.to_string()),
        );
        metadata.insert(
            "general.alignment".to_string(),
            GGUFValue::UInt32(self.alignment as u32),
        );
        metadata.insert(
            "general.file_type".to_string(),
            GGUFValue::UInt32(tensor_type.file_type()),
        );
        metadata.insert(
            "general.quantization_version".to_string(),
            GGUFValue::UInt32(2),
        );
        metadata.insert(
            "general.parameter_count".to_string(),
            GGUFValue::UInt64(model.num_parameters() as u64),
        );

        // Values the caller declared in the export configuration. They describe the
        // export request, not anything measured from the model.
        if let Some(context_length) = config.sequence_length {
            metadata.insert(
                format!("{architecture}.context_length"),
                GGUFValue::UInt64(context_length as u64),
            );
        }
        if let Some(vocab_size) = config.vocab_size {
            metadata.insert(
                format!("{architecture}.vocab_size"),
                GGUFValue::UInt64(vocab_size as u64),
            );
        }

        payload_from_tensors(metadata, &tensors, tensor_type)
    }
}

impl ModelExporter for GGUFExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::GGUF {
            return Err(anyhow!("GGUFExporter only supports GGUF format"));
        }

        let payload = self.build_payload(model, config)?;
        let output_path = format!("{}.gguf", config.output_path);
        write_gguf_file(&output_path, &payload)?;

        log::info!(
            "wrote {} tensors ({} parameters) to {}",
            payload.tensors.len(),
            model.num_parameters(),
            output_path
        );
        Ok(())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::GGUF]
    }

    fn validate_model<M: Model>(&self, model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::GGUF {
            return Err(anyhow!("GGUFExporter only supports GGUF format"));
        }
        collect_model_tensors(model)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::gguf_format::read_gguf_file;
    use crate::export::test_support::TestModel;
    use crate::export::ExportPrecision;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn test_gguf_exporter_creation() {
        let exporter = GGUFExporter::new();
        assert_eq!(exporter.alignment(), 32);
        assert_eq!(exporter.with_alignment(64).alignment(), 64);
    }

    #[test]
    fn test_gguf_constants() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn test_gguf_tensor_type_conversion() {
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::FP32) as u32,
            0
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::FP16) as u32,
            1
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::INT8) as u32,
            8
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::INT4) as u32,
            2
        );
    }

    #[test]
    fn test_supported_formats() {
        let exporter = GGUFExporter::new();
        assert_eq!(exporter.supported_formats(), vec![ExportFormat::GGUF]);
    }

    #[test]
    fn test_offset_alignment() {
        let exporter = GGUFExporter::new().with_alignment(32);
        assert_eq!(exporter.align_offset(0), 0);
        assert_eq!(exporter.align_offset(1), 32);
        assert_eq!(exporter.align_offset(32), 32);
        assert_eq!(exporter.align_offset(33), 64);
    }

    /// The exporter must write the model's real tensors, not a fixed skeleton.
    #[test]
    fn export_writes_the_models_real_tensors() {
        let dir = temp_dir("trustformers_gguf_export_real");
        let output = dir.join("model");

        let model = TestModel::with_seed(0.5);
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            precision: ExportPrecision::FP32,
            ..Default::default()
        };

        GGUFExporter::new().export(&model, &config).expect("export");

        let parsed = read_gguf_file(output.with_extension("gguf")).expect("re-read");
        assert_eq!(parsed.tensors.len(), 3);
        assert_eq!(
            parsed.metadata.get("general.architecture").and_then(GGUFValue::as_str),
            Some("test_transformer")
        );

        for (name, tensor) in model.named_tensors() {
            let expected = tensor.to_vec_f32().expect("f32");
            let actual = parsed.tensor_f32(&name).expect("tensor present");
            assert_eq!(actual, expected, "tensor '{name}' must round-trip exactly");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression test: the old exporter produced identical bytes for every model.
    #[test]
    fn export_output_varies_with_the_model_weights() {
        let dir = temp_dir("trustformers_gguf_export_varies");

        let write = |seed: f32, name: &str| {
            let output = dir.join(name);
            let config = ExportConfig {
                format: ExportFormat::GGUF,
                output_path: output.to_string_lossy().to_string(),
                ..Default::default()
            };
            GGUFExporter::new()
                .export(&TestModel::with_seed(seed), &config)
                .expect("export");
            std::fs::read(output.with_extension("gguf")).expect("read back")
        };

        let a = write(0.0, "a");
        let b = write(7.0, "b");
        assert_eq!(a.len(), b.len(), "same shapes give the same file size");
        assert_ne!(a, b, "different weights must give different files");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_refuses_a_model_without_named_tensors() {
        let dir = temp_dir("trustformers_gguf_export_empty");
        let output = dir.join("model");
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = GGUFExporter::new()
            .export(&TestModel::empty(), &config)
            .expect_err("must not invent weights");
        assert!(err.to_string().contains("named_tensors"), "{err}");
        assert!(
            !output.with_extension("gguf").exists(),
            "no file may be written"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantized_export_round_trips_within_tolerance() {
        let dir = temp_dir("trustformers_gguf_export_q8");
        let output = dir.join("model");

        // Q8_0 needs every tensor to be a multiple of 32 elements.
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();
        let model = TestModel::new(
            Default::default(),
            vec![(
                "w".to_string(),
                crate::tensor::Tensor::from_vec(values.clone(), &[2, 32]).expect("tensor"),
            )],
        );
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            precision: ExportPrecision::INT8,
            ..Default::default()
        };

        GGUFExporter::new().export(&model, &config).expect("export");
        let parsed = read_gguf_file(output.with_extension("gguf")).expect("re-read");
        assert_eq!(parsed.tensors[0].0.tensor_type, GGUFTensorType::Q8_0);

        let recovered = parsed.tensor_f32("w").expect("tensor");
        for (original, actual) in values.iter().zip(recovered.iter()) {
            assert!(
                (original - actual).abs() < 4.0 / 127.0,
                "{original} -> {actual}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantized_export_rejects_tensors_that_do_not_fill_a_block() {
        let dir = temp_dir("trustformers_gguf_export_ragged");
        let output = dir.join("model");
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            precision: ExportPrecision::INT4,
            ..Default::default()
        };

        // The default test model has a 4-element tensor, which no 32-wide block can hold.
        let err = GGUFExporter::new()
            .export(&TestModel::with_seed(1.0), &config)
            .expect_err("ragged tensors must be rejected, not padded silently");
        assert!(err.to_string().contains("block size"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
