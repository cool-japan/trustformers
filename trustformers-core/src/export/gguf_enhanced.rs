//! Enhanced GGUF export: explicit quantization choice and caller-supplied metadata.
//!
//! This exporter shares the binary implementation in [`super::gguf_format`] with
//! [`super::gguf`]; it adds an explicit [`GGUFTensorType`] selection and a builder
//! for extra metadata keys.
//!
//! Like every exporter in this crate it writes the model's **real** parameters,
//! taken from [`Model::named_tensors`]. A
//! previous revision of this file ignored the model and emitted a hard-coded
//! 12-block GPT-2 skeleton of `0.1f32` weights; that code is gone.

use super::gguf_format::{
    dequantize, payload_from_tensors, quantize, read_gguf_file, write_gguf_file, GGUFPayload,
    GGUF_DEFAULT_ALIGNMENT, GGUF_VERSION,
};
use super::{collect_model_tensors, ExportConfig, ExportFormat, ModelExporter};
use crate::traits::{Config, Model};
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;
use std::path::Path;

pub use super::gguf_format::{
    GGUFHeader, GGUFTensorInfo, GGUFTensorType, GGUFValue, GGUFValueType,
};

/// GGUF exporter with an explicit quantization type and extensible metadata.
#[derive(Clone, Debug)]
pub struct GGUFExporter {
    quantization_type: GGUFTensorType,
    metadata: BTreeMap<String, GGUFValue>,
}

impl Default for GGUFExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GGUFExporter {
    /// Create an exporter that writes `F32` tensors.
    pub fn new() -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.alignment".to_string(),
            GGUFValue::UInt32(GGUF_DEFAULT_ALIGNMENT as u32),
        );
        metadata.insert(
            "general.quantization_version".to_string(),
            GGUFValue::UInt32(2),
        );
        metadata.insert(
            "general.file_type".to_string(),
            GGUFValue::UInt32(GGUFTensorType::F32.file_type()),
        );

        Self {
            quantization_type: GGUFTensorType::F32,
            metadata,
        }
    }

    /// Select the tensor type used for every exported weight.
    pub fn with_quantization(mut self, tensor_type: GGUFTensorType) -> Self {
        self.quantization_type = tensor_type;
        self.metadata.insert(
            "general.file_type".to_string(),
            GGUFValue::UInt32(tensor_type.file_type()),
        );
        self
    }

    /// The tensor type this exporter will write.
    pub fn quantization_type(&self) -> GGUFTensorType {
        self.quantization_type
    }

    /// Add or replace a metadata key.
    pub fn add_metadata(mut self, key: String, value: GGUFValue) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Read-only view of the metadata that will be written.
    pub fn metadata(&self) -> &BTreeMap<String, GGUFValue> {
        &self.metadata
    }

    /// Declare llama-family architecture metadata.
    ///
    /// These values come from the caller; nothing here is measured from a model.
    /// No tokenizer keys are written, because this exporter has no tokenizer: a
    /// consumer that needs `tokenizer.ggml.*` must add it with [`Self::add_metadata`].
    #[allow(clippy::too_many_arguments)]
    pub fn set_architecture_metadata(
        mut self,
        context_length: u64,
        embedding_length: u64,
        block_count: u64,
        feed_forward_length: u64,
        head_count: u64,
        head_count_kv: Option<u64>,
        vocab_size: u64,
    ) -> Self {
        self.metadata.insert(
            "llama.context_length".to_string(),
            GGUFValue::UInt64(context_length),
        );
        self.metadata.insert(
            "llama.embedding_length".to_string(),
            GGUFValue::UInt64(embedding_length),
        );
        self.metadata.insert(
            "llama.block_count".to_string(),
            GGUFValue::UInt64(block_count),
        );
        self.metadata.insert(
            "llama.feed_forward_length".to_string(),
            GGUFValue::UInt64(feed_forward_length),
        );
        self.metadata.insert(
            "llama.attention.head_count".to_string(),
            GGUFValue::UInt64(head_count),
        );
        if let Some(kv_heads) = head_count_kv {
            self.metadata.insert(
                "llama.attention.head_count_kv".to_string(),
                GGUFValue::UInt64(kv_heads),
            );
        }
        self.metadata.insert(
            "llama.vocab_size".to_string(),
            GGUFValue::UInt64(vocab_size),
        );
        self
    }

    /// Build the in-memory payload for `model`, using only its real parameters.
    pub fn build_payload<M: Model>(&self, model: &M) -> Result<GGUFPayload> {
        let tensors = collect_model_tensors(model)?;

        let mut metadata = self.metadata.clone();
        metadata
            .entry("general.architecture".to_string())
            .or_insert_with(|| GGUFValue::String(model.get_config().architecture().to_string()));
        metadata.insert(
            "general.parameter_count".to_string(),
            GGUFValue::UInt64(model.num_parameters() as u64),
        );

        payload_from_tensors(metadata, &tensors, self.quantization_type)
    }
}

impl ModelExporter for GGUFExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::GGUF {
            return Err(anyhow!("GGUFExporter only supports GGUF format"));
        }

        let payload = self.build_payload(model)?;
        let output_path = format!("{}.gguf", config.output_path);
        write_gguf_file(&output_path, &payload)?;

        log::info!(
            "wrote {} tensors as {:?} to {}",
            payload.tensors.len(),
            self.quantization_type,
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

/// GGUF file utilities.
pub struct GGUFConverter;

impl GGUFConverter {
    /// Requantize every tensor of a GGUF file into `target_type`.
    ///
    /// This performs the real conversion: each tensor is decoded to `f32` with the
    /// codec its directory entry declares, re-encoded into `target_type`, and the
    /// file is rebuilt with fresh offsets and an updated `general.file_type`.
    /// Metadata other than `general.file_type` is preserved verbatim.
    ///
    /// A previous revision of this function copied the input file byte for byte
    /// while printing that it had requantized the model.
    ///
    /// # Errors
    ///
    /// Fails if the source uses a codec this crate cannot decode, if `target_type`
    /// has no encoder here, or if a tensor's element count is not a multiple of the
    /// target block size. In every case no output file is written.
    pub fn convert_quantization<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        target_type: GGUFTensorType,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        if !target_type.is_supported_codec() {
            return Err(crate::errors::unsupported_operation(
                format!("GGUF requantization to {target_type:?}"),
                "TrustformeRS implements the F32, F16, F64, Q8_0 and Q4_0 GGUF codecs",
            )
            .into());
        }

        let source = read_gguf_file(input_path)?;

        let mut metadata = source.metadata.clone();
        metadata.insert(
            "general.file_type".to_string(),
            GGUFValue::UInt32(target_type.file_type()),
        );
        let alignment = source.alignment();

        let mut converted = GGUFPayload {
            metadata,
            tensors: Vec::with_capacity(source.tensors.len()),
        };

        let mut offset = 0u64;
        for (info, data) in &source.tensors {
            let element_count = info.element_count() as usize;
            let values = dequantize(data, info.tensor_type, element_count)
                .map_err(|e| anyhow!("tensor '{}': {e}", info.name))?;
            let encoded = quantize(&values, target_type)
                .map_err(|e| anyhow!("tensor '{}': {e}", info.name))?;

            let mut new_info = info.clone();
            new_info.tensor_type = target_type;
            new_info.offset = offset;
            offset = super::gguf_format::align_up(offset + encoded.len() as u64, alignment);
            converted.tensors.push((new_info, encoded));
        }

        write_gguf_file(output_path, &converted)?;
        Ok(())
    }

    /// Validate a GGUF file by parsing it in full.
    ///
    /// Every tensor's declared extent is checked against the file length, so a
    /// truncated or corrupt file fails here rather than at load time.
    pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<GGUFValidationReport> {
        let path = path.as_ref();
        let file_size = std::fs::metadata(path)?.len();
        let payload = read_gguf_file(path)?;

        let mut warnings = Vec::new();
        for (info, _) in &payload.tensors {
            if !info.tensor_type.is_supported_codec() {
                warnings.push(format!(
                    "tensor '{}' uses {:?}, which this build cannot decode",
                    info.name, info.tensor_type
                ));
            }
        }
        if !payload.metadata.contains_key("general.architecture") {
            warnings.push("missing 'general.architecture' metadata key".to_string());
        }

        Ok(GGUFValidationReport {
            is_valid: true,
            version: GGUF_VERSION,
            tensor_count: payload.tensors.len() as u64,
            metadata_count: payload.metadata.len() as u64,
            file_size,
            errors: Vec::new(),
            warnings,
        })
    }
}

/// Result of [`GGUFConverter::validate_file`].
#[derive(Debug, Clone)]
pub struct GGUFValidationReport {
    pub is_valid: bool,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub file_size: u64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::test_support::{TestConfig, TestModel};
    use crate::tensor::Tensor;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// A model whose single tensor fills whole 32-element quantization blocks.
    fn block_aligned_model(seed: f32) -> TestModel {
        let values: Vec<f32> = (0..64).map(|i| seed + (i as f32 - 32.0) * 0.125).collect();
        TestModel::new(
            TestConfig::default(),
            vec![(
                "blk.0.attn_q.weight".to_string(),
                Tensor::from_vec(values, &[2, 32]).expect("tensor"),
            )],
        )
    }

    #[test]
    fn test_gguf_tensor_type_properties() {
        assert_eq!(GGUFTensorType::F32.type_size(), 4);
        assert_eq!(GGUFTensorType::F16.type_size(), 2);
        assert_eq!(GGUFTensorType::Q4_0.block_size(), 32);
        assert_eq!(GGUFTensorType::Q4_0.type_size(), 18);
        assert_eq!(GGUFTensorType::Q8_0.type_size(), 34);
        assert!(GGUFTensorType::Q8_0.is_quantized());
        assert!(!GGUFTensorType::F32.is_quantized());
    }

    #[test]
    fn test_gguf_value_types() {
        assert!(matches!(
            GGUFValue::Int32(42).value_type(),
            GGUFValueType::Int32
        ));
        assert!(matches!(
            GGUFValue::String("test".to_string()).value_type(),
            GGUFValueType::String
        ));
        assert!(matches!(
            GGUFValue::Array(GGUFValueType::Int32, vec![GGUFValue::Int32(1)]).value_type(),
            GGUFValueType::Array
        ));
    }

    #[test]
    fn test_gguf_exporter_creation() {
        let exporter = GGUFExporter::new();
        assert_eq!(exporter.quantization_type(), GGUFTensorType::F32);
        assert!(!exporter.metadata().is_empty());
    }

    #[test]
    fn test_gguf_exporter_with_quantization() {
        let exporter = GGUFExporter::new().with_quantization(GGUFTensorType::Q4_0);
        assert_eq!(exporter.quantization_type(), GGUFTensorType::Q4_0);
    }

    #[test]
    fn test_gguf_exporter_metadata() {
        let exporter = GGUFExporter::new()
            .add_metadata(
                "custom.key".to_string(),
                GGUFValue::String("value".to_string()),
            )
            .set_architecture_metadata(2048, 768, 12, 3072, 12, Some(12), 50257);

        assert!(exporter.metadata().contains_key("custom.key"));
        assert!(exporter.metadata().contains_key("llama.context_length"));
        assert!(
            !exporter.metadata().contains_key("tokenizer.ggml.tokens"),
            "no empty tokenizer vocabulary may be claimed"
        );
    }

    #[test]
    fn test_gguf_value_serialization() -> Result<()> {
        let mut buffer = Vec::new();
        GGUFValue::String("test".to_string()).write_to_buffer(&mut buffer)?;
        assert_eq!(buffer.len(), 12, "u64 length prefix plus four bytes");
        Ok(())
    }

    #[test]
    fn test_supported_formats() {
        assert_eq!(
            GGUFExporter::new().supported_formats(),
            vec![ExportFormat::GGUF]
        );
    }

    #[test]
    fn export_writes_real_weights_and_varies_with_them() {
        let dir = temp_dir("trustformers_gguf_enhanced_export");

        let write = |seed: f32, name: &str| {
            let output = dir.join(name);
            let config = ExportConfig {
                format: ExportFormat::GGUF,
                output_path: output.to_string_lossy().to_string(),
                ..Default::default()
            };
            GGUFExporter::new().export(&block_aligned_model(seed), &config).expect("export");
            output.with_extension("gguf")
        };

        let path_a = write(0.0, "a");
        let path_b = write(5.0, "b");

        let bytes_a = std::fs::read(&path_a).expect("read a");
        let bytes_b = std::fs::read(&path_b).expect("read b");
        assert_ne!(
            bytes_a, bytes_b,
            "different weights must give different files"
        );

        let parsed = read_gguf_file(&path_a).expect("parse");
        let expected: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();
        assert_eq!(
            parsed.tensor_f32("blk.0.attn_q.weight").expect("tensor"),
            expected
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_refuses_a_model_without_named_tensors() {
        let dir = temp_dir("trustformers_gguf_enhanced_empty");
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
        assert!(!output.with_extension("gguf").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression test for the converter that used to `std::fs::copy` the input.
    #[test]
    fn convert_quantization_actually_requantizes() {
        let dir = temp_dir("trustformers_gguf_requantize");
        let source = dir.join("f32");
        let target = dir.join("q8.gguf");

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: source.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&block_aligned_model(0.0), &config).expect("export");
        let source_path = source.with_extension("gguf");

        GGUFConverter::convert_quantization(&source_path, &target, GGUFTensorType::Q8_0)
            .expect("requantize");

        let original_bytes = std::fs::read(&source_path).expect("read source");
        let converted_bytes = std::fs::read(&target).expect("read target");
        assert_ne!(
            original_bytes, converted_bytes,
            "a requantized file must not be a byte copy of the input"
        );
        assert!(
            converted_bytes.len() < original_bytes.len(),
            "Q8_0 must be smaller than F32: {} vs {}",
            converted_bytes.len(),
            original_bytes.len()
        );

        let parsed = read_gguf_file(&target).expect("parse target");
        assert_eq!(parsed.tensors[0].0.tensor_type, GGUFTensorType::Q8_0);
        assert_eq!(
            parsed.metadata.get("general.file_type"),
            Some(&GGUFValue::UInt32(GGUFTensorType::Q8_0.file_type()))
        );

        let expected: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();
        let recovered = parsed.tensor_f32("blk.0.attn_q.weight").expect("tensor");
        for (original, actual) in expected.iter().zip(recovered.iter()) {
            assert!(
                (original - actual).abs() < 4.0 / 127.0,
                "{original} -> {actual}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn convert_quantization_refuses_unimplemented_targets() {
        let dir = temp_dir("trustformers_gguf_requantize_unsupported");
        let source = dir.join("f32");
        let target = dir.join("q6k.gguf");

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: source.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&block_aligned_model(0.0), &config).expect("export");

        let err = GGUFConverter::convert_quantization(
            source.with_extension("gguf"),
            &target,
            GGUFTensorType::Q6K,
        )
        .expect_err("no Q6_K encoder exists");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
        assert!(!target.exists(), "no output file may be produced");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_gguf_converter_validation() -> Result<()> {
        let dir = temp_dir("trustformers_gguf_validate");
        let source = dir.join("model");

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: source.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&block_aligned_model(0.0), &config)?;

        let report = GGUFConverter::validate_file(source.with_extension("gguf"))?;
        assert!(report.is_valid);
        assert_eq!(report.version, GGUF_VERSION);
        assert_eq!(report.tensor_count, 1);
        assert!(report.metadata_count >= 4);

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn validate_file_rejects_truncated_files() {
        let dir = temp_dir("trustformers_gguf_truncated");
        let source = dir.join("model");
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: source.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&block_aligned_model(0.0), &config).expect("export");

        let path = source.with_extension("gguf");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes.truncate(bytes.len() - 64);
        std::fs::write(&path, &bytes).expect("truncate");

        let err = GGUFConverter::validate_file(&path).expect_err("truncated file must fail");
        assert!(err.to_string().contains("extends to byte"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn magic_constant_is_gguf_ascii() {
        assert_eq!(
            crate::export::gguf_format::GGUF_MAGIC.to_le_bytes(),
            *b"GGUF"
        );
    }
}
