//! OpenVINO export surface.
//!
//! OpenVINO IR is an *operation graph* (`model.xml`) paired with a flat weight blob
//! (`model.bin`) whose byte offsets are referenced from the XML.
//!
//! # Why no IR pair is written
//!
//! [`Model`] exposes parameters (via [`Model::named_tensors`]) but not topology, so
//! the `<layers>`/`<edges>` sections of the XML cannot be derived. Earlier revisions
//! emitted a fixed 768-wide, 12-block transformer XML and a `model.bin` of generated
//! filler bytes for every model, then reported success. That artifact described a
//! model that did not exist, so it is no longer produced:
//! [`OpenVINOExporter::export`] now returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation).
//!
//! Use the GGUF or GGML exporters to write the model's real parameters; they are
//! tensor containers and need no topology.

use crate::errors::unsupported_operation;
use crate::export::{ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::traits::Model;
use anyhow::{anyhow, Result};

/// Explanation attached to every refusal to write an OpenVINO IR pair.
pub const OPENVINO_UNSUPPORTED_REASON: &str =
    "an OpenVINO IR pair requires the model's operation graph and the byte offsets of \
     every weight inside `model.bin`; the `Model` trait exposes parameters only \
     (`named_tensors`), so TrustformeRS will not emit a synthesized graph under a \
     real model's name.";

/// OpenVINO exporter for TrustformeRS models
#[derive(Clone)]
pub struct OpenVINOExporter {
    version: String,
    optimize_for_device: String,
    precision_config: OpenVINOPrecisionConfig,
}

/// OpenVINO precision configuration
#[derive(Clone, Debug)]
pub struct OpenVINOPrecisionConfig {
    pub input_precision: String,
    pub output_precision: String,
    pub weights_precision: String,
    pub enable_int8_calibration: bool,
}

impl OpenVINOExporter {
    /// Create a new OpenVINO exporter
    pub fn new() -> Self {
        Self {
            version: "2024.3".to_string(),
            optimize_for_device: "CPU".to_string(),
            precision_config: OpenVINOPrecisionConfig::default(),
        }
    }

    /// Create OpenVINO exporter with custom configuration
    pub fn with_config(
        version: String,
        device: String,
        precision_config: OpenVINOPrecisionConfig,
    ) -> Self {
        Self {
            version,
            optimize_for_device: device,
            precision_config,
        }
    }

    /// The OpenVINO toolkit version this exporter would declare.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The device this exporter was configured to optimize for.
    pub fn target_device(&self) -> &str {
        &self.optimize_for_device
    }

    /// The precision configuration this exporter was built with.
    pub fn precision_config(&self) -> &OpenVINOPrecisionConfig {
        &self.precision_config
    }

    /// Convert export precision to the corresponding OpenVINO element type name.
    pub fn precision_to_openvino_type(&self, precision: ExportPrecision) -> &'static str {
        match precision {
            ExportPrecision::FP32 => "f32",
            ExportPrecision::FP16 => "f16",
            ExportPrecision::INT8 => "i8",
            ExportPrecision::INT4 => "i4",
        }
    }

    /// Validate an OpenVINO export configuration.
    pub fn validate_config(&self, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::OpenVINO {
            return Err(anyhow!(
                "Invalid format for OpenVINO exporter: {:?}",
                config.format
            ));
        }

        // Validate device support
        if !["CPU", "GPU", "MYRIAD", "HDDL", "GNA"].contains(&self.optimize_for_device.as_str()) {
            return Err(anyhow!(
                "Unsupported OpenVINO device: {}",
                self.optimize_for_device
            ));
        }

        // Validate precision support per device
        match (self.optimize_for_device.as_str(), config.precision) {
            ("CPU", ExportPrecision::FP32)
            | ("CPU", ExportPrecision::FP16)
            | ("CPU", ExportPrecision::INT8) => {},
            ("GPU", ExportPrecision::FP32) | ("GPU", ExportPrecision::FP16) => {},
            ("MYRIAD", ExportPrecision::FP16) => {},
            ("GNA", ExportPrecision::INT8) => {},
            (device, precision) => {
                return Err(anyhow!(
                    "Precision {:?} not supported on device {}",
                    precision,
                    device
                ));
            },
        }

        Ok(())
    }
}

impl Default for OpenVINOPrecisionConfig {
    fn default() -> Self {
        Self {
            input_precision: "FP32".to_string(),
            output_precision: "FP32".to_string(),
            weights_precision: "FP32".to_string(),
            enable_int8_calibration: false,
        }
    }
}

impl ModelExporter for OpenVINOExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no IR pair is written.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.validate_config(config)?;
        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing topology is a limitation of the `Model` trait.
        let _tensors = crate::export::collect_model_tensors(model)?;
        Err(unsupported_operation("OpenVINO IR export", OPENVINO_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::OpenVINO]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::OpenVINO {
            return Err(anyhow!("OpenVINO exporter only supports OpenVINO format"));
        }
        Ok(())
    }
}

impl Default for OpenVINOExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::export::test_support::TestModel;

    #[test]
    fn test_openvino_exporter_creation() {
        let exporter = OpenVINOExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats, vec![ExportFormat::OpenVINO]);
        assert_eq!(exporter.version, "2024.3");
        assert_eq!(exporter.optimize_for_device, "CPU");
    }

    #[test]
    fn test_openvino_exporter_with_config() {
        let precision_config = OpenVINOPrecisionConfig {
            input_precision: "FP16".to_string(),
            output_precision: "FP16".to_string(),
            weights_precision: "FP16".to_string(),
            enable_int8_calibration: true,
        };

        let exporter = OpenVINOExporter::with_config(
            "2024.4".to_string(),
            "GPU".to_string(),
            precision_config,
        );

        assert_eq!(exporter.version, "2024.4");
        assert_eq!(exporter.optimize_for_device, "GPU");
        assert_eq!(exporter.precision_config.input_precision, "FP16");
        assert!(exporter.precision_config.enable_int8_calibration);
    }

    #[test]
    fn test_precision_to_openvino_type() {
        let exporter = OpenVINOExporter::new();
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::FP32),
            "f32"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::FP16),
            "f16"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::INT8),
            "i8"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::INT4),
            "i4"
        );
    }

    #[test]
    fn test_validate_config_success() {
        let exporter = OpenVINOExporter::new();
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            precision: ExportPrecision::FP32,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_format() {
        let exporter = OpenVINOExporter::new();
        let config = ExportConfig {
            format: ExportFormat::ONNX,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_unsupported_device() {
        let exporter = OpenVINOExporter::with_config(
            "2024.3".to_string(),
            "INVALID_DEVICE".to_string(),
            OpenVINOPrecisionConfig::default(),
        );
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_unsupported_precision_device_combo() {
        let exporter = OpenVINOExporter::with_config(
            "2024.3".to_string(),
            "MYRIAD".to_string(),
            OpenVINOPrecisionConfig::default(),
        );
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            precision: ExportPrecision::FP32, // MYRIAD only supports FP16
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_model_success() {
        let exporter = OpenVINOExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::OpenVINO).is_ok());
    }

    #[test]
    fn test_validate_model_wrong_format() {
        let exporter = OpenVINOExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::ONNX).is_err());
    }

    #[test]
    fn test_openvino_precision_config_default() {
        let config = OpenVINOPrecisionConfig::default();
        assert_eq!(config.input_precision, "FP32");
        assert_eq!(config.output_precision, "FP32");
        assert_eq!(config.weights_precision, "FP32");
        assert!(!config.enable_int8_calibration);
    }

    /// Regression test for the exporter that used to write a fixed 12-block
    /// transformer `model.xml` plus a generated `model.bin` for any model.
    #[test]
    fn export_refuses_to_write_a_synthesized_ir() {
        let dir = std::env::temp_dir().join("trustformers_openvino_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let exporter = OpenVINOExporter::new();
        let model = TestModel::with_seed(2.0);
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("must not fabricate an IR");
        assert!(
            err.to_string().contains("Unsupported operation"),
            "expected UnsupportedOperation, got: {err}"
        );
        assert!(
            !output.with_extension("xml").exists(),
            "no model.xml may be produced"
        );
        assert!(
            !output.with_extension("bin").exists(),
            "no model.bin may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_reports_missing_weights_before_missing_topology() {
        let exporter = OpenVINOExporter::new();
        let model = TestModel::empty();
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("no weights, no export");
        assert!(
            err.to_string().contains("named_tensors"),
            "expected the missing-weights diagnostic, got: {err}"
        );
    }
}
