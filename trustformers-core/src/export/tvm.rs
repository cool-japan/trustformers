//! TVM (Tensor Virtual Machine) export functionality
//!
//! This module provides TVM export capabilities for TrustformeRS models.
//! TVM is Apache's open-source deep learning compilation framework.

//! # Why no TVM module is written
//!
//! A TVM deployment artifact is a compiled shared object plus a Relay IR graph and
//! a parameter blob. Producing it requires (a) the model's operation graph and
//! (b) the TVM compiler itself. [`Model`] exposes parameters (via
//! [`Model::named_tensors`]) but not topology, and TVM is not a Rust dependency of
//! this workspace.
//!
//! Earlier revisions emitted a fixed 12-block transformer Relay IR, a generated
//! parameter blob and a JSON "placeholder module" under the `.so` name, and
//! reported success. Those artifacts described a model that did not exist, so they
//! are no longer produced: [`TVMExporter::export`] returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation).
//!
//! Use the GGUF or GGML exporters to write the model's real parameters; they are
//! tensor containers and need no topology.

use crate::errors::unsupported_operation;
use crate::export::{ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::traits::Model;
use anyhow::{anyhow, Result};

/// Explanation attached to every refusal to write a TVM module.
pub const TVM_UNSUPPORTED_REASON: &str =
    "a TVM deployment artifact is the output of the TVM compiler over the model's \
     Relay IR graph; the `Model` trait exposes parameters only (`named_tensors`), \
     and the TVM compiler is not available in this pure-Rust build. TrustformeRS \
     will not emit a synthesized module under a real model's name.";

/// TVM exporter for TrustformeRS models
#[derive(Clone)]
pub struct TVMExporter {
    target: String,
    target_host: Option<String>,
    optimization_level: u8,
    enable_auto_scheduler: bool,
    enable_meta_schedule: bool,
}

/// TVM target configuration
#[derive(Clone, Debug)]
pub struct TVMTargetConfig {
    pub device: String,
    pub arch: Option<String>,
    pub keys: Vec<String>,
    pub libs: Vec<String>,
}

impl TVMExporter {
    /// Create a new TVM exporter
    pub fn new() -> Self {
        Self {
            target: "llvm".to_string(),
            target_host: None,
            optimization_level: 3,
            enable_auto_scheduler: true,
            enable_meta_schedule: false,
        }
    }

    /// Create TVM exporter with custom configuration
    pub fn with_config(
        target: String,
        target_host: Option<String>,
        optimization_level: u8,
        enable_auto_scheduler: bool,
        enable_meta_schedule: bool,
    ) -> Self {
        Self {
            target,
            target_host,
            optimization_level,
            enable_auto_scheduler,
            enable_meta_schedule,
        }
    }

    /// The TVM target triple this exporter was configured with.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// The TVM host target, when one was configured.
    pub fn target_host(&self) -> Option<&str> {
        self.target_host.as_deref()
    }

    /// Whether the auto-scheduler was requested.
    pub fn auto_scheduler_enabled(&self) -> bool {
        self.enable_auto_scheduler
    }

    /// Whether meta-schedule tuning was requested.
    pub fn meta_schedule_enabled(&self) -> bool {
        self.enable_meta_schedule
    }

    /// Convert export precision to the corresponding TVM dtype string.
    pub fn precision_to_tvm_dtype(&self, precision: ExportPrecision) -> &'static str {
        match precision {
            ExportPrecision::FP32 => "float32",
            ExportPrecision::FP16 => "float16",
            ExportPrecision::INT8 => "int8",
            ExportPrecision::INT4 => "int4",
        }
    }

    /// Get device type for runtime
    pub fn get_device_type(&self) -> u8 {
        match self.target.as_str() {
            target if target.starts_with("cuda") => 2,   // kDLCUDA
            target if target.starts_with("opencl") => 4, // kDLOpenCL
            target if target.starts_with("vulkan") => 7, // kDLVulkan
            target if target.starts_with("metal") => 8,  // kDLMetal
            _ => 1,                                      // kDLCPU
        }
    }

    /// Get number of threads for CPU target
    pub fn get_num_threads(&self) -> u8 {
        if self.target.starts_with("llvm") {
            std::thread::available_parallelism().map(|n| n.get() as u8).unwrap_or(4)
        } else {
            1
        }
    }

    /// Validate TVM export configuration
    pub fn validate_config(&self, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::TVM {
            return Err(anyhow!(
                "Invalid format for TVM exporter: {:?}",
                config.format
            ));
        }

        // Validate target
        let valid_targets = [
            "llvm", "cuda", "opencl", "vulkan", "metal", "rocm", "hexagon",
        ];

        if !valid_targets.iter().any(|&t| self.target.starts_with(t)) {
            return Err(anyhow!("Unsupported TVM target: {}", self.target));
        }

        // Validate optimization level
        if self.optimization_level > 4 {
            return Err(anyhow!(
                "Invalid optimization level: {}",
                self.optimization_level
            ));
        }

        Ok(())
    }
}

impl ModelExporter for TVMExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no TVM module is written.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.validate_config(config)?;
        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing topology is a limitation of the `Model` trait.
        let _tensors = crate::export::collect_model_tensors(model)?;
        Err(unsupported_operation("TVM module export", TVM_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::TVM]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::TVM {
            return Err(anyhow!("TVM exporter only supports TVM format"));
        }
        Ok(())
    }
}

impl Default for TVMExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::export::test_support::TestModel;

    #[test]
    fn test_tvm_exporter_creation() {
        let exporter = TVMExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats, vec![ExportFormat::TVM]);
        assert_eq!(exporter.target, "llvm");
        assert_eq!(exporter.optimization_level, 3);
        assert!(exporter.enable_auto_scheduler);
        assert!(!exporter.enable_meta_schedule);
    }

    #[test]
    fn test_tvm_exporter_with_config() {
        let exporter =
            TVMExporter::with_config("cuda".to_string(), Some("llvm".to_string()), 4, false, true);

        assert_eq!(exporter.target, "cuda");
        assert_eq!(exporter.target_host, Some("llvm".to_string()));
        assert_eq!(exporter.optimization_level, 4);
        assert!(!exporter.enable_auto_scheduler);
        assert!(exporter.enable_meta_schedule);
    }

    #[test]
    fn test_precision_to_tvm_dtype() {
        let exporter = TVMExporter::new();
        assert_eq!(
            exporter.precision_to_tvm_dtype(ExportPrecision::FP32),
            "float32"
        );
        assert_eq!(
            exporter.precision_to_tvm_dtype(ExportPrecision::FP16),
            "float16"
        );
        assert_eq!(
            exporter.precision_to_tvm_dtype(ExportPrecision::INT8),
            "int8"
        );
        assert_eq!(
            exporter.precision_to_tvm_dtype(ExportPrecision::INT4),
            "int4"
        );
    }

    #[test]
    fn test_get_device_type() {
        let llvm_exporter = TVMExporter::with_config("llvm".to_string(), None, 3, true, false);
        let cuda_exporter = TVMExporter::with_config("cuda".to_string(), None, 3, true, false);
        let opencl_exporter = TVMExporter::with_config("opencl".to_string(), None, 3, true, false);

        assert_eq!(llvm_exporter.get_device_type(), 1); // CPU
        assert_eq!(cuda_exporter.get_device_type(), 2); // CUDA
        assert_eq!(opencl_exporter.get_device_type(), 4); // OpenCL
    }

    #[test]
    fn test_validate_config_success() {
        let exporter = TVMExporter::new();
        let config = ExportConfig {
            format: ExportFormat::TVM,
            precision: ExportPrecision::FP32,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_format() {
        let exporter = TVMExporter::new();
        let config = ExportConfig {
            format: ExportFormat::ONNX,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_invalid_target() {
        let exporter = TVMExporter::with_config("invalid_target".to_string(), None, 3, true, false);
        let config = ExportConfig {
            format: ExportFormat::TVM,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_invalid_optimization_level() {
        let exporter = TVMExporter::with_config(
            "llvm".to_string(),
            None,
            5, // Invalid level
            true,
            false,
        );
        let config = ExportConfig {
            format: ExportFormat::TVM,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_model_success() {
        let exporter = TVMExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::TVM).is_ok());
    }

    #[test]
    fn test_validate_model_wrong_format() {
        let exporter = TVMExporter::new();
        let model = TestModel::with_seed(1.0);

        assert!(exporter.validate_model(&model, ExportFormat::ONNX).is_err());
    }

    /// Regression test for the exporter that used to write a fixed 12-block Relay
    /// IR plus a generated parameter blob for any model.
    #[test]
    fn export_refuses_to_write_a_synthesized_module() {
        let dir = std::env::temp_dir().join("trustformers_tvm_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let exporter = TVMExporter::new();
        let model = TestModel::with_seed(4.0);
        let config = ExportConfig {
            format: ExportFormat::TVM,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("must not fabricate a module");
        assert!(
            err.to_string().contains("Unsupported operation"),
            "expected UnsupportedOperation, got: {err}"
        );
        assert!(
            !output.with_extension("so").exists(),
            "no .so may be produced"
        );
        assert!(
            !output.with_extension("json").exists(),
            "no Relay IR may be produced"
        );
        assert!(
            !output.with_extension("params").exists(),
            "no params blob may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_reports_missing_weights_before_missing_topology() {
        let exporter = TVMExporter::new();
        let model = TestModel::empty();
        let config = ExportConfig {
            format: ExportFormat::TVM,
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("no weights, no export");
        assert!(
            err.to_string().contains("named_tensors"),
            "expected the missing-weights diagnostic, got: {err}"
        );
    }
}
