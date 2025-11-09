//! Export factory for creating format-specific exporters
//!
//! This module provides a factory pattern for creating format-specific exporters
//! with proper configuration and validation.

use crate::export::*;
use crate::traits::Model;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::process::Command;

/// Factory for creating format-specific exporters
pub struct ExporterFactory {
    registered_exporters: HashMap<ExportFormat, Box<dyn ExporterProvider>>,
}

/// Trait for providing format-specific exporters
pub trait ExporterProvider: Send + Sync {
    fn create_exporter(&self) -> ConcreteExporter;
    fn get_format(&self) -> ExportFormat;
    fn get_requirements(&self) -> ExporterRequirements;
    fn validate_environment(&self) -> Result<()>;
}

/// Requirements for a specific exporter
#[derive(Debug, Clone)]
pub struct ExporterRequirements {
    pub required_features: Vec<String>,
    pub optional_features: Vec<String>,
    pub minimum_memory_gb: Option<f64>,
    pub supported_precisions: Vec<ExportPrecision>,
    pub max_model_size_gb: Option<f64>,
}

impl Default for ExporterFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl ExporterFactory {
    /// Create a new exporter factory
    pub fn new() -> Self {
        let mut factory = Self {
            registered_exporters: HashMap::new(),
        };

        // Register default exporters
        factory.register_default_exporters();
        factory
    }

    /// Register default exporters for all supported formats
    fn register_default_exporters(&mut self) {
        self.register_exporter(Box::new(ONNXProvider::new()));
        self.register_exporter(Box::new(GGMLProvider::new()));
        self.register_exporter(Box::new(GGUFProvider::new()));
        self.register_exporter(Box::new(NNEFProvider::new()));
        self.register_exporter(Box::new(OpenVINOProvider::new()));
        self.register_exporter(Box::new(TensorRTProvider::new()));
        self.register_exporter(Box::new(TVMProvider::new()));
        self.register_exporter(Box::new(CoreMLProvider::new()));
    }

    /// Register a new exporter provider
    pub fn register_exporter(&mut self, provider: Box<dyn ExporterProvider>) {
        let format = provider.get_format();
        self.registered_exporters.insert(format, provider);
    }

    /// Create an exporter for a specific format
    pub fn create_exporter(&self, format: ExportFormat) -> Result<ConcreteExporter> {
        let provider = self
            .registered_exporters
            .get(&format)
            .ok_or_else(|| anyhow!("No exporter registered for format: {:?}", format))?;

        // Validate environment before creating exporter
        provider.validate_environment()?;

        Ok(provider.create_exporter())
    }

    /// Get requirements for a specific format
    pub fn get_requirements(&self, format: ExportFormat) -> Result<ExporterRequirements> {
        let provider = self
            .registered_exporters
            .get(&format)
            .ok_or_else(|| anyhow!("No exporter registered for format: {:?}", format))?;

        Ok(provider.get_requirements())
    }

    /// Get all supported formats
    pub fn supported_formats(&self) -> Vec<ExportFormat> {
        self.registered_exporters.keys().cloned().collect()
    }

    /// Validate if a format is supported for a specific model
    pub fn validate_export<M: Model>(
        &self,
        model: &M,
        config: &ExportConfig,
    ) -> Result<ValidationResult> {
        let provider = self
            .registered_exporters
            .get(&config.format)
            .ok_or_else(|| anyhow!("Unsupported export format: {:?}", config.format))?;

        let requirements = provider.get_requirements();
        let mut validation = ValidationResult::new();

        // Check precision support
        if !requirements.supported_precisions.contains(&config.precision) {
            validation.add_error(format!(
                "Precision {:?} not supported for format {:?}. Supported: {:?}",
                config.precision, config.format, requirements.supported_precisions
            ));
        }

        // Check model size if applicable
        if let Some(max_size) = requirements.max_model_size_gb {
            let model_size_gb = estimate_model_size(model)?;
            if model_size_gb > max_size {
                validation.add_warning(format!(
                    "Model size ({:.2} GB) exceeds recommended maximum ({:.2} GB) for format {:?}",
                    model_size_gb, max_size, config.format
                ));
            }
        }

        // Check memory requirements
        if let Some(min_memory) = requirements.minimum_memory_gb {
            let available_memory = get_available_memory_gb()?;
            if available_memory < min_memory {
                validation.add_error(format!(
                    "Insufficient memory: {:.2} GB available, {:.2} GB required",
                    available_memory, min_memory
                ));
            }
        }

        // Validate environment
        if let Err(e) = provider.validate_environment() {
            validation.add_error(format!("Environment validation failed: {}", e));
        }

        Ok(validation)
    }

    /// Export with automatic format selection based on requirements
    pub fn export_with_best_format<M: Model>(
        &self,
        model: &M,
        output_path: &str,
        constraints: &ExportConstraints,
    ) -> Result<ExportResult> {
        let best_format = self.select_best_format(model, constraints)?;

        let config = ExportConfig {
            format: best_format,
            output_path: output_path.to_string(),
            precision: constraints.preferred_precision,
            optimize: constraints.optimize,
            ..Default::default()
        };

        let start_time = Instant::now();
        let exporter = self.create_exporter(best_format)?;

        // Track optimizations that will be applied
        let mut optimizations_applied = Vec::new();
        if config.optimize {
            optimizations_applied.push("General optimization".to_string());
        }
        if config.precision != ExportPrecision::FP32 {
            optimizations_applied.push(format!("Precision optimization: {:?}", config.precision));
        }

        exporter.export(model, &config)?;
        let export_time_ms = start_time.elapsed().as_millis() as u64;

        // Measure output file size
        let output_size_bytes = if Path::new(output_path).exists() {
            fs::metadata(output_path).map(|metadata| metadata.len()).unwrap_or(0)
        } else {
            0
        };

        Ok(ExportResult {
            format: best_format,
            output_path: output_path.to_string(),
            optimizations_applied,
            export_time_ms,
            output_size_bytes,
        })
    }

    /// Select the best export format based on constraints
    fn select_best_format<M: Model>(
        &self,
        model: &M,
        constraints: &ExportConstraints,
    ) -> Result<ExportFormat> {
        let mut candidates = Vec::new();

        for &format in &self.supported_formats() {
            let validation = self.validate_export(
                model,
                &ExportConfig {
                    format,
                    precision: constraints.preferred_precision,
                    ..Default::default()
                },
            )?;

            if validation.is_valid() {
                let score = self.score_format(format, constraints);
                candidates.push((format, score));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        candidates
            .first()
            .map(|(format, _)| *format)
            .ok_or_else(|| anyhow!("No suitable export format found"))
    }

    /// Score a format based on constraints (higher is better)
    fn score_format(&self, format: ExportFormat, constraints: &ExportConstraints) -> f64 {
        let mut score = 0.0;

        // Prefer formats based on target platform
        match (&constraints.target_platform, format) {
            (Some(TargetPlatform::Mobile), ExportFormat::CoreML) => score += 10.0,
            (Some(TargetPlatform::Mobile), ExportFormat::ONNX) => score += 8.0,
            (Some(TargetPlatform::Server), ExportFormat::TensorRT) => score += 10.0,
            (Some(TargetPlatform::Server), ExportFormat::ONNX) => score += 9.0,
            (Some(TargetPlatform::Edge), ExportFormat::GGUF) => score += 10.0,
            (Some(TargetPlatform::Edge), ExportFormat::GGML) => score += 9.0,
            _ => score += 5.0,
        }

        // Prefer formats that support required features
        if let Ok(requirements) = self.get_requirements(format) {
            if requirements.supported_precisions.contains(&constraints.preferred_precision) {
                score += 5.0;
            }
        }

        score
    }
}

/// Constraints for automatic format selection
#[derive(Debug, Clone)]
pub struct ExportConstraints {
    pub target_platform: Option<TargetPlatform>,
    pub preferred_precision: ExportPrecision,
    pub max_file_size_mb: Option<f64>,
    pub optimize: bool,
    pub require_quantization: bool,
}

/// Target platforms for deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    Mobile,
    Server,
    Edge,
    Browser,
    Embedded,
}

/// Result of export operation
#[derive(Debug, Clone)]
pub struct ExportResult {
    pub format: ExportFormat,
    pub output_path: String,
    pub optimizations_applied: Vec<String>,
    pub export_time_ms: u64,
    pub output_size_bytes: u64,
}

/// Validation result for export operations
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

// Provider implementations for each format
struct ONNXProvider;
struct GGMLProvider;
struct GGUFProvider;
struct NNEFProvider;
struct OpenVINOProvider;
struct TensorRTProvider;
struct TVMProvider;
struct CoreMLProvider;

impl ONNXProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for ONNXProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::ONNX(ONNXExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::ONNX
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec!["onnx".to_string()],
            optional_features: vec!["onnxruntime".to_string()],
            minimum_memory_gb: Some(2.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
            ],
            max_model_size_gb: Some(10.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // Check if ONNX is available
        // This would check for onnx library availability
        Ok(())
    }
}

impl GGMLProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for GGMLProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::GGML(GGMLExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::GGML
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec![],
            optional_features: vec!["quantization".to_string()],
            minimum_memory_gb: Some(1.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
                ExportPrecision::INT4,
            ],
            max_model_size_gb: None,
        }
    }

    fn validate_environment(&self) -> Result<()> {
        Ok(())
    }
}

impl GGUFProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for GGUFProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::GGUF(GGUFExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::GGUF
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec![],
            optional_features: vec!["quantization".to_string()],
            minimum_memory_gb: Some(1.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
                ExportPrecision::INT4,
            ],
            max_model_size_gb: None,
        }
    }

    fn validate_environment(&self) -> Result<()> {
        Ok(())
    }
}

impl NNEFProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for NNEFProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::NNEF(NNEFExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::NNEF
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec![],
            optional_features: vec!["quantization".to_string()],
            minimum_memory_gb: Some(1.5),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
                ExportPrecision::INT4,
            ],
            max_model_size_gb: Some(15.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // NNEF is a standard format with no specific dependencies
        Ok(())
    }
}

impl OpenVINOProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for OpenVINOProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::OpenVINO(OpenVINOExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::OpenVINO
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec!["openvino".to_string()],
            optional_features: vec!["gpu".to_string(), "vpu".to_string()],
            minimum_memory_gb: Some(2.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
            ],
            max_model_size_gb: Some(20.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // Check for OpenVINO runtime availability
        // This would check for openvino libraries
        Ok(())
    }
}

impl TensorRTProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for TensorRTProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::TensorRT(TensorRTExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::TensorRT
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec!["cuda".to_string(), "tensorrt".to_string()],
            optional_features: vec![],
            minimum_memory_gb: Some(4.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
            ],
            max_model_size_gb: Some(20.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // Check for CUDA and TensorRT availability
        Ok(())
    }
}

impl TVMProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for TVMProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::TVM(TVMExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::TVM
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec!["tvm".to_string()],
            optional_features: vec![
                "cuda".to_string(),
                "opencl".to_string(),
                "vulkan".to_string(),
            ],
            minimum_memory_gb: Some(3.0),
            supported_precisions: vec![
                ExportPrecision::FP32,
                ExportPrecision::FP16,
                ExportPrecision::INT8,
                ExportPrecision::INT4,
            ],
            max_model_size_gb: Some(25.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // Check for TVM runtime availability
        // This would check for TVM libraries and compilation tools
        Ok(())
    }
}

impl CoreMLProvider {
    fn new() -> Self {
        Self
    }
}

impl ExporterProvider for CoreMLProvider {
    fn create_exporter(&self) -> ConcreteExporter {
        ConcreteExporter::CoreML(CoreMLExporter::new())
    }

    fn get_format(&self) -> ExportFormat {
        ExportFormat::CoreML
    }

    fn get_requirements(&self) -> ExporterRequirements {
        ExporterRequirements {
            required_features: vec!["coreml".to_string()],
            optional_features: vec!["metal".to_string()],
            minimum_memory_gb: Some(2.0),
            supported_precisions: vec![ExportPrecision::FP32, ExportPrecision::FP16],
            max_model_size_gb: Some(5.0),
        }
    }

    fn validate_environment(&self) -> Result<()> {
        // Check if running on macOS
        #[cfg(not(target_os = "macos"))]
        return Err(anyhow!("CoreML export is only supported on macOS"));

        #[cfg(target_os = "macos")]
        Ok(())
    }
}

// Utility functions
fn estimate_model_size<M: Model>(model: &M) -> Result<f64> {
    // Estimate model size based on parameters and data types
    let num_params = model.num_parameters();

    // Default to FP32 (4 bytes per parameter) for base estimate
    let mut base_size_bytes = num_params * 4;

    // Add overhead for model structure, metadata, etc. (typically 10-20%)
    let overhead_factor = 1.15;
    base_size_bytes = (base_size_bytes as f64 * overhead_factor) as usize;

    // Additional considerations for different model types
    // - Embeddings typically have additional vocabulary size overhead
    // - Transformer models have attention patterns and position embeddings
    // - Convolutional models have kernel weights

    // Estimate based on model architecture patterns
    let architecture_multiplier = if num_params > 1_000_000_000 {
        // Large models (1B+ params)
        1.2 // Extra overhead for large model optimizations
    } else if num_params > 100_000_000 {
        // Medium models (100M+ params)
        1.1 // Moderate overhead
    } else {
        1.0 // Small models have minimal overhead
    };

    let total_size_bytes = (base_size_bytes as f64 * architecture_multiplier) as usize;

    // Convert to GB
    let size_gb = total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    Ok(size_gb)
}

fn get_available_memory_gb() -> Result<f64> {
    // Check available system memory using system information

    #[cfg(target_os = "linux")]
    {
        // On Linux, parse /proc/meminfo
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            let gb = kb as f64 / (1024.0 * 1024.0);
                            return Ok(gb);
                        }
                    }
                }
            }
        }

        // Fallback: try to get total memory and estimate available as 70%
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            let total_gb = kb as f64 / (1024.0 * 1024.0);
                            return Ok(total_gb * 0.7); // Estimate 70% available
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, use vm_stat command
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                let mut pages_free = 0u64;
                let mut page_size = 4096u64; // Default page size

                for line in output_str.lines() {
                    if line.contains("Pages free:") {
                        if let Some(free_str) = line.split(':').nth(1) {
                            if let Ok(free) = free_str.trim().trim_end_matches('.').parse::<u64>() {
                                pages_free = free;
                            }
                        }
                    }
                    if line.contains("Mach Virtual Memory Statistics:") {
                        // Try to get page size, though it's usually 4KB on x86_64
                        page_size = 4096;
                    }
                }

                if pages_free > 0 {
                    let free_bytes = pages_free * page_size;
                    let free_gb = free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    return Ok(free_gb);
                }
            }
        }

        // Fallback: use sysctl to get total memory and estimate
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(total_bytes) = mem_str.trim().parse::<u64>() {
                    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    return Ok(total_gb * 0.6); // Conservative estimate: 60% available
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // On Windows, try to use wmic command
        if let Ok(output) = Command::new("wmic")
            .args(&["OS", "get", "FreePhysicalMemory", "/value"])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    if line.starts_with("FreePhysicalMemory=") {
                        if let Some(kb_str) = line.split('=').nth(1) {
                            if let Ok(kb) = kb_str.trim().parse::<u64>() {
                                let gb = kb as f64 / (1024.0 * 1024.0);
                                return Ok(gb);
                            }
                        }
                    }
                }
            }
        }

        // Fallback: get total memory and estimate
        if let Ok(output) = Command::new("wmic")
            .args(&["computersystem", "get", "TotalPhysicalMemory", "/value"])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    if line.starts_with("TotalPhysicalMemory=") {
                        if let Some(bytes_str) = line.split('=').nth(1) {
                            if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                                let total_gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                                return Ok(total_gb * 0.6); // Conservative estimate
                            }
                        }
                    }
                }
            }
        }
    }

    // Ultimate fallback: return a conservative estimate
    // Modern systems typically have at least 8GB, assume 4GB available
    Ok(4.0)
}

impl Default for ExportConstraints {
    fn default() -> Self {
        Self {
            target_platform: None,
            preferred_precision: ExportPrecision::FP32,
            max_file_size_mb: None,
            optimize: true,
            require_quantization: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exporter_factory_creation() {
        let factory = ExporterFactory::new();
        let formats = factory.supported_formats();

        assert!(formats.contains(&ExportFormat::ONNX));
        assert!(formats.contains(&ExportFormat::GGUF));
        assert!(formats.contains(&ExportFormat::NNEF));
        assert!(formats.contains(&ExportFormat::OpenVINO));
        assert!(formats.contains(&ExportFormat::TVM));
        assert!(formats.len() >= 6);
    }

    #[test]
    fn test_exporter_creation() {
        let factory = ExporterFactory::new();
        let exporter = factory.create_exporter(ExportFormat::ONNX);

        assert!(exporter.is_ok());
    }

    #[test]
    fn test_nnef_exporter_creation() {
        let factory = ExporterFactory::new();
        let exporter = factory.create_exporter(ExportFormat::NNEF);

        assert!(exporter.is_ok());
    }

    #[test]
    fn test_openvino_exporter_creation() {
        let factory = ExporterFactory::new();
        let exporter = factory.create_exporter(ExportFormat::OpenVINO);

        assert!(exporter.is_ok());
    }

    #[test]
    fn test_tvm_exporter_creation() {
        let factory = ExporterFactory::new();
        let exporter = factory.create_exporter(ExportFormat::TVM);

        assert!(exporter.is_ok());
    }

    #[test]
    fn test_requirements_access() {
        let factory = ExporterFactory::new();
        let requirements = factory.get_requirements(ExportFormat::ONNX);

        assert!(requirements.is_ok());
        let req = requirements.unwrap();
        assert!(!req.supported_precisions.is_empty());
    }

    #[test]
    fn test_validation_result() {
        let mut validation = ValidationResult::new();
        assert!(validation.is_valid());

        validation.add_warning("Test warning".to_string());
        assert!(validation.is_valid());

        validation.add_error("Test error".to_string());
        assert!(!validation.is_valid());
    }

    #[test]
    fn test_export_constraints_default() {
        let constraints = ExportConstraints::default();
        assert!(matches!(
            constraints.preferred_precision,
            ExportPrecision::FP32
        ));
        assert!(constraints.optimize);
        assert!(!constraints.require_quantization);
    }
}
