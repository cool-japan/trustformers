use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Comprehensive error code system for TrustformeRS Training
///
/// Error codes follow the pattern: COMPONENT_CATEGORY_SPECIFIC
/// where:
/// - COMPONENT: The module/component where the error occurred
/// - CATEGORY: The type of error (CONFIG, RUNTIME, RESOURCE, etc.)
/// - SPECIFIC: Specific error within the category

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCodeInfo {
    pub code: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub severity: &'static str,
    pub causes: Vec<&'static str>,
    pub solutions: Vec<&'static str>,
    pub documentation_url: Option<&'static str>,
    pub related_codes: Vec<&'static str>,
}

/// Comprehensive error code registry
pub struct ErrorCodeRegistry {
    codes: HashMap<&'static str, ErrorCodeInfo>,
}

impl Default for ErrorCodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorCodeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            codes: HashMap::new(),
        };
        registry.register_all_codes();
        registry
    }

    pub fn get_code_info(&self, code: &str) -> Option<&ErrorCodeInfo> {
        self.codes.get(code)
    }

    pub fn list_codes_by_component(&self, component: &str) -> Vec<&ErrorCodeInfo> {
        self.codes.values().filter(|info| info.code.starts_with(component)).collect()
    }

    fn register_all_codes(&mut self) {
        // Configuration Errors (CONFIG_*)
        self.register_code(ErrorCodeInfo {
            code: "CONFIG_INVALID_PARAM",
            name: "Invalid Parameter",
            description: "A configuration parameter has an invalid value",
            severity: "HIGH",
            causes: vec![
                "Parameter value outside valid range",
                "Incorrect parameter type",
                "Missing required parameter",
            ],
            solutions: vec![
                "Check parameter documentation",
                "Validate parameter ranges",
                "Use configuration schema validation",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/config"),
            related_codes: vec!["CONFIG_MISSING_PARAM", "CONFIG_TYPE_MISMATCH"],
        });

        self.register_code(ErrorCodeInfo {
            code: "CONFIG_MISSING_PARAM",
            name: "Missing Required Parameter",
            description: "A required configuration parameter is missing",
            severity: "CRITICAL",
            causes: vec![
                "Incomplete configuration file",
                "Parameter not provided in CLI",
                "Environment variable not set",
            ],
            solutions: vec![
                "Add missing parameter to config",
                "Check required parameters list",
                "Use configuration template",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/config"),
            related_codes: vec!["CONFIG_INVALID_PARAM"],
        });

        self.register_code(ErrorCodeInfo {
            code: "CONFIG_TYPE_MISMATCH",
            name: "Parameter Type Mismatch",
            description: "Configuration parameter has wrong type",
            severity: "HIGH",
            causes: vec![
                "String provided where number expected",
                "Invalid enum value",
                "Incorrect data structure",
            ],
            solutions: vec![
                "Check parameter type requirements",
                "Use correct data type",
                "Validate against schema",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/config"),
            related_codes: vec!["CONFIG_INVALID_PARAM"],
        });

        // Data Loading Errors (DATA_*)
        self.register_code(ErrorCodeInfo {
            code: "DATA_FILE_NOT_FOUND",
            name: "Dataset File Not Found",
            description: "Specified dataset file does not exist",
            severity: "CRITICAL",
            causes: vec![
                "Incorrect file path",
                "File moved or deleted",
                "Permission issues",
            ],
            solutions: vec![
                "Check file path spelling",
                "Verify file exists",
                "Check file permissions",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/data"),
            related_codes: vec!["DATA_PERMISSION_DENIED", "DATA_CORRUPT"],
        });

        self.register_code(ErrorCodeInfo {
            code: "DATA_CORRUPT",
            name: "Corrupted Dataset",
            description: "Dataset file is corrupted or malformed",
            severity: "HIGH",
            causes: vec![
                "Incomplete download",
                "File corruption",
                "Unsupported format",
            ],
            solutions: vec![
                "Re-download dataset",
                "Verify file integrity",
                "Check format compatibility",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/data"),
            related_codes: vec!["DATA_FORMAT_UNSUPPORTED"],
        });

        self.register_code(ErrorCodeInfo {
            code: "DATA_PERMISSION_DENIED",
            name: "Data Access Permission Denied",
            description: "Insufficient permissions to access dataset",
            severity: "CRITICAL",
            causes: vec![
                "File permission restrictions",
                "Directory access denied",
                "Network access blocked",
            ],
            solutions: vec![
                "Check file permissions",
                "Run with appropriate user",
                "Adjust filesystem permissions",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/data"),
            related_codes: vec!["DATA_FILE_NOT_FOUND"],
        });

        // Training Errors (TRAIN_*)
        self.register_code(ErrorCodeInfo {
            code: "TRAIN_NAN_LOSS",
            name: "NaN Loss Detected",
            description: "Training loss has become NaN (Not a Number)",
            severity: "CRITICAL",
            causes: vec![
                "Learning rate too high",
                "Gradient explosion",
                "Numerical instability",
                "Division by zero",
            ],
            solutions: vec![
                "Reduce learning rate",
                "Enable gradient clipping",
                "Check input data normalization",
                "Use mixed precision training",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/training"),
            related_codes: vec!["TRAIN_INF_LOSS", "TRAIN_GRADIENT_EXPLOSION"],
        });

        self.register_code(ErrorCodeInfo {
            code: "TRAIN_INF_LOSS",
            name: "Infinite Loss Detected",
            description: "Training loss has become infinite",
            severity: "CRITICAL",
            causes: vec![
                "Learning rate too high",
                "Unstable model architecture",
                "Overflow in calculations",
            ],
            solutions: vec![
                "Reduce learning rate",
                "Use gradient clipping",
                "Check model stability",
                "Enable loss scaling",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/training"),
            related_codes: vec!["TRAIN_NAN_LOSS", "TRAIN_GRADIENT_EXPLOSION"],
        });

        self.register_code(ErrorCodeInfo {
            code: "TRAIN_GRADIENT_EXPLOSION",
            name: "Gradient Explosion",
            description: "Gradients have grown too large",
            severity: "HIGH",
            causes: vec![
                "Learning rate too high",
                "Poor weight initialization",
                "Unstable optimization",
            ],
            solutions: vec![
                "Enable gradient clipping",
                "Reduce learning rate",
                "Improve weight initialization",
                "Use batch normalization",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/training"),
            related_codes: vec!["TRAIN_NAN_LOSS", "TRAIN_CONVERGENCE_FAILURE"],
        });

        self.register_code(ErrorCodeInfo {
            code: "TRAIN_CONVERGENCE_FAILURE",
            name: "Training Not Converging",
            description: "Model is not learning or converging",
            severity: "MEDIUM",
            causes: vec![
                "Learning rate too low",
                "Poor data quality",
                "Model capacity mismatch",
                "Optimization algorithm unsuitable",
            ],
            solutions: vec![
                "Increase learning rate",
                "Check data quality",
                "Adjust model size",
                "Try different optimizer",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/training"),
            related_codes: vec!["TRAIN_OVERFITTING", "TRAIN_UNDERFITTING"],
        });

        // Resource Errors (RESOURCE_*)
        self.register_code(ErrorCodeInfo {
            code: "RESOURCE_OOM",
            name: "Out of Memory",
            description: "System has run out of available memory",
            severity: "CRITICAL",
            causes: vec![
                "Batch size too large",
                "Model too large for available memory",
                "Memory leak",
                "Insufficient system memory",
            ],
            solutions: vec![
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use model parallelism",
                "Upgrade system memory",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/resource"),
            related_codes: vec!["RESOURCE_GPU_OOM", "RESOURCE_CPU_LIMIT"],
        });

        self.register_code(ErrorCodeInfo {
            code: "RESOURCE_GPU_OOM",
            name: "GPU Out of Memory",
            description: "GPU has run out of available memory",
            severity: "CRITICAL",
            causes: vec![
                "Batch size too large for GPU",
                "Model parameters exceed GPU memory",
                "Multiple processes using GPU",
            ],
            solutions: vec![
                "Reduce batch size",
                "Use gradient accumulation",
                "Enable GPU memory optimization",
                "Use model sharding",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/resource"),
            related_codes: vec!["RESOURCE_OOM", "RESOURCE_GPU_UNAVAILABLE"],
        });

        self.register_code(ErrorCodeInfo {
            code: "RESOURCE_GPU_UNAVAILABLE",
            name: "GPU Not Available",
            description: "Required GPU resources are not available",
            severity: "HIGH",
            causes: vec![
                "GPU not detected",
                "Driver issues",
                "GPU already in use",
                "CUDA/ROCm not installed",
            ],
            solutions: vec![
                "Check GPU installation",
                "Update GPU drivers",
                "Install CUDA/ROCm",
                "Use CPU fallback",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/resource"),
            related_codes: vec!["RESOURCE_CUDA_ERROR"],
        });

        // Network Errors (NETWORK_*)
        self.register_code(ErrorCodeInfo {
            code: "NETWORK_CONNECTION_TIMEOUT",
            name: "Network Connection Timeout",
            description: "Network connection timed out",
            severity: "MEDIUM",
            causes: vec![
                "Slow network connection",
                "Server overloaded",
                "Network configuration issues",
            ],
            solutions: vec![
                "Increase timeout value",
                "Check network connectivity",
                "Try different server",
                "Use offline mode",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/network"),
            related_codes: vec!["NETWORK_CONNECTION_REFUSED"],
        });

        // Hardware Errors (HARDWARE_*)
        self.register_code(ErrorCodeInfo {
            code: "HARDWARE_THERMAL_THROTTLING",
            name: "Thermal Throttling Detected",
            description: "Hardware is throttling due to temperature",
            severity: "MEDIUM",
            causes: vec![
                "High ambient temperature",
                "Inadequate cooling",
                "Intensive computational load",
            ],
            solutions: vec![
                "Improve cooling",
                "Reduce computational load",
                "Lower power limits",
                "Schedule training during cooler periods",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/hardware"),
            related_codes: vec!["HARDWARE_POWER_LIMIT"],
        });

        // Model Errors (MODEL_*)
        self.register_code(ErrorCodeInfo {
            code: "MODEL_INCOMPATIBLE_WEIGHTS",
            name: "Incompatible Model Weights",
            description: "Model weights are not compatible with architecture",
            severity: "HIGH",
            causes: vec![
                "Architecture mismatch",
                "Version incompatibility",
                "Corrupted checkpoint",
            ],
            solutions: vec![
                "Check model architecture",
                "Verify checkpoint version",
                "Re-download checkpoint",
                "Use compatible model version",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/model"),
            related_codes: vec!["MODEL_CHECKPOINT_CORRUPT"],
        });

        // Quantization Errors (QUANT_*)
        self.register_code(ErrorCodeInfo {
            code: "QUANT_CALIBRATION_FAILED",
            name: "Quantization Calibration Failed",
            description: "Failed to calibrate quantization parameters",
            severity: "HIGH",
            causes: vec![
                "Insufficient calibration data",
                "Extreme activation ranges",
                "Unsupported layer type",
            ],
            solutions: vec![
                "Increase calibration dataset size",
                "Check activation ranges",
                "Use different quantization method",
                "Pre-process calibration data",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/quantization"),
            related_codes: vec!["QUANT_UNSUPPORTED_OP"],
        });

        // Distributed Training Errors (DIST_*)
        self.register_code(ErrorCodeInfo {
            code: "DIST_COMMUNICATION_FAILURE",
            name: "Distributed Communication Failure",
            description: "Communication between nodes failed",
            severity: "HIGH",
            causes: vec![
                "Network partition",
                "Node failure",
                "Communication timeout",
                "Process crash",
            ],
            solutions: vec![
                "Check network connectivity",
                "Restart failed nodes",
                "Increase timeout values",
                "Use fault-tolerant training",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/distributed"),
            related_codes: vec!["DIST_NODE_FAILURE", "DIST_SYNC_TIMEOUT"],
        });

        self.register_code(ErrorCodeInfo {
            code: "DIST_RANK_MISMATCH",
            name: "Process Rank Mismatch",
            description: "Process rank configuration mismatch",
            severity: "CRITICAL",
            causes: vec![
                "Incorrect world size",
                "Duplicate rank assignment",
                "Process group misconfiguration",
            ],
            solutions: vec![
                "Check process configuration",
                "Verify world size setting",
                "Ensure unique rank assignment",
                "Restart distributed training",
            ],
            documentation_url: Some("https://docs.trustformers.rs/errors/distributed"),
            related_codes: vec!["DIST_COMMUNICATION_FAILURE"],
        });
    }

    fn register_code(&mut self, info: ErrorCodeInfo) {
        self.codes.insert(info.code, info);
    }

    /// Generate markdown documentation for all error codes
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::from("# TrustformeRS Training Error Code Reference\n\n");

        doc.push_str("This document provides comprehensive information about all error codes used in TrustformeRS Training.\n\n");

        // Group by component
        let mut components: HashMap<String, Vec<&ErrorCodeInfo>> = HashMap::new();
        for info in self.codes.values() {
            let component = info.code.split('_').next().unwrap_or("UNKNOWN").to_string();
            components.entry(component).or_default().push(info);
        }

        for (component, codes) in components {
            doc.push_str(&format!("## {} Errors\n\n", component));

            for code_info in codes {
                doc.push_str(&format!("### {} - {}\n\n", code_info.code, code_info.name));
                doc.push_str(&format!("**Severity**: {}\n\n", code_info.severity));
                doc.push_str(&format!("**Description**: {}\n\n", code_info.description));

                doc.push_str("**Common Causes**:\n");
                for cause in &code_info.causes {
                    doc.push_str(&format!("- {}\n", cause));
                }
                doc.push('\n');

                doc.push_str("**Solutions**:\n");
                for solution in &code_info.solutions {
                    doc.push_str(&format!("- {}\n", solution));
                }
                doc.push('\n');

                if !code_info.related_codes.is_empty() {
                    doc.push_str("**Related Error Codes**:\n");
                    for related in &code_info.related_codes {
                        doc.push_str(&format!("- {}\n", related));
                    }
                    doc.push('\n');
                }

                if let Some(url) = code_info.documentation_url {
                    doc.push_str(&format!("**Documentation**: [{}]({})\n\n", url, url));
                }

                doc.push_str("---\n\n");
            }
        }

        doc
    }

    /// Get recovery actions for a specific error code
    pub fn get_recovery_actions(&self, code: &str) -> Vec<String> {
        if let Some(info) = self.get_code_info(code) {
            info.solutions.iter().map(|s| s.to_string()).collect()
        } else {
            vec!["Unknown error code - check documentation".to_string()]
        }
    }

    /// Check if an error code is critical
    pub fn is_critical(&self, code: &str) -> bool {
        if let Some(info) = self.get_code_info(code) {
            info.severity == "CRITICAL"
        } else {
            false
        }
    }

    /// Get all error codes for a severity level
    pub fn get_codes_by_severity(&self, severity: &str) -> Vec<&str> {
        self.codes
            .values()
            .filter(|info| info.severity == severity)
            .map(|info| info.code)
            .collect()
    }
}

/// Global error code registry instance
static ERROR_CODE_REGISTRY: OnceLock<ErrorCodeRegistry> = OnceLock::new();

/// Get the global error code registry instance
fn get_registry() -> &'static ErrorCodeRegistry {
    ERROR_CODE_REGISTRY.get_or_init(ErrorCodeRegistry::new)
}

/// Helper function to get error code information
pub fn get_error_info(code: &str) -> Option<&'static ErrorCodeInfo> {
    get_registry().get_code_info(code)
}

/// Helper function to check if error is critical
pub fn is_critical_error(code: &str) -> bool {
    get_registry().is_critical(code)
}

/// Helper function to get recovery actions
pub fn get_recovery_actions(code: &str) -> Vec<String> {
    get_registry().get_recovery_actions(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_registry() {
        let registry = ErrorCodeRegistry::new();

        // Test getting code info
        let info = registry.get_code_info("TRAIN_NAN_LOSS");
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.code, "TRAIN_NAN_LOSS");
        assert_eq!(info.severity, "CRITICAL");
        assert!(!info.causes.is_empty());
        assert!(!info.solutions.is_empty());
    }

    #[test]
    fn test_severity_classification() {
        let registry = ErrorCodeRegistry::new();

        assert!(registry.is_critical("TRAIN_NAN_LOSS"));
        assert!(registry.is_critical("CONFIG_MISSING_PARAM"));
        assert!(!registry.is_critical("TRAIN_CONVERGENCE_FAILURE"));
    }

    #[test]
    fn test_recovery_actions() {
        let registry = ErrorCodeRegistry::new();

        let actions = registry.get_recovery_actions("RESOURCE_OOM");
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.contains("batch size")));
    }

    #[test]
    fn test_component_grouping() {
        let registry = ErrorCodeRegistry::new();

        let config_codes = registry.list_codes_by_component("CONFIG");
        assert!(!config_codes.is_empty());

        for code in config_codes {
            assert!(code.code.starts_with("CONFIG"));
        }
    }

    #[test]
    fn test_documentation_generation() {
        let registry = ErrorCodeRegistry::new();

        let doc = registry.generate_documentation();
        assert!(!doc.is_empty());
        assert!(doc.contains("TRAIN_NAN_LOSS"));
        assert!(doc.contains("# TrustformeRS Training Error Code Reference"));
    }

    #[test]
    fn test_global_registry_access() {
        let info = get_error_info("TRAIN_NAN_LOSS");
        assert!(info.is_some());

        assert!(is_critical_error("CONFIG_MISSING_PARAM"));
        assert!(!is_critical_error("TRAIN_CONVERGENCE_FAILURE"));

        let actions = get_recovery_actions("RESOURCE_GPU_OOM");
        assert!(!actions.is_empty());
    }
}
