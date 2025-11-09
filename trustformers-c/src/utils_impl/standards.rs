//! Naming Convention and Error Handling Standards
//!
//! This module provides standardized naming conventions and error handling patterns
//! for the TrustformeRS codebase to ensure consistency and maintainability.

use crate::error::TrustformersError;

/// Naming convention styles supported by the standards framework
#[derive(Debug, Clone, PartialEq)]
pub enum NamingStyle {
    SnakeCase,      // snake_case
    PascalCase,     // PascalCase
    CamelCase,      // camelCase
    ScreamingSnake, // SCREAMING_SNAKE_CASE
    KebabCase,      // kebab-case
}

/// Naming conventions configuration
#[derive(Debug, Clone)]
pub struct NamingConventions {
    /// C API function prefix (should be "trustformers_")
    pub c_api_prefix: String,
    /// Rust function naming style (snake_case)
    pub rust_function_style: NamingStyle,
    /// Rust struct naming style (PascalCase)
    pub rust_struct_style: NamingStyle,
    /// Rust const naming style (SCREAMING_SNAKE_CASE)
    pub rust_const_style: NamingStyle,
    /// C struct naming style (PascalCase with prefix)
    pub c_struct_style: NamingStyle,
    /// Error enum variant style (PascalCase)
    pub error_variant_style: NamingStyle,
}

impl Default for NamingConventions {
    fn default() -> Self {
        Self {
            c_api_prefix: "trustformers_".to_string(),
            rust_function_style: NamingStyle::SnakeCase,
            rust_struct_style: NamingStyle::PascalCase,
            rust_const_style: NamingStyle::ScreamingSnake,
            c_struct_style: NamingStyle::PascalCase,
            error_variant_style: NamingStyle::PascalCase,
        }
    }
}

/// Error handling standards configuration
#[derive(Debug, Clone)]
pub struct ErrorHandlingStandards {
    /// Required error types for C API functions
    pub required_null_checks: bool,
    /// Whether to use Result<T> for fallible Rust functions
    pub use_result_types: bool,
    /// Whether to log errors before returning
    pub log_errors: bool,
    /// Whether to include context in error messages
    pub include_context: bool,
    /// Maximum error message length
    pub max_error_message_length: usize,
    /// Whether to track errors globally
    pub track_errors_globally: bool,
}

impl Default for ErrorHandlingStandards {
    fn default() -> Self {
        Self {
            required_null_checks: true,
            use_result_types: true,
            log_errors: true,
            include_context: true,
            max_error_message_length: 512,
            track_errors_globally: true,
        }
    }
}

/// Code quality standards configuration
#[derive(Debug, Clone)]
pub struct CodeQualityStandards {
    /// Maximum function length in lines
    pub max_function_length: usize,
    /// Maximum file length in lines
    pub max_file_length: usize,
    /// Required documentation for public functions
    pub require_documentation: bool,
    /// Require examples in documentation
    pub require_examples: bool,
    /// Maximum cyclomatic complexity
    pub max_complexity: usize,
    /// Require tests for public functions
    pub require_tests: bool,
}

impl Default for CodeQualityStandards {
    fn default() -> Self {
        Self {
            max_function_length: 100,
            max_file_length: 2000,
            require_documentation: true,
            require_examples: false,
            max_complexity: 10,
            require_tests: true,
        }
    }
}

/// Violation categories for standards checking
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationCategory {
    Naming,
    ErrorHandling,
    CodeQuality,
    Documentation,
    Testing,
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    Error,
    Warning,
    Info,
}

/// Enforcement levels for standards
#[derive(Debug, Clone, PartialEq)]
pub enum EnforcementLevel {
    Strict,   // Fail on any violation
    Warning,  // Warn on violations but continue
    Advisory, // Log violations but take no action
    Disabled, // No enforcement
}

/// A standards violation record
#[derive(Debug, Clone)]
pub struct StandardsViolation {
    pub category: ViolationCategory,
    pub severity: ViolationSeverity,
    pub message: String,
    pub file_path: Option<String>,
    pub line_number: Option<usize>,
    pub suggestion: Option<String>,
}

/// Validation result containing violations and suggestions
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<StandardsViolation>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Combined standards configuration
#[derive(Debug, Clone)]
pub struct StandardsConfig {
    pub naming: NamingConventions,
    pub error_handling: ErrorHandlingStandards,
    pub code_quality: CodeQualityStandards,
    pub enforcement_level: EnforcementLevel,
}

impl Default for StandardsConfig {
    fn default() -> Self {
        Self {
            naming: NamingConventions::default(),
            error_handling: ErrorHandlingStandards::default(),
            code_quality: CodeQualityStandards::default(),
            enforcement_level: EnforcementLevel::Warning,
        }
    }
}

/// Get default standards configuration
pub fn get_standards() -> StandardsConfig {
    StandardsConfig::default()
}

/// Generate a simple compliance report
pub fn generate_compliance_report(violations: &[StandardsViolation]) -> String {
    let mut report = String::new();
    report.push_str("=== TrustformeRS Standards Compliance Report ===\n\n");

    let error_count = violations.iter().filter(|v| v.severity == ViolationSeverity::Error).count();
    let warning_count =
        violations.iter().filter(|v| v.severity == ViolationSeverity::Warning).count();
    let info_count = violations.iter().filter(|v| v.severity == ViolationSeverity::Info).count();

    report.push_str(&format!("Summary:\n"));
    report.push_str(&format!("  Errors: {}\n", error_count));
    report.push_str(&format!("  Warnings: {}\n", warning_count));
    report.push_str(&format!("  Info: {}\n\n", info_count));

    if !violations.is_empty() {
        report.push_str("Violations:\n");
        for (i, violation) in violations.iter().enumerate() {
            report.push_str(&format!(
                "{}. [{:?}] {:?}: {}\n",
                i + 1,
                violation.severity,
                violation.category,
                violation.message
            ));
            if let Some(suggestion) = &violation.suggestion {
                report.push_str(&format!("   Suggestion: {}\n", suggestion));
            }
            report.push('\n');
        }
    } else {
        report.push_str("✅ All standards checks passed!\n");
    }

    report
}

/// Standardized error handling macros

/// Macro for standardized null pointer checking in C API functions
#[macro_export]
macro_rules! trustformers_null_check {
    ($ptr:expr, $error_type:expr) => {
        if $ptr.is_null() {
            return $error_type;
        }
    };
    ($ptr:expr) => {
        if $ptr.is_null() {
            return crate::error::TrustformersError::NullPointer;
        }
    };
}

/// Macro for standardized parameter validation
#[macro_export]
macro_rules! trustformers_validate_param {
    ($condition:expr, $error_msg:expr, $error_type:expr) => {
        if !($condition) {
            eprintln!("Parameter validation failed: {}", $error_msg);
            return $error_type;
        }
    };
    ($condition:expr, $error_msg:expr) => {
        if !($condition) {
            eprintln!("Parameter validation failed: {}", $error_msg);
            return crate::error::TrustformersError::InvalidParameter;
        }
    };
}

/// Macro for standardized error logging and return
#[macro_export]
macro_rules! trustformers_error_return {
    ($error:expr, $context:expr) => {{
        eprintln!("TrustformeRS Error in {}: {:?}", $context, $error);
        $error
    }};
    ($error:expr) => {{
        eprintln!("TrustformeRS Error: {:?}", $error);
        $error
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standards_config_default() {
        let config = StandardsConfig::default();
        assert_eq!(config.naming.c_api_prefix, "trustformers_");
        assert_eq!(config.naming.rust_function_style, NamingStyle::SnakeCase);
        assert!(config.error_handling.required_null_checks);
        assert_eq!(config.code_quality.max_file_length, 2000);
    }

    #[test]
    fn test_violation_creation() {
        let violation = StandardsViolation {
            category: ViolationCategory::Naming,
            severity: ViolationSeverity::Warning,
            message: "Test violation".to_string(),
            file_path: Some("test.rs".to_string()),
            line_number: Some(42),
            suggestion: Some("Fix the name".to_string()),
        };

        assert_eq!(violation.category, ViolationCategory::Naming);
        assert_eq!(violation.severity, ViolationSeverity::Warning);
    }

    #[test]
    fn test_compliance_report() {
        let violations = vec![StandardsViolation {
            category: ViolationCategory::Naming,
            severity: ViolationSeverity::Error,
            message: "Function name should use snake_case".to_string(),
            file_path: None,
            line_number: None,
            suggestion: Some("Rename to snake_case".to_string()),
        }];

        let report = generate_compliance_report(&violations);
        assert!(report.contains("Errors: 1"));
        assert!(report.contains("Function name should use snake_case"));
    }

    #[test]
    fn test_empty_compliance_report() {
        let violations = vec![];
        let report = generate_compliance_report(&violations);
        assert!(report.contains("✅ All standards checks passed!"));
    }
}
