//! Data validation: rules, severities, strategies and the `DataValidator` engine.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::pipeline::DataSample;

/// Data validation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Validation strategy
    pub strategy: ValidationStrategy,
    /// Error handling
    pub error_handling: ErrorHandling,
}
pub struct DataValidator {
    pub config: DataValidationConfig,
    pub validators: Vec<Box<dyn Validator>>,
    pub stats: ValidationStats,
}
impl DataValidator {
    /// Validate one sample against the declarative rules and the registered validators.
    ///
    /// Rule semantics (all parameters come from `ValidationRule::parameters`):
    ///
    /// | rule | parameters | check |
    /// |------|------------|-------|
    /// | `Schema` | `field` | the sample must carry that tensor |
    /// | `Range` | `field`, `min`, `max` | every element must lie inside `[min, max]` |
    /// | `Format` | `field`, `shape` (comma-separated) | the tensor must have that shape |
    /// | `Quality` | `field` | the tensor must be non-empty and free of `NaN`/`inf` |
    /// | `Consistency` | `field`, `matches` | the two tensors must share a shape |
    /// | `Custom` | `validator_name` | delegated to a registered [`Validator`] of that name |
    ///
    /// `ValidationSeverity::Error` produces a [`ValidationError`] (and marks the sample
    /// invalid); `Warning` and `Info` produce a [`ValidationWarning`].
    pub fn validate_sample(&mut self, sample: &DataSample) -> Result<ValidationResult> {
        let started = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        for rule in &self.config.rules {
            if let Some(message) = Self::check_rule(rule, sample) {
                match rule.severity {
                    ValidationSeverity::Error => errors.push(ValidationError {
                        rule_name: rule.name.clone(),
                        message,
                        severity: ValidationSeverity::Error,
                    }),
                    ValidationSeverity::Warning | ValidationSeverity::Info => {
                        warnings.push(ValidationWarning {
                            rule_name: rule.name.clone(),
                            message,
                        })
                    },
                }
            }
        }
        for validator in &self.validators {
            let result = validator.validate(sample)?;
            errors.extend(result.errors);
            warnings.extend(result.warnings);
        }
        self.stats.samples_validated += 1;
        self.stats.validation_time += started.elapsed();
        for error in &errors {
            *self.stats.errors_detected.entry(error.rule_name.clone()).or_insert(0) += 1;
        }
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
    /// Evaluate one declarative rule, returning a message when it fails.
    pub(super) fn check_rule(rule: &ValidationRule, sample: &DataSample) -> Option<String> {
        let field = rule.parameters.get("field").map(String::as_str);
        match &rule.rule_type {
            ValidationRuleType::Schema => {
                let field = field?;
                if sample.data.contains_key(field) {
                    None
                } else {
                    Some(format!("required field `{field}` is missing"))
                }
            },
            ValidationRuleType::Range => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let min = rule.parameters.get("min").and_then(|v| v.parse::<f32>().ok());
                let max = rule.parameters.get("max").and_then(|v| v.parse::<f32>().ok());
                let values = tensor.data().ok()?;
                for value in values {
                    if let Some(min) = min {
                        if value < min {
                            return Some(format!("`{field}` value {value} is below minimum {min}"));
                        }
                    }
                    if let Some(max) = max {
                        if value > max {
                            return Some(format!("`{field}` value {value} exceeds maximum {max}"));
                        }
                    }
                }
                None
            },
            ValidationRuleType::Format => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let expected: Vec<usize> = rule
                    .parameters
                    .get("shape")?
                    .split(',')
                    .filter_map(|part| part.trim().parse::<usize>().ok())
                    .collect();
                if tensor.shape() == expected.as_slice() {
                    None
                } else {
                    Some(format!(
                        "`{field}` has shape {:?}, expected {:?}",
                        tensor.shape(),
                        expected
                    ))
                }
            },
            ValidationRuleType::Quality => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let values = tensor.data().ok()?;
                if values.is_empty() {
                    return Some(format!("`{field}` is empty"));
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Some(format!("`{field}` contains NaN or inf"));
                }
                None
            },
            ValidationRuleType::Consistency => {
                let field = field?;
                let other_name = rule.parameters.get("matches")?;
                let a = sample.data.get(field)?;
                let b = sample.data.get(other_name)?;
                if a.shape() == b.shape() {
                    None
                } else {
                    Some(format!(
                        "`{field}` shape {:?} is inconsistent with `{other_name}` shape {:?}",
                        a.shape(),
                        b.shape()
                    ))
                }
            },
            ValidationRuleType::Custom { validator_name } => Some(format!(
                "custom rule `{validator_name}` has no registered Validator; \
                 add one to DataValidator::validators"
            )),
        }
    }
    /// Register a [`Validator`] trait object that runs on every sample.
    pub fn add_validator(&mut self, validator: Box<dyn Validator>) {
        self.validators.push(validator);
    }
    /// Replace the declarative validation configuration.
    pub fn set_config(&mut self, config: DataValidationConfig) {
        self.config = config;
    }
}
impl DataValidator {
    pub fn new() -> Self {
        Self {
            config: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::LogAndContinue,
            },
            validators: vec![],
            stats: ValidationStats {
                samples_validated: 0,
                errors_detected: HashMap::new(),
                validation_time: Duration::from_secs(0),
            },
        }
    }
}
impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Fail on any error
    Strict,
    /// Skip invalid samples
    Skip,
    /// Attempt to fix errors
    Fix,
    /// Log and continue
    LogAndContinue,
}
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub rule_name: String,
    pub message: String,
    pub severity: ValidationSeverity,
}
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Parameters
    pub parameters: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Schema validation
    Schema,
    /// Range validation
    Range,
    /// Format validation
    Format,
    /// Consistency validation
    Consistency,
    /// Quality validation
    Quality,
    /// Custom validation
    Custom { validator_name: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub samples_validated: usize,
    pub errors_detected: HashMap<String, usize>,
    pub validation_time: Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// Validate all data
    All,
    /// Sample-based validation
    Sample { sample_rate: f64 },
    /// Batch-based validation
    Batch { batch_interval: usize },
}
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub rule_name: String,
    pub message: String,
}
pub trait Validator: Send + Sync {
    fn validate(&self, sample: &DataSample) -> Result<ValidationResult>;
}
