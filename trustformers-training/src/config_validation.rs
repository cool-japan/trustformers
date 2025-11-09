use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Configuration validation framework
pub trait Validatable {
    fn validate(&self) -> Result<(), Vec<ValidationError>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub severity: Severity,
    pub error_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "ERROR"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Info => write!(f, "INFO"),
        }
    }
}

/// Configuration validator with rules
pub struct ConfigValidator {
    rules: Vec<Box<dyn ValidationRule>>,
}

pub trait ValidationRule: Send + Sync {
    fn validate(&self, config: &dyn std::any::Any) -> Vec<ValidationError>;
    fn rule_name(&self) -> &str;
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidator {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule<R: ValidationRule + 'static>(mut self, rule: R) -> Self {
        self.rules.push(Box::new(rule));
        self
    }

    pub fn validate<T: 'static>(
        &self,
        config: &T,
    ) -> Result<ValidationReport, Vec<ValidationError>> {
        let mut all_errors = Vec::new();
        let mut warnings = Vec::new();
        let mut infos = Vec::new();

        for rule in &self.rules {
            let errors = rule.validate(config as &dyn std::any::Any);
            for error in errors {
                match error.severity {
                    Severity::Error => all_errors.push(error),
                    Severity::Warning => warnings.push(error),
                    Severity::Info => infos.push(error),
                }
            }
        }

        if !all_errors.is_empty() {
            return Err(all_errors);
        }

        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings,
            infos,
            rules_applied: self.rules.len(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationError>,
    pub infos: Vec<ValidationError>,
    pub rules_applied: usize,
}

impl ValidationReport {
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn has_infos(&self) -> bool {
        !self.infos.is_empty()
    }

    pub fn print_summary(&self) {
        println!("🔍 Validation Report");
        println!(
            "   Status: {}",
            if self.is_valid { "✅ Valid" } else { "❌ Invalid" }
        );
        println!("   Rules Applied: {}", self.rules_applied);

        if !self.errors.is_empty() {
            println!("   ❌ Errors: {}", self.errors.len());
            for error in &self.errors {
                println!("      {}", error);
            }
        }

        if !self.warnings.is_empty() {
            println!("   ⚠️  Warnings: {}", self.warnings.len());
            for warning in &self.warnings {
                println!("      {}", warning);
            }
        }

        if !self.infos.is_empty() {
            println!("   ℹ️  Infos: {}", self.infos.len());
            for info in &self.infos {
                println!("      {}", info);
            }
        }
    }
}

/// Common validation rules
pub struct RangeRule<T> {
    field_name: String,
    min: Option<T>,
    max: Option<T>,
    extractor: fn(&dyn std::any::Any) -> Option<T>,
}

impl<T> RangeRule<T>
where
    T: PartialOrd + Copy + fmt::Display + 'static,
{
    pub fn new(
        field_name: String,
        min: Option<T>,
        max: Option<T>,
        extractor: fn(&dyn std::any::Any) -> Option<T>,
    ) -> Self {
        Self {
            field_name,
            min,
            max,
            extractor,
        }
    }
}

impl<T> ValidationRule for RangeRule<T>
where
    T: PartialOrd + Copy + fmt::Display + 'static + Sync + Send,
{
    fn validate(&self, config: &dyn std::any::Any) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Some(value) = (self.extractor)(config) {
            if let Some(min) = self.min {
                if value < min {
                    errors.push(ValidationError {
                        field: self.field_name.clone(),
                        message: format!("Value {} is below minimum {}", value, min),
                        severity: Severity::Error,
                        error_code: "RANGE_BELOW_MIN".to_string(),
                    });
                }
            }

            if let Some(max) = self.max {
                if value > max {
                    errors.push(ValidationError {
                        field: self.field_name.clone(),
                        message: format!("Value {} exceeds maximum {}", value, max),
                        severity: Severity::Error,
                        error_code: "RANGE_ABOVE_MAX".to_string(),
                    });
                }
            }
        }

        errors
    }

    fn rule_name(&self) -> &str {
        "RangeRule"
    }
}

pub struct RequiredFieldRule {
    field_name: String,
    checker: fn(&dyn std::any::Any) -> bool,
}

impl RequiredFieldRule {
    pub fn new(field_name: String, checker: fn(&dyn std::any::Any) -> bool) -> Self {
        Self {
            field_name,
            checker,
        }
    }
}

impl ValidationRule for RequiredFieldRule {
    fn validate(&self, config: &dyn std::any::Any) -> Vec<ValidationError> {
        if !(self.checker)(config) {
            vec![ValidationError {
                field: self.field_name.clone(),
                message: "Required field is missing or invalid".to_string(),
                severity: Severity::Error,
                error_code: "REQUIRED_FIELD_MISSING".to_string(),
            }]
        } else {
            Vec::new()
        }
    }

    fn rule_name(&self) -> &str {
        "RequiredFieldRule"
    }
}

pub struct CompatibilityRule {
    name: String,
    checker: fn(&dyn std::any::Any) -> Vec<ValidationError>,
}

impl CompatibilityRule {
    pub fn new(name: String, checker: fn(&dyn std::any::Any) -> Vec<ValidationError>) -> Self {
        Self { name, checker }
    }
}

impl ValidationRule for CompatibilityRule {
    fn validate(&self, config: &dyn std::any::Any) -> Vec<ValidationError> {
        (self.checker)(config)
    }

    fn rule_name(&self) -> &str {
        &self.name
    }
}

/// Helper macros for common validation patterns
#[macro_export]
macro_rules! validate_range {
    ($field:ident, $min:expr, $max:expr, $type:ty) => {
        RangeRule::new(stringify!($field).to_string(), $min, $max, |config| {
            config.downcast_ref::<Self>().map(|c| c.$field as $type)
        })
    };
}

#[macro_export]
macro_rules! validate_required {
    ($field:ident) => {
        RequiredFieldRule::new(stringify!($field).to_string(), |config| {
            config.downcast_ref::<Self>().map(|c| !c.$field.is_empty()).unwrap_or(false)
        })
    };
}

/// Built-in validators for common training configurations
use crate::training_args::TrainingArguments;

impl Validatable for TrainingArguments {
    fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let validator = ConfigValidator::new()
            .add_rule(RangeRule::new(
                "learning_rate".to_string(),
                Some(1e-10_f64),
                Some(1.0_f64),
                |config| {
                    config
                        .downcast_ref::<TrainingArguments>()
                        .map(|c| c.learning_rate as f64)
                },
            ))
            .add_rule(RangeRule::new(
                "per_device_train_batch_size".to_string(),
                Some(1_usize),
                Some(1024_usize),
                |config| {
                    config
                        .downcast_ref::<TrainingArguments>()
                        .map(|c| c.per_device_train_batch_size)
                },
            ))
            .add_rule(RangeRule::new(
                "num_train_epochs".to_string(),
                Some(1_u32),
                Some(10000_u32),
                |config| {
                    config
                        .downcast_ref::<TrainingArguments>()
                        .map(|c| c.num_train_epochs as u32)
                },
            ))
            .add_rule(RequiredFieldRule::new(
                "output_dir".to_string(),
                |config| {
                    config
                        .downcast_ref::<TrainingArguments>()
                        .map(|c| !c.output_dir.to_string_lossy().is_empty())
                        .unwrap_or(false)
                },
            ))
            .add_rule(CompatibilityRule::new(
                "eval_strategy_compatibility".to_string(),
                |config| {
                    if let Some(args) = config.downcast_ref::<TrainingArguments>() {
                        let mut errors = Vec::new();

                        // Check eval strategy and eval steps compatibility
                        if args.evaluation_strategy == crate::training_args::EvaluationStrategy::Steps && args.eval_steps == 0 {
                            errors.push(ValidationError {
                                field: "eval_steps".to_string(),
                                message: "eval_steps must be greater than 0 when evaluation_strategy is Steps".to_string(),
                                severity: Severity::Warning,
                                error_code: "EVAL_STRATEGY_COMPATIBILITY".to_string(),
                            });
                        }

                        // Check gradient accumulation and batch size
                        if args.gradient_accumulation_steps > 1 && args.per_device_train_batch_size > 64 {
                            errors.push(ValidationError {
                                field: "batch_size_gradient_accumulation".to_string(),
                                message: "Large batch size with gradient accumulation may cause memory issues".to_string(),
                                severity: Severity::Warning,
                                error_code: "MEMORY_WARNING".to_string(),
                            });
                        }

                        errors
                    } else {
                        Vec::new()
                    }
                },
            ));

        match validator.validate(self) {
            Ok(report) => {
                if report.has_warnings() || report.has_infos() {
                    report.print_summary();
                }
                Ok(())
            },
            Err(errors) => Err(errors),
        }
    }
}

/// Type-safe configuration builder with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedConfig<T> {
    inner: T,
    validation_report: Option<ValidationReport>,
}

impl<T> ValidatedConfig<T>
where
    T: Validatable + Clone + 'static,
{
    pub fn new(config: T) -> Result<Self, Vec<ValidationError>> {
        match config.validate() {
            Ok(_) => Ok(Self {
                inner: config,
                validation_report: None,
            }),
            Err(errors) => Err(errors),
        }
    }

    pub fn new_with_warnings(config: T) -> Result<Self, Vec<ValidationError>> {
        let validator = ConfigValidator::new();
        match validator.validate(&config) {
            Ok(report) => Ok(Self {
                inner: config,
                validation_report: Some(report),
            }),
            Err(errors) => Err(errors),
        }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }

    pub fn get(&self) -> &T {
        &self.inner
    }

    pub fn get_validation_report(&self) -> Option<&ValidationReport> {
        self.validation_report.as_ref()
    }

    /// Update the configuration with validation
    pub fn update<F>(mut self, updater: F) -> Result<Self, Vec<ValidationError>>
    where
        F: FnOnce(&mut T),
    {
        updater(&mut self.inner);
        match self.inner.validate() {
            Ok(_) => Ok(self),
            Err(errors) => Err(errors),
        }
    }
}

/// Configuration schema for runtime validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub fields: HashMap<String, FieldSchema>,
    pub required_fields: Vec<String>,
    pub dependencies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub field_type: FieldType,
    pub constraints: Vec<Constraint>,
    pub description: String,
    pub default_value: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array { item_type: Box<FieldType> },
    Object { schema: Box<ConfigSchema> },
    Union { types: Vec<FieldType> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    Range {
        min: Option<f64>,
        max: Option<f64>,
    },
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
    Pattern {
        regex: String,
    },
    OneOf {
        values: Vec<serde_json::Value>,
    },
    Custom {
        name: String,
        description: String,
    },
}

impl Default for ConfigSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigSchema {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            required_fields: Vec::new(),
            dependencies: HashMap::new(),
        }
    }

    pub fn add_field(mut self, name: String, schema: FieldSchema) -> Self {
        self.fields.insert(name, schema);
        self
    }

    pub fn require_field(mut self, name: String) -> Self {
        self.required_fields.push(name);
        self
    }

    pub fn add_dependency(mut self, field: String, depends_on: Vec<String>) -> Self {
        self.dependencies.insert(field, depends_on);
        self
    }

    pub fn validate_json(&self, value: &serde_json::Value) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let serde_json::Value::Object(obj) = value {
            // Check required fields
            for required in &self.required_fields {
                if !obj.contains_key(required) {
                    errors.push(ValidationError {
                        field: required.clone(),
                        message: "Required field is missing".to_string(),
                        severity: Severity::Error,
                        error_code: "REQUIRED_FIELD_MISSING".to_string(),
                    });
                }
            }

            // Validate each field
            for (field_name, field_value) in obj {
                if let Some(field_schema) = self.fields.get(field_name) {
                    errors.extend(self.validate_field_value(field_name, field_value, field_schema));
                }
            }

            // Check dependencies
            for (field_name, dependencies) in &self.dependencies {
                if obj.contains_key(field_name) {
                    for dep in dependencies {
                        if !obj.contains_key(dep) {
                            errors.push(ValidationError {
                                field: field_name.clone(),
                                message: format!(
                                    "Field {} requires {} to be present",
                                    field_name, dep
                                ),
                                severity: Severity::Error,
                                error_code: "DEPENDENCY_MISSING".to_string(),
                            });
                        }
                    }
                }
            }
        } else {
            errors.push(ValidationError {
                field: "root".to_string(),
                message: "Expected object at root level".to_string(),
                severity: Severity::Error,
                error_code: "INVALID_ROOT_TYPE".to_string(),
            });
        }

        errors
    }

    fn validate_field_value(
        &self,
        field_name: &str,
        value: &serde_json::Value,
        schema: &FieldSchema,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Check type compatibility
        if !self.is_type_compatible(value, &schema.field_type) {
            errors.push(ValidationError {
                field: field_name.to_string(),
                message: format!("Type mismatch: expected {:?}", schema.field_type),
                severity: Severity::Error,
                error_code: "TYPE_MISMATCH".to_string(),
            });
            return errors; // Don't check constraints if type is wrong
        }

        // Check constraints
        for constraint in &schema.constraints {
            if let Some(error) = self.check_constraint(field_name, value, constraint) {
                errors.push(error);
            }
        }

        errors
    }

    fn is_type_compatible(&self, value: &serde_json::Value, field_type: &FieldType) -> bool {
        match (value, field_type) {
            (serde_json::Value::String(_), FieldType::String) => true,
            (serde_json::Value::Number(n), FieldType::Integer) => n.is_i64(),
            (serde_json::Value::Number(_), FieldType::Float) => true,
            (serde_json::Value::Bool(_), FieldType::Boolean) => true,
            (serde_json::Value::Array(_), FieldType::Array { .. }) => true,
            (serde_json::Value::Object(_), FieldType::Object { .. }) => true,
            (val, FieldType::Union { types }) => {
                types.iter().any(|t| self.is_type_compatible(val, t))
            },
            _ => false,
        }
    }

    fn check_constraint(
        &self,
        field_name: &str,
        value: &serde_json::Value,
        constraint: &Constraint,
    ) -> Option<ValidationError> {
        match constraint {
            Constraint::Range { min, max } => {
                if let serde_json::Value::Number(n) = value {
                    let val = n.as_f64().unwrap();
                    if let Some(min_val) = min {
                        if val < *min_val {
                            return Some(ValidationError {
                                field: field_name.to_string(),
                                message: format!("Value {} is below minimum {}", val, min_val),
                                severity: Severity::Error,
                                error_code: "RANGE_BELOW_MIN".to_string(),
                            });
                        }
                    }
                    if let Some(max_val) = max {
                        if val > *max_val {
                            return Some(ValidationError {
                                field: field_name.to_string(),
                                message: format!("Value {} exceeds maximum {}", val, max_val),
                                severity: Severity::Error,
                                error_code: "RANGE_ABOVE_MAX".to_string(),
                            });
                        }
                    }
                }
            },
            Constraint::Length { min, max } => {
                let len = match value {
                    serde_json::Value::String(s) => s.len(),
                    serde_json::Value::Array(a) => a.len(),
                    _ => return None,
                };

                if let Some(min_len) = min {
                    if len < *min_len {
                        return Some(ValidationError {
                            field: field_name.to_string(),
                            message: format!("Length {} is below minimum {}", len, min_len),
                            severity: Severity::Error,
                            error_code: "LENGTH_BELOW_MIN".to_string(),
                        });
                    }
                }
                if let Some(max_len) = max {
                    if len > *max_len {
                        return Some(ValidationError {
                            field: field_name.to_string(),
                            message: format!("Length {} exceeds maximum {}", len, max_len),
                            severity: Severity::Error,
                            error_code: "LENGTH_ABOVE_MAX".to_string(),
                        });
                    }
                }
            },
            Constraint::OneOf { values } => {
                if !values.contains(value) {
                    return Some(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Value must be one of: {:?}", values),
                        severity: Severity::Error,
                        error_code: "VALUE_NOT_IN_SET".to_string(),
                    });
                }
            },
            Constraint::Pattern { regex } => {
                if let serde_json::Value::String(s) = value {
                    // In a real implementation, would use regex crate
                    if s.is_empty() {
                        return Some(ValidationError {
                            field: field_name.to_string(),
                            message: format!("String does not match pattern: {}", regex),
                            severity: Severity::Error,
                            error_code: "PATTERN_MISMATCH".to_string(),
                        });
                    }
                }
            },
            Constraint::Custom {
                name: _,
                description: _,
            } => {
                // Custom constraints would be implemented here
                // For now, just a placeholder
            },
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestConfig {
        learning_rate: f64,
        #[allow(dead_code)]
        batch_size: usize,
        output_dir: String,
    }

    impl Validatable for TestConfig {
        fn validate(&self) -> Result<(), Vec<ValidationError>> {
            let validator = ConfigValidator::new()
                .add_rule(RangeRule::new(
                    "learning_rate".to_string(),
                    Some(1e-6_f64),
                    Some(1.0_f64),
                    |config| config.downcast_ref::<TestConfig>().map(|c| c.learning_rate),
                ))
                .add_rule(RequiredFieldRule::new("output_dir".to_string(), |config| {
                    config
                        .downcast_ref::<TestConfig>()
                        .map(|c| !c.output_dir.is_empty())
                        .unwrap_or(false)
                }));

            match validator.validate(self) {
                Ok(_) => Ok(()),
                Err(errors) => Err(errors),
            }
        }
    }

    #[test]
    fn test_valid_config() {
        let config = TestConfig {
            learning_rate: 0.001,
            batch_size: 32,
            output_dir: "/tmp/output".to_string(),
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let config = TestConfig {
            learning_rate: 2.0, // Too high
            batch_size: 32,
            output_dir: "/tmp/output".to_string(),
        };

        let result = config.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "learning_rate");
        assert_eq!(errors[0].error_code, "RANGE_ABOVE_MAX");
    }

    #[test]
    fn test_missing_output_dir() {
        let config = TestConfig {
            learning_rate: 0.001,
            batch_size: 32,
            output_dir: "".to_string(), // Empty
        };

        let result = config.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "output_dir");
        assert_eq!(errors[0].error_code, "REQUIRED_FIELD_MISSING");
    }

    #[test]
    fn test_validated_config() {
        let config = TestConfig {
            learning_rate: 0.001,
            batch_size: 32,
            output_dir: "/tmp/output".to_string(),
        };

        let validated = ValidatedConfig::new(config.clone()).unwrap();
        assert_eq!(validated.get().learning_rate, 0.001);

        let inner = validated.into_inner();
        assert_eq!(inner.learning_rate, 0.001);
    }

    #[test]
    fn test_config_schema_validation() {
        let schema = ConfigSchema::new()
            .add_field(
                "learning_rate".to_string(),
                FieldSchema {
                    field_type: FieldType::Float,
                    constraints: vec![Constraint::Range {
                        min: Some(1e-6),
                        max: Some(1.0),
                    }],
                    description: "Learning rate for training".to_string(),
                    default_value: Some(serde_json::json!(0.001)),
                },
            )
            .require_field("learning_rate".to_string());

        // Valid JSON
        let valid_json = serde_json::json!({
            "learning_rate": 0.001
        });
        let errors = schema.validate_json(&valid_json);
        assert!(errors.is_empty());

        // Invalid JSON - missing required field
        let invalid_json = serde_json::json!({});
        let errors = schema.validate_json(&invalid_json);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].error_code, "REQUIRED_FIELD_MISSING");

        // Invalid JSON - out of range
        let invalid_json = serde_json::json!({
            "learning_rate": 2.0
        });
        let errors = schema.validate_json(&invalid_json);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].error_code, "RANGE_ABOVE_MAX");
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: vec![ValidationError {
                field: "test".to_string(),
                message: "Test warning".to_string(),
                severity: Severity::Warning,
                error_code: "TEST_WARNING".to_string(),
            }],
            infos: Vec::new(),
            rules_applied: 1,
        };

        assert!(report.has_warnings());
        assert!(!report.has_infos());

        // Test display
        report.print_summary(); // Should not panic
    }
}
