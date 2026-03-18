//! Configuration Management System for TrustformeRS
//!
//! This module provides comprehensive configuration management tools including:
//! - Configuration validation and schema checking
//! - Configuration migration between versions
//! - Configuration recommendations and optimization suggestions
//! - Configuration templates and presets
//! - Configuration diff and comparison tools

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

/// Main configuration manager
pub struct ConfigurationManager {
    /// Schema registry for different configuration types
    schema_registry: HashMap<String, ConfigSchema>,
    /// Migration registry for version upgrades
    migration_registry: HashMap<String, Vec<Migration>>,
    /// Validator for configuration validation
    validator: ConfigValidator,
    /// Recommendation engine
    recommender: ConfigRecommender,
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let mut manager = Self {
            schema_registry: HashMap::new(),
            migration_registry: HashMap::new(),
            validator: ConfigValidator::new(),
            recommender: ConfigRecommender::new(),
        };

        // Register built-in schemas
        manager.register_builtin_schemas();
        manager.register_builtin_migrations();

        manager
    }

    /// Register a configuration schema
    pub fn register_schema(&mut self, config_type: String, schema: ConfigSchema) {
        self.schema_registry.insert(config_type, schema);
    }

    /// Register a migration
    pub fn register_migration(&mut self, config_type: String, migration: Migration) {
        self.migration_registry.entry(config_type).or_default().push(migration);
    }

    /// Validate a configuration against its schema
    pub fn validate_config(
        &self,
        config_type: &str,
        config: &serde_json::Value,
    ) -> ValidationResult {
        if let Some(schema) = self.schema_registry.get(config_type) {
            self.validator.validate(config, schema)
        } else {
            ValidationResult {
                is_valid: false,
                errors: vec![ValidationError {
                    field: None,
                    error_type: ValidationErrorType::UnknownConfigType,
                    message: format!("Unknown configuration type: {}", config_type),
                    severity: ValidationSeverity::Error,
                    suggestion: Some("Check available configuration types".to_string()),
                }],
                warnings: vec![],
            }
        }
    }

    /// Migrate configuration from one version to another
    pub fn migrate_config(
        &self,
        config_type: &str,
        config: &serde_json::Value,
        from_version: &str,
        to_version: &str,
    ) -> Result<serde_json::Value> {
        if let Some(migrations) = self.migration_registry.get(config_type) {
            let mut current_config = config.clone();
            let mut current_version = from_version.to_string();

            // Find migration path
            for migration in migrations {
                if migration.from_version == current_version
                    && (migration.to_version == to_version
                        || self.is_version_on_path(&migration.to_version, to_version, migrations))
                {
                    current_config = migration.apply(&current_config)?;
                    current_version = migration.to_version.clone();

                    if current_version == to_version {
                        break;
                    }
                }
            }

            if current_version == to_version {
                Ok(current_config)
            } else {
                Err(anyhow!(
                    "No migration path found from {} to {}",
                    from_version,
                    to_version
                ))
            }
        } else {
            Err(anyhow!(
                "No migrations registered for config type: {}",
                config_type
            ))
        }
    }

    /// Get configuration recommendations
    pub fn get_recommendations(
        &self,
        config_type: &str,
        config: &serde_json::Value,
        context: &RecommendationContext,
    ) -> Vec<ConfigRecommendation> {
        self.recommender.generate_recommendations(config_type, config, context)
    }

    /// Load configuration from file with validation
    pub fn load_config_file<P: AsRef<Path>>(
        &self,
        path: P,
        config_type: &str,
    ) -> Result<serde_json::Value> {
        let content = std::fs::read_to_string(path)?;
        let config: serde_json::Value = if content.trim().starts_with('{') {
            serde_json::from_str(&content)?
        } else {
            // Try YAML format
            serde_yaml::from_str(&content)?
        };

        let validation_result = self.validate_config(config_type, &config);
        if !validation_result.is_valid {
            return Err(anyhow!(
                "Configuration validation failed: {:?}",
                validation_result.errors
            ));
        }

        Ok(config)
    }

    /// Save configuration to file
    pub fn save_config_file<P: AsRef<Path>>(
        &self,
        config: &serde_json::Value,
        path: P,
        format: ConfigFormat,
    ) -> Result<()> {
        let content = match format {
            ConfigFormat::Json => serde_json::to_string_pretty(config)?,
            ConfigFormat::Yaml => serde_yaml::to_string(config)?,
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Generate configuration template
    pub fn generate_template(&self, config_type: &str) -> Option<serde_json::Value> {
        self.schema_registry.get(config_type).map(|schema| schema.generate_template())
    }

    /// Compare two configurations
    pub fn compare_configs(
        &self,
        config1: &serde_json::Value,
        config2: &serde_json::Value,
    ) -> ConfigComparison {
        ConfigDiffer::new().compare(config1, config2)
    }

    /// Get available configuration presets
    pub fn get_presets(&self, config_type: &str) -> Vec<ConfigPreset> {
        self.recommender.get_presets(config_type)
    }

    /// Create configuration from preset
    pub fn create_from_preset(
        &self,
        config_type: &str,
        preset_name: &str,
        overrides: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<serde_json::Value> {
        let presets = self.get_presets(config_type);
        if let Some(preset) = presets.iter().find(|p| p.name == preset_name) {
            let mut config = preset.config.clone();

            if let Some(overrides) = overrides {
                for (key, value) in overrides {
                    self.set_nested_value(&mut config, &key, value)?;
                }
            }

            Ok(config)
        } else {
            Err(anyhow!("Preset not found: {}", preset_name))
        }
    }

    // Private helper methods
    fn register_builtin_schemas(&mut self) {
        // Register common configuration schemas
        self.register_schema("training".to_string(), create_training_schema());
        self.register_schema("model".to_string(), create_model_schema());
        self.register_schema("pipeline".to_string(), create_pipeline_schema());
        self.register_schema("conversational".to_string(), create_conversational_schema());
        self.register_schema("hub".to_string(), create_hub_schema());
    }

    fn register_builtin_migrations(&mut self) {
        // Register common migrations
        self.register_migration("training".to_string(), create_training_migration_v1_to_v2());
        self.register_migration("model".to_string(), create_model_migration_v1_to_v2());
    }

    fn is_version_on_path(&self, version: &str, target: &str, migrations: &[Migration]) -> bool {
        // Simplified version path checking - in real implementation would use proper version comparison
        migrations.iter().any(|m| m.from_version == version && m.to_version == target)
    }

    fn set_nested_value(
        &self,
        config: &mut serde_json::Value,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let parts: Vec<&str> = key.split('.').collect();
        let mut current = config;

        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Last part - set the value
                if let serde_json::Value::Object(map) = current {
                    map.insert(part.to_string(), value);
                }
                break; // Exit the loop after setting the value
            } else {
                // Navigate to nested object
                if let serde_json::Value::Object(map) = current {
                    current = map
                        .entry(part.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                }
            }
        }

        Ok(())
    }
}

/// Configuration schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub name: String,
    pub version: String,
    pub description: String,
    pub fields: HashMap<String, FieldSchema>,
    pub required_fields: HashSet<String>,
    pub conditional_requirements: Vec<ConditionalRequirement>,
}

impl ConfigSchema {
    /// Generate a template configuration based on this schema
    pub fn generate_template(&self) -> serde_json::Value {
        let mut template = serde_json::Map::new();

        for (field_name, field_schema) in &self.fields {
            if self.required_fields.contains(field_name) {
                template.insert(field_name.clone(), field_schema.get_default_value());
            }
        }

        serde_json::Value::Object(template)
    }
}

/// Field schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub field_type: FieldType,
    pub description: String,
    pub default_value: Option<serde_json::Value>,
    pub constraints: Vec<FieldConstraint>,
    pub examples: Vec<serde_json::Value>,
}

impl FieldSchema {
    pub fn get_default_value(&self) -> serde_json::Value {
        self.default_value.clone().unwrap_or_else(|| match &self.field_type {
            FieldType::String => serde_json::Value::String("".to_string()),
            FieldType::Number => serde_json::Value::Number(serde_json::Number::from(0)),
            FieldType::Boolean => serde_json::Value::Bool(false),
            FieldType::Array => serde_json::Value::Array(vec![]),
            FieldType::Object => serde_json::Value::Object(serde_json::Map::new()),
            FieldType::Enum { options } => {
                if let Some(first_option) = options.first() {
                    serde_json::Value::String(first_option.clone())
                } else {
                    serde_json::Value::Null
                }
            },
        })
    }
}

/// Field type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Enum { options: Vec<String> },
}

/// Field constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConstraint {
    MinLength(usize),
    MaxLength(usize),
    MinValue(f64),
    MaxValue(f64),
    Pattern(String),
    OneOf(Vec<serde_json::Value>),
    Custom { name: String, description: String },
}

/// Conditional requirement definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalRequirement {
    pub condition: String, // Field path and condition
    pub required_fields: Vec<String>,
}

/// Configuration migration definition
pub struct Migration {
    pub from_version: String,
    pub to_version: String,
    pub description: String,
    pub migration_fn: Box<dyn Fn(&serde_json::Value) -> Result<serde_json::Value> + Send + Sync>,
}

impl Migration {
    pub fn new<F>(from: &str, to: &str, description: &str, migration_fn: F) -> Self
    where
        F: Fn(&serde_json::Value) -> Result<serde_json::Value> + Send + Sync + 'static,
    {
        Self {
            from_version: from.to_string(),
            to_version: to.to_string(),
            description: description.to_string(),
            migration_fn: Box::new(migration_fn),
        }
    }

    pub fn apply(&self, config: &serde_json::Value) -> Result<serde_json::Value> {
        (self.migration_fn)(config)
    }
}

/// Configuration validator
pub struct ConfigValidator {
    custom_validators: HashMap<String, Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>>,
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidator {
    pub fn new() -> Self {
        Self {
            custom_validators: HashMap::new(),
        }
    }

    pub fn register_custom_validator<F>(&mut self, name: String, validator: F)
    where
        F: Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    {
        self.custom_validators.insert(name, Box::new(validator));
    }

    pub fn validate(&self, config: &serde_json::Value, schema: &ConfigSchema) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
        };

        // Check required fields
        if let serde_json::Value::Object(config_map) = config {
            for required_field in &schema.required_fields {
                if !config_map.contains_key(required_field) {
                    result.errors.push(ValidationError {
                        field: Some(required_field.clone()),
                        error_type: ValidationErrorType::MissingRequiredField,
                        message: format!("Required field '{}' is missing", required_field),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(
                            "Add the required field to your configuration".to_string(),
                        ),
                    });
                    result.is_valid = false;
                }
            }

            // Validate field types and constraints
            for (field_name, field_value) in config_map {
                if let Some(field_schema) = schema.fields.get(field_name) {
                    self.validate_field(field_name, field_value, field_schema, &mut result);
                } else {
                    result.warnings.push(ValidationWarning {
                        field: Some(field_name.clone()),
                        message: format!(
                            "Unknown field '{}' - this field is not defined in the schema",
                            field_name
                        ),
                        suggestion: Some(
                            "Remove this field or check if it's spelled correctly".to_string(),
                        ),
                    });
                }
            }

            // Check conditional requirements
            for conditional in &schema.conditional_requirements {
                if self.evaluate_condition(&conditional.condition, config_map) {
                    for required_field in &conditional.required_fields {
                        if !config_map.contains_key(required_field) {
                            result.errors.push(ValidationError {
                                field: Some(required_field.clone()),
                                error_type: ValidationErrorType::ConditionalRequirementNotMet,
                                message: format!(
                                    "Field '{}' is required when {}",
                                    required_field, conditional.condition
                                ),
                                severity: ValidationSeverity::Error,
                                suggestion: Some(
                                    "Add the conditionally required field".to_string(),
                                ),
                            });
                            result.is_valid = false;
                        }
                    }
                }
            }
        } else {
            result.errors.push(ValidationError {
                field: None,
                error_type: ValidationErrorType::InvalidFormat,
                message: "Configuration must be a JSON object".to_string(),
                severity: ValidationSeverity::Error,
                suggestion: Some("Ensure your configuration is a valid JSON object".to_string()),
            });
            result.is_valid = false;
        }

        result
    }

    fn validate_field(
        &self,
        field_name: &str,
        field_value: &serde_json::Value,
        field_schema: &FieldSchema,
        result: &mut ValidationResult,
    ) {
        // Type validation
        if !self.is_type_compatible(field_value, &field_schema.field_type) {
            result.errors.push(ValidationError {
                field: Some(field_name.to_string()),
                error_type: ValidationErrorType::TypeMismatch,
                message: format!("Field '{}' has incorrect type", field_name),
                severity: ValidationSeverity::Error,
                suggestion: Some(format!("Expected type: {:?}", field_schema.field_type)),
            });
            result.is_valid = false;
            return;
        }

        // Constraint validation
        for constraint in &field_schema.constraints {
            if !self.validate_constraint(field_value, constraint) {
                result.errors.push(ValidationError {
                    field: Some(field_name.to_string()),
                    error_type: ValidationErrorType::ConstraintViolation,
                    message: format!(
                        "Field '{}' violates constraint: {:?}",
                        field_name, constraint
                    ),
                    severity: ValidationSeverity::Error,
                    suggestion: Some("Adjust the field value to meet the constraint".to_string()),
                });
                result.is_valid = false;
            }
        }
    }

    fn is_type_compatible(&self, value: &serde_json::Value, field_type: &FieldType) -> bool {
        match field_type {
            FieldType::String => value.is_string(),
            FieldType::Number => value.is_number(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Array => value.is_array(),
            FieldType::Object => value.is_object(),
            FieldType::Enum { options } => {
                if let Some(string_value) = value.as_str() {
                    options.contains(&string_value.to_string())
                } else {
                    false
                }
            },
        }
    }

    fn validate_constraint(&self, value: &serde_json::Value, constraint: &FieldConstraint) -> bool {
        match constraint {
            FieldConstraint::MinLength(min_len) => {
                if let Some(s) = value.as_str() {
                    s.len() >= *min_len
                } else {
                    true // Not applicable for non-strings
                }
            },
            FieldConstraint::MaxLength(max_len) => {
                if let Some(s) = value.as_str() {
                    s.len() <= *max_len
                } else {
                    true
                }
            },
            FieldConstraint::MinValue(min_val) => {
                if let Some(n) = value.as_f64() {
                    n >= *min_val
                } else {
                    true
                }
            },
            FieldConstraint::MaxValue(max_val) => {
                if let Some(n) = value.as_f64() {
                    n <= *max_val
                } else {
                    true
                }
            },
            FieldConstraint::Pattern(pattern) => {
                if let Some(s) = value.as_str() {
                    if let Ok(regex) = regex::Regex::new(pattern) {
                        regex.is_match(s)
                    } else {
                        false
                    }
                } else {
                    true
                }
            },
            FieldConstraint::OneOf(valid_values) => valid_values.contains(value),
            FieldConstraint::Custom { name, .. } => {
                if let Some(validator) = self.custom_validators.get(name) {
                    validator(value)
                } else {
                    true // Unknown custom validator - assume valid
                }
            },
        }
    }

    fn evaluate_condition(
        &self,
        condition: &str,
        config: &serde_json::Map<String, serde_json::Value>,
    ) -> bool {
        // Simplified condition evaluation - real implementation would have a proper parser
        if condition.contains("==") {
            let parts: Vec<&str> = condition.split("==").collect();
            if parts.len() == 2 {
                let field = parts[0].trim();
                let expected_value = parts[1].trim().trim_matches('"');

                if let Some(actual_value) = config.get(field) {
                    if let Some(actual_str) = actual_value.as_str() {
                        return actual_str == expected_value;
                    }
                }
            }
        }

        false
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: Option<String>,
    pub error_type: ValidationErrorType,
    pub message: String,
    pub severity: ValidationSeverity,
    pub suggestion: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub field: Option<String>,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    MissingRequiredField,
    TypeMismatch,
    ConstraintViolation,
    ConditionalRequirementNotMet,
    InvalidFormat,
    UnknownConfigType,
}

/// Validation severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

/// Configuration recommendation engine
pub struct ConfigRecommender {
    presets: HashMap<String, Vec<ConfigPreset>>,
    optimization_rules: Vec<OptimizationRule>,
}

impl Default for ConfigRecommender {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigRecommender {
    pub fn new() -> Self {
        let mut recommender = Self {
            presets: HashMap::new(),
            optimization_rules: vec![],
        };

        recommender.register_default_presets();
        recommender.register_optimization_rules();

        recommender
    }

    pub fn generate_recommendations(
        &self,
        config_type: &str,
        config: &serde_json::Value,
        context: &RecommendationContext,
    ) -> Vec<ConfigRecommendation> {
        let mut recommendations = vec![];

        // Apply optimization rules
        for rule in &self.optimization_rules {
            if rule.applies_to_config_type(config_type) {
                if let Some(recommendation) = rule.evaluate(config, context) {
                    recommendations.push(recommendation);
                }
            }
        }

        recommendations
    }

    pub fn get_presets(&self, config_type: &str) -> Vec<ConfigPreset> {
        self.presets.get(config_type).cloned().unwrap_or_default()
    }

    fn register_default_presets(&mut self) {
        // Register common presets
        self.presets.insert(
            "training".to_string(),
            vec![
                ConfigPreset {
                    name: "fast_development".to_string(),
                    description: "Fast training for development and testing".to_string(),
                    config: serde_json::json!({
                        "num_epochs": 3,
                        "batch_size": 8,
                        "learning_rate": 5e-5,
                        "warmup_steps": 100
                    }),
                },
                ConfigPreset {
                    name: "production_training".to_string(),
                    description: "Production training with optimal settings".to_string(),
                    config: serde_json::json!({
                        "num_epochs": 10,
                        "batch_size": 32,
                        "learning_rate": 2e-5,
                        "warmup_steps": 1000,
                        "gradient_accumulation_steps": 4
                    }),
                },
            ],
        );

        self.presets.insert(
            "conversational".to_string(),
            vec![
                ConfigPreset {
                    name: "chatbot".to_string(),
                    description: "Configuration for casual chatbot conversations".to_string(),
                    config: serde_json::json!({
                        "max_history_turns": 20,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "conversation_mode": "Chat"
                    }),
                },
                ConfigPreset {
                    name: "assistant".to_string(),
                    description: "Configuration for task-oriented assistant".to_string(),
                    config: serde_json::json!({
                        "max_history_turns": 10,
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "conversation_mode": "Assistant",
                        "enable_safety_filter": true
                    }),
                },
            ],
        );
    }

    fn register_optimization_rules(&mut self) {
        self.optimization_rules.push(OptimizationRule {
            name: "batch_size_optimization".to_string(),
            config_types: vec!["training".to_string()],
            condition: Box::new(|config, context| {
                if let (Some(batch_size), Some(gpu_memory)) = (
                    config.get("batch_size").and_then(|v| v.as_u64()),
                    context.hardware_info.get("gpu_memory_gb").and_then(|v| v.as_f64())
                ) {
                    // Recommend larger batch size if GPU has plenty of memory
                    batch_size < 16 && gpu_memory > 8.0
                } else {
                    false
                }
            }),
            recommendation: Box::new(|_config, context| {
                let gpu_memory = context.hardware_info.get("gpu_memory_gb")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0);

                let recommended_batch_size = if gpu_memory > 16.0 {
                    64
                } else if gpu_memory > 8.0 {
                    32
                } else {
                    16
                };

                ConfigRecommendation {
                    field: "batch_size".to_string(),
                    current_value: None,
                    recommended_value: serde_json::Value::Number(serde_json::Number::from(recommended_batch_size)),
                    reason: format!("You have {:.1} GB of GPU memory. Consider increasing batch size to {} for better GPU utilization.", gpu_memory, recommended_batch_size),
                    impact: RecommendationImpact::Performance,
                    confidence: 0.8,
                }
            }),
        });

        self.optimization_rules.push(OptimizationRule {
            name: "learning_rate_optimization".to_string(),
            config_types: vec!["training".to_string()],
            condition: Box::new(|config, _context| {
                if let Some(lr) = config.get("learning_rate").and_then(|v| v.as_f64()) {
                    lr > 1e-3 // Learning rate too high
                } else {
                    false
                }
            }),
            recommendation: Box::new(|config, _context| {
                let current_lr = config.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.0);

                ConfigRecommendation {
                    field: "learning_rate".to_string(),
                    current_value: serde_json::Number::from_f64(current_lr)
                        .map(serde_json::Value::Number),
                    recommended_value: serde_json::Value::Number(
                        serde_json::Number::from_f64(2e-5)
                            .expect("2e-5 is a valid float constant"),
                    ),
                    reason: format!("Learning rate of {} is very high and may cause training instability. Consider using 2e-5 for transformer models.", current_lr),
                    impact: RecommendationImpact::Stability,
                    confidence: 0.9,
                }
            }),
        });
    }
}

/// Configuration preset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPreset {
    pub name: String,
    pub description: String,
    pub config: serde_json::Value,
}

/// Optimization rule
pub struct OptimizationRule {
    pub name: String,
    pub config_types: Vec<String>,
    pub condition: Box<dyn Fn(&serde_json::Value, &RecommendationContext) -> bool + Send + Sync>,
    pub recommendation: Box<
        dyn Fn(&serde_json::Value, &RecommendationContext) -> ConfigRecommendation + Send + Sync,
    >,
}

impl OptimizationRule {
    pub fn applies_to_config_type(&self, config_type: &str) -> bool {
        self.config_types.iter().any(|ct| ct == config_type)
    }

    pub fn evaluate(
        &self,
        config: &serde_json::Value,
        context: &RecommendationContext,
    ) -> Option<ConfigRecommendation> {
        if (self.condition)(config, context) {
            Some((self.recommendation)(config, context))
        } else {
            None
        }
    }
}

/// Recommendation context
#[derive(Debug, Clone)]
pub struct RecommendationContext {
    pub hardware_info: HashMap<String, serde_json::Value>,
    pub use_case: String,
    pub performance_requirements: PerformanceRequirements,
    pub constraints: Vec<String>,
}

/// Performance requirements
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub max_latency_ms: Option<f64>,
    pub min_throughput: Option<f64>,
    pub memory_budget_gb: Option<f64>,
    pub power_budget_watts: Option<f64>,
}

/// Configuration recommendation
#[derive(Debug, Clone)]
pub struct ConfigRecommendation {
    pub field: String,
    pub current_value: Option<serde_json::Value>,
    pub recommended_value: serde_json::Value,
    pub reason: String,
    pub impact: RecommendationImpact,
    pub confidence: f64,
}

/// Recommendation impact
#[derive(Debug, Clone)]
pub enum RecommendationImpact {
    Performance,
    Memory,
    Accuracy,
    Stability,
    Security,
    Usability,
}

/// Configuration format
#[derive(Debug, Clone)]
pub enum ConfigFormat {
    Json,
    Yaml,
}

/// Configuration comparison utilities
pub struct ConfigDiffer;

impl Default for ConfigDiffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigDiffer {
    pub fn new() -> Self {
        Self
    }

    pub fn compare(
        &self,
        config1: &serde_json::Value,
        config2: &serde_json::Value,
    ) -> ConfigComparison {
        let mut comparison = ConfigComparison {
            added_fields: vec![],
            removed_fields: vec![],
            modified_fields: vec![],
            identical_fields: vec![],
        };

        self.compare_values(config1, config2, "", &mut comparison);

        comparison
    }

    fn compare_values(
        &self,
        val1: &serde_json::Value,
        val2: &serde_json::Value,
        path: &str,
        comparison: &mut ConfigComparison,
    ) {
        match (val1, val2) {
            (serde_json::Value::Object(obj1), serde_json::Value::Object(obj2)) => {
                // Find added and modified fields
                for (key, value2) in obj2 {
                    let current_path =
                        if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };

                    if let Some(value1) = obj1.get(key) {
                        if value1 != value2 {
                            if value1.is_object() && value2.is_object() {
                                self.compare_values(value1, value2, &current_path, comparison);
                            } else {
                                comparison.modified_fields.push(FieldDiff {
                                    field: current_path,
                                    old_value: Some(value1.clone()),
                                    new_value: Some(value2.clone()),
                                });
                            }
                        } else {
                            comparison.identical_fields.push(current_path);
                        }
                    } else {
                        comparison.added_fields.push(FieldDiff {
                            field: current_path,
                            old_value: None,
                            new_value: Some(value2.clone()),
                        });
                    }
                }

                // Find removed fields
                for (key, value1) in obj1 {
                    if !obj2.contains_key(key) {
                        let current_path =
                            if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };

                        comparison.removed_fields.push(FieldDiff {
                            field: current_path,
                            old_value: Some(value1.clone()),
                            new_value: None,
                        });
                    }
                }
            },
            _ => {
                if val1 != val2 {
                    comparison.modified_fields.push(FieldDiff {
                        field: path.to_string(),
                        old_value: Some(val1.clone()),
                        new_value: Some(val2.clone()),
                    });
                } else {
                    comparison.identical_fields.push(path.to_string());
                }
            },
        }
    }
}

/// Configuration comparison result
#[derive(Debug, Clone)]
pub struct ConfigComparison {
    pub added_fields: Vec<FieldDiff>,
    pub removed_fields: Vec<FieldDiff>,
    pub modified_fields: Vec<FieldDiff>,
    pub identical_fields: Vec<String>,
}

/// Field difference
#[derive(Debug, Clone)]
pub struct FieldDiff {
    pub field: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
}

impl fmt::Display for ConfigComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Configuration Comparison:")?;

        if !self.added_fields.is_empty() {
            writeln!(f, "\nAdded Fields:")?;
            for field in &self.added_fields {
                writeln!(f, "  + {}: {:?}", field.field, field.new_value)?;
            }
        }

        if !self.removed_fields.is_empty() {
            writeln!(f, "\nRemoved Fields:")?;
            for field in &self.removed_fields {
                writeln!(f, "  - {}: {:?}", field.field, field.old_value)?;
            }
        }

        if !self.modified_fields.is_empty() {
            writeln!(f, "\nModified Fields:")?;
            for field in &self.modified_fields {
                writeln!(
                    f,
                    "  ~ {}: {:?} -> {:?}",
                    field.field, field.old_value, field.new_value
                )?;
            }
        }

        writeln!(f, "\nIdentical Fields: {}", self.identical_fields.len())?;

        Ok(())
    }
}

// Schema creation functions for built-in configuration types

fn create_training_schema() -> ConfigSchema {
    let mut fields = HashMap::new();

    fields.insert(
        "num_epochs".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Number of training epochs".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(3))),
            constraints: vec![
                FieldConstraint::MinValue(1.0),
                FieldConstraint::MaxValue(1000.0),
            ],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(5))],
        },
    );

    fields.insert(
        "batch_size".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Training batch size".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(16))),
            constraints: vec![
                FieldConstraint::MinValue(1.0),
                FieldConstraint::MaxValue(1024.0),
            ],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(32))],
        },
    );

    fields.insert(
        "learning_rate".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Learning rate for optimization".to_string(),
            default_value: Some(serde_json::Value::Number(
                serde_json::Number::from_f64(2e-5).expect("2e-5 is a valid float constant"),
            )),
            constraints: vec![
                FieldConstraint::MinValue(1e-8),
                FieldConstraint::MaxValue(1.0),
            ],
            examples: vec![serde_json::Value::Number(
                serde_json::Number::from_f64(5e-5).expect("5e-5 is a valid float constant"),
            )],
        },
    );

    ConfigSchema {
        name: "Training Configuration".to_string(),
        version: "1.0.0".to_string(),
        description: "Configuration for model training".to_string(),
        fields,
        required_fields: ["num_epochs", "batch_size", "learning_rate"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        conditional_requirements: vec![],
    }
}

fn create_model_schema() -> ConfigSchema {
    let mut fields = HashMap::new();

    fields.insert(
        "model_name".to_string(),
        FieldSchema {
            field_type: FieldType::String,
            description: "Name of the pre-trained model".to_string(),
            default_value: Some(serde_json::Value::String("bert-base-uncased".to_string())),
            constraints: vec![FieldConstraint::MinLength(1)],
            examples: vec![serde_json::Value::String("roberta-base".to_string())],
        },
    );

    fields.insert(
        "model_type".to_string(),
        FieldSchema {
            field_type: FieldType::Enum {
                options: vec![
                    "bert".to_string(),
                    "gpt2".to_string(),
                    "t5".to_string(),
                    "roberta".to_string(),
                ],
            },
            description: "Type of the model architecture".to_string(),
            default_value: Some(serde_json::Value::String("bert".to_string())),
            constraints: vec![],
            examples: vec![serde_json::Value::String("bert".to_string())],
        },
    );

    fields.insert(
        "hidden_size".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Hidden size of the model".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(768))),
            constraints: vec![FieldConstraint::MinValue(1.0)],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(768))],
        },
    );

    fields.insert(
        "num_attention_heads".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Number of attention heads".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(12))),
            constraints: vec![FieldConstraint::MinValue(1.0)],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(12))],
        },
    );

    fields.insert(
        "num_hidden_layers".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Number of hidden layers".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(12))),
            constraints: vec![FieldConstraint::MinValue(1.0)],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(12))],
        },
    );

    ConfigSchema {
        name: "Model Configuration".to_string(),
        version: "1.0.0".to_string(),
        description: "Configuration for model selection and setup".to_string(),
        fields,
        required_fields: ["model_type"].iter().map(|s| s.to_string()).collect(),
        conditional_requirements: vec![],
    }
}

fn create_pipeline_schema() -> ConfigSchema {
    let mut fields = HashMap::new();

    fields.insert(
        "task".to_string(),
        FieldSchema {
            field_type: FieldType::Enum {
                options: vec![
                    "text-classification".to_string(),
                    "text-generation".to_string(),
                    "question-answering".to_string(),
                    "conversational".to_string(),
                ],
            },
            description: "Pipeline task type".to_string(),
            default_value: Some(serde_json::Value::String("text-classification".to_string())),
            constraints: vec![],
            examples: vec![serde_json::Value::String("text-generation".to_string())],
        },
    );

    fields.insert(
        "max_length".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Maximum sequence length".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(512))),
            constraints: vec![
                FieldConstraint::MinValue(1.0),
                FieldConstraint::MaxValue(8192.0),
            ],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(1024))],
        },
    );

    ConfigSchema {
        name: "Pipeline Configuration".to_string(),
        version: "1.0.0".to_string(),
        description: "Configuration for pipeline setup".to_string(),
        fields,
        required_fields: ["task"].iter().map(|s| s.to_string()).collect(),
        conditional_requirements: vec![],
    }
}

fn create_conversational_schema() -> ConfigSchema {
    let mut fields = HashMap::new();

    fields.insert(
        "max_history_turns".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Maximum number of conversation turns to keep in history".to_string(),
            default_value: Some(serde_json::Value::Number(serde_json::Number::from(20))),
            constraints: vec![
                FieldConstraint::MinValue(1.0),
                FieldConstraint::MaxValue(100.0),
            ],
            examples: vec![serde_json::Value::Number(serde_json::Number::from(15))],
        },
    );

    fields.insert(
        "temperature".to_string(),
        FieldSchema {
            field_type: FieldType::Number,
            description: "Temperature for response generation".to_string(),
            default_value: Some(serde_json::Value::Number(
                serde_json::Number::from_f64(0.7).expect("0.7 is a valid float constant"),
            )),
            constraints: vec![
                FieldConstraint::MinValue(0.0),
                FieldConstraint::MaxValue(2.0),
            ],
            examples: vec![serde_json::Value::Number(
                serde_json::Number::from_f64(0.8).expect("0.8 is a valid float constant"),
            )],
        },
    );

    fields.insert(
        "conversation_mode".to_string(),
        FieldSchema {
            field_type: FieldType::Enum {
                options: vec![
                    "Chat".to_string(),
                    "Assistant".to_string(),
                    "InstructionFollowing".to_string(),
                    "QuestionAnswering".to_string(),
                    "RolePlay".to_string(),
                    "Educational".to_string(),
                ],
            },
            description: "Conversation mode".to_string(),
            default_value: Some(serde_json::Value::String("Chat".to_string())),
            constraints: vec![],
            examples: vec![serde_json::Value::String("Assistant".to_string())],
        },
    );

    ConfigSchema {
        name: "Conversational Configuration".to_string(),
        version: "1.0.0".to_string(),
        description: "Configuration for conversational pipelines".to_string(),
        fields,
        required_fields: ["conversation_mode"].iter().map(|s| s.to_string()).collect(),
        conditional_requirements: vec![],
    }
}

fn create_hub_schema() -> ConfigSchema {
    let mut fields = HashMap::new();

    fields.insert(
        "cache_dir".to_string(),
        FieldSchema {
            field_type: FieldType::String,
            description: "Directory for caching downloaded models".to_string(),
            default_value: Some(serde_json::Value::String(
                "~/.cache/trustformers".to_string(),
            )),
            constraints: vec![FieldConstraint::MinLength(1)],
            examples: vec![serde_json::Value::String("/tmp/models".to_string())],
        },
    );

    fields.insert(
        "use_auth_token".to_string(),
        FieldSchema {
            field_type: FieldType::Boolean,
            description: "Whether to use authentication token for private models".to_string(),
            default_value: Some(serde_json::Value::Bool(false)),
            constraints: vec![],
            examples: vec![serde_json::Value::Bool(true)],
        },
    );

    ConfigSchema {
        name: "Hub Configuration".to_string(),
        version: "1.0.0".to_string(),
        description: "Configuration for Hugging Face Hub integration".to_string(),
        fields,
        required_fields: std::collections::HashSet::<String>::new(),
        conditional_requirements: vec![],
    }
}

// Migration functions for version upgrades

fn create_training_migration_v1_to_v2() -> Migration {
    Migration::new(
        "1.0.0",
        "2.0.0",
        "Add gradient accumulation steps and warmup steps",
        |config| {
            let mut new_config = config.clone();

            if let serde_json::Value::Object(map) = &mut new_config {
                // Add new fields with default values
                if !map.contains_key("gradient_accumulation_steps") {
                    map.insert(
                        "gradient_accumulation_steps".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(1)),
                    );
                }

                if !map.contains_key("warmup_steps") {
                    map.insert(
                        "warmup_steps".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(500)),
                    );
                }
            }

            Ok(new_config)
        },
    )
}

fn create_model_migration_v1_to_v2() -> Migration {
    Migration::new(
        "1.0.0",
        "2.0.0",
        "Rename model_name to model_id and add revision field",
        |config| {
            let mut new_config = config.clone();

            if let serde_json::Value::Object(map) = &mut new_config {
                // Rename model_name to model_id
                if let Some(model_name) = map.remove("model_name") {
                    map.insert("model_id".to_string(), model_name);
                }

                // Add revision field
                if !map.contains_key("revision") {
                    map.insert(
                        "revision".to_string(),
                        serde_json::Value::String("main".to_string()),
                    );
                }
            }

            Ok(new_config)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_manager_creation() {
        let manager = ConfigurationManager::new();
        assert!(manager.schema_registry.contains_key("training"));
        assert!(manager.schema_registry.contains_key("model"));
        assert!(manager.schema_registry.contains_key("conversational"));
    }

    #[test]
    fn test_validation_success() {
        let manager = ConfigurationManager::new();
        let config = serde_json::json!({
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5
        });

        let result = manager.validate_config("training", &config);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_missing_required_field() {
        let manager = ConfigurationManager::new();
        let config = serde_json::json!({
            "num_epochs": 5,
            "batch_size": 32
            // missing learning_rate
        });

        let result = manager.validate_config("training", &config);
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert!(matches!(
            result.errors[0].error_type,
            ValidationErrorType::MissingRequiredField
        ));
    }

    #[test]
    fn test_validation_type_mismatch() {
        let manager = ConfigurationManager::new();
        let config = serde_json::json!({
            "num_epochs": "not_a_number",
            "batch_size": 32,
            "learning_rate": 2e-5
        });

        let result = manager.validate_config("training", &config);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::TypeMismatch)));
    }

    #[test]
    fn test_migration() {
        let manager = ConfigurationManager::new();
        let old_config = serde_json::json!({
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5
        });

        let migrated = manager.migrate_config("training", &old_config, "1.0.0", "2.0.0").unwrap();

        assert!(migrated.get("gradient_accumulation_steps").is_some());
        assert!(migrated.get("warmup_steps").is_some());
    }

    #[test]
    fn test_template_generation() {
        let manager = ConfigurationManager::new();
        let template = manager.generate_template("training").unwrap();

        assert!(template.get("num_epochs").is_some());
        assert!(template.get("batch_size").is_some());
        assert!(template.get("learning_rate").is_some());
    }

    #[test]
    fn test_config_comparison() {
        let manager = ConfigurationManager::new();

        let config1 = serde_json::json!({
            "num_epochs": 5,
            "batch_size": 32
        });

        let config2 = serde_json::json!({
            "num_epochs": 10,
            "learning_rate": 2e-5
        });

        let comparison = manager.compare_configs(&config1, &config2);

        assert_eq!(comparison.modified_fields.len(), 1); // num_epochs changed
        assert_eq!(comparison.added_fields.len(), 1); // learning_rate added
        assert_eq!(comparison.removed_fields.len(), 1); // batch_size removed
    }

    #[test]
    fn test_preset_creation() {
        let manager = ConfigurationManager::new();

        let config = manager
            .create_from_preset(
                "training",
                "fast_development",
                Some(HashMap::from([(
                    "batch_size".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(16)),
                )])),
            )
            .unwrap();

        assert_eq!(
            config.get("num_epochs").unwrap(),
            &serde_json::Value::Number(serde_json::Number::from(3))
        );
        assert_eq!(
            config.get("batch_size").unwrap(),
            &serde_json::Value::Number(serde_json::Number::from(16))
        ); // overridden
    }

    #[test]
    fn test_recommendations() {
        let manager = ConfigurationManager::new();

        let config = serde_json::json!({
            "batch_size": 8,
            "learning_rate": 1e-2 // Very high learning rate
        });

        let context = RecommendationContext {
            hardware_info: HashMap::from([(
                "gpu_memory_gb".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(16.0).unwrap()),
            )]),
            use_case: "production".to_string(),
            performance_requirements: PerformanceRequirements {
                max_latency_ms: None,
                min_throughput: None,
                memory_budget_gb: None,
                power_budget_watts: None,
            },
            constraints: vec![],
        };

        let recommendations = manager.get_recommendations("training", &config, &context);

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.field == "batch_size"));
        assert!(recommendations.iter().any(|r| r.field == "learning_rate"));
    }

    #[test]
    fn test_unknown_config_type() {
        let manager = ConfigurationManager::new();
        let config = serde_json::json!({"test": "value"});

        let result = manager.validate_config("unknown_type", &config);
        assert!(!result.is_valid);
        assert!(matches!(
            result.errors[0].error_type,
            ValidationErrorType::UnknownConfigType
        ));
    }

    #[test]
    fn test_constraint_validation() {
        let manager = ConfigurationManager::new();
        let config = serde_json::json!({
            "num_epochs": -1, // Violates minimum value constraint
            "batch_size": 32,
            "learning_rate": 2e-5
        });

        let result = manager.validate_config("training", &config);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::ConstraintViolation)));
    }
}
