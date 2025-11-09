use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use trustformers_core::errors::{Result, TrustformersError};

/// Output validation framework for pipeline outputs
/// Provides comprehensive validation of model outputs for quality assurance

/// Validation result for pipeline outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub score: f64,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub metrics: ValidationMetrics,
    pub suggestions: Vec<ValidationSuggestion>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            score: 1.0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
            suggestions: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.score = (self.score * 0.8).min(0.5); // Penalize score for errors
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.score = (self.score * 0.95).max(0.1); // Minor penalty for warnings
        self.warnings.push(warning);
    }

    pub fn add_suggestion(&mut self, suggestion: ValidationSuggestion) {
        self.suggestions.push(suggestion);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
    pub severity: ValidationSeverity,
    pub field: Option<String>,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub code: String,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub warning_type: ValidationWarningType,
    pub message: String,
    pub field: Option<String>,
    pub recommendation: Option<String>,
    pub code: String,
}

/// Validation suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSuggestion {
    pub suggestion_type: ValidationSuggestionType,
    pub message: String,
    pub improvement: String,
    pub impact: ValidationImpact,
    pub code: String,
}

/// Types of validation errors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationErrorType {
    /// Output format is invalid
    FormatError,
    /// Output content is invalid
    ContentError,
    /// Output violates constraints
    ConstraintViolation,
    /// Output quality is below threshold
    QualityError,
    /// Output contains harmful content
    SafetyError,
    /// Output schema validation failed
    SchemaError,
    /// Output encoding/decoding error
    EncodingError,
    /// Output size exceeds limits
    SizeError,
}

/// Types of validation warnings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationWarningType {
    /// Output quality could be improved
    QualityWarning,
    /// Output may have minor formatting issues
    FormatWarning,
    /// Output performance could be optimized
    PerformanceWarning,
    /// Output may not follow best practices
    BestPracticeWarning,
    /// Output consistency issues
    ConsistencyWarning,
    /// Output completeness issues
    CompletenessWarning,
}

/// Types of validation suggestions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationSuggestionType {
    /// Suggest quality improvements
    QualityImprovement,
    /// Suggest format optimizations
    FormatOptimization,
    /// Suggest content enhancements
    ContentEnhancement,
    /// Suggest performance improvements
    PerformanceOptimization,
    /// Suggest safety improvements
    SafetyEnhancement,
    /// Suggest best practice adoption
    BestPracticeAdoption,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Validation impact levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationImpact {
    Low,
    Medium,
    High,
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationMetrics {
    pub quality_score: f64,
    pub safety_score: f64,
    pub performance_score: f64,
    pub consistency_score: f64,
    pub completeness_score: f64,
    pub overall_score: f64,
    pub validation_time_ms: f64,
    pub checks_performed: u32,
    pub checks_passed: u32,
    pub checks_failed: u32,
}

impl ValidationMetrics {
    pub fn calculate_overall_score(&mut self) {
        let scores = vec![
            self.quality_score,
            self.safety_score,
            self.performance_score,
            self.consistency_score,
            self.completeness_score,
        ];

        let valid_scores: Vec<f64> = scores.into_iter().filter(|&s| s >= 0.0).collect();

        if !valid_scores.is_empty() {
            self.overall_score = valid_scores.iter().sum::<f64>() / valid_scores.len() as f64;
        } else {
            self.overall_score = 0.0;
        }
    }

    pub fn pass_rate(&self) -> f64 {
        if self.checks_performed > 0 {
            self.checks_passed as f64 / self.checks_performed as f64
        } else {
            0.0
        }
    }
}

/// Output validator trait
pub trait OutputValidator: std::fmt::Debug {
    type Input;
    type Output;

    fn validate(&self, input: &Self::Input, output: &Self::Output) -> Result<ValidationResult>;
    fn get_validator_name(&self) -> &str;
    fn get_validator_version(&self) -> &str;
    fn supports_output_type(&self, output_type: &str) -> bool;
}

/// Text output validator
#[derive(Debug, Clone)]
pub struct TextOutputValidator {
    config: TextValidationConfig,
}

#[derive(Debug, Clone)]
pub struct TextValidationConfig {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub check_encoding: bool,
    pub check_language: bool,
    pub check_toxicity: bool,
    pub check_coherence: bool,
    pub check_fluency: bool,
    pub allowed_languages: Vec<String>,
    pub quality_threshold: f64,
    pub safety_threshold: f64,
}

impl Default for TextValidationConfig {
    fn default() -> Self {
        Self {
            min_length: Some(1),
            max_length: Some(10000),
            check_encoding: true,
            check_language: true,
            check_toxicity: true,
            check_coherence: true,
            check_fluency: true,
            allowed_languages: vec!["en".to_string()],
            quality_threshold: 0.7,
            safety_threshold: 0.9,
        }
    }
}

impl TextOutputValidator {
    pub fn new(config: TextValidationConfig) -> Self {
        Self { config }
    }

    fn validate_length(&self, text: &str, result: &mut ValidationResult) {
        let length = text.len();

        if let Some(min_len) = self.config.min_length {
            if length < min_len {
                result.add_error(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    message: format!("Text length {} is below minimum {}", length, min_len),
                    severity: ValidationSeverity::Medium,
                    field: Some("length".to_string()),
                    expected: Some(format!(">= {}", min_len)),
                    actual: Some(length.to_string()),
                    code: "TEXT_TOO_SHORT".to_string(),
                });
            }
        }

        if let Some(max_len) = self.config.max_length {
            if length > max_len {
                result.add_error(ValidationError {
                    error_type: ValidationErrorType::SizeError,
                    message: format!("Text length {} exceeds maximum {}", length, max_len),
                    severity: ValidationSeverity::High,
                    field: Some("length".to_string()),
                    expected: Some(format!("<= {}", max_len)),
                    actual: Some(length.to_string()),
                    code: "TEXT_TOO_LONG".to_string(),
                });
            }
        }
    }

    fn validate_encoding(&self, text: &str, result: &mut ValidationResult) {
        if !self.config.check_encoding {
            return;
        }

        // Check for valid UTF-8 encoding
        if !text.is_ascii()
            && !text
                .chars()
                .all(|c| c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation())
        {
            result.add_warning(ValidationWarning {
                warning_type: ValidationWarningType::FormatWarning,
                message: "Text contains non-standard characters".to_string(),
                field: Some("encoding".to_string()),
                recommendation: Some("Consider using standard character sets".to_string()),
                code: "ENCODING_WARNING".to_string(),
            });
        }

        // Check for common encoding issues
        if text.contains("�") {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::EncodingError,
                message: "Text contains replacement characters indicating encoding issues"
                    .to_string(),
                severity: ValidationSeverity::High,
                field: Some("encoding".to_string()),
                expected: Some("Valid UTF-8 text".to_string()),
                actual: Some("Text with replacement characters".to_string()),
                code: "INVALID_ENCODING".to_string(),
            });
        }
    }

    fn validate_quality(&self, text: &str, result: &mut ValidationResult) {
        let mut quality_score = 1.0;

        // Check for repetitive patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 1 {
            let mut repetition_score = 0.0;
            for window in words.windows(2) {
                if window[0] == window[1] {
                    repetition_score += 1.0;
                }
            }

            let repetition_ratio = repetition_score / (words.len() - 1) as f64;
            if repetition_ratio > 0.3 {
                quality_score *= 0.7;
                result.add_warning(ValidationWarning {
                    warning_type: ValidationWarningType::QualityWarning,
                    message: "High repetition detected in text".to_string(),
                    field: Some("quality".to_string()),
                    recommendation: Some("Consider reducing repetitive patterns".to_string()),
                    code: "HIGH_REPETITION".to_string(),
                });
            }
        }

        // Check for sentence structure
        let sentence_count =
            text.matches('.').count() + text.matches('!').count() + text.matches('?').count();
        if sentence_count == 0 && text.len() > 50 {
            quality_score *= 0.8;
            result.add_warning(ValidationWarning {
                warning_type: ValidationWarningType::FormatWarning,
                message: "No sentence boundaries detected in long text".to_string(),
                field: Some("structure".to_string()),
                recommendation: Some("Consider adding proper punctuation".to_string()),
                code: "NO_SENTENCES".to_string(),
            });
        }

        // Check for basic grammar patterns
        if text.chars().filter(|&c| c.is_uppercase()).count()
            == text.chars().filter(|&c| c.is_alphabetic()).count()
        {
            quality_score *= 0.6;
            result.add_warning(ValidationWarning {
                warning_type: ValidationWarningType::FormatWarning,
                message: "Text is all uppercase".to_string(),
                field: Some("format".to_string()),
                recommendation: Some("Consider using proper case".to_string()),
                code: "ALL_UPPERCASE".to_string(),
            });
        }

        result.metrics.quality_score = quality_score;

        if quality_score < self.config.quality_threshold {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::QualityError,
                message: format!(
                    "Quality score {:.2} below threshold {:.2}",
                    quality_score, self.config.quality_threshold
                ),
                severity: ValidationSeverity::Medium,
                field: Some("quality".to_string()),
                expected: Some(format!(">= {:.2}", self.config.quality_threshold)),
                actual: Some(format!("{:.2}", quality_score)),
                code: "LOW_QUALITY".to_string(),
            });
        }
    }

    fn validate_safety(&self, text: &str, result: &mut ValidationResult) {
        if !self.config.check_toxicity {
            result.metrics.safety_score = 1.0;
            return;
        }

        let mut safety_score = 1.0;

        // Basic toxicity patterns (simplified)
        let toxic_patterns = vec!["hate", "kill", "die", "stupid", "idiot", "fool"];

        let text_lower = text.to_lowercase();
        let mut toxic_count = 0;

        for pattern in &toxic_patterns {
            if text_lower.contains(pattern) {
                toxic_count += 1;
                safety_score *= 0.8;
            }
        }

        if toxic_count > 0 {
            result.add_warning(ValidationWarning {
                warning_type: ValidationWarningType::BestPracticeWarning,
                message: format!(
                    "Potentially harmful content detected ({} patterns)",
                    toxic_count
                ),
                field: Some("safety".to_string()),
                recommendation: Some("Review content for appropriateness".to_string()),
                code: "POTENTIAL_TOXICITY".to_string(),
            });
        }

        result.metrics.safety_score = safety_score;

        if safety_score < self.config.safety_threshold {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::SafetyError,
                message: format!(
                    "Safety score {:.2} below threshold {:.2}",
                    safety_score, self.config.safety_threshold
                ),
                severity: ValidationSeverity::High,
                field: Some("safety".to_string()),
                expected: Some(format!(">= {:.2}", self.config.safety_threshold)),
                actual: Some(format!("{:.2}", safety_score)),
                code: "UNSAFE_CONTENT".to_string(),
            });
        }
    }
}

impl OutputValidator for TextOutputValidator {
    type Input = String;
    type Output = String;

    fn validate(&self, _input: &Self::Input, output: &Self::Output) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        let mut result = ValidationResult::new();

        // Perform validation checks
        self.validate_length(output, &mut result);
        self.validate_encoding(output, &mut result);
        self.validate_quality(output, &mut result);
        self.validate_safety(output, &mut result);

        // Update metrics
        result.metrics.validation_time_ms = start_time.elapsed().as_millis() as f64;
        result.metrics.checks_performed = 4;
        result.metrics.checks_passed = 4 - result.error_count() as u32;
        result.metrics.checks_failed = result.error_count() as u32;
        result.metrics.calculate_overall_score();

        // Add suggestions based on validation results
        if result.metrics.quality_score < 0.8 {
            result.add_suggestion(ValidationSuggestion {
                suggestion_type: ValidationSuggestionType::QualityImprovement,
                message: "Consider improving text quality".to_string(),
                improvement: "Add more varied vocabulary and sentence structures".to_string(),
                impact: ValidationImpact::Medium,
                code: "IMPROVE_QUALITY".to_string(),
            });
        }

        if result.warning_count() > 0 {
            result.add_suggestion(ValidationSuggestion {
                suggestion_type: ValidationSuggestionType::BestPracticeAdoption,
                message: "Address validation warnings".to_string(),
                improvement: "Follow best practices for text generation".to_string(),
                impact: ValidationImpact::Low,
                code: "ADDRESS_WARNINGS".to_string(),
            });
        }

        Ok(result)
    }

    fn get_validator_name(&self) -> &str {
        "TextOutputValidator"
    }

    fn get_validator_version(&self) -> &str {
        "1.0.0"
    }

    fn supports_output_type(&self, output_type: &str) -> bool {
        output_type == "text" || output_type == "string"
    }
}

/// Classification output validator
#[derive(Debug, Clone)]
pub struct ClassificationOutputValidator {
    config: ClassificationValidationConfig,
}

#[derive(Debug, Clone)]
pub struct ClassificationValidationConfig {
    pub expected_classes: Option<Vec<String>>,
    pub min_confidence: Option<f64>,
    pub max_classes: Option<usize>,
    pub require_probabilities: bool,
    pub probability_threshold: f64,
}

impl Default for ClassificationValidationConfig {
    fn default() -> Self {
        Self {
            expected_classes: None,
            min_confidence: Some(0.5),
            max_classes: None,
            require_probabilities: true,
            probability_threshold: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutput {
    pub label: String,
    pub confidence: f64,
    pub probabilities: Option<HashMap<String, f64>>,
}

impl ClassificationOutputValidator {
    pub fn new(config: ClassificationValidationConfig) -> Self {
        Self { config }
    }

    fn validate_label(&self, output: &ClassificationOutput, result: &mut ValidationResult) {
        if let Some(ref expected_classes) = self.config.expected_classes {
            if !expected_classes.contains(&output.label) {
                result.add_error(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    message: format!("Unexpected class label: {}", output.label),
                    severity: ValidationSeverity::High,
                    field: Some("label".to_string()),
                    expected: Some(format!("One of: {:?}", expected_classes)),
                    actual: Some(output.label.clone()),
                    code: "INVALID_CLASS".to_string(),
                });
            }
        }

        if output.label.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::ContentError,
                message: "Empty class label".to_string(),
                severity: ValidationSeverity::High,
                field: Some("label".to_string()),
                expected: Some("Non-empty string".to_string()),
                actual: Some("Empty string".to_string()),
                code: "EMPTY_LABEL".to_string(),
            });
        }
    }

    fn validate_confidence(&self, output: &ClassificationOutput, result: &mut ValidationResult) {
        if output.confidence < 0.0 || output.confidence > 1.0 {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::ConstraintViolation,
                message: format!("Invalid confidence score: {}", output.confidence),
                severity: ValidationSeverity::High,
                field: Some("confidence".to_string()),
                expected: Some("0.0 <= confidence <= 1.0".to_string()),
                actual: Some(output.confidence.to_string()),
                code: "INVALID_CONFIDENCE".to_string(),
            });
        }

        if let Some(min_conf) = self.config.min_confidence {
            if output.confidence < min_conf {
                result.add_warning(ValidationWarning {
                    warning_type: ValidationWarningType::QualityWarning,
                    message: format!("Low confidence score: {:.3}", output.confidence),
                    field: Some("confidence".to_string()),
                    recommendation: Some("Consider improving model certainty".to_string()),
                    code: "LOW_CONFIDENCE".to_string(),
                });
            }
        }
    }

    fn validate_probabilities(&self, output: &ClassificationOutput, result: &mut ValidationResult) {
        if self.config.require_probabilities && output.probabilities.is_none() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::ContentError,
                message: "Missing required probabilities".to_string(),
                severity: ValidationSeverity::Medium,
                field: Some("probabilities".to_string()),
                expected: Some("Probability distribution".to_string()),
                actual: Some("None".to_string()),
                code: "MISSING_PROBABILITIES".to_string(),
            });
            return;
        }

        if let Some(ref probs) = output.probabilities {
            let total: f64 = probs.values().sum();
            if (total - 1.0).abs() > 0.01 {
                result.add_error(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    message: format!("Probabilities don't sum to 1.0: {:.3}", total),
                    severity: ValidationSeverity::Medium,
                    field: Some("probabilities".to_string()),
                    expected: Some("Sum = 1.0".to_string()),
                    actual: Some(format!("Sum = {:.3}", total)),
                    code: "INVALID_PROBABILITY_SUM".to_string(),
                });
            }

            for (class, &prob) in probs {
                if !(0.0..=1.0).contains(&prob) {
                    result.add_error(ValidationError {
                        error_type: ValidationErrorType::ConstraintViolation,
                        message: format!("Invalid probability for class {}: {}", class, prob),
                        severity: ValidationSeverity::Medium,
                        field: Some("probabilities".to_string()),
                        expected: Some("0.0 <= probability <= 1.0".to_string()),
                        actual: Some(format!("{}: {}", class, prob)),
                        code: "INVALID_PROBABILITY".to_string(),
                    });
                }

                if prob < self.config.probability_threshold {
                    result.add_warning(ValidationWarning {
                        warning_type: ValidationWarningType::QualityWarning,
                        message: format!("Very low probability for class {}: {:.6}", class, prob),
                        field: Some("probabilities".to_string()),
                        recommendation: Some(
                            "Consider filtering out very low probability classes".to_string(),
                        ),
                        code: "LOW_PROBABILITY".to_string(),
                    });
                }
            }

            if let Some(max_classes) = self.config.max_classes {
                if probs.len() > max_classes {
                    result.add_warning(ValidationWarning {
                        warning_type: ValidationWarningType::PerformanceWarning,
                        message: format!("Too many classes: {} > {}", probs.len(), max_classes),
                        field: Some("probabilities".to_string()),
                        recommendation: Some(
                            "Consider reducing the number of output classes".to_string(),
                        ),
                        code: "TOO_MANY_CLASSES".to_string(),
                    });
                }
            }
        }
    }
}

impl OutputValidator for ClassificationOutputValidator {
    type Input = String;
    type Output = ClassificationOutput;

    fn validate(&self, _input: &Self::Input, output: &Self::Output) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        let mut result = ValidationResult::new();

        self.validate_label(output, &mut result);
        self.validate_confidence(output, &mut result);
        self.validate_probabilities(output, &mut result);

        result.metrics.validation_time_ms = start_time.elapsed().as_millis() as f64;
        result.metrics.checks_performed = 3;
        result.metrics.checks_passed = 3 - result.error_count() as u32;
        result.metrics.checks_failed = result.error_count() as u32;
        result.metrics.calculate_overall_score();

        Ok(result)
    }

    fn get_validator_name(&self) -> &str {
        "ClassificationOutputValidator"
    }

    fn get_validator_version(&self) -> &str {
        "1.0.0"
    }

    fn supports_output_type(&self, output_type: &str) -> bool {
        output_type == "classification" || output_type == "class"
    }
}

/// Composite validator that can handle multiple output types
#[derive(Debug)]
pub struct CompositeValidator {
    validators: HashMap<String, Box<dyn OutputValidator<Input = String, Output = String>>>,
}

impl CompositeValidator {
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
        }
    }

    pub fn add_text_validator(&mut self, name: String, validator: TextOutputValidator) {
        self.validators.insert(name, Box::new(validator));
    }

    pub fn validate_text(
        &self,
        validator_name: &str,
        input: &str,
        output: &str,
    ) -> Result<ValidationResult> {
        if let Some(validator) = self.validators.get(validator_name) {
            let input_string = input.to_string();
            let output_string = output.to_string();
            validator.validate(&input_string, &output_string)
        } else {
            Err(TrustformersError::invalid_input(format!(
                "Validator {} not found (expected existing validator name, got {})",
                validator_name, validator_name
            )))
        }
    }

    pub fn list_validators(&self) -> Vec<String> {
        self.validators.keys().cloned().collect()
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation manager for coordinating different validators
pub struct ValidationManager {
    composite_validator: CompositeValidator,
    default_config: ValidationManagerConfig,
}

#[derive(Debug, Clone)]
pub struct ValidationManagerConfig {
    pub enable_validation: bool,
    pub fail_on_error: bool,
    pub log_warnings: bool,
    pub collect_metrics: bool,
    pub default_text_config: TextValidationConfig,
    pub default_classification_config: ClassificationValidationConfig,
}

impl Default for ValidationManagerConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            fail_on_error: false,
            log_warnings: true,
            collect_metrics: true,
            default_text_config: TextValidationConfig::default(),
            default_classification_config: ClassificationValidationConfig::default(),
        }
    }
}

impl ValidationManager {
    pub fn new(config: ValidationManagerConfig) -> Self {
        let mut composite_validator = CompositeValidator::new();

        // Add default validators
        composite_validator.add_text_validator(
            "default_text".to_string(),
            TextOutputValidator::new(config.default_text_config.clone()),
        );

        Self {
            composite_validator,
            default_config: config,
        }
    }

    pub fn validate_text_output(&self, input: &str, output: &str) -> Result<ValidationResult> {
        if !self.default_config.enable_validation {
            return Ok(ValidationResult::default());
        }

        let result = self.composite_validator.validate_text("default_text", input, output)?;

        if self.default_config.log_warnings && result.has_warnings() {
            for warning in &result.warnings {
                tracing::warn!("Validation warning: {}", warning.message);
            }
        }

        if self.default_config.fail_on_error && result.has_errors() {
            return Err(TrustformersError::invalid_input(                format!("Validation failed with {} errors (expected valid output, got {} validation errors)", result.error_count(), result.error_count())
            ));
        }

        Ok(result)
    }

    pub fn add_custom_text_validator(&mut self, name: String, config: TextValidationConfig) {
        self.composite_validator
            .add_text_validator(name, TextOutputValidator::new(config));
    }

    pub fn get_available_validators(&self) -> Vec<String> {
        self.composite_validator.list_validators()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Validation Result:")?;
        writeln!(f, "  Valid: {}", self.is_valid)?;
        writeln!(f, "  Score: {:.3}", self.score)?;
        writeln!(f, "  Errors: {}", self.errors.len())?;
        writeln!(f, "  Warnings: {}", self.warnings.len())?;
        writeln!(f, "  Suggestions: {}", self.suggestions.len())?;
        writeln!(f, "  Overall Score: {:.3}", self.metrics.overall_score)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_validator_basic() {
        let config = TextValidationConfig::default();
        let validator = TextOutputValidator::new(config);

        let input = "Hello world";
        let output = "This is a test response.";

        let result = validator.validate(&input.to_string(), &output.to_string()).unwrap();
        assert!(result.is_valid);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_text_validator_length_constraints() {
        let config = TextValidationConfig {
            min_length: Some(10),
            max_length: Some(50),
            ..Default::default()
        };
        let validator = TextOutputValidator::new(config);

        // Test too short
        let result = validator.validate(&"input".to_string(), &"short".to_string()).unwrap();
        assert!(!result.is_valid);
        assert!(result.has_errors());

        // Test too long
        let long_text = "a".repeat(100);
        let result = validator.validate(&"input".to_string(), &long_text).unwrap();
        assert!(!result.is_valid);
        assert!(result.has_errors());

        // Test valid length
        let result = validator
            .validate(
                &"input".to_string(),
                &"This is a valid length text.".to_string(),
            )
            .unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_classification_validator() {
        let config = ClassificationValidationConfig {
            expected_classes: Some(vec!["positive".to_string(), "negative".to_string()]),
            min_confidence: Some(0.7),
            ..Default::default()
        };
        let validator = ClassificationOutputValidator::new(config);

        let output = ClassificationOutput {
            label: "positive".to_string(),
            confidence: 0.85,
            probabilities: Some({
                let mut map = HashMap::new();
                map.insert("positive".to_string(), 0.85);
                map.insert("negative".to_string(), 0.15);
                map
            }),
        };

        let result = validator.validate(&"test input".to_string(), &output).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_validation_manager() {
        let config = ValidationManagerConfig::default();
        let manager = ValidationManager::new(config);

        let result = manager.validate_text_output("input", "This is a test output.").unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_validation_metrics() {
        let mut metrics = ValidationMetrics::default();
        metrics.quality_score = 0.8;
        metrics.safety_score = 0.9;
        metrics.performance_score = 0.7;
        metrics.consistency_score = 0.85;
        metrics.completeness_score = 0.75;

        metrics.calculate_overall_score();
        assert!((metrics.overall_score - 0.8).abs() < 0.01);
    }
}
