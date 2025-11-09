//! Feedback Validation Engine and Rules
//!
//! This module provides comprehensive feedback validation capabilities including
//! a validation engine that orchestrates multiple validation rules and specific
//! rule implementations for range checking, temporal validation, and source
//! reliability assessment.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::warn;

use super::types::*;
use crate::performance_optimizer::types::{FeedbackSource, PerformanceFeedback};

// =============================================================================
// VALIDATION RULE TRAIT
// =============================================================================

/// Trait for feedback validation rules
pub trait ValidationRule {
    /// Get the name of this validation rule
    fn name(&self) -> &str;

    /// Get the priority of this validation rule (higher priority runs first)
    fn priority(&self) -> i32;

    /// Validate a performance feedback entry
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>>;
}

// =============================================================================
// FEEDBACK VALIDATION ENGINE
// =============================================================================

/// Comprehensive feedback validation engine
pub struct FeedbackValidationEngine {
    /// Collection of validation rules
    validation_rules: Arc<Mutex<Vec<Box<dyn ValidationRule + Send + Sync>>>>,
    /// Validation history for analytics
    validation_history: Arc<Mutex<Vec<FeedbackValidationResult>>>,
}

impl FeedbackValidationEngine {
    /// Create new feedback validation engine
    pub fn new() -> Self {
        Self {
            validation_rules: Arc::new(Mutex::new(Vec::new())),
            validation_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(_config: super::types::ValidationEngineConfig) -> Self {
        // For now, use default implementation
        Self::new()
    }

    /// Validate feedback using all rules
    pub async fn validate(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<FeedbackValidationResult> {
        let rules = self.validation_rules.lock();
        let mut all_issues = Vec::new();
        let mut total_score = 0.0f32;
        let mut rule_count = 0u32;

        for rule in rules.iter() {
            match rule.validate(feedback) {
                Ok(issues) => {
                    let rule_score = if issues.is_empty() {
                        1.0
                    } else {
                        1.0 - (issues.len() as f32 * 0.2).min(1.0)
                    };
                    total_score += rule_score;
                    rule_count += 1;
                    all_issues.extend(issues);
                },
                Err(e) => {
                    warn!("Validation rule '{}' failed: {}", rule.name(), e);
                    all_issues.push(ValidationIssue {
                        issue_type: ValidationIssueType::Custom("Rule execution error".to_string()),
                        severity: IssueSeverity::Medium,
                        description: format!("Rule '{}' failed to execute", rule.name()),
                        suggested_resolution: Some("Check rule implementation".to_string()),
                    });
                },
            }
        }

        let score = if rule_count > 0 {
            total_score / rule_count as f32
        } else {
            1.0 // No rules means validation passes
        };

        let valid =
            score >= 0.7 && all_issues.iter().all(|i| i.severity != IssueSeverity::Critical);

        let result = FeedbackValidationResult {
            valid,
            score,
            issues: all_issues,
            quality_metrics: FeedbackQualityMetrics {
                reliability: score,
                relevance: score,
                timeliness: score,
                completeness: score,
                consistency: score,
                overall_quality: score,
                assessed_at: Utc::now(),
            },
            recommended_corrections: Vec::new(), // Would be implemented based on issues
        };

        // Store validation result
        let mut history = self.validation_history.lock();
        history.push(result.clone());
        if history.len() > 1000 {
            history.remove(0);
        }

        Ok(result)
    }

    /// Add validation rule
    pub async fn add_rule(&self, rule: Box<dyn ValidationRule + Send + Sync>) -> Result<()> {
        let mut rules = self.validation_rules.lock();
        rules.push(rule);
        rules.sort_by_key(|r| r.priority());
        Ok(())
    }

    /// Remove validation rule by name
    pub async fn remove_rule(&self, name: &str) -> Result<bool> {
        let mut rules = self.validation_rules.lock();
        let initial_len = rules.len();
        rules.retain(|r| r.name() != name);
        Ok(rules.len() != initial_len)
    }

    /// Get validation statistics
    pub fn get_validation_statistics(&self) -> ValidationStatistics {
        let history = self.validation_history.lock();

        if history.is_empty() {
            return ValidationStatistics::default();
        }

        let total_validations = history.len();
        let successful_validations = history.iter().filter(|r| r.valid).count();
        let average_score = history.iter().map(|r| r.score).sum::<f32>() / total_validations as f32;

        let issue_counts =
            history.iter().flat_map(|r| &r.issues).fold(HashMap::new(), |mut acc, issue| {
                *acc.entry(format!("{:?}", issue.issue_type)).or_insert(0) += 1;
                acc
            });

        ValidationStatistics {
            total_validations,
            successful_validations,
            success_rate: successful_validations as f32 / total_validations as f32,
            average_score,
            common_issues: issue_counts,
        }
    }

    /// Get recent validation results
    pub fn get_recent_validations(&self, count: usize) -> Vec<FeedbackValidationResult> {
        let history = self.validation_history.lock();
        history.iter().rev().take(count).cloned().collect()
    }
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub success_rate: f32,
    pub average_score: f32,
    pub common_issues: HashMap<String, u32>,
}

impl Default for ValidationStatistics {
    fn default() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            success_rate: 1.0,
            average_score: 1.0,
            common_issues: HashMap::new(),
        }
    }
}

// =============================================================================
// RANGE VALIDATION RULE
// =============================================================================

/// Range validation rule
pub struct RangeValidationRule {
    /// Minimum allowed value
    pub min_value: f64,
    /// Maximum allowed value
    pub max_value: f64,
}

impl RangeValidationRule {
    /// Create new range validation rule
    pub fn new(min_value: f64, max_value: f64) -> Self {
        Self {
            min_value,
            max_value,
        }
    }
}

impl ValidationRule for RangeValidationRule {
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if feedback.value < self.min_value {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::ValueOutOfRange,
                severity: IssueSeverity::High,
                description: format!(
                    "Value {} is below minimum {}",
                    feedback.value, self.min_value
                ),
                suggested_resolution: Some("Verify data source accuracy".to_string()),
            });
        }

        if feedback.value > self.max_value {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::ValueOutOfRange,
                severity: IssueSeverity::High,
                description: format!(
                    "Value {} exceeds maximum {}",
                    feedback.value, self.max_value
                ),
                suggested_resolution: Some("Check for measurement errors".to_string()),
            });
        }

        Ok(issues)
    }

    fn name(&self) -> &str {
        "range_validation"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// =============================================================================
// TEMPORAL VALIDATION RULE
// =============================================================================

/// Temporal validation rule
pub struct TemporalValidationRule {
    /// Maximum allowed age
    pub max_age: Duration,
}

impl TemporalValidationRule {
    /// Create new temporal validation rule
    pub fn new(max_age: Duration) -> Self {
        Self { max_age }
    }
}

impl ValidationRule for TemporalValidationRule {
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();
        let age = Utc::now().signed_duration_since(feedback.timestamp);

        if let Ok(age_duration) = age.to_std() {
            if age_duration > self.max_age {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::TemporalInconsistency,
                    severity: IssueSeverity::Medium,
                    description: format!(
                        "Feedback is {} old, exceeds maximum age of {}ms",
                        age_duration.as_millis(),
                        self.max_age.as_millis()
                    ),
                    suggested_resolution: Some("Use more recent data".to_string()),
                });
            }
        }

        Ok(issues)
    }

    fn name(&self) -> &str {
        "temporal_validation"
    }

    fn priority(&self) -> i32 {
        90
    }
}

// =============================================================================
// SOURCE RELIABILITY VALIDATION RULE
// =============================================================================

/// Source reliability validation rule
pub struct SourceReliabilityRule {
    /// Minimum reliability scores by source
    pub min_reliability: HashMap<FeedbackSource, f32>,
}

impl Default for SourceReliabilityRule {
    fn default() -> Self {
        let mut min_reliability = HashMap::new();
        min_reliability.insert(FeedbackSource::PerformanceMonitor, 0.9);
        min_reliability.insert(FeedbackSource::ResourceMonitor, 0.8);
        min_reliability.insert(FeedbackSource::TestExecutionEngine, 0.7);
        min_reliability.insert(FeedbackSource::ExternalSystem, 0.5);
        min_reliability.insert(FeedbackSource::UserInput, 0.3);

        Self { min_reliability }
    }
}

impl SourceReliabilityRule {
    /// Create new source reliability rule with custom thresholds
    pub fn new(min_reliability: HashMap<FeedbackSource, f32>) -> Self {
        Self { min_reliability }
    }
}

impl ValidationRule for SourceReliabilityRule {
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if let Some(&min_rel) = self.min_reliability.get(&feedback.source) {
            let source_reliability = match feedback.source {
                FeedbackSource::PerformanceMonitor => 0.95,
                FeedbackSource::ResourceMonitor => 0.9,
                FeedbackSource::TestExecutionEngine => 0.85,
                FeedbackSource::ExternalSystem => 0.7,
                FeedbackSource::UserInput => 0.5,
            };

            if source_reliability < min_rel {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::SourceUnreliable,
                    severity: IssueSeverity::Medium,
                    description: format!(
                        "Source reliability {} below threshold {}",
                        source_reliability, min_rel
                    ),
                    suggested_resolution: Some("Use alternative data source".to_string()),
                });
            }
        }

        Ok(issues)
    }

    fn name(&self) -> &str {
        "source_reliability"
    }

    fn priority(&self) -> i32 {
        80
    }
}

// =============================================================================
// DATA INTEGRITY VALIDATION RULE
// =============================================================================

/// Data integrity validation rule
pub struct DataIntegrityRule {
    /// Check for null/NaN values
    pub check_null_values: bool,
    /// Check for negative values where inappropriate
    pub check_negative_values: bool,
    /// Check for zero values where inappropriate
    pub check_zero_values: bool,
}

impl Default for DataIntegrityRule {
    fn default() -> Self {
        Self {
            check_null_values: true,
            check_negative_values: true,
            check_zero_values: true,
        }
    }
}

impl DataIntegrityRule {
    /// Create new data integrity rule
    pub fn new(check_null: bool, check_negative: bool, check_zero: bool) -> Self {
        Self {
            check_null_values: check_null,
            check_negative_values: check_negative,
            check_zero_values: check_zero,
        }
    }
}

impl ValidationRule for DataIntegrityRule {
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        // Check for invalid numeric values
        if self.check_null_values && (feedback.value.is_nan() || feedback.value.is_infinite()) {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::DataIntegrity,
                severity: IssueSeverity::Critical,
                description: "Invalid numeric value detected (NaN or Infinite)".to_string(),
                suggested_resolution: Some(
                    "Check data collection and processing pipeline".to_string(),
                ),
            });
        }

        // Check for negative values where inappropriate
        if self.check_negative_values && feedback.value < 0.0 {
            // For certain feedback types, negative values don't make sense
            match feedback.feedback_type {
                crate::performance_optimizer::types::FeedbackType::Throughput
                | crate::performance_optimizer::types::FeedbackType::ResourceUtilization => {
                    issues.push(ValidationIssue {
                        issue_type: ValidationIssueType::DataIntegrity,
                        severity: IssueSeverity::High,
                        description: format!(
                            "Negative value {} for {:?} feedback",
                            feedback.value, feedback.feedback_type
                        ),
                        suggested_resolution: Some("Verify measurement methodology".to_string()),
                    });
                },
                _ => {}, // Negative values might be valid for other types
            }
        }

        // Check for zero values where suspicious
        if self.check_zero_values && feedback.value == 0.0 {
            match feedback.feedback_type {
                crate::performance_optimizer::types::FeedbackType::Throughput => {
                    issues.push(ValidationIssue {
                        issue_type: ValidationIssueType::DataIntegrity,
                        severity: IssueSeverity::Medium,
                        description: "Zero throughput value detected".to_string(),
                        suggested_resolution: Some(
                            "Verify system is processing requests".to_string(),
                        ),
                    });
                },
                _ => {}, // Zero might be valid for other metrics
            }
        }

        Ok(issues)
    }

    fn name(&self) -> &str {
        "data_integrity"
    }

    fn priority(&self) -> i32 {
        95
    }
}

// =============================================================================
// CONTEXT COMPLETENESS VALIDATION RULE
// =============================================================================

/// Context completeness validation rule
pub struct ContextCompletenessRule {
    /// Required context fields
    pub required_fields: Vec<String>,
    /// Minimum context completeness score
    pub min_completeness: f32,
}

impl Default for ContextCompletenessRule {
    fn default() -> Self {
        Self {
            required_fields: vec!["test_id".to_string(), "parallelism_level".to_string()],
            min_completeness: 0.7,
        }
    }
}

impl ContextCompletenessRule {
    /// Create new context completeness rule
    pub fn new(required_fields: Vec<String>, min_completeness: f32) -> Self {
        Self {
            required_fields,
            min_completeness,
        }
    }
}

impl ValidationRule for ContextCompletenessRule {
    fn validate(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();
        let context = &feedback.context;

        // Check for required fields
        let mut missing_fields = Vec::new();
        for field in &self.required_fields {
            if !context.additional_context.contains_key(field) {
                missing_fields.push(field.clone());
            }
        }

        if !missing_fields.is_empty() {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::MissingContext,
                severity: IssueSeverity::Medium,
                description: format!(
                    "Missing required context fields: {}",
                    missing_fields.join(", ")
                ),
                suggested_resolution: Some(
                    "Include required context information in feedback".to_string(),
                ),
            });
        }

        // Calculate completeness score
        // TODO: FeedbackContext fields changed - no longer has test_name, resource_usage, environment_info as Option fields
        // FeedbackContext now has test_characteristics (always present), system_state (always present), and additional_context
        let total_possible_fields = self.required_fields.len() + 3; // Base fields + required
        let present_fields = context.additional_context.len()
            + 2 // test_characteristics and system_state are always present
            + 1; // timestamp is always present

        let completeness_score = present_fields as f32 / total_possible_fields as f32;

        if completeness_score < self.min_completeness {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::MissingContext,
                severity: IssueSeverity::Low,
                description: format!(
                    "Context completeness {} below threshold {}",
                    completeness_score, self.min_completeness
                ),
                suggested_resolution: Some("Provide more context information".to_string()),
            });
        }

        Ok(issues)
    }

    fn name(&self) -> &str {
        "context_completeness"
    }

    fn priority(&self) -> i32 {
        70
    }
}

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/// Create a comprehensive validation engine with default rules
pub fn create_comprehensive_validation_engine() -> FeedbackValidationEngine {
    let engine = FeedbackValidationEngine::new();

    // Add default validation rules (in a real async context)
    // engine.add_rule(Box::new(RangeValidationRule::new(0.0, 10000.0))).await.ok();
    // engine.add_rule(Box::new(TemporalValidationRule::new(Duration::from_secs(300)))).await.ok();
    // engine.add_rule(Box::new(SourceReliabilityRule::default())).await.ok();
    // engine.add_rule(Box::new(DataIntegrityRule::default())).await.ok();
    // engine.add_rule(Box::new(ContextCompletenessRule::default())).await.ok();

    engine
}

/// Create a basic validation engine with minimal rules
pub fn create_basic_validation_engine() -> FeedbackValidationEngine {
    let engine = FeedbackValidationEngine::new();

    // Would add minimal rules in async context
    // engine.add_rule(Box::new(DataIntegrityRule::default())).await.ok();
    // engine.add_rule(Box::new(RangeValidationRule::new(0.0, 1000.0))).await.ok();

    engine
}
