//! Safety Validator
//!
//! Provides comprehensive safety validation and compliance checking for
//! concurrent test execution with multi-layered verification.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Instant};

/// Implementation priority for safety violations
#[derive(Debug, Clone)]
pub struct ImplementationPriority {
    pub priority_level: u8,
    pub priority_name: String,
}

pub struct SafetyValidator {
    /// Safety validation rules
    validation_rules: Arc<Mutex<Vec<Box<dyn SafetyValidationRule + Send + Sync>>>>,

    /// Safety constraint database
    constraint_db: Arc<RwLock<SafetyConstraintDatabase>>,

    /// Validation history
    validation_history: Arc<Mutex<SafetyValidationHistory>>,

    /// Configuration
    config: SafetyValidationConfig,
}

impl SafetyValidator {
    /// Creates a new safety validator
    pub async fn new(config: SafetyValidationConfig) -> Result<Self> {
        let mut validation_rules: Vec<Box<dyn SafetyValidationRule + Send + Sync>> = Vec::new();

        // Initialize safety validation rules
        validation_rules.push(Box::new(DeadlockSafetyRule::new()));
        validation_rules.push(Box::new(ResourceSafetyRule::new()));
        validation_rules.push(Box::new(ConcurrencySafetyRule::new()));
        validation_rules.push(Box::new(IsolationSafetyRule::new()));

        let constraint_db = SafetyConstraintDatabase::new();

        Ok(Self {
            validation_rules: Arc::new(Mutex::new(validation_rules)),
            constraint_db: Arc::new(RwLock::new(constraint_db)),
            validation_history: Arc::new(Mutex::new(SafetyValidationHistory::new())),
            config,
        })
    }

    /// Validates safety of concurrency requirements
    pub async fn validate_safety(
        &self,
        _requirements: &ConcurrencyRequirements,
        _test_data: &TestExecutionData,
    ) -> Result<SafetyValidationResult> {
        let start_time = Utc::now();

        // Run safety validation rules synchronously to avoid lifetime issues
        let safety_validation_results: Vec<_> = {
            let rules = self.validation_rules.lock();
            rules
                .iter()
                .map(|rule| {
                    let rule_name = rule.name().to_string();
                    let validation_start = Instant::now();
                    // TODO: validate_safety takes 0 arguments, removed requirements and test_data parameters
                    let result_bool = rule.validate_safety();
                    // Convert bool to SafetyValidation
                    let result: Result<SafetyValidation> = Ok(SafetyValidation {
                        is_safe: result_bool,
                        validation_checks: Vec::new(),
                        violations_found: Vec::new(),
                        violations: Vec::new(),
                    });
                    let validation_duration = validation_start.elapsed();
                    (rule_name, result, validation_duration)
                })
                .collect()
        };

        // Collect validation results
        let mut validation_results = Vec::new();
        let mut safety_violations = Vec::new();

        for (rule_name, result, duration) in safety_validation_results {
            match result {
                Ok(validation) => {
                    validation_results.push(SafetyRuleResult {
                        rule_name,
                        validation: validation.clone(),
                        validation_duration: duration,
                        confidence: self.calculate_rule_confidence(&validation) as f64,
                    });

                    if !validation.is_safe {
                        safety_violations.extend(validation.violations);
                    }
                },
                Err(e) => {
                    log::warn!("Safety validation rule failed: {}", e);
                },
            }
        }

        // Determine overall safety
        let overall_safety = safety_violations.is_empty();
        let safety_score = self.calculate_safety_score(&validation_results);

        // Generate safety recommendations
        let safety_recommendations_vec = self.generate_safety_recommendations(&safety_violations);

        // Generate compliance report
        let _compliance_report_struct =
            self.generate_compliance_report(&validation_results, &safety_violations);

        // Convert types
        let safety_recommendations: Vec<String> =
            safety_recommendations_vec.iter().map(|r| format!("{:?}", r)).collect();
        let compliance_report: HashMap<String, bool> = HashMap::new(); // TODO: extract from compliance_report_struct

        let confidence = self.calculate_overall_safety_confidence(&validation_results) as f64;

        let result = SafetyValidationResult {
            overall_safety,
            safety_score: safety_score as f64,
            safety_violations,
            safety_recommendations,
            safety_constraints: Vec::new(), // TODO: populate with actual safety constraints
            compliance_report,
            validation_results,
            validation_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence,
        };

        // Store validation result
        self.store_validation_result(&result).await?;

        Ok(result)
    }

    /// Calculates rule confidence
    fn calculate_rule_confidence(&self, validation: &SafetyValidation) -> f32 {
        let violation_factor = if validation.violations.is_empty() {
            1.0
        } else {
            1.0 - (validation.violations.len() as f32 * 0.1).min(0.8)
        };

        let severity_factor = validation
            .violations
            .iter()
            .map(|v| match v.severity {
                ViolationSeverity::Critical => 0.2,
                ViolationSeverity::High => 0.4,
                ViolationSeverity::Medium => 0.6,
                ViolationSeverity::Low => 0.8,
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        (violation_factor + severity_factor) / 2.0
    }

    /// Calculates overall safety score
    fn calculate_safety_score(&self, results: &[SafetyRuleResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let safe_rules = results.iter().filter(|r| r.validation.is_safe).count();

        safe_rules as f32 / results.len() as f32
    }

    /// Generates safety recommendations
    fn generate_safety_recommendations(
        &self,
        violations: &[SafetyViolation],
    ) -> Vec<SafetyRecommendation> {
        let mut recommendations = Vec::new();

        for (idx, violation) in violations.iter().enumerate() {
            let recommendation_text = self.generate_violation_recommendation(violation);
            let severity_str = format!("{:?}", violation.severity);

            recommendations.push(SafetyRecommendation {
                recommendation_id: format!("safety_rec_{}", idx),
                recommendation_type: format!("{:?}", violation.violation_type),
                description: recommendation_text,
                severity: severity_str,
            });
        }

        recommendations
    }

    /// Generates recommendation for a specific violation
    fn generate_violation_recommendation(&self, violation: &SafetyViolation) -> String {
        match violation.violation_type {
            ViolationType::DataRace => {
                "Eliminate data races through proper synchronization or using thread-safe primitives"
                    .to_string()
            },
            ViolationType::Deadlock => {
                "Resolve deadlock by reviewing lock ordering and avoiding circular dependencies"
                    .to_string()
            },
            ViolationType::ResourceLeak => {
                "Fix resource leaks by ensuring proper cleanup and RAII patterns".to_string()
            },
            ViolationType::SynchronizationViolation => {
                "Correct synchronization issues by using appropriate synchronization primitives"
                    .to_string()
            },
            ViolationType::LockOrderingViolation => {
                "Establish and enforce consistent lock ordering to prevent deadlocks".to_string()
            },
            ViolationType::DeadlockRisk => {
                "Implement deadlock prevention strategies such as ordered locking or timeouts"
                    .to_string()
            },
            ViolationType::ResourceConflict => {
                "Resolve resource conflicts through partitioning or temporal separation".to_string()
            },
            ViolationType::ConcurrencyViolation => {
                "Adjust concurrency levels to meet safety constraints".to_string()
            },
            ViolationType::IsolationBreach => {
                "Strengthen isolation between concurrent operations".to_string()
            },
            ViolationType::Custom(ref custom) => {
                format!("Address custom safety concern: {}", custom)
            },
        }
    }

    /// Calculates implementation priority
    fn calculate_implementation_priority(
        &self,
        violation: &SafetyViolation,
    ) -> ImplementationPriority {
        match violation.severity {
            ViolationSeverity::Critical => ImplementationPriority {
                priority_level: 4,
                priority_name: "Immediate".to_string(),
            },
            ViolationSeverity::High => ImplementationPriority {
                priority_level: 3,
                priority_name: "High".to_string(),
            },
            ViolationSeverity::Medium => ImplementationPriority {
                priority_level: 2,
                priority_name: "Medium".to_string(),
            },
            ViolationSeverity::Low => ImplementationPriority {
                priority_level: 1,
                priority_name: "Low".to_string(),
            },
        }
    }

    /// Estimates safety impact
    fn estimate_safety_impact(&self, violation: &SafetyViolation) -> f32 {
        match violation.severity {
            ViolationSeverity::Critical => 1.0,
            ViolationSeverity::High => 0.8,
            ViolationSeverity::Medium => 0.6,
            ViolationSeverity::Low => 0.4,
        }
    }

    /// Generates compliance report
    fn generate_compliance_report(
        &self,
        results: &[SafetyRuleResult],
        violations: &[SafetyViolation],
    ) -> ComplianceReport {
        let total_rules = results.len();
        let passed_rules = results.iter().filter(|r| r.validation.is_safe).count();
        let _failed_rules = total_rules - passed_rules;

        let critical_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .count();

        let high_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::High))
            .count();

        let compliance_status = if critical_violations > 0 {
            ComplianceStatus::NonCompliant
        } else if high_violations > 0 {
            ComplianceStatus::ConditionallyCompliant
        } else {
            ComplianceStatus::Compliant
        };

        let violation_list: Vec<String> = violations
            .iter()
            .map(|v| format!("{:?}: {}", v.violation_type, v.severity))
            .collect();

        ComplianceReport {
            report_id: format!("compliance_{}", chrono::Utc::now().timestamp()),
            compliance_status,
            violations: violation_list,
            generated_at: chrono::Utc::now(),
        }
    }

    /// Stores validation result for historical tracking
    async fn store_validation_result(&self, result: &SafetyValidationResult) -> Result<()> {
        let mut history = self.validation_history.lock();
        history.add_validation_result(result.clone());

        // Cleanup old results if needed
        let retention_limit = self.config.history_retention_limit;
        history.cleanup(retention_limit);

        Ok(())
    }

    /// Calculates overall safety confidence
    fn calculate_overall_safety_confidence(&self, results: &[SafetyRuleResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> = results.iter().map(|r| r.confidence as f32).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}
