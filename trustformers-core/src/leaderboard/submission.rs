//! Submission handling and validation for leaderboard entries

#![allow(unused_variables)] // Leaderboard submission

use super::{HardwareInfo, LeaderboardCategory, PerformanceMetrics, SoftwareInfo, SubmitterInfo};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Leaderboard submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardSubmission {
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Category
    pub category: LeaderboardCategory,
    /// Hardware configuration
    pub hardware: HardwareInfo,
    /// Software configuration
    pub software: SoftwareInfo,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Submitter information
    pub submitter: SubmitterInfo,
    /// Tags
    pub tags: Vec<String>,
    /// Benchmark report (optional)
    pub benchmark_report: Option<serde_json::Value>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the submission is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Field that failed validation
    pub field: String,
    /// Error message
    pub message: String,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Field with warning
    pub field: String,
    /// Warning message
    pub message: String,
}

/// Trait for submission validators
#[async_trait]
pub trait SubmissionValidator: Send + Sync {
    /// Validate a submission
    async fn validate(&self, submission: &LeaderboardSubmission) -> Result<ValidationResult>;
}

/// Default submission validator
pub struct DefaultValidator {
    /// Minimum model name length
    min_model_name_length: usize,
    /// Maximum model name length
    max_model_name_length: usize,
    /// Required tags
    required_tags: Vec<String>,
    /// Allowed benchmarks
    allowed_benchmarks: Option<Vec<String>>,
    /// Metric bounds
    metric_bounds: MetricBounds,
}

/// Metric bounds for validation
#[derive(Debug, Clone)]
pub struct MetricBounds {
    /// Minimum latency (ms)
    pub min_latency: f64,
    /// Maximum latency (ms)
    pub max_latency: f64,
    /// Minimum throughput
    pub min_throughput: Option<f64>,
    /// Maximum throughput
    pub max_throughput: Option<f64>,
    /// Minimum memory (MB)
    pub min_memory: Option<f64>,
    /// Maximum memory (MB)
    pub max_memory: Option<f64>,
    /// Minimum accuracy (0-1)
    pub min_accuracy: Option<f64>,
    /// Maximum accuracy (0-1)
    pub max_accuracy: Option<f64>,
}

impl Default for MetricBounds {
    fn default() -> Self {
        Self {
            min_latency: 0.001,     // 1 microsecond
            max_latency: 3600000.0, // 1 hour
            min_throughput: Some(0.001),
            max_throughput: Some(1e9), // 1 billion items/sec
            min_memory: Some(0.1),     // 100 KB
            max_memory: Some(1e6),     // 1 TB
            min_accuracy: Some(0.0),
            max_accuracy: Some(1.0),
        }
    }
}

impl DefaultValidator {
    /// Create new default validator
    pub fn new() -> Self {
        Self {
            min_model_name_length: 3,
            max_model_name_length: 100,
            required_tags: vec![],
            allowed_benchmarks: None,
            metric_bounds: MetricBounds::default(),
        }
    }

    /// Set required tags
    pub fn with_required_tags(mut self, tags: Vec<String>) -> Self {
        self.required_tags = tags;
        self
    }

    /// Set allowed benchmarks
    pub fn with_allowed_benchmarks(mut self, benchmarks: Vec<String>) -> Self {
        self.allowed_benchmarks = Some(benchmarks);
        self
    }

    /// Set metric bounds
    pub fn with_metric_bounds(mut self, bounds: MetricBounds) -> Self {
        self.metric_bounds = bounds;
        self
    }
}

impl Default for DefaultValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SubmissionValidator for DefaultValidator {
    async fn validate(&self, submission: &LeaderboardSubmission) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate model name
        if submission.model_name.len() < self.min_model_name_length {
            errors.push(ValidationError {
                field: "model_name".to_string(),
                message: format!(
                    "Model name must be at least {} characters",
                    self.min_model_name_length
                ),
            });
        }

        if submission.model_name.len() > self.max_model_name_length {
            errors.push(ValidationError {
                field: "model_name".to_string(),
                message: format!(
                    "Model name must be at most {} characters",
                    self.max_model_name_length
                ),
            });
        }

        // Validate benchmark name
        if let Some(allowed) = &self.allowed_benchmarks {
            if !allowed.contains(&submission.benchmark_name) {
                errors.push(ValidationError {
                    field: "benchmark_name".to_string(),
                    message: format!(
                        "Benchmark '{}' is not in the allowed list",
                        submission.benchmark_name
                    ),
                });
            }
        }

        // Validate required tags
        for required_tag in &self.required_tags {
            if !submission.tags.contains(required_tag) {
                warnings.push(ValidationWarning {
                    field: "tags".to_string(),
                    message: format!("Missing recommended tag: {}", required_tag),
                });
            }
        }

        // Validate metrics
        self.validate_metrics(&submission.metrics, &mut errors, &mut warnings);

        // Validate hardware info
        if submission.hardware.cpu_cores == 0 {
            errors.push(ValidationError {
                field: "hardware.cpu_cores".to_string(),
                message: "CPU cores must be greater than 0".to_string(),
            });
        }

        if submission.hardware.memory_gb <= 0.0 {
            errors.push(ValidationError {
                field: "hardware.memory_gb".to_string(),
                message: "Memory must be greater than 0".to_string(),
            });
        }

        // Validate submitter info
        if submission.submitter.name.is_empty() {
            errors.push(ValidationError {
                field: "submitter.name".to_string(),
                message: "Submitter name is required".to_string(),
            });
        }

        // Check for suspicious values
        if submission.metrics.latency_ms < 0.1 {
            warnings.push(ValidationWarning {
                field: "metrics.latency_ms".to_string(),
                message: "Extremely low latency detected - please verify".to_string(),
            });
        }

        if let Some(accuracy) = submission.metrics.accuracy {
            if accuracy > 0.99 {
                warnings.push(ValidationWarning {
                    field: "metrics.accuracy".to_string(),
                    message: "Very high accuracy detected - please verify".to_string(),
                });
            }
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

impl DefaultValidator {
    fn validate_metrics(
        &self,
        metrics: &PerformanceMetrics,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationWarning>,
    ) {
        // Validate latency
        if metrics.latency_ms < self.metric_bounds.min_latency {
            errors.push(ValidationError {
                field: "metrics.latency_ms".to_string(),
                message: format!(
                    "Latency {} ms is below minimum {} ms",
                    metrics.latency_ms, self.metric_bounds.min_latency
                ),
            });
        }

        if metrics.latency_ms > self.metric_bounds.max_latency {
            errors.push(ValidationError {
                field: "metrics.latency_ms".to_string(),
                message: format!(
                    "Latency {} ms exceeds maximum {} ms",
                    metrics.latency_ms, self.metric_bounds.max_latency
                ),
            });
        }

        // Validate latency percentiles
        let percentiles = [
            ("p50", metrics.latency_percentiles.p50),
            ("p90", metrics.latency_percentiles.p90),
            ("p95", metrics.latency_percentiles.p95),
            ("p99", metrics.latency_percentiles.p99),
            ("p999", metrics.latency_percentiles.p999),
        ];

        let mut prev_value = 0.0;
        for (name, value) in percentiles {
            if value < prev_value {
                errors.push(ValidationError {
                    field: format!("metrics.latency_percentiles.{}", name),
                    message: "Percentiles must be in ascending order".to_string(),
                });
            }
            prev_value = value;
        }

        // Validate throughput
        if let Some(throughput) = metrics.throughput {
            if let Some(min) = self.metric_bounds.min_throughput {
                if throughput < min {
                    errors.push(ValidationError {
                        field: "metrics.throughput".to_string(),
                        message: format!("Throughput {} is below minimum {}", throughput, min),
                    });
                }
            }

            if let Some(max) = self.metric_bounds.max_throughput {
                if throughput > max {
                    errors.push(ValidationError {
                        field: "metrics.throughput".to_string(),
                        message: format!("Throughput {} exceeds maximum {}", throughput, max),
                    });
                }
            }
        }

        // Validate memory
        if let Some(memory) = metrics.memory_mb {
            if let Some(min) = self.metric_bounds.min_memory {
                if memory < min {
                    errors.push(ValidationError {
                        field: "metrics.memory_mb".to_string(),
                        message: format!("Memory {} MB is below minimum {} MB", memory, min),
                    });
                }
            }

            if let Some(max) = self.metric_bounds.max_memory {
                if memory > max {
                    errors.push(ValidationError {
                        field: "metrics.memory_mb".to_string(),
                        message: format!("Memory {} MB exceeds maximum {} MB", memory, max),
                    });
                }
            }

            // Check peak memory consistency
            if let Some(peak) = metrics.peak_memory_mb {
                if peak < memory {
                    errors.push(ValidationError {
                        field: "metrics.peak_memory_mb".to_string(),
                        message: "Peak memory cannot be less than average memory".to_string(),
                    });
                }
            }
        }

        // Validate accuracy
        if let Some(accuracy) = metrics.accuracy {
            if let Some(min) = self.metric_bounds.min_accuracy {
                if accuracy < min {
                    errors.push(ValidationError {
                        field: "metrics.accuracy".to_string(),
                        message: format!("Accuracy {} is below minimum {}", accuracy, min),
                    });
                }
            }

            if let Some(max) = self.metric_bounds.max_accuracy {
                if accuracy > max {
                    errors.push(ValidationError {
                        field: "metrics.accuracy".to_string(),
                        message: format!("Accuracy {} exceeds maximum {}", accuracy, max),
                    });
                }
            }
        }

        // Validate GPU utilization
        if let Some(gpu_util) = metrics.gpu_utilization {
            if !(0.0..=100.0).contains(&gpu_util) {
                errors.push(ValidationError {
                    field: "metrics.gpu_utilization".to_string(),
                    message: "GPU utilization must be between 0 and 100".to_string(),
                });
            }
        }
    }
}

/// Chain validator that runs multiple validators
pub struct ChainValidator {
    validators: Vec<Box<dyn SubmissionValidator>>,
}

impl ChainValidator {
    /// Create new chain validator
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the chain
    pub fn add_validator(mut self, validator: Box<dyn SubmissionValidator>) -> Self {
        self.validators.push(validator);
        self
    }
}

impl Default for ChainValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SubmissionValidator for ChainValidator {
    async fn validate(&self, submission: &LeaderboardSubmission) -> Result<ValidationResult> {
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();

        for validator in &self.validators {
            let result = validator.validate(submission).await?;
            all_errors.extend(result.errors);
            all_warnings.extend(result.warnings);
        }

        Ok(ValidationResult {
            is_valid: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;

    fn create_test_submission() -> LeaderboardSubmission {
        LeaderboardSubmission {
            model_name: "test_model".to_string(),
            model_version: "1.0".to_string(),
            benchmark_name: "test_benchmark".to_string(),
            category: LeaderboardCategory::Inference,
            hardware: HardwareInfo {
                cpu: "Test CPU".to_string(),
                cpu_cores: 8,
                gpu: None,
                gpu_count: None,
                memory_gb: 16.0,
                accelerator: None,
                platform: "test".to_string(),
            },
            software: SoftwareInfo {
                framework_version: "0.1.0".to_string(),
                rust_version: "1.75".to_string(),
                os: "Test OS".to_string(),
                optimization_level: OptimizationLevel::O2,
                precision: Precision::FP32,
                quantization: None,
                compiler_flags: vec![],
            },
            metrics: PerformanceMetrics {
                latency_ms: 50.0,
                latency_percentiles: LatencyPercentiles {
                    p50: 45.0,
                    p90: 60.0,
                    p95: 70.0,
                    p99: 85.0,
                    p999: 100.0,
                },
                throughput: Some(20.0),
                tokens_per_second: None,
                memory_mb: Some(512.0),
                peak_memory_mb: Some(768.0),
                gpu_utilization: None,
                accuracy: Some(0.85),
                energy_watts: None,
                custom_metrics: HashMap::new(),
            },
            metadata: HashMap::new(),
            submitter: SubmitterInfo {
                name: "Test User".to_string(),
                organization: None,
                email: None,
                github: None,
            },
            tags: vec!["test".to_string()],
            benchmark_report: None,
        }
    }

    #[tokio::test]
    async fn test_valid_submission() {
        let validator = DefaultValidator::new();
        let submission = create_test_submission();

        let result = validator.validate(&submission).await.unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_invalid_model_name() {
        let validator = DefaultValidator::new();
        let mut submission = create_test_submission();
        submission.model_name = "ab".to_string(); // Too short

        let result = validator.validate(&submission).await.unwrap();
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].field, "model_name");
    }

    #[tokio::test]
    async fn test_invalid_metrics() {
        let validator = DefaultValidator::new();
        let mut submission = create_test_submission();
        submission.metrics.latency_ms = -5.0; // Negative latency

        let result = validator.validate(&submission).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.field == "metrics.latency_ms"));
    }

    #[tokio::test]
    async fn test_percentile_order() {
        let validator = DefaultValidator::new();
        let mut submission = create_test_submission();
        submission.metrics.latency_percentiles.p90 = 40.0; // Less than p50

        let result = validator.validate(&submission).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.field.contains("percentiles")));
    }

    #[tokio::test]
    async fn test_chain_validator() {
        let chain = ChainValidator::new().add_validator(Box::new(DefaultValidator::new()));

        let submission = create_test_submission();
        let result = chain.validate(&submission).await.unwrap();
        assert!(result.is_valid);
    }
}
