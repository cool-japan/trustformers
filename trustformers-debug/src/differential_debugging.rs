//! # Differential Debugging System
//!
//! Advanced model comparison, A/B analysis, version diff tracking, regression identification,
//! and performance delta analysis for TrustformeRS models.

use anyhow::Result;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
// use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for differential debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialDebuggingConfig {
    /// Enable model comparison analysis
    pub enable_model_comparison: bool,
    /// Enable A/B testing analysis
    pub enable_ab_analysis: bool,
    /// Enable version diff tracking
    pub enable_version_diff: bool,
    /// Enable regression identification
    pub enable_regression_detection: bool,
    /// Enable performance delta analysis
    pub enable_performance_delta: bool,
    /// Statistical significance threshold for comparisons
    pub significance_threshold: f64,
    /// Maximum number of models to compare simultaneously
    pub max_comparison_models: usize,
    /// Regression detection sensitivity (0.0 to 1.0)
    pub regression_sensitivity: f64,
    /// Performance delta threshold (percentage)
    pub performance_delta_threshold: f64,
}

impl Default for DifferentialDebuggingConfig {
    fn default() -> Self {
        Self {
            enable_model_comparison: true,
            enable_ab_analysis: true,
            enable_version_diff: true,
            enable_regression_detection: true,
            enable_performance_delta: true,
            significance_threshold: 0.05,
            max_comparison_models: 10,
            regression_sensitivity: 0.8,
            performance_delta_threshold: 5.0,
        }
    }
}

/// Model snapshot for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    /// Unique identifier for the model snapshot
    pub id: Uuid,
    /// Model name or version identifier
    pub name: String,
    /// Timestamp when snapshot was created
    pub timestamp: DateTime<Utc>,
    /// Model version information
    pub version: String,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Model performance metrics
    pub metrics: ModelMetrics,
    /// Model architecture information
    pub architecture: ArchitectureInfo,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Model weights summary statistics
    pub weights_summary: WeightsSummary,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance metrics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Training accuracy
    pub train_accuracy: f64,
    /// Validation accuracy
    pub val_accuracy: f64,
    /// Test accuracy (if available)
    pub test_accuracy: Option<f64>,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: f64,
    /// Test loss (if available)
    pub test_loss: Option<f64>,
    /// Inference latency (ms)
    pub inference_latency_ms: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Model size (MB)
    pub model_size_mb: f64,
    /// FLOPS count
    pub flops: u64,
    /// Training time (seconds)
    pub training_time_s: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    /// Number of parameters
    pub parameter_count: u64,
    /// Number of layers
    pub layer_count: u32,
    /// Model depth
    pub depth: u32,
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Number of attention heads
    pub num_heads: Option<u32>,
    /// Feed-forward dimension
    pub ff_dim: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Sequence length
    pub max_seq_length: Option<u32>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: u32,
    /// Number of epochs
    pub epochs: u32,
    /// Optimizer type
    pub optimizer: String,
    /// Learning rate schedule
    pub lr_schedule: Option<String>,
    /// Regularization parameters
    pub regularization: HashMap<String, f64>,
}

/// Summary statistics for model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsSummary {
    /// Mean weight value
    pub mean: f64,
    /// Standard deviation of weights
    pub std_dev: f64,
    /// Minimum weight value
    pub min: f64,
    /// Maximum weight value
    pub max: f64,
    /// Weight distribution percentiles
    pub percentiles: HashMap<String, f64>,
    /// Number of zero weights
    pub zero_count: u64,
    /// Sparsity ratio
    pub sparsity: f64,
}

/// Result of model comparison analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    /// Models being compared
    pub models: Vec<String>,
    /// Comparison timestamp
    pub timestamp: DateTime<Utc>,
    /// Performance comparison
    pub performance_comparison: PerformanceComparison,
    /// Architecture differences
    pub architecture_diff: ArchitectureDiff,
    /// Statistical significance results
    pub statistical_analysis: StatisticalAnalysis,
    /// Overall comparison summary
    pub summary: ComparisonSummary,
}

/// Performance comparison between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Accuracy comparison
    pub accuracy_comparison: MetricComparison,
    /// Loss comparison
    pub loss_comparison: MetricComparison,
    /// Latency comparison
    pub latency_comparison: MetricComparison,
    /// Memory usage comparison
    pub memory_comparison: MetricComparison,
    /// Model size comparison
    pub size_comparison: MetricComparison,
    /// Custom metric comparisons
    pub custom_comparisons: HashMap<String, MetricComparison>,
}

/// Comparison result for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Values for each model
    pub values: HashMap<String, f64>,
    /// Best performing model for this metric
    pub best_model: String,
    /// Worst performing model for this metric
    pub worst_model: String,
    /// Performance differences (relative to best)
    pub differences: HashMap<String, f64>,
    /// Statistical significance of differences
    pub significant_differences: HashMap<String, bool>,
}

/// Architecture differences between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDiff {
    /// Parameter count differences
    pub parameter_diff: HashMap<String, i64>,
    /// Layer count differences
    pub layer_diff: HashMap<String, i32>,
    /// Architecture similarity score (0.0 to 1.0)
    pub similarity_score: f64,
    /// Notable differences
    pub notable_differences: Vec<String>,
}

/// Statistical analysis of comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// P-values for metric comparisons
    pub p_values: HashMap<String, f64>,
    /// Effect sizes (Cohen's d)
    pub effect_sizes: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Statistical significance summary
    pub significance_summary: HashMap<String, bool>,
}

/// Overall comparison summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Overall best model
    pub best_model: String,
    /// Model rankings by different criteria
    pub rankings: HashMap<String, Vec<String>>,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// A/B test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Test name
    pub name: String,
    /// Model A identifier
    pub model_a: String,
    /// Model B identifier
    pub model_b: String,
    /// Test duration (if applicable)
    pub duration_hours: Option<u32>,
    /// Sample size for each group
    pub sample_size: u32,
    /// Metrics to track
    pub tracked_metrics: Vec<String>,
    /// Minimum detectable effect size
    pub min_effect_size: f64,
    /// Statistical power
    pub power: f64,
}

/// A/B test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResult {
    /// Test configuration
    pub config: ABTestConfig,
    /// Test start time
    pub start_time: DateTime<Utc>,
    /// Test end time
    pub end_time: Option<DateTime<Utc>>,
    /// Model A results
    pub model_a_results: ABTestMetrics,
    /// Model B results
    pub model_b_results: ABTestMetrics,
    /// Statistical test results
    pub statistical_tests: HashMap<String, StatisticalTestResult>,
    /// Test conclusion
    pub conclusion: ABTestConclusion,
}

/// Metrics for A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestMetrics {
    /// Sample size
    pub sample_size: u32,
    /// Metric values
    pub metrics: HashMap<String, Vec<f64>>,
    /// Summary statistics
    pub summary_stats: HashMap<String, SummaryStats>,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test type (t-test, Mann-Whitney U, etc.)
    pub test_type: String,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval for difference
    pub confidence_interval: (f64, f64),
    /// Is result statistically significant?
    pub is_significant: bool,
}

/// A/B test conclusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConclusion {
    /// Winner (if any)
    pub winner: Option<String>,
    /// Confidence level
    pub confidence: f64,
    /// Practical significance
    pub practical_significance: bool,
    /// Recommendation
    pub recommendation: String,
    /// Summary
    pub summary: String,
}

/// Version diff tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Previous version
    pub from_version: String,
    /// Current version
    pub to_version: String,
    /// Diff timestamp
    pub timestamp: DateTime<Utc>,
    /// Performance changes
    pub performance_delta: PerformanceDelta,
    /// Architecture changes
    pub architecture_changes: Vec<ArchitectureChange>,
    /// Configuration changes
    pub config_changes: Vec<ConfigChange>,
    /// Weight changes summary
    pub weight_changes: WeightChangesSummary,
}

/// Performance delta between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    /// Accuracy change
    pub accuracy_delta: f64,
    /// Loss change
    pub loss_delta: f64,
    /// Latency change
    pub latency_delta: f64,
    /// Memory usage change
    pub memory_delta: f64,
    /// Model size change
    pub size_delta: f64,
    /// Training time change
    pub training_time_delta: f64,
    /// Custom metric changes
    pub custom_deltas: HashMap<String, f64>,
}

/// Architecture change description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureChange {
    /// Type of change
    pub change_type: String,
    /// Description
    pub description: String,
    /// Impact assessment
    pub impact: String,
}

/// Configuration change description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    /// Parameter name
    pub parameter: String,
    /// Old value
    pub old_value: String,
    /// New value
    pub new_value: String,
    /// Change impact
    pub impact: String,
}

/// Summary of weight changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightChangesSummary {
    /// Average magnitude of weight changes
    pub avg_magnitude: f64,
    /// Maximum weight change
    pub max_change: f64,
    /// Percentage of weights that changed significantly
    pub significant_change_ratio: f64,
    /// Layer-wise change summary
    pub layer_changes: HashMap<String, f64>,
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Detected regressions
    pub regressions: Vec<Regression>,
    /// Performance improvements
    pub improvements: Vec<Improvement>,
    /// Overall assessment
    pub overall_assessment: RegressionAssessment,
}

/// Detected regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Metric that regressed
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Previous value
    pub previous_value: f64,
    /// Regression magnitude
    pub magnitude: f64,
    /// Severity level
    pub severity: RegressionSeverity,
    /// Possible causes
    pub possible_causes: Vec<String>,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
}

/// Performance improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Metric that improved
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Previous value
    pub previous_value: f64,
    /// Improvement magnitude
    pub magnitude: f64,
    /// Likely causes
    pub likely_causes: Vec<String>,
}

/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Critical,
    Major,
    Minor,
    Negligible,
}

/// Overall regression assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAssessment {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,
    /// Number of critical regressions
    pub critical_regressions: usize,
    /// Number of improvements
    pub improvements: usize,
    /// Recommendation
    pub recommendation: String,
}

/// Main differential debugging analyzer
#[derive(Debug)]
pub struct DifferentialDebugger {
    config: DifferentialDebuggingConfig,
    model_snapshots: IndexMap<String, ModelSnapshot>,
    comparison_history: Vec<ModelComparisonResult>,
    ab_tests: Vec<ABTestResult>,
    version_diffs: Vec<VersionDiff>,
    regression_history: Vec<RegressionDetectionResult>,
}

impl DifferentialDebugger {
    /// Create a new differential debugger
    pub fn new(config: DifferentialDebuggingConfig) -> Self {
        Self {
            config,
            model_snapshots: IndexMap::new(),
            comparison_history: Vec::new(),
            ab_tests: Vec::new(),
            version_diffs: Vec::new(),
            regression_history: Vec::new(),
        }
    }

    /// Add a model snapshot for comparison
    pub fn add_model_snapshot(&mut self, snapshot: ModelSnapshot) -> Result<()> {
        if self.model_snapshots.len() >= self.config.max_comparison_models {
            // Remove oldest snapshot
            self.model_snapshots.shift_remove_index(0);
        }

        self.model_snapshots.insert(snapshot.name.clone(), snapshot);
        Ok(())
    }

    /// Compare two or more models
    pub async fn compare_models(
        &mut self,
        model_names: Vec<String>,
    ) -> Result<ModelComparisonResult> {
        if !self.config.enable_model_comparison {
            return Err(anyhow::anyhow!("Model comparison is disabled"));
        }

        if model_names.len() < 2 {
            return Err(anyhow::anyhow!(
                "At least two models are required for comparison"
            ));
        }

        // Get model snapshots
        let models: Vec<&ModelSnapshot> = model_names
            .iter()
            .map(|name| {
                self.model_snapshots
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", name))
            })
            .collect::<Result<Vec<_>>>()?;

        // Perform comparison analysis
        let performance_comparison = self.compare_performance(&models)?;
        let architecture_diff = self.analyze_architecture_differences(&models)?;
        let statistical_analysis = self.perform_statistical_analysis(&models)?;
        let summary = self.generate_comparison_summary(
            &models,
            &performance_comparison,
            &statistical_analysis,
        )?;

        let result = ModelComparisonResult {
            models: model_names,
            timestamp: Utc::now(),
            performance_comparison,
            architecture_diff,
            statistical_analysis,
            summary,
        };

        self.comparison_history.push(result.clone());
        Ok(result)
    }

    /// Run A/B test analysis
    pub async fn run_ab_test(
        &mut self,
        config: ABTestConfig,
        model_a_data: Vec<f64>,
        model_b_data: Vec<f64>,
    ) -> Result<ABTestResult> {
        if !self.config.enable_ab_analysis {
            return Err(anyhow::anyhow!("A/B analysis is disabled"));
        }

        let start_time = Utc::now();

        // Calculate summary statistics for both models
        let model_a_stats = self.calculate_summary_stats(&model_a_data);
        let model_b_stats = self.calculate_summary_stats(&model_b_data);

        let model_a_results = ABTestMetrics {
            sample_size: model_a_data.len() as u32,
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("primary_metric".to_string(), model_a_data);
                metrics
            },
            summary_stats: {
                let mut stats = HashMap::new();
                stats.insert("primary_metric".to_string(), model_a_stats);
                stats
            },
        };

        let model_b_results = ABTestMetrics {
            sample_size: model_b_data.len() as u32,
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("primary_metric".to_string(), model_b_data);
                metrics
            },
            summary_stats: {
                let mut stats = HashMap::new();
                stats.insert("primary_metric".to_string(), model_b_stats);
                stats
            },
        };

        // Perform statistical tests
        let statistical_tests =
            self.perform_ab_statistical_tests(&model_a_results, &model_b_results)?;

        // Generate conclusion
        let conclusion = self.generate_ab_conclusion(
            &config,
            &model_a_results,
            &model_b_results,
            &statistical_tests,
        )?;

        let result = ABTestResult {
            config,
            start_time,
            end_time: Some(Utc::now()),
            model_a_results,
            model_b_results,
            statistical_tests,
            conclusion,
        };

        self.ab_tests.push(result.clone());
        Ok(result)
    }

    /// Track version differences
    pub async fn track_version_diff(
        &mut self,
        from_model: &str,
        to_model: &str,
    ) -> Result<VersionDiff> {
        if !self.config.enable_version_diff {
            return Err(anyhow::anyhow!("Version diff tracking is disabled"));
        }

        let from_snapshot = self
            .model_snapshots
            .get(from_model)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", from_model))?;
        let to_snapshot = self
            .model_snapshots
            .get(to_model)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", to_model))?;

        let performance_delta = self.calculate_performance_delta(from_snapshot, to_snapshot)?;
        let architecture_changes = self.detect_architecture_changes(from_snapshot, to_snapshot)?;
        let config_changes = self.detect_config_changes(from_snapshot, to_snapshot)?;
        let weight_changes = self.analyze_weight_changes(from_snapshot, to_snapshot)?;

        let diff = VersionDiff {
            from_version: from_snapshot.version.clone(),
            to_version: to_snapshot.version.clone(),
            timestamp: Utc::now(),
            performance_delta,
            architecture_changes,
            config_changes,
            weight_changes,
        };

        self.version_diffs.push(diff.clone());
        Ok(diff)
    }

    /// Detect performance regressions
    pub async fn detect_regressions(
        &mut self,
        current_model: &str,
        baseline_model: &str,
    ) -> Result<RegressionDetectionResult> {
        if !self.config.enable_regression_detection {
            return Err(anyhow::anyhow!("Regression detection is disabled"));
        }

        let current = self
            .model_snapshots
            .get(current_model)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", current_model))?;
        let baseline = self
            .model_snapshots
            .get(baseline_model)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", baseline_model))?;

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Check accuracy regression
        if current.metrics.val_accuracy < baseline.metrics.val_accuracy {
            let magnitude = baseline.metrics.val_accuracy - current.metrics.val_accuracy;
            if magnitude > self.config.regression_sensitivity * 0.01 {
                regressions.push(Regression {
                    metric: "validation_accuracy".to_string(),
                    current_value: current.metrics.val_accuracy,
                    previous_value: baseline.metrics.val_accuracy,
                    magnitude,
                    severity: self.classify_regression_severity(magnitude, "accuracy"),
                    possible_causes: vec![
                        "Learning rate too high".to_string(),
                        "Insufficient training".to_string(),
                        "Data distribution shift".to_string(),
                    ],
                    suggested_fixes: vec![
                        "Reduce learning rate".to_string(),
                        "Increase training epochs".to_string(),
                        "Check data quality".to_string(),
                    ],
                });
            }
        } else if current.metrics.val_accuracy > baseline.metrics.val_accuracy {
            let magnitude = current.metrics.val_accuracy - baseline.metrics.val_accuracy;
            improvements.push(Improvement {
                metric: "validation_accuracy".to_string(),
                current_value: current.metrics.val_accuracy,
                previous_value: baseline.metrics.val_accuracy,
                magnitude,
                likely_causes: vec![
                    "Better optimization".to_string(),
                    "Improved architecture".to_string(),
                    "Better hyperparameters".to_string(),
                ],
            });
        }

        // Check latency regression
        if current.metrics.inference_latency_ms > baseline.metrics.inference_latency_ms {
            let magnitude =
                current.metrics.inference_latency_ms - baseline.metrics.inference_latency_ms;
            let relative_change = magnitude / baseline.metrics.inference_latency_ms * 100.0;
            if relative_change > self.config.performance_delta_threshold {
                regressions.push(Regression {
                    metric: "inference_latency".to_string(),
                    current_value: current.metrics.inference_latency_ms,
                    previous_value: baseline.metrics.inference_latency_ms,
                    magnitude,
                    severity: self.classify_regression_severity(relative_change, "latency"),
                    possible_causes: vec![
                        "Model complexity increased".to_string(),
                        "Inefficient implementation".to_string(),
                        "Hardware degradation".to_string(),
                    ],
                    suggested_fixes: vec![
                        "Profile and optimize bottlenecks".to_string(),
                        "Consider model compression".to_string(),
                        "Check hardware configuration".to_string(),
                    ],
                });
            }
        }

        let critical_regressions = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
            .count();

        let health_score = if critical_regressions > 0 {
            0.0
        } else {
            1.0 - (regressions.len() as f64 * 0.1).min(1.0)
        };

        let recommendation = if critical_regressions > 0 {
            "Critical regressions detected. Immediate action required.".to_string()
        } else if !regressions.is_empty() {
            "Some regressions detected. Review and address if necessary.".to_string()
        } else {
            "No significant regressions detected.".to_string()
        };

        let overall_assessment = RegressionAssessment {
            health_score,
            critical_regressions,
            improvements: improvements.len(),
            recommendation,
        };

        let result = RegressionDetectionResult {
            timestamp: Utc::now(),
            regressions,
            improvements,
            overall_assessment,
        };

        self.regression_history.push(result.clone());
        Ok(result)
    }

    /// Generate comprehensive differential debugging report
    pub async fn generate_report(&self) -> Result<DifferentialDebuggingReport> {
        Ok(DifferentialDebuggingReport {
            timestamp: Utc::now(),
            config: self.config.clone(),
            total_models: self.model_snapshots.len(),
            comparison_count: self.comparison_history.len(),
            ab_test_count: self.ab_tests.len(),
            version_diff_count: self.version_diffs.len(),
            regression_detection_count: self.regression_history.len(),
            recent_comparisons: self.comparison_history.iter().rev().take(5).cloned().collect(),
            recent_regressions: self.regression_history.iter().rev().take(3).cloned().collect(),
            model_summary: self.generate_model_summary(),
        })
    }

    // Helper methods

    fn compare_performance(&self, models: &[&ModelSnapshot]) -> Result<PerformanceComparison> {
        let mut accuracy_values = HashMap::new();
        let mut loss_values = HashMap::new();
        let mut latency_values = HashMap::new();
        let mut memory_values = HashMap::new();
        let mut size_values = HashMap::new();

        for model in models {
            accuracy_values.insert(model.name.clone(), model.metrics.val_accuracy);
            loss_values.insert(model.name.clone(), model.metrics.val_loss);
            latency_values.insert(model.name.clone(), model.metrics.inference_latency_ms);
            memory_values.insert(model.name.clone(), model.metrics.memory_usage_mb);
            size_values.insert(model.name.clone(), model.metrics.model_size_mb);
        }

        Ok(PerformanceComparison {
            accuracy_comparison: self.create_metric_comparison(accuracy_values, true)?,
            loss_comparison: self.create_metric_comparison(loss_values, false)?,
            latency_comparison: self.create_metric_comparison(latency_values, false)?,
            memory_comparison: self.create_metric_comparison(memory_values, false)?,
            size_comparison: self.create_metric_comparison(size_values, false)?,
            custom_comparisons: HashMap::new(),
        })
    }

    fn create_metric_comparison(
        &self,
        values: HashMap<String, f64>,
        higher_is_better: bool,
    ) -> Result<MetricComparison> {
        let best_model = if higher_is_better {
            values.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone()
        } else {
            values.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone()
        };

        let worst_model = if higher_is_better {
            values.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone()
        } else {
            values.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone()
        };

        let best_value = values[&best_model];
        let mut differences = HashMap::new();
        let mut significant_differences = HashMap::new();

        for (model, value) in &values {
            let diff = if higher_is_better {
                (value - best_value) / best_value * 100.0
            } else {
                (best_value - value) / best_value * 100.0
            };
            differences.insert(model.clone(), diff);
            significant_differences.insert(model.clone(), diff.abs() > 1.0); // 1% threshold
        }

        Ok(MetricComparison {
            values,
            best_model,
            worst_model,
            differences,
            significant_differences,
        })
    }

    fn analyze_architecture_differences(
        &self,
        models: &[&ModelSnapshot],
    ) -> Result<ArchitectureDiff> {
        if models.len() < 2 {
            return Err(anyhow::anyhow!(
                "Need at least 2 models for architecture diff"
            ));
        }

        let base_model = models[0];
        let mut parameter_diff = HashMap::new();
        let mut layer_diff = HashMap::new();
        let mut notable_differences = Vec::new();

        for model in models.iter().skip(1) {
            let param_diff = model.architecture.parameter_count as i64
                - base_model.architecture.parameter_count as i64;
            let layer_diff_val =
                model.architecture.layer_count as i32 - base_model.architecture.layer_count as i32;

            parameter_diff.insert(model.name.clone(), param_diff);
            layer_diff.insert(model.name.clone(), layer_diff_val);

            if param_diff.abs() > 1_000_000 {
                notable_differences.push(format!(
                    "Model '{}' has {} parameter difference",
                    model.name, param_diff
                ));
            }

            if layer_diff_val != 0 {
                notable_differences.push(format!(
                    "Model '{}' has {} layer difference",
                    model.name, layer_diff_val
                ));
            }
        }

        // Calculate similarity score based on architecture features
        let mut similarity_scores = Vec::new();
        for model in models.iter().skip(1) {
            let score = self
                .calculate_architecture_similarity(&base_model.architecture, &model.architecture);
            similarity_scores.push(score);
        }
        let similarity_score =
            similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64;

        Ok(ArchitectureDiff {
            parameter_diff,
            layer_diff,
            similarity_score,
            notable_differences,
        })
    }

    fn calculate_architecture_similarity(
        &self,
        arch1: &ArchitectureInfo,
        arch2: &ArchitectureInfo,
    ) -> f64 {
        let mut similarity = 0.0;
        let mut features = 0;

        // Parameter count similarity
        let param_ratio = (arch1.parameter_count.min(arch2.parameter_count) as f64)
            / (arch1.parameter_count.max(arch2.parameter_count) as f64);
        similarity += param_ratio;
        features += 1;

        // Layer count similarity
        let layer_ratio = (arch1.layer_count.min(arch2.layer_count) as f64)
            / (arch1.layer_count.max(arch2.layer_count) as f64);
        similarity += layer_ratio;
        features += 1;

        // Hidden size similarity (if available)
        let hidden_ratio = (arch1.hidden_size.min(arch2.hidden_size) as f64)
            / (arch1.hidden_size.max(arch2.hidden_size) as f64);
        similarity += hidden_ratio;
        features += 1;

        similarity / features as f64
    }

    fn perform_statistical_analysis(
        &self,
        _models: &[&ModelSnapshot],
    ) -> Result<StatisticalAnalysis> {
        // For now, return placeholder analysis
        // In a real implementation, this would perform proper statistical tests
        Ok(StatisticalAnalysis {
            p_values: HashMap::new(),
            effect_sizes: HashMap::new(),
            confidence_intervals: HashMap::new(),
            significance_summary: HashMap::new(),
        })
    }

    fn generate_comparison_summary(
        &self,
        _models: &[&ModelSnapshot],
        performance: &PerformanceComparison,
        _statistical: &StatisticalAnalysis,
    ) -> Result<ComparisonSummary> {
        let best_model = performance.accuracy_comparison.best_model.clone();

        let mut rankings = HashMap::new();
        rankings.insert(
            "accuracy".to_string(),
            vec![performance.accuracy_comparison.best_model.clone()],
        );
        rankings.insert(
            "latency".to_string(),
            vec![performance.latency_comparison.best_model.clone()],
        );

        let key_findings = vec![
            format!(
                "Best accuracy: {} ({:.2}%)",
                performance.accuracy_comparison.best_model,
                performance.accuracy_comparison.values[&performance.accuracy_comparison.best_model]
                    * 100.0
            ),
            format!(
                "Fastest inference: {} ({:.2}ms)",
                performance.latency_comparison.best_model,
                performance.latency_comparison.values[&performance.latency_comparison.best_model]
            ),
        ];

        let recommendations = vec![
            "Consider the trade-offs between accuracy and latency".to_string(),
            "Monitor memory usage for production deployment".to_string(),
        ];

        Ok(ComparisonSummary {
            best_model,
            rankings,
            key_findings,
            recommendations,
        })
    }

    fn calculate_summary_stats(&self, data: &[f64]) -> SummaryStats {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.variance();
        let std_dev = variance.sqrt();

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let median = sorted_data[sorted_data.len() / 2];
        let q25 = sorted_data[sorted_data.len() / 4];
        let q75 = sorted_data[3 * sorted_data.len() / 4];

        SummaryStats {
            mean,
            std_dev,
            min,
            max,
            median,
            q25,
            q75,
        }
    }

    fn perform_ab_statistical_tests(
        &self,
        model_a: &ABTestMetrics,
        model_b: &ABTestMetrics,
    ) -> Result<HashMap<String, StatisticalTestResult>> {
        let mut results = HashMap::new();

        // Simple t-test for primary metric
        if let (Some(a_data), Some(b_data)) = (
            model_a.metrics.get("primary_metric"),
            model_b.metrics.get("primary_metric"),
        ) {
            let a_mean = a_data.mean();
            let b_mean = b_data.mean();
            let a_var = a_data.variance();
            let b_var = b_data.variance();

            // Simplified t-test calculation
            let pooled_std = ((a_var + b_var) / 2.0).sqrt();
            let standard_error =
                pooled_std * (1.0 / a_data.len() as f64 + 1.0 / b_data.len() as f64).sqrt();
            let t_statistic = (a_mean - b_mean) / standard_error;

            // Simplified p-value (would use proper statistical functions in real implementation)
            let p_value = if t_statistic.abs() > 2.0 { 0.01 } else { 0.1 };

            let effect_size = (a_mean - b_mean) / pooled_std; // Cohen's d
            let margin_of_error = 1.96 * standard_error; // 95% CI

            results.insert(
                "primary_metric".to_string(),
                StatisticalTestResult {
                    test_type: "Welch's t-test".to_string(),
                    statistic: t_statistic,
                    p_value,
                    effect_size,
                    confidence_interval: (
                        a_mean - b_mean - margin_of_error,
                        a_mean - b_mean + margin_of_error,
                    ),
                    is_significant: p_value < 0.05,
                },
            );
        }

        Ok(results)
    }

    fn generate_ab_conclusion(
        &self,
        config: &ABTestConfig,
        _model_a: &ABTestMetrics,
        _model_b: &ABTestMetrics,
        tests: &HashMap<String, StatisticalTestResult>,
    ) -> Result<ABTestConclusion> {
        let primary_test = tests.get("primary_metric");

        let (winner, confidence, practical_significance) = if let Some(test) = primary_test {
            let winner = if test.effect_size > 0.0 {
                Some(config.model_a.clone())
            } else {
                Some(config.model_b.clone())
            };

            let confidence = if test.is_significant { 0.95 } else { 0.5 };
            let practical_significance = test.effect_size.abs() > config.min_effect_size;

            (winner, confidence, practical_significance)
        } else {
            (None, 0.5, false)
        };

        let recommendation = if practical_significance && confidence > 0.9 {
            format!("Recommend deploying {}", winner.as_ref().unwrap())
        } else {
            "Insufficient evidence for a clear recommendation".to_string()
        };

        let summary = format!(
            "A/B test completed with {} confidence",
            if confidence > 0.9 { "high" } else { "low" }
        );

        Ok(ABTestConclusion {
            winner,
            confidence,
            practical_significance,
            recommendation,
            summary,
        })
    }

    fn calculate_performance_delta(
        &self,
        from: &ModelSnapshot,
        to: &ModelSnapshot,
    ) -> Result<PerformanceDelta> {
        Ok(PerformanceDelta {
            accuracy_delta: to.metrics.val_accuracy - from.metrics.val_accuracy,
            loss_delta: to.metrics.val_loss - from.metrics.val_loss,
            latency_delta: to.metrics.inference_latency_ms - from.metrics.inference_latency_ms,
            memory_delta: to.metrics.memory_usage_mb - from.metrics.memory_usage_mb,
            size_delta: to.metrics.model_size_mb - from.metrics.model_size_mb,
            training_time_delta: to.metrics.training_time_s - from.metrics.training_time_s,
            custom_deltas: HashMap::new(),
        })
    }

    fn detect_architecture_changes(
        &self,
        from: &ModelSnapshot,
        to: &ModelSnapshot,
    ) -> Result<Vec<ArchitectureChange>> {
        let mut changes = Vec::new();

        if from.architecture.parameter_count != to.architecture.parameter_count {
            changes.push(ArchitectureChange {
                change_type: "Parameter Count".to_string(),
                description: format!(
                    "Changed from {} to {} parameters",
                    from.architecture.parameter_count, to.architecture.parameter_count
                ),
                impact: "Affects model capacity and memory usage".to_string(),
            });
        }

        if from.architecture.layer_count != to.architecture.layer_count {
            changes.push(ArchitectureChange {
                change_type: "Layer Count".to_string(),
                description: format!(
                    "Changed from {} to {} layers",
                    from.architecture.layer_count, to.architecture.layer_count
                ),
                impact: "Affects model depth and training dynamics".to_string(),
            });
        }

        Ok(changes)
    }

    fn detect_config_changes(
        &self,
        from: &ModelSnapshot,
        to: &ModelSnapshot,
    ) -> Result<Vec<ConfigChange>> {
        let mut changes = Vec::new();

        if from.training_config.learning_rate != to.training_config.learning_rate {
            changes.push(ConfigChange {
                parameter: "learning_rate".to_string(),
                old_value: from.training_config.learning_rate.to_string(),
                new_value: to.training_config.learning_rate.to_string(),
                impact: "Affects training speed and convergence".to_string(),
            });
        }

        if from.training_config.batch_size != to.training_config.batch_size {
            changes.push(ConfigChange {
                parameter: "batch_size".to_string(),
                old_value: from.training_config.batch_size.to_string(),
                new_value: to.training_config.batch_size.to_string(),
                impact: "Affects gradient noise and memory usage".to_string(),
            });
        }

        Ok(changes)
    }

    fn analyze_weight_changes(
        &self,
        from: &ModelSnapshot,
        to: &ModelSnapshot,
    ) -> Result<WeightChangesSummary> {
        // Simplified weight change analysis
        let avg_magnitude = (to.weights_summary.mean - from.weights_summary.mean).abs();
        let max_change = (to.weights_summary.max - from.weights_summary.max).abs();
        let significant_change_ratio = if avg_magnitude > 0.01 { 0.8 } else { 0.2 };

        Ok(WeightChangesSummary {
            avg_magnitude,
            max_change,
            significant_change_ratio,
            layer_changes: HashMap::new(),
        })
    }

    fn classify_regression_severity(
        &self,
        magnitude: f64,
        metric_type: &str,
    ) -> RegressionSeverity {
        match metric_type {
            "accuracy" => {
                if magnitude > 0.1 {
                    RegressionSeverity::Critical
                } else if magnitude > 0.05 {
                    RegressionSeverity::Major
                } else if magnitude > 0.02 {
                    RegressionSeverity::Minor
                } else {
                    RegressionSeverity::Negligible
                }
            },
            "latency" => {
                if magnitude > 50.0 {
                    RegressionSeverity::Critical
                } else if magnitude > 20.0 {
                    RegressionSeverity::Major
                } else if magnitude > 10.0 {
                    RegressionSeverity::Minor
                } else {
                    RegressionSeverity::Negligible
                }
            },
            _ => RegressionSeverity::Minor,
        }
    }

    fn generate_model_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();

        if let Some((best_name, best_model)) = self
            .model_snapshots
            .iter()
            .max_by(|a, b| a.1.metrics.val_accuracy.partial_cmp(&b.1.metrics.val_accuracy).unwrap())
        {
            summary.insert("best_accuracy_model".to_string(), best_name.clone());
            summary.insert(
                "best_accuracy_value".to_string(),
                format!("{:.4}", best_model.metrics.val_accuracy),
            );
        }

        if let Some((fastest_name, fastest_model)) = self.model_snapshots.iter().min_by(|a, b| {
            a.1.metrics
                .inference_latency_ms
                .partial_cmp(&b.1.metrics.inference_latency_ms)
                .unwrap()
        }) {
            summary.insert("fastest_model".to_string(), fastest_name.clone());
            summary.insert(
                "fastest_latency".to_string(),
                format!("{:.2}ms", fastest_model.metrics.inference_latency_ms),
            );
        }

        summary.insert(
            "total_models".to_string(),
            self.model_snapshots.len().to_string(),
        );
        summary
    }
}

/// Comprehensive differential debugging report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialDebuggingReport {
    pub timestamp: DateTime<Utc>,
    pub config: DifferentialDebuggingConfig,
    pub total_models: usize,
    pub comparison_count: usize,
    pub ab_test_count: usize,
    pub version_diff_count: usize,
    pub regression_detection_count: usize,
    pub recent_comparisons: Vec<ModelComparisonResult>,
    pub recent_regressions: Vec<RegressionDetectionResult>,
    pub model_summary: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_differential_debugger_creation() {
        let config = DifferentialDebuggingConfig::default();
        let debugger = DifferentialDebugger::new(config);
        assert_eq!(debugger.model_snapshots.len(), 0);
    }

    #[tokio::test]
    async fn test_model_snapshot_addition() {
        let config = DifferentialDebuggingConfig::default();
        let mut debugger = DifferentialDebugger::new(config);

        let snapshot = create_test_snapshot("test_model");
        debugger.add_model_snapshot(snapshot).unwrap();
        assert_eq!(debugger.model_snapshots.len(), 1);
    }

    #[tokio::test]
    async fn test_model_comparison() {
        let config = DifferentialDebuggingConfig::default();
        let mut debugger = DifferentialDebugger::new(config);

        // Add two test models
        let snapshot1 = create_test_snapshot("model_a");
        let snapshot2 = create_test_snapshot("model_b");

        debugger.add_model_snapshot(snapshot1).unwrap();
        debugger.add_model_snapshot(snapshot2).unwrap();

        let result = debugger
            .compare_models(vec!["model_a".to_string(), "model_b".to_string()])
            .await;
        assert!(result.is_ok());
    }

    fn create_test_snapshot(name: &str) -> ModelSnapshot {
        ModelSnapshot {
            id: Uuid::new_v4(),
            name: name.to_string(),
            timestamp: Utc::now(),
            version: "1.0.0".to_string(),
            commit_hash: Some("abc123".to_string()),
            metrics: ModelMetrics {
                train_accuracy: 0.95,
                val_accuracy: 0.90,
                test_accuracy: Some(0.88),
                train_loss: 0.05,
                val_loss: 0.10,
                test_loss: Some(0.12),
                inference_latency_ms: 50.0,
                memory_usage_mb: 2048.0,
                model_size_mb: 500.0,
                flops: 1_000_000_000,
                training_time_s: 3600.0,
                custom_metrics: HashMap::new(),
            },
            architecture: ArchitectureInfo {
                parameter_count: 175_000_000,
                layer_count: 24,
                depth: 24,
                hidden_size: 1024,
                num_heads: Some(16),
                ff_dim: Some(4096),
                vocab_size: Some(50257),
                max_seq_length: Some(2048),
            },
            training_config: TrainingConfig {
                learning_rate: 1e-4,
                batch_size: 32,
                epochs: 10,
                optimizer: "AdamW".to_string(),
                lr_schedule: Some("cosine".to_string()),
                regularization: HashMap::new(),
            },
            weights_summary: WeightsSummary {
                mean: 0.0,
                std_dev: 0.1,
                min: -0.5,
                max: 0.5,
                percentiles: HashMap::new(),
                zero_count: 1000,
                sparsity: 0.01,
            },
            metadata: HashMap::new(),
        }
    }
}
