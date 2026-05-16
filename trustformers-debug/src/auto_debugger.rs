//! Automated debugging system for common issues and optimization suggestions

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::{
    AnomalyDetectorReport, DashboardMetrics, DebugConfig, GradientDebugReport, ProfilerReport,
};

/// Automated debugging system
#[derive(Debug)]
#[allow(dead_code)]
pub struct AutoDebugger {
    #[allow(dead_code)]
    config: DebugConfig,
    issue_detectors: Vec<Box<dyn IssueDetector>>,
    fix_suggestions: HashMap<IssueType, Vec<FixSuggestion>>,
    optimization_history: Vec<OptimizationAttempt>,
    knowledge_base: KnowledgeBase,
}

/// Common training and model issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueType {
    // Training Issues
    VanishingGradients,
    ExplodingGradients,
    LearningRateProblems,
    OverfittingDetected,
    UnderfittingDetected,
    TrainingStalled,
    LossNotDecreasing,
    UnstableTraining,
    MemoryIssues,

    // Model Architecture Issues
    ModelTooLarge,
    ModelTooSmall,
    InappropriateArchitecture,
    LayerMismatch,
    ActivationProblems,

    // Data Issues
    DataImbalance,
    DataLeakage,
    InsufficientData,
    DataQualityIssues,
    BatchSizeProblems,

    // Performance Issues
    SlowTraining,
    LowGpuUtilization,
    MemoryBottleneck,
    IoBottleneck,
    ComputeBottleneck,

    // Hyperparameter Issues
    LearningRateTooHigh,
    LearningRateTooLow,
    BatchSizeTooLarge,
    BatchSizeTooSmall,
    RegularizationIssues,
}

/// Issue detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub confidence: f64,
    pub description: String,
    pub evidence: Vec<Evidence>,
    pub metrics: HashMap<String, f64>,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub metric_name: String,
    pub observed_value: f64,
    pub expected_range: (f64, f64),
    pub explanation: String,
}

/// Fix suggestion with implementation guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    pub fix_id: String,
    pub fix_type: FixType,
    pub title: String,
    pub description: String,
    pub implementation_steps: Vec<String>,
    pub expected_impact: ExpectedImpact,
    pub priority: FixPriority,
    pub estimated_effort: EstimatedEffort,
    pub prerequisites: Vec<String>,
    pub code_examples: Vec<CodeExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixType {
    HyperparameterAdjustment,
    ArchitectureChange,
    TrainingProcedure,
    DataProcessing,
    OptimizationTechnique,
    EnvironmentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    pub performance_improvement: f64,
    pub training_speed_improvement: f64,
    pub stability_improvement: f64,
    pub memory_usage_change: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EstimatedEffort {
    Trivial, // < 5 minutes
    Easy,    // 5-30 minutes
    Medium,  // 30 minutes - 2 hours
    Hard,    // 2-8 hours
    Complex, // > 8 hours
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub language: String,
    pub code: String,
    pub explanation: String,
}

/// Optimization attempt tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAttempt {
    pub attempt_id: String,
    pub issue_addressed: IssueType,
    pub fix_applied: String,
    pub before_metrics: HashMap<String, f64>,
    pub after_metrics: Option<HashMap<String, f64>>,
    pub success: Option<bool>,
    pub notes: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Knowledge base for common patterns and solutions
#[derive(Debug)]
#[allow(dead_code)]
pub struct KnowledgeBase {
    #[allow(dead_code)]
    issue_patterns: HashMap<IssueType, IssuePattern>,
    hyperparameter_recommendations: HashMap<String, HyperparameterAdvice>,
    architecture_patterns: Vec<ArchitecturePattern>,
    best_practices: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct IssuePattern {
    pub symptoms: Vec<String>,
    pub common_causes: Vec<String>,
    pub diagnostic_metrics: Vec<String>,
    pub typical_solutions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HyperparameterAdvice {
    pub parameter_name: String,
    pub recommended_range: (f64, f64),
    pub tuning_strategy: String,
    pub dependencies: Vec<String>,
    pub common_mistakes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ArchitecturePattern {
    pub pattern_name: String,
    pub use_cases: Vec<String>,
    pub typical_layers: Vec<String>,
    pub hyperparameter_suggestions: HashMap<String, f64>,
    pub performance_characteristics: String,
}

/// Issue detector trait for modular detection
pub trait IssueDetector: std::fmt::Debug {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>>;
    fn get_detector_name(&self) -> &str;
    fn get_supported_issues(&self) -> Vec<IssueType>;
}

/// Context for issue detection
#[derive(Debug)]
pub struct DebugContext<'a> {
    pub profiler_report: Option<&'a ProfilerReport>,
    pub gradient_report: Option<&'a GradientDebugReport>,
    pub anomaly_report: Option<&'a AnomalyDetectorReport>,
    pub recent_metrics: &'a [DashboardMetrics],
    pub training_duration: Duration,
    pub model_info: Option<&'a ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_type: String,
    pub parameter_count: usize,
    pub layer_count: usize,
    pub architecture_details: HashMap<String, String>,
}

impl AutoDebugger {
    /// Create new auto-debugger with default detectors
    pub fn new(config: &DebugConfig) -> Self {
        let mut auto_debugger = Self {
            config: config.clone(),
            issue_detectors: Vec::new(),
            fix_suggestions: HashMap::new(),
            optimization_history: Vec::new(),
            knowledge_base: KnowledgeBase::new(),
        };

        // Register default detectors
        auto_debugger.register_default_detectors();
        auto_debugger.initialize_fix_suggestions();

        auto_debugger
    }

    /// Register all default issue detectors
    fn register_default_detectors(&mut self) {
        self.issue_detectors.push(Box::new(GradientIssueDetector::new()));
        self.issue_detectors.push(Box::new(TrainingIssueDetector::new()));
        self.issue_detectors.push(Box::new(PerformanceIssueDetector::new()));
        self.issue_detectors.push(Box::new(HyperparameterIssueDetector::new()));
        self.issue_detectors.push(Box::new(ArchitectureIssueDetector::new()));
        self.issue_detectors.push(Box::new(DataIssueDetector::new()));
    }

    /// Initialize fix suggestions for common issues
    fn initialize_fix_suggestions(&mut self) {
        // Vanishing gradients fixes
        self.fix_suggestions.insert(
            IssueType::VanishingGradients,
            vec![
                FixSuggestion {
                    fix_id: "vg_001".to_string(),
                    fix_type: FixType::ArchitectureChange,
                    title: "Add Residual Connections".to_string(),
                    description:
                        "Implement skip connections to help gradients flow through deep networks"
                            .to_string(),
                    implementation_steps: vec![
                        "Add residual blocks to your model architecture".to_string(),
                        "Ensure input and output dimensions match for residual connections"
                            .to_string(),
                        "Consider using batch normalization within residual blocks".to_string(),
                    ],
                    expected_impact: ExpectedImpact {
                        performance_improvement: 0.15,
                        training_speed_improvement: 0.05,
                        stability_improvement: 0.25,
                        memory_usage_change: 0.02,
                    },
                    priority: FixPriority::High,
                    estimated_effort: EstimatedEffort::Medium,
                    prerequisites: vec!["Model architecture access".to_string()],
                    code_examples: vec![CodeExample {
                        language: "python".to_string(),
                        code: r#"
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)
"#
                        .to_string(),
                        explanation: "Basic residual block implementation with skip connection"
                            .to_string(),
                    }],
                },
                FixSuggestion {
                    fix_id: "vg_002".to_string(),
                    fix_type: FixType::HyperparameterAdjustment,
                    title: "Adjust Learning Rate".to_string(),
                    description:
                        "Increase learning rate to help gradients propagate more effectively"
                            .to_string(),
                    implementation_steps: vec![
                        "Increase learning rate by 2-5x".to_string(),
                        "Monitor training stability".to_string(),
                        "Consider learning rate scheduling".to_string(),
                    ],
                    expected_impact: ExpectedImpact {
                        performance_improvement: 0.08,
                        training_speed_improvement: 0.10,
                        stability_improvement: -0.05,
                        memory_usage_change: 0.0,
                    },
                    priority: FixPriority::Medium,
                    estimated_effort: EstimatedEffort::Trivial,
                    prerequisites: vec![],
                    code_examples: vec![CodeExample {
                        language: "python".to_string(),
                        code: "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)"
                            .to_string(),
                        explanation: "Increase learning rate to help overcome vanishing gradients"
                            .to_string(),
                    }],
                },
            ],
        );

        // Exploding gradients fixes
        self.fix_suggestions.insert(
            IssueType::ExplodingGradients,
            vec![FixSuggestion {
                fix_id: "eg_001".to_string(),
                fix_type: FixType::TrainingProcedure,
                title: "Apply Gradient Clipping".to_string(),
                description: "Clip gradients to prevent explosion during backpropagation"
                    .to_string(),
                implementation_steps: vec![
                    "Add gradient clipping to your training loop".to_string(),
                    "Start with clip value of 1.0 and adjust based on results".to_string(),
                    "Monitor gradient norms to ensure clipping is effective".to_string(),
                ],
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.10,
                    training_speed_improvement: 0.0,
                    stability_improvement: 0.30,
                    memory_usage_change: 0.0,
                },
                priority: FixPriority::Critical,
                estimated_effort: EstimatedEffort::Easy,
                prerequisites: vec![],
                code_examples: vec![CodeExample {
                    language: "python".to_string(),
                    code: "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
                        .to_string(),
                    explanation: "Clip gradients before optimizer step".to_string(),
                }],
            }],
        );

        // Learning rate issues
        self.fix_suggestions.insert(
            IssueType::LearningRateTooHigh,
            vec![FixSuggestion {
                fix_id: "lr_high_001".to_string(),
                fix_type: FixType::HyperparameterAdjustment,
                title: "Reduce Learning Rate".to_string(),
                description: "Lower the learning rate to improve training stability".to_string(),
                implementation_steps: vec![
                    "Reduce learning rate by 2-10x".to_string(),
                    "Consider learning rate scheduling".to_string(),
                    "Monitor loss convergence".to_string(),
                ],
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.12,
                    training_speed_improvement: -0.05,
                    stability_improvement: 0.25,
                    memory_usage_change: 0.0,
                },
                priority: FixPriority::High,
                estimated_effort: EstimatedEffort::Trivial,
                prerequisites: vec![],
                code_examples: vec![CodeExample {
                    language: "python".to_string(),
                    code: "optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)".to_string(),
                    explanation: "Reduce learning rate for more stable training".to_string(),
                }],
            }],
        );

        // Performance issues
        self.fix_suggestions.insert(
            IssueType::LowGpuUtilization,
            vec![FixSuggestion {
                fix_id: "gpu_001".to_string(),
                fix_type: FixType::HyperparameterAdjustment,
                title: "Increase Batch Size".to_string(),
                description: "Increase batch size to better utilize GPU compute capacity"
                    .to_string(),
                implementation_steps: vec![
                    "Double the current batch size".to_string(),
                    "Monitor memory usage to avoid OOM".to_string(),
                    "Adjust learning rate proportionally".to_string(),
                ],
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.05,
                    training_speed_improvement: 0.30,
                    stability_improvement: 0.0,
                    memory_usage_change: 0.20,
                },
                priority: FixPriority::Medium,
                estimated_effort: EstimatedEffort::Easy,
                prerequisites: vec!["Available GPU memory".to_string()],
                code_examples: vec![CodeExample {
                    language: "python".to_string(),
                    code: "train_loader = DataLoader(dataset, batch_size=64, shuffle=True)"
                        .to_string(),
                    explanation: "Increase batch size to improve GPU utilization".to_string(),
                }],
            }],
        );
    }

    /// Analyze debug context and detect issues
    pub fn analyze_issues(&self, context: &DebugContext) -> Result<AutoDebugReport> {
        let mut all_issues = Vec::new();

        // Run all issue detectors
        for detector in &self.issue_detectors {
            match detector.detect_issues(context) {
                Ok(mut issues) => all_issues.append(&mut issues),
                Err(e) => {
                    tracing::warn!(
                        "Issue detector '{}' failed: {}",
                        detector.get_detector_name(),
                        e
                    );
                },
            }
        }

        // Sort issues by severity and confidence
        all_issues.sort_by(|a, b| {
            let severity_order = |s: &IssueSeverity| match s {
                IssueSeverity::Critical => 0,
                IssueSeverity::High => 1,
                IssueSeverity::Medium => 2,
                IssueSeverity::Low => 3,
                IssueSeverity::Info => 4,
            };

            let severity_cmp = severity_order(&a.severity).cmp(&severity_order(&b.severity));
            if severity_cmp == std::cmp::Ordering::Equal {
                b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                severity_cmp
            }
        });

        // Generate fix recommendations
        let fix_recommendations = self.generate_fix_recommendations(&all_issues);

        // Generate hyperparameter recommendations
        let hyperparameter_recommendations = self.generate_hyperparameter_recommendations(context);

        // Generate architecture suggestions
        let architecture_suggestions = self.generate_architecture_suggestions(context);

        // Generate training recipe optimization
        let training_recipe = self.generate_training_recipe_optimization(context);

        Ok(AutoDebugReport {
            detected_issues: all_issues,
            fix_recommendations: fix_recommendations.clone(),
            hyperparameter_recommendations,
            architecture_suggestions,
            training_recipe,
            analysis_summary: self.generate_analysis_summary(&fix_recommendations),
            confidence_score: self.calculate_overall_confidence(&fix_recommendations),
        })
    }

    /// Generate fix recommendations for detected issues
    fn generate_fix_recommendations(&self, issues: &[DetectedIssue]) -> Vec<FixRecommendation> {
        let mut recommendations = Vec::new();

        for issue in issues {
            if let Some(suggestions) = self.fix_suggestions.get(&issue.issue_type) {
                for suggestion in suggestions {
                    recommendations.push(FixRecommendation {
                        issue: issue.clone(),
                        fix_suggestion: suggestion.clone(),
                        confidence: issue.confidence * 0.9, // Slightly reduce confidence
                        urgency: self.calculate_urgency(issue),
                    });
                }
            }
        }

        // Sort by urgency and confidence
        recommendations.sort_by(|a, b| {
            let urgency_cmp =
                b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal);
            if urgency_cmp == std::cmp::Ordering::Equal {
                b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                urgency_cmp
            }
        });

        recommendations
    }

    fn calculate_urgency(&self, issue: &DetectedIssue) -> f64 {
        let severity_multiplier = match issue.severity {
            IssueSeverity::Critical => 1.0,
            IssueSeverity::High => 0.8,
            IssueSeverity::Medium => 0.6,
            IssueSeverity::Low => 0.4,
            IssueSeverity::Info => 0.2,
        };

        issue.confidence * severity_multiplier
    }

    /// Generate hyperparameter recommendations
    fn generate_hyperparameter_recommendations(
        &self,
        context: &DebugContext,
    ) -> Vec<HyperparameterRecommendation> {
        let mut recommendations = Vec::new();

        // Learning rate recommendations
        if let Some(metrics) = context.recent_metrics.last() {
            if let Some(loss) = metrics.loss {
                if loss > 1.0 {
                    recommendations.push(HyperparameterRecommendation {
                        parameter: "learning_rate".to_string(),
                        current_value: None,
                        recommended_value: 0.001,
                        reason: "High loss suggests learning rate might be too low".to_string(),
                        confidence: 0.7,
                    });
                }
            }
        }

        // Batch size recommendations based on GPU utilization
        if let Some(_profiler_report) = context.profiler_report {
            // Simplified logic - in practice would analyze detailed metrics
            recommendations.push(HyperparameterRecommendation {
                parameter: "batch_size".to_string(),
                current_value: None,
                recommended_value: 32.0,
                reason: "Optimize batch size for better GPU utilization".to_string(),
                confidence: 0.6,
            });
        }

        recommendations
    }

    /// Generate architecture suggestions
    fn generate_architecture_suggestions(
        &self,
        context: &DebugContext,
    ) -> Vec<ArchitectureSuggestion> {
        let mut suggestions = Vec::new();

        // Analyze model size vs performance
        if let Some(model_info) = context.model_info {
            if model_info.parameter_count > 100_000_000 {
                suggestions.push(ArchitectureSuggestion {
                    suggestion_type: "model_compression".to_string(),
                    title: "Consider Model Compression".to_string(),
                    description: "Large model may benefit from pruning or distillation".to_string(),
                    impact_assessment: "Reduce memory usage by 20-50% with minimal accuracy loss"
                        .to_string(),
                    implementation_difficulty: "Medium".to_string(),
                });
            }

            if model_info.layer_count > 50 {
                suggestions.push(ArchitectureSuggestion {
                    suggestion_type: "depth_optimization".to_string(),
                    title: "Optimize Network Depth".to_string(),
                    description: "Very deep network may suffer from gradient flow issues"
                        .to_string(),
                    impact_assessment: "Improve training stability and convergence speed"
                        .to_string(),
                    implementation_difficulty: "High".to_string(),
                });
            }
        }

        suggestions
    }

    /// Generate training recipe optimization
    fn generate_training_recipe_optimization(
        &self,
        context: &DebugContext,
    ) -> TrainingRecipeOptimization {
        let mut optimizations = Vec::new();

        // Analyze training duration and suggest optimizations
        if context.training_duration > Duration::from_secs(3600) {
            optimizations
                .push("Consider learning rate scheduling to speed up convergence".to_string());
            optimizations.push("Implement early stopping to avoid overtraining".to_string());
        }

        // Analyze recent metrics for training recipe suggestions
        if context.recent_metrics.len() > 10 {
            let recent_losses: Vec<f64> =
                context.recent_metrics.iter().rev().take(10).filter_map(|m| m.loss).collect();

            if recent_losses.len() >= 5 {
                let variance = self.calculate_variance(&recent_losses);
                if variance > 0.1 {
                    optimizations.push(
                        "Training loss is unstable - consider reducing learning rate".to_string(),
                    );
                }
            }
        }

        TrainingRecipeOptimization {
            recommended_optimizations: optimizations,
            training_schedule: TrainingSchedule {
                warmup_steps: 1000,
                learning_rate_schedule: "cosine_annealing".to_string(),
                batch_size_schedule: "constant".to_string(),
                early_stopping: true,
                checkpoint_frequency: 1000,
            },
            data_strategy: DataStrategy {
                data_augmentation: vec!["horizontal_flip".to_string(), "random_crop".to_string()],
                sampling_strategy: "balanced".to_string(),
                preprocessing_optimizations: vec![
                    "normalization".to_string(),
                    "standardization".to_string(),
                ],
            },
        }
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance
    }

    fn generate_analysis_summary(&self, recommendations: &[FixRecommendation]) -> String {
        let critical_count = recommendations
            .iter()
            .filter(|r| matches!(r.issue.severity, IssueSeverity::Critical))
            .count();

        let high_count = recommendations
            .iter()
            .filter(|r| matches!(r.issue.severity, IssueSeverity::High))
            .count();

        if critical_count > 0 {
            format!("Found {} critical issues requiring immediate attention. {} high-priority issues also detected.",
                   critical_count, high_count)
        } else if high_count > 0 {
            format!(
                "Found {} high-priority issues that should be addressed soon.",
                high_count
            )
        } else if !recommendations.is_empty() {
            "Found some optimization opportunities to improve training performance.".to_string()
        } else {
            "No significant issues detected. Training appears to be proceeding normally."
                .to_string()
        }
    }

    fn calculate_overall_confidence(&self, recommendations: &[FixRecommendation]) -> f64 {
        if recommendations.is_empty() {
            return 1.0;
        }

        let sum_confidence: f64 = recommendations.iter().map(|r| r.confidence).sum();
        sum_confidence / recommendations.len() as f64
    }

    /// Record optimization attempt for learning
    pub fn record_optimization_attempt(&mut self, attempt: OptimizationAttempt) {
        self.optimization_history.push(attempt);

        // Keep only recent attempts to prevent unbounded growth
        if self.optimization_history.len() > 1000 {
            self.optimization_history.drain(0..500);
        }
    }

    /// Get optimization history for analysis
    pub fn get_optimization_history(&self) -> &[OptimizationAttempt] {
        &self.optimization_history
    }
}

// Issue detector implementations

#[derive(Debug)]
struct GradientIssueDetector;

impl GradientIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for GradientIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        let mut issues = Vec::new();

        if let Some(gradient_report) = context.gradient_report {
            // Check for vanishing gradients
            if gradient_report.has_vanishing_gradients() {
                issues.push(DetectedIssue {
                    issue_type: IssueType::VanishingGradients,
                    severity: IssueSeverity::High,
                    confidence: 0.9,
                    description: "Vanishing gradients detected in multiple layers".to_string(),
                    evidence: vec![Evidence {
                        metric_name: "gradient_norm".to_string(),
                        observed_value: 0.001,
                        expected_range: (0.01, 1.0),
                        explanation: "Gradient norms are significantly below normal range"
                            .to_string(),
                    }],
                    metrics: HashMap::new(),
                    detected_at: chrono::Utc::now(),
                });
            }

            // Check for exploding gradients
            if gradient_report.has_exploding_gradients() {
                issues.push(DetectedIssue {
                    issue_type: IssueType::ExplodingGradients,
                    severity: IssueSeverity::Critical,
                    confidence: 0.95,
                    description: "Exploding gradients detected - training instability likely"
                        .to_string(),
                    evidence: vec![Evidence {
                        metric_name: "gradient_norm".to_string(),
                        observed_value: 100.0,
                        expected_range: (0.01, 10.0),
                        explanation: "Gradient norms are extremely high".to_string(),
                    }],
                    metrics: HashMap::new(),
                    detected_at: chrono::Utc::now(),
                });
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "GradientIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![IssueType::VanishingGradients, IssueType::ExplodingGradients]
    }
}

#[derive(Debug)]
struct TrainingIssueDetector;

impl TrainingIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for TrainingIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        let mut issues = Vec::new();

        // Analyze recent training metrics
        if context.recent_metrics.len() >= 10 {
            let recent_losses: Vec<f64> =
                context.recent_metrics.iter().rev().take(10).filter_map(|m| m.loss).collect();

            if recent_losses.len() >= 5 {
                // Check for stalled training
                let first_half_avg = recent_losses[..recent_losses.len() / 2].iter().sum::<f64>()
                    / (recent_losses.len() / 2) as f64;
                let second_half_avg = recent_losses[recent_losses.len() / 2..].iter().sum::<f64>()
                    / (recent_losses.len() - recent_losses.len() / 2) as f64;

                if (first_half_avg - second_half_avg).abs() / first_half_avg < 0.01 {
                    issues.push(DetectedIssue {
                        issue_type: IssueType::TrainingStalled,
                        severity: IssueSeverity::Medium,
                        confidence: 0.8,
                        description: "Training appears to have stalled - loss not decreasing"
                            .to_string(),
                        evidence: vec![Evidence {
                            metric_name: "loss_change".to_string(),
                            observed_value: (first_half_avg - second_half_avg).abs()
                                / first_half_avg,
                            expected_range: (0.05, 1.0),
                            explanation: "Loss change is below expected threshold".to_string(),
                        }],
                        metrics: HashMap::new(),
                        detected_at: chrono::Utc::now(),
                    });
                }
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "TrainingIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![
            IssueType::TrainingStalled,
            IssueType::LossNotDecreasing,
            IssueType::UnstableTraining,
        ]
    }
}

#[derive(Debug)]
struct PerformanceIssueDetector;

impl PerformanceIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for PerformanceIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        let mut issues = Vec::new();

        // Check GPU utilization
        if let Some(metrics) = context.recent_metrics.last() {
            if let Some(gpu_util) = metrics.gpu_utilization {
                if gpu_util < 0.5 {
                    issues.push(DetectedIssue {
                        issue_type: IssueType::LowGpuUtilization,
                        severity: IssueSeverity::Medium,
                        confidence: 0.8,
                        description:
                            "Low GPU utilization detected - compute resources underutilized"
                                .to_string(),
                        evidence: vec![Evidence {
                            metric_name: "gpu_utilization".to_string(),
                            observed_value: gpu_util,
                            expected_range: (0.7, 1.0),
                            explanation: "GPU utilization is below optimal range".to_string(),
                        }],
                        metrics: HashMap::new(),
                        detected_at: chrono::Utc::now(),
                    });
                }
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "PerformanceIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![
            IssueType::LowGpuUtilization,
            IssueType::SlowTraining,
            IssueType::MemoryBottleneck,
        ]
    }
}

#[derive(Debug)]
struct HyperparameterIssueDetector;

impl HyperparameterIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for HyperparameterIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        let mut issues = Vec::new();

        if let Some(metrics) = context.recent_metrics.last() {
            // Check learning rate issues
            if let Some(lr) = metrics.learning_rate {
                if lr > 0.1 {
                    issues.push(DetectedIssue {
                        issue_type: IssueType::LearningRateTooHigh,
                        severity: IssueSeverity::High,
                        confidence: 0.7,
                        description:
                            "Learning rate appears too high - may cause training instability"
                                .to_string(),
                        evidence: vec![Evidence {
                            metric_name: "learning_rate".to_string(),
                            observed_value: lr,
                            expected_range: (0.0001, 0.01),
                            explanation: "Learning rate is above typical range".to_string(),
                        }],
                        metrics: HashMap::new(),
                        detected_at: chrono::Utc::now(),
                    });
                } else if lr < 0.00001 {
                    issues.push(DetectedIssue {
                        issue_type: IssueType::LearningRateTooLow,
                        severity: IssueSeverity::Medium,
                        confidence: 0.6,
                        description: "Learning rate might be too low - training could be slow"
                            .to_string(),
                        evidence: vec![Evidence {
                            metric_name: "learning_rate".to_string(),
                            observed_value: lr,
                            expected_range: (0.0001, 0.01),
                            explanation: "Learning rate is below typical range".to_string(),
                        }],
                        metrics: HashMap::new(),
                        detected_at: chrono::Utc::now(),
                    });
                }
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "HyperparameterIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![
            IssueType::LearningRateTooHigh,
            IssueType::LearningRateTooLow,
        ]
    }
}

#[derive(Debug)]
struct ArchitectureIssueDetector;

impl ArchitectureIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for ArchitectureIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        let mut issues = Vec::new();

        if let Some(model_info) = context.model_info {
            // Check model size
            if model_info.parameter_count > 1_000_000_000 {
                issues.push(DetectedIssue {
                    issue_type: IssueType::ModelTooLarge,
                    severity: IssueSeverity::Medium,
                    confidence: 0.6,
                    description:
                        "Model has very large number of parameters - consider optimization"
                            .to_string(),
                    evidence: vec![Evidence {
                        metric_name: "parameter_count".to_string(),
                        observed_value: model_info.parameter_count as f64,
                        expected_range: (1_000_000.0, 100_000_000.0),
                        explanation: "Parameter count is extremely high".to_string(),
                    }],
                    metrics: HashMap::new(),
                    detected_at: chrono::Utc::now(),
                });
            }

            if model_info.layer_count > 100 {
                issues.push(DetectedIssue {
                    issue_type: IssueType::InappropriateArchitecture,
                    severity: IssueSeverity::Low,
                    confidence: 0.5,
                    description: "Very deep model - may have gradient flow issues".to_string(),
                    evidence: vec![Evidence {
                        metric_name: "layer_count".to_string(),
                        observed_value: model_info.layer_count as f64,
                        expected_range: (10.0, 50.0),
                        explanation: "Layer count is very high".to_string(),
                    }],
                    metrics: HashMap::new(),
                    detected_at: chrono::Utc::now(),
                });
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "ArchitectureIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![
            IssueType::ModelTooLarge,
            IssueType::InappropriateArchitecture,
        ]
    }
}

#[derive(Debug)]
struct DataIssueDetector;

impl DataIssueDetector {
    fn new() -> Self {
        Self
    }
}

impl IssueDetector for DataIssueDetector {
    fn detect_issues(&self, context: &DebugContext) -> Result<Vec<DetectedIssue>> {
        // Detect three classes of data-related issues from dashboard metrics:
        //
        //  - BatchSizeProblems  : sustained low GPU utilisation paired with low
        //                         tokens/sec is a strong signal that the batch is
        //                         too small to saturate the device.
        //  - DataImbalance      : accuracy pinned at a near-trivial value (high
        //                         floor or low ceiling) while loss continues to
        //                         change is the canonical signature of a model
        //                         collapsing onto a majority class.
        //  - InsufficientData   : loss decreases but accuracy oscillates wildly,
        //                         indicating the model is memorising rather than
        //                         generalising — a classic small-dataset failure
        //                         mode.
        //
        // Heuristics use thresholds tuned for a "typical" supervised-learning
        // setup; they are intentionally conservative so we do not produce false
        // positives on small recent_metrics windows.
        let mut issues = Vec::new();

        const MIN_WINDOW: usize = 5;
        if context.recent_metrics.len() < MIN_WINDOW {
            return Ok(issues);
        }

        // Sample the freshest MIN_WINDOW metrics for analysis.
        let window: Vec<&DashboardMetrics> =
            context.recent_metrics.iter().rev().take(MIN_WINDOW * 2).collect();

        // --- BatchSizeProblems ---------------------------------------------
        let gpu_samples: Vec<f64> = window.iter().filter_map(|m| m.gpu_utilization).collect();
        let tps_samples: Vec<f64> = window.iter().filter_map(|m| m.tokens_per_second).collect();
        if gpu_samples.len() >= MIN_WINDOW && tps_samples.len() >= MIN_WINDOW {
            let gpu_mean = gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64;
            let tps_mean = tps_samples.iter().sum::<f64>() / tps_samples.len() as f64;
            // Low GPU and low throughput together strongly suggest the batch is
            // starving the device. We use 0.5 (50% utilisation) and 100 tok/s as
            // canonical thresholds, matching the project's other detectors.
            if gpu_mean < 0.5 && tps_mean < 100.0 {
                let mut metrics = HashMap::new();
                metrics.insert("avg_gpu_utilization".to_string(), gpu_mean);
                metrics.insert("avg_tokens_per_second".to_string(), tps_mean);
                issues.push(DetectedIssue {
                    issue_type: IssueType::BatchSizeProblems,
                    severity: IssueSeverity::Medium,
                    confidence: 0.7,
                    description:
                        "Sustained low GPU utilisation and throughput suggest batch size may be \
                         too small to saturate the device"
                            .to_string(),
                    evidence: vec![
                        Evidence {
                            metric_name: "gpu_utilization".to_string(),
                            observed_value: gpu_mean,
                            expected_range: (0.7, 1.0),
                            explanation:
                                "Average GPU utilisation is below the typical training range"
                                    .to_string(),
                        },
                        Evidence {
                            metric_name: "tokens_per_second".to_string(),
                            observed_value: tps_mean,
                            expected_range: (100.0, f64::INFINITY),
                            explanation: "Throughput is below the typical training floor"
                                .to_string(),
                        },
                    ],
                    metrics,
                    detected_at: chrono::Utc::now(),
                });
            }
        }

        // --- DataImbalance / InsufficientData ------------------------------
        let acc_samples: Vec<f64> = window.iter().filter_map(|m| m.accuracy).collect();
        let loss_samples: Vec<f64> = window.iter().filter_map(|m| m.loss).collect();

        if acc_samples.len() >= MIN_WINDOW && loss_samples.len() >= MIN_WINDOW {
            let acc_mean = acc_samples.iter().sum::<f64>() / acc_samples.len() as f64;
            let acc_var = acc_samples
                .iter()
                .map(|a| {
                    let d = a - acc_mean;
                    d * d
                })
                .sum::<f64>()
                / acc_samples.len() as f64;
            let acc_stddev = acc_var.sqrt();

            // `window` (and therefore `loss_samples`) is ordered newest-first.
            // Compare the older half (end of the slice) to the newer half
            // (start of the slice) to decide whether loss is decreasing.
            let half = loss_samples.len() / 2;
            let newer_half = &loss_samples[..half];
            let older_half = &loss_samples[loss_samples.len() - half..];
            let newer_avg = if newer_half.is_empty() {
                0.0
            } else {
                newer_half.iter().sum::<f64>() / newer_half.len() as f64
            };
            let older_avg = if older_half.is_empty() {
                0.0
            } else {
                older_half.iter().sum::<f64>() / older_half.len() as f64
            };
            // Positive => loss decreasing over time.
            let loss_relative_change = if older_avg.abs() > f64::EPSILON {
                (older_avg - newer_avg) / older_avg.abs()
            } else {
                0.0
            };

            // DataImbalance: accuracy is pinned (very low variance) at an
            // extreme value (either trivially low or near-perfect) while the
            // loss continues to move meaningfully. Models collapsing onto the
            // majority class show exactly this signature.
            let acc_pinned_extreme =
                acc_stddev < 0.01 && !(0.2..=0.95).contains(&acc_mean);
            let loss_changing = loss_relative_change.abs() > 0.05;
            if acc_pinned_extreme && loss_changing {
                let mut metrics = HashMap::new();
                metrics.insert("accuracy_mean".to_string(), acc_mean);
                metrics.insert("accuracy_stddev".to_string(), acc_stddev);
                metrics.insert("loss_relative_change".to_string(), loss_relative_change);
                issues.push(DetectedIssue {
                    issue_type: IssueType::DataImbalance,
                    severity: IssueSeverity::High,
                    confidence: 0.75,
                    description:
                        "Accuracy is pinned at an extreme value while loss continues to change \
                         — model may be collapsing onto a majority class"
                            .to_string(),
                    evidence: vec![Evidence {
                        metric_name: "accuracy_stddev".to_string(),
                        observed_value: acc_stddev,
                        expected_range: (0.01, 0.5),
                        explanation:
                            "Accuracy variance is far below the range expected during healthy \
                             training"
                                .to_string(),
                    }],
                    metrics,
                    detected_at: chrono::Utc::now(),
                });
            }

            // InsufficientData: loss is steadily decreasing (model fitting the
            // training set) but accuracy is highly volatile, suggesting the
            // model is memorising rather than generalising — a classic failure
            // mode of training on too little data.
            if loss_relative_change > 0.10 && acc_stddev > 0.15 {
                let mut metrics = HashMap::new();
                metrics.insert("accuracy_stddev".to_string(), acc_stddev);
                metrics.insert("loss_relative_change".to_string(), loss_relative_change);
                issues.push(DetectedIssue {
                    issue_type: IssueType::InsufficientData,
                    severity: IssueSeverity::Medium,
                    confidence: 0.6,
                    description:
                        "Loss is decreasing but accuracy fluctuates wildly — the dataset may be \
                         too small, leading to memorisation rather than generalisation"
                            .to_string(),
                    evidence: vec![Evidence {
                        metric_name: "accuracy_stddev".to_string(),
                        observed_value: acc_stddev,
                        expected_range: (0.0, 0.10),
                        explanation:
                            "Accuracy variance is well above what is expected when the model \
                             is generalising"
                                .to_string(),
                    }],
                    metrics,
                    detected_at: chrono::Utc::now(),
                });
            }
        }

        Ok(issues)
    }

    fn get_detector_name(&self) -> &str {
        "DataIssueDetector"
    }

    fn get_supported_issues(&self) -> Vec<IssueType> {
        vec![
            IssueType::DataImbalance,
            IssueType::BatchSizeProblems,
            IssueType::InsufficientData,
        ]
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            issue_patterns: HashMap::new(),
            hyperparameter_recommendations: HashMap::new(),
            architecture_patterns: Vec::new(),
            best_practices: HashMap::new(),
        }
    }
}

// Report structures

#[derive(Debug, Serialize, Deserialize)]
pub struct AutoDebugReport {
    pub detected_issues: Vec<DetectedIssue>,
    pub fix_recommendations: Vec<FixRecommendation>,
    pub hyperparameter_recommendations: Vec<HyperparameterRecommendation>,
    pub architecture_suggestions: Vec<ArchitectureSuggestion>,
    pub training_recipe: TrainingRecipeOptimization,
    pub analysis_summary: String,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixRecommendation {
    pub issue: DetectedIssue,
    pub fix_suggestion: FixSuggestion,
    pub confidence: f64,
    pub urgency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterRecommendation {
    pub parameter: String,
    pub current_value: Option<f64>,
    pub recommended_value: f64,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSuggestion {
    pub suggestion_type: String,
    pub title: String,
    pub description: String,
    pub impact_assessment: String,
    pub implementation_difficulty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecipeOptimization {
    pub recommended_optimizations: Vec<String>,
    pub training_schedule: TrainingSchedule,
    pub data_strategy: DataStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSchedule {
    pub warmup_steps: u32,
    pub learning_rate_schedule: String,
    pub batch_size_schedule: String,
    pub early_stopping: bool,
    pub checkpoint_frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStrategy {
    pub data_augmentation: Vec<String>,
    pub sampling_strategy: String,
    pub preprocessing_optimizations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DebugConfig {
        DebugConfig::default()
    }

    #[test]
    fn test_knowledge_base_new() {
        let kb = KnowledgeBase::new();
        assert!(kb.issue_patterns.is_empty());
        assert!(kb.hyperparameter_recommendations.is_empty());
        assert!(kb.architecture_patterns.is_empty());
        assert!(kb.best_practices.is_empty());
    }

    #[test]
    fn test_knowledge_base_default() {
        let kb = KnowledgeBase::default();
        assert!(kb.issue_patterns.is_empty());
    }

    #[test]
    fn test_auto_debugger_new() {
        let config = make_config();
        let debugger = AutoDebugger::new(&config);
        assert!(!debugger.issue_detectors.is_empty());
        assert!(!debugger.fix_suggestions.is_empty());
        assert!(debugger.optimization_history.is_empty());
    }

    #[test]
    fn test_auto_debugger_has_default_detectors() {
        let config = make_config();
        let debugger = AutoDebugger::new(&config);
        assert_eq!(debugger.issue_detectors.len(), 6);
    }

    #[test]
    fn test_auto_debugger_has_fix_suggestions() {
        let config = make_config();
        let debugger = AutoDebugger::new(&config);
        assert!(debugger.fix_suggestions.contains_key(&IssueType::VanishingGradients));
        assert!(debugger.fix_suggestions.contains_key(&IssueType::ExplodingGradients));
    }

    #[test]
    fn test_gradient_issue_detector_name() {
        let detector = GradientIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "GradientIssueDetector");
    }

    #[test]
    fn test_gradient_issue_detector_supported_issues() {
        let detector = GradientIssueDetector::new();
        let issues = detector.get_supported_issues();
        assert!(issues.contains(&IssueType::VanishingGradients));
        assert!(issues.contains(&IssueType::ExplodingGradients));
    }

    #[test]
    fn test_training_issue_detector_name() {
        let detector = TrainingIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "TrainingIssueDetector");
    }

    #[test]
    fn test_training_issue_detector_supported_issues() {
        let detector = TrainingIssueDetector::new();
        let issues = detector.get_supported_issues();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_performance_issue_detector_name() {
        let detector = PerformanceIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "PerformanceIssueDetector");
    }

    #[test]
    fn test_hyperparameter_issue_detector_name() {
        let detector = HyperparameterIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "HyperparameterIssueDetector");
    }

    #[test]
    fn test_architecture_issue_detector_name() {
        let detector = ArchitectureIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "ArchitectureIssueDetector");
    }

    #[test]
    fn test_data_issue_detector_name() {
        let detector = DataIssueDetector::new();
        assert_eq!(detector.get_detector_name(), "DataIssueDetector");
    }

    #[test]
    fn test_issue_type_equality() {
        assert_eq!(IssueType::VanishingGradients, IssueType::VanishingGradients);
        assert_ne!(IssueType::VanishingGradients, IssueType::ExplodingGradients);
    }

    #[test]
    fn test_issue_type_hash_compatible() {
        let mut map = HashMap::new();
        map.insert(IssueType::OverfittingDetected, "fix");
        assert!(map.contains_key(&IssueType::OverfittingDetected));
        assert!(!map.contains_key(&IssueType::UnderfittingDetected));
    }

    #[test]
    fn test_evidence_construction() {
        let evidence = Evidence {
            metric_name: "gradient_norm".to_string(),
            observed_value: 0.001,
            expected_range: (0.01, 1.0),
            explanation: "Gradient norm too low".to_string(),
        };
        assert_eq!(evidence.metric_name, "gradient_norm");
        assert!(evidence.observed_value < evidence.expected_range.0);
    }

    #[test]
    fn test_expected_impact_fields() {
        let impact = ExpectedImpact {
            performance_improvement: 0.15,
            training_speed_improvement: 0.05,
            stability_improvement: 0.25,
            memory_usage_change: 0.02,
        };
        assert!(impact.performance_improvement > 0.0);
        assert!(impact.stability_improvement > impact.performance_improvement);
    }

    #[test]
    fn test_model_info_construction() {
        let info = ModelInfo {
            model_type: "transformer".to_string(),
            parameter_count: 1_000_000,
            layer_count: 12,
            architecture_details: HashMap::new(),
        };
        assert_eq!(info.model_type, "transformer");
        assert_eq!(info.parameter_count, 1_000_000);
    }

    #[test]
    fn test_issue_pattern_construction() {
        let pattern = IssuePattern {
            symptoms: vec!["low gradient norm".to_string()],
            common_causes: vec!["deep network".to_string()],
            diagnostic_metrics: vec!["gradient_norm".to_string()],
            typical_solutions: vec!["add skip connections".to_string()],
        };
        assert_eq!(pattern.symptoms.len(), 1);
        assert_eq!(pattern.common_causes.len(), 1);
    }

    #[test]
    fn test_hyperparameter_advice_construction() {
        let advice = HyperparameterAdvice {
            parameter_name: "learning_rate".to_string(),
            recommended_range: (1e-5, 1e-2),
            tuning_strategy: "grid_search".to_string(),
            dependencies: vec!["batch_size".to_string()],
            common_mistakes: vec!["too high initial lr".to_string()],
        };
        assert!(advice.recommended_range.0 < advice.recommended_range.1);
    }

    fn make_metric(
        loss: Option<f64>,
        accuracy: Option<f64>,
        gpu: Option<f64>,
        tps: Option<f64>,
    ) -> DashboardMetrics {
        DashboardMetrics {
            timestamp: std::time::SystemTime::now(),
            loss,
            accuracy,
            learning_rate: Some(1e-3),
            memory_usage_mb: 1024.0,
            gpu_utilization: gpu,
            tokens_per_second: tps,
            gradient_norm: Some(0.5),
            epoch: Some(0),
            step: Some(0),
        }
    }

    #[test]
    fn test_data_issue_detector_returns_empty_with_no_metrics() {
        let detector = DataIssueDetector::new();
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &[],
            training_duration: Duration::from_secs(60),
            model_info: None,
        };
        let issues = detector.detect_issues(&context).expect("detect_issues should succeed");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_data_issue_detector_flags_batch_size_problem() {
        let detector = DataIssueDetector::new();
        // Simulate a long stretch of low GPU utilisation and low throughput.
        let metrics: Vec<DashboardMetrics> = (0..10)
            .map(|i| make_metric(Some(2.0 - i as f64 * 0.01), Some(0.6), Some(0.2), Some(50.0)))
            .collect();
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(600),
            model_info: None,
        };
        let issues = detector.detect_issues(&context).expect("detect_issues should succeed");
        assert!(
            issues.iter().any(|i| i.issue_type == IssueType::BatchSizeProblems),
            "expected BatchSizeProblems to be flagged, got: {:?}",
            issues.iter().map(|i| &i.issue_type).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_data_issue_detector_flags_data_imbalance_when_accuracy_pinned() {
        let detector = DataIssueDetector::new();
        // Accuracy pinned at ~0.97 with virtually no variance, while loss
        // continues to fall: a classic majority-class collapse.
        let metrics: Vec<DashboardMetrics> = (0..10)
            .map(|i| {
                make_metric(Some(2.0 - i as f64 * 0.10), Some(0.97), Some(0.85), Some(500.0))
            })
            .collect();
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(600),
            model_info: None,
        };
        let issues = detector.detect_issues(&context).expect("detect_issues should succeed");
        assert!(
            issues.iter().any(|i| i.issue_type == IssueType::DataImbalance),
            "expected DataImbalance to be flagged, got: {:?}",
            issues.iter().map(|i| &i.issue_type).collect::<Vec<_>>()
        );
    }
}
