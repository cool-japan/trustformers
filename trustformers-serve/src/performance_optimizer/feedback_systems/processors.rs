//! Enhanced Feedback Processors
//!
//! This module provides specialized feedback processors for different types of
//! performance metrics including throughput, latency, and resource utilization.
//! Each processor includes advanced capabilities like quality assessment, anomaly
//! detection, trend analysis, and enhanced recommendation generation.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use super::types::*;
use crate::performance_optimizer::types::{
    ActionType, FeedbackSource, FeedbackType, PerformanceFeedback, RecommendedAction,
};

// =============================================================================
// ENHANCED THROUGHPUT PROCESSOR
// =============================================================================

/// Comprehensive feedback processor for throughput optimization
pub struct EnhancedThroughputProcessor {
    /// Historical throughput data
    historical_data: Arc<Mutex<VecDeque<f64>>>,
    /// Quality assessment thresholds
    quality_thresholds: QualityThresholds,
    /// Trend analysis enabled
    trend_analysis: bool,
    /// Anomaly detection enabled
    anomaly_detection: bool,
}

/// Quality assessment thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum reliability threshold
    pub min_reliability: f32,
    /// Minimum relevance threshold
    pub min_relevance: f32,
    /// Maximum age for timeliness
    pub max_age: Duration,
    /// Minimum completeness threshold
    pub min_completeness: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_reliability: 0.7,
            min_relevance: 0.8,
            max_age: Duration::from_secs(300), // 5 minutes
            min_completeness: 0.9,
        }
    }
}

impl Default for EnhancedThroughputProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedThroughputProcessor {
    /// Create new enhanced throughput processor
    pub fn new() -> Self {
        Self {
            historical_data: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            quality_thresholds: QualityThresholds::default(),
            trend_analysis: true,
            anomaly_detection: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(_config: super::types::ThroughputProcessorConfig) -> Self {
        // For now, use default implementation
        // Can be enhanced later to use config values
        Self::new()
    }

    /// Assess feedback quality
    pub fn assess_quality(&self, feedback: &PerformanceFeedback) -> Result<FeedbackQualityMetrics> {
        let age = Utc::now().signed_duration_since(feedback.timestamp);
        let timeliness = if age <= chrono::Duration::from_std(self.quality_thresholds.max_age)? {
            1.0 - (age.num_seconds() as f32 / self.quality_thresholds.max_age.as_secs() as f32)
        } else {
            0.0
        };

        // Calculate reliability based on source
        let reliability = match feedback.source {
            FeedbackSource::PerformanceMonitor => 0.9,
            FeedbackSource::ResourceMonitor => 0.85,
            FeedbackSource::TestExecutionEngine => 0.8,
            FeedbackSource::ExternalSystem => 0.6,
            FeedbackSource::UserInput => 0.4,
        };

        // Calculate relevance based on feedback type and context
        let relevance = match feedback.feedback_type {
            FeedbackType::Throughput => 1.0,
            FeedbackType::Latency => 0.8,
            FeedbackType::ResourceUtilization => 0.7,
            FeedbackType::Quality => 0.6,
            FeedbackType::ErrorRate => 0.5,
            FeedbackType::Custom(_) => 0.5,
        };

        // Calculate completeness (simplified)
        let completeness = if feedback.context.additional_context.is_empty() { 0.5 } else { 0.9 };

        // Calculate consistency with historical data
        let consistency = self.calculate_consistency(feedback.value)?;

        let overall_quality =
            (reliability + relevance + timeliness + completeness + consistency) / 5.0;

        Ok(FeedbackQualityMetrics {
            reliability,
            relevance,
            timeliness,
            completeness,
            consistency,
            overall_quality,
            assessed_at: Utc::now(),
        })
    }

    /// Calculate consistency with historical data
    fn calculate_consistency(&self, value: f64) -> Result<f32> {
        let historical = self.historical_data.lock();
        if historical.len() < 3 {
            return Ok(0.8); // Default consistency for insufficient data
        }

        let mean: f64 = historical.iter().sum::<f64>() / historical.len() as f64;
        let variance: f64 =
            historical.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / historical.len() as f64;
        let std_dev = variance.sqrt();

        let z_score = if std_dev > 0.0 { (value - mean).abs() / std_dev } else { 0.0 };

        // Convert z-score to consistency score (higher z-score = lower consistency)
        let consistency = (1.0 / (1.0 + z_score * 0.5)).clamp(0.0, 1.0);
        Ok(consistency as f32)
    }

    /// Detect anomalies in feedback
    pub fn detect_anomalies(&self, feedback: &PerformanceFeedback) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        // Check for value range anomalies
        if feedback.value < 0.0 {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::ValueOutOfRange,
                severity: IssueSeverity::High,
                description: "Negative throughput value detected".to_string(),
                suggested_resolution: Some("Verify data source and collection method".to_string()),
            });
        }

        // Check for extreme values
        let historical = self.historical_data.lock();
        if !historical.is_empty() {
            let max_historical = historical.iter().fold(0.0f64, |a, &b| a.max(b));
            if feedback.value > max_historical * 10.0 {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::ValueOutOfRange,
                    severity: IssueSeverity::Medium,
                    description: "Unusually high throughput value detected".to_string(),
                    suggested_resolution: Some("Verify measurement accuracy".to_string()),
                });
            }
        }

        // Check temporal consistency
        let age = Utc::now().signed_duration_since(feedback.timestamp);
        if age > chrono::Duration::hours(1) {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::TemporalInconsistency,
                severity: IssueSeverity::Medium,
                description: "Feedback data is stale".to_string(),
                suggested_resolution: Some("Use more recent measurements".to_string()),
            });
        }

        Ok(issues)
    }

    /// Generate enhanced recommendations
    pub fn generate_enhanced_recommendation(
        &self,
        feedback: &PerformanceFeedback,
        quality_metrics: &FeedbackQualityMetrics,
    ) -> Result<Option<EnhancedRecommendedAction>> {
        if quality_metrics.overall_quality < 0.5 {
            return Ok(None); // Don't generate recommendations for low-quality feedback
        }

        let base_action = RecommendedAction {
            action_type: if feedback.value > 100.0 {
                ActionType::IncreaseParallelism
            } else {
                ActionType::OptimizeTestBatching
            },
            parameters: HashMap::new(),
            priority: quality_metrics.overall_quality,
            expected_impact: 0.15 * quality_metrics.overall_quality,
            reversible: true,
            estimated_duration: Duration::from_secs(5),
        };

        let enhanced_action = EnhancedRecommendedAction {
            base_action,
            category: ActionCategory::PerformanceOptimization,
            priority_level: if quality_metrics.overall_quality > 0.8 {
                ActionPriority::High
            } else {
                ActionPriority::Medium
            },
            resource_requirements: ResourceRequirements {
                cpu_requirements: 0.1,
                memory_requirements: 100,
                io_requirements: 0.05,
                network_requirements: 0.02,
                execution_time: Duration::from_secs(30),
                additional_resources: HashMap::new(),
            },
            complexity: ActionComplexity::Moderate,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                potential_risks: vec![Risk {
                    risk_type: RiskType::PerformanceDegradation,
                    probability: 0.1,
                    impact: RiskImpact::Minor,
                    description: "Temporary performance impact during adjustment".to_string(),
                }],
                mitigation_strategies: vec![MitigationStrategy {
                    strategy_type: MitigationType::GradualRollout,
                    steps: vec!["Implement change gradually".to_string()],
                    effectiveness: 0.9,
                    implementation_cost: 0.2,
                }],
                rollback_available: true,
                recovery_time: Some(Duration::from_secs(60)),
            },
            success_probability: quality_metrics.overall_quality,
            alternatives: vec![],
            dependencies: vec![],
            contraindications: vec![],
            action: format!(
                "Optimize throughput by adjusting parallelism (current: {:.2})",
                feedback.value
            ),
            confidence: quality_metrics.overall_quality,
            rationale: "Based on throughput analysis and quality assessment".to_string(),
        };

        Ok(Some(enhanced_action))
    }

    /// Update historical data
    pub fn update_historical_data(&self, value: f64) -> Result<()> {
        let mut historical = self.historical_data.lock();
        historical.push_back(value);

        // Keep only the last 1000 values
        while historical.len() > 1000 {
            historical.pop_front();
        }

        Ok(())
    }

    /// Process feedback with quality assessment
    pub async fn process_with_quality(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<crate::performance_optimizer::types::ProcessedFeedback> {
        let quality = self.assess_quality(feedback)?;
        let enhanced_recommendation = self.generate_enhanced_recommendation(feedback, &quality)?;

        Ok(crate::performance_optimizer::types::ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value: feedback.value,
            processing_method: "enhanced_throughput".to_string(),
            confidence: quality.overall_quality,
            recommended_action: enhanced_recommendation.map(|er| er.base_action),
        })
    }
}

// =============================================================================
// LATENCY FEEDBACK PROCESSOR
// =============================================================================

/// Latency-focused feedback processor
pub struct LatencyFeedbackProcessor {
    /// Latency percentile targets
    percentile_targets: HashMap<u8, Duration>,
    /// SLA thresholds
    sla_thresholds: SlaThresholds,
    /// Performance degradation detection
    degradation_detection: bool,
}

/// SLA threshold configuration
#[derive(Debug, Clone)]
pub struct SlaThresholds {
    /// P50 latency threshold
    pub p50_threshold: Duration,
    /// P95 latency threshold
    pub p95_threshold: Duration,
    /// P99 latency threshold
    pub p99_threshold: Duration,
    /// Maximum acceptable latency
    pub max_latency: Duration,
}

impl Default for SlaThresholds {
    fn default() -> Self {
        Self {
            p50_threshold: Duration::from_millis(100),
            p95_threshold: Duration::from_millis(500),
            p99_threshold: Duration::from_millis(1000),
            max_latency: Duration::from_millis(5000),
        }
    }
}

impl Default for LatencyFeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyFeedbackProcessor {
    /// Create new latency feedback processor
    pub fn new() -> Self {
        let mut percentile_targets = HashMap::new();
        percentile_targets.insert(50, Duration::from_millis(100));
        percentile_targets.insert(95, Duration::from_millis(500));
        percentile_targets.insert(99, Duration::from_millis(1000));

        Self {
            percentile_targets,
            sla_thresholds: SlaThresholds::default(),
            degradation_detection: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(_config: super::types::LatencyProcessorConfig) -> Self {
        // For now, use default implementation
        Self::new()
    }

    /// Assess latency quality against SLA thresholds
    pub fn assess_latency_quality(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<FeedbackQualityMetrics> {
        let latency_ms = feedback.value;
        let latency_duration = Duration::from_millis(latency_ms as u64);

        // Calculate quality based on SLA compliance
        let relevance = if latency_duration <= self.sla_thresholds.p50_threshold {
            1.0
        } else if latency_duration <= self.sla_thresholds.p95_threshold {
            0.8
        } else if latency_duration <= self.sla_thresholds.p99_threshold {
            0.6
        } else if latency_duration <= self.sla_thresholds.max_latency {
            0.4
        } else {
            0.2
        };

        // Calculate reliability based on source consistency
        let reliability = match feedback.source {
            FeedbackSource::PerformanceMonitor => 0.9,
            FeedbackSource::TestExecutionEngine => 0.85,
            FeedbackSource::ResourceMonitor => 0.7,
            FeedbackSource::ExternalSystem => 0.6,
            FeedbackSource::UserInput => 0.3,
        };

        let age = Utc::now().signed_duration_since(feedback.timestamp);
        let timeliness = if age <= chrono::Duration::seconds(60) {
            1.0 - (age.num_seconds() as f32 / 60.0)
        } else {
            0.0
        };

        let completeness = 0.8; // Simplified for latency measurements
        let consistency = 0.8; // Would calculate from historical data in full implementation

        let overall_quality =
            (reliability + relevance + timeliness + completeness + consistency) / 5.0;

        Ok(FeedbackQualityMetrics {
            reliability,
            relevance,
            timeliness,
            completeness,
            consistency,
            overall_quality,
            assessed_at: Utc::now(),
        })
    }

    /// Generate latency-specific recommendations
    pub fn generate_latency_recommendation(
        &self,
        feedback: &PerformanceFeedback,
        quality_metrics: &FeedbackQualityMetrics,
    ) -> Result<Option<EnhancedRecommendedAction>> {
        if quality_metrics.overall_quality < 0.4 {
            return Ok(None);
        }

        let latency_ms = feedback.value;
        let latency_duration = Duration::from_millis(latency_ms as u64);

        let (action_type, priority, complexity) =
            if latency_duration > self.sla_thresholds.max_latency {
                (
                    ActionType::DecreaseParallelism,
                    ActionPriority::Critical,
                    ActionComplexity::Simple,
                )
            } else if latency_duration > self.sla_thresholds.p99_threshold {
                (
                    ActionType::OptimizeTestBatching,
                    ActionPriority::High,
                    ActionComplexity::Moderate,
                )
            } else {
                (
                    ActionType::IncreaseParallelism,
                    ActionPriority::Medium,
                    ActionComplexity::Moderate,
                )
            };

        let base_action = RecommendedAction {
            action_type,
            parameters: HashMap::new(),
            priority: quality_metrics.overall_quality,
            expected_impact: 0.2,
            reversible: true,
            estimated_duration: Duration::from_secs(3),
        };

        let enhanced_action = EnhancedRecommendedAction {
            base_action,
            category: ActionCategory::PerformanceOptimization,
            priority_level: priority,
            resource_requirements: ResourceRequirements::default(),
            complexity,
            risk_assessment: RiskAssessment::default(),
            success_probability: quality_metrics.overall_quality,
            alternatives: vec![],
            dependencies: vec![],
            contraindications: vec![],
            action: format!(
                "Adjust parallelism to reduce latency (current: {:.2}ms)",
                latency_ms
            ),
            confidence: quality_metrics.overall_quality,
            rationale: "Based on latency SLA compliance analysis".to_string(),
        };

        Ok(Some(enhanced_action))
    }

    /// Process feedback with quality assessment
    pub async fn process_with_quality(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<crate::performance_optimizer::types::ProcessedFeedback> {
        let quality = self.assess_latency_quality(feedback)?;
        let enhanced_recommendation = self.generate_latency_recommendation(feedback, &quality)?;

        Ok(crate::performance_optimizer::types::ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value: feedback.value,
            processing_method: "latency_analysis".to_string(),
            confidence: quality.overall_quality,
            recommended_action: enhanced_recommendation.map(|er| er.base_action),
        })
    }
}

// =============================================================================
// RESOURCE UTILIZATION PROCESSOR
// =============================================================================

/// Resource utilization feedback processor
pub struct ResourceUtilizationProcessor {
    /// Resource thresholds
    resource_thresholds: ResourceThresholds,
    /// Efficiency calculation method
    efficiency_method: EfficiencyCalculationMethod,
    /// Bottleneck detection enabled
    bottleneck_detection: bool,
}

/// Resource utilization thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// CPU utilization thresholds
    pub cpu_thresholds: UtilizationThresholds,
    /// Memory utilization thresholds
    pub memory_thresholds: UtilizationThresholds,
    /// I/O utilization thresholds
    pub io_thresholds: UtilizationThresholds,
    /// Network utilization thresholds
    pub network_thresholds: UtilizationThresholds,
}

/// Utilization thresholds for a resource
#[derive(Debug, Clone)]
pub struct UtilizationThresholds {
    /// Low utilization threshold
    pub low_threshold: f32,
    /// Optimal utilization range
    pub optimal_range: (f32, f32),
    /// High utilization threshold
    pub high_threshold: f32,
    /// Critical utilization threshold
    pub critical_threshold: f32,
}

impl Default for UtilizationThresholds {
    fn default() -> Self {
        Self {
            low_threshold: 0.2,
            optimal_range: (0.5, 0.8),
            high_threshold: 0.9,
            critical_threshold: 0.95,
        }
    }
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_thresholds: UtilizationThresholds::default(),
            memory_thresholds: UtilizationThresholds::default(),
            io_thresholds: UtilizationThresholds::default(),
            network_thresholds: UtilizationThresholds::default(),
        }
    }
}

/// Efficiency calculation methods
#[derive(Debug, Clone)]
pub enum EfficiencyCalculationMethod {
    /// Simple ratio
    SimpleRatio,
    /// Weighted efficiency
    WeightedEfficiency,
    /// Data envelopment analysis
    DataEnvelopmentAnalysis,
    /// Custom method
    Custom(String),
}

impl Default for ResourceUtilizationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceUtilizationProcessor {
    /// Create new resource utilization processor
    pub fn new() -> Self {
        Self {
            resource_thresholds: ResourceThresholds::default(),
            efficiency_method: EfficiencyCalculationMethod::WeightedEfficiency,
            bottleneck_detection: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(_config: super::types::ResourceProcessorConfig) -> Self {
        // For now, use default implementation
        Self::new()
    }

    /// Assess resource utilization quality
    pub fn assess_utilization_quality(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<FeedbackQualityMetrics> {
        let utilization = feedback.value as f32;

        // Determine optimal range based on feedback type
        let optimal_range = match feedback.feedback_type {
            FeedbackType::ResourceUtilization => {
                // Use CPU thresholds as default
                self.resource_thresholds.cpu_thresholds.optimal_range
            },
            _ => (0.5, 0.8), // Default range
        };

        // Calculate relevance based on whether utilization is in optimal range
        let relevance = if utilization >= optimal_range.0 && utilization <= optimal_range.1 {
            1.0
        } else if utilization < self.resource_thresholds.cpu_thresholds.low_threshold {
            0.6 // Under-utilization
        } else if utilization > self.resource_thresholds.cpu_thresholds.high_threshold {
            0.4 // Over-utilization
        } else {
            0.8 // Acceptable but not optimal
        };

        let reliability = 0.8; // Would be calculated from source reliability
        let timeliness = 0.9; // Would be calculated from feedback age
        let completeness = 0.8; // Would be calculated from context completeness
        let consistency = 0.8; // Would be calculated from historical data

        let overall_quality =
            (reliability + relevance + timeliness + completeness + consistency) / 5.0;

        Ok(FeedbackQualityMetrics {
            reliability,
            relevance,
            timeliness,
            completeness,
            consistency,
            overall_quality,
            assessed_at: Utc::now(),
        })
    }

    /// Generate resource utilization recommendations
    pub fn generate_utilization_recommendation(
        &self,
        feedback: &PerformanceFeedback,
        quality_metrics: &FeedbackQualityMetrics,
    ) -> Result<Option<EnhancedRecommendedAction>> {
        let utilization = feedback.value as f32;
        let thresholds = &self.resource_thresholds.cpu_thresholds;

        if quality_metrics.overall_quality < 0.5 {
            return Ok(None);
        }

        let (action_type, priority, risk_level) = if utilization > thresholds.critical_threshold {
            (
                ActionType::DecreaseParallelism,
                ActionPriority::Critical,
                RiskLevel::High,
            )
        } else if utilization > thresholds.high_threshold {
            (
                ActionType::OptimizeTestBatching,
                ActionPriority::High,
                RiskLevel::Medium,
            )
        } else if utilization < thresholds.low_threshold {
            (
                ActionType::IncreaseParallelism,
                ActionPriority::Medium,
                RiskLevel::Low,
            )
        } else {
            return Ok(None); // Utilization is in acceptable range
        };

        let base_action = RecommendedAction {
            action_type,
            parameters: HashMap::new(),
            priority: quality_metrics.overall_quality,
            expected_impact: 0.25,
            reversible: true,
            estimated_duration: Duration::from_secs(5),
        };

        let enhanced_action = EnhancedRecommendedAction {
            base_action,
            category: ActionCategory::ResourceManagement,
            priority_level: priority,
            resource_requirements: ResourceRequirements::default(),
            complexity: ActionComplexity::Moderate,
            risk_assessment: RiskAssessment {
                risk_level,
                ..RiskAssessment::default()
            },
            success_probability: quality_metrics.overall_quality,
            alternatives: vec![],
            dependencies: vec![],
            contraindications: vec![],
            action: format!(
                "Adjust resource utilization (current: {:.1}%)",
                utilization * 100.0
            ),
            confidence: quality_metrics.overall_quality,
            rationale: "Based on resource utilization threshold analysis".to_string(),
        };

        Ok(Some(enhanced_action))
    }

    /// Process feedback with quality assessment
    pub async fn process_with_quality(
        &self,
        feedback: &PerformanceFeedback,
    ) -> Result<crate::performance_optimizer::types::ProcessedFeedback> {
        let quality = self.assess_utilization_quality(feedback)?;
        let enhanced_recommendation =
            self.generate_utilization_recommendation(feedback, &quality)?;

        Ok(crate::performance_optimizer::types::ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value: feedback.value,
            processing_method: "resource_utilization".to_string(),
            confidence: quality.overall_quality,
            recommended_action: enhanced_recommendation.map(|er| er.base_action),
        })
    }
}
