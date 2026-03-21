//! Enhanced Feedback Systems Module
//!
//! This module provides comprehensive feedback processing capabilities for performance
//! optimization including quality assessment, validation, aggregation strategies,
//! and advanced analytics. The system supports multiple feedback sources, sophisticated
//! validation rules, and multi-objective optimization strategies.
//!
//! # Architecture
//!
//! The feedback system is organized into specialized modules:
//! - `types`: Core type definitions and data structures
//! - `processors`: Enhanced feedback processors for different metric types
//! - `aggregation`: Sophisticated aggregation strategies with statistical analysis
//! - `validation`: Comprehensive validation engine with rule-based system
//! - `quality_assessment`: Quality assessment with trend analysis and statistics
//!
//! # Features
//!
//! - **Multi-source feedback processing** with reliability scoring
//! - **Advanced quality assessment** with comprehensive metrics
//! - **Sophisticated aggregation** strategies including time-series and consensus
//! - **Comprehensive validation** with customizable rules and issue detection
//! - **Statistical analysis** with trend detection and anomaly identification
//! - **Machine learning integration** for enhanced processing capabilities
//! - **Risk assessment** with mitigation strategies
//! - **Multi-objective optimization** with Pareto analysis
//!
//! # Usage
//!
//! ```rust
//! use trustformers_serve::performance_optimizer::feedback_systems::{
//!     FeedbackQualityAssessor, EnhancedThroughputProcessor, ValidationEngine
//! };
//!
//! // Create enhanced feedback processor
//! let processor = EnhancedThroughputProcessor::new();
//!
//! // Process feedback with quality assessment
//! let processed = processor.process_with_quality(&feedback)?;
//!
//! // Validate feedback
//! let validator = ValidationEngine::new();
//! let validation_result = validator.validate(&feedback)?;
//! ```

use anyhow::Result;

// Re-export all public types and interfaces
pub use processors::*;
pub use quality_assessment::*;
pub use types::*;
pub use validation::*;

// Module declarations
pub mod aggregation;
pub mod processors;
pub mod quality_assessment;
pub mod types;
pub mod validation;

use crate::performance_optimizer::types::{AggregatedFeedback, FeedbackType, ProcessedFeedback};

// Re-export types from parent module
pub use crate::performance_optimizer::types::PerformanceFeedback;

// =============================================================================
// MAIN FEEDBACK SYSTEM COORDINATOR
// =============================================================================

/// Main feedback system coordinator that orchestrates all components
pub struct FeedbackSystemCoordinator {
    /// Throughput processor
    throughput_processor: EnhancedThroughputProcessor,
    /// Latency processor
    latency_processor: LatencyFeedbackProcessor,
    /// Resource utilization processor
    resource_processor: ResourceUtilizationProcessor,
    /// Validation engine
    validation_engine: FeedbackValidationEngine,
    /// Quality assessor
    quality_assessor: FeedbackQualityAssessor,
    /// Aggregation manager (using weighted confidence strategy)
    aggregation_manager: WeightedConfidenceAggregationStrategy,
}

impl FeedbackSystemCoordinator {
    /// Create new feedback system coordinator
    pub fn new() -> Self {
        Self {
            throughput_processor: EnhancedThroughputProcessor::new(),
            latency_processor: LatencyFeedbackProcessor::new(),
            resource_processor: ResourceUtilizationProcessor::new(),
            validation_engine: FeedbackValidationEngine::new(),
            quality_assessor: FeedbackQualityAssessor::new(),
            aggregation_manager: WeightedConfidenceAggregationStrategy::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        throughput_config: ThroughputProcessorConfig,
        latency_config: LatencyProcessorConfig,
        resource_config: ResourceProcessorConfig,
        validation_config: ValidationEngineConfig,
        quality_thresholds: QualityThresholds,
        aggregation_config: AggregationManagerConfig,
    ) -> Self {
        Self {
            throughput_processor: EnhancedThroughputProcessor::with_config(throughput_config),
            latency_processor: LatencyFeedbackProcessor::with_config(latency_config),
            resource_processor: ResourceUtilizationProcessor::with_config(resource_config),
            validation_engine: FeedbackValidationEngine::with_config(validation_config),
            quality_assessor: FeedbackQualityAssessor::with_thresholds(quality_thresholds),
            aggregation_manager: WeightedConfidenceAggregationStrategy::with_config(
                aggregation_config,
            ),
        }
    }

    /// Process feedback comprehensively through all stages
    pub async fn process_feedback_comprehensive(
        &mut self,
        feedback: &PerformanceFeedback,
    ) -> Result<ComprehensiveFeedbackResult> {
        // Stage 1: Validation
        let validation_result = self.validation_engine.validate(feedback).await?;
        if !validation_result.valid {
            return Ok(ComprehensiveFeedbackResult {
                validation_result: Some(validation_result),
                quality_metrics: None,
                processed_feedback: None,
                recommendations: Vec::new(),
                aggregation_result: None,
            });
        }

        // Stage 2: Quality Assessment
        let quality_metrics = self.quality_assessor.assess_quality(feedback)?;

        // Stage 3: Processing based on feedback type
        let processed_feedback = match feedback.feedback_type {
            FeedbackType::Throughput => {
                Some(self.throughput_processor.process_with_quality(feedback).await?)
            },
            FeedbackType::Latency => {
                Some(self.latency_processor.process_with_quality(feedback).await?)
            },
            FeedbackType::ResourceUtilization => {
                Some(self.resource_processor.process_with_quality(feedback).await?)
            },
            _ => {
                // Default processing for other types
                Some(ProcessedFeedback {
                    original_feedback: feedback.clone(),
                    processed_value: feedback.value,
                    processing_method: "default".to_string(),
                    confidence: quality_metrics.overall_quality,
                    recommended_action: None,
                })
            },
        };

        // Stage 4: Generate recommendations
        let recommendations = if let Some(ref processed) = processed_feedback {
            if let Some(ref recommended_action) = processed.recommended_action {
                vec![EnhancedRecommendedAction {
                    base_action: recommended_action.clone(),
                    category: ActionCategory::PerformanceOptimization,
                    priority_level: ActionPriority::Medium,
                    resource_requirements: ResourceRequirements::default(),
                    complexity: ActionComplexity::Moderate,
                    risk_assessment: RiskAssessment::default(),
                    success_probability: processed.confidence,
                    alternatives: Vec::new(),
                    dependencies: Vec::new(),
                    contraindications: Vec::new(),
                    action: format!("{:?}", recommended_action.action_type),
                    confidence: processed.confidence,
                    rationale: processed.processing_method.clone(),
                }]
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Ok(ComprehensiveFeedbackResult {
            validation_result: Some(validation_result),
            quality_metrics: Some(quality_metrics),
            processed_feedback,
            recommendations,
            aggregation_result: None, // Would be populated when aggregating multiple feedbacks
        })
    }

    /// Process multiple feedbacks with aggregation
    pub async fn process_multiple_feedbacks(
        &mut self,
        feedbacks: &[PerformanceFeedback],
        strategy: AggregationStrategyType,
    ) -> Result<MultipleProcessingResult> {
        let mut processed_feedbacks = Vec::new();
        let mut validation_results = Vec::new();
        let mut quality_metrics = Vec::new();

        // Process each feedback individually
        for feedback in feedbacks {
            let comprehensive_result = self.process_feedback_comprehensive(feedback).await?;

            if let Some(validation) = comprehensive_result.validation_result {
                validation_results.push(validation);
            }

            if let Some(quality) = comprehensive_result.quality_metrics {
                quality_metrics.push(quality);
            }

            if let Some(processed) = comprehensive_result.processed_feedback {
                processed_feedbacks.push(processed);
            }
        }

        // Aggregate processed feedbacks
        let aggregation_result = if !processed_feedbacks.is_empty() {
            Some(
                self.aggregation_manager
                    .aggregate_with_strategy(&processed_feedbacks, strategy)
                    .await?,
            )
        } else {
            None
        };

        let overall_statistics = self.calculate_overall_statistics(&quality_metrics);

        Ok(MultipleProcessingResult {
            individual_results: processed_feedbacks,
            validation_results,
            quality_metrics,
            aggregation_result,
            overall_statistics,
        })
    }

    /// Get system health and statistics
    pub fn get_system_health(&self) -> SystemHealthStatus {
        let quality_stats = self.quality_assessor.get_quality_statistics();
        let quality_trend = self.quality_assessor.get_quality_trend(20);
        let quality_breakdown = self.quality_assessor.get_quality_breakdown();

        SystemHealthStatus {
            overall_health: self.calculate_overall_health(&quality_stats),
            quality_statistics: quality_stats,
            quality_trend,
            quality_breakdown,
            processor_status: ProcessorStatus {
                throughput_processor_active: true,
                latency_processor_active: true,
                resource_processor_active: true,
                validation_engine_active: true,
                quality_assessor_active: true,
                aggregation_manager_active: true,
            },
            last_updated: chrono::Utc::now(),
        }
    }

    /// Calculate overall health score
    fn calculate_overall_health(&self, quality_stats: &QualityStatistics) -> f32 {
        if quality_stats.total_assessments == 0 {
            return 0.5; // Neutral health with no data
        }

        let mut health_score = quality_stats.average_quality * 0.4;

        // Factor in distribution of quality levels
        let total = quality_stats.total_assessments as f32;
        let high_quality_ratio = quality_stats.high_quality_count as f32 / total;
        let low_quality_ratio = quality_stats.low_quality_count as f32 / total;

        health_score += high_quality_ratio * 0.3;
        health_score -= low_quality_ratio * 0.2;

        // Factor in component averages
        let component_avg = (quality_stats.average_reliability
            + quality_stats.average_relevance
            + quality_stats.average_timeliness
            + quality_stats.average_completeness
            + quality_stats.average_consistency)
            / 5.0;

        health_score += component_avg * 0.3;

        health_score.clamp(0.0, 1.0)
    }

    /// Calculate overall statistics from quality metrics
    fn calculate_overall_statistics(
        &self,
        quality_metrics: &[FeedbackQualityMetrics],
    ) -> OverallStatistics {
        if quality_metrics.is_empty() {
            return OverallStatistics::default();
        }

        let total_count = quality_metrics.len();
        let average_quality =
            quality_metrics.iter().map(|q| q.overall_quality).sum::<f32>() / total_count as f32;

        let high_quality_count =
            quality_metrics.iter().filter(|q| q.overall_quality >= 0.8).count();
        let medium_quality_count = quality_metrics
            .iter()
            .filter(|q| q.overall_quality >= 0.6 && q.overall_quality < 0.8)
            .count();
        let low_quality_count = quality_metrics.iter().filter(|q| q.overall_quality < 0.6).count();

        OverallStatistics {
            total_processed: total_count,
            average_quality,
            high_quality_count,
            medium_quality_count,
            low_quality_count,
            processing_success_rate: 1.0, // All processed successfully if we reach here
        }
    }
}

impl Default for FeedbackSystemCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Comprehensive feedback processing result
#[derive(Debug, Clone)]
pub struct ComprehensiveFeedbackResult {
    pub validation_result: Option<FeedbackValidationResult>,
    pub quality_metrics: Option<FeedbackQualityMetrics>,
    pub processed_feedback: Option<ProcessedFeedback>,
    pub recommendations: Vec<EnhancedRecommendedAction>,
    pub aggregation_result: Option<AggregatedFeedback>,
}

/// Multiple feedback processing result
#[derive(Debug, Clone)]
pub struct MultipleProcessingResult {
    pub individual_results: Vec<ProcessedFeedback>,
    pub validation_results: Vec<FeedbackValidationResult>,
    pub quality_metrics: Vec<FeedbackQualityMetrics>,
    pub aggregation_result: Option<AggregatedFeedback>,
    pub overall_statistics: OverallStatistics,
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealthStatus {
    pub overall_health: f32,
    pub quality_statistics: QualityStatistics,
    pub quality_trend: QualityTrend,
    pub quality_breakdown: QualityBreakdown,
    pub processor_status: ProcessorStatus,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Processor status information
#[derive(Debug, Clone)]
pub struct ProcessorStatus {
    pub throughput_processor_active: bool,
    pub latency_processor_active: bool,
    pub resource_processor_active: bool,
    pub validation_engine_active: bool,
    pub quality_assessor_active: bool,
    pub aggregation_manager_active: bool,
}

/// Overall processing statistics
#[derive(Debug, Clone)]
pub struct OverallStatistics {
    pub total_processed: usize,
    pub average_quality: f32,
    pub high_quality_count: usize,
    pub medium_quality_count: usize,
    pub low_quality_count: usize,
    pub processing_success_rate: f32,
}

impl Default for OverallStatistics {
    fn default() -> Self {
        Self {
            total_processed: 0,
            average_quality: 0.0,
            high_quality_count: 0,
            medium_quality_count: 0,
            low_quality_count: 0,
            processing_success_rate: 0.0,
        }
    }
}
