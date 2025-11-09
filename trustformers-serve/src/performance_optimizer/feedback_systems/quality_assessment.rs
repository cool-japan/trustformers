//! Feedback Quality Assessment
//!
//! This module provides comprehensive quality assessment capabilities for
//! performance feedback including quality metrics calculation, trend analysis,
//! and historical quality tracking.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::VecDeque, sync::Arc};

use super::{processors::QualityThresholds, types::*};
use crate::performance_optimizer::types::{FeedbackSource, FeedbackType, PerformanceFeedback};

// =============================================================================
// FEEDBACK QUALITY ASSESSOR
// =============================================================================

/// Feedback quality assessor
pub struct FeedbackQualityAssessor {
    /// Quality thresholds
    thresholds: QualityThresholds,
    /// Historical quality data
    quality_history: Arc<Mutex<VecDeque<FeedbackQualityMetrics>>>,
}

impl FeedbackQualityAssessor {
    /// Create new feedback quality assessor
    pub fn new() -> Self {
        Self {
            thresholds: QualityThresholds::default(),
            quality_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: QualityThresholds) -> Self {
        Self {
            thresholds,
            quality_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Assess feedback quality comprehensively
    pub fn assess_quality(&self, feedback: &PerformanceFeedback) -> Result<FeedbackQualityMetrics> {
        let age = Utc::now().signed_duration_since(feedback.timestamp);

        // Calculate timeliness
        let timeliness = if age <= chrono::Duration::from_std(self.thresholds.max_age)? {
            1.0 - (age.num_seconds() as f32 / self.thresholds.max_age.as_secs() as f32)
        } else {
            0.0
        };

        // Calculate reliability based on source
        let reliability = self.calculate_source_reliability(&feedback.source);

        // Calculate relevance based on feedback type
        let relevance = self.calculate_type_relevance(&feedback.feedback_type);

        // Calculate completeness based on context
        let completeness = self.calculate_completeness(feedback);

        // Calculate consistency with historical data
        let consistency = self.calculate_consistency(feedback.value)?;

        let overall_quality =
            (reliability + relevance + timeliness + completeness + consistency) / 5.0;

        let quality_metrics = FeedbackQualityMetrics {
            reliability,
            relevance,
            timeliness,
            completeness,
            consistency,
            overall_quality,
            assessed_at: Utc::now(),
        };

        // Store in history
        self.store_quality_metrics(&quality_metrics)?;

        Ok(quality_metrics)
    }

    /// Calculate source reliability score
    fn calculate_source_reliability(&self, source: &FeedbackSource) -> f32 {
        match source {
            FeedbackSource::PerformanceMonitor => 0.95,
            FeedbackSource::ResourceMonitor => 0.9,
            FeedbackSource::TestExecutionEngine => 0.85,
            FeedbackSource::ExternalSystem => 0.7,
            FeedbackSource::UserInput => 0.5,
        }
    }

    /// Calculate feedback type relevance
    fn calculate_type_relevance(&self, feedback_type: &FeedbackType) -> f32 {
        match feedback_type {
            FeedbackType::Throughput => 1.0,
            FeedbackType::Latency => 0.9,
            FeedbackType::ResourceUtilization => 0.85,
            FeedbackType::Quality => 0.8,
            FeedbackType::ErrorRate => 0.75,
            FeedbackType::Custom(_) => 0.6,
        }
    }

    /// Calculate completeness based on context richness
    fn calculate_completeness(&self, feedback: &PerformanceFeedback) -> f32 {
        let mut completeness_score = 0.0;
        let mut max_score = 0.0;

        // Base context fields
        // TODO: FeedbackContext fields changed - no longer has test_name, resource_usage, environment_info as Option fields
        // FeedbackContext now has test_characteristics, system_state (not optional), and additional_context
        max_score += 1.0;
        // Always has test_characteristics
        completeness_score += 0.2;

        max_score += 1.0;
        // Always has system_state
        completeness_score += 0.2;

        // Additional context
        max_score += 1.0;
        if !feedback.context.additional_context.is_empty() {
            completeness_score += 0.2;
        }

        if max_score > 0.0 {
            completeness_score / max_score
        } else {
            0.5
        }
    }

    /// Calculate consistency with historical data
    fn calculate_consistency(&self, _value: f64) -> Result<f32> {
        let history = self.quality_history.lock();

        // Need recent quality data to determine consistency
        if history.len() < 3 {
            return Ok(0.8); // Default consistency for insufficient data
        }

        // Use recent overall quality scores as a proxy
        let recent_scores: Vec<f32> =
            history.iter().rev().take(10).map(|q| q.overall_quality).collect();

        let mean = recent_scores.iter().sum::<f32>() / recent_scores.len() as f32;
        let variance = recent_scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / recent_scores.len() as f32;
        let std_dev = variance.sqrt();

        // For value consistency, we need to compare with historical feedback values
        // This is a simplified version - in reality we'd store historical feedback values
        let consistency = if std_dev > 0.0 {
            let stability_factor = 1.0 / (1.0 + std_dev * 2.0);
            stability_factor.clamp(0.0, 1.0)
        } else {
            1.0
        };

        Ok(consistency)
    }

    /// Store quality metrics in history
    fn store_quality_metrics(&self, metrics: &FeedbackQualityMetrics) -> Result<()> {
        let mut history = self.quality_history.lock();
        history.push_back(metrics.clone());

        // Limit history size
        if history.len() > 1000 {
            history.pop_front();
        }

        Ok(())
    }

    /// Get quality statistics
    pub fn get_quality_statistics(&self) -> QualityStatistics {
        let history = self.quality_history.lock();

        if history.is_empty() {
            return QualityStatistics::default();
        }

        let total_assessments = history.len();
        let average_quality =
            history.iter().map(|q| q.overall_quality).sum::<f32>() / total_assessments as f32;

        let high_quality_count = history.iter().filter(|q| q.overall_quality >= 0.8).count();
        let medium_quality_count = history
            .iter()
            .filter(|q| q.overall_quality >= 0.6 && q.overall_quality < 0.8)
            .count();
        let low_quality_count = history.iter().filter(|q| q.overall_quality < 0.6).count();

        // Calculate component averages
        let avg_reliability =
            history.iter().map(|q| q.reliability).sum::<f32>() / total_assessments as f32;
        let avg_relevance =
            history.iter().map(|q| q.relevance).sum::<f32>() / total_assessments as f32;
        let avg_timeliness =
            history.iter().map(|q| q.timeliness).sum::<f32>() / total_assessments as f32;
        let avg_completeness =
            history.iter().map(|q| q.completeness).sum::<f32>() / total_assessments as f32;
        let avg_consistency =
            history.iter().map(|q| q.consistency).sum::<f32>() / total_assessments as f32;

        QualityStatistics {
            total_assessments,
            average_quality,
            high_quality_count,
            medium_quality_count,
            low_quality_count,
            average_reliability: avg_reliability,
            average_relevance: avg_relevance,
            average_timeliness: avg_timeliness,
            average_completeness: avg_completeness,
            average_consistency: avg_consistency,
        }
    }

    /// Get recent quality trend
    pub fn get_quality_trend(&self, window_size: usize) -> QualityTrend {
        let history = self.quality_history.lock();

        if history.len() < 2 {
            return QualityTrend::Stable;
        }

        let window = window_size.min(history.len());
        let recent_scores: Vec<f32> =
            history.iter().rev().take(window).map(|q| q.overall_quality).collect();

        if recent_scores.len() < 2 {
            return QualityTrend::Stable;
        }

        // Simple trend calculation
        let first_half_avg =
            recent_scores.iter().skip(window / 2).sum::<f32>() / (window - window / 2) as f32;
        let second_half_avg =
            recent_scores.iter().take(window / 2).sum::<f32>() / (window / 2) as f32;

        let change = second_half_avg - first_half_avg;

        if change > 0.1 {
            QualityTrend::Improving
        } else if change < -0.1 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }

    /// Get quality breakdown by component
    pub fn get_quality_breakdown(&self) -> QualityBreakdown {
        let history = self.quality_history.lock();

        if history.is_empty() {
            return QualityBreakdown::default();
        }

        let recent_metrics: Vec<&FeedbackQualityMetrics> = history.iter().rev().take(50).collect();

        let reliability_avg =
            recent_metrics.iter().map(|q| q.reliability).sum::<f32>() / recent_metrics.len() as f32;
        let relevance_avg =
            recent_metrics.iter().map(|q| q.relevance).sum::<f32>() / recent_metrics.len() as f32;
        let timeliness_avg =
            recent_metrics.iter().map(|q| q.timeliness).sum::<f32>() / recent_metrics.len() as f32;
        let completeness_avg = recent_metrics.iter().map(|q| q.completeness).sum::<f32>()
            / recent_metrics.len() as f32;
        let consistency_avg =
            recent_metrics.iter().map(|q| q.consistency).sum::<f32>() / recent_metrics.len() as f32;

        QualityBreakdown {
            reliability: reliability_avg,
            relevance: relevance_avg,
            timeliness: timeliness_avg,
            completeness: completeness_avg,
            consistency: consistency_avg,
        }
    }
}

// =============================================================================
// QUALITY STATISTICS AND TRENDS
// =============================================================================

/// Quality statistics
#[derive(Debug, Clone)]
pub struct QualityStatistics {
    pub total_assessments: usize,
    pub average_quality: f32,
    pub high_quality_count: usize,
    pub medium_quality_count: usize,
    pub low_quality_count: usize,
    pub average_reliability: f32,
    pub average_relevance: f32,
    pub average_timeliness: f32,
    pub average_completeness: f32,
    pub average_consistency: f32,
}

impl Default for QualityStatistics {
    fn default() -> Self {
        Self {
            total_assessments: 0,
            average_quality: 0.0,
            high_quality_count: 0,
            medium_quality_count: 0,
            low_quality_count: 0,
            average_reliability: 0.0,
            average_relevance: 0.0,
            average_timeliness: 0.0,
            average_completeness: 0.0,
            average_consistency: 0.0,
        }
    }
}

/// Quality trend direction
#[derive(Debug, Clone)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Quality breakdown by component
#[derive(Debug, Clone)]
pub struct QualityBreakdown {
    pub reliability: f32,
    pub relevance: f32,
    pub timeliness: f32,
    pub completeness: f32,
    pub consistency: f32,
}

impl Default for QualityBreakdown {
    fn default() -> Self {
        Self {
            reliability: 0.0,
            relevance: 0.0,
            timeliness: 0.0,
            completeness: 0.0,
            consistency: 0.0,
        }
    }
}
